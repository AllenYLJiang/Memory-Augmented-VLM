"""Blind full-clip review and one-shot, source-clustered held-out reporting."""
from __future__ import annotations

import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from ..contracts import iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..evaluation import metrics_with_predictions, grouped_bootstrap_delta
from ..hard_trial import continuous_target
from .protocol import advance, at_least, immutable, stable_hash, verify_manifest


def packet(out):
    rows = list(iter_jsonl(out / "enrollment/windows.jsonl"))
    media = read_json(out / "media/manifest.json")
    public, private = [], {}
    for row in rows:
        uid = row["window_uid"]
        blind = "V_" + semantic_sha256(["v912_blind_clip", uid])[:20]
        folder = out / "review/public/media"
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / (blind + ".mp4")
        source = out / "media" / uid / "clip.mp4"
        if not dst.exists():
            shutil.copy2(source, dst)
        if stable_hash(dst) != media[uid]["files"]["clip.mp4"]:
            raise ValueError("Blind clip changed")
        public.append({"blind_id": blind, "clip": "media/" + dst.name, "clip_sha256": stable_hash(dst), "frames": 96})
        private[blind] = {"window_uid": uid, "role": row["role"]}
    public.sort(key=lambda r: r["blind_id"])
    immutable(out / "review/public_manifest.json", public)
    immutable(out / "review/private_map.json", private)
    packet_hash = semantic_sha256(public)
    for reviewer in ("R1", "R2"):
        path = out / "review/returns" / (reviewer + ".jsonl")
        if not path.exists():
            write_jsonl(path, [{"reviewer_id": reviewer, "blind_id": r["blind_id"], "packet_sha256": packet_hash,
                "current_window_visual_label": "pending", "direct_mechanism": "pending",
                "visible_event_intervals_local": [], "event_classes": [], "normal_mechanism": "pending",
                "normal_explains_suspicious_action": "pending", "same_actor_support": "pending", "same_time_support": "pending",
                "unexplained_anomalous_mechanism": "pending", "review_confidence": "pending", "note": "",
                "context_review": {"status": "not_used", "context_used": False, "audio_used": False}}
                for r in public])
    videos = "\n".join(f'<section><h2>{r["blind_id"]}</h2><video controls preload="none" src="{r["clip"]}"></video></section>' for r in public)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind clip review</title><style>body{font:16px system-ui;margin:24px;color:#171717;background:#fff}main{max-width:960px;margin:auto}section{border-top:1px solid #bbb;padding:20px 0}h1{font-size:26px}h2{font-size:16px;overflow-wrap:anywhere}video{width:100%;max-height:540px;background:#111}</style><main><h1>Blind Clip Review</h1>' + videos + '</main></html>'
    (out / "review/public/index.html").write_text(page, encoding="utf-8")
    if not (out / "protocol/review_approval.json").exists():
        write_json(out / "protocol/review_approval.json", {
            "reviewer": "", "enrollment_sha256": stable_hash(out / "enrollment/windows.jsonl"), "smoke_sha256": "",
            "history_scope_checked": False, "source_aliases_checked": False, "enrollment_checked": False,
            "full_clip_media_checked": False, "label_scope_checked": False, "smoke_checked": False,
            "weak_labels_not_gold": True, "locked_outcomes_not_viewed": True})
    return {"blind_clips": len(public), "packet_sha256": packet_hash, "two_reviewers_required": True}


def validate_review(r, packet_hash):
    if r.get("packet_sha256") != packet_hash or not r.get("reviewer_id"):
        raise ValueError("Review packet hash or reviewer missing")
    label = r.get("current_window_visual_label")
    if label not in ("normal", "anomalous", "uncertain", "unobservable"):
        raise ValueError("Pending/invalid visual label")
    for field in ("direct_mechanism", "normal_mechanism"):
        if not isinstance(r.get(field), str) or not r[field].strip() or r[field] == "pending":
            raise ValueError("Pending mechanism: use explicit none/unknown if appropriate")
    for field in ("same_actor_support", "same_time_support"):
        if r.get(field) not in ("supported", "contradicted", "unknown"):
            raise ValueError("Invalid " + field)
    if r.get("normal_explains_suspicious_action") not in ("yes", "partly", "no", "unknown"):
        raise ValueError("Invalid normal explanation")
    if r.get("unexplained_anomalous_mechanism") not in ("yes", "no", "unknown"):
        raise ValueError("Invalid residual observation")
    if r.get("review_confidence") not in ("high", "medium", "low"):
        raise ValueError("Invalid confidence")
    if r.get("context_review", {}).get("audio_used") or r.get("context_review", {}).get("context_used"):
        raise ValueError("Primary visual judgment must not use outside context/audio")
    if not isinstance(r.get("event_classes"), list) or any(c not in ("B1", "B2", "B4", "B5", "B6", "G", "unknown") for c in r["event_classes"]):
        raise ValueError("Invalid event classes")
    target = continuous_target(r.get("visible_event_intervals_local", []))
    if label == "normal" and r.get("visible_event_intervals_local"):
        raise ValueError("Normal label conflicts with event intervals")
    # Short visible anomaly intervals are reported but not silently made normal.
    return 0 if label == "normal" else 1 if label == "anomalous" and target == 1 else None


def import_reviews(out):
    public = read_json(out / "review/public_manifest.json")
    packet_hash = semantic_sha256(public)
    expected = {r["blind_id"] for r in public}
    reviews, errors = {}, []
    files = sorted((out / "review/returns").glob("*.jsonl"))
    for path in files:
        rows = list(iter_jsonl(path))
        for r in rows:
            try:
                label = validate_review(r, packet_hash)
                if r.get("blind_id") not in expected:
                    raise ValueError("Unknown blind ID")
                rid = r["reviewer_id"]
                if rid in reviews.setdefault(r["blind_id"], {}):
                    raise ValueError("Duplicate reviewer/case record")
                reviews[r["blind_id"]][rid] = {"raw": r, "target": label}
            except ValueError as exc:
                errors.append({"file": path.name, "blind_id": r.get("blind_id"), "reason": str(exc)})
    adjudications = {r["blind_id"]: r for r in iter_jsonl(out / "review/adjudications.jsonl")} if (out / "review/adjudications.jsonl").exists() else {}
    accepted, todo = {}, []
    for blind in sorted(expected):
        pair = reviews.get(blind, {})
        if len(pair) < 2:
            errors.append({"blind_id": blind, "reason": "two independent completed reviews required"})
            continue
        targets = {r["target"] for r in pair.values()}
        labels = {r["raw"]["current_window_visual_label"] for r in pair.values()}
        semantic_fields = ("current_window_visual_label", "visible_event_intervals_local", "event_classes")
        signatures = {semantic_sha256({k: r["raw"][k] for k in semantic_fields}) for r in pair.values()}
        pair_hash = semantic_sha256({rid: r["raw"] for rid, r in sorted(pair.items())})
        chosen = next(iter(pair.values()))
        if len(targets) != 1 or len(labels) != 1 or len(signatures) != 1:
            adj = adjudications.get(blind)
            if not adj or adj.get("review_pair_sha256") != pair_hash or adj.get("reviewer_id") in pair:
                todo.append({"blind_id": blind, "review_pair_sha256": pair_hash, "packet_sha256": packet_hash,
                             "reviewer_id": "R3", "reason": "adjudicate label/interval/class disagreements with the full clip"})
                continue
            chosen = {"raw": adj, "target": validate_review(adj, packet_hash)}
        raw = chosen["raw"]
        accepted[blind] = {"target": chosen["target"], "visual_label": raw["current_window_visual_label"],
                           "event_classes": raw["event_classes"], "review_pair_sha256": pair_hash,
                           "hard_normal_semantics_confirmed": all(r["raw"]["current_window_visual_label"] == "normal" and
                                r["raw"]["normal_explains_suspicious_action"] in ("yes", "partly") and
                                r["raw"]["normal_mechanism"].lower() not in ("none", "unknown") for r in pair.values())}
    write_jsonl(out / "review/adjudication_required.jsonl", todo)
    write_json(out / "review/import_summary.json", {"expected": len(expected), "accepted": len(accepted),
               "errors": errors, "unresolved": len(todo), "ready": not errors and not todo,
               "use_policy": "locked_evaluation_only; adaptation audit_only"})
    if errors or todo:
        raise ValueError("WAITING_FOR_HUMAN_REVIEW: review/import_summary.json and adjudication_required.jsonl")
    return accepted


def metric(rows, name, mode):
    usable = [r for r in rows if r["prediction"][mode].get(name, {}).get("score") is not None]
    values = [r["prediction"][mode][name] for r in usable]
    result = metrics_with_predictions([r["y"] for r in usable], [v["score"] for v in values], [v["prediction"] for v in values])
    result.update(enrolled_or_slice_n=len(rows), score_coverage=len(usable) / len(rows) if rows else None,
                  abstentions=len(rows) - len(usable), fpr=result["fp"] / result["negative_n"] if result["negative_n"] else None)
    result["threshold"] = "frozen_family_threshold_with_recorded_fallback"
    return result


def comparison(rows, base, candidate, mode, config):
    paired = [r for r in rows if all(r["prediction"][mode].get(k, {}).get("score") is not None for k in (base, candidate))]
    a, b = metric(paired, base, mode), metric(paired, candidate, mode)
    helps = sum(r["prediction"][mode][base]["prediction"] != r["y"] == r["prediction"][mode][candidate]["prediction"] for r in paired)
    hurts = sum(r["prediction"][mode][base]["prediction"] == r["y"] != r["prediction"][mode][candidate]["prediction"] for r in paired)
    boot_rows = [{"source_group": r["source_group"], "y": r["y"], "base": r["prediction"][mode][base]["score"],
                 "candidate": r["prediction"][mode][candidate]["score"], "base_pred": r["prediction"][mode][base]["prediction"],
                 "candidate_pred": r["prediction"][mode][candidate]["prediction"]} for r in paired]
    boot = grouped_bootstrap_delta(boot_rows, "base", "candidate", config["bootstrap_samples"], config["seed"])
    boot["p_positive_interpretation"] = "bootstrap positive fraction, NOT a hypothesis-test p-value"
    slices = {}
    for name in ("hard_normal", "easy_label_A", "B1", "B4", "context_unverified", "other_class"):
        subset = [r for r in paired if r["stratum"] == "hard_label_A" and r.get("hard_normal_semantics_confirmed") and r["y"] == 0] if name == "hard_normal" else []
        if name in ("B1", "B4"):
            subset = [r for r in paired if name in r.get("event_classes", []) and r["y"] == 1]
        elif name in ("easy_label_A", "context_unverified"):
            subset = [r for r in paired if r["stratum"] == name]
        elif name == "other_class":
            subset = [r for r in paired if set(r.get("event_classes", [])) & {"B2", "B5", "B6", "G"}]
        slices[name] = {"status": "EVALUATED" if subset else "NOT_EVALUATED", "base": metric(subset, base, mode), "candidate": metric(subset, candidate, mode)}
    ap_delta = b["ap"] - a["ap"] if a["ap"] is not None and b["ap"] is not None else None
    ba_delta = b["balanced_accuracy"] - a["balanced_accuracy"] if a["balanced_accuracy"] is not None and b["balanced_accuracy"] is not None else None
    risk = slices["hard_normal"]
    safety = risk["base"]["fpr"] is not None and risk["candidate"]["fpr"] <= risk["base"]["fpr"]
    for code in ("B1", "B4"):
        s = slices[code]
        safety = safety and s["base"]["recall"] is not None and s["candidate"]["recall"] >= s["base"]["recall"]
    if ap_delta is None or ba_delta is None:
        verdict = "INCOMPLETE_EVIDENCE"
    elif ap_delta <= 0 or helps <= hurts or not safety:
        verdict = "NO_DIRECTIONAL_GAIN"
    else:
        ci = boot.get("ap_delta", {}).get("ci95", [None, None])
        verdict = "POSITIVE_SCOPED_PILOT" if ci[0] is not None and ci[0] > 0 and ba_delta >= 0 else "PROMISING_BUT_INCONCLUSIVE"
    return {"base": a, "candidate": b, "paired_n": len(paired), "helps": helps, "hurts": hurts,
            "input_cohort_n": len(rows), "paired_coverage": len(paired) / len(rows) if rows else None,
            "unpaired_full_cohort_coverage": {"base": metric(rows, base, mode), "candidate": metric(rows, candidate, mode)},
            "ap_delta": ap_delta, "balanced_accuracy_delta": ba_delta, "bootstrap": boot, "slices": slices,
            "safety_slices_established_nonworse": bool(safety), "verdict": verdict, "deployment_authorized": False}


def cost_report(out):
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    logical = [read_json(p) for p in (out / "cost/logical_requests").glob("*.json")]
    times = [r["latency_seconds"] for r in attempts if "latency_seconds" in r]
    return {"logical_requests_attempted": len({r["request_key"] for r in attempts}), "physical_attempts": len(attempts),
            "status_counts": dict(Counter(r["status"] for r in attempts)), "stage_counts": dict(Counter(r["stage"] for r in attempts)),
            "latency_seconds_p50": float(np.quantile(times, .5)) if times else None,
            "latency_seconds_p95": float(np.quantile(times, .95)) if times else None,
            "usage_reported_attempts": sum(r.get("usage") is not None for r in attempts),
            "logical_cache_hits": sum(r["cache_hit"] for r in logical),
            "local_parser_repairs": sum(r.get("local_parser_repair", False) for r in logical),
            "valid_parser_receipts": len(logical), "remote_schema_repair_receipts": sum(r["remote_schema_repair_index"] > 0 for r in logical),
            "usage_records": [r.get("usage") for r in attempts], "latency_not_local_8B_benchmark": True}


def evaluate(out, config):
    if not at_least(out, "PREDICTIONS_COMMITTED"):
        raise ValueError("Commit predictions before reading any evaluation labels")
    commit = read_json(out / "predictions/commit.json")
    verify_manifest(commit["files"])
    if (out / "evaluation/report.json").exists():
        report = read_json(out / "evaluation/report.json")
        verify_manifest(report["review_files"])
        return report
    reviews = import_reviews(out)
    private = read_json(out / "review/private_map.json")
    by_uid = {private[bid]["window_uid"]: value for bid, value in reviews.items()}
    manifest = {r["window_uid"]: r for r in iter_jsonl(out / "enrollment/windows.jsonl") if r["role"] == "locked_evaluation"}
    predictions = list(iter_jsonl(out / "predictions/predictions_committed.jsonl"))
    if {r["window_uid"] for r in predictions} != set(manifest):
        raise ValueError("Prediction/locked manifest coverage mismatch")
    primary, weak = [], []
    for pred in predictions:
        uid = pred["window_uid"]
        row = manifest[uid]
        human = by_uid[uid]
        common = {"window_uid": uid, "source_group": row["source_group"], "stratum": row["stratum"],
                  "event_classes": human["event_classes"], "hard_normal_semantics_confirmed": human["hard_normal_semantics_confirmed"], "prediction": pred}
        if human["target"] is not None:
            primary.append({**common, "y": human["target"]})
        if row["weak_target"] is not None:
            weak.append({**common, "y": row["weak_target"]})
    review_files = sorted((out / "review/returns").glob("*.jsonl"))
    if (out / "review/adjudications.jsonl").exists():
        review_files.append(out / "review/adjudications.jsonl")
    unblind = {"commit_sha256": stable_hash(out / "predictions/commit.json"),
               "review_files": [{"path": str(p.resolve()), "sha256": stable_hash(p)} for p in review_files]}
    immutable(out / "evaluation/unblind.json", unblind)
    advance(out, "EVALUATION_UNBLINDED")
    comparisons = {}
    pairs = [("T0_M0", "T1_DIRECT2", "core"), ("B0_C1_ONLY", "B1_SAME_EVENT", "binding"),
             ("T1_DIRECT2", "T2_EVENT6", "core"), ("B_BAG", "B1_SAME_EVENT", "binding")]
    for view, rows in (("adjudicated_human_primary", primary), ("weak_dataset_secondary", weak)):
        comparisons[view] = {}
        for base, candidate, cohort in pairs:
            direct_rows = [r for r in rows if r["prediction"][cohort + "_complete"]]
            comparisons[view][candidate + "_vs_" + base] = {
                "common_complete": comparison(direct_rows, base, candidate, "direct", config),
                "full_manifest_fallback": comparison(rows, base, candidate, "full_manifest", config)}
            comparisons[view][candidate + "_vs_" + base]["common_complete"]["feature_coverage_of_labeled_view"] = len(direct_rows) / len(rows) if rows else None
    report = {"version": "v912_scoped_pilot_report1", "enrolled_locked_n": len(predictions),
              "synthetic_mock_only": config.get("mock", False),
              "primary_human_label_n": len(primary), "primary_label_coverage": len(primary) / len(predictions) if predictions else 0,
              "uncertain_or_unobservable_or_sub8_anomaly_n": len(predictions) - len(primary),
              "comparisons": comparisons, "cost": cost_report(out), "review_files": unblind["review_files"],
              "operational_separately_fitted_baselines": {name: metric(primary, name, "direct") for name in ("T0_BASE_OPERATIONAL", "B0_C1_OPERATIONAL")},
              "claim_limit": "B1/B4 hard-normal train-source pilot; not six-class benchmark or deployment authorization",
              "deployment_authorized": False, "branch_closed": True}
    if report["primary_label_coverage"] < config["minimum_primary_human_coverage"]:
        report["conclusion"] = "INCOMPLETE_EVIDENCE"
    else:
        t = comparisons["adjudicated_human_primary"]["T1_DIRECT2_vs_T0_M0"]["common_complete"]["verdict"]
        b = comparisons["adjudicated_human_primary"]["B1_SAME_EVENT_vs_B0_C1_ONLY"]["common_complete"]["verdict"]
        control = comparisons["adjudicated_human_primary"]["B1_SAME_EVENT_vs_B_BAG"]["common_complete"]["verdict"]
        report["binding_conclusion"] = b if control in ("POSITIVE_SCOPED_PILOT", "PROMISING_BUT_INCONCLUSIVE") else "BINDING_INCREMENT_NOT_ESTABLISHED"
        report["conclusion"] = "NO_VALIDATED_INCREMENT" if t == b == "NO_DIRECTIONAL_GAIN" else "SCOPED_RESULTS_REQUIRE_REPLICATION"
    write_json(out / "evaluation/report.json", report)
    lines = ["# V9.12 B1/B4 Scoped Pilot", "", report["claim_limit"], "", "Conclusion: " + report["conclusion"],
             f"Primary human labels: {len(primary)}/{len(predictions)}. Unknown labels are excluded, not normal.", "",
             "| Contrast | Cohort | N | AP delta | BA delta | Helps | Hurts | Verdict |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, result in comparisons["adjudicated_human_primary"].items():
        for cohort, r in result.items():
            lines.append(f'| {name} | {cohort} | {r["paired_n"]} | {r["ap_delta"]} | {r["balanced_accuracy_delta"]} | {r["helps"]} | {r["hurts"]} | {r["verdict"]} |')
    report_md = (out / "evaluation/REPORT_SYNTHETIC.md") if config.get("mock") else (
        Path(__file__).resolve().parents[3] / "docs" / ("GOVERNED_V912_" + out.name + "_RESULTS.md"))
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "evaluation/index.html").write_text('<!doctype html><html><meta charset="utf-8"><title>V9.12 Results</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V9.12 Results</h1><pre>' + html.escape("\n".join(lines)) + '</pre></html>', encoding="utf-8")
    advance(out, "REPORTED"); advance(out, "BRANCH_CLOSED")
    return report
