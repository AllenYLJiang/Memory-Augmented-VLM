"""Read-only review interpretation and prospective single-C2 comparison."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from . import claim_scope_v915 as trial
from .contracts import read_json, write_json, semantic_sha256
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest
from .role_scoped import portable

VERSION = "v915_review_followup_snapshot_1"
LIMITS = {
    "hockey_local_benignity": "A categorical violence judgment flags a disagreement with benign-play claims; it is not a benchmark label or a proof of sport-rule legality. Preserve the cited local action evidence.",
    "fire_person_binding": "Stationarity alone does not logically exclude a person. Keep the reviewer's contradicted judgment as attributed feedback; do not turn this inference into a proven object identity or an anomaly/normal label.",
    "vehicle_display_control": "Uncertain display evidence does not establish that automatic control is on or off, and does not establish safety or demonstration intent.",
}
COMPARISON_POLICY = {
    "version": VERSION,
    "scope": "Single previously missing hockey C2; claim-level comparison, never accuracy or visual verification.",
    "native_validity": "Unchanged validate_native and raw/parent/receipt hashes are checked first.",
    "accepted_explanation": "At least one known event has bound > 0 under the existing diagnostic binding rule.",
    "explicit_assertion": "visible_normal_mechanism=true with visible_benign_activity and remains=no, reported even when binding is unknown.",
    "absence": "No accepted explanation is not proof of harmfulness or that C1 is true.",
    "review": "Supported/contradicted/uncertain remain attributed diagnostic assessments, not gold labels.",
    "selection": "All terminal outcomes are reported; no outcome-dependent retries or preferred-response selection.",
}


def compare(c1, result, review_assessment):
    outcome = {"model_status": result["status"] if result else "not_requested",
               "review_assessment": review_assessment, "model_explanation_state": "not_available",
               "comparison": "NOT_YET_OBSERVED", "diagnostic_conditional_not_anomaly_score": None,
               "explicit_benign_assertions": [], "visual_truth_verified": False,
               "anomaly_label": None, "acceptance_or_deployment": False}
    if not result or result["status"] != "valid":
        if result:
            outcome.update(comparison="TECHNICAL_FAILURE_NO_SEMANTIC_CONCLUSION", reason=result.get("reason"))
        return outcome
    c2 = result["parsed"]
    trial.prior.validate_native(c1, c2)
    diagnostic = trial.prior.diagnostic_binding(c1, c2)
    if semantic_sha256(diagnostic) != semantic_sha256(result["diagnostic_conditional_not_anomaly_score"]):
        raise ValueError("stored C2 diagnostic differs from unchanged replay")
    assertions = [{"event_id": row["event_id"], "explanation": ex}
                  for row in c2["event_bindings"] for ex in row["explanations"]
                  if ex["visible_normal_mechanism"] and ex["benignity_basis"] == "visible_benign_activity"
                  and ex["unexplained_anomalous_mechanism_remains"] == "no"]
    accepted = any(r["known"] and r["bound"] is not None and r["bound"] > 0 for r in diagnostic["events"])
    state = "accepted_benign_binding" if accepted else "binding_uncertain" if not diagnostic["observed"] else "no_accepted_benign_binding"
    if review_assessment == "uncertain":
        relationship = "HUMAN_UNCERTAIN_NO_CORRECTNESS_TARGET"
    elif state == "binding_uncertain":
        relationship = "MODEL_UNCERTAIN_NO_ALIGNMENT_CLAIM"
    elif state == "accepted_benign_binding":
        relationship = "CLAIM_DISAGREEMENT_NOT_MEASURED_ERROR" if review_assessment == "contradicted" else "CLAIM_AGREEMENT_NOT_VISUAL_VERIFICATION"
    else:
        relationship = "NO_BENIGN_VETO_CONSISTENT_WITH_REVIEW_NOT_PROOF_OF_ANOMALY" if review_assessment == "contradicted" else "NO_ACCEPTED_BINDING_DESPITE_SUPPORTED_REVIEW"
    outcome.update(model_explanation_state=state, comparison=relationship,
                   diagnostic_conditional_not_anomaly_score=diagnostic,
                   explicit_benign_assertions=assertions,
                   explicit_assertion_conflicts_with_contradicted_review=bool(assertions) and review_assessment == "contradicted")
    return outcome


def source_snapshot(source):
    if (source / ".operation.lock").exists():
        raise ValueError("source operation still running; inspect only after it finishes")
    return trial.audit.snapshot([source])


def check_out(source, out):
    protocol = read_json(source / "protocol.json")
    roots = [source] + [portable(p).resolve() for p in protocol["source_roots"]]
    for root in roots:
        trial.audit.check_output(root, out, root)


def verify_result_payload(source, result):
    if not result or result["status"] != "valid":
        return
    files = result.get("raw_manifest", [])
    if len(files) != 1:
        raise ValueError("single C2 requires one raw response")
    raw_path = portable(files[0]["path"]).resolve()
    if raw_path.parent != (source / "cache/raw").resolve():
        raise ValueError("raw path outside the prepared run")
    raw = read_json(raw_path)
    receipt_path = portable(raw["receipt"]).resolve()
    if receipt_path.parent != (source / "cost/attempts").resolve():
        raise ValueError("physical receipt outside the prepared run")
    receipt = read_json(receipt_path)
    c1, media, request, config = (read_json(source / f) for f in ("parent_C1.json", "media.json", "request.json", "config.json"))
    identity = {"version": trial.VERSION, "stage": trial.STAGE, "window_uid": c1["window_id"],
                "config": config, "parent_sha256": semantic_sha256(c1), "images": media["image_sha256"],
                "frame_indices": media["frame_indices"], "prompt": request["prompt"]}
    if (raw["identity"] != identity or semantic_sha256(identity) != raw_path.stem
            or receipt.get("raw_file_sha256") != stable_hash(raw_path)
            or receipt.get("request_key") != raw_path.stem or receipt.get("status") != "success"):
        raise ValueError("C2 raw/physical receipt identity mismatch")
    if semantic_sha256(trial.audit.parse_raw(raw["raw"])) != semantic_sha256(result["parsed"]):
        raise ValueError("saved parsed C2 differs from raw response")


def build(project, source, out):
    source, out = source.resolve(), out.resolve()
    check_out(source, out)
    protocol = trial.verify(source)
    review = trial.accepted_review(source)
    before = source_snapshot(source)
    immutable(out / "study.json", {"version": VERSION, "source_run": trial.prior.canonical(source)})
    code_paths = [Path(__file__), project / "tools/claim_scope_followup_cli.py", project / "run_claim_scope_followup.sh"]
    code = [{"path": trial.prior.canonical(p), "sha256": stable_hash(p)} for p in code_paths]
    identity = {"version": VERSION, "source_snapshot": before, "implementation": code, "comparison_policy": COMPARISON_POLICY}
    identifier = semantic_sha256(identity)
    snap = out / "snapshots" / identifier[:20]
    if (snap / "completion.json").exists():
        completion = read_json(snap / "completion.json")
        if (completion["identity_sha256"] != identifier
                or completion["output_manifest_sha256"] != stable_hash(snap / "output_manifest.json")):
            raise ValueError("followup snapshot completion changed")
        verify_manifest(read_json(snap / "output_manifest.json"))
        summary = read_json(snap / "summary.json")
    else:
        immutable(snap / "input_identity.json", identity)
        result = trial.get_result(source)
        verify_result_payload(source, result)
        c1 = read_json(source / "parent_C1.json")
        packet = read_json(source / "review/packet.json")
        reviewed = {r["claim_id"]: r for r in review["claims"]}
        hockey = compare(c1, result, reviewed["hockey_local_benignity"]["assessment"])
        rows = []
        for card in packet["cards"]:
            raw_review = reviewed[card["claim_id"]]
            rows.append({"claim_id": card["claim_id"], "window_uid": card["window_uid"],
                "claim": card["claim"], "review": raw_review,
                "review_assessment_available": raw_review["assessment"] != "uncertain",
                "reviewer_support_value": {"supported": True, "contradicted": False, "uncertain": None}[raw_review["assessment"]],
                "visual_truth_established_by_import": False, "training_label": None,
                "interpretation_limit": LIMITS[card["claim_id"]],
                "frame_indices": card["frame_indices"], "image_paths": card["image_paths"],
                "new_model_request": card["window_uid"] == protocol["window_uid"]})
        attempts = [read_json(p) for p in (source / "cost/attempts").glob("*.json")]
        logical = [read_json(p) for p in (source / "cost/logical_requests").glob("*.json")]
        cost = trial.prior.costs(attempts, logical)
        saved_summary = read_json(source / "summary.json")
        consistent = (saved_summary.get("claim_reviews_imported") is True
                      and saved_summary.get("single_C2_status") == hockey["model_status"]
                      and saved_summary.get("cost", {}).get("physical_attempts") == cost["physical_attempts"])
        authorization = read_json(source / "authorizations/missing_C2.json")
        state = "EXPLICIT_BUDGET_THEN_ONE_PREPARED_C2" if not result else "INSPECT_C2_TECHNICAL_FAILURE_NO_REBUY" if result["status"] != "valid" else "INSPECT_CLAIM_COMPARISON_NO_EXPANSION"
        summary = {"version": VERSION, "snapshot": identifier[:20], "reviewer_id": review["reviewer_id"],
            "review_receipt_verified": True, "overlay_matches_imported_review": True,
            "imported_claims": len(rows), "assessments": {r["claim_id"]: r["review"]["assessment"] for r in rows},
            "saved_summary_consistent": consistent, "C2_status": hockey["model_status"],
            "claim_level_comparison": hockey["comparison"], "next": state,
            "new_review_task_required": False, "current_authorization_present": authorization.get("authorized") is True,
            "reporting_calls": 0, "source_request_cost": cost,
            "new_scores_written_to_original": False, "formal_accuracy": None, "formal_AP": None,
            "larger_acquisition_authorized": False, "graph_OT_changes_authorized": False,
            "training_authorized": False, "complete_four_case_comparison": False,
            "original_run_files_verified_unchanged": len(before)}
        immutable(snap / "review_interpretation.json", {"version": VERSION, "claims": rows,
            "review_sha256": semantic_sha256(review), "use_policy": trial.USE,
            "observed_in_old_overlay_means_categorical_review_not_direct_visual_proof": True})
        immutable(snap / "comparison_policy.json", COMPARISON_POLICY)
        immutable(snap / "hockey_C2_comparison.json", hockey)
        immutable(snap / "summary.json", summary)
        render(snap, source, rows, hockey, summary)
        if source_snapshot(source) != before:
            raise ValueError("source changed during report; do not use incomplete snapshot")
        outputs = [{"path": trial.prior.canonical(p), "sha256": stable_hash(p)} for p in sorted(snap.iterdir()) if p.is_file()]
        immutable(snap / "output_manifest.json", outputs)
        immutable(snap / "completion.json", {"identity_sha256": identifier,
                  "output_manifest_sha256": stable_hash(snap / "output_manifest.json")})
    if source_snapshot(source) != before:
        raise ValueError("source changed during verification")
    write_json(out / "latest.json", {"snapshot": identifier[:20], "summary": f"snapshots/{identifier[:20]}/summary.json"})
    write_json(out / "summary.json", summary)
    (out / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=snapshots/{identifier[:20]}/index.html"><a href="snapshots/{identifier[:20]}/index.html">Latest review and C2 follow-up</a>', encoding="utf-8")
    return summary


def render(out, source, rows, hockey, summary):
    esc = lambda value: html.escape(str(value))
    def href(path):
        try:
            return quote(Path(os.path.relpath(path, out)).as_posix(), safe="/")
        except ValueError:
            return Path(path).as_uri()
    blocks = []
    for row in rows:
        review = row["review"]
        frames = ''.join(f'<figure><img loading="lazy" src="{href(portable(path))}" alt="T{k}"><figcaption>T{k} / frame {row["frame_indices"][k]}</figcaption></figure>' for k, path in enumerate(row["image_paths"]))
        blocks.append(f'<section><h2>{esc(row["claim_id"])}</h2><p>{esc(row["claim"])}</p><p><b>Reviewer {esc(summary["reviewer_id"])}:</b> {esc(review["assessment"])} | cited frames: {esc(", ".join(review["frame_ids"]))}</p><blockquote>{esc(review["notes"])}</blockquote><p class="limit">{esc(row["interpretation_limit"])}</p><div class="frames">{frames}</div></section>')
    data = esc(json.dumps(hockey, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.15 Review and C2 Follow-up</title><style>
body{margin:0;background:#fcfcfc;color:#202a25;font:15px system-ui}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:20px;overflow-wrap:anywhere}section{border-top:1px solid #b6c5be;padding:20px 0}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}blockquote{margin:12px 0;padding:12px;background:#eef4ef}.limit{border-left:4px solid #b65a36;padding:12px;background:#fff1eb}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff2f2;padding:12px}a{color:#116b5f}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><main><h1>V9.15 Review and C2 Follow-up</h1>'''
    page += f'<p>Review receipt verified. C2: <b>{esc(summary["C2_status"])}</b>. Physical attempts: {summary["source_request_cost"]["physical_attempts"]}.</p><p class="limit">Attributed human judgments are not gold labels. Unknown is not normal. No graph/OT change, training or expansion approval.</p><p><a href="summary.json">Snapshot summary</a> | <a href="review_interpretation.json">Review interpretation</a> | <a href="hockey_C2_comparison.json">C2 comparison</a></p>'
    page += ''.join(blocks) + f'<section><h2>Single hockey dependency</h2><p>{esc(hockey["comparison"])}</p><pre>{data}</pre><a href="{href(source / "request.json")}">Frozen request and parent hash</a></section></main></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
