#!/usr/bin/env python3
"""Zero-API smoke audit, native C2 benignity draft, four-case diagnostic review.

Never replaces acquired records, scores, training targets, protocol or budgets.
There is deliberately no provider execution entry point in this utility.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import html
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from event_decision.contracts import read_json, iter_jsonl, semantic_sha256, write_json
from event_decision.role_scoped import portable
from event_decision.binding import binding_prompt, validate_binding, validate_proposal
from event_decision.b1b4_trial.features import feature_record, NAMES
from event_decision.b1b4_trial.protocol import stable_hash, verify_manifest, run_lock, immutable, now
import v912_prepared_run_handoff as prepared

VERSION = "v912_smoke_semantic_audit_1"
SCHEMA = "same_event_benign_binding_v2_draft1"
TYPES = {"structured_sport_or_play", "peaceful_collective_activity", "assistance_or_rescue",
         "ordinary_nonharmful_activity", "visibly_controlled_performance", "other_visible_benign_activity", "unresolved"}
BASES = {"visible_benign_activity", "natural_physical_cause_only", "external_context_only", "unknown"}
REMAINS = {"yes", "no", "unknown"}


def draft_contract():
    return {"schema_version": SCHEMA, "status": "DRAFT_NOT_DEPLOYED",
            "type_allowed": sorted(TYPES), "benignity_basis_allowed": sorted(BASES),
            "unexplained_anomalous_mechanism_remains_allowed": sorted(REMAINS),
            "new_fields_per_explanation": ["benignity_basis", "unexplained_anomalous_mechanism_remains"],
            "old_relation_reference_checks_retained": True, "automatic_legacy_migration": False,
            "unknown_is_not_normal": True, "new_scores_authorized": False}


def native_prompt(c1):
    validate_proposal(c1)
    prefix = f'''This is a diagnostic contract recheck, schema {SCHEMA}.
NORMAL means a visibly BENIGN, NON-ANOMALOUS explanation for this exact event.
It does NOT mean natural, physically lawful, common, or causally explainable.
Combustion explains flames physically but does not make a burning person benign.
Gas dispersion, momentum, impact, gravity, injury mechanisms, or a weapon operating
normally do not establish benignity. Do not relabel them as ordinary activity.
Assistance may explain a rescuer's action but does not erase a separate injury or attack.
Sports context alone does not establish that an act is ordinary play; require local evidence.
Do not assume a performance, filming, automated driving, consent, harmlessness or intent
from a title or an off-screen story. Uncertain appearance stays unknown.
Retain the frozen C1 events and evidence, including any questionable proposals.
Use existing mechanism, reason and normal_evidence fields for observable justification.
For each explanation, type must be one of {sorted(TYPES)}.
Add benignity_basis, one of {sorted(BASES)}; and
unexplained_anomalous_mechanism_remains, one of {sorted(REMAINS)}.
visible_normal_mechanism may be true ONLY when type is not unresolved,
benignity_basis=visible_benign_activity, and unexplained_anomalous_mechanism_remains=no.
This is necessary but not sufficient: all original participant, time, event and
coverage requirements still apply. Causal explanation alone is never a veto.
If no benign explanation applies, return the event with an empty explanations list.
Do not manufacture a benign explanation to fill the schema. Unknown means unknown.
The following original reference contract has the new version name. All fields remain
required, with the two additional fields and stricter type vocabulary above.
'''
    return prefix + binding_prompt(c1).replace("schema_version=same_event_normal_binding_v1", "schema_version=" + SCHEMA)


def validate_native(c1, c2):
    if not isinstance(c2, dict) or c2.get("schema_version") != SCHEMA:
        raise ValueError("native v2 response required; do not silently migrate legacy records")
    # Validate the shared reference structure on a copy, never mutate raw or stored scores.
    validate_binding(c1, {**c2, "schema_version": "same_event_normal_binding_v1"})
    for row in c2["event_bindings"]:
        for ex in row["explanations"]:
            if ex.get("type") not in TYPES or ex.get("benignity_basis") not in BASES or ex.get("unexplained_anomalous_mechanism_remains") not in REMAINS:
                raise ValueError("missing/invalid native benignity fields")
            if ex["visible_normal_mechanism"] and (ex["type"] == "unresolved" or ex["benignity_basis"] != "visible_benign_activity" or ex["unexplained_anomalous_mechanism_remains"] != "no"):
                raise ValueError("non-benign or unresolved cause cannot assert visible benign mechanism")
    return c2


def costs(attempts, logical):
    import numpy as np
    lat = [a["latency_seconds"] for a in attempts if isinstance(a.get("latency_seconds"), (int, float))]
    starts = [datetime.fromisoformat(a["started_at"]) for a in attempts]
    ends = [datetime.fromisoformat(a["started_at"]) + timedelta(seconds=a.get("latency_seconds", 0)) for a in attempts]
    total = {k:sum((a.get("usage") or {}).get(k, 0) for a in attempts) for k in ("input_tokens", "output_tokens", "image_tokens", "total_tokens")}
    reasoning = sum((a.get("usage") or {}).get("output_tokens_details", {}).get("reasoning_tokens", 0) for a in attempts)
    return {"physical_attempts": len(attempts), "statuses": dict(Counter(a["status"] for a in attempts)),
            "stage_families": dict(Counter(a["stage"].split("/")[0] for a in attempts)),
            "logical_receipts": len(logical), "unique_valid_request_keys": len({a["request_key"] for a in logical}),
            "remote_format_repairs": sum(a["remote_schema_repair_index"] > 0 for a in logical),
            "local_repairs": sum(a["local_parser_repair"] for a in logical), "cache_hits": sum(a["cache_hit"] for a in logical),
            "known_usage": total, "reasoning_tokens": reasoning,
            "reasoning_fraction_of_known_output": reasoning / total["output_tokens"] if total["output_tokens"] else None,
            "missing_usage_receipts": sum(a.get("usage") is None for a in attempts),
            "billing_unknown_receipts": sum(a.get("billing_unknown", False) for a in attempts),
            "in_flight": sum(a["status"] == "in_flight" for a in attempts),
            "first_request": min(starts).isoformat() if starts else None,
            "last_response": max(ends).isoformat() if ends else None,
            "request_span_seconds": (max(ends)-min(starts)).total_seconds() if starts else None,
            "sum_request_latency_seconds": sum(lat),
            "request_latency_p50_p95_seconds": np.quantile(lat, [.5,.95]).tolist() if lat else None,
            "failures": [a for a in attempts if a["status"] != "success"],
            "not_a_currency_bill": True, "unknown_usage_is_not_zero_cost": True}


def format_repair_changes(out):
    from event_decision.b1b4_trial.evidence import parse_local
    grouped = {}
    for path in (out / "cache/raw").glob("*.json"):
        value = read_json(path)
        identity = value["identity"]
        if identity["stage"] == "C0":
            parsed, _ = parse_local(value["raw"])
            grouped.setdefault(identity["window_uid"], {})[identity["repair"]] = parsed
    result = []
    for uid, pair in grouped.items():
        if 0 not in pair or 1 not in pair:
            continue
        before, after = pair[0], pair[1]
        keys = ("current_window_active_occupancy_probability", "current_window_normal_confound_probability", "uncertainty")
        changes = {k: {"before": before.get(k), "after": after.get(k)} for k in keys if before.get(k) != after.get(k)}
        for phase in ("active", "aftermath"):
            for node, item in before.get("phase_nodes", {}).get(phase, {}).items():
                other = after.get("phase_nodes", {}).get(phase, {}).get(node, {})
                for field in ("presence_probability", "uncertainty", "null_probability"):
                    if item.get(field) != other.get(field):
                        changes[f"{phase}/{node}/{field}"] = {"before": item.get(field), "after": other.get(field)}
        result.append({"window_uid": uid, "scalar_changes_during_format_repair": changes,
                       "not_a_controlled_semantic_improvement": True})
    return result


def diagnostic_rows(out):
    smoke_ids = [r["window_uid"] for r in iter_jsonl(out / "enrollment/smoke_manifest.jsonl")]
    if len(set(smoke_ids)) != 4:
        raise ValueError("expected the entire fixed four-window smoke")
    enrollment = {r["window_uid"]:r for r in iter_jsonl(out / "enrollment/windows.jsonl")}
    media = read_json(out / "media/manifest.json")
    rows, manifests = [], []
    for uid in smoke_ids:
        if enrollment[uid]["role"] != "adaptation":
            raise ValueError("diagnostic utility must not open locked outcomes")
        path = out / "private_acquisition/adaptation" / (uid + ".json")
        value = read_json(path)
        if value.get("window_uid") != uid or value.get("human_layers_loaded") is not False:
            raise ValueError("result identity or human-overlay scope invalid")
        verify_manifest(value["raw_request_manifest"])
        validate_proposal(value["C1"], window_id=uid, evidence_signature=semantic_sha256(media[uid]["image_sha256"]))
        validate_binding(value["C1"], value["C2"])
        f = feature_record(uid, value)
        manifests.extend(value["raw_request_manifest"])
        manifests.append({"path": str(path.resolve()), "sha256": stable_hash(path)})
        for file, sha in media[uid]["files"].items():
            manifests.append({"path": str((out / "media" / uid / file).resolve()), "sha256": sha})
        rows.append({"window_uid": uid, "role": "adaptation", "stratum": enrollment[uid]["stratum"],
                     "video_id": enrollment[uid]["video_id"], "frame_start": enrollment[uid]["start_frame"],
                     "frame_end_exclusive": enrollment[uid]["end_frame_exclusive"],
                     "weak_target_audit_only": enrollment[uid]["weak_target"],
                     "values": f["values"], "errors": value["errors"], "binding_trace": f["binding_trace"],
                     "C0": value["C0"], "C1": value["C1"], "C2": value["C2"]})
    verify_manifest(manifests)
    return rows, manifests


def immutable_inputs(path, inputs):
    # A Windows/WSL spelling change is not a new source; retain original bytes/hash.
    if path.exists():
        existing = read_json(path)
        verify_manifest(existing)
        identity = lambda items: Counter((str(portable(r["path"]).resolve()), r["sha256"]) for r in items)
        if identity(existing) != identity(inputs):
            raise ValueError("Frozen diagnostic inputs changed; preserve TAG")
    else:
        immutable(path, inputs)


def verify_packet(root):
    packet = read_json(root / "packet.json")
    if packet["input_manifest_sha256"] != stable_hash(root / "input_manifest.json"):
        raise ValueError("diagnostic input manifest changed")
    verify_manifest(read_json(root / "input_manifest.json"))
    if packet["contract_sha256"] != semantic_sha256(read_json(root / "C2_BENIGNITY_DRAFT.json")):
        raise ValueError("diagnostic contract changed")
    for item in packet["prompts"]:
        request = read_json(root / "draft_requests" / (item["window_uid"] + ".json"))
        if (request["window_uid"] != item["window_uid"] or request["execution_authorized"] is not False
                or semantic_sha256(request["prompt"]) != item["prompt_sha256"]
                or request["draft_contract_sha256"] != packet["contract_sha256"]):
            raise ValueError("diagnostic draft request changed")
    return packet


def render(out, root, rows):
    esc = lambda value: html.escape(str(value))
    rel = lambda path: esc(os.path.relpath(path, root).replace("\\", "/"))
    sections = []
    for number, row in enumerate(rows, 1):
        uid = row["window_uid"]
        table = "".join(f'<tr><td>{esc(k)}</td><td>{esc("missing" if v is None else format(v,".6f"))}</td></tr>' for k,v in row["values"].items())
        frames = "".join(f'<figure><img loading="lazy" src="{rel(out / "media" / uid / ("T"+str(k)+".jpg"))}" alt="T{k}"><figcaption>T{k}</figcaption></figure>' for k in range(8))
        raw = "".join(f'<details><summary>{name} model record</summary><pre>{esc(json.dumps(row[name],ensure_ascii=False,indent=2))}</pre></details>' for name in ("C0", "C1", "C2"))
        sections.append(f'<section id="case-{number}"><h2>Case {number} | {esc(uid[:12])}</h2><video controls preload="metadata" src="{rel(out / "media" / uid / "clip.mp4")}"></video><div class="frames">{frames}</div><table><thead><tr><th>Saved feature</th><th>Value</th></tr></thead><tbody>{table}</tbody></table>{raw}</section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.12 Smoke Diagnostic Review</title><style>body{font:15px system-ui;color:#192326;background:#fafcfc;margin:0}main{max-width:1040px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:19px;overflow-wrap:anywhere}section{border-top:1px solid #c8d2d2;padding:24px 0}video{width:100%;max-height:440px;background:#111}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:16px 0}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{border-bottom:1px solid #ddd;text-align:left;padding:6px}th{background:#e5eff0}td:last-child{font-variant-numeric:tabular-nums}details{padding:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f0f2f2;padding:12px}aside{border-left:4px solid #b45532;padding:10px;background:#fff1e9}@media(max-width:600px){main{padding:14px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main><h1>V9.12 Smoke Diagnostic Review</h1><aside>Development evidence audit. Model statements are not verified facts. These four cases are not the independent locked-evaluation review.</aside>' + "".join(sections) + '</main></html>'
    (root / "index.html").write_text(page, encoding="utf-8")


def prepare(out, full_audit=True):
    if full_audit:
        base = prepared.audit(out)
        if not base["technical_prepared_ready"]:
            raise ValueError("prepared-run integrity audit failed")
    rows, inputs = diagnostic_rows(out)
    summary = read_json(out / "smoke/summary.json")
    observed = {k:sum(r["values"][k] is not None for r in rows) for k in NAMES}
    if observed != summary["observed_counts"] or summary["windows"] != 4:
        raise ValueError("smoke summary does not match reconstructed features")
    root = out / "handoff/smoke_semantic_review"
    root.mkdir(parents=True, exist_ok=True)
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json") if read_json(p)["phase"] == "smoke"]
    logical = [read_json(p) for p in (out / "cost/logical_requests").glob("*.json") if read_json(p)["phase"] == "smoke"]
    allowed = {r["window_uid"] for r in rows}
    if any(a["window_uid"] not in allowed for a in attempts):
        raise ValueError("smoke receipt outside fixed cohort")
    for r in rows:
        for ref in read_json(out / "private_acquisition/adaptation" / (r["window_uid"] + ".json"))["raw_request_manifest"]:
            raw = read_json(portable(ref["path"]))
            receipt = read_json(portable(raw["receipt"]))
            if receipt.get("raw_file_sha256") != ref["sha256"] or receipt["status"] != "success":
                raise ValueError("raw response lacks matching successful attempt receipt")
    invalid = [read_json(p) for p in (out / "cache/invalid").glob("*.json")]
    result = {"version": VERSION, "at": now(), "state": read_json(out / "state.json")["state"],
              "technical_complete": all(n == 4 for n in observed.values()) and all(not r["errors"] for r in rows),
              "observed_counts": observed, "costs": costs(attempts, logical), "prior_format_failures": invalid,
              "format_repair_changes": format_repair_changes(out),
              "per_window": [{k:r[k] for k in ("window_uid","stratum","video_id","frame_start","frame_end_exclusive","weak_target_audit_only","values","errors")} for r in rows],
              "explanation_types": dict(Counter(e["type"] for r in rows for b in r["C2"]["event_bindings"] for e in b["explanations"])),
              "semantic_truth_established": False, "formal_accuracy": None, "formal_AP": None,
              "next_recommendation": "HOLD_LARGER_ACQUISITION_REVIEW_BENIGNITY_AND_C0_C1_DISAGREEMENTS",
              "new_API_calls": 0, "scores_changed": False, "protocol_gate_overridden": False}
    write_json(root / "audit_summary.json", result)
    inputs.extend({"path": str((out / path).resolve()), "sha256": stable_hash(out / path)} for path in ("smoke/summary.json", "enrollment/smoke_manifest.jsonl", "seal/legacy_inventory.json"))
    immutable_inputs(root / "input_manifest.json", inputs)
    contract = draft_contract()
    immutable(root / "C2_BENIGNITY_DRAFT.json", contract)
    prompts = []
    for r in rows:
        prompt = native_prompt(r["C1"])
        immutable(root / "draft_requests" / (r["window_uid"] + ".json"), {"window_uid": r["window_uid"], "C1_sha256": semantic_sha256(r["C1"]),
            "evidence_signature": r["C1"]["evidence_signature"], "prompt": prompt, "draft_contract_sha256": semantic_sha256(contract), "execution_authorized": False})
        prompts.append({"window_uid": r["window_uid"], "prompt_sha256": semantic_sha256(prompt)})
    packet = {"version": VERSION, "case_ids": [r["window_uid"] for r in rows], "input_manifest_sha256": stable_hash(root / "input_manifest.json"), "contract_sha256": semantic_sha256(contract), "prompts": prompts}
    immutable(root / "packet.json", packet)
    if not (root / "review.json").exists():
        write_json(root / "review.json", {"packet_sha256": semantic_sha256(packet), "reviewer_id": "", "use_policy": "diagnostic_only_not_training_or_gold",
            "cases": [{"window_uid": r["window_uid"], "visible_event_summary": "pending", "benign_explanation_supported": "pending",
                       "unexplained_harmful_event_remains": "pending", "c0_c1_relation": "pending", "evidence_notes": "pending"} for r in rows]})
    render(out, root, rows)
    return result


def import_review(out, path):
    root = out / "handoff/smoke_semantic_review"
    packet = verify_packet(root)
    value = read_json(path)
    if not value.get("reviewer_id", "").strip() or value.get("packet_sha256") != semantic_sha256(packet) or value.get("use_policy") != "diagnostic_only_not_training_or_gold":
        raise ValueError("reviewer, packet hash or diagnostic scope missing")
    cases = prepared.index_rows(value["cases"], "window_uid")
    if set(cases) != set(packet["case_ids"]):
        raise ValueError("review must cover all four fixed cases")
    for row in cases.values():
        for field in ("visible_event_summary", "evidence_notes"):
            if not isinstance(row.get(field), str) or row[field].strip().lower() in ("", "pending"):
                raise ValueError("review evidence required: " + field)
        for field in ("benign_explanation_supported", "unexplained_harmful_event_remains"):
            if row.get(field) not in ("yes", "no", "unclear"):
                raise ValueError("use yes/no/unclear: " + field)
        if row.get("c0_c1_relation") not in ("consistent", "c0_better_supported", "c1_better_supported", "both_partly_supported", "unclear"):
            raise ValueError("invalid c0_c1_relation")
    result = {"version": VERSION, "reviewer_id": value["reviewer_id"], "reviewed_cases": 4,
              "review_sha256": stable_hash(path), "review_file": str(path.resolve()), "cases": list(cases.values()),
              "next": "DECIDE_NEW_TAG_FOR_FIXED_FOUR_C2_ONLY_RECHECK; no automatic paid execution",
              "existing_scores_changed": False, "training_or_gold_labels_created": False,
              "new_API_calls": 0, "adaptation_authorized": False}
    write_json(root / "review_imports" / (str(time.time_ns()) + ".json"), result)
    write_json(root / "review_import_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "import-review"))
    parser.add_argument("--run", type=Path, default=PROJECT / "runs/governed_v912_b1b4_minimal_effect_20260915")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        with run_lock(args.run):
            result = prepare(args.run) if args.command == "prepare" else import_review(args.run, args.review or args.run / "handoff/smoke_semantic_review/review.json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
