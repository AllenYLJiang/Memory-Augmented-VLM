"""Three narrow claim reviews and one optional, bounded missing-C2 acquisition."""
from __future__ import annotations

import copy
import html
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

from . import smoke_offline_v914 as audit
from . import smoke_recheck_v913 as prior
from .contracts import read_json, write_json, write_jsonl, iter_jsonl, semantic_sha256
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest, now
from .b1b4_trial.evidence import Budget, StopAcquisition, RequestRefused, dashscope_once
from .role_scoped import portable

VERSION = "v915_three_claim_scope_review_and_single_missing_C2_1"
PHASE = "missing_C2"
STAGE = "C2_normalized_C1_scope_clarified_v915"
USE = "diagnostic_claim_scope_only_not_training_or_gold"
CLAIMS = [
    {"claim_id": "hockey_local_benignity", "window_uid": "edb6091cadf84583ee6ac99cf67cd7f67d338a6b089cbeb95d90776cc38cea90",
     "title": "Hockey: local benign explanation", "claim": "The local contact and fall are visibly explained as benign play, beyond the rink/uniform context alone.",
     "not_implied": "This is not a judgment of rule legality or a benchmark anomaly label."},
    {"claim_id": "fire_person_binding", "window_uid": "01dfb1e8b7208657a6aa60f460fdb448d60afa2c8e33271ceadc63a627d4ef8a",
     "title": "Street fire: burning-object identity", "claim": "The visible flames can be bound to a person, rather than an unidentified nearby object.",
     "not_implied": "Fire/smoke being visible does not by itself identify what is burning."},
    {"claim_id": "vehicle_display_control", "window_uid": "e5896297709fc4b12ed74a3a2fce6a66c8b1841c890620936b8c74c91f1b2067",
     "title": "Vehicle: central-screen evidence", "claim": "The large central touchscreen visibly confirms that an automatic driving-control mode is active.",
     "not_implied": "An unverified display claim does not prove the mode is off. Even support here does not establish safety or demonstration intent."},
]


def prompt_for(c1):
    return """V9.15 scoped-reference clarification. This is one missing dependency,
not a new independent observation. Keep the frozen C1 unchanged; it is a proposal,
not guaranteed visual truth. Use strings for all IDs, including numeric-looking IDs.
normal_evidence_ids MUST refer only to entries you return in C2.normal_evidence.
Use N-prefixed IDs (N1, N2, ...) for those entries, never cite a C1 evidence ID there.
unexplained_direct_evidence_ids MUST refer only to this C1 event's evidence_ids.
participant_correspondence uses the exact C1 entity IDs; event_id uses the exact C1 event ID.
Do not invent benign evidence to complete the schema. A scene category alone is not
a local benign explanation. If the premise or explanation cannot be verified,
retain uncertainty: assessment_complete=false and/or the existing unknown fields.
An empty explanation list is allowed, but does not establish that C1 is visually true.
No reviewer answers, earlier scores or desired outcome are supplied.
""" + prior.native_prompt(c1)


def verify(out):
    seal = read_json(out / "seal.json")
    if not seal or seal.get("version") != VERSION:
        raise ValueError("prepare a separate V9.15 TAG first")
    for rel, digest in seal["files"].items():
        if stable_hash(out / rel) != digest:
            raise ValueError("frozen V9.15 artifact changed: " + rel)
    verify_manifest(read_json(out / "source_manifest.json"))
    verify_manifest(read_json(out / "code_manifest.json"))
    protocol = read_json(out / "protocol.json")
    if protocol["run_root"] != prior.canonical(out):
        raise ValueError("prepared run moved")
    # New attempts in old runs must not slip through a file-only inventory check.
    files = sorted(prior.canonical(p) for root in protocol["source_roots"]
                   for p in portable(root).rglob("*") if p.is_file())
    if files != protocol["source_file_paths"]:
        raise ValueError("source file inventory changed; preserve this TAG")
    return protocol


def prepare(project, source, out):
    source, out = source.resolve(), out.resolve()
    binding = read_json(source / "input_binding.json")
    if not binding or binding.get("version") != audit.VERSION or not (source / "completion.json").exists():
        raise ValueError("completed V9.14 audit required")
    old_run = portable(binding["source"]).resolve()
    old_protocol = prior.verify(old_run)
    original = portable(old_protocol["source_run"]).resolve()
    roots = [original, old_run, source]
    for root in roots:
        audit.check_output(root, out, root)
    if (out / "seal.json").exists():
        if verify(out)["source_audit"] != prior.canonical(source):
            raise ValueError("source changed for existing TAG")
        return report(out)
    if any(p.name != ".operation.lock" for p in out.iterdir()):
        raise ValueError("new empty TAG required")
    audit.build(project, old_run, source)  # Completed audit: verify only, no writes/calls.
    summary = read_json(source / "summary.json")
    decision = read_json(source / "next_request_decision.json")
    candidates = decision["missing_dependent_requests_for_future_decision_only"]
    uid = CLAIMS[0]["window_uid"]
    if len(candidates) != 1 or candidates[0]["window_uid"] != uid or candidates[0]["response_exists"]:
        raise ValueError("only the one recorded missing hockey dependency is eligible")
    if len(summary["lossless_C1_recoveries"]) != 1:
        raise ValueError("one verified lossless C1 recovery required")
    normalized = read_json(source / "normalized_C1" / (uid + ".json"))
    c1 = normalized["normalized"]
    if semantic_sha256(c1) != candidates[0]["recovered_parent"]["normalized_payload_sha256"]:
        raise ValueError("normalized parent hash mismatch")
    media = read_json(old_run / "media/manifest.json")
    prior.validate_fresh(c1, uid, semantic_sha256(media[uid]["image_sha256"]))
    before = audit.snapshot(roots)
    immutable(out / "source_manifest.json", before)
    immutable(out / "parent_C1.json", c1)
    immutable(out / "parent_provenance.json", candidates[0]["recovered_parent"])
    immutable(out / "media.json", media[uid])
    immutable(out / "request.json", {"stage": STAGE, "window_uid": uid, "prompt": prompt_for(c1),
              "parent_sha256": semantic_sha256(c1), "human_answers_in_input": False})
    config = read_json(old_run / "config.json")
    config.update(workers=1, transport_attempts=2, schema_repair_attempts=0, semantic_retries=0)
    immutable(out / "config.json", config)
    role_map = list(iter_jsonl(old_run / "enrollment/role_map.jsonl"))
    write_jsonl(out / "enrollment/role_map.jsonl", [r for r in role_map if r["window_uid"] == uid])
    humans = {r["window_uid"]: r for r in read_json(old_run / "review_reference/imported_review.json")["cases"]}
    cards = [{**spec, "existing_R3_reference_not_gold": humans[spec["window_uid"]],
              "frame_indices": media[spec["window_uid"]]["frame_indices"],
              "image_paths": media[spec["window_uid"]]["image_paths"],
              "image_sha256": media[spec["window_uid"]]["image_sha256"]} for spec in CLAIMS]
    packet = {"version": VERSION, "use_policy": USE, "blind": False, "cards": cards,
              "review_is_not_API_authorization": True}
    immutable(out / "review/packet.json", packet)
    template = {"version": VERSION, "packet_sha256": semantic_sha256(packet), "reviewer_id": "",
                "use_policy": USE, "claims": [{"claim_id": r["claim_id"], "assessment": "pending",
                    "frame_ids": [], "notes": ""} for r in cards]}
    immutable(out / "review/review.json", template)
    render_review(project, out, packet, template)
    plan = {"version": VERSION, "phase": PHASE, "windows": [{"window_uid": uid}],
            "logical_requests_upper_bound": 1, "physical_attempts_upper_bound": 2,
            "output_tokens_upper_bound": 2 * config["max_output_tokens"], "input_tokens": None,
            "parent_sha256": semantic_sha256(c1), "request_sha256": stable_hash(out / "request.json"),
            "graph_C0_C1_discovery_calls": 0, "human_answers_in_input": False,
            "semantic_retries": 0, "schema_repairs": 0, "success_criterion": "dependency observed, not preferred answer"}
    immutable(out / "plans" / (PHASE + ".json"), plan)
    paths = [Path(__file__), project / "tools/claim_scope_v915_cli.py", project / "run_claim_scope_v915.sh",
             project / "tools/static/claim_scope_review_v915.js", project / "tools/event_decision/safety.py"]
    immutable(out / "code_manifest.json", [{"path": prior.canonical(p), "sha256": stable_hash(p)} for p in paths]
              + read_json(source / "input_binding.json")["code"] + read_json(old_run / "code_manifest.json"))
    immutable(out / "protocol.json", {"version": VERSION, "run_root": prior.canonical(out),
        "source_audit": prior.canonical(source), "source_roots": [prior.canonical(p) for p in roots],
        "source_file_paths": sorted(r["path"] for r in before), "window_uid": uid,
        "review_use": USE, "fixed_claims": [r["claim_id"] for r in cards],
        "training_authorized": False, "graph_OT_changes_authorized": False})
    if audit.snapshot(roots) != before:
        raise ValueError("source changed during prepare")
    files = {p.relative_to(out).as_posix(): stable_hash(p) for p in out.rglob("*")
             if p.is_file() and p.name != ".operation.lock" and p != out / "review/review.json"}
    immutable(out / "seal.json", {"version": VERSION, "files": files})
    immutable(out / "authorizations" / (PHASE + ".json"), {"authorized": False, "approved_by": "",
        "plan_sha256": stable_hash(out / "plans" / (PHASE + ".json")), "max_physical_attempts": None,
        "max_output_tokens": None, "review_sha256": None, "resume_uncertain_attempts": False})
    return report(out)


def validate_review(packet, review):
    if not isinstance(review, dict) or set(review) != {"version", "packet_sha256", "reviewer_id", "use_policy", "claims"}:
        raise ValueError("review must contain exactly the template fields")
    if (review.get("version") != VERSION or review.get("packet_sha256") != semantic_sha256(packet)
            or review.get("use_policy") != USE):
        raise ValueError("review packet/version/diagnostic scope mismatch")
    reviewer = review.get("reviewer_id")
    if not isinstance(reviewer, str) or reviewer.strip().lower() in {"", "pending", "your_name"}:
        raise ValueError("reviewer_id required")
    rows = review.get("claims")
    expected = {r["claim_id"] for r in packet["cards"]}
    if not isinstance(rows, list) or len(rows) != len(expected) or {r.get("claim_id") for r in rows} != expected:
        raise ValueError("exactly the three fixed claims are required")
    for row in rows:
        if set(row) != {"claim_id", "assessment", "frame_ids", "notes"}:
            raise ValueError("unknown/missing review fields")
        if row["assessment"] not in {"supported", "contradicted", "uncertain"}:
            raise ValueError("finish each claim; uncertain is an acceptable answer")
        frames = row["frame_ids"]
        if (not isinstance(frames, list) or any(type(f) is not str for f in frames)
                or len(frames) != len(set(frames)) or not set(frames) <= {f"T{i}" for i in range(8)}):
            raise ValueError("frame_ids must be unique T0..T7")
        if row["assessment"] != "uncertain" and not frames:
            raise ValueError("supported/contradicted needs at least one cited frame")
        if not isinstance(row["notes"], str) or not row["notes"].strip():
            raise ValueError("a short evidence note is required, including for uncertain")
    return {**review, "reviewer_id": reviewer.strip()}


def scope_overlay(review):
    rows = [{**r, "observed": r["assessment"] != "uncertain",
             "claim_supported": {"supported": True, "contradicted": False, "uncertain": None}[r["assessment"]],
             "source": "human_diagnostic_claim_review", "review_sha256": semantic_sha256(review),
             "can_set_window_normal": False, "can_set_anomaly_label": False} for r in review["claims"]]
    return {"version": VERSION, "claims": rows,
        "scope_complete": True, "all_claims_supported_by_reviewer": all(r["claim_supported"] is True for r in rows),
        "unknown_is_not_normal": True, "vehicle_safety_or_intent_verified": False,
        "sports_rule_legality_verified": False, "original_scores_changed": False,
        "training_or_gold_labels_created": False, "API_authorized": False}


def import_review(out, path):
    verify(out)
    review = validate_review(read_json(out / "review/packet.json"), read_json(path))
    immutable(out / "review_import/accepted_review.json", review)
    immutable(out / "review_import/receipt.json", {"review_sha256": semantic_sha256(review),
              "accepted_file_sha256": stable_hash(out / "review_import/accepted_review.json"), "use_policy": USE})
    immutable(out / "claim_scope_overlay.json", scope_overlay(review))
    return report(out)


def accepted_review(out):
    review = read_json(out / "review_import/accepted_review.json")
    receipt = read_json(out / "review_import/receipt.json")
    if not review or not receipt:
        raise ValueError("finish/import the three claim reviews first; uncertain is allowed")
    validate_review(read_json(out / "review/packet.json"), review)
    if (receipt["review_sha256"] != semantic_sha256(review)
            or receipt["accepted_file_sha256"] != stable_hash(out / "review_import/accepted_review.json")):
        raise ValueError("accepted review changed")
    if read_json(out / "claim_scope_overlay.json") != scope_overlay(review):
        raise ValueError("claim-scope overlay changed")
    return review


def authorize(out, reviewer, cap):
    verify(out)
    review = accepted_review(out)
    if not isinstance(reviewer, str) or reviewer.strip().lower() in {"", "pending", "your_name"} or type(cap) is not int or cap not in (1, 2):
        raise ValueError("explicit APPROVED_BY and MAX_ATTEMPTS=1 or 2 required")
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    if any(r["status"] == "in_flight" for r in attempts):
        raise ValueError("uncertain billing remains; do not automatically retry")
    path = out / "authorizations" / (PHASE + ".json")
    old = read_json(path)
    config = read_json(out / "config.json")
    value = {**old, "authorized": True, "approved_by": reviewer.strip(), "max_physical_attempts": cap,
             "max_output_tokens": cap * config["max_output_tokens"], "review_sha256": semantic_sha256(review)}
    if old.get("authorized") and value != old:
        raise ValueError("authorization already frozen; no budget reset or review replacement")
    if cap < len(attempts):
        raise ValueError("cap smaller than attempts already used")
    if value != old:
        immutable(out / "authorization_receipt.json", {"previous": old, "approved": value})
        write_json(path, value)


def get_result(out):
    result = read_json(out / "result.json")
    if result is None:
        return None
    receipt = read_json(out / "result_receipt.json")
    if not receipt or receipt.get("sha256") != stable_hash(out / "result.json"):
        raise ValueError("result changed; do not rebuy")
    verify_manifest(result.get("raw_manifest", []))
    return result


def run(out, provider=dashscope_once):
    protocol = verify(out)
    if get_result(out) is not None:
        return report(out)
    review = accepted_review(out)
    config = read_json(out / "config.json")
    budget = Budget(out, PHASE, config)
    if budget.auth.get("review_sha256") != semantic_sha256(review):
        raise ValueError("authorization does not bind the imported review")
    request, c1, media = (read_json(out / file) for file in ("request.json", "parent_C1.json", "media.json"))
    identity = {"version": VERSION, "stage": STAGE, "window_uid": protocol["window_uid"],
                "config": config, "parent_sha256": semantic_sha256(c1), "images": media["image_sha256"],
                "frame_indices": media["frame_indices"], "prompt": request["prompt"]}
    key = semantic_sha256(identity)
    raw_path = out / "cache/raw" / (key + ".json")
    raw = read_json(raw_path)
    try:
        if raw is None:
            if provider is dashscope_once:
                if not os.environ.get("DASHSCOPE_API_KEY"):
                    raise StopAcquisition("Missing DASHSCOPE_API_KEY; no physical attempt started")
                import dashscope  # noqa: F401
            for attempt in range(config["transport_attempts"]):
                receipt_path = budget.begin(protocol["window_uid"], key, STAGE)
                started = time.monotonic()
                try:
                    text, usage = provider(config, media, request["prompt"])
                except Exception as exc:
                    write_json(receipt_path, {**read_json(receipt_path), "status": "provider_failure",
                        "error_type": type(exc).__name__, "error": str(exc)[:300], "billing_unknown": True,
                        "latency_seconds": time.monotonic() - started})
                    if isinstance(exc, (StopAcquisition, RequestRefused)) or attempt + 1 == config["transport_attempts"]:
                        raise
                    time.sleep(2 ** attempt)
                    continue
                # Persistence failure must not purchase another successful provider response.
                raw = {"identity": identity, "raw": text, "usage": usage, "receipt": prior.canonical(receipt_path)}
                try:
                    immutable(raw_path, raw)
                    write_json(receipt_path, {**read_json(receipt_path), "status": "success", "usage": usage,
                        "latency_seconds": time.monotonic() - started, "raw_file_sha256": stable_hash(raw_path)})
                except Exception as exc:
                    raise StopAcquisition("response persistence interrupted; inspect in-flight receipt, do not rebuy") from exc
                break
        receipt = read_json(portable(raw["receipt"]))
        if raw["identity"] != identity or receipt.get("raw_file_sha256") != stable_hash(raw_path) or receipt.get("status") != "success":
            raise StopAcquisition("raw identity/receipt mismatch; inspect, never rebuy")
        parsed = audit.parse_raw(raw["raw"])
        prior.validate_native(c1, parsed)
        result = {"status": "valid", "parsed": parsed, "diagnostic_conditional_not_anomaly_score": prior.diagnostic_binding(c1, parsed),
                  "premise_truth_verified": False, "human_review_not_used_as_input": True}
        immutable(out / "cost/logical_requests" / (key + ".json"), {"phase": PHASE, "stage": STAGE,
            "request_key": key, "cache_hit": False, "remote_schema_repair_index": 0, "local_parser_repair": False})
    except StopAcquisition as exc:
        write_json(out / "last_pause.json", {"reason": str(exc), "at": now(), "budget_reset": False})
        report(out)
        raise
    except Exception as exc:
        result = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)[:1000],
                  "no_semantic_retry": True}
    result["raw_manifest"] = [{"path": prior.canonical(raw_path), "sha256": stable_hash(raw_path)}] if raw_path.exists() else []
    immutable(out / "result.json", result)
    immutable(out / "result_receipt.json", {"sha256": stable_hash(out / "result.json")})
    return report(out)


def report(out):
    verify(out)
    imported = (out / "review_import/accepted_review.json").exists()
    if imported:
        accepted_review(out)
    result = get_result(out)
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    logical = [read_json(p) for p in (out / "cost/logical_requests").glob("*.json")]
    state = "REVIEW_THREE_LOCAL_CLAIMS" if not imported else "OPTIONAL_SINGLE_C2_REQUIRES_EXPLICIT_BUDGET" if result is None else "INSPECT_SINGLE_DEPENDENCY_NO_EXPANSION"
    summary = {"version": VERSION, "claim_reviews_imported": imported,
        "single_C2_status": result["status"] if result else "not_requested", "next": state,
        "cost": prior.costs(attempts, logical), "formal_accuracy": None, "formal_AP": None,
        "old_scores_changed": False, "larger_acquisition_authorized": False, "training_authorized": False,
        "graph_OT_changes_authorized": False, "complete_four_case_comparison": False,
        "vehicle_C2_still_originally_invalid": True}
    write_json(out / "summary.json", summary)
    return summary


def render_review(project, out, packet, template):
    esc = lambda x: html.escape(str(x))
    root = out / "review"
    pieces = []
    for spec in packet["cards"]:
        cid = spec["claim_id"]
        frames = ''.join(f'<figure><label><img src="{quote(Path(os.path.relpath(portable(p), root)).as_posix(), safe="/")}" alt="T{i}"><figcaption><input type="checkbox" name="{cid}_frames" value="T{i}"> T{i} / frame {spec["frame_indices"][i]}</figcaption></label></figure>' for i, p in enumerate(spec["image_paths"]))
        pieces.append(f'<section data-claim="{cid}"><h2>{esc(spec["title"])}</h2><p class="claim">{esc(spec["claim"])}</p><p>{esc(spec["not_implied"])}</p><div class="frames">{frames}</div><label class="field">Assessment<select id="{cid}_assessment"><option value="pending">Pending</option><option value="supported">Supported by visible evidence</option><option value="contradicted">Contradicted by visible evidence</option><option value="uncertain">Uncertain / not established</option></select></label><label class="field">Evidence note<textarea id="{cid}_notes" rows="3"></textarea></label><details><summary>Existing R3 diagnostic reference, not gold</summary><p>{esc(spec["existing_R3_reference_not_gold"]["visible_event_summary"])}</p></details></section>')
    data = json.dumps(template, ensure_ascii=True).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.15 Three Claim Review</title><style>
body{margin:0;background:#fbfcfc;color:#202923;font:15px system-ui}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:20px}section{padding:24px 0;border-top:1px solid #b9c8c0}.claim{font-weight:600}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px;overflow-wrap:anywhere}.field{display:block;margin:14px 0}input[type=text],select,textarea{font:inherit;padding:8px;border:1px solid #889a91;border-radius:4px;box-sizing:border-box;max-width:100%}textarea{display:block;width:100%}select{display:block;min-height:40px}button{background:#166e60;color:white;border:0;border-radius:4px;padding:10px 18px;font:inherit;cursor:pointer}aside{background:#fff0eb;border-left:4px solid #b85431;padding:12px}footer{padding:18px 0;border-top:1px solid #b9c8c0}#status{color:#8f281e;overflow-wrap:anywhere}details{margin-top:16px}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><main><h1>V9.15 Three Claim Review</h1><aside>Diagnostic follow-up, not a blind evaluation or gold labeling task. Uncertain is a valid conclusion. Reviews never authorize API calls or change existing scores.</aside><label class="field">Reviewer ID <input type="text" id="reviewer_id" autocomplete="off"></label>'''
    page += ''.join(pieces) + '<footer><button id="download" type="button" title="Download the three claim assessments">Download review.json</button><p id="status" role="status"></p></footer>'
    page += f'<script type="application/json" id="review-template">{data}</script><script src="review.js"></script></main></html>'
    (root / "index.html").write_text(page, encoding="utf-8")
    prior.frozen_copy(project / "tools/static/claim_scope_review_v915.js", root / "review.js")
