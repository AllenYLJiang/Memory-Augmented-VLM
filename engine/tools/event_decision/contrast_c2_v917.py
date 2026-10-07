"""Four frozen-parent C2 requests, with no score update or semantic retry."""
from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

from . import benignity_audit_v916 as previous
from .contracts import read_json, write_json, write_jsonl, semantic_sha256, iter_jsonl
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest, now
from .b1b4_trial.evidence import Budget, StopAcquisition, RequestRefused, dashscope_once
from .role_scoped import portable

prior = previous.prior
VERSION = "v917_four_fixed_parent_contrast_C2_1"
PHASE = "four_C2_contrast"
STAGE = "C2_contrast_same_parent_v917"
FRAME_IDS = {f"T{i}" for i in range(8)}
CONTRACT = {
    "version": VERSION,
    "purpose": "Diagnostic observations and explanation contrast, not anomaly scoring.",
    "premise_support": ["supported", "contradicted", "unknown"],
    "discrimination": ["benign", "harmful", "neither", "unknown"],
    "neither": "Cited evidence is compatible with both alternatives and does not distinguish them.",
    "unknown": "Visibility, identity or available local evidence is insufficient to decide.",
    "no_numeric_output_scores": True,
    "no_new_C1_or_C0": True,
    "reviewer_answers_and_old_C2_not_sent": True,
    "frozen_C1_is_a_proposal_not_verified_truth": True,
    "human_truth_verified": False,
    "deployment_authorized": False,
}


def template(c1):
    return {"schema_version": VERSION, "window_id": c1["window_id"],
        "proposal_sha256": semantic_sha256(c1), "evidence_signature": c1["evidence_signature"],
        "observations": [], "event_assessments": [{"event_id": e["event_id"],
            "premise_support": "unknown", "benign_alternative": "unknown",
            "harmful_alternative": "unknown", "evidence_ids": [], "discrimination": "unknown",
            "contrast_reason": "Insufficient visible evidence to distinguish alternatives.",
            "limits": "No claim of visual verification."} for e in c1["events"]]}


def prompt_for(c1):
    allowed = {"schema_version", "window_id", "evidence_signature", "scan_complete", "overflow",
               "observation_sufficient", "uncertainty", "entities", "evidence", "events"}
    if set(c1) != allowed:
        raise ValueError("C1 contains unexpected fields; do not send private metadata or silently change the parent")
    for group, fields in (
        ("entities", {"entity_id", "kind", "visual_descriptor", "observed_frame_ids"}),
        ("evidence", {"evidence_id", "frame_ids", "description"}),
        ("events", {"event_id", "participant_ids", "observed_frame_ids", "observed_action", "evidence_ids",
                    "direct_mechanism_probability", "direct_evidence_quality", "visibility", "time_relation_unknown"}),
    ):
        if any(set(row) != fields for row in c1[group]):
            raise ValueError("C1 contains unexpected fields in " + group)
    return """Inspect only the supplied exact eight frames, labeled T0 through T7.
The frozen C1 below is an earlier proposal, NOT visual truth. Do not update it.
Its confidence numbers are not evidence of truth or a target to reproduce.
For EVERY C1 event, first say whether its literal proposed action/object binding
is supported, contradicted, or unknown in these frames. Do not invent identities.
Then consider one benign explanation and one harmful explanation of that SAME
local event. Give the concrete visible observation that distinguishes the two,
or explicitly retain neither/unknown. These alternatives are hypotheses, not facts.
Scene category, uniforms, equipment, ordinary co-occurrence, or recovery afterwards
alone cannot establish local benignity. Conversely, contact/falling alone does not
establish harmful intent. Do not invent rules, intent, control modes or unseen causes.
Do not infer a particular cause for calm/ordinary activity when only that activity
is visible. Failure to find a benign explanation does not prove the C1 premise.

Return one JSON object matching the template. No markdown, scores, or extra fields.
observations entries have exactly: observation_id (N-prefixed string), frame_ids
(nonempty unique T0..T7), participant_ids (unique exact C1 entity IDs, or [] for
background/uncertain identity), description (literal visible detail, not an inferred
verdict). Multiple observations may cite the same frame. Do not put C1 evidence IDs
in evidence_ids: that field refers ONLY to your observations[].observation_id.
event_assessments must include each C1 event exactly once, with the template fields.
All text fields must be nonempty; 'unknown' is allowed for an unresolved alternative.
premise_support: supported | contradicted | unknown.
discrimination: benign | harmful | neither | unknown.
If premise_support is not supported, discrimination MUST remain unknown, while
literal observations are retained. Contradicting a premise is not proof of normality.
For supported/contradicted premises, cite at least one observation. For benign or
harmful discrimination, cite a local frame overlapping the C1 event and observations
covering its proposed participants; contrast_reason must explain why that actual
detail favors one alternative, not just name the activity or repeat the enum.
neither means the observations fit both alternatives; unknown means insufficient
observability or identity evidence. Both are legitimate, non-normal, non-abnormal masks.
limits must say what is NOT established. No external video, audio, filenames,
reviewer answers, old C2, desired labels, graph scores, or ground truth is provided.
Output completeness is required; a preferred semantic answer is NOT required.

TEMPLATE (unknown is a valid outcome, not an instruction to choose it):
""" + json.dumps(template(c1), ensure_ascii=False) + "\nFROZEN C1:\n" + json.dumps(c1, ensure_ascii=False)


def text(value, name):
    if type(value) is not str or not value.strip() or len(value) > 4000:
        raise ValueError(name + " requires nonempty text, at most 4000 characters")


def refs(value, allowed, name, nonempty=False):
    if (not isinstance(value, list) or any(type(x) is not str for x in value)
            or len(value) != len(set(value)) or not set(value) <= allowed or nonempty and not value):
        raise ValueError(name + " has invalid, duplicate, empty or wrong-scope references")


def validate(c1, value):
    required = {"schema_version", "window_id", "proposal_sha256", "evidence_signature", "observations", "event_assessments"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("exact C2 root fields required")
    for field, expected in (("schema_version", VERSION), ("window_id", c1["window_id"]),
                            ("proposal_sha256", semantic_sha256(c1)), ("evidence_signature", c1["evidence_signature"])):
        if value[field] != expected:
            raise ValueError("wrong parent/window/schema/media: " + field)
    entities = {e["entity_id"] for e in c1["entities"]}
    observations = value["observations"]
    if not isinstance(observations, list) or len(observations) > 16:
        raise ValueError("observations must be a list of at most 16 items")
    defined = {}
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"observation_id", "frame_ids", "participant_ids", "description"}:
            raise ValueError("exact observation fields required")
        oid = row["observation_id"]
        if type(oid) is not str or not oid.startswith("N") or oid.strip() != oid or oid in defined:
            raise ValueError("unique N-prefixed string observation IDs required")
        refs(row["frame_ids"], FRAME_IDS, "observation frames", True)
        refs(row["participant_ids"], entities, "observation participants")
        text(row["description"], "literal description")
        defined[oid] = row
    fields = {"event_id", "premise_support", "benign_alternative", "harmful_alternative", "evidence_ids",
              "discrimination", "contrast_reason", "limits"}
    rows = value["event_assessments"]
    if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != fields for r in rows):
        raise ValueError("exact event assessment fields required")
    ids = [r["event_id"] for r in rows]
    expected = {e["event_id"]: e for e in c1["events"]}
    if any(type(e) is not str for e in ids) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("every frozen event must appear exactly once")
    for row in rows:
        if row["premise_support"] not in CONTRACT["premise_support"] or row["discrimination"] not in CONTRACT["discrimination"]:
            raise ValueError("unknown premise/discrimination state")
        refs(row["evidence_ids"], set(defined), "C2 evidence_ids",
             row["premise_support"] != "unknown" or row["discrimination"] != "unknown")
        for field in ("benign_alternative", "harmful_alternative", "contrast_reason", "limits"):
            text(row[field], field)
        if row["premise_support"] != "supported" and row["discrimination"] != "unknown":
            raise ValueError("unconfirmed premise must preserve unknown discrimination")
        if row["discrimination"] in {"benign", "harmful"}:
            event = expected[row["event_id"]]
            local = [defined[i] for i in row["evidence_ids"]
                     if set(defined[i]["frame_ids"]) & set(event["observed_frame_ids"])]
            participants = set().union(*(set(o["participant_ids"]) for o in local))
            if not local or not set(event["participant_ids"]) <= participants:
                raise ValueError("discrimination needs local evidence covering proposed event participants")
            if row["benign_alternative"].strip().lower() == row["harmful_alternative"].strip().lower():
                raise ValueError("discrimination requires distinct alternatives")
    return value


def input_identity(config, media, c1, prompt):
    return {"version": VERSION, "stage": STAGE, "window_uid": c1["window_id"],
        "parent_sha256": semantic_sha256(c1), "config": config,
        "image_sha256": media["image_sha256"], "frame_indices": media["frame_indices"], "prompt": prompt}


def verify(out):
    seal = read_json(out / "seal.json")
    if not seal or seal.get("version") != VERSION:
        raise ValueError("prepare a separate V9.17 TAG first")
    for rel, digest in seal["files"].items():
        if stable_hash(out / rel) != digest:
            raise ValueError("frozen V9.17 file changed: " + rel)
    verify_manifest(read_json(out / "code_manifest.json"))
    verify_manifest(read_json(out / "source_manifest.json"))
    protocol = read_json(out / "protocol.json")
    if protocol["run_root"] != prior.canonical(out):
        raise ValueError("prepared run moved")
    return protocol


def prepare(project, source, out, max_output_tokens=8192):
    source, out = source.resolve(), out.resolve()
    previous.followup.check_out(source, out)
    if (out / "seal.json").exists():
        protocol = verify(out)
        if protocol["source_run"] != prior.canonical(source) or read_json(out / "config.json")["max_output_tokens"] != max_output_tokens:
            raise ValueError("source/budget changed for existing TAG")
        return report(out)
    if any(p.name != ".operation.lock" for p in out.iterdir()):
        raise ValueError("new empty TAG required; preserve incomplete preparation")
    if type(max_output_tokens) is not int or not 512 <= max_output_tokens <= 8192:
        raise ValueError("freeze MAX_OUTPUT_TOKENS between 512 and 8192")
    source_protocol = read_json(source / "protocol.json")
    roots = [source] + [portable(p).resolve() for p in source_protocol["source_roots"]]
    before = previous.audit.snapshot(roots)
    records, review, _, v913 = previous.collect(source)
    ids = prior.verify(v913)["case_ids"]
    media = read_json(v913 / "media/manifest.json")
    config = read_json(source / "config.json")
    config.update(workers=1, transport_attempts=1, schema_repair_attempts=0,
                  semantic_retries=0, max_output_tokens=max_output_tokens)
    immutable(out / "config.json", config)
    immutable(out / "contract.json", CONTRACT)
    immutable(out / "source_manifest.json", before)
    immutable(out / "reference_only/human.json", {"R1": review,
        "R3": read_json(v913 / "review_reference/imported_review.json"), "sent_to_provider": False,
        "use": "post-hoc diagnostic disagreement only, never training or gold"})
    cases = []
    for uid in ids:
        stage = "v915_missing_C2" if uid == source_protocol["window_uid"] else "v913_C2_reobserved_C1"
        same = [r for r in records if r["window_uid"] == uid and r["stage"] == stage]
        if len(same) != 1:
            raise ValueError("exactly one prespecified same-parent baseline required")
        same = same[0]
        parent = same["C1"]
        prior.validate_fresh(parent, uid, semantic_sha256(media[uid]["image_sha256"]))
        evidence = media[uid]
        if len(evidence["image_paths"]) != 8 or len(evidence["image_sha256"]) != 8 or len(evidence["frame_indices"]) != 8:
            raise ValueError("exactly eight frozen frames required")
        if (any(type(i) is not int for i in evidence["frame_indices"])
                or sorted(set(evidence["frame_indices"])) != evidence["frame_indices"]):
            raise ValueError("frame indices must be eight distinct ordered integers")
        for path, digest in zip(evidence["image_paths"], evidence["image_sha256"]):
            if stable_hash(portable(path)) != digest:
                raise ValueError("saved frame changed")
        prompt = prompt_for(parent)
        request = input_identity(config, evidence, parent, prompt)
        folder = out / "frozen" / uid
        immutable(folder / "C1.json", parent)
        immutable(folder / "media.json", evidence)
        immutable(folder / "request.json", request)
        immutable(folder / "same_parent_C2.json", same)
        others = [r for r in records if r["window_uid"] == uid and r["stage"] != stage]
        immutable(folder / "different_parent_history.json", {"records": [r for r in others if r["C1_sha256"] != semantic_sha256(parent)],
            "not_a_paired_effect_estimate": True})
        immutable(folder / "other_same_parent_history.json", {"records": [r for r in others if r["C1_sha256"] == semantic_sha256(parent)],
            "not_the_prespecified_baseline": True})
        cases.append({"window_uid": uid, "name": previous.CASE_NOTES[uid[:12]]["name"],
            "parent_sha256": semantic_sha256(parent), "same_parent_stage": stage,
            "same_parent_technical_valid": same["technical_validation"]["valid"],
            "same_parent_C2_sha256": same["C2_sha256"], "request_key": semantic_sha256(request),
            "frames_sha256": evidence["image_sha256"], "frame_indices": evidence["frame_indices"],
            "prompt_characters": len(prompt)})
    immutable(out / "cohort.json", cases)
    roles = {r["window_uid"]: r for r in iter_jsonl(v913 / "enrollment/role_map.jsonl")}
    write_jsonl(out / "enrollment/role_map.jsonl", [roles[u] for u in ids])
    immutable(out / "plans" / (PHASE + ".json"), {
        "version": VERSION, "phase": PHASE, "windows": cases,
        "logical_requests_upper_bound": 4, "physical_attempts_upper_bound": 4,
        "max_attempts_per_window": 1, "max_output_tokens_per_attempt": max_output_tokens,
        "output_tokens_upper_bound": 4 * max_output_tokens,
        "input_tokens": None, "input_budget_note": "Exact prompts, eight image hashes and max pixels are frozen. Provider input/image token accounting is unavailable before upload; this is not a total-token or currency cap.",
        "semantic_retries": 0, "format_repairs": 0, "transport_retries": 0,
        "new_C0_C1_graph_discovery_calls": 0, "prior_authorizations_inherited": False,
        "all_terminal_outcomes_retained": True,
        "success_criterion": "Inspectable local evidence contrast, not higher positive rate or agreement with reviewers.",
        "no_automatic_score_delta": True,
    })
    immutable(out / "protocol.json", {"version": VERSION, "run_root": prior.canonical(out),
        "source_run": prior.canonical(source), "windows": ids,
        "design_exposed_development_only": True,
        "graph_OT_changes_authorized": False, "training_authorized": False})
    own = [Path(__file__), project / "tools/contrast_c2_v917_cli.py", project / "run_contrast_c2_v917.sh",
           Path(previous.__file__), Path(previous.followup.__file__)]
    code = {r["path"]: r for r in read_json(source / "code_manifest.json")}
    code.update({prior.canonical(p): {"path": prior.canonical(p), "sha256": stable_hash(p)} for p in own})
    immutable(out / "code_manifest.json", sorted(code.values(), key=lambda r: r["path"]))
    if previous.audit.snapshot(roots) != before:
        raise ValueError("source changed during preparation")
    immutable(out / "seal.json", {"version": VERSION, "files": {p.relative_to(out).as_posix(): stable_hash(p)
              for p in sorted(out.rglob("*")) if p.is_file() and p.name != ".operation.lock"}})
    immutable(out / "authorizations" / (PHASE + ".json"), {"authorized": False, "approved_by": "",
        "plan_sha256": stable_hash(out / "plans" / (PHASE + ".json")),
        "max_physical_attempts": None, "max_output_tokens": None, "resume_uncertain_attempts": False})
    return report(out)


def authorize(out, approved_by, max_attempts):
    verify(out)
    if type(approved_by) is not str or approved_by.strip().lower() in {"", "pending", "your_name"} or type(max_attempts) is not int or max_attempts != 4:
        raise ValueError("explicit APPROVED_BY and MAX_ATTEMPTS=4 required for the whole fixed panel")
    path = out / "authorizations" / (PHASE + ".json")
    current = read_json(path)
    approved = {"authorized": True, "approved_by": approved_by.strip(),
        "plan_sha256": stable_hash(out / "plans" / (PHASE + ".json")),
        "max_physical_attempts": 4, "max_output_tokens": 4 * read_json(out / "config.json")["max_output_tokens"],
        "resume_uncertain_attempts": False}
    receipt = read_json(out / "authorization_receipt.json")
    if receipt:
        if receipt["approved"] != approved or current not in (receipt["previous"], approved):
            raise ValueError("approval is frozen; no reset or budget extension")
    else:
        if current.get("authorized"):
            raise ValueError("authorization lacks approval receipt")
        immutable(out / "authorization_receipt.json", {"previous": current, "approved": approved})
    write_json(path, approved)


def check_authorization(out):
    if (out / "authorizations/approval.json").exists():
        raise StopAcquisition("bundle overrides are forbidden in this four-call protocol")
    receipt = read_json(out / "authorization_receipt.json")
    if not receipt or read_json(out / "authorizations" / (PHASE + ".json")) != receipt["approved"]:
        raise StopAcquisition("explicit matching V9.17 budget approval required")


def verify_raw(out, case, raw_path):
    raw = read_json(raw_path)
    request = read_json(out / "frozen" / case["window_uid"] / "request.json")
    receipt_path = portable(raw["receipt"]).resolve()
    if receipt_path.parent != (out / "cost/attempts").resolve():
        raise ValueError("receipt outside this run")
    receipt = read_json(receipt_path)
    if (raw["identity"] != request or semantic_sha256(request) != case["request_key"]
            or raw_path.stem != case["request_key"] or receipt.get("request_key") != case["request_key"]
            or receipt.get("status") != "success" or receipt.get("raw_file_sha256") != stable_hash(raw_path)):
        raise StopAcquisition("raw/receipt identity mismatch or incomplete persistence; never rebuy automatically")
    return raw


def parse_result(out, case, raw_path):
    raw = verify_raw(out, case, raw_path)
    c1 = read_json(out / "frozen" / case["window_uid"] / "C1.json")
    try:
        parsed = previous.audit.parse_raw(raw["raw"])
        validate(c1, parsed)
        result = {"status": "valid", "parsed": parsed}
    except (ValueError, TypeError, KeyError) as exc:
        result = {"status": "schema_failed", "reason": str(exc)[:1000]}
    return {**result, "window_uid": case["window_uid"], "request_key": case["request_key"],
        "raw_manifest": [{"path": prior.canonical(raw_path), "sha256": stable_hash(raw_path)}],
        "human_answers_used_as_input": False, "semantic_truth_verified": False,
        "anomaly_score": None, "no_semantic_retry": True}


def save_result(out, case, result):
    path = out / "results" / (case["window_uid"] + ".json")
    immutable(path, result)
    immutable(path.with_suffix(".receipt.json"), {"sha256": stable_hash(path)})
    if result["status"] in {"valid", "schema_failed"}:
        immutable(out / "cost/logical_requests" / (case["request_key"] + ".json"), {
            "phase": PHASE, "stage": STAGE, "request_key": case["request_key"],
            "status": result["status"], "cache_hit": False,
            "remote_schema_repair_index": 0, "local_parser_repair": False})


def recover_local_commits(out, cases):
    """Finish a verified persisted response after interruption, without a provider call."""
    for case in cases:
        raw_path = out / "cache/raw" / (case["request_key"] + ".json")
        path = out / "results" / (case["window_uid"] + ".json")
        saved = read_json(path)
        physical_paths = [p for p in (out / "cost/attempts").glob("*.json")
                          if read_json(p).get("request_key") == case["request_key"]]
        if len(physical_paths) > 1:
            raise StopAcquisition("more than one physical attempt for a frozen request")
        if raw_path.exists():
            expected = parse_result(out, case, raw_path)
            if saved is not None and saved != expected:
                raise StopAcquisition("saved result disagrees with raw; no automatic overwrite or rebuy")
            save_result(out, case, expected)
        elif saved and saved.get("status") in {"provider_failed", "refused"}:
            receipt = portable(saved["physical_receipt"]).resolve()
            if receipt.parent != (out / "cost/attempts").resolve() or stable_hash(receipt) != saved["physical_receipt_sha256"]:
                raise StopAcquisition("incomplete failure receipt changed")
            physical = read_json(receipt)
            if physical.get("window_uid") != case["window_uid"] or physical.get("request_key") != case["request_key"] or physical.get("status") != saved["status"]:
                raise StopAcquisition("failure identity changed")
            save_result(out, case, saved)
        elif saved is None and physical_paths:
            physical_path = physical_paths[0]
            physical = read_json(physical_path)
            if physical.get("phase") != PHASE or physical.get("window_uid") != case["window_uid"]:
                raise StopAcquisition("physical receipt outside the frozen identity")
            if physical["status"] in {"provider_failed", "refused"}:
                save_result(out, case, {"status": physical["status"], "window_uid": case["window_uid"],
                    "request_key": case["request_key"], "error_type": physical["error_type"],
                    "physical_receipt": prior.canonical(physical_path),
                    "physical_receipt_sha256": stable_hash(physical_path), "raw_manifest": [],
                    "no_semantic_retry": True, "anomaly_score": None})
            elif physical["status"] == "success":
                raise StopAcquisition("successful paid receipt has no raw response; inspect, do not rebuy")


def get_result(out, case):
    path = out / "results" / (case["window_uid"] + ".json")
    result = read_json(path)
    if result is None:
        return None
    receipt = read_json(path.with_suffix(".receipt.json"))
    if not receipt or receipt.get("sha256") != stable_hash(path):
        raise StopAcquisition("result receipt incomplete or changed; inspect saved raw, never rebuy")
    if result.get("request_key") != case["request_key"] or result.get("window_uid") != case["window_uid"]:
        raise ValueError("result identity mismatch")
    verify_manifest(result.get("raw_manifest", []))
    if result["status"] in {"valid", "schema_failed"}:
        raw_path = out / "cache/raw" / (case["request_key"] + ".json")
        if parse_result(out, case, raw_path) != result:
            raise ValueError("saved result differs from exact raw replay")
    elif result["status"] in {"provider_failed", "refused"}:
        physical = portable(result["physical_receipt"]).resolve()
        if physical.parent != (out / "cost/attempts").resolve() or stable_hash(physical) != result["physical_receipt_sha256"]:
            raise ValueError("failure receipt changed")
    else:
        raise ValueError("unknown saved terminal status")
    return result


def run(out, provider=dashscope_once):
    verify(out)
    cases = read_json(out / "cohort.json")
    recover_local_commits(out, cases)
    if all(get_result(out, c) is not None for c in cases):
        return report(out)
    check_authorization(out)
    config = read_json(out / "config.json")
    budget = Budget(out, PHASE, config)
    if provider is dashscope_once:
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise StopAcquisition("Missing DASHSCOPE_API_KEY; no request started")
        import dashscope  # noqa: F401
    try:
        for case in cases:
            if get_result(out, case) is not None:
                continue
            uid, key = case["window_uid"], case["request_key"]
            raw_path = out / "cache/raw" / (key + ".json")
            if not raw_path.exists():
                request = read_json(out / "frozen" / uid / "request.json")
                media = read_json(out / "frozen" / uid / "media.json")
                receipt_path = budget.begin(uid, key, STAGE)
                start = time.monotonic()
                try:
                    response, usage = provider(config, media, request["prompt"])
                except Exception as exc:
                    status = "refused" if isinstance(exc, RequestRefused) else "provider_failed"
                    # Do not serialize arbitrary provider exception strings, which may contain credentials.
                    write_json(receipt_path, {**read_json(receipt_path), "status": status,
                        "error_type": type(exc).__name__, "billing_unknown": True,
                        "latency_seconds": time.monotonic() - start})
                    save_result(out, case, {"status": status, "window_uid": uid, "request_key": key,
                        "error_type": type(exc).__name__, "physical_receipt": prior.canonical(receipt_path),
                        "physical_receipt_sha256": stable_hash(receipt_path), "raw_manifest": [],
                        "no_semantic_retry": True, "anomaly_score": None})
                    print(f"[v917] {uid[:12]}: {status}; no rebuy", flush=True)
                    if isinstance(exc, StopAcquisition):
                        raise StopAcquisition("provider account/rate stop; saved failure retained, resume only unattempted windows") from exc
                    continue
                raw = {"identity": request, "raw": response, "usage": usage, "receipt": prior.canonical(receipt_path)}
                try:
                    immutable(raw_path, raw)
                    write_json(receipt_path, {**read_json(receipt_path), "status": "success", "usage": usage,
                        "latency_seconds": time.monotonic() - start, "raw_file_sha256": stable_hash(raw_path)})
                except Exception as exc:
                    raise StopAcquisition("response persistence interrupted; inspect in-flight receipt, never rebuy") from exc
            result = parse_result(out, case, raw_path)
            save_result(out, case, result)
            print(f"[v917] {uid[:12]}: {result['status']}; all semantic outcomes retained", flush=True)
            report(out, verify_inputs=False)
    except (StopAcquisition, RuntimeError, OSError, ValueError) as exc:
        write_json(out / "last_pause.json", {"at": now(), "reason": str(exc), "budget_reset": False})
        report(out, verify_inputs=False)
        raise
    return report(out)


def comparison(out, case, result):
    frozen = out / "frozen" / case["window_uid"]
    baseline = read_json(frozen / "same_parent_C2.json")
    parent = read_json(frozen / "C1.json")
    if baseline["C1_sha256"] != semantic_sha256(parent) or baseline["C2"]["proposal_sha256"] != semantic_sha256(parent):
        raise ValueError("baseline is not the exact same parent")
    native_valid = baseline["technical_validation"]["valid"]
    old = prior.diagnostic_binding(parent, baseline["C2"]) if native_valid else None
    old_events = {r["event_id"]: r for r in old["events"]} if old else {}
    new_rows = result["parsed"]["event_assessments"] if result and result["status"] == "valid" else []
    observations = {r["observation_id"]: r for r in result["parsed"]["observations"]} if new_rows else {}
    rows = []
    for row in new_rows:
        legacy = old_events.get(row["event_id"])
        accepted = bool(legacy["known"] and legacy["bound"] is not None and legacy["bound"] > 0) if legacy else None
        rows.append({**row, "cited_observations": [observations[k] for k in row["evidence_ids"]],
            "old_benign_binding_accepted": accepted,
            "old_benign_to_new_not_benign_claim_change": accepted is True and row["discrimination"] != "benign",
            "not_a_same_scale_score_delta": True, "semantic_truth_verified": False})
    basketball = case["name"] == "Basketball"
    return {"window_uid": case["window_uid"], "name": case["name"], "same_parent_sha256": semantic_sha256(parent),
        "same_eight_frames": True, "baseline_stage": case["same_parent_stage"],
        "baseline_technical_valid": native_valid, "baseline_error": baseline["technical_validation"]["error"],
        "old_diagnostic_only_if_valid": old, "new_status": result["status"] if result else "not_requested",
        "new_error": (result or {}).get("reason") or (result or {}).get("error_type"), "events": rows,
        "unknown_events": sum(r["discrimination"] == "unknown" for r in rows),
        "neither_events": sum(r["discrimination"] == "neither" for r in rows),
        "unresolved_or_contradicted_premises": sum(r["premise_support"] != "supported" for r in rows),
        "basketball_new_harmful_claim_flag": basketball and any(r["discrimination"] == "harmful" for r in rows),
        "basketball_new_unknown_flag": basketball and any(r["discrimination"] in {"unknown", "neither"} for r in rows),
        "basketball_loss_of_old_benign_claim_flag": basketball and any(r["old_benign_to_new_not_benign_claim_change"] for r in rows),
        "flags_are_not_measured_degradation": True, "anomaly_score_delta": None,
        "semantic_truth_verified": False, "training_or_gold_labels_created": False}


def report(out, verify_inputs=True):
    if verify_inputs:
        verify(out)
    cases = read_json(out / "cohort.json")
    rows = [comparison(out, c, get_result(out, c)) for c in cases]
    statuses = Counter(r["new_status"] for r in rows)
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    logical = [read_json(p) for p in (out / "cost/logical_requests").glob("*.json")]
    cost = prior.costs(attempts, logical)
    cost["unique_response_keys"] = len({r["request_key"] for r in logical})
    cost["unique_valid_request_keys"] = len({r["request_key"] for r in logical if r["status"] == "valid"})
    complete = all(r["new_status"] != "not_requested" for r in rows)
    valid = statuses["valid"] == 4
    summary = {"version": VERSION, "windows": 4, "status_counts": dict(statuses),
        "complete_terminal_accounting": complete, "technical_all_four_valid": valid,
        "baseline_valid_pairs": sum(r["baseline_technical_valid"] for r in rows),
        "baseline_invalid_pairs_preserved": sum(not r["baseline_technical_valid"] for r in rows),
        "unknown_events": sum(r["unknown_events"] for r in rows),
        "neither_events": sum(r["neither_events"] for r in rows),
        "unresolved_or_contradicted_premises": sum(r["unresolved_or_contradicted_premises"] for r in rows),
        "basketball_flags": {k: any(r[k] for r in rows) for k in (
            "basketball_new_harmful_claim_flag", "basketball_new_unknown_flag", "basketball_loss_of_old_benign_claim_flag")},
        "cost": cost,
        "next": "INSPECT_ALL_FOUR_OBSERVATIONS_NO_AUTO_DEPLOYMENT" if valid else "INSPECT_TECHNICAL_FAILURES_NO_SEMANTIC_REBUY" if complete else "EXPLICIT_APPROVAL_OR_RESUME_UNATTEMPTED_WINDOWS",
        "all_observations_unverified_not_only_enum": True, "semantic_ready": False,
        "human_review_activated": False, "formal_accuracy": None, "formal_AP": None,
        "anomaly_score_delta": None, "original_scores_changed": False,
        "graph_OT_changes_authorized": False, "training_authorized": False,
        "broader_acquisition_authorized": False, "design_exposed_development_only": True}
    write_json(out / "summary.json", summary)
    write_jsonl(out / "paired_comparison.jsonl", rows)
    # Retain every state/report snapshot, rather than selecting the most favorable response.
    identifier = semantic_sha256({"summary": summary, "rows": rows})[:20]
    immutable(out / "report_snapshots" / identifier / "summary.json", summary)
    immutable(out / "report_snapshots" / identifier / "comparisons.json", rows)
    render(out, cases, rows, summary)
    return summary


def render(out, cases, rows, summary):
    esc = lambda v: html.escape(str(v))
    def href(path):
        try:
            return quote(Path(os.path.relpath(portable(path), out)).as_posix(), safe="/")
        except ValueError:
            return portable(path).as_uri()
    sections = []
    for case, row in zip(cases, rows):
        media = read_json(out / "frozen" / case["window_uid"] / "media.json")
        frames = ''.join(f'<figure><img src="{href(p)}" alt="T{i}"><figcaption>T{i} / frame {media["frame_indices"][i]}</figcaption></figure>' for i, p in enumerate(media["image_paths"]))
        assessments = []
        for e in row["events"]:
            quoted = ''.join(f'<li><b>{esc(o["observation_id"])} / {esc(", ".join(o["frame_ids"]))}</b>: {esc(o["description"])}<br>Participants: {esc(", ".join(o["participant_ids"]))}</li>' for o in e["cited_observations"])
            assessments.append(f'<article><h3>Event {esc(e["event_id"])}</h3><p>Premise: <b>{esc(e["premise_support"])}</b>; distinction claimed: <b>{esc(e["discrimination"])}</b>.</p><p><b>Benign alternative:</b> {esc(e["benign_alternative"])}</p><p><b>Harmful alternative:</b> {esc(e["harmful_alternative"])}</p><ul>{quoted}</ul><p><b>Why this distinguishes them:</b> {esc(e["contrast_reason"])}</p><p class="limit"><b>Limits:</b> {esc(e["limits"])}</p></article>')
        frozen = "frozen/" + case["window_uid"]
        baseline = read_json(out / frozen / "same_parent_C2.json")
        old_quotes = ''.join(f'<p><b>{esc(e["event_id"])} / {esc(e["explanation_id"])}:</b> {esc(e["saved_mechanism"])}</p><blockquote>{esc(e["saved_reason"])}</blockquote>' for e in baseline["explanations"])
        old_panel = '<details><summary>Previous same-parent explanation, not verified truth</summary>' + (old_quotes or '<p>No previous benign explanation; this does not verify C1.</p>') + '</details>'
        old_values = row["old_diagnostic_only_if_valid"]
        old_panel += (f'<p>Old conditional diagnostic only: Q={esc(old_values["Q"])}; U={esc(old_values["U_conditional_on_C1"])}. No comparable new anomaly score.</p>' if old_values else '<p class="limit">Old C2 is technically invalid: no valid old diagnostic or before/after score delta.</p>')
        flags = [k for k in ("basketball_new_harmful_claim_flag", "basketball_new_unknown_flag", "basketball_loss_of_old_benign_claim_flag") if row[k]]
        flag_text = f'<p class="limit">Control flags (not measured errors): {esc(", ".join(flags))}</p>' if flags else ''
        sections.append(f'<section><h2>{esc(case["name"])}</h2><p>New: <b>{esc(row["new_status"])}</b>. Same-parent old C2 valid: <b>{row["baseline_technical_valid"]}</b>.</p><p>{esc(row["baseline_error"] or "")}</p><p>{esc(row["new_error"] or "")}</p><p class="hash">Parent: {esc(row["same_parent_sha256"])}</p><div class="frames">{frames}</div>' + old_panel + flag_text + ''.join(assessments) + f'<p><a href="{frozen}/same_parent_C2.json">Exact same-parent baseline</a> | <a href="{frozen}/different_parent_history.json">Different-parent history (not a paired effect)</a> | <a href="{frozen}/other_same_parent_history.json">Other same-parent records (not selected baseline)</a> | <a href="{frozen}/request.json">Frozen request</a></p></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.17 Four-case C2 Contrast</title><style>
body{margin:0;background:#fafcfc;color:#202925;font:15px system-ui}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:22px}h3{font-size:18px}section{border-top:2px solid #b3c3ba;padding:20px 0}article{border-top:1px solid #d6dfda;padding-top:12px;margin-top:16px}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}.limit{border-left:4px solid #b34f33;background:#fff0ea;padding:12px}p,li,.hash{overflow-wrap:anywhere}.hash{font-size:12px}a{color:#116653}li{margin:10px 0}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main><h1>Four frozen-parent C2 contrasts</h1>'''
    page += f'<p>Status: {esc(summary["status_counts"])}. Physical attempts: {summary["cost"]["physical_attempts"]}.</p><p class="limit">All four design-exposed windows are retained. Read literal observations and contrast reasons, not just enums. Same-parent does not remove prompt and sampling confounds. Unknown is not normal. No new anomaly scores, accuracy/AP or automatic deployment.</p><p><a href="summary.json">Summary</a> | <a href="paired_comparison.jsonl">All comparisons and control flags</a> | <a href="plans/{PHASE}.json">Frozen budget</a> | <a href="reference_only/human.json">Human reference (not sent to model)</a></p>'
    (out / "index.html").write_text(page + ''.join(sections) + '</main></html>', encoding="utf-8")
