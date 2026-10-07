"""Fixed four-case development recheck; never trains or updates the V9.12 run."""
from __future__ import annotations

import copy
import html
import json
from pathlib import Path
import shutil
import time

from .binding import binding_features, proposal_prompt, validate_proposal
from .contracts import iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from .role_scoped import portable
from .b1b4_trial.evidence import Budget, TrialRuntime, StopAcquisition, dashscope_once
from .b1b4_trial.features import feature_record, c1_quality
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest, now
from v912_smoke_semantic_audit import verify_packet, native_prompt, validate_native, draft_contract, costs

VERSION = "v913_fixed_four_premise_and_benignity_recheck_1"
PHASE = "diagnostic_recheck"
STAGES = ("C2_fixed_C1", "C1_reobserved", "C2_reobserved_C1")


def canonical(path):
    value = portable(path).resolve().as_posix()
    return "/mnt/" + value[0].lower() + value[2:] if len(value) > 2 and value[1] == ":" else value


def fresh_prompt(uid, signature):
    return """Independent observation contract v913. Use only the supplied eight frames.
Do not assume that an abnormal event must be present. An empty event list is valid.
Distinguish direct visible actions/states from inferred intention, event identity,
physical consequences, and background context. Contact, proximity, ordinary motion
or a scene category alone do not establish aggression or an active harmful event.
Do not attach an object's visible state to a nearby person without visible evidence.
Do not invent an unseen cause, participant interaction, control system or intent.
For every proposed event describe the literal action and cite local frame evidence.
Participants must be observed in frames supporting that event; maintain local roles.
Use uncertainty/observation_sufficient to retain ambiguity rather than fabricating
an event or treating insufficient visibility as evidence of normality.
No prior answer, reviewer conclusion, filename, category, or graph score is provided.
""" + proposal_prompt(uid, signature)


def validate_fresh(value, uid, signature):
    validate_proposal(value, window_id=uid, evidence_signature=signature)
    entities = {e["entity_id"]: e for e in value["entities"]}
    for event in value["events"]:
        for eid in event["participant_ids"]:
            if not set(entities[eid]["observed_frame_ids"]) & set(event["observed_frame_ids"]):
                raise ValueError("participant has no observed frame in its proposed event")
    return value


def diagnostic_binding(c1, c2):
    """Conditional residual only: neither premise truth nor deployed anomaly score."""
    validate_native(c1, c2)
    clone = copy.deepcopy(c2)
    clone["schema_version"] = "same_event_normal_binding_v1"
    for row in clone["event_bindings"]:
        for ex in row["explanations"]:
            if (ex["benignity_basis"] in {"unknown", "external_context_only"}
                    or ex["unexplained_anomalous_mechanism_remains"] == "unknown"):
                row["assessment_complete"] = False
    result = binding_features(c1, clone)
    return {"Q": result["Q"], "U_conditional_on_C1": result["U"],
            "observed": result["U_observed"], "events": result["events"],
            "premise_truth_verified": False, "deployment_authorized": False}


def frozen_copy(source, target):
    if target.exists():
        if stable_hash(source) != stable_hash(target):
            raise ValueError("existing snapshot differs: " + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify(out):
    seal = read_json(out / "seal.json")
    if not seal or seal["version"] != VERSION:
        raise ValueError("not a prepared V9.13 recheck")
    for rel, digest in seal["files"].items():
        if stable_hash(out / rel) != digest:
            raise ValueError("frozen recheck file changed: " + rel)
    verify_manifest(read_json(out / "source_manifest.json"))
    verify_manifest(read_json(out / "code_manifest.json"))
    protocol = read_json(out / "protocol.json")
    if canonical(out) != protocol["run_root"]:
        raise ValueError("recheck moved; use its prepared directory")
    return protocol


def prepare(project, source, out):
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("new sibling TAG required; never write inside the original run")
    if (out / "seal.json").exists():
        protocol = verify(out)
        if protocol["source_run"] != canonical(source):
            raise ValueError("source changed for existing TAG")
        return report(out)
    if any((out / p).exists() for p in ("cost", "results", "authorizations")):
        raise ValueError("unsealed run has acquisition artifacts; preserve and inspect")
    packet_root = source / "handoff/smoke_semantic_review"
    packet = verify_packet(packet_root)
    review = read_json(packet_root / "review.json")
    imported = read_json(packet_root / "review_import_summary.json")
    if (not imported or imported.get("reviewed_cases") != 4
            or imported.get("review_sha256") != stable_hash(packet_root / "review.json")
            or imported.get("cases") != review.get("cases")
            or imported.get("reviewer_id") != review.get("reviewer_id")
            or review.get("packet_sha256") != semantic_sha256(packet)
            or review.get("use_policy") != "diagnostic_only_not_training_or_gold"):
        raise ValueError("current four-case review must be successfully imported first")
    ids = packet["case_ids"]
    if len(ids) != 4 or len(set(ids)) != 4 or {r["window_uid"] for r in review["cases"]} != set(ids):
        raise ValueError("entire fixed four-case cohort required")
    enrollment = {r["window_uid"]: r for r in iter_jsonl(source / "enrollment/windows.jsonl")}
    if any(enrollment[uid]["role"] != "adaptation" for uid in ids):
        raise ValueError("locked outcomes are forbidden")
    verify_manifest(read_json(source / "seal/legacy_inventory.json"))
    source_inputs = list(read_json(packet_root / "input_manifest.json"))
    extra = [packet_root / p for p in ("review.json", "review_import_summary.json", "packet.json")]
    extra += [source / p for p in ("protocol/config.json", "enrollment/windows.jsonl", "media/manifest.json")]
    source_inputs += [{"path": canonical(p), "sha256": stable_hash(p)} for p in extra]
    immutable(out / "source_manifest.json", source_inputs)
    frozen_copy(packet_root / "review_import_summary.json", out / "review_reference/imported_review.json")
    config = read_json(source / "protocol/config.json")
    config = {key: config[key] for key in ("model", "provider_revision", "temperature", "max_output_tokens", "image_max_pixels")}
    config.update(workers=1, transport_attempts=2, schema_repair_attempts=0, semantic_retries=0)
    immutable(out / "config.json", config)
    media = read_json(source / "media/manifest.json")
    selected_media = {}
    for uid in ids:
        old = read_json(source / "private_acquisition/adaptation" / (uid + ".json"))
        validate_proposal(old["C1"], window_id=uid, evidence_signature=semantic_sha256(media[uid]["image_sha256"]))
        frozen_copy(source / "private_acquisition/adaptation" / (uid + ".json"), out / "reference" / (uid + ".json"))
        local = copy.deepcopy(media[uid])
        for name, digest in local["files"].items():
            origin = source / "media" / uid / name
            if stable_hash(origin) != digest:
                raise ValueError("original media changed")
            frozen_copy(origin, out / "media" / uid / name)
        local["image_paths"] = [canonical(out / "media" / uid / f"T{k}.jpg") for k in range(8)]
        selected_media[uid] = local
        signature = semantic_sha256(local["image_sha256"])
        immutable(out / "requests" / (uid + ".json"), {
            "C2_fixed_C1": native_prompt(old["C1"]), "C1_reobserved": fresh_prompt(uid, signature),
            "C2_reobserved_C1": "resolved from the canonical, valid new C1; never reviewer text"})
    immutable(out / "media/manifest.json", selected_media)
    immutable(out / "C2_CONTRACT.json", draft_contract())
    write_jsonl(out / "enrollment/role_map.jsonl", [{"window_uid": uid, "source_group": enrollment[uid]["source_group"]} for uid in ids])
    own_code = [Path(__file__), project / "tools/smoke_recheck_v913_cli.py", project / "run_smoke_recheck_v913.sh",
                project / "docs/v912_smoke_semantic_audit.py", project / "docs/v912_prepared_run_handoff.py"]
    code = [r for r in read_json(source / "seal/legacy_inventory.json") if r.get("use") == "implementation_reference" and r["path"].endswith(".py")]
    code += [{"path": canonical(p), "sha256": stable_hash(p)} for p in own_code]
    immutable(out / "code_manifest.json", code)
    immutable(out / "protocol.json", {"version": VERSION, "source_run": canonical(source), "run_root": canonical(out),
        "case_ids": ids, "stages": list(STAGES), "review_use": "diagnostic_design_reference_only_not_gold_or_model_input",
        "new_graph_or_C0_calls": 0, "same_evidence_semantic_retries": 0, "formal_accuracy": None, "formal_AP": None,
        "training_authorized": False, "adaptation_authorized": False, "deployment_authorized": False})
    plan = {"version": VERSION, "phase": PHASE, "windows": [{"window_uid": uid} for uid in ids],
            "logical_requests_upper_bound": 12, "physical_attempts_upper_bound": 24,
            "output_tokens_upper_bound": 24 * config["max_output_tokens"], "workers": 1,
            "format_repairs": 0, "transport_attempts_per_request_max": 2,
            "input_tokens": None, "human_answers_in_model_input": False,
            "empty_new_C1": "skip dependent C2, preserving observation sufficiency mask"}
    immutable(out / "plans" / (PHASE + ".json"), plan)
    files = {p.relative_to(out).as_posix(): stable_hash(p) for p in out.rglob("*") if p.is_file() and p.name != ".operation.lock"}
    immutable(out / "seal.json", {"version": VERSION, "files": files})
    immutable(out / "authorizations" / (PHASE + ".json"), {"approved_by": "", "authorized": False,
        "plan_sha256": stable_hash(out / "plans" / (PHASE + ".json")), "max_physical_attempts": None,
        "max_output_tokens": None, "resume_uncertain_attempts": False})
    return report(out)


def authorize(out, approved_by, cap):
    verify(out)
    if not approved_by or approved_by.strip().lower() in {"pending", "your_name"} or type(cap) is not int or not 1 <= cap <= 24:
        raise ValueError("explicit approver and MAX_ATTEMPTS=1..24 required")
    path = out / "authorizations" / (PHASE + ".json")
    old = read_json(path)
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    if any(r["status"] == "in_flight" for r in attempts):
        raise ValueError("uncertain in-flight billing: inspect receipts; no automatic acknowledgement")
    if cap < len(attempts):
        raise ValueError("cap cannot be smaller than already reserved physical attempts")
    config = read_json(out / "config.json")
    value = {**old, "authorized": True, "approved_by": approved_by.strip(), "max_physical_attempts": cap,
             "max_output_tokens": cap * config["max_output_tokens"]}
    if old.get("authorized") and value != old:
        raise ValueError("authorization already frozen; do not reset budget on resume")
    if value != old:
        write_json(path, value)
        write_json(out / "authorization_receipts" / (str(time.time_ns()) + ".json"), {"previous": old, "next": value, "at": now()})


def empty_result(c1):
    q = c1_quality(c1)
    return {"status": "skipped_empty_C1", "diagnostic": {"Q": q, "U_conditional_on_C1": q,
            "observed": q is not None, "premise_truth_verified": False, "deployment_authorized": False},
            "reason": "complete_empty" if q is not None else "insufficient_or_incomplete_observation"}


def save_result(out, uid, stage, result):
    path = out / "results" / uid / (stage + ".json")
    immutable(path, result)
    immutable(out / "result_receipts" / uid / (stage + ".json"), {"sha256": stable_hash(path)})


def load_result(out, uid, stage):
    path = out / "results" / uid / (stage + ".json")
    if not path.exists():
        return {"status": "pending"}
    receipt = read_json(out / "result_receipts" / uid / (stage + ".json"), {})
    if receipt.get("sha256") != stable_hash(path):
        raise ValueError("result receipt missing or changed; inspect without rebuying: " + str(path))
    value = read_json(path)
    verify_manifest(value.get("raw_manifest", []))
    return value


def run(out, provider=dashscope_once):
    protocol = verify(out)
    config = read_json(out / "config.json")
    budget = Budget(out, PHASE, config)
    media = read_json(out / "media/manifest.json")
    try:
        for uid in protocol["case_ids"]:
            old = read_json(out / "reference" / (uid + ".json"))
            prompts = read_json(out / "requests" / (uid + ".json"))
            runtime = TrialRuntime(out, uid, media[uid], config, budget, provider=provider)
            signature = semantic_sha256(media[uid]["image_sha256"])
            for stage in STAGES:
                target = out / "results" / uid / (stage + ".json")
                if target.exists():
                    load_result(out, uid, stage)
                    continue
                runtime.keys = []
                c1 = old["C1"]
                if stage == "C2_reobserved_C1":
                    fresh = load_result(out, uid, "C1_reobserved")
                    if fresh["status"] != "valid":
                        save_result(out, uid, stage, {"status": "blocked_by_invalid_C1", "raw_manifest": []})
                        continue
                    c1 = fresh["parsed"]
                    if not c1["events"]:
                        save_result(out, uid, stage, empty_result(c1))
                        continue
                is_c1 = stage == "C1_reobserved"
                prompt = prompts[stage] if stage != "C2_reobserved_C1" else native_prompt(c1)
                runtime.parents = [] if is_c1 else [semantic_sha256(c1)]
                write_json(out / "resolved_requests" / uid / (stage + ".json"), {
                    "prompt": prompt, "C1_parent_sha256": None if is_c1 else semantic_sha256(c1),
                    "image_sha256": media[uid]["image_sha256"], "human_answers_in_input": False})
                validator = (lambda p: validate_fresh(p, uid, signature)) if is_c1 else (lambda p: validate_native(c1, p))
                try:
                    response = runtime.request_json(case=None, prompt=prompt, namespace=stage, validator=validator)
                    result = {"status": "valid", "parsed": response["parsed"]}
                    if not is_c1:
                        result["diagnostic"] = diagnostic_binding(c1, response["parsed"])
                except StopAcquisition:
                    raise
                except Exception as exc:
                    result = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)[:1000]}
                result["raw_manifest"] = [{"path": canonical(out / "cache/raw" / (k + ".json")),
                    "sha256": stable_hash(out / "cache/raw" / (k + ".json"))} for k in runtime.keys if (out / "cache/raw" / (k + ".json")).exists()]
                save_result(out, uid, stage, result)
                print(f"[v913] {uid[:12]} {stage}: {result['status']}; attempts={budget.used}", flush=True)
            report(out)
    except StopAcquisition as exc:
        write_json(out / "last_pause.json", {"reason": str(exc), "at": now(), "resume_resets_budget": False})
        report(out)
        raise
    return report(out)


def report(out):
    protocol = verify(out)
    rows = []
    for uid in protocol["case_ids"]:
        old = read_json(out / "reference" / (uid + ".json"))
        stages = {stage: load_result(out, uid, stage) for stage in STAGES}
        rows.append({"window_uid": uid, "old_values": feature_record(uid, old)["values"], "stages": stages})
    terminal = all(v["status"] != "pending" for r in rows for v in r["stages"].values())
    valid = all(v["status"] in {"valid", "skipped_empty_C1"} for r in rows for v in r["stages"].values())
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    logical = [read_json(p) for p in (out / "cost/logical_requests").glob("*.json")]
    summary = {"version": VERSION, "at": now(), "windows": 4, "complete_accounting": terminal,
        "technical_ready_for_all_four_comparison": valid, "per_window": rows, "cost": costs(attempts, logical),
        "next": "INSPECT_ALL_FOUR_PREMISE_AND_BENIGNITY_CHANGES" if valid else "INSPECT_FAILURES_NO_SEMANTIC_RETRY" if terminal else "WAITING_FOR_AUTHORIZED_FIXED_RECHECK",
        "formal_accuracy": None, "formal_AP": None, "human_truth_verified": False,
        "old_scores_changed": False, "adaptation_authorized": False, "training_authorized": False}
    write_json(out / "summary.json", summary)
    render(out, rows)
    return {k:v for k,v in summary.items() if k != "per_window"}


def render(out, rows):
    esc = lambda x: html.escape(str(x))
    review = read_json(out / "review_reference/imported_review.json")
    humans = {r["window_uid"]: r for r in review["cases"]}
    parts = []
    for row in rows:
        uid = row["window_uid"]
        old = read_json(out / "reference" / (uid + ".json"))
        frames = ''.join(f'<figure><img src="media/{uid}/T{k}.jpg" alt="T{k}"><figcaption>T{k}</figcaption></figure>' for k in range(8))
        stage_rows = []
        for stage, val in row["stages"].items():
            d = val.get("diagnostic", {})
            stage_rows.append(f'<tr><td>{stage}</td><td>{esc(val["status"])}</td><td>{esc(d.get("Q", "not applicable"))}</td><td>{esc(d.get("U_conditional_on_C1", "not applicable"))}</td></tr>')
        records = {"R3 diagnostic reference (not gold)": humans[uid], "Original C0": old["C0"], "Original C1": old["C1"], "Original C2": old["C2"], **row["stages"]}
        details = ''.join(f'<details><summary>{esc(k)}</summary><pre>{esc(json.dumps(v,ensure_ascii=False,indent=2))}</pre></details>' for k,v in records.items())
        parts.append(f'<section><h2>{esc(uid[:12])}</h2><video controls preload="metadata" src="media/{uid}/clip.mp4"></video><div class="frames">{frames}</div><p>Original Q={row["old_values"]["c1_Q"]:.6f}; U={row["old_values"]["binding_U"]:.6f}. Original graph scores are unchanged.</p><table><thead><tr><th>Branch</th><th>Status</th><th>Q</th><th>U conditional on C1</th></tr></thead><tbody>{"".join(stage_rows)}</tbody></table>{details}</section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.13 Four-Case Recheck</title><style>body{margin:0;background:#fafcfc;color:#192326;font:15px system-ui}main{max-width:1050px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px}section{padding:24px 0;border-top:1px solid #ccd3d3}video{width:100%;max-height:420px;background:#111}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:12px 0}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ccc;padding:8px;overflow-wrap:anywhere}details{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f1;padding:12px}aside{background:#fff1e9;border-left:4px solid #b45532;padding:12px}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}table{font-size:12px}td,th{padding:4px}}</style><main><h1>V9.13 Four-Case Recheck</h1><aside>Development diagnostic only. A higher residual is not necessarily better. False event premises can also create high scores. No accuracy, deployment or training approval.</aside>' + ''.join(parts) + '</main></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
