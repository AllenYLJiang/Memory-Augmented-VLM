"""Offline evidence-sufficiency audit. No provider, score update, or review task."""
from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from . import claim_scope_followup as followup
from .contracts import read_json, write_jsonl, semantic_sha256
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest
from .role_scoped import portable

trial = followup.trial
prior = trial.prior
audit = trial.audit
VERSION = "v916_binding_is_not_benignity_offline_1"
SCOPES = {"unassessed", "context_only", "action_compatible", "discriminative_action", "unknown"}
CONTRACT = {
    "version": VERSION,
    "status": "offline_draft_not_a_new_model_response",
    "separate_questions": [
        "Does the explanation refer to the same participants, time and event?",
        "Does its cited local evidence distinguish a benign mechanism from a harmful alternative?",
        "Is the original C1 premise itself visually established?",
    ],
    "support_scope": sorted(SCOPES),
    "meaning": {
        "unassessed": "Legacy response did not answer this question. Never infer an answer from its score.",
        "context_only": "Setting, uniforms or roles, without a discriminating local action.",
        "action_compatible": "Local action fits a benign explanation, but also fits the alternative.",
        "discriminative_action": "A concrete local detail is claimed to distinguish the alternatives; still needs semantic verification.",
        "unknown": "Insufficient evidence to classify the support.",
    },
    "no_automatic_semantic_typing": True,
    "no_keyword_classifier": True,
    "no_probability_from_text_or_human_answers": True,
    "legacy_score_is_not_benignity_verification": True,
    "empty_explanations_do_not_verify_C1": True,
    "unknown_is_not_normal_or_abnormal": True,
    "syntactic_success_does_not_prove_visual_truth": True,
    "deployment_authorized": False,
}

CASE_NOTES = {
    "edb6091cadf8": {
        "name": "Hockey",
        "finding": "V9.15 N1 describes rink/equipment/uniform context. The reason adds body-check legality/routine-play and recovery claims; these are not separately grounded as discriminating normal evidence. Same-event binding does not settle benignity.",
        "unresolved": "Whether this particular contact/stick action is benign; sporting legality and severity are not established by the record or by the reviewer adjective alone.",
        "do_not": "Do not change a score to match R1/R3, and do not treat recovery as proof that an action was benign.",
    },
    "01dfb1e8b720": {
        "name": "Street fire",
        "finding": "Both V9.13 C2 variants return no benign explanations. This leaves the premise, including the identity of what is burning, unverified.",
        "unresolved": "Flames/smoke and a burning-person identity are different claims. Stationarity alone does not exclude a person.",
        "do_not": "Do not call an empty explanation a verified true positive, or replace an uncertain identity with a fabricated object.",
    },
    "e5896297709f": {
        "name": "Vehicle",
        "finding": "The fixed-parent C2 is structurally valid but infers active automation and demonstration intent. The reobserved-parent C2 also has a preserved cross-namespace reference failure.",
        "unresolved": "R1 marks central-screen control evidence uncertain. Display appearance, active mode, intent and safety are distinct claims.",
        "do_not": "Do not interpret unknown control state as either disabled control or safe driving; do not repair cross-scope evidence by silently merging IDs.",
    },
    "be18a4a445c6": {
        "name": "Basketball",
        "finding": "R3 reports calm walking/standing. The model adds scuffle/intervention or timeout/substitution/minor-foul interpretations across versions.",
        "unresolved": "Ordinary gathering can remain supported without proving a particular stoppage cause. The original direct-event premise also needs attention.",
        "do_not": "Do not remove all sport explanations to fix hockey; preserve this normal-looking control and distinguish literal observations from the named cause.",
    },
}


def explanations(c2):
    return [(row["event_id"], ex) for row in c2["event_bindings"] for ex in row["explanations"]]


def empty_assessment(c1, c2):
    return {"version": VERSION, "C1_sha256": semantic_sha256(c1), "C2_sha256": semantic_sha256(c2),
        "explanations": [{"event_id": eid, "explanation_id": ex["explanation_id"],
            "support_scope": "unassessed", "evidence_ids": [], "frame_ids": [],
            "literal_observation": "", "discriminating_detail": "", "limitations": ""}
            for eid, ex in explanations(c2)]}


def validate_assessment(c1, c2, value):
    """Validate obligations, NOT whether a claimed discriminator is visually true."""
    prior.validate_native(c1, c2)
    if not isinstance(value, dict) or set(value) != {"version", "C1_sha256", "C2_sha256", "explanations"}:
        raise ValueError("exact assessment fields required")
    if (value["version"] != VERSION or value["C1_sha256"] != semantic_sha256(c1)
            or value["C2_sha256"] != semantic_sha256(c2)):
        raise ValueError("assessment parent/version mismatch")
    expected = {(eid, ex["explanation_id"]): ex for eid, ex in explanations(c2)}
    if len(expected) != len(explanations(c2)):
        raise ValueError("duplicate source explanation")
    rows = value["explanations"]
    fields = {"event_id", "explanation_id", "support_scope", "evidence_ids", "frame_ids",
              "literal_observation", "discriminating_detail", "limitations"}
    if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != fields for r in rows):
        raise ValueError("exact explanation fields required")
    keys = [(r["event_id"], r["explanation_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("assess every explanation once; no selection by outcome")
    events = {e["event_id"]: e for e in c1["events"]}
    assessments = {b["event_id"]: b["assessment_complete"] for b in c2["event_bindings"]}
    evidence = {e["evidence_id"]: e for e in c2["normal_evidence"]}
    result = []
    for row in rows:
        ex = expected[(row["event_id"], row["explanation_id"])]
        scope = row["support_scope"]
        if scope not in SCOPES:
            raise ValueError("unknown support scope")
        for field in ("evidence_ids", "frame_ids"):
            refs = row[field]
            if not isinstance(refs, list) or any(type(x) is not str for x in refs) or len(refs) != len(set(refs)):
                raise ValueError("unique string references required")
        if not set(row["evidence_ids"]) <= set(ex["normal_evidence_ids"]):
            raise ValueError("evidence must be cited by this C2 explanation")
        frames = set(row["frame_ids"])
        pool = set().union(*(set(evidence[e]["frame_ids"]) for e in row["evidence_ids"]))
        allowed = pool & set(ex["observed_frame_ids"]) & set(events[row["event_id"]]["observed_frame_ids"])
        if not frames <= allowed:
            raise ValueError("frame not jointly supported by event, explanation and evidence")
        if any(not set(evidence[e]["frame_ids"]) & frames for e in row["evidence_ids"]):
            raise ValueError("each reference needs a cited local frame")
        for field in ("literal_observation", "discriminating_detail", "limitations"):
            if type(row[field]) is not str:
                raise ValueError("text must be a string")
        if scope == "unassessed" and any(row[f] for f in fields - {"event_id", "explanation_id", "support_scope"}):
            raise ValueError("unassessed cannot conceal an inferred assessment")
        if scope != "unassessed" and not row["limitations"].strip():
            raise ValueError("explicit evidence limits required")
        if scope not in {"unassessed", "unknown"} and (not frames or not row["literal_observation"].strip()):
            raise ValueError("non-unknown support needs literal local evidence")
        if scope == "discriminative_action" and not row["discriminating_detail"].strip():
            raise ValueError("discriminative support must identify the contrast")
        binding = all(ex[k] == "supported" for k in ("same_participant", "same_time", "same_event"))
        complete = c2["complete"] and assessments[row["event_id"]]
        eligible = None if scope in {"unassessed", "unknown"} or not complete else scope == "discriminative_action" and binding
        result.append({"event_id": row["event_id"], "explanation_id": row["explanation_id"],
            "same_event_binding_claimed": binding, "support_scope": scope,
            "structurally_eligible_for_semantic_check": eligible,
            "semantic_truth_verified": False, "benignity_score": None,
            "suppress_direct_event_authorized": False, "anomaly_label": None})
    return result


def record(uid, stage, c1, c2, origin, old_status, old_diagnostic):
    native = audit.validate_status(prior.validate_native, c1, c2)
    if stage == "v912_original":
        native = audit.validate_status(prior.binding_features, c1, c2)
    trace = []
    defs = {e["evidence_id"]: e for e in c2.get("normal_evidence", [])}
    for eid, ex in explanations(c2):
        trace.append({"event_id": eid, "explanation_id": ex["explanation_id"],
            "same_event_binding_claimed": all(ex[k] == "supported" for k in ("same_participant", "same_time", "same_event")),
            "saved_bound_support_score": ex["bound_support_score"],
            "saved_benignity_basis": ex.get("benignity_basis"),
            "saved_reason": ex["reason"], "saved_mechanism": ex["mechanism"],
            "explanation_frame_ids": ex["observed_frame_ids"],
            "cited_evidence": [{"evidence_id": e, "declared_scope": "C2", "record": defs.get(e)} for e in ex["normal_evidence_ids"]],
            "discriminative_support": None, "discriminative_support_observed": False,
            "reason": "Not explicitly assessed under the new contract; no automatic text classification.",
            "semantic_truth_verified": False, "new_score": None})
    return {"window_uid": uid, "stage": stage, "origin": prior.canonical(origin),
        "origin_sha256": stable_hash(origin), "C1_sha256": semantic_sha256(c1), "C2_sha256": semantic_sha256(c2),
        "saved_status": old_status, "technical_validation": native,
        "saved_diagnostic_not_recomputed": old_diagnostic,
        "C1_premise_truth_verified": False, "empty_explanations_verify_C1": False,
        "explanations": trace, "C1": c1, "C2": c2,
        "anomaly_label": None, "new_score": None}


def collect(source):
    protocol = trial.verify(source)
    review = trial.accepted_review(source)
    result = trial.get_result(source)
    if not result or result["status"] != "valid":
        raise ValueError("completed valid single V9.15 C2 required; no request is made here")
    followup.verify_result_payload(source, result)
    c1 = read_json(source / "parent_C1.json")
    comparison = followup.compare(c1, result, next(r["assessment"] for r in review["claims"] if r["claim_id"] == "hockey_local_benignity"))
    previous_audit = portable(protocol["source_audit"]).resolve()
    v913 = portable(read_json(previous_audit / "input_binding.json")["source"]).resolve()
    old_protocol = prior.verify(v913)
    # These checks read sealed outputs only; they cannot acquire or rebuild old reports.
    completion = read_json(previous_audit / "completion.json")
    if completion["output_manifest_sha256"] != stable_hash(previous_audit / "output_manifest.json"):
        raise ValueError("V9.14 output seal changed")
    verify_manifest(read_json(previous_audit / "output_manifest.json"))
    ids = old_protocol["case_ids"]
    if len(ids) != 4 or {u[:12] for u in ids} != set(CASE_NOTES):
        raise ValueError("fixed full four-case cohort required")
    rows = []
    for uid in ids:
        path = v913 / "reference" / (uid + ".json")
        old = read_json(path)
        rows.append(record(uid, "v912_original", old["C1"], old["C2"], path, "saved_valid", old.get("values")))
    for path in sorted((previous_audit / "response_audits").glob("*.json")):
        item = read_json(path)
        if not item["stage"].startswith("C2_"):
            continue
        uid = item["window_uid"]
        parent = (read_json(v913 / "reference" / (uid + ".json"))["C1"] if item["stage"] == "C2_fixed_C1"
                  else prior.load_result(v913, uid, "C1_reobserved")["parsed"])
        saved = prior.load_result(v913, uid, item["stage"])
        rows.append(record(uid, "v913_" + item["stage"], parent, item["original_payload"], path,
                           item["original_status"], saved.get("diagnostic")))
    rows.append(record(c1["window_id"], "v915_missing_C2", c1, result["parsed"], source / "result.json",
                       "valid", result["diagnostic_conditional_not_anomaly_score"]))
    if len(rows) != 12 or Counter(r["window_uid"] for r in rows) != Counter({uid: 3 for uid in ids}):
        raise ValueError("expected all 12 saved C2 records: four original, seven V9.13, one V9.15")
    if any(r["saved_status"] in {"valid", "saved_valid"} and not r["technical_validation"]["valid"] for r in rows):
        raise ValueError("previously valid C2 fails unchanged validator")
    return rows, review, comparison, v913


def build(project, source, out):
    source, out = source.resolve(), out.resolve()
    followup.check_out(source, out)
    roots = [source] + [portable(p).resolve() for p in read_json(source / "protocol.json")["source_roots"]]
    before = audit.snapshot(roots)
    rows, review, comparison, v913 = collect(source)
    code = [Path(__file__), project / "tools/benignity_audit_v916_cli.py", project / "run_benignity_audit_v916.sh"]
    binding = {"version": VERSION, "source": prior.canonical(source), "source_sha256": semantic_sha256(before),
        "code": [{"path": prior.canonical(p), "sha256": stable_hash(p)} for p in code], "contract": CONTRACT,
        "analyst_notes": CASE_NOTES}
    if (out / "completion.json").exists():
        completion = read_json(out / "completion.json")
        if read_json(out / "input_binding.json") != binding or completion["output_manifest_sha256"] != stable_hash(out / "output_manifest.json"):
            raise ValueError("source/code/output changed; preserve TAG and use a new one")
        verify_manifest(read_json(out / "output_manifest.json"))
        if audit.snapshot(roots) != before:
            raise ValueError("source changed during verification")
        return read_json(out / "summary.json")
    if not (out / "input_binding.json").exists() and any(p.name != ".operation.lock" for p in out.iterdir()):
        raise ValueError("new empty output required")
    immutable(out / "input_binding.json", binding)
    immutable(out / "source_snapshot.json", before)
    immutable(out / "minimal_benignity_contract.json", CONTRACT)
    immutable(out / "human_reference.json", {"review": review, "source_sha256": stable_hash(source / "review_import/accepted_review.json"),
        "older_R3": read_json(v913 / "review_reference/imported_review.json"), "diagnostic_only": True})
    immutable(out / "hockey_completed_dependency.json", comparison)
    write_jsonl(out / "all_C2_records.jsonl", rows)
    notes = [{"window_uid": uid, **CASE_NOTES[uid[:12]], "author": "assistant_textual_audit_not_new_visual_or_human_truth",
              "applies_to": [{"stage": r["stage"], "C2_sha256": r["C2_sha256"]} for r in rows if r["window_uid"] == uid]}
             for uid in sorted({r["window_uid"] for r in rows})]
    immutable(out / "case_scope_notes.json", {"notes": notes, "used_for_scoring": False})
    templates = 0
    for row in rows:
        if row["stage"] == "v912_original" or not row["technical_validation"]["valid"]:
            continue
        value = empty_assessment(row["C1"], row["C2"])
        validated = validate_assessment(row["C1"], row["C2"], value)
        immutable(out / "unassessed_traces" / (row["window_uid"][:12] + "_" + row["stage"] + ".json"),
                  {"template": value, "structural_readout": validated, "human_task_activated": False})
        templates += 1
    costs = prior.costs([read_json(p) for p in (source / "cost/attempts").glob("*.json")],
                        [read_json(p) for p in (source / "cost/logical_requests").glob("*.json")])
    if costs != read_json(source / "summary.json")["cost"]:
        raise ValueError("saved V9.15 cost summary differs from receipt accounting")
    explanation_rows = [ex for row in rows for ex in row["explanations"]]
    summary = {"version": VERSION, "fixed_windows": 4, "saved_C2_records": len(rows),
        "records_by_stage": dict(Counter(r["stage"] for r in rows)),
        "technical_valid_C2_records": sum(r["technical_validation"]["valid"] for r in rows),
        "invalid_records_preserved": sum(not r["technical_validation"]["valid"] for r in rows),
        "explanation_records": len(explanation_rows),
        "same_event_binding_claimed": sum(e["same_event_binding_claimed"] for e in explanation_rows),
        "explicitly_assessed_discriminative_support": 0,
        "unassessed_trace_files": templates,
        "hockey": comparison, "completed_C2_cost": costs,
        "original_files_verified_unchanged": len(before), "new_API_calls": 0, "new_media_decodes": 0,
        "new_review_tasks": 0, "original_scores_changed": False, "formal_accuracy": None, "formal_AP": None,
        "remote_execution_authorized": False, "graph_OT_changes_authorized": False, "training_authorized": False,
        "next": "INSPECT_EVIDENCE_SUFFICIENCY_DRAFT_NOT_ANOTHER_SEMANTIC_RETRY"}
    immutable(out / "summary.json", summary)
    immutable(out / "next_step_plan.json", {
        "status": "draft_only_no_paid_runner", "existing_budget_inherited": False,
        "immediate_action": "Close the missing dependency. Inspect all-four evidence audit; do not rerun V9.15.",
        "before_any_new_request": [
            "Freeze a minimal contrast: same-event compatibility versus action-level benign discrimination; no extra category taxonomy.",
            "If approved later, use the entire fixed four-case panel, identical eight frames and exact frozen C1 parents; preserve invalid vehicle C2 as an old failure.",
            "Do not supply R1/R3 answers or desired labels. Report all outcomes, including more unknowns and harm to the basketball control.",
            "Compare only against the corresponding same-parent C2; never attribute different-parent or stochastic changes to this contract alone.",
            "Inspect actual cited observations, not only a model's discriminative_action flag. Stable syntax does not authorize scoring.",
        ],
        "possible_future_C2_only_logical_calls": 4, "authorized_calls": 0,
        "new_C0_C1_graph_discovery_calls": 0, "human_review_required_now": False,
        "success_is_not_higher_positive_rate": True,
        "claim_limit": "Four design-exposed windows cannot estimate generalization, AP or accuracy.",
    })
    render(out, rows, notes, read_json(v913 / "media/manifest.json"), summary)
    if audit.snapshot(roots) != before:
        raise ValueError("original files changed during offline audit")
    outputs = [{"path": prior.canonical(p), "sha256": stable_hash(p)} for p in sorted(out.rglob("*"))
               if p.is_file() and p.name not in {".operation.lock", "output_manifest.json", "completion.json"}]
    immutable(out / "output_manifest.json", outputs)
    immutable(out / "completion.json", {"output_manifest_sha256": stable_hash(out / "output_manifest.json")})
    return summary


def render(out, rows, notes, media, summary):
    esc = lambda v: html.escape(str(v))
    def href(path):
        try:
            return quote(Path(os.path.relpath(portable(path), out)).as_posix(), safe="/")
        except ValueError:
            return portable(path).as_uri()
    sections = []
    for note in notes:
        uid = note["window_uid"]
        frames = ''.join(f'<figure><img src="{href(p)}" alt="T{i}"><figcaption>T{i} / frame {media[uid]["frame_indices"][i]}</figcaption></figure>' for i, p in enumerate(media[uid]["image_paths"]))
        records = []
        for row in (r for r in rows if r["window_uid"] == uid):
            body = []
            for ex in row["explanations"]:
                quotes = ''.join(f'<li><b>{esc(e["evidence_id"])} (C2)</b>: {esc(e["record"]["description"] if e["record"] else "UNRESOLVED IN C2")}</li>' for e in ex["cited_evidence"])
                body.append(f'<h4>{esc(ex["event_id"])} / {esc(ex["explanation_id"])}</h4><p>{esc(ex["saved_mechanism"])}</p><p>Saved binding support: <b>{esc(ex["saved_bound_support_score"])}</b>. Discriminative support: <b>unassessed</b>.</p><ul>{quotes}</ul><details><summary>Model reasoning (not verified facts)</summary><p>{esc(ex["saved_reason"])}</p></details>')
            status = row["technical_validation"]
            records.append(f'<article><h3>{esc(row["stage"])}</h3><p>Reference/format check: <b>{"valid" if status["valid"] else "INVALID"}</b> {esc(status["error"] or "")}</p>' + (''.join(body) or '<p>No explanation. This does not verify the C1 premise or establish anomaly.</p>') + '</article>')
        sections.append(f'<section><h2>{esc(note["name"])}</h2><p>{esc(note["finding"])}</p><p class="limit">{esc(note["unresolved"])}</p><div class="frames">{frames}</div>' + ''.join(records) + f'<p>{esc(note["do_not"])}</p></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.16 Evidence Sufficiency Audit</title><style>
body{margin:0;background:#fafcfc;color:#202925;font:15px system-ui}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:22px}h3{font-size:18px}h4{font-size:16px}section{border-top:2px solid #b3c3ba;padding:20px 0}article{border-top:1px solid #d5dfda;margin-top:18px;padding-top:12px}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}.limit{border-left:4px solid #b34f33;padding:12px;background:#fff0ea}li{margin:8px 0}p,li,h3,h4{overflow-wrap:anywhere}a{color:#116653}details{margin-bottom:18px}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main><h1>Binding is not benignity</h1>'''
    saved = summary["hockey"]["diagnostic_conditional_not_anomaly_score"]
    page += f'<p>All four windows; {summary["saved_C2_records"]} saved C2 records. No API calls, no new review, no score changes.</p><p class="limit">Hockey: saved Q = {esc(saved["Q"])}; saved conditional residual = {esc(saved["U_conditional_on_C1"])}. These are not calibrated anomaly probabilities. The cited N1 is scene context, while the reason asserts routine play. This is an evidence-sufficiency concern, not a measured classification error.</p><p><a href="summary.json">Summary</a> | <a href="all_C2_records.jsonl">All records</a> | <a href="minimal_benignity_contract.json">Draft contract</a> | <a href="next_step_plan.json">Next-step limits</a></p>'
    (out / "index.html").write_text(page + ''.join(sections) + '</main></html>', encoding="utf-8")
