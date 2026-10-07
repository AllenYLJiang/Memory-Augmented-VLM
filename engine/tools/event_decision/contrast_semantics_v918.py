"""Offline, claim-scoped review of the complete frozen V9.17 panel."""
from __future__ import annotations

from collections import Counter
import copy
import html
import json
from pathlib import Path
import shutil

from . import contrast_c2_v917 as source_trial
from .contracts import read_json, write_json, write_jsonl, semantic_sha256, iter_jsonl
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest
from .role_scoped import portable

VERSION = "v918_four_window_semantic_scope_audit_1"
USE = "diagnostic_subclaim_only_not_window_labels_or_training"
DEFAULT_SOURCE = "governed_v917_four_C2_contrast_20260917"
DEFAULT_TAG = "governed_v918_contrast_semantic_audit_20260917"
FIXED_UIDS = {
    "Hockey": "edb6091cadf84583ee6ac99cf67cd7f67d338a6b089cbeb95d90776cc38cea90",
    "Street fire": "01dfb1e8b7208657a6aa60f460fdb448d60afa2c8e33271ceadc63a627d4ef8a",
    "Vehicle": "e5896297709fc4b12ed74a3a2fce6a66c8b1841c890620936b8c74c91f1b2067",
    "Basketball": "be18a4a445c66c95a245d94ccbb468f03dc7199ac4c320f4018605dcecca3bae",
}
# These are analyst-written questions, not automatic visual verdicts or new prompts.
QUESTIONS = {
    "Hockey": [
        ("hockey_contact_fall", "The blue player makes physical contact with the white player, who then falls at the boards.",
         "Review the visible interaction only, not intent, rule legality or a benchmark label.",
         ["/observations/0/description", "/observations/2/description"], ["1"]),
        ("hockey_stick_height", "The blue player's stick is raised above shoulder height during the contact.",
         "Stick height is separate from stick-to-person contact; support here does not prove high-sticking.",
         ["/observations/1/description"], ["1"]),
    ],
    "Street fire": [
        ("fire_burning_person", "The burning target on the left can be visually identified as a person.",
         "Flames or a person-like silhouette alone may leave identity uncertain. Stationarity does not prove an object either.",
         ["/observations/0/description", "/event_assessments/0/contrast_reason"], ["evt_1"]),
        ("fire_smoke", "Thick white smoke near ground level obscures part of the scene.",
         "This does not establish its substance, source, authorization or harmlessness.",
         ["/observations/1/description"], ["evt_2"]),
        ("fire_projectile", "The visible trail can be bound across frames to the motion of a projectile.",
         "A streak or smoke shape is not automatically a tracked projectile. Use uncertain when continuity is not visible.",
         ["/observations/2/description"], ["evt_2"]),
        ("fire_people_moving", "People in the background move quickly away from the central area.",
         "Movement can be visible while its cause, motivation or the danger level remains unknown.",
         ["/observations/3/description"], ["evt_3"]),
    ],
    "Vehicle": [
        ("vehicle_cluster_content", "The instrument cluster BEHIND THE STEERING WHEEL visibly shows a vehicle between blue lane lines.",
         "Inspect this specific display region. The earlier R1 review concerned the LARGE CENTRAL TOUCHSCREEN, not this region.",
         ["/observations/1/description"], ["event_01"]),
        ("vehicle_control_active", "These frames establish that automatic lateral steering control is actually engaged.",
         "A display depiction and actual control engagement are different claims. Even support here would not establish safety or attention.",
         ["/event_assessments/0/contrast_reason", "/event_assessments/0/limits"], ["event_01"]),
    ],
    "Basketball": [
        ("basketball_ground_person", "A distinct person low to the ground or seated can actually be identified in the cluster.",
         "Do not infer a hidden person solely from somebody bending over. This claim does not establish injury.",
         ["/observations/0/description"], ["EVENT1"]),
        ("basketball_direct_help", "A concrete checking-on or helping-up action toward that person is visible.",
         "Calm bystanders and a walking referee are context, not direct proof of help or injury. Unknown does not mean violence.",
         ["/event_assessments/0/benign_alternative", "/event_assessments/0/contrast_reason"], ["EVENT1"]),
    ],
}
SCOPE_NOTES = {
    "Hockey": "Unknown now concerns stick contact/rule legality, not necessarily whether force and a fall are visible. Retreat from a benign claim is not an anomaly-score improvement.",
    "Street fire": "Accidental ignition and self-immolation are not inherently benign. Authorization does not establish harmlessness. A safety reaction does not explain away the surrounding hazard. The burning-person premise is still asserted, not independently verified.",
    "Vehicle": "The reason asserts engaged steering control while limits say engagement cannot be verified. The new instrument-cluster claim is not the same region as the old R1 touchscreen review. No valid same-parent numeric baseline exists.",
    "Basketball": "No enum-level regression was flagged. Nevertheless, a fallen/injured person and a helping action are additional claims, not established by passive bystanders. A benign enum alone is not a successful semantic validation.",
}
RULES = {
    "version": VERSION,
    "scope": USE,
    "unknown_dimensions": ["object_identity", "local_action", "cross_frame_binding", "cause", "intent", "legality", "control_engagement"],
    "analyst_notes_are_not_truth": True,
    "unknown_cause_or_intent_does_not_negate_visible_event": True,
    "accident_self_harm_or_authorization_does_not_imply_benignity": True,
    "context_compatibility_is_not_direct_local_mechanism": True,
    "review_subclaim_does_not_validate_entire_observation_or_event": True,
    "unknown_is_masked_not_normal": True,
    "no_model_or_score_interface_changed": True,
    "paid_runner_available": False,
    "human_review_does_not_authorize_acquisition": True,
}


def pointer(value, path):
    for key in path.split("/")[1:]:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def boundaries():
    return {"new_API_calls": 0, "new_video_decodes": 0, "original_scores_changed": False,
        "formal_accuracy": None, "formal_AP": None, "anomaly_score_delta": None,
        "graph_OT_changes_authorized": False, "training_authorized": False,
        "broader_acquisition_authorized": False, "automatic_pipeline_integration_authorized": False,
        "design_exposed_development_only": True, "event_semantic_ready": False}


def check_destination(source, out):
    pending, seen = [source.resolve()], set()
    while pending:
        root = pending.pop()
        if root in seen:
            continue
        seen.add(root)
        source_trial.previous.audit.check_output(root, out.resolve(), root)
        protocol = read_json(root / "protocol.json", {})
        links = protocol.get("source_roots", []) + [protocol.get(k) for k in ("source_run", "source_audit")]
        pending.extend(portable(p).resolve() for p in links if p)


def checked_source(source):
    source_trial.verify(source)
    cohort = read_json(source / "cohort.json")
    if len(cohort) != 4 or {c["name"]: c["window_uid"] for c in cohort} != FIXED_UIDS:
        raise ValueError("the complete fixed four-window cohort is required")
    rows = []
    for case in cohort:
        result = source_trial.get_result(source, case)
        if not result or result["status"] != "valid":
            raise ValueError("all four technical responses must be valid before semantic review")
        rows.append(source_trial.comparison(source, case, result))
    if rows != list(iter_jsonl(source / "paired_comparison.jsonl")):
        raise ValueError("stored comparison disagrees with raw replay")
    attempts = [read_json(p) for p in sorted((source / "cost/attempts").glob("*.json"))]
    logical = [read_json(p) for p in sorted((source / "cost/logical_requests").glob("*.json"))]
    keys = {c["request_key"] for c in cohort}
    if (len(attempts) != 4 or len(logical) != 4
            or {r["request_key"] for r in attempts} != keys
            or {r["request_key"] for r in logical} != keys
            or any(r["status"] != "success" for r in attempts)
            or any(r["status"] != "valid" for r in logical)):
        raise ValueError("four complete physical/logical receipt pairs required")
    summary = read_json(source / "summary.json")
    costs = source_trial.prior.costs(attempts, logical)
    for key in ("physical_attempts", "known_usage", "reasoning_tokens", "in_flight", "missing_usage_receipts"):
        if summary["cost"][key] != costs[key]:
            raise ValueError("summary cost disagrees with receipts: " + key)
    derived = {"windows": 4, "technical_all_four_valid": True,
        "baseline_valid_pairs": sum(r["baseline_technical_valid"] for r in rows),
        "baseline_invalid_pairs_preserved": sum(not r["baseline_technical_valid"] for r in rows),
        "unknown_events": sum(r["unknown_events"] for r in rows),
        "unresolved_or_contradicted_premises": sum(r["unresolved_or_contradicted_premises"] for r in rows)}
    if any(summary[k] != v for k, v in derived.items()):
        raise ValueError("summary outcomes disagree with exact replay")
    return cohort, rows, summary


def verify(out):
    seal = read_json(out / "seal.json")
    if not seal or seal.get("version") != VERSION:
        raise ValueError("prepare a new V9.18 TAG first")
    for rel, digest in seal["files"].items():
        if stable_hash(out / rel) != digest:
            raise ValueError("frozen audit changed: " + rel)
    binding = read_json(out / "input_binding.json")
    if binding["out"] != source_trial.prior.canonical(out.resolve()):
        raise ValueError("prepared output moved")
    verify_manifest(read_json(out / "code_manifest.json"))
    source = portable(binding["source"])
    source_trial.verify(source)
    if source_trial.previous.audit.snapshot([source]) != read_json(out / "source_manifest.json"):
        raise ValueError("source inventory/content changed; preserve this audit and use a new TAG")
    return binding


def prepare(project, source, out):
    source, out = source.resolve(), out.resolve()
    check_destination(source, out)
    if (out / "seal.json").exists():
        if verify(out)["source"] != source_trial.prior.canonical(source):
            raise ValueError("different source for an existing TAG")
        return report(out)
    if any(p.name != ".operation.lock" for p in out.iterdir()):
        raise ValueError("new empty TAG required; never overwrite an earlier audit")
    before = source_trial.previous.audit.snapshot([source])
    cohort, comparisons, summary = checked_source(source)
    immutable(out / "source_manifest.json", before)
    immutable(out / "rules.json", RULES)
    immutable(out / "source_summary.json", summary)
    immutable(out / "reference_only/historical_human.json", read_json(source / "reference_only/human.json"))
    cases = []
    for case, comparison in zip(cohort, comparisons):
        uid = case["window_uid"]
        frozen = source / "frozen" / uid
        result = source_trial.get_result(source, case)
        parsed = result["parsed"]
        media = read_json(frozen / "media.json")
        if (media["frame_indices"] != case["frame_indices"] or media["image_sha256"] != case["frames_sha256"]
                or len(media["image_paths"]) != 8):
            raise ValueError("exact media identity mismatch")
        images = []
        for i, (path, digest) in enumerate(zip(media["image_paths"], media["image_sha256"])):
            path = portable(path)
            if stable_hash(path) != digest:
                raise ValueError("source frame changed")
            rel = f"media/{uid}/T{i}.jpg"
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if stable_hash(target) != digest:
                raise ValueError("frame copy checksum mismatch")
            images.append(rel)
        # Keep every observation/event, including those without a targeted question.
        evidence = {"C1": read_json(frozen / "C1.json"), "C2": parsed, "comparison": comparison,
            "different_parent_history": read_json(frozen / "different_parent_history.json"),
            "other_same_parent_history": read_json(frozen / "other_same_parent_history.json"),
            "same_parent_C2": read_json(frozen / "same_parent_C2.json"),
            "raw_manifest": result["raw_manifest"]}
        rel = f"evidence/{uid}.json"
        immutable(out / rel, evidence)
        events = {e["event_id"] for e in parsed["event_assessments"]}
        questions = []
        for qid, statement, scope, paths, event_ids in QUESTIONS[case["name"]]:
            if not set(event_ids) <= events:
                raise ValueError("question refers to missing frozen event")
            anchors = [{"json_pointer": p, "exact_text": pointer(parsed, p)} for p in paths]
            if any(type(a["exact_text"]) is not str for a in anchors):
                raise ValueError("question anchor must be an exact text field")
            questions.append({"claim_id": qid, "statement": statement, "scope_limit": scope,
                "event_ids": event_ids, "source_fields": anchors,
                "question_origin": "analyst_scoped_subclaim_not_model_quote_or_visual_verdict"})
        if set().union(*(set(q["event_ids"]) for q in questions)) != events:
            raise ValueError("targeted questions must cover all frozen events")
        cases.append({**case, "images": images, "evidence_file": rel,
            "result_file_sha256": stable_hash(source / "results" / (uid + ".json")),
            "C2_sha256": semantic_sha256(parsed), "questions": questions,
            "scope_note": SCOPE_NOTES[case["name"]], "scope_note_origin": "analyst_not_human_truth"})
    packet = {"version": VERSION, "use_policy": USE, "blind": False, "cases": cases,
        "no_preferred_answer": True, "all_six_events_retained": True}
    immutable(out / "review/packet.json", packet)
    template = {"version": VERSION, "packet_sha256": semantic_sha256(packet), "use_policy": USE,
        "reviewer_id": "", "claims": [{"claim_id": q["claim_id"], "assessment": "pending", "frame_ids": [], "notes": ""}
            for c in cases for q in c["questions"]]}
    immutable(out / "review/template.json", template)
    immutable(out / "review/review.json", template)
    paths = [Path(__file__), project / "tools/contrast_semantics_v918_cli.py",
        project / "tools/static/contrast_semantics_v918.js", project / "run_contrast_semantics_v918.sh",
        project / "tools/event_decision/safety.py"]
    immutable(out / "code_manifest.json", [{"path": source_trial.prior.canonical(p), "sha256": stable_hash(p)} for p in paths])
    immutable(out / "input_binding.json", {"version": VERSION, "source": source_trial.prior.canonical(source),
        "out": source_trial.prior.canonical(out), "source_manifest_sha256": stable_hash(out / "source_manifest.json")})
    shutil.copy2(project / "tools/static/contrast_semantics_v918.js", out / "review.js")
    render(out, packet, template)
    if source_trial.previous.audit.snapshot([source]) != before:
        raise ValueError("source changed during preparation")
    files = {p.relative_to(out).as_posix(): stable_hash(p) for p in out.rglob("*")
        if p.is_file() and p.name != ".operation.lock" and p != out / "review/review.json"}
    immutable(out / "seal.json", {"version": VERSION, "files": files})
    return report(out)


def validate_review(packet, review):
    root = {"version", "packet_sha256", "use_policy", "reviewer_id", "claims"}
    if not isinstance(review, dict) or set(review) != root:
        raise ValueError("exact review template fields required")
    if (review["version"] != VERSION or review["packet_sha256"] != semantic_sha256(packet) or review["use_policy"] != USE):
        raise ValueError("review packet/version/scope mismatch")
    if type(review["reviewer_id"]) is not str or review["reviewer_id"].strip().lower() in {"", "pending", "your_name"}:
        raise ValueError("reviewer_id required")
    expected = [q["claim_id"] for c in packet["cases"] for q in c["questions"]]
    rows = review["claims"]
    if (not isinstance(rows, list) or len(rows) != len(expected) or any(not isinstance(r, dict) for r in rows)
            or sorted(r.get("claim_id", "") for r in rows) != sorted(expected)):
        raise ValueError("all targeted claims, exactly once, including the normal control, are required")
    for row in rows:
        if set(row) != {"claim_id", "assessment", "frame_ids", "notes"}:
            raise ValueError("unknown or missing claim fields")
        if row["assessment"] not in {"supported", "contradicted", "uncertain"}:
            raise ValueError("finish each claim; uncertain is a completed, acceptable answer")
        source_trial.refs(row["frame_ids"], source_trial.FRAME_IDS, "review frames", row["assessment"] != "uncertain")
        source_trial.text(row["notes"], "brief visible-evidence/uncertainty note")
    normalized = copy.deepcopy(review)
    normalized["reviewer_id"] = normalized["reviewer_id"].strip()
    by_id = {r["claim_id"]: r for r in rows}
    normalized["claims"] = [by_id[qid] for qid in expected]
    return normalized


def accepted_review(out, packet):
    receipt = read_json(out / "review_import/receipt.json")
    path = out / "review_import/accepted_review.json"
    if not receipt:
        if path.exists():
            raise ValueError("incomplete review import; rerun import with the identical review")
        return None
    if stable_hash(path) != receipt["accepted_file_sha256"]:
        raise ValueError("accepted review changed")
    review = validate_review(packet, read_json(path))
    if receipt["review_sha256"] != semantic_sha256(review) or receipt["packet_sha256"] != semantic_sha256(packet):
        raise ValueError("review receipt binding mismatch")
    return review


def import_review(out, review_path):
    verify(out)
    packet = read_json(out / "review/packet.json")
    # Strict parsing also rejects duplicate keys and non-finite values.
    review = source_trial.previous.audit.parse_raw(review_path.read_text(encoding="utf-8-sig"))
    accepted = validate_review(packet, review)
    dest = out / "review_import/accepted_review.json"
    immutable(dest, accepted)
    immutable(out / "review_import/receipt.json", {"version": VERSION,
        "packet_sha256": semantic_sha256(packet), "review_sha256": semantic_sha256(accepted),
        "accepted_file_sha256": stable_hash(dest), "reviewer_id": accepted["reviewer_id"], "use_policy": USE})
    return report(out)


def trace_rows(out, packet, review):
    answers = {r["claim_id"]: r for r in review["claims"]} if review else {}
    digest = semantic_sha256(review) if review else None
    rows = []
    for case in packet["cases"]:
        evidence = read_json(out / case["evidence_file"])
        claims = []
        for q in case["questions"]:
            answer = answers.get(q["claim_id"])
            state = answer["assessment"] if answer else "not_reviewed"
            claims.append({**q, "human_assessment": state,
                "review_observed_mask": state in {"supported", "contradicted"},
                "review_supported_value": {"supported": True, "contradicted": False}.get(state),
                "reviewer_id": review["reviewer_id"] if review else None,
                "review_sha256": digest, "frame_ids": answer["frame_ids"] if answer else [],
                "notes": answer["notes"] if answer else "",
                "mask_scope": "only_this_question_not_whole_source_field_event_or_window"})
        rows.append({"version": VERSION, "window_uid": case["window_uid"], "name": case["name"],
            "parent_C1_sha256": case["parent_sha256"], "C2_sha256": case["C2_sha256"],
            "result_file_sha256": case["result_file_sha256"], "raw_manifest": evidence["raw_manifest"],
            "frame_indices": case["frame_indices"], "frames_sha256": case["frames_sha256"],
            "raw_model_observations": evidence["C2"]["observations"],
            "raw_model_event_assessments": evidence["C2"]["event_assessments"],
            "raw_model_fields_are_not_semantically_verified": True,
            "human_subclaims": claims, "same_parent_comparison": evidence["comparison"],
            "different_parent_history_kept_separate": True, "diagnostic_sidecar_only": True,
            "anomaly_label": None, "anomaly_score": None, **boundaries()})
    return rows


def report(out):
    verify(out)
    packet = read_json(out / "review/packet.json")
    review = accepted_review(out, packet)
    rows = trace_rows(out, packet, review)
    source_summary = read_json(out / "source_summary.json")
    summary = {"version": VERSION, "windows": len(rows),
        "events": sum(len(r["raw_model_event_assessments"]) for r in rows),
        "observations": sum(len(r["raw_model_observations"]) for r in rows),
        "targeted_claims": sum(len(r["human_subclaims"]) for r in rows),
        "model_self_reported_premise_counts_not_truth": dict(Counter(e["premise_support"] for r in rows for e in r["raw_model_event_assessments"])),
        "model_discrimination_counts_not_labels": dict(Counter(e["discrimination"] for r in rows for e in r["raw_model_event_assessments"])),
        "human_review_activated": True, "human_review_imported": review is not None,
        "human_claim_counts": dict(Counter(r["assessment"] for r in review["claims"])) if review else {},
        "diagnostic_sidecar_ready": review is not None,
        "all_observations_semantically_verified": False,
        "original_source_files_verified_unchanged": len(read_json(out / "source_manifest.json")),
        "source_technical_all_four_valid": source_summary["technical_all_four_valid"],
        "source_cost_not_new_cost": source_summary["cost"],
        "next": "INSPECT_SCOPED_REVIEW_AND_MASKED_TRACE_NO_AUTO_EXPANSION" if review else "REVIEW_TEN_SCOPED_CLAIMS_ALL_FOUR_WINDOWS",
        **boundaries()}
    write_json(out / "summary.json", summary)
    write_jsonl(out / "diagnostic_trace.jsonl", rows)
    return summary


def render(out, packet, template):
    esc = html.escape
    body = []
    for case in packet["cases"]:
        ev = read_json(out / case["evidence_file"])
        cmp = ev["comparison"]
        frames = "".join(f'<figure><a href="{path}" target="_blank"><img src="{path}" alt="{esc(case["name"])} T{i}"></a><figcaption>T{i} / frame {case["frame_indices"][i]}</figcaption></figure>' for i, path in enumerate(case["images"]))
        questions = []
        for q in case["questions"]:
            refs = "".join(f'<li><code>{esc(a["json_pointer"])}</code><br>{esc(a["exact_text"])}</li>' for a in q["source_fields"])
            boxes = " ".join(f'<label><input type="checkbox" value="T{i}">T{i}</label>' for i in range(8))
            questions.append(f'''<fieldset class="claim" data-id="{q['claim_id']}"><legend>{esc(q['claim_id'])}</legend>
<p class="statement">{esc(q['statement'])}</p><p>{esc(q['scope_limit'])}</p>
<label>Assessment <select aria-label="{q['claim_id']} assessment"><option value="pending">Pending</option><option value="supported">Supported / 可见支持</option><option value="contradicted">Contradicted / 可见反证</option><option value="uncertain">Uncertain / 无法确定</option></select></label>
<div class="checks">{boxes}</div><label>Evidence / limitation <textarea maxlength="4000" rows="2" aria-label="{q['claim_id']} notes"></textarea></label>
<details><summary>Exact model text, not verified truth</summary><ul>{refs}</ul></details></fieldset>''')
        event_rows = "".join(f'<tr><td>{esc(r["event_id"])}</td><td>{esc(r["premise_support"])}</td><td>{esc(r["discrimination"])}</td><td>{esc(r["contrast_reason"])}</td><td>{esc(r["limits"])}</td></tr>' for r in ev["C2"]["event_assessments"])
        baseline = 'Valid same-parent diagnostic only: ' + json.dumps(cmp["old_diagnostic_only_if_valid"]) if cmp["baseline_technical_valid"] else 'INVALID old C2. No numeric before/after: ' + str(cmp['baseline_error'])
        body.append(f'''<section id="{case['window_uid']}"><h2>{esc(case['name'])}</h2><div class="frames">{frames}</div>
<div class="questions">{''.join(questions)}</div>
<details><summary>All model events and same-parent comparison</summary><p>{esc(baseline)}</p><div class="scroll"><table><thead><tr><th>Event</th><th>Premise</th><th>Contrast</th><th>Reason</th><th>Limits</th></tr></thead><tbody>{event_rows}</tbody></table></div><pre>{esc(json.dumps(ev['C2']['observations'],ensure_ascii=False,indent=2))}</pre></details>
<details><summary>Analyst concern and provenance (not a reviewer answer)</summary><p>{esc(case['scope_note'])}</p><p>Parent C1: <code>{case['parent_sha256']}</code><br>C2: <code>{case['C2_sha256']}</code></p><a href="{case['evidence_file']}">Full evidence, separate different-parent history</a></details></section>''')
    data = json.dumps(template, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.18 | Four-window claim review</title>
<style>*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#182323;background:#fff;margin:0;letter-spacing:0}header,main,footer{max-width:1400px;margin:auto;padding:20px}header{border-bottom:2px solid #16756b}h1{font-size:26px;margin:0}h2{font-size:22px}section{padding:20px 0;border-bottom:1px solid #a8b8b5}p{max-width:100ch}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}pre{font-size:13px}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frames figure{margin:0;min-width:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}.questions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:20px 0}fieldset{min-width:0;border:1px solid #adbab7;border-radius:4px;padding:14px}legend{font-size:14px;overflow-wrap:anywhere}.statement{font-weight:600}select,textarea,input,button{font:inherit}textarea{width:100%;display:block}select{max-width:100%}.checks{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}details{margin:12px 0}summary{cursor:pointer;color:#175e9f}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:650px}td,th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #ccc}nav{display:flex;flex-wrap:wrap;gap:16px;margin-top:15px}.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}button{padding:8px 14px;border:1px solid #28756c;border-radius:4px;background:#e8f4f0;cursor:pointer}#status{color:#8a3825}figcaption{font-size:13px}.notice{color:#565d62}@media(max-width:700px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}.questions{grid-template-columns:1fr}header,main,footer{padding:14px}h1{font-size:23px}}</style>
<header><h1>V9.18 · Four-window scoped review</h1><p>四窗、同一八帧、十项核查。只评价题目中的具体陈述，不填写异常标签；无法看清可选 uncertain。Supported / contradicted 需勾选证据帧，所有回答需一句依据或限制。旧人审不预填；这不是盲评或准确率测试。</p>
<p class="notice">Unknown cause, legality or intent does not make a visible event normal. A supported subclaim does not validate an entire C1/C2. No scoring, training or API execution.</p>
<nav>''' + ''.join(f'<a href="#{c["window_uid"]}">{esc(c["name"])}</a>' for c in packet['cases']) + '''</nav>
<div class="actions"><label>Reviewer ID <input id="reviewer" maxlength="80"></label><button id="download">Download review JSON</button><label>Load saved JSON <input id="load" type="file" accept="application/json,.json"></label></div><p id="status" role="status">0 / 10 completed</p></header><main>''' + ''.join(body) + '''</main><footer><a href="summary.json">Audit status</a> · <a href="diagnostic_trace.jsonl">Masked diagnostic trace</a> · <a href="reference_only/historical_human.json">Historical R1/R3 reference, not current answers</a><p>No requests will run from this page. Download and import the review locally.</p></footer><script type="application/json" id="template">''' + data + '''</script><script src="review.js"></script></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
