"""Read-only V9.13 replay with a separate, non-deployable compatibility layer."""
from __future__ import annotations

from collections import Counter
import copy
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from . import smoke_recheck_v913 as prior
from .contracts import read_json, semantic_sha256, write_json, write_jsonl
from .b1b4_trial.protocol import immutable, stable_hash, verify_manifest
from .role_scoped import portable

VERSION = "v914_zero_api_typed_id_and_evidence_scope_audit_1"
POLICY = {
    "version": VERSION,
    "C1": "Only nonnegative JSON integer IDs at declared ID paths become decimal strings; no coercion of refs across types.",
    "collision": "Reject duplicate or post-conversion colliding definitions within each typed namespace.",
    "C2": "normal_evidence_ids belong to C2; unexplained_direct_evidence_ids belong to the current C1 event.",
    "parent_scope_fallback": "Diagnostic resolution only; never promotes C1 observations to C2 benign evidence.",
    "semantics": "No text, frame, probability, visibility or observation sufficiency edits; no inferred identity.",
    "missing_dependent_C2": "Stays missing. New normalized parent has a new hash; no cached C2 is reparented.",
    "requests": "No acquisition entry point, authorization or inherited budget.",
}


def parse_raw(text):
    """Accept JSON or one JSON fence, never heuristic extraction/repair."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out

    def nonfinite(value):
        raise ValueError("non-finite JSON value: " + value)

    value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def typed_id(value):
    if type(value) is str and value and value == value.strip():
        return ("str", value), value
    if type(value) is int and value >= 0:
        return ("int", value), str(value)
    raise ValueError("ID must be a nonempty string or nonnegative integer, not bool/float/null")


def normalize_c1(c1, uid, signature):
    normalized = copy.deepcopy(c1)
    changes = []
    maps = {}

    def assign(container, key, value, path, namespace):
        previous = container[key]
        if type(previous) is not type(value) or previous != value:
            changes.append({"path": path, "namespace": namespace, "before": previous, "after": value})
            container[key] = value

    for group, key in (("entities", "entity_id"), ("evidence", "evidence_id"), ("events", "event_id")):
        if not isinstance(c1.get(group), list):
            raise ValueError("missing ID definition array: " + group)
        table, targets = {}, set()
        for index, row in enumerate(normalized[group]):
            token, value = typed_id(row[key])
            if token in table or value in targets:
                raise ValueError("duplicate or conversion collision in " + group)
            table[token] = value
            targets.add(value)
            assign(row, key, value, f"/{group}/{index}/{key}", group)
        maps[group] = table
    for index, event in enumerate(normalized["events"]):
        for field, namespace in (("participant_ids", "entities"), ("evidence_ids", "evidence")):
            if not isinstance(event.get(field), list):
                raise ValueError("missing reference array: " + field)
            seen = set()
            for j, ref in enumerate(event[field]):
                token, _ = typed_id(ref)
                if token not in maps[namespace] or token in seen:
                    raise ValueError("unknown, type-mismatched or duplicate reference in " + field)
                seen.add(token)
                assign(event[field], j, maps[namespace][token], f"/events/{index}/{field}/{j}", namespace)
    # Reverse the explicit path edits: everything else must be byte-equivalent JSON data.
    reverse = copy.deepcopy(normalized)
    for change in changes:
        parts = change["path"].strip("/").split("/")
        target = reverse
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        key = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[key] = change["before"]
    if semantic_sha256(reverse) != semantic_sha256(c1):
        raise ValueError("non-ID data changed")
    prior.validate_fresh(normalized, uid, signature)
    return {"normalized": normalized, "changes": changes,
            "original_payload_sha256": semantic_sha256(c1),
            "normalized_payload_sha256": semantic_sha256(normalized),
            "reverse_transform_matches_original": True,
            "semantic_fields_unchanged": True, "visual_truth_verified": False}


def evidence_scope(c1, c2):
    """Resolve under the original declared scope first, not an untyped ID union."""
    prior.validate_proposal(c1)
    if c2.get("proposal_sha256") != semantic_sha256(c1):
        raise ValueError("C2 parent hash mismatch; no automatic reparenting")
    if any(c2.get(k) != c1.get(k) for k in ("window_id", "evidence_signature")):
        raise ValueError("C2 parent window/media mismatch")
    pools = {"C1": c1["evidence"], "C2": c2.get("normal_evidence", [])}
    for scope, items in pools.items():
        if not isinstance(items, list):
            raise ValueError("invalid evidence list")
        ids = [row.get("evidence_id") for row in items]
        if any(type(x) is not str or not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("invalid or duplicate evidence definitions in " + scope)
    definitions = {scope: {r["evidence_id"]: (i, r) for i, r in enumerate(items)} for scope, items in pools.items()}
    hashes = {"C1": semantic_sha256(c1), "C2": semantic_sha256(c2)}
    events = {r["event_id"]: r for r in c1["events"]}
    refs = []
    for i, row in enumerate(c2.get("event_bindings", [])):
        if row.get("event_id") not in events:
            raise ValueError("unknown event in C2")
        event = events[row["event_id"]]
        for j, ex in enumerate(row.get("explanations", [])):
            for field, expected in (("normal_evidence_ids", "C2"), ("unexplained_direct_evidence_ids", "C1")):
                values = ex.get(field)
                if not isinstance(values, list) or any(type(v) is not str for v in values):
                    raise ValueError("invalid evidence reference array")
                for k, value in enumerate(values):
                    available = [s for s in ("C1", "C2") if value in definitions[s]]
                    scope = expected if expected in available else available[0] if len(available) == 1 else None
                    issue = "declared_scope_resolved" if scope == expected else "cross_scope_only" if scope else "unresolved"
                    if len(values) != len(set(values)):
                        issue = "duplicate_reference"
                    if expected == "C1" and value not in event["evidence_ids"] and scope == "C1":
                        issue = "different_event_evidence"
                    reference = {"path": f"/event_bindings/{i}/explanations/{j}/{field}/{k}",
                                 "unqualified_id": value, "expected_scope": expected,
                                 "defined_in": available, "status": issue, "resolved_record": None}
                    if scope:
                        index, item = definitions[scope][value]
                        reference["resolved_record"] = {"scope": scope, "evidence_id": value,
                            "record_sha256": hashes[scope], "pointer": f"/{'evidence' if scope == 'C1' else 'normal_evidence'}/{index}",
                            "evidence": copy.deepcopy(item), "semantically_verified": False}
                    refs.append(reference)
    return {"references": refs, "counts": dict(Counter(r["status"] for r in refs)),
            "all_references_in_declared_scope": all(r["status"] == "declared_scope_resolved" for r in refs),
            "cross_scope_refs_are_not_normal_evidence": True, "original_C2_unchanged": True,
            "benignity_verified": False, "new_U": None}


def snapshot(roots):
    result = []
    for root in roots:
        if (root / ".operation.lock").exists():
            raise ValueError("source still running; wait for a stable read-only snapshot")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result.append({"path": prior.canonical(path), "sha256": stable_hash(path)})
    return sorted(result, key=lambda r: r["path"])


def check_output(source, out, original):
    for root in (source, original):
        if out == root or root in out.parents or out in root.parents:
            raise ValueError("new sibling output required; original runs are read-only")


def validate_status(function, *args):
    try:
        function(*args)
        return {"valid": True, "error": None}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"valid": False, "error": str(exc)}


def semantic_questions(uid):
    # Analyst questions from the prior report, not generated labels or visual verdicts.
    questions = {
        "edb6091cadf8": ["Does rink/uniform context establish that THIS collision is benign? R3 reports a knockdown and no benign explanation; that does not itself establish rule legality or anomaly gold."],
        "01dfb1e8b720": ["Do flames belong to a person, or to a nearby object? R3 confirms street fire/smoke and a photographer, not the model's burning-person identity. Keep the identity claim unresolved."],
        "e5896297709f": ["Does the large central screen show active vehicle-control lane lines, rather than a map? Neither hands-off motion nor R3's 'likely autonomous' proves active automatic control, demonstration intent or safety."],
        "be18a4a445c6": ["R3 supports calm court activity. Do the frames establish a timeout, substitution or minor foul, or only ordinary gathering? Do not promote the specific inferred cause to an observation."],
    }
    return questions.get(uid[:12], ["Compare literal model claims with the existing diagnostic review; no automatic visual verification."])


def build(project, source, out):
    source, out = source.resolve(), out.resolve()
    protocol = prior.verify(source)
    original = portable(protocol["source_run"]).resolve()
    check_output(source, out, original)
    roots = [original, source]
    before = snapshot(roots)
    own_paths = [Path(__file__), project / "tools/smoke_offline_v914_cli.py", project / "run_smoke_offline_v914.sh",
                 project / "tools/event_decision/safety.py"]
    code = [{"path": prior.canonical(p), "sha256": stable_hash(p)} for p in own_paths]
    binding = {"version": VERSION, "source": prior.canonical(source), "out": prior.canonical(out),
               "input_sha256": semantic_sha256(before), "code": code, "policy": POLICY}
    if (out / "completion.json").exists():
        if read_json(out / "input_binding.json") != binding:
            raise ValueError("source/code changed for existing TAG; preserve it and choose a new TAG")
        completion = read_json(out / "completion.json")
        if (completion.get("input_sha256") != semantic_sha256(before)
                or completion.get("output_manifest_sha256") != stable_hash(out / "output_manifest.json")):
            raise ValueError("offline completion seal changed")
        verify_manifest(read_json(out / "output_manifest.json"))
        if snapshot(roots) != before:
            raise ValueError("source changed during verification")
        return read_json(out / "summary.json")
    if not (out / "input_binding.json").exists() and any(p.name != ".operation.lock" for p in out.iterdir()):
        raise ValueError("refuse to overwrite a non-audit output folder")
    immutable(out / "input_binding.json", binding)
    immutable(out / "source_snapshot.json", before)
    immutable(out / "policy.json", POLICY)
    immutable(out / "reference_contract_draft.json", {
        "version": VERSION, "status": "diagnostic_design_only_not_deployed",
        "qualified_evidence_reference": {"scope": "C1 or C2", "record_sha256": "exact canonical parent/response hash",
            "evidence_id": "ID in the declared namespace", "pointer": "path into that immutable record"},
        "scope_rules": ["C1 supports a direct observation; it is not automatically benign evidence.",
            "C2 supports a proposed explanation; reference validity is not visual verification.",
            "If the same ID occurs in both scopes, the declared field scope controls; never union the namespaces.",
            "Cross-event, missing, wrong-parent and duplicate references remain errors.",
            "Unobserved control modes, intention and event identity remain unverified; unknown is not normal."],
        "migrated_existing_C2": False, "scoring_enabled": False})
    media = read_json(source / "media/manifest.json")
    review = read_json(source / "review_reference/imported_review.json")
    humans = {row["window_uid"]: row for row in review["cases"]}
    ids = protocol["case_ids"]
    if len(ids) != 4 or len(set(ids)) != 4 or set(humans) != set(ids):
        raise ValueError("all fixed four cases and their diagnostic references are required")
    cells, audits, claims, missing = [], [], [], []
    linked_raw, recoveries = set(), []
    for uid in ids:
        old = read_json(source / "reference" / (uid + ".json"))
        signature = semantic_sha256(media[uid]["image_sha256"])
        fresh_record = prior.load_result(source, uid, "C1_reobserved")
        fresh = fresh_record.get("parsed")
        case_audits = []
        for stage in prior.STAGES:
            record = prior.load_result(source, uid, stage)
            cell = {"window_uid": uid, "stage": stage, "original_status": record["status"],
                    "original_reason": record.get("reason"), "response_audit_files": [],
                    "original_diagnostic": record.get("diagnostic"), "new_U": None}
            raws = record.get("raw_manifest", [])
            if record["status"] == "valid" and len(raws) != 1:
                raise ValueError("valid response requires one unambiguous raw payload in this no-repair run")
            for item in raws:
                path = portable(item["path"]).resolve()
                if path.parent != (source / "cache/raw").resolve() or path in linked_raw:
                    raise ValueError("raw payload outside source cache or linked more than once")
                linked_raw.add(path)
                raw = read_json(path)
                identity = raw["identity"]
                receipt_path = portable(raw["receipt"]).resolve()
                if receipt_path.parent != (source / "cost/attempts").resolve():
                    raise ValueError("receipt outside source run")
                receipt = read_json(receipt_path)
                expected = {"window_uid": uid, "stage": stage}
                if any(identity.get(k) != v or receipt.get(k) != v for k, v in expected.items()):
                    raise ValueError("raw/receipt stage identity mismatch")
                if (receipt.get("raw_file_sha256") != stable_hash(path) or receipt.get("request_key") != path.stem
                        or receipt.get("status") != "success" or identity.get("images") != media[uid]["image_sha256"]):
                    raise ValueError("raw receipt/media identity mismatch")
                parent = old["C1"] if stage == "C2_fixed_C1" else fresh
                expected_parents = [] if stage == "C1_reobserved" else [semantic_sha256(parent)] if parent else None
                if identity.get("parents") != expected_parents:
                    raise ValueError("raw parent identity mismatch")
                audit = {**expected, "original_status": record["status"], "raw_path": item["path"],
                         "raw_sha256": stable_hash(path), "physical_receipt_sha256": stable_hash(receipt_path),
                         "raw_parse_valid": False, "native": None, "compatibility": None, "scope_audit": None}
                try:
                    payload = parse_raw(raw["raw"])
                    audit.update(raw_parse_valid=True, original_payload=payload)
                    if record["status"] == "valid" and semantic_sha256(payload) != semantic_sha256(record["parsed"]):
                        raise RuntimeError("saved parsed response differs from strict raw parse")
                    if stage == "C1_reobserved":
                        audit["native"] = validate_status(prior.validate_fresh, payload, uid, signature)
                        try:
                            normalized = normalize_c1(payload, uid, signature)
                            audit["compatibility"] = {"valid": True, **normalized}
                            if not audit["native"]["valid"]:
                                recoveries.append({"window_uid": uid, "stage": stage,
                                    "normalized_payload_sha256": normalized["normalized_payload_sha256"],
                                    "raw_sha256": stable_hash(path), "changed_ID_occurrences": len(normalized["changes"])})
                                immutable(out / "normalized_C1" / (uid + ".json"), normalized)
                        except (ValueError, TypeError, KeyError) as exc:
                            audit["compatibility"] = {"valid": False, "error": str(exc)}
                    else:
                        if parent is None:
                            raise ValueError("missing original parent; cannot audit C2")
                        audit["native"] = validate_status(prior.validate_native, parent, payload)
                        audit["scope_audit"] = evidence_scope(parent, payload)
                    if record["status"] == "valid" and not audit["native"]["valid"]:
                        raise RuntimeError("original valid status fails unchanged native validator")
                except (ValueError, TypeError, KeyError) as exc:
                    audit["audit_error"] = str(exc)
                relative = "response_audits/" + path.stem + ".json"
                immutable(out / relative, audit)
                cell["response_audit_files"].append(relative)
                audits.append(audit)
                case_audits.append(audit)
            if record["status"] == "blocked_by_invalid_C1":
                missing.append({"window_uid": uid, "stage": stage, "original_status": record["status"],
                                "response_exists": bool(raws), "new_score": None})
            cells.append(cell)
        claim = {"window_uid": uid, "status": "semantic_claims_unverified_not_a_label",
                 "existing_R3_diagnostic_not_gold": humans[uid], "review_source_sha256": stable_hash(source / "review_reference/imported_review.json"),
                 "questions": semantic_questions(uid), "frame_indices": media[uid]["frame_indices"],
                 "frames": [prior.canonical(source / "media" / uid / f"T{k}.jpg") for k in range(8)],
                 "model_claims": [{"stage": r["stage"], "raw_sha256": r["raw_sha256"],
                      "payload": r.get("original_payload")} for r in case_audits],
                 "reviewer_task_activated": False, "human_truth_verified": False}
        claims.append(claim)
    raw_paths = set((source / "cache/raw").glob("*.json"))
    if linked_raw != raw_paths:
        raise ValueError("not every saved raw response was audited; inspect unlinked cache records")
    attempts = [read_json(p) for p in (source / "cost/attempts").glob("*.json")]
    if any(r.get("status") == "in_flight" for r in attempts):
        raise ValueError("source has in-flight requests")
    native_valid = sum(bool(r.get("native", {}).get("valid")) for r in audits if r.get("native"))
    gap = [r for r in audits if r.get("scope_audit") and not r["scope_audit"]["all_references_in_declared_scope"]]
    candidates = [{**m, "recovered_parent": next((r for r in recoveries if r["window_uid"] == m["window_uid"]), None)} for m in missing]
    decision = {"version": VERSION, "decision": "DEFER_NEW_REQUESTS_PENDING_CLAIM_SCOPE_CHECK",
                "remote_execution_authorized": False, "new_requests_required_now": False,
                "budget_inherited": False, "proposed_budget": None,
                "missing_dependent_requests_for_future_decision_only": candidates,
                "candidate_is_not_approved": True,
                "before_any_future_request": ["Decide whether completing the missing dependency adds diagnostic information.",
                    "Use a new request version and the exact normalized parent hash; preserve the old blocked record.",
                    "Explicitly approve a bounded budget; no full four-case rerun or response selection by preferred semantics.",
                    "Resolve or retain masks for burning-person identity, vehicle display/intent and sports benignity."],
                "new_human_review_form_required_now": False,
                "ready_for_larger_acquisition": False, "ready_for_shadow_integration": False,
                "scoring_authorized": False, "training_authorized": False, "formal_accuracy": None, "formal_AP": None}
    summary = {"version": VERSION, "windows": len(ids), "stage_cells": len(cells),
               "original_statuses": dict(Counter(r["original_status"] for r in cells)),
               "saved_raw_responses_audited": len(audits), "original_native_valid_responses": native_valid,
               "lossless_C1_recoveries": recoveries, "C2_scope_gap_responses": len(gap),
               "cross_scope_reference_occurrences": sum(r["scope_audit"]["counts"].get("cross_scope_only", 0) for r in gap),
               "missing_dependent_C2": len(missing), "complete_four_case_comparison": False,
               "structural_recoverability_is_not_visual_truth": True,
               "source_files_verified_unchanged": len(before), "new_API_calls": 0, "new_media_decodes": 0,
               "new_scores_computed": 0, "old_scores_changed": False, "remote_execution_authorized": False,
               "human_review_activated": False, "scoring_authorized": False, "training_authorized": False,
               "formal_accuracy": None, "formal_AP": None, "next": decision["decision"]}
    write_jsonl(out / "stage_accounting.jsonl", cells)
    write_jsonl(out / "semantic_claim_ledger.jsonl", claims)
    immutable(out / "next_request_decision.json", decision)
    render(out, source, cells, claims, summary)
    if snapshot(roots) != before:
        raise ValueError("source changed during audit; do not use this incomplete output")
    prior.verify(source)
    immutable(out / "summary.json", summary)
    output = [{"path": prior.canonical(p), "sha256": stable_hash(p)} for p in sorted(out.rglob("*"))
              if p.is_file() and p.name not in {".operation.lock", "output_manifest.json", "completion.json"}]
    immutable(out / "output_manifest.json", output)
    immutable(out / "completion.json", {"version": VERSION, "output_manifest_sha256": stable_hash(out / "output_manifest.json"),
                                      "input_sha256": semantic_sha256(before), "source_unchanged": True})
    return summary


def render(out, source, cells, claims, summary):
    esc = lambda value: html.escape(str(value))
    def relative(path):
        try:
            return quote(Path(os.path.relpath(path, out)).as_posix(), safe="/")
        except ValueError:
            return Path(path).as_uri()
    sections = []
    for claim in claims:
        uid = claim["window_uid"]
        frames = ''.join(f'<figure><img loading="lazy" src="{relative(source / "media" / uid / f"T{k}.jpg")}" alt="T{k}"><figcaption>T{k} / frame {index}</figcaption></figure>' for k, index in enumerate(claim["frame_indices"]))
        rows = []
        for cell in (c for c in cells if c["window_uid"] == uid):
            findings = []
            for file in cell["response_audit_files"]:
                audit = read_json(out / file)
                compatibility = audit.get("compatibility") or {}
                if compatibility.get("valid") and compatibility.get("changes"):
                    text = f'Lossless ID view: {len(compatibility["changes"])} edits. Original failed status retained; C2 still missing.'
                elif audit.get("scope_audit") and not audit["scope_audit"]["all_references_in_declared_scope"]:
                    text = 'C1/C2 reference scope mismatch. No evidence merged/deleted; no new U.'
                else:
                    text = 'Native validation replayed; visual meaning is not verified.'
                findings.append(f'<p>{esc(text)} <a href="{file}">Detailed audit</a></p>')
            rows.append(f'<tr><td>{esc(cell["stage"])}</td><td>{esc(cell["original_status"])}</td><td>{"".join(findings) or "No response; not reconstructed."}</td></tr>')
        questions = ''.join(f'<li>{esc(q)}</li>' for q in claim["questions"])
        sections.append(f'<section><h2>{uid[:12]}</h2><div class="frames">{frames}</div><table><thead><tr><th>Branch</th><th>Original status</th><th>Offline finding</th></tr></thead><tbody>{"".join(rows)}</tbody></table><h3>Unverified semantic claims</h3><ul>{questions}</ul><p><b>Existing R3 diagnostic, not gold:</b> {esc(claim["existing_R3_diagnostic_not_gold"]["visible_event_summary"])}</p><details><summary>Original model claims and review provenance</summary><pre>{esc(json.dumps(claim, ensure_ascii=False, indent=2))}</pre></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9.14 Offline Contract Audit</title><style>
body{margin:0;background:#fcfcfc;color:#242a28;font:15px system-ui}main{max-width:1150px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:21px}h3{font-size:17px}section{padding:24px 0;border-top:1px solid #bacbc2}aside{border-left:4px solid #a4442a;background:#fff0eb;padding:14px}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:16px 0}figure{margin:0}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figcaption{font-size:12px}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f1;padding:12px}a{color:#126f64}li{margin:8px 0}details{margin-top:14px}@media(max-width:600px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}table{font-size:12px}td,th{padding:4px}}
</style><main><h1>V9.14 Offline Contract Audit</h1><aside>Zero API calls. Original responses and scores unchanged. ID compatibility and resolvable references do not verify visual truth. No scoring, training or new acquisition approval.</aside>'''
    page += f'<p>{summary["windows"]} fixed cases; {summary["saved_raw_responses_audited"]} saved responses; {summary["original_native_valid_responses"]} native-valid; {len(summary["lossless_C1_recoveries"])} lossless C1 recovery; {summary["C2_scope_gap_responses"]} C2 scope gap; {summary["missing_dependent_C2"]} missing C2.</p><p><a href="summary.json">Summary</a> | <a href="next_request_decision.json">Next-request decision: not authorized</a> | <a href="semantic_claim_ledger.jsonl">Claim ledger</a></p>'
    (out / "index.html").write_text(page + ''.join(sections) + '</main></html>', encoding="utf-8")
