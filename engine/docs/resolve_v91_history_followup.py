#!/usr/bin/env python3
"""Offline follow-up resolver: preserve text, trace index evidence, check capacity.

No reviewer assertions are changed or inferred as approvals. Only new outputs
are written. This is not a release-list writer or an enrollment-gate bypass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from event_decision.contracts import file_sha256, write_json, write_jsonl
from event_decision.hard_trial import group_id
from event_decision.safety import OfflineGuard

spec = importlib.util.spec_from_file_location("v91_canary_audit", Path(__file__).with_name("audit_v91_canary_sources.py"))
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)

VERSION = "v91_history_followup_resolution_v1"
DEFAULT_AUDIT = PROJECT / "runs/governed_v91_canary_source_audit_20260912"
OLD_REVIEW = PROJECT / "runs/governed_v91_reviewed_hard_normal_binding_20260911/history/review_input_snapshot.jsonl"
CHECK_FIELDS = ("checked_actual_fit_threshold_evaluation_use",
                "checked_discovery_prompt_design_human_viewing",
                "checked_external_history_and_source_aliases")
FALSE_FIELDS = ("eligible_for_enrollment", "import_supported", "automatic_release_effect")
NOTE_PATTERN = re.compile(r'"evidence_notes"\s*:\s*"(?P<body>.*?)"(?=\s*,\s*"eligible_for_enrollment"\s*:)', re.S)
REF_PATTERN = re.compile(r'(?P<path>(?:[A-Za-z]:[\\/]|/mnt/)[^\r\n<>\uff0c\uff1b]*?\.jsonl)\s*(?P<line>\d+)\s*\u884c')
USED_TEXT = "\u5f53\u524d\u7ed3\u8bba\uff1a\u53d1\u73b0\u4f7f\u7528"
NO_USE_TEXT = "\u5b9e\u9645\u6267\u884c/\u62df\u5408/\u8bc4\u4f30\uff1a\u672a\u53d1\u73b0\u5df2\u7528"
DECISIONS = {"pending", "actual_use_confirmed", "index_only_no_other_use_confirmed",
             "no_use_in_declared_scope_confirmed", "unknown"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_stream(text):
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    rows, offset = [], 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        row, offset = decoder.raw_decode(text, offset)
        if not isinstance(row, dict):
            raise ValueError("review stream must contain JSON objects, not arrays or prose")
        rows.append(row)
    return rows


def parse_review(raw):
    """Repair only malformed note strings in the known draft layout, never other fields."""
    text = raw.decode("utf-8-sig")
    try:
        return decode_stream(text), []
    except json.JSONDecodeError as original:
        repairs = []

        def repair(match):
            body = match.group("body")
            # Raw Windows paths have JSON escape collisions (e.g. \r in \runs).
            # In this malformed format, retain the exact literal note characters.
            raw_windows = bool(re.search(r"[A-Za-z]:\\(?!\\)", body))
            if raw_windows:
                note = body
                method = "verbatim_raw_windows_note"
            else:
                try:
                    note = json.loads('"' + body + '"', strict=False)
                    method = "escape_literal_control_characters"
                except json.JSONDecodeError:
                    raise ValueError("unsupported malformed note; no speculative repair performed")
            repairs.append({"note_ordinal": len(repairs) + 1, "method": method,
                            "raw_note_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                            "normalized_note_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest(),
                            "contains_literal_newline": "\n" in body,
                            "human_conclusion_changed": False})
            return '"evidence_notes": ' + json.dumps(note, ensure_ascii=False)

        repaired = NOTE_PATTERN.sub(repair, text)
        if not repairs:
            raise ValueError(f"malformed review outside supported evidence_notes layout: {original}")
        rows = decode_stream(repaired)
        if len(rows) != len(repairs):
            raise ValueError("ambiguous note boundaries; refuse automatic formatting repair")
        return rows, repairs


class Reader:
    def __init__(self):
        self.inputs, self.cache = {}, {}

    def raw(self, path):
        path = canary.portable(path).resolve()
        if path not in self.cache:
            before = canary.signature(path)
            if before[0] > 1024**3:
                raise ValueError(f"file over 1 GiB read budget: {path}")
            raw = path.read_bytes()
            if before != canary.signature(path):
                raise ValueError(f"input changed during read: {path}")
            self.cache[path] = raw
            self.inputs[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return self.cache[path]

    def json(self, path):
        return json.loads(self.raw(path), object_pairs_hook=unique_object)

    def jsonl(self, path):
        return [json.loads(line, object_pairs_hook=unique_object) for line in self.raw(path).splitlines() if line.strip()]


def validate_reviews(rows, sources):
    expected = {r["source_group"]: r for r in sources if r["primary_status"] not in canary.RETAIN}
    seen = set()
    if len(rows) != len(expected):
        raise ValueError("review must cover exactly the audit's non-retained source rows")
    for row in rows:
        group = row.get("source_group")
        if group not in expected or group in seen:
            raise ValueError(f"unknown or duplicate reviewed source: {group}")
        seen.add(group)
        original = expected[group]
        if row.get("schema") != canary.VERSION + "_manual_review_draft":
            raise ValueError("unexpected review schema")
        if sorted(row.get("video_ids", [])) != sorted(original["video_ids"]) or sorted(row.get("target_categories", [])) != sorted(original["target_categories"]):
            raise ValueError(f"review source identity/categories changed: {group}")
        if row.get("audit_status") != original["primary_status"]:
            raise ValueError(f"review rewrites the audit status: {group}")
        if any(row.get(f) is not False for f in FALSE_FIELDS):
            raise ValueError("review draft cannot authorize release/import/enrollment")
        if any(type(row.get(f)) is not bool for f in CHECK_FIELDS):
            raise ValueError("review check fields must be actual JSON Booleans")
        if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip():
            raise ValueError("reviewer_id is required")
        if not isinstance(row.get("evidence_notes"), str):
            raise ValueError("evidence_notes must be a string")


def trace_index(reader, path, line, group):
    """Resolve the index row to its hash-checked underlying record, not its filename."""
    from audit_v91_history import classify, mock_context, record_groups
    path = canary.portable(path).resolve()
    result = {"index_path": str(path), "index_line": line, "source_group": group,
              "proves_actual_execution": False}
    try:
        lines = reader.raw(path).splitlines()
        if not 1 <= line <= len(lines):
            raise ValueError("cited line is outside file")
        row = json.loads(lines[line - 1])
        if row.get("source_group") != group:
            raise ValueError("cited line belongs to another source")
        if path.name != "source_exposure_evidence.jsonl":
            raise ValueError("unsupported citation type; not automatically treated as execution")
        target = canary.portable(row["path"]).resolve()
        raw = reader.raw(target)
        current_sha = hashlib.sha256(raw).hexdigest()
        result.update(origin_path=str(target), origin_sha256=current_sha, index_expected_sha256=row.get("sha256"))
        if current_sha != row.get("sha256"):
            raise ValueError("origin changed since the cited index")
        mock, config, error = mock_context(target)
        if config:
            reader.raw(config)
        if error:
            raise ValueError(error)
        kinds, matching_lines = set(), []
        if target.suffix == ".jsonl":
            records = ((i, json.loads(value)) for i, value in enumerate(raw.splitlines(), 1) if value.strip())
        else:
            records = [(1, json.loads(raw))]
        for ordinal, origin in records:
            if group in record_groups(origin):
                kinds.add(classify(target.name, mock, origin))
                matching_lines.append(ordinal)
        if not kinds:
            raise ValueError("origin contains no matching source")
        result.update(status="TRACED", evidence_kinds=sorted(kinds),
                      origin_matching_lines=matching_lines,
                      proves_actual_execution="execution_record" in kinds,
                      planned_only=kinds == {"planned_registration"})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(status="TRACE_UNRESOLVED", error=str(exc))
    return result


def capacity_report(sources, videos):
    """Optimistic necessary condition only; never equate anchors with permission."""
    reviewable = {s["source_group"] for s in sources if s["primary_status"] not in canary.RETAIN}
    result = {}
    for code in ("B2", "B5", "B6"):
        all_groups = {v["source_group"] for v in videos if v["source_group"] in reviewable and code in v["target_categories"]}
        anchored = {v["source_group"] for v in videos if v["source_group"] in reviewable and code in v["target_categories"]
                    and v["positive_anchor"]["structurally_traceable_anchor"]}
        result[code] = {"minimum_independent_source_groups": 2,
                        "optimistic_sources_if_all_histories_cleared": sorted(all_groups),
                        "optimistic_sources_with_existing_traceable_anchors": sorted(anchored),
                        "source_gap_even_if_missing_anchors_repaired": max(0, 2-len(all_groups)),
                        "source_gap_with_existing_anchors": max(0, 2-len(anchored))}
    blocked = [code for code, value in result.items() if value["source_gap_even_if_missing_anchors_repaired"]]
    return {"classes": result, "blocked_even_under_optimistic_assumptions": blocked,
            "human_confirmation_required_now_for_enrollment": not bool(blocked),
            "scope": "B2/B5/B6 necessary source-capacity bound; not a 144-window allocation or G/media/label check",
            "not_an_approval": True, "ready_for_reenrollment": False}


def inspect_review(rows, sources, reader, old_reviews):
    by_source = {r["source_group"]: r for r in sources}
    previous = {r["source_group"]: r for r in old_reviews}
    resolved, traces, questions = [], [], []
    for row in rows:
        group, notes = row["source_group"], row["evidence_notes"]
        citations = [trace_index(reader, m["path"], int(m["line"]), group) for m in REF_PATTERN.finditer(notes)]
        traces.extend(citations)
        used = USED_TEXT in notes
        no_use = NO_USE_TEXT in notes
        flags = []
        if used and no_use:
            flags.append("CONTRADICTORY_USED_AND_NOT_FOUND_STATEMENTS")
        if used and citations and all(c.get("planned_only") for c in citations):
            flags.append("USED_CONCLUSION_CITES_PLAN_INDEX_ONLY")
        if any(c["status"] != "TRACED" for c in citations):
            flags.append("CITATION_NEEDS_CORRECTION_OR_FRESH_EVIDENCE")
        if any(not row[f] for f in CHECK_FIELDS):
            flags.append("LEGACY_BOOLEAN_ATTESTATIONS_INCOMPLETE")
        old = previous.get(group)
        if old and old.get("disposition") == "not_used_after_audit" and used:
            flags.append("NEW_USED_STATEMENT_CONFLICTS_WITH_PRIOR_UNUSED_ATTESTATION")
        if not citations and not used and not no_use:
            flags.append("NO_MACHINE_INTERPRETABLE_CONCLUSION_NOT_ASSUMED_UNUSED")
        source = by_source[group]
        resolved.append({"source_group": group, "reviewer_id": row["reviewer_id"],
                         "original_disposition": row.get("disposition"), "human_notes_unchanged": notes,
                         "reported_used_phrase": used, "reported_not_found_phrase": no_use,
                         "diagnostic_flags": flags, "citation_count": len(citations),
                         "unique_origin_paths": sorted({c["origin_path"] for c in citations if "origin_path" in c}),
                         "prior_review": old, "prior_release_in_audit_snapshot": source["prior_release_in_input_snapshot"],
                         "technical_flags": source["technical_flags"], "eligible_for_enrollment": False})
        question = ("Earlier execution notes say no use was found, but the conclusion says used. Which is correct?"
                    if used and no_use else
                    "The cited index points only to planned records. Was actual use independently confirmed, or was 'used' based only on that index?"
                    if "USED_CONCLUSION_CITES_PLAN_INDEX_ONLY" in flags else
                    "Confirm actual use / reviewed non-use / unresolved status; do not infer non-use from missing records.")
        questions.append({"source_group": group, "question": question,
                          "decision": "pending", "correction_or_evidence_note": ""})
    return resolved, traces, questions


def import_confirmations(path, reader, review_sha, questions, original_rows):
    data = reader.json(path)
    if data.get("schema") != VERSION + "_minimal_confirmation" or data.get("review_input_sha256") != review_sha:
        raise ValueError("confirmation schema or original-review hash mismatch")
    reviewer = data.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("confirmation reviewer_id is required")
    if reviewer not in {r["reviewer_id"] for r in original_rows}:
        raise ValueError("correction must be confirmed by an original reviewer")
    statements = data.get("attestations", {})
    required = {"local_execution_checked", "discovery_design_viewing_checked",
                "external_use_and_aliases_checked", "no_new_use_since_audit_checked"}
    if set(statements) != required or any(type(v) is not bool for v in statements.values()):
        raise ValueError("invalid confirmation attestations")
    expected = {q["source_group"] for q in questions}
    decisions, seen = [], set()
    for entry in data.get("decisions", []):
        group, decision = entry.get("source_group"), entry.get("decision")
        note = entry.get("correction_or_evidence_note")
        if group not in expected or group in seen or decision not in DECISIONS or not isinstance(note, str):
            raise ValueError("unknown/duplicate source, decision or note in confirmations")
        seen.add(group)
        if decision not in {"pending", "unknown"} and not note.strip():
            raise ValueError("resolved decisions require a short correction/evidence note")
        if decision in {"index_only_no_other_use_confirmed", "no_use_in_declared_scope_confirmed"} and not all(statements.values()):
            raise ValueError("reviewed non-use requires explicit completion of all scope attestations")
        decisions.append({"source_group": group, "reviewer_id": reviewer, "human_confirmed_decision": decision,
                          "correction_or_evidence_note": note, "attestations": statements,
                          "requires_future_fresh_history_gate": True, "eligible_for_enrollment": False,
                          "source_exclusions_changed": False})
    if seen != expected:
        raise ValueError("confirmation source set must match the submitted review")
    return decisions


def render_report(out, summary, resolved, capacity):
    lines = ["# History Follow-up Resolution", "", f"Status: `{summary['status']}`", "",
             "No source was released. Original reviewer text and files are unchanged.", "",
             "## What Was Automated", "",
             f"- Recovered {summary['review_rows']} review objects; formatting repairs: {summary['format_repairs']}.",
             f"- Traced {summary['cited_index_rows']} cited index rows to {summary['unique_origin_files']} unique original files.",
             "- Compared prior reviews and contradictory statements, without deciding what the reviewer meant.",
             "- Checked optimistic independent-source capacity BEFORE asking people to repeat reviews.", "",
             "## Immediate Next Action", ""]
    if capacity["blocked_even_under_optimistic_assumptions"]:
        lines += ["STOP enrollment work: even clearing all unresolved histories and repairing missing anchors cannot cover "
                  + ", ".join(capacity["blocked_even_under_optimistic_assumptions"]) + " with two independent sources.",
                  "Do not spend reviewer time trying to make these eight rows pass. Obtain a legitimate additional source or explicitly redesign the study in a separate protocol.",
                  "The short confirmation file is optional/deferred; it cannot repair missing source capacity."]
    else:
        lines += ["Resolve only the pending short confirmations, then implement/check fresh history and media gates before enrollment."]
    lines += ["", "## Source Findings", "", "| Source | Findings | Existing technical flags |",
              "| --- | --- | --- |"]
    for row in resolved:
        lines.append(f"| {row['source_group']} | {', '.join(row['diagnostic_flags'])} | {', '.join(row['technical_flags'])} |")
    lines += ["", "## Files", "",
              "- `review_original.txt`: exact original bytes, including malformed formatting.",
              "- `review_normalized.jsonl`: valid JSONL copy; conclusions/Booleans unchanged.",
              "- `evidence_chain.jsonl`: index row, actual source file, hash and matched original lines.",
              "- `resolved_review_findings.jsonl`: contradictions, prior attestations and technical flags.",
              "- `capacity_report.json`: optimistic capacity is not scientific approval.",
              "- `review_minimal_confirmations.json`: only short corrections; no need to retype old notes.",
              "- `input_manifest.json`: provenance, input/code hashes and scope limitations.",
              "- `human_confirmation_import.jsonl`: only present when --confirmations was supplied; records assertions, not source releases.", "",
              "No full fresh-history scan, video decoding, API calls, candidate generation or accuracy/AP measurement was performed."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    reader = Reader()
    raw = reader.raw(args.review)
    review_sha = hashlib.sha256(raw).hexdigest()
    rows, repairs = parse_review(raw)
    sources = reader.jsonl(args.audit / "canary_source_audit.jsonl")
    videos = reader.jsonl(args.audit / "canary_video_audit.jsonl")
    audit_summary = reader.json(args.audit / "audit_summary.json")
    if not audit_summary.get("status", "").startswith("AUDIT_COMPLETE"):
        raise ValueError("source audit is not complete; inspect its issues before resolving reviews")
    validate_reviews(rows, sources)
    old = reader.jsonl(args.prior_review) if args.prior_review.is_file() else []
    resolved, traces, questions = inspect_review(rows, sources, reader, old)
    capacity = capacity_report(sources, videos)
    confirmations = import_confirmations(args.confirmations, reader, review_sha, questions, rows) if args.confirmations else None
    for path in reader.inputs:
        if canary.within(canary.portable(path), args.out):
            raise ValueError("an input would be contained in the output directory")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "review_original.txt").write_bytes(raw)
    write_jsonl(args.out / "review_normalized.jsonl", rows)
    write_json(args.out / "format_repair_report.json", {"input_sha256": review_sha, "repairs": repairs,
               "original_modified": False, "conclusions_or_booleans_changed": False})
    write_jsonl(args.out / "resolved_review_findings.jsonl", resolved)
    write_jsonl(args.out / "evidence_chain.jsonl", traces)
    write_json(args.out / "capacity_report.json", capacity)
    write_json(args.out / "review_minimal_confirmations.json", {
        "schema": VERSION + "_minimal_confirmation", "review_input_sha256": review_sha,
        "reviewer_id": rows[0]["reviewer_id"] if rows else "",
        "deferred_until_source_capacity_is_resolved": bool(capacity["blocked_even_under_optimistic_assumptions"]),
        "attestations": {"local_execution_checked": False, "discovery_design_viewing_checked": False,
                         "external_use_and_aliases_checked": False, "no_new_use_since_audit_checked": False},
        "allowed_decisions": sorted(DECISIONS), "decisions": questions})
    if confirmations is not None:
        write_jsonl(args.out / "human_confirmation_import.jsonl", confirmations)
    status = "STOP_NEW_SOURCE_CAPACITY_REQUIRED" if capacity["blocked_even_under_optimistic_assumptions"] else "WAITING_FOR_TARGETED_CONFIRMATION_AND_FRESH_HISTORY"
    summary = {"version": VERSION, "status": status, "review_rows": len(rows), "format_repairs": len(repairs),
               "cited_index_rows": len(traces), "unique_origin_files": len({t["origin_path"] for t in traces if "origin_path" in t}),
               "planned_only_citations": sum(t.get("planned_only", False) for t in traces),
               "trace_errors": sum(t["status"] != "TRACED" for t in traces),
               "finding_counts": dict(Counter(flag for r in resolved for flag in r["diagnostic_flags"])),
               "blocked_categories": capacity["blocked_even_under_optimistic_assumptions"],
               "confirmation_rows_imported_as_assertions_only": len(confirmations or []),
               "immediate_review_required_for_enrollment": capacity["human_confirmation_required_now_for_enrollment"],
               "released_sources": 0, "ready_for_reenrollment": False, "remote_calls": 0,
               "new_video_decoding_performed": False, "full_history_refresh_performed": False}
    write_json(args.out / "followup_summary.json", summary)
    code = [Path(__file__), Path(canary.__file__), PROJECT / "tools/audit_v91_history.py",
            PROJECT / "tools/event_decision/hard_trial.py"]
    write_json(args.out / "input_manifest.json", {"version": VERSION, "input_files": reader.inputs,
               "code_hashes": {str(p): file_sha256(p) for p in code},
               "history_scope": "saved source audit plus freshly read cited index/origin files only",
               "no_source_release_or_model_authorization": True})
    render_report(args.out, summary, resolved, capacity)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {args.out / 'README.md'}", flush=True)
    return summary


def main(argv=None):
    guard = OfflineGuard()
    guard.install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=canary.portable, default=DEFAULT_AUDIT)
    parser.add_argument("--review", type=canary.portable)
    parser.add_argument("--prior-review", type=canary.portable, default=OLD_REVIEW)
    parser.add_argument("--confirmations", type=canary.portable, help="optional separately completed compact confirmation file")
    parser.add_argument("--out", type=canary.portable)
    args = parser.parse_args(argv)
    args.audit = args.audit.resolve()
    args.review = (args.review or args.audit / "review_returns/reviewer_R3_history_followup_v1.jsonl").resolve()
    args.prior_review = args.prior_review.resolve()
    args.out = (args.out or PROJECT / "runs" / ("governed_v91_followup_resolution_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if args.out.exists() or canary.within(args.out, args.audit) or canary.within(args.audit, args.out):
        parser.error("use a new output directory separate from the original audit")
    try:
        run(args)
        guard.assert_no_remote_calls()
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
