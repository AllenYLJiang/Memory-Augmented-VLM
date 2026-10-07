#!/usr/bin/env python3
"""Step 1B: read-only training-source audit, NOT an enrollment/release tool.

Run with the project's existing WSL Python environment. The default does not
open video streams. --probe-metadata optionally reads local container nb_frames;
it never counts/decodes frames, creates clips, imports reviews, or calls an API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Do not create __pycache__ files in the existing, hash-bound code tree.
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from event_decision.contracts import file_sha256, read_json, write_json, write_jsonl
from event_decision.hard_trial import HISTORY_NAMES, PRUNE, anomaly_codes, group_id
from event_decision.safety import OfflineGuard

VERSION = "v91_canary_source_audit_v1"
CATEGORIES = {"B1", "B2", "B4", "B5", "B6", "G"}
DEFAULT_TRIAL = PROJECT / "runs/governed_v91_reenrolled_scope_review_20260911"
DEFAULT_ANCHORS = PROJECT.parent / (
    "Transformer_semantic_components_select_anomaly/top_anomalous_frames_72B_positive_segments")
RETAIN = {"RETAIN_EXCLUSION_EXECUTION_EVIDENCE", "RETAIN_EXCLUSION_UNVERIFIED_EXECUTION"}
STATUS_HELP = {
    "RETAIN_EXCLUSION_EXECUTION_EVIDENCE": "Recognized execution with explicit real-mode config; keep excluded.",
    "RETAIN_EXCLUSION_UNVERIFIED_EXECUTION": "Execution-like record with unverified mode; keep excluded pending provenance review.",
    "REVIEW_PLANNED_OR_MOCK_ONLY": "Only planned/mock evidence found; human use/design history must still be checked.",
    "REVIEW_UNRESOLVED_HISTORY": "History, aliases or registrations are unresolved; not cleared.",
    "NEEDS_ANCHOR_OR_MEDIA_EVIDENCE": "No valid technical basis yet; do not fabricate anchors or media verification.",
    "POTENTIAL_FOR_TECHNICAL_SCREENING_NOT_APPROVED": "No recognized exclusion found and an anchor exists; still NOT approved.",
}


def portable(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        return Path(value[5].upper() + ":/" + value[7:])
    if os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        return Path("/mnt/" + value[0].lower() + "/" + value[3:])
    return Path(value)


def signature(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def read_bound_json(path, inputs):
    before = signature(path)
    raw = path.read_bytes()
    if before != signature(path):
        raise ValueError(f"input changed during read: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    inputs[str(path.resolve())] = {"sha256": digest, "bytes": len(raw)}
    return json.loads(raw), digest


def read_bound_jsonl(path, inputs):
    before = signature(path)
    digest, rows = hashlib.sha256(), []
    with path.open("rb") as handle:
        for line, raw in enumerate(handle, 1):
            digest.update(raw)
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"expected object: {path}:{line}")
                rows.append(row)
    if before != signature(path):
        raise ValueError(f"input changed during read: {path}")
    inputs[str(path.resolve())] = {"sha256": digest.hexdigest(), "bytes": before[0]}
    return rows


def load_snapshot(trial, inputs):
    """Accept the current refresh pointer or an older inventory, never guess a release."""
    latest = trial / "history_checks/latest.json"
    if latest.is_file():
        pointer, _ = read_bound_json(latest, inputs)
        check = portable(pointer["path"]).resolve()
        report, digest = read_bound_json(check / "report.json", inputs)
        if digest != pointer["report_sha256"]:
            raise ValueError("history_checks/latest.json report hash mismatch")
        history, _ = read_bound_json(check / "reconciled_history.json", inputs)
        files = read_bound_jsonl(check / "scanned_files.jsonl", inputs)
        snapshot = str(check)
        prior_issues = list(report.get("errors", []))
    else:
        candidates = [trial / "history/reconciled_history.json", trial / "history/history_inventory.json"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError("trial needs history_checks/latest.json or a history inventory")
        history, _ = read_bound_json(path, inputs)
        files, prior_issues, snapshot = history.get("files", []), [], str(path)
        if not files:
            raise ValueError("no history file inventory; use the current re-enrollment trial")
    if not isinstance(history.get("source_groups"), list):
        raise ValueError("invalid source_groups in history snapshot")
    old = {portable(row["path"]).resolve(): row["sha256"] for row in files}
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in old.values()):
        raise ValueError("invalid history-file hash")
    return history, old, snapshot, prior_issues + list(history.get("issues", []))


def within(path, parent):
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def list_videos(root, classes, issues):
    by_video, total = defaultdict(list), 0
    def onerror(exc):
        issues.append({"scope": "media", "issue": "TRAIN_DIRECTORY_UNREADABLE", "path": str(exc.filename)})
    for directory, dirs, files in os.walk(root, onerror=onerror, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in PRUNE)
        for name in sorted(files):
            if Path(name).suffix.lower() != ".mp4":
                continue
            total += 1
            video = Path(name).stem
            if "_label_" not in video or not (anomaly_codes(video) & classes):
                continue
            path = (Path(directory) / name).resolve()
            if not within(path, root):
                issues.append({"scope": "media", "issue": "VIDEO_LINK_OUTSIDE_TRAIN_ROOT", "path": str(path)})
                continue
            by_video[video].append(path)
    return by_video, total


def anchor_audit(video, root, inputs):
    """Parse the existing selected_segments schema strictly; never use score to rank."""
    files, spans, problems = [], set(), []
    per_file_spans = []
    for path in (root / video / "selected_frames.json", root / (video + ".mp4") / "selected_frames.json"):
        if not path.is_file():
            continue
        entry = {"path": str(path.resolve()), "coordinate_convention": "zero_based_inclusive"}
        files.append(entry)
        current = set()
        try:
            data, digest = read_bound_json(path, inputs)
            entry["sha256"] = digest
            if not isinstance(data, dict) or not isinstance(data.get("selected_segments"), list):
                raise ValueError("selected_segments must be a list")
            declared = data.get("video")
            if declared is not None and Path(str(declared).replace("\\", "/")).name.removesuffix(".mp4") != video:
                raise ValueError("declared video differs from the directory video ID")
            if data.get("covered_by_72B") is False:
                raise ValueError("anchor file explicitly denies 72B coverage")
            entry["declared_original_video_path"] = data.get("video_path")
            entry["covered_by_72B"] = data.get("covered_by_72B")
            entry["selection_mode"] = data.get("selection_mode")
            for ordinal, segment in enumerate(data["selected_segments"]):
                if not isinstance(segment, dict):
                    raise ValueError(f"segment {ordinal} is not an object")
                start = segment.get("segment_start")
                end = segment.get("segment_end_inclusive")
                length = segment.get("segment_len")
                if type(start) is not int or start < 0:
                    raise ValueError(f"segment {ordinal} has invalid start")
                if end is None and type(length) is int and length > 0:
                    end = start + length - 1
                if type(end) is not int or end < start:
                    raise ValueError(f"segment {ordinal} has no valid explicit span")
                if length is not None and (type(length) is not int or length != end - start + 1):
                    raise ValueError(f"segment {ordinal} has inconsistent length")
                current.add((start, end))
            entry["valid_record_count"] = len(current)
            per_file_spans.append(current)
            spans.update(current)
        except (OSError, ValueError, TypeError) as exc:
            entry["error"] = str(exc)
            problems.append("INVALID_OR_CHANGED_ANCHOR_FILE")
    if len(per_file_spans) > 1 and any(s != per_file_spans[0] for s in per_file_spans[1:]):
        problems.append("CONFLICTING_ANCHOR_FILES")
    qualifying = sorted((a, b) for a, b in spans if b - a + 1 >= 8)
    if not files:
        problems.append("MISSING_POSITIVE_ANCHOR_FILE")
    elif not qualifying:
        problems.append("NO_EXPLICIT_AT_LEAST_8_FRAME_POSITIVE_ANCHOR")
    return {"files": files, "positive_spans_inclusive": [list(s) for s in sorted(spans)],
            "qualifying_anchor_count": len(qualifying), "problems": sorted(set(problems)),
            "structurally_traceable_anchor": bool(qualifying) and not problems,
            "label_scope": "existing weak positive anchor, not full-window gold or class-specific confirmation",
            "scores_used_for_selection": False}


def probe_metadata(path, executable):
    # No -count_frames, no frame extraction, no duration*fps frame-count estimate.
    result = subprocess.run([executable, "-v", "error", "-protocol_whitelist", "file",
                             "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                             "-of", "json", str(path)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    value = streams[0].get("nb_frames") if streams else None
    count = int(value) if isinstance(value, (str, int)) and str(value).isdigit() else None
    return {"frame_count": count if count and count > 0 else None,
            "method": "ffprobe_container_nb_frames_only", "verified_by_decoding": False}


def example_windows(anchor, frame_count):
    result = []
    for a, b in anchor["positive_spans_inclusive"]:
        if b - a + 1 < 8:
            continue
        if frame_count is not None and (frame_count < 96 or a >= frame_count or b >= frame_count):
            continue
        start = max(0, (a + b) // 2 - 48)
        if frame_count is not None:
            start = min(start, frame_count - 96)
        if min(start + 96, b + 1) - max(start, a) < 8:
            continue
        if result and start < result[-1]["end_frame_exclusive"]:
            continue
        result.append({"start_frame": start, "end_frame_exclusive": start + 96,
                       "sampled_indices_if_later_screened": [start + int(i * 95 / 7) for i in range(8)],
                       "range_check": "container_metadata_only" if frame_count is not None else "PENDING_FRAME_COUNT",
                       "media_verified": False, "not_an_enrollment_candidate": True})
        if len(result) == 2:
            break
    return result


def video_audit(video, paths, args, inputs, issues):
    anchor = anchor_audit(video, args.anchors_root, inputs)
    flags, media = list(anchor["problems"]), []
    for path in sorted(paths):
        entry = {"path": str(path)}
        try:
            before = signature(path)
            entry.update(size_bytes=before[0], mtime_ns=before[1])
            if before[0] <= 0:
                flags.append("EMPTY_MEDIA_FILE")
            if args.probe_metadata and len(paths) == 1 and before[0] > 0:
                entry.update(probe_metadata(path, args.ffprobe))
                if before != signature(path):
                    raise ValueError("video changed during metadata probe")
            else:
                entry.update(frame_count=None, method="not_probed", verified_by_decoding=False)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            flags.append("MEDIA_METADATA_UNAVAILABLE")
            issues.append({"scope": "media", "path": str(path), "issue": type(exc).__name__, "detail": str(exc)[:300]})
        media.append(entry)
    if len(paths) > 1:
        flags.append("DUPLICATE_VIDEO_ID_PATHS_NOT_CONTENT_VERIFIED")
    count = media[0].get("frame_count") if len(media) == 1 else None
    if count is None:
        flags.append("FULL_WINDOW_FRAME_RANGE_PENDING")
    elif count < 96:
        flags.append("MEDIA_SHORTER_THAN_96_FRAMES")
    if count is not None and any(b >= count for _, b in anchor["positive_spans_inclusive"]):
        flags.append("ANCHOR_OUTSIDE_CONTAINER_FRAME_RANGE")
    codes = sorted(anomaly_codes(video))
    legacy = "B1_weak_positive" if "B1" in codes else "B4_weak_positive" if "B4" in codes else "other_class_canary"
    return {"video_id": video, "source_group": group_id(video), "filename_categories": codes,
            "target_categories": sorted(set(codes) & set(args.classes)), "media_paths": media,
            "positive_anchor": anchor, "illustrative_windows_not_screened": example_windows(anchor, count),
            "legacy_positive_pool_priority": legacy,
            "multilabel_canary_priority_warning": legacy != "other_class_canary",
            "technical_flags": sorted(set(flags)), "eligible_for_enrollment": False}


def history_files(xd_root, extra_roots, old_files, out, names, issues):
    files = set(old_files)
    roots = sorted({*(p / "runs" for p in xd_root.iterdir() if (p / "runs").is_dir()), *extra_roots})
    def onerror(exc):
        issues.append({"scope": "history", "issue": "HISTORY_DIRECTORY_UNREADABLE", "path": str(exc.filename)})
    for root in roots:
        if not root.is_dir():
            issues.append({"scope": "history", "issue": "MISSING_HISTORY_ROOT", "path": str(root)})
            continue
        for directory, dirs, found in os.walk(root, onerror=onerror, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and not within((Path(directory) / d).resolve(), out))
            files.update((Path(directory) / n).resolve() for n in set(found) & names)
    return sorted(files), roots


def alias_key(value):
    # A REVIEW flag only; never merge or release case/punctuation variants.
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def decide_status(kinds, excluded, history_incomplete, aliases, technical_ok):
    if "execution_record" in kinds:
        return "RETAIN_EXCLUSION_EXECUTION_EVIDENCE"
    if kinds & {"execution_record_unverified_mode", "conflicting_mock_evidence"}:
        return "RETAIN_EXCLUSION_UNVERIFIED_EXECUTION"
    if history_incomplete or aliases:
        return "REVIEW_UNRESOLVED_HISTORY"
    if kinds and kinds <= {"planned_registration", "mock_config_execution", "mock_marker_execution"}:
        return "REVIEW_PLANNED_OR_MOCK_ONLY"
    if kinds or excluded:
        return "REVIEW_UNRESOLVED_HISTORY"
    if not technical_ok:
        return "NEEDS_ANCHOR_OR_MEDIA_EVIDENCE"
    return "POTENTIAL_FOR_TECHNICAL_SCREENING_NOT_APPROVED"


def source_reports(videos, evidence, history, incomplete, all_known_groups):
    by_group, by_evidence, aliases = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in videos:
        by_group[row["source_group"]].append(row)
    for row in evidence:
        by_evidence[row["source_group"]].append(row)
    for group in all_known_groups | set(by_group):
        aliases[alias_key(group)].add(group)
    result, reviews = [], []
    bad_media = {"EMPTY_MEDIA_FILE", "MEDIA_METADATA_UNAVAILABLE", "MEDIA_SHORTER_THAN_96_FRAMES",
                 "DUPLICATE_VIDEO_ID_PATHS_NOT_CONTENT_VERIFIED", "ANCHOR_OUTSIDE_CONTAINER_FRAME_RANGE"}
    for group, members in sorted(by_group.items()):
        records = by_evidence[group]
        kinds = {r["kind"] for r in records}
        suspects = sorted(aliases[alias_key(group)] - {group})
        technical = any(v["positive_anchor"]["structurally_traceable_anchor"] and
                        v["illustrative_windows_not_screened"] and not (set(v["technical_flags"]) & bad_media)
                        for v in members)
        excluded = group in history["source_groups"]
        status = decide_status(kinds, excluded, incomplete, suspects, technical)
        row = {"source_group": group, "primary_status": status, "status_explanation": STATUS_HELP[status],
               "video_ids": sorted(v["video_id"] for v in members),
               "target_categories": sorted({c for v in members for c in v["target_categories"]}),
               "filename_categories": sorted({c for v in members for c in v["filename_categories"]}),
               "excluded_in_input_snapshot": excluded,
               "prior_release_in_input_snapshot": group in history.get("released_source_groups", []),
               "history_evidence_kinds": sorted(kinds), "history_evidence": records,
               "possible_alias_source_groups_not_merged": suspects,
               "technical_flags": sorted({f for v in members for f in v["technical_flags"]}),
               "structurally_traceable_anchor_videos": sum(v["positive_anchor"]["structurally_traceable_anchor"] for v in members),
               "has_provisional_anchor_window": technical, "eligible_for_enrollment": False,
               "remote_execution_authorized": False}
        result.append(row)
        if status not in RETAIN:
            reviews.append({"schema": VERSION + "_manual_review_draft", "source_group": group,
                            "audit_status": status, "target_categories": row["target_categories"],
                            "video_ids": row["video_ids"], "reviewer_id": "", "disposition": "pending",
                            "checked_actual_fit_threshold_evaluation_use": False,
                            "checked_discovery_prompt_design_human_viewing": False,
                            "checked_external_history_and_source_aliases": False,
                            "evidence_notes": "", "eligible_for_enrollment": False,
                            "import_supported": False, "automatic_release_effect": False})
    return result, reviews


def run(args):
    from audit_v91_history import SUPPLEMENTAL, scan_file

    started = time.time()
    inputs, issues = {}, []
    history, old_files, snapshot, prior_issues = load_snapshot(args.trial, inputs)
    issues.extend({"scope": "history", "issue": "PRIOR_SNAPSHOT_ISSUE", "detail": item} for item in prior_issues)
    video_paths, total_paths = list_videos(args.train_root, set(args.classes), issues)
    targets = {group_id(v) for v in video_paths}
    files, roots = history_files(PROJECT.parent, args.history_root, old_files, args.out,
                                HISTORY_NAMES | SUPPLEMENTAL, issues)
    if not files:
        raise ValueError("no recognized history files found")
    for path in list(inputs) + [str(p) for p in files]:
        if within(portable(path).resolve(), args.out):
            raise ValueError("audit output would contain a history input")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "audit_summary.json", {"version": VERSION, "status": "RUNNING",
               "remote_calls": 0, "released_sources": 0, "eligible_for_enrollment": False})
    print(f"[videos] matched={len(video_paths)} source_groups={len(targets)} total_paths={total_paths}; API=0", flush=True)
    videos = []
    for ordinal, (video, paths) in enumerate(sorted(video_paths.items()), 1):
        videos.append(video_audit(video, paths, args, inputs, issues))
        if ordinal % 50 == 0 or ordinal == len(video_paths):
            print(f"[anchors] {ordinal}/{len(video_paths)} videos; no image decoding", flush=True)
    write_jsonl(args.out / "canary_video_audit.jsonl", videos)
    evidence, scanned, changed, all_known = [], [], [], set(history["source_groups"])
    for ordinal, path in enumerate(files, 1):
        try:
            before = signature(path) if path.is_file() else None
            rows, errors, meta = scan_file(path, targets, args.max_history_file_mb * 1024**2)
            if before and (not path.is_file() or before != signature(path)):
                errors.append({"path": str(path), "issue": "HISTORY_CHANGED_DURING_SCAN"})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rows, meta = [], {}
            errors = [{"path": str(path), "issue": "HISTORY_SCAN_FAILED", "detail": str(exc)[:300]}]
        for row in rows:
            row["new_since_inventory"] = path not in old_files
        evidence.extend(rows)
        issues.extend({**error, "scope": "history"} for error in errors)
        if meta:
            scanned.append(meta)
            all_known.update(meta["source_groups_found"])
            inputs[str(path)] = {"sha256": meta["sha256"], "bytes": meta["bytes"]}
            if old_files.get(path) != meta["sha256"]:
                changed.append({"path": str(path), "previous_sha256": old_files.get(path), "current_sha256": meta["sha256"]})
        if ordinal % 10 == 0 or ordinal == len(files):
            print(f"[history] {ordinal}/{len(files)} files; issues={len(issues)}; API=0", flush=True)
    incomplete = any(i["scope"] == "history" for i in issues)
    sources, reviews = source_reports(videos, evidence, history, incomplete, all_known)
    code_inputs = [Path(__file__), PROJECT / "tools/audit_v91_history.py",
                   PROJECT / "tools/event_decision/hard_trial.py", PROJECT / "tools/event_decision/contracts.py",
                   PROJECT / "tools/event_decision/safety.py"]
    outputs = {"canary_source_audit.jsonl": sources, "manual_review_template.jsonl": reviews,
               "history_evidence.jsonl": evidence, "scanned_history_files.jsonl": scanned,
               "audit_issues.jsonl": issues, "changed_history_files.jsonl": changed}
    for name, rows in outputs.items():
        write_jsonl(args.out / name, rows)
    write_json(args.out / "input_manifest.json", {
        "version": VERSION, "trial": str(args.trial), "snapshot": snapshot,
        "train_root": str(args.train_root), "anchors_root": str(args.anchors_root),
        "history_roots": [str(r) for r in roots], "classes": args.classes,
        "history_file_max_bytes": args.max_history_file_mb * 1024**2,
        "input_files": inputs, "code_hashes": {str(p): file_sha256(p) for p in code_inputs},
        "video_integrity": "path/size/mtime only, not a full-video content or visual-duplicate verification",
        "probe_metadata": args.probe_metadata, "no_source_release_or_import": True})
    counts = dict(Counter(s["primary_status"] for s in sources))
    per_class = {}
    for code in args.classes:
        matching = [s for s in sources if code in s["target_categories"]]
        per_class[code] = {"videos": sum(code in v["target_categories"] for v in videos),
                           "source_groups": len(matching),
                           "source_status_counts": dict(Counter(s["primary_status"] for s in matching)),
                           "traceable_anchor_source_groups": len({v["source_group"] for v in videos
                               if code in v["target_categories"] and v["positive_anchor"]["structurally_traceable_anchor"]}),
                           "approved_sources": 0}
    summary = {"version": VERSION, "status": "AUDIT_INCOMPLETE_REVIEW_ISSUES" if issues or not videos else "AUDIT_COMPLETE_REQUIRES_SOURCE_REVIEW",
               "train_video_paths_scanned": total_paths, "matched_video_ids": len(videos),
               "matched_video_paths": sum(len(p) for p in video_paths.values()), "source_groups": len(sources),
               "class_counts": per_class, "source_status_counts": counts,
               "history_files_requested": len(files), "history_files_scanned": len(scanned),
               "history_scan_incomplete": incomplete, "history_files_new_or_changed": len(changed),
               "issue_count": len(issues), "manual_review_rows": len(reviews),
               "metadata_probe_enabled": args.probe_metadata, "new_video_decoding_performed": False,
               "remote_calls": 0, "released_sources": 0, "enrolled_windows": 0,
               "import_supported": False, "ready_for_reenrollment": False,
               "remote_execution_authorized": False, "elapsed_seconds": round(time.time() - started, 3),
               "limitations": ["Recognized local history only; no hit is not proof of non-use.",
                   "Mock runs can still expose sources through viewing or design.",
                   "Filename labels and weak anchors are not full-window semantic gold.",
                   "Container frame counts are not a complete media or fingerprint verification.",
                   "Alias flags are heuristic, not a comprehensive identity or visual-duplicate audit.",
                   "Existing exclusions and prior 35 human reviews were not modified.",
                   "Step 1C import and Step 1D candidate generation are NOT implemented here."]}
    write_json(args.out / "audit_summary.json", summary)
    lines = ["# Canary Source Audit (Step 1B)", "", "This is a coordinator audit, not a blind visual review or an allow-list.",
             "No sources were released. Do not pass these files to --additional-candidates.", "",
             f"Status: `{summary['status']}`. Sources: {len(sources)}. Issues: {len(issues)}.", "",
             "## Read in This Order", "",
             "1. `audit_summary.json`: counts and incomplete-scan warnings.",
             "2. `canary_source_audit.jsonl`: one source per row, status and historical file/line evidence.",
             "3. `canary_video_audit.jsonl`: exact video paths, anchor hashes, spans and pending 96-frame checks.",
             "4. `manual_review_template.jsonl`: pending HISTORY-use questions, NOT anomaly labels; no importer/release effect.",
             "5. `audit_issues.jsonl`: missing/oversized/changing/unreadable history or media must not be ignored.", "",
             "History reviewers should inspect the referenced local records, not rewatch clips to choose favorable model cases.",
             "Keep the generated template unchanged; store actual reviewer answers separately for a future audited importer.", "",
             "## Status Meanings", ""]
    lines.extend(f"- `{status}`: {message}" for status, message in STATUS_HELP.items())
    lines.extend(["", "## Source Index", "", "| Source | Categories | Status | Videos |", "| --- | --- | --- | ---: |"])
    for s in sources:
        source = s["source_group"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {source} | {', '.join(s['target_categories'])} | {s['primary_status']} | {len(s['video_ids'])} |")
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] {args.out / 'README.md'}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=portable, default=DEFAULT_TRIAL)
    parser.add_argument("--out", type=portable, required=True, help="new independent output directory; existing paths are refused")
    parser.add_argument("--train-root", type=portable, default=portable("G:/Dataset/XDViolence/train"))
    parser.add_argument("--anchors-root", type=portable, default=DEFAULT_ANCHORS)
    parser.add_argument("--classes", nargs="+", choices=sorted(CATEGORIES), default=["B2", "B5", "B6"])
    parser.add_argument("--history-root", action="append", type=portable, default=[], help="additional history root (repeatable); does not replace normal history search")
    parser.add_argument("--max-history-file-mb", type=int, default=1024)
    parser.add_argument("--probe-metadata", action="store_true", help="optional local ffprobe nb_frames only; no frame decoding")
    args = parser.parse_args(argv)
    args.classes = sorted(set(args.classes))
    for key in ("trial", "out", "train_root", "anchors_root"):
        setattr(args, key, getattr(args, key).resolve())
    args.history_root = [r.resolve() for r in args.history_root]
    if args.max_history_file_mb <= 0:
        parser.error("--max-history-file-mb must be positive")
    for root in (args.trial, args.train_root, args.anchors_root):
        if not root.is_dir():
            parser.error(f"required directory missing: {root}")
        if within(args.out, root) or within(root, args.out):
            parser.error("output must be separate from trial, video and anchor inputs")
    if args.out.exists():
        parser.error("output already exists; preserve it and choose a new TAG")
    args.ffprobe = shutil.which("ffprobe") if args.probe_metadata else None
    if args.probe_metadata and not args.ffprobe:
        parser.error("ffprobe not found; omit --probe-metadata for JSON/directory-only audit")
    return args


def main(argv=None):
    guard = OfflineGuard()
    guard.install()
    args = parse_args(argv)
    try:
        summary = run(args)
        guard.assert_no_remote_calls()
        return 2 if summary["status"].startswith("AUDIT_INCOMPLETE") else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if (args.out / "audit_summary.json").is_file():
            write_json(args.out / "audit_summary.json", {"version": VERSION, "status": "AUDIT_FAILED",
                       "error": str(exc), "ready_for_reenrollment": False, "remote_execution_authorized": False})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
