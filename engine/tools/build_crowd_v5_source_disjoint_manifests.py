#!/usr/bin/env python3
"""Build source-disjoint V5 packets from confident training anchors."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, write_json, write_jsonl
from selection import is_pure_normal_video, label_codes, source_group_id


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256("\0".join([str(seed), *map(str, parts)]).encode()).hexdigest()


def _confidence(row: Mapping[str, Any]) -> float | None:
    for key in ("anchor_confidence", "confidence", "teacher_confidence"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    if str(row.get("anchor_status", row.get("anchor_type", ""))).lower() in {"confident", "confident_positive", "confident_negative"}:
        return 1.0
    return None


def _phase(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("event_phase", row.get("gt", {}).get("event_phase", ""))).lower()
    if "post" in explicit or "after" in explicit:
        return "post_event"
    if "pre" in explicit or "bound" in explicit or "onset" in explicit:
        return "pre_or_boundary"
    if int(row.get("y_true", 0)) == 1:
        return "active"
    return "not_applicable"


def _stratum(row: Mapping[str, Any]) -> str:
    video_id = str(row.get("video_id", ""))
    labels = set(label_codes(video_id)) | {str(value) for value in row.get("label_codes", [])}
    phase = _phase(row)
    y_true = int(row.get("y_true", 0))
    if y_true == 1 and "B4" in labels:
        return "b4_active_anchor" if phase == "active" else "b4_boundary_anchor"
    if y_true == 1 and "B1" in labels:
        return "b1_crowd_fight_anchor"
    if y_true == 1:
        return "other_positive_anchor"
    if is_pure_normal_video(video_id):
        return "pure_normal_anchor"
    if phase == "post_event":
        return "post_event_normal_anchor"
    if bool(row.get("hard_context_normal")) or "hard" in str(row.get("anchor_type", "")).lower():
        return "hard_context_normal_anchor"
    return "other_normal_anchor"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--windows-per-source-group", type=int, default=2)
    parser.add_argument("--max-per-stratum-per-split", type=int, default=40)
    parser.add_argument("--max-pure-normal-per-split", type=int, default=100)
    parser.add_argument("--minimum-anchor-confidence", type=float, default=0.8)
    parser.add_argument("--allow-missing-anchor-confidence", action="store_true")
    parser.add_argument("--allow-nontraining-source", action="store_true")
    parser.add_argument("--minimum-calibration-windows", type=int, default=40)
    parser.add_argument("--minimum-validation-windows", type=int, default=40)
    parser.add_argument("--minimum-tail-normals-per-split", type=int, default=5)
    parser.add_argument("--minimum-pure-normal-per-split", type=int, default=50)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.95:
        raise SystemExit("calibration fraction must be in [0.05, 0.95]")
    excluded_groups = set()
    for path in args.exclude_manifest:
        if path.is_file():
            for row in iter_jsonl(path):
                excluded_groups.add(str(row.get("source_group", "")) or source_group_id(str(row.get("video_id", ""))))
    by_group: dict[str, list[dict]] = defaultdict(list)
    rejected = Counter()
    for raw in iter_jsonl(args.records):
        row = dict(raw)
        key, video_id = str(row.get("segment_key", "")), str(row.get("video_id", ""))
        group = str(row.get("source_group", "")) or source_group_id(video_id)
        if not key or not video_id or not group or group in excluded_groups:
            rejected["missing_identity_or_excluded"] += 1
            continue
        split = str(row.get("dataset_split", row.get("split", row.get("source_split", "")))).lower()
        source = str(row.get("source", "")).lower()
        if not args.allow_nontraining_source and "train" not in split and "train" not in source:
            rejected["not_explicitly_training"] += 1
            continue
        confidence = _confidence(row)
        if confidence is None and not args.allow_missing_anchor_confidence:
            rejected["missing_anchor_confidence"] += 1
            continue
        if confidence is not None and confidence < args.minimum_anchor_confidence:
            rejected["low_anchor_confidence"] += 1
            continue
        if row.get("y_true") not in (0, 1):
            rejected["missing_binary_anchor_label"] += 1
            continue
        stratum = _stratum(row)
        if stratum == "pure_normal_anchor" and not is_pure_normal_video(video_id):
            rejected["invalid_pure_normal_identity"] += 1
            continue
        row.update({
            "source_group": group, "dataset_split": "train",
            "anchor_confidence": confidence, "crowd_v5_stratum": stratum,
            "event_phase": _phase(row),
        })
        by_group[group].append(row)
    if len(by_group) < 2:
        raise SystemExit(f"insufficient confident training source groups; rejected={dict(rejected)}")
    assignments = {}
    for group in sorted(by_group):
        value = int(_rank(args.seed, "split", group)[:16], 16) / float(16**16)
        assignments[group] = "calibration" if value < args.calibration_fraction else "validation"
    if len(set(assignments.values())) < 2:
        ordered = sorted(assignments, key=lambda group: _rank(args.seed, "rebalance", group))
        assignments[ordered[0]], assignments[ordered[-1]] = "calibration", "validation"
    outputs: dict[str, list[dict]] = {"calibration": [], "validation": []}
    selected_source: dict[str, dict] = {}
    counts = {"calibration": Counter(), "validation": Counter()}
    for split_name in outputs:
        candidates = []
        for group, rows in by_group.items():
            if assignments[group] != split_name:
                continue
            by_stratum: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_stratum[str(row["crowd_v5_stratum"])].append(row)
            for stratum, stratum_rows in by_stratum.items():
                ordered = sorted(stratum_rows, key=lambda row: _rank(args.seed, split_name, group, stratum, row["segment_key"]))
                candidates.extend(ordered[:max(1, args.windows_per_source_group)])
        candidates.sort(key=lambda row: (str(row["crowd_v5_stratum"]), _rank(args.seed, split_name, row["segment_key"])))
        for row in candidates:
            stratum = str(row["crowd_v5_stratum"])
            stratum_limit = (
                args.max_pure_normal_per_split
                if stratum == "pure_normal_anchor"
                else args.max_per_stratum_per_split
            )
            if stratum_limit > 0 and counts[split_name][stratum] >= stratum_limit:
                continue
            outputs[split_name].append({
                "segment_key": row["segment_key"], "video_id": row["video_id"],
                "source_group": row["source_group"], "split_role": f"v5_{split_name}",
                "stratum": stratum, "event_phase": row["event_phase"],
                "anchor_confidence": row.get("anchor_confidence"),
            })
            selected_source[str(row["segment_key"])] = row
            counts[split_name][stratum] += 1
    calibration_groups = {row["source_group"] for row in outputs["calibration"]}
    validation_groups = {row["source_group"] for row in outputs["validation"]}
    overlap = calibration_groups & validation_groups
    if overlap:
        raise RuntimeError(f"source-group leakage: {sorted(overlap)[0]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in outputs.items():
        write_jsonl(args.out_dir / f"{split_name}_manifest.jsonl", rows)
    write_jsonl(args.out_dir / "selected_source_records.jsonl", sorted(selected_source.values(), key=lambda row: str(row["segment_key"])))
    tail_strata = {"post_event_normal_anchor", "hard_context_normal_anchor"}
    tail_counts = {
        split_name: sum(counts[split_name].get(stratum, 0) for stratum in tail_strata)
        for split_name in outputs
    }
    pure_normal_counts = {
        split_name: counts[split_name].get("pure_normal_anchor", 0)
        for split_name in outputs
    }
    ready = (
        len(outputs["calibration"]) >= args.minimum_calibration_windows
        and len(outputs["validation"]) >= args.minimum_validation_windows
        and all(value >= args.minimum_tail_normals_per_split for value in tail_counts.values())
        and all(value >= args.minimum_pure_normal_per_split for value in pure_normal_counts.values())
    )
    summary = {
        "version": "crowd_v5_confident_training_anchor_split_v2_true_pure_normal",
        "records": str(args.records), "records_sha256": file_sha256(args.records),
        "seed": args.seed, "rejected": dict(rejected), "excluded_source_groups": sorted(excluded_groups),
        "calibration": {"windows": len(outputs["calibration"]), "source_groups": len(calibration_groups), "strata": dict(counts["calibration"])},
        "validation": {"windows": len(outputs["validation"]), "source_groups": len(validation_groups), "strata": dict(counts["validation"])},
        "tail_normal_counts": tail_counts, "pure_normal_counts": pure_normal_counts,
        "source_group_overlap": sorted(overlap),
        "ready_for_api": ready,
        "readiness_checks": {
            "minimum_calibration_windows": args.minimum_calibration_windows,
            "minimum_validation_windows": args.minimum_validation_windows,
            "minimum_tail_normals_per_split": args.minimum_tail_normals_per_split,
            "minimum_pure_normal_per_split": args.minimum_pure_normal_per_split,
            "training_provenance_required": not args.allow_nontraining_source,
            "anchor_confidence_required": not args.allow_missing_anchor_confidence,
        },
    }
    write_json(args.out_dir / "split_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
