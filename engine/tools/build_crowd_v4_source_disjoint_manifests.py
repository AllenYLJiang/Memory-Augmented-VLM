#!/usr/bin/env python3
"""Build deterministic Crowd V4 calibration/validation manifests split by source group."""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, write_json, write_jsonl
from gt_annotations import label_window, load_annotations
from selection import is_pure_normal_video, label_codes, source_group_id


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256("\0".join([str(seed), *map(str, parts)]).encode()).hexdigest()


def _phase(row: Mapping[str, Any]) -> str:
    gt = row.get("gt", {}) if isinstance(row.get("gt"), Mapping) else {}
    explicit = str(gt.get("event_phase", "")).lower()
    subset = str(gt.get("temporal_subset", "")).lower()
    value = explicit or subset
    if "after" in value or "ending" in value:
        return "aftermath"
    if "bound" in value or "pre" in value or "onset" in value:
        return "boundary"
    return "active"


def _stratum(row: Mapping[str, Any]) -> str:
    video_id = str(row.get("video_id", ""))
    labels = label_codes(video_id)
    y_true = int(row.get("y_true", 0))
    if y_true == 1 and "B4" in labels:
        return f"b4_{_phase(row)}"
    if y_true == 1 and "B1" in labels:
        return "b1_crowd_fight"
    if y_true == 0 and is_pure_normal_video(video_id):
        return "pure_label_a"
    if y_true == 0:
        return "hard_or_context_negative"
    return "other_positive_canary"


def _excluded_groups(paths: list[Path]) -> set[str]:
    groups = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in iter_jsonl(path):
            group = str(row.get("source_group", "")) or source_group_id(str(row.get("video_id", "")))
            if group:
                groups.add(group)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--calibration-fraction", type=float, default=0.4)
    parser.add_argument("--windows-per-source-group", type=int, default=2)
    parser.add_argument("--max-per-stratum-per-split", type=int, default=0)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--minimum-calibration-windows", type=int, default=30)
    parser.add_argument("--minimum-validation-windows", type=int, default=50)
    parser.add_argument("--minimum-per-required-validation-stratum", type=int, default=3)
    parser.add_argument("--exclude-development-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.95:
        raise SystemExit("calibration fraction must be in [0.05, 0.95]")

    excluded = _excluded_groups(args.exclude_development_manifest)
    annotations = load_annotations(args.annotations) if args.annotations else None
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in iter_jsonl(args.records):
        key = str(row.get("segment_key", ""))
        video_id = str(row.get("video_id", ""))
        group = source_group_id(video_id)
        if key and video_id and group not in excluded:
            value = dict(row)
            video_path = str(value.get("video_path", ""))
            if len(video_path) >= 3 and video_path[1:3] in {":\\", ":/"}:
                value["video_path"] = f"/mnt/{video_path[0].lower()}/" + video_path[3:].replace("\\", "/")
            if annotations is not None:
                gt = label_window(
                    video_id, int(value.get("start_frame", 0)), int(value.get("end_frame", 0)),
                    annotations, known_normal=is_pure_normal_video(video_id),
                )
                if not gt.get("annotation_found") and not gt.get("known_normal"):
                    continue
                value["gt"] = gt
                value["y_true"] = int(gt["y_true_operational"])
            value["source_group"] = group
            value["crowd_v4_stratum"] = _stratum(value)
            by_group[group].append(value)
    if not by_group:
        raise SystemExit("no eligible source groups after development exclusion")

    assignments = {}
    for group in sorted(by_group):
        raw = int(_rank(args.seed, "split", group)[:16], 16) / float(16**16)
        assignments[group] = "calibration" if raw < args.calibration_fraction else "validation"
    # Ensure both sides exist even for very small smoke pools.
    if len(set(assignments.values())) < 2 and len(assignments) >= 2:
        ordered = sorted(assignments, key=lambda group: _rank(args.seed, "rebalance", group))
        assignments[ordered[0]] = "calibration"
        assignments[ordered[-1]] = "validation"

    outputs = {"calibration": [], "validation": []}
    counts = {"calibration": Counter(), "validation": Counter()}
    for split in ("calibration", "validation"):
        candidates = []
        for group, rows in by_group.items():
            if assignments[group] != split:
                continue
            by_stratum: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_stratum[str(row["crowd_v4_stratum"])].append(row)
            for stratum, stratum_rows in by_stratum.items():
                ordered = sorted(
                    stratum_rows,
                    key=lambda row: _rank(args.seed, split, group, stratum, row["segment_key"]),
                )
                candidates.extend(ordered[:max(1, args.windows_per_source_group)])
        candidates.sort(key=lambda row: (
            str(row["crowd_v4_stratum"]), _rank(args.seed, split, row["source_group"], row["segment_key"])
        ))
        for row in candidates:
            stratum = str(row["crowd_v4_stratum"])
            if args.max_per_stratum_per_split > 0 and counts[split][stratum] >= args.max_per_stratum_per_split:
                continue
            outputs[split].append({
                "segment_key": row["segment_key"], "video_id": row["video_id"],
                "source_group": row["source_group"], "roles": [f"crowd_v4:{stratum}"],
                "event_phase": _phase(row), "crowd_v4_stratum": stratum,
            })
            counts[split][stratum] += 1

    calibration_groups = {row["source_group"] for row in outputs["calibration"]}
    validation_groups = {row["source_group"] for row in outputs["validation"]}
    overlap = calibration_groups & validation_groups
    if overlap:
        raise RuntimeError(f"internal source-group leakage: {sorted(overlap)[0]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        write_jsonl(args.out_dir / f"{split}_manifest.jsonl", rows)
    selected_keys = {
        str(row["segment_key"])
        for split in outputs.values() for row in split
    }
    selected_source_rows = sorted(
        [row for rows in by_group.values() for row in rows if str(row["segment_key"]) in selected_keys],
        key=lambda row: str(row["segment_key"]),
    )
    write_jsonl(args.out_dir / "selected_source_records.jsonl", selected_source_rows)
    required_strata = {
        "b4_active", "b4_aftermath", "b4_boundary", "b1_crowd_fight",
        "pure_label_a", "hard_or_context_negative",
    }
    missing_required = sorted(
        stratum for stratum in required_strata
        if counts["validation"].get(stratum, 0) < args.minimum_per_required_validation_stratum
    )
    ready = (
        len(outputs["calibration"]) >= args.minimum_calibration_windows
        and len(outputs["validation"]) >= args.minimum_validation_windows
        and not missing_required
    )
    summary = {
        "version": "crowd_v4_source_disjoint_split_v1",
        "records": str(args.records), "records_sha256": file_sha256(args.records),
        "seed": args.seed, "calibration_fraction": args.calibration_fraction,
        "windows_per_source_group": args.windows_per_source_group,
        "excluded_development_source_groups": sorted(excluded),
        "excluded_development_source_groups_count": len(excluded),
        "calibration": {
            "windows": len(outputs["calibration"]), "source_groups": len(calibration_groups),
            "strata": dict(sorted(counts["calibration"].items())),
        },
        "validation": {
            "windows": len(outputs["validation"]), "source_groups": len(validation_groups),
            "strata": dict(sorted(counts["validation"].items())),
        },
        "source_group_overlap": sorted(overlap),
        "selected_source_records": str(args.out_dir / "selected_source_records.jsonl"),
        "annotations": str(args.annotations or ""),
        "ready_for_api": ready,
        "readiness_checks": {
            "minimum_calibration_windows": args.minimum_calibration_windows,
            "minimum_validation_windows": args.minimum_validation_windows,
            "minimum_per_required_validation_stratum": args.minimum_per_required_validation_stratum,
            "missing_or_underfilled_validation_strata": missing_required,
        },
        "warning": (
            "This splitter cannot create missing strata. Inspect counts before API runs and supply a larger source pool if needed."
        ),
    }
    write_json(args.out_dir / "split_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
