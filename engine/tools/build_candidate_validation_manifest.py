#!/usr/bin/env python3
"""Build a deterministic, source-disjoint candidate-directed window manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, write_json, write_jsonl
from graph_catalog import infer_family
from selection import is_pure_normal_video, label_codes, source_group_id


CLASS_FAMILY = {
    "B1": "human_interaction", "B2": "impulse_or_blast", "B4": "crowd",
    "B5": "human_interaction", "B6": "traffic", "G": "impulse_or_blast",
}


def _rank(seed: int, *parts: Any) -> str:
    return hashlib.sha256("\0".join([str(seed), *map(str, parts)]).encode()).hexdigest()


def _candidate_family(row: Mapping[str, Any]) -> str:
    graph = row.get("graph", {})
    return str(graph.get("family") or infer_family(
        str(graph.get("key", "")), str(graph.get("title", "")), str(graph.get("polarity", "normal")),
    ))


def _target_labels(row: Mapping[str, Any]) -> list[str]:
    key = str(row.get("graph", {}).get("key", ""))
    family = _candidate_family(row)
    if key == "rescue_or_assist_interaction" or family in {"assist_or_rescue", "human_interaction"}:
        return ["B1", "B5"]
    if family == "traffic" or "traffic" in key:
        return ["B6"]
    if family == "crowd" or "crowd" in key:
        return ["B4"]
    if family == "impulse_or_blast":
        return ["B2", "G"]
    return []


def _phase(record: Mapping[str, Any]) -> str:
    subset = str(record.get("gt", {}).get("temporal_subset", ""))
    if "boundary" in subset:
        return "boundary"
    if "core" in subset:
        return "active"
    overlap = float(record.get("gt", {}).get("overlap_fraction", 0.0) or 0.0)
    return "active" if overlap >= 2.0 / 3.0 else "unknown"


def _event_phase(record: Mapping[str, Any]) -> str:
    value = str(record.get("gt", {}).get("event_phase", ""))
    return value or _phase(record)


def _matches_slice(record: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    video_id = str(record.get("video_id", ""))
    labels = label_codes(video_id)
    required_labels = {str(value) for value in spec.get("labels", [])}
    if required_labels and not (labels & required_labels):
        return False
    if "y_true" in spec and int(record.get("y_true", 0)) != int(spec["y_true"]):
        return False
    if "known_normal" in spec and is_pure_normal_video(video_id) != bool(spec["known_normal"]):
        return False
    phases = {str(value) for value in spec.get("event_phases", [])}
    if phases and _event_phase(record) not in phases:
        return False
    return True


def build(records_path: Path, registry_path: Path, out_path: Path, seed: int,
          windows_per_video: int, max_per_role: int,
          include_hypothesis_ids: list[str] | None = None,
          exclude_hypothesis_ids: list[str] | None = None,
          exclude_videos_files: list[Path] | None = None,
          exclude_source_groups_files: list[Path] | None = None,
          slice_config: Path | None = None) -> dict:
    records = list(iter_jsonl(records_path))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_candidates = [row for row in registry.get("candidate_graphs", []) if row.get("status") == "candidate"]
    include = {str(value) for value in (include_hypothesis_ids or []) if str(value)}
    exclude = {str(value) for value in (exclude_hypothesis_ids or []) if str(value)}
    candidates = [
        row for row in registry_candidates
        if (not include or str(row.get("id")) in include) and str(row.get("id")) not in exclude
    ]
    missing = include - {str(row.get("id")) for row in registry_candidates}
    if missing:
        raise ValueError("unknown included hypotheses: " + ", ".join(sorted(missing)))
    excluded_videos = {
        line.strip() for path in (exclude_videos_files or []) if Path(path).is_file()
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    }
    explicit_excluded_groups = {
        line.strip() for path in (exclude_source_groups_files or []) if Path(path).is_file()
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    }
    slice_policy: dict[str, Any] = {}
    if slice_config:
        import yaml  # type: ignore
        slice_policy = yaml.safe_load(Path(slice_config).read_text(encoding="utf-8")) or {}
    role_max = {str(key): int(value) for key, value in slice_policy.get("role_max", {}).items()}
    global_excluded = {
        str(value) for candidate in candidates for value in candidate.get("support_source_groups", [])
    }
    chosen: dict[str, dict] = {}
    coverage = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        polarity = str(candidate.get("graph", {}).get("polarity", "normal"))
        family = _candidate_family(candidate)
        target_labels = _target_labels(candidate)
        excluded = set(str(value) for value in candidate.get("support_source_groups", []))
        pools: dict[str, list[dict]] = defaultdict(list)
        eligible_records = []
        for record in records:
            video_id = str(record.get("video_id", ""))
            group = source_group_id(str(record.get("video_id", "")))
            if video_id in excluded_videos or group in global_excluded or group in explicit_excluded_groups:
                continue
            eligible_records.append(record)
            y_true = int(record.get("y_true", 0))
            labels = label_codes(str(record.get("video_id", "")))
            expected_families = {CLASS_FAMILY[code] for code in labels if code in CLASS_FAMILY}
            phase = _phase(record)
            if phase == "boundary":
                role = "boundary"
            elif y_true == 0 and is_pure_normal_video(video_id):
                role = "pure_normal"
            elif polarity == "abnormal" and y_true == 1 and family in expected_families:
                role = "target_positive"
            elif polarity == "normal" and y_true == 0:
                role = "target_positive"
            elif (polarity == "abnormal" and y_true == 0) or (polarity == "normal" and y_true == 1):
                role = "hard_normal"
            else:
                role = "canary"
            pools[role].append(record)
        counts = {}
        for role in ("target_positive", "pure_normal", "hard_normal", "boundary", "canary"):
            by_video: dict[str, list[dict]] = defaultdict(list)
            for record in pools.get(role, []):
                by_video[str(record.get("video_id", ""))].append(record)
            selected = []
            for video_id, values in sorted(by_video.items(), key=lambda item: (_rank(seed, candidate_id, role, item[0]), item[0])):
                values.sort(key=lambda row: (_rank(seed, candidate_id, role, row.get("segment_key")), str(row.get("segment_key"))))
                selected.extend(values[:max(1, int(windows_per_video))])
                role_limit = int(role_max.get(role, max_per_role))
                if role_limit > 0 and len(selected) >= role_limit:
                    selected = selected[:role_limit]
                    break
            counts[role] = len(selected)
            for record in selected:
                key = str(record.get("segment_key"))
                row = chosen.setdefault(key, {
                    "segment_key": key,
                    "video_id": record.get("video_id"),
                    "source_group": source_group_id(str(record.get("video_id", ""))),
                    "candidate_ids": [],
                    "roles": [],
                    "selection_reasons": [],
                    "event_phase": _event_phase(record),
                })
                if candidate_id not in row["candidate_ids"]:
                    row["candidate_ids"].append(candidate_id)
                if role not in row["roles"]:
                    row["roles"].append(role)
                row["selection_reasons"].append(f"{candidate_id}:{role}:{family}")
        slice_counts = {}
        configured_slices = slice_policy.get("required_slices", [])
        target_slice_id = str(slice_policy.get("target_hypothesis_id", ""))
        if isinstance(configured_slices, list) and (not target_slice_id or target_slice_id == candidate_id):
            for spec in configured_slices:
                if not isinstance(spec, Mapping) or not spec.get("name"):
                    continue
                slice_name = str(spec["name"])
                pool = [record for record in eligible_records if _matches_slice(record, spec)]
                by_video: dict[str, list[dict]] = defaultdict(list)
                for record in pool:
                    by_video[str(record.get("video_id", ""))].append(record)
                selected_slice = []
                per_video = max(1, int(spec.get("windows_per_video", windows_per_video)))
                limit = int(spec.get("max_windows", 0))
                for video_id, values in sorted(
                    by_video.items(), key=lambda item: (_rank(seed, candidate_id, slice_name, item[0]), item[0])
                ):
                    values.sort(key=lambda row: (_rank(seed, candidate_id, slice_name, row.get("segment_key")), str(row.get("segment_key"))))
                    selected_slice.extend(values[:per_video])
                    if limit > 0 and len(selected_slice) >= limit:
                        selected_slice = selected_slice[:limit]
                        break
                for record in selected_slice:
                    key = str(record.get("segment_key"))
                    output = chosen.setdefault(key, {
                        "segment_key": key, "video_id": record.get("video_id"),
                        "source_group": source_group_id(str(record.get("video_id", ""))),
                        "candidate_ids": [], "roles": [], "selection_reasons": [],
                        "event_phase": _event_phase(record),
                    })
                    if candidate_id not in output["candidate_ids"]:
                        output["candidate_ids"].append(candidate_id)
                    role = f"slice:{slice_name}"
                    if role not in output["roles"]:
                        output["roles"].append(role)
                    output["selection_reasons"].append(f"{candidate_id}:{role}:{family}")
                slice_counts[slice_name] = {
                    "available_windows": len(pool),
                    "available_videos": len(by_video),
                    "requested_max_windows": limit,
                    "selected_windows": len(selected_slice),
                }
        coverage[candidate_id] = {
            "family": family,
            "polarity": polarity,
            "target_label_codes": target_labels,
            "expected_error_kind": "false_negative" if polarity == "abnormal" else "false_positive",
            "target_metric": "recall" if polarity == "abnormal" else "specificity",
            "target_direction": "increase",
            "excluded_source_groups": len(excluded),
            "roles": counts,
            "slices": slice_counts,
        }
    rows = sorted(chosen.values(), key=lambda row: (_rank(seed, row["source_group"], row["segment_key"]), row["segment_key"]))
    write_jsonl(out_path, rows)
    manifest = {
        "version": "candidate_directed_window_manifest_v1",
        "records": str(records_path),
        "registry": str(registry_path),
        "seed": int(seed),
        "windows_per_video": int(windows_per_video),
        "max_per_role": int(max_per_role),
        "windows": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "globally_excluded_discovery_source_groups": sorted(global_excluded),
        "explicit_excluded_videos": sorted(excluded_videos),
        "explicit_excluded_source_groups": sorted(explicit_excluded_groups),
        "exclusion_file_sha256": {
            str(path): file_sha256(path) for path in (exclude_videos_files or []) + (exclude_source_groups_files or [])
            if Path(path).is_file()
        },
        "registry_candidate_count": len(registry_candidates),
        "selected_candidate_count": len(candidates),
        "selected_hypothesis_ids": [str(row.get("id")) for row in candidates],
        "excluded_hypothesis_ids": sorted(exclude),
        "candidate_coverage": coverage,
        "slice_config": str(slice_config or ""),
        "slice_policy": slice_policy,
    }
    write_json(out_path.with_suffix(".summary.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--windows-per-video", type=int, default=3)
    parser.add_argument("--max-per-role", type=int, default=20)
    parser.add_argument("--include-hypothesis-ids", nargs="*", default=[])
    parser.add_argument("--exclude-hypothesis-ids", nargs="*", default=[])
    parser.add_argument("--exclude-videos-file", action="append", type=Path, default=[])
    parser.add_argument("--exclude-source-groups-file", action="append", type=Path, default=[])
    parser.add_argument("--slice-config", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        args.records, args.registry, args.out, args.seed, args.windows_per_video, args.max_per_role,
        args.include_hypothesis_ids, args.exclude_hypothesis_ids,
        args.exclude_videos_file, args.exclude_source_groups_file,
        args.slice_config,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
