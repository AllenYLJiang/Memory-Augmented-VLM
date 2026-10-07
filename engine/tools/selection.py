#!/usr/bin/env python3
"""GT-blind source-window loading and deterministic per-class video sampling."""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from common import file_sha256, iter_jsonl, write_json, write_jsonl


CLASS_NAMES = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}
CLASS_CODES = tuple(CLASS_NAMES)


def _label_tokens(video_id: str) -> Set[str]:
    """Return exact filename label tokens without treating movie-title dots as suffixes."""
    name = Path(str(video_id)).name
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    match = re.search(r"(?:^|_)label_([^/]+)$", name)
    if not match:
        return set()
    return {token for token in match.group(1).split("-") if token}


def label_codes(video_id: str) -> Set[str]:
    # Path.stem treats dots in movie titles as suffix separators even when there is
    # no file extension (for example Bad.Boys.II.2003__..._label_B2-G-0).
    return {token for token in _label_tokens(video_id) if token in CLASS_NAMES}


def is_pure_normal_video(video_id: str) -> bool:
    """True only for an explicit ``label_A`` filename with no anomaly-code token."""
    tokens = _label_tokens(video_id)
    return "A" in tokens and not any(code in tokens for code in CLASS_CODES)


def discover_prediction_files(source: Path) -> List[Path]:
    """Prefer a merged file only when it is at least as new as every shard.

    Long graph-vs-node runs are resumable; a stale merged file must not silently hide newer
    shard rows.
    """
    source = Path(source)
    if source.is_file():
        return [source]
    preferred = source / "merged" / "predictions.jsonl"
    shards = sorted(source.glob("shard_*/test/predictions.jsonl"))
    if preferred.is_file() and (
        not shards or preferred.stat().st_mtime >= max(path.stat().st_mtime for path in shards)
    ):
        return [preferred]
    files = shards or sorted(source.rglob("predictions.jsonl"))
    files = [path for path in files if path != preferred]
    if not files:
        raise FileNotFoundError(f"no predictions.jsonl found under {source}")
    return files


def source_group_id(video_id: str) -> str:
    """Movie-level source group used to prevent test-derived graph leakage across clips."""
    value = str(video_id)
    if value.startswith("v="):
        return value.split("__#", 1)[0]
    return value.split("__#", 1)[0]


def _quality(record: Mapping[str, Any]) -> tuple:
    ablation = record.get("node_graph_ablation")
    if not isinstance(ablation, Mapping):
        ablation = {}
    completeness = ablation.get("evidence_completeness")
    if not isinstance(completeness, Mapping):
        completeness = {}
    return (
        int(bool(ablation)),
        float(completeness.get("graph_entry_coverage", 0.0) or 0.0),
        float(completeness.get("model_node_state_coverage", 0.0) or 0.0),
        int(bool(record.get("segment_path"))),
    )


def source_graph_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    """Read a frozen source pair without using GT; nested ablation fields are fallbacks."""
    candidates = [(record.get("best_abnormal_graph"), record.get("best_normal_graph"))]
    ablation = record.get("node_graph_ablation")
    if isinstance(ablation, Mapping):
        for key in ("operational_graph", "required_structural_graph", "strict_complete_graph", "node_only"):
            value = ablation.get(key)
            if isinstance(value, Mapping):
                candidates.append((value.get("best_abnormal_graph"), value.get("best_normal_graph")))
    for abnormal, normal in candidates:
        abnormal_key = str(abnormal or "")
        normal_key = str(normal or "")
        if abnormal_key not in {"", "NONE"} and normal_key not in {"", "NONE"}:
            return abnormal_key, normal_key
    return "NONE", "NONE"


def load_deduplicated_records(paths: Sequence[Path]) -> tuple[List[dict], dict]:
    selected: Dict[str, dict] = {}
    attempts: Dict[str, int] = defaultdict(int)
    raw_lines = 0
    for path in paths:
        for record in iter_jsonl(path):
            raw_lines += 1
            key = str(record.get("segment_key", "") or "")
            if not key:
                continue
            attempts[key] += 1
            if key not in selected or _quality(record) > _quality(selected[key]):
                selected[key] = record
    records = []
    for source in selected.values():
        record = dict(source)
        abnormal_key, normal_key = source_graph_pair(record)
        record["best_abnormal_graph"] = abnormal_key
        record["best_normal_graph"] = normal_key
        records.append(record)
    records.sort(key=lambda item: str(item.get("segment_key", "")))
    return records, {
        "prediction_files": [str(path) for path in paths],
        "prediction_file_sha256": {str(path): file_sha256(path) for path in paths},
        "raw_lines": raw_lines,
        "unique_segments": len(records),
        "duplicate_attempts": sum(max(0, count - 1) for count in attempts.values()),
    }


def _rank(seed: int, group: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{group}\0{value}".encode("utf-8")).hexdigest()


def sample_video_ids(records: Sequence[Mapping[str, Any]], fraction: float, seed: int) -> tuple[Set[str], dict]:
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("video sample fraction must satisfy 0 < fraction <= 1")
    all_ids = sorted({str(record.get("video_id", "") or "") for record in records if record.get("video_id")})
    pools = {code: [video_id for video_id in all_ids if code in label_codes(video_id)] for code in CLASS_CODES}
    draws: Dict[str, List[str]] = {}
    selected_by: Dict[str, Set[str]] = defaultdict(set)
    for code, pool in pools.items():
        target = min(len(pool), max(1, int(math.floor(len(pool) * float(fraction) + 0.5)))) if pool else 0
        draw = sorted(pool, key=lambda video_id: (_rank(seed, code, video_id), video_id))[:target]
        draws[code] = sorted(draw)
        for video_id in draw:
            selected_by[video_id].add(code)
    selected = set(selected_by)
    class_stats = {}
    for code, pool in pools.items():
        union_count = sum(video_id in selected for video_id in pool)
        class_stats[code] = {
            "name": CLASS_NAMES[code],
            "source_videos_containing_class": len(pool),
            "target_independent_draw": len(draws[code]),
            "union_videos_containing_class": union_count,
        }
    manifest = {
        "version": "ot_graph_video_sample_v1",
        "policy": "independent deterministic sample per filename class, followed by a multi-label union",
        "fraction": float(fraction),
        "seed": int(seed),
        "source_anomaly_videos": len(all_ids),
        "selected_union_videos": len(selected),
        "class_stats": class_stats,
        "selected_videos": [
            {"video_id": video_id, "labels": sorted(label_codes(video_id)), "selected_by": sorted(selected_by[video_id])}
            for video_id in sorted(selected)
        ],
        "class_draws": draws,
    }
    return selected, manifest


def select_windows(
    records: Sequence[dict],
    selected_video_ids: Set[str],
    seed: int,
    max_windows_per_video: int,
    max_total_windows: int,
    allowed_graph_keys: Set[str] | None = None,
    require_source_pair: bool = True,
) -> List[dict]:
    by_video: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        video_id = str(record.get("video_id", "") or "")
        if video_id not in selected_video_ids:
            continue
        abnormal_key = str(record.get("best_abnormal_graph", "") or "")
        normal_key = str(record.get("best_normal_graph", "") or "")
        if require_source_pair:
            if abnormal_key in {"", "NONE"} or normal_key in {"", "NONE"}:
                continue
            if allowed_graph_keys is not None and (
                abnormal_key not in allowed_graph_keys or normal_key not in allowed_graph_keys
            ):
                continue
        by_video[video_id].append(record)
    chosen: List[dict] = []
    for video_id in sorted(by_video):
        windows = sorted(
            by_video[video_id],
            key=lambda record: (_rank(seed, "window", str(record.get("segment_key", ""))), str(record.get("segment_key", ""))),
        )
        if max_windows_per_video > 0:
            windows = windows[:max_windows_per_video]
        chosen.extend(windows)
    chosen.sort(key=lambda record: (_rank(seed, "global", str(record.get("segment_key", ""))), str(record.get("segment_key", ""))))
    if max_total_windows > 0:
        chosen = chosen[:max_total_windows]
    return chosen


def write_selection(
    out_dir: Path,
    records: Sequence[dict],
    manifest: dict,
    input_stats: dict,
    require_source_pair: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest.update(input_stats)
    manifest["selected_windows"] = len(records)
    sampled_ids = {str(item.get("video_id", "")) for item in manifest.get("selected_videos", [])}
    matched_ids = {str(record.get("video_id", "")) for record in records}
    manifest["videos_with_selected_windows"] = len(matched_ids)
    manifest["selection_requires_source_pair"] = bool(require_source_pair)
    manifest["sampled_videos_without_selected_window"] = sorted(sampled_ids - matched_ids)
    manifest["sampled_videos_without_valid_source_graph_pair"] = (
        sorted(sampled_ids - matched_ids) if require_source_pair else []
    )
    write_json(out_dir / "selection_summary.json", manifest)
    write_jsonl(out_dir / "window_manifest.jsonl", [
        {
            "segment_key": record.get("segment_key"),
            "video_id": record.get("video_id"),
            "video_path": record.get("video_path"),
            "start_frame": record.get("start_frame"),
            "end_frame": record.get("end_frame"),
            "source_best_abnormal_graph": record.get("best_abnormal_graph"),
            "source_best_normal_graph": record.get("best_normal_graph"),
        }
        for record in records
    ])
