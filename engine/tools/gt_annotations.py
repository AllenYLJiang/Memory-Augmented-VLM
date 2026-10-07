#!/usr/bin/env python3
"""Frame-level XD-Violence ground truth used by the OT experiment.

The project-wide operational rule is ``max overlap with one merged anomaly interval >= 8``.
A second core-event label (strictly more than two thirds of the window anomalous) is retained
for structural analyses.  Main metrics and graph discovery must use the operational label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from selection import is_pure_normal_video

Interval = Tuple[int, int]

TRAINING_ANCHOR_SOURCE = "xdviolence_train_72b_selected_confident_anchor"
EXPLICIT_NORMAL_SOURCE = "xdviolence_train_explicit_label_A_known_normal"
TRUSTED_TRAINING_SOURCES = {TRAINING_ANCHOR_SOURCE, EXPLICIT_NORMAL_SOURCE}


def label_window_from_source_anchor(
    record: Mapping[str, Any],
    *,
    core_overlap_threshold: float = 2.0 / 3.0,
) -> dict:
    """Return an auditable weak label for a selected training-anchor window.

    This is intentionally strict. It must never become an implicit fallback for a
    missing benchmark annotation: callers have to select the source-anchor policy,
    and every record must carry the provenance emitted by the V5 anchor exporter.
    """
    segment_key = str(record.get("segment_key", ""))
    if str(record.get("dataset_split", "")) != "train":
        raise ValueError(f"source-anchor GT requires dataset_split=train: {segment_key}")
    source = str(record.get("source", ""))
    if source not in TRUSTED_TRAINING_SOURCES:
        raise ValueError(f"source-anchor GT has untrusted source provenance: {segment_key}")
    if "y_true" not in record:
        raise ValueError(f"source-anchor GT is missing y_true: {segment_key}")
    try:
        y_true = int(record["y_true"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source-anchor GT has invalid y_true: {segment_key}") from exc
    if y_true not in (0, 1):
        raise ValueError(f"source-anchor GT requires binary y_true: {segment_key}")

    explicit_normal = source == EXPLICIT_NORMAL_SOURCE
    if explicit_normal and (
        y_true != 0
        or not is_pure_normal_video(str(record.get("video_id", "")))
        or str(record.get("anchor_type", "")) != "pure_normal"
    ):
        raise ValueError(f"explicit label-A source failed known-normal identity checks: {segment_key}")
    expected_status = (
        "known_normal" if explicit_normal
        else ("confident_positive" if y_true else "confident_negative")
    )
    if str(record.get("anchor_status", "")) != expected_status:
        raise ValueError(f"source-anchor GT status disagrees with y_true: {segment_key}")
    try:
        confidence = float(record["anchor_confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"source-anchor GT is missing valid anchor_confidence: {segment_key}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"source-anchor GT confidence is outside [0, 1]: {segment_key}")

    start, end = int(record.get("start_frame", 0)), int(record.get("end_frame", 0))
    if end < start:
        raise ValueError(f"invalid source-anchor frame span: {start}-{end}")
    phase = str(record.get("event_phase", "") or "not_applicable")
    subset = (
        "training_explicit_label_A_normal"
        if explicit_normal
        else (
            "training_anchor_active_positive"
            if y_true
            else f"training_anchor_{phase}_negative"
        )
    )
    return {
        "y_true": y_true,
        "y_true_operational": y_true,
        # A weak window label cannot establish frame-level core occupancy.
        "y_true_core": None,
        "window_frames": end - start + 1,
        "max_contiguous_overlap_frames": None,
        "union_overlap_frames": None,
        "overlap_frames": None,
        "overlap_fraction": None,
        "min_overlap_frames": None,
        "core_overlap_threshold": float(core_overlap_threshold),
        "core_comparison": "not_applicable_without_frame_annotations",
        "temporal_subset": subset,
        "event_phase": phase,
        "boundary_position": "unknown_without_frame_annotations",
        "distance_to_event_start_frames": None,
        "distance_to_event_end_frames": None,
        "distance_to_nearest_event_start": None,
        "distance_to_nearest_event_end": None,
        "annotation_found": False,
        "label_available": True,
        "ground_truth_source": "source_record_anchor",
        "known_normal": explicit_normal,
        "known_normal_from_video_label": explicit_normal,
        "intervals": [],
        "anchor_status": expected_status,
        "anchor_confidence": confidence,
        "anchor_confidence_semantics": str(record.get("anchor_confidence_semantics", "")),
        "anchor_label_source": str(record.get("label_source", "")),
    }


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    values = sorted((int(a), int(b)) for a, b in intervals)
    merged: List[Interval] = []
    for start, end in values:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def load_annotations(path: Path) -> Dict[str, List[Interval]]:
    annotations: Dict[str, List[Interval]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) < 3 or (len(parts) - 1) % 2:
                raise ValueError(f"invalid annotation line {line_no}: {raw.rstrip()}")
            values = [int(value) for value in parts[1:]]
            intervals = []
            for index in range(0, len(values), 2):
                start, end = values[index], values[index + 1]
                if start < 0 or end < start:
                    raise ValueError(f"invalid interval on line {line_no}: {start} {end}")
                intervals.append((start, end))
            annotations[parts[0]] = merge_intervals(intervals)
    return annotations


def _overlap(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start) + 1)


def label_window(
    video_id: str,
    start_frame: int,
    end_frame: int,
    annotations: Mapping[str, Sequence[Interval]],
    min_overlap_frames: int = 8,
    core_overlap_threshold: float = 2.0 / 3.0,
    known_normal: bool = False,
) -> dict:
    start, end = int(start_frame), int(end_frame)
    if end < start:
        raise ValueError(f"invalid frame span: {start}-{end}")
    annotation_found = str(video_id) in annotations
    intervals = merge_intervals([(int(a), int(b)) for a, b in annotations.get(str(video_id), [])])
    if not annotation_found and not known_normal:
        intervals = []
    window_frames = end - start + 1
    overlaps = [_overlap(start, end, a, b) for a, b in intervals]
    max_overlap = max(overlaps, default=0)
    # Intervals are merged/non-overlapping, so the sum is the union overlap.
    union_overlap = sum(overlaps)
    overlap_fraction = union_overlap / float(window_frames)
    operational = int(max_overlap >= max(1, int(min_overlap_frames)))
    core = int(overlap_fraction > float(core_overlap_threshold))
    overlapping_intervals = [(a, b) for (a, b), overlap in zip(intervals, overlaps) if overlap > 0]
    contains_start = any(start <= a <= end for a, _ in overlapping_intervals)
    contains_end = any(start <= b <= end for _, b in overlapping_intervals)
    if contains_start and contains_end:
        boundary_position = "spans_full_event"
    elif contains_start:
        boundary_position = "onset"
    elif contains_end:
        boundary_position = "ending"
    else:
        boundary_position = "none"

    def distance_to_window(point: int) -> int:
        if start <= point <= end:
            return 0
        return min(abs(point - start), abs(point - end))

    event_starts = [a for a, _ in intervals]
    event_ends = [b for _, b in intervals]
    distance_to_event_start = min((distance_to_window(value) for value in event_starts), default=None)
    distance_to_event_end = min((distance_to_window(value) for value in event_ends), default=None)

    if known_normal and not annotation_found:
        subset = "pure_normal"
        event_phase = "pure_normal"
    elif operational == 0:
        subset = "normal"
        event_phase = "post_event" if intervals and all(b < start for _, b in intervals) else "normal"
    elif boundary_position != "none":
        subset = "core_anomaly" if core == 1 else "partial_or_boundary_anomaly"
        event_phase = {
            "onset": "onset",
            "ending": "ending",
            "spans_full_event": "spans_full_event",
        }[boundary_position]
    elif core == 1:
        subset = "core_anomaly"
        event_phase = "core"
    else:
        subset = "partial_or_boundary_anomaly"
        event_phase = {
            "onset": "onset",
            "ending": "ending",
            "spans_full_event": "spans_full_event",
        }.get(boundary_position, "partial")
    return {
        "y_true": operational,
        "y_true_operational": operational,
        "y_true_core": core,
        "window_frames": window_frames,
        "max_contiguous_overlap_frames": max_overlap,
        "union_overlap_frames": union_overlap,
        # Backward-compatible aliases.
        "overlap_frames": union_overlap,
        "overlap_fraction": overlap_fraction,
        "min_overlap_frames": int(min_overlap_frames),
        "core_overlap_threshold": float(core_overlap_threshold),
        "core_comparison": "strictly_greater_than",
        "temporal_subset": subset,
        "event_phase": event_phase,
        "boundary_position": boundary_position,
        "distance_to_event_start_frames": distance_to_event_start,
        "distance_to_event_end_frames": distance_to_event_end,
        "distance_to_nearest_event_start": distance_to_event_start,
        "distance_to_nearest_event_end": distance_to_event_end,
        "annotation_found": annotation_found,
        "known_normal": bool(known_normal),
        "known_normal_from_video_label": bool(known_normal),
        "intervals": [[a, b] for a, b in intervals],
    }
