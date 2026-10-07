#!/usr/bin/env python3
"""Audit how completely the source prediction run covers each video."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def expected_window_count(num_frames: int, window: int, stride: int) -> int:
    if num_frames <= 0:
        return 0
    if num_frames <= window:
        return 1
    return 1 + int(math.ceil((num_frames - window) / float(stride)))


def _num_frames(path: str, fallback: int) -> tuple[int, str]:
    video = Path(path)
    if video.is_file():
        try:
            import cv2  # type: ignore
            capture = cv2.VideoCapture(str(video))
            if capture.isOpened():
                value = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                if value > 0:
                    return value, "video_metadata"
            capture.release()
        except Exception:
            pass
    return max(0, int(fallback)), "observed_max_frame_fallback"


def audit_video_completion(records: Sequence[Mapping], window: int = 96, stride: int = 16) -> list[dict]:
    by_video = defaultdict(list)
    for record in records:
        video_id = str(record.get("video_id", "") or "")
        if video_id:
            by_video[video_id].append(record)
    rows = []
    for video_id, values in sorted(by_video.items()):
        observed_keys = {str(value.get("segment_key", "")) for value in values if value.get("segment_key")}
        observed_max = max((int(value.get("end_frame", -1)) for value in values), default=-1) + 1
        video_path = next((str(value.get("video_path", "")) for value in values if value.get("video_path")), "")
        num_frames, source = _num_frames(video_path, observed_max)
        expected = expected_window_count(num_frames, int(window), int(stride))
        observed = len(observed_keys)
        coverage = observed / float(expected) if expected else 0.0
        rows.append({
            "video_id": video_id,
            "video_path": video_path,
            "num_frames": num_frames,
            "num_frames_source": source,
            "expected_windows": expected,
            "observed_unique_windows": observed,
            "coverage": min(1.0, coverage),
            "complete_0p99": bool(expected > 0 and coverage >= 0.99),
        })
    return rows
