#!/usr/bin/env python3
"""GT-blind lightweight motion observations for crowd Event-State V4."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from common import stable_sha1
from schemas import WindowCase


def _read_gray(capture, index: int):
    import cv2  # type: ignore

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, 360.0 / max(height, width, 1))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(8, int(width * scale)), max(8, int(height * scale))))
    return gray


def compute_crowd_motion_features(case: WindowCase, bins: int = 8) -> dict[str, Any]:
    """Compute neutral flow magnitude/change features without labels or category information."""
    import cv2  # type: ignore

    video = Path(case.video_path)
    if not video.is_file():
        return {"version": "crowd_motion_v1", "complete": False, "reason": "video_unavailable"}
    start, end = int(case.start_frame), int(case.end_frame)
    boundaries = [int(round(start + (end - start) * index / bins)) for index in range(bins + 1)]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"version": "crowd_motion_v1", "complete": False, "reason": "video_open_failed"}
    magnitudes, dispersions, opposing = [], [], []
    try:
        for index in range(bins):
            left = _read_gray(capture, boundaries[index])
            right = _read_gray(capture, max(boundaries[index], boundaries[index + 1] - 1))
            if left is None or right is None or left.shape != right.shape:
                magnitudes.append(0.0); dispersions.append(0.0); opposing.append(0.0)
                continue
            flow = cv2.calcOpticalFlowFarneback(left, right, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = flow[..., 0], flow[..., 1]
            magnitude = np.sqrt(fx * fx + fy * fy)
            angle = np.arctan2(fy, fx)
            weights = magnitude / max(float(magnitude.sum()), 1e-6)
            resultant = math.sqrt(
                float(np.sum(weights * np.cos(angle))) ** 2
                + float(np.sum(weights * np.sin(angle))) ** 2
            )
            mid = fx.shape[1] // 2
            left_dx = float(np.mean(fx[:, :mid])) if mid else 0.0
            right_dx = float(np.mean(fx[:, mid:])) if mid < fx.shape[1] else 0.0
            magnitudes.append(float(np.percentile(magnitude, 75)))
            dispersions.append(max(0.0, min(1.0, 1.0 - resultant)))
            opposing.append(float(max(0.0, -left_dx * right_dx) ** 0.5))
    finally:
        capture.release()
    acceleration = [abs(right - left) for left, right in zip(magnitudes, magnitudes[1:])]
    change_bin = int(np.argmax(acceleration)) + 1 if acceleration else 0
    value = {
        "version": "crowd_motion_v1",
        "complete": True,
        "temporal_bins": bins,
        "motion_magnitude_by_bin": magnitudes,
        "directional_dispersion_by_bin": dispersions,
        "opposing_flow_by_bin": opposing,
        "motion_acceleration_by_transition": acceleration,
        "strongest_change_bin": change_bin,
        "mean_motion_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0,
        "mean_directional_dispersion": float(np.mean(dispersions)) if dispersions else 0.0,
        "mean_opposing_flow": float(np.mean(opposing)) if opposing else 0.0,
        "gt_blind": True,
    }
    value["feature_sha1"] = stable_sha1(value, size=40)
    return value
