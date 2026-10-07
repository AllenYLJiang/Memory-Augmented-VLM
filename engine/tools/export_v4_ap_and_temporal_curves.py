#!/usr/bin/env python3
"""Export V4 AP audits and honest temporal anomaly-score visualizations.

The completed V4 experiment evaluates one window per video.  Therefore it cannot
produce a dense V4 temporal curve.  This tool keeps that distinction explicit:

* AP/PR artifacts are computed from every completed V4 record.
* Dense curves use the saved Part A+C source predictions from the run config.
* The single V4 M0 and M3c margins are overlaid as isolated markers.

No model or API is called.  Representative video frames are read locally with
WSL ffmpeg when available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - environment-specific message
    raise SystemExit("Pillow is required to render temporal curves") from exc


METHODS = (
    "independent_direct_nodes",
    "shared_unary_rowmax",
    "unary_ot",
    "conditional_rowmax",
    "conditional_ot_no_coherence",
    "conditional_ot_full",
)
METHOD_LABELS = {
    "independent_direct_nodes": "M0 independent nodes",
    "shared_unary_rowmax": "M1 shared unary row-max",
    "unary_ot": "M2 unary OT",
    "conditional_rowmax": "M3a conditional row-max",
    "conditional_ot_no_coherence": "M3b conditional OT",
    "conditional_ot_full": "M3c full conditional OT",
}
CLASS_NAMES = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 3 or (len(parts) - 1) % 2:
            raise ValueError(f"invalid annotation line {line_no}: {raw}")
        values = [int(value) for value in parts[1:]]
        result[parts[0]] = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return result


def overlap_frames(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(end, b) - max(start, a) + 1) for a, b in intervals)


def label_core(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> int:
    return int(overlap_frames(start, end, intervals) / max(end - start + 1, 1) > 2.0 / 3.0)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Match the project's rank-based AP implementation exactly."""
    if not labels or sum(labels) == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if int(labels[index]) == 1:
            hits += 1
            total += hits / rank
    return total / hits


def precision_recall_rows(labels: Sequence[int], scores: Sequence[float]) -> list[dict[str, Any]]:
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    positives = max(sum(int(value) for value in labels), 1)
    tp = 0
    fp = 0
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order, 1):
        if int(labels[index]):
            tp += 1
        else:
            fp += 1
        rows.append({
            "rank": rank,
            "threshold_margin": float(scores[index]),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / positives,
            "label_at_rank": int(labels[index]),
        })
    return rows


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    tp = sum(t == 1 and p == 1 for t, p in zip(labels, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(labels, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(labels, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
    }


def windows_path(path_text: str) -> Path:
    text = str(path_text)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if match:
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return Path(text)


def wsl_path(path: Path | str) -> str:
    text = str(path).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return text


def safe_name(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:72]
    return f"{stem}_{digest}"


def label_codes(video_id: str) -> list[str]:
    tail = video_id.split("_label_", 1)[1] if "_label_" in video_id else video_id
    found = re.findall(r"(?:^|-)(B1|B2|B4|B5|B6|G)(?=-|$)", tail)
    return list(dict.fromkeys(found))


def competition(record: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return record.get("competitions", {}).get(method, {}) or {}


def load_v4_records(run_dir: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((run_dir / "records").glob("*.json"))]


def load_source_predictions(
    paths: Sequence[Path],
    keep_videos: set[str],
    annotations: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_segment: dict[str, dict[str, Any]] = {}
    raw_lines = 0
    malformed = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                raw_lines += 1
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                video_id = str(record.get("video_id", ""))
                if video_id not in keep_videos:
                    continue
                start = int(record.get("start_frame", 0) or 0)
                end = int(record.get("end_frame", start) or start)
                abnormal = float(record.get("best_abnormal_confidence", 0.0) or 0.0)
                normal = float(record.get("best_normal_confidence", 0.0) or 0.0)
                compact = {
                    "segment_key": str(record.get("segment_key", "")),
                    "video_id": video_id,
                    "video_path": str(record.get("video_path", "")),
                    "start_frame": start,
                    "end_frame": end,
                    "center_frame": 0.5 * (start + end),
                    "score": abnormal - normal,
                    "abnormal_confidence": abnormal,
                    "normal_confidence": normal,
                    "y_true_operational": int(record.get("y_true", 0) or 0),
                    "y_true_core": label_core(start, end, annotations.get(video_id, ())),
                    "y_pred": int(record.get("y_pred", 0) or 0),
                }
                by_segment[compact["segment_key"]] = compact
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in by_segment.values():
        by_video[record["video_id"]].append(record)
    for records in by_video.values():
        records.sort(key=lambda value: (value["center_frame"], value["segment_key"]))
    return dict(by_video), {
        "prediction_files": [str(path) for path in paths],
        "raw_lines_scanned": raw_lines,
        "unique_kept_segments": len(by_segment),
        "kept_videos": len(by_video),
        "malformed_lines": malformed,
    }


def parse_fraction(value: str) -> float:
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return float(value)


def probe_video(video_path: str) -> dict[str, Any]:
    command = [
        "wsl", "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
        "-of", "json", wsl_path(windows_path(video_path)),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        stream = (json.loads(result.stdout).get("streams") or [{}])[0]
        fps = parse_fraction(str(stream.get("avg_frame_rate", "0")))
        frame_count = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or 0.0)
        if frame_count <= 0 and fps > 0 and duration > 0:
            frame_count = int(round(fps * duration))
        return {"fps": fps or 24.0, "frame_count": frame_count, "duration": duration, "error": ""}
    except Exception as exc:  # pragma: no cover - depends on local ffprobe/video
        return {"fps": 24.0, "frame_count": 0, "duration": 0.0, "error": str(exc)}


def extract_frame(video_path: str, frame: int, fps: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seconds = max(float(frame) / max(fps, 0.001), 0.0)
    command = [
        "wsl", "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seconds:.6f}", "-i", wsl_path(windows_path(video_path)),
        "-frames:v", "1", "-vf", "scale=260:-2", wsl_path(out_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=90, check=True)
        return out_path.is_file() and out_path.stat().st_size > 0
    except Exception:  # pragma: no cover - depends on local ffmpeg/video
        return False


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_line_series(
    draw: ImageDraw.ImageDraw,
    points: Sequence[dict[str, Any]],
    map_x: Any,
    map_y: Any,
    color: str,
) -> None:
    if len(points) < 2:
        return
    strides = [points[i]["center_frame"] - points[i - 1]["center_frame"] for i in range(1, len(points))]
    positive = [value for value in strides if value > 0]
    gap_limit = (statistics.median(positive) * 2.5) if positive else float("inf")
    segment: list[tuple[float, float]] = []
    last_frame: float | None = None
    for value in points:
        frame = float(value["center_frame"])
        xy = (map_x(frame), map_y(float(value["score"])))
        if last_frame is not None and frame - last_frame > gap_limit:
            if len(segment) > 1:
                draw.line(segment, fill=color, width=3)
            segment = []
        segment.append(xy)
        last_frame = frame
    if len(segment) > 1:
        draw.line(segment, fill=color, width=3)
    for value in points:
        x = map_x(float(value["center_frame"]))
        y = map_y(float(value["score"]))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def render_curve(
    out_path: Path,
    video_id: str,
    source_rows: Sequence[dict[str, Any]],
    v4_record: Mapping[str, Any],
    intervals: Sequence[tuple[int, int]],
    frames_per_video: int,
) -> dict[str, Any]:
    video_path = str(v4_record.get("video_path") or source_rows[0].get("video_path") or "")
    probe = probe_video(video_path)
    fps = float(probe["fps"] or 24.0)
    max_saved = max([int(row["end_frame"]) for row in source_rows] + [int(v4_record.get("end_frame", 0) or 0)])
    max_gt = max([b for _, b in intervals], default=0)
    max_frame = max(int(probe.get("frame_count", 0) or 0) - 1, max_saved, max_gt, 1)

    width, height = 1560, 820
    left, right = 105, 1515
    strip_top, strip_bottom = 120, 320
    chart_top, chart_bottom = 405, 745
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)

    title = video_id if len(video_id) <= 110 else video_id[:107] + "..."
    draw.text((left, 28), title, font=font(25, True), fill="#18202a")
    codes = ", ".join(f"{code} {CLASS_NAMES.get(code, '')}" for code in label_codes(video_id)) or "unknown"
    draw.text(
        (left, 70),
        f"labels: {codes} | dense saved windows: {len(source_rows)} | fps: {fps:.3f}",
        font=font(17), fill="#4b5563",
    )

    def map_x(frame_value: float) -> float:
        return left + (right - left) * max(0.0, min(frame_value / max_frame, 1.0))

    def map_y(score_value: float) -> float:
        clipped = max(-1.0, min(score_value, 1.0))
        return chart_bottom - (clipped + 1.0) * 0.5 * (chart_bottom - chart_top)

    for start, end in intervals:
        x0, x1 = map_x(start), map_x(end)
        draw.rectangle((x0, strip_top, x1, strip_bottom), fill="#fde2e2")
        draw.rectangle((x0, chart_top, x1, chart_bottom), fill="#fde2e2")

    frame_values = sorted({int(round(i * max_frame / max(frames_per_video - 1, 1))) for i in range(frames_per_video)})
    frame_dir = out_path.parent / (out_path.stem + "_frames")
    thumb_width = max(90, min(132, int((right - left) / max(len(frame_values), 1)) - 8))
    for index, frame_value in enumerate(frame_values):
        frame_path = frame_dir / f"frame_{frame_value:07d}.jpg"
        if not frame_path.is_file():
            extract_frame(video_path, frame_value, fps, frame_path)
        center_x = map_x(frame_value)
        x0 = int(center_x - thumb_width / 2)
        x0 = max(left, min(x0, right - thumb_width))
        if frame_path.is_file():
            try:
                thumb = Image.open(frame_path).convert("RGB")
                thumb.thumbnail((thumb_width, strip_bottom - strip_top - 34))
                y0 = strip_top + 4 + max(0, (strip_bottom - strip_top - 34 - thumb.height) // 2)
                image.paste(thumb, (x0, y0))
                border = "#c83838" if any(a <= frame_value <= b for a, b in intervals) else "#697586"
                draw.rectangle((x0, y0, x0 + thumb.width, y0 + thumb.height), outline=border, width=3)
            except OSError:
                pass
        label = f"f{frame_value} / {frame_value / fps:.1f}s"
        draw.text((x0, strip_bottom - 25), label, font=font(12), fill="#303947")

    draw.rectangle((left, strip_top, right, strip_bottom), outline="#9aa4b2", width=2)
    draw.rectangle((left, chart_top, right, chart_bottom), outline="#697586", width=2)
    for score_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = map_y(score_value)
        draw.line((left, y, right, y), fill="#cfd5dd" if score_value else "#59636f", width=1 if score_value else 2)
        draw.text((42, y - 10), f"{score_value:+.1f}", font=font(14), fill="#4b5563")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        frame_value = int(round(max_frame * fraction))
        x = map_x(frame_value)
        draw.line((x, chart_bottom, x, chart_bottom + 7), fill="#4b5563", width=2)
        draw.text((x - 24, chart_bottom + 12), f"{frame_value / fps:.1f}s", font=font(13), fill="#4b5563")

    draw.text((left, 365), "Anomaly margin = abnormal confidence/aggregate - normal confidence/aggregate", font=font(16, True), fill="#253140")
    draw_line_series(draw, source_rows, map_x, map_y, "#2d6cdf")

    v4_start = int(v4_record.get("start_frame", 0) or 0)
    v4_end = int(v4_record.get("end_frame", v4_start) or v4_start)
    draw.rectangle((map_x(v4_start), chart_top, map_x(v4_end), chart_bottom), outline="#7c3aed", width=2)
    center = 0.5 * (v4_start + v4_end)
    marker_values = []
    for method, color in (("independent_direct_nodes", "#e08700"), ("conditional_ot_full", "#7c3aed")):
        value = competition(v4_record, method)
        margin = float(value.get("margin", 0.0) or 0.0)
        x, y = map_x(center), map_y(margin)
        draw.polygon(((x, y - 9), (x + 9, y), (x, y + 9), (x - 9, y)), fill=color, outline="#ffffff")
        marker_values.append({"method": method, "margin": margin, "y_pred": int(value.get("y_pred", 0) or 0)})

    legend_y = 780
    draw.line((left, legend_y, left + 42, legend_y), fill="#2d6cdf", width=4)
    draw.text((left + 50, legend_y - 11), "Dense saved Part A+C margin", font=font(14), fill="#253140")
    draw.polygon(((left + 340, legend_y - 9), (left + 349, legend_y), (left + 340, legend_y + 9), (left + 331, legend_y)), fill="#e08700")
    draw.text((left + 360, legend_y - 11), "V4 M0 (one window)", font=font(14), fill="#253140")
    draw.polygon(((left + 575, legend_y - 9), (left + 584, legend_y), (left + 575, legend_y + 9), (left + 566, legend_y)), fill="#7c3aed")
    draw.text((left + 595, legend_y - 11), "V4 M3c (one window)", font=font(14), fill="#253140")
    draw.rectangle((left + 835, legend_y - 9, left + 870, legend_y + 9), fill="#fde2e2", outline="#c83838")
    draw.text((left + 880, legend_y - 11), "GT anomaly interval", font=font(14), fill="#253140")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)
    return {
        "video_id": video_id,
        "video_path": video_path,
        "image": str(out_path),
        "fps": fps,
        "frame_count": max_frame + 1,
        "probe": probe,
        "dense_saved_windows": len(source_rows),
        "dense_saved_span": [int(source_rows[0]["start_frame"]), int(source_rows[-1]["end_frame"])],
        "v4_window": [v4_start, v4_end],
        "v4_markers": marker_values,
        "gt_intervals": [list(value) for value in intervals],
        "frame_values": frame_values,
    }


def choose_representatives(
    source_by_video: Mapping[str, Sequence[dict[str, Any]]],
    v4_by_video: Mapping[str, Mapping[str, Any]],
    count: int,
    min_points: int,
) -> list[tuple[str, str]]:
    candidates: list[dict[str, Any]] = []
    for video_id, source_rows in source_by_video.items():
        if len(source_rows) < min_points or video_id not in v4_by_video:
            continue
        labels = [int(row["y_true_operational"]) for row in source_rows]
        scores = [float(row["score"]) for row in source_rows]
        record = v4_by_video[video_id]
        m0 = competition(record, "independent_direct_nodes")
        m3 = competition(record, "conditional_ot_full")
        gt = int(record.get("y_true", 0) or 0)
        graph_helps = int(m0.get("y_pred", 0) or 0) != gt and int(m3.get("y_pred", 0) or 0) == gt
        both_labels = int(any(labels) and not all(labels))
        candidates.append({
            "video_id": video_id,
            "codes": label_codes(video_id),
            "points": len(source_rows),
            "both_labels": both_labels,
            "graph_helps": int(graph_helps),
            "m3_correct": int(int(m3.get("y_pred", 0) or 0) == gt),
            "ap": average_precision(labels, scores),
            "range": max(scores) - min(scores) if scores else 0.0,
        })
    selected: list[tuple[str, str]] = []
    used: set[str] = set()
    for code in CLASS_NAMES:
        pool = [value for value in candidates if code in value["codes"] and value["video_id"] not in used]
        if not pool:
            continue
        pool.sort(
            key=lambda value: (
                value["both_labels"], value["graph_helps"], value["m3_correct"],
                min(value["points"], 300), value["range"], value["ap"],
            ),
            reverse=True,
        )
        chosen = pool[0]
        selected.append((chosen["video_id"], code))
        used.add(chosen["video_id"])
        if len(selected) >= count:
            return selected
    remaining = [value for value in candidates if value["video_id"] not in used]
    remaining.sort(
        key=lambda value: (value["both_labels"], value["graph_helps"], value["points"], value["range"]),
        reverse=True,
    )
    for value in remaining:
        selected.append((value["video_id"], value["codes"][0] if value["codes"] else "other"))
        if len(selected) >= count:
            break
    return selected


def fmt(value: float) -> str:
    return f"{value:.4f}"


def build_analysis(
    run_dir: Path,
    out_dir: Path,
    selection: Mapping[str, Any],
    errors: int,
    metrics: Sequence[Mapping[str, Any]],
    core_metrics: Sequence[Mapping[str, Any]],
    source_audit: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    curve_manifest: Sequence[Mapping[str, Any]],
) -> str:
    by_method = {row["method"]: row for row in metrics}
    core_by_method = {row["method"]: row for row in core_metrics}
    m0 = by_method["independent_direct_nodes"]
    m3 = by_method["conditional_ot_full"]
    c0 = core_by_method["independent_direct_nodes"]
    c3 = core_by_method["conditional_ot_full"]
    lines = [
        "# V4 Combined-Library Evaluation and Temporal Curves",
        "",
        "## Scope and the meaning of 'whole dataset'",
        "",
        f"The source artifact contains **{selection.get('source_anomaly_videos', 0)} videos with anomalous filename codes**. "
        f"The leakage-control policy excluded **{len(selection.get('excluded_source_groups', []))} discovery-source groups**, "
        f"leaving **{selection.get('selected_windows', 0)} selected videos/windows**. The run completed "
        f"**{len(metrics) and metrics[0]['n']}/{selection.get('selected_windows', 0)}** windows and recorded **{errors} API errors**.",
        "",
        "Because `MAX_WINDOWS_PER_VIDEO=1`, every completed video contributes exactly one window. Consequently, the AP below "
        "is the AP over all completed, leakage-controlled V4 windows, not a frame-level or all-window XD-Violence AP. "
        "A dense V4 temporal AP cannot be recovered without evaluating more windows with V4.",
        "",
        "## AP and classification metrics",
        "",
        "The score ranked by AP is the polarity margin",
        "",
        "$$s(x)=A(x)-N(x),$$",
        "",
        "where $A$ and $N$ are the aggregated abnormal-graph and normal-graph scores for the method.",
        "",
        "| Method | AP (8-frame GT) | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {fmt(row['ap'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['specificity'])} | {fmt(row['f1'])} |"
        )
    lines += [
        "",
        f"Under the project's operational 8-frame rule, **M3c AP is {fmt(m3['ap'])}**. M0 AP is {fmt(m0['ap'])}; "
        f"therefore M3c changes AP by **{100.0 * (m3['ap'] - m0['ap']):+.2f} percentage points**. "
        f"M3c nevertheless raises accuracy by **{100.0 * (m3['accuracy'] - m0['accuracy']):+.2f} points**, "
        f"balanced accuracy by **{100.0 * (m3['balanced_accuracy'] - m0['balanced_accuracy']):+.2f} points**, and "
        f"precision by **{100.0 * (m3['precision'] - m0['precision']):+.2f} points**. This means the current OT decision "
        "rule is better at the fixed operating threshold but is not better at globally ranking every positive above every negative.",
        "",
        "### Strict core-event GT (> 2/3 overlap)",
        "",
        "| Method | AP | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in core_metrics:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {fmt(row['ap'])} | {fmt(row['accuracy'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['specificity'])} | {fmt(row['f1'])} |"
        )
    lines += [
        "",
        f"With strict core-event GT, **M3c AP is {fmt(c3['ap'])}** versus M0 {fmt(c0['ap'])}. M3c still improves "
        f"accuracy by **{100.0 * (c3['accuracy'] - c0['accuracy']):+.2f} points**, but its AP changes by "
        f"**{100.0 * (c3['ap'] - c0['ap']):+.2f} points**. Boundary labeling therefore materially changes the ranking result.",
        "",
        "## Paired evidence and uncertainty",
        "",
        "The saved paired comparison reports 31 M0 errors corrected by M3c and 19 M0-correct cases regressed by M3c, "
        "for a net gain of 12 windows. Of these, the blind verifier accepted 13 helps and 11 hurts. The video-clustered "
        "bootstrap estimated a +3.44-point mean accuracy change with 95% CI [-0.58, +7.54], while the AP change was "
        "-1.98 points with 95% CI [-7.33, +3.19]. Both intervals cross zero, so this 345-video run is promising at the "
        "chosen threshold but does not establish a statistically decisive AP improvement.",
        "",
        "## Temporal anomaly-score curves",
        "",
        f"The dense blue curve uses **{source_audit.get('unique_kept_segments', 0)} unique saved Part A+C windows** from "
        f"{source_audit.get('kept_videos', 0)} V4-completed videos. Its score is `best_abnormal_confidence - "
        "best_normal_confidence`. The orange and purple diamonds are the one saved V4 M0 and M3c margins. They are points, "
        "not V4 curves. Red bands and red frame borders are frame-level GT anomaly intervals.",
        "",
        f"For reference, the dense saved Part A+C subset has operational AP **{fmt(source_metrics['operational_ap'])}** "
        f"and strict core-event AP **{fmt(source_metrics['core_ap'])}**. These are diagnostic source-model numbers, not V4 numbers.",
        "",
        "| Class | Video | Dense windows | Saved span | V4 window | V4 M0 margin | V4 M3c margin | Figure |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for item in curve_manifest:
        markers = {value["method"]: value for value in item["v4_markers"]}
        image_rel = Path(item["image"]).relative_to(out_dir).as_posix()
        lines.append(
            f"| {item['representative_class']} | `{item['video_id']}` | {item['dense_saved_windows']} | "
            f"{item['dense_saved_span'][0]}-{item['dense_saved_span'][1]} | {item['v4_window'][0]}-{item['v4_window'][1]} | "
            f"{markers['independent_direct_nodes']['margin']:+.4f} | {markers['conditional_ot_full']['margin']:+.4f} | "
            f"[{Path(image_rel).name}]({image_rel}) |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `ap_summary.json`: recomputed V4 AP and coverage audit.",
        "- `ap_metrics.csv`: all six methods under both GT policies.",
        "- `pr_curves/`: rank-level precision-recall data for M0 and M3c.",
        "- `curves/`: aligned frame-strip and anomaly-margin figures.",
        "- `curve_points/`: exact dense score values behind each figure.",
        "- `index.html`: browsable figure report.",
        "",
        "## What requires a new run",
        "",
        "A true V4 temporal curve requires `MAX_WINDOWS_PER_VIDEO=0` (or a sufficiently large number) on a small selected "
        "video list, preferably with stride 16 retained. The current one-window artifacts cannot be interpolated into a V4 "
        "curve without inventing scores. The figures here deliberately preserve that limitation.",
    ]
    return "\n".join(lines) + "\n"


def build_html(out_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    cards = []
    for item in manifest:
        markers = {value["method"]: value for value in item["v4_markers"]}
        rel = Path(item["image"]).relative_to(out_dir).as_posix()
        cards.append(f"""
<section>
  <h2>{html.escape(item['representative_class'])}: {html.escape(item['video_id'])}</h2>
  <p><b>Dense saved windows:</b> {item['dense_saved_windows']} &nbsp; <b>V4 window:</b> {item['v4_window'][0]}-{item['v4_window'][1]}
     &nbsp; <b>M0 margin:</b> {markers['independent_direct_nodes']['margin']:+.4f}
     &nbsp; <b>M3c margin:</b> {markers['conditional_ot_full']['margin']:+.4f}</p>
  <img src="{html.escape(rel)}" alt="Temporal anomaly score curve for {html.escape(item['video_id'])}">
</section>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4 temporal anomaly-score curves</title>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;color:#17202a;background:#f7f8fa}}main{{max-width:1600px;margin:auto;padding:28px}}
h1{{font-size:32px;margin:0 0 8px}}h2{{font-size:20px;margin:0 0 8px}}p{{line-height:1.5;color:#45515f}}
section{{padding:24px 0;border-top:1px solid #ccd3dc}}img{{display:block;width:100%;height:auto;border:1px solid #adb6c2;background:white}}
.note{{max-width:1050px;padding:14px 18px;border-left:4px solid #7c3aed;background:#fff}}
</style></head><body><main>
<h1>V4 temporal anomaly-score curves</h1>
<p class="note">Blue is the dense saved Part A+C margin. Orange and purple are the single V4 M0 and M3c points. Red bands are frame-level GT anomaly intervals. The completed V4 run used one window per video, so it does not contain a dense V4 curve.</p>
{''.join(cards)}
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--representative-count", type=int, default=6)
    parser.add_argument("--min-dense-points", type=int, default=20)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "temporal_anomaly_curves")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = read_json(run_dir / "run_config.json")
    selection = read_json(run_dir / "selection_summary.json")
    annotations_path = args.annotations or windows_path(run_config["annotations"])
    annotations = load_annotations(annotations_path)
    records = load_v4_records(run_dir)
    v4_by_video = {str(record["video_id"]): record for record in records}

    ap_rows: list[dict[str, Any]] = []
    ap_summary: dict[str, Any] = {
        "version": "v4_ap_coverage_audit_v1",
        "run_dir": str(run_dir),
        "score": "best_abnormal_score - best_normal_score (stored competition margin)",
        "selected_windows": int(selection.get("selected_windows", 0) or 0),
        "completed_windows": len(records),
        "errors": sum(1 for line in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()) if (run_dir / "errors.jsonl").is_file() else 0,
        "source_anomaly_videos": int(selection.get("source_anomaly_videos", 0) or 0),
        "excluded_source_groups": len(selection.get("excluded_source_groups", [])),
        "max_windows_per_video": int(run_config.get("max_windows_per_video", 0) or 0),
        "methods": {},
    }
    for label_name, label_field in (("operational_8_frame", "y_true"), ("core_gt_over_two_thirds", "y_true_core")):
        labels = [int(record.get(label_field, 0) or 0) for record in records]
        for method in METHODS:
            values = [competition(record, method) for record in records]
            scores = [float(value.get("margin", 0.0) or 0.0) for value in values]
            predictions = [int(value.get("y_pred", 0) or 0) for value in values]
            row = {
                "gt_policy": label_name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "ap": average_precision(labels, scores),
                **binary_metrics(labels, predictions),
            }
            ap_rows.append(row)
            ap_summary["methods"].setdefault(method, {})[label_name] = row
            if method in ("independent_direct_nodes", "conditional_ot_full"):
                write_csv(out_dir / "pr_curves" / f"{method}_{label_name}.csv", precision_recall_rows(labels, scores))
    write_csv(out_dir / "ap_metrics.csv", ap_rows)

    source_paths = [windows_path(path) for path in run_config.get("prediction_files", [])]
    source_by_video, source_audit = load_source_predictions(source_paths, set(v4_by_video), annotations)
    all_source_rows = [row for rows in source_by_video.values() for row in rows]
    source_scores = [float(row["score"]) for row in all_source_rows]
    source_metrics = {
        "operational_ap": average_precision([int(row["y_true_operational"]) for row in all_source_rows], source_scores),
        "core_ap": average_precision([int(row["y_true_core"]) for row in all_source_rows], source_scores),
    }
    ap_summary["source_dense_diagnostic"] = {**source_audit, **source_metrics}
    write_json(out_dir / "ap_summary.json", ap_summary)

    chosen = choose_representatives(
        source_by_video, v4_by_video,
        count=max(1, args.representative_count),
        min_points=max(1, args.min_dense_points),
    )
    curve_manifest: list[dict[str, Any]] = []
    for index, (video_id, representative_class) in enumerate(chosen, 1):
        source_rows = source_by_video[video_id]
        record = v4_by_video[video_id]
        stem = safe_name(video_id)
        points_rows = [{
            "segment_key": row["segment_key"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "center_frame": row["center_frame"],
            "partac_abnormal_confidence": row["abnormal_confidence"],
            "partac_normal_confidence": row["normal_confidence"],
            "partac_anomaly_margin": row["score"],
            "y_true_operational": row["y_true_operational"],
            "y_true_core": row["y_true_core"],
        } for row in source_rows]
        write_csv(out_dir / "curve_points" / f"{stem}.csv", points_rows)
        out_path = out_dir / "curves" / f"{stem}.png"
        frames_per_video = 0 if args.no_frames else max(2, args.frames_per_video)
        item = render_curve(
            out_path, video_id, source_rows, record, annotations.get(video_id, ()),
            frames_per_video=frames_per_video,
        )
        item["representative_class"] = f"{representative_class} {CLASS_NAMES.get(representative_class, '')}".strip()
        item["dense_operational_ap"] = average_precision(
            [int(row["y_true_operational"]) for row in source_rows],
            [float(row["score"]) for row in source_rows],
        )
        item["dense_core_ap"] = average_precision(
            [int(row["y_true_core"]) for row in source_rows],
            [float(row["score"]) for row in source_rows],
        )
        curve_manifest.append(item)
        print(f"[curve {index}/{len(chosen)}] {video_id} -> {out_path}", flush=True)
    write_json(out_dir / "curves_manifest.json", curve_manifest)

    metric_rows = [row for row in ap_rows if row["gt_policy"] == "operational_8_frame"]
    core_rows = [row for row in ap_rows if row["gt_policy"] == "core_gt_over_two_thirds"]
    error_count = int(ap_summary["errors"])
    analysis = build_analysis(
        run_dir, out_dir, selection, error_count, metric_rows, core_rows,
        source_audit, source_metrics, curve_manifest,
    )
    (out_dir / "V4_AP_AND_TEMPORAL_CURVES_ANALYSIS.md").write_text(analysis, encoding="utf-8")
    (out_dir / "index.html").write_text(build_html(out_dir, curve_manifest), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "completed_v4_windows": len(records),
        "m3c_operational_ap": ap_summary["methods"]["conditional_ot_full"]["operational_8_frame"]["ap"],
        "m3c_core_ap": ap_summary["methods"]["conditional_ot_full"]["core_gt_over_two_thirds"]["ap"],
        "representative_curves": len(curve_manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
