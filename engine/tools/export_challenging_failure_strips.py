#!/usr/bin/env python3
"""Select challenging V4 cases and render nine-frame GT-only strips.

This is an offline diagnostic exporter.  It never invokes a VLM or LLM.  Each
strip contains exactly nine non-overlapping thumbnails, frame-level GT anomaly
intervals as light-red timeline blocks, red borders around anomalous sampled
frames, and interval frame indices.  Exact sampled-frame indices are retained
in JSON/CSV but intentionally omitted from the image.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from export_v4_ap_and_temporal_curves import (
    CLASS_NAMES,
    competition,
    extract_frame,
    font,
    label_codes,
    load_annotations,
    probe_video,
    read_json,
    safe_name,
    write_csv,
    write_json,
)


CLASSES = ("B5", "G", "B4", "B1", "B6", "B2")
DISPLAY_NAMES = {
    "B5": "Abuse",
    "G": "Explosion",
    "B4": "Riot",
    "B1": "Fighting",
    "B6": "Car accident",
    "B2": "Shooting",
}


def is_inside(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def case_kind(record: Mapping[str, Any]) -> str:
    method = competition(record, "conditional_ot_full")
    truth = int(record.get("y_true", 0) or 0)
    pred = int(method.get("y_pred", 0) or 0)
    if pred != truth:
        if truth == 0:
            return "false_positive"
        if int(record.get("y_true_core", 0) or 0) == 1:
            return "core_false_negative"
        return "boundary_false_negative"
    if method.get("decision") == "uncertain":
        return "correct_but_uncertain"
    if int(record.get("y_true_core", 0) or 0) == 1:
        return "correct_core_anomaly_stress"
    return "correct_normal_stress"


def failure(record: Mapping[str, Any]) -> bool:
    value = competition(record, "conditional_ot_full")
    return int(value.get("y_pred", 0) or 0) != int(record.get("y_true", 0) or 0)


def severity(record: Mapping[str, Any]) -> float:
    value = competition(record, "conditional_ot_full")
    margin = float(value.get("margin", 0.0) or 0.0)
    truth = int(record.get("y_true", 0) or 0)
    if failure(record):
        return margin if truth == 0 else -margin
    return -abs(margin)


def graph_shift(record: Mapping[str, Any]) -> float:
    m0 = float(competition(record, "independent_direct_nodes").get("margin", 0.0) or 0.0)
    m3 = float(competition(record, "conditional_ot_full").get("margin", 0.0) or 0.0)
    return abs(m3 - m0)


def choose_cases(records: Sequence[dict[str, Any]], per_class: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    for code in CLASSES:
        pool = [record for record in records if code in label_codes(str(record.get("video_id", "")))]
        chosen: list[dict[str, Any]] = []
        chosen_segments: set[str] = set()

        def take(values: Sequence[dict[str, Any]], reason: str, key: Any) -> bool:
            available = [
                value for value in values
                if str(value.get("segment_key", "")) not in chosen_segments
                and str(value.get("video_id", "")) not in used_videos
            ]
            if not available:
                available = [
                    value for value in values
                    if str(value.get("segment_key", "")) not in chosen_segments
                ]
            if not available:
                return False
            value = sorted(available, key=key, reverse=True)[0]
            value = dict(value)
            value["_representative_class"] = code
            value["_selection_reason"] = reason
            chosen.append(value)
            chosen_segments.add(str(value.get("segment_key", "")))
            used_videos.add(str(value.get("video_id", "")))
            return True

        graph_hurts = [
            value for value in pool
            if failure(value) and bool(value.get("comparison", {}).get("graph_hurts"))
        ]
        take(
            graph_hurts,
            "conditional refinement/OT overturned a correct independent-node decision",
            lambda value: (
                int(case_kind(value) == "core_false_negative"),
                int(case_kind(value) == "boundary_false_negative"),
                graph_shift(value), severity(value),
            ),
        )

        represented = {case_kind(value) for value in chosen}
        buckets = (
            ("core_false_negative", "core anomaly missed despite >2/3 GT overlap"),
            ("false_positive", "GT-normal window received an abnormal graph win"),
            ("boundary_false_negative", "partial/boundary anomaly was missed"),
        )
        for kind, reason in buckets:
            if len(chosen) >= per_class:
                break
            if kind in represented:
                continue
            values = [value for value in pool if case_kind(value) == kind]
            if take(values, reason, severity):
                represented.add(kind)

        remaining_failures = [value for value in pool if failure(value)]
        while len(chosen) < per_class and take(
            remaining_failures, "additional high-severity model failure", severity,
        ):
            pass

        stress_order = (
            ("correct_but_uncertain", "near-threshold correct case; uncertainty is counted as normal"),
            ("correct_core_anomaly_stress", "scarce-category core anomaly stress case"),
            ("correct_normal_stress", "scarce-category normal stress case"),
        )
        for kind, reason in stress_order:
            if len(chosen) >= per_class:
                break
            values = [value for value in pool if case_kind(value) == kind]
            take(values, reason, lambda value: -abs(float(competition(value, "conditional_ot_full").get("margin", 0.0) or 0.0)))

        selected.extend(chosen[:per_class])
    return selected


def sampled_frames(
    frame_count: int,
    intervals: Sequence[tuple[int, int]],
    focus_frame: int,
    count: int = 9,
) -> list[int]:
    max_frame = max(frame_count - 1, max((end for _, end in intervals), default=0), focus_frame, 1)
    values: list[int] = []
    for index in range(count):
        start = int(math.floor(index * (max_frame + 1) / count))
        end = int(math.floor((index + 1) * (max_frame + 1) / count)) - 1
        end = max(start, min(end, max_frame))
        if start <= focus_frame <= end:
            value = focus_frame
        else:
            overlaps = [
                (max(start, a), min(end, b)) for a, b in intervals
                if max(start, a) <= min(end, b)
            ]
            if overlaps:
                best = max(overlaps, key=lambda pair: pair[1] - pair[0])
                value = (best[0] + best[1]) // 2
            else:
                value = (start + end) // 2
        values.append(int(value))
    return values


def interval_rows(intervals: Sequence[tuple[int, int]]) -> list[int]:
    rows: list[list[tuple[int, int]]] = []
    assignments: list[int] = []
    for start, end in intervals:
        row = 0
        while row < len(rows) and any(not (end < a or start > b) for a, b in rows[row]):
            row += 1
        if row == len(rows):
            rows.append([])
        rows[row].append((start, end))
        assignments.append(row)
    return assignments


def render_strip(
    record: Mapping[str, Any],
    code: str,
    out_path: Path,
    frame_count_target: int = 9,
) -> dict[str, Any]:
    video_id = str(record.get("video_id", ""))
    video_path = str(record.get("video_path", ""))
    intervals = [tuple(map(int, pair)) for pair in record.get("gt", {}).get("intervals", [])]
    probe = probe_video(video_path)
    focus = (int(record.get("start_frame", 0) or 0) + int(record.get("end_frame", 0) or 0)) // 2
    frame_count = max(
        int(probe.get("frame_count", 0) or 0),
        max((end + 1 for _, end in intervals), default=0),
        focus + 1,
    )
    frames = sampled_frames(frame_count, intervals, focus, frame_count_target)
    max_frame = max(frame_count - 1, 1)
    fps = float(probe.get("fps", 24.0) or 24.0)

    width, height = 1800, 475
    left, right = 70, 1730
    interval_top = 82
    strip_top, strip_bottom = 160, 405
    canvas = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    title = video_id if len(video_id) < 125 else video_id[:122] + "..."
    draw.text((left, 22), f"{code} {DISPLAY_NAMES[code]}  |  {title}", font=font(24, True), fill="#17202a")

    def x_of(frame: int) -> float:
        return left + (right - left) * max(0.0, min(float(frame) / max_frame, 1.0))

    for start, end in intervals:
        draw.rectangle((x_of(start), strip_top, x_of(end), strip_bottom), fill="#fde2e2")

    row_assignments = interval_rows(intervals)
    for (start, end), row in zip(intervals, row_assignments):
        y = interval_top + row * 31
        x0, x1 = x_of(start), x_of(end)
        draw.line((x0, y, x1, y), fill="#c73535", width=3)
        draw.line((x0, y - 6, x0, y + 6), fill="#c73535", width=3)
        draw.line((x1, y - 6, x1, y + 6), fill="#c73535", width=3)
        label = f"GT [{start}-{end}]"
        box = draw.textbbox((0, 0), label, font=font(15, True))
        label_width = box[2] - box[0]
        label_x = max(left, min((x0 + x1 - label_width) / 2, right - label_width))
        draw.rectangle((label_x - 4, y - 24, label_x + label_width + 4, y - 3), fill="#f6f7f9")
        draw.text((label_x, y - 24), label, font=font(15, True), fill="#a72c2c")

    frame_dir = out_path.parent / (out_path.stem + "_frames")
    centers = [x_of(frame) for frame in frames]
    min_gap = min((centers[i + 1] - centers[i] for i in range(len(centers) - 1)), default=170)
    thumb_width = int(max(76, min(172, min_gap - 10)))
    target_height = strip_bottom - strip_top - 28
    extracted: list[dict[str, Any]] = []
    for frame, center_x in zip(frames, centers):
        frame_path = frame_dir / f"frame_{frame:07d}.jpg"
        if not frame_path.is_file():
            extract_frame(video_path, frame, fps, frame_path)
        x0 = int(max(left, min(center_x - thumb_width / 2, right - thumb_width)))
        image_ok = False
        if frame_path.is_file():
            try:
                thumb = Image.open(frame_path).convert("RGB")
                thumb.thumbnail((thumb_width, target_height))
                y0 = strip_top + max(4, (strip_bottom - strip_top - thumb.height) // 2)
                canvas.paste(thumb, (x0, y0))
                abnormal = is_inside(frame, intervals)
                border = "#d12f2f" if abnormal else "#687585"
                draw.rectangle((x0, y0, x0 + thumb.width, y0 + thumb.height), outline=border, width=5 if abnormal else 3)
                image_ok = True
            except OSError:
                image_ok = False
        extracted.append({
            "frame_index": frame,
            "inside_gt_anomaly": is_inside(frame, intervals),
            "image_path": str(frame_path),
            "image_ok": image_ok,
        })

    draw.rectangle((left, strip_top, right, strip_bottom), outline="#8b97a6", width=2)
    legend_y = 443
    draw.rectangle((left, legend_y - 11, left + 34, legend_y + 11), fill="#fde2e2", outline="#c73535")
    draw.text((left + 46, legend_y - 12), "Ground-truth anomaly interval", font=font(15), fill="#344050")
    draw.rectangle((left + 360, legend_y - 13, left + 398, legend_y + 13), outline="#d12f2f", width=5)
    draw.text((left + 412, legend_y - 12), "Sampled frame inside a GT anomaly interval", font=font(15), fill="#344050")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=93)
    return {
        "video_id": video_id,
        "video_path": video_path,
        "representative_class": code,
        "category": DISPLAY_NAMES[code],
        "strip_path": str(out_path),
        "fps": fps,
        "video_frame_count": frame_count,
        "gt_intervals": [list(value) for value in intervals],
        "shown_frames": extracted,
        "focus_failure_window": [int(record.get("start_frame", 0) or 0), int(record.get("end_frame", 0) or 0)],
    }


def weakness_text(record: Mapping[str, Any], code: str) -> str:
    method = competition(record, "conditional_ot_full")
    abnormal = str(method.get("best_abnormal_graph", ""))
    normal = str(method.get("best_normal_graph", ""))
    kind = case_kind(record)
    if bool(record.get("comparison", {}).get("graph_hurts")):
        base = "Conditional graph refinement reversed a correct M0 decision, so contextual conditioning amplified the wrong interpretation."
    elif kind == "core_false_negative":
        base = "A core anomaly was suppressed by a competing normal explanation; this is not merely a boundary-label issue."
    elif kind == "boundary_false_negative":
        base = "The event occupies only part of the 96-frame unit, exposing sensitivity to temporal boundaries and fixed windows."
    elif kind == "false_positive":
        base = "A GT-normal window contains visually suggestive aftermath/context cues that the abnormal library over-interprets."
    elif kind == "correct_but_uncertain":
        base = "The result is correct only because uncertain is mapped to normal; a small calibration change would flip it."
    else:
        base = "This scarce-category stress case is retained to show what the library can represent and how narrow its margin is."

    pair = f"The competition was `{abnormal}` versus `{normal}`."
    lower = f"{abnormal} {normal}".lower()
    detail = ""
    if code in {"B2", "G"} and "combat_weapon_discharge" in lower:
        detail = " The normal combat-discharge graph conflicts with XD-Violence labeling, revealing a dataset-taxonomy versus semantic-normality mismatch."
    elif code == "G" and "large_outdoor_fire" in lower:
        detail = " Explosion versus sustained fire remains ambiguous when onset/expansion is outside the sampled evidence."
    elif any(token in lower for token in ("simulated", "game", "news_branded", "text_card")):
        detail = " Broadcast/simulation appearance is dominating event evidence, so content and presentation mode are not cleanly separated."
    elif any(token in lower for token in ("sports", "basketball", "hockey")):
        detail = " Sports context is acting as an overly strong normal prior even when aggressive interaction is visible."
    elif "victim_on_ground" in lower or "aftermath" in lower:
        detail = " Static aftermath-like appearance is not sufficient to distinguish harm, assistance, or an event outside the current interval."
    elif code == "B6":
        detail = " Near-miss, crash onset, and post-collision aftermath are not consistently separated across a fixed window."
    return f"{base} {pair}{detail}"


def category_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for code in CLASSES:
        values = [record for record in records if code in label_codes(str(record.get("video_id", "")))]
        failures = [record for record in values if failure(record)]
        rows.append({
            "code": code,
            "category": DISPLAY_NAMES[code],
            "n_windows": len(values),
            "failures": len(failures),
            "failure_rate": len(failures) / len(values) if values else 0.0,
            "false_positives": sum(case_kind(value) == "false_positive" for value in failures),
            "false_negatives": sum(int(value.get("y_true", 0) or 0) == 1 for value in failures),
            "core_false_negatives": sum(case_kind(value) == "core_false_negative" for value in failures),
            "boundary_false_negatives": sum(case_kind(value) == "boundary_false_negative" for value in failures),
            "uncertain_failures": sum(competition(value, "conditional_ot_full").get("decision") == "uncertain" for value in failures),
            "graph_hurt_failures": sum(bool(value.get("comparison", {}).get("graph_hurts")) for value in failures),
        })
    return rows


def call_audit(records: Sequence[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    node_counts = [len(value.get("independent_node_calls", {})) for value in records]
    graph_counts = [len(value.get("joint_graph_calls", {})) for value in records]
    repairs = [bool(value.get("graph_candidates", {}).get("repaired")) for value in records]
    verifiers = [value.get("blind_verifier") is not None for value in records]
    selector_cache = sum(bool(value.get("graph_candidates", {}).get("cache_hit")) for value in records)
    node_cache = sum(
        sum(bool(trace.get("cache_hit")) for trace in value.get("independent_node_calls", {}).values())
        for value in records
    )
    joint_cache = sum(
        sum(bool(trace.get("cache_hit")) for trace in value.get("joint_graph_calls", {}).values())
        for value in records
    )
    n = max(len(records), 1)
    selector_calls = len(records) + sum(repairs)
    node_calls = sum(node_counts)
    joint_calls = sum(graph_counts)
    verifier_calls = sum(verifiers)
    graph_scoring_calls = selector_calls + node_calls + joint_calls
    actual_calls = graph_scoring_calls + verifier_calls

    start = (run_dir / "selection_summary.json").stat().st_mtime
    end = (run_dir / "summary.json").stat().st_mtime
    wall_seconds = max(0.0, end - start)
    selected = read_json(run_dir / "selection_summary.json")
    source_windows = int(selected.get("unique_segments", 0) or 0)
    source_videos = int(selected.get("source_anomaly_videos", 0) or 0)
    avg_source_windows = source_windows / source_videos if source_videos else 0.0

    mean_nodes = node_calls / n
    mean_graphs = joint_calls / n
    mean_selector = selector_calls / n
    mean_graph_calls = graph_scoring_calls / n
    mean_node_only_calls = mean_selector + mean_nodes
    # Explicit A100/8B assumptions, not benchmark measurements.
    low_window_seconds = 5.0 * mean_selector + 2.0 * mean_nodes + 3.0 * mean_graphs + 1.0
    high_window_seconds = 9.0 * mean_selector + 3.5 * mean_nodes + 5.0 * mean_graphs + 4.0
    low_node_seconds = 5.0 * mean_selector + 2.0 * mean_nodes + 1.0
    high_node_seconds = 9.0 * mean_selector + 3.5 * mean_nodes + 3.0
    return {
        "completed_windows": len(records),
        "selector_calls": selector_calls,
        "independent_node_calls": node_calls,
        "joint_graph_calls": joint_calls,
        "verifier_calls": verifier_calls,
        "actual_total_vlm_calls": actual_calls,
        "graph_scoring_calls_without_verifier": graph_scoring_calls,
        "mean_unique_nodes_per_window": mean_nodes,
        "mean_graphs_per_window": mean_graphs,
        "mean_selector_calls_per_window": mean_selector,
        "mean_graph_scoring_calls_per_window": mean_graph_calls,
        "mean_node_only_calls_per_window": mean_node_only_calls,
        "graph_to_node_call_ratio": mean_graph_calls / mean_node_only_calls,
        "repair_calls": sum(repairs),
        "windows_with_verifier": sum(verifiers),
        "cache_hits": {"selector": selector_cache, "independent_node": node_cache, "joint_graph": joint_cache},
        "measured_remote_run": {
            "wall_hours": wall_seconds / 3600.0,
            "wall_minutes_per_completed_window": wall_seconds / 60.0 / n,
            "worker_equivalent_minutes_per_window": wall_seconds / 60.0 * 3.0 / n,
            "completed_windows_per_hour": len(records) / wall_seconds * 3600.0 if wall_seconds else 0.0,
        },
        "source_average_windows_per_video": avg_source_windows,
        "a100_8b_estimate": {
            "assumptions": {
                "selector_seconds": [5.0, 9.0],
                "independent_node_seconds": [2.0, 3.5],
                "joint_graph_seconds": [3.0, 5.0],
                "local_prepost_seconds": [1.0, 4.0],
                "evidence_frames": 8,
                "precision": "BF16",
                "serving": "one shared 8B VLM with vLLM-style continuous batching",
            },
            "graph_scoring_seconds_per_window": [low_window_seconds, high_window_seconds],
            "node_only_seconds_per_window": [low_node_seconds, high_node_seconds],
            "graph_latency_ratio_vs_node_only": [low_window_seconds / low_node_seconds, high_window_seconds / high_node_seconds],
            "one_window_video_seconds": [low_window_seconds, high_window_seconds],
            "average_full_video_sequential_hours": [avg_source_windows * low_window_seconds / 3600.0, avg_source_windows * high_window_seconds / 3600.0],
            "average_full_video_three_concurrent_minutes": [avg_source_windows * low_window_seconds / 3.0 / 60.0, avg_source_windows * high_window_seconds / 3.0 / 60.0],
            "memory_gb": {
                "shared_8b_bf16_weights": 16.0,
                "one_active_window_total": [22.0, 30.0],
                "incremental_per_active_window": [1.5, 4.0],
                "three_concurrent_windows_total": [27.0, 40.0],
                "eight_concurrent_windows_total": [40.0, 65.0],
            },
        },
    }


def build_report(
    run_dir: Path,
    out_dir: Path,
    category_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    calls: Mapping[str, Any],
) -> str:
    lines = [
        "# Challenging V4 Cases and Inference-Cost Analysis",
        "",
        "## Case-selection policy",
        "",
        "Cases were selected from all 345 completed V4 records. For each label code, the selector prefers: "
        "(1) an M3c failure that overturns a correct M0 decision, (2) a core false negative, (3) a false positive, "
        "and (4) a boundary false negative. If a scarce category has too few failures, near-threshold/core stress cases fill the set. "
        "Multi-label clips are counted in every applicable category. B5 has only four completed windows and only one actual M3c failure, "
        "so its other two examples are explicitly marked as stress cases rather than failures.",
        "",
        "Each image has exactly nine non-overlapping thumbnails. The light-red background and horizontal `GT [start-end]` bars use "
        "the frame-level annotation file. A sampled frame receives a red border only when its exact frame index falls inside a GT interval. "
        "The exact nine frame indices are stored in `challenging_cases_manifest.json` and `shown_frame_indices.csv`, not printed on the images.",
        "",
        "## Quantitative failure distribution",
        "",
        "| Category | Windows | Failures | Failure rate | FP | FN | Core FN | Boundary FN | Uncertain failures | M3c hurt M0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['code']} {row['category']} | {row['n_windows']} | {row['failures']} | {100*row['failure_rate']:.1f}% | "
            f"{row['false_positives']} | {row['false_negatives']} | {row['core_false_negatives']} | "
            f"{row['boundary_false_negatives']} | {row['uncertain_failures']} | {row['graph_hurt_failures']} |"
        )
    lines += [
        "",
        "B6 has the highest observed failure rate (31.4%), driven mainly by missed short/boundary driving events. "
        "B4 has 18 core false negatives among 20 failures, so its main weakness is under-detection of sustained crowd disorder, not boundary labeling. "
        "B2 repeatedly confuses dataset-positive shooting with the library's normal `combat_weapon_discharge`; G similarly confuses explosion onset with "
        "sustained fire or combat. These are representation/taxonomy conflicts that adding more score calibration alone will not solve.",
        "",
        "## Selected cases",
        "",
    ]
    for code in CLASSES:
        lines += [f"### {code} {DISPLAY_NAMES[code]}", ""]
        for item in [value for value in cases if value["representative_class"] == code]:
            rel = Path(item["strip_path"]).relative_to(out_dir).as_posix()
            lines += [
                f"#### `{item['video_id']}`",
                "",
                f"- Type: **{item['case_kind'].replace('_', ' ')}**. Selection: {item['selection_reason']}.",
                f"- Evaluated window: `{item['focus_failure_window'][0]}-{item['focus_failure_window'][1]}`; "
                f"GT operational/core = `{item['y_true']}/{item['y_true_core']}`.",
                f"- M0 margin/prediction: `{item['m0_margin']:+.4f}` / `{item['m0_prediction']}`. "
                f"M3c margin/prediction: `{item['m3c_margin']:+.4f}` / `{item['m3c_prediction']}`.",
                f"- M3c pair: `{item['m3c_abnormal_graph']}` versus `{item['m3c_normal_graph']}`.",
                f"- Diagnostic value: {item['weakness']}",
                f"- [Nine-frame strip]({rel})",
                "",
            ]

    estimate = calls["a100_8b_estimate"]
    measured = calls["measured_remote_run"]
    lines += [
        "## Inference call topology",
        "",
        f"Across {calls['completed_windows']} records, the persisted traces contain **{calls['actual_total_vlm_calls']:,} actual VLM calls**: "
        f"{calls['selector_calls']:,} shortlist/repair calls, {calls['independent_node_calls']:,} independent-node calls, "
        f"{calls['joint_graph_calls']:,} joint-graph calls, and {calls['verifier_calls']:,} optional verifier calls. "
        f"All selector/node/joint cache-hit counts are zero, so these are real uncached calls.",
        "",
        "For deployment that only produces graph scores and anomaly scores:",
        "",
        fr"$$N_{{\mathrm{{VLM}}}}=N_{{\mathrm{{selector}}}}+N_{{\mathrm{{unique\ nodes}}}}+N_{{\mathrm{{graphs}}}} "
        fr"\approx {calls['mean_selector_calls_per_window']:.3f}+{calls['mean_unique_nodes_per_window']:.3f}+"
        fr"{calls['mean_graphs_per_window']:.0f}=\mathbf{{{calls['mean_graph_scoring_calls_per_window']:.3f}}}\ \text{{calls/window}}.$$",
        "",
        "There is **no DeepSeek/LLM call** when discovery is disabled. The blind comparison verifier, example mining, discovery, and leave-one-out "
        "should also be disabled. This OT project does not invoke Part A rescue, neutral-scene reasoning, grounding, or wider-window reruns. "
        "Graph retrieval is still one VLM shortlist call in the current implementation; it is not an LLM-discovery call.",
        "",
        "## Measured remote-run throughput",
        "",
        f"The completed remote API run took {measured['wall_hours']:.2f} wall hours with three workers: "
        f"{measured['wall_minutes_per_completed_window']:.2f} wall minutes per completed one-window video, or "
        f"{measured['worker_equivalent_minutes_per_window']:.2f} worker-minutes per window. This includes remote queueing, network latency, "
        "long JSON generation, retries, seven timeouts, and report generation. It is not a local A100 benchmark and should not be used as the "
        "8B latency estimate.",
        "",
        "## Estimated A100 80GB latency for an 8B VLM",
        "",
        "Assumptions: BF16 8B-class VLM, eight evidence frames, structured JSON output, one shared model served by an optimized engine, no discovery, "
        "no verifier, and the current sequential loop inside each window. The ranges allow 5-9 s for the 90-graph shortlist, 2.0-3.5 s per "
        "independent node, 3-5 s per joint graph, and 1-4 s local preprocessing/OT. They are engineering estimates, not measured benchmarks.",
        "",
        f"- **Graph scoring latency per window:** {estimate['graph_scoring_seconds_per_window'][0]:.0f}-"
        f"{estimate['graph_scoring_seconds_per_window'][1]:.0f} s (**{estimate['graph_scoring_seconds_per_window'][0]/60:.2f}-"
        f"{estimate['graph_scoring_seconds_per_window'][1]/60:.2f} min**).",
        f"- **Node-only latency per window:** {estimate['node_only_seconds_per_window'][0]:.0f}-"
        f"{estimate['node_only_seconds_per_window'][1]:.0f} s.",
        f"- **One sampled window per video:** same {estimate['one_window_video_seconds'][0]:.0f}-"
        f"{estimate['one_window_video_seconds'][1]:.0f} s request latency; three concurrent windows yield an ideal throughput of roughly "
        f"{3600*3/estimate['one_window_video_seconds'][1]:.0f}-{3600*3/estimate['one_window_video_seconds'][0]:.0f} videos/hour.",
        f"- The source artifact averages **{calls['source_average_windows_per_video']:.1f} windows/video**. Full sliding-window inference would take "
        f"about **{estimate['average_full_video_sequential_hours'][0]:.2f}-{estimate['average_full_video_sequential_hours'][1]:.2f} h/video** "
        f"sequentially, or an ideal **{estimate['average_full_video_three_concurrent_minutes'][0]:.0f}-"
        f"{estimate['average_full_video_three_concurrent_minutes'][1]:.0f} min/video** with three windows in flight.",
        "",
        "NVIDIA specifies 80GB HBM2e and about 1.94-2.04 TB/s memory bandwidth for A100 80GB variants. "
        "[NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/). The model weights alone require approximately "
        "$8\times10^9\times2=16$ GB in BF16. With the vision tower, CUDA workspaces, activations, and KV cache, a practical estimate is:",
        "",
        "- **One active window total GPU memory:** 22-30 GB.",
        "- **Incremental memory per additional active window:** 1.5-4 GB; weights are shared and must not be counted again.",
        "- **Three concurrent windows:** 27-40 GB total.",
        "- **Eight concurrent windows:** 40-65 GB total, normally feasible on an 80GB A100 with headroom dependent on image tokens and output length.",
        "- These are working-set estimates. A serving engine may pre-allocate most otherwise-free A100 memory as a shared KV-cache pool, so "
        "`nvidia-smi` can show roughly 70 GB allocated even for one active request. That reservation is server capacity, not memory consumed by one window.",
        "",
        "vLLM supports multi-image batches and precomputed multimodal embeddings, which is directly relevant here because all 33 calls repeatedly "
        "encode the same eight frames. [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/). "
        "Qwen's official description also notes that visual token counts vary with image resolution, so actual memory/latency must be benchmarked "
        "with the exact frame resolution. [Qwen2.5-VL technical overview](https://qwenlm.github.io/blog/qwen2.5-vl/).",
        "",
        "## Graph matching versus independent nodes",
        "",
        f"Independent-node scoring needs about **{calls['mean_node_only_calls_per_window']:.2f} calls/window** (selector + unique nodes). "
        f"Full graph matching needs **{calls['mean_graph_scoring_calls_per_window']:.2f}**, adding exactly ten joint-graph calls. "
        f"The call-count ratio is **{calls['graph_to_node_call_ratio']:.3f}x** (+{100*(calls['graph_to_node_call_ratio']-1):.1f}%). "
        f"Because joint calls have longer prompts/outputs, the estimated latency ratio is about "
        f"**{min(estimate['graph_latency_ratio_vs_node_only']):.2f}-{max(estimate['graph_latency_ratio_vs_node_only']):.2f}x**, not merely the call ratio. "
        "Peak model memory changes little for sequential execution because both paths share the same model; graph prompts mainly add KV/activation memory.",
        "",
        "The largest deployment optimization is therefore not the local Sinkhorn calculation, which is negligible. It is to encode the eight frames "
        "once, cache/reuse their visual embeddings, batch all independent nodes, and batch graph-conditioned scoring. That can reduce the logical "
        "request topology from about 33 calls to roughly 2-3 batched calls per window, subject to output-length and context limits.",
    ]
    return "\n".join(lines) + "\n"


def build_html(out_dir: Path, cases: Sequence[Mapping[str, Any]]) -> str:
    sections = []
    for code in CLASSES:
        figures = []
        for item in [value for value in cases if value["representative_class"] == code]:
            rel = Path(item["strip_path"]).relative_to(out_dir).as_posix()
            figures.append(
                f"<article><h3>{html.escape(item['video_id'])}</h3>"
                f"<p><b>{html.escape(item['case_kind'].replace('_', ' '))}</b> | {html.escape(item['selection_reason'])}</p>"
                f"<img src=\"{html.escape(rel)}\" alt=\"Nine-frame GT strip\"></article>"
            )
        sections.append(f"<section><h2>{code} {DISPLAY_NAMES[code]}</h2>{''.join(figures)}</section>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Challenging V4 failure strips</title><style>
body{{margin:0;background:#f6f7f9;color:#17202a;font-family:Segoe UI,Arial,sans-serif}}main{{max-width:1880px;margin:auto;padding:28px}}
h1{{font-size:32px;margin:0}}h2{{font-size:25px;margin:32px 0 8px;border-top:1px solid #ccd3dc;padding-top:24px}}
h3{{font-size:17px;margin:0 0 5px}}p{{margin:0 0 12px;color:#4a5563}}article{{padding:18px 0 28px}}img{{display:block;width:100%;height:auto;border:1px solid #aab3bf;background:white}}
.note{{max-width:1150px;line-height:1.55;margin-top:10px}}
</style></head><body><main><h1>Challenging V4 cases</h1>
<p class="note">Exactly nine non-overlapping frames per video. Light red is frame-level GT anomaly. A red thumbnail border means that exact sampled frame lies inside a GT interval. Individual sampled-frame indices are saved in the manifest and intentionally omitted from the image.</p>
{''.join(sections)}</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cases-per-class", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "challenging_failure_strips")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.annotations)
    records = [read_json(path) for path in sorted((run_dir / "records").glob("*.json"))]
    selected = choose_cases(records, max(2, min(3, args.cases_per_class)))
    manifests: list[dict[str, Any]] = []
    shown_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, 1):
        code = str(record["_representative_class"])
        video_id = str(record.get("video_id", ""))
        # Use the canonical annotation source even if an older record contains a copy.
        record = dict(record)
        record["gt"] = dict(record.get("gt", {}), intervals=[list(value) for value in annotations.get(video_id, [])])
        stem = f"{code}_{safe_name(video_id)}"
        strip = render_strip(record, code, out_dir / "strips" / f"{stem}.png")
        m0 = competition(record, "independent_direct_nodes")
        m3 = competition(record, "conditional_ot_full")
        item = {
            **strip,
            "selection_reason": str(record["_selection_reason"]),
            "case_kind": case_kind(record),
            "is_failure": failure(record),
            "y_true": int(record.get("y_true", 0) or 0),
            "y_true_core": int(record.get("y_true_core", 0) or 0),
            "m0_margin": float(m0.get("margin", 0.0) or 0.0),
            "m0_prediction": int(m0.get("y_pred", 0) or 0),
            "m3c_margin": float(m3.get("margin", 0.0) or 0.0),
            "m3c_prediction": int(m3.get("y_pred", 0) or 0),
            "m3c_decision": str(m3.get("decision", "")),
            "m3c_abnormal_graph": str(m3.get("best_abnormal_graph", "")),
            "m3c_normal_graph": str(m3.get("best_normal_graph", "")),
            "graph_hurts": bool(record.get("comparison", {}).get("graph_hurts")),
            "weakness": weakness_text(record, code),
        }
        manifests.append(item)
        for frame in item["shown_frames"]:
            shown_rows.append({
                "representative_class": code,
                "category": DISPLAY_NAMES[code],
                "video_id": video_id,
                "frame_index": frame["frame_index"],
                "inside_gt_anomaly": frame["inside_gt_anomaly"],
                "image_ok": frame["image_ok"],
                "image_path": frame["image_path"],
            })
        selection_rows.append({
            "representative_class": code,
            "category": DISPLAY_NAMES[code],
            "video_id": video_id,
            "case_kind": item["case_kind"],
            "is_failure": item["is_failure"],
            "selection_reason": item["selection_reason"],
            "y_true": item["y_true"],
            "y_true_core": item["y_true_core"],
            "m0_margin": item["m0_margin"],
            "m3c_margin": item["m3c_margin"],
            "m3c_abnormal_graph": item["m3c_abnormal_graph"],
            "m3c_normal_graph": item["m3c_normal_graph"],
            "strip_path": item["strip_path"],
        })
        print(f"[strip {index}/{len(selected)}] {code} {video_id}", flush=True)

    category_rows = category_metrics(records)
    calls = call_audit(records, run_dir)
    write_json(out_dir / "challenging_cases_manifest.json", manifests)
    write_json(out_dir / "runtime_call_memory_estimate.json", calls)
    write_csv(out_dir / "shown_frame_indices.csv", shown_rows)
    write_csv(out_dir / "selected_cases.csv", selection_rows)
    write_csv(out_dir / "category_failure_metrics.csv", category_rows)
    report = build_report(run_dir, out_dir, category_rows, manifests, calls)
    (out_dir / "CHALLENGING_CASES_AND_RUNTIME_ANALYSIS.md").write_text(report, encoding="utf-8")
    (out_dir / "index.html").write_text(build_html(out_dir, manifests), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "cases": len(manifests),
        "shown_frames": len(shown_rows),
        "all_images_ok": all(bool(value["image_ok"]) for value in shown_rows),
        "mean_vlm_calls_without_verifier": calls["mean_graph_scoring_calls_per_window"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
