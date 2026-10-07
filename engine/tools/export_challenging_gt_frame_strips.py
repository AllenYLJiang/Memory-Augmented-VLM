#!/usr/bin/env python3
"""Select challenging videos and render GT-only nine-frame strips.

Selection is reproducible and uses only persisted results.  V4 errors and small
M3c margins are prioritized, followed by dense Part A+C error rate.  The image
contains no anomaly-score curve and no sampled-frame labels; exact sampled frame
indices are retained in JSON and CSV manifests.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from export_v4_ap_and_temporal_curves import (
    CLASS_NAMES,
    average_precision,
    competition,
    extract_frame,
    font,
    label_codes,
    load_annotations,
    load_source_predictions,
    load_v4_records,
    read_json,
    safe_name,
    windows_path,
    write_csv,
    write_json,
)


def in_interval(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def challenge_row(
    video_id: str,
    source_rows: Sequence[Mapping[str, Any]],
    v4_record: Mapping[str, Any],
) -> dict[str, Any]:
    m0 = competition(v4_record, "independent_direct_nodes")
    m3 = competition(v4_record, "conditional_ot_full")
    y_true = int(v4_record.get("y_true", 0) or 0)
    source_labels = [int(row["y_true_operational"]) for row in source_rows]
    source_scores = [float(row["score"]) for row in source_rows]
    source_predictions = [int(row["y_pred"]) for row in source_rows]
    source_errors = sum(t != p for t, p in zip(source_labels, source_predictions))
    m3_pred = int(m3.get("y_pred", 0) or 0)
    m0_pred = int(m0.get("y_pred", 0) or 0)
    margin = float(m3.get("margin", 0.0) or 0.0)
    decision = str(m3.get("decision", ""))
    return {
        "video_id": video_id,
        "label_codes": label_codes(video_id),
        "dense_windows": len(source_rows),
        "dense_error_count": source_errors,
        "dense_error_rate": source_errors / len(source_rows) if source_rows else 0.0,
        "dense_operational_ap": average_precision(source_labels, source_scores),
        "v4_segment_key": str(v4_record.get("segment_key", "")),
        "v4_start_frame": int(v4_record.get("start_frame", 0) or 0),
        "v4_end_frame": int(v4_record.get("end_frame", 0) or 0),
        "v4_y_true": y_true,
        "v4_m0_pred": m0_pred,
        "v4_m0_margin": float(m0.get("margin", 0.0) or 0.0),
        "v4_m3c_pred": m3_pred,
        "v4_m3c_margin": margin,
        "v4_m3c_decision": decision,
        "v4_m3c_wrong": int(m3_pred != y_true),
        "v4_m3c_uncertain": int(decision == "uncertain"),
        "v4_graph_helps": int(m0_pred != y_true and m3_pred == y_true),
        "v4_graph_hurts": int(m0_pred == y_true and m3_pred != y_true),
    }


def challenge_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["v4_m3c_wrong"]),
        int(row["v4_m3c_uncertain"]),
        float(row["dense_error_rate"]),
        -abs(float(row["v4_m3c_margin"])),
        -float(row["dense_operational_ap"]),
        int(row["dense_windows"]),
    )


def select_by_class(rows: Sequence[dict[str, Any]], per_class: int) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for code in CLASS_NAMES:
        pool = [row for row in rows if code in row["label_codes"]]
        pool.sort(key=challenge_sort_key, reverse=True)
        unique = [row for row in pool if row["video_id"] not in used]
        chosen = unique[:per_class]
        if len(chosen) < per_class:
            chosen.extend(row for row in pool if row not in chosen)
            chosen = chosen[:per_class]
        for row in chosen:
            used.add(row["video_id"])
        selected[code] = chosen
    return selected


def probe_video(video_path: str) -> dict[str, Any]:
    # Imported lazily to keep this module's public surface small.
    from export_v4_ap_and_temporal_curves import probe_video as _probe_video
    return _probe_video(video_path)


def assign_label_lanes(
    intervals: Sequence[tuple[int, int]],
    map_x: Any,
    draw: ImageDraw.ImageDraw,
    label_font: Any,
    left: int,
    right: int,
) -> list[tuple[str, float, int]]:
    lane_ends = [float("-inf")] * 5
    output: list[tuple[str, float, int]] = []
    for start, end in intervals:
        text = f"{start}-{end}"
        box = draw.textbbox((0, 0), text, font=label_font)
        text_width = box[2] - box[0]
        center = 0.5 * (map_x(start) + map_x(end))
        x = max(left, min(center - text_width / 2.0, right - text_width))
        lane = next((index for index, lane_end in enumerate(lane_ends) if x > lane_end + 8), 4)
        lane_ends[lane] = x + text_width
        output.append((text, x, lane))
    return output


def render_strip(
    out_path: Path,
    category: str,
    row: Mapping[str, Any],
    v4_record: Mapping[str, Any],
    intervals: Sequence[tuple[int, int]],
    frame_count: int = 9,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    video_path = str(v4_record.get("video_path", ""))
    probe = probe_video(video_path)
    fps = float(probe.get("fps", 24.0) or 24.0)
    max_gt = max((end for _, end in intervals), default=0)
    max_frame = max(int(probe.get("frame_count", 0) or 0) - 1, max_gt, int(v4_record.get("end_frame", 0) or 0), frame_count)
    sampled_frames = sorted({
        min(max_frame, max(0, int(round((index + 0.5) * (max_frame + 1) / frame_count - 0.5))))
        for index in range(frame_count)
    })
    # Very short/corrupt videos still receive exactly nine deterministic indices.
    while len(sampled_frames) < frame_count:
        sampled_frames.append(min(max_frame, sampled_frames[-1] + 1 if sampled_frames else 0))

    width, height = 1740, 500
    left, right = 90, 1650
    strip_top, strip_bottom = 115, 335
    interval_bar_top, interval_bar_bottom = 355, 378
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    display_id = video_id if len(video_id) <= 115 else video_id[:112] + "..."
    draw.text((left, 28), f"{category} | {display_id}", font=font(25, True), fill="#17202a")

    def map_x(frame: int | float) -> float:
        return left + (right - left) * max(0.0, min(float(frame) / max(max_frame, 1), 1.0))

    for start, end in intervals:
        draw.rectangle((map_x(start), strip_top, map_x(end), strip_bottom), fill="#fde2e2")
    draw.rectangle((left, strip_top, right, strip_bottom), outline="#8b96a5", width=2)

    slot_width = (right - left) / frame_count
    thumb_width = int(slot_width - 12)
    thumb_height = strip_bottom - strip_top - 18
    frame_dir = out_path.parent / (out_path.stem + "_frames")
    frame_rows = []
    for index, frame_value in enumerate(sampled_frames[:frame_count]):
        frame_path = frame_dir / f"frame_{frame_value:07d}.jpg"
        if not frame_path.is_file():
            extract_frame(video_path, frame_value, fps, frame_path)
        slot_left = int(left + index * slot_width + 6)
        x0 = slot_left
        y0 = strip_top + 9
        if frame_path.is_file():
            try:
                source = Image.open(frame_path).convert("RGB")
                source.thumbnail((thumb_width, thumb_height))
                x0 = slot_left + max(0, (thumb_width - source.width) // 2)
                y0 = strip_top + 9 + max(0, (thumb_height - source.height) // 2)
                image.paste(source, (x0, y0))
                abnormal = in_interval(frame_value, intervals)
                draw.rectangle(
                    (x0, y0, x0 + source.width, y0 + source.height),
                    outline="#c62828" if abnormal else "#596575",
                    width=5 if abnormal else 2,
                )
            except OSError:
                abnormal = in_interval(frame_value, intervals)
        else:
            abnormal = in_interval(frame_value, intervals)
        frame_rows.append({
            "slot": index + 1,
            "frame_index": frame_value,
            "is_gt_abnormal": bool(abnormal),
            "frame_file": str(frame_path),
        })

    draw.text((left, 382), "GT anomaly intervals (frame indices)", font=font(15, True), fill="#3c4653")
    draw.rectangle((left, interval_bar_top, right, interval_bar_bottom), fill="#ffffff", outline="#9aa4b2", width=1)
    for start, end in intervals:
        draw.rectangle((map_x(start), interval_bar_top, map_x(end), interval_bar_bottom), fill="#f6bcbc", outline="#c62828", width=1)
    label_font = font(13, True)
    labels = assign_label_lanes(intervals, map_x, draw, label_font, left, right)
    for text, x, lane in labels:
        y = 410 + lane * 17
        start, end = (int(value) for value in text.split("-", 1))
        center = 0.5 * (map_x(start) + map_x(end))
        draw.line((center, interval_bar_bottom, center, y - 2), fill="#b42323", width=1)
        draw.text((x, y), text, font=label_font, fill="#8f1d1d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=93)
    return {
        **dict(row),
        "category": category,
        "video_path": video_path,
        "image": str(out_path),
        "fps_for_extraction_only": fps,
        "video_frame_count": max_frame + 1,
        "gt_intervals": [list(value) for value in intervals],
        "sampled_frames": frame_rows,
        "probe": probe,
    }


def runtime_summary(run_dir: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selection = read_json(run_dir / "selection_summary.json")
    config = read_json(run_dir / "run_config.json")
    selection_time = (run_dir / "selection_summary.json").stat().st_mtime
    summary_time = (run_dir / "summary.json").stat().st_mtime
    wall_seconds = max(0.0, summary_time - selection_time)
    independent_counts = [len(record.get("independent_node_calls", {})) for record in records]
    joint_counts = [len(record.get("joint_graph_calls", {})) for record in records]
    verifier_count = sum(record.get("blind_verifier") is not None for record in records)
    repair_count = sum(bool(record.get("graph_candidates", {}).get("repaired")) for record in records)
    selector_count = len(records)
    total_calls = selector_count + sum(independent_counts) + sum(joint_counts) + verifier_count + repair_count
    inference_only_calls = selector_count + sum(independent_counts) + sum(joint_counts) + repair_count
    completed = len(records)
    workers = int(config.get("workers", 1) or 1)
    evidence_bytes = sum(path.stat().st_size for path in (run_dir / "cache" / "evidence_frames").rglob("*") if path.is_file())
    response_bytes = sum(path.stat().st_size for path in (run_dir / "cache" / "responses").rglob("*.json"))
    average_inference_calls = inference_only_calls / completed if completed else 0.0
    worker_seconds_per_video = wall_seconds * workers / completed if completed else 0.0
    scenarios = []
    for seconds_per_call in (1.0, 2.0, 5.0, 10.0):
        scenarios.append({
            "assumed_seconds_per_local_vlm_request": seconds_per_call,
            "current_code_serial_seconds_per_window": average_inference_calls * seconds_per_call,
            "batched_nodes_12_call_seconds_per_window": 12.0 * seconds_per_call,
            "fully_batched_3_call_seconds_per_window": 3.0 * seconds_per_call,
        })
    return {
        "version": "v4_inference_cost_audit_v1",
        "observed_remote_api_run": {
            "selected_windows": int(selection.get("selected_windows", 0) or 0),
            "completed_windows": completed,
            "errors": int(selection.get("selected_windows", 0) or 0) - completed,
            "workers": workers,
            "wall_seconds": wall_seconds,
            "wall_hours": wall_seconds / 3600.0,
            "wall_minutes_per_completed_video": wall_seconds / 60.0 / completed if completed else 0.0,
            "worker_equivalent_minutes_per_video": worker_seconds_per_video / 60.0,
            "effective_seconds_per_logical_vlm_call": worker_seconds_per_video / (total_calls / completed) if completed and total_calls else 0.0,
            "note": "Artifact-wall estimate; includes API stalls, retries, startup, and report generation.",
        },
        "logical_vlm_calls": {
            "candidate_selector_total": selector_count,
            "candidate_selector_repair_total": repair_count,
            "independent_node_total": sum(independent_counts),
            "independent_node_per_window_mean": sum(independent_counts) / completed if completed else 0.0,
            "independent_node_per_window_min": min(independent_counts, default=0),
            "independent_node_per_window_max": max(independent_counts, default=0),
            "conditional_graph_total": sum(joint_counts),
            "conditional_graph_per_window_mean": sum(joint_counts) / completed if completed else 0.0,
            "blind_verifier_total": verifier_count,
            "all_saved_successful_calls": total_calls,
            "all_saved_calls_per_window": total_calls / completed if completed else 0.0,
            "inference_only_calls_without_verifier": inference_only_calls,
            "inference_only_calls_per_window": average_inference_calls,
        },
        "disk_cache": {
            "evidence_frame_bytes": evidence_bytes,
            "response_json_bytes": response_bytes,
            "total_bytes": evidence_bytes + response_bytes,
        },
        "local_8b_memory_planning": {
            "bf16_weight_gb_decimal": 16.0,
            "int8_weight_gb_decimal": 8.0,
            "int4_weight_gb_decimal": 4.0,
            "bf16_batch1_total_estimate_gb": [20, 30],
            "int8_batch1_total_estimate_gb": [12, 20],
            "int4_batch1_total_estimate_gb": [8, 14],
            "caution": "Total estimates include a planning allowance for the vision tower, activations, KV cache, and framework workspace; measure the selected model and image resolution before fixing batch size.",
        },
        "local_latency_scenarios": scenarios,
        "window_scaling": {
            "window_frames": int(config.get("source_window", 96) or 96),
            "stride_frames": int(config.get("source_stride", 16) or 16),
            "formula": "W=max(1,floor((F-window_frames)/stride_frames)+1); calls_per_video approximately W*calls_per_window",
        },
    }


def build_report(
    out_dir: Path,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    runtime: Mapping[str, Any],
) -> str:
    observed = runtime["observed_remote_api_run"]
    calls = runtime["logical_vlm_calls"]
    memory = runtime["local_8b_memory_planning"]
    full_scan = runtime["representative_full_video_scan"]
    lines = [
        "# Challenging Videos and Inference-Cost Audit",
        "",
        "## Challenging-video selection",
        "",
        "Each category contributes three videos. Selection prioritizes a wrong V4 M3c decision, then an uncertain/low-margin "
        "V4 decision, then high error rate in the saved dense Part A+C windows. This is a diagnostic challenge ranking, not a "
        "new benchmark split.",
        "",
    ]
    for code, rows in selected.items():
        lines += [
            f"### {code} {CLASS_NAMES[code]}",
            "",
            "| Video | V4 GT/pred | M3c margin | Dense error rate | Dense AP | Strip |",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in rows:
            rel = Path(str(row["image"])).relative_to(out_dir).as_posix()
            lines.append(
                f"| `{row['video_id']}` | {row['v4_y_true']}/{row['v4_m3c_pred']} | "
                f"{float(row['v4_m3c_margin']):+.4f} | {100.0 * float(row['dense_error_rate']):.1f}% | "
                f"{float(row['dense_operational_ap']):.4f} | [{Path(rel).name}]({rel}) |"
            )
        lines.append("")
    lines += [
        "## Observed runtime and VLM calls",
        "",
        f"The completed remote-API run took **{observed['wall_hours']:.2f} wall hours** with "
        f"**{observed['workers']} workers** for {observed['completed_windows']} completed one-window videos. This is "
        f"**{observed['wall_minutes_per_completed_video']:.2f} wall minutes/video** at run-level throughput, or "
        f"**{observed['worker_equivalent_minutes_per_video']:.2f} worker-equivalent minutes/video**. The latter implies "
        f"about **{observed['effective_seconds_per_logical_vlm_call']:.1f} seconds/logical call**, but it includes outages, "
        "retries, queueing, startup, and report generation and is not clean model latency.",
        "",
        "| Logical request | Total | Mean/window |",
        "|---|---:|---:|",
        f"| Candidate shortlist | {calls['candidate_selector_total']} | {calls['candidate_selector_total'] / observed['completed_windows']:.3f} |",
        f"| Candidate repair | {calls['candidate_selector_repair_total']} | {calls['candidate_selector_repair_total'] / observed['completed_windows']:.3f} |",
        f"| Independent node evidence | {calls['independent_node_total']} | {calls['independent_node_per_window_mean']:.3f} |",
        f"| Conditional graph refinement | {calls['conditional_graph_total']} | {calls['conditional_graph_per_window_mean']:.3f} |",
        f"| Blind graph-vs-node verifier | {calls['blind_verifier_total']} | {calls['blind_verifier_total'] / observed['completed_windows']:.3f} |",
        f"| Total saved successful calls | {calls['all_saved_successful_calls']} | {calls['all_saved_calls_per_window']:.3f} |",
        "",
        f"With graph discovery, example verification, and leave-one-out disabled, the unchanged M3c implementation still needs "
        f"**{calls['inference_only_calls_per_window']:.3f} VLM requests/window**: one shortlist call, an average of "
        f"{calls['independent_node_per_window_mean']:.2f} unique-node calls, and ten graph calls. Disabling the blind verifier "
        f"removes only {calls['blind_verifier_total']} calls ({100.0 * calls['blind_verifier_total'] / calls['all_saved_successful_calls']:.2f}% of this run).",
        "",
        "## Is an LLM needed at inference?",
        "",
        "**No DeepSeek/teacher LLM is needed.** The graph library is already fixed and `DISCOVER_GRAPHS=0`. Also disable "
        "`RUN_BLIND_VERIFIER` and `RUN_LEAVE_ONE_OUT` when only final graph/anomaly scores are required. A VLM is still needed "
        "for visual evidence. Importantly, current M3c uses independent-node probabilities to build unary affinity and initial "
        "OT before conditional graph refinement, so those node calls cannot simply be deleted without changing the algorithm.",
        "",
        "## 8B VLM memory",
        "",
        "Parameter memory follows $M_{weights}=N_{params}b/8$. For 8 billion parameters this is about 16 GB in BF16/FP16, "
        "8 GB in INT8, or 4 GB at 4-bit. Practical batch-1 planning envelopes are approximately "
        f"**{memory['bf16_batch1_total_estimate_gb'][0]}-{memory['bf16_batch1_total_estimate_gb'][1]} GB BF16**, "
        f"**{memory['int8_batch1_total_estimate_gb'][0]}-{memory['int8_batch1_total_estimate_gb'][1]} GB INT8**, and "
        f"**{memory['int4_batch1_total_estimate_gb'][0]}-{memory['int4_batch1_total_estimate_gb'][1]} GB 4-bit**, after allowing "
        "for the vision tower, activations, KV cache, and framework workspace. An 80 GB GPU is ample for one 8B model and "
        "moderate batching, but the real peak must be measured for the exact VLM, image resolution, and output length.",
        "",
        "## Local 8B latency scenarios",
        "",
        "Let $t$ be measured seconds per local eight-frame VLM request. Current code has",
        "",
        rf"$$T_{{window}} \approx {calls['inference_only_calls_per_window']:.3f}t.$$",
        "| Assumed local request latency | Current 33.4-call code | Batch all node queries (12 calls) | Batch nodes and graphs (3 calls) |",
        "|---:|---:|---:|---:|",
    ]
    for row in runtime["local_latency_scenarios"]:
        lines.append(
            f"| {row['assumed_seconds_per_local_vlm_request']:.0f} s | "
            f"{row['current_code_serial_seconds_per_window']:.1f} s | "
            f"{row['batched_nodes_12_call_seconds_per_window']:.1f} s | "
            f"{row['fully_batched_3_call_seconds_per_window']:.1f} s |"
        )
    lines += [
        "",
        "The 12-call and 3-call columns require code changes and accuracy revalidation. The simplest high-value optimization for "
        "a local 8B model is to encode the current eight inference frames once, reuse the vision embeddings, batch all independent-node "
        "prompts, and then batch the ten selected graph prompts. Repeatedly re-encoding the same frames for every graph is wasteful.",
        "",
        "For full temporal inference, multiply by the number of windows",
        "",
        r"$$W=\max\left(1,\left\lfloor\frac{F-96}{16}\right\rfloor+1\right),\qquad "
        r"N_{calls/video}\approx W\,N_{calls/window}.$$",
        "",
        f"Across the 18 rendered challenging clips, the mean video has **{full_scan['mean_frame_count']:.1f} frames** and "
        f"**{full_scan['mean_windows_per_video']:.1f} sliding windows** (median {full_scan['median_windows_per_video']:.1f}). "
        f"The unchanged inference-only code would therefore make about **{full_scan['mean_current_calls_per_video']:.0f} "
        "VLM requests per full video**. The following is a scenario estimate, not a measured local-8B benchmark:",
        "",
        "| Local request latency | Current code/full video | 12-call batching/full video | 3-call batching/full video |",
        "|---:|---:|---:|---:|",
    ]
    for row in full_scan["latency_scenarios"]:
        lines.append(
            f"| {row['seconds_per_request']:.0f} s | {row['current_minutes_per_video']:.1f} min | "
            f"{row['batched_nodes_minutes_per_video']:.1f} min | {row['fully_batched_minutes_per_video']:.1f} min |"
        )
    lines += [
        "",
        "The completed experiment used one selected window/video, so its 17.73-minute figure must not be reported as latency "
        "for scanning every temporal window of a full video.",
        "The remote run did not record local GPU peak memory. The memory numbers above are deployment estimates; the persisted "
        f"evidence/response cache itself occupies only {runtime['disk_cache']['total_bytes'] / 1e6:.1f} MB on disk.",
        "",
        "## Files",
        "",
        "- `index.html`: all 18 GT-only strips grouped by class.",
        "- `selection_manifest.json`: challenge scores, GT intervals, and exact sampled frame indices.",
        "- `sampled_frame_indices.csv`: one row per displayed frame; indices are intentionally absent from the images.",
        "- `inference_cost_summary.json`: machine-readable timing, calls, cache, memory, and latency scenarios.",
    ]
    return "\n".join(lines) + "\n"


def build_html(selected: Mapping[str, Sequence[Mapping[str, Any]]], out_dir: Path) -> str:
    sections = []
    for code, rows in selected.items():
        figures = []
        for row in rows:
            rel = Path(str(row["image"])).relative_to(out_dir).as_posix()
            figures.append(f"""
<article>
  <h3>{html.escape(str(row['video_id']))}</h3>
  <p>V4 GT/pred {row['v4_y_true']}/{row['v4_m3c_pred']}; M3c margin {float(row['v4_m3c_margin']):+.4f}; dense error {100.0 * float(row['dense_error_rate']):.1f}%</p>
  <img src="{html.escape(rel)}" alt="Nine-frame GT strip for {html.escape(str(row['video_id']))}">
</article>""")
        sections.append(f"<section><h2>{code} {html.escape(CLASS_NAMES[code])}</h2>{''.join(figures)}</section>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Challenging XD-Violence GT frame strips</title><style>
body{{margin:0;background:#f7f8fa;color:#17202a;font-family:Segoe UI,Arial,sans-serif}}main{{max-width:1780px;margin:auto;padding:28px}}
h1{{font-size:32px}}h2{{margin-top:44px;padding-top:22px;border-top:2px solid #8d98a8}}h3{{font-size:18px;margin:0 0 4px}}
article{{padding:18px 0 28px;border-bottom:1px solid #ccd3dc}}p{{color:#4b5563}}img{{display:block;width:100%;height:auto;background:white;border:1px solid #aab3bf}}
.note{{max-width:1100px;padding:14px 18px;border-left:4px solid #c62828;background:white;line-height:1.5}}
</style></head><body><main><h1>Challenging videos: nine-frame GT strips</h1>
<p class="note">Light red background denotes a frame-level GT anomaly interval. A displayed frame has a red border only when its frame index lies inside a GT anomaly interval. Exact sampled frame indices are stored in the manifest and are intentionally not printed on the image.</p>
{''.join(sections)}</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "challenging_gt_frame_strips")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "run_config.json")
    annotations_path = args.annotations or windows_path(config["annotations"])
    annotations = load_annotations(annotations_path)
    records = load_v4_records(run_dir)
    v4_by_video = {str(record["video_id"]): record for record in records}
    source_paths = [windows_path(path) for path in config.get("prediction_files", [])]
    source_by_video, source_audit = load_source_predictions(source_paths, set(v4_by_video), annotations)

    candidates = [
        challenge_row(video_id, source_by_video.get(video_id, ()), record)
        for video_id, record in v4_by_video.items()
        if video_id in source_by_video and Path(windows_path(str(record.get("video_path", "")))).is_file()
    ]
    selected_base = select_by_class(candidates, max(2, min(3, args.per_class)))
    selected_rendered: dict[str, list[dict[str, Any]]] = {}
    sampled_rows: list[dict[str, Any]] = []
    total = sum(len(rows) for rows in selected_base.values())
    counter = 0
    for code, rows in selected_base.items():
        selected_rendered[code] = []
        for row in rows:
            counter += 1
            video_id = str(row["video_id"])
            out_path = out_dir / "strips" / code / f"{safe_name(video_id)}.png"
            rendered = render_strip(
                out_path,
                f"{code} {CLASS_NAMES[code]}",
                row,
                v4_by_video[video_id],
                annotations.get(video_id, ()),
                frame_count=9,
            )
            selected_rendered[code].append(rendered)
            for frame in rendered["sampled_frames"]:
                sampled_rows.append({
                    "category_code": code,
                    "category_name": CLASS_NAMES[code],
                    "video_id": video_id,
                    "slot": frame["slot"],
                    "frame_index": frame["frame_index"],
                    "is_gt_abnormal": int(frame["is_gt_abnormal"]),
                    "frame_file": frame["frame_file"],
                    "strip_image": rendered["image"],
                })
            print(f"[strip {counter}/{total}] {code} {video_id}", flush=True)

    runtime = runtime_summary(run_dir, records)
    rendered_flat = [row for rows in selected_rendered.values() for row in rows]
    frame_counts = [int(row["video_frame_count"]) for row in rendered_flat]
    source_window = int(runtime["window_scaling"]["window_frames"])
    source_stride = int(runtime["window_scaling"]["stride_frames"])
    window_counts = [max(1, math.floor((count - source_window) / source_stride) + 1) for count in frame_counts]
    ordered_windows = sorted(window_counts)
    mean_windows = sum(window_counts) / len(window_counts)
    calls_per_window = float(runtime["logical_vlm_calls"]["inference_only_calls_per_window"])
    runtime["representative_full_video_scan"] = {
        "n_videos": len(rendered_flat),
        "mean_frame_count": sum(frame_counts) / len(frame_counts),
        "mean_windows_per_video": mean_windows,
        "median_windows_per_video": float(ordered_windows[len(ordered_windows) // 2]),
        "mean_current_calls_per_video": mean_windows * calls_per_window,
        "latency_scenarios": [{
            "seconds_per_request": seconds,
            "current_minutes_per_video": mean_windows * calls_per_window * seconds / 60.0,
            "batched_nodes_minutes_per_video": mean_windows * 12.0 * seconds / 60.0,
            "fully_batched_minutes_per_video": mean_windows * 3.0 * seconds / 60.0,
        } for seconds in (1.0, 2.0, 5.0, 10.0)],
        "scope_note": "Derived from the 18 rendered challenging clips, not all 800 test videos.",
    }
    write_json(out_dir / "selection_manifest.json", {
        "version": "challenging_gt_frame_strips_v1",
        "selection_policy": "V4 M3c wrong, then uncertain, dense Part A+C error rate, small absolute M3c margin, low dense AP",
        "source_audit": source_audit,
        "per_class": selected_rendered,
    })
    write_csv(out_dir / "sampled_frame_indices.csv", sampled_rows)
    write_json(out_dir / "inference_cost_summary.json", runtime)
    report = build_report(out_dir, selected_rendered, runtime)
    (out_dir / "CHALLENGING_VIDEOS_AND_INFERENCE_COST.md").write_text(report, encoding="utf-8")
    (out_dir / "index.html").write_text(build_html(selected_rendered, out_dir), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "categories": len(selected_rendered),
        "videos": sum(len(rows) for rows in selected_rendered.values()),
        "sampled_frames": len(sampled_rows),
        "inference_only_vlm_calls_per_window": runtime["logical_vlm_calls"]["inference_only_calls_per_window"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
