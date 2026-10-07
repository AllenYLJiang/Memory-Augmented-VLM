#!/usr/bin/env python3
"""Export challenging per-class GT frame strips and an inference cost audit.

This tool is offline. It reads completed V4 records and local videos; it does
not call a VLM or an LLM.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from export_v4_ap_and_temporal_curves import (
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


CLASS_ORDER = ("B5", "G", "B4", "B1", "B6", "B2")
CLASS_DISPLAY = {
    "B5": "Abuse",
    "G": "Explosion",
    "B4": "Riot",
    "B1": "Fighting",
    "B6": "Car accident",
    "B2": "Shooting",
}


def challenge_info(record: Mapping[str, Any]) -> dict[str, Any]:
    gt = int(record.get("y_true", 0) or 0)
    m0 = competition(record, "independent_direct_nodes")
    m3 = competition(record, "conditional_ot_full")
    p0 = int(m0.get("y_pred", 0) or 0)
    p3 = int(m3.get("y_pred", 0) or 0)
    margin0 = float(m0.get("margin", 0.0) or 0.0)
    margin3 = float(m3.get("margin", 0.0) or 0.0)
    comparison = record.get("comparison", {}) or {}
    if p0 != gt and p3 == gt:
        kind = "graph_help"
    elif p0 == gt and p3 != gt:
        kind = "graph_hurt"
    elif p3 != gt:
        kind = "m3c_failure"
    else:
        kind = "low_margin_correct"
    direction = 1.0 if gt else -1.0
    return {
        "kind": kind,
        "gt": gt,
        "gt_core": int(record.get("y_true_core", 0) or 0),
        "m0_pred": p0,
        "m3c_pred": p3,
        "m0_margin": margin0,
        "m3c_margin": margin3,
        "directed_margin_gain": direction * (margin3 - margin0),
        "verified_help": bool(comparison.get("verified_graph_help")),
        "verified_hurt": bool(comparison.get("verified_graph_hurt")),
    }


def rank_for_target(value: Mapping[str, Any], target: str) -> tuple[Any, ...]:
    info = value["challenge"]
    if target == "graph_help":
        return (
            info["kind"] == "graph_help",
            info["verified_help"],
            info["directed_margin_gain"],
            abs(info["m0_margin"]),
        )
    if target == "failure":
        return (
            info["kind"] == "graph_hurt",
            info["kind"] in ("graph_hurt", "m3c_failure"),
            info["verified_hurt"],
            abs(info["m3c_margin"]),
        )
    return (
        info["kind"] == "low_margin_correct",
        -abs(info["m3c_margin"]),
        info["m3c_pred"] == info["gt"],
    )


def choose_examples(records: Sequence[Mapping[str, Any]], per_class: int) -> dict[str, list[dict[str, Any]]]:
    values = [{
        "record": record,
        "video_id": str(record.get("video_id", "")),
        "codes": label_codes(str(record.get("video_id", ""))),
        "challenge": challenge_info(record),
    } for record in records]
    selected: dict[str, list[dict[str, Any]]] = {code: [] for code in CLASS_ORDER}
    globally_used: set[str] = set()
    for code in CLASS_ORDER:
        pool = [value for value in values if code in value["codes"]]
        class_used: set[str] = set()
        for target in ("graph_help", "failure", "boundary"):
            candidates = [
                value for value in pool
                if value["video_id"] not in class_used and value["video_id"] not in globally_used
            ]
            if not candidates:
                candidates = [value for value in pool if value["video_id"] not in class_used]
            if not candidates:
                continue
            candidates.sort(key=lambda value: rank_for_target(value, target), reverse=True)
            chosen = candidates[0]
            selected[code].append(chosen)
            class_used.add(chosen["video_id"])
            globally_used.add(chosen["video_id"])
            if len(selected[code]) >= per_class:
                break
    return selected


def frame_in_intervals(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def choose_nine_frames(frame_count: int, intervals: Sequence[tuple[int, int]]) -> list[int]:
    """Choose one ordered sample per equal temporal bin, preferring GT content."""
    count = max(int(frame_count), 1)
    selected: list[int] = []
    for index in range(9):
        bin_start = int(math.floor(index * count / 9.0))
        bin_end = max(bin_start, int(math.floor((index + 1) * count / 9.0)) - 1)
        center = (bin_start + bin_end) // 2
        overlaps: list[tuple[int, int]] = []
        for gt_start, gt_end in intervals:
            start = max(bin_start, gt_start)
            end = min(bin_end, gt_end)
            if start <= end:
                overlaps.append((start, end))
        if overlaps:
            start, end = max(overlaps, key=lambda value: value[1] - value[0])
            selected.append((start + end) // 2)
        else:
            selected.append(center)
    return selected


def interval_label_rows(intervals: Sequence[tuple[int, int]], max_frame: int) -> list[int]:
    rows: list[list[tuple[float, float]]] = [[], [], []]
    assignments: list[int] = []
    for start, end in intervals:
        x0 = start / max(max_frame, 1)
        x1 = end / max(max_frame, 1)
        padded = (x0 - 0.045, x1 + 0.045)
        assigned = 0
        for row_index, values in enumerate(rows):
            if all(padded[1] < a or padded[0] > b for a, b in values):
                assigned = row_index
                values.append(padded)
                break
        assignments.append(assigned)
    return assignments


def render_strip(
    out_path: Path,
    record: Mapping[str, Any],
    class_code: str,
    challenge: Mapping[str, Any],
    intervals: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    video_id = str(record.get("video_id", ""))
    video_path = str(record.get("video_path", ""))
    probe = probe_video(video_path)
    fps = float(probe.get("fps", 24.0) or 24.0)
    max_record = max(int(record.get("end_frame", 0) or 0), max((end for _, end in intervals), default=0))
    frame_count = max(int(probe.get("frame_count", 0) or 0), max_record + 1, 1)
    max_frame = frame_count - 1
    shown_frames = choose_nine_frames(frame_count, intervals)

    width, height = 1600, 500
    left, right = 70, 1530
    strip_top, strip_bottom = 120, 345
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    title = video_id if len(video_id) <= 112 else video_id[:109] + "..."
    draw.text((left, 25), title, font=font(25, True), fill="#17202a")
    subtitle = (
        f"{class_code} {CLASS_DISPLAY[class_code]} | challenge: "
        f"{challenge['kind'].replace('_', ' ')} | 9 ordered samples"
    )
    draw.text((left, 68), subtitle, font=font(17), fill="#4b5563")

    def x_of(frame: int) -> float:
        return left + (right - left) * frame / max(max_frame, 1)

    for start, end in intervals:
        draw.rectangle((x_of(start), strip_top, x_of(end), strip_bottom), fill="#fde2e2")
    draw.rectangle((left, strip_top, right, strip_bottom), outline="#9aa4b2", width=2)

    cell_width = (right - left) / 9.0
    thumb_width = int(cell_width - 13)
    frame_dir = out_path.parent / (out_path.stem + "_frames")
    frame_records: list[dict[str, Any]] = []
    for index, frame_value in enumerate(shown_frames):
        frame_path = frame_dir / f"frame_{frame_value:07d}.jpg"
        if not frame_path.is_file():
            extract_frame(video_path, frame_value, fps, frame_path)
        x0 = int(left + index * cell_width + 6)
        available_height = strip_bottom - strip_top - 24
        abnormal = frame_in_intervals(frame_value, intervals)
        pasted = False
        if frame_path.is_file():
            try:
                thumb = Image.open(frame_path).convert("RGB")
                thumb.thumbnail((thumb_width, available_height))
                y0 = strip_top + 12 + max(0, (available_height - thumb.height) // 2)
                image.paste(thumb, (x0, y0))
                border = "#d12f2f" if abnormal else "#626e7e"
                draw.rectangle((x0, y0, x0 + thumb.width, y0 + thumb.height), outline=border, width=4)
                pasted = True
            except OSError:
                pass
        frame_records.append({
            "order": index + 1,
            "frame_index": frame_value,
            "gt_abnormal": abnormal,
            "image_path": str(frame_path),
            "rendered": pasted,
        })

    track_top = 380
    draw.text((left, track_top - 27), "GT anomaly intervals (frame indices)", font=font(15, True), fill="#7f1d1d")
    row_assignments = interval_label_rows(intervals, max_frame)
    for (start, end), row_index in zip(intervals, row_assignments):
        y = track_top + row_index * 31
        x0, x1 = x_of(start), x_of(end)
        draw.line((x0, y, x1, y), fill="#d12f2f", width=4)
        draw.line((x0, y - 6, x0, y + 6), fill="#d12f2f", width=3)
        draw.line((x1, y - 6, x1, y + 6), fill="#d12f2f", width=3)
        label = f"{start}-{end}"
        box = draw.textbbox((0, 0), label, font=font(14, True))
        label_width = box[2] - box[0]
        label_x = max(left, min((x0 + x1 - label_width) / 2, right - label_width))
        draw.rectangle((label_x - 4, y + 7, label_x + label_width + 4, y + 28), fill="#f7f8fa")
        draw.text((label_x, y + 7), label, font=font(14, True), fill="#9b2020")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=94)
    return {
        "category_code": class_code,
        "category": CLASS_DISPLAY[class_code],
        "video_id": video_id,
        "video_path": video_path,
        "strip_image": str(out_path),
        "challenge": dict(challenge),
        "gt_intervals": [list(value) for value in intervals],
        "shown_frames": frame_records,
        "fps": fps,
        "frame_count": frame_count,
        "probe_error": probe.get("error", ""),
    }


def call_audit(records: Sequence[Mapping[str, Any]], run_dir: Path) -> dict[str, Any]:
    independent = [len(record.get("independent_node_calls") or {}) for record in records]
    joint = [len(record.get("joint_graph_calls") or {}) for record in records]
    verifier = [int(record.get("blind_verifier") is not None) for record in records]
    repair = [int(bool((record.get("graph_candidates") or {}).get("repair_used"))) for record in records]
    uncertain = [
        int(competition(record, "conditional_ot_full").get("decision") == "uncertain")
        for record in records
    ]

    def raw_chars(field: str) -> list[int]:
        values: list[int] = []
        for record in records:
            for call in (record.get(field) or {}).values():
                values.append(len(str(call.get("raw") or "")))
        return values

    independent_chars = raw_chars("independent_node_calls")
    joint_chars = raw_chars("joint_graph_calls")
    shortlist_chars = [len(str((record.get("graph_candidates") or {}).get("raw") or "")) for record in records]
    verifier_chars = [
        len(str((record.get("blind_verifier") or {}).get("raw") or ""))
        for record in records if record.get("blind_verifier") is not None
    ]
    n = max(len(records), 1)
    means = {
        "shortlist_calls": 1.0,
        "independent_node_calls": statistics.mean(independent),
        "joint_graph_calls": statistics.mean(joint),
        "verifier_calls": statistics.mean(verifier),
        "shortlist_repair_calls": statistics.mean(repair),
        "wider_window_calls": 0.0,
        "discovery_llm_calls": 0.0,
    }
    means["total_vlm_calls"] = sum(means[key] for key in (
        "shortlist_calls", "independent_node_calls", "joint_graph_calls",
        "verifier_calls", "shortlist_repair_calls", "wider_window_calls",
    ))
    stage_tokens = {
        "shortlist": sum(shortlist_chars) / n / 4.0,
        "independent_nodes": sum(independent_chars) / n / 4.0,
        "joint_graphs": sum(joint_chars) / n / 4.0,
        "verifier": sum(verifier_chars) / n / 4.0,
    }
    selection_time = (run_dir / "selection_summary.json").stat().st_mtime
    summary_time = (run_dir / "summary.json").stat().st_mtime
    wall_seconds = max(0.0, summary_time - selection_time)
    workers = int(read_json(run_dir / "run_config.json").get("workers", 1) or 1)
    return {
        "version": "v4_inference_call_audit_v1",
        "completed_windows": len(records),
        "workers": workers,
        "means_per_window": means,
        "count_distributions": {
            "independent_node_calls": dict(Counter(independent)),
            "joint_graph_calls": dict(Counter(joint)),
        },
        "m3c_uncertain_windows": sum(uncertain),
        "m3c_uncertain_fraction": statistics.mean(uncertain),
        "estimated_output_tokens_per_window_chars_div_4": sum(stage_tokens.values()),
        "mean_output_tokens_by_stage_chars_div_4": stage_tokens,
        "eight_frame_presentations_per_window": means["total_vlm_calls"] * 8.0,
        "observed_remote_run": {
            "wall_hours": wall_seconds / 3600.0,
            "wall_minutes_per_completed_window": wall_seconds / 60.0 / n,
            "throughput_windows_per_hour": n / wall_seconds * 3600.0 if wall_seconds else 0.0,
            "worker_equivalent_minutes_per_window": wall_seconds / 60.0 / n * workers,
            "note": "Remote qwen3.6-plus/API run; not an A100 8B benchmark.",
        },
    }


def runtime_markdown(audit: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]) -> str:
    means = audit["means_per_window"]
    tokens = audit["mean_output_tokens_by_stage_chars_div_4"]
    observed = audit["observed_remote_run"]
    node_calls = means["shortlist_calls"] + means["independent_node_calls"]
    graph_calls = node_calls + means["joint_graph_calls"]
    node_tokens = tokens["shortlist"] + tokens["independent_nodes"]
    graph_tokens = node_tokens + tokens["joint_graphs"]
    lines = [
        "# Challenging Examples and 8B A100 Inference Cost Audit",
        "",
        "## Challenging video selection",
        "",
        "Three videos were selected for each class where available. Selection prioritizes one M0-to-M3c correction, "
        "one M3c failure or regression, and one correct low-margin boundary case. The operational 8-frame label is "
        "used for challenge ranking; every strip shows the complete frame-level GT intervals.",
        "",
        "Each strip has exactly nine non-overlapping thumbnails in temporal order. Light red background blocks are GT "
        "anomaly intervals, and a red thumbnail border means the sampled frame lies inside an interval. Sampled frame "
        "indices are intentionally absent from the image and retained in selected_frame_indices.csv.",
        "",
        "| Category | Video | Challenge | GT | M0 | M3c |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in manifest:
        c = item["challenge"]
        lines.append(
            f"| {item['category']} | {item['video_id']} | {c['kind']} | {c['gt']} | {c['m0_pred']} | {c['m3c_pred']} |"
        )
    lines += [
        "",
        "## Does inference require an LLM?",
        "",
        "No separate teacher or DeepSeek LLM is required after the graph library is frozen. Graph discovery is an offline "
        "training/analysis operation. Inference still needs the 8B VLM, unless that scorer is later distilled into a "
        "non-VLM model, plus inexpensive local OT aggregation. Discovery was disabled in the completed V4 run.",
        "",
        "## Measured call structure of the completed V4 code",
        "",
        "| Stage | Mean VLM calls/window | Production graph score? |",
        "|---|---:|---|",
        f"| Blind graph shortlist | {means['shortlist_calls']:.3f} | Yes, unless local retrieval replaces it |",
        f"| Independent node scoring | {means['independent_node_calls']:.3f} | Yes for current conditional OT |",
        f"| Joint graph scoring | {means['joint_graph_calls']:.3f} | Yes |",
        f"| Graph-help verifier | {means['verifier_calls']:.3f} | No |",
        f"| Wider-window reasoning | {means['wider_window_calls']:.3f} | Not present |",
        f"| Discovery LLM | {means['discovery_llm_calls']:.3f} | No |",
        f"| **Total research pipeline** | **{means['total_vlm_calls']:.3f}** | Includes ablation/verification |",
        "",
        f"The current verbose schema emits roughly **{audit['estimated_output_tokens_per_window_chars_div_4']:.0f} output "
        f"tokens/window** by a characters/4 estimate and presents the same eight images about "
        f"**{audit['eight_frame_presentations_per_window']:.1f} times/window**. Repeated multimodal calls, rather than "
        "Sinkhorn OT arithmetic, dominate runtime.",
        "",
        f"The completed remote run took **{observed['wall_hours']:.2f} wall-hours** with three workers, or "
        f"**{observed['wall_minutes_per_completed_window']:.2f} wall-minutes/completed window** and "
        f"**{observed['throughput_windows_per_hour']:.2f} windows/hour**. This includes API queueing, retries, timeouts, "
        "and verbose output; it is not a local A100 result.",
        "",
        "## A100 80GB and 8B VLM estimates",
        "",
        "These are engineering ranges, not measured benchmarks. Assumptions: BF16, FlashAttention/vLLM-style serving, "
        "eight frames, warm weights, and concise score-only JSON. The A100 80GB has 80GB HBM2e and about 2TB/s memory "
        "bandwidth ([NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)); the model scale is similar to the "
        "7B-8B multimodal family described by the [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923).",
        "",
        "| Implementation | Calls/window | Estimated latency/window | Peak GPU memory |",
        "|---|---:|---:|---:|",
        "| Current unbatched verbose research schema | 33.57 | 80-180 s | 22-32 GB |",
        "| Compact conditional OT: local retrieval + batched node + batched graph | 2 | **7-18 s** | **22-32 GB** |",
        "| Compact conditional OT with VLM shortlist | 3 | 10-25 s | 22-32 GB |",
        "| Direct joint graph scoring without independent prior | 1 | 4-10 s | 20-28 GB |",
        "",
        "The one-call direct row is cheaper but is not the current M3c algorithm because M3c conditions on independent "
        "node probabilities. The recommended deployment is two compact calls: one response containing all shortlisted "
        "node scalars, followed by one response containing all ten conditional graph scores. Reuse one visual encoding "
        "and remove evidence prose and graph-over-node verification.",
        "",
        "Eight billion BF16 parameters occupy about 16GB. Vision components, CUDA workspaces, image tokens, KV cache, and "
        "allocator headroom produce the estimated 22-32GB single-request peak. Once weights are resident, a useful "
        "marginal estimate is 3-8GB per active window. Server-reserved KV memory is not memory consumed by one window.",
        "",
        "## Latency per full video",
        "",
        "For F frames, window 96 and stride 16 give:",
        "",
        "$$N_w = 1 + \\left\\lfloor\\frac{F-96}{16}\\right\\rfloor.$$",
        "",
        "The earlier 500-video manifest contained 44,436 windows, or about 88.9 windows/video:",
        "",
        "| Implementation | Serial latency/video | Batched wall estimate |",
        "|---|---:|---:|",
        "| Current verbose local 8B estimate | 2.0-4.5 h | 0.7-2.0 h |",
        "| Compact conditional OT | **10.4-26.7 min** | **5-15 min** |",
        "| Compact OT plus one wider call on 29.6% uncertain windows | 13-33 min | 6-18 min |",
        "",
        "The batched range assumes two to four concurrent windows and shared model weights. It is not exact linear "
        "division because image prefill and autoregressive decode contend for the same GPU.",
        "",
        "## Graph matching versus independent-node cost",
        "",
        f"Current node-only inference uses about **{node_calls:.2f} calls/window**. Conditional graph inference adds ten "
        f"joint calls for **{graph_calls:.2f} calls/window**, or **{graph_calls / node_calls:.2f}x** the calls. Estimated "
        f"output rises from **{node_tokens:.0f}** to **{graph_tokens:.0f} tokens/window** "
        f"(**{graph_tokens / node_tokens:.2f}x**).",
        "",
        "After compact batching and shared visual encoding, node-only needs one score call and conditional graph matching "
        "needs two. Graph matching should then be about 1.4-2.0x node-only latency, instead of the present 2.5x output "
        "burden. Local OT and polarity aggregation are negligible compared with VLM encoding and JSON generation.",
        "",
        "## Recommended deployment profile",
        "",
        "1. Freeze V4 graphs; disable discovery, verifier, graph-help mining, and report prose.",
        "2. Replace VLM shortlist with local embedding retrieval, or retain one concise shortlist call.",
        "3. Batch all shortlisted node scores into one compact VLM response.",
        "4. Batch all ten graph scores into one compact conditional response and reuse visual embeddings.",
        "5. Trigger one compact wider-window call only for uncertain margins.",
        "6. Benchmark 100 representative windows on the actual A100 before publishing final latency.",
    ]
    return "\n".join(lines) + "\n"


def html_report(out_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for code in CLASS_ORDER:
        items = [item for item in manifest if item["category_code"] == code]
        figures: list[str] = []
        for item in items:
            rel = Path(item["strip_image"]).relative_to(out_dir).as_posix()
            figures.append(
                f'<figure><h3>{html.escape(item["video_id"])}</h3>'
                f'<p>{html.escape(item["challenge"]["kind"].replace("_", " "))}</p>'
                f'<img src="{html.escape(rel)}" alt="GT frame strip for {html.escape(item["video_id"])}"></figure>'
            )
        sections.append(f'<section><h2>{code} {CLASS_DISPLAY[code]}</h2>{"".join(figures)}</section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Challenging GT frame strips</title>
<style>body{{margin:0;background:#f7f8fa;color:#17202a;font-family:Segoe UI,Arial,sans-serif}}main{{max-width:1640px;margin:auto;padding:28px}}h1{{font-size:32px}}h2{{font-size:25px;margin-top:40px;border-bottom:2px solid #b8c1cc;padding-bottom:8px}}h3{{font-size:18px;margin:0 0 4px}}p{{color:#566170;margin:0 0 10px}}figure{{margin:24px 0 34px}}img{{display:block;width:100%;height:auto;border:1px solid #aab3bf;background:white}}</style>
</head><body><main><h1>Challenging videos: sampled frames and frame-level GT</h1>
<p>Light red blocks are GT anomaly intervals. Red thumbnail borders identify samples inside those intervals.</p>
{''.join(sections)}</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--examples-per-class", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "challenging_gt_frame_strips")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "run_config.json")
    annotations_path = args.annotations
    if annotations_path is None:
        text = str(config["annotations"])
        if text.startswith("/mnt/"):
            annotations_path = Path(f"{text[5].upper()}:/{text[7:]}")
        else:
            annotations_path = Path(text)
    annotations = load_annotations(annotations_path)
    records = [read_json(path) for path in sorted((run_dir / "records").glob("*.json"))]
    selected = choose_examples(records, max(2, min(args.examples_per_class, 3)))

    manifest: list[dict[str, Any]] = []
    for code in CLASS_ORDER:
        for index, value in enumerate(selected[code], 1):
            record = value["record"]
            video_id = value["video_id"]
            out_path = out_dir / "strips" / code / f"{index:02d}_{safe_name(video_id)}.png"
            item = render_strip(out_path, record, code, value["challenge"], annotations.get(video_id, ()))
            manifest.append(item)
            print(f"[strip] {code} {index}: {video_id} -> {out_path}", flush=True)

    frame_rows: list[dict[str, Any]] = []
    for item in manifest:
        for frame in item["shown_frames"]:
            frame_rows.append({
                "category_code": item["category_code"],
                "category": item["category"],
                "video_id": item["video_id"],
                "challenge_kind": item["challenge"]["kind"],
                "sample_order": frame["order"],
                "frame_index": frame["frame_index"],
                "gt_abnormal": int(frame["gt_abnormal"]),
                "frame_image_path": frame["image_path"],
            })
    write_csv(out_dir / "selected_frame_indices.csv", frame_rows)
    write_json(out_dir / "challenging_examples_manifest.json", manifest)

    audit = call_audit(records, run_dir)
    write_json(out_dir / "inference_call_audit.json", audit)
    (out_dir / "CHALLENGING_EXAMPLES_AND_RUNTIME_ANALYSIS.md").write_text(
        runtime_markdown(audit, manifest), encoding="utf-8"
    )
    (out_dir / "index.html").write_text(html_report(out_dir, manifest), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "examples": len(manifest),
        "frames": len(frame_rows),
        "mean_vlm_calls_per_window": audit["means_per_window"]["total_vlm_calls"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
