#!/usr/bin/env python3
"""Select challenging videos and render GT-only nine-frame strips.

Selection is deterministic and based on saved V4 and dense Part A+C results.
No API or model inference is performed.  The strip contains exactly one frame
from each of nine temporal bins.  A bin samples an annotated anomaly when one
is present, otherwise it samples the bin center.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from export_v4_ap_and_temporal_curves import (
    CLASS_NAMES,
    average_precision,
    competition,
    extract_frame,
    label_codes,
    load_annotations,
    load_source_predictions,
    load_v4_records,
    probe_video,
    read_json,
    safe_name,
    windows_path,
    write_json,
)


CLASS_ORDER = ("B5", "G", "B4", "B1", "B6", "B2")


def is_abnormal(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def choose_bin_frames(max_frame: int, intervals: Sequence[tuple[int, int]], count: int = 9) -> list[int]:
    values: list[int] = []
    for index in range(count):
        bin_start = int(round(index * (max_frame + 1) / count))
        bin_end = int(round((index + 1) * (max_frame + 1) / count)) - 1
        bin_end = max(bin_start, min(bin_end, max_frame))
        intersections = []
        for start, end in intervals:
            left, right = max(bin_start, start), min(bin_end, end)
            if left <= right:
                intersections.append((right - left + 1, left, right))
        if intersections:
            _, left, right = max(intersections)
            value = (left + right) // 2
        else:
            value = (bin_start + bin_end) // 2
        values.append(max(0, min(value, max_frame)))
    # The bins are disjoint, so this should only matter for extremely short clips.
    for index in range(1, len(values)):
        if values[index] <= values[index - 1]:
            values[index] = min(max_frame, values[index - 1] + 1)
    return values


def dense_difficulty(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["y_true_operational"]) for row in rows]
    predictions = [int(row["y_pred"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    positives = sum(labels)
    true_positives = sum(t == 1 and p == 1 for t, p in zip(labels, predictions))
    return {
        "dense_windows": len(rows),
        "dense_positive_windows": positives,
        "dense_accuracy": sum(t == p for t, p in zip(labels, predictions)) / len(rows) if rows else 0.0,
        "dense_recall": true_positives / positives if positives else 0.0,
        "dense_ap": average_precision(labels, scores),
    }


def select_challenging(
    v4_records: Sequence[Mapping[str, Any]],
    source_by_video: Mapping[str, Sequence[dict[str, Any]]],
    per_class: int,
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in v4_records:
        video_id = str(record["video_id"])
        source_rows = source_by_video.get(video_id, ())
        if not source_rows:
            continue
        dense = dense_difficulty(source_rows)
        if dense["dense_positive_windows"] == 0:
            continue
        m0 = competition(record, "independent_direct_nodes")
        m3 = competition(record, "conditional_ot_full")
        gt = int(record.get("y_true", 0) or 0)
        gt_core = int(record.get("y_true_core", 0) or 0)
        m0_pred = int(m0.get("y_pred", 0) or 0)
        m3_pred = int(m3.get("y_pred", 0) or 0)
        margin = float(m3.get("margin", 0.0) or 0.0)
        value = {
            "video_id": video_id,
            "video_path": str(record.get("video_path", "")),
            "segment_key": str(record.get("segment_key", "")),
            "start_frame": int(record.get("start_frame", 0) or 0),
            "end_frame": int(record.get("end_frame", 0) or 0),
            "y_true": gt,
            "y_true_core": gt_core,
            "m0_pred": m0_pred,
            "m0_margin": float(m0.get("margin", 0.0) or 0.0),
            "m3c_pred": m3_pred,
            "m3c_margin": margin,
            "m3c_wrong": int(m3_pred != gt),
            "m3c_false_negative": int(gt == 1 and m3_pred == 0),
            "m0_m3_disagree": int(m0_pred != m3_pred),
            **dense,
        }
        for code in label_codes(video_id):
            candidates[code].append(dict(value, category=code))

    selected: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set()
    for code in CLASS_ORDER:
        pool = [value for value in candidates.get(code, ()) if value["video_id"] not in used]
        pool.sort(key=lambda value: (
            value["m3c_false_negative"],
            value["m3c_wrong"],
            1.0 - value["dense_recall"],
            1.0 - value["dense_accuracy"],
            value["m0_m3_disagree"],
            -abs(value["m3c_margin"]),
            value["dense_windows"],
        ), reverse=True)
        chosen = pool[:per_class]
        if len(chosen) < per_class:
            fallback = [value for value in candidates.get(code, ()) if value["video_id"] not in {x["video_id"] for x in chosen}]
            fallback.sort(key=lambda value: (value["m3c_wrong"], 1.0 - value["dense_recall"], -abs(value["m3c_margin"])), reverse=True)
            chosen.extend(fallback[:per_class - len(chosen)])
        selected[code] = chosen
        used.update(value["video_id"] for value in chosen)
    return selected


def render_strip(
    out_path: Path,
    video_path: str,
    intervals: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    probe = probe_video(video_path)
    fps = float(probe.get("fps", 24.0) or 24.0)
    max_gt = max((end for _, end in intervals), default=0)
    max_frame = max(int(probe.get("frame_count", 0) or 0) - 1, max_gt, 8)
    frame_values = choose_bin_frames(max_frame, intervals, count=9)

    tile_width, tile_height, gap = 188, 142, 8
    margin = 8
    timeline_height = 18
    width = margin * 2 + tile_width * 9 + gap * 8
    height = margin * 2 + tile_height + 14 + timeline_height
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    frame_dir = out_path.parent / (out_path.stem + "_frames")

    for index, frame_value in enumerate(frame_values):
        frame_path = frame_dir / f"frame_{frame_value:07d}.jpg"
        if not frame_path.is_file():
            extract_frame(video_path, frame_value, fps, frame_path)
        x = margin + index * (tile_width + gap)
        y = margin
        tile = Image.new("RGB", (tile_width, tile_height), "#121417")
        if frame_path.is_file():
            try:
                source = Image.open(frame_path).convert("RGB")
                tile = ImageOps.fit(source, (tile_width, tile_height), method=Image.Resampling.LANCZOS)
            except OSError:
                pass
        canvas.paste(tile, (x, y))
        abnormal = is_abnormal(frame_value, intervals)
        draw.rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            outline="#d21f2b" if abnormal else "#697586",
            width=6 if abnormal else 2,
        )

    bar_y = margin + tile_height + 14
    bar_left, bar_right = margin, width - margin
    draw.rectangle((bar_left, bar_y, bar_right, bar_y + timeline_height), fill="#d8dde5", outline="#697586", width=1)
    for start, end in intervals:
        x0 = bar_left + (bar_right - bar_left) * max(0.0, min(start / max_frame, 1.0))
        x1 = bar_left + (bar_right - bar_left) * max(0.0, min(end / max_frame, 1.0))
        draw.rectangle((x0, bar_y, max(x0 + 2, x1), bar_y + timeline_height), fill="#d21f2b")
    for index in range(1, 9):
        x = bar_left + (bar_right - bar_left) * index / 9.0
        draw.line((x, bar_y, x, bar_y + timeline_height), fill="#ffffff", width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=93)
    return {
        "image": str(out_path),
        "fps": fps,
        "frame_count": max_frame + 1,
        "sampled_frames": frame_values,
        "sampled_frame_gt": [int(is_abnormal(value, intervals)) for value in frame_values],
        "gt_intervals": [list(value) for value in intervals],
        "probe_error": str(probe.get("error", "")),
    }


def runtime_analysis() -> str:
    return r"""## Runtime and inference budget

### Measured V4 run

The completed run selected 352 one-window videos and completed 345; seven windows ended in API read timeouts. From `selection_summary.json` to `summary.json`, wall time was **101.97 hours** with three asynchronous workers, equivalent to **17.73 wall minutes per completed one-window video** and **3.38 completed videos/hour**.

The 345 successful records contain **11,582 persisted VLM calls**:

| Call type | Total | Mean per window |
|---|---:|---:|
| Blind graph-candidate shortlist | 345 | 1.000 |
| Independent node unary evidence | 7,739 | 22.432 |
| Conditional graph refinement | 3,450 | 10.000 |
| Blind M0-vs-M3 verifier | 47 | 0.136 |
| Shortlist repair | 1 | 0.003 |
| **Total** | **11,582** | **33.571** |

The response cache contains **11,582 successful VLM response files**, exactly matching the completed-record tally above. The cache also contains 352 non-response JSON metadata files. Failed HTTP attempts and backend retries do not produce response files, so their count cannot be recovered; real HTTP attempts were greater than 11,582. The run stored 251.9 MiB of cache data: 148.7 MiB responses and 103.2 MiB evidence frames.

The effective serial worker time was approximately $3\times17.73=53.20$ minutes per completed window. Dividing by 33.57 calls gives about **95 seconds per persisted cloud-VLM call**, including upload, queueing, generation, parsing, and retry overhead.

### Is an LLM needed at inference?

No DeepSeek or other text-only LLM is needed once the graph library is frozen. Graph discovery was already disabled in this V4 evaluation. Inference still requires a vision-language model to produce the graph shortlist, unary node evidence, and conditional graph evidence; OT and abnormal-versus-normal aggregation are local deterministic computations. Disabling graph-over-node example mining removes the optional blind verifier, report rendering, and discovery calls.

However, with the current code, independent-node VLM calls cannot simply be removed: their probabilities are the unary priors used by unary OT and conditional OT. With discovery and verifier disabled, the current architecture still needs approximately

$$1 + 22.43 + 10 = 33.43\ \text{VLM calls per window}. $$

### 8B VLM memory estimate

Memory was not logged by this API run, so the following is a deployment estimate rather than a measurement. An 8B model needs approximately 16 GB for BF16 weights, 8 GB for INT8 weights, or 4--5 GB for 4-bit weights. Including the visual encoder, KV cache for eight images, activations, CUDA workspaces, and runtime fragmentation, practical planning ranges are:

| Deployment | Batch/concurrency 1 | Concurrency about 3 |
|---|---:|---:|
| BF16 | 22--30 GiB | 32--50 GiB |
| INT8 | 14--22 GiB | 24--40 GiB |
| 4-bit | 10--16 GiB | 18--32 GiB |

An 80 GiB GPU is sufficient for BF16 inference and modest request batching. Actual peak memory depends strongly on image resolution, visual-token count, prompt length, and generated JSON length and must be profiled with `torch.cuda.max_memory_allocated()`.

### 8B latency estimate

Let $t_8$ be measured seconds for one eight-frame 8B-VLM request. Current unbatched graph-only inference takes approximately $33.43t_8$ seconds for one window. Using a planning range of $t_8=3$--$10$ seconds on an 80 GiB high-end GPU gives **100--334 seconds (1.7--5.6 minutes) per one-window video**.

For a dense temporal pass, the earlier 500-video manifest contained 44,436 windows, or about 88.9 windows/video. The current unbatched design would therefore make roughly **2,972 VLM calls/video**, requiring about **2.5--8.3 serial GPU-hours/video** at the same 3--10 second assumption.

The useful production optimization is request batching, not removing local OT:

| Production design | Calls/window | One window at 3--10 s/call | 88.9-window video |
|---|---:|---:|---:|
| Current unbatched M3c, verifier off | 33.43 | 1.7--5.6 min | 2.5--8.3 h |
| Batched shortlist + all unary nodes + all conditional graphs | 3 | 9--30 s | 13--45 min |
| Combined shortlist/unary + batched conditional graphs | 2 | 6--20 s | 9--30 min |
| One direct multi-graph scoring call | 1 | 3--10 s | 4--15 min |

The three-call design preserves the current two-stage semantics most closely. The one-call design is fastest but is a different model because it removes the explicit independent-to-conditional update. GPU batching can improve throughput across videos, but it does not reduce the single-video critical path unless windows are processed concurrently.
"""


def build_html(out_dir: Path, selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    sections = []
    for code in CLASS_ORDER:
        cards = []
        for item in selected.get(code, ()):
            rel = Path(item["strip"]["image"]).relative_to(out_dir).as_posix()
            verdict = "FN" if item["m3c_false_negative"] else "error" if item["m3c_wrong"] else "low-margin/dense challenge"
            cards.append(f"""
<article><h3>{html.escape(item['video_id'])}</h3>
<p><b>Selection:</b> {verdict}; dense recall {item['dense_recall']:.3f}; dense AP {item['dense_ap']:.3f}; V4 M3c margin {item['m3c_margin']:+.4f}.</p>
<img src="{html.escape(rel)}" alt="Nine-frame GT strip for {html.escape(item['video_id'])}"></article>""")
        sections.append(f"<section><h2>{code} {CLASS_NAMES[code]}</h2>{''.join(cards)}</section>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Challenging XD-Violence GT strips</title><style>
body{{margin:0;background:#f7f8fa;color:#17202a;font-family:Segoe UI,Arial,sans-serif}}main{{max-width:1900px;margin:auto;padding:28px}}
h1{{font-size:32px}}h2{{padding-top:24px;border-top:1px solid #c9d1dc}}h3{{font-size:18px;margin:0 0 6px}}p{{color:#4b5563}}
article{{margin:22px 0 34px}}img{{display:block;width:100%;height:auto;border:1px solid #aab4c0;background:white}}
.key{{border-left:4px solid #d21f2b;padding:10px 16px;background:white;max-width:1100px}}
</style></head><body><main><h1>Challenging videos: nine-frame GT strips</h1>
<p class="key">Each strip contains exactly one non-overlapping frame from each of nine temporal bins. Red frame borders and red timeline spans indicate frame-level GT anomaly regions. No anomaly scores are drawn.</p>
{''.join(sections)}</main></body></html>"""


def build_markdown(out_dir: Path, selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    lines = [
        "# Challenging Videos and Inference-Cost Audit", "",
        "## Selection policy", "",
        "Each category contributes three deterministic challenging videos. Selection prioritizes V4 M3c false negatives, "
        "then other M3c errors, low dense Part A+C recall/accuracy, M0--M3 disagreement, and small absolute M3c margin. "
        "Every strip contains exactly nine disjoint temporal-bin samples. If an anomaly interval intersects a bin, that bin "
        "samples from the anomaly; otherwise it samples the bin center. Red borders and red timeline spans come directly "
        "from `annotations_uniform_format.txt`.", "",
    ]
    for code in CLASS_ORDER:
        lines += [f"### {code} {CLASS_NAMES[code]}", "", "| Video | V4 GT/pred | M3c margin | Dense recall | Dense AP | Strip |", "|---|---|---:|---:|---:|---|"]
        for item in selected.get(code, ()):
            rel = Path(item["strip"]["image"]).relative_to(out_dir).as_posix()
            lines.append(
                f"| `{item['video_id']}` | {item['y_true']}/{item['m3c_pred']} | {item['m3c_margin']:+.4f} | "
                f"{item['dense_recall']:.3f} | {item['dense_ap']:.3f} | [{Path(rel).name}]({rel}) |"
            )
        lines.append("")
    lines.append(runtime_analysis())
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "challenging_gt_strips")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "run_config.json")
    annotations_path = args.annotations or windows_path(config["annotations"])
    annotations = load_annotations(annotations_path)
    v4_records = load_v4_records(run_dir)
    v4_videos = {str(record["video_id"]) for record in v4_records}
    source_paths = [windows_path(path) for path in config.get("prediction_files", [])]
    source_by_video, source_audit = load_source_predictions(source_paths, v4_videos, annotations)
    selected = select_challenging(v4_records, source_by_video, max(2, min(args.per_class, 3)))

    rendered: dict[str, list[dict[str, Any]]] = {}
    total = sum(len(values) for values in selected.values())
    done = 0
    for code in CLASS_ORDER:
        rendered[code] = []
        for item in selected.get(code, ()):
            done += 1
            stem = safe_name(item["video_id"])
            strip_path = out_dir / "strips" / code / f"{stem}.png"
            strip = render_strip(strip_path, item["video_path"], annotations.get(item["video_id"], ()))
            value = dict(item, strip=strip)
            rendered[code].append(value)
            print(f"[strip {done}/{total}] {code} {item['video_id']} -> {strip_path}", flush=True)

    manifest = {
        "version": "challenging_gt_strips_v1",
        "run_dir": str(run_dir),
        "annotations": str(annotations_path),
        "selection_policy": "M3c FN/error, then dense recall/accuracy, disagreement, and low absolute margin",
        "frames_per_video": 9,
        "source_audit": source_audit,
        "categories": rendered,
    }
    write_json(out_dir / "manifest.json", manifest)
    (out_dir / "index.html").write_text(build_html(out_dir, rendered), encoding="utf-8")
    (out_dir / "CHALLENGING_VIDEOS_AND_INFERENCE_COST.md").write_text(build_markdown(out_dir, rendered), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "videos": total, "frames": total * 9}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
