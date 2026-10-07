#!/usr/bin/env python3
"""Export existing 72B-selected XD-Violence training anchors for the V5 packet builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from common import file_sha256, write_json, write_jsonl
from selection import is_pure_normal_video, label_codes, source_group_id


def _rank(seed: int, *parts: object) -> str:
    payload = "\0".join([str(seed), *map(str, parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-tools", required=True, type=Path)
    parser.add_argument("--select-root", required=True, type=Path)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--anchor-overlap", choices=("contains", "any"), default="contains")
    parser.add_argument("--top-k-per-video", type=int, default=2)
    parser.add_argument("--pure-normal-videos", type=int, default=500)
    parser.add_argument("--pure-normal-windows-per-video", type=int, default=1)
    parser.add_argument("--pure-normal-seed", type=int, default=20260902)
    args = parser.parse_args()
    sys.path.insert(0, str(args.pipeline_tools.resolve()))
    import train_from_selected_segments as driver  # type: ignore

    stem_index = driver.build_stem_index(args.videos_root)
    counter = driver.FrameCounter()
    jobs, stats = driver.enumerate_jobs(
        args.select_root, stem_index, args.window, args.stride,
        args.anchor_overlap, args.top_k_per_video, counter,
    )
    positive_anchors = driver.gather_anchors(args.select_root / driver.POSITIVE_SUBDIR)
    output = []
    phase_counts = Counter()
    for raw in jobs:
        row = dict(raw)
        stem = str(row["video_id"])
        labels = label_codes(stem)
        phase = "active" if int(row["y_true"]) == 1 else "normal"
        if int(row["y_true"]) == 0 and positive_anchors.get(stem):
            starts = [int(item[0]) for item in positive_anchors[stem]]
            ends = [int(item[1]) for item in positive_anchors[stem]]
            if int(row["start_frame"]) > max(ends):
                phase = "post_event"
            elif int(row["end_frame"]) < min(starts):
                phase = "pre_event"
            else:
                phase = "inter_event_normal"
        hard_context = int(row["y_true"]) == 0 and bool(set(labels) & {"B1", "B4"})
        row.update({
            "dataset_split": "train",
            "source": "xdviolence_train_72b_selected_confident_anchor",
            "source_group": source_group_id(stem),
            "anchor_status": "confident_positive" if int(row["y_true"]) == 1 else "confident_negative",
            "anchor_type": "confident_positive" if int(row["y_true"]) == 1 else "confident_negative",
            "anchor_confidence": 1.0,
            "anchor_confidence_semantics": "membership in upstream 72B-selected positive/negative anchor set; not a calibrated probability",
            "event_phase": phase,
            "hard_context_normal": hard_context,
            "label_codes": labels,
        })
        output.append(row)
        phase_counts[phase] += 1

    pure_normal_candidates = sorted(
        (
            (stem, path) for stem, path in stem_index.items()
            if is_pure_normal_video(stem)
        ),
        key=lambda item: _rank(args.pure_normal_seed, "pure-normal-video", item[0]),
    )
    if args.pure_normal_videos > 0:
        pure_normal_candidates = pure_normal_candidates[:args.pure_normal_videos]
    pure_normal_stats = Counter()
    for stem, video_path in pure_normal_candidates:
        nframes = counter.num_frames(video_path)
        if nframes <= 0:
            pure_normal_stats["skipped_no_frames"] += 1
            continue
        windows = list(driver.iter_windows(nframes, args.window, args.stride))
        ordered = sorted(
            windows,
            key=lambda item: _rank(args.pure_normal_seed, stem, item[0], item[1], item[2]),
        )
        if args.pure_normal_windows_per_video > 0:
            ordered = ordered[:args.pure_normal_windows_per_video]
        for seg_idx, start_frame, end_frame in ordered:
            output.append({
                "segment_key": f"{stem}__seg{seg_idx}_f{start_frame}-{end_frame}",
                "video_id": stem,
                "video_path": str(video_path),
                "seg_idx": seg_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "gt_anom_fraction": 0.0,
                "y_true": 0,
                "label_source": "explicit_filename_label_A",
                "dataset_split": "train",
                "source": "xdviolence_train_explicit_label_A_known_normal",
                "source_group": source_group_id(stem),
                "anchor_status": "known_normal",
                "anchor_type": "pure_normal",
                "anchor_confidence": 1.0,
                "anchor_confidence_semantics": (
                    "explicit filename label_A with no anomaly-code token; known-normal training source"
                ),
                "event_phase": "not_applicable",
                "hard_context_normal": False,
                "known_normal": True,
                "known_normal_from_video_label": True,
                "label_codes": [],
            })
            phase_counts["pure_normal"] += 1
            pure_normal_stats["windows"] += 1
        pure_normal_stats["videos"] += 1
    output.sort(key=lambda row: str(row["segment_key"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, output)
    summary = {
        "version": "crowd_v5_training_anchor_export_v2_true_pure_normal",
        "records": len(output), "videos_root": str(args.videos_root),
        "select_root": str(args.select_root), "window": args.window, "stride": args.stride,
        "anchor_overlap": args.anchor_overlap, "top_k_per_video": args.top_k_per_video,
        "phase_counts": dict(sorted(phase_counts.items())), "enumeration_stats": stats,
        "pure_normal_sampling": {
            "seed": args.pure_normal_seed,
            "requested_videos": args.pure_normal_videos,
            "windows_per_video": args.pure_normal_windows_per_video,
            **dict(pure_normal_stats),
        },
        "output": str(args.out), "output_sha256": file_sha256(args.out),
        "caveat": (
            "pre/post labels are weak temporal roles inferred relative to selected positive anchors; "
            "the V5 source-group gate and negative-tail constraints remain mandatory"
        ),
    }
    write_json(args.out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
