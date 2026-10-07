#!/usr/bin/env python3
"""Build a metadata-only dense validation source index without calling a VLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import file_sha256, write_json, write_jsonl
from gt_annotations import label_window, load_annotations
from selection import is_pure_normal_video, source_group_id
from video_completion import expected_window_count


def _lines(path: Path | None) -> set[str]:
    if not path or not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _frame_count(path: Path) -> int:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    finally:
        capture.release()


def build(
    videos_root: Path,
    videos_glob: str,
    annotations_path: Path,
    out_dir: Path,
    window: int = 96,
    stride: int = 16,
    include_videos_file: Path | None = None,
    exclude_videos_file: Path | None = None,
    exclude_source_groups_file: Path | None = None,
) -> dict:
    annotations = load_annotations(annotations_path)
    include = _lines(include_videos_file)
    excluded_videos = _lines(exclude_videos_file)
    excluded_groups = _lines(exclude_source_groups_file)
    rows, video_rows, skipped = [], [], []
    for video_path in sorted(Path(videos_root).glob(videos_glob)):
        video_id = video_path.stem
        if include and video_id not in include and video_path.name not in include and str(video_path) not in include:
            continue
        if video_id in excluded_videos or video_path.name in excluded_videos:
            continue
        group = source_group_id(video_id)
        if group in excluded_groups:
            continue
        known_normal = is_pure_normal_video(video_id)
        if video_id not in annotations and not known_normal:
            skipped.append({"video_id": video_id, "reason": "unknown_gt"})
            continue
        num_frames = _frame_count(video_path)
        if num_frames <= 0:
            skipped.append({"video_id": video_id, "reason": "unreadable_video"})
            continue
        count = expected_window_count(num_frames, window, stride)
        phases: dict[str, int] = {}
        for index in range(count):
            start = index * int(stride)
            end = min(num_frames - 1, start + int(window) - 1)
            gt = label_window(
                video_id, start, end, annotations,
                min_overlap_frames=8, core_overlap_threshold=2.0 / 3.0,
                known_normal=known_normal,
            )
            phase = str(gt.get("event_phase", "unknown"))
            phases[phase] = phases.get(phase, 0) + 1
            rows.append({
                "segment_key": f"{video_id}__seg{index}_f{start}-{end}",
                "video_id": video_id,
                "video_path": str(video_path),
                "start_frame": start,
                "end_frame": end,
                "y_true": gt["y_true_operational"],
                "gt": gt,
                "source_group": group,
                "source_index_only": True,
            })
        video_rows.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "source_group": group,
            "num_frames": num_frames,
            "windows": count,
            "known_normal": known_normal,
            "phase_counts": phases,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_jsonl(out_dir / "videos.jsonl", video_rows)
    summary = {
        "version": "dense_validation_source_index_v1_no_vlm",
        "videos_root": str(videos_root),
        "videos_glob": videos_glob,
        "annotations": str(annotations_path),
        "window_frames": int(window),
        "stride_frames": int(stride),
        "videos": len(video_rows),
        "windows": len(rows),
        "pure_normal_videos": sum(bool(row["known_normal"]) for row in video_rows),
        "pure_normal_windows": sum(bool(row["gt"].get("known_normal")) for row in rows),
        "phase_counts": {
            phase: sum(int(row["phase_counts"].get(phase, 0)) for row in video_rows)
            for phase in sorted({key for row in video_rows for key in row["phase_counts"]})
        },
        "skipped": skipped,
        "api_calls": 0,
        "input_filter_sha256": {
            str(path): file_sha256(path)
            for path in (include_videos_file, exclude_videos_file, exclude_source_groups_file)
            if path and path.is_file()
        },
    }
    write_json(out_dir / "source_index_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--videos-glob", default="*.mp4")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--include-videos-file", type=Path)
    parser.add_argument("--exclude-videos-file", type=Path)
    parser.add_argument("--exclude-source-groups-file", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        args.videos_root, args.videos_glob, args.annotations, args.out_dir,
        args.window, args.stride, args.include_videos_file,
        args.exclude_videos_file, args.exclude_source_groups_file,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
