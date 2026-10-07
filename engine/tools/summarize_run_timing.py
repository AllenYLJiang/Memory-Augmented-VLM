#!/usr/bin/env python3
"""Estimate end-to-end wall throughput from persisted run artifact times."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from common import write_json


def _stamp(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def summarize(run_dir: Path) -> dict:
    selection_path = run_dir / "selection_summary.json"
    summary_path = run_dir / "summary.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = sorted((run_dir / "records").glob("*.json"))
    errors_path = run_dir / "errors.jsonl"
    errors = sum(1 for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()) if errors_path.is_file() else 0

    start = selection_path.stat().st_mtime
    end = summary_path.stat().st_mtime
    wall = max(0.0, end - start)
    completed_segments = len(records)
    completed_videos = int(summary.get("n_videos", completed_segments) or completed_segments)
    selected_segments = int(selection.get("selected_windows", completed_segments + errors) or 0)
    selected_videos = int(selection.get("videos_with_selected_windows", selected_segments) or 0)
    workers = int(json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")).get("workers", 1))

    result = {
        "version": "artifact_wall_timing_v1",
        "run_dir": str(run_dir),
        "start_basis": str(selection_path),
        "end_basis": str(summary_path),
        "start_local": _stamp(start),
        "end_local": _stamp(end),
        "wall_seconds": wall,
        "wall_hms": _duration(wall),
        "workers": workers,
        "selected_segments": selected_segments,
        "completed_segments": completed_segments,
        "selected_videos": selected_videos,
        "completed_videos": completed_videos,
        "errors": errors,
        "completion_fraction": completed_segments / selected_segments if selected_segments else 0.0,
        "wall_seconds_per_completed_segment": wall / completed_segments if completed_segments else None,
        "wall_minutes_per_completed_segment": wall / completed_segments / 60.0 if completed_segments else None,
        "wall_seconds_per_completed_video": wall / completed_videos if completed_videos else None,
        "wall_minutes_per_completed_video": wall / completed_videos / 60.0 if completed_videos else None,
        "completed_segments_per_hour": completed_segments / wall * 3600.0 if wall else None,
        "completed_videos_per_hour": completed_videos / wall * 3600.0 if wall else None,
        "interpretation": (
            "End-to-end wall throughput derived from file modification times. It includes concurrency, "
            "retries, API pauses, live discovery, leave-one-out calls, and final report generation; it is "
            "not the service latency of one segment. Per-video and per-segment values are identical here "
            "because the run selected one segment per video."
        ),
    }
    write_json(run_dir / "timing_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.run_dir)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
