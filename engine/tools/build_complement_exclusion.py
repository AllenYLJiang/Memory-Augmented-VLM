#!/usr/bin/env python3
"""Build an exact complement exclusion list from an earlier sampled run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_lines(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build(selection_path: Path, active_source_path: Path | None, out_path: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sampled = {
        str(item.get("video_id", "")).strip()
        for item in selection.get("selected_videos", [])
        if isinstance(item, dict) and str(item.get("video_id", "")).strip()
    }
    active_sources = _read_lines(active_source_path)
    exclusions = sampled | active_sources
    source_count = int(selection.get("source_anomaly_videos", 0) or 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(f"{video_id}\n" for video_id in sorted(exclusions)), encoding="utf-8"
    )
    manifest = {
        "version": "conditional_ot_exact_complement_v1",
        "source_selection": str(selection_path),
        "active_library_sources": str(active_source_path or ""),
        "source_anomaly_videos": source_count,
        "prior_sample_videos": len(sampled),
        "active_library_source_videos": len(active_sources),
        "overlap_prior_sample_and_active_sources": len(sampled & active_sources),
        "excluded_union_videos": len(exclusions),
        "safe_remaining_videos_before_window_filter": max(0, source_count - len(exclusions)),
        "exclusion_file": str(out_path),
        "policy": (
            "Sample all available videos, then exclude the earlier sample and every active-graph "
            "discovery source by exact video id. Source groups are not expanded."
        ),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--active-library-sources", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.selection, args.active_library_sources, args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
