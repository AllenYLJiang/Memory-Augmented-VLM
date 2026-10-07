#!/usr/bin/env python3
"""Select only failures belonging to recurring clusters before blind VLM observation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import iter_jsonl, write_json, write_jsonl


def select(input_path: Path, clusters_path: Path, out_path: Path, max_clusters: int) -> dict:
    clusters = [row for row in iter_jsonl(clusters_path) if row.get("status") == "recurring"]
    if max_clusters > 0:
        clusters = clusters[:max_clusters]
    selected_ids = {
        str(case_id)
        for cluster in clusters
        for case_id in cluster.get("member_case_ids", [])
    }
    rows = [row for row in iter_jsonl(input_path) if str(row.get("case_id")) in selected_ids]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    summary = {
        "version": "recurring_cluster_failure_selection_v1",
        "input_failures": sum(1 for _ in iter_jsonl(input_path)),
        "selected_clusters": len(clusters),
        "selected_failures": len(rows),
        "cluster_ids": [cluster.get("id") for cluster in clusters],
        "output": str(out_path),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-clusters", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(select(args.input, args.clusters, args.out, args.max_clusters), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
