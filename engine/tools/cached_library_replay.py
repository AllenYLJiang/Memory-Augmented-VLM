#!/usr/bin/env python3
"""No-API replay of graph-library subsets over graph scores already present in cached records.

This is an availability-limited diagnostic: a pruned graph not shortlisted in the historical
run has no cached score and cannot be recovered without a new VLM run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import iter_jsonl, write_json
from competition import _aggregate
from validate_graph_candidates import _metrics


METHOD = "conditional_ot_full"


def _keys(path: Path) -> tuple[set[str], dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    polarity = {
        str(graph.get("key")): side
        for side in ("abnormal", "normal")
        for graph in value.get(side, [])
        if graph.get("active", True) is not False and str(graph.get("status", "active")) not in {"candidate", "retired", "rejected"}
    }
    return set(polarity), polarity


def replay(records_path: Path, catalogs: list[tuple[str, Path]], out_path: Path) -> dict:
    records = list(iter_jsonl(records_path))
    results = []
    for name, path in catalogs:
        allowed, polarity = _keys(path)
        metric_rows, missing_sides = [], 0
        candidate_counts = []
        for row in records:
            competition = row.get("competitions", {}).get(METHOD, {})
            graph_results = row.get("graph_results", {}).get(METHOD, {})
            abnormal = [float(value.get("graph_score", 0.0)) for key, value in graph_results.items() if key in allowed and polarity.get(key) == "abnormal"]
            normal = [float(value.get("graph_score", 0.0)) for key, value in graph_results.items() if key in allowed and polarity.get(key) == "normal"]
            if not abnormal or not normal:
                missing_sides += 1
                continue
            aggregation = str(competition.get("aggregation", "logmeanexp"))
            temperature = float(competition.get("temperature", 0.1) or 0.1)
            margin = _aggregate(abnormal, aggregation, temperature) - _aggregate(normal, aggregation, temperature)
            threshold = float(competition.get("decision_margin_threshold", 0.03) or 0.03)
            metric_rows.append((int(row.get("y_true", 0)), int(margin > threshold), margin, str(row.get("video_id", ""))))
            candidate_counts.append((len(abnormal), len(normal)))
        metrics = _metrics(metric_rows)
        results.append({
            "name": name, "catalog": str(path), "catalog_graphs": len(allowed),
            "eligible_cached_windows": len(metric_rows), "unavailable_cached_windows": missing_sides,
            "coverage": len(metric_rows) / len(records) if records else 0.0,
            "mean_available_abnormal": sum(value[0] for value in candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
            "mean_available_normal": sum(value[1] for value in candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
            "metrics": metrics,
        })
    summary = {
        "version": "availability_limited_cached_library_replay_v1", "records": len(records),
        "warning": "Compare metrics only when coverage is high and similar. Missing historical shortlist scores require a new VLM run.",
        "libraries": results,
    }
    write_json(out_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--catalog", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    catalogs = []
    for value in args.catalog:
        name, separator, path = value.partition("=")
        if not separator:
            raise SystemExit(f"--catalog must be NAME=PATH: {value}")
        catalogs.append((name, Path(path)))
    print(json.dumps(replay(args.records, catalogs, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
