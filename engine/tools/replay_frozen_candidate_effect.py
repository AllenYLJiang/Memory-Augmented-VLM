#!/usr/bin/env python3
"""Offline B0 replay: replace only one target graph result in frozen baseline competition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, read_json, stable_sha1, write_json, write_jsonl
from competition import _aggregate
from validate_graph_candidates import _delta, _metrics


METHODS = ("conditional_rowmax", "conditional_ot_no_coherence", "conditional_ot_full")


def _records(run: Path) -> dict[str, dict]:
    path = run / "ot_window_results.jsonl"
    if path.is_file():
        return {str(row.get("segment_key")): row for row in iter_jsonl(path)}
    return {
        str(row.get("segment_key")): row
        for path in sorted((run / "records").glob("*.json"))
        if isinstance((row := read_json(path, None)), dict)
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _selected(row: Mapping[str, Any]) -> set[str]:
    value = row.get("graph_candidates", {})
    return set(value.get("selected_abnormal", [])) | set(value.get("selected_normal", []))


def _node_keys(row: Mapping[str, Any], graph_key: str) -> set[str]:
    result = row.get("graph_results", {}).get("conditional_ot_full", {}).get(graph_key, {})
    return set((result.get("node_presence") or {}).keys())


def _evidence_signature(row: Mapping[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    hashes = []
    for value in evidence.get("image_paths", []):
        path = Path(str(value))
        hashes.append(file_sha256(path) if path.is_file() else path.name)
    return stable_sha1(evidence.get("mode"), evidence.get("frame_indices"), hashes, size=40)


def _recompute(base: Mapping[str, Any], candidate: Mapping[str, Any], key: str, method: str, polarity: str) -> dict:
    results = {name: dict(value) for name, value in base.get("graph_results", {}).get(method, {}).items()}
    results[key] = dict(candidate["graph_results"][method][key])
    selected = base.get("graph_candidates", {})
    abnormal_keys = list(selected.get("selected_abnormal", []))
    normal_keys = list(selected.get("selected_normal", []))
    if key not in abnormal_keys and key not in normal_keys:
        (abnormal_keys if polarity == "abnormal" else normal_keys).append(key)
    abnormal = [float(results[name].get("graph_score", 0.0)) for name in abnormal_keys if name in results]
    normal = [float(results[name].get("graph_score", 0.0)) for name in normal_keys if name in results]
    frozen = base.get("competitions", {}).get(method, {})
    aggregation = str(frozen.get("aggregation", "logmeanexp"))
    temperature = float(frozen.get("temperature", 0.1) or 0.1)
    threshold = float(frozen.get("decision_margin_threshold", 0.03) or 0.03)
    margin = _aggregate(abnormal, aggregation, temperature) - _aggregate(normal, aggregation, temperature)
    y_pred = int(margin > threshold)
    return {
        "method": method, "margin": margin, "y_pred": y_pred,
        "decision": "abnormal" if y_pred else "normal",
        "aggregation": aggregation, "temperature": temperature,
        "decision_margin_threshold": threshold,
        "frozen_non_target_results": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--candidate-graph-key", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline, candidate = _records(args.baseline_run), _records(args.candidate_run)
    rows, audit, effects = [], [], []
    for segment in sorted(set(baseline) & set(candidate)):
        base, cand = baseline[segment], candidate[segment]
        reasons = []
        if args.candidate_graph_key not in _selected(base) or args.candidate_graph_key not in _selected(cand):
            reasons.append("target_not_in_both_shortlists")
        if _evidence_signature(base) != _evidence_signature(cand):
            reasons.append("evidence_differs")
        base_nodes, candidate_nodes = _node_keys(base, args.candidate_graph_key), _node_keys(cand, args.candidate_graph_key)
        if not base_nodes or not candidate_nodes or base_nodes != candidate_nodes:
            reasons.append("target_node_keys_incompatible")
        if not bool(cand.get("joint_graph_calls", {}).get(args.candidate_graph_key, {}).get("complete")):
            reasons.append("target_json_incomplete")
        eligible = not reasons
        audit.append({"segment_key": segment, "eligible": eligible, "reasons": ";".join(reasons), "baseline_nodes": ";".join(sorted(base_nodes)), "candidate_nodes": ";".join(sorted(candidate_nodes))})
        if not eligible:
            continue
        selected = cand.get("graph_candidates", {})
        polarity = "abnormal" if args.candidate_graph_key in selected.get("selected_abnormal", []) else "normal"
        competitions = {method: _recompute(base, cand, args.candidate_graph_key, method, polarity) for method in METHODS}
        row = {
            "segment_key": segment, "video_id": base.get("video_id"), "y_true": base.get("y_true"),
            "metric_eligible": base.get("metric_eligible", True), "competitions": competitions,
            "candidate_graph_key": args.candidate_graph_key, "candidate_polarity": polarity,
        }
        rows.append(row)
        before = base.get("competitions", {}).get("conditional_ot_full", {})
        after = competitions["conditional_ot_full"]
        y = int(base.get("y_true", 0))
        if before.get("y_pred") != after.get("y_pred"):
            effects.append({"segment_key": segment, "video_id": base.get("video_id"), "y_true": y, "baseline_pred": before.get("y_pred"), "hybrid_pred": after.get("y_pred"), "effect": "help" if before.get("y_pred") != y and after.get("y_pred") == y else "hurt" if before.get("y_pred") == y and after.get("y_pred") != y else "changed_unresolved", "baseline_margin": before.get("margin"), "hybrid_margin": after.get("margin")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "frozen_hybrid_results.jsonl", rows)
    _write_csv(args.out_dir / "eligibility_audit.csv", audit)
    _write_csv(args.out_dir / "frozen_help_hurt.csv", effects)
    base_rows = [baseline[row["segment_key"]] for row in rows]
    baseline_metrics = _metrics((int(row.get("y_true", 0)), int(row.get("competitions", {}).get("conditional_ot_full", {}).get("y_pred", -1)), float(row.get("competitions", {}).get("conditional_ot_full", {}).get("margin", 0.0)), str(row.get("video_id", ""))) for row in base_rows)
    hybrid_metrics = _metrics((int(row.get("y_true", 0)), int(row["competitions"]["conditional_ot_full"]["y_pred"]), float(row["competitions"]["conditional_ot_full"]["margin"]), str(row.get("video_id", ""))) for row in rows)
    summary = {"version": "frozen_candidate_offline_replay_v1", "matched": len(set(baseline) & set(candidate)), "eligible": len(rows), "ineligible": len(audit)-len(rows), "baseline": baseline_metrics, "hybrid": hybrid_metrics, "delta": _delta(hybrid_metrics, baseline_metrics), "helps": sum(row["effect"] == "help" for row in effects), "hurts": sum(row["effect"] == "hurt" for row in effects)}
    write_json(args.out_dir / "frozen_hybrid_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
