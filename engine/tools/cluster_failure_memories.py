#!/usr/bin/env python3
"""Build recurring, source-group-separated failure clusters."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl


def _prediction(value: Mapping[str, Any]) -> int:
    prediction = value.get("y_pred")
    return int(prediction) if prediction in {0, 1} else -1


def _graph_family(graph_key: Any, graph_map: Mapping[str, Mapping[str, Any]]) -> str:
    graph = graph_map.get(str(graph_key), {})
    return str(graph.get("family", "unknown"))


def _catalog(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(graph.get("key")): graph
        for polarity in ("abnormal", "normal")
        for graph in raw.get(polarity, []) if isinstance(graph, Mapping)
    }


def _counterfactual_map(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(family): {str(value) for value in values}
        for family, values in raw.get("counterfactual_families", {}).items()
        if isinstance(values, list)
    }


def _family_pair(abnormal_family: str, normal_family: str,
                 counterfactuals: Mapping[str, set[str]], cluster_mode: str) -> tuple[str, str, bool]:
    if cluster_mode != "counterfactual_axis":
        return abnormal_family, normal_family, False
    linked = normal_family in counterfactuals.get(abnormal_family, set())
    if linked:
        return abnormal_family, f"counterfactuals_of_{abnormal_family}", True
    return abnormal_family, normal_family, False


def _record_signature(record: Mapping[str, Any], graph_map: Mapping[str, Mapping[str, Any]]) -> tuple:
    competition = record.get("competitions", {}).get("conditional_ot_full", {})
    y_true = int(record.get("y_true", 0))
    y_pred = _prediction(competition)
    failure_kind = (
        "unresolved" if y_pred == -1 else
        "fn" if y_true == 1 and y_pred == 0 else
        "fp" if y_true == 0 and y_pred == 1 else "correct"
    )
    return (
        failure_kind,
        _graph_family(competition.get("best_abnormal_graph"), graph_map),
        _graph_family(competition.get("best_normal_graph"), graph_map),
    )


def _compact_balanced(record: Mapping[str, Any]) -> dict:
    competition = record.get("competitions", {}).get("conditional_ot_full", {})
    return {
        "anonymous_case_id": record.get("case_id"),
        "source_group_token": stable_sha1(record.get("video_id", ""), size=10),
        "truth": "abnormal" if int(record.get("y_true", 0)) else "normal",
        "prediction": "unresolved" if _prediction(competition) == -1 else ("abnormal" if _prediction(competition) else "normal"),
        "margin": competition.get("margin"),
        "winning_abnormal_graph": competition.get("best_abnormal_graph"),
        "winning_normal_graph": competition.get("best_normal_graph"),
    }


def cluster(input_path: Path, graph_catalog: Path, out_dir: Path, min_support_groups: int,
            max_failures: int, max_balanced: int, records_path: Path | None = None,
            constitution_path: Path | None = None, cluster_mode: str = "error_signature") -> dict:
    graphs = _catalog(graph_catalog)
    counterfactuals = _counterfactual_map(constitution_path)
    memories = list(iter_jsonl(input_path))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for memory in memories:
        winners = memory.get("winning_graphs", {})
        abnormal_family = _graph_family(winners.get("abnormal"), graphs)
        normal_family = _graph_family(winners.get("normal"), graphs)
        pair_abnormal, pair_normal, linked = _family_pair(
            abnormal_family, normal_family, counterfactuals, cluster_mode,
        )
        if cluster_mode == "counterfactual_axis":
            key = ("mixed", "counterfactual_axis" if linked else "off_axis", pair_abnormal, pair_normal)
        else:
            key = (
                str(memory.get("failure_kind", "")),
                str(memory.get("failure_category", "")),
                pair_abnormal,
                pair_normal,
            )
        groups[key].append(memory)

    all_records = list(iter_jsonl(records_path)) if records_path and records_path.is_file() else []
    clusters = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            0 if item[0][1] == "counterfactual_axis" else 1,
            -len({str(row.get("source_group", "")) for row in item[1] if row.get("source_group")}),
            item[0],
        ),
    )
    for signature, rows in ordered_groups:
        source_groups = sorted({str(row.get("source_group", "")) for row in rows if row.get("source_group")})
        cluster_id = "cluster_" + stable_sha1(*signature, size=16)
        failures = sorted(rows, key=lambda row: (str(row.get("source_group", "")), str(row.get("case_id", ""))))[:max_failures]
        correct_positive, hard_normal, contradicting = [], [], []
        for record in all_records:
            rec_sig = _record_signature(record, graphs)
            rec_pair = _family_pair(rec_sig[1], rec_sig[2], counterfactuals, cluster_mode)
            if rec_pair[:2] != signature[2:]:
                continue
            competition = record.get("competitions", {}).get("conditional_ot_full", {})
            y_true = int(record.get("y_true", 0))
            y_pred = _prediction(competition)
            compact = _compact_balanced(record)
            if y_true == y_pred == 1 and len(correct_positive) < 2:
                correct_positive.append(compact)
            elif y_true == y_pred == 0 and abs(float(competition.get("margin", 0.0) or 0.0)) <= 0.15 and len(hard_normal) < 2:
                hard_normal.append(compact)
            elif y_true != y_pred and str(record.get("case_id")) not in {str(row.get("case_id")) for row in failures} and len(contradicting) < 1:
                contradicting.append(compact)
        clusters.append({
            "id": cluster_id,
            "memory_type": "clustered_semantic",
            "status": "recurring" if len(source_groups) >= min_support_groups else "insufficient_recurrence",
            "signature": {
                "failure_kind": signature[0], "failure_category": signature[1],
                "winning_abnormal_family": signature[2], "winning_normal_family": signature[3],
                "cluster_mode": cluster_mode,
                "counterfactual_axis": signature[1] == "counterfactual_axis",
            },
            "failure_kind_distribution": dict(Counter(str(row.get("failure_kind", "")) for row in rows)),
            "failure_category_distribution": dict(Counter(str(row.get("failure_category", "")) for row in rows)),
            "winning_graph_pair_distribution": {
                f"{pair[0]} :: {pair[1]}": count
                for pair, count in Counter(
                    (
                        str(value.get("winning_graphs", {}).get("abnormal")),
                        str(value.get("winning_graphs", {}).get("normal")),
                    )
                    for value in rows
                ).items()
            },
            "member_case_ids": [row.get("case_id") for row in rows],
            "member_count": len(rows),
            "support_source_groups": source_groups,
            "support_video_ids": sorted({str(row.get("window", {}).get("video_id", "")) for row in rows if row.get("window", {}).get("video_id")}),
            "support_group_count": len(source_groups),
            "failure_examples": failures,
            "balanced_examples": {
                "correct_positive_cases": correct_positive,
                "hard_normal_cases": hard_normal,
                "contradicting_cases": contradicting,
            },
            "balanced_evidence_requirement": {
                "failure_cases": min(3, len(rows)), "correct_positive_cases": 2,
                "hard_normal_cases": 2, "contradicting_cases": 1,
            },
            "max_balanced_cases": max_balanced,
        })
    write_jsonl(out_dir / "failure_clusters.jsonl", clusters)
    recurring = [cluster for cluster in clusters if cluster["status"] == "recurring"]
    summary = {
        "version": "failure_cluster_memory_v1", "failures": len(memories),
        "clusters": len(clusters), "recurring_clusters": len(recurring),
        "min_support_groups": min_support_groups,
        "cluster_mode": cluster_mode,
        "output": str(out_dir / "failure_clusters.jsonl"),
    }
    write_json(out_dir / "cluster_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-support-groups", type=int, default=3)
    parser.add_argument("--max-failures", type=int, default=8)
    parser.add_argument("--max-balanced-cases", type=int, default=8)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--constitution", type=Path)
    parser.add_argument("--cluster-mode", choices=("error_signature", "counterfactual_axis"), default="error_signature")
    args = parser.parse_args()
    print(json.dumps(cluster(args.input, args.graph_catalog, args.out_dir, args.min_support_groups,
        args.max_failures, args.max_balanced_cases, args.records, args.constitution,
        args.cluster_mode), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
