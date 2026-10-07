#!/usr/bin/env python3
"""Freeze, audit, and demote the polluted V4 graph library without API calls."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from common import write_json, write_jsonl
from event_constitution import audit_graph, load_constitution, semantic_overlap
from graph_catalog import infer_family


def _load(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid graph catalog: {path}")
    return value


def _graphs(catalog: Mapping[str, Any]):
    for polarity in ("abnormal", "normal"):
        for graph in catalog.get(polarity, []):
            if isinstance(graph, Mapping):
                value = dict(graph)
                value["polarity"] = polarity
                yield value


def _source_groups(graph: Mapping[str, Any]) -> set[str]:
    values = graph.get("discovery_sources", [])
    groups = {
        str(item.get("source_group_id", ""))
        for item in values if isinstance(item, Mapping) and item.get("source_group_id")
    }
    if graph.get("source_group_id"):
        groups.add(str(graph["source_group_id"]))
    return groups


def _counterfactual_links(graph: Mapping[str, Any], all_graphs: list[dict], constitution: Mapping[str, Any]) -> list[str]:
    current = graph.get("counterfactual_links")
    if isinstance(current, list) and current:
        return [str(value) for value in current]
    wanted = set(constitution.get("counterfactual_families", {}).get(str(graph.get("family", "")), []))
    opposite = "normal" if graph.get("polarity") == "abnormal" else "abnormal"
    candidates = [
        value for value in all_graphs
        if value.get("polarity") == opposite and (not wanted or value.get("family") in wanted)
    ]
    candidates.sort(key=lambda value: (-semantic_overlap(graph, value), str(value.get("key", ""))))
    return [str(value.get("key")) for value in candidates[:2] if value.get("key")]


def audit(input_catalog: Path, stable_catalog: Path, constitution_path: Path, out_dir: Path) -> dict:
    constitution = load_constitution(constitution_path)
    source = _load(input_catalog)
    stable = _load(stable_catalog)
    stable_keys = {str(graph.get("key")) for graph in _graphs(stable)}
    all_graphs = list(_graphs(source))
    for graph in all_graphs:
        graph["family"] = str(graph.get("family") or infer_family(
            str(graph.get("key", "")), str(graph.get("title", "")), str(graph.get("polarity", "normal")),
        ))
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_overlaps: dict[str, list[dict]] = {str(graph.get("key")): [] for graph in all_graphs}
    for index, left in enumerate(all_graphs):
        for right in all_graphs[index + 1:]:
            score = semantic_overlap(left, right)
            if score < 0.65:
                continue
            pair = {"other_key": right.get("key"), "overlap": round(score, 4), "other_polarity": right.get("polarity")}
            pair_overlaps[str(left.get("key"))].append(pair)
            pair_overlaps[str(right.get("key"))].append({
                "other_key": left.get("key"), "overlap": round(score, 4), "other_polarity": left.get("polarity"),
            })

    records, active = [], {"version": "theory_governed_active_graph_library_v1", "abnormal": [], "normal": []}
    for graph in all_graphs:
        key = str(graph.get("key", ""))
        theory = audit_graph(graph, constitution)
        groups = _source_groups(graph)
        prior_validation = graph.get("validation", {}) if isinstance(graph.get("validation"), Mapping) else {}
        validated = str(prior_validation.get("status", "")) == "validated"
        stable_original = key in stable_keys
        if validated and theory["passes_hard_invariants"]:
            status, reason = "active", "independently_validated"
        elif stable_original:
            status, reason = "active", "frozen_original_baseline"
        elif not theory["passes_hard_invariants"]:
            status, reason = "retired", "event_constitution_hard_failure"
        else:
            status, reason = "candidate", "unvalidated_discovered_hypothesis"

        graph_record = dict(graph)
        default_falsifiers = [
            "a required visual anchor is absent or unresolved",
            "the nodes cannot be assigned to compatible evidence locations in one episode",
        ]
        if graph.get("polarity") == "normal":
            default_falsifiers.append("a visible harmful mechanism or state change contradicts the benign explanation")
        graph_record.update({
            "status": status,
            "active": status == "active",
            "confidence": float(graph.get("confidence", 0.8 if stable_original else 0.25)),
            "utility": graph.get("utility", {"global": 0.0, "by_class": {}, "fp_delta": 0, "fn_delta": 0}),
            "counterfactual_links": _counterfactual_links(graph, all_graphs, constitution),
            "canonical_factors": graph.get("canonical_factors") or [
                str(node.get("key")) for node in graph.get("nodes", []) if isinstance(node, Mapping)
            ],
            "applicability": graph.get("applicability") or [
                "the required directly visible factors can be evaluated in the current window",
            ],
            "falsifiers": graph.get("falsifiers") or default_falsifiers,
            "requires_independent_validation": status != "active",
        })
        record = {
            "id": key,
            "version": 1,
            "memory_type": "semantic",
            "status": status,
            "activation_reason": reason,
            "graph": graph_record,
            "stable_original": stable_original,
            "support_source_groups": sorted(groups),
            "support_group_count": len(groups),
            "theory_audit": theory,
            "near_duplicates": sorted(pair_overlaps.get(key, []), key=lambda item: -item["overlap"]),
            "provenance": {
                "source_catalog": str(input_catalog),
                "origin": graph.get("origin", "original_partac_graph"),
                "source_case_id": graph.get("source_case_id"),
            },
        }
        records.append(record)
        if status == "active":
            active[str(graph_record["polarity"])].append(graph_record)

    for polarity in ("abnormal", "normal"):
        active[polarity].sort(key=lambda graph: str(graph.get("key", "")))
    write_jsonl(out_dir / "graph_memory_registry.jsonl", records)
    write_json(out_dir / "active_library" / "graph_catalog_v2.json", active)
    write_json(out_dir / "event_constitution_snapshot.json", constitution)

    status_counts = Counter(record["status"] for record in records)
    issue_counts = Counter(
        issue["code"] for record in records for issue in record["theory_audit"]["issues"]
    )
    summary = {
        "version": "theory_governed_graph_audit_v1",
        "source_catalog": str(input_catalog),
        "stable_catalog": str(stable_catalog),
        "input_graphs": len(records),
        "status_counts": dict(status_counts),
        "active_counts": {polarity: len(active[polarity]) for polarity in ("abnormal", "normal")},
        "issue_counts": dict(issue_counts),
        "single_source_graphs": sum(record["support_group_count"] <= 1 and not record["stable_original"] for record in records),
        "active_catalog": str(out_dir / "active_library" / "graph_catalog_v2.json"),
    }
    write_json(out_dir / "audit_summary.json", summary)

    lines = [
        "# Theory-Governed Graph Memory Audit", "",
        f"- Input graphs: {summary['input_graphs']}",
        f"- Active: {status_counts['active']}",
        f"- Candidate: {status_counts['candidate']}",
        f"- Retired: {status_counts['retired']}",
        f"- Single-source discovered graphs: {summary['single_source_graphs']}", "",
        "Only frozen original graphs or independently validated hypotheses are active. No history was deleted.", "",
        "## Graphs", "",
        "| status | polarity | graph | support groups | reason | hard theory issues |", "|---|---|---|---:|---|---|",
    ]
    for record in sorted(records, key=lambda item: (item["status"], item["graph"]["polarity"], item["id"])):
        issues = ", ".join(record["theory_audit"]["hard_issue_codes"]) or "none"
        lines.append(
            f"| {record['status']} | {record['graph']['polarity']} | `{record['id']}` | "
            f"{record['support_group_count']} | {record['activation_reason']} | {issues} |"
        )
    (out_dir / "GRAPH_MEMORY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-catalog", type=Path, default=root / "graph_library_v4_active_discovered" / "graph_catalog_v2.json")
    parser.add_argument("--stable-catalog", type=Path, default=root / "graph_catalog_v2.json")
    parser.add_argument("--constitution", type=Path, default=root / "config" / "event_constitution.yaml")
    parser.add_argument("--out-dir", type=Path, default=root / "graph_memory_governed_v5")
    args = parser.parse_args()
    print(json.dumps(audit(args.input_catalog, args.stable_catalog, args.constitution, args.out_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
