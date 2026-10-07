#!/usr/bin/env python3
"""DeepSeek cluster reflection under the Event Constitution.

This stage may create hypotheses, never active graphs. A cluster with insufficient source-group
recurrence is recorded as NOOP without an API call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from discover_ot_failures import DeepSeekClient, _validate_graph
from event_constitution import audit_graph, load_constitution
from hypothesis_contract import atomic_contract_errors, clean_targets, graph_schema


OPERATIONS = {"ADD", "UPDATE", "MERGE", "SPLIT", "LINK_COUNTERFACTUAL", "RETIRE", "NOOP", "REPRESENTATION_ONLY"}


def _catalog_summaries(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{
        "key": graph.get("key"), "polarity": polarity, "family": graph.get("family"),
        "joint_semantics": graph.get("joint_semantics"),
        "nodes": [node.get("title", node.get("key")) for node in graph.get("nodes", [])],
        "counterfactual_links": graph.get("counterfactual_links", []),
    } for polarity in ("abnormal", "normal") for graph in raw.get(polarity, []) if isinstance(graph, Mapping)]


def _case_for_llm(memory: Mapping[str, Any]) -> dict:
    observed = memory.get("observed_event_state", {})
    observation = observed.get("observation", {}) if isinstance(observed, Mapping) else {}
    return {
        "anonymous_case_id": memory.get("case_id"),
        "source_group_token": stable_sha1(memory.get("source_group", ""), size=10),
        "frozen_blind_observation": observation,
        "posthoc_error": {
            "kind": memory.get("failure_kind"),
            "correct_conclusion": memory.get("posthoc_truth", {}).get("correct_conclusion"),
            "prediction": memory.get("posthoc_truth", {}).get("prediction"),
        },
        "winning_graphs": memory.get("winning_graphs", {}),
        "probability_flow": memory.get("probability_flow", [])[:20],
        "root_cause_hint": memory.get("failure_category"),
    }


def _schema() -> dict:
    return {
        "diagnosis": {
            "failure_class": "selector_failure|node_semantic_failure|temporal_localization_failure|ot_allocation_failure|competition_calibration_failure|catalog_gap|representation_gap|label_noise|insufficient_evidence",
            "shared_invariant_context": [], "stable_mechanism": [], "nuisances": [],
            "discriminative_residual": [], "reason": "",
        },
        "hypotheses": [{
            "operation": "ADD|UPDATE|NOOP|REPRESENTATION_ONLY",
            "target": "exactly one existing graph key for UPDATE; empty for ADD",
            "reason": "", "confidence": 0.0, "graph_hypothesis": graph_schema(),
        }],
    }


def _prompt(cluster: Mapping[str, Any], active_graphs: list[dict], constitution: Mapping[str, Any]) -> str:
    cases = [_case_for_llm(row) for row in cluster.get("failure_examples", [])]
    cluster_payload = dict(cluster)
    cluster_payload["failure_examples"] = cases
    cluster_payload.pop("support_video_ids", None)
    return f"""You are the Cluster Reflector and Graph Editor in a theory-governed memory system.
Blind observations were frozen before labels were revealed. Work across the entire recurring
cluster, not from one case. Separate shared context, stable mechanism, nuisance and the
abnormal/normal discriminative residual.

Return one or more atomic memory hypotheses. Each UPDATE must name exactly one existing target and
its graph key must equal that target. If a counterfactual pair needs two edits, emit two UPDATE
items. ADD has no target and is allowed only if multiple independent source groups share a directly
visible missing mechanism that no active graph expresses. Prefer UPDATE over synonyms. Use NOOP for
perception, selector, calibration, label-noise or insufficient-evidence failures. Use
REPRESENTATION_ONLY for phase, boundary, binding, uncertainty or continuous state factors that a
fixed node-set graph cannot stably express. Do not use absence-only required nodes; put exclusions
in falsifiers. Every graph hypothesis is inactive.

Event Constitution:
{json.dumps(constitution, ensure_ascii=False, indent=2)}

Current active graph summaries:
{json.dumps(active_graphs, ensure_ascii=False, indent=2)}

Recurring cluster:
{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(_schema(), ensure_ascii=False, indent=2)}"""


def _mock_reflection(cluster: Mapping[str, Any]) -> dict:
    return {
        "diagnosis": {
            "failure_class": "insufficient_evidence", "shared_invariant_context": [],
            "stable_mechanism": [], "nuisances": [], "discriminative_residual": [],
            "reason": "mock reflection preserves memory without graph creation",
        },
        "hypotheses": [{
            "operation": "NOOP", "target": "", "reason": "mock", "confidence": 0.5,
            "graph_hypothesis": {},
        }],
    }


def _response_hypotheses(response: Mapping[str, Any]) -> list[dict]:
    """Accept the new atomic schema and legacy single-graph cached responses."""
    values = response.get("hypotheses")
    if isinstance(values, list):
        return [dict(value) for value in values if isinstance(value, Mapping)]
    return [{
        "operation": response.get("operation", "NOOP"),
        "targets": response.get("targets", []),
        "reason": response.get("reason", ""),
        "confidence": response.get("confidence", 0.0),
        "graph_hypothesis": response.get("graph_hypothesis", {}),
        "_legacy": True,
    }]


def reflect(clusters_path: Path, graph_catalog: Path, constitution_path: Path, out_dir: Path,
            *, model: str, base_url: str, key_env: str, max_clusters: int, retries: int,
            mock: bool) -> dict:
    clusters = list(iter_jsonl(clusters_path))
    recurring = [cluster for cluster in clusters if cluster.get("status") == "recurring"]
    if max_clusters > 0:
        recurring = recurring[:max_clusters]
    active_graphs = _catalog_summaries(graph_catalog)
    raw_catalog = json.loads(graph_catalog.read_text(encoding="utf-8"))
    catalog_index = {
        str(graph.get("key")): graph
        for polarity in ("abnormal", "normal")
        for graph in raw_catalog.get(polarity, [])
        if isinstance(graph, Mapping) and graph.get("key")
    }
    catalog_keys = set(catalog_index)
    constitution = load_constitution(constitution_path)
    teacher = None if mock else DeepSeekClient(model, base_url, key_env, retries)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    for index, cluster in enumerate(recurring, 1):
        cid = str(cluster.get("id"))
        output = out_dir / "reflection_cases" / f"{cid}.json"
        try:
            if output.is_file():
                response = json.loads(output.read_text(encoding="utf-8"))
            elif mock:
                response = _mock_reflection(cluster)
                write_json(output, response)
            else:
                response = teacher.call(
                    "You edit semantic memory conservatively under a machine-checkable Event Constitution.",
                    _prompt(cluster, active_graphs, constitution),
                )
                write_json(output, response)
            items = _response_hypotheses(response)
            if not items:
                raise ValueError("reflection response contains no hypothesis items")
            bundle_id = "reflection_bundle_" + stable_sha1(cid, size=16)
            for item_index, item in enumerate(items, 1):
                operation = str(item.get("operation", "NOOP")).upper()
                if operation not in OPERATIONS:
                    operation = "NOOP"
                graph = item.get("graph_hypothesis", {})
                graph = dict(graph) if isinstance(graph, Mapping) else {}
                declared_targets = clean_targets(item.get("targets", []))
                target = str(item.get("target", "")).strip()
                if target:
                    declared_targets = [target]
                targets = list(declared_targets)
                unmaterialized_targets: list[str] = []
                if operation == "ADD":
                    targets = []
                elif operation == "UPDATE" and len(targets) > 1 and str(graph.get("key", "")) in targets:
                    # Legacy cached responses promised several updates but carried only one graph.
                    # Keep the materialized update atomic and expose the missing companions to revision.
                    selected = str(graph.get("key"))
                    unmaterialized_targets = [value for value in targets if value != selected]
                    targets = [selected]
                contract_row = {"operation": operation, "targets": targets, "graph": graph}
                expected_polarity = None
                if operation == "UPDATE" and len(targets) == 1 and targets[0] in catalog_index:
                    expected_polarity = str(catalog_index[targets[0]].get("polarity", "")) or None
                schema_errors = _validate_graph(graph, expected_polarity) if operation in {"ADD", "UPDATE", "MERGE", "SPLIT"} else []
                schema_errors.extend(atomic_contract_errors(contract_row, catalog_keys))
                theory = audit_graph(graph, constitution) if graph else {
                    "passes_hard_invariants": operation in {"NOOP", "REPRESENTATION_ONLY", "RETIRE", "LINK_COUNTERFACTUAL"},
                    "issues": [], "hard_issue_codes": [],
                }
                status = "proposed"
                if schema_errors or not theory.get("passes_hard_invariants", False):
                    status = "theory_rejected"
                if operation in {"NOOP", "REPRESENTATION_ONLY"}:
                    status = "diagnostic_only"
                row = {
                    "id": "hypothesis_" + stable_sha1(cid, operation, graph.get("key", ""), size=16),
                    "memory_type": "graph_hypothesis",
                    "status": status, "active": False, "version": 1,
                    "operation": operation, "targets": targets, "parents": targets,
                    "declared_targets": declared_targets, "unmaterialized_targets": unmaterialized_targets,
                    "bundle_id": bundle_id, "bundle_item": item_index, "cluster_id": cid,
                    "support_case_ids": cluster.get("member_case_ids", []),
                    "support_source_groups": cluster.get("support_source_groups", []),
                    "support_video_ids": cluster.get("support_video_ids", []),
                    "contradicting_source_groups": [],
                    "diagnosis": response.get("diagnosis", {}),
                    "reason": item.get("reason", ""), "confidence": item.get("confidence", 0.0),
                    "graph": graph, "schema_errors": sorted(set(schema_errors)), "theory_checks": theory,
                    "validation": {"status": "pending"},
                    "provenance": {
                        "cluster_file": str(clusters_path), "model": model,
                        "legacy_response": bool(item.get("_legacy")),
                    },
                }
                rows.append(row)
                write_json(out_dir / "live_hypotheses" / f"{row['id']}.json", row)
                print(
                    f"+ REFLECTED [{index}/{len(recurring)}] cluster={cid} operation={operation} "
                    f"target={(targets or ['-'])[0]} status={status}", flush=True,
                )
        except Exception as exc:
            error = {"cluster_id": cid, "type": type(exc).__name__, "error": str(exc)}
            if teacher is not None and teacher.last_trace:
                error["deepseek_trace"] = teacher.last_trace
            errors.append(error)
            print(f"[reflection-error] cluster={cid}: {exc}", flush=True)
    write_jsonl(out_dir / "graph_hypotheses.jsonl", rows)
    write_jsonl(out_dir / "reflection_errors.jsonl", errors)
    summary = {
        "version": "cluster_reflection_v1", "recurring_clusters": len(recurring),
        "hypotheses": len(rows), "graph_operations": sum(row["operation"] not in {"NOOP", "REPRESENTATION_ONLY"} for row in rows),
        "diagnostic_only": sum(row["status"] == "diagnostic_only" for row in rows),
        "theory_rejected": sum(row["status"] == "theory_rejected" for row in rows), "errors": len(errors),
    }
    write_json(out_dir / "reflection_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--constitution", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-clusters", type=int, default=0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reflect(args.clusters, args.graph_catalog, args.constitution, args.out_dir,
        model=args.model, base_url=args.base_url, key_env=args.key_env, max_clusters=args.max_clusters,
        retries=args.retries, mock=args.mock), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
