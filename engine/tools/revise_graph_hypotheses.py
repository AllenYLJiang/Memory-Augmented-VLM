#!/usr/bin/env python3
"""Revise critic-returned hypotheses into atomic, constitution-audited mutations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from discover_ot_failures import DeepSeekClient, _validate_graph
from event_constitution import audit_graph, load_constitution
from hypothesis_contract import (
    atomic_contract_errors, bundle_contract_errors, clean_targets, graph_schema,
)
from reflect_failure_clusters import _case_for_llm


REVISION_OPERATIONS = {"ADD", "UPDATE", "NOOP", "REPRESENTATION_ONLY"}
REVISION_PROMPT_VERSION = "atomic_graph_revision_v2_bundle_unique"


def _catalog(path: Path) -> tuple[dict[str, dict], set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for polarity in ("abnormal", "normal"):
        for graph in raw.get(polarity, []):
            if isinstance(graph, Mapping) and graph.get("key"):
                index[str(graph["key"])] = dict(graph)
    return index, set(index)


def _cluster_payload(cluster: Mapping[str, Any] | None) -> dict:
    if not cluster:
        return {}
    payload = {
        key: value for key, value in cluster.items()
        if key not in {"failure_examples", "support_video_ids"}
    }
    payload["failure_examples"] = [_case_for_llm(row) for row in cluster.get("failure_examples", [])]
    return payload


def _relevant_graphs(row: Mapping[str, Any], catalog: Mapping[str, dict]) -> list[dict]:
    graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
    requested = clean_targets(row.get("declared_targets", row.get("targets", [])))
    requested.extend(clean_targets(graph.get("counterfactual_links", [])))
    family = str(graph.get("family", ""))
    relevant = []
    for key, value in catalog.items():
        if key in requested or (family and str(value.get("family", "")) == family):
            relevant.append(value)
    return relevant[:12]


def _schema() -> dict:
    return {
        "revision_summary": "",
        "revisions": [{
            "operation": "ADD|UPDATE|NOOP|REPRESENTATION_ONLY",
            "target": "one existing graph key for UPDATE; empty for ADD",
            "reason": "",
            "confidence": 0.0,
            "graph_hypothesis": graph_schema(),
        }],
    }


def _input_bundle_id(row: Mapping[str, Any]) -> str:
    revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
    return str(revision.get("bundle_id") or row.get("bundle_id") or row.get("id") or "")


def _bundle_context(row: Mapping[str, Any], rows: list[dict]) -> dict:
    bundle_id = _input_bundle_id(row)
    members = [value for value in rows if _input_bundle_id(value) == bundle_id]
    return {
        "parent_bundle_id": bundle_id,
        "sibling_hypotheses": [{
            "id": value.get("id"), "operation": value.get("operation"),
            "targets": value.get("targets", []), "declared_targets": value.get("declared_targets", []),
            "graph_key": (value.get("graph") or {}).get("key"),
            "critic": value.get("critic", {}),
        } for value in members],
    }


def _revision_request_fingerprint(
    row: Mapping[str, Any], bundle_context: Mapping[str, Any],
    cluster: Mapping[str, Any] | None, relevant_graphs: list[dict],
    constitution: Mapping[str, Any], revision_round: int,
) -> str:
    return stable_sha1(
        REVISION_PROMPT_VERSION,
        json.dumps(row, ensure_ascii=False, sort_keys=True),
        json.dumps(bundle_context, ensure_ascii=False, sort_keys=True),
        json.dumps(_cluster_payload(cluster), ensure_ascii=False, sort_keys=True),
        json.dumps(relevant_graphs, ensure_ascii=False, sort_keys=True),
        json.dumps(constitution, ensure_ascii=False, sort_keys=True),
        revision_round,
        size=40,
    )


def _prompt(row: Mapping[str, Any], cluster: Mapping[str, Any] | None,
            catalog: Mapping[str, dict], constitution: Mapping[str, Any],
            bundle_context: Mapping[str, Any]) -> str:
    declared_targets = clean_targets(row.get("declared_targets", row.get("targets", [])))
    return f"""You are the Graph Revision Editor. The latest adversarial critic requested revision.
Repair the hypothesis conservatively; do not defend the old draft.

Every graph mutation must be atomic:
- UPDATE has exactly one existing target and graph_hypothesis.key must equal that target.
- ADD has no target and must use a genuinely new key.
- If the old proposal named multiple targets, emit one revision item per graph that really needs
  editing. Give every declared target either an UPDATE item or an explicit NOOP item explaining why
  it should remain unchanged. Never promise a companion update without emitting it.
- The sibling bundle context may contain another draft for the same target. Emit at most one
  materialized UPDATE for any target in this response. A later reconciliation gate will reject
  unresolved cross-response target collisions; never rely on row order or last-write-wins.
- Do not encode absence as a required positive node. Put contradictory/absent evidence in falsifiers
  and use positive, directly visible body, object, motion, trajectory, contact, or state-change cues.
- Use REPRESENTATION_ONLY when the critic's request cannot be expressed by a stable fixed graph.
- A revised graph remains inactive and can only be recommended for held-out validation.

Event Constitution:
{json.dumps(constitution, ensure_ascii=False, indent=2)}

Original hypothesis and first critic:
{json.dumps(row, ensure_ascii=False, indent=2)}

Sibling bundle context:
{json.dumps(bundle_context, ensure_ascii=False, indent=2)}

Declared targets requiring an explicit disposition:
{json.dumps(declared_targets, ensure_ascii=False)}

Relevant active graph definitions:
{json.dumps(_relevant_graphs(row, catalog), ensure_ascii=False, indent=2)}

Frozen recurring evidence (anonymous and observer-first):
{json.dumps(_cluster_payload(cluster), ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(_schema(), ensure_ascii=False, indent=2)}"""


def _mock_response(row: Mapping[str, Any], catalog: Mapping[str, dict]) -> dict:
    operation = str(row.get("operation", "NOOP")).upper()
    original_graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
    targets = clean_targets(row.get("declared_targets", row.get("targets", [])))
    revisions = []
    if operation == "UPDATE":
        for target in targets or clean_targets(row.get("targets", [])):
            value = dict(original_graph) if str(original_graph.get("key")) == target else dict(catalog.get(target, {}))
            revisions.append({
                "operation": "UPDATE", "target": target, "reason": "mock atomic revision",
                "confidence": 0.5, "graph_hypothesis": value,
            })
    elif operation == "ADD" and original_graph:
        revisions.append({
            "operation": "ADD", "target": "", "reason": "mock atomic revision",
            "confidence": 0.5, "graph_hypothesis": dict(original_graph),
        })
    else:
        revisions.append({
            "operation": "NOOP", "target": "", "reason": "mock diagnostic disposition",
            "confidence": 0.5, "graph_hypothesis": {},
        })
    return {"revision_summary": "mock revision", "revisions": revisions}


def _revision_items(response: Mapping[str, Any]) -> list[dict]:
    values = response.get("revisions", [])
    return [dict(value) for value in values if isinstance(value, Mapping)] if isinstance(values, list) else []


def revise(input_path: Path, clusters_path: Path, graph_catalog: Path, constitution_path: Path,
           out_dir: Path, *, model: str, base_url: str, key_env: str, retries: int,
           revision_round: int, mock: bool) -> dict:
    rows = list(iter_jsonl(input_path))
    clusters = {str(row.get("id")): row for row in iter_jsonl(clusters_path)}
    catalog, catalog_keys = _catalog(graph_catalog)
    constitution = load_constitution(constitution_path)
    teacher = None if mock else DeepSeekClient(model, base_url, key_env, retries)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict] = []
    errors: list[dict] = []
    revision_requested = 0
    incomplete_bundles = 0
    for index, row in enumerate(rows, 1):
        recommendation = str(row.get("critic", {}).get("recommendation", ""))
        if recommendation != "revise":
            output_rows.append(dict(row))
            continue
        revision_requested += 1
        hid = str(row.get("id", index))
        output = out_dir / "revision_cases" / f"{hid}.json"
        try:
            cluster = clusters.get(str(row.get("cluster_id")))
            context = _bundle_context(row, rows)
            relevant_graphs = _relevant_graphs(row, catalog)
            request_fingerprint = _revision_request_fingerprint(
                row, context, cluster, relevant_graphs, constitution, revision_round
            )
            cached = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
            cache_matches = bool(
                cached and cached.get("_revision_request_fingerprint") == request_fingerprint
            )
            if cache_matches:
                response = cached
            elif mock:
                response = _mock_response(row, catalog)
            else:
                if cached:
                    print(f"[revision-cache-stale] hypothesis={hid} refreshing v2 revision", flush=True)
                response = teacher.call(
                    "You revise graph memory conservatively and obey atomic mutation contracts.",
                    _prompt(row, cluster, catalog, constitution, context),
                )
            response["_revision_prompt_version"] = REVISION_PROMPT_VERSION
            response["_revision_request_fingerprint"] = request_fingerprint
            response["_revision_bundle_id"] = context.get("parent_bundle_id", "")
            if not cache_matches:
                write_json(output, response)

            items = _revision_items(response)
            declared = clean_targets(row.get("declared_targets", row.get("targets", [])))
            dispositions = {
                str(item.get("target", "")).strip() for item in items
                if str(item.get("target", "")).strip()
            }
            missing = sorted(set(declared) - dispositions)
            if missing:
                incomplete_bundles += 1
            parent_revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
            parent_bundle_id = str(parent_revision.get("bundle_id", ""))
            bundle_basis = parent_bundle_id or hid
            bundle_id = "revision_bundle_" + stable_sha1(bundle_basis, revision_round, size=16)

            if not items:
                raise ValueError("revision response contains no revision items")
            for item_index, item in enumerate(items, 1):
                operation = str(item.get("operation", "NOOP")).upper()
                if operation not in REVISION_OPERATIONS:
                    operation = "NOOP"
                target = str(item.get("target", "")).strip()
                graph = item.get("graph_hypothesis", {})
                graph = dict(graph) if isinstance(graph, Mapping) else {}
                targets = [target] if target else []
                contract_row = {"operation": operation, "targets": targets, "graph": graph}
                expected_polarity = None
                if operation == "UPDATE" and target in catalog:
                    expected_polarity = str(catalog[target].get("polarity", "")) or None
                schema_errors = _validate_graph(graph, expected_polarity) if operation in {"ADD", "UPDATE"} else []
                schema_errors.extend(atomic_contract_errors(contract_row, catalog_keys))
                if missing and operation in {"ADD", "UPDATE"}:
                    schema_errors.append("incomplete_target_disposition:" + ",".join(missing))
                theory = audit_graph(graph, constitution) if graph else {
                    "passes_hard_invariants": operation in {"NOOP", "REPRESENTATION_ONLY"},
                    "issues": [], "hard_issue_codes": [],
                }
                status = "proposed"
                if schema_errors or not theory.get("passes_hard_invariants", False):
                    status = "theory_rejected"
                if operation in {"NOOP", "REPRESENTATION_ONLY"}:
                    status = "diagnostic_only"
                child_id = "hypothesis_" + stable_sha1(
                    hid, revision_round, item_index, operation, target, graph.get("key", ""), size=16,
                )
                child = {
                    key: value for key, value in row.items()
                    if key not in {"id", "status", "critic", "graph", "operation", "targets", "parents",
                                   "schema_errors", "theory_checks", "validation", "version"}
                }
                prior_critic_history = row.get("critic_history", [])
                if not isinstance(prior_critic_history, list):
                    prior_critic_history = []
                child.update({
                    "id": child_id, "memory_type": "graph_hypothesis", "status": status,
                    "active": False, "version": int(row.get("version", 1) or 1) + 1,
                    "operation": operation, "targets": targets, "parents": targets,
                    "parent_hypothesis_id": hid,
                    # The child is an atomic mutation. Keep the original pair/set only at bundle
                    # level so a row-level critic cannot mistake peer targets for missing payloads.
                    "declared_targets": targets,
                    "reason": item.get("reason", ""), "confidence": item.get("confidence", 0.0),
                    "graph": graph, "schema_errors": sorted(set(schema_errors)),
                    "theory_checks": theory, "validation": {"status": "pending"},
                    "critic_history": [
                        *prior_critic_history, row.get("critic", {}),
                    ],
                    "revision": {
                        "round": revision_round, "bundle_id": bundle_id,
                        "parent_bundle_id": parent_bundle_id,
                        "summary": response.get("revision_summary", ""),
                        "bundle_declared_targets": declared,
                        "target_dispositions": sorted(dispositions),
                        "missing_target_dispositions": missing,
                    },
                    "provenance": {
                        **dict(row.get("provenance", {})), "revision_model": model,
                        "revision_input": str(input_path),
                    },
                })
                output_rows.append(child)
                write_json(out_dir / "live_revisions" / f"{child_id}.json", child)
                print(
                    f"+ REVISED [{index}/{len(rows)}] parent={hid} operation={operation} "
                    f"target={target or '-'} status={status}", flush=True,
                )
        except Exception as exc:
            error = {"hypothesis_id": hid, "type": type(exc).__name__, "error": str(exc)}
            if teacher is not None and teacher.last_trace:
                error["deepseek_trace"] = teacher.last_trace
            errors.append(error)
            failed = dict(row)
            failed["status"] = "revision_error"
            failed["revision_error"] = error
            output_rows.append(failed)
            print(f"[revision-error] hypothesis={hid}: {exc}", flush=True)

    write_jsonl(out_dir / "revised_hypotheses.jsonl", output_rows)
    write_jsonl(out_dir / "revision_errors.jsonl", errors)
    bundle_errors = bundle_contract_errors(output_rows, catalog_keys)
    write_json(out_dir / "bundle_contract_summary.json", {
        "version": "bundle_mutation_contract_v1",
        "stage": "pre_reconciliation",
        "rows": len(output_rows),
        "rows_with_errors": len(bundle_errors),
        "errors_by_row": bundle_errors,
    })
    summary = {
        "version": REVISION_PROMPT_VERSION, "input": len(rows),
        "revision_requested": revision_requested, "output": len(output_rows),
        "proposed_for_second_critic": sum(row.get("status") == "proposed" for row in output_rows),
        "diagnostic_only": sum(row.get("status") == "diagnostic_only" for row in output_rows),
        "theory_rejected": sum(row.get("status") == "theory_rejected" for row in output_rows),
        "incomplete_bundles": incomplete_bundles, "errors": len(errors),
        "revision_round": revision_round,
        "bundle_contract_error_rows": len(bundle_errors),
        "prompt_version": REVISION_PROMPT_VERSION,
    }
    write_json(out_dir / "revision_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--constitution", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--revision-round", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(json.dumps(revise(
        args.input, args.clusters, args.graph_catalog, args.constitution, args.out_dir,
        model=args.model, base_url=args.base_url, key_env=args.key_env, retries=args.retries,
        revision_round=args.revision_round, mock=args.mock,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
