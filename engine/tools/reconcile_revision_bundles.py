#!/usr/bin/env python3
"""Resolve duplicate graph mutations in a revision bundle before critic review."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from discover_ot_failures import DeepSeekClient, _validate_graph
from event_constitution import audit_graph, load_constitution
from hypothesis_contract import atomic_contract_errors, bundle_contract_errors, clean_targets


PROMPT_VERSION = "revision_bundle_reconciliation_v1"


def _bundle_id(row: Mapping[str, Any]) -> str:
    revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
    return str(revision.get("bundle_id") or row.get("bundle_id") or row.get("id") or "")


def _catalog(path: Path) -> tuple[dict[str, dict], set[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for polarity in ("abnormal", "normal"):
        for graph in raw.get(polarity, []):
            if isinstance(graph, Mapping) and graph.get("key"):
                index[str(graph["key"])] = dict(graph)
    return index, set(index)


def _mutation_identity(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if row.get("status") != "proposed":
        return None
    operation = str(row.get("operation", "NOOP")).upper()
    graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
    if operation == "UPDATE":
        targets = clean_targets(row.get("targets", []))
        return (_bundle_id(row), operation, targets[0]) if len(targets) == 1 else None
    if operation == "ADD" and graph.get("key"):
        return (_bundle_id(row), operation, str(graph["key"]))
    return None


def _collision_groups(rows: list[dict]) -> list[tuple[tuple[str, str, str], list[dict]]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        identity = _mutation_identity(row)
        if identity:
            groups[identity].append(row)
    return [(identity, values) for identity, values in groups.items() if len(values) > 1]


def _score(row: Mapping[str, Any]) -> tuple[int, float, str]:
    graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
    return (
        len(graph.get("nodes", [])) if isinstance(graph.get("nodes"), list) else 0,
        float(row.get("confidence", 0.0) or 0.0),
        str(row.get("id", "")),
    )


def _prompt(identity: tuple[str, str, str], rows: list[dict], active_graph: dict,
            constitution: dict) -> str:
    schema = {
        "resolution": "select|merge_into_one|noop|representation_only",
        "target": identity[2],
        "selected_parent_ids": [],
        "superseded_parent_ids": [],
        "reason": "",
        "confidence": 0.0,
        "graph_hypothesis": {},
    }
    return f"""Reconcile one ambiguous revision bundle. This is a bounded mutation-contract step,
not a new graph-discovery round. Multiple drafts mutate the same target/key. Select one draft or
merge them into exactly one atomic graph mutation. Do not return two mutations and do not use row
order. Preserve only positive, directly visible required nodes; keep absence/contradiction in
falsifiers. Use noop or representation_only when no safe fixed-graph mutation exists.

Collision identity:
{json.dumps(identity, ensure_ascii=False, indent=2)}

Current active graph, if any:
{json.dumps(active_graph, ensure_ascii=False, indent=2)}

Competing drafts and critic history:
{json.dumps(rows, ensure_ascii=False, indent=2)}

Event Constitution:
{json.dumps(constitution, ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""


def _mock_response(identity: tuple[str, str, str], rows: list[dict]) -> dict:
    selected = max(rows, key=_score)
    selected_id = str(selected.get("id"))
    return {
        "resolution": "select", "target": identity[2],
        "selected_parent_ids": [selected_id],
        "superseded_parent_ids": [
            str(row.get("id")) for row in rows if str(row.get("id")) != selected_id
        ],
        "reason": "mock reconciliation selects the most complete atomic draft",
        "confidence": 0.5, "graph_hypothesis": selected.get("graph", {}),
    }


def _superseded(row: Mapping[str, Any], canonical_id: str, reason: str) -> dict:
    value = dict(row)
    value["status"] = "superseded_revision_draft"
    value["active"] = False
    value["reconciliation"] = {
        "status": "superseded", "canonical_hypothesis_id": canonical_id,
        "reason": reason,
    }
    return value


def reconcile(input_path: Path, graph_catalog: Path, constitution_path: Path, out_dir: Path,
              *, model: str, base_url: str, key_env: str, retries: int, mock: bool) -> dict:
    rows = list(iter_jsonl(input_path))
    catalog, catalog_keys = _catalog(graph_catalog)
    constitution = load_constitution(constitution_path)
    teacher = None if mock else DeepSeekClient(model, base_url, key_env, retries)
    collisions = _collision_groups(rows)
    collision_ids = {str(row.get("id")) for _, values in collisions for row in values}
    output_rows = [dict(row) for row in rows if str(row.get("id")) not in collision_ids]
    errors: list[dict] = []
    reconciled = 0
    superseded_count = 0

    for collision_index, (identity, drafts) in enumerate(collisions, 1):
        bundle_id, operation, target = identity
        case_id = stable_sha1(bundle_id, operation, target, size=16)
        case_path = out_dir / "reconciliation_cases" / f"{case_id}.json"
        fingerprint = stable_sha1(
            PROMPT_VERSION,
            json.dumps(drafts, ensure_ascii=False, sort_keys=True),
            json.dumps(catalog.get(target, {}), ensure_ascii=False, sort_keys=True),
            json.dumps(constitution, ensure_ascii=False, sort_keys=True),
            size=40,
        )
        try:
            cached = json.loads(case_path.read_text(encoding="utf-8")) if case_path.is_file() else None
            cache_matches = bool(
                cached and cached.get("_reconciliation_request_fingerprint") == fingerprint
            )
            if cache_matches:
                response = cached
            elif mock:
                response = _mock_response(identity, drafts)
            else:
                response = teacher.call(
                    "You reconcile duplicate graph mutations conservatively into one atomic disposition.",
                    _prompt(identity, drafts, catalog.get(target, {}), constitution),
                )
            response["_reconciliation_prompt_version"] = PROMPT_VERSION
            response["_reconciliation_request_fingerprint"] = fingerprint
            response["_reconciliation_bundle_id"] = bundle_id
            if not cache_matches:
                write_json(case_path, response)

            resolution = str(response.get("resolution", "")).lower()
            selected_ids = {str(value) for value in response.get("selected_parent_ids", [])}
            base = next(
                (dict(row) for row in drafts if str(row.get("id")) in selected_ids),
                dict(max(drafts, key=_score)),
            )
            canonical: dict | None = None
            if resolution in {"select", "merge_into_one"}:
                graph = base.get("graph", {})
                if resolution == "merge_into_one":
                    graph = response.get("graph_hypothesis", {})
                graph = dict(graph) if isinstance(graph, Mapping) else {}
                canonical = dict(base)
                canonical["graph"] = graph
                canonical["operation"] = operation
                canonical["targets"] = [target] if operation == "UPDATE" else []
                canonical["parents"] = list(canonical["targets"])
                canonical["status"] = "proposed"
                canonical["active"] = False
                if resolution == "merge_into_one":
                    canonical["id"] = "hypothesis_" + stable_sha1(
                        *sorted(str(row.get("id")) for row in drafts), target, graph.get("key", ""),
                        size=16,
                    )
                expected_polarity = str(catalog.get(target, {}).get("polarity", "")) or None
                schema_errors = _validate_graph(graph, expected_polarity)
                schema_errors.extend(atomic_contract_errors(canonical, catalog_keys))
                theory = audit_graph(graph, constitution) if graph else {
                    "passes_hard_invariants": False, "issues": ["missing graph"],
                    "hard_issue_codes": ["missing_graph"],
                }
                if schema_errors or not theory.get("passes_hard_invariants", False):
                    raise ValueError(
                        "unsafe reconciliation: " + "; ".join(
                            [*schema_errors, *theory.get("hard_issue_codes", [])]
                        )
                    )
                canonical["schema_errors"] = []
                canonical["theory_checks"] = theory
                canonical["reconciliation"] = {
                    "status": "canonical", "resolution": resolution,
                    "source_hypothesis_ids": [str(row.get("id")) for row in drafts],
                    "reason": response.get("reason", ""),
                }
                output_rows.append(canonical)
                canonical_id = str(canonical.get("id"))
                for draft in drafts:
                    if str(draft.get("id")) != canonical_id:
                        output_rows.append(_superseded(draft, canonical_id, str(response.get("reason", ""))))
                        superseded_count += 1
                reconciled += 1
                print(
                    f"+ RECONCILED [{collision_index}/{len(collisions)}] "
                    f"bundle={bundle_id} target={target} resolution={resolution}", flush=True,
                )
            else:
                diagnostic_status = (
                    "diagnostic_only" if resolution in {"noop", "representation_only"}
                    else "bundle_contract_rejected"
                )
                for draft in drafts:
                    value = dict(draft)
                    value["status"] = diagnostic_status
                    value["active"] = False
                    value["reconciliation"] = {
                        "status": diagnostic_status, "resolution": resolution or "invalid",
                        "reason": response.get("reason", ""),
                    }
                    output_rows.append(value)
        except Exception as exc:
            error = {
                "bundle_id": bundle_id, "operation": operation, "target": target,
                "type": type(exc).__name__, "error": str(exc),
            }
            if teacher is not None and teacher.last_trace:
                error["deepseek_trace"] = teacher.last_trace
            errors.append(error)
            for draft in drafts:
                value = dict(draft)
                value["status"] = "bundle_contract_rejected"
                value["active"] = False
                value["reconciliation_error"] = error
                output_rows.append(value)
            print(f"[reconciliation-error] bundle={bundle_id} target={target}: {exc}", flush=True)

    post_errors = bundle_contract_errors(output_rows, catalog_keys)
    if post_errors:
        repaired_rows = []
        for row in output_rows:
            row_id = str(row.get("id"))
            if row_id in post_errors and row.get("status") == "proposed":
                row = dict(row)
                row["status"] = "bundle_contract_rejected"
                row["bundle_contract_errors"] = post_errors[row_id]
            repaired_rows.append(row)
        output_rows = repaired_rows
        post_errors = bundle_contract_errors(output_rows, catalog_keys)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "reconciled_hypotheses.jsonl", output_rows)
    write_jsonl(out_dir / "reconciliation_errors.jsonl", errors)
    contract_summary = {
        "version": "bundle_mutation_contract_v1", "stage": "post_reconciliation",
        "rows": len(output_rows), "rows_with_errors": len(post_errors),
        "errors_by_row": post_errors,
    }
    write_json(out_dir / "bundle_contract_summary.json", contract_summary)
    summary = {
        "version": PROMPT_VERSION, "input": len(rows), "collision_groups": len(collisions),
        "reconciled_groups": reconciled, "superseded_drafts": superseded_count,
        "output": len(output_rows), "post_contract_error_rows": len(post_errors),
        "errors": len(errors), "output_path": str(out_dir / "reconciled_hypotheses.jsonl"),
    }
    write_json(out_dir / "reconciliation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--constitution", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reconcile(
        args.input, args.graph_catalog, args.constitution, args.out_dir,
        model=args.model, base_url=args.base_url, key_env=args.key_env,
        retries=args.retries, mock=args.mock,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
