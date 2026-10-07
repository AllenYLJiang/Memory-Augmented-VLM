#!/usr/bin/env python3
"""Build an explicitly non-deployable catalog for held-out candidate validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from common import write_json
from hypothesis_contract import (
    atomic_contract_errors, clean_targets, validation_action_set_errors,
)


def _find(catalog: dict, key: str) -> tuple[str, int] | None:
    for polarity in ("abnormal", "normal"):
        for index, graph in enumerate(catalog.get(polarity, [])):
            if str(graph.get("key")) == key:
                return polarity, index
    return None


def _apply(catalog: dict, row: Mapping[str, Any]) -> str:
    operation = str(row.get("operation", "ADD")).upper()
    graph = dict(row.get("graph", {}))
    targets = clean_targets(row.get("targets", []))
    if not graph or row.get("status") != "candidate":
        return "skipped_not_candidate"
    catalog_keys = {
        str(value.get("key")) for polarity in ("abnormal", "normal")
        for value in catalog.get(polarity, []) if value.get("key")
    }
    contract_errors = atomic_contract_errors(row, catalog_keys)
    if contract_errors:
        raise ValueError(
            f"unsafe non-atomic validation candidate {row.get('id')}: {', '.join(contract_errors)}"
        )
    if operation == "UPDATE":
        found = _find(catalog, targets[0])
        if found and str(graph.get("polarity")) != found[0]:
            raise ValueError(
                f"unsafe polarity-changing UPDATE {row.get('id')}: "
                f"target={found[0]} graph={graph.get('polarity')}"
            )
    graph.update({
        "active": True,
        "status": "validation_candidate",
        "confidence": float(row.get("confidence", 0.25) or 0.25),
        "utility": {"global": 0.0, "by_class": {}, "fp_delta": 0, "fn_delta": 0},
        "support_source_groups": row.get("support_source_groups", []),
        "hypothesis_id": row.get("id"),
    })
    key = str(graph.get("key", ""))
    if operation == "ADD":
        if _find(catalog, key):
            return "skipped_duplicate_key"
        catalog[str(graph.get("polarity"))].append(graph)
        return "added_for_validation"
    if operation == "UPDATE":
        found = _find(catalog, targets[0])
        if found:
            del catalog[found[0]][found[1]]
        found = _find(catalog, key)
        if found:
            del catalog[found[0]][found[1]]
        catalog[str(graph.get("polarity"))].append(graph)
        return f"{operation.lower()}_applied_for_validation"
    return "skipped_non_graph_operation"


def build(base_path: Path, registry_path: Path, out_dir: Path) -> dict:
    source_catalog = json.loads(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    catalog = {
        "version": "held_out_validation_catalog_v1_not_for_deployment",
        "abnormal": [dict(value) for value in source_catalog.get("abnormal", [])],
        "normal": [dict(value) for value in source_catalog.get("normal", [])],
    }
    candidate_rows = [
        row for row in registry.get("candidate_graphs", []) if row.get("status") == "candidate"
    ]
    catalog_keys = {
        str(value.get("key")) for polarity in ("abnormal", "normal")
        for value in catalog.get(polarity, []) if value.get("key")
    }
    action_errors = validation_action_set_errors(candidate_rows, catalog_keys)
    for row in candidate_rows:
        if str(row.get("operation", "")).upper() != "UPDATE":
            continue
        targets = clean_targets(row.get("targets", []))
        found = _find(catalog, targets[0]) if len(targets) == 1 else None
        graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
        if found and str(graph.get("polarity", "")) != found[0]:
            action_errors.append({
                "row_id": row.get("id"),
                "code": f"polarity_changing_update:{found[0]}->{graph.get('polarity')}",
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    if action_errors:
        write_json(out_dir / "graph_catalog_v2.json", catalog)
        blocked_manifest = {
            "version": "held_out_validation_catalog_manifest_v2_transactional",
            "deployable": False, "blocked": True,
            "base_catalog": str(base_path), "hypothesis_registry": str(registry_path),
            "candidate_input_count": len(candidate_rows), "actions": [],
            "blocked_actions": action_errors, "empty_reason": "unsafe_validation_action_set",
            "upstream_gate_counts": registry.get("gate_counts", {}),
            "counts": {polarity: len(catalog[polarity]) for polarity in ("abnormal", "normal")},
        }
        write_json(out_dir / "manifest.json", blocked_manifest)
        raise ValueError(
            "unsafe validation action set: "
            + "; ".join(f"{value['row_id']}:{value['code']}" for value in action_errors)
        )
    actions = []
    for row in candidate_rows:
        actions.append({"hypothesis_id": row.get("id"), "graph_key": row.get("graph", {}).get("key"), "result": _apply(catalog, row)})
    for polarity in ("abnormal", "normal"):
        catalog[polarity].sort(key=lambda graph: str(graph.get("key", "")))
    write_json(out_dir / "graph_catalog_v2.json", catalog)
    manifest = {
        "version": "held_out_validation_catalog_manifest_v2_transactional",
        "deployable": False, "blocked": False,
        "base_catalog": str(base_path), "hypothesis_registry": str(registry_path),
        "actions": actions,
        "candidate_input_count": len(candidate_rows),
        "empty_reason": "no_validation_ready_hypotheses" if not candidate_rows else "",
        "upstream_gate_counts": registry.get("gate_counts", {}),
        "blocked_actions": [],
        "counts": {polarity: len(catalog[polarity]) for polarity in ("abnormal", "normal")},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.registry, args.out_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
