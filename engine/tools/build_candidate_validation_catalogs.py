#!/usr/bin/env python3
"""Build baseline, combined, per-hypothesis, and per-bundle validation catalogs."""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import file_sha256, safe_name, write_json
from hypothesis_contract import atomic_contract_errors, clean_targets, validation_action_set_errors


def _find(catalog: Mapping[str, Any], key: str) -> tuple[str, int] | None:
    for polarity in ("abnormal", "normal"):
        for index, graph in enumerate(catalog.get(polarity, [])):
            if str(graph.get("key")) == key:
                return polarity, index
    return None


def _catalog_keys(catalog: Mapping[str, Any]) -> set[str]:
    return {
        str(graph.get("key")) for polarity in ("abnormal", "normal")
        for graph in catalog.get(polarity, []) if isinstance(graph, Mapping) and graph.get("key")
    }


def _apply(catalog: dict, row: Mapping[str, Any]) -> dict:
    operation = str(row.get("operation", "ADD")).upper()
    graph = copy.deepcopy(row.get("graph", {}))
    targets = clean_targets(row.get("targets", []))
    errors = atomic_contract_errors(row, _catalog_keys(catalog))
    if errors:
        raise ValueError(f"candidate {row.get('id')} violates atomic contract: {', '.join(errors)}")
    graph.update({
        "active": True,
        "status": "validation_candidate",
        "hypothesis_id": row.get("id"),
        "support_source_groups": row.get("support_source_groups", []),
    })
    if operation == "UPDATE":
        found = _find(catalog, targets[0])
        if found:
            del catalog[found[0]][found[1]]
    found = _find(catalog, str(graph.get("key", "")))
    if found:
        del catalog[found[0]][found[1]]
    catalog[str(graph.get("polarity"))].append(graph)
    return {
        "hypothesis_id": row.get("id"),
        "bundle_id": row.get("bundle_id") or row.get("revision", {}).get("bundle_id"),
        "operation": operation,
        "targets": targets,
        "graph_key": graph.get("key"),
    }


def _write_catalog(root: Path, name: str, base: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict:
    catalog = {
        "version": "exact_candidate_validation_catalog_v1_not_for_deployment",
        "abnormal": copy.deepcopy(base.get("abnormal", [])),
        "normal": copy.deepcopy(base.get("normal", [])),
    }
    actions = [_apply(catalog, row) for row in rows]
    for polarity in ("abnormal", "normal"):
        catalog[polarity].sort(key=lambda graph: str(graph.get("key", "")))
    directory = root / name
    path = directory / "graph_catalog_v2.json"
    write_json(path, catalog)
    return {
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
        "hypothesis_ids": [str(row.get("id")) for row in rows],
        "actions": actions,
        "counts": {polarity: len(catalog[polarity]) for polarity in ("abnormal", "normal")},
    }


def build(
    base_path: Path,
    registry_path: Path,
    out_dir: Path,
    include_hypothesis_ids: Sequence[str] | None = None,
    exclude_hypothesis_ids: Sequence[str] | None = None,
    skip_combined: bool = False,
) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_rows = [row for row in registry.get("candidate_graphs", []) if row.get("status") == "candidate"]
    include = {str(value) for value in (include_hypothesis_ids or []) if str(value)}
    exclude = {str(value) for value in (exclude_hypothesis_ids or []) if str(value)}
    rows = [
        row for row in registry_rows
        if (not include or str(row.get("id")) in include) and str(row.get("id")) not in exclude
    ]
    missing = include - {str(row.get("id")) for row in registry_rows}
    if missing:
        raise ValueError("unknown --include-hypothesis-ids: " + ", ".join(sorted(missing)))
    action_errors = validation_action_set_errors(rows, _catalog_keys(base))
    if action_errors:
        raise ValueError(
            "unsafe validation action set: "
            + "; ".join(f"{value['row_id']}:{value['code']}" for value in action_errors)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _write_catalog(out_dir, "baseline", base, [])
    combined = None if skip_combined else _write_catalog(out_dir, "combined", base, rows)
    hypotheses = {}
    for row in rows:
        row_id = str(row.get("id"))
        hypotheses[row_id] = _write_catalog(out_dir, f"hypothesis_{safe_name(row_id)}", base, [row])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        bundle = str(row.get("bundle_id") or row.get("revision", {}).get("bundle_id") or "")
        if bundle:
            grouped[bundle].append(row)
    bundles = {
        bundle: _write_catalog(out_dir, f"bundle_{safe_name(bundle)}", base, members)
        for bundle, members in sorted(grouped.items())
    }
    manifest = {
        "version": "exact_candidate_validation_catalogs_manifest_v1",
        "deployable": False,
        "base_catalog": str(base_path),
        "base_catalog_sha256": file_sha256(base_path),
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "registry_candidate_count": len(registry_rows),
        "selected_candidate_count": len(rows),
        "candidate_count": len(rows),
        "selected_hypothesis_ids": [str(row.get("id")) for row in rows],
        "excluded_hypothesis_ids": sorted(exclude),
        "combined_requested": not skip_combined,
        "baseline": baseline,
        "combined": combined,
        "hypotheses": hypotheses,
        "bundles": bundles,
        "scientific_rule": "activation evidence requires each exact run versus baseline on one identical window manifest",
    }
    write_json(out_dir / "catalogs_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--include-hypothesis-ids", nargs="*", default=[])
    parser.add_argument("--exclude-hypothesis-ids", nargs="*", default=[])
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(
        args.base, args.registry, args.out_dir,
        include_hypothesis_ids=args.include_hypothesis_ids,
        exclude_hypothesis_ids=args.exclude_hypothesis_ids,
        skip_combined=args.skip_combined,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
