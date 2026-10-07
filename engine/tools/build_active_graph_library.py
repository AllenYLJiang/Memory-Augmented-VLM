#!/usr/bin/env python3
"""The sole activation authority for validated graph-memory operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from common import write_json


def _find(catalog: dict, key: str):
    for polarity in ("abnormal", "normal"):
        for index, graph in enumerate(catalog.get(polarity, [])):
            if str(graph.get("key")) == key:
                return polarity, index
    return None


def build(base_path: Path, validated_path: Path, out_dir: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    memory = json.loads(validated_path.read_text(encoding="utf-8"))
    catalog = {
        "version": "theory_governed_active_graph_library_v2",
        "abnormal": [dict(value) for value in base.get("abnormal", [])],
        "normal": [dict(value) for value in base.get("normal", [])],
    }
    actions = []
    for row in memory.get("candidate_graphs", []):
        validation = row.get("validation", {})
        if str(validation.get("status")) != "validated":
            actions.append({"hypothesis_id": row.get("id"), "result": "archived_not_validated"})
            continue
        operation = str(row.get("operation", "ADD"))
        graph = dict(row.get("graph", {}))
        targets = [str(value) for value in row.get("targets", [])]
        contribution = validation.get("contribution_vs_baseline_exact", validation.get("contribution_vs_removal", {}))
        graph.update({
            "active": True, "status": "active", "requires_independent_validation": False,
            "confidence": float(row.get("confidence", 0.5) or 0.5),
            "utility": {
                "global": contribution.get("balanced_accuracy", 0.0),
                "ap_delta": contribution.get("ap", 0.0),
                "fp_delta": contribution.get("fp", 0),
                "fn_delta": contribution.get("fn", 0),
            },
            "validation": validation, "operation_history": [{
                "operation": operation, "parents": targets, "reason": row.get("reason", ""),
            }],
        })
        if operation == "RETIRE":
            for target in targets:
                found = _find(catalog, target)
                if found:
                    del catalog[found[0]][found[1]]
            actions.append({"hypothesis_id": row.get("id"), "operation": operation, "result": "retired_targets"})
            continue
        if operation == "LINK_COUNTERFACTUAL":
            for target in targets:
                found = _find(catalog, target)
                if found:
                    links = catalog[found[0]][found[1]].setdefault("counterfactual_links", [])
                    for link in graph.get("counterfactual_links", []):
                        if link not in links:
                            links.append(link)
            actions.append({"hypothesis_id": row.get("id"), "operation": operation, "result": "links_updated"})
            continue
        if operation in {"UPDATE", "MERGE", "SPLIT"}:
            for target in targets:
                found = _find(catalog, target)
                if found:
                    del catalog[found[0]][found[1]]
        existing = _find(catalog, str(graph.get("key", "")))
        if existing:
            del catalog[existing[0]][existing[1]]
        catalog[str(graph.get("polarity"))].append(graph)
        actions.append({"hypothesis_id": row.get("id"), "operation": operation, "graph_key": graph.get("key"), "result": "activated"})
    for polarity in ("abnormal", "normal"):
        catalog[polarity].sort(key=lambda graph: str(graph.get("key", "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "graph_catalog_v2.json", catalog)
    write_json(out_dir / "validated_graph_memory.json", memory)
    manifest = {
        "version": "theory_governed_active_library_manifest_v2", "base": str(base_path),
        "validated_memory": str(validated_path), "actions": actions,
        "activated_this_build": sum(action.get("result") == "activated" for action in actions),
        "counts": {polarity: len(catalog[polarity]) for polarity in ("abnormal", "normal")},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--validated-memory", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.validated_memory, args.out_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
