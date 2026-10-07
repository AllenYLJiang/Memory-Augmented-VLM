from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .adapters import BASELINE_RELATIVE, STATE_RELATIVE, load_baseline_snapshot, load_state_snapshot
from .contracts import file_sha256, iter_jsonl, read_json, write_json
from .safety import SEAL_INPUTS


def build_inventory(legacy_run: Path, work_root: Path, base_catalog: Path) -> dict[str, Any]:
    legacy_run, work_root = Path(legacy_run), Path(work_root)
    files = []
    for relative in SEAL_INPUTS:
        path = legacy_run / relative
        files.append({
            "relative_path": relative,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
            "jsonl_rows": sum(1 for _ in iter_jsonl(path)) if path.suffix == ".jsonl" and path.is_file() else None,
        })
    baseline, baseline_conflicts = load_baseline_snapshot(legacy_run)
    states, state_conflicts = load_state_snapshot(legacy_run)
    sources = list(iter_jsonl(legacy_run / "source_disjoint_training_anchors/selected_source_records.jsonl"))
    catalog = read_json(base_catalog, {})
    graph_counts = {polarity: sum(1 for g in catalog.get(polarity, []) if g.get("active", True)) for polarity in ("abnormal", "normal")}
    result = {
        "version": "event_decision_inventory_v1",
        "created_unix": time.time(),
        "legacy_run": str(legacy_run.resolve()),
        "work_root": str(work_root.resolve()),
        "input_policy": "exact_whitelist_no_recursive_result_discovery",
        "files": files,
        "counts": {
            "selected_source_records": len(sources),
            "baseline_unique": len(baseline),
            "baseline_calibration": sum(r["historical_split"] == "calibration" for r in baseline),
            "baseline_validation": sum(r["historical_split"] == "validation" for r in baseline),
            "semantic_state_unique": len(states),
            "baseline_conflicts": len(baseline_conflicts),
            "state_conflicts": len(state_conflicts),
            "active_graphs": graph_counts,
        },
        "excluded_patterns": ["**/backup/**", "**/cache/**", "provider response copies", "merged duplicates"],
    }
    audit = Path(work_root) / "audit"
    write_json(audit / "inventory.json", result)
    write_json(Path(work_root) / "config_snapshot.json", {"legacy_run": str(legacy_run), "base_catalog": str(base_catalog), "inventory_version": result["version"]})
    return result
