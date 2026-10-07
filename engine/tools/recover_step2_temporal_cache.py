#!/usr/bin/env python3
"""Inventory targeted repairs and snapshot existing evidence without API calls."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from common import file_sha256, iter_jsonl, read_json, write_json
from temporal_contract import CONTRACT_VERSION, raw_object, response_errors, window_errors
from vlm_runtime import CachedVideoVLM


def inspect_run(run_dir: Path, manifest: Path) -> dict:
    requested = {str(row["segment_key"]) for row in iter_jsonl(manifest)}
    rows = {str(row["segment_key"]): row for path in (run_dir / "records").glob("*.json")
            if isinstance(row := read_json(path, None), dict) and row.get("segment_key") in requested}
    invalid = [{"segment_key": key, "errors": issues} for key, row in rows.items()
               if (issues := window_errors(row))]
    transient, restart_recoverable, terminal, malformed = [], [], [], []
    # Follow known failures, rather than reread thousands of valid response files on /mnt/c.
    response_paths = set()

    def add_cache_path(value):
        text = str(value or "").replace("\\", "/")
        if "/cache/responses/" in text:
            path = (run_dir / "cache" / "responses" / text.split("/cache/responses/", 1)[1]).resolve()
            path.relative_to((run_dir / "cache" / "responses").resolve())
            if path.is_file():
                response_paths.add(path)

    for row in rows.values():
        for field, kind in (("independent_node_calls", "independent"), ("joint_graph_calls", "joint")):
            for trace in row.get(field, {}).values():
                try:
                    bad = response_errors(raw_object(trace), kind)
                except (ValueError, TypeError):
                    bad = ["unparseable raw response"]
                if bad:
                    add_cache_path(trace.get("cache_path"))
    if (run_dir / "errors.jsonl").is_file():
        for error in iter_jsonl(run_dir / "errors.jsonl"):
            if error.get("segment_key") in requested:
                match = re.search(r"cached=(.+)$", str(error.get("error", "")))
                if match:
                    add_cache_path(match.group(1))
    for path in sorted(response_paths):
        row = read_json(path, {})
        if row.get("segment_key") not in requested or not isinstance(row.get("parsed"), dict):
            continue
        value = row["parsed"]
        item = {"path": str(path.relative_to(run_dir.resolve())), "sha256": file_sha256(path),
                "segment_key": row["segment_key"], "namespace": row.get("namespace", "")}
        error = CachedVideoVLM._provider_error(value)
        if error:
            item["error"] = error
            if CachedVideoVLM._retryable_provider_error(value):
                transient.append(item)
            elif CachedVideoVLM._recoverable_cached_provider_error(value):
                restart_recoverable.append(item)
            else:
                terminal.append(item)
            continue
        namespace = str(row.get("namespace", ""))
        kind = "independent" if namespace.startswith("independent/") else "joint" if namespace.startswith("conditional_refinement/") else ""
        if issues := response_errors(value, kind):
            item["errors"] = issues
            malformed.append(item)
    return {
        "version": "step2_targeted_recovery_v1", "contract": CONTRACT_VERSION,
        "inventory_scope": "all saved window raw traces plus recorded failed request paths",
        "run_dir": str(run_dir.resolve()), "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest), "requested_windows": len(requested),
        "existing_windows": len(rows), "valid_windows": len(rows) - len(invalid),
        "invalid_windows": invalid, "missing_windows": sorted(requested - rows.keys()),
        "transient_error_caches": transient,
        "restart_recoverable_error_caches": restart_recoverable,
        "terminal_error_caches": terminal,
        "malformed_response_caches": malformed,
        "policy": "Preserve valid responses; archive and retry transient/invalid responses on resume. Terminal content rejections remain unavailable.",
    }


def snapshot(run_dir: Path, out: Path, plan: dict) -> Path:
    destination = out / "snapshot"
    marker = out / "snapshot_complete.json"
    if marker.exists():
        saved = read_json(marker, {})
        if saved.get("manifest_sha256") != plan["manifest_sha256"]:
            raise ValueError("snapshot belongs to a different manifest")
        for relative, digest in saved.get("sha256", {}).items():
            if file_sha256(destination / relative) != digest:
                raise ValueError(f"snapshot modified: {relative}")
        return destination / "run_config.json"
    files = [p for p in run_dir.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl", ".csv"}]
    files += list((run_dir / "records").glob("*.json"))
    for field in ("transient_error_caches", "restart_recoverable_error_caches",
                  "terminal_error_caches", "malformed_response_caches"):
        files += [run_dir / item["path"] for item in plan[field]]
    hashes = {}
    for path in sorted(set(files)):
        relative = path.resolve().relative_to(run_dir.resolve())
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = file_sha256(path)
        if target.exists() and file_sha256(target) != digest:
            raise ValueError(f"refusing to overwrite a different snapshot: {target}")
        if not target.exists():
            shutil.copy2(path, target)
        if file_sha256(target) != digest:
            raise ValueError(f"snapshot verification failed: {target}")
        hashes[relative.as_posix()] = digest
    write_json(marker, {"manifest_sha256": plan["manifest_sha256"], "sha256": hashes})
    return destination / "run_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="Create verified snapshots; no calls or cache deletion")
    args = parser.parse_args()
    plan = inspect_run(args.run_dir, args.manifest)
    if args.prepare:
        plan["snapshot_config"] = str(snapshot(args.run_dir, args.out_dir, plan))
    write_json(args.out_dir / "repair_plan.json", plan)
    print(json.dumps({
        "requested": plan["requested_windows"], "existing": plan["existing_windows"],
        "valid_reusable_windows": plan["valid_windows"], "invalid_windows": len(plan["invalid_windows"]),
        "missing_windows": len(plan["missing_windows"]),
        "transient_error_caches": len(plan["transient_error_caches"]),
        "restart_recoverable_error_caches": len(plan["restart_recoverable_error_caches"]),
        "terminal_error_caches": len(plan["terminal_error_caches"]),
        "malformed_response_caches": len(plan["malformed_response_caches"]),
        "repair_plan": str(args.out_dir / "repair_plan.json"),
        "snapshot_config": plan.get("snapshot_config"), "api_calls": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
