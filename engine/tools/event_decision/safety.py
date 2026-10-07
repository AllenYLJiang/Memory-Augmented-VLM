from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    OfflineNetworkViolation,
    SealViolation,
    file_sha256,
    read_json,
    semantic_sha256,
    write_json,
)


class OfflineGuard:
    """Process-local hard network block installed before business modules run."""

    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        guard = self

        def blocked(*args: Any, **kwargs: Any):
            target = repr(args[:2])
            guard.attempts.append({"target": target, "time": time.time()})
            raise OfflineNetworkViolation(f"offline guard blocked network access: {target}")

        socket.create_connection = blocked  # type: ignore[assignment]
        original_socket = socket.socket

        class GuardedSocket(original_socket):
            def connect(self, address: Any) -> None:
                blocked(address)

            def connect_ex(self, address: Any) -> int:
                blocked(address)
                return 1

        socket.socket = GuardedSocket  # type: ignore[assignment,misc]

    def assert_no_remote_calls(self) -> None:
        if self.attempts:
            raise OfflineNetworkViolation(f"{len(self.attempts)} remote call(s) attempted")


class SealedRunGuard:
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)

    def verify(self, seal_path: Path) -> dict[str, Any]:
        return verify_v8_seal(Path(seal_path).parent.parent)

    def refuse_legacy_write(self, run_root: Path, requested_stage: str) -> None:
        refuse_sealed_write(run_root, requested_stage, self.registry_path)


@contextmanager
def execution_receipt(work_root: Path, stage: str, guard: OfflineGuard):
    started = time.time()
    status = "completed"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row = {
            "version": "offline_execution_receipt_v1",
            "stage": stage,
            "status": status,
            "started_unix": started,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "logical_provider_calls": 0,
            "remote_attempts": len(guard.attempts),
            "network_bytes": 0,
            "error": error,
        }
        path = Path(work_root) / "offline_execution_receipts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


SEAL_INPUTS = (
    "source_disjoint_training_anchors/selected_source_records.jsonl",
    "source_disjoint_training_anchors/calibration_manifest.jsonl",
    "source_disjoint_training_anchors/validation_manifest.jsonl",
    "training_anchor_source/crowd_v5_training_anchors.jsonl",
    "baseline/calibration/ot_window_results.jsonl",
    "baseline/validation/ot_window_results.jsonl",
    "baseline/calibration/run_config.json",
    "baseline/validation/run_config.json",
    "v5_calibration_collect/frozen_pair_results.jsonl",
    "validation_r1/frozen_pair_results.jsonl",
    "calibration/crowd_event_state_v5_calibration.json",
    "gate_r1/progression_gate_v3.json",
)


def _stable_file(path: Path, pause: float = 0.05) -> tuple[int, int]:
    before = path.stat()
    time.sleep(pause)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SealViolation(f"source is actively changing: {path}")
    return after.st_size, after.st_mtime_ns


def _catalog_counts(catalog: dict[str, Any]) -> dict[str, int]:
    return {
        polarity: sum(1 for graph in catalog.get(polarity, []) if graph.get("active", True))
        for polarity in ("abnormal", "normal")
    }


def create_v8_seal(
    legacy_run: Path,
    work_root: Path,
    project_root: Path,
    base_catalog: Path,
    registry_path: Path,
) -> dict[str, Any]:
    legacy_run, work_root = Path(legacy_run).resolve(), Path(work_root).resolve()
    if legacy_run == work_root or legacy_run in work_root.parents:
        raise SealViolation("work_root must be separate from the legacy run")
    gate_path = legacy_run / "gate_r1/progression_gate_v3.json"
    gate = read_json(gate_path, {})
    if gate.get("decision") != "STOP_AFTER_REPLICATE_1":
        raise SealViolation(f"unexpected V8 gate decision: {gate.get('decision')}")
    stop_marker = legacy_run / "gate_r1/STOPPED_BY_GATE"
    if not stop_marker.is_file() or "STOP_AFTER_REPLICATE_1" not in stop_marker.read_text(encoding="utf-8", errors="replace"):
        raise SealViolation(f"completed V8 stop marker is missing or inconsistent: {stop_marker}")
    lock_files = [p for p in legacy_run.rglob("*.lock") if p.is_file() and p.stat().st_size]
    if lock_files:
        raise SealViolation(f"legacy run has active-looking lock files: {lock_files[:3]}")
    rows: list[dict[str, Any]] = []
    for relative in SEAL_INPUTS:
        path = legacy_run / relative
        if not path.is_file():
            raise SealViolation(f"required seal input missing: {path}")
        size, mtime_ns = _stable_file(path)
        rows.append({"relative_path": relative, "bytes": size, "mtime_ns": mtime_ns, "sha256": file_sha256(path)})
    catalog = read_json(base_catalog, {})
    graph_counts = _catalog_counts(catalog)
    seal = {
        "version": "governed_v8_read_only_seal_v1",
        "legacy_run": str(legacy_run),
        "legacy_tag": legacy_run.name,
        "project_root": str(Path(project_root).resolve()),
        "created_unix": time.time(),
        "gate_decision": gate.get("decision"),
        "stop_marker": str(stop_marker),
        "stop_marker_sha256": file_sha256(stop_marker),
        "calibration_id": read_json(legacy_run / "calibration/crowd_event_state_v5_calibration.json", {}).get("calibration_id"),
        "base_catalog": str(Path(base_catalog).resolve()),
        "base_catalog_sha256": file_sha256(base_catalog),
        "base_catalog_semantic_sha256": semantic_sha256(catalog),
        "active_graph_counts": graph_counts,
        "active_graph_total": sum(graph_counts.values()),
        "files_manifest_sha256": semantic_sha256(rows),
        "file_count": len(rows),
        "read_only_contract": True,
    }
    archive = work_root / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    write_json(archive / "v8_seal.json", seal)
    from .contracts import write_jsonl
    write_jsonl(archive / "v8_files.sha256.jsonl", rows)
    code_rows = []
    for path in sorted(Path(project_root).rglob("*")):
        if path.is_file() and path.name != "sealed_run_registry.json" and path.suffix.lower() in {".py", ".sh", ".json", ".yaml", ".yml"} and "runs" not in path.parts and not any("backup" in part.lower() for part in path.parts):
            code_rows.append({"path": str(path.relative_to(project_root)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(archive / "code_manifest.json", {"files": code_rows, "manifest_sha256": semantic_sha256(code_rows)})
    registry = read_json(registry_path, {"version": "sealed_run_registry_v1", "runs": []})
    runs = [r for r in registry.get("runs", []) if Path(str(r.get("run_root", ""))).resolve() != legacy_run]
    runs.append({"run_root": str(legacy_run), "tag": legacy_run.name, "seal_path": str(archive / "v8_seal.json"), "sealed_unix": seal["created_unix"]})
    registry["runs"] = runs
    write_json(registry_path, registry)
    return seal


def verify_v8_seal(work_root: Path) -> dict[str, Any]:
    seal_path = Path(work_root) / "archive/v8_seal.json"
    manifest_path = Path(work_root) / "archive/v8_files.sha256.jsonl"
    seal = read_json(seal_path)
    if not seal:
        raise SealViolation(f"seal missing: {seal_path}")
    from .contracts import iter_jsonl
    mismatches = []
    root = Path(seal["legacy_run"])
    for row in iter_jsonl(manifest_path):
        path = root / row["relative_path"]
        actual = file_sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"path": str(path), "expected": row["sha256"], "actual": actual})
    catalog_path = Path(seal["base_catalog"])
    if not catalog_path.is_file() or file_sha256(catalog_path) != seal["base_catalog_sha256"]:
        mismatches.append({"path": str(catalog_path), "reason": "catalog_hash_changed"})
    stop_marker = Path(seal.get("stop_marker", ""))
    if not stop_marker.is_file() or file_sha256(stop_marker) != seal.get("stop_marker_sha256"):
        mismatches.append({"path": str(stop_marker), "reason": "stop_marker_changed"})
    code_manifest = read_json(Path(work_root) / "archive/code_manifest.json", {})
    project_root = Path(seal.get("project_root", "."))
    for row in code_manifest.get("files", []):
        path = project_root / row["path"]
        actual = file_sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"path": str(path), "reason": "sealed_code_changed", "expected": row["sha256"], "actual": actual})
    result = {"verified": not mismatches, "mismatches": mismatches, "checked_unix": time.time()}
    write_json(Path(work_root) / "archive/seal_verification.json", result)
    if mismatches:
        raise SealViolation(f"sealed V8 inputs changed ({len(mismatches)} mismatch(es))")
    return result


def refuse_sealed_write(run_root: Path, requested_stage: str, registry_path: Path, force: bool = False) -> None:
    registry = read_json(registry_path, {"runs": []})
    target = Path(run_root).resolve()
    for row in registry.get("runs", []):
        if Path(str(row.get("run_root", ""))).resolve() == target:
            raise SealViolation(
                f"run is sealed read-only: {target}; requested stage={requested_stage}. "
                "Create a new TAG. Sealed Step 8 cannot be forced."
            )
