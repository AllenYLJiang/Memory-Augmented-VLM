"""Immutable inputs, explicit transitions and budget-independent local operations."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import file_sha256, read_json, semantic_sha256, write_json, write_jsonl
from ..role_scoped import portable
from . import VERSION

STATES = ["DRAFT", "LEGACY_SEALED", "SOURCE_CAPACITY_VERIFIED", "ROLES_RESERVED",
          "DEV_SMOKE_COMPLETE", "PROTOCOL_LOCKED", "ADAPTATION_FEATURES_FROZEN",
          "MODELS_FROZEN", "LOCKED_FEATURES_COMPLETE_OR_DECLARED_MISSING",
          "PREDICTIONS_COMMITTED", "EVALUATION_UNBLINDED", "REPORTED", "BRANCH_CLOSED"]


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_hash(path):
    path = Path(path)
    before = path.stat()
    value = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("WAITING_FOR_STABLE_SOURCE: " + str(path))
    return value


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen artifact changed; preserve TAG: " + str(path))
    else:
        write_json(path, value)
    return value


@contextmanager
def run_lock(out):
    path = Path(out) / ".operation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("WAITING_FOR_OPERATION_LOCK: inspect owner before removing " + str(path)) from exc
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "time": now()}).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def advance(out, state):
    current = read_json(Path(out) / "state.json", {"state": "DRAFT"})
    if STATES.index(state) > STATES.index(current["state"]):
        write_json(Path(out) / "state.json", {"state": state, "at": now(),
                   "deployment_authorized": False})


def at_least(out, state):
    return STATES.index(read_json(Path(out) / "state.json", {"state": "DRAFT"})["state"]) >= STATES.index(state)


def verify_manifest(manifest):
    for item in manifest:
        path = portable(item["path"])
        if not path.is_file() or stable_hash(path) != item["sha256"]:
            raise ValueError("SEALED_INPUT_CHANGED: " + str(path))


def input_inventory(project, config):
    paths = set()
    for rel in (config["graph_catalog"], config["phase_catalog"]):
        paths.add((project / rel).resolve())
    # Actual results, including failures and human layers, are reference-only.
    prefixes = tuple("governed_v" + v for v in ("91_", "92_", "93_", "94_", "95_", "96_", "97_", "98_", "99_", "910_", "911_"))
    names = {"selection.json", "protocol.json", "integrity.json", "summary.json",
             "review_import_summary.json", "model.json", "fit_summary.json", "fit_manifest.json",
             "numeric_certificate.json", "certificate.json", "source_exposure_evidence.jsonl",
             "reconciled_history.json", "preflight_report.json", "manifest.json", "results.jsonl",
             "predictions.jsonl", "failed_requests.jsonl", "source_manifest.json"}
    for root in sorted((project / "runs").iterdir()):
        if not root.is_dir() or not root.name.startswith(prefixes):
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {"frames", "media", "__pycache__", "cache", "raw_cache"}]
            paths.update(Path(directory) / n for n in files if n in names or
                         Path(n).suffix in {".json", ".jsonl", ".npy", ".npz"})
    paths.update(project.glob("docs/v99_diagnostic_review*.json"))
    paths.update(project.glob("docs/GOVERNED_V911*.md"))
    paths.add(project / "docs/GOVERNED_V912_B1B4_MINIMAL_EFFECT_TRIAL_CODEX_GUIDE_20260915.md")
    # Seal all scoring/parser dependencies, not just the new wrapper.
    paths.update((project / "tools").rglob("*.py"))
    paths.add(project.parent / "Previous_code/structural_vlm_binary_v40_15_qwen36_highres_graph_vs_node_revision/src/structural_vlm_binary/vlm/dashscope_backend.py")
    return [{"path": str(p.resolve()), "sha256": stable_hash(p), "bytes": p.stat().st_size,
             "use": "reference_only_never_new_features_or_labels" if "runs" in p.parts else "implementation_reference"}
            for p in sorted(paths) if p.is_file()]


def initialize(project, out, config):
    if config.get("version") != VERSION or config.get("adaptation_label_policy") != "weak_only":
        raise ValueError("Unknown protocol or supervision policy")
    if config.get("deployment_authorized") or config.get("semantic_retries") != 0:
        raise ValueError("Deployment / semantic retries are not authorized")
    from graph_catalog import read_catalog_json
    catalog = read_catalog_json(project / config["graph_catalog"])
    if len(catalog) != 13:
        raise ValueError("This registered trial requires the fixed governed 13-graph catalog")
    immutable(out / "seal/graph_objects.json", {key: semantic_sha256(asdict(graph)) for key, graph in catalog.items()})
    immutable(out / "protocol/config.json", config)
    seal = read_json(out / "seal/legacy_inventory.json")
    if seal is None:
        seal = input_inventory(project, config)
        write_json(out / "seal/legacy_inventory.json", seal)
        immutable(out / "seal/identity.json", {"inventory_sha256": stable_hash(out / "seal/legacy_inventory.json")})
    else:
        if stable_hash(out / "seal/legacy_inventory.json") != read_json(out / "seal/identity.json")["inventory_sha256"]:
            raise ValueError("seal inventory changed")
        verify_manifest(seal)
    advance(out, "LEGACY_SEALED")
    return seal


def verify_frozen(out):
    frozen = read_json(out / "protocol/frozen.json")
    if not frozen:
        raise ValueError("WAITING_FOR_PROTOCOL_LOCK")
    verify_manifest(frozen["files"])
    return frozen


def freeze(out):
    if not at_least(out, "DEV_SMOKE_COMPLETE"):
        raise ValueError("WAITING_FOR_DEV_SMOKE")
    if not read_json(out / "smoke/summary.json", {}).get("core_ready_for_lock"):
        raise ValueError("WAITING_FOR_SMOKE_CONTRACT_REPAIR: missing M0/C0; inspect saved raw responses offline, no semantic retries")
    approval = read_json(out / "protocol/review_approval.json", {})
    required = ("history_scope_checked", "source_aliases_checked", "enrollment_checked",
                "full_clip_media_checked", "label_scope_checked", "smoke_checked",
                "weak_labels_not_gold", "locked_outcomes_not_viewed")
    enrollment_hash = stable_hash(out / "enrollment/windows.jsonl")
    smoke_hash = stable_hash(out / "smoke/summary.json")
    if (not approval.get("reviewer") or any(approval.get(k) is not True for k in required)
            or approval.get("enrollment_sha256") != enrollment_hash or approval.get("smoke_sha256") != smoke_hash):
        raise ValueError("WAITING_FOR_ENROLLMENT_MEDIA_SCOPE_REVIEW: protocol/review_approval.json")
    files = ["protocol/config.json", "protocol/review_approval.json", "seal/identity.json",
             "seal/legacy_inventory.json", "enrollment/windows.jsonl", "enrollment/feature_inputs.jsonl",
             "enrollment/adaptation_labels.jsonl", "enrollment/history_scope.json", "enrollment/role_map.jsonl", "enrollment/smoke_manifest.jsonl", "smoke/summary.json",
             "enrollment/negative_anchor_snapshot.json",
             "media/manifest.json", "review/private_map.json", "review/public_manifest.json"]
    value = {"version": VERSION, "files": [{"path": str((out / x).resolve()), "sha256": stable_hash(out / x)} for x in files],
             "adaptation_labels": "weak_only", "human_layers": "evaluation_or_audit_only", "deployment_authorized": False}
    immutable(out / "protocol/frozen.json", value)
    advance(out, "PROTOCOL_LOCKED")
    return value
