"""Opt-in, exact-file runtime upgrade; frozen scientific configuration stays intact."""
from pathlib import Path

from ..contracts import file_sha256, read_json, semantic_sha256
from ..b1b4_trial.protocol import immutable, now


PATCH_ID = "v919_terminal_refusal_v2"
PREDECESSOR_ID = "v919_transport_retry_v1"
PATCH_PATHS = {
    "run_effectiveness_v919.sh",
    "tools/effectiveness_v919_cli.py",
    "tools/event_decision/effectiveness_v919/acquisition.py",
    "tools/event_decision/effectiveness_v919/pipeline.py",
    "tools/event_decision/effectiveness_v919/transport_patch.py",
}


def compatible_change(project, out, config, current, patch_id=PATCH_ID):
    if patch_id not in (PATCH_ID, PREDECESSOR_ID):
        raise ValueError("Unknown runtime patch")
    frozen = read_json(out / "code_inventory.json")
    if not frozen or semantic_sha256(frozen) != config["code_sha256"]:
        raise ValueError("Original frozen code inventory does not match protocol")
    path = project / "runtime_patches" / (patch_id + ".json")
    manifest = read_json(path)
    if not manifest or manifest.get("patch_id") != patch_id or set(manifest["changes"]) != PATCH_PATHS:
        raise ValueError("Missing/invalid exact transport patch manifest")
    changed = {k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k)}
    if changed != PATCH_PATHS:
        raise ValueError("Upgrade refused: code changes are not exactly the audited transport patch")
    for rel, hashes in manifest["changes"].items():
        if hashes != {"before": frozen.get(rel), "after": current.get(rel)}:
            raise ValueError("Upgrade refused: unexpected before/after hash for " + rel)
    if file_sha256(out / "graph_catalog.json") != config["graph_sha256"]:
        raise ValueError("Frozen graph library changed")
    artifacts = {"protocol.json": file_sha256(out / "protocol.json"),
                 "code_inventory.json": file_sha256(out / "code_inventory.json"),
                 "graph_catalog.json": file_sha256(out / "graph_catalog.json")}
    for phase in ("pilot", "dense"):
        plan = read_json(out / phase / "plan.json")
        if not plan:
            continue
        if plan["protocol_sha256"] != artifacts["protocol.json"] or file_sha256(out / phase / "inputs.jsonl") != plan["manifest_sha256"]:
            raise ValueError("Frozen plan/protocol/inputs changed")
        artifacts[phase + "/plan.json"] = file_sha256(out / phase / "plan.json")
        artifacts[phase + "/inputs.jsonl"] = plan["manifest_sha256"]
        if plan.get("labels_sha256"):
            if file_sha256(out / "private/pilot_labels.jsonl") != plan["labels_sha256"]:
                raise ValueError("Frozen labels changed")
            artifacts["private/pilot_labels.jsonl"] = plan["labels_sha256"]
    result = {"patch_id": patch_id, "patch_manifest_sha256": file_sha256(path),
            "frozen_code_sha256": config["code_sha256"], "runtime_code_sha256": semantic_sha256(current),
            "changes": manifest["changes"], "frozen_artifacts": artifacts,
            "scientific_configuration_changed": False, "old_request_identity_preserved": True}
    if patch_id == PATCH_ID:
        previous_receipt = out / "runtime_upgrades" / (PREDECESSOR_ID + ".json")
        result["predecessor_upgrade_sha256"] = None
        if previous_receipt.exists():
            previous_path = project / "runtime_patches" / (PREDECESSOR_ID + ".json")
            if file_sha256(previous_path) != manifest.get("predecessor_manifest_sha256"):
                raise ValueError("Previous runtime manifest changed")
            previous = read_json(previous_path)
            intermediate = dict(frozen)
            for rel, hashes in previous["changes"].items():
                if frozen.get(rel) != hashes["before"]:
                    raise ValueError("Previous runtime patch origin changed")
                intermediate[rel] = hashes["after"]
            verify_upgrade(project, out, config, intermediate, patch_id=PREDECESSOR_ID)
            result["predecessor_upgrade_sha256"] = file_sha256(previous_receipt)
        result["refusal_policy"] = "terminal_missing_evidence_no_requery_no_window_drop"
    return result


def verify_upgrade(project, out, config, current, patch_id=PATCH_ID):
    expected = compatible_change(project, out, config, current, patch_id=patch_id)
    receipt = read_json(out / "runtime_upgrades" / (patch_id + ".json"))
    if not receipt or not receipt.get("approved_by"):
        raise ValueError("Frozen code differs only by runtime patch; explicitly run STAGE=2 ACTION=upgrade APPROVED_BY=... first")
    # A dense plan may be created after the pilot transport upgrade.
    for key, value in expected.items():
        if key == "frozen_artifacts":
            if any(value.get(k) != h for k, h in receipt[key].items()):
                raise ValueError("Runtime upgrade frozen artifact identity changed")
        elif receipt.get(key) != value:
            raise ValueError("Runtime upgrade receipt does not match current code: " + key)
    snapshot = out / receipt["source_snapshot_file"]
    if file_sha256(snapshot) != receipt["source_snapshot_sha256"]:
        raise ValueError("Pre-upgrade source snapshot changed")
    return receipt


def accept_upgrade(project, out, approved_by, current, patch_id=PATCH_ID):
    if not approved_by.strip():
        raise ValueError("Explicit APPROVED_BY required for the offline transport upgrade")
    config = read_json(out / "protocol.json")
    if not config:
        raise ValueError("No frozen experiment to upgrade")
    receipt_path = out / "runtime_upgrades" / (patch_id + ".json")
    if receipt_path.exists():
        return verify_upgrade(project, out, config, current, patch_id=patch_id)
    expected = compatible_change(project, out, config, current, patch_id=patch_id)
    # Validate existing response envelopes before certifying their reuse. No API/decoding.
    for path in (out / "cache").glob("*.json"):
        envelope = read_json(path)
        body = envelope.get("body", {})
        if (semantic_sha256(body) != envelope.get("sha256") or semantic_sha256(body.get("identity")) != path.stem or
                body.get("identity", {}).get("configuration") != config):
            raise ValueError("Existing raw request identity/hash mismatch: " + path.name)
    for path in out.glob("*/results/*.json"):
        result = read_json(path)
        if result.get("record_sha256") != semantic_sha256({k: v for k, v in result.items() if k != "record_sha256"}):
            raise ValueError("Existing window result changed: " + path.name)
        if any(file_sha256(out / "cache" / (k + ".json")) != h for k, h in result["cache_sha256"].items()):
            raise ValueError("Existing window evidence changed: " + path.name)
        from .acquisition import verify_refusals
        verify_refusals(out, result)
    snapshot = {p.relative_to(out).as_posix(): file_sha256(p) for p in out.rglob("*")
                if p.is_file() and p.name != ".operation.lock" and "runtime_upgrades" not in p.relative_to(out).parts}
    relative = "runtime_upgrades/" + patch_id + "_source_files.json"
    immutable(out / relative, snapshot)
    receipt = {**expected, "approved_by": approved_by, "at": now(), "source_files": len(snapshot),
               "source_snapshot_file": relative, "source_snapshot_sha256": file_sha256(out / relative),
               "paid_execution_authorized": False, "new_API_calls": 0}
    immutable(receipt_path, receipt)
    return receipt
