from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters import load_baseline_snapshot, load_state_snapshot
from .contracts import FEATURE_NAMES, FEATURE_SCHEMA_ID, canonical_window, file_sha256, iter_jsonl, json_pointer, semantic_sha256, write_json, write_jsonl


FEATURE_POINTERS = {
    "m0_margin": "/competitions/independent_direct_nodes/margin",
    "m3a_margin": "/competitions/conditional_rowmax/margin",
    "m3c_margin": "/competitions/conditional_ot_full/margin",
    "o_active": "/candidate_event_state/current_window_active_occupancy_probability",
    "q_direct": "/candidate_event_state/phase_nodes/active/direct_physical_escalation_mechanism",
    "state_uncertainty": "/candidate_event_state/uncertainty",
    "normal_bound_probability": "unavailable_without_structured_binding",
    "unexplained_direct_evidence": "unavailable_without_structured_binding",
}


def _finite(value: Any) -> tuple[float, bool]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan, False
    return (number, True) if math.isfinite(number) else (math.nan, False)


def _q_direct(state: Mapping[str, Any]) -> tuple[float, bool]:
    node = json_pointer(state, "/phase_nodes/active/direct_physical_escalation_mechanism")
    if not isinstance(node, Mapping):
        return math.nan, False
    p, present = _finite(node.get("presence_probability", node.get("conditional_presence")))
    quality = node.get("evidence_quality_by_bin")
    if not present or not isinstance(quality, list) or not quality:
        return math.nan, False
    finite_quality = [float(v) for v in quality if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return (p * max(finite_quality), True) if finite_quality else (math.nan, False)


def _evidence_uid(record: Mapping[str, Any], input_contract: str) -> tuple[str, list[dict[str, Any]]]:
    evidence = record.get("evidence", {}) if isinstance(record.get("evidence"), Mapping) else {}
    indices = evidence.get("frame_indices", [])
    image_rows = []
    hashes = []
    for raw in evidence.get("image_paths", []) if isinstance(evidence.get("image_paths"), list) else []:
        path = Path(str(raw))
        digest = file_sha256(path) if path.is_file() else None
        image_rows.append({"path": str(path), "sha256": digest, "exists": path.is_file()})
        hashes.append(digest or f"MISSING:{path}")
    return semantic_sha256({"indices": indices, "image_hashes": hashes, "contract": input_contract}), image_rows


class FrozenFeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, contract_id: str, row_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        root = self.root / contract_id
        values, observed = np.load(root / "values.npy"), np.load(root / "observed.npy")
        if row_ids is None:
            return values, observed
        rows = {r["window_uid"]: int(r["row_index"]) for r in iter_jsonl(root / "rows.jsonl")}
        idx = [rows[row] for row in row_ids]
        return values[idx], observed[idx]

    def import_snapshot(self, inventory: Mapping[str, Any], *, cache_only: bool = True) -> dict[str, Any]:
        legacy_run = inventory.get("legacy_run")
        work_root = inventory.get("work_root")
        if not legacy_run or not work_root:
            raise ValueError("inventory must provide legacy_run and work_root")
        return export_features(Path(str(legacy_run)), Path(str(work_root)), "source_frozen", cache_only)

    def plan_missing(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(r, remote_execution_authorized=False) for r in requests]


def export_features(legacy_run: Path, work_root: Path, baseline_origin: str = "source_frozen", cache_only: bool = True) -> dict[str, Any]:
    if baseline_origin != "source_frozen":
        raise ValueError("V9 primary store accepts source_frozen baseline only")
    if not cache_only:
        raise ValueError("semantic extraction is not authorized; use --cache-only")
    baseline, _ = load_baseline_snapshot(legacy_run)
    state_rows, _ = load_state_snapshot(legacy_run)
    input_hashes = {
        "baseline_calibration": file_sha256(Path(legacy_run) / "baseline/calibration/ot_window_results.jsonl"),
        "baseline_validation": file_sha256(Path(legacy_run) / "baseline/validation/ot_window_results.jsonl"),
        "state_calibration": file_sha256(Path(legacy_run) / "v5_calibration_collect/frozen_pair_results.jsonl"),
        "state_validation": file_sha256(Path(legacy_run) / "validation_r1/frozen_pair_results.jsonl"),
    }
    frozen_call_contract = semantic_sha256({"origin": baseline_origin, "features": FEATURE_NAMES, "input_hashes": input_hashes})
    contract_id = f"{FEATURE_SCHEMA_ID}_{frozen_call_contract[:16]}"
    out = Path(work_root) / "feature_store" / contract_id
    out.mkdir(parents=True, exist_ok=True)
    values = np.full((len(baseline), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    rows, lineage, auxiliary, missing = [], [], [], []
    for index, base in enumerate(baseline):
        uid = base["canonical_window_uid"]
        state_wrap = state_rows.get(uid)
        raw_state = state_wrap["state"] if state_wrap else None
        # Parser/provider failures sometimes retain a shaped object with fallback
        # zeros. Those are unavailable observations, not measured zero evidence.
        state = raw_state if isinstance(raw_state, Mapping) and raw_state.get("complete") is True else None
        mapping = {
            "m0_margin": json_pointer(base, "/competitions/independent_direct_nodes/margin"),
            "m3a_margin": json_pointer(base, "/competitions/conditional_rowmax/margin"),
            "m3c_margin": json_pointer(base, "/competitions/conditional_ot_full/margin"),
            "o_active": json_pointer(state or {}, "/current_window_active_occupancy_probability"),
            "state_uncertainty": json_pointer(state or {}, "/uncertainty"),
        }
        q_value, q_seen = _q_direct(state or {})
        for col, name in enumerate(FEATURE_NAMES):
            if name == "q_direct":
                number, seen = q_value, q_seen
            elif name in {"normal_bound_probability", "unexplained_direct_evidence"}:
                number, seen = math.nan, False
            else:
                number, seen = _finite(mapping.get(name))
            values[index, col], observed[index, col] = number, seen
            lineage.append({"window_uid": uid, "feature": name, "observed": seen, "source_pointer": FEATURE_POINTERS[name], "source_origin": "state_frozen" if col >= 3 else "baseline_source_frozen"})
            if not seen:
                reason = "binding_not_identifiable_from_current_cache" if col >= 6 else "semantic_state_not_available" if col >= 3 else "baseline_field_missing_or_invalid"
                missing.append({"window_uid": uid, "feature": name, "reason": reason, "cache_only": True, "remote_execution_authorized": False})
        evidence_uid, image_rows = _evidence_uid(base, frozen_call_contract)
        window = canonical_window(base)
        rows.append({"row_index": index, "window_uid": uid, "evidence_uid": evidence_uid, "segment_key": base.get("segment_key"), "video_id": window.video_id, "start_frame": window.start_frame, "end_frame_exclusive": window.end_frame_exclusive, "source_group": base.get("source_group"), "historical_split": base.get("historical_split"), "baseline_origin": baseline_origin})
        auxiliary.append({"window_uid": uid, "legacy_y_true": base.get("y_true"), "label_codes": base.get("gt", {}).get("label_codes", []), "o_normal_global_diagnostic": (state or {}).get("current_window_normal_confound_probability") if state else None, "state_available": raw_state is not None, "state_complete": bool((raw_state or {}).get("complete")), "evidence_images": image_rows, "temporal_bins": (base.get("evidence") or {}).get("temporal_bins"), "capacity_slots_per_bin": 2})
    np.save(out / "values.npy", values)
    np.save(out / "observed.npy", observed)
    write_jsonl(out / "rows.jsonl", rows)
    write_jsonl(out / "feature_lineage.jsonl", lineage)
    write_jsonl(out / "auxiliary_traces.jsonl", auxiliary)
    write_jsonl(out / "missing_requests.jsonl", missing)
    schema = {"version": "event_feature_schema_v1", "schema_id": FEATURE_SCHEMA_ID, "contract_id": contract_id, "frozen_call_contract_id": frozen_call_contract, "shape": list(values.shape), "dtype": "float32", "features": [{"index": i, "name": n, "pointer": FEATURE_POINTERS[n]} for i, n in enumerate(FEATURE_NAMES)], "baseline_origin": baseline_origin, "missing_semantics": "NaN_plus_observed_false", "T0_T7_semantics": "eight_sampled_temporal_bins_not_sixteen_frames", "input_hashes": input_hashes}
    write_json(out / "schema.json", schema)
    availability = {name: {"observed": int(observed[:, i].sum()), "total": len(rows), "fraction": float(observed[:, i].mean()) if len(rows) else 0.0} for i, name in enumerate(FEATURE_NAMES)}
    write_json(out / "availability.json", {"features": availability, "F2_event_bound_status": "BINDING_NOT_IDENTIFIABLE_FROM_CURRENT_CACHE"})
    write_json(Path(work_root) / "feature_store/CURRENT.json", {"contract_id": contract_id, "path": str(out)})
    return {"contract_id": contract_id, "rows": len(rows), "dimensions": len(FEATURE_NAMES), "missing_values": int((~observed).sum()), "path": str(out)}
