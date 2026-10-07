"""Nine immutable columns, independent C1 validity and paired C2 controls."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..binding import binding_features, validate_binding, validate_proposal
from ..contracts import read_json, semantic_sha256, write_json, write_jsonl
from ..feature_store import _q_direct
from .protocol import immutable, stable_hash, verify_manifest

NAMES = ("m0_margin", "m3a_margin", "m3c_margin", "o_active", "q_direct", "state_uncertainty",
         "c1_Q", "binding_U", "normal_bag_U")
SCHEMA = "v912_nine_features_native_c0_c1_c2_v1"


def c1_quality(c1):
    if not isinstance(c1, dict):
        raise ValueError("missing_or_nonobject_C1")
    validate_proposal(c1)
    complete = c1["scan_complete"] and c1["observation_sufficient"] and not c1["overflow"]
    q = max([e["direct_mechanism_probability"] * e["direct_evidence_quality"] for e in c1["events"]] or [0.0])
    return q if complete else None


def paired_binding(c1, c2):
    q = c1_quality(c1)
    if q is not None and not c1["events"]:
        return {"Q": q, "U": 0.0, "U_bag": 0.0, "U_bag_natural": 0.0, "paired_observed": True,
                "C2_skipped": True, "reason": "complete_empty_C1", "event_trace": []}
    try:
        if not isinstance(c2, dict):
            raise ValueError("missing_C2")
        validate_binding(c1, c2)
        bound = binding_features(c1, c2)
        normal = {r["evidence_id"]: r for r in c2["normal_evidence"]}
        scores = []
        assessed = c2["complete"] and not c2["unmapped_event_ids"]
        for item in c2["event_bindings"]:
            assessed = assessed and item["assessment_complete"]
            for explanation in item["explanations"]:
                refs = explanation["normal_evidence_ids"]
                if explanation["visible_normal_mechanism"] and explanation["mechanism"].strip() and refs and all(ref in normal for ref in refs):
                    scores.append(explanation["bound_support_score"])
        natural_bag = q * (1 - max(scores, default=0.0)) if q is not None and assessed else None
        u = bound["U"]
        paired = q is not None and u is not None and natural_bag is not None
        return {"Q": q, "U": u if paired else None, "U_bag": natural_bag if paired else None,
                "U_bag_natural": natural_bag, "paired_observed": paired,
                "C2_skipped": False, "event_trace": bound.get("event_trace", bound.get("events", [])),
                "binding_trace": bound}
    except (ValueError, KeyError, TypeError) as exc:
        return {"Q": q, "U": None, "U_bag": None, "U_bag_natural": None,
                "paired_observed": False, "reason": str(exc), "C2_skipped": False}


def feature_record(uid, result):
    values = {name: None for name in NAMES}
    lineage = {}
    baseline = result.get("baseline") or {}
    complete = baseline.get("completeness", {})
    for name, method in zip(NAMES[:3], ("independent_direct_nodes", "conditional_rowmax", "conditional_ot_full")):
        good = complete.get("independent") and (name == "m0_margin" or complete.get("joint"))
        val = baseline.get("competitions", {}).get(method, {}).get("margin")
        if good and type(val) in (int, float) and math.isfinite(val):
            values[name] = float(val)
        lineage[name] = {"pointer": "/baseline/competitions/" + method + "/margin", "observed": values[name] is not None}
    state = result.get("C0") or {}
    if state.get("complete") is True:
        for name, key in (("o_active", "current_window_active_occupancy_probability"), ("state_uncertainty", "uncertainty")):
            val = state.get(key)
            if type(val) in (int, float) and math.isfinite(val) and 0 <= val <= 1:
                values[name] = float(val)
        val, observed = _q_direct(state)
        if observed and 0 <= val <= 1:
            values["q_direct"] = float(val)
    for name in NAMES[3:6]:
        lineage[name] = {"pointer": "/C0", "observed": values[name] is not None,
                         "transform": "presence_probability * max(evidence_quality_by_bin)" if name == "q_direct" else "native_v5"}
    trace = {"reason": "missing_or_invalid_C1"}
    try:
        trace = paired_binding(result.get("C1"), result.get("C2"))
        for name, key in zip(NAMES[6:], ("Q", "U", "U_bag")):
            values[name] = trace[key]
    except (ValueError, TypeError, KeyError):
        pass
    for name in NAMES[6:]:
        lineage[name] = {"pointer": "/C1" if name == "c1_Q" else "/C2", "observed": values[name] is not None,
                         "C1_canonical_hash": semantic_sha256(result["C1"]) if result.get("C1") else None}
    for item in lineage.values():
        item["raw_records"] = result.get("raw_request_manifest", [])
        item["parser_contract"] = SCHEMA
    return {"window_uid": uid, "values": values, "lineage": lineage, "binding_trace": trace}


def export_store(out, role, inputs):
    records, hashes = [], {}
    for row in inputs:
        uid = row["window_uid"]
        path = out / "private_acquisition" / role / (uid + ".json")
        result = read_json(path, {})
        verify_manifest(result.get("raw_request_manifest", []))
        hashes[uid] = stable_hash(path) if path.exists() else None
        records.append(feature_record(uid, result))
    identity = semantic_sha256([SCHEMA, hashes])
    root = out / "feature_store" / role / identity
    if not (root / "manifest.json").exists():
        root.mkdir(parents=True, exist_ok=True)
        x = np.array([[r["values"][k] if r["values"][k] is not None else np.nan for k in NAMES] for r in records], dtype=np.float64).reshape(-1, 9)
        np.save(root / "values.npy", x)
        np.save(root / "observed.npy", np.isfinite(x))
        write_jsonl(root / "rows.jsonl", [{"window_uid": r["window_uid"], "row_index": i} for i, r in enumerate(records)])
        write_jsonl(root / "lineage.jsonl", [{"window_uid": r["window_uid"], "columns": r["lineage"]} for r in records])
        write_jsonl(root / "binding_trace.jsonl", [{"window_uid": r["window_uid"], **r["binding_trace"]} for r in records])
        write_json(root / "manifest.json", {"schema": SCHEMA, "names": list(NAMES), "input_hashes": hashes,
                   "human_overlay_loaded": False, "columns_observed": dict(zip(NAMES, np.isfinite(x).sum(axis=0).tolist())),
                   "files": {p.name: stable_hash(p) for p in root.iterdir() if p.is_file()}})
    immutable(out / "feature_store" / role / "CURRENT.json", {"identity": identity})
    return root
