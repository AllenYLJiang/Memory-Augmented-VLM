"""Validate frame coordinates before any truncation, scoring, or frozen reuse."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping


CONTRACT_VERSION = "eight_frame_temporal_contract_v1"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def node_errors(node: Any, prefix: str = "node") -> list[str]:
    if not isinstance(node, Mapping):
        return [f"{prefix}: missing node object"]
    errors = []
    for field in ("presence_probability", "null_probability", "uncertainty"):
        if field not in node and field != "presence_probability":
            continue
        value = node.get(field)
        if not _number(value) or not 0 <= value <= 1:
            errors.append(f"{prefix}.{field}: expected finite probability in [0,1]")
    for field in ("location_distribution_given_present", "evidence_quality_by_bin"):
        values = node.get(field)
        if not isinstance(values, list) or len(values) != 8:
            length = len(values) if isinstance(values, list) else None
            errors.append(f"{prefix}.{field}: expected 8 frame bins, got {length}")
        elif any(not _number(v) or not 0 <= v <= 1 for v in values):
            errors.append(f"{prefix}.{field}: expected finite values in [0,1]")
        elif field.startswith("location") and sum(values) <= 0:
            errors.append(f"{prefix}.{field}: zero total mass")
    best = node.get("best_bin")
    if best is not None and (type(best) is not int or not 0 <= best < 8):
        errors.append(f"{prefix}.best_bin: expected null or integer 0..7, got {best}")
    return errors


def response_errors(value: Any, kind: str, node_ids: list[str] | None = None) -> list[str]:
    if kind not in {"independent", "joint"}:
        return []
    if not isinstance(value, Mapping):
        return ["response: expected JSON object"]
    if kind == "independent":
        return node_errors(value)
    nodes = value.get("nodes")
    if not isinstance(nodes, Mapping) or not nodes:
        return ["response.nodes: missing node objects"]
    errors = []
    for key in node_ids if node_ids is not None else nodes:
        errors.extend(node_errors(nodes.get(key), str(key)))
    coherence = value.get("graph_coherence")
    if not _number(coherence) or not 0 <= coherence <= 1:
        errors.append("graph_coherence: expected finite probability in [0,1]")
    span = value.get("episode_span_bins", [])
    if not isinstance(span, list) or any(type(v) is not int or not 0 <= v < 8 for v in span):
        errors.append("episode_span_bins: expected frame indices 0..7")
    return errors


def raw_object(trace: Mapping[str, Any]) -> dict:
    if isinstance(trace.get("parsed"), dict):
        return trace["parsed"]
    raw = str(trace.get("raw", "")).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("raw response is not a JSON object")
    return value


def window_errors(row: Mapping[str, Any]) -> list[str]:
    """Inspect raw responses, never the already-truncated visualization fields."""
    errors = []
    for field, kind in (("independent_node_calls", "independent"), ("joint_graph_calls", "joint")):
        traces = row.get(field)
        if not isinstance(traces, Mapping) or not traces:
            errors.append(f"{field}: missing traces")
            continue
        for key, trace in traces.items():
            try:
                value = raw_object(trace)
                node_ids = [n["anonymous_id"] for n in trace["nodes"].values()] if kind == "joint" else None
                issues = response_errors(value, kind, node_ids)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                issues = [f"cannot validate original response: {exc}"]
            errors.extend(f"{field}/{key}: {issue}" for issue in issues)
    candidates = row.get("graph_candidates", {})
    selected = set(candidates.get("selected_abnormal", [])) | set(candidates.get("selected_normal", []))
    missing = selected - set(row.get("joint_graph_calls", {}))
    if missing:
        errors.append(f"missing selected graph traces: {sorted(missing)}")
    flags = row.get("completeness", {})
    if flags.get("independent") is not True or flags.get("joint") is not True:
        errors.append("window completeness flags are not both true")
    return errors
