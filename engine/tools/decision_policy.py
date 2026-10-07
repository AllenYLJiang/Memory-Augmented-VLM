#!/usr/bin/env python3
"""Score extraction, calibration, and decision policies for cached OT records.

The module is deliberately API-free. It separates representational scores from an
operating-point decision so existing VLM traces can be recalibrated without inference.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_SCHEMA_VERSION = "decision_features_v1"
M0 = "independent_direct_nodes"
M2 = "unary_ot"
M3 = "conditional_ot_full"
POLICIES = {
    "legacy_graph", "weighted_fusion", "gated_normal_veto", "asymmetric", "ternary",
    "crowd_state_residual_v1",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _mean(values: Any, default: float = 1.0) -> float:
    if isinstance(values, Mapping):
        values = list(values.values())
    if not isinstance(values, (list, tuple)):
        return _float(values, default)
    numbers = [_float(value, default) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else float(default)


@dataclass(frozen=True)
class DecisionFeatures:
    node_margin: float
    unary_ot_margin: float
    conditional_graph_margin: float
    best_abnormal_score: float
    best_normal_score: float
    abnormal_null_mass: float
    normal_null_mass: float
    abnormal_uncertainty: float
    normal_uncertainty: float
    graph_parse_complete: bool
    crowd_state_log_odds: float = 0.0
    crowd_state_uncertainty: float = 1.0
    crowd_state_complete: bool = False
    crowd_target_exposed: bool = False
    crowd_observation_adequate: bool = False

    def to_dict(self) -> dict:
        value = asdict(self)
        value["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        return value


def _winner_diagnostics(record: Mapping[str, Any], graph_key: str) -> tuple[float, float]:
    graph = record.get("graph_results", {}).get(M3, {}).get(graph_key, {})
    diagnostics = graph.get("diagnostics", {}) if isinstance(graph, Mapping) else {}
    null_mass = _mean(diagnostics.get("null_mass_by_node", diagnostics.get("null_mass", [])), 1.0)
    joint = record.get("joint_graph_calls", {}).get(graph_key, {})
    nodes = joint.get("nodes", {}) if isinstance(joint, Mapping) else {}
    uncertainties = [
        value.get("uncertainty", 1.0) for value in nodes.values() if isinstance(value, Mapping)
    ] if isinstance(nodes, Mapping) else []
    uncertainty = _mean(uncertainties, 1.0)
    return max(0.0, min(1.0, null_mass)), max(0.0, min(1.0, uncertainty))


def extract_decision_features(record: Mapping[str, Any]) -> DecisionFeatures:
    competitions = record.get("competitions", {})
    m0 = competitions.get(M0, {})
    m2 = competitions.get(M2, {})
    m3 = competitions.get(M3, {})
    abnormal_key = str(m3.get("best_abnormal_graph", "NONE"))
    normal_key = str(m3.get("best_normal_graph", "NONE"))
    abnormal_null, abnormal_uncertainty = _winner_diagnostics(record, abnormal_key)
    normal_null, normal_uncertainty = _winner_diagnostics(record, normal_key)
    completeness = record.get("completeness", {})
    candidates = record.get("graph_candidates", {})
    complete = bool(completeness.get("independent", False) and completeness.get("joint", False))
    if isinstance(candidates, Mapping) and "complete" in candidates:
        complete = complete and bool(candidates.get("complete"))
    return DecisionFeatures(
        node_margin=_float(m0.get("margin")),
        unary_ot_margin=_float(m2.get("margin")),
        conditional_graph_margin=_float(m3.get("margin")),
        best_abnormal_score=_float(m3.get("best_abnormal_score", m3.get("best_abnormal_graph_score"))),
        best_normal_score=_float(m3.get("best_normal_score", m3.get("best_normal_graph_score"))),
        abnormal_null_mass=abnormal_null,
        normal_null_mass=normal_null,
        abnormal_uncertainty=abnormal_uncertainty,
        normal_uncertainty=normal_uncertainty,
        graph_parse_complete=complete,
    )


def normal_graph_reliability(features: DecisionFeatures, config: Mapping[str, Any]) -> float:
    max_null = _float(config.get("normal_max_null_mass", 0.35), 0.35)
    max_uncertainty = _float(config.get("normal_max_uncertainty", 0.35), 0.35)
    min_score = _float(config.get("normal_min_score", 0.55), 0.55)
    if (
        not features.graph_parse_complete
        or features.best_normal_score < min_score
        or features.normal_null_mass > max_null
        or features.normal_uncertainty > max_uncertainty
    ):
        return 0.0
    return (
        features.best_normal_score
        * (1.0 - features.normal_null_mass)
        * (1.0 - features.normal_uncertainty)
    )


def policy_score(features: DecisionFeatures, config: Mapping[str, Any]) -> tuple[float, dict]:
    policy = str(config.get("policy", "legacy_graph"))
    if policy not in POLICIES:
        raise ValueError(f"unknown decision policy: {policy}")
    details: dict[str, Any] = {}
    if policy == "legacy_graph":
        return features.conditional_graph_margin, details
    if policy == "crowd_state_residual_v1":
        uncertainty_max = max(0.0, min(1.0, _float(config.get("uncertainty_max", 0.8), 0.8)))
        noop_reasons = []
        if not features.crowd_state_complete:
            noop_reasons.append("incomplete_state_output")
        if not features.crowd_target_exposed:
            noop_reasons.append("crowd_target_not_exposed")
        if not features.crowd_observation_adequate:
            noop_reasons.append("observation_inadequate")
        if features.crowd_state_uncertainty > uncertainty_max:
            noop_reasons.append("uncertainty_above_limit")
        beta = max(0.0, _float(config.get("beta", 0.0)))
        bound = max(0.0, _float(config.get("residual_bound", 0.0)))
        residual = 0.0 if noop_reasons else max(
            -bound, min(bound, beta * features.crowd_state_log_odds)
        )
        details.update({
            "beta": beta, "residual_bound": bound, "state_residual": residual,
            "noop_reasons": noop_reasons, "repeated_gate_count": 0,
        })
        return features.conditional_graph_margin + residual, details
    alpha = max(0.0, min(1.0, _float(config.get("alpha", 0.5), 0.5)))
    fused = alpha * features.node_margin + (1.0 - alpha) * features.conditional_graph_margin
    if policy in {"weighted_fusion", "ternary"}:
        details["alpha"] = alpha
        return fused, details
    reliability = normal_graph_reliability(features, config)
    beta = max(0.0, _float(config.get("beta", 0.5), 0.5))
    details.update({"alpha": alpha, "beta": beta, "normal_reliability": reliability})
    if policy == "gated_normal_veto":
        return features.node_margin - beta * reliability, details
    abnormal_weight = max(0.0, _float(config.get("abnormal_weight", 0.5), 0.5))
    abnormal_support = max(0.0, features.conditional_graph_margin) * (
        1.0 - features.abnormal_null_mass
    ) * (1.0 - features.abnormal_uncertainty)
    details.update({"abnormal_weight": abnormal_weight, "abnormal_support": abnormal_support})
    return features.node_margin + abnormal_weight * abnormal_support - beta * reliability, details


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def calibrated_probability(score: float, config: Mapping[str, Any]) -> float:
    if "platt_a" in config or "platt_b" in config:
        return sigmoid(_float(config.get("platt_a", 1.0), 1.0) * score + _float(config.get("platt_b", 0.0)))
    # Raw margins are centered at zero; this fallback is for diagnostics only.
    return sigmoid(score)


def apply_decision_policy(features: DecisionFeatures, config: Mapping[str, Any] | None = None) -> dict:
    config = dict(config or {"policy": "legacy_graph"})
    policy = str(config.get("policy", "legacy_graph"))
    score, details = policy_score(features, config)
    score_based = policy in {"legacy_graph", "crowd_state_residual_v1"}
    threshold = _float(config.get("threshold", 0.03 if score_based else 0.5))
    probability = calibrated_probability(score, config)
    unresolved_margin = max(0.0, _float(config.get("unresolved_margin", threshold if score_based else 0.05)))
    if score_based:
        unresolved = abs(score) <= threshold
        decision = "unresolved" if unresolved else ("abnormal" if score > threshold else "normal")
        y_pred = 0 if unresolved else int(score > threshold)
        decision_variable = "score"
    else:
        unresolved = policy == "ternary" and abs(probability - threshold) <= unresolved_margin
        decision = "unresolved" if unresolved else ("abnormal" if probability >= threshold else "normal")
        y_pred = None if unresolved else int(probability >= threshold)
        decision_variable = "probability"
    return {
        "name": policy,
        "score": float(score),
        "calibrated_probability": float(probability),
        "threshold": float(threshold),
        "decision_variable": decision_variable,
        "decision": decision,
        "y_pred": y_pred,
        "unresolved": bool(unresolved),
        "details": details,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def load_policy_config(path: Path | None, policy: str = "legacy_graph", legacy_threshold: float = 0.03) -> dict:
    value: dict[str, Any] = {}
    if path is not None:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"decision policy file must contain a JSON object: {path}")
        value.update(parsed)
    value.setdefault("policy", policy)
    if value["policy"] in {"legacy_graph", "crowd_state_residual_v1"}:
        value.setdefault("threshold", float(legacy_threshold))
    return value


def fit_platt(scores: Sequence[float], labels: Sequence[int], max_iter: int = 100) -> tuple[float, float]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        raise ValueError("scores and labels must be non-empty and have equal length")
    if len(set(int(value) for value in y.tolist())) < 2:
        return 1.0, math.log((float(y.mean()) + 1e-3) / (1.0 - float(y.mean()) + 1e-3))
    a, b = 1.0, math.log((float(y.mean()) + 1e-3) / (1.0 - float(y.mean()) + 1e-3))
    regularization = 1e-4
    for _ in range(max(1, int(max_iter))):
        z = np.clip(a * x + b, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-8)
        gradient = np.array([np.sum((p - y) * x) + regularization * a, np.sum(p - y)])
        hessian = np.array([
            [np.sum(w * x * x) + regularization, np.sum(w * x)],
            [np.sum(w * x), np.sum(w) + regularization],
        ])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        a -= float(step[0])
        b -= float(step[1])
        if float(np.linalg.norm(step)) < 1e-7:
            break
    return float(a), float(b)
