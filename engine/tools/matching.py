#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic matching math for M0/M1/M2/M3a/M3b/M3c.

M3 is decomposed so that a gain cannot be vaguely attributed to "OT":

* M3a: conditional evidence, independent row-max allocation;
* M3b: the same conditional evidence with OT, graph coherence disabled;
* M3c: M3b plus the graph-level coherence factor.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import sinkhorn_ot as ot
from schemas import ConditionalAffinity, GraphMatchResult, GraphTemplateV2, UnaryAffinity

EPS = 1e-9


def graph_score(q: np.ndarray, weights: np.ndarray, coherence: float, coherence_weight: float = 0.25) -> float:
    """Node geometric score times a size-independent coherence factor."""
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    node_score = float(np.exp(np.sum(w * np.log(q)) / max(float(np.sum(w)), EPS)))
    return float(node_score * (max(float(coherence), EPS) ** max(0.0, float(coherence_weight))))


def _coverage(presence: np.ndarray, graph: GraphTemplateV2, threshold: float = 0.5):
    required = np.array([node.required for node in graph.nodes], dtype=bool)
    required_coverage = float(np.mean(presence[required] >= threshold)) if required.any() else 1.0
    optional = ~required
    optional_coverage = float(np.mean(presence[optional] >= threshold)) if optional.any() else 1.0
    return required_coverage, optional_coverage


def _result(
    graph: GraphTemplateV2,
    method: str,
    node_keys,
    presence,
    q,
    expected_time,
    plan,
    coherence,
    assignments,
    diagnostics,
    coherence_weight: float = 0.25,
) -> GraphMatchResult:
    weights = graph.weights
    required_coverage, optional_coverage = _coverage(np.asarray(presence), graph)
    node_geometric = float(
        np.exp(np.sum(weights * np.log(np.clip(q, EPS, 1.0))) / max(float(np.sum(weights)), EPS))
    )
    score = graph_score(q, weights, coherence, coherence_weight)
    required_mask = np.array([node.required for node in graph.nodes], dtype=bool)
    complete = bool(np.all(np.asarray(presence)[required_mask] >= 0.5)) if required_mask.any() else True
    return GraphMatchResult(
        graph_key=graph.key,
        method=method,
        node_support={key: float(q[index]) for index, key in enumerate(node_keys)},
        node_presence={key: float(presence[index]) for index, key in enumerate(node_keys)},
        expected_time={key: float(expected_time[index]) for index, key in enumerate(node_keys)},
        assignments=assignments,
        transport_plan=(plan.tolist() if plan is not None else None),
        node_geometric_score=node_geometric,
        graph_coherence=float(coherence),
        graph_score=score,
        required_coverage=required_coverage,
        optional_coverage=optional_coverage,
        complete=complete,
        diagnostics=diagnostics,
    )


def _assignments(node_keys, evidence_ids, location, presence, null_mass):
    output = []
    for index, key in enumerate(node_keys):
        real_peak = float(location[index].max()) if location.shape[1] else 0.0
        if presence[index] < 0.5 and null_mass[index] >= real_peak:
            output.append({
                "node_key": key,
                "assigned": "NULL",
                "mass": float(null_mass[index]),
                "presence": float(presence[index]),
            })
        else:
            best = int(location[index].argmax()) if location.shape[1] else -1
            output.append({
                "node_key": key,
                "assigned": evidence_ids[best] if best >= 0 else "NULL",
                "mass": float(location[index, best]) if best >= 0 else 0.0,
                "presence": float(presence[index]),
            })
    return output


def _collisions(location, presence, evidence_ids=None):
    picks = []
    for index in range(location.shape[0]):
        if presence[index] < 0.5 or not location.shape[1]:
            continue
        column = int(location[index].argmax())
        value = evidence_ids[column] if evidence_ids else column
        # Slot IDs such as T2#0/T2#1 share the same temporal bin.  Report both slot and bin collisions.
        picks.append(value)
    slot_collisions = len(picks) - len(set(picks))
    bins = [str(value).split("#", 1)[0] for value in picks]
    bin_collisions = len(bins) - len(set(bins))
    return {"slot_collisions": slot_collisions, "temporal_bin_collisions": bin_collisions}


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0)
    return array / np.maximum(array.sum(axis=1, keepdims=True), EPS)


def _align_columns(values: np.ndarray, target_columns: int, *, normalize: bool) -> np.ndarray:
    """Align a KxL location/quality matrix to the KxM evidence field.

    Live v3 runs already emit exactly one value per evidence slot.  The interpolation path
    is deliberately retained for older caches and small synthetic tests whose temporal
    distributions have eight bins while the evidence field has a different number of
    units.  It is deterministic and never invents a new peak outside the original range.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape={array.shape}")
    if target_columns <= 0:
        raise ValueError("target_columns must be positive")
    if array.shape[1] == target_columns:
        aligned = array.copy()
    elif array.shape[1] == 1:
        aligned = np.repeat(array, target_columns, axis=1)
    elif target_columns % array.shape[1] == 0:
        repeats = target_columns // array.shape[1]
        aligned = np.repeat(array, repeats, axis=1)
        if normalize and repeats > 1:
            aligned /= repeats
    else:
        source_x = np.linspace(0.0, 1.0, array.shape[1])
        target_x = np.linspace(0.0, 1.0, target_columns)
        aligned = np.vstack([np.interp(target_x, source_x, row) for row in array])
    aligned = np.clip(aligned, EPS, 1.0)
    return _normalized_rows(aligned) if normalize else aligned


def _unary_presence(unary: UnaryAffinity, indices) -> np.ndarray:
    if unary.node_presence_priors is not None:
        return np.clip(np.asarray(unary.node_presence_priors, dtype=np.float64)[indices], EPS, 1.0)
    # Backward compatibility: infer a prior from null probability.
    return np.clip(1.0 - np.asarray(unary.null_scores, dtype=np.float64)[indices], EPS, 1.0)


def unary_effective_affinity(unary: UnaryAffinity, indices) -> np.ndarray:
    quality = np.clip(np.asarray(unary.scores, dtype=np.float64)[indices], EPS, 1.0)
    location = _align_columns(
        np.asarray(unary.phase_scores, dtype=np.float64)[indices],
        quality.shape[1],
        normalize=True,
    )
    presence = _unary_presence(unary, indices)[:, None]
    return np.clip(presence * np.sqrt(quality * location), EPS, 1.0)


def conditional_effective_affinity(
    unary: UnaryAffinity,
    cond: ConditionalAffinity,
    unary_indices,
    conditional_indices,
    temporal: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    unary_quality = np.clip(np.asarray(unary.scores)[unary_indices], EPS, 1.0)
    unary_location = _align_columns(
        np.asarray(unary.phase_scores)[unary_indices], unary_quality.shape[1], normalize=True
    )
    unary_presence = _unary_presence(unary, unary_indices)

    conditional_quality = _align_columns(
        np.asarray(cond.scores)[conditional_indices], unary_quality.shape[1], normalize=False
    )
    conditional_location = _align_columns(
        np.asarray(cond.phase_scores)[conditional_indices], unary_quality.shape[1], normalize=True
    )
    conditional_presence = np.clip(np.asarray(cond.node_presence_priors)[conditional_indices], EPS, 1.0)

    # Presence is used exactly once.  Location and evidence quality are conditional distributions.
    presence = np.sqrt(unary_presence * conditional_presence)
    spatial = np.sqrt(
        np.sqrt(unary_quality * unary_location)
        * np.sqrt(conditional_quality * conditional_location)
    )
    affinity = presence[:, None] * spatial
    if temporal is not None:
        affinity *= np.sqrt(np.clip(np.asarray(temporal, dtype=np.float64), EPS, 1.0))
    null_affinity = np.sqrt(
        np.clip(np.asarray(unary.null_scores)[unary_indices], EPS, 1.0)
        * np.clip(np.asarray(cond.null_scores)[conditional_indices], EPS, 1.0)
        * np.clip(1.0 - presence, EPS, 1.0)
    )
    return np.clip(affinity, EPS, 1.0), np.clip(null_affinity, EPS, 1.0)


def _support_quality(plan, affinity, presence):
    """Real transport presence times mean visual affinity under the allocated mass."""
    node_count, evidence_count = affinity.shape
    quality = np.zeros(node_count)
    for index in range(node_count):
        mass = plan[index, :evidence_count]
        denominator = mass.sum()
        mean_affinity = float(np.sum(mass * affinity[index])) / (denominator + EPS)
        quality[index] = float(presence[index]) * mean_affinity
    return quality


def _transport_diagnostics(graph, evidence_ids, summary, affinity):
    collision_details = _collisions(summary["node_location"], summary["node_presence"], evidence_ids)
    return {
        "collision_evidence": int(collision_details["slot_collisions"]),
        "collision_details": collision_details,
        "entropy": summary["entropy"].tolist(),
        "null_mass": summary["null_mass"].tolist(),
        "null_mass_by_node": {
            node.key: float(summary["null_mass"][index]) for index, node in enumerate(graph.nodes)
        },
        "node_location": {
            node.key: summary["node_location"][index].tolist() for index, node in enumerate(graph.nodes)
        },
        "effective_affinity": np.asarray(affinity).tolist(),
        "fused_affinity": np.asarray(affinity).tolist(),
    }


# M0 -------------------------------------------------------------------------
def match_graph_independent_direct(
    graph: GraphTemplateV2,
    node_presence: Dict[str, float],
    node_time: Optional[Dict[str, float]] = None,
) -> GraphMatchResult:
    keys = graph.node_keys
    presence = np.array([float(node_presence.get(key, 0.0)) for key in keys])
    expected_time = np.array([float((node_time or {}).get(key, 0.0)) for key in keys])
    assignments = [
        {"node_key": key, "assigned": "independent", "presence": float(presence[index])}
        for index, key in enumerate(keys)
    ]
    return _result(
        graph, "independent_direct_nodes", keys, presence, np.clip(presence, EPS, 1.0),
        expected_time, None, 1.0, assignments, {}, coherence_weight=0.0,
    )


# M1 -------------------------------------------------------------------------
def match_graph_shared_rowmax(graph: GraphTemplateV2, unary: UnaryAffinity, evidence_centers) -> GraphMatchResult:
    indices = [unary.node_keys.index(node.key) for node in graph.nodes]
    affinity = unary_effective_affinity(unary, indices)
    null_values = np.asarray(unary.null_scores)[indices]
    node_count, evidence_count = affinity.shape
    presence = np.zeros(node_count)
    location = np.zeros((node_count, evidence_count))
    quality = np.zeros(node_count)
    expected_time = np.zeros(node_count)
    null_mass = np.zeros(node_count)
    centers = np.asarray(evidence_centers, dtype=np.float64)
    center_scale = max(float(np.max(centers)), 1.0) if centers.size else 1.0
    for index in range(node_count):
        best = int(affinity[index].argmax())
        real = float(affinity[index, best])
        if real >= null_values[index]:
            presence[index] = _unary_presence(unary, indices)[index]
            location[index, best] = 1.0
            quality[index] = real
            expected_time[index] = centers[best] / center_scale
        else:
            null_mass[index] = float(null_values[index])
            quality[index] = EPS
    assignments = _assignments(graph.node_keys, unary.evidence_ids, location, presence, null_mass)
    collision_details = _collisions(location, presence, unary.evidence_ids)
    diagnostics = {
        "collision_evidence": int(collision_details["slot_collisions"]),
        "collision_details": collision_details,
        "node_location": {node.key: location[i].tolist() for i, node in enumerate(graph.nodes)},
        "null_mass_by_node": {node.key: float(null_mass[i]) for i, node in enumerate(graph.nodes)},
        "effective_affinity": affinity.tolist(),
    }
    return _result(
        graph, "shared_unary_rowmax", graph.node_keys, presence, quality, expected_time,
        None, 1.0, assignments, diagnostics, coherence_weight=0.0,
    )


# M2 -------------------------------------------------------------------------
def match_graph_unary_ot(
    graph: GraphTemplateV2,
    unary: UnaryAffinity,
    evidence_centers,
    iterations: int = 60,
) -> GraphMatchResult:
    indices = [unary.node_keys.index(node.key) for node in graph.nodes]
    affinity = unary_effective_affinity(unary, indices)
    null_values = np.asarray(unary.null_scores)[indices]
    plan = ot.solve_dustbin_ot(ot.build_augmented_log_scores(affinity, null_values), iterations)
    summary = ot.summarize_transport(plan, evidence_centers)
    quality = _support_quality(plan, affinity, summary["node_presence"])
    assignments = _assignments(
        graph.node_keys, unary.evidence_ids, summary["node_location"],
        summary["node_presence"], summary["null_mass"],
    )
    diagnostics = _transport_diagnostics(graph, unary.evidence_ids, summary, affinity)
    return _result(
        graph, "unary_ot", graph.node_keys, summary["node_presence"], quality,
        summary["expected_time"], plan, 1.0, assignments, diagnostics, coherence_weight=0.0,
    )


# M3a ------------------------------------------------------------------------
def match_graph_conditional_rowmax(
    graph: GraphTemplateV2,
    unary: UnaryAffinity,
    cond: ConditionalAffinity,
    evidence_centers,
    temporal: Optional[np.ndarray] = None,
) -> GraphMatchResult:
    unary_indices = [unary.node_keys.index(node.key) for node in graph.nodes]
    conditional_indices = [cond.node_keys.index(node.key) for node in graph.nodes]
    affinity, null_values = conditional_effective_affinity(
        unary, cond, unary_indices, conditional_indices, temporal,
    )
    node_count, evidence_count = affinity.shape
    presence = np.zeros(node_count)
    location = np.zeros((node_count, evidence_count))
    quality = np.zeros(node_count)
    expected_time = np.zeros(node_count)
    null_mass = np.zeros(node_count)
    centers = np.asarray(evidence_centers, dtype=np.float64)
    center_scale = max(float(np.max(centers)), 1.0) if centers.size else 1.0
    conditional_presence = np.asarray(cond.node_presence_priors)[conditional_indices]
    for index in range(node_count):
        best = int(affinity[index].argmax())
        if float(affinity[index, best]) >= float(null_values[index]):
            presence[index] = float(conditional_presence[index])
            location[index, best] = 1.0
            quality[index] = float(affinity[index, best])
            expected_time[index] = centers[best] / center_scale
        else:
            null_mass[index] = float(null_values[index])
            quality[index] = EPS
    assignments = _assignments(graph.node_keys, unary.evidence_ids, location, presence, null_mass)
    collision_details = _collisions(location, presence, unary.evidence_ids)
    diagnostics = {
        "collision_evidence": int(collision_details["slot_collisions"]),
        "collision_details": collision_details,
        "node_location": {node.key: location[i].tolist() for i, node in enumerate(graph.nodes)},
        "null_mass_by_node": {node.key: float(null_mass[i]) for i, node in enumerate(graph.nodes)},
        "effective_affinity": affinity.tolist(),
    }
    return _result(
        graph, "conditional_rowmax", graph.node_keys, presence, quality, expected_time,
        None, 1.0, assignments, diagnostics, coherence_weight=0.0,
    )


# M3b/M3c --------------------------------------------------------------------
def match_graph_conditional_ot(
    graph: GraphTemplateV2,
    unary: UnaryAffinity,
    cond: ConditionalAffinity,
    evidence_centers,
    temporal: Optional[np.ndarray] = None,
    iterations: int = 60,
    *,
    use_coherence: bool = True,
    coherence_weight: float = 0.25,
    method_name: Optional[str] = None,
) -> GraphMatchResult:
    unary_indices = [unary.node_keys.index(node.key) for node in graph.nodes]
    conditional_indices = [cond.node_keys.index(node.key) for node in graph.nodes]
    affinity, null_values = conditional_effective_affinity(
        unary, cond, unary_indices, conditional_indices, temporal,
    )
    plan = ot.solve_dustbin_ot(ot.build_augmented_log_scores(affinity, null_values), iterations)
    summary = ot.summarize_transport(plan, evidence_centers)
    quality = _support_quality(plan, affinity, summary["node_presence"])
    assignments = _assignments(
        graph.node_keys, unary.evidence_ids, summary["node_location"],
        summary["node_presence"], summary["null_mass"],
    )
    coherence = float(cond.graph_coherence) if use_coherence else 1.0
    diagnostics = _transport_diagnostics(graph, unary.evidence_ids, summary, affinity)
    diagnostics["conditional_supporting_nodes"] = cond.supporting_nodes
    diagnostics["conditional_suppressing_nodes"] = cond.suppressing_nodes
    method = method_name or ("conditional_ot_full" if use_coherence else "conditional_ot_no_coherence")
    return _result(
        graph, method, graph.node_keys, summary["node_presence"], quality,
        summary["expected_time"], plan, coherence, assignments, diagnostics,
        coherence_weight=coherence_weight if use_coherence else 0.0,
    )
