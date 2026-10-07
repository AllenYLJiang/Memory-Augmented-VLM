#!/usr/bin/env python3
"""Deterministic post-hoc routing for graph-system errors."""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from graph_catalog import infer_family
from selection import label_codes


class FailureRoute(str, Enum):
    GRAPH_CATALOG_GAP = "graph_catalog_gap"
    GRAPH_DEFINITION_GAP = "graph_definition_gap"
    RETRIEVAL_FAILURE = "retrieval_failure"
    NODE_PERCEPTION_FAILURE = "node_perception_failure"
    CONDITIONAL_REFINEMENT_FAILURE = "conditional_refinement_failure"
    OT_ALLOCATION_FAILURE = "ot_allocation_failure"
    CALIBRATION_FAILURE = "calibration_failure"
    TEMPORAL_CONTEXT_FAILURE = "temporal_context_failure"
    REPRESENTATION_GAP = "representation_gap"
    LABEL_NOISE = "label_noise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


METHODS = (
    "independent_direct_nodes",
    "shared_unary_rowmax",
    "unary_ot",
    "conditional_rowmax",
    "conditional_ot_no_coherence",
    "conditional_ot_full",
)
EXPECTED_FAMILIES = {
    "B1": {"human_interaction"},
    "B2": {"impulse_or_blast"},
    "B4": {"crowd"},
    "B5": {"human_interaction"},
    "B6": {"traffic"},
    "G": {"impulse_or_blast"},
}
INTERVENTION_LAYER = {
    FailureRoute.GRAPH_CATALOG_GAP: "graph_catalog",
    FailureRoute.GRAPH_DEFINITION_GAP: "graph_definition",
    FailureRoute.RETRIEVAL_FAILURE: "retrieval",
    FailureRoute.NODE_PERCEPTION_FAILURE: "node_perception",
    FailureRoute.CONDITIONAL_REFINEMENT_FAILURE: "conditional_refinement",
    FailureRoute.OT_ALLOCATION_FAILURE: "ot_matching",
    FailureRoute.CALIBRATION_FAILURE: "decision_policy",
    FailureRoute.TEMPORAL_CONTEXT_FAILURE: "temporal_context",
    FailureRoute.REPRESENTATION_GAP: "event_state_representation",
    FailureRoute.LABEL_NOISE: "annotation_audit",
    FailureRoute.INSUFFICIENT_EVIDENCE: "evidence_collection",
}


def _correct(value: Mapping[str, Any], y_true: int) -> bool:
    return value.get("y_pred") in {0, 1} and int(value.get("y_pred")) == int(y_true)


def _expected_families(case: Mapping[str, Any]) -> set[str]:
    values = set()
    for code in label_codes(str(case.get("video_id", ""))):
        values.update(EXPECTED_FAMILIES.get(code, set()))
    return values


def _candidate_families(case: Mapping[str, Any], polarity: str) -> set[str]:
    candidates = case.get("graph_candidates", {})
    rows = candidates.get(f"{polarity}_ranking", []) if isinstance(candidates, Mapping) else []
    families = {
        str(row.get("family")) for row in rows
        if isinstance(row, Mapping) and row.get("family")
    }
    selected = candidates.get(f"selected_{polarity}", []) if isinstance(candidates, Mapping) else []
    families.update(infer_family(str(key), polarity=polarity) for key in selected)
    return families


def _winner_quality(case: Mapping[str, Any], graph_key: str, method: str = "independent_direct_nodes") -> tuple[float, float]:
    value = case.get("graph_results", {}).get(method, {}).get(graph_key, {})
    coverage = float(value.get("required_coverage", 0.0) or 0.0) if isinstance(value, Mapping) else 0.0
    joint = case.get("joint_graph_calls", {}).get(graph_key, {})
    nodes = joint.get("nodes", {}) if isinstance(joint, Mapping) else {}
    uncertainties = [
        float(node.get("uncertainty", 1.0) or 1.0)
        for node in nodes.values() if isinstance(node, Mapping)
    ] if isinstance(nodes, Mapping) else []
    uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 1.0
    return coverage, uncertainty


def route_failure(case: Mapping[str, Any]) -> dict:
    y_true = int(case.get("y_true", 0))
    competitions = case.get("competitions", {})
    m = {name: competitions.get(name, {}) for name in METHODS}
    final = m["conditional_ot_full"]
    evidence: list[str] = []
    secondary: list[FailureRoute] = []
    gt = case.get("gt", {})
    completeness = case.get("completeness", {})
    candidates = case.get("graph_candidates", {})

    if not bool(gt.get("label_available", gt.get("annotation_found", True))):
        primary, confidence = FailureRoute.LABEL_NOISE, 0.95
        evidence.append("ground_truth_label_unavailable")
    elif not (
        bool(completeness.get("independent", False))
        and bool(completeness.get("joint", False))
        and bool(candidates.get("complete", False))
    ):
        primary, confidence = FailureRoute.INSUFFICIENT_EVIDENCE, 0.95
        evidence.append("incomplete_vlm_or_candidate_parse")
    else:
        margin = float(final.get("margin", 0.0) or 0.0)
        threshold = float(final.get("decision_margin_threshold", 0.0) or 0.0)
        raw_rank_correct = (y_true == 1 and margin > 0.0) or (y_true == 0 and margin < 0.0)
        if raw_rank_correct and not _correct(final, y_true):
            primary, confidence = FailureRoute.CALIBRATION_FAILURE, 0.98
            evidence.extend(["raw_margin_has_correct_sign", "thresholded_decision_is_wrong"])
        elif _correct(m["shared_unary_rowmax"], y_true) and not _correct(m["unary_ot"], y_true):
            primary, confidence = FailureRoute.OT_ALLOCATION_FAILURE, 0.94
            evidence.append("shared_unary_rowmax_correct_but_unary_ot_wrong")
        elif _correct(m["unary_ot"], y_true) and not _correct(m["conditional_rowmax"], y_true):
            primary, confidence = FailureRoute.CONDITIONAL_REFINEMENT_FAILURE, 0.94
            evidence.append("unary_ot_correct_but_conditional_refinement_wrong")
        elif _correct(m["conditional_rowmax"], y_true) and not _correct(m["conditional_ot_no_coherence"], y_true):
            primary, confidence = FailureRoute.OT_ALLOCATION_FAILURE, 0.92
            evidence.append("conditional_rowmax_correct_but_conditional_ot_wrong")
        elif _correct(m["conditional_ot_no_coherence"], y_true) and not _correct(final, y_true):
            primary, confidence = FailureRoute.CALIBRATION_FAILURE, 0.88
            evidence.append("coherence_term_flipped_correct_decision")
        else:
            expected = _expected_families(case) if y_true else set()
            available = _candidate_families(case, "abnormal")
            if expected and not bool(expected & available):
                primary, confidence = FailureRoute.RETRIEVAL_FAILURE, 0.88
                evidence.append("expected_anomaly_family_absent_from_shortlist_trace")
            else:
                winner_key = str(final.get("best_abnormal_graph" if y_true else "best_normal_graph", "NONE"))
                coverage, uncertainty = _winner_quality(case, winner_key)
                if coverage < 0.45 or uncertainty > 0.65:
                    primary, confidence = FailureRoute.NODE_PERCEPTION_FAILURE, 0.76
                    evidence.extend([f"winner_required_coverage={coverage:.3f}", f"winner_uncertainty={uncertainty:.3f}"])
                elif winner_key not in {"", "NONE"} and float(final.get(
                    "best_abnormal_graph_score" if y_true else "best_normal_graph_score", 0.0
                ) or 0.0) >= 0.55:
                    primary, confidence = FailureRoute.GRAPH_DEFINITION_GAP, 0.62
                    evidence.append("relevant_graph_scored_high_but_semantics_did_not_separate_classes")
                else:
                    primary, confidence = FailureRoute.REPRESENTATION_GAP, 0.60
                    evidence.append("no_deterministic_graph_catalog_gap_evidence")

    temporal_subset = str(gt.get("temporal_subset", ""))
    overlap = float(gt.get("overlap_fraction", 0.0) or 0.0)
    if "boundary" in temporal_subset or (0.0 < overlap < 1.0):
        if primary != FailureRoute.TEMPORAL_CONTEXT_FAILURE:
            secondary.append(FailureRoute.TEMPORAL_CONTEXT_FAILURE)
        evidence.append("window_intersects_event_boundary")
    return {
        "version": "failure_router_v1",
        "primary_route": primary.value,
        "secondary_routes": [value.value for value in secondary],
        "confidence": float(confidence),
        "evidence": evidence,
        "recommended_intervention_layer": INTERVENTION_LAYER[primary],
    }


def diagnose_failure_source(case: Mapping[str, Any]) -> str:
    return str(route_failure(case)["primary_route"])
