#!/usr/bin/env python3
"""Parsing and deterministic scoring for the crowd event-state V3 candidate."""
from __future__ import annotations

from typing import Any, Mapping

from common import clip01


STATES = (
    "active_escalation",
    "causally_linked_aftermath",
    "benign_or_pre_event_context",
    "none_or_unobservable",
)


def _bins(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted({int(item) for item in value if isinstance(item, int) and 0 <= item < 8})


def parse_event_state_response(response: Mapping[str, Any]) -> dict:
    parsed = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
    raw_states = parsed.get("state_probabilities", {}) if isinstance(parsed.get("state_probabilities"), Mapping) else {}
    values = {state: clip01(raw_states.get(state, 0.0)) for state in STATES}
    total = sum(values.values())
    if total <= 1e-9:
        values = {state: 0.0 for state in STATES}
        values["none_or_unobservable"] = 1.0
    else:
        values = {state: value / total for state, value in values.items()}
    complete = all(state in raw_states for state in STATES) and all(
        key in parsed for key in (
            "transition_observed_probability",
            "aftermath_causal_link_probability",
            "same_episode_probability",
        )
    )
    return {
        "version": "crowd_event_state_trace_v3",
        "state_probabilities": values,
        "transition_observed_probability": clip01(parsed.get("transition_observed_probability", 0.0)),
        "aftermath_causal_link_probability": clip01(parsed.get("aftermath_causal_link_probability", 0.0)),
        "same_episode_probability": clip01(parsed.get("same_episode_probability", 0.0)),
        "active_transition_bins": _bins(parsed.get("active_transition_bins")),
        "aftermath_evidence_bins": _bins(parsed.get("aftermath_evidence_bins")),
        "benign_context_bins": _bins(parsed.get("benign_context_bins")),
        "observed_transition_evidence": str(parsed.get("observed_transition_evidence", "")),
        "aftermath_causal_evidence": str(parsed.get("aftermath_causal_evidence", "")),
        "benign_counterfactual_evidence": str(parsed.get("benign_counterfactual_evidence", "")),
        "decision_reason": str(parsed.get("decision_reason", "")),
        "uncertainty": clip01(parsed.get("uncertainty", 0.5)),
        "complete": bool(complete),
        "cache_hit": bool(response.get("cache_hit")),
        "cache_path": response.get("cache_path"),
        "evidence": response.get("evidence", {}),
        "raw": response.get("raw", ""),
    }


def score_event_state(
    trace: Mapping[str, Any],
    phase_results: Mapping[str, Mapping[str, Any]],
    active_key: str,
    aftermath_key: str,
    target_key: str,
) -> tuple[dict, dict]:
    states = trace.get("state_probabilities", {})
    active_score = clip01(phase_results.get(active_key, {}).get("graph_score", 0.0))
    aftermath_score = clip01(phase_results.get(aftermath_key, {}).get("graph_score", 0.0))
    p_active = clip01(states.get("active_escalation", 0.0))
    p_aftermath = clip01(states.get("causally_linked_aftermath", 0.0))
    p_benign = clip01(states.get("benign_or_pre_event_context", 0.0))
    transition = clip01(trace.get("transition_observed_probability", 0.0))
    causal_link = clip01(trace.get("aftermath_causal_link_probability", 0.0))
    same_episode = clip01(trace.get("same_episode_probability", 0.0))
    active_component = p_active * transition * same_episode * active_score
    aftermath_component = p_aftermath * causal_link * same_episode * aftermath_score
    benign_suppression = 1.0 - p_benign
    final_score = clip01((active_component + aftermath_component) * benign_suppression)
    winner = active_key if active_component >= aftermath_component else aftermath_key
    details = {
        "version": "crowd_event_state_score_v3",
        "active_key": active_key,
        "aftermath_key": aftermath_key,
        "active_unary_ot_score": active_score,
        "aftermath_unary_ot_score": aftermath_score,
        "active_component": active_component,
        "aftermath_component": aftermath_component,
        "benign_suppression": benign_suppression,
        "same_episode_probability": same_episode,
        "winner": winner,
        "graph_score": final_score,
    }
    result = {
        "graph_key": target_key,
        "method": "event_state_gated_unary_ot_v3",
        "node_support": {
            active_key: active_score,
            aftermath_key: aftermath_score,
        },
        "node_presence": dict(states),
        "expected_time": {},
        "assignments": [],
        "transport_plan": None,
        "node_geometric_score": active_score + aftermath_score,
        "graph_coherence": same_episode,
        "graph_score": final_score,
        "required_coverage": p_active + p_aftermath,
        "optional_coverage": benign_suppression,
        "complete": bool(trace.get("complete")),
        "diagnostics": {"event_state": details},
    }
    return result, details

