#!/usr/bin/env python3
"""Parsing and equal-contract scoring for Crowd Event-State V4."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import clip01, file_sha256, stable_sha1
from live_matching import _expand_bins, _normalize_distribution, _quality_bins
from schemas import ConditionalAffinity, GraphTemplateV2


STATES = (
    "active_or_ongoing_physical_escalation",
    "causally_linked_aftermath",
    "pre_event_tension_or_flight",
    "benign_collective_activity",
    "none_or_unobservable",
)
POSITIVE_STATES = STATES[:2]
METHODS = ("conditional_rowmax", "conditional_ot_no_coherence", "conditional_ot_full")
V5_CONTEXT_STATES = ("active", "aftermath", "pre", "benign", "none")
V5_NORMAL_CONFOUND_STATES = (
    "structured_sport_or_play",
    "peaceful_protest_or_ceremony",
    "assistance_or_rescue",
    "ordinary_object_interaction",
    "staged_or_performed_action",
    "none",
)


def _unit(value: Any, default: float = 0.0) -> float:
    """Clamp to [0, 1] while preserving exact no-op coefficients 0 and 1."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def load_calibration(path: Path | None) -> dict:
    if path is None:
        return {
            "version": "crowd_state_calibration_trace_only_v1",
            "frozen": False,
            "refinement_mode": "method_blend",
            "alpha": 0.0,
            "uncertainty_max": 1.0,
            "none_max": 1.0,
            "method_calibration": {},
            "source_groups": [],
            "source_groups_sha256": stable_sha1([], size=40),
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"state calibration must be a JSON object: {path}")
    value = dict(value)
    value["path"] = str(path)
    value["sha256"] = file_sha256(path)
    return value


def _phase_conditional(
    raw_nodes: Mapping[str, Any], graph: GraphTemplateV2, id_to_key: Mapping[str, str],
    evidence_ids: Sequence[str], slots_per_bin: int, coherence: float,
) -> tuple[ConditionalAffinity, dict, bool]:
    key_to_id = {key: anonymous_id for anonymous_id, key in id_to_key.items()}
    qualities, locations, null_scores, priors, uncertainty = [], [], [], [], []
    parsed_nodes: dict[str, dict] = {}
    complete = True
    for node in graph.nodes:
        anonymous_id = key_to_id[node.key]
        item = raw_nodes.get(anonymous_id, {}) if isinstance(raw_nodes.get(anonymous_id), Mapping) else {}
        legacy = item.get("bin_affinity")
        location, location_complete = _normalize_distribution(
            item.get("location_distribution_given_present", legacy)
        )
        quality, quality_complete = _quality_bins(item.get("evidence_quality_by_bin"), legacy)
        presence = clip01(item.get("presence_probability", item.get("conditional_presence", 0.01)))
        null_probability = clip01(item.get("null_probability", 1.0 - presence))
        item_uncertainty = clip01(item.get("uncertainty", 0.5))
        qualities.append(_expand_bins(quality, slots_per_bin, distribution=False))
        locations.append(_expand_bins(location, slots_per_bin, distribution=True))
        null_scores.append(null_probability)
        priors.append(presence)
        uncertainty.append(item_uncertainty)
        parsed_nodes[node.key] = {
            "anonymous_id": anonymous_id,
            "conditional_presence": presence,
            "presence_probability": presence,
            "null_probability": null_probability,
            "location_distribution_given_present": location.tolist(),
            "evidence_quality_by_bin": quality.tolist(),
            "best_bin": int(item.get("best_bin")) if isinstance(item.get("best_bin"), int) else int(np.argmax(location * quality)),
            "visible_evidence": str(item.get("visible_evidence", "")),
            "probability_update_reason": str(item.get("probability_update_reason", "")),
            "uncertainty": item_uncertainty,
        }
        complete = complete and location_complete and quality_complete and (
            "presence_probability" in item or "conditional_presence" in item
        )
    cond = ConditionalAffinity(
        node_keys=graph.node_keys,
        evidence_ids=list(evidence_ids),
        scores=np.vstack(qualities),
        null_scores=np.asarray(null_scores, dtype=np.float64),
        node_presence_priors=np.asarray(priors, dtype=np.float64),
        phase_scores=np.vstack(locations),
        graph_coherence=clip01(coherence),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        cache_sha1="crowd_event_state_v4_shared_response",
    )
    return cond, parsed_nodes, bool(complete)


def parse_event_state_v4_response(
    response: Mapping[str, Any],
    active_graph: GraphTemplateV2,
    aftermath_graph: GraphTemplateV2,
    phase_mappings: Mapping[str, Mapping[str, str]],
    evidence_ids: Sequence[str],
    slots_per_bin: int,
) -> tuple[dict, dict[str, ConditionalAffinity]]:
    parsed = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
    raw_states = parsed.get("state_probabilities", {}) if isinstance(parsed.get("state_probabilities"), Mapping) else {}
    states = {name: _unit(raw_states.get(name, 0.0)) for name in STATES}
    total = sum(states.values())
    if total <= 1e-9:
        states = {name: 0.0 for name in STATES}
        states["none_or_unobservable"] = 1.0
    else:
        states = {name: value / total for name, value in states.items()}
    raw_phases = parsed.get("phase_nodes", {}) if isinstance(parsed.get("phase_nodes"), Mapping) else {}
    coherence = parsed.get("phase_coherence", {}) if isinstance(parsed.get("phase_coherence"), Mapping) else {}
    active_cond, active_nodes, active_complete = _phase_conditional(
        raw_phases.get("active", {}) if isinstance(raw_phases.get("active"), Mapping) else {},
        active_graph, phase_mappings["active"], evidence_ids, slots_per_bin,
        _unit(coherence.get("active", 0.01)),
    )
    aftermath_cond, aftermath_nodes, aftermath_complete = _phase_conditional(
        raw_phases.get("aftermath", {}) if isinstance(raw_phases.get("aftermath"), Mapping) else {},
        aftermath_graph, phase_mappings["aftermath"], evidence_ids, slots_per_bin,
        _unit(coherence.get("aftermath", 0.01)),
    )
    complete = all(name in raw_states for name in STATES) and active_complete and aftermath_complete
    trace = {
        "version": "crowd_event_state_trace_v4",
        "state_probabilities": states,
        "phase_coherence": {
            "active": active_cond.graph_coherence,
            "aftermath": aftermath_cond.graph_coherence,
        },
        "phase_nodes": {"active": active_nodes, "aftermath": aftermath_nodes},
        "active_evidence_bins": list(parsed.get("active_evidence_bins", [])),
        "aftermath_evidence_bins": list(parsed.get("aftermath_evidence_bins", [])),
        "pre_event_bins": list(parsed.get("pre_event_bins", [])),
        "benign_bins": list(parsed.get("benign_bins", [])),
        "visible_evidence": parsed.get("visible_evidence", {}),
        "counterfactual_evidence": parsed.get("counterfactual_evidence", {}),
        "decision_reason": str(parsed.get("decision_reason", "")),
        "uncertainty": _unit(parsed.get("uncertainty", 0.5)),
        "complete": bool(complete),
        "cache_hit": bool(response.get("cache_hit")),
        "cache_path": response.get("cache_path"),
        "evidence": response.get("evidence", {}),
        "raw": response.get("raw", ""),
    }
    return trace, {"active": active_cond, "aftermath": aftermath_cond}


def parse_event_state_v5_response(
    response: Mapping[str, Any],
    active_graph: GraphTemplateV2,
    aftermath_graph: GraphTemplateV2,
    phase_mappings: Mapping[str, Mapping[str, str]],
    evidence_ids: Sequence[str],
    slots_per_bin: int,
) -> tuple[dict, dict[str, ConditionalAffinity]]:
    """Parse V5's separate event-context and current-window occupancy heads."""
    parsed = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
    raw_context = (
        parsed.get("event_context_state_probabilities", {})
        if isinstance(parsed.get("event_context_state_probabilities"), Mapping)
        else {}
    )
    context = {name: _unit(raw_context.get(name, 0.0)) for name in V5_CONTEXT_STATES}
    total = sum(context.values())
    if total <= 1e-9:
        context = {name: 0.0 for name in V5_CONTEXT_STATES}
        context["none"] = 1.0
    else:
        context = {name: value / total for name, value in context.items()}
    legacy_parsed = dict(parsed)
    legacy_parsed["state_probabilities"] = {
        STATES[0]: context["active"], STATES[1]: context["aftermath"],
        STATES[2]: context["pre"], STATES[3]: context["benign"], STATES[4]: context["none"],
    }
    legacy_response = dict(response)
    legacy_response["parsed"] = legacy_parsed
    trace, conditionals = parse_event_state_v4_response(
        legacy_response, active_graph, aftermath_graph, phase_mappings,
        evidence_ids, slots_per_bin,
    )
    occupancy_present = "current_window_active_occupancy_probability" in parsed
    normal_confound_present = "current_window_normal_confound_probability" in parsed
    raw_normal_confound = (
        parsed.get("normal_confound_probabilities", {})
        if isinstance(parsed.get("normal_confound_probabilities"), Mapping)
        else {}
    )
    normal_confound = {
        name: _unit(raw_normal_confound.get(name, 0.0))
        for name in V5_NORMAL_CONFOUND_STATES
    }
    normal_total = sum(normal_confound.values())
    if normal_total <= 1e-9:
        normal_confound = {name: 0.0 for name in V5_NORMAL_CONFOUND_STATES}
        normal_confound["none"] = 1.0
    else:
        normal_confound = {name: value / normal_total for name, value in normal_confound.items()}
    context_complete = all(name in raw_context for name in V5_CONTEXT_STATES)
    normal_confound_complete = all(name in raw_normal_confound for name in V5_NORMAL_CONFOUND_STATES)
    trace.update({
        "version": "crowd_event_state_trace_v5_signed_confound",
        "event_context_state_probabilities": context,
        "current_window_anomaly_occupancy": _unit(
            parsed.get("current_window_active_occupancy_probability", 0.0)
        ),
        "current_window_active_occupancy_probability": _unit(
            parsed.get("current_window_active_occupancy_probability", 0.0)
        ),
        "current_window_normal_confound_probability": _unit(
            parsed.get("current_window_normal_confound_probability", 0.0)
        ),
        "normal_confound_probabilities": normal_confound,
        "aftermath_context_probability": _unit(
            parsed.get("aftermath_context_probability", context["aftermath"])
        ),
        "aftermath_current_window_occupancy_support_probability": _unit(
            parsed.get("aftermath_current_window_occupancy_support_probability", 0.0)
        ),
        "direct_active_bins": list(parsed.get("direct_active_bins", [])),
        "context_only_bins": list(parsed.get("context_only_bins", [])),
        "normal_confound_bins": list(parsed.get("normal_confound_bins", [])),
        "normal_confound_evidence": parsed.get("normal_confound_evidence", {}),
        "complete": bool(
            trace.get("complete") and occupancy_present and context_complete
            and normal_confound_present and normal_confound_complete
        ),
    })
    return trace, conditionals


def _mapped_score(value: float, config: Mapping[str, Any], method: str) -> float:
    methods = config.get("method_calibration", {}) if isinstance(config.get("method_calibration"), Mapping) else {}
    method_config = methods.get(method, {}) if isinstance(methods.get(method), Mapping) else {}
    scale = float(method_config.get("scale", 1.0))
    offset = float(method_config.get("offset", 0.0))
    return _unit(scale * float(value) + offset)


def score_method_blend(
    trace: Mapping[str, Any],
    base_results: Mapping[str, Mapping[str, Any]],
    phase_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    target_key: str,
    calibration: Mapping[str, Any],
) -> tuple[dict[str, dict], dict]:
    states = trace.get("state_probabilities", {})
    p_active = _unit(states.get(STATES[0], 0.0))
    p_aftermath = _unit(states.get(STATES[1], 0.0))
    p_none = _unit(states.get(STATES[4], 0.0))
    alpha = _unit(calibration.get("alpha", 0.0))
    uncertainty_max = _unit(calibration.get("uncertainty_max", 1.0))
    none_max = _unit(calibration.get("none_max", 1.0))
    noop_reason = ""
    if not trace.get("complete"):
        noop_reason = "incomplete_state_output"
    elif _unit(trace.get("uncertainty", 1.0)) > uncertainty_max:
        noop_reason = "uncertainty_above_calibrated_limit"
    elif p_none > none_max:
        noop_reason = "none_or_unobservable_above_limit"
    effective_alpha = 0.0 if noop_reason else alpha
    replacements: dict[str, dict] = {}
    method_details: dict[str, dict] = {}
    for method in METHODS:
        active_score = _unit(phase_results["active"][method].get("graph_score", 0.0))
        aftermath_score = _unit(phase_results["aftermath"][method].get("graph_score", 0.0))
        state_score = _unit(p_active * active_score + p_aftermath * aftermath_score)
        mapped = _mapped_score(state_score, calibration, method)
        base_score = _unit(base_results[method].get("graph_score", 0.0))
        final_score = _unit((1.0 - effective_alpha) * base_score + effective_alpha * mapped)
        value = dict(base_results[method])
        value.update({
            "graph_key": target_key,
            "method": f"conditional_ot_state_blend_v4/{method}",
            "graph_score": final_score,
            "diagnostics": {
                **dict(base_results[method].get("diagnostics", {})),
                "crowd_event_state_v4": {
                    "base_score": base_score,
                    "active_method_score": active_score,
                    "aftermath_method_score": aftermath_score,
                    "state_score": state_score,
                    "mapped_state_score": mapped,
                    "configured_alpha": alpha,
                    "effective_alpha": effective_alpha,
                    "noop_reason": noop_reason,
                    "repeated_gate_count": 0,
                },
            },
        })
        replacements[method] = value
        method_details[method] = value["diagnostics"]["crowd_event_state_v4"]
    provenance = {
        "baseline_score_contract": "method_specific_conditional_ot_v3",
        "candidate_score_contract": "method_specific_conditional_ot_state_blend_v4",
        "calibration_id": str(calibration.get("calibration_id", "trace_only")),
        "calibration_sha256": str(calibration.get("sha256", "")),
        "calibration_source_groups_sha256": str(calibration.get("source_groups_sha256", "")),
        "method_matched": True,
        "repeated_gate_count": 0,
        "refinement_mode": "method_blend",
        "noop_reason": noop_reason,
        "method_details": method_details,
    }
    return replacements, provenance


def bounded_margin_residual(
    trace: Mapping[str, Any], base_competition: Mapping[str, Any], calibration: Mapping[str, Any],
) -> tuple[dict, dict]:
    value = dict(base_competition)
    states = trace.get("state_probabilities", {})
    positive = _unit(states.get(STATES[0], 0.0)) + _unit(states.get(STATES[1], 0.0))
    negative = _unit(states.get(STATES[2], 0.0)) + _unit(states.get(STATES[3], 0.0)) + _unit(states.get(STATES[4], 0.0))
    beta = max(0.0, float(calibration.get("beta", 0.0)))
    bound = max(0.0, float(calibration.get("residual_bound", 0.0)))
    uncertainty_max = _unit(calibration.get("uncertainty_max", 1.0))
    none_max = _unit(calibration.get("none_max", 1.0))
    noop_reason = ""
    if not trace.get("complete"):
        noop_reason = "incomplete_state_output"
    elif _unit(trace.get("uncertainty", 1.0)) > uncertainty_max:
        noop_reason = "uncertainty_above_calibrated_limit"
    elif _unit(states.get(STATES[4], 0.0)) > none_max:
        noop_reason = "none_or_unobservable_above_limit"
    odds = math.log((positive + 1e-6) / (negative + 1e-6))
    residual = 0.0 if noop_reason else max(-bound, min(bound, beta * odds))
    margin = float(value.get("margin", 0.0)) + residual
    threshold = float(value.get("decision_margin_threshold", 0.03) or 0.03)
    value.update({
        "margin": margin,
        "decision": "abnormal" if margin > threshold else "normal",
        "y_pred": int(margin > threshold),
        "state_residual": residual,
    })
    return value, {
        "state_log_odds": odds, "beta": beta, "residual_bound": bound,
        "residual": residual, "noop_reason": noop_reason, "repeated_gate_count": 0,
    }


def score_phase_aware_occupancy(
    trace: Mapping[str, Any], base_competition: Mapping[str, Any], calibration: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Apply a bounded V5 residual whose direct positive signal is current occupancy.

    Aftermath context contributes only through an independently observable occupancy-support
    term and a frozen coefficient constrained by the calibration fitter.
    """
    value = dict(base_competition)
    occupancy = _unit(trace.get("current_window_active_occupancy_probability", 0.0))
    normal_confound = _unit(trace.get("current_window_normal_confound_probability", 0.0))
    aftermath = _unit(trace.get("aftermath_context_probability", 0.0))
    aftermath_support = _unit(trace.get("aftermath_current_window_occupancy_support_probability", 0.0))
    beta_active = max(0.0, float(calibration.get("beta_active", 0.0)))
    beta_aftermath = max(0.0, float(calibration.get("beta_aftermath", 0.0)))
    beta_normal_confound = max(0.0, float(calibration.get("beta_normal_confound", 0.0)))
    if beta_aftermath > 0.5 * beta_active + 1e-12:
        raise ValueError("beta_aftermath must be <= 0.5 * beta_active")
    bound = max(0.0, float(calibration.get("residual_bound", 0.0)))
    uncertainty_max = _unit(calibration.get("uncertainty_max", 1.0))
    none_max = _unit(calibration.get("none_max", 1.0))
    context = trace.get("event_context_state_probabilities", {})
    p_none = _unit(context.get("none", 0.0)) if isinstance(context, Mapping) else 0.0
    noop_reason = ""
    if not trace.get("complete"):
        noop_reason = "incomplete_v5_output"
    elif _unit(trace.get("uncertainty", 1.0)) > uncertainty_max:
        noop_reason = "uncertainty_above_calibrated_limit"
    elif p_none > none_max:
        noop_reason = "none_or_unobservable_above_limit"
    eps = 1e-6
    active_logit = math.log((occupancy + eps) / (1.0 - occupancy + eps))
    normal_confound_logit = math.log(
        (normal_confound + eps) / (1.0 - normal_confound + eps)
    )
    normal_confound_evidence = max(0.0, normal_confound_logit)
    aftermath_occupancy = aftermath * aftermath_support
    raw = (
        beta_active * active_logit
        + beta_aftermath * aftermath_occupancy
        - beta_normal_confound * normal_confound_evidence
    )
    residual = 0.0 if noop_reason or bound <= 0.0 else bound * math.tanh(raw / bound)
    margin = float(value.get("margin", 0.0)) + residual
    threshold = float(value.get("decision_margin_threshold", 0.03) or 0.03)
    value.update({
        "margin": margin,
        "decision": "abnormal" if margin > threshold else "normal",
        "y_pred": int(margin > threshold),
        "state_residual": residual,
    })
    return value, {
        "current_window_active_occupancy_probability": occupancy,
        "active_occupancy_logit": active_logit,
        "current_window_normal_confound_probability": normal_confound,
        "normal_confound_logit": normal_confound_logit,
        "normal_confound_positive_evidence": normal_confound_evidence,
        "aftermath_context_probability": aftermath,
        "aftermath_current_window_occupancy_support_probability": aftermath_support,
        "aftermath_occupancy_interaction": aftermath_occupancy,
        "beta_active": beta_active,
        "beta_aftermath": beta_aftermath,
        "beta_normal_confound": beta_normal_confound,
        "residual_bound": bound,
        "raw_residual_input": raw,
        "residual": residual,
        "saturation_ratio": abs(residual) / bound if bound > 0.0 else 0.0,
        "noop_reason": noop_reason,
        "repeated_gate_count": 0,
        "phase_specific_weights_frozen": bool(calibration.get("frozen", False)),
    }
