#!/usr/bin/env python3
"""Blind prompts for independent nodes, two-stage conditional refinement and verification."""
from __future__ import annotations

import json
from typing import Any, Mapping

from schemas import GraphNodeV2, GraphTemplateV2


LEGACY_PROMPT_VERSION = "conditional_ot_multigraph_v3_initial_ot_feedback_pool_v2"
PROMPT_VERSION = "conditional_ot_multigraph_v3_eight_frame_contract_v3"
FORBIDDEN = (
    "y_true", "correct answer", "anomaly overlap", "label_b1", "label_b2",
    "label_b4", "label_b5", "label_b6",
)


def assert_blind(prompt: str) -> None:
    lowered = prompt.lower()
    leaked = [token for token in FORBIDDEN if token in lowered]
    if leaked:
        raise ValueError(f"prompt contains forbidden evaluation information: {leaked}")


def _node_text(node: GraphNodeV2, anonymous_id: str) -> dict:
    return {
        "id": anonymous_id,
        "description": node.title,
        "visual_cues": list(node.cue_bundle),
    }


def _distribution_schema() -> dict:
    return {
        "presence_probability": 0.0,
        "null_probability": 0.0,
        "location_distribution_given_present": [0.125] * 8,
        "evidence_quality_by_bin": [0.0] * 8,
        "best_bin": None,
        "region": "",
        "visible_evidence": "",
        "uncertainty": 0.0,
    }


def independent_node_prompt(node: GraphNodeV2, evidence_mode: str = "frames8") -> str:
    schema = _distribution_schema()
    evidence_description = (
        "The sixteen supplied images are eight close chronological pairs T0a/T0b through "
        "T7a/T7b. Return one value per pair/bin, so every output array still has length eight."
        if evidence_mode == "paired16" else
        "The eight supplied frames are T0..T7 in chronological order."
    )
    prompt = f"""TASK: INDEPENDENT_NODE_MATCHING
{evidence_description} They are the complete visual field for this experiment. Inspect them
for exactly ONE visual concept. Do not infer or
search for a larger event template and do not assume related concepts exist.

Concept:
{json.dumps(_node_text(node, 'N'), ensure_ascii=False, indent=2)}

Return JSON only in this exact shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Definitions:
- `presence_probability`: P(the concept is visibly present | these frames, this concept only).
- `location_distribution_given_present`: eight non-negative values that sum to 1; temporal
  location conditional on the concept being present. If presence is very low, return a
  near-uniform distribution rather than inventing a location.
- `evidence_quality_by_bin`: direct visual correspondence quality in [0,1] for each frame.
- `null_probability`: probability that no adequate visible support exists.

Describe only directly visible evidence. Filename, category, graph membership, other nodes,
and expected answer are unavailable."""
    assert_blind(prompt)
    return prompt


def _frame_distribution(values: list) -> list[float]:
    if not values:
        return [0.125] * 8
    if len(values) % 8:
        raise ValueError("initial OT slots cannot be mapped to eight frame bins")
    slots = len(values) // 8
    masses = [sum(values[i * slots:(i + 1) * slots]) for i in range(8)]
    total = sum(masses)
    return [v / total for v in masses] if total > 0 else [0.125] * 8


def _initial_ot_payload(graph: GraphTemplateV2, standalone_nodes: Mapping[str, Mapping[str, Any]], initial_ot: Mapping[str, Any], *, legacy_slot_layout: bool = False) -> tuple[list[dict], dict[str, str]]:
    id_to_key = {f"N{index}": node.key for index, node in enumerate(graph.nodes)}
    assignments = {
        str(item.get("node_key")): item
        for item in initial_ot.get("assignments", [])
        if isinstance(item, Mapping)
    }
    diagnostics = initial_ot.get("diagnostics", {}) if isinstance(initial_ot.get("diagnostics"), Mapping) else {}
    location_by_node = diagnostics.get("node_location", {}) if isinstance(diagnostics.get("node_location"), Mapping) else {}
    null_by_node = diagnostics.get("null_mass_by_node", {}) if isinstance(diagnostics.get("null_mass_by_node"), Mapping) else {}
    nodes = []
    for anonymous_id, node in zip(id_to_key, graph.nodes):
        standalone = standalone_nodes.get(node.key, {})
        nodes.append({
            **_node_text(node, anonymous_id),
            "standalone_estimate": {
                "presence_probability": standalone.get("presence", 0.0),
                "location_distribution_given_present": standalone.get("location_distribution_given_present", [0.125] * 8),
                "evidence_quality_by_bin": standalone.get("evidence_quality_by_bin", standalone.get("bin_affinity", [0.0] * 8)),
                "best_bin": standalone.get("best_bin"),
                "visible_evidence": standalone.get("visible_evidence", ""),
            },
            "initial_ot_estimate": {
                "presence_probability": initial_ot.get("node_presence", {}).get(node.key, 0.0),
                "expected_time": initial_ot.get("expected_time", {}).get(node.key, 0.0),
                "assigned_evidence": assignments.get(node.key, {}).get("assigned", "NULL") if legacy_slot_layout else str(assignments.get(node.key, {}).get("assigned", "NULL")).split("#")[0],
                "location_distribution": location_by_node.get(node.key, []) if legacy_slot_layout else _frame_distribution(location_by_node.get(node.key, [])),
                "null_mass": null_by_node.get(node.key, 0.0),
            },
        })
    return nodes, id_to_key


def conditional_refinement_prompt(
    graph: GraphTemplateV2,
    standalone_nodes: Mapping[str, Mapping[str, Any]],
    initial_ot: Mapping[str, Any],
    evidence_mode: str = "frames8",
    *,
    legacy_slot_layout: bool = False,
) -> tuple[str, dict[str, str]]:
    nodes, id_to_key = _initial_ot_payload(graph, standalone_nodes, initial_ot, legacy_slot_layout=legacy_slot_layout)
    node_schema = {
        anonymous_id: {
            **_distribution_schema(),
            "context_increases_from": [],
            "context_decreases_from": [],
            "probability_update_reason": "",
        }
        for anonymous_id in id_to_key
    }
    schema = {
        "graph_coherence": 0.0,
        "episode_span_bins": [],
        "episode_summary": "",
        "nodes": node_schema,
    }
    evidence_description = (
        "The sixteen images are eight close chronological pairs T0a/T0b through T7a/T7b; "
        "all returned temporal arrays contain one value per pair/bin."
        if evidence_mode == "paired16" else
        "The eight supplied frames are T0..T7 in chronological order."
    )
    prompt = f"""TASK: CONDITIONAL_NODE_SET_REFINEMENT_AFTER_INITIAL_OT
{evidence_description} A first-stage independent
analysis and a first optimal-transport allocation have already produced tentative node
locations. Treat the listed semantic nodes as one candidate explanation and refine each
node using the other nodes' tentative existence and locations.

Do not infer or output a fixed relation schema. The graph semantics are joint
conditional inference over one shared evidence field:
- decide whether the initial allocation is visually justified;
- re-localize a node when another node reveals the correct episode;
- suppress a standalone match when it belongs to an unrelated moment;
- allow NULL when the node is absent or not observable;
- do not force nodes to be in different frames when distinct concepts genuinely co-occur.

Candidate joint semantics: {graph.joint_semantics}
Initial state:
{json.dumps(nodes, ensure_ascii=False, indent=2)}

For each node return P(node | video, all other listed nodes and their initial OT state).
`context_increases_from` and `context_decreases_from` contain anonymous node IDs only. They
mean conditional probability influence, not physical causality.

Return JSON only in this exact shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

`location_distribution_given_present` must sum to 1. `evidence_quality_by_bin` measures only
direct visual support. `graph_coherence` is the probability that the node set describes one
compatible episode rather than unrelated moments. Use only visible evidence. Filename,
class label, graph polarity and expected answer are unavailable."""
    if not legacy_slot_layout:
        prompt += "\nTEMPORAL CONTRACT: All input and output location/quality arrays have exactly eight entries, one per T0..T7. Initial OT slot mass has already been summed into frame bins. Internal capacity slots are NOT extra frames. best_bin is null or 0..7; episode_span_bins contains only indices 0..7. Never return 16 or 17 temporal entries."
    assert_blind(prompt)
    return prompt, id_to_key


def crowd_event_state_prompt(
    active_graph: GraphTemplateV2,
    aftermath_graph: GraphTemplateV2,
    standalone_nodes: Mapping[str, Mapping[str, Any]],
    initial_ot: Mapping[str, Mapping[str, Any]],
) -> str:
    def phase_payload(graph: GraphTemplateV2, phase_id: str) -> dict:
        return {
            "phase_id": phase_id,
            "joint_semantics": graph.joint_semantics,
            "falsifiers": list(graph.falsifiers),
            "initial_unary_ot": {
                "graph_score": initial_ot.get(graph.key, {}).get("graph_score", 0.0),
                "node_presence": initial_ot.get(graph.key, {}).get("node_presence", {}),
                "assignments": initial_ot.get(graph.key, {}).get("assignments", []),
            },
            "nodes": [
                {
                    "description": node.title,
                    "visual_cues": list(node.cue_bundle),
                    "standalone_presence": standalone_nodes.get(node.key, {}).get("presence", 0.0),
                    "standalone_best_bin": standalone_nodes.get(node.key, {}).get("best_bin"),
                    "standalone_visible_evidence": standalone_nodes.get(node.key, {}).get("visible_evidence", ""),
                }
                for node in graph.nodes
            ],
        }

    schema = {
        "state_probabilities": {
            "active_escalation": 0.0,
            "causally_linked_aftermath": 0.0,
            "benign_or_pre_event_context": 0.0,
            "none_or_unobservable": 0.0,
        },
        "transition_observed_probability": 0.0,
        "aftermath_causal_link_probability": 0.0,
        "same_episode_probability": 0.0,
        "active_transition_bins": [],
        "aftermath_evidence_bins": [],
        "benign_context_bins": [],
        "observed_transition_evidence": "",
        "aftermath_causal_evidence": "",
        "benign_counterfactual_evidence": "",
        "decision_reason": "",
        "uncertainty": 0.0,
    }
    phases = [
        phase_payload(active_graph, "active_escalation"),
        phase_payload(aftermath_graph, "causally_linked_aftermath"),
    ]
    prompt = f"""TASK: MUTUALLY_EXCLUSIVE_CROWD_EVENT_STATE_AUDIT_V3
The eight supplied frames T0..T7 are chronological. Audit two tentative phase matches and
decide which ONE event state best describes the directly visible evidence. The four state
probabilities must be non-negative and sum to 1.

States:
- `active_escalation`: a new violent or physically coercive crowd transition is visibly
  occurring inside T0..T7. Require a visible mechanism such as contact, striking, throwing,
  impact, barrier attack, forced dispersal, or a clear ordered change from tense/orderly to
  violent/disordered. Running, chasing, gesturing, crowd density, or camera motion alone is
  insufficient.
- `causally_linked_aftermath`: damage, debris, fire, smoke, or emergency response is visibly
  produced by, or directly tied to, the same event represented in T0..T7. Static old damage,
  a pre-existing cracked windshield, generic military presence, or background smoke is
  insufficient without a visible current-event link.
- `benign_or_pre_event_context`: tension, flight, chasing, crowd movement, military context,
  static damage, ceremonial activity, sports, stage effects, or other scene context is visible
  but no qualifying transition or current-event causal link is established.
- `none_or_unobservable`: the supplied frames do not support any phase strongly enough.

Text visible inside frames may provide context, but must never be the sole evidence for an
event state. Do not infer hidden contact or a transition between scene cuts. Evidence assigned
to the two positive phases must belong to one coherent episode.

Tentative phase evidence:
{json.dumps(phases, ensure_ascii=False, indent=2)}

Return JSON only in this exact shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

All probabilities are in [0,1]. Bin lists contain only integers 0..7. Use only directly
visible evidence. Filename, category, graph polarity, annotation and expected answer are
unavailable."""
    assert_blind(prompt)
    return prompt


def crowd_event_state_v4_prompt(
    active_graph: GraphTemplateV2,
    aftermath_graph: GraphTemplateV2,
    standalone_nodes: Mapping[str, Mapping[str, Any]],
    initial_ot: Mapping[str, Mapping[str, Any]],
    evidence_metadata: Mapping[str, Any] | None = None,
    motion_features: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, dict[str, str]]]:
    """One blind call returning state probabilities and phase-conditional node evidence."""
    mappings = {
        "active": {f"A{index}": node.key for index, node in enumerate(active_graph.nodes)},
        "aftermath": {f"F{index}": node.key for index, node in enumerate(aftermath_graph.nodes)},
    }

    def node_schema() -> dict:
        return {
            "presence_probability": 0.0,
            "null_probability": 0.0,
            "location_distribution_given_present": [0.0] * 8,
            "evidence_quality_by_bin": [0.0] * 8,
            "best_bin": 0,
            "visible_evidence": "",
            "probability_update_reason": "",
            "uncertainty": 0.0,
        }

    def phase_payload(name: str, graph: GraphTemplateV2) -> dict:
        reverse = {key: anonymous for anonymous, key in mappings[name].items()}
        return {
            "joint_semantics": graph.joint_semantics,
            "falsifiers": list(graph.falsifiers),
            "initial_unary_ot": {
                "graph_score": initial_ot.get(graph.key, {}).get("graph_score", 0.0),
                "node_presence": initial_ot.get(graph.key, {}).get("node_presence", {}),
                "assignments": initial_ot.get(graph.key, {}).get("assignments", []),
            },
            "nodes": [
                {
                    "id": reverse[node.key],
                    "description": node.title,
                    "visual_cues": list(node.cue_bundle),
                    "standalone_presence": standalone_nodes.get(node.key, {}).get("presence", 0.0),
                    "standalone_best_bin": standalone_nodes.get(node.key, {}).get("best_bin"),
                    "standalone_visible_evidence": standalone_nodes.get(node.key, {}).get("visible_evidence", ""),
                }
                for node in graph.nodes
            ],
        }

    schema = {
        "state_probabilities": {
            "active_or_ongoing_physical_escalation": 0.0,
            "causally_linked_aftermath": 0.0,
            "pre_event_tension_or_flight": 0.0,
            "benign_collective_activity": 0.0,
            "none_or_unobservable": 0.0,
        },
        "phase_coherence": {"active": 0.0, "aftermath": 0.0},
        "phase_nodes": {
            "active": {anonymous: node_schema() for anonymous in mappings["active"]},
            "aftermath": {anonymous: node_schema() for anonymous in mappings["aftermath"]},
        },
        "active_evidence_bins": [],
        "aftermath_evidence_bins": [],
        "pre_event_bins": [],
        "benign_bins": [],
        "visible_evidence": {"active": "", "aftermath": ""},
        "counterfactual_evidence": {"pre_event": "", "benign": ""},
        "decision_reason": "",
        "uncertainty": 0.0,
    }
    evidence = dict(evidence_metadata or {})
    observation = {
        "mode": evidence.get("mode", "frames"),
        "bin_labels": evidence.get("bin_labels", []),
        "pair_map": evidence.get("pair_map", []),
        "motion_features": dict(motion_features or {}),
    }
    phases = {
        "active": phase_payload("active", active_graph),
        "aftermath": phase_payload("aftermath", aftermath_graph),
    }
    prompt = f"""TASK: MUTUALLY_EXCLUSIVE_CROWD_EVENT_STATE_AUDIT_V4
The supplied visual evidence is chronological and grouped into eight temporal bins. In
`paired16` mode, Ta and Tb are a close adjacent pair belonging to the same bin T. First assign
one mutually exclusive event-state posterior, then condition every tentative phase node on the
whole phase explanation. Use only directly visible evidence.

States:
- `active_or_ongoing_physical_escalation`: a new onset OR ongoing directly visible physical
  mechanism is present, including striking, grappling, forceful coercion, throwing, impact,
  barrier attack, forced dispersal, or sustained physical confrontation. The onset need not
  occur after the first supplied frame.
- `causally_linked_aftermath`: damage, debris, fire, smoke, injury, or response is visibly tied
  to the same current episode. Static old damage and generic emergency or military context are
  insufficient.
- `pre_event_tension_or_flight`: tension, running, chasing, flight, or standoff is visible, but
  no direct physical mechanism or linked aftermath is observable.
- `benign_collective_activity`: ceremony, protest without physical escalation, sports, stage
  activity, ordinary crowd motion, or another benign collective explanation is supported.
- `none_or_unobservable`: the packet cannot support any state strongly enough.

The five probabilities must be non-negative and sum to one. They already encode transition,
causal-link, same-episode, and counterfactual reasoning; do not create a separate duplicate
confidence gate. For each phase node, return conditional presence, an eight-bin location
distribution given presence, and eight evidence-quality values. Phase-node evidence must belong
to one coherent episode. Frame text may provide context but cannot be sole event evidence. Do
not bridge unrelated scene cuts.

Observation metadata and neutral local motion measurements:
{json.dumps(observation, ensure_ascii=False, indent=2)}

Tentative phase evidence:
{json.dumps(phases, ensure_ascii=False, indent=2)}

Return JSON only in this exact shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

All probabilities are in [0,1]. Bin arrays contain exactly eight values and bin lists contain
integers 0..7. Filename, category, annotation, graph polarity, and expected answer are unavailable."""
    assert_blind(prompt)
    return prompt, mappings


def crowd_event_state_v5_prompt(
    active_graph: GraphTemplateV2,
    aftermath_graph: GraphTemplateV2,
    standalone_nodes: Mapping[str, Mapping[str, Any]],
    initial_ot: Mapping[str, Mapping[str, Any]],
    evidence_metadata: Mapping[str, Any] | None = None,
    motion_features: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, dict[str, str]]]:
    """Blind dual-head prompt separating event context from current-window occupancy."""
    mappings = {
        "active": {f"A{index}": node.key for index, node in enumerate(active_graph.nodes)},
        "aftermath": {f"F{index}": node.key for index, node in enumerate(aftermath_graph.nodes)},
    }

    def node_schema() -> dict:
        return {
            "presence_probability": 0.0, "null_probability": 0.0,
            "location_distribution_given_present": [0.0] * 8,
            "evidence_quality_by_bin": [0.0] * 8, "best_bin": 0,
            "visible_evidence": "", "probability_update_reason": "", "uncertainty": 0.0,
        }

    def phase_payload(name: str, graph: GraphTemplateV2) -> dict:
        reverse = {key: anonymous for anonymous, key in mappings[name].items()}
        return {
            "joint_semantics": graph.joint_semantics,
            "falsifiers": list(graph.falsifiers),
            "initial_unary_ot": {
                "graph_score": initial_ot.get(graph.key, {}).get("graph_score", 0.0),
                "node_presence": initial_ot.get(graph.key, {}).get("node_presence", {}),
                "assignments": initial_ot.get(graph.key, {}).get("assignments", []),
            },
            "nodes": [{
                "id": reverse[node.key], "description": node.title,
                "visual_cues": list(node.cue_bundle),
                "standalone_presence": standalone_nodes.get(node.key, {}).get("presence", 0.0),
                "standalone_best_bin": standalone_nodes.get(node.key, {}).get("best_bin"),
                "standalone_visible_evidence": standalone_nodes.get(node.key, {}).get("visible_evidence", ""),
            } for node in graph.nodes],
        }

    schema = {
        "event_context_state_probabilities": {
            "active": 0.0, "aftermath": 0.0, "pre": 0.0, "benign": 0.0, "none": 0.0,
        },
        "current_window_active_occupancy_probability": 0.0,
        "current_window_normal_confound_probability": 0.0,
        "normal_confound_probabilities": {
            "structured_sport_or_play": 0.0,
            "peaceful_protest_or_ceremony": 0.0,
            "assistance_or_rescue": 0.0,
            "ordinary_object_interaction": 0.0,
            "staged_or_performed_action": 0.0,
            "none": 0.0,
        },
        "aftermath_context_probability": 0.0,
        "aftermath_current_window_occupancy_support_probability": 0.0,
        "phase_coherence": {"active": 0.0, "aftermath": 0.0},
        "phase_nodes": {
            "active": {anonymous: node_schema() for anonymous in mappings["active"]},
            "aftermath": {anonymous: node_schema() for anonymous in mappings["aftermath"]},
        },
        "direct_active_bins": [], "context_only_bins": [], "normal_confound_bins": [],
        "active_evidence_bins": [], "aftermath_evidence_bins": [],
        "pre_event_bins": [], "benign_bins": [],
        "visible_evidence": {"active": "", "aftermath": ""},
        "counterfactual_evidence": {"pre_event": "", "benign": ""},
        "normal_confound_evidence": {"type": "", "visible_evidence": "", "reason": ""},
        "decision_reason": "", "uncertainty": 0.0,
    }
    evidence = dict(evidence_metadata or {})
    observation = {
        "mode": evidence.get("mode", "frames"), "bin_labels": evidence.get("bin_labels", []),
        "pair_map": evidence.get("pair_map", []), "motion_features": dict(motion_features or {}),
    }
    phases = {
        "active": phase_payload("active", active_graph),
        "aftermath": phase_payload("aftermath", aftermath_graph),
    }
    prompt = f"""TASK: CROWD_EVENT_CONTEXT_AND_CURRENT_OCCUPANCY_AUDIT_V5
The supplied visual evidence is chronological and grouped into eight temporal bins. Produce two
separate judgments from the same evidence packet.

Head A, event context, assigns probabilities that sum to one across active, aftermath, pre,
benign, and none. This head describes which phase or context the packet belongs to.

Head B, current-window active occupancy, estimates whether this window itself contains a directly
visible active anomalous mechanism: striking, grappling, forceful coercion, throwing, impact,
barrier attack, forced dispersal, or sustained physical confrontation. Event membership alone is
not enough. Forensic evidence, debris, police response, emergency presence, smoke, fire, injury,
or old damage may support aftermath context but do not by themselves prove current-window active
anomaly occupancy. Put those observations in context_only_bins unless a direct active mechanism is
also visible in the current packet.

For this task, `barrier attack` requires visible force intended to breach, overturn, burn, destroy,
or weaponize a barrier/object as part of harmful confrontation. Merely standing near, opening,
touching, pushing, or reaching into a public object without visible damage, harmful consequence,
or confrontation is not sufficient. Mere body contact is also insufficient when a structured sport,
play, assistance, rescue, ceremony, or peaceful protest visibly explains it.

Head C, current-window normal confound, independently estimates whether a directly visible coherent
normal mechanism explains the active-looking observations. Its subtype probabilities sum to one:
- `structured_sport_or_play`: rules, rink/field/court, uniforms, game roles, or ordinary play;
- `peaceful_protest_or_ceremony`: marching, signs, gathering, or coordinated activity without a
  visible assault or destructive mechanism;
- `assistance_or_rescue`: helping, protecting, medical response, or safety-motivated restraint;
- `ordinary_object_interaction`: ordinary opening, moving, carrying, inspecting, or using an object
  without visible damage or harmful confrontation;
- `staged_or_performed_action`: acting, stunt, rehearsal, stage, filming, or performance evidence;
- `none`: no positively visible normal mechanism explains the packet.
`current_window_normal_confound_probability` must be high only when positive visual evidence supports
one of those explanations; absence of anomaly alone is not enough. Head B and Head C are reported
separately so calibration can measure their competition rather than forcing one from the other.

`aftermath_current_window_occupancy_support_probability` asks whether aftermath evidence is
accompanied by direct current-window evidence that the anomalous mechanism still occupies this
window. It must be low for calm post-event investigation, static debris, old damage, or response
activity without ongoing anomalous action.

Condition every phase node on the whole phase explanation. Evidence for a phase must belong to one
coherent episode. Text may provide context but cannot be the sole event evidence. Do not bridge
unrelated scene cuts and do not infer hidden contact.

Observation metadata and neutral local motion measurements:
{json.dumps(observation, ensure_ascii=False, indent=2)}

Tentative phase evidence:
{json.dumps(phases, ensure_ascii=False, indent=2)}

Return JSON only in this exact shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

All probabilities are in [0,1]. Bin arrays contain exactly eight values and bin lists contain
integers 0..7. Filename, category, annotation, graph polarity, and expected answer are unavailable."""
    assert_blind(prompt)
    return prompt, mappings


def conditional_graph_prompt(graph: GraphTemplateV2) -> tuple[str, dict[str, str]]:
    """Backward-compatible prompt helper used by old tests/tools.

    New live code must call :func:`conditional_refinement_prompt` with the initial OT state.
    """
    standalone = {
        node.key: {
            "presence": 0.5,
            "location_distribution_given_present": [0.125] * 8,
            "evidence_quality_by_bin": [0.5] * 8,
            "best_bin": None,
            "visible_evidence": "not supplied in compatibility mode",
        }
        for node in graph.nodes
    }
    initial = {
        "node_presence": {node.key: 0.5 for node in graph.nodes},
        "expected_time": {node.key: 0.5 for node in graph.nodes},
        "assignments": [],
        "diagnostics": {},
    }
    return conditional_refinement_prompt(graph, standalone, initial)


def graph_shortlist_prompt(catalog: Mapping[str, GraphTemplateV2], top_k_abnormal: int, top_k_normal: int) -> tuple[str, dict[str, str]]:
    abnormal = sorted((g for g in catalog.values() if g.polarity == "abnormal"), key=lambda g: g.key)
    normal = sorted((g for g in catalog.values() if g.polarity == "normal"), key=lambda g: g.key)
    id_to_key = {
        **{f"A{index}": graph.key for index, graph in enumerate(abnormal)},
        **{f"N{index}": graph.key for index, graph in enumerate(normal)},
    }

    def compact(graph: GraphTemplateV2, candidate_id: str) -> dict:
        return {
            "id": candidate_id,
            "family": graph.family,
            "joint_semantics": graph.joint_semantics,
            "applicability": list(graph.applicability),
            "falsifiers": list(graph.falsifiers),
            "canonical_factors": list(graph.canonical_factors),
            "nodes": [
                {"description": node.title, "visual_cues": list(node.cue_bundle[:3])}
                for node in graph.nodes
            ],
        }

    schema = {
        "abnormal_candidates": [{"id": "A0", "visual_support": 0.0, "visible_reason": ""}],
        "normal_candidates": [{"id": "N0", "visual_support": 0.0, "visible_reason": ""}],
    }
    prompt = f"""TASK: BLIND_GRAPH_CANDIDATE_SHORTLIST
Select candidate node-set explanations before either matching method runs. The same shortlist
will be used by all methods. Use only visible frames; do not infer from filename or expected
answer.

Return exactly {max(1, int(top_k_abnormal))} abnormal and {max(2, int(top_k_normal))}
    normal/confound candidates. Check applicability and falsifiers against direct visual evidence.
    Normal candidates must be family-diverse when visually plausible:
prefer at most one candidate from a family before selecting a second from that family.

Abnormal candidates:
{json.dumps([compact(g, f'A{i}') for i, g in enumerate(abnormal)], ensure_ascii=False, indent=2)}

Normal/confound candidates:
{json.dumps([compact(g, f'N{i}') for i, g in enumerate(normal)], ensure_ascii=False, indent=2)}

Return JSON only and use candidate IDs exactly:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""
    assert_blind(prompt)
    return prompt, id_to_key


def graph_shortlist_repair_prompt(previous: Mapping[str, Any], id_to_key: Mapping[str, str], top_k_abnormal: int, top_k_normal: int) -> str:
    schema = {
        "abnormal_candidates": [{"id": "A0", "visual_support": 0.0, "visible_reason": ""}],
        "normal_candidates": [{"id": "N0", "visual_support": 0.0, "visible_reason": ""}],
    }
    prompt = f"""TASK: REPAIR_BLIND_GRAPH_SHORTLIST
The previous candidate shortlist was incomplete or duplicated. Re-read the same eight frames
and return exactly {top_k_abnormal} distinct abnormal IDs and {top_k_normal} distinct normal
IDs. Use only IDs from this allowed map and keep the normal candidates family-diverse.

Allowed IDs:
{json.dumps(dict(id_to_key), ensure_ascii=False, indent=2)}

Previous response:
{json.dumps(dict(previous), ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""
    assert_blind(prompt)
    return prompt


def verifier_prompt(method_x: dict, method_y: dict) -> str:
    schema = {
        "preferred_method": "X|Y|tie",
        "confidence": 0.0,
        "visual_reason": "",
        "method_x_failure": "",
        "method_y_failure": "",
    }
    prompt = f"""TASK: BLIND_MATCHING_VERIFIER
Two anonymous methods analyzed the same eight frames. Judge which result is more faithfully
grounded. You do not know the dataset label, method identity or which one was scored correct.
Both summaries use the same schema. Prefer a method only when its node probabilities,
locations, NULL assignments and graph competition are visibly better justified.

METHOD X:
{json.dumps(method_x, ensure_ascii=False, indent=2)}

METHOD Y:
{json.dumps(method_y, ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""
    assert_blind(prompt)
    return prompt
