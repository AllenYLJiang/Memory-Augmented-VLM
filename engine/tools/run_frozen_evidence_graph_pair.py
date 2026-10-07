#!/usr/bin/env python3
"""B1 paired target-graph recomputation with frozen images, shortlist and non-target scores."""
from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent import futures
from pathlib import Path
from typing import Any, Mapping

import matching
from common import iter_jsonl, read_json, stable_sha1, write_json, write_jsonl
from competition import _aggregate
from crowd_event_state import parse_event_state_response, score_event_state
from crowd_event_state_v4 import (
    METHODS as V4_METHODS,
    bounded_margin_residual,
    load_calibration,
    parse_event_state_v4_response,
    parse_event_state_v5_response,
    score_phase_aware_occupancy,
    score_method_blend,
)
from crowd_motion_features import compute_crowd_motion_features
from graph_catalog import read_catalog_json
from live_matching import _evidence_layout, _parse_independent, _parse_joint, _result_dict, _unary
from prompts import (
    conditional_refinement_prompt,
    crowd_event_state_prompt,
    crowd_event_state_v4_prompt,
    crowd_event_state_v5_prompt,
    independent_node_prompt,
)
from schemas import WindowCase
from validate_graph_candidates import _delta, _metrics
from vlm_runtime import CachedVideoVLM, RuntimeConfig


METHODS = ("conditional_rowmax", "conditional_ot_no_coherence", "conditional_ot_full")


def _records(path: Path) -> dict[str, dict]:
    if path.is_dir():
        merged = path / "ot_window_results.jsonl"
        if merged.is_file():
            path = merged
        else:
            return {
                str(row.get("segment_key")): row
                for item in sorted((path / "records").glob("*.json"))
                if isinstance((row := read_json(item, None)), dict)
            }
    return {str(row.get("segment_key")): row for row in iter_jsonl(path)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _replace_competition(base: Mapping[str, Any], key: str, replacement: Mapping[str, Any], method: str, polarity: str) -> dict:
    results = {name: dict(value) for name, value in base.get("graph_results", {}).get(method, {}).items()}
    results[key] = dict(replacement)
    shortlist = base.get("graph_candidates", {})
    abnormal_keys = list(shortlist.get("selected_abnormal", []))
    normal_keys = list(shortlist.get("selected_normal", []))
    if key not in abnormal_keys and key not in normal_keys:
        (abnormal_keys if polarity == "abnormal" else normal_keys).append(key)
    abnormal_scores = [float(results[name].get("graph_score", 0.0)) for name in abnormal_keys if name in results]
    normal_scores = [float(results[name].get("graph_score", 0.0)) for name in normal_keys if name in results]
    frozen = base.get("competitions", {}).get(method, {})
    aggregation = str(frozen.get("aggregation", "logmeanexp"))
    temperature = float(frozen.get("temperature", 0.1) or 0.1)
    threshold = float(frozen.get("decision_margin_threshold", 0.03) or 0.03)
    abnormal = _aggregate(abnormal_scores, aggregation, temperature)
    normal = _aggregate(normal_scores, aggregation, temperature)
    margin = abnormal - normal
    return {
        "method": method, "aggregation": aggregation, "temperature": temperature,
        "decision_margin_threshold": threshold, "best_abnormal_score": abnormal,
        "best_normal_score": normal, "margin": margin,
        "decision": "abnormal" if margin > threshold else "normal",
        "y_pred": int(margin > threshold), "frozen_non_target_results": True,
    }


def _select_phase_result(candidate_targets: Mapping[str, Mapping[str, Any]], method: str, aggregation: str) -> tuple[str, dict]:
    """Select the phase explanation that occupies the single frozen target slot."""
    if aggregation != "max":
        raise ValueError(f"unsupported candidate phase aggregation: {aggregation}")
    choices = [
        (str(graph_key), dict(value["results"][method]))
        for graph_key, value in candidate_targets.items()
    ]
    if not choices:
        raise ValueError("candidate phase ensemble is empty")
    return max(choices, key=lambda item: (float(item[1].get("graph_score", 0.0)), item[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--candidate-catalog", required=True, type=Path)
    parser.add_argument("--target-graph-key", default="crowd_escalation_chain")
    parser.add_argument("--candidate-target-graph-keys", nargs="*", default=[])
    parser.add_argument("--candidate-phase-aggregation", choices=("max",), default="max")
    parser.add_argument(
        "--candidate-scoring-mode",
        choices=("phase_ensemble_v2", "event_state_v3", "event_state_v4", "event_state_v5"),
        default="phase_ensemble_v2",
    )
    parser.add_argument("--event-state-active-key", default="")
    parser.add_argument("--event-state-aftermath-key", default="")
    parser.add_argument("--window-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--evidence-slots-per-bin", type=int, default=2)
    parser.add_argument("--coherence-weight", type=float, default=0.25)
    parser.add_argument("--crowd-evidence-mode", choices=("frames8", "paired16"), default="frames8")
    parser.add_argument("--crowd-pair-gap", type=int, default=2)
    parser.add_argument("--state-calibration-file", type=Path)
    parser.add_argument(
        "--state-refinement-mode",
        choices=("method_blend", "margin_residual", "phase_aware_occupancy"),
        default="method_blend",
    )
    parser.add_argument("--require-frozen-calibration", action="store_true")
    parser.add_argument("--use-motion-features", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    baseline = _records(args.baseline_records)
    base_catalog, candidate_catalog = read_catalog_json(args.base_catalog), read_catalog_json(args.candidate_catalog)
    base_graph = base_catalog.get(args.target_graph_key)
    candidate_keys = list(dict.fromkeys(args.candidate_target_graph_keys or [args.target_graph_key]))
    candidate_graphs = [candidate_catalog.get(key) for key in candidate_keys]
    missing_candidate_keys = [key for key, graph in zip(candidate_keys, candidate_graphs) if graph is None]
    if base_graph is None or missing_candidate_keys:
        details = ", ".join(missing_candidate_keys) if missing_candidate_keys else args.target_graph_key
        raise SystemExit(f"target graph(s) missing from catalogs: {details}")
    candidate_graphs = [graph for graph in candidate_graphs if graph is not None]
    if any(graph.polarity != base_graph.polarity for graph in candidate_graphs):
        raise SystemExit("all candidate phase graphs must have the same polarity as the frozen target slot")
    event_active_graph = event_aftermath_graph = None
    if args.candidate_scoring_mode in {"event_state_v3", "event_state_v4", "event_state_v5"}:
        if len(candidate_graphs) != 2:
            raise SystemExit(f"{args.candidate_scoring_mode} requires exactly two candidate phase graphs")
        active_key = args.event_state_active_key or candidate_keys[0]
        aftermath_key = args.event_state_aftermath_key or candidate_keys[1]
        event_active_graph = candidate_catalog.get(active_key)
        event_aftermath_graph = candidate_catalog.get(aftermath_key)
        if event_active_graph is None or event_aftermath_graph is None or active_key == aftermath_key:
            raise SystemExit("event-state active/aftermath graph keys are invalid")
    calibration = load_calibration(args.state_calibration_file) if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else {}
    if args.candidate_scoring_mode == "event_state_v5" and args.state_calibration_file is None:
        calibration = {
            **calibration,
            "version": "crowd_event_state_v5_signed_confound_trace_only_v2",
            "refinement_mode": "phase_aware_occupancy",
            "beta_active": 0.0, "beta_aftermath": 0.0,
            "beta_normal_confound": 0.0, "residual_bound": 0.0,
            "phase_specific_weights_frozen": False,
        }
    if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}:
        if tuple(METHODS) != tuple(V4_METHODS):
            raise SystemExit("V4 method contract does not match frozen runner methods")
        if args.require_frozen_calibration and not bool(calibration.get("frozen")):
            raise SystemExit(f"held-out {args.candidate_scoring_mode} validation requires a calibration file with frozen=true")
        configured_mode = str(calibration.get("refinement_mode", args.state_refinement_mode))
        if configured_mode != args.state_refinement_mode:
            raise SystemExit(
                f"state refinement mode mismatch: CLI={args.state_refinement_mode} calibration={configured_mode}"
            )
        if args.candidate_scoring_mode == "event_state_v5" and args.state_refinement_mode != "phase_aware_occupancy":
            raise SystemExit("event_state_v5 requires --state-refinement-mode phase_aware_occupancy")
    candidate_node_keys = set().union(*(set(graph.node_keys) for graph in candidate_graphs))
    manifest = list(iter_jsonl(args.window_manifest))
    manifest_by_key = {str(row.get("segment_key", "")): row for row in manifest if row.get("segment_key")}
    keys = [str(row.get("segment_key", "")) for row in manifest if row.get("segment_key")]
    if args.max_windows > 0:
        keys = keys[:args.max_windows]
    if args.candidate_scoring_mode == "event_state_v5":
        from temporal_contract import window_errors
        invalid = {key: window_errors(baseline[key]) for key in keys if key in baseline}
        invalid = {key: issues for key, issues in invalid.items() if issues}
        if invalid:
            raise SystemExit(f"frozen V5 baseline has {len(invalid)} invalid temporal records; repair Step 2 first: {next(iter(invalid))}")
    if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} and args.require_frozen_calibration:
        calibration_groups = {str(value) for value in calibration.get("source_groups", [])}
        manifest_groups = {
            str(row.get("source_group", "")) for row in manifest
            if str(row.get("segment_key", "")) in set(keys) and row.get("source_group")
        }
        overlap = sorted(calibration_groups & manifest_groups)
        if overlap:
            raise SystemExit(f"calibration/validation source-group overlap; first={overlap[0]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def runtime() -> CachedVideoVLM:
        value = getattr(local, "runtime", None)
        if value is None:
            value = CachedVideoVLM(RuntimeConfig(
                code_dir=args.code_dir, cache_dir=args.out_dir / "cache", model=args.model,
                mock=args.mock,
                evidence_mode="paired16" if args.crowd_evidence_mode == "paired16" else "frames",
                evidence_frames=8, pair_gap=args.crowd_pair_gap, cache_salt=args.cache_salt,
            ))
            local.runtime = value
        return value

    def process(segment: str) -> tuple[dict | None, dict]:
        base = baseline.get(segment)
        if not base:
            return None, {"segment_key": segment, "eligible": False, "reason": "missing_baseline_record"}
        manifest_row = manifest_by_key.get(segment, {})
        selected = set(base.get("graph_candidates", {}).get("selected_abnormal", [])) | set(base.get("graph_candidates", {}).get("selected_normal", []))
        if args.target_graph_key not in selected:
            frozen_competitions = {
                method: dict(base.get("competitions", {}).get(method, {})) for method in METHODS
            }
            result = {
                "version": "frozen_evidence_target_pair_full_packet_noop_v1",
                "segment_key": segment,
                "video_id": base.get("video_id"), "video_path": base.get("video_path"),
                "start_frame": base.get("start_frame"), "end_frame": base.get("end_frame"),
                "y_true": base.get("y_true"), "gt": base.get("gt", {}),
                "metric_eligible": base.get("metric_eligible", True),
                "evaluation_stratum": manifest_row.get("stratum", manifest_row.get("crowd_v4_stratum", "")),
                "event_phase": manifest_row.get("event_phase", ""),
                "source_group": manifest_row.get("source_group", ""),
                "split_role": manifest_row.get("split_role", manifest_row.get("role", "")),
                "eligible_for_target_effect": False, "full_packet_eligible": True,
                "target_exposure_reason": "target_not_exposed",
                "candidate_scoring_mode": args.candidate_scoring_mode,
                "frozen_shortlist": base.get("graph_candidates", {}),
                "frozen_non_target_graph_results": True,
                "base_competitions": frozen_competitions,
                "candidate_competitions": {method: dict(value) for method, value in frozen_competitions.items()},
                "paired_delta": {method: 0.0 for method in METHODS},
                "candidate_event_state": None,
                "score_contract": {
                    "method_matched": True, "repeated_gate_count": 0,
                    "refinement_mode": "full_packet_noop", "noop_reason": "target_not_exposed",
                    "phase_specific_weights_frozen": bool(calibration.get("frozen", False)),
                    "method_details": {},
                },
                "state_calibration": {
                    "path": str(args.state_calibration_file) if args.state_calibration_file else "",
                    "sha256": str(calibration.get("sha256", "")),
                    "calibration_id": str(calibration.get("calibration_id", "trace_only")),
                    "frozen": bool(calibration.get("frozen", False)),
                    "source_groups_sha256": str(calibration.get("source_groups_sha256", "")),
                    "source_groups": list(calibration.get("source_groups", [])),
                } if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else None,
                "candidate_only_independent_calls": 0, "fresh_conditional_calls": 0,
            }
            return result, {
                "segment_key": segment, "eligible": False, "full_packet_eligible": True,
                "reason": "target_not_in_frozen_shortlist", "candidate_only_nodes": 0,
                "fresh_conditional_calls": 0,
            }
        frozen_evidence = base.get("evidence") or {}
        if frozen_evidence.get("mode") != "frames" or (not args.mock and not frozen_evidence.get("image_paths")):
            return None, {"segment_key": segment, "eligible": False, "reason": "missing_frozen_frame_evidence"}
        case = WindowCase(
            segment_key=segment, video_id=str(base.get("video_id", "")), video_path=str(base.get("video_path", "")),
            start_frame=int(base.get("start_frame", 0)), end_frame=int(base.get("end_frame", 0)),
            y_true=int(base.get("y_true", 0)), source_record=dict(base),
        )
        paired_observation = args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} and args.crowd_evidence_mode == "paired16"
        evidence = runtime().evidence_metadata(case) if paired_observation else dict(frozen_evidence)
        independent = (
            {} if paired_observation else
            {key: dict(value) for key, value in base.get("independent_node_calls", {}).items()}
        )
        missing_base = [node.key for node in base_graph.nodes if node.key not in independent]
        if missing_base and not paired_observation:
            return None, {"segment_key": segment, "eligible": False, "reason": "missing_baseline_independent_nodes", "details": ";".join(missing_base)}
        new_node_calls = 0
        candidate_nodes = {}
        graphs_requiring_independent = ([base_graph] + candidate_graphs) if paired_observation else candidate_graphs
        for graph in graphs_requiring_independent:
            for node in graph.nodes:
                candidate_nodes.setdefault(node.key, node)
        for node_key, node in candidate_nodes.items():
            if node_key not in independent:
                response = runtime().request_json(
                    case=case,
                    prompt=independent_node_prompt(node, args.crowd_evidence_mode),
                    namespace=f"paired_candidate_only/{args.crowd_evidence_mode}/{node.key}",
                    mock_spec={"kind": "independent", "node_key": node.key}, evidence_override=evidence,
                )
                independent[node.key] = _parse_independent(response, node.key)
                new_node_calls += int(not response.get("cache_hit"))
        evidence_ids, evidence_centers = _evidence_layout(args.evidence_slots_per_bin)

        def target(graph, variant: str):
            unary = _unary(graph.nodes, independent, args.evidence_slots_per_bin)
            m2 = matching.match_graph_unary_ot(graph, unary, evidence_centers)
            prompt, mapping = conditional_refinement_prompt(
                graph, independent, m2.to_dict(), args.crowd_evidence_mode,
            )
            response = runtime().request_json(
                case=case, prompt=prompt, namespace=f"paired_target/{variant}/{graph.key}",
                mock_spec={"kind": "joint", "graph_key": graph.key, "node_keys": graph.node_keys, "initial_presence": m2.node_presence},
                evidence_override=evidence,
            )
            cond, trace = _parse_joint(response, graph, mapping, evidence_ids, args.evidence_slots_per_bin)
            trace["initial_ot"] = m2.to_dict()
            m3a = matching.match_graph_conditional_rowmax(graph, unary, cond, evidence_centers)
            m3b = matching.match_graph_conditional_ot(graph, unary, cond, evidence_centers, use_coherence=False, method_name="conditional_ot_no_coherence")
            m3c = matching.match_graph_conditional_ot(graph, unary, cond, evidence_centers, use_coherence=True, coherence_weight=args.coherence_weight, method_name="conditional_ot_full")
            return {
                "unary_ot": _result_dict(m2), "conditional_rowmax": _result_dict(m3a),
                "conditional_ot_no_coherence": _result_dict(m3b), "conditional_ot_full": _result_dict(m3c),
            }, trace, int(not response.get("cache_hit")), {
                "prompt_sha1": stable_sha1(prompt, size=40),
                "response": response.get("parsed", {}),
                "cache_path": response.get("cache_path"),
                "cache_hit": bool(response.get("cache_hit")),
            }

        base_target, base_trace, base_calls, base_call_meta = target(base_graph, "base")
        base_comp = {method: _replace_competition(base, args.target_graph_key, base_target[method], method, base_graph.polarity) for method in METHODS}
        candidate_targets = {}
        candidate_calls = 0
        candidate_event_state = None
        candidate_target_prompt_sha = {}
        candidate_target_response = {}
        score_contract = None
        motion_features = None
        if args.candidate_scoring_mode == "event_state_v3":
            assert event_active_graph is not None and event_aftermath_graph is not None
            initial_ot = {}
            for graph in candidate_graphs:
                unary = _unary(graph.nodes, independent, args.evidence_slots_per_bin)
                m2 = matching.match_graph_unary_ot(graph, unary, evidence_centers)
                m2_result = _result_dict(m2)
                initial_ot[graph.key] = m2_result
                candidate_targets[graph.key] = {
                    "graph": graph.key,
                    "results": {"unary_ot": m2_result},
                    "joint_trace": None,
                    "response": None,
                }
            prompt = crowd_event_state_prompt(
                event_active_graph, event_aftermath_graph, independent, initial_ot,
            )
            response = runtime().request_json(
                case=case, prompt=prompt, namespace="paired_event_state_v3",
                mock_spec={"kind": "crowd_event_state"}, evidence_override=evidence,
            )
            candidate_calls = int(not response.get("cache_hit"))
            candidate_event_state = parse_event_state_response(response)
            event_result, event_score = score_event_state(
                candidate_event_state, initial_ot,
                event_active_graph.key, event_aftermath_graph.key, args.target_graph_key,
            )
            candidate_phase_winners = {method: event_score["winner"] for method in METHODS}
            candidate_replacements = {
                method: {**event_result, "method": f"event_state_gated_unary_ot_v3/{method}"}
                for method in METHODS
            }
            candidate_target_prompt_sha = {"event_state_v3": stable_sha1(prompt, size=40)}
            candidate_target_response = {
                "event_state_v3": {
                    "prompt_sha1": stable_sha1(prompt, size=40),
                    "response": response.get("parsed", {}),
                    "cache_path": response.get("cache_path"),
                    "cache_hit": bool(response.get("cache_hit")),
                }
            }
            candidate_event_state["score"] = event_score
            candidate_event_state["initial_unary_ot"] = initial_ot
        elif args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}:
            assert event_active_graph is not None and event_aftermath_graph is not None
            phase_graphs = {
                "active": event_active_graph,
                "aftermath": event_aftermath_graph,
            }
            initial_ot: dict[str, dict] = {}
            phase_unary = {}
            for phase, graph in phase_graphs.items():
                unary = _unary(graph.nodes, independent, args.evidence_slots_per_bin)
                phase_unary[phase] = unary
                m2 = matching.match_graph_unary_ot(graph, unary, evidence_centers)
                initial_ot[graph.key] = _result_dict(m2)

            motion_features = (
                compute_crowd_motion_features(case, bins=8)
                if args.use_motion_features and not args.mock
                else {"available": False, "reason": "disabled_or_mock"}
            )
            prompt_builder = (
                crowd_event_state_v5_prompt
                if args.candidate_scoring_mode == "event_state_v5"
                else crowd_event_state_v4_prompt
            )
            prompt, phase_mappings = prompt_builder(
                event_active_graph,
                event_aftermath_graph,
                independent,
                initial_ot,
                evidence,
                motion_features,
            )
            response = runtime().request_json(
                case=case,
                prompt=prompt,
                namespace=f"paired_{args.candidate_scoring_mode}/{args.crowd_evidence_mode}",
                mock_spec={
                    "kind": args.candidate_scoring_mode,
                    "active_ids": list(phase_mappings["active"]),
                    "aftermath_ids": list(phase_mappings["aftermath"]),
                    "active_initial_presence": {
                        anonymous_id: initial_ot[event_active_graph.key].get("node_presence", {}).get(node_key, 0.5)
                        for anonymous_id, node_key in phase_mappings["active"].items()
                    },
                    "aftermath_initial_presence": {
                        anonymous_id: initial_ot[event_aftermath_graph.key].get("node_presence", {}).get(node_key, 0.5)
                        for anonymous_id, node_key in phase_mappings["aftermath"].items()
                    },
                },
                evidence_override=evidence,
            )
            candidate_calls = int(not response.get("cache_hit"))
            parser = (
                parse_event_state_v5_response
                if args.candidate_scoring_mode == "event_state_v5"
                else parse_event_state_v4_response
            )
            candidate_event_state, phase_conditionals = parser(
                response,
                event_active_graph,
                event_aftermath_graph,
                phase_mappings,
                evidence_ids,
                args.evidence_slots_per_bin,
            )
            phase_results: dict[str, dict[str, dict]] = {}
            for phase, graph in phase_graphs.items():
                unary = phase_unary[phase]
                conditional = phase_conditionals[phase]
                m3a = matching.match_graph_conditional_rowmax(graph, unary, conditional, evidence_centers)
                m3b = matching.match_graph_conditional_ot(
                    graph, unary, conditional, evidence_centers,
                    use_coherence=False, method_name="conditional_ot_no_coherence",
                )
                m3c = matching.match_graph_conditional_ot(
                    graph, unary, conditional, evidence_centers,
                    use_coherence=True, coherence_weight=args.coherence_weight,
                    method_name="conditional_ot_full",
                )
                results_for_phase = {
                    "unary_ot": initial_ot[graph.key],
                    "conditional_rowmax": _result_dict(m3a),
                    "conditional_ot_no_coherence": _result_dict(m3b),
                    "conditional_ot_full": _result_dict(m3c),
                }
                phase_results[phase] = results_for_phase
                candidate_targets[graph.key] = {
                    "graph": graph.key,
                    "phase": phase,
                    "results": results_for_phase,
                    "joint_trace": candidate_event_state.get("phase_nodes", {}).get(phase, {}),
                    "response": None,
                }

            states = candidate_event_state.get("state_probabilities", {})
            p_active = float(states.get("active_or_ongoing_physical_escalation", 0.0))
            p_aftermath = float(states.get("causally_linked_aftermath", 0.0))
            candidate_phase_winners = {
                method: (
                    event_active_graph.key
                    if p_active * float(phase_results["active"][method].get("graph_score", 0.0))
                    >= p_aftermath * float(phase_results["aftermath"][method].get("graph_score", 0.0))
                    else event_aftermath_graph.key
                )
                for method in METHODS
            }
            if args.candidate_scoring_mode == "event_state_v4" and args.state_refinement_mode == "method_blend":
                candidate_replacements, score_contract = score_method_blend(
                    candidate_event_state,
                    base_target,
                    phase_results,
                    args.target_graph_key,
                    calibration,
                )
            else:
                # Margin residual leaves all graph scores frozen and is applied after competition.
                candidate_replacements = {method: dict(base_target[method]) for method in METHODS}
                score_contract = {
                    "baseline_score_contract": "frozen_method_specific_conditional_ot_v3",
                    "candidate_score_contract": (
                        "frozen_graph_scores_plus_signed_occupancy_confound_residual_v5"
                        if args.candidate_scoring_mode == "event_state_v5"
                        else "frozen_graph_scores_plus_bounded_state_margin_residual_v4"
                    ),
                    "calibration_id": str(calibration.get("calibration_id", "trace_only")),
                    "calibration_sha256": str(calibration.get("sha256", "")),
                    "calibration_source_groups_sha256": str(calibration.get("source_groups_sha256", "")),
                    "method_matched": True,
                    "repeated_gate_count": 0,
                    "refinement_mode": (
                        "signed_phase_aware_occupancy_residual"
                        if args.candidate_scoring_mode == "event_state_v5"
                        else "margin_residual"
                    ),
                    "phase_specific_weights_frozen": bool(calibration.get("frozen", False)),
                    "method_details": {},
                }
            candidate_target_prompt_sha = {args.candidate_scoring_mode: stable_sha1(prompt, size=40)}
            candidate_target_response = {
                args.candidate_scoring_mode: {
                    "prompt_sha1": stable_sha1(prompt, size=40),
                    "response": response.get("parsed", {}),
                    "cache_path": response.get("cache_path"),
                    "cache_hit": bool(response.get("cache_hit")),
                }
            }
            candidate_event_state["initial_unary_ot"] = initial_ot
            candidate_event_state["method_phase_scores"] = phase_results
            candidate_event_state["score_contract"] = score_contract
        else:
            for graph in candidate_graphs:
                graph_results, graph_trace, graph_calls, graph_call_meta = target(graph, f"candidate/{graph.key}")
                candidate_calls += graph_calls
                candidate_targets[graph.key] = {
                    "graph": graph.key, "results": graph_results, "joint_trace": graph_trace,
                    "response": graph_call_meta,
                }
            candidate_phase_winners = {}
            candidate_replacements = {}
            for method in METHODS:
                winner_key, winner_result = _select_phase_result(candidate_targets, method, args.candidate_phase_aggregation)
                candidate_phase_winners[method] = winner_key
                candidate_replacements[method] = winner_result
            candidate_target_prompt_sha = {
                key: value["response"]["prompt_sha1"] for key, value in candidate_targets.items()
            }
            candidate_target_response = {
                key: value["response"] for key, value in candidate_targets.items()
            }
        candidate_comp = {
            method: _replace_competition(
                base, args.target_graph_key, candidate_replacements[method], method, base_graph.polarity,
            )
            for method in METHODS
        }
        if args.candidate_scoring_mode == "event_state_v4" and args.state_refinement_mode == "margin_residual":
            residual_details = {}
            for method in METHODS:
                candidate_comp[method], residual_details[method] = bounded_margin_residual(
                    candidate_event_state or {}, base_comp[method], calibration,
                )
            score_contract = {**dict(score_contract or {}), "method_details": residual_details}
            if candidate_event_state is not None:
                candidate_event_state["score_contract"] = score_contract
        elif args.candidate_scoring_mode == "event_state_v5":
            residual_details = {}
            for method in METHODS:
                candidate_comp[method], residual_details[method] = score_phase_aware_occupancy(
                    candidate_event_state or {}, base_comp[method], calibration,
                )
            score_contract = {**dict(score_contract or {}), "method_details": residual_details}
            if candidate_event_state is not None:
                candidate_event_state["score_contract"] = score_contract
        multi_phase = len(candidate_graphs) > 1
        evidence_version = (
            "crowd_event_state_pair_v3"
            if args.candidate_scoring_mode == "event_state_v3"
            else (
                f"crowd_event_state_pair_{args.candidate_scoring_mode}_{args.crowd_evidence_mode}"
                if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}
                else ("crowd_phase_pair_v2" if multi_phase else "crowd_pair_v1")
            )
        )
        result = {
            "version": (
                "frozen_evidence_target_pair_v3"
                if args.candidate_scoring_mode == "event_state_v3"
                else (
                    f"frozen_evidence_target_pair_{args.candidate_scoring_mode}"
                    if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}
                    else "frozen_evidence_target_pair_v1"
                )
            ),
            "frozen_evidence_version": evidence_version, "segment_key": segment,
            "video_id": base.get("video_id"), "y_true": base.get("y_true"), "gt": base.get("gt", {}),
            "video_path": base.get("video_path"), "start_frame": base.get("start_frame"),
            "end_frame": base.get("end_frame"),
            "evaluation_stratum": manifest_row.get("stratum", manifest_row.get("crowd_v4_stratum", "")),
            "event_phase": manifest_row.get("event_phase", ""),
            "source_group": manifest_row.get("source_group", ""),
            "split_role": manifest_row.get("split_role", manifest_row.get("role", "")),
            "eligible_for_target_effect": True, "full_packet_eligible": True,
            "target_exposure_reason": "target_exposed",
            "metric_eligible": base.get("metric_eligible", True), "evidence": evidence,
            "frozen_shortlist": base.get("graph_candidates", {}),
            "frozen_non_target_graph_results": True,
            "frozen_competition_context": {
                "graph_candidates": base.get("graph_candidates", {}),
                "graph_results": {
                    method: {
                        key: value for key, value in base.get("graph_results", {}).get(method, {}).items()
                    }
                    for method in METHODS
                },
                "competitions": {
                    method: dict(base.get("competitions", {}).get(method, {}))
                    for method in METHODS
                },
            },
            "evidence_signature": stable_sha1(evidence, size=40),
            "baseline_shortlist_sha256": stable_sha1(base.get("graph_candidates", {}), size=40),
            "candidate_target_graph_keys": candidate_keys,
            "candidate_scoring_mode": args.candidate_scoring_mode,
            "candidate_phase_aggregation": (
                "event_state_gated_weighted_sum"
                if args.candidate_scoring_mode == "event_state_v3"
                else (
                    f"{args.state_refinement_mode}_method_matched_{args.candidate_scoring_mode}"
                    if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}
                    else args.candidate_phase_aggregation
                )
            ),
            "candidate_phase_winner_by_method": candidate_phase_winners,
            "shared_node_keys": sorted(set(base_graph.node_keys) & candidate_node_keys),
            "baseline_only_node_keys": sorted(set(base_graph.node_keys) - candidate_node_keys),
            "candidate_only_node_keys": sorted(candidate_node_keys - set(base_graph.node_keys)),
            "baseline_target_prompt_sha256": base_call_meta["prompt_sha1"],
            "candidate_target_prompt_sha256": candidate_target_prompt_sha,
            "baseline_target_response": base_call_meta,
            "candidate_target_response": candidate_target_response,
            "non_target_graph_results_sha256": stable_sha1({
                method: {key: value for key, value in base.get("graph_results", {}).get(method, {}).items() if key != args.target_graph_key}
                for method in METHODS
            }, size=40),
            "base_target": {"graph": base_graph.key, "results": base_target, "joint_trace": base_trace},
            "candidate_targets": candidate_targets,
            "candidate_event_state": candidate_event_state,
            "score_contract": score_contract,
            "state_calibration": {
                "path": str(args.state_calibration_file) if args.state_calibration_file else "",
                "sha256": str(calibration.get("sha256", "")),
                "calibration_id": str(calibration.get("calibration_id", "trace_only")),
                "frozen": bool(calibration.get("frozen", False)),
                "source_groups_sha256": str(calibration.get("source_groups_sha256", "")),
                "source_groups": list(calibration.get("source_groups", [])),
            } if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else None,
            "crowd_observation": {
                "mode": args.crowd_evidence_mode,
                "pair_gap": args.crowd_pair_gap if args.crowd_evidence_mode == "paired16" else None,
                "pair_map": evidence.get("pair_map", []),
                "motion_features": motion_features,
            } if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else None,
            "base_competitions": base_comp, "candidate_competitions": candidate_comp,
            "paired_delta": {
                method: candidate_comp[method]["margin"] - base_comp[method]["margin"]
                for method in METHODS
            },
            "candidate_only_independent_calls": new_node_calls,
            "fresh_conditional_calls": base_calls + candidate_calls,
        }
        if not multi_phase and candidate_targets:
            result["candidate_target"] = next(iter(candidate_targets.values()))
        return result, {"segment_key": segment, "eligible": True, "full_packet_eligible": True, "reason": "", "candidate_only_nodes": new_node_calls, "fresh_conditional_calls": base_calls + candidate_calls}

    results, audits = [], []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        jobs = {executor.submit(process, key): key for key in keys}
        for index, future in enumerate(futures.as_completed(jobs), 1):
            key = jobs[future]
            try:
                result, audit = future.result()
            except Exception as exc:
                result, audit = None, {"segment_key": key, "eligible": False, "reason": f"{type(exc).__name__}:{exc}"}
            audits.append(audit)
            if result:
                results.append(result)
            print(f"[frozen-pair {index}/{len(keys)}] {'OK' if result else 'SKIP'} {key}", flush=True)
    results.sort(key=lambda row: str(row["segment_key"])); audits.sort(key=lambda row: str(row["segment_key"]))
    write_jsonl(args.out_dir / "frozen_pair_results.jsonl", results)
    _write_csv(args.out_dir / "eligibility_audit.csv", audits)
    target_results = [row for row in results if bool(row.get("eligible_for_target_effect", True))]
    full_packet_results = [row for row in results if bool(row.get("full_packet_eligible", True))]
    before_values = [(int(row["y_true"]), int(row["base_competitions"]["conditional_ot_full"]["y_pred"]), float(row["base_competitions"]["conditional_ot_full"]["margin"]), str(row["video_id"])) for row in target_results]
    after_values = [(int(row["y_true"]), int(row["candidate_competitions"]["conditional_ot_full"]["y_pred"]), float(row["candidate_competitions"]["conditional_ot_full"]["margin"]), str(row["video_id"])) for row in target_results]
    before, after = _metrics(before_values), _metrics(after_values)
    full_before_values = [(int(row["y_true"]), int(row["base_competitions"]["conditional_ot_full"]["y_pred"]), float(row["base_competitions"]["conditional_ot_full"]["margin"]), str(row["video_id"])) for row in full_packet_results]
    full_after_values = [(int(row["y_true"]), int(row["candidate_competitions"]["conditional_ot_full"]["y_pred"]), float(row["candidate_competitions"]["conditional_ot_full"]["margin"]), str(row["video_id"])) for row in full_packet_results]
    full_before, full_after = _metrics(full_before_values), _metrics(full_after_values)
    helps = sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(before_values, after_values))
    hurts = sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(before_values, after_values))
    phase_winner_counts = {
        method: {
            key: sum(row.get("candidate_phase_winner_by_method", {}).get(method) == key for row in target_results)
            for key in candidate_keys
        }
        for method in METHODS
    }
    event_state_summary = None
    if args.candidate_scoring_mode in {"event_state_v3", "event_state_v4", "event_state_v5"}:
        state_names = (
            (
                "active_escalation", "causally_linked_aftermath",
                "benign_or_pre_event_context", "none_or_unobservable",
            )
            if args.candidate_scoring_mode == "event_state_v3"
            else (
                "active_or_ongoing_physical_escalation", "causally_linked_aftermath",
                "pre_event_tension_or_flight", "benign_collective_activity",
                "none_or_unobservable",
            )
        )

        def state_means(selected_rows):
            count = len(selected_rows)
            return {
                state: (
                    sum(float(row.get("candidate_event_state", {}).get("state_probabilities", {}).get(state, 0.0)) for row in selected_rows) / count
                    if count else None
                )
                for state in state_names
            }

        event_state_summary = {
            "complete": sum(bool(row.get("candidate_event_state", {}).get("complete")) for row in target_results),
            "incomplete": sum(not bool(row.get("candidate_event_state", {}).get("complete")) for row in target_results),
            "mean_state_probabilities": state_means(target_results),
            "mean_state_probabilities_by_gt": {
                str(label): state_means([row for row in target_results if int(row.get("y_true", 0)) == label])
                for label in (0, 1)
            },
        }
        if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}:
            event_state_summary.update({
                "observation_mode": args.crowd_evidence_mode,
                "state_refinement_mode": args.state_refinement_mode,
                "method_matched_score_contracts": sum(
                    bool(row.get("score_contract", {}).get("method_matched")) for row in target_results
                ),
                "repeated_gate_violations": sum(
                    int(row.get("score_contract", {}).get("repeated_gate_count", 0) or 0)
                    for row in target_results
                ),
                "noop_reasons": {
                    reason: sum(row.get("score_contract", {}).get("noop_reason", "") == reason for row in target_results)
                    for reason in sorted({
                        str(row.get("score_contract", {}).get("noop_reason", ""))
                        for row in target_results
                        if row.get("score_contract", {}).get("noop_reason")
                    })
                },
                "calibration_id": str(calibration.get("calibration_id", "trace_only")),
                "calibration_sha256": str(calibration.get("sha256", "")),
                "calibration_frozen": bool(calibration.get("frozen", False)),
            })
        if args.candidate_scoring_mode == "event_state_v5":
            event_state_summary["mean_current_window_active_occupancy_probability"] = (
                sum(float(row.get("candidate_event_state", {}).get("current_window_active_occupancy_probability", 0.0)) for row in target_results)
                / len(target_results) if target_results else None
            )
            event_state_summary["mean_occupancy_probability_by_gt"] = {
                str(label): (
                    sum(float(row.get("candidate_event_state", {}).get("current_window_active_occupancy_probability", 0.0)) for row in target_results if int(row.get("y_true", 0)) == label)
                    / sum(int(row.get("y_true", 0)) == label for row in target_results)
                    if any(int(row.get("y_true", 0)) == label for row in target_results) else None
                )
                for label in (0, 1)
            }
            event_state_summary["mean_current_window_normal_confound_probability"] = (
                sum(float(row.get("candidate_event_state", {}).get("current_window_normal_confound_probability", 0.0)) for row in target_results)
                / len(target_results) if target_results else None
            )
            event_state_summary["mean_normal_confound_probability_by_gt"] = {
                str(label): (
                    sum(float(row.get("candidate_event_state", {}).get("current_window_normal_confound_probability", 0.0)) for row in target_results if int(row.get("y_true", 0)) == label)
                    / sum(int(row.get("y_true", 0)) == label for row in target_results)
                    if any(int(row.get("y_true", 0)) == label for row in target_results) else None
                )
                for label in (0, 1)
            }
    summary = {
        "version": (
            "frozen_evidence_target_pair_summary_v3"
            if args.candidate_scoring_mode == "event_state_v3"
            else (
                f"frozen_evidence_target_pair_summary_{args.candidate_scoring_mode}"
                if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}
                else ("frozen_evidence_target_pair_summary_v2" if len(candidate_graphs) > 1 else "frozen_evidence_target_pair_summary_v1")
            )
        ),
        "requested": len(keys), "eligible": len(target_results),
        "target_exposed_eligible": len(target_results),
        "full_packet_eligible": len(full_packet_results),
        "target_not_exposed_noop": len(full_packet_results) - len(target_results),
        "ineligible": len(keys) - len(full_packet_results),
        "cache_salt": args.cache_salt, "target_graph_key": args.target_graph_key,
        "candidate_target_graph_keys": candidate_keys,
        "candidate_scoring_mode": args.candidate_scoring_mode,
        "candidate_phase_aggregation": (
            "event_state_gated_weighted_sum"
            if args.candidate_scoring_mode == "event_state_v3"
            else (
                f"{args.state_refinement_mode}_method_matched_{args.candidate_scoring_mode}"
                if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"}
                else args.candidate_phase_aggregation
            )
        ),
        "crowd_evidence_mode": args.crowd_evidence_mode if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else None,
        "state_refinement_mode": args.state_refinement_mode if args.candidate_scoring_mode in {"event_state_v4", "event_state_v5"} else None,
        "candidate_phase_winner_counts": phase_winner_counts,
        "event_state_summary": event_state_summary,
        "base": before, "candidate": after, "delta": _delta(after, before),
        "helps": helps, "hurts": hurts,
        "full_packet": {
            "base": full_before, "candidate": full_after, "delta": _delta(full_after, full_before),
            "helps": sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(full_before_values, full_after_values)),
            "hurts": sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(full_before_values, full_after_values)),
        },
        "fresh_conditional_calls": sum(row.get("fresh_conditional_calls", 0) for row in audits),
        "candidate_only_independent_calls": sum(row.get("candidate_only_nodes", 0) for row in audits),
    }
    write_json(args.out_dir / "frozen_pair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
