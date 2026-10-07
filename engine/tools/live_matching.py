#!/usr/bin/env python3
"""Live comparison: independent nodes versus two-stage conditional OT graph matching."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from math import ceil
from typing import Any, Dict, Mapping, Sequence

import numpy as np

import competition
import matching
from common import case_id, clip01, jsonable
from decision_policy import apply_decision_policy, extract_decision_features
from event_constitution import retrieval_multiplier
from leave_one_out import delta_matrix
from prompts import (
    conditional_refinement_prompt,
    graph_shortlist_prompt,
    graph_shortlist_repair_prompt,
    independent_node_prompt,
    verifier_prompt,
)
from schemas import ConditionalAffinity, GraphNodeV2, GraphTemplateV2, UnaryAffinity, WindowCase
from vlm_runtime import CachedVideoVLM
from temporal_contract import response_errors


METHODS = (
    "independent_direct_nodes",
    "shared_unary_rowmax",
    "unary_ot",
    "conditional_rowmax",
    "conditional_ot_no_coherence",
    "conditional_ot_full",
)
PRIMARY_GRAPH_METHOD = "conditional_ot_full"


def _normalize_distribution(value: Any, length: int = 8) -> tuple[np.ndarray, bool]:
    complete = isinstance(value, list) and len(value) == length
    if not complete:
        raise ValueError(f"temporal distribution must contain exactly {length} bins; refusing truncation/padding")
    values = list(value) if isinstance(value, list) else []
    values = values[:length] + [0.0] * max(0, length - len(values))
    array = np.array([max(0.0, float(item or 0.0)) for item in values], dtype=np.float64)
    total = float(array.sum())
    if total <= 1e-9:
        array[:] = 1.0 / length
    else:
        array /= total
    return array, complete


def _quality_bins(value: Any, fallback: Any = None, length: int = 8) -> tuple[np.ndarray, bool]:
    selected = value if isinstance(value, list) else fallback
    complete = isinstance(selected, list) and len(selected) == length
    if not complete:
        raise ValueError(f"temporal quality must contain exactly {length} bins; refusing truncation/padding")
    values = list(selected) if isinstance(selected, list) else []
    values = values[:length] + [0.01] * max(0, length - len(values))
    return np.array([clip01(item) for item in values], dtype=np.float64), complete


def _expand_bins(values: np.ndarray, slots_per_bin: int, *, distribution: bool) -> np.ndarray:
    slots = max(1, int(slots_per_bin))
    expanded = np.repeat(np.asarray(values, dtype=np.float64), slots)
    if distribution and slots > 1:
        expanded /= slots
    return expanded


def _evidence_layout(slots_per_bin: int) -> tuple[list[str], list[float]]:
    slots = max(1, int(slots_per_bin))
    ids, centers = [], []
    for bin_index in range(8):
        for slot in range(slots):
            ids.append(f"T{bin_index}#{slot}" if slots > 1 else f"T{bin_index}")
            centers.append(float(bin_index))
    return ids, centers


def _parse_independent(response: Mapping[str, Any], node_key: str) -> dict:
    value = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
    issues = response_errors(value, "independent")
    if issues:
        raise ValueError("; ".join(issues))
    legacy_bins = value.get("bin_affinity")
    location, location_complete = _normalize_distribution(
        value.get("location_distribution_given_present", legacy_bins)
    )
    quality, quality_complete = _quality_bins(value.get("evidence_quality_by_bin"), legacy_bins)
    presence = clip01(value.get("presence_probability", value.get("presence", 0.01)))
    null_probability = clip01(value.get("null_probability", 1.0 - presence))
    best_bin = value.get("best_bin")
    if not isinstance(best_bin, int) or not 0 <= best_bin < 8:
        best_bin = int(np.argmax(location * quality))
    return {
        "node_key": node_key,
        "presence": presence,
        "null_probability": null_probability,
        "location_distribution_given_present": location.tolist(),
        "evidence_quality_by_bin": quality.tolist(),
        # Backward-compatible visualization field.
        "bin_affinity": (presence * np.sqrt(location * quality)).tolist(),
        "best_bin": best_bin,
        "region": str(value.get("region", "")),
        "visible_evidence": str(value.get("visible_evidence", value.get("evidence", ""))),
        "uncertainty": clip01(value.get("uncertainty", 0.5)),
        "complete": bool(
            location_complete and quality_complete and "presence_probability" in value
        ),
        "cache_hit": bool(response.get("cache_hit")),
        "cache_path": response.get("cache_path"),
        "evidence": response.get("evidence", {}),
        "raw": response.get("raw", ""),
        "cache_reuse": response.get("cache_reuse"),
        "response_validation_version": response.get("response_validation_version"),
    }


def _parse_joint(
    response: Mapping[str, Any],
    graph: GraphTemplateV2,
    id_to_key: Mapping[str, str],
    evidence_ids: Sequence[str],
    slots_per_bin: int,
) -> tuple[ConditionalAffinity, dict]:
    value = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
    issues = response_errors(value, "joint", list(id_to_key))
    if issues:
        raise ValueError("; ".join(issues))
    raw_nodes = value.get("nodes", {}) if isinstance(value.get("nodes"), Mapping) else {}
    qualities, locations, null_scores, priors, uncertainty = [], [], [], [], []
    supporting: Dict[str, list[str]] = {}
    suppressing: Dict[str, list[str]] = {}
    parsed_nodes: Dict[str, dict] = {}
    complete = True
    key_to_id = {key: anonymous_id for anonymous_id, key in id_to_key.items()}
    for node in graph.nodes:
        anonymous_id = key_to_id[node.key]
        item = raw_nodes.get(anonymous_id, {}) if isinstance(raw_nodes.get(anonymous_id), Mapping) else {}
        legacy_bins = item.get("bin_affinity")
        location, location_complete = _normalize_distribution(
            item.get("location_distribution_given_present", legacy_bins)
        )
        quality, quality_complete = _quality_bins(item.get("evidence_quality_by_bin"), legacy_bins)
        presence = clip01(item.get("presence_probability", item.get("conditional_presence", 0.01)))
        null_probability = clip01(item.get("null_probability", 1.0 - presence))
        positive_ids = item.get("context_increases_from", item.get("supporting_nodes", []))
        negative_ids = item.get("context_decreases_from", item.get("suppressing_nodes", []))
        support_keys = [id_to_key[token] for token in positive_ids if token in id_to_key]
        suppress_keys = [id_to_key[token] for token in negative_ids if token in id_to_key]
        qualities.append(_expand_bins(quality, slots_per_bin, distribution=False))
        locations.append(_expand_bins(location, slots_per_bin, distribution=True))
        null_scores.append(null_probability)
        priors.append(presence)
        uncertainty.append(clip01(item.get("uncertainty", 0.5)))
        supporting[node.key] = support_keys
        suppressing[node.key] = suppress_keys
        best_bin = item.get("best_bin")
        if not isinstance(best_bin, int) or not 0 <= best_bin < 8:
            best_bin = int(np.argmax(location * quality))
        parsed_nodes[node.key] = {
            "anonymous_id": anonymous_id,
            "conditional_presence": presence,
            "presence_probability": presence,
            "null_probability": null_probability,
            "location_distribution_given_present": location.tolist(),
            "evidence_quality_by_bin": quality.tolist(),
            "bin_affinity": (presence * np.sqrt(location * quality)).tolist(),
            "best_bin": best_bin,
            "region": str(item.get("region", "")),
            "visible_evidence": str(item.get("visible_evidence", item.get("evidence", ""))),
            "context_increases_from": support_keys,
            "context_decreases_from": suppress_keys,
            "probability_update_reason": str(item.get("probability_update_reason", "")),
            "uncertainty": uncertainty[-1],
        }
        complete = complete and location_complete and quality_complete and (
            "presence_probability" in item or "conditional_presence" in item
        )
    coherence = clip01(value.get("graph_coherence", 0.01))
    cond = ConditionalAffinity(
        node_keys=graph.node_keys,
        evidence_ids=list(evidence_ids),
        scores=np.vstack(qualities),
        null_scores=np.array(null_scores, dtype=np.float64),
        node_presence_priors=np.array(priors, dtype=np.float64),
        phase_scores=np.vstack(locations),
        graph_coherence=coherence,
        supporting_nodes=supporting,
        suppressing_nodes=suppressing,
        uncertainty=np.array(uncertainty, dtype=np.float64),
        cache_sha1=str(response.get("cache_path", "")),
    )
    trace = {
        "graph_key": graph.key,
        "graph_coherence": coherence,
        "episode_span_bins": value.get("episode_span_bins", []),
        "episode_summary": str(value.get("episode_summary", "")),
        "nodes": parsed_nodes,
        "complete": bool(complete),
        "cache_hit": bool(response.get("cache_hit")),
        "cache_path": response.get("cache_path"),
        "evidence": response.get("evidence", {}),
        "raw": response.get("raw", ""),
        "cache_reuse": response.get("cache_reuse"),
        "response_validation_version": response.get("response_validation_version"),
    }
    return cond, trace


def _union_nodes(graphs: Sequence[GraphTemplateV2]) -> list[GraphNodeV2]:
    by_key: Dict[str, GraphNodeV2] = {}
    for graph in graphs:
        for node in graph.nodes:
            by_key.setdefault(node.key, node)
    return list(by_key.values())


def _unary(
    nodes: Sequence[GraphNodeV2],
    traces: Mapping[str, Mapping[str, Any]],
    slots_per_bin: int,
) -> UnaryAffinity:
    evidence_ids, _ = _evidence_layout(slots_per_bin)
    return UnaryAffinity(
        node_keys=[node.key for node in nodes],
        evidence_ids=evidence_ids,
        scores=np.array([
            _expand_bins(np.asarray(traces[node.key]["evidence_quality_by_bin"]), slots_per_bin, distribution=False)
            for node in nodes
        ], dtype=np.float64),
        null_scores=np.array([traces[node.key]["null_probability"] for node in nodes], dtype=np.float64),
        phase_scores=np.array([
            _expand_bins(np.asarray(traces[node.key]["location_distribution_given_present"]), slots_per_bin, distribution=True)
            for node in nodes
        ], dtype=np.float64),
        uncertainty=np.array([traces[node.key]["uncertainty"] for node in nodes], dtype=np.float64),
        node_presence_priors=np.array([traces[node.key]["presence"] for node in nodes], dtype=np.float64),
        cache_sha1="|".join(str(traces[node.key].get("cache_path", "")) for node in nodes),
    )


def _compete(
    method: str,
    abnormal,
    normal,
    margin_threshold: float,
    aggregation: str,
    temperature: float,
) -> dict:
    result = competition.compete(
        method, list(abnormal), list(normal), aggregation=aggregation, temperature=temperature,
    )
    margin = float(result["margin"])
    raw = {
        "abnormal_score": float(result.get("best_abnormal_score", 0.0) or 0.0),
        "normal_score": float(result.get("best_normal_score", 0.0) or 0.0),
        "margin": margin,
    }
    if abs(margin) <= float(margin_threshold):
        result["decision"] = "uncertain"
        result["y_pred"] = 0
    result["decision_margin_threshold"] = float(margin_threshold)
    result["raw"] = raw
    result["legacy_decision"] = {
        "threshold": float(margin_threshold),
        "decision": result["decision"],
        "y_pred": result["y_pred"],
    }
    return result


def _result_dict(value) -> dict:
    return jsonable(value.to_dict())


def _proxy_results(values: Mapping[str, Mapping[str, Any]], graphs: Sequence[GraphTemplateV2]):
    return [_dict_result_proxy(values[graph.key]) for graph in graphs]


class WindowMatcher:
    def __init__(
        self,
        runtime: CachedVideoVLM,
        catalog: Mapping[str, GraphTemplateV2],
        margin_threshold: float = 0.03,
        top_k_abnormal: int = 2,
        top_k_normal: int = 4,
        shortlist_pool_multiplier: float = 1.0,
        use_catalog_shortlist: bool = True,
        evidence_slots_per_bin: int = 2,
        coherence_weight: float = 0.25,
        competition_aggregation: str = "logmeanexp",
        competition_temperature: float = 0.1,
        run_leave_one_out: bool = False,
        decision_policy_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.catalog = catalog
        self.margin_threshold = float(margin_threshold)
        self.top_k_abnormal = max(1, int(top_k_abnormal))
        self.top_k_normal = max(2, int(top_k_normal))
        self.shortlist_pool_multiplier = max(1.0, float(shortlist_pool_multiplier))
        self.use_catalog_shortlist = bool(use_catalog_shortlist)
        self.evidence_slots_per_bin = max(1, int(evidence_slots_per_bin))
        self.coherence_weight = max(0.0, float(coherence_weight))
        self.competition_aggregation = str(competition_aggregation)
        self.competition_temperature = max(1e-6, float(competition_temperature))
        self.run_leave_one_out = bool(run_leave_one_out)
        self.decision_policy_config = dict(decision_policy_config or {
            "policy": "legacy_graph", "threshold": self.margin_threshold,
        })

    def _ranked_candidates(self, value, field, prefix, limit, id_to_key):
        raw = value.get(field, []) if isinstance(value.get(field), list) else []
        scored, seen = [], set()
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            candidate_id = str(item.get("id", ""))
            graph_key = id_to_key.get(candidate_id)
            if not graph_key or not candidate_id.startswith(prefix) or graph_key in seen:
                continue
            seen.add(graph_key)
            graph = self.catalog[graph_key]
            visual_support = clip01(item.get("visual_support", 0.0), eps=0.0)
            multiplier = retrieval_multiplier({
                "confidence": graph.confidence, "utility": {"global": graph.utility_global},
            })
            scored.append({
                "graph_key": graph_key,
                "family": graph.family,
                "visual_support": visual_support,
                "confidence": graph.confidence,
                "utility_global": graph.utility_global,
                "retrieval_multiplier": multiplier,
                "retrieval_score": visual_support * multiplier,
                "visible_reason": str(item.get("visible_reason", "")),
            })
        scored.sort(key=lambda item: (-item["retrieval_score"], -item["visual_support"], item["graph_key"]))
        if prefix == "N":
            selected, used_families = [], set()
            for item in scored:
                if item["family"] in used_families:
                    continue
                selected.append(item["graph_key"])
                used_families.add(item["family"])
                if len(selected) >= limit:
                    break
            for item in scored:
                if item["graph_key"] not in selected:
                    selected.append(item["graph_key"])
                if len(selected) >= limit:
                    break
        else:
            selected = [item["graph_key"] for item in scored[:limit]]
        return selected[:limit], scored

    def _counterfactual_completion(self, selected_abnormal, selected_normal, abnormal_ranking, normal_ranking):
        """Reserve shortlist capacity for validated linked opposite explanations."""
        forced_abnormal, forced_normal = [], []
        for key in selected_normal:
            for linked in self.catalog[key].counterfactual_links:
                if linked in self.catalog and self.catalog[linked].polarity == "abnormal":
                    forced_abnormal.append(linked)
                    break
        for key in selected_abnormal:
            for linked in self.catalog[key].counterfactual_links:
                if linked in self.catalog and self.catalog[linked].polarity == "normal":
                    forced_normal.append(linked)
                    break

        def complete(selected, forced, ranking, limit):
            ordered = []
            # Keep at least half of the direct visual ranking; counterfactuals fill the rest.
            direct_slots = max(1, (int(limit) + 1) // 2)
            for key in selected[:direct_slots] + forced + selected[direct_slots:] + [item["graph_key"] for item in ranking]:
                if key not in ordered:
                    ordered.append(key)
                if len(ordered) >= limit:
                    break
            return ordered

        abnormal = complete(selected_abnormal, forced_abnormal, abnormal_ranking, self.top_k_abnormal)
        normal = complete(selected_normal, forced_normal, normal_ranking, self.top_k_normal)
        return abnormal, normal, {
            "forced_abnormal": [key for key in abnormal if key in forced_abnormal],
            "forced_normal": [key for key in normal if key in forced_normal],
        }

    def _shortlist(self, case: WindowCase) -> tuple[list[GraphTemplateV2], dict]:
        source = case.source_record
        source_abnormal = str(source.get("best_abnormal_graph", "") or "")
        source_normal = str(source.get("best_normal_graph", "") or "")
        if not self.use_catalog_shortlist:
            if source_abnormal not in self.catalog or source_normal not in self.catalog:
                raise KeyError(f"source graph pair absent from catalog: {source_abnormal} vs {source_normal}")
            return [self.catalog[source_abnormal], self.catalog[source_normal]], {
                "mode": "frozen_source_pair",
                "selected_abnormal": [source_abnormal],
                "selected_normal": [source_normal],
                "complete": True,
                "cache_hit": None,
            }

        num_abnormal = sum(graph.polarity == "abnormal" for graph in self.catalog.values())
        num_normal = sum(graph.polarity == "normal" for graph in self.catalog.values())
        pool_abnormal = min(num_abnormal, max(
            self.top_k_abnormal, ceil(self.top_k_abnormal * self.shortlist_pool_multiplier),
        ))
        pool_normal = min(num_normal, max(
            self.top_k_normal, ceil(self.top_k_normal * self.shortlist_pool_multiplier),
        ))
        prompt, id_to_key = graph_shortlist_prompt(self.catalog, pool_abnormal, pool_normal)
        abnormal_ids = [candidate_id for candidate_id in id_to_key if candidate_id.startswith("A")]
        normal_ids = [candidate_id for candidate_id in id_to_key if candidate_id.startswith("N")]
        response = self.runtime.request_json(
            case=case,
            prompt=prompt,
            namespace="candidate_selector",
            mock_spec={
                "kind": "selector",
                "abnormal_ids": abnormal_ids,
                "normal_ids": normal_ids,
                "top_k_abnormal": pool_abnormal,
                "top_k_normal": pool_normal,
            },
        )
        value = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
        selected_abnormal, abnormal_ranking = self._ranked_candidates(
            value, "abnormal_candidates", "A", self.top_k_abnormal, id_to_key,
        )
        selected_normal, normal_ranking = self._ranked_candidates(
            value, "normal_candidates", "N", self.top_k_normal, id_to_key,
        )
        repaired = False
        if len(selected_abnormal) < self.top_k_abnormal or len(selected_normal) < self.top_k_normal:
            repair_prompt = graph_shortlist_repair_prompt(
                value, id_to_key, self.top_k_abnormal, self.top_k_normal,
            )
            repaired_response = self.runtime.request_json(
                case=case,
                prompt=repair_prompt,
                namespace="candidate_selector_repair",
                mock_spec={
                    "kind": "selector_repair",
                    "abnormal_ids": abnormal_ids,
                    "normal_ids": normal_ids,
                    "top_k_abnormal": self.top_k_abnormal,
                    "top_k_normal": self.top_k_normal,
                },
            )
            repaired_value = repaired_response.get("parsed", {}) if isinstance(repaired_response.get("parsed"), Mapping) else {}
            selected_abnormal, abnormal_ranking = self._ranked_candidates(
                repaired_value, "abnormal_candidates", "A", self.top_k_abnormal, id_to_key,
            )
            selected_normal, normal_ranking = self._ranked_candidates(
                repaired_value, "normal_candidates", "N", self.top_k_normal, id_to_key,
            )
            response = repaired_response
            value = repaired_value
            repaired = True
        selected_abnormal, selected_normal, counterfactual_trace = self._counterfactual_completion(
            selected_abnormal, selected_normal, abnormal_ranking, normal_ranking,
        )
        complete = len(selected_abnormal) == self.top_k_abnormal and len(selected_normal) == self.top_k_normal
        if not complete:
            raise ValueError(
                f"candidate shortlist incomplete after repair: abnormal={len(selected_abnormal)}/"
                f"{self.top_k_abnormal}, normal={len(selected_normal)}/{self.top_k_normal}"
            )
        selected_keys = selected_abnormal + selected_normal
        trace = {
            "mode": "blind_catalog_shortlist",
            "selected_abnormal": selected_abnormal,
            "selected_normal": selected_normal,
            "abnormal_ranking": abnormal_ranking,
            "normal_ranking": normal_ranking,
            "counterfactual_completion": counterfactual_trace,
            "retrieval_policy": "visual_support_x_confidence_x_utility_then_counterfactual_completion",
            "source_pair": {"abnormal": source_abnormal, "normal": source_normal},
            "selector_pool": {"abnormal": pool_abnormal, "normal": pool_normal},
            "shortlist_pool_multiplier": self.shortlist_pool_multiplier,
            "complete": True,
            "repaired": repaired,
            "cache_hit": bool(response.get("cache_hit")),
            "cache_path": response.get("cache_path"),
            "raw": response.get("raw", ""),
        }
        return [self.catalog[key] for key in selected_keys], trace

    def match(self, case: WindowCase, *, run_verifier: bool = True) -> dict:
        source = case.source_record
        evidence_metadata = self.runtime.evidence_metadata(case)
        evidence_ids, evidence_centers = _evidence_layout(self.evidence_slots_per_bin)
        graphs, shortlist = self._shortlist(case)
        abnormal_graphs = [graph for graph in graphs if graph.polarity == "abnormal"]
        normal_graphs = [graph for graph in graphs if graph.polarity == "normal"]
        if not abnormal_graphs or not normal_graphs:
            raise ValueError("candidate shortlist must contain abnormal and normal graphs")
        nodes = _union_nodes(graphs)

        independent: Dict[str, dict] = {}
        for node in nodes:
            response = self.runtime.request_json(
                case=case,
                prompt=independent_node_prompt(node),
                namespace=f"independent/{node.key}",
                mock_spec={"kind": "independent", "node_key": node.key},
            )
            independent[node.key] = _parse_independent(response, node.key)
        unary = _unary(nodes, independent, self.evidence_slots_per_bin)

        joint: Dict[str, dict] = {}
        graph_results: Dict[str, Dict[str, dict]] = {method: {} for method in METHODS}
        conditional_objects: Dict[str, ConditionalAffinity] = {}
        initial_ot_objects = {}
        for graph in graphs:
            direct_presence = {node.key: independent[node.key]["presence"] for node in graph.nodes}
            direct_time = {
                node.key: float(independent[node.key].get("best_bin") or 0) / 7.0 for node in graph.nodes
            }
            m0 = matching.match_graph_independent_direct(graph, direct_presence, direct_time)
            m1 = matching.match_graph_shared_rowmax(graph, unary, evidence_centers)
            m2 = matching.match_graph_unary_ot(graph, unary, evidence_centers)
            initial_ot_objects[graph.key] = m2
            prompt, id_to_key = conditional_refinement_prompt(graph, independent, m2.to_dict())
            legacy_prompt = None
            if self.runtime.config.allow_legacy_temporal_cache:
                legacy_prompt, _ = conditional_refinement_prompt(graph, independent, m2.to_dict(), legacy_slot_layout=True)
            response = self.runtime.request_json(
                case=case,
                prompt=prompt,
                namespace=f"conditional_refinement/{graph.key}",
                legacy_prompt=legacy_prompt,
                mock_spec={
                    "kind": "joint",
                    "graph_key": graph.key,
                    "node_keys": graph.node_keys,
                    "initial_presence": m2.node_presence,
                },
            )
            cond, trace = _parse_joint(
                response, graph, id_to_key, evidence_ids, self.evidence_slots_per_bin,
            )
            trace["initial_ot"] = m2.to_dict()
            joint[graph.key] = trace
            conditional_objects[graph.key] = cond
            m3a = matching.match_graph_conditional_rowmax(graph, unary, cond, evidence_centers)
            m3b = matching.match_graph_conditional_ot(
                graph, unary, cond, evidence_centers,
                use_coherence=False,
                method_name="conditional_ot_no_coherence",
            )
            m3c = matching.match_graph_conditional_ot(
                graph, unary, cond, evidence_centers,
                use_coherence=True,
                coherence_weight=self.coherence_weight,
                method_name="conditional_ot_full",
            )
            for method, value in zip(METHODS, (m0, m1, m2, m3a, m3b, m3c)):
                graph_results[method][graph.key] = _result_dict(value)

        competitions, competition_ablations, top_n_sensitivity = {}, {}, {}
        for method in METHODS:
            abnormal = _proxy_results(graph_results[method], abnormal_graphs)
            normal = _proxy_results(graph_results[method], normal_graphs)
            competitions[method] = _compete(
                method, abnormal, normal, self.margin_threshold,
                self.competition_aggregation, self.competition_temperature,
            )
            competition_ablations[method] = {
                "max": _compete(
                    method, abnormal, normal, self.margin_threshold, "max", self.competition_temperature,
                ),
                "logmeanexp": _compete(
                    method, abnormal, normal, self.margin_threshold, "logmeanexp", self.competition_temperature,
                ),
            }
            top_n_sensitivity[method] = competition.top_n_sensitivity(
                method, abnormal, normal,
                aggregation=self.competition_aggregation,
                temperature=self.competition_temperature,
            )

        independent_complete = all(value["complete"] for value in independent.values())
        joint_complete = all(value["complete"] for value in joint.values())
        feature_record = {
            "competitions": competitions,
            "graph_results": graph_results,
            "joint_graph_calls": joint,
            "graph_candidates": shortlist,
            "completeness": {"independent": independent_complete, "joint": joint_complete},
        }
        decision_features = extract_decision_features(feature_record)
        selected_policy = apply_decision_policy(decision_features, self.decision_policy_config)
        primary = competitions[PRIMARY_GRAPH_METHOD]
        primary["decision_policy"] = selected_policy
        if selected_policy["name"] != "legacy_graph":
            primary["decision"] = selected_policy["decision"]
            primary["y_pred"] = selected_policy["y_pred"]

        m0_correct = int(competitions["independent_direct_nodes"]["y_pred"] == case.y_true)
        legacy_graph_correct = int(primary["legacy_decision"]["y_pred"] == case.y_true)
        graph_correct = int(primary["y_pred"] == case.y_true)
        graph_helps = bool(not m0_correct and graph_correct)
        graph_hurts = bool(m0_correct and not graph_correct)
        discordant = competitions["independent_direct_nodes"]["y_pred"] != competitions[PRIMARY_GRAPH_METHOD]["y_pred"]
        verifier = None
        if discordant and independent_complete and joint_complete and shortlist["complete"] and run_verifier:
            verifier = self.verify(case, independent, joint, graph_results, competitions)
        verifier_prefers_graph = bool(verifier and verifier.get("preferred_method") == "graph")
        verifier_prefers_independent = bool(verifier and verifier.get("preferred_method") == "independent")
        verifier_confident = bool(verifier and float(verifier.get("confidence", 0.0)) >= 0.55)
        verified_help = bool(graph_helps and (not run_verifier or (verifier_prefers_graph and verifier_confident)))
        verified_hurt = bool(graph_hurts and (not run_verifier or (verifier_prefers_independent and verifier_confident)))

        conditionality_audit = None
        if self.run_leave_one_out and discordant:
            conditionality_audit = self._leave_one_out_audit(
                case, unary, independent, graphs, competitions[PRIMARY_GRAPH_METHOD], evidence_ids, evidence_centers,
            )

        probability_flow = []
        for graph in graphs:
            for node in graph.nodes:
                probability_flow.append({
                    "graph_key": graph.key,
                    "node_key": node.key,
                    "independent_probability": float(independent[node.key]["presence"]),
                    "initial_ot_presence": float(graph_results["unary_ot"][graph.key]["node_presence"][node.key]),
                    "conditional_probability": float(joint[graph.key]["nodes"][node.key]["conditional_presence"]),
                    "conditional_delta": float(joint[graph.key]["nodes"][node.key]["conditional_presence"])
                        - float(independent[node.key]["presence"]),
                    "conditional_rowmax_presence": float(graph_results["conditional_rowmax"][graph.key]["node_presence"][node.key]),
                    "conditional_ot_no_coherence_presence": float(graph_results["conditional_ot_no_coherence"][graph.key]["node_presence"][node.key]),
                    "final_ot_presence": float(graph_results[PRIMARY_GRAPH_METHOD][graph.key]["node_presence"][node.key]),
                })

        operational_pred = int(source.get("y_pred", 0) or 0)
        result = {
            "version": "multi_candidate_conditional_ot_v3",
            "case_id": case_id(case.segment_key),
            "segment_key": case.segment_key,
            "video_id": case.video_id,
            "video_path": case.video_path,
            "start_frame": int(case.start_frame),
            "end_frame": int(case.end_frame),
            "y_true": case.y_true,
            "y_true_operational": case.y_true,
            "y_true_core": source.get("_corrected_gt", {}).get("y_true_core"),
            "gt": source.get("_corrected_gt", {}),
            "metric_eligible": bool(source.get("_metric_eligible", True)),
            "evaluation_scope": source.get("_evaluation_scope", {}),
            "source_video_completion": source.get("_video_completion", {}),
            "evidence": evidence_metadata,
            "graph_pair": shortlist.get("source_pair", {}),
            "graph_candidates": shortlist,
            "independent_node_calls": independent,
            "joint_graph_calls": joint,
            "graph_results": graph_results,
            "competitions": competitions,
            "decision_features": decision_features.to_dict(),
            "selected_decision_policy": selected_policy,
            "competition_ablations": competition_ablations,
            "top_n_sensitivity": top_n_sensitivity,
            "operational_partac": {
                "y_pred": operational_pred,
                "decision": source.get("binary_decision", "abnormal" if operational_pred else "normal"),
                "correct": int(operational_pred == case.y_true),
            },
            "probability_flow": probability_flow,
            "conditionality_audit": conditionality_audit,
            "comparison": {
                "independent_correct": m0_correct,
                "conditional_ot_correct": graph_correct,
                "legacy_conditional_ot_correct": legacy_graph_correct,
                "graph_helps": graph_helps,
                "graph_hurts": graph_hurts,
                "discordant": bool(discordant),
                "verified_graph_help": verified_help,
                "verified_graph_hurt": verified_hurt,
                "conditional_ot_failure": bool(not graph_correct),
            },
            "blind_verifier": verifier,
            "completeness": {"independent": independent_complete, "joint": joint_complete},
        }
        return result

    def _method_summary(self, method: str, independent: dict, joint: dict, graph_results: dict, competitions: dict) -> dict:
        competition_value = competitions[method]
        graphs = {}
        for graph_key, value in graph_results[method].items():
            node_rows = {}
            for node_key, presence in value.get("node_presence", {}).items():
                evidence = independent.get(node_key, {}) if method == "independent_direct_nodes" else joint.get(graph_key, {}).get("nodes", {}).get(node_key, {})
                node_rows[node_key] = {
                    "presence": presence,
                    "expected_time": value.get("expected_time", {}).get(node_key),
                    "assignment": next((item.get("assigned") for item in value.get("assignments", []) if item.get("node_key") == node_key), ""),
                    "visible_evidence": evidence.get("visible_evidence", ""),
                }
            graphs[graph_key] = {"graph_score": value.get("graph_score"), "nodes": node_rows}
        return {
            "decision": competition_value.get("decision"),
            "best_abnormal_graph": competition_value.get("best_abnormal_graph"),
            "best_normal_graph": competition_value.get("best_normal_graph"),
            "margin": competition_value.get("margin"),
            "graphs": graphs,
        }

    def verify(self, case: WindowCase, independent: dict, joint: dict, graph_results: dict, competitions: dict) -> dict:
        independent_summary = self._method_summary(
            "independent_direct_nodes", independent, joint, graph_results, competitions,
        )
        graph_summary = self._method_summary(
            PRIMARY_GRAPH_METHOD, independent, joint, graph_results, competitions,
        )
        swap = int(hashlib.sha1(case.segment_key.encode()).hexdigest()[-1], 16) % 2 == 1
        method_x = graph_summary if swap else independent_summary
        method_y = independent_summary if swap else graph_summary
        response = self.runtime.request_json(
            case=case,
            prompt=verifier_prompt(method_x, method_y),
            namespace="verifier_symmetric",
            mock_spec={"kind": "verifier", "preferred_method": "X" if swap else "Y"},
        )
        value = response.get("parsed", {}) if isinstance(response.get("parsed"), Mapping) else {}
        preferred = str(value.get("preferred_method", "tie")).upper()
        if preferred not in {"X", "Y", "TIE"}:
            preferred = "TIE"
        if preferred == "TIE":
            mapped = "tie"
        elif (preferred == "X" and swap) or (preferred == "Y" and not swap):
            mapped = "graph"
        else:
            mapped = "independent"
        return {
            "preferred_method": mapped,
            "raw_preference": preferred,
            "method_x": "graph" if swap else "independent",
            "method_y": "independent" if swap else "graph",
            "confidence": clip01(value.get("confidence", 0.0), eps=0.0),
            "visual_reason": str(value.get("visual_reason", "")),
            "method_x_failure": str(value.get("method_x_failure", "")),
            "method_y_failure": str(value.get("method_y_failure", "")),
            "cache_hit": bool(response.get("cache_hit")),
            "cache_path": response.get("cache_path"),
            "raw": response.get("raw", ""),
        }

    def _leave_one_out_audit(self, case, unary, independent, graphs, primary_competition, evidence_ids, evidence_centers):
        decision = primary_competition.get("decision")
        graph_key = primary_competition.get("best_abnormal_graph") if decision == "abnormal" else primary_competition.get("best_normal_graph")
        graph = next((value for value in graphs if value.key == graph_key), None)
        if graph is None or len(graph.nodes) < 2:
            return None
        full_presence = {}
        subset_presence = {}
        # Reuse the full conditional call from the regular path through its deterministic cache.
        full_m2 = matching.match_graph_unary_ot(graph, unary, evidence_centers)
        full_prompt, full_map = conditional_refinement_prompt(graph, independent, full_m2.to_dict())
        full_response = self.runtime.request_json(
            case=case,
            prompt=full_prompt,
            namespace=f"conditional_refinement/{graph.key}",
            mock_spec={"kind": "joint", "graph_key": graph.key, "node_keys": graph.node_keys, "initial_presence": full_m2.node_presence},
        )
        full_cond, _ = _parse_joint(full_response, graph, full_map, evidence_ids, self.evidence_slots_per_bin)
        full_presence = {key: float(full_cond.node_presence_priors[index]) for index, key in enumerate(graph.node_keys)}
        for removed in graph.nodes:
            subset_nodes = tuple(node for node in graph.nodes if node.key != removed.key)
            subset = replace(graph, key=f"{graph.key}__without__{removed.key}", nodes=subset_nodes)
            subset_m2 = matching.match_graph_unary_ot(subset, unary, evidence_centers)
            prompt, mapping = conditional_refinement_prompt(subset, independent, subset_m2.to_dict())
            response = self.runtime.request_json(
                case=case,
                prompt=prompt,
                namespace=f"leave_one_out/{graph.key}/without_{removed.key}",
                mock_spec={"kind": "joint_subset", "graph_key": graph.key, "node_keys": subset.node_keys, "initial_presence": subset_m2.node_presence},
            )
            cond, _ = _parse_joint(response, subset, mapping, evidence_ids, self.evidence_slots_per_bin)
            subset_presence[removed.key] = {
                key: float(cond.node_presence_priors[index]) for index, key in enumerate(subset.node_keys)
            }
        return {
            "graph_key": graph.key,
            "full_presence": full_presence,
            "subset_presence": subset_presence,
            "deltas": delta_matrix(full_presence, subset_presence),
        }


class _dict_result_proxy:
    def __init__(self, value: Mapping[str, Any]) -> None:
        self.graph_key = str(value["graph_key"])
        self.graph_score = float(value["graph_score"])
