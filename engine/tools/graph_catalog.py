#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph-catalog helpers for OT node-set matching.

Legacy cause/precedes/supports edges are intentionally not imported.  The catalog retains
node sets, soft phase metadata and a coarse graph family used only to make the normal
shortlist diverse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from schemas import GraphNodeV2, GraphTemplateV2

_TIME_TO_PHASE = {
    "onset": "onset", "same_or_onset": "onset",
    "before_or_same": "prelude", "same_or_before": "prelude",
    "same": "active",
    "after": "aftermath", "immediate_after": "aftermath", "same_or_after": "aftermath",
}


def infer_family(key: str, title: str = "", polarity: str = "normal") -> str:
    text = f"{key} {title}".lower()
    rules = (
        ("text_or_non_scene", ("text_card", "non_scene", "title_card")),
        ("stage_or_simulation", ("simulation", "simulated", "video_game", "stage", "firework", "staged", "movie")),
        ("media_or_edit", ("camera", "edit", "text_card", "news", "screen", "graphic")),
        ("sports", ("sport", "hockey", "basketball", "ceremonial_contact", "body_check", "game", "player")),
        ("traffic", ("vehicle", "traffic", "driving", "road", "highway", "collision")),
        ("crowd", ("crowd", "riot", "gathering", "protest", "ceremonial_crowd")),
        ("assist_or_rescue", ("rescue", "assist", "aid", "help")),
        ("human_interaction", ("fight", "conflict", "control", "abuse", "contact", "interpersonal")),
        ("impulse_or_blast", ("shoot", "discharge", "blast", "explosion", "flash")),
    )
    for family, tokens in rules:
        if any(token in text for token in tokens):
            return family
    return "abnormal_other" if polarity == "abnormal" else "normal_other"


def _load_detector(code_dir: Optional[str]):
    if code_dir:
        src = Path(code_dir) / "src"
        if src.exists():
            sys.path.insert(0, str(src))
    from structural_vlm_binary.graph_schema import ABNORMAL_GRAPHS, NORMAL_GRAPHS, node_evidence_class  # type: ignore
    return ABNORMAL_GRAPHS, NORMAL_GRAPHS, node_evidence_class


def _phase_hint(node) -> str:
    return _TIME_TO_PHASE.get(str(getattr(node, "time_role", "same")), "any")


def _convert(g, nec) -> GraphTemplateV2:
    ordered = len({str(getattr(n, "time_role", "same")) for n in g.nodes}) > 1
    nodes = tuple(
        GraphNodeV2(
            key=n.key,
            title=n.title,
            cue_bundle=tuple(n.cue_bundle),
            required=bool(n.required),
            anchor=bool(n.anchor),
            weight=float(getattr(n, "weight", 1.0 if n.required else 0.5)),
            role=str(nec(g.key, n.key)),
            phase_hint=_phase_hint(n),
        )
        for n in g.nodes
    )
    return GraphTemplateV2(
        key=g.key,
        title=g.title,
        polarity=g.polarity,
        joint_semantics=f"the node set should describe one coherent {g.title}",
        ordered=ordered,
        nodes=nodes,
        matching_policy={"allow_null": True, "use_conditional_refinement": True, "temporal_mode": "soft_auto"},
        family=infer_family(g.key, g.title, g.polarity),
    )


def load_graph_catalog_v2(code_dir: Optional[str]) -> Dict[str, GraphTemplateV2]:
    abnormal, normal, nec = _load_detector(code_dir)
    catalog: Dict[str, GraphTemplateV2] = {}
    for graph in list(abnormal) + list(normal):
        catalog[graph.key] = _convert(graph, nec)
    return catalog


def template_to_dict(template: GraphTemplateV2) -> dict:
    return {
        "key": template.key,
        "title": template.title,
        "polarity": template.polarity,
        "family": template.family,
        "joint_semantics": template.joint_semantics,
        "ordered": template.ordered,
        "matching_policy": template.matching_policy,
        "status": template.status,
        "active": template.status == "active",
        "confidence": template.confidence,
        "utility": {"global": template.utility_global},
        "applicability": list(template.applicability),
        "falsifiers": list(template.falsifiers),
        "counterfactual_links": list(template.counterfactual_links),
        "canonical_factors": list(template.canonical_factors),
        "nodes": [
            {
                "key": node.key,
                "title": node.title,
                "cue_bundle": list(node.cue_bundle),
                "required": node.required,
                "anchor": node.anchor,
                "weight": node.weight,
                "role": node.role,
                "phase_hint": node.phase_hint,
            }
            for node in template.nodes
        ],
    }


def export_catalog_json(catalog: Dict[str, GraphTemplateV2], path: Path) -> None:
    data = {
        "abnormal": [template_to_dict(t) for t in catalog.values() if t.polarity == "abnormal"],
        "normal": [template_to_dict(t) for t in catalog.values() if t.polarity != "abnormal"],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def template_from_dict(value: dict) -> GraphTemplateV2:
    nodes = tuple(
        GraphNodeV2(
            key=str(node["key"]),
            title=str(node.get("title", node["key"])),
            cue_bundle=tuple(str(cue) for cue in node.get("cue_bundle", [])),
            required=bool(node.get("required", True)),
            anchor=bool(node.get("anchor", False)),
            weight=float(node.get("weight", 1.0 if node.get("required", True) else 0.5)),
            role=str(node.get("role", "evidence")),
            phase_hint=str(node.get("phase_hint", "any")),
        )
        for node in value.get("nodes", [])
    )
    if not nodes:
        raise ValueError(f"graph template has no nodes: {value.get('key')}")
    polarity = str(value.get("polarity", "normal"))
    title = str(value.get("title", value["key"]))
    utility = value.get("utility", {})
    if isinstance(utility, dict):
        utility = utility.get("global", 0.0)
    return GraphTemplateV2(
        key=str(value["key"]),
        title=title,
        polarity=polarity,
        family=str(value.get("family") or infer_family(str(value["key"]), title, polarity)),
        joint_semantics=str(value.get("joint_semantics", "the nodes describe one coherent episode")),
        ordered=bool(value.get("ordered", False)),
        nodes=nodes,
        matching_policy=dict(value.get("matching_policy", {})),
        status=str(value.get("status", "active" if value.get("active", True) else "candidate")),
        confidence=float(value.get("confidence", 1.0) or 0.0),
        utility_global=float(utility or 0.0),
        applicability=tuple(str(item) for item in value.get("applicability", [])),
        falsifiers=tuple(str(item) for item in value.get("falsifiers", [])),
        counterfactual_links=tuple(str(item) for item in value.get("counterfactual_links", [])),
        canonical_factors=tuple(str(item) for item in value.get("canonical_factors", [])),
    )


def read_catalog_json(path: Path) -> Dict[str, GraphTemplateV2]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = []
    if isinstance(raw, dict):
        for group in ("abnormal", "normal"):
            group_values = raw.get(group, [])
            if isinstance(group_values, list):
                values.extend(
                    value for value in group_values
                    if isinstance(value, dict)
                    and value.get("active", True) is not False
                    and str(value.get("status", "active")) not in {"candidate", "retired", "rejected"}
                )
    catalog = {str(value["key"]): template_from_dict(value) for value in values if isinstance(value, dict)}
    if not catalog:
        raise ValueError(f"graph catalog is empty: {path}")
    return catalog
