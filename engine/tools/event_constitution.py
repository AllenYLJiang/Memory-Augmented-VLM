#!/usr/bin/env python3
"""Machine-checkable invariants for reflective graph memory.

The constitution file is JSON syntax, which is also valid YAML. Keeping the parser
dependency-free makes governance available in the same lightweight environment as the
offline report tools.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


HARD_ISSUES = {
    "invalid_polarity",
    "too_few_nodes",
    "duplicate_node_keys",
    "forbidden_legacy_edges",
    "nuisance_anchor",
    "identity_or_source_style_dependency",
    "abnormal_without_mechanism",
    "normal_without_benign_mechanism",
    "non_visual_node",
}


def load_constitution(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("version"):
        raise ValueError(f"invalid Event Constitution: {path}")
    return value


def _snake(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def graph_text(graph: Mapping[str, Any]) -> str:
    fields = {
        "key": graph.get("key"),
        "title": graph.get("title"),
        "joint_semantics": graph.get("joint_semantics"),
        "canonical_factors": graph.get("canonical_factors"),
        "applicability": graph.get("applicability"),
        "falsifiers": graph.get("falsifiers"),
        "nodes": graph.get("nodes"),
    }
    return _snake(_flatten_text(fields))


def semantic_tokens(graph: Mapping[str, Any]) -> set[str]:
    stop = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of",
        "on", "or", "the", "to", "with", "visible", "scene", "event", "graph",
        "node", "normal", "abnormal",
    }
    return {token for token in graph_text(graph).split("_") if len(token) > 2 and token not in stop}


def semantic_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a, b = semantic_tokens(left), semantic_tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def _contains_any(text: str, terms: Iterable[str]) -> list[str]:
    return sorted({_snake(term) for term in terms if _snake(term) and _snake(term) in text})


def audit_graph(graph: Mapping[str, Any], constitution: Mapping[str, Any]) -> dict:
    issues: list[dict] = []

    def add(code: str, detail: Any = "", hard: bool | None = None) -> None:
        issues.append({
            "code": code,
            "detail": detail,
            "hard": code in HARD_ISSUES if hard is None else bool(hard),
        })

    polarity = str(graph.get("polarity", ""))
    if polarity not in {"abnormal", "normal"}:
        add("invalid_polarity", polarity)
    if "edges" in graph:
        add("forbidden_legacy_edges")
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) < 2:
        add("too_few_nodes", len(nodes) if isinstance(nodes, list) else 0)
        nodes = []
    keys = [str(node.get("key", "")) for node in nodes if isinstance(node, Mapping)]
    if len(keys) != len(set(keys)):
        add("duplicate_node_keys")

    text = graph_text(graph)
    nuisance = _contains_any(text, constitution.get("forbidden_nuisance_anchors", []))
    if nuisance:
        add("nuisance_anchor", nuisance)
    if any(term in text for term in ("actor_identity", "celebrity", "country_specific", "video_source")):
        add("identity_or_source_style_dependency")

    visual_terms = constitution.get("direct_visual_terms", [])
    for node in nodes:
        if not isinstance(node, Mapping):
            add("non_visual_node", "non-object node")
            continue
        node_text = _snake(_flatten_text({
            "key": node.get("key"), "title": node.get("title"), "cue_bundle": node.get("cue_bundle"),
        }))
        cues = node.get("cue_bundle")
        if not isinstance(cues, list) or not cues:
            add("non_visual_node", node.get("key"))
        elif visual_terms and not _contains_any(node_text, visual_terms):
            add("non_visual_node", node.get("key"), hard=False)

    if polarity == "abnormal" and not _contains_any(text, constitution.get("abnormal_mechanism_terms", [])):
        add("abnormal_without_mechanism")
    if polarity == "normal" and not _contains_any(text, constitution.get("normal_mechanism_terms", [])):
        add("normal_without_benign_mechanism")

    if not graph.get("applicability"):
        add("missing_applicability", hard=False)
    if not graph.get("falsifiers"):
        add("missing_falsifiers", hard=False)
    links = graph.get("counterfactual_links") or graph.get("counterfactual_graph")
    if not links:
        add("missing_counterfactual", hard=False)
    if not graph.get("canonical_factors"):
        add("missing_canonical_factors", hard=False)

    hard = [item for item in issues if item["hard"]]
    complexity = len(nodes) + 0.25 * sum(len(node.get("cue_bundle", [])) for node in nodes if isinstance(node, Mapping))
    return {
        "graph_key": graph.get("key"),
        "polarity": polarity,
        "passes_hard_invariants": not hard,
        "issues": issues,
        "hard_issue_codes": sorted({item["code"] for item in hard}),
        "complexity": round(float(complexity), 4),
        "semantic_tokens": sorted(semantic_tokens(graph)),
    }


def utility_confidence(graph: Mapping[str, Any]) -> tuple[float, float]:
    utility = graph.get("utility", {})
    if isinstance(utility, Mapping):
        utility = utility.get("global", 0.0)
    try:
        utility_value = float(utility or 0.0)
    except (TypeError, ValueError):
        utility_value = 0.0
    try:
        confidence = float(graph.get("confidence", 1.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 1.0
    return max(-1.0, min(1.0, utility_value)), max(0.0, min(1.0, confidence))


def retrieval_multiplier(graph: Mapping[str, Any]) -> float:
    utility, confidence = utility_confidence(graph)
    return (0.5 + 0.5 * confidence) * math.exp(max(-0.25, min(0.25, utility)))
