#!/usr/bin/env python3
"""DeepSeek post-hoc discovery for conditional-OT failures.

Discovery first diagnoses whether the failure is selector, conditional-refinement,
coherence/calibration, catalog, or representation related.  Only a declared ``catalog_gap``
may register a new node-set graph.  Test-derived candidates always remain inactive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from common import iter_jsonl, read_json, safe_name, stable_sha1, write_json, write_jsonl
from failure_router import diagnose_failure_source, route_failure
from selection import source_group_id

PRIMARY_METHOD = "conditional_ot_full"
ACTIONS = {
    "no_new_graph_needed",
    "add_abnormal_graph",
    "add_normal_graph",
    "representation_only",
}
FAILURE_CATEGORIES = {
    "graph_catalog_gap",
    "graph_definition_gap",
    "retrieval_failure",
    "node_perception_failure",
    "conditional_refinement_failure",
    "ot_allocation_failure",
    "calibration_failure",
    "temporal_context_failure",
    "representation_gap",
    "label_noise",
    "insufficient_evidence",
}


def _parse_json_like(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(stripped)
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        depth, in_string, escaped = 0, False, False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
            elif current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start:end + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    raise ValueError("DeepSeek response did not contain a JSON object")


class DeepSeekClient:
    def __init__(self, model: str, base_url: str, key_env: str, retries: int) -> None:
        key = os.environ.get(key_env, "")
        if not key:
            raise EnvironmentError(f"missing {key_env}")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("OpenAI SDK is required: pip install openai") from exc
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=180.0)
        self.model = model
        self.retries = max(0, int(retries))
        self.max_tokens = max(1000, int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16000")))
        self.last_trace: dict = {}

    def call(self, system: str, user: str) -> dict:
        base = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        structured = dict(base)
        structured["response_format"] = {"type": "json_object"}
        thinking_json = dict(structured)
        thinking_json.update({"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}})
        nonthinking_json = dict(structured)
        nonthinking_json["extra_body"] = {"thinking": {"type": "disabled"}}
        nonthinking_plain = dict(base)
        nonthinking_plain["extra_body"] = {"thinking": {"type": "disabled"}}
        variants = [
            ("thinking_json", thinking_json),
            ("nonthinking_json", nonthinking_json),
            ("nonthinking_plain", nonthinking_plain),
        ]
        last_error: Optional[Exception] = None
        variant_index = 0
        traces: list[dict] = []
        for attempt in range(self.retries + 1):
            variant_name, payload = variants[variant_index]
            try:
                response = self.client.chat.completions.create(**payload)
                choice = response.choices[0]
                message = choice.message
                text = message.content or ""
                reasoning = getattr(message, "reasoning_content", None) or ""
                usage = getattr(response, "usage", None)
                if usage is not None and hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                trace = {
                    "attempt": attempt + 1,
                    "variant": variant_name,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_chars": len(text),
                    "reasoning_content_chars": len(reasoning),
                    "content_preview": text[:2000],
                    "usage": usage if isinstance(usage, Mapping) else None,
                }
                traces.append(trace)
                self.last_trace = {"model": self.model, "max_tokens": self.max_tokens, "attempts": traces}
                if not text.strip():
                    raise ValueError(
                        "DeepSeek returned empty final content "
                        f"(variant={variant_name}, finish_reason={trace['finish_reason']}, "
                        f"reasoning_chars={len(reasoning)})"
                    )
                value = _parse_json_like(text)
                value["_raw"] = text
                value["_response_meta"] = {
                    key: trace[key] for key in (
                        "attempt", "variant", "finish_reason", "content_chars",
                        "reasoning_content_chars", "usage",
                    )
                }
                return value
            except Exception as exc:
                last_error = exc
                if not traces or traces[-1].get("attempt") != attempt + 1:
                    traces.append({
                        "attempt": attempt + 1,
                        "variant": variant_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                else:
                    traces[-1]["error_type"] = type(exc).__name__
                    traces[-1]["error"] = str(exc)
                self.last_trace = {"model": self.model, "max_tokens": self.max_tokens, "attempts": traces}
                if attempt >= self.retries:
                    raise
                message = str(exc).lower()
                if isinstance(exc, ValueError) or any(token in message for token in (
                    "reasoning_effort", "extra_body", "thinking", "response_format",
                    "unexpected keyword", "empty final content", "json object",
                )):
                    variant_index = min(variant_index + 1, len(variants) - 1)
                time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(str(last_error))


def _catalog_by_key(path: Path) -> tuple[dict, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    by_key = {}
    for polarity in ("abnormal", "normal"):
        for graph in raw.get(polarity, []):
            by_key[str(graph.get("key"))] = graph
    return raw, by_key


def _compact_failure(case: Mapping[str, Any], graphs: Mapping[str, Any]) -> dict:
    competition = case.get("competitions", {}).get(PRIMARY_METHOD, {})
    candidates = case.get("graph_candidates", {})
    keys = list(candidates.get("selected_abnormal", [])) + list(candidates.get("selected_normal", []))
    y_true = int(case.get("y_true", 0))
    y_pred_raw = competition.get("y_pred")
    y_pred = int(y_pred_raw) if y_pred_raw in {0, 1} else None
    return {
        "anonymous_case_id": case.get("case_id"),
        "failure_category_hint": diagnose_failure_source(case),
        "failure_route": route_failure(case),
        "failure_type": (
            "unresolved" if y_pred is None else
            "false_negative" if y_true == 1 and y_pred == 0 else "false_positive"
        ),
        "correct_binary_conclusion": "abnormal" if y_true else "normal",
        "method_decisions": {
            method: value.get("decision") for method, value in case.get("competitions", {}).items()
        },
        "conditional_ot_winners": {
            "abnormal": competition.get("best_abnormal_graph"),
            "normal": competition.get("best_normal_graph"),
        },
        "current_graphs": [graphs[key] for key in keys if key in graphs],
        "independent_nodes": {
            key: {field: value.get(field) for field in (
                "presence", "location_distribution_given_present", "evidence_quality_by_bin",
                "best_bin", "region", "visible_evidence", "uncertainty",
            )}
            for key, value in case.get("independent_node_calls", {}).items()
        },
        "joint_graphs": case.get("joint_graph_calls", {}),
        "method_results": {
            method: case.get("graph_results", {}).get(method, {})
            for method in ("unary_ot", "conditional_rowmax", "conditional_ot_no_coherence", PRIMARY_METHOD)
        },
        "probability_flow": case.get("probability_flow", []),
        "conditionality_audit": case.get("conditionality_audit"),
    }


def _prompt(case: Mapping[str, Any], graphs: Mapping[str, Any]) -> str:
    y_true = int(case.get("y_true", 0))
    expected_polarity = "abnormal" if y_true else "normal"
    add_action = f"add_{expected_polarity}_graph"
    failure_type = "false_negative" if y_true else "false_positive"
    category_hint = diagnose_failure_source(case)
    schema = {
        "action": "no_new_graph_needed|representation_only|" + add_action,
        "failure_category": "graph_catalog_gap|graph_definition_gap|retrieval_failure|node_perception_failure|conditional_refinement_failure|ot_allocation_failure|calibration_failure|temporal_context_failure|representation_gap|label_noise|insufficient_evidence",
        "diagnosis": "",
        "failure_signature": "",
        "representation_direction": "",
        "confidence": 0.0,
        "proposed_graphs": [{
            "key": "snake_case_key",
            "title": "",
            "polarity": expected_polarity,
            "family": "",
            "joint_semantics": "one coherent visible episode described by this node set",
            "ordered": False,
            "matching_policy": {"allow_null": True, "use_conditional_refinement": True, "temporal_mode": "soft_auto"},
            "nodes": [{
                "key": "snake_case_node",
                "title": "directly visible concept",
                "cue_bundle": ["concrete visual cue"],
                "required": True,
                "anchor": True,
                "weight": 1.0,
                "role": "event_anchor|confound_anchor|context|effect",
                "phase_hint": "any|prelude|onset|active|aftermath",
            }],
        }],
    }
    evidence = _compact_failure(case, graphs)
    return f"""This is post-hoc analysis of a {failure_type} from two-stage conditional OT on an
anonymous test segment. The correct binary conclusion is supplied only for audit; never invent
visual evidence.

First classify the failure. The local heuristic suggests `{category_hint}`, but you must check
all traces. Do NOT add a graph for selector, calibration, OT-allocation, conditional-refinement,
or generic representation failures. Use `representation_only` or `no_new_graph_needed` there.
Only when the existing catalog genuinely lacks a reusable node-set explanation may you set
`failure_category=graph_catalog_gap` and action=`{add_action}`. Any proposed graph must have polarity
`{expected_polarity}`, at least two directly visible nodes, no legacy `edges`, and remain inactive
until held-out validation.

Failure evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""


def _graph_signature(graph: Mapping[str, Any]) -> str:
    canonical = {
        "polarity": graph.get("polarity"),
        "family": graph.get("family"),
        "joint_semantics": graph.get("joint_semantics"),
        "nodes": graph.get("nodes", []),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _validate_graph(graph: Mapping[str, Any], expected_polarity: Optional[str] = None) -> List[str]:
    errors = []
    if graph.get("polarity") not in {"abnormal", "normal"}:
        errors.append("invalid_polarity")
    if expected_polarity and graph.get("polarity") != expected_polarity:
        errors.append(f"wrong_polarity_expected_{expected_polarity}")
    if "edges" in graph:
        errors.append("legacy_edges_are_forbidden")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        errors.append("at_least_two_nodes_required")
        nodes = []
    keys = []
    for node in nodes:
        if not isinstance(node, Mapping) or not node.get("key") or not node.get("title"):
            errors.append("invalid_node")
            continue
        keys.append(str(node["key"]))
        cues = node.get("cue_bundle")
        if not isinstance(cues, list) or not cues:
            errors.append(f"node_without_cues:{node.get('key')}")
    if len(keys) != len(set(keys)):
        errors.append("duplicate_node_keys")
    if not graph.get("key") or not graph.get("joint_semantics"):
        errors.append("missing_graph_identity_or_joint_semantics")
    return sorted(set(errors))


def _mock_proposal(case: Mapping[str, Any]) -> dict:
    polarity = "abnormal" if int(case.get("y_true", 0)) else "normal"
    cid = str(case.get("case_id", "case"))
    return {
        "action": f"add_{polarity}_graph",
        "failure_category": "catalog_gap",
        "diagnosis": "mock catalog gap",
        "failure_signature": "mock_node_set_gap",
        "representation_direction": "validate on held-out videos",
        "confidence": 0.75,
        "proposed_graphs": [{
            "key": f"mock_{polarity}_{cid}",
            "title": "Mock discovered node set",
            "polarity": polarity,
            "family": "mock",
            "joint_semantics": "two visible concepts occupy compatible evidence locations",
            "ordered": False,
            "matching_policy": {"allow_null": True, "use_conditional_refinement": True, "temporal_mode": "soft_auto"},
            "nodes": [
                {"key": "mock_anchor", "title": "visible anchor", "cue_bundle": ["one visible anchor"], "required": True, "anchor": True, "weight": 1.0, "role": "event_anchor", "phase_hint": "any"},
                {"key": "mock_context", "title": "compatible context", "cue_bundle": ["one compatible context cue"], "required": True, "anchor": False, "weight": 1.0, "role": "context", "phase_hint": "any"},
            ],
        }],
        "_raw": "mock",
    }


def _write_augmented(base: dict, registry: dict, output_dir: Path) -> None:
    augmented = json.loads(json.dumps(base))
    candidates = registry.get("candidate_graphs", [])
    augmented["inactive_test_derived_candidates"] = candidates
    write_json(output_dir / "graph_catalog_v3_augmented.json", augmented)
    lines = [
        "# Conditional-OT graph library candidates", "",
        "All entries are test-derived and `active=false`; held-out validation is mandatory.", "",
    ]
    for item in candidates:
        graph = item.get("graph", {})
        lines.extend([
            f"## {graph.get('title', graph.get('key', 'candidate'))}", "",
            f"- key: `{graph.get('key')}`",
            f"- polarity: `{graph.get('polarity')}`",
            f"- family: `{graph.get('family', '')}`",
            f"- source group: `{item.get('source_group_id', '')}`",
            f"- active: `{item.get('active', False)}`",
            f"- semantics: {graph.get('joint_semantics', '')}",
            "- nodes: " + ", ".join(f"`{node.get('key')}`" for node in graph.get("nodes", [])), "",
        ])
    (output_dir / "CURRENT_OT_GRAPH_LIBRARY_AUGMENTED.md").write_text("\n".join(lines), encoding="utf-8")


class LiveDiscovery:
    def __init__(
        self,
        *,
        out_dir: Path,
        graph_catalog: Path,
        registry_path: Path,
        teacher_model: str = "deepseek-v4-pro",
        teacher_base_url: str = "https://api.deepseek.com",
        teacher_key_env: str = "DEEPSEEK_API_KEY",
        retries: int = 6,
        mock: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(registry_path)
        self.base_catalog, self.graph_by_key = _catalog_by_key(graph_catalog)
        self.teacher_model = teacher_model
        self.teacher_base_url = teacher_base_url
        self.teacher_key_env = teacher_key_env
        self.retries = int(retries)
        self.mock = bool(mock)
        self.teacher = None
        self.registry = read_json(self.registry_path, {"version": "ot_graph_registry_v3", "candidate_graphs": []})
        if not isinstance(self.registry, dict):
            self.registry = {"version": "ot_graph_registry_v3", "candidate_graphs": []}
        self.candidates = self.registry.setdefault("candidate_graphs", [])
        self.known_signatures = {
            str(item.get("signature", "")) for item in self.candidates
            if isinstance(item, Mapping) and item.get("signature")
        }
        self.proposals: List[dict] = []
        self.accepted: List[dict] = []
        self.errors: List[dict] = []
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.rejections = 0

    def discover(self, case: Mapping[str, Any], index: int = 0, total: int = 0) -> dict:
        cid = str(case.get("case_id", stable_sha1(case.get("segment_key", ""))))
        output = self.out_dir / "cases" / f"{cid}.json"
        proposal = read_json(output, None)
        try:
            if not isinstance(proposal, dict):
                if self.mock:
                    proposal = _mock_proposal(case)
                else:
                    if self.teacher is None:
                        self.teacher = DeepSeekClient(
                            self.teacher_model, self.teacher_base_url, self.teacher_key_env, self.retries,
                        )
                    proposal = self.teacher.call(
                        "You audit conditional matching failures and add graphs only for genuine catalog gaps.",
                        _prompt(case, self.graph_by_key),
                    )
                write_json(output, proposal)
        except Exception as exc:
            error = {"case_id": cid, "segment_key": case.get("segment_key"), "type": type(exc).__name__, "error": str(exc)}
            if self.teacher is not None and self.teacher.last_trace:
                error["deepseek_trace"] = self.teacher.last_trace
            self.errors.append(error)
            print(f"[discovery-error {index or 'live'}] case={cid}: {exc}", flush=True)
            return {"case_id": cid, "error": error}

        expected_polarity = "abnormal" if int(case.get("y_true", 0)) else "normal"
        expected_add_action = f"add_{expected_polarity}_graph"
        action = str(proposal.get("action", "no_new_graph_needed"))
        if action not in ACTIONS:
            action = "no_new_graph_needed"
        category = str(proposal.get("failure_category", diagnose_failure_source(case)))
        if category not in FAILURE_CATEGORIES:
            category = diagnose_failure_source(case)
        self.action_counts[action] += 1
        self.category_counts[category] += 1
        proposal_record = dict(proposal)
        proposal_record.update({
            "case_id": cid,
            "segment_key": case.get("segment_key"),
            "action": action,
            "failure_category": category,
            "expected_add_action": expected_add_action,
        })
        self.proposals.append(proposal_record)
        print(f"+ DIAGNOSED [{index or 'live'}] case={cid} category={category} action={action}", flush=True)

        new_candidates = []
        registration_allowed = category == "graph_catalog_gap" and action == expected_add_action
        if action.startswith("add_") and not registration_allowed:
            self.rejections += 1
            print(f"[candidate-rejected] case={cid} category={category} action={action}", flush=True)
        if registration_allowed:
            for graph_value in proposal.get("proposed_graphs", []):
                if not isinstance(graph_value, Mapping):
                    continue
                graph = dict(graph_value)
                validation_errors = _validate_graph(graph, expected_polarity)
                signature = _graph_signature(graph)
                if validation_errors or signature in self.known_signatures:
                    if validation_errors:
                        print(f"[candidate-rejected] case={cid} graph={graph.get('key')} errors={','.join(validation_errors)}", flush=True)
                    continue
                video_id = str(case.get("video_id", ""))
                candidate = {
                    "graph": graph,
                    "active": False,
                    "test_derived": True,
                    "requires_held_out_validation": True,
                    "validation_status": "schema_valid_test_derived_candidate",
                    "source_case_id": cid,
                    "source_segment_key": case.get("segment_key"),
                    "source_video_id": video_id,
                    "source_group_id": source_group_id(video_id),
                    "failure_category": category,
                    "proposal_action": action,
                    "proposal_confidence": proposal.get("confidence"),
                    "signature": signature,
                }
                self.candidates.append(candidate)
                self.accepted.append(candidate)
                new_candidates.append(candidate)
                self.known_signatures.add(signature)
                write_json(self.registry_path, self.registry)
                live_path = self.out_dir / "live_discovered_graphs" / f"{cid}__{safe_name(str(graph.get('key')))}.json"
                write_json(live_path, candidate)
                _write_augmented(self.base_catalog, self.registry, self.registry_path.parent)
                print(f"+ DISCOVERED graph={graph.get('key')} active=false record={live_path}", flush=True)
        return {"case_id": cid, "proposal": proposal_record, "new_candidates": new_candidates}

    def finalize(self, selected_failures: int) -> dict:
        write_json(self.registry_path, self.registry)
        _write_augmented(self.base_catalog, self.registry, self.registry_path.parent)
        write_jsonl(self.out_dir / "graph_proposals.jsonl", self.proposals)
        write_jsonl(self.out_dir / "accepted_graph_candidates.jsonl", self.accepted)
        write_jsonl(self.out_dir / "discovery_errors.jsonl", self.errors)
        summary = {
            "selected_failures": int(selected_failures),
            "proposals": len(self.proposals),
            "errors": len(self.errors),
            "actions": dict(self.action_counts),
            "failure_categories": dict(self.category_counts),
            "new_graphs_this_run": len(self.accepted),
            "registration_rejections": self.rejections,
            "registry_total": len(self.candidates),
            "registry": str(self.registry_path),
            "analysis_only": True,
        }
        write_json(self.out_dir / "discovery_summary.json", summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover inactive node-set graphs from conditional-OT failures")
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-cases-per-video", type=int, default=1)
    parser.add_argument("--teacher-model", default="deepseek-v4-pro")
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    parser.add_argument("--teacher-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    failures = list(iter_jsonl(args.failures)) if args.failures.is_file() else []
    failures.sort(key=lambda case: str(case.get("segment_key", "")))
    if args.max_cases_per_video > 0:
        counts: Dict[str, int] = defaultdict(int)
        capped = []
        for case in failures:
            video_id = str(case.get("video_id", ""))
            if counts[video_id] >= args.max_cases_per_video:
                continue
            counts[video_id] += 1
            capped.append(case)
        failures = capped
    if args.max_cases > 0:
        failures = failures[:args.max_cases]
    print(f"[discovery] selected_failures={len(failures)}", flush=True)
    session = LiveDiscovery(
        out_dir=args.out_dir,
        graph_catalog=args.graph_catalog,
        registry_path=args.registry,
        teacher_model=args.teacher_model,
        teacher_base_url=args.teacher_base_url,
        teacher_key_env=args.teacher_key_env,
        retries=args.retries,
        mock=args.mock,
    )
    for index, case in enumerate(failures, 1):
        session.discover(case, index, len(failures))
    print(json.dumps(session.finalize(len(failures)), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
