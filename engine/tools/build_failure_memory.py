#!/usr/bin/env python3
"""Convert OT failures into immutable episodic memory records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from discover_ot_failures import PRIMARY_METHOD
from failure_router import route_failure
from selection import source_group_id


def _compact_nodes(case: Mapping[str, Any]) -> dict:
    result = {}
    for key, value in case.get("independent_node_calls", {}).items():
        if not isinstance(value, Mapping):
            continue
        result[str(key)] = {
            "presence": value.get("presence"),
            "best_bin": value.get("best_bin"),
            "region": value.get("region"),
            "visible_evidence": value.get("visible_evidence"),
            "uncertainty": value.get("uncertainty"),
        }
    return result


def _trace_observation(case: Mapping[str, Any]) -> dict:
    descriptions = []
    for value in case.get("independent_node_calls", {}).values():
        if isinstance(value, Mapping) and value.get("visible_evidence"):
            descriptions.append(str(value["visible_evidence"]))
    for value in case.get("joint_graph_calls", {}).values():
        if not isinstance(value, Mapping):
            continue
        if value.get("episode_summary"):
            descriptions.append(str(value["episode_summary"]))
    return {
        "status": "trace_synthesis_pending_blind_observer",
        "visible_descriptions": list(dict.fromkeys(descriptions))[:16],
        "uncertainty_note": "This is a trace synthesis, not a frozen blind observer result.",
    }


def _correct(case: Mapping[str, Any], method: str) -> bool:
    value = case.get("competitions", {}).get(method, {})
    return value.get("y_pred") in {0, 1} and int(value.get("y_pred")) == int(case.get("y_true", 0))


def _roles(case: Mapping[str, Any], options: Mapping[str, Any]) -> list[str]:
    comparison = case.get("comparison", {})
    competition = case.get("competitions", {}).get(PRIMARY_METHOD, {})
    roles = []
    if not _correct(case, PRIMARY_METHOD):
        roles.append("final_error")
    if options.get("include_graph_helps") and bool(comparison.get("graph_helps")):
        roles.append("graph_help")
    if options.get("include_graph_hurts") and bool(comparison.get("graph_hurts")):
        roles.append("graph_hurt")
    decisions = {
        value.get("y_pred") for value in case.get("competitions", {}).values()
        if isinstance(value, Mapping) and value.get("y_pred") in {0, 1}
    }
    if options.get("include_method_disagreements") and len(decisions) > 1:
        roles.append("method_disagreement")
    margin = abs(float(competition.get("margin", 0.0) or 0.0))
    if (
        options.get("include_near_threshold_correct")
        and _correct(case, PRIMARY_METHOD)
        and margin <= float(options.get("near_threshold_margin", 0.05))
    ):
        roles.append("fragile_correct")
    return list(dict.fromkeys(roles))


def _method_outcomes(case: Mapping[str, Any]) -> dict:
    aliases = {
        "m0": "independent_direct_nodes",
        "m1": "shared_unary_rowmax",
        "m2": "unary_ot",
        "m3a": "conditional_rowmax",
        "m3b": "conditional_ot_no_coherence",
        "m3c": PRIMARY_METHOD,
    }
    return {
        alias: {
            "method": method,
            "outcome": "correct" if _correct(case, method) else "wrong",
            "y_pred": case.get("competitions", {}).get(method, {}).get("y_pred"),
            "decision": case.get("competitions", {}).get(method, {}).get("decision"),
            "margin": case.get("competitions", {}).get(method, {}).get("margin"),
        }
        for alias, method in aliases.items()
    }


def build(
    records_path: Path,
    out_dir: Path,
    *,
    include_graph_helps: bool = False,
    include_graph_hurts: bool = False,
    include_method_disagreements: bool = False,
    include_near_threshold_correct: bool = False,
    near_threshold_margin: float = 0.05,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "include_graph_helps": bool(include_graph_helps),
        "include_graph_hurts": bool(include_graph_hurts),
        "include_method_disagreements": bool(include_method_disagreements),
        "include_near_threshold_correct": bool(include_near_threshold_correct),
        "near_threshold_margin": float(near_threshold_margin),
    }
    rows = []
    for case in iter_jsonl(records_path):
        roles = _roles(case, options)
        if not roles:
            continue
        competition = case.get("competitions", {}).get(PRIMARY_METHOD, {})
        y_true = int(case.get("y_true", 0))
        y_pred_raw = competition.get("y_pred")
        y_pred = int(y_pred_raw) if y_pred_raw in {0, 1} else None
        video_id = str(case.get("video_id", ""))
        cid = str(case.get("case_id") or stable_sha1(case.get("segment_key", "")))
        failure_route = route_failure(case) if "final_error" in roles else None
        row = {
            "id": f"memory_{cid}",
            "case_id": cid,
            "memory_type": "episodic",
            "memory_role": roles[0],
            "memory_roles": roles,
            "status": "raw",
            "immutable": True,
            "failure_kind": (
                "fn" if y_true == 1 and y_pred == 0 else
                "fp" if y_true == 0 and y_pred == 1 else
                "unresolved" if "final_error" in roles else "none"
            ),
            "failure_category": failure_route.get("primary_route") if failure_route else None,
            "failure_route": failure_route,
            "method_outcomes": _method_outcomes(case),
            "source_group": source_group_id(video_id),
            "window": {
                "segment_key": case.get("segment_key"),
                "video_id": video_id,
                "video_path": case.get("video_path"),
                "start_frame": case.get("start_frame"),
                "end_frame": case.get("end_frame"),
            },
            "posthoc_truth": {
                "y_true": y_true,
                "correct_conclusion": "abnormal" if y_true else "normal",
                "prediction": "unresolved" if y_pred is None else ("abnormal" if y_pred else "normal"),
                "gt_audit": case.get("gt", {}),
            },
            "winning_graphs": {
                "abnormal": competition.get("best_abnormal_graph"),
                "normal": competition.get("best_normal_graph"),
                "margin": competition.get("margin"),
            },
            "candidate_trace": case.get("graph_candidates", {}),
            "node_scores": _compact_nodes(case),
            "ot_trace": case.get("graph_results", {}).get(PRIMARY_METHOD, {}),
            "probability_flow": case.get("probability_flow", []),
            "parse_completeness": case.get("completeness", {}),
            "observed_event_state": _trace_observation(case),
            "root_cause_hypotheses": [failure_route.get("primary_route")] if failure_route else [],
            "raw_evidence": {
                "evidence": case.get("evidence", {}),
                "source_records_file": str(records_path),
                "source_record_sha1": stable_sha1(json.dumps(case, sort_keys=True, ensure_ascii=False), size=40),
            },
        }
        rows.append(row)
    write_jsonl(out_dir / "episodic_failures.jsonl", rows)
    summary = {
        "version": "episodic_failure_memory_v2_routed_contrastive",
        "records_source": str(records_path),
        "records": len(rows),
        "failures": sum("final_error" in row["memory_roles"] for row in rows),
        "false_positives": sum(row["failure_kind"] == "fp" for row in rows),
        "false_negatives": sum(row["failure_kind"] == "fn" for row in rows),
        "unresolved_errors": sum(row["failure_kind"] == "unresolved" for row in rows),
        "role_counts": {
            role: sum(role in row["memory_roles"] for row in rows)
            for role in ("final_error", "graph_help", "graph_hurt", "fragile_correct", "method_disagreement")
        },
        "route_counts": {
            route: sum(row.get("failure_category") == route for row in rows)
            for route in sorted({str(row.get("failure_category")) for row in rows if row.get("failure_category")})
        },
        "source_groups": len({row["source_group"] for row in rows}),
        "observer_status": "pending",
        "selection_options": options,
    }
    write_json(out_dir / "failure_memory_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--include-graph-helps", action="store_true")
    parser.add_argument("--include-graph-hurts", action="store_true")
    parser.add_argument("--include-method-disagreements", action="store_true")
    parser.add_argument("--include-near-threshold-correct", action="store_true")
    parser.add_argument("--near-threshold-margin", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(build(
        args.records, args.out_dir,
        include_graph_helps=args.include_graph_helps,
        include_graph_hurts=args.include_graph_hurts,
        include_method_disagreements=args.include_method_disagreements,
        include_near_threshold_correct=args.include_near_threshold_correct,
        near_threshold_margin=args.near_threshold_margin,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
