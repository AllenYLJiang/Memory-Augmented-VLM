#!/usr/bin/env python3
"""Consolidate critic-approved hypotheses; never activate them."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from common import iter_jsonl, write_json, write_jsonl
from discover_ot_failures import _graph_signature
from hypothesis_contract import atomic_contract_errors, bundle_contract_errors


def consolidate(input_path: Path, out_dir: Path, min_support_groups: int) -> dict:
    rows = list(iter_jsonl(input_path))
    bundle_errors = bundle_contract_errors(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    passthrough = []
    for row in rows:
        graph = row.get("graph", {})
        contract_errors = atomic_contract_errors(row)
        row_id = str(row.get("id", ""))
        row_bundle_errors = bundle_errors.get(row_id, [])
        if (
            row.get("status") == "critic_approved_candidate"
            and isinstance(graph, Mapping) and graph
            and not contract_errors and not row_bundle_errors
        ):
            signature = _graph_signature(graph)
            operation = str(row.get("operation", "ADD")).upper()
            # Equivalent ADD proposals may consolidate across clusters. UPDATE utility belongs to
            # one exact target and must never be merged with another target that happens to have
            # similar node semantics.
            target = str((row.get("targets") or [""])[0]) if operation == "UPDATE" else ""
            grouped[f"{operation}:{target}:{signature}"].append(row)
        else:
            value = dict(row)
            if row.get("status") == "critic_approved_candidate" and (contract_errors or row_bundle_errors):
                value["status"] = "contract_rejected"
                value["contract_errors"] = sorted(set([*contract_errors, *row_bundle_errors]))
            passthrough.append(value)
    candidates = []
    for signature, values in grouped.items():
        first = dict(values[0])
        support_groups = sorted({
            str(group) for value in values for group in value.get("support_source_groups", []) if group
        })
        support_cases = sorted({
            str(case) for value in values for case in value.get("support_case_ids", []) if case
        })
        support_videos = sorted({
            str(video) for value in values for video in value.get("support_video_ids", []) if video
        })
        first["support_source_groups"] = support_groups
        first["support_case_ids"] = support_cases
        first["support_video_ids"] = support_videos
        first["consolidated_hypothesis_ids"] = [value.get("id") for value in values]
        first["signature"] = signature
        first["active"] = False
        first["status"] = "candidate" if len(support_groups) >= min_support_groups else "insufficient_recurrence"
        first["validation"] = {"status": "pending", "requires_held_out_source_groups": True}
        first["held_out_validation_risks"] = first.get("critic", {}).get(
            "held_out_validation_risks", []
        )
        candidates.append(first)
    recurrence_sufficient_before_critic = sum(
        isinstance(row.get("graph"), Mapping)
        and bool(row.get("graph"))
        and not atomic_contract_errors(row)
        and not bundle_errors.get(str(row.get("id", "")), [])
        and len({str(value) for value in row.get("support_source_groups", []) if value})
            >= min_support_groups
        for row in rows
    )
    recurrence_sufficient_after_critic = sum(row["status"] == "candidate" for row in candidates)
    gate_counts = {
        "critic_approved": sum(row.get("status") == "critic_approved_candidate" for row in rows),
        "risk_only_promoted": sum(
            row.get("critic", {}).get("raw_recommendation") == "revise"
            and row.get("critic", {}).get("recommendation") == "accept_for_validation"
            and row.get("status") == "critic_approved_candidate"
            for row in rows
        ),
        "validation_ready_with_risks": sum(
            row.get("status") == "critic_approved_candidate"
            and bool(row.get("critic", {}).get("held_out_validation_risks", []))
            for row in rows
        ),
        "blocked_by_critic": sum(row.get("status") in {
            "revision_requested", "human_review_required", "critic_rejected", "diagnostic_only",
        } for row in rows),
        "blocked_by_recurrence": sum(row.get("status") == "insufficient_recurrence" for row in candidates),
        "blocked_by_contract": sum(bool(atomic_contract_errors(row)) for row in rows),
        "blocked_by_bundle_contract": len(bundle_errors),
        "recurrence_sufficient_before_critic": recurrence_sufficient_before_critic,
        "recurrence_sufficient_after_critic": recurrence_sufficient_after_critic,
    }
    registry = {
        "version": "theory_governed_graph_hypothesis_registry_v1",
        "candidate_graphs": candidates,
        "diagnostic_memories": passthrough,
        "gate_counts": gate_counts,
        "activation_policy": "Only validate_graph_candidates.py may set validation.status=validated; only build_active_graph_library.py may activate.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "graph_hypothesis_registry.json", registry)
    write_jsonl(out_dir / "graph_hypotheses_consolidated.jsonl", candidates)
    source_groups = sorted({group for row in candidates for group in row.get("support_source_groups", [])})
    source_videos = sorted({video for row in candidates for video in row.get("support_video_ids", [])})
    (out_dir / "discovery_source_groups.txt").write_text("".join(f"{value}\n" for value in source_groups), encoding="utf-8")
    (out_dir / "discovery_source_videos.txt").write_text("".join(f"{value}\n" for value in source_videos), encoding="utf-8")
    summary = {
        "version": "graph_memory_consolidation_v1", "input": len(rows),
        "candidate_graphs": len(candidates),
        "recurrence_eligible": recurrence_sufficient_after_critic,
        "recurrence_sufficient_before_critic": recurrence_sufficient_before_critic,
        "recurrence_sufficient_after_critic": recurrence_sufficient_after_critic,
        "validation_ready_but_insufficient_recurrence": gate_counts["blocked_by_recurrence"],
        "critic_approved": gate_counts["critic_approved"],
        "risk_only_promoted": gate_counts["risk_only_promoted"],
        "validation_ready_with_risks": gate_counts["validation_ready_with_risks"],
        "blocked_by_critic": gate_counts["blocked_by_critic"],
        "blocked_by_bundle_contract": gate_counts["blocked_by_bundle_contract"],
        "diagnostic_memories": len(passthrough),
        "contract_rejected": sum(row.get("status") == "contract_rejected" for row in passthrough),
        "min_support_groups": min_support_groups,
        "gate_counts": gate_counts,
    }
    write_json(out_dir / "consolidation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-support-groups", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.input, args.out_dir, args.min_support_groups), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
