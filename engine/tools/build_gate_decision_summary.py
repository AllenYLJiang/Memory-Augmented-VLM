#!/usr/bin/env python3
"""Combine critic, consolidation, and catalog gate results into one summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json


def build(critic_path: Path, consolidation_path: Path, manifest_path: Path,
          out_path: Path, final_critic_round: int) -> dict:
    critic = json.loads(critic_path.read_text(encoding="utf-8"))
    consolidation = json.loads(consolidation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = {
        "version": "reflective_gate_decision_summary_v1",
        "final_critic_round": final_critic_round,
        "input": critic.get("input", 0),
        "validation_ready": critic.get("approved_for_validation", 0),
        "risk_only_promoted": critic.get("risk_only_promoted", 0),
        "validation_ready_with_risks": consolidation.get("validation_ready_with_risks", 0),
        "revision_requested": critic.get("revision_requested", 0),
        "revision_blocked_at_max_rounds": critic.get("human_review_required", 0),
        "terminal_rejected": critic.get("terminal_rejected", 0),
        "representation_only": critic.get("representation_only", 0),
        "recurrence_sufficient_before_critic": consolidation.get(
            "recurrence_sufficient_before_critic", 0
        ),
        "recurrence_sufficient_after_critic": consolidation.get(
            "recurrence_sufficient_after_critic", 0
        ),
        "candidate_graphs": consolidation.get("candidate_graphs", 0),
        "validation_actions": len(manifest.get("actions", [])),
        "catalog_blocked": bool(manifest.get("blocked", False)),
        "catalog_empty_reason": manifest.get("empty_reason", ""),
        "deployable": False,
        "sources": {
            "critic_summary": str(critic_path),
            "consolidation_summary": str(consolidation_path),
            "validation_catalog_manifest": str(manifest_path),
        },
    }
    write_json(out_path, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-summary", required=True, type=Path)
    parser.add_argument("--consolidation-summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--final-critic-round", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(build(
        args.critic_summary, args.consolidation_summary, args.manifest,
        args.out, args.final_critic_round,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
