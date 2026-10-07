#!/usr/bin/env python3
"""Audit whether a frozen baseline exposes the V5 target in required negative tails."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import iter_jsonl, write_json
from temporal_contract import CONTRACT_VERSION, window_errors


def _records(path: Path) -> dict[str, dict]:
    if path.is_dir():
        path = path / "ot_window_results.jsonl"
    return {str(row.get("segment_key", "")): row for row in iter_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-key", default="crowd_escalation_chain")
    parser.add_argument("--minimum-hard-context-exposed", type=int, default=5)
    parser.add_argument("--minimum-post-event-exposed", type=int, default=2)
    parser.add_argument("--minimum-pure-normal-exposed", type=int, default=5)
    parser.add_argument("--minimum-baseline-coverage", type=float, default=0.99)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    records = _records(args.baseline_records)
    all_counts, found_counts, exposed_counts, missing_counts = Counter(), Counter(), Counter(), Counter()
    missing = []
    invalid = []
    for meta in iter_jsonl(args.manifest):
        key = str(meta.get("segment_key", ""))
        stratum = str(meta.get("stratum", meta.get("crowd_v4_stratum", "unstratified")))
        all_counts[stratum] += 1
        row = records.get(key)
        if row is None:
            missing.append(key)
            missing_counts[stratum] += 1
            continue
        found_counts[stratum] += 1
        issues = window_errors(row)
        if issues:
            invalid.append({"segment_key": key, "errors": issues})
            continue
        shortlist = row.get("graph_candidates", {})
        selected = set(shortlist.get("selected_abnormal", [])) | set(shortlist.get("selected_normal", []))
        if args.target_key in selected:
            exposed_counts[stratum] += 1
    hard_exposed = sum(value for key, value in exposed_counts.items() if "hard_context" in key or "hard_or_context" in key)
    post_exposed = sum(value for key, value in exposed_counts.items() if "post_event" in key)
    pure_exposed = sum(value for key, value in exposed_counts.items() if "pure_normal" in key)
    manifest_windows = sum(all_counts.values())
    found_windows = sum(found_counts.values())
    baseline_coverage = found_windows / manifest_windows if manifest_windows else 0.0
    conditions = {
        "baseline_coverage_met": baseline_coverage >= float(args.minimum_baseline_coverage),
        "frozen_evidence_valid": not invalid,
        "hard_context_exposure_met": hard_exposed >= args.minimum_hard_context_exposed,
        "post_event_exposure_met": post_exposed >= args.minimum_post_event_exposed,
        "pure_normal_exposure_met": pure_exposed >= args.minimum_pure_normal_exposed,
    }
    ready = bool(manifest_windows) and all(conditions.values())
    result = {
        "version": "crowd_v5_target_exposure_audit_v4_temporal_contract", "target_key": args.target_key,
        "temporal_contract_version": CONTRACT_VERSION,
        "invalid_baseline_records": invalid,
        "valid_baseline_records": found_windows - len(invalid),
        "valid_baseline_coverage": (found_windows - len(invalid)) / manifest_windows if manifest_windows else 0.0,
        "manifest_windows": manifest_windows, "baseline_records_found": found_windows,
        "baseline_coverage": baseline_coverage,
        "minimum_baseline_coverage": float(args.minimum_baseline_coverage),
        "missing_baseline_records": missing, "all_by_stratum": dict(sorted(all_counts.items())),
        "baseline_found_by_stratum": dict(sorted(found_counts.items())),
        "missing_by_stratum": dict(sorted(missing_counts.items())),
        "target_exposed_by_stratum": dict(sorted(exposed_counts.items())),
        "hard_context_target_exposed": hard_exposed, "post_event_target_exposed": post_exposed,
        "pure_normal_target_exposed": pure_exposed,
        "minimum_hard_context_target_exposed": args.minimum_hard_context_exposed,
        "minimum_post_event_target_exposed": args.minimum_post_event_exposed,
        "minimum_pure_normal_target_exposed": args.minimum_pure_normal_exposed,
        "ready_conditions": conditions,
        "ready_for_v5_calls": ready,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
