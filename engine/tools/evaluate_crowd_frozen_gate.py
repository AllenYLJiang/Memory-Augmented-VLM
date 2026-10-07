#!/usr/bin/env python3
"""Evaluate the preregistered crowd frozen-evidence progression gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import iter_jsonl, read_json, write_json


def _number(value, default=0.0) -> float:
    return float(default if value is None else value)


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


def _prediction(row: dict, side: str = "competitions") -> int:
    return int(row.get(side, {}).get("conditional_ot_full", {}).get("y_pred", 0))


def evaluate(
    summary: dict,
    rows: list[dict],
    original_baseline: dict[str, dict],
    original_candidate: dict[str, dict],
    minimum_eligible: int,
) -> dict:
    delta = summary.get("delta", {})
    class_delta = delta.get("recall_by_class", {})
    b1_hurts = 0
    direct_flips = 0
    total_flips = 0
    for row in rows:
        base = int(row.get("base_competitions", {}).get("conditional_ot_full", {}).get("y_pred", 0))
        candidate = int(row.get("candidate_competitions", {}).get("conditional_ot_full", {}).get("y_pred", 0))
        truth = int(row.get("y_true", 0))
        if base != candidate:
            total_flips += 1
            direct_flips += int(bool(row.get("frozen_non_target_graph_results")))
        gt = row.get("gt", {})
        classes = set(gt.get("anomaly_codes", []) or gt.get("label_codes", []) or [])
        video_id = str(row.get("video_id", ""))
        if ("B1" in classes or "label_B1" in video_id) and base == truth and candidate != truth:
            b1_hurts += 1

    original_help_keys = {
        key for key, baseline in original_baseline.items()
        if key in original_candidate
        and _prediction(baseline) != int(baseline.get("y_true", 0))
        and _prediction(original_candidate[key]) == int(baseline.get("y_true", 0))
    }
    eligible_keys = {str(row.get("segment_key", "")) for row in rows}
    eligible_original_help_keys = original_help_keys & eligible_keys
    surviving_original_help_keys = {
        str(row.get("segment_key", "")) for row in rows
        if str(row.get("segment_key", "")) in eligible_original_help_keys
        and int(row.get("base_competitions", {}).get("conditional_ot_full", {}).get("y_pred", 0)) != int(row.get("y_true", 0))
        and int(row.get("candidate_competitions", {}).get("conditional_ot_full", {}).get("y_pred", 0)) == int(row.get("y_true", 0))
    }
    required_surviving_helps = math.ceil(len(eligible_original_help_keys) / 2)
    checks = {
        "minimum_eligible_packet": int(summary.get("eligible", 0)) >= minimum_eligible,
        "ap_delta_positive": _number(delta.get("ap")) > 0.0,
        "balanced_accuracy_delta_positive": _number(delta.get("balanced_accuracy")) > 0.0,
        "b4_recall_delta_positive": _number(class_delta.get("B4")) > 0.0,
        "helps_exceed_hurts": int(summary.get("helps", 0)) > int(summary.get("hurts", 0)),
        "b1_harm_bounded": b1_hurts <= 1,
        "half_original_helps_survive": bool(eligible_original_help_keys) and len(surviving_original_help_keys) >= required_surviving_helps,
        "flips_are_direct_target_effects": total_flips > 0 and direct_flips == total_flips,
    }
    passed = all(checks.values())
    return {
        "version": "crowd_frozen_progression_gate_v2",
        "decision": "PROCEED_TO_STEP_4" if passed else "DO_NOT_PROCEED_TO_STEP_4",
        "passed": passed,
        "checks": checks,
        "diagnostics": {
            "eligible": int(summary.get("eligible", 0)),
            "minimum_eligible": minimum_eligible,
            "original_helps_total": len(original_help_keys),
            "original_helps_eligible": len(eligible_original_help_keys),
            "original_help_segment_keys_eligible": sorted(eligible_original_help_keys),
            "required_surviving_helps": required_surviving_helps,
            "surviving_original_helps": len(surviving_original_help_keys),
            "surviving_original_help_segment_keys": sorted(surviving_original_help_keys),
            "observed_helps": int(summary.get("helps", 0)),
            "observed_hurts": int(summary.get("hurts", 0)),
            "b1_hurts": b1_hurts,
            "total_prediction_flips": total_flips,
            "direct_target_flips": direct_flips,
            "ap_delta": delta.get("ap"),
            "balanced_accuracy_delta": delta.get("balanced_accuracy"),
            "b4_recall_delta": class_delta.get("B4"),
            "b1_recall_delta": class_delta.get("B1"),
            "fp_delta": delta.get("fp"),
            "fn_delta": delta.get("fn"),
        },
        "interpretation": (
            "Passing authorizes Step 4 research only; it does not activate or deploy either phase graph."
            if passed else
            "Keep the governed 13-graph library unchanged and do not run Step 4 from this candidate result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--original-baseline-records", required=True, type=Path)
    parser.add_argument("--original-candidate-records", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--minimum-eligible", type=int, default=20)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(args.results))
    result = evaluate(
        summary, rows,
        _records(args.original_baseline_records),
        _records(args.original_candidate_records),
        args.minimum_eligible,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "progression_gate.json", result)
    lines = [
        "# Crowd Frozen-Evidence Progression Gate", "",
        f"**Decision: `{result['decision']}`**", "", "## Checks", "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in result["checks"].items()
    )
    lines.extend(["", "## Diagnostics", "", "```json", json.dumps(result["diagnostics"], indent=2), "```", "", result["interpretation"], ""])
    (args.out_dir / "PROGRESSION_GATE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
