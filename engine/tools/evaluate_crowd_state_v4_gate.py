#!/usr/bin/env python3
"""Evaluate the preregistered source-disjoint Crowd Event-State V4 gate."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import iter_jsonl, write_json
from selection import is_pure_normal_video, label_codes, source_group_id
from validate_graph_candidates import _delta, _metrics


METHOD = "conditional_ot_full"


def _values(rows: list[dict], side: str) -> list[tuple[int, int, float, str]]:
    return [
        (
            int(row.get("y_true", 0)),
            int(row.get(side, {}).get(METHOD, {}).get("y_pred", 0)),
            float(row.get(side, {}).get(METHOD, {}).get("margin", 0.0)),
            str(row.get("video_id", "")),
        )
        for row in rows
    ]


def _is_post_event_normal(row: dict) -> bool:
    if int(row.get("y_true", 0)) != 0:
        return False
    phase = str(row.get("event_phase", row.get("gt", {}).get("event_phase", ""))).lower()
    if phase == "post_event" or "post_event" in phase:
        return True
    start = int(row.get("start_frame", row.get("source_record", {}).get("start_frame", 0)) or 0)
    intervals = row.get("gt", {}).get("intervals", [])
    ends = [int(item[1]) for item in intervals if isinstance(item, (list, tuple)) and len(item) >= 2]
    return bool(ends and max(ends) < start)


def _is_hard_context_normal(row: dict) -> bool:
    if int(row.get("y_true", 0)) != 0:
        return False
    stratum = str(row.get("evaluation_stratum", row.get("stratum", ""))).lower()
    return bool(
        row.get("hard_context_normal")
        or "hard" in stratum
        or "context" in stratum
        or "post_event" in stratum
        or _is_post_event_normal(row)
    )


def _subset_fp(rows: list[dict], predicate) -> dict[str, int]:
    selected = [row for row in rows if predicate(row)]
    return {
        "n": len(selected),
        "before": sum(int(row.get("base_competitions", {}).get(METHOD, {}).get("y_pred", 0)) == 1 for row in selected),
        "after": sum(int(row.get("candidate_competitions", {}).get(METHOD, {}).get("y_pred", 0)) == 1 for row in selected),
    }


def _residual_saturation(rows: list[dict]) -> dict[str, float | int]:
    values: list[tuple[float, float, float | None]] = []
    for row in rows:
        detail = row.get("score_contract", {}).get("method_details", {}).get(METHOD, {})
        residual = abs(float(detail.get("residual", 0.0) or 0.0))
        bound = abs(float(detail.get("residual_bound", 0.0) or 0.0))
        if bound > 0.0:
            explicit_ratio = detail.get("saturation_ratio")
            values.append((residual, bound, float(explicit_ratio) if explicit_ratio is not None else None))
    saturated = sum(
        (ratio >= 0.95 if ratio is not None else residual >= bound - 1e-9)
        for residual, bound, ratio in values
    )
    return {
        "n_with_bound": len(values),
        "saturated": saturated,
        "rate": saturated / len(values) if values else 0.0,
    }


def _bootstrap_probability(rows: list[dict], samples: int, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_group_id(str(row.get("video_id", "")))].append(row)
    groups = sorted(grouped)
    rng = random.Random(seed)
    ba_positive = ap_positive = valid = 0
    for _ in range(max(1, samples)):
        sampled = [item for _ in groups for item in grouped[rng.choice(groups)]]
        if not sampled:
            continue
        before, after = _metrics(_values(sampled, "base_competitions")), _metrics(_values(sampled, "candidate_competitions"))
        delta = _delta(after, before)
        ba_positive += int(float(delta.get("balanced_accuracy", 0.0)) > 0.0)
        ap_positive += int(float(delta.get("ap", 0.0)) > 0.0)
        valid += 1
    return {
        "source_groups": len(groups), "samples": valid,
        "p_balanced_accuracy_delta_positive": ba_positive / valid if valid else 0.0,
        "p_ap_delta_positive": ap_positive / valid if valid else 0.0,
    }


def _single_run(rows: list[dict], summary: dict, args) -> dict:
    target_rows = [row for row in rows if bool(row.get("eligible_for_target_effect", True))]
    # Legacy V4 files contain only the target-exposed subset. Full-packet coverage must
    # be explicit; otherwise an old 120/208 subset would be mislabeled as the whole packet.
    full_packet_rows = [row for row in rows if bool(row.get("full_packet_eligible", False))]
    full_before_values = _values(full_packet_rows, "base_competitions")
    full_after_values = _values(full_packet_rows, "candidate_competitions")
    full_before = _metrics(full_before_values)
    full_after = _metrics(full_after_values)
    full_helps = sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(full_before_values, full_after_values))
    full_hurts = sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(full_before_values, full_after_values))
    rows = target_rows
    before_values, after_values = _values(rows, "base_competitions"), _values(rows, "candidate_competitions")
    before, after = _metrics(before_values), _metrics(after_values)
    delta = _delta(after, before)
    helps = sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(before_values, after_values))
    hurts = sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(before_values, after_values))
    flips = [row for row, b, a in zip(rows, before_values, after_values) if b[1] != a[1]]
    pure_normal_fp_before = sum(y == 0 and p == 1 and is_pure_normal_video(video) for y, p, _, video in before_values)
    pure_normal_fp_after = sum(y == 0 and p == 1 and is_pure_normal_video(video) for y, p, _, video in after_values)
    pure_normal_n = sum(y == 0 and is_pure_normal_video(video) for y, _, _, video in before_values)
    b4_positive_n = sum(y == 1 and "B4" in label_codes(video) for y, _, _, video in before_values)
    b1_positive_n = sum(y == 1 and "B1" in label_codes(video) for y, _, _, video in before_values)
    calibration_groups = {
        str(value)
        for row in rows for value in row.get("state_calibration", {}).get("source_groups", [])
    }
    # Current records store a hash in the score trace and source groups in the frozen calibration summary.
    validation_groups = {source_group_id(str(row.get("video_id", ""))) for row in rows}
    explicit_overlap = calibration_groups & validation_groups
    matched = all(bool(row.get("score_contract", {}).get("method_matched")) for row in rows)
    no_repeated_gates = all(int(row.get("score_contract", {}).get("repeated_gate_count", 0) or 0) == 0 for row in rows)
    frozen_calibration = all(bool(row.get("state_calibration", {}).get("frozen")) for row in rows)
    direct_flips = all(bool(row.get("frozen_non_target_graph_results")) for row in flips)
    bootstrap = _bootstrap_probability(rows, args.bootstrap_samples, args.bootstrap_seed)
    hard_context_fp = _subset_fp(rows, _is_hard_context_normal)
    post_event_fp = _subset_fp(rows, _is_post_event_normal)
    saturation = _residual_saturation(rows)
    b1_delta = delta.get("recall_by_class", {}).get("B1")
    b4_delta = delta.get("recall_by_class", {}).get("B4")
    checks = {
        "minimum_eligible_packet": len(rows) >= args.minimum_eligible,
        "minimum_pure_label_a_coverage": pure_normal_n >= args.minimum_pure_label_a,
        "minimum_b4_positive_coverage": b4_positive_n >= args.minimum_b4_positive,
        "minimum_b1_canary_coverage": b1_positive_n >= args.minimum_b1_positive,
        "ap_delta_positive": float(delta.get("ap", 0.0)) > 0.0,
        "balanced_accuracy_delta_positive": float(delta.get("balanced_accuracy", 0.0)) > 0.0,
        "b4_recall_delta_positive": b4_delta is not None and float(b4_delta) > 0.0,
        "helps_exceed_hurts": helps > hurts,
        "pure_label_a_fp_not_increased": pure_normal_fp_after <= pure_normal_fp_before,
        "minimum_hard_context_coverage": hard_context_fp["n"] >= args.minimum_hard_context_normal,
        "minimum_post_event_normal_coverage": post_event_fp["n"] >= args.minimum_post_event_normal,
        "hard_context_fp_not_increased": hard_context_fp["after"] <= hard_context_fp["before"],
        "post_event_normal_fp_not_increased": post_event_fp["after"] <= post_event_fp["before"],
        "b1_regression_bounded": b1_delta is None or float(b1_delta) >= -args.max_b1_recall_regression,
        "flips_are_direct_target_effects": bool(flips) and direct_flips,
        "bootstrap_supports_ba_gain": bootstrap["p_balanced_accuracy_delta_positive"] >= args.bootstrap_probability_threshold,
        "method_score_contracts_matched": matched,
        "no_repeated_state_gates": no_repeated_gates,
        "frozen_calibration_used": frozen_calibration,
        "calibration_validation_source_disjoint": bool(calibration_groups) and not explicit_overlap,
        "residual_saturation_rate_bounded": (
            saturation["n_with_bound"] == 0 or saturation["rate"] <= args.max_residual_saturation_rate
        ),
        "phase_specific_weights_frozen": all(
            bool(row.get("score_contract", {}).get("phase_specific_weights_frozen", False))
            or str(row.get("candidate_scoring_mode", "")) == "event_state_v4"
            for row in rows
        ),
        "target_exposed_effect_reported": bool(target_rows),
        "whole_packet_effect_reported": bool(full_packet_rows),
    }
    ranking_checks = {
        key: checks[key]
        for key in (
            "minimum_eligible_packet", "ap_delta_positive", "method_score_contracts_matched",
            "no_repeated_state_gates", "frozen_calibration_used",
            "calibration_validation_source_disjoint", "target_exposed_effect_reported",
            "whole_packet_effect_reported",
        )
    }
    binary_checks = {key: value for key, value in checks.items() if key != "ap_delta_positive"}
    return {
        "checks": checks,
        "ranking_checks": ranking_checks,
        "binary_checks": binary_checks,
        "ranking_gate_passed": all(ranking_checks.values()),
        "binary_gate_passed": all(binary_checks.values()),
        "passed": all(checks.values()), "base": before, "candidate": after,
        "delta": delta, "helps": helps, "hurts": hurts, "prediction_flips": len(flips),
        "pure_label_a_fp_before": pure_normal_fp_before, "pure_label_a_fp_after": pure_normal_fp_after,
        "pure_label_a_n": pure_normal_n, "b4_positive_n": b4_positive_n, "b1_positive_n": b1_positive_n,
        "hard_context_fp": hard_context_fp, "post_event_normal_fp": post_event_fp,
        "residual_saturation": saturation,
        "target_exposed_n": len(target_rows), "whole_packet_n": len(full_packet_rows),
        "whole_packet_effect": {
            "base": full_before, "candidate": full_after, "delta": _delta(full_after, full_before),
            "helps": full_helps, "hurts": full_hurts,
        },
        "bootstrap": bootstrap, "summary_declared_eligible": int(summary.get("eligible", 0)),
        "validation_source_groups": sorted(validation_groups), "explicit_source_group_overlap": sorted(explicit_overlap),
    }


def _load(summary_path: Path, results_path: Path, args) -> tuple[dict, list[dict], dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(results_path))
    return summary, rows, _single_run(rows, summary, args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--replicate-2-summary", type=Path)
    parser.add_argument("--replicate-2-results", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--minimum-eligible", type=int, default=50)
    parser.add_argument("--minimum-pure-label-a", type=int, default=5)
    parser.add_argument("--minimum-b4-positive", type=int, default=20)
    parser.add_argument("--minimum-b1-positive", type=int, default=10)
    parser.add_argument("--minimum-hard-context-normal", type=int, default=5)
    parser.add_argument("--minimum-post-event-normal", type=int, default=2)
    parser.add_argument("--max-b1-recall-regression", type=float, default=0.05)
    parser.add_argument("--max-residual-saturation-rate", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--bootstrap-probability-threshold", type=float, default=0.8)
    parser.add_argument("--replicate-prediction-agreement", type=float, default=0.8)
    args = parser.parse_args()
    _, rows1, run1 = _load(args.summary, args.results, args)
    run2 = None
    replicate = {"available": False, "passed": False}
    if bool(args.replicate_2_summary) != bool(args.replicate_2_results):
        raise SystemExit("replicate 2 requires both summary and results")
    if args.replicate_2_summary and args.replicate_2_results:
        _, rows2, run2 = _load(args.replicate_2_summary, args.replicate_2_results, args)
        by_key2 = {str(row.get("segment_key")): row for row in rows2}
        shared = [row for row in rows1 if str(row.get("segment_key")) in by_key2]
        agreement = sum(
            int(row.get("candidate_competitions", {}).get(METHOD, {}).get("y_pred", 0))
            == int(by_key2[str(row.get("segment_key"))].get("candidate_competitions", {}).get(METHOD, {}).get("y_pred", 0))
            for row in shared
        ) / len(shared) if shared else 0.0
        replicate = {
            "available": True, "shared_windows": len(shared), "candidate_prediction_agreement": agreement,
            "both_runs_pass_core_gate": bool(run1["passed"] and run2["passed"]),
            "passed": bool(run1["passed"] and run2["passed"] and agreement >= args.replicate_prediction_agreement),
        }
    if not run1["binary_gate_passed"]:
        decision = "STOP_AFTER_REPLICATE_1"
        legacy_decision = "DO_NOT_PROCEED_TO_STEP_4"
    elif not replicate["available"]:
        decision = "RUN_REPLICATE_2"
        legacy_decision = "RUN_REPLICATE_2"
    elif replicate["passed"]:
        decision = "PROCEED_TO_NEW_RESEARCH_STAGE"
        legacy_decision = "PROCEED_TO_FRESH_STEP_4"
    else:
        decision = "STOP_AFTER_REPLICATE_1"
        legacy_decision = "DO_NOT_PROCEED_TO_STEP_4"
    result = {
        "version": "crowd_event_state_v4_progression_gate_v4",
        "decision": decision, "legacy_decision": legacy_decision,
        "ranking_gate_passed": bool(run1["ranking_gate_passed"]),
        "binary_gate_passed": bool(run1["binary_gate_passed"]),
        "passed": decision == "PROCEED_TO_NEW_RESEARCH_STAGE",
        "replicate_1": run1, "replicate_2": run2, "replicate_agreement": replicate,
        "original_help_survival": "diagnostic_only_not_a_gate",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "progression_gate_v3.json", result)
    lines = ["# Crowd Event-State V4 Progression Gate", "", f"**Decision: `{decision}`**", "", "## Replicate 1", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{key}`" for key, value in run1["checks"].items())
    lines.extend(["", "## Replicate", "", "```json", json.dumps(replicate, indent=2), "```", ""])
    (args.out_dir / "PROGRESSION_GATE_V3.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
