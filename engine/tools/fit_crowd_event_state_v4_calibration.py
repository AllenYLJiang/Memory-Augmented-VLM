#!/usr/bin/env python3
"""Fit and freeze a bounded Crowd Event-State V4 score contract on calibration data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, stable_sha1, write_json
from competition import _aggregate
from selection import is_pure_normal_video, label_codes
from validate_graph_candidates import _delta, _metrics


METHOD = "conditional_ot_full"
TARGET = "crowd_escalation_chain"


def _float_grid(text: str) -> list[float]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _competition_with_target(row: Mapping[str, Any], target_score: float) -> dict:
    context = row.get("frozen_competition_context", {})
    results = {
        str(key): dict(value)
        for key, value in context.get("graph_results", {}).get(METHOD, {}).items()
        if isinstance(value, Mapping)
    }
    if not results:
        raise ValueError("record lacks frozen_competition_context.graph_results")
    target = dict(results.get(TARGET, {}))
    target["graph_score"] = max(0.0, min(1.0, float(target_score)))
    results[TARGET] = target
    shortlist = context.get("graph_candidates", {})
    abnormal = [
        float(results[key].get("graph_score", 0.0))
        for key in shortlist.get("selected_abnormal", []) if key in results
    ]
    normal = [
        float(results[key].get("graph_score", 0.0))
        for key in shortlist.get("selected_normal", []) if key in results
    ]
    frozen = context.get("competitions", {}).get(METHOD, {})
    aggregation = str(frozen.get("aggregation", "logmeanexp"))
    temperature = float(frozen.get("temperature", 0.1) or 0.1)
    threshold = float(frozen.get("decision_margin_threshold", 0.03) or 0.03)
    margin = _aggregate(abnormal, aggregation, temperature) - _aggregate(normal, aggregation, temperature)
    return {"margin": margin, "y_pred": int(margin > threshold)}


def _values(
    rows: list[dict], mode: str, parameter: float, bound: float,
    uncertainty_max: float, none_max: float,
) -> list[tuple[int, int, float, str]]:
    values = []
    for row in rows:
        base_comp = row.get("base_competitions", {}).get(METHOD, {})
        event = row.get("candidate_event_state", {})
        states = event.get("state_probabilities", {})
        noop = (
            not bool(event.get("complete"))
            or float(event.get("uncertainty", 1.0)) > uncertainty_max
            or float(states.get("none_or_unobservable", 0.0)) > none_max
        )
        effective_parameter = 0.0 if noop else parameter
        if mode == "method_blend":
            details = row.get("score_contract", {}).get("method_details", {}).get(METHOD, {})
            base_score = float(details.get(
                "base_score", row.get("base_target", {}).get("results", {}).get(METHOD, {}).get("graph_score", 0.0)
            ))
            state_score = float(details.get("state_score", base_score))
            competition = _competition_with_target(
                row, (1.0 - effective_parameter) * base_score + effective_parameter * state_score,
            )
        else:
            positive = float(states.get("active_or_ongoing_physical_escalation", 0.0)) + float(
                states.get("causally_linked_aftermath", 0.0)
            )
            negative = float(states.get("pre_event_tension_or_flight", 0.0)) + float(
                states.get("benign_collective_activity", 0.0)
            ) + float(states.get("none_or_unobservable", 0.0))
            odds = math.log((positive + 1e-6) / (negative + 1e-6))
            residual = max(-bound, min(bound, effective_parameter * odds))
            margin = float(base_comp.get("margin", 0.0)) + residual
            threshold = float(base_comp.get("decision_margin_threshold", 0.03) or 0.03)
            competition = {"margin": margin, "y_pred": int(margin > threshold)}
        values.append((
            int(row.get("y_true", 0)), int(competition["y_pred"]), float(competition["margin"]),
            str(row.get("video_id", "")),
        ))
    return values


def _paired_effects(before, after) -> tuple[int, int]:
    helps = sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(before, after))
    hurts = sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(before, after))
    return helps, hurts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--refinement-mode", choices=("method_blend", "margin_residual"), default="method_blend")
    parser.add_argument("--alpha-grid", type=_float_grid, default=_float_grid("0,0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--beta-grid", type=_float_grid, default=_float_grid("0,0.01,0.02,0.03,0.05"))
    parser.add_argument("--residual-bounds", type=_float_grid, default=_float_grid("0.02,0.03,0.05"))
    parser.add_argument("--uncertainty-max", type=float, default=0.8)
    parser.add_argument("--none-max", type=float, default=0.8)
    parser.add_argument("--max-b1-recall-regression", type=float, default=0.05)
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.records) if row.get("candidate_scoring_mode") == "event_state_v4"]
    if not rows:
        raise SystemExit("no Event-State V4 calibration records")
    if any(not row.get("score_contract", {}).get("method_matched") for row in rows):
        raise SystemExit("calibration records contain unmatched score contracts")
    manifest_rows = list(iter_jsonl(args.manifest))
    manifest_by_key = {str(row.get("segment_key")): row for row in manifest_rows}
    missing = [str(row.get("segment_key")) for row in rows if str(row.get("segment_key")) not in manifest_by_key]
    if missing:
        raise SystemExit(f"records are not covered by calibration manifest; first={missing[0]}")
    source_groups = sorted({
        str(manifest_by_key[str(row.get("segment_key"))].get("source_group", ""))
        for row in rows
        if manifest_by_key[str(row.get("segment_key"))].get("source_group")
    })
    before = [
        (
            int(row.get("y_true", 0)),
            int(row.get("base_competitions", {}).get(METHOD, {}).get("y_pred", 0)),
            float(row.get("base_competitions", {}).get(METHOD, {}).get("margin", 0.0)),
            str(row.get("video_id", "")),
        )
        for row in rows
    ]
    base_metrics = _metrics(before)
    grid_rows = []
    parameters = args.alpha_grid if args.refinement_mode == "method_blend" else args.beta_grid
    bounds = [0.0] if args.refinement_mode == "method_blend" else args.residual_bounds
    for parameter in parameters:
        for bound in bounds:
            after_values = _values(
                rows, args.refinement_mode, parameter, bound,
                max(0.0, min(1.0, args.uncertainty_max)),
                max(0.0, min(1.0, args.none_max)),
            )
            metrics = _metrics(after_values)
            delta = _delta(metrics, base_metrics)
            helps, hurts = _paired_effects(before, after_values)
            b1_delta = delta.get("recall_by_class", {}).get("B1")
            feasible = (
                int(delta.get("pure_normal_fp", 0)) <= 0
                and (b1_delta is None or float(b1_delta) >= -args.max_b1_recall_regression)
            )
            grid_rows.append({
                "parameter": parameter, "residual_bound": bound, "feasible": feasible,
                "ap_delta": delta.get("ap", 0.0), "balanced_accuracy_delta": delta.get("balanced_accuracy", 0.0),
                "b4_recall_delta": delta.get("recall_by_class", {}).get("B4"),
                "b1_recall_delta": b1_delta, "pure_normal_fp_delta": delta.get("pure_normal_fp", 0),
                "helps": helps, "hurts": hurts,
            })
    selected = max(
        grid_rows,
        key=lambda row: (
            bool(row["feasible"]), float(row["ap_delta"]), float(row["balanced_accuracy_delta"]),
            float(row["b4_recall_delta"] or 0.0), int(row["helps"]) - int(row["hurts"]),
            -float(row["parameter"]), -float(row["residual_bound"]),
        ),
    )
    positive_signal = bool(
        selected["feasible"] and float(selected["parameter"]) > 0.0
        and float(selected["ap_delta"]) > 0.0 and float(selected["balanced_accuracy_delta"]) >= 0.0
    )
    config = {
        "version": "crowd_event_state_v4_frozen_calibration_v1",
        "calibration_id": stable_sha1({
            "records": file_sha256(args.records), "manifest": file_sha256(args.manifest),
            "mode": args.refinement_mode, "selection": selected,
        }, size=20),
        "frozen": True,
        "deployable_signal_on_calibration": positive_signal,
        "refinement_mode": args.refinement_mode,
        "alpha": float(selected["parameter"]) if args.refinement_mode == "method_blend" else 0.0,
        "beta": float(selected["parameter"]) if args.refinement_mode == "margin_residual" else 0.0,
        "residual_bound": float(selected["residual_bound"]),
        "uncertainty_max": max(0.0, min(1.0, args.uncertainty_max)),
        "none_max": max(0.0, min(1.0, args.none_max)),
        "method_calibration": {METHOD: {"scale": 1.0, "offset": 0.0}},
        "source_groups": source_groups,
        "source_groups_sha256": stable_sha1(source_groups, size=40),
        "calibration_records": str(args.records),
        "calibration_records_sha256": file_sha256(args.records),
        "calibration_manifest": str(args.manifest),
        "calibration_manifest_sha256": file_sha256(args.manifest),
        "selection": selected,
        "selection_rule": "feasible_then_max_AP_then_BA_then_B4_then_net_fixes_then_smallest_coefficient",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, config)
    grid_path = args.out.with_name("calibration_grid.csv")
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader(); writer.writerows(grid_rows)
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
