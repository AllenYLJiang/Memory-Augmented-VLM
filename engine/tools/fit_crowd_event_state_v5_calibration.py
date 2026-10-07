#!/usr/bin/env python3
"""Fit a frozen, source-group-audited Crowd Event-State V5 occupancy contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from common import file_sha256, iter_jsonl, stable_sha1, write_json
from selection import is_pure_normal_video, label_codes
from validate_graph_candidates import _delta, _metrics


METHOD = "conditional_ot_full"


def _float_grid(text: str) -> list[float]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def _fold(group: str, folds: int, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _manifest_meta(row: Mapping[str, Any], manifest_by_key: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    return manifest_by_key.get(str(row.get("segment_key", "")), {})


def _is_post_event_normal(row: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    return int(row.get("y_true", 0)) == 0 and str(meta.get("event_phase", row.get("event_phase", ""))).lower() == "post_event"


def _is_hard_context_normal(row: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    if int(row.get("y_true", 0)) != 0:
        return False
    stratum = str(meta.get("stratum", row.get("evaluation_stratum", ""))).lower()
    return "hard" in stratum or "context" in stratum or "post_event" in stratum or _is_post_event_normal(row, meta)


def _is_pure_normal(row: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    if int(row.get("y_true", 0)) != 0:
        return False
    stratum = str(meta.get("stratum", row.get("evaluation_stratum", ""))).lower()
    return "pure_normal" in stratum and is_pure_normal_video(str(row.get("video_id", "")))


def _base_value(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    comp = row.get("base_competitions", {}).get(METHOD, {})
    return (
        int(row.get("y_true", 0)), int(comp.get("y_pred", 0)), float(comp.get("margin", 0.0)),
        str(row.get("video_id", "")),
    )


def _candidate_value(
    row: Mapping[str, Any], beta_active: float, beta_aftermath: float,
    beta_normal_confound: float, bound: float,
    uncertainty_max: float, none_max: float,
) -> tuple[tuple[int, int, float, str], float]:
    comp = row.get("base_competitions", {}).get(METHOD, {})
    event = row.get("candidate_event_state", {})
    context = event.get("event_context_state_probabilities", {})
    occupancy = max(0.0, min(1.0, float(event.get("current_window_active_occupancy_probability", 0.0))))
    normal_confound = max(
        0.0, min(1.0, float(event.get("current_window_normal_confound_probability", 0.0)))
    )
    aftermath = max(0.0, min(1.0, float(event.get("aftermath_context_probability", 0.0))))
    support = max(0.0, min(1.0, float(event.get("aftermath_current_window_occupancy_support_probability", 0.0))))
    noop = (
        not bool(event.get("complete"))
        or float(event.get("uncertainty", 1.0)) > uncertainty_max
        or float(context.get("none", 0.0)) > none_max
    )
    logit = math.log((occupancy + 1e-6) / (1.0 - occupancy + 1e-6))
    normal_logit = math.log((normal_confound + 1e-6) / (1.0 - normal_confound + 1e-6))
    raw = (
        beta_active * logit
        + beta_aftermath * aftermath * support
        - beta_normal_confound * max(0.0, normal_logit)
    )
    residual = 0.0 if noop or bound <= 0.0 else bound * math.tanh(raw / bound)
    margin = float(comp.get("margin", 0.0)) + residual
    threshold = float(comp.get("decision_margin_threshold", 0.03) or 0.03)
    value = (int(row.get("y_true", 0)), int(margin > threshold), margin, str(row.get("video_id", "")))
    return value, abs(residual) / bound if bound > 0.0 else 0.0


def _paired_effects(before, after) -> tuple[int, int]:
    return (
        sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(before, after)),
        sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(before, after)),
    )


def _fp(rows, values, manifest_by_key, predicate) -> tuple[int, int, int]:
    selected = [
        index for index, row in enumerate(rows)
        if predicate(row, _manifest_meta(row, manifest_by_key))
    ]
    before = sum(_base_value(rows[index])[1] == 1 for index in selected)
    after = sum(values[index][1] == 1 for index in selected)
    return len(selected), before, after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--ablation",
        choices=("active_only", "phase_aware", "signed_confound", "signed_phase_aware"),
        default="signed_confound",
    )
    parser.add_argument("--beta-active-grid", type=_float_grid, default=_float_grid("0,0.0025,0.005,0.01,0.02"))
    parser.add_argument("--aftermath-ratio-grid", type=_float_grid, default=_float_grid("0,0.25,0.5"))
    parser.add_argument(
        "--normal-confound-ratio-grid", type=_float_grid,
        default=_float_grid("0,0.5,1,1.5,2"),
    )
    parser.add_argument("--residual-bounds", type=_float_grid, default=_float_grid("0.02,0.03,0.05"))
    parser.add_argument("--uncertainty-max", type=float, default=0.8)
    parser.add_argument("--none-max", type=float, default=0.8)
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--group-fold-seed", type=int, default=20260831)
    parser.add_argument("--max-b1-recall-regression", type=float, default=0.05)
    parser.add_argument("--max-saturation-rate", type=float, default=0.5)
    parser.add_argument("--minimum-hard-context-normal", type=int, default=5)
    parser.add_argument("--minimum-post-event-normal", type=int, default=2)
    parser.add_argument("--minimum-pure-normal", type=int, default=5)
    args = parser.parse_args()

    all_rows = list(iter_jsonl(args.records))
    rows = [
        row for row in all_rows
        if row.get("candidate_scoring_mode") == "event_state_v5"
        and bool(row.get("eligible_for_target_effect", True))
    ]
    if not rows:
        raise SystemExit("no target-exposed Event-State V5 calibration records")
    if any(not row.get("score_contract", {}).get("method_matched") for row in rows):
        raise SystemExit("calibration records contain unmatched score contracts")
    manifest_rows = list(iter_jsonl(args.manifest))
    manifest_by_key = {str(row.get("segment_key")): row for row in manifest_rows}
    missing = [str(row.get("segment_key")) for row in rows if str(row.get("segment_key")) not in manifest_by_key]
    if missing:
        raise SystemExit(f"records are not covered by calibration manifest; first={missing[0]}")
    source_groups = sorted({
        str(manifest_by_key[str(row.get("segment_key"))].get("source_group", ""))
        for row in rows if manifest_by_key[str(row.get("segment_key"))].get("source_group")
    })
    if len(source_groups) < max(2, args.group_folds):
        raise SystemExit("insufficient source groups for requested cross-validation folds")
    before = [_base_value(row) for row in rows]
    base_metrics = _metrics(before)
    ratios = (
        args.aftermath_ratio_grid
        if args.ablation in {"phase_aware", "signed_phase_aware"}
        else [0.0]
    )
    confound_ratios = (
        args.normal_confound_ratio_grid
        if args.ablation in {"signed_confound", "signed_phase_aware"}
        else [0.0]
    )
    grid_rows: list[dict[str, Any]] = []
    for beta_active in args.beta_active_grid:
        for ratio in ratios:
            beta_aftermath = beta_active * ratio
            if beta_aftermath > 0.5 * beta_active + 1e-12:
                continue
            for normal_confound_ratio in confound_ratios:
                beta_normal_confound = beta_active * normal_confound_ratio
                for bound in args.residual_bounds:
                    predicted = [
                        _candidate_value(
                            row, beta_active, beta_aftermath, beta_normal_confound, bound,
                            max(0.0, min(1.0, args.uncertainty_max)),
                            max(0.0, min(1.0, args.none_max)),
                        )
                        for row in rows
                    ]
                    after = [value for value, _ in predicted]
                    saturation = sum(ratio_value >= 0.95 for _, ratio_value in predicted) / len(predicted)
                    metrics = _metrics(after)
                    delta = _delta(metrics, base_metrics)
                    helps, hurts = _paired_effects(before, after)
                    hard_n, hard_before, hard_after = _fp(rows, after, manifest_by_key, _is_hard_context_normal)
                    post_n, post_before, post_after = _fp(rows, after, manifest_by_key, _is_post_event_normal)
                    pure_n, pure_before, pure_after = _fp(rows, after, manifest_by_key, _is_pure_normal)
                    fold_ap, fold_ba = [], []
                    for fold_index in range(args.group_folds):
                        indices = [
                            index for index, row in enumerate(rows)
                            if _fold(str(manifest_by_key[str(row.get("segment_key"))].get("source_group", "")), args.group_folds, args.group_fold_seed) == fold_index
                        ]
                        if not indices:
                            continue
                        fold_delta = _delta(_metrics([after[index] for index in indices]), _metrics([before[index] for index in indices]))
                        fold_ap.append(float(fold_delta.get("ap", 0.0)))
                        fold_ba.append(float(fold_delta.get("balanced_accuracy", 0.0)))
                    cv_ap = sum(fold_ap) / len(fold_ap) if fold_ap else -1.0
                    cv_ba = sum(fold_ba) / len(fold_ba) if fold_ba else -1.0
                    b1_delta = delta.get("recall_by_class", {}).get("B1")
                    feasible = (
                        pure_n >= args.minimum_pure_normal
                        and pure_after <= pure_before
                        and hard_n >= args.minimum_hard_context_normal
                        and post_n >= args.minimum_post_event_normal
                        and hard_after <= hard_before
                        and post_after <= post_before
                        and (b1_delta is None or float(b1_delta) >= -args.max_b1_recall_regression)
                        and saturation <= args.max_saturation_rate
                    )
                    grid_rows.append({
                        "beta_active": beta_active, "aftermath_ratio": ratio,
                        "beta_aftermath": beta_aftermath,
                        "normal_confound_ratio": normal_confound_ratio,
                        "beta_normal_confound": beta_normal_confound,
                        "residual_bound": bound,
                        "feasible": feasible, "cv_ap_delta_mean": cv_ap,
                        "cv_balanced_accuracy_delta_mean": cv_ba,
                        "ap_delta": delta.get("ap", 0.0),
                        "balanced_accuracy_delta": delta.get("balanced_accuracy", 0.0),
                        "b4_recall_delta": delta.get("recall_by_class", {}).get("B4"),
                        "b1_recall_delta": b1_delta,
                        "pure_normal_n": pure_n,
                        "pure_normal_fp_before": pure_before,
                        "pure_normal_fp_after": pure_after,
                        "pure_normal_fp_delta": pure_after - pure_before,
                        "hard_context_n": hard_n, "hard_context_fp_before": hard_before,
                        "hard_context_fp_after": hard_after, "post_event_normal_n": post_n,
                        "post_event_fp_before": post_before, "post_event_fp_after": post_after,
                        "residual_saturation_rate": saturation, "helps": helps, "hurts": hurts,
                    })
    selected = max(grid_rows, key=lambda row: (
        bool(row["feasible"]), float(row["cv_ap_delta_mean"]),
        float(row["cv_balanced_accuracy_delta_mean"]), float(row["ap_delta"]),
        float(row["balanced_accuracy_delta"]), int(row["helps"]) - int(row["hurts"]),
        -float(row["beta_aftermath"]), -float(row["beta_normal_confound"]),
        -float(row["beta_active"]), -float(row["residual_bound"]),
    ))
    positive_signal = bool(
        selected["feasible"] and float(selected["beta_active"]) > 0.0
        and float(selected["cv_ap_delta_mean"]) > 0.0
        and float(selected["cv_balanced_accuracy_delta_mean"]) >= 0.0
    )
    config = {
        "version": "crowd_event_state_v5_frozen_signed_confound_calibration_v2",
        "calibration_id": stable_sha1({
            "records": file_sha256(args.records), "manifest": file_sha256(args.manifest),
            "ablation": args.ablation, "selection": selected,
        }, size=20),
        "frozen": True, "deployable_signal_on_calibration": positive_signal,
        "refinement_mode": "phase_aware_occupancy", "ablation": args.ablation,
        "beta_active": float(selected["beta_active"]),
        "beta_aftermath": float(selected["beta_aftermath"]),
        "beta_normal_confound": float(selected["beta_normal_confound"]),
        "normal_confound_ratio": float(selected["normal_confound_ratio"]),
        "aftermath_ratio": float(selected["aftermath_ratio"]),
        "residual_bound": float(selected["residual_bound"]),
        "uncertainty_max": max(0.0, min(1.0, args.uncertainty_max)),
        "none_max": max(0.0, min(1.0, args.none_max)),
        "phase_specific_weights_frozen": True,
        "source_group_cross_validation": {
            "folds": args.group_folds, "seed": args.group_fold_seed,
            "selection_metric_order": ["cv_ap_delta_mean", "cv_balanced_accuracy_delta_mean"],
        },
        "source_groups": source_groups, "source_groups_sha256": stable_sha1(source_groups, size=40),
        "calibration_records": str(args.records), "calibration_records_sha256": file_sha256(args.records),
        "calibration_manifest": str(args.manifest), "calibration_manifest_sha256": file_sha256(args.manifest),
        "selection": selected,
        "selection_rule": "hard_constraints_then_group_cv_AP_then_group_cv_BA_then_global_metrics_then_simplest",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, config)
    grid_path = args.out.with_name("crowd_event_state_v5_calibration_grid.csv")
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader(); writer.writerows(grid_rows)
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
