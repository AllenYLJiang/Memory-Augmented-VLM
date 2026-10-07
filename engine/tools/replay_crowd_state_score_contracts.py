#!/usr/bin/env python3
"""Exploratory, zero-API replay of bounded crowd state score contracts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from common import iter_jsonl, read_json, stable_sha1, write_json
from run_frozen_evidence_graph_pair import _records, _replace_competition
from selection import source_group_id
from validate_graph_candidates import _delta, _metrics


METHOD = "conditional_ot_full"
EPS = 1e-6


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _features(row: Mapping[str, Any]) -> dict:
    event = row.get("candidate_event_state", {})
    states = event.get("state_probabilities", {}) if isinstance(event.get("state_probabilities"), Mapping) else {}
    score = event.get("score", {}) if isinstance(event.get("score"), Mapping) else {}
    p_active = float(states.get("active_escalation", 0.0))
    p_aftermath = float(states.get("causally_linked_aftermath", 0.0))
    p_benign = float(states.get("benign_or_pre_event_context", 0.0))
    p_none = float(states.get("none_or_unobservable", 0.0))
    q = _clip(p_active + p_aftermath)
    return {
        "p_active": p_active,
        "p_aftermath": p_aftermath,
        "p_benign": p_benign,
        "p_none": p_none,
        "q": q,
        "state_log_odds": math.log((q + EPS) / (p_benign + p_none + EPS)),
        "active_unary": float(score.get("active_unary_ot_score", 0.0)),
        "aftermath_unary": float(score.get("aftermath_unary_ot_score", 0.0)),
        "transition": float(event.get("transition_observed_probability", 0.0)),
        "causal_link": float(event.get("aftermath_causal_link_probability", 0.0)),
        "same_episode": float(event.get("same_episode_probability", 0.0)),
    }


def _state_unary(features: Mapping[str, float]) -> float:
    return _clip(
        features["p_active"] * features["active_unary"]
        + features["p_aftermath"] * features["aftermath_unary"]
    )


def _metric_rows(rows: list[dict], predictor: Callable[[dict], tuple[int, float]]) -> tuple[dict, list[dict]]:
    values, records = [], []
    for row in rows:
        pred, score = predictor(row)
        truth, video = int(row["y_true"]), str(row["video_id"])
        values.append((truth, int(pred), float(score), video))
        records.append({
            "segment_key": row["segment_key"], "video_id": video,
            "source_group": source_group_id(video), "y_true": truth,
            "y_pred": int(pred), "score": float(score),
        })
    return _metrics(values), records


def _target_competition(
    row: Mapping[str, Any], baseline: Mapping[str, Any], target_score: float, target_key: str,
) -> dict:
    base_result = dict(row["base_target"]["results"][METHOD])
    base_result["graph_score"] = _clip(target_score)
    return _replace_competition(baseline, target_key, base_result, METHOD, "abnormal")


def _evaluate_grid(rows: list[dict], baseline: Mapping[str, dict], target_key: str) -> list[dict]:
    grid: list[dict] = []

    def target_formula(name: str, alpha: float) -> None:
        def predict(row: dict) -> tuple[int, float]:
            features = row["_features"]
            state = _state_unary(features)
            base_target = float(row["base_target"]["results"][METHOD]["graph_score"])
            target = state if name == "nonduplicative_unary" else (1.0 - alpha) * base_target + alpha * state
            comp = _target_competition(row, baseline[row["segment_key"]], target, target_key)
            return int(comp["y_pred"]), float(comp["margin"])

        metrics, _ = _metric_rows(rows, predict)
        grid.append({"family": name, "alpha": alpha, "beta": "", "bound": "", **_compact(metrics)})

    target_formula("nonduplicative_unary", 1.0)
    for alpha in [index / 10 for index in range(0, 11)]:
        target_formula("bounded_target_blend", alpha)

    for beta in (0.005, 0.01, 0.02, 0.04, 0.08):
        for bound in (0.01, 0.02, 0.04, 0.08):
            def predict(row: dict, beta=beta, bound=bound) -> tuple[int, float]:
                base = row["base_competitions"][METHOD]
                residual = _clip(beta * row["_features"]["state_log_odds"], -bound, bound)
                margin = float(base["margin"]) + residual
                threshold = float(base.get("decision_margin_threshold", 0.03) or 0.03)
                return int(margin > threshold), margin

            metrics, _ = _metric_rows(rows, predict)
            grid.append({"family": "bounded_margin_residual", "alpha": "", "beta": beta, "bound": bound, **_compact(metrics)})
    return grid


def _compact(metrics: Mapping[str, Any]) -> dict:
    return {
        key: metrics.get(key) for key in (
            "accuracy", "balanced_accuracy", "ap", "auc", "precision", "recall",
            "specificity", "f1", "tp", "tn", "fp", "fn",
        )
    }


def _params(item: Mapping[str, Any]) -> dict:
    return {key: item[key] for key in ("family", "alpha", "beta", "bound")}


def _predictor(params: Mapping[str, Any], baseline: Mapping[str, dict], target_key: str):
    family = str(params["family"])

    def predict(row: dict) -> tuple[int, float]:
        if family in {"nonduplicative_unary", "bounded_target_blend"}:
            state = _state_unary(row["_features"])
            alpha = 1.0 if family == "nonduplicative_unary" else float(params["alpha"])
            base_target = float(row["base_target"]["results"][METHOD]["graph_score"])
            target = (1.0 - alpha) * base_target + alpha * state
            comp = _target_competition(row, baseline[row["segment_key"]], target, target_key)
            return int(comp["y_pred"]), float(comp["margin"])
        beta, bound = float(params["beta"]), float(params["bound"])
        base = row["base_competitions"][METHOD]
        margin = float(base["margin"]) + _clip(beta * row["_features"]["state_log_odds"], -bound, bound)
        threshold = float(base.get("decision_margin_threshold", 0.03) or 0.03)
        return int(margin > threshold), margin

    return predict


def _logo_cv(rows: list[dict], baseline: Mapping[str, dict], target_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[source_group_id(str(row["video_id"]))].append(row)
    predictions = []
    selections = []
    for held_out in sorted(groups):
        train = [row for group, values in groups.items() if group != held_out for row in values]
        valid = groups[held_out]
        if not train or len({int(row["y_true"]) for row in train}) < 2:
            continue
        grid = _evaluate_grid(train, baseline, target_key)
        # Exploratory selection only. Prefer BA, then AP, then the smaller perturbation.
        selected = max(
            grid,
            key=lambda item: (
                float(item["balanced_accuracy"]), float(item["ap"]),
                -float(item["alpha"] or 0.0), -float(item["bound"] or 0.0),
            ),
        )
        predictor = _predictor(selected, baseline, target_key)
        _, held_rows = _metric_rows(valid, predictor)
        predictions.extend(held_rows)
        selections.append({"held_out_source_group": held_out, "selected": _params(selected), "n_train": len(train), "n_valid": len(valid)})
    metrics = _metrics([
        (int(row["y_true"]), int(row["y_pred"]), float(row["score"]), str(row["video_id"]))
        for row in predictions
    ]) if predictions else {}
    return {
        "policy": "leave_one_source_group_out_exploratory",
        "n_source_groups": len(groups),
        "metrics": metrics,
        "fold_selections": selections,
        "predictions": predictions,
    }


def replay(results_path: Path, baseline_path: Path, out_dir: Path, target_key: str) -> dict:
    rows = list(iter_jsonl(results_path))
    baseline = _records(baseline_path)
    missing = [str(row.get("segment_key", "")) for row in rows if str(row.get("segment_key", "")) not in baseline]
    if missing:
        raise ValueError(f"baseline records missing {len(missing)} frozen segments; first={missing[0]}")
    for row in rows:
        row["_features"] = _features(row)

    base_metrics, _ = _metric_rows(rows, lambda row: (
        int(row["base_competitions"][METHOD]["y_pred"]),
        float(row["base_competitions"][METHOD]["margin"]),
    ))
    posterior_metrics, _ = _metric_rows(rows, lambda row: (
        int(row["_features"]["q"] >= 0.5), float(row["_features"]["q"]),
    ))
    grid = _evaluate_grid(rows, baseline, target_key)
    logo = _logo_cv(rows, baseline, target_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(grid[0]) if grid else []
    with (out_dir / "formula_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(grid)
    write_json(out_dir / "logo_cv.json", logo)
    summary = {
        "version": "crowd_state_score_contract_replay_v1",
        "analysis_scope": "development_only_zero_api_not_for_deployment",
        "results": str(results_path),
        "baseline_records": str(baseline_path),
        "source_groups_sha256": stable_sha1(sorted({source_group_id(str(row["video_id"])) for row in rows}), size=40),
        "baseline_metrics": base_metrics,
        "posterior_only_diagnostic": posterior_metrics,
        "formula_grid_rows": len(grid),
        "logo_cv_metrics": logo.get("metrics", {}),
        "interpretation": (
            "This replay may select a formula family for a separate calibration split. "
            "It must not freeze deployment coefficients or claim held-out improvement."
        ),
    }
    write_json(out_dir / "score_contract_replay_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-graph-key", default="crowd_escalation_chain")
    args = parser.parse_args()
    print(json.dumps(replay(args.results, args.baseline_records, args.out_dir, args.target_graph_key), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
