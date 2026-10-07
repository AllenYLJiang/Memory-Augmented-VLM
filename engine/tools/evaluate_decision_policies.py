#!/usr/bin/env python3
"""Fit decision policies on source-disjoint calibration groups and evaluate them."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from decision_policy import (
    FEATURE_SCHEMA_VERSION,
    apply_decision_policy,
    extract_decision_features,
    fit_platt,
    policy_score,
)
from selection import CLASS_CODES, label_codes, source_group_id


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(int(value) for value in labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda value: value[0], reverse=True)
    hits = 0
    total = 0.0
    for index, (_, label) in enumerate(ranked, 1):
        if int(label):
            hits += 1
            total += hits / index
    return total / positives


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return 0.0
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _reliability(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> list[dict]:
    rows = []
    for index in range(max(1, int(bins))):
        low, high = index / bins, (index + 1) / bins
        selected = [
            (int(label), float(probability)) for label, probability in zip(labels, probabilities)
            if probability >= low and (probability < high or index == bins - 1)
        ]
        rows.append({
            "bin": index,
            "lower": low,
            "upper": high,
            "n": len(selected),
            "mean_probability": sum(value[1] for value in selected) / len(selected) if selected else None,
            "positive_fraction": sum(value[0] for value in selected) / len(selected) if selected else None,
        })
    return rows


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    labels = [int(row["y_true"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predictions = [row.get("y_pred") for row in rows]
    resolved = [index for index, value in enumerate(predictions) if value in {0, 1}]
    tp = sum(labels[i] == 1 and predictions[i] == 1 for i in resolved)
    tn = sum(labels[i] == 0 and predictions[i] == 0 for i in resolved)
    fp = sum(labels[i] == 0 and predictions[i] == 1 for i in resolved)
    fn = sum(labels[i] == 1 and predictions[i] == 0 for i in resolved)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    reliability = _reliability(labels, probabilities)
    ece = sum(
        row["n"] / max(len(rows), 1) * abs(float(row["mean_probability"]) - float(row["positive_fraction"]))
        for row in reliability if row["n"]
    )
    pure_normal = [i for i, row in enumerate(rows) if not label_codes(str(row.get("video_id", "")))]
    per_class = {}
    for code in CLASS_CODES:
        subset = [i for i, row in enumerate(rows) if code in label_codes(str(row.get("video_id", ""))) and labels[i] == 1]
        per_class[code] = (
            sum(predictions[i] == 1 for i in subset if predictions[i] in {0, 1})
            / sum(predictions[i] in {0, 1} for i in subset)
            if any(predictions[i] in {0, 1} for i in subset) else None
        )
    return {
        "n": len(rows),
        "resolved": len(resolved),
        "unresolved": len(rows) - len(resolved),
        "coverage": len(resolved) / len(rows) if rows else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy_resolved": (tp + tn) / len(resolved) if resolved else 0.0,
        "balanced_accuracy_resolved": 0.5 * (recall + specificity),
        "precision_resolved": precision,
        "recall_resolved": recall,
        "specificity_resolved": specificity,
        "f1_resolved": f1,
        "ap": _average_precision(labels, scores),
        "auc": _auc(labels, scores),
        "brier": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(rows) if rows else 0.0,
        "ece": ece,
        "reliability_bins": reliability,
        "pure_normal_n": len(pure_normal),
        "pure_normal_fp": sum(predictions[i] == 1 for i in pure_normal),
        "recall_by_class": per_class,
        "margin_mean": float(np.mean(scores)) if scores else 0.0,
        "margin_std": float(np.std(scores)) if scores else 0.0,
        "margin_quantiles": {
            str(q): float(np.quantile(scores, q)) if scores else 0.0 for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }


def _rank(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()


def source_group_split(records: Sequence[Mapping[str, Any]], calibration_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    groups = sorted({source_group_id(str(row.get("video_id", ""))) for row in records}, key=lambda value: (_rank(seed, value), value))
    if len(groups) < 2:
        raise ValueError("at least two source groups are required for calibration/validation separation")
    count = min(len(groups) - 1, max(1, int(round(len(groups) * float(calibration_fraction)))))
    return set(groups[:count]), set(groups[count:])


def _candidate_configs() -> list[dict]:
    values = [{"policy": "legacy_graph"}]
    values.extend({"policy": "weighted_fusion", "alpha": round(index / 10.0, 1)} for index in range(11))
    values.extend({"policy": "gated_normal_veto", "beta": beta} for beta in (0.25, 0.5, 0.75, 1.0))
    values.extend(
        {"policy": "asymmetric", "beta": beta, "abnormal_weight": weight}
        for beta in (0.25, 0.5, 0.75) for weight in (0.25, 0.5, 0.75)
    )
    return values


def _threshold(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    best_rank = (float("-inf"), float("-inf"), float("-inf"))
    selected_threshold = 0.5
    for threshold in np.linspace(0.1, 0.9, 161):
        rows = [
            {"y_true": int(label), "score": float(probability), "probability": float(probability),
             "y_pred": int(probability >= threshold), "video_id": ""}
            for probability, label in zip(probabilities, labels)
        ]
        value = metrics(rows)
        rank = (value["balanced_accuracy_resolved"], value["f1_resolved"], -abs(float(threshold) - 0.5))
        if rank > best_rank:
            best_rank = rank
            selected_threshold = float(threshold)
    return selected_threshold


def _evaluate_config(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict], dict]:
    rows = []
    for record in records:
        features = extract_decision_features(record)
        decision = apply_decision_policy(features, config)
        rows.append({
            "segment_key": record.get("segment_key"),
            "video_id": record.get("video_id"),
            "source_group": source_group_id(str(record.get("video_id", ""))),
            "y_true": int(record.get("y_true", 0)),
            "score": decision["score"],
            "probability": decision["calibrated_probability"],
            "y_pred": decision["y_pred"],
            "decision": decision["decision"],
        })
    return rows, metrics(rows)


def analyze_dataset(records_path: Path, out_dir: Path, calibration_fraction: float, seed: int) -> dict:
    records = [row for row in iter_jsonl(records_path) if bool(row.get("metric_eligible", True))]
    calibration_groups, validation_groups = source_group_split(records, calibration_fraction, seed)
    calibration = [row for row in records if source_group_id(str(row.get("video_id", ""))) in calibration_groups]
    validation = [row for row in records if source_group_id(str(row.get("video_id", ""))) in validation_groups]
    evaluated = []
    for base in _candidate_configs():
        raw_scores = [policy_score(extract_decision_features(row), base)[0] for row in calibration]
        labels = [int(row.get("y_true", 0)) for row in calibration]
        a, b = fit_platt(raw_scores, labels)
        probabilities = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, a * score + b)))) for score in raw_scores]
        fitted = dict(
            base,
            platt_a=a,
            platt_b=b,
            threshold=(0.03 if base["policy"] == "legacy_graph" else _threshold(probabilities, labels)),
        )
        _, calibration_metrics = _evaluate_config(calibration, fitted)
        evaluated.append({"config": fitted, "calibration_metrics": calibration_metrics})
    selected = max(
        evaluated,
        key=lambda row: (
            row["calibration_metrics"]["balanced_accuracy_resolved"],
            row["calibration_metrics"]["ap"],
            -row["calibration_metrics"]["pure_normal_fp"],
        ),
    )
    selected_config = dict(selected["config"])
    selected_config.update({
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fit_source_groups_sha256": stable_sha1(*sorted(calibration_groups), size=40),
        "fit_source_group_count": len(calibration_groups),
        "selection_uses_validation_labels": False,
    })
    calibration_rows, calibration_metrics = _evaluate_config(calibration, selected_config)
    validation_rows, validation_metrics = _evaluate_config(validation, selected_config)
    ternary = dict(selected_config, policy="ternary", unresolved_margin=0.05)
    _, ternary_metrics = _evaluate_config(validation, ternary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "decision_policy.json", selected_config)
    write_jsonl(out_dir / "calibration_predictions.jsonl", calibration_rows)
    write_jsonl(out_dir / "validation_predictions.jsonl", validation_rows)
    write_json(out_dir / "policy_sweep.json", evaluated)
    summary = {
        "version": "decision_policy_evaluation_v1",
        "records": str(records_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "source_groups": {
            "calibration": sorted(calibration_groups),
            "validation": sorted(validation_groups),
            "overlap": sorted(calibration_groups & validation_groups),
        },
        "selected_policy": selected_config,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "validation_ternary_metrics": ternary_metrics,
    }
    write_json(out_dir / "evaluation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--candidate-records", type=Path)
    parser.add_argument("--group-by", choices=("source_group",), default="source_group")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be between 0 and 1")
    baseline = analyze_dataset(args.records, args.out_dir / "baseline", args.calibration_fraction, args.seed)
    output: dict[str, Any] = {"baseline": baseline}
    if args.candidate_records:
        candidate = analyze_dataset(args.candidate_records, args.out_dir / "candidate", args.calibration_fraction, args.seed)
        output["candidate"] = candidate
        output["validation_delta_candidate_minus_baseline"] = {
            key: candidate["validation_metrics"][key] - baseline["validation_metrics"][key]
            for key in ("ap", "auc", "balanced_accuracy_resolved", "recall_resolved", "specificity_resolved", "f1_resolved")
        }
    write_json(args.out_dir / "comparison_summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
