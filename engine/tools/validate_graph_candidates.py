#!/usr/bin/env python3
"""Deterministic held-out validation and per-candidate removal ablation."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import iter_jsonl, write_json, write_jsonl
from competition import _aggregate
from event_constitution import load_constitution
from selection import is_pure_normal_video, label_codes, source_group_id


METHOD = "conditional_ot_full"
ANOMALY_CODES = ("B1", "B2", "B4", "B5", "B6", "G")


def _target_labels(row: Mapping[str, Any]) -> list[str]:
    key = str(row.get("graph", {}).get("key", ""))
    family = str(row.get("graph", {}).get("family", ""))
    if key == "rescue_or_assist_interaction" or family in {"assist_or_rescue", "human_interaction"}:
        return ["B1", "B5"]
    if family == "traffic" or "traffic" in key:
        return ["B6"]
    if family == "crowd" or "crowd" in key:
        return ["B4"]
    if family == "impulse_or_blast":
        return ["B2", "G"]
    return []


def _records(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "ot_window_results.jsonl"
    return {str(row.get("segment_key")): row for row in iter_jsonl(path)}


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    total = 0.0
    for index, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            total += hits / index
    return total / positives


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _metrics(rows: Iterable[tuple[int, int, float, str]]) -> dict:
    values = list(rows)
    resolved = [value for value in values if value[1] in {0, 1}]
    tp = sum(y == 1 and p == 1 for y, p, _, _ in resolved)
    tn = sum(y == 0 and p == 0 for y, p, _, _ in resolved)
    fp = sum(y == 0 and p == 1 for y, p, _, _ in resolved)
    fn = sum(y == 1 and p == 0 for y, p, _, _ in resolved)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * tpr / (precision + tpr) if precision + tpr else 0.0
    labels = [y for y, _, _, _ in values]
    scores = [score for _, _, score, _ in values]
    probabilities = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))) for score in scores]
    reliability = []
    for index in range(10):
        selected = [
            (label, probability) for label, probability in zip(labels, probabilities)
            if probability >= index / 10 and (probability < (index + 1) / 10 or index == 9)
        ]
        reliability.append({
            "bin": index, "n": len(selected),
            "mean_probability": sum(value[1] for value in selected) / len(selected) if selected else None,
            "positive_fraction": sum(value[0] for value in selected) / len(selected) if selected else None,
        })
    ece = sum(
        row["n"] / max(len(values), 1) * abs(float(row["mean_probability"]) - float(row["positive_fraction"]))
        for row in reliability if row["n"]
    )
    by_class = {}
    for code in ANOMALY_CODES:
        subset = [(y, p) for y, p, _, video_id in values if code in label_codes(video_id) and y == 1]
        by_class[code] = sum(p == 1 for _, p in subset) / len(subset) if subset else None
    return {
        "n": len(values), "resolved": len(resolved), "unresolved": len(values) - len(resolved),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(resolved) if resolved else 0.0,
        "balanced_accuracy": 0.5 * (tpr + tnr), "precision": precision,
        "recall": tpr, "specificity": tnr, "f1": f1,
        "ap": _average_precision(labels, scores), "auc": _auc(labels, scores),
        "brier_uncalibrated": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(values) if values else 0.0,
        "ece_uncalibrated": ece, "reliability_bins_uncalibrated": reliability,
        "pure_normal_n": sum(y == 0 and is_pure_normal_video(video_id) for y, _, _, video_id in values),
        "pure_normal_fp": sum(
            y == 0 and p == 1 and is_pure_normal_video(video_id) for y, p, _, video_id in values
        ),
        "pure_normal_specificity": (
            sum(y == 0 and p == 0 and is_pure_normal_video(video_id) for y, p, _, video_id in values)
            / sum(y == 0 and is_pure_normal_video(video_id) for y, _, _, video_id in values)
            if any(y == 0 and is_pure_normal_video(video_id) for y, _, _, video_id in values) else None
        ),
        "recall_by_class": by_class,
    }


def _competition_tuple(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    value = row.get("competitions", {}).get(METHOD, {})
    prediction = value.get("y_pred")
    return (
        int(row.get("y_true", 0)), int(prediction) if prediction in {0, 1} else -1,
        float(value.get("margin", 0.0) or 0.0), str(row.get("video_id", "")),
    )


def _without_graph(row: Mapping[str, Any], graph_key: str, polarity: Mapping[str, str]) -> tuple[int, int, float, str]:
    full = row.get("competitions", {}).get(METHOD, {})
    scores = row.get("graph_results", {}).get(METHOD, {})
    abnormal, normal = [], []
    for key, value in scores.items():
        if key == graph_key or not isinstance(value, Mapping):
            continue
        score = float(value.get("graph_score", 0.0) or 0.0)
        (abnormal if polarity.get(str(key)) == "abnormal" else normal).append(score)
    if not abnormal or not normal:
        return _competition_tuple(row)
    aggregation = str(full.get("aggregation", "logmeanexp"))
    temperature = float(full.get("temperature", 0.1) or 0.1)
    margin = _aggregate(abnormal, aggregation, temperature) - _aggregate(normal, aggregation, temperature)
    threshold = float(full.get("decision_margin_threshold", 0.03) or 0.03)
    pred = int(margin > threshold)
    return int(row.get("y_true", 0)), pred, float(margin), str(row.get("video_id", ""))


def _delta(full: Mapping[str, Any], reference: Mapping[str, Any]) -> dict:
    return {
        "ap": full["ap"] - reference["ap"],
        "balanced_accuracy": full["balanced_accuracy"] - reference["balanced_accuracy"],
        "accuracy": full["accuracy"] - reference["accuracy"],
        "auc": full["auc"] - reference["auc"],
        "precision": full["precision"] - reference["precision"],
        "recall": full["recall"] - reference["recall"],
        "specificity": full["specificity"] - reference["specificity"],
        "f1": full["f1"] - reference["f1"],
        "brier_uncalibrated": full["brier_uncalibrated"] - reference["brier_uncalibrated"],
        "ece_uncalibrated": full["ece_uncalibrated"] - reference["ece_uncalibrated"],
        "pure_normal_fp": full["pure_normal_fp"] - reference["pure_normal_fp"],
        "fp": full["fp"] - reference["fp"], "fn": full["fn"] - reference["fn"],
        "recall_by_class": {
            code: (None if full["recall_by_class"][code] is None or reference["recall_by_class"][code] is None
                   else full["recall_by_class"][code] - reference["recall_by_class"][code])
            for code in ANOMALY_CODES
        },
    }


def _manifest_run_path(value: Any, manifest_path: Path) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("run_dir") or value.get("path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def _paired_effects(baseline: Mapping[str, dict], variant: Mapping[str, dict], shared: list[str]) -> dict:
    helps = hurts = 0
    help_groups, hurt_groups, help_cases, hurt_cases = set(), set(), [], []
    for segment in shared:
        y, before, _, video_id = _competition_tuple(baseline[segment])
        _, after, _, _ = _competition_tuple(variant[segment])
        group = source_group_id(video_id)
        if after == y and before != y:
            helps += 1; help_groups.add(group); help_cases.append(segment)
        elif after != y and before == y:
            hurts += 1; hurt_groups.add(group); hurt_cases.append(segment)
    return {
        "helps": helps, "hurts": hurts,
        "help_source_groups": sorted(help_groups), "hurt_source_groups": sorted(hurt_groups),
        "help_case_ids": help_cases, "hurt_case_ids": hurt_cases,
    }


def _scope_counts(records: Mapping[str, dict], keys: Iterable[str]) -> dict:
    fields = (
        "candidate_directed_exact_window_eligible", "full_source_video_completion_eligible",
        "natural_distribution_eligible", "pure_normal_coverage", "dense_temporal_coverage",
    )
    selected = [records[key] for key in keys if key in records]
    return {
        field: sum(bool((row.get("evaluation_scope") or {}).get(field)) for row in selected)
        for field in fields
    }


def validate(registry_path: Path, baseline_run: Path, candidate_run: Path | None, candidate_catalog: Path,
             constitution_path: Path, out_dir: Path, candidate_runs_manifest: Path | None = None) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    constitution = load_constitution(constitution_path)
    policy = constitution["activation"]
    catalog = json.loads(candidate_catalog.read_text(encoding="utf-8"))
    polarity = {str(graph.get("key")): side for side in ("abnormal", "normal") for graph in catalog.get(side, [])}
    exact_manifest = {}
    exact_hypothesis_runs: dict[str, Path] = {}
    exact_bundle_runs: dict[str, Path] = {}
    if candidate_runs_manifest:
        exact_manifest = json.loads(candidate_runs_manifest.read_text(encoding="utf-8"))
        manifest_baseline = _manifest_run_path(exact_manifest.get("baseline"), candidate_runs_manifest)
        if manifest_baseline is not None:
            baseline_run = manifest_baseline
        candidate_run = _manifest_run_path(exact_manifest.get("combined"), candidate_runs_manifest) or candidate_run
        exact_hypothesis_runs = {
            str(key): path for key, value in exact_manifest.get("hypotheses", {}).items()
            if (path := _manifest_run_path(value, candidate_runs_manifest)) is not None
        }
        exact_bundle_runs = {
            str(key): path for key, value in exact_manifest.get("bundles", {}).items()
            if (path := _manifest_run_path(value, candidate_runs_manifest)) is not None
        }
    if candidate_run is None and exact_hypothesis_runs:
        candidate_run = next(iter(exact_hypothesis_runs.values()))
    if candidate_run is None:
        raise ValueError("candidate_run, a combined run, or at least one exact hypothesis run is required")
    baseline, candidate = _records(baseline_run), _records(candidate_run)
    shared = sorted(set(baseline) & set(candidate))
    baseline_metrics = _metrics(_competition_tuple(baseline[key]) for key in shared)
    candidate_metrics = _metrics(_competition_tuple(candidate[key]) for key in shared)
    global_delta = _delta(candidate_metrics, baseline_metrics)
    combined_global_gate = (
        global_delta["ap"] >= -float(policy["max_ap_regression"])
        and global_delta["balanced_accuracy"] >= -float(policy["max_balanced_accuracy_regression"])
        and global_delta["fp"] <= 0
    )

    validated_rows = []
    selected_ids = set(exact_hypothesis_runs) if candidate_runs_manifest else set()
    registry_rows = [
        row for row in registry.get("candidate_graphs", [])
        if not selected_ids or str(row.get("id", "")) in selected_ids
    ]
    for row in registry_rows:
        updated = dict(row)
        graph = row.get("graph", {})
        key = str(graph.get("key", ""))
        support = set(str(value) for value in row.get("support_source_groups", []))
        row_id = str(row.get("id", ""))
        target_labels = _target_labels(row)
        updated["target_label_codes"] = target_labels
        updated["target_family"] = str(graph.get("family", ""))
        updated["target_error_kind"] = "FN" if graph.get("polarity") == "abnormal" else "FP"
        updated["expected_metric"] = (
            "|".join(f"recall_by_class.{code}" for code in target_labels)
            if graph.get("polarity") == "abnormal" else "specificity"
        )
        updated["expected_direction"] = "increase"
        exact_path = exact_hypothesis_runs.get(row_id)
        exact_records = _records(exact_path) if exact_path and exact_path.is_dir() else {}
        exact_shared = sorted(set(baseline) & set(exact_records))
        same_manifest = bool(exact_records) and set(exact_records) == set(baseline)
        evaluation_records = exact_records if exact_records else candidate
        evaluation_shared = exact_shared if exact_records else shared
        exact_metrics = _metrics(_competition_tuple(evaluation_records[segment]) for segment in evaluation_shared)
        exact_baseline_metrics = _metrics(_competition_tuple(baseline[segment]) for segment in evaluation_shared)
        exact_contribution = _delta(exact_metrics, exact_baseline_metrics)
        exact_effects = _paired_effects(baseline, evaluation_records, evaluation_shared)
        exposed = [
            evaluation_records[segment] for segment in evaluation_shared
            if key in evaluation_records[segment].get("graph_results", {}).get(METHOD, {})
        ]
        exposure_groups = {source_group_id(str(value.get("video_id", ""))) for value in exposed}
        counterexample_groups = {
            source_group_id(str(value.get("video_id", ""))) for value in exposed
            if int(value.get("y_true", 0)) != int(graph.get("polarity") == "abnormal")
        }
        full_metrics = _metrics(_competition_tuple(candidate[segment]) for segment in shared)
        removed_metrics = _metrics(_without_graph(candidate[segment], key, polarity) for segment in shared)
        diagnostic_removal = _delta(full_metrics, removed_metrics)
        contribution = exact_contribution
        helps, hurts = exact_effects["helps"], exact_effects["hurts"]
        help_groups = set(exact_effects["help_source_groups"])
        hurt_groups = set(exact_effects["hurt_source_groups"])
        other_class_ok = all(
            delta is None or delta >= -float(policy["max_other_class_recall_regression"])
            for code, delta in contribution["recall_by_class"].items() if code not in target_labels
        )
        target_deltas = {
            code: contribution["recall_by_class"].get(code) for code in target_labels
        }
        any_class_gain = any(
            value is not None and value > 0.0 for value in contribution["recall_by_class"].values()
        )
        target_gain = (
            any(value is not None and value > 0.0 for value in target_deltas.values())
            if target_labels else any_class_gain
        )
        off_target_gain = any(
            value is not None and value > 0.0
            for code, value in contribution["recall_by_class"].items() if code not in target_labels
        ) if target_labels else False
        recurrence_ok = len(support) >= int(policy["min_support_groups"])
        exposure_ok = len(exposure_groups) >= int(policy["min_validation_exposure_groups"])
        counterexample_ok = len(counterexample_groups) >= int(policy["min_counterexample_groups"])
        contribution_ok = (
            contribution["ap"] >= -float(policy["max_ap_regression"])
            and contribution["balanced_accuracy"] >= -float(policy["max_balanced_accuracy_regression"])
            and contribution["fp"] <= 0 and other_class_ok and helps > hurts
        )
        exact_global_gate = (
            contribution["ap"] >= -float(policy["max_ap_regression"])
            and contribution["balanced_accuracy"] >= -float(policy["max_balanced_accuracy_regression"])
            and contribution["fp"] <= 0
        )
        gates = {
            "exact_candidate_run_available": bool(exact_records),
            "identical_window_manifest": same_manifest,
            "global_library_non_inferior": exact_global_gate, "recurrence": recurrence_ok,
            "held_out_exposure": exposure_ok, "counterexamples": counterexample_ok,
            "candidate_non_inferior_and_useful": contribution_ok,
            "source_disjoint": not bool(support & exposure_groups),
        }
        passed = all(gates.values())
        if not exact_records or not same_manifest or not exposure_ok:
            status = "insufficient_validation_exposure"
        elif target_gain and not other_class_ok:
            status = "promising_but_cross_class_harmful"
        elif target_gain and not exact_global_gate:
            status = "target_family_useful_global_harmful"
        elif off_target_gain and not target_gain:
            status = "off_target_class_shift"
        elif contribution["ap"] > 0.0 and not passed:
            status = "promising_but_uncalibrated"
        else:
            status = "validated" if passed else "rejected_empirical"
        bundle_id = str(row.get("bundle_id") or row.get("revision", {}).get("bundle_id") or "")
        bundle_path = exact_bundle_runs.get(bundle_id)
        bundle_records = _records(bundle_path) if bundle_path and bundle_path.is_dir() else {}
        bundle_shared = sorted(set(baseline) & set(bundle_records))
        bundle_delta = (
            _delta(
                _metrics(_competition_tuple(bundle_records[segment]) for segment in bundle_shared),
                _metrics(_competition_tuple(baseline[segment]) for segment in bundle_shared),
            ) if bundle_records else None
        )
        updated["validation"] = {
            "status": status, "gates": gates, "exposed_windows": len(exposed),
            "exposure_source_groups": sorted(exposure_groups),
            "counterexample_source_groups": sorted(counterexample_groups),
            "activation_evidence": "exact_candidate_run" if exact_records else "missing_exact_candidate_run",
            "exact_candidate_run": str(exact_path or ""),
            "exact_shared_windows": len(exact_shared),
            "helps": helps, "hurts": hurts, "help_source_groups": sorted(help_groups),
            "hurt_source_groups": sorted(hurt_groups),
            "help_case_ids": exact_effects["help_case_ids"], "hurt_case_ids": exact_effects["hurt_case_ids"],
            "contribution_vs_baseline_exact": contribution,
            "contribution_vs_removal": diagnostic_removal,
            "diagnostic_posthoc_removal_only": True,
            "bundle_id": bundle_id, "bundle_run": str(bundle_path or ""),
            "bundle_delta_vs_baseline_exact": bundle_delta,
            "global_candidate_library_delta_vs_baseline": global_delta,
            "combined_library_non_inferior": combined_global_gate,
            "target_label_codes": target_labels,
            "target_family": str(graph.get("family", "")),
            "expected_error_kind": "false_negative" if graph.get("polarity") == "abnormal" else "false_positive",
            "target_metric": "recall" if graph.get("polarity") == "abnormal" else "specificity",
            "target_direction": "increase",
            "target_recall_delta": target_deltas,
            "off_target_recall_gain": off_target_gain,
        }
        updated["status"] = status
        updated["active"] = False
        validated_rows.append(updated)
        print(f"[validate] graph={key} status={status} helps={helps} hurts={hurts} exposure_groups={len(exposure_groups)}", flush=True)

    output_registry = dict(registry)
    output_registry["candidate_graphs"] = validated_rows
    output_registry["validation_run"] = {
        "baseline": str(baseline_run), "candidate": str(candidate_run), "shared_windows": len(shared),
        "candidate_runs_manifest": str(candidate_runs_manifest or ""),
        "baseline_metrics": baseline_metrics, "candidate_metrics": candidate_metrics,
        "global_delta": global_delta, "global_gate": combined_global_gate,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "validated_graph_memory.json", output_registry)
    write_jsonl(out_dir / "candidate_validation.jsonl", validated_rows)
    summary = {
        "version": "held_out_graph_validation_v2_exact_actions", "shared_windows": len(shared),
        "validated": sum(row["validation"]["status"] == "validated" for row in validated_rows),
        "rejected_empirical": sum(row["validation"]["status"] == "rejected_empirical" for row in validated_rows),
        "promising_but_uncalibrated": sum(row["validation"]["status"] == "promising_but_uncalibrated" for row in validated_rows),
        "promising_but_cross_class_harmful": sum(row["validation"]["status"] == "promising_but_cross_class_harmful" for row in validated_rows),
        "target_family_useful_global_harmful": sum(row["validation"]["status"] == "target_family_useful_global_harmful" for row in validated_rows),
        "off_target_class_shift": sum(row["validation"]["status"] == "off_target_class_shift" for row in validated_rows),
        "insufficient_exposure": sum(row["validation"]["status"] == "insufficient_validation_exposure" for row in validated_rows),
        "baseline_metrics": baseline_metrics, "candidate_metrics": candidate_metrics, "global_delta": global_delta,
        "evaluation_scope": {
            "baseline": _scope_counts(baseline, shared),
            "candidate": _scope_counts(candidate, shared),
        },
    }
    write_json(out_dir / "validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--candidate-runs-manifest", type=Path)
    parser.add_argument("--candidate-catalog", required=True, type=Path)
    parser.add_argument("--constitution", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.registry, args.baseline_run, args.candidate_run, args.candidate_catalog,
        args.constitution, args.out_dir, args.candidate_runs_manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
