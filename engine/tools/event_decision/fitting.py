from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import FEATURE_NAMES, MissingLocalData, file_sha256, iter_jsonl, read_json, write_json, write_jsonl
from .evaluation import binary_metrics
from .models import RidgeLogisticScorer, Standardizer, group_weights, weighted_bce


FAMILIES = {
    "F0_m0_calibrated": {"features": (0,), "lower": {0: 1e-6}},
    "F1_graph_optional": {"features": (0, 1, 2), "lower": {0: 0.0}},
    "F2_event_active": {"features": (0, 1, 2, 3, 4, 5), "lower": {0: 0.0, 3: 0.0, 4: 0.0}},
    "F2_event_bound": {"features": tuple(range(8)), "lower": {0: 0.0, 3: 0.0, 4: 0.0}, "upper": {6: 0.0}},
}


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _split_model_threshold(indices: list[int], groups: Sequence[str], y: np.ndarray, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    unique = sorted({groups[i] for i in indices}, key=lambda g: _rank(seed, g))
    threshold_n = max(1, int(round(len(unique) * fraction)))
    threshold_groups = set(unique[:threshold_n])
    # Expand deterministically until the threshold pool represents both classes.
    for group in unique[threshold_n:]:
        current = [i for i in indices if groups[i] in threshold_groups]
        if len(set(y[current].tolist())) == 2:
            break
        threshold_groups.add(group)
    threshold = [i for i in indices if groups[i] in threshold_groups]
    model = [i for i in indices if groups[i] not in threshold_groups]
    return model, threshold


def _inner_folds(indices: list[int], groups: Sequence[str], folds: int, seed: int) -> list[list[int]]:
    unique = sorted({groups[i] for i in indices}, key=lambda g: _rank(seed, g))
    assignment = {group: n % folds for n, group in enumerate(unique)}
    return [[i for i in indices if assignment[groups[i]] == fold] for fold in range(folds)]


def _fit_model(X: np.ndarray, y: np.ndarray, groups: Sequence[str], indices: list[int], family: str, regularization: float, config: Mapping[str, Any]) -> tuple[Standardizer, RidgeLogisticScorer]:
    spec = FAMILIES[family]
    columns = list(spec["features"])
    scaler = Standardizer.fit(X[np.ix_(indices, columns)])
    matrix = scaler.transform(X[np.ix_(indices, columns)])
    model = RidgeLogisticScorer(regularization, int(config.get("max_iter", 1000)), float(config.get("tolerance", 1e-7)), spec.get("lower"), spec.get("upper"))
    model.fit(matrix, y[indices], group_weights([groups[i] for i in indices]))
    return scaler, model


def _select_lambda(X: np.ndarray, y: np.ndarray, groups: Sequence[str], indices: list[int], family: str, config: Mapping[str, Any], seed: int) -> tuple[float, list[dict[str, Any]]]:
    grid = [float(x) for x in config.get("regularization_grid", [.01, .1, 1.0])]
    folds = min(int(config.get("inner_group_folds", 3)), len(set(groups[i] for i in indices)))
    if folds < 2:
        return grid[-1], [{"regularization": grid[-1], "reason": "insufficient_inner_groups"}]
    heldouts = _inner_folds(indices, groups, folds, seed)
    rows = []
    for regularization in grid:
        losses = []
        for heldout in heldouts:
            train = [i for i in indices if i not in set(heldout)]
            if not heldout or len(set(y[train].tolist())) < 2:
                continue
            scaler, model = _fit_model(X, y, groups, train, family, regularization, config)
            cols = list(FAMILIES[family]["features"])
            probability = model.predict_proba(scaler.transform(X[np.ix_(heldout, cols)]))[:, 1]
            losses.append(weighted_bce(y[heldout], probability, group_weights([groups[i] for i in heldout])))
        rows.append({"regularization": regularization, "fold_losses": losses, "mean_bce": float(np.mean(losses)) if losses else math.inf, "se_bce": float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0})
    best = min(rows, key=lambda r: r["mean_bce"])
    cutoff = best["mean_bce"] + best["se_bce"]
    eligible = [r for r in rows if r["mean_bce"] <= cutoff]
    selected = max(eligible, key=lambda r: r["regularization"])["regularization"]
    return float(selected), rows


def _choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set([0.5, *probability.tolist()]))
    evaluated = []
    for threshold in candidates:
        metric = binary_metrics(y, probability, threshold)
        ba = metric["balanced_accuracy"] if metric["balanced_accuracy"] is not None else -1.0
        evaluated.append((ba, -metric["fp"], threshold, metric))
    best = max(evaluated, key=lambda item: (item[0], item[1], item[2]))
    return float(best[2]), best[3]


def _load_arrays(work_root: Path):
    current = read_json(Path(work_root) / "feature_store/CURRENT.json")
    if not current:
        raise MissingLocalData("feature_store/CURRENT.json is missing")
    root = Path(current["path"])
    X, observed = np.load(root / "values.npy"), np.load(root / "observed.npy")
    feature_rows = list(iter_jsonl(root / "rows.jsonl"))
    row_index = {row["window_uid"]: int(row["row_index"]) for row in feature_rows}
    plan = read_json(Path(work_root) / "splits/outer_inner_threshold_plan.json")
    records = [row for row in plan.get("rows", []) if row["window_uid"] in row_index]
    indices = np.asarray([row_index[row["window_uid"]] for row in records], dtype=int)
    y = np.asarray([int(row["y"]) for row in records], dtype=np.int8)
    groups = [str(row["source_group"]) for row in records]
    folds = np.asarray([int(row["outer_fold"]) for row in records], dtype=int)
    return current, root, X[indices], observed[indices], records, y, groups, folds


def fit_registered_models(work_root: Path, families: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    current, store_root, X, observed, rows, y, groups, outer = _load_arrays(work_root)
    if len(set(y.tolist())) < 2:
        raise MissingLocalData("scope-aligned labels contain fewer than two classes")
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    output_rows, summary = [], {}
    seed = int(config.get("seed", 20260907))
    for family in families:
        spec = FAMILIES[family]
        cols = list(spec["features"])
        available = observed[:, cols].all(axis=1) & np.isfinite(X[:, cols]).all(axis=1)
        if family == "F2_event_bound" and not available.any():
            out = Path(work_root) / "models" / family
            write_json(out / "skipped_reason.json", {"status": "BINDING_NOT_IDENTIFIABLE_FROM_CURRENT_CACHE", "reason": "both binding columns are structurally missing", "raw_global_normal_substitution_forbidden": True})
            summary[family] = {"status": "skipped", "available": 0}
            continue
        family_rows = []
        for fold in sorted(set(outer.tolist())):
            train_all = [i for i in range(len(rows)) if outer[i] != fold]
            test = [i for i in range(len(rows)) if outer[i] == fold]
            model_pool, threshold_pool = _split_model_threshold(train_all, groups, y, float(config.get("threshold_pool_fraction", .2)), seed + fold)
            candidate_pool = [i for i in model_pool if available[i]]
            candidate_threshold = [i for i in threshold_pool if available[i]]
            candidate_test = [i for i in test if available[i]]
            f0_pool = model_pool
            f0_threshold_pool = threshold_pool
            if len(set(y[candidate_pool].tolist())) < 2 or len(set(y[candidate_threshold].tolist())) < 2 or not candidate_test:
                summary.setdefault(family, {}).setdefault("fold_skips", []).append({"fold": fold, "reason": "insufficient_feature_complete_classes_or_test"})
                continue
            reg, cv = _select_lambda(X, y, groups, candidate_pool, family, config, seed + fold * 17)
            scaler, model = _fit_model(X, y, groups, candidate_pool, family, reg, config)
            candidate_threshold_probs = model.predict_proba(scaler.transform(X[np.ix_(candidate_threshold, cols)]))[:, 1]
            threshold, threshold_metric = _choose_threshold(y[candidate_threshold], candidate_threshold_probs)
            f0_reg, f0_cv = _select_lambda(X, y, groups, f0_pool, "F0_m0_calibrated", config, seed + fold * 17)
            f0_scaler, f0_model = _fit_model(X, y, groups, f0_pool, "F0_m0_calibrated", f0_reg, config)
            f0_threshold_probs = f0_model.predict_proba(f0_scaler.transform(X[np.ix_(f0_threshold_pool, [0])]))[:, 1]
            f0_threshold, _ = _choose_threshold(y[f0_threshold_pool], f0_threshold_probs)
            candidate_probability = model.predict_proba(scaler.transform(X[np.ix_(candidate_test, cols)]))[:, 1]
            f0_probability_all = f0_model.predict_proba(f0_scaler.transform(X[np.ix_(test, [0])]))[:, 1]
            f0_by_index = dict(zip(test, f0_probability_all))
            candidate_by_index = dict(zip(candidate_test, candidate_probability))
            fold_dir = Path(work_root) / "models" / family / f"outer_fold_{fold}"
            write_json(fold_dir / "model.json", {"family": family, "feature_indices": cols, "feature_names": [FEATURE_NAMES[i] for i in cols], "regularization": reg, "standardizer": scaler.to_json(), "model": model.to_json(), "threshold": threshold, "threshold_metric": threshold_metric, "inner_cv": cv, "model_pool_groups": sorted({groups[i] for i in model_pool}), "threshold_pool_groups": sorted({groups[i] for i in threshold_pool}), "test_groups": sorted({groups[i] for i in test})})
            for i in test:
                candidate_seen = i in candidate_by_index
                result = {"window_uid": rows[i]["window_uid"], "source_group": groups[i], "y": int(y[i]), "outer_fold": int(fold), "family": family, "candidate_available": candidate_seen, "candidate_probability": candidate_by_index.get(i), "candidate_threshold": threshold, "candidate_y_pred": int(candidate_by_index[i] >= threshold) if candidate_seen else None, "matched_f0_probability": float(f0_by_index[i]), "matched_f0_threshold": f0_threshold, "matched_f0_y_pred": int(f0_by_index[i] >= f0_threshold), "whole_packet_probability": float(candidate_by_index[i]) if candidate_seen else float(f0_by_index[i]), "whole_packet_y_pred": int((candidate_by_index[i] if candidate_seen else f0_by_index[i]) >= (threshold if candidate_seen else f0_threshold)), "fallback_used": not candidate_seen, "review_label_used": False}
                output_rows.append(result); family_rows.append(result)
        family_dir = Path(work_root) / "models" / family
        complete_indices = [i for i in range(len(rows)) if available[i]]
        frozen_status: dict[str, Any]
        if family_rows and len(set(y[complete_indices].tolist())) == 2:
            final_model_pool, final_threshold_pool = _split_model_threshold(complete_indices, groups, y, float(config.get("threshold_pool_fraction", .2)), seed + 9001)
            if len(set(y[final_model_pool].tolist())) == 2 and len(set(y[final_threshold_pool].tolist())) == 2:
                final_reg, final_cv = _select_lambda(X, y, groups, final_model_pool, family, config, seed + 9002)
                final_scaler, final_model = _fit_model(X, y, groups, final_model_pool, family, final_reg, config)
                final_probability = final_model.predict_proba(final_scaler.transform(X[np.ix_(final_threshold_pool, cols)]))[:, 1]
                final_threshold, final_threshold_metric = _choose_threshold(y[final_threshold_pool], final_probability)
                frozen = {
                    "version": "frozen_development_model_v1", "family": family, "feature_contract_id": current["contract_id"],
                    "feature_indices": cols, "feature_names": [FEATURE_NAMES[i] for i in cols],
                    "label_ledger_sha256": file_sha256(Path(work_root) / "labels/label_ledger.jsonl"),
                    "split_plan_sha256": file_sha256(Path(work_root) / "splits/outer_inner_threshold_plan.json"),
                    "regularization": final_reg, "standardizer": final_scaler.to_json(), "model": final_model.to_json(),
                    "threshold": final_threshold, "threshold_metric": final_threshold_metric, "inner_cv": final_cv,
                    "fit_rows": len(final_model_pool), "threshold_rows": len(final_threshold_pool),
                    "fit_source_groups": sorted({groups[i] for i in final_model_pool}), "threshold_source_groups": sorted({groups[i] for i in final_threshold_pool}),
                    "human_audit_labels_used": False, "development_only": True, "remote_execution_authorized": False,
                }
                write_json(family_dir / "frozen_development_model.json", frozen)
                frozen_status = {"status": "fit", "fit_rows": len(final_model_pool), "threshold_rows": len(final_threshold_pool), "converged": final_model.converged_}
            else:
                frozen_status = {"status": "insufficient_classes_in_final_model_or_threshold_pool"}
        else:
            frozen_status = {"status": "insufficient_feature_complete_classes"}
        write_json(family_dir / "model_manifest.json", {"family": family, "outer_fold_artifacts": len(list(family_dir.glob("outer_fold_*/model.json"))), "frozen_development_model": frozen_status, "feature_contract_id": current["contract_id"], "review_labels_used": False})
        summary[family] = {**summary.get(family, {}), "status": "fit" if family_rows else "insufficient", "oof_rows": len(family_rows), "candidate_available": sum(r["candidate_available"] for r in family_rows), "fallback_rows": sum(r["fallback_used"] for r in family_rows), "frozen_development_model": frozen_status}
    write_jsonl(Path(work_root) / "evaluation/oof_predictions.jsonl", output_rows)
    write_json(Path(work_root) / "models/fit_summary.json", {"families": summary, "labels": len(rows), "source_groups": len(set(groups)), "feature_contract_id": current["contract_id"], "human_audit_labels_used": False})
    return summary


def evaluate_fitted_models(work_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(iter_jsonl(Path(work_root) / "evaluation/oof_predictions.jsonl"))
    if not rows:
        raise MissingLocalData("no OOF predictions available")
    evaluation: dict[str, Any] = {"version": "scope_aligned_oof_evaluation_v1", "scope": "scope_valid_selected_training_windows", "evidence_origin": "development_nested_group_oof", "families": {}}
    incremental = []
    from .evaluation import grouped_bootstrap_delta, metrics_with_predictions
    repetitions = int(config.get("bootstrap_repetitions", 5000)); bootstrap_seed = int(config.get("bootstrap_seed", 20260907))
    bootstrap = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        matched = [r for r in subset if r["candidate_available"]]
        y = [r["y"] for r in matched]
        cand = [r["candidate_probability"] for r in matched]
        f0 = [r["matched_f0_probability"] for r in matched]
        candidate_metric = metrics_with_predictions(y, cand, [r["candidate_y_pred"] for r in matched]) if matched else {}
        f0_metric = metrics_with_predictions(y, f0, [r["matched_f0_y_pred"] for r in matched]) if matched else {}
        whole = metrics_with_predictions([r["y"] for r in subset], [r["whole_packet_probability"] for r in subset], [r["whole_packet_y_pred"] for r in subset])
        delta = {key: candidate_metric.get(key) - f0_metric.get(key) if candidate_metric.get(key) is not None and f0_metric.get(key) is not None else None for key in ("ap", "auc", "balanced_accuracy", "accuracy", "recall", "specificity")}
        evaluation["families"][family] = {"feature_matched": {"n": len(matched), "candidate": candidate_metric, "same_fold_F0": f0_metric, "delta": delta}, "whole_packet_with_F0_fallback": whole, "fallback_n": sum(r["fallback_used"] for r in subset)}
        incremental.append({"family": family, "n": len(matched), **{f"delta_{k}": v for k, v in delta.items()}})
        b_rows = [{"source_group": r["source_group"], "y": r["y"], "base": r["matched_f0_probability"], "candidate": r["candidate_probability"], "base_pred": r["matched_f0_y_pred"], "candidate_pred": r["candidate_y_pred"]} for r in matched]
        bootstrap[family] = grouped_bootstrap_delta(b_rows, "base", "candidate", repetitions, bootstrap_seed) if b_rows else {"status": "no_matched_rows"}
    out = Path(work_root) / "evaluation"
    write_json(out / "scope_aligned_metrics.json", evaluation)
    write_json(out / "whole_packet_fallback_metrics.json", {family: data["whole_packet_with_F0_fallback"] for family, data in evaluation["families"].items()})
    write_json(out / "source_group_bootstrap.json", bootstrap)
    write_json(out / "feature_incremental_value.json", incremental)
    write_json(out / "legacy_proxy_metrics.json", {"status": "diagnostic_only", "source": "audit/baseline_methods_tie_aware.csv"})
    write_json(out / "blind_challenge_metrics.json", {"status": "audit_only", "human_labels_used_for_model_selection": False})
    write_json(out / "calibration_reliability.json", {"status": "not_a_separate_probability_calibration_stage", "models": "ridge_logistic_outputs"})
    return evaluation
