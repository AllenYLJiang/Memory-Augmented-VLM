"""Registered families on shared cohorts, certified existing float64 optimizer."""
from __future__ import annotations

import math

import numpy as np

from ..contracts import iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..fitting import _inner_folds, _choose_threshold
from ..models import Standardizer, group_weights, weighted_bce
from ..numerics import CertifiedRidgeLogisticScorer
from .features import NAMES
from .protocol import advance, immutable, stable_hash, verify_manifest

SPECS = {
    "T0_M0": {"columns": [0], "lower": {0: 1e-6}, "cohort": "core"},
    "T1_DIRECT2": {"columns": [3, 4], "lower": {0: 0.0, 1: 0.0}, "cohort": "core"},
    "T2_EVENT6": {"columns": list(range(6)), "lower": {0: 0.0, 3: 0.0, 4: 0.0}, "cohort": "core"},
    "B0_C1_ONLY": {"columns": [3, 4, 6], "lower": {0: 0.0, 1: 0.0, 2: 0.0}, "cohort": "binding"},
    "B1_SAME_EVENT": {"columns": [3, 4, 6, 7], "lower": dict.fromkeys(range(4), 0.0), "cohort": "binding"},
    "B_BAG": {"columns": [3, 4, 6, 8], "lower": dict.fromkeys(range(4), 0.0), "cohort": "binding"},
    "T0_BASE_OPERATIONAL": {"columns": [0], "lower": {0: 1e-6}, "cohort": "base"},
    "B0_C1_OPERATIONAL": {"columns": [3, 4, 6], "lower": dict.fromkeys(range(3), 0.0), "cohort": "c1"},
}


def load_store(out, role):
    current = read_json(out / "feature_store" / role / "CURRENT.json")
    if not current:
        raise ValueError("Missing feature store " + role)
    root = out / "feature_store" / role / current["identity"]
    manifest = read_json(root / "manifest.json")
    for name, digest in manifest["files"].items():
        if stable_hash(root / name) != digest:
            raise ValueError("FEATURE_STORE_CHANGED")
    x, observed = np.load(root / "values.npy"), np.load(root / "observed.npy")
    if x.dtype != np.float64 or observed.dtype != bool or x.shape != observed.shape or x.shape[1] != 9:
        raise ValueError("Feature array contract mismatch")
    if not np.array_equal(observed, np.isfinite(x)):
        raise ValueError("Missingness mask mismatch")
    return x, observed, list(iter_jsonl(root / "rows.jsonl")), root


def score(model, X):
    if model.get("status") != "CERTIFIED":
        raise ValueError("Uncertified model")
    scaler = model["standardizer"]
    matrix = (X[:, model["columns"]] - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    logits = float(model["model"]["intercept"]) + matrix @ np.asarray(model["model"]["coef"])
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def fit_once(X, y, groups, ids, indices, spec, regularization, config):
    if len(indices) < 2 or len(set(y[indices])) != 2:
        raise ValueError("FIT_SINGLE_CLASS_OR_EMPTY")
    scaler = Standardizer.fit(X[np.ix_(indices, spec["columns"])])
    matrix = scaler.transform(X[np.ix_(indices, spec["columns"])])
    model = CertifiedRidgeLogisticScorer(regularization, config["max_iter"], config["tolerance"], spec["lower"], {})
    model.fit(matrix, y[indices], group_weights([groups[i] for i in indices]),
              fit_context={"uids": [ids[i] for i in indices], "feature_names": [NAMES[c] for c in spec["columns"]], "scaler": scaler.to_json()})
    payload = model.to_json()
    if payload["numeric_certificate"]["status"] != "CERTIFIED":
        raise ValueError("NUMERIC_FIT_NOT_CERTIFIED")
    return {"status": "CERTIFIED", "columns": spec["columns"], "standardizer": scaler.to_json(), "model": payload}


def fit_family(X, y, groups, ids, fit, threshold, spec, config):
    if len(set(y[fit])) != 2 or len(set(y[threshold])) != 2:
        return {"status": "UNESTIMABLE", "reason": "shared fit/threshold cohort lacks both classes"}
    folds = _inner_folds(fit, groups, config["inner_group_folds"], config["seed"])
    trials = []
    for reg in config["regularization_grid"]:
        losses, errors = [], []
        for validation in folds:
            train = [i for i in fit if i not in set(validation)]
            try:
                if len(set(y[validation])) != 2:
                    raise ValueError("INNER_VALIDATION_SINGLE_CLASS_OR_EMPTY")
                fitted = fit_once(X, y, groups, ids, train, spec, reg, config)
                losses.append(weighted_bce(y[validation], score(fitted, X[validation]), group_weights([groups[i] for i in validation])))
            except ValueError as exc:
                errors.append(str(exc))
        trials.append({"lambda": reg, "losses": losses, "errors": errors,
                       "mean": float(np.mean(losses)) if len(losses) == len(folds) and not errors else None,
                       "se": float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else None})
    valid = [r for r in trials if r["mean"] is not None]
    if not valid:
        return {"status": "UNESTIMABLE", "reason": "no fully certified three-fold lambda", "inner_cv": trials}
    best = min(valid, key=lambda r: (r["mean"], -r["lambda"]))
    chosen = max(r["lambda"] for r in valid if r["mean"] <= best["mean"] + best["se"])
    try:
        fitted = fit_once(X, y, groups, ids, fit, spec, chosen, config)
    except ValueError as exc:
        return {"status": "UNESTIMABLE", "reason": str(exc), "inner_cv": trials}
    cutoff, metric = _choose_threshold(y[threshold], score(fitted, X[threshold]))
    return {**fitted, "threshold": cutoff, "threshold_metrics_weak_only": metric,
            "selected_lambda": chosen, "inner_cv": trials,
            "fit_uids": [ids[i] for i in fit], "threshold_uids": [ids[i] for i in threshold],
            "fit_groups": sorted({groups[i] for i in fit}), "threshold_groups": sorted({groups[i] for i in threshold})}


def fit_models(out, config):
    X, observed, rows, root = load_store(out, "adaptation")
    labels = {r["window_uid"]: r for r in iter_jsonl(out / "enrollment/adaptation_labels.jsonl")}
    metadata = {r["window_uid"]: r for r in iter_jsonl(out / "enrollment/role_map.jsonl")}
    ids = [r["window_uid"] for r in rows]
    if any(metadata[uid]["role"] != "adaptation" or labels[uid].get("supervision") != "weak_only" for uid in ids):
        raise ValueError("TRAINING_SCOPE_VIOLATION")
    y = np.asarray([labels[u]["target"] if labels[u]["loss_mask"] else -1 for u in ids], dtype=int)
    groups = [metadata[u]["source_group"] for u in ids]
    unique = sorted(set(groups), key=lambda g: semantic_sha256([config["seed"], "threshold", g]))
    threshold_groups = set(unique[:max(1, round(len(unique) * config["threshold_fraction"]))])
    cohort_masks = {"core": observed[:, :6].all(axis=1), "base": observed[:, 0],
                    "binding": observed[:, [3, 4, 6, 7, 8]].all(axis=1), "c1": observed[:, [3, 4, 6]].all(axis=1)}
    cohorts = {}
    for name, mask in cohort_masks.items():
        eligible = [i for i in range(len(ids)) if mask[i] and y[i] in (0, 1)]
        cohorts[name] = {"fit": [i for i in eligible if groups[i] not in threshold_groups],
                         "threshold": [i for i in eligible if groups[i] in threshold_groups]}
    models = {}
    for name, spec in SPECS.items():
        c = cohorts[spec["cohort"]]
        models[name] = fit_family(X, y, groups, ids, c["fit"], c["threshold"], spec, config)
    result = {"version": "v912_certified_models1", "models": models,
              "cohorts": {k: {part: [ids[i] for i in indices] for part, indices in v.items()} for k, v in cohorts.items()},
              "threshold_source_groups": sorted(threshold_groups), "human_label_policy": "never_read",
              "locked_labels_read": False, "feature_manifest_sha256": stable_hash(root / "manifest.json"),
              "supervision_sha256": stable_hash(out / "enrollment/adaptation_labels.jsonl"), "deployment_authorized": False}
    immutable(out / "models/frozen_models.json", result)
    immutable(out / "models/identity.json", {"sha256": stable_hash(out / "models/frozen_models.json")})
    advance(out, "MODELS_FROZEN")
    return {name: m["status"] for name, m in models.items()}


def predict_and_commit(out):
    X, observed, rows, root = load_store(out, "locked_evaluation")
    model_path = out / "models/frozen_models.json"
    if stable_hash(model_path) != read_json(out / "models/identity.json")["sha256"]:
        raise ValueError("FROZEN_MODEL_CHANGED")
    models = read_json(model_path)["models"]
    results = []
    for i, row in enumerate(rows):
        direct = {}
        for name, model in models.items():
            if model.get("status") == "CERTIFIED" and observed[i, model["columns"]].all():
                probability = float(score(model, X[i:i+1])[0])
                direct[name] = {"score": probability, "prediction": int(probability >= model["threshold"]), "threshold": model["threshold"], "source_model": name}
        binding_complete = bool(observed[i, [3, 4, 6, 7, 8]].all())
        main = {}
        for name in list(SPECS)[:6]:
            if name.startswith("B"):
                chain = [name] if binding_complete else []
                chain += ["B0_C1_OPERATIONAL", "T1_DIRECT2", "T0_M0", "T0_BASE_OPERATIONAL"]
            else:
                chain = [name, "T0_M0", "T0_BASE_OPERATIONAL"]
            chosen = next((direct[k] for k in chain if k in direct), None)
            main[name] = {**chosen, "fallback": chosen["source_model"] != name} if chosen else {
                "score": None, "prediction": None, "source_model": None, "fallback": True, "reason": "technical_missing"}
        results.append({"window_uid": row["window_uid"], "direct": direct, "full_manifest": main,
                        "core_complete": bool(observed[i, :6].all()), "binding_complete": binding_complete})
    path = out / "predictions/predictions_committed.jsonl"
    if path.exists():
        if list(iter_jsonl(path)) != results:
            raise ValueError("Predictions changed after commitment")
    else:
        write_jsonl(path, results)
    files = [model_path, out / "protocol/frozen.json", out / "enrollment/role_map.jsonl", root / "manifest.json", path]
    commit = {"version": "v912_prediction_commit1", "files": [{"path": str(p.resolve()), "sha256": stable_hash(p)} for p in files],
              "evaluation_labels_loaded": False, "n": len(results)}
    immutable(out / "predictions/commit.json", commit)
    advance(out, "PREDICTIONS_COMMITTED")
    return {"committed_windows": len(results), "evaluation_labels_loaded": False}
