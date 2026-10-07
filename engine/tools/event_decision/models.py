from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Standardizer.fit requires a finite 2-D matrix")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.scale

    def to_json(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit_on_training_only": True}


class RidgeLogisticScorer:
    def __init__(
        self,
        regularization: float = 0.1,
        max_iter: int = 1000,
        tolerance: float = 1e-7,
        lower_bounds: Mapping[int, float] | None = None,
        upper_bounds: Mapping[int, float] | None = None,
    ) -> None:
        self.regularization = float(regularization)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.lower_bounds = dict(lower_bounds or {})
        self.upper_bounds = dict(upper_bounds or {})
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.converged_: bool = False
        self.iterations_: int = 0
        self.objective_: float | None = None

    def _project(self, coef: np.ndarray) -> np.ndarray:
        result = coef.copy()
        for index, bound in self.lower_bounds.items():
            result[index] = max(result[index], bound)
        for index, bound in self.upper_bounds.items():
            result[index] = min(result[index], bound)
        return result

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RidgeLogisticScorer":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("invalid or non-finite fit arrays")
        if len(set(y.tolist())) < 2:
            raise ValueError("logistic fit requires both classes")
        weights = np.ones(len(X), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = weights / weights.sum()
        intercept = math.log(np.clip(np.average(y, weights=weights), 1e-5, 1 - 1e-5) / np.clip(1 - np.average(y, weights=weights), 1e-5, 1))
        coef = self._project(np.zeros(X.shape[1], dtype=float))

        def objective(b: float, w: np.ndarray) -> float:
            z = np.clip(b + X @ w, -50, 50)
            loss = np.sum(weights * (np.logaddexp(0.0, z) - y * z))
            return float(loss + .5 * self.regularization * np.dot(w, w))

        old = objective(intercept, coef)
        for iteration in range(1, self.max_iter + 1):
            probability = sigmoid(intercept + X @ coef)
            residual = weights * (probability - y)
            grad_b = float(residual.sum())
            grad_w = X.T @ residual + self.regularization * coef
            grad_norm = float(np.sqrt(grad_b * grad_b + np.dot(grad_w, grad_w)))
            if grad_norm < self.tolerance:
                self.converged_, self.iterations_ = True, iteration
                break
            step = 1.0
            accepted = False
            for _ in range(40):
                trial_b = intercept - step * grad_b
                trial_w = self._project(coef - step * grad_w)
                trial = objective(trial_b, trial_w)
                if trial <= old - 1e-4 * step * grad_norm * grad_norm or trial < old:
                    intercept, coef, old, accepted = trial_b, trial_w, trial, True
                    break
                step *= .5
            if not accepted:
                self.iterations_ = iteration
                break
            if step * grad_norm < self.tolerance:
                self.converged_, self.iterations_ = True, iteration
                break
        else:
            self.iterations_ = self.max_iter
        self.intercept_, self.coef_, self.objective_ = float(intercept), coef, float(old)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("model is not fit")
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - probability, probability])

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "ridge_logistic_scorer_v1", "intercept": self.intercept_,
            "coef": None if self.coef_ is None else self.coef_.tolist(), "regularization": self.regularization,
            "lower_bounds": self.lower_bounds, "upper_bounds": self.upper_bounds,
            "converged": self.converged_, "iterations": self.iterations_, "objective": self.objective_,
        }


def weighted_bce(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weight, dtype=float)
    return float(np.sum(w * (-(y * np.log(p) + (1 - y) * np.log(1 - p)))) / np.sum(w))


def group_weights(groups: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for group in groups:
        counts[str(group)] = counts.get(str(group), 0) + 1
    values = np.asarray([1.0 / counts[str(group)] for group in groups], dtype=float)
    return values * (len(values) / values.sum())

