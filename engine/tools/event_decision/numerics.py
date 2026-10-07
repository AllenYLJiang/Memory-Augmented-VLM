"""Stable box-constrained ridge logistic regression with a verifiable certificate.

The V9 scorer remains available for historical replay. New fits use this module.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from .contracts import semantic_sha256
from .models import RidgeLogisticScorer, sigmoid


OBJECTIVE = {"version": "weighted_ridge_logistic_float64_v2", "weights": "normalized",
             "intercept_regularized": False, "loss": "logaddexp_without_clipping"}


def objective_gradient(theta, X, y, weights, regularization):
    z = theta[0] + X @ theta[1:]
    loss = np.where(y == 1, np.logaddexp(0., -z), np.logaddexp(0., z))
    residual = weights * (sigmoid(z) - y)
    gradient = np.r_[residual.sum(), X.T @ residual + regularization * theta[1:]]
    objective = weights @ loss + .5 * regularization * (theta[1:] @ theta[1:])
    return float(objective), gradient


def certificate(theta, X, y, weights, regularization, lower, upper, tolerance=1e-7,
                iterations=0, termination="POSTHOC", fit_context=None):
    objective, gradient = objective_gradient(theta, X, y, weights, regularization)
    pg = theta - np.clip(theta - gradient, lower, upper)
    kkt = np.abs(gradient)
    at_lower, at_upper = theta <= lower + 1e-12, theta >= upper - 1e-12
    kkt[at_lower] = np.maximum(0., -gradient[at_lower])
    kkt[at_upper] = np.maximum(0., gradient[at_upper])
    kkt[lower == upper] = 0.
    primal = max(0., float(np.max(lower - theta)), float(np.max(theta - upper)))
    finite = bool(np.isfinite(theta).all() and np.isfinite(gradient).all() and np.isfinite(objective))
    certified = bool(finite and np.max(np.abs(pg)) <= tolerance and np.max(kkt) <= tolerance and primal <= 1e-10)
    digest = hashlib.sha256()
    for array in (X, y, weights, lower, upper):
        a = np.ascontiguousarray(array, dtype='<f8')
        digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    digest.update(semantic_sha256({"lambda": regularization, "context": fit_context}).encode())
    return {"version": "numeric_certificate_v2", "objective": objective,
            "raw_gradient_l2": float(np.linalg.norm(gradient)), "raw_gradient_inf": float(np.max(np.abs(gradient))),
            "projected_gradient_inf": float(np.max(np.abs(pg))), "kkt_inf": float(np.max(kkt)),
            "primal_violation_inf": primal, "tolerance": tolerance, "primal_tolerance": 1e-10,
            "finite": finite, "iterations": iterations, "termination": termination,
            "status": "CERTIFIED" if certified else termination + "_NOT_CERTIFIED", "certified": certified,
            "objective_contract_sha256": semantic_sha256(OBJECTIVE), "fit_arrays_sha256": digest.hexdigest()}


class CertifiedRidgeLogisticScorer(RidgeLogisticScorer):
    def fit(self, X, y, sample_weight=None, *, fit_context=None):
        self.intercept_, self.coef_, self.objective_ = None, None, None
        self.converged_, self.iterations_, self.certificate_ = False, 0, None
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or not 0 < X.shape[1] <= 12 or y.shape != (len(X),) or not len(X):
            raise ValueError("INVALID_INPUT: matrix dimensions")
        if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("INVALID_INPUT: finite X and binary y required")
        if len(np.unique(y)) != 2:
            raise ValueError("INSUFFICIENT_CLASSES")
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("INVALID_INPUT: nonnegative finite weights with positive sum required")
        weights = weights / weights.sum()
        if any(weights[y == label].sum() <= 0 for label in (0, 1)):
            raise ValueError("INSUFFICIENT_CLASSES: zero weighted class")
        if not np.isfinite(self.regularization) or self.regularization < 0 or self.max_iter < 0 or not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("INVALID_INPUT: solver configuration")
        lower, upper = np.full(X.shape[1] + 1, -np.inf), np.full(X.shape[1] + 1, np.inf)
        for bounds, destination in ((self.lower_bounds, lower), (self.upper_bounds, upper)):
            for index, bound in bounds.items():
                if not isinstance(index, int) or not 0 <= index < X.shape[1] or not np.isfinite(bound):
                    raise ValueError("INVALID_INPUT: coefficient bound")
                destination[index + 1] = bound
        if np.any(lower > upper):
            raise ValueError("INFEASIBLE_BOUNDS")
        prevalence = weights @ y
        theta = np.clip(np.r_[np.log(prevalence / (1 - prevalence)), np.zeros(X.shape[1])], lower, upper)
        reason = "MAX_ITER"
        for iteration in range(self.max_iter):
            current = certificate(theta, X, y, weights, self.regularization, lower, upper, self.tolerance)
            if current['certified']:
                reason = "PROJECTED_KKT"; break
            old, gradient = objective_gradient(theta, X, y, weights, self.regularization)
            if not np.isfinite(old) or not np.isfinite(gradient).all():
                reason = "NONFINITE"; break
            step, accepted = 1., False
            for _ in range(40):
                trial = np.clip(theta - step * gradient, lower, upper)
                delta = trial - theta
                value, _ = objective_gradient(trial, X, y, weights, self.regularization)
                if np.any(delta) and np.isfinite(value) and value <= old + 1e-4 * (gradient @ delta):
                    theta, accepted = trial, True; break
                step *= .5
            self.iterations_ = iteration + 1
            if not accepted:
                reason = "LINE_SEARCH_FAILED"; break
        self.certificate_ = certificate(theta, X, y, weights, self.regularization, lower, upper,
                                        self.tolerance, self.iterations_, reason, fit_context)
        self.intercept_, self.coef_ = float(theta[0]), theta[1:]
        self.objective_ = self.certificate_['objective']
        self.converged_ = self.certificate_['certified']
        return self

    def to_json(self):
        return {**super().to_json(), "type": "ridge_logistic_scorer_v2", "numeric_certificate": getattr(self, 'certificate_', None)}


def named_bounds(feature_names, bounds: Mapping[str, float]):
    unknown = set(bounds) - set(feature_names)
    if unknown:
        raise ValueError(f"unknown bounded features: {sorted(unknown)}")
    return {feature_names.index(name): float(value) for name, value in bounds.items()}
