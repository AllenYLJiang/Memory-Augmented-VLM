#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dustbin (soft) optimal transport in log space (spec 12).

A generalized SuperGlue-style dustbin Sinkhorn: K nodes vs M evidence units, plus one
node-absent dustbin column and one evidence-unmatched dustbin row. Higher affinity => more
transport mass. Pure numpy so it is verifiable offline; a torch backend is a drop-in
replacement (identical math). No POT dependency.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

EPS = 1e-9


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m.squeeze(axis) + np.log(np.sum(np.exp(a - m), axis=axis) + EPS)
    return out


def ordinal_to_p(s) -> np.ndarray:
    """Map ordinal score s in {0..4} to a bounded affinity p=(s+0.5)/5 in (0.1..0.9) (spec 12.3)."""
    return (np.asarray(s, dtype=np.float64) + 0.5) / 5.0


def build_augmented_log_scores(real_affinity: np.ndarray,
                               node_null_affinity: np.ndarray,
                               evidence_unmatched_affinity: Optional[np.ndarray] = None,
                               dustbin_default: float = 0.5) -> np.ndarray:
    """Return [K+1, M+1] log-affinity matrix (spec 12.4).

    real cells -> log(affinity); node->dustbin col -> log(null); dustbin row->evidence ->
    log(unmatched) (constant if not given); dustbin<->dustbin -> log(1)=0."""
    real = np.asarray(real_affinity, dtype=np.float64)
    K, M = real.shape
    nullv = np.asarray(node_null_affinity, dtype=np.float64).reshape(K)
    if evidence_unmatched_affinity is None:
        unm = np.full(M, float(dustbin_default), dtype=np.float64)
    else:
        unm = np.asarray(evidence_unmatched_affinity, dtype=np.float64).reshape(M)
    aug = np.zeros((K + 1, M + 1), dtype=np.float64)
    aug[:K, :M] = np.log(np.clip(real, EPS, 1.0))
    aug[:K, M] = np.log(np.clip(nullv, EPS, 1.0))
    aug[K, :M] = np.log(np.clip(unm, EPS, 1.0))
    aug[K, M] = 0.0
    return aug


def _marginals(K: int, M: int):
    total = float(K + M)
    log_mu = np.log(np.concatenate([np.ones(K), [float(M)]]) / total + EPS)   # [K+1]
    log_nu = np.log(np.concatenate([np.ones(M), [float(K)]]) / total + EPS)   # [M+1]
    return log_mu, log_nu


def log_sinkhorn_iterations(log_scores: np.ndarray, log_mu: np.ndarray, log_nu: np.ndarray,
                            iterations: int = 50) -> np.ndarray:
    """Log-domain Sinkhorn. Returns the log transport plan [K+1, M+1] with row/col
    marginals exp(log_mu)/exp(log_nu)."""
    u = np.zeros(log_scores.shape[0], dtype=np.float64)
    v = np.zeros(log_scores.shape[1], dtype=np.float64)
    for _ in range(int(iterations)):
        u = log_mu - _logsumexp(log_scores + v[None, :], axis=1)
        v = log_nu - _logsumexp(log_scores + u[:, None], axis=0)
    return log_scores + u[:, None] + v[None, :]


def solve_dustbin_ot(augmented_log_scores: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Solve the dustbin OT; return the transport plan [K+1, M+1] (total mass 1)."""
    K1, M1 = augmented_log_scores.shape
    K, M = K1 - 1, M1 - 1
    log_mu, log_nu = _marginals(K, M)
    log_plan = log_sinkhorn_iterations(augmented_log_scores, log_mu, log_nu, iterations)
    return np.exp(log_plan)


def summarize_transport(plan: np.ndarray, evidence_centers: List[float]) -> dict:
    """Per-node presence / location / expected-time / null-mass / entropy (spec 12.5)."""
    K = plan.shape[0] - 1
    M = plan.shape[1] - 1
    centers = np.asarray(evidence_centers, dtype=np.float64).reshape(M) if M > 0 else np.zeros(0)
    presence = np.zeros(K); null_mass = np.zeros(K); expected = np.zeros(K)
    entropy = np.zeros(K); location = np.zeros((K, M))
    for i in range(K):
        row = plan[i, :]
        row_mass = row.sum()
        real_mass = plan[i, :M].sum()
        null_mass[i] = plan[i, M]
        presence[i] = real_mass / max(row_mass, EPS)
        if M > 0:
            loc = plan[i, :M] / max(real_mass, EPS)
            location[i] = loc
            scale = max(float(np.max(centers)), 1.0)
            expected[i] = float(np.sum(loc * (centers / scale)))
            pnz = loc[loc > EPS]
            entropy[i] = float(-np.sum(pnz * np.log(pnz))) if pnz.size else 0.0
    return {"node_presence": presence, "node_location": location, "expected_time": expected,
            "null_mass": null_mass, "entropy": entropy}
