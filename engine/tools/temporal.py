#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal compatibility without legacy edges (spec 14). Soft distributions only; no hard
'A before B' gate."""
from __future__ import annotations
import numpy as np
from schemas import N_BINS

EPS = 1e-9


def evidence_time_mask(units) -> np.ndarray:
    """[M, 8] normalized temporal mask per evidence unit (spec 14.1)."""
    M = len(units)
    mask = np.zeros((M, N_BINS), dtype=np.float64)
    for j, u in enumerate(units):
        s, e = int(u.start_bin), int(u.end_bin)
        s, e = max(0, min(N_BINS - 1, s)), max(0, min(N_BINS - 1, e))
        if e < s:
            s, e = e, s
        span = e - s + 1
        mask[j, s:e + 1] = 1.0 / span
    return mask


def phase_softmax(phase_ordinal: np.ndarray) -> np.ndarray:
    """Ordinal [K,8] -> softmax phase distribution per node (spec 14.2)."""
    logits = np.asarray(phase_ordinal, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def temporal_compatibility(phase_dist: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    """T[i,j] = sum_b q_i(b) m_j(b)  (spec 14.2). phase_dist [K,8], time_mask [M,8] -> [K,M]."""
    return np.clip(phase_dist @ time_mask.T, 0.0, 1.0)


def crossing_score(plan_real: np.ndarray, expected_time: np.ndarray) -> float:
    """Diagnostic soft monotonicity crossing R_cross (spec 14.3). plan_real [K,M].
    Reported only, never a hard gate."""
    K, M = plan_real.shape
    order = np.argsort(expected_time)  # node order by time
    r = 0.0
    # penalize pairs (i<k in graph order) whose assigned evidence times invert
    for i in range(K):
        for k in range(i + 1, K):
            for j in range(M):
                for l in range(M):
                    if l < j:  # k assigned earlier evidence than i -> crossing
                        r += float(plan_real[i, j] * plan_real[k, l])
    return r
