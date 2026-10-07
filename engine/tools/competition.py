#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Abnormal-vs-normal graph competition with candidate-count-normalized aggregation."""
from __future__ import annotations

import math
from typing import List, Sequence

from schemas import GraphMatchResult


def _aggregate(values: Sequence[float], mode: str, temperature: float) -> float:
    if not values:
        return 0.0
    numbers = [float(value) for value in values]
    if mode == "max":
        return max(numbers)
    if mode != "logmeanexp":
        raise ValueError(f"unknown competition aggregation: {mode}")
    tau = max(float(temperature), 1e-6)
    maximum = max(numbers)
    # Normalized log-mean-exp: adding more candidates does not automatically raise the score.
    return maximum + tau * math.log(sum(math.exp((value - maximum) / tau) for value in numbers) / len(numbers))


def compete(
    method: str,
    abnormal_results: List[GraphMatchResult],
    normal_results: List[GraphMatchResult],
    *,
    aggregation: str = "logmeanexp",
    temperature: float = 0.1,
) -> dict:
    best_abnormal = max(abnormal_results, key=lambda result: result.graph_score) if abnormal_results else None
    best_normal = max(normal_results, key=lambda result: result.graph_score) if normal_results else None
    abnormal_score = _aggregate([result.graph_score for result in abnormal_results], aggregation, temperature)
    normal_score = _aggregate([result.graph_score for result in normal_results], aggregation, temperature)
    margin = abnormal_score - normal_score
    decision = "abnormal" if margin > 0 else "normal" if margin < 0 else "uncertain"
    return {
        "method": method,
        "aggregation": aggregation,
        "temperature": float(temperature),
        "best_abnormal_graph": best_abnormal.graph_key if best_abnormal else "NONE",
        "best_abnormal_graph_score": best_abnormal.graph_score if best_abnormal else 0.0,
        "best_abnormal_score": abnormal_score,
        "best_normal_graph": best_normal.graph_key if best_normal else "NONE",
        "best_normal_graph_score": best_normal.graph_score if best_normal else 0.0,
        "best_normal_score": normal_score,
        "margin": margin,
        "decision": decision,
        "y_pred": 1 if decision == "abnormal" else 0,
        "candidate_counts": {"abnormal": len(abnormal_results), "normal": len(normal_results)},
    }


def top_n_sensitivity(
    method: str,
    abnormal_results: List[GraphMatchResult],
    normal_results: List[GraphMatchResult],
    *,
    aggregation: str,
    temperature: float,
) -> list[dict]:
    abnormal = sorted(abnormal_results, key=lambda result: result.graph_score, reverse=True)
    normal = sorted(normal_results, key=lambda result: result.graph_score, reverse=True)
    rows = []
    for normal_count in range(1, len(normal) + 1):
        result = compete(
            method,
            abnormal,
            normal[:normal_count],
            aggregation=aggregation,
            temperature=temperature,
        )
        result["normal_top_n"] = normal_count
        rows.append(result)
    return rows
