#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-API demonstration of the six controlled v3 methods."""
from __future__ import annotations

import matching as mm
import mock
from competition import compete
from leave_one_out import delta_matrix


def demo():
    graph_abnormal = mock.mk_graph(["A", "B", "C"], required=[True, True, False], polarity="abnormal")
    graph_normal = mock.mk_graph(["X", "Y"], required=[True, True], polarity="normal")
    evidence_centers = [0.5, 3.5, 6.5]

    unary_abnormal = mock.mk_unary(
        ["A", "B", "C"], ["E1", "E2", "E3"],
        [[0.88, 0.30, 0.20], [0.86, 0.75, 0.25], [0.84, 0.20, 0.20]],
        [0.15, 0.20, 0.60],
    )
    unary_normal = mock.mk_unary(
        ["X", "Y"], ["E1", "E2", "E3"],
        [[0.50, 0.30, 0.20], [0.45, 0.30, 0.20]], [0.50, 0.55],
    )
    conditional = mock.mk_conditional(
        ["A", "B", "C"], ["E1", "E2", "E3"],
        [[0.85, 0.30, 0.20], [0.30, 0.85, 0.25], [0.20, 0.20, 0.20]],
        [0.15, 0.20, 0.80], priors=[0.85, 0.82, 0.10], coherence=0.82,
    )

    m0 = mm.match_graph_independent_direct(
        graph_abnormal,
        {key: value for key, value in zip(["A", "B", "C"], [0.85, 0.80, 0.20])},
    )
    m1 = mm.match_graph_shared_rowmax(graph_abnormal, unary_abnormal, evidence_centers)
    m2 = mm.match_graph_unary_ot(graph_abnormal, unary_abnormal, evidence_centers)
    m3a = mm.match_graph_conditional_rowmax(graph_abnormal, unary_abnormal, conditional, evidence_centers)
    m3b = mm.match_graph_conditional_ot(
        graph_abnormal, unary_abnormal, conditional, evidence_centers, use_coherence=False,
    )
    m3c = mm.match_graph_conditional_ot(
        graph_abnormal, unary_abnormal, conditional, evidence_centers, use_coherence=True,
    )
    normal = mm.match_graph_unary_ot(graph_normal, unary_normal, evidence_centers)

    methods = [m0, m1, m2, m3a, m3b, m3c]
    print("NODE-PROBABILITY FLOW")
    print(f"{'node':4} " + " ".join(f"{value.method[:13]:>13}" for value in methods))
    for key in ["A", "B", "C"]:
        print(f"{key:4} " + " ".join(f"{value.node_presence[key]:>13.3f}" for value in methods))
    print(f"\nM1 slot collisions: {m1.diagnostics['collision_evidence']}")
    print(f"M2 slot collisions: {m2.diagnostics['collision_evidence']}")
    print(f"C final NULL/presence suppression: {m3c.node_presence['C'] < 0.5}")

    print("\nGRAPH SCORE + COMPETITION")
    for value in methods:
        result = compete(value.method, [value], [normal])
        print(
            f"  {value.method:30} abnormal={value.graph_score:.3f} "
            f"normal={normal.graph_score:.3f} margin={result['margin']:+.3f} -> {result['decision']}"
        )

    full = m3c.node_presence
    subset = {"B": {"A": 0.55, "C": 0.05}, "C": {"A": max(0.05, full["A"] - 0.02), "B": full["B"]}}
    row = next(value for value in delta_matrix(full, subset) if value["target_node"] == "A")
    print(
        f"\nLeave-one-out: P(A|full)={row['full_graph_presence']:.3f}; "
        f"P(A|without B)={row.get('without_B_presence')}; "
        f"delta_B->A={row.get('delta_B_to_A'):+.3f}"
    )


if __name__ == "__main__":
    demo()
