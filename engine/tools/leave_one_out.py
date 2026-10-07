#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-node-out conditionality delta (spec 17). Deterministic given the per-node
presences of the full graph and each leave-one-out subgraph (produced by VLM in the live
pipeline, or mocks in tests)."""
from __future__ import annotations
from typing import Dict, List


def delta_matrix(full_presence: Dict[str, float], subset_presence: Dict[str, Dict[str, float]]) -> List[dict]:
    """subset_presence[removed_node][target_node] = presence of target when `removed` is dropped.
    delta_{k->i} = p_i(G) - p_i(G\\{k})  (spec 17)."""
    rows = []
    for target, full_p in full_presence.items():
        entry = {"target_node": target, "full_graph_presence": float(full_p)}
        for removed, pres in subset_presence.items():
            if removed == target or target not in pres:
                continue
            entry[f"without_{removed}_presence"] = float(pres[target])
            entry[f"delta_{removed}_to_{target}"] = float(full_p) - float(pres[target])
        rows.append(entry)
    return rows
