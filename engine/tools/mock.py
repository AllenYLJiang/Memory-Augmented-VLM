#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny builders for offline tests (no VLM)."""
from __future__ import annotations
import numpy as np
from schemas import (EvidenceUnit, EvidenceSet, GraphNodeV2, GraphTemplateV2,
                     UnaryAffinity, ConditionalAffinity, N_BINS)


def mk_evidence(n=3, segment_key="seg"):
    units = tuple(EvidenceUnit(id=f"E{j+1}", start_bin=j*(N_BINS//max(1,n)), end_bin=j*(N_BINS//max(1,n)),
                               entity_ids=(f"ent{j}",), region="full_frame", description=f"fact {j+1}",
                               visibility=3, salience=3, uncertainty=1) for j in range(n))
    return EvidenceSet(segment_key=segment_key, units=units, scene_summary="neutral",
                       scene_cut_bins=tuple(), global_visibility=3, uncertainty=1, prompt_version="v3")


def mk_graph(keys, required=None, polarity="abnormal", ordered=True):
    required = required or [True]*len(keys)
    nodes = tuple(GraphNodeV2(key=k, title=k, cue_bundle=(k,), required=required[i], anchor=(i==0),
                              weight=1.0 if required[i] else 0.5, role="event_anchor" if i==0 else "active",
                              phase_hint="any") for i, k in enumerate(keys))
    return GraphTemplateV2(
        key="G", title="G", polarity=polarity, joint_semantics="joint", ordered=ordered,
        nodes=nodes, family="synthetic",
        matching_policy={"allow_null": True, "use_conditional_refinement": True, "temporal_mode": "soft_auto"},
    )


def mk_unary(node_keys, evidence_ids, scores, null, phase=None):
    scores = np.asarray(scores, float); null = np.asarray(null, float)
    K, M = len(node_keys), len(evidence_ids)
    phase = np.asarray(phase, float) if phase is not None else np.full((K, M), 1.0 / max(M, 1))
    return UnaryAffinity(
        node_keys=list(node_keys), evidence_ids=list(evidence_ids), scores=scores,
        null_scores=null, phase_scores=phase, uncertainty=np.full(K, 0.2),
        node_presence_priors=np.clip(1.0 - null, 0.0, 1.0),
    )


def mk_conditional(node_keys, evidence_ids, scores, null, priors, coherence=0.8, phase=None):
    scores = np.asarray(scores, float); null = np.asarray(null, float); priors = np.asarray(priors, float)
    K, M = len(node_keys), len(evidence_ids)
    phase = np.asarray(phase, float) if phase is not None else np.full((K, M), 1.0 / max(M, 1))
    return ConditionalAffinity(
        node_keys=list(node_keys), evidence_ids=list(evidence_ids), scores=scores,
        null_scores=null, node_presence_priors=priors, phase_scores=phase,
        graph_coherence=coherence, uncertainty=np.full(K, 0.2),
    )
