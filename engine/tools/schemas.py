#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataclasses for conditional spatio-temporal OT graph matching.

The v3 schema separates three quantities that were conflated in v2:

* node presence probability;
* location distribution conditional on presence;
* visual evidence quality at each location.

Keeping these separate avoids multiplying the same VLM confidence several times and makes
M3a/M3b/M3c ablations interpretable.  All structures are JSON-serializable through the
normal ``dataclasses.asdict`` path used by :mod:`matching`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PHASES = ("any", "prelude", "onset", "active", "aftermath")
N_BINS = 8


@dataclass(frozen=True)
class WindowCase:
    segment_key: str
    video_id: str
    video_path: str
    start_frame: int
    end_frame: int
    y_true: Optional[int]
    source_record: dict


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    start_bin: int
    end_bin: int
    entity_ids: Tuple[str, ...]
    region: str
    description: str
    visibility: float
    salience: float
    uncertainty: float

    @property
    def center_bin(self) -> float:
        return 0.5 * (int(self.start_bin) + int(self.end_bin))


@dataclass(frozen=True)
class EvidenceSet:
    segment_key: str
    units: Tuple[EvidenceUnit, ...]
    scene_summary: str
    scene_cut_bins: Tuple[int, ...]
    global_visibility: float
    uncertainty: float
    prompt_version: str

    @property
    def ids(self) -> List[str]:
        return [u.id for u in self.units]


@dataclass(frozen=True)
class GraphNodeV2:
    key: str
    title: str
    cue_bundle: Tuple[str, ...]
    required: bool
    anchor: bool
    weight: float
    role: str
    phase_hint: str


@dataclass(frozen=True)
class GraphTemplateV2:
    key: str
    title: str
    polarity: str
    joint_semantics: str
    ordered: bool
    nodes: Tuple[GraphNodeV2, ...]
    matching_policy: dict
    family: str = "other"
    status: str = "active"
    confidence: float = 1.0
    utility_global: float = 0.0
    applicability: Tuple[str, ...] = ()
    falsifiers: Tuple[str, ...] = ()
    counterfactual_links: Tuple[str, ...] = ()
    canonical_factors: Tuple[str, ...] = ()

    @property
    def node_keys(self) -> List[str]:
        return [n.key for n in self.nodes]

    @property
    def weights(self) -> np.ndarray:
        # Respect the catalog value.  v2 silently replaced custom weights with 1/0.5.
        return np.array([float(n.weight) for n in self.nodes], dtype=np.float64)


@dataclass
class UnaryAffinity:
    node_keys: List[str]                  # K
    evidence_ids: List[str]               # M
    scores: np.ndarray                    # [K, M] evidence quality in (0,1)
    null_scores: np.ndarray               # [K] in (0,1)
    phase_scores: np.ndarray              # [K, M] location P(evidence | node present)
    uncertainty: np.ndarray               # [K]
    node_presence_priors: Optional[np.ndarray] = None  # [K]
    cache_sha1: str = ""


@dataclass
class ConditionalAffinity:
    node_keys: List[str]
    evidence_ids: List[str]
    scores: np.ndarray                    # [K, M] conditional evidence quality
    null_scores: np.ndarray               # [K]
    node_presence_priors: np.ndarray      # [K] P(node | whole node set, initial OT)
    phase_scores: np.ndarray              # [K, M] location distribution given present
    graph_coherence: float
    supporting_nodes: Dict[str, List[str]] = field(default_factory=dict)
    suppressing_nodes: Dict[str, List[str]] = field(default_factory=dict)
    uncertainty: Optional[np.ndarray] = None
    cache_sha1: str = ""


@dataclass
class OTPlanSummary:
    plan: np.ndarray                      # [K+1, M+1]
    node_presence: np.ndarray             # [K]
    node_location: np.ndarray             # [K, M]
    expected_time: np.ndarray             # [K], normalized 0..1
    null_mass: np.ndarray                 # [K]
    entropy: np.ndarray                   # [K]


@dataclass
class GraphMatchResult:
    graph_key: str
    method: str
    node_support: Dict[str, float]
    node_presence: Dict[str, float]
    expected_time: Dict[str, float]
    assignments: List[dict]
    transport_plan: Optional[List[List[float]]]
    node_geometric_score: float
    graph_coherence: float
    graph_score: float
    required_coverage: float
    optional_coverage: float
    complete: bool
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
