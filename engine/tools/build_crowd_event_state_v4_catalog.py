#!/usr/bin/env python3
"""Build the non-deployable equal-contract Crowd Event-State V4 catalog."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import file_sha256, write_json


TARGET_KEY = "crowd_escalation_chain"
ACTIVE_KEY = "crowd_active_or_ongoing_physical_escalation_v4"
AFTERMATH_KEY = "crowd_causally_linked_aftermath_v4"


def _node(key: str, title: str, cues: list[str], role: str, phase: str) -> dict:
    return {
        "key": key, "title": title, "cue_bundle": cues,
        "required": True, "anchor": True, "weight": 1.0,
        "role": role, "phase_hint": phase,
    }


def build(base_path: Path, out_dir: Path, candidate_scoring_mode: str = "event_state_v4") -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    catalog = {
        "version": "crowd_event_state_v4_equal_contract_validation_catalog",
        "abnormal": copy.deepcopy(base.get("abnormal", [])),
        "normal": copy.deepcopy(base.get("normal", [])),
    }
    catalog["abnormal"] = [
        graph for graph in catalog["abnormal"]
        if str(graph.get("key")) not in {ACTIVE_KEY, AFTERMATH_KEY}
    ]
    shared = {
        "polarity": "abnormal", "family": "crowd", "active": True,
        "status": "validation_candidate", "ordered": False,
        "matching_policy": {"allow_null": True, "temporal_mode": "soft_auto"},
        "counterfactual_links": ["ceremonial_crowd_gathering"],
        "hypothesis_id": "crowd_event_state_v4_equal_contract",
    }
    active = dict(shared, **{
        "key": ACTIVE_KEY,
        "title": "Active or ongoing physical crowd escalation",
        "canonical_factors": ["crowd", "active_or_ongoing_escalation", "visible_physical_mechanism"],
        "joint_semantics": (
            "one coherent episode contains directly visible new or ongoing physical coercion, "
            "violence, impact, throwing, barrier attack, or forced dispersal"
        ),
        "applicability": ["new onset or ongoing directly visible physical crowd escalation"],
        "falsifiers": [
            "running, flight, chasing, gesturing, or density without a visible physical mechanism",
            "text overlay is the only evidence",
            "apparent contact is inferred across unrelated cuts",
        ],
        "nodes": [
            _node(
                "direct_physical_escalation_mechanism",
                "directly visible new or ongoing physical escalation mechanism",
                [
                    "striking, grappling, pushing, or forceful coercion",
                    "projectile or object throwing", "barrier attack or impact",
                    "forced dispersal or ongoing physical confrontation",
                ],
                "event_anchor", "active",
            ),
            _node(
                "visible_disorder_or_scatter", "visible disorder, panic, or abrupt loss of order",
                ["people scatter or break formation", "panic", "sustained confrontation or disorder"],
                "context", "active",
            ),
            _node(
                "opposing_crowd_motion", "opposing or aggressive collective motion",
                ["opposing groups", "pushing or resisting", "aggressive crowd or police-line motion"],
                "context", "active",
            ),
        ],
    })
    aftermath = dict(shared, **{
        "key": AFTERMATH_KEY,
        "title": "Crowd-event aftermath with a visible current-event link",
        "canonical_factors": ["crowd", "aftermath", "current_event_link"],
        "joint_semantics": (
            "damage, debris, fire, smoke, injury, or emergency response is visibly tied to the "
            "same current crowd event rather than pre-existing context"
        ),
        "applicability": ["current-event aftermath with a directly supported causal or episode link"],
        "falsifiers": [
            "old static damage", "background smoke or stage effects",
            "generic military or emergency context without a current-event link",
        ],
        "nodes": [
            _node(
                "current_event_aftermath_link", "visible link from the current event to its aftermath",
                [
                    "freshly produced damage or debris", "response visibly tied to the same incident",
                    "before/after evidence connects disorder to the aftermath",
                ],
                "event_anchor", "aftermath",
            ),
            _node(
                "damage_debris_smoke_fire_aftermath", "damage, debris, fire, smoke, injury, or emergency aftermath",
                ["physical damage", "debris", "fire or smoke with damage", "injury or emergency response"],
                "effect", "aftermath",
            ),
            _node(
                "visible_disorder_or_scatter", "crowd disorder belonging to the same episode",
                ["crowd disorder", "panic or scatter", "aftermath of confrontation"],
                "context", "any",
            ),
        ],
    })
    catalog["abnormal"].extend((active, aftermath))
    catalog["abnormal"].sort(key=lambda graph: str(graph.get("key", "")))
    catalog["event_state_candidates"] = {
        TARGET_KEY: {
            "version": "crowd_event_state_candidate_v4",
            "members": [ACTIVE_KEY, AFTERMATH_KEY],
            "states": [
                "active_or_ongoing_physical_escalation", "causally_linked_aftermath",
                "pre_event_tension_or_flight", "benign_collective_activity", "none_or_unobservable",
            ],
            "scoring": "method-matched conditional OT with bounded calibrated refinement",
            "deployable": False,
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "graph_catalog_v2.json"
    write_json(catalog_path, catalog)
    manifest = {
        "version": "crowd_event_state_v4_catalog_manifest",
        "deployable": False,
        "base_catalog": str(base_path), "base_catalog_sha256": file_sha256(base_path),
        "catalog": str(catalog_path), "catalog_sha256": file_sha256(catalog_path),
        "target_slot": TARGET_KEY,
        "candidate_target_graph_keys": [ACTIVE_KEY, AFTERMATH_KEY],
        "candidate_scoring_mode": candidate_scoring_mode,
        "scientific_change": (
            "ongoing physical escalation is observable; one shared response supplies phase-conditional "
            "node evidence; each method is recomputed locally; repeated multiplicative gates are removed"
        ),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate-scoring-mode",
        choices=("event_state_v4", "event_state_v5"),
        default="event_state_v4",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.base_catalog, args.out_dir, args.candidate_scoring_mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
