#!/usr/bin/env python3
"""Build the non-deployable crowd event-state V3 catalog for frozen Step 3."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import file_sha256, write_json


TARGET_KEY = "crowd_escalation_chain"
ACTIVE_KEY = "crowd_escalation_active_transition_v3"
AFTERMATH_KEY = "crowd_escalation_causal_aftermath_v3"


def _node(key: str, title: str, cues: list[str], role: str, phase: str) -> dict:
    return {
        "key": key, "title": title, "cue_bundle": cues,
        "required": True, "anchor": True, "weight": 1.0,
        "role": role, "phase_hint": phase,
    }


def build(base_path: Path, out_dir: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    catalog = {
        "version": "crowd_event_state_v3_frozen_validation_catalog",
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
        "counterfactual_links": [
            "ceremonial_crowd_gathering",
            "benign_or_pre_event_crowd_context_v3",
        ],
        "hypothesis_id": "crowd_event_state_v3",
    }
    active = dict(shared, **{
        "key": ACTIVE_KEY,
        "title": "Crowd escalation with a visible current-window transition",
        "canonical_factors": ["crowd", "active_escalation", "visible_transition"],
        "joint_semantics": (
            "visible crowd disorder and opposing motion are tentative support only; a separate "
            "event-state audit must verify a new physical escalation mechanism inside the window"
        ),
        "applicability": ["crowd disorder with a directly observable current-window escalation"],
        "falsifiers": [
            "running, fleeing, chasing, gesturing, or crowd density without visible physical escalation",
            "text overlay is the only evidence of violence",
            "the apparent escalation is inferred across unrelated scene cuts",
        ],
        "nodes": [
            _node(
                "visible_disorder_or_scatter", "visible disorder, panic, or abrupt loss of order",
                ["abrupt visible disorder", "people scatter or break formation", "panic or confrontation"],
                "event_anchor", "active",
            ),
            _node(
                "opposing_crowd_motion", "opposing or aggressive collective motion",
                ["opposing groups move against one another", "pushing or resisting", "aggressive crowd surge"],
                "event_anchor", "active",
            ),
        ],
    })
    aftermath = dict(shared, **{
        "key": AFTERMATH_KEY,
        "title": "Crowd escalation with causally linked current-event aftermath",
        "canonical_factors": ["crowd", "aftermath", "visible_current_event_link"],
        "joint_semantics": (
            "crowd disorder and damage or emergency evidence are tentative support only; a separate "
            "event-state audit must verify that the aftermath is tied to the current visible event"
        ),
        "applicability": ["crowd disorder with aftermath visibly linked to the same current event"],
        "falsifiers": [
            "old or static damage without an observed producing event",
            "generic military or emergency context without a current-event link",
            "isolated smoke, stage effects, or pre-existing cracked glass",
        ],
        "nodes": [
            _node(
                "visible_disorder_or_scatter", "visible disorder, panic, or abrupt loss of order",
                ["abrupt visible disorder", "people scatter or break formation", "panic or confrontation"],
                "event_anchor", "any",
            ),
            _node(
                "damage_debris_smoke_fire_aftermath", "damage, debris, fire, smoke, or emergency aftermath",
                [
                    "visible physical damage", "debris", "visible fire or burning",
                    "smoke with damage", "emergency response tied to an incident",
                ],
                "effect", "aftermath",
            ),
        ],
    })
    catalog["abnormal"].extend((active, aftermath))
    catalog["abnormal"].sort(key=lambda graph: str(graph.get("key", "")))
    catalog["event_state_candidates"] = {
        TARGET_KEY: {
            "version": "crowd_event_state_candidate_v3",
            "members": [ACTIVE_KEY, AFTERMATH_KEY],
            "states": [
                "active_escalation", "causally_linked_aftermath",
                "benign_or_pre_event_context", "none_or_unobservable",
            ],
            "scoring": "event-state-gated unary OT in one frozen target slot",
            "deployable": False,
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "graph_catalog_v2.json"
    write_json(catalog_path, catalog)
    manifest = {
        "version": "crowd_event_state_v3_catalog_manifest",
        "deployable": False,
        "base_catalog": str(base_path), "base_catalog_sha256": file_sha256(base_path),
        "catalog": str(catalog_path), "catalog_sha256": file_sha256(catalog_path),
        "target_slot": TARGET_KEY,
        "candidate_target_graph_keys": [ACTIVE_KEY, AFTERMATH_KEY],
        "candidate_scoring_mode": "event_state_v3",
        "scientific_change": (
            "one mutually exclusive event-state audit distinguishes current-window active transition, "
            "causally linked aftermath, benign/pre-event context, and unobservable evidence"
        ),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.base_catalog, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

