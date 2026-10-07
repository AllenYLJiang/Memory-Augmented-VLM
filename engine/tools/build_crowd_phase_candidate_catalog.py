#!/usr/bin/env python3
"""Build the controlled active/aftermath crowd phase candidate used by frozen Step 3."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import file_sha256, write_json


TARGET_KEY = "crowd_escalation_chain"
ACTIVE_KEY = "crowd_escalation_active_phase"
AFTERMATH_KEY = "crowd_escalation_aftermath_phase"


def _node(key: str, title: str, cues: list[str], role: str, phase: str) -> dict:
    return {
        "key": key, "title": title, "cue_bundle": cues,
        "required": True, "anchor": True, "weight": 1.0,
        "role": role, "phase_hint": phase,
    }


def build(base_path: Path, out_dir: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    catalog = {
        "version": "crowd_phase_v2_frozen_validation_catalog",
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
        "matching_policy": {
            "allow_null": True, "use_conditional_refinement": True,
            "temporal_mode": "soft_auto",
        },
        "counterfactual_links": ["ceremonial_crowd_gathering"],
        "hypothesis_id": "crowd_phase_v2_active_or_aftermath",
    }
    active = dict(shared, **{
        "key": ACTIVE_KEY,
        "title": "Crowd escalation: active conflict phase",
        "canonical_factors": ["crowd_escalation", "active_disorder", "opposing_motion"],
        "joint_semantics": (
            "the two required nodes must describe one active crowd-escalation episode: "
            "visible disorder, panic, scatter, or confrontation together with opposing, "
            "aggressive, pushing, or resisting crowd motion"
        ),
        "applicability": [
            "active crowd disorder with opposing or aggressive collective motion in the same episode"
        ],
        "falsifiers": [
            "orderly or ceremonial crowd movement",
            "celebratory running without confrontation, resistance, panic, or disorder",
            "camera motion or unrelated people moving in different events",
        ],
        "nodes": [
            _node(
                "visible_disorder_or_scatter", "visible disorder, panic, or scatter",
                [
                    "visible disorder", "people scatter, run, duck, or break formation suddenly",
                    "objects thrown, barriers struck, or crowd becomes visibly chaotic",
                    "panic", "confrontation",
                ], "event_anchor", "active",
            ),
            _node(
                "opposing_crowd_motion", "opposing or aggressive crowd motion",
                [
                    "opposing groups move against one another", "aggressive crowd surge",
                    "people push, resist, or clash", "collective motion visibly intensifies confrontation",
                ], "event_anchor", "active",
            ),
        ],
    })
    aftermath = dict(shared, **{
        "key": AFTERMATH_KEY,
        "title": "Crowd escalation: damage or emergency aftermath phase",
        "canonical_factors": ["crowd_escalation", "disorder_context", "damage_or_emergency_aftermath"],
        "joint_semantics": (
            "the two required nodes must describe one crowd-escalation episode: visible disorder, "
            "panic, scatter, or confrontation together with damage, debris, fire, burning, or "
            "emergency-response evidence; isolated smoke is insufficient"
        ),
        "applicability": [
            "crowd disorder with visible damage, debris, fire, burning, or emergency response from the same episode"
        ],
        "falsifiers": [
            "orderly festival or concert with benign stage smoke or fireworks",
            "cooking or grill smoke without crowd disorder or physical damage",
            "damage or emergency lights unrelated to the visible crowd event",
        ],
        "nodes": [
            _node(
                "visible_disorder_or_scatter", "visible disorder, panic, or scatter",
                [
                    "visible disorder", "people scatter, run, duck, or break formation suddenly",
                    "objects thrown, barriers struck, or crowd becomes visibly chaotic",
                    "panic", "confrontation",
                ], "event_anchor", "any",
            ),
            _node(
                "damage_debris_smoke_fire_aftermath", "damage, debris, fire, or emergency aftermath",
                [
                    "visible physical damage", "debris from the crowd incident", "visible fire or burning",
                    "smoke accompanied by fire, damage, or debris", "emergency lights or response tied to the incident",
                ], "effect", "aftermath",
            ),
        ],
    })
    catalog["abnormal"].extend((active, aftermath))
    catalog["abnormal"].sort(key=lambda graph: str(graph.get("key", "")))
    catalog["phase_ensembles"] = {
        TARGET_KEY: {
            "version": "crowd_phase_ensemble_v2",
            "aggregation": "max",
            "members": [ACTIVE_KEY, AFTERMATH_KEY],
            "slot_policy": "replace one frozen baseline target slot with the strongest phase explanation",
            "deployable": False,
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "graph_catalog_v2.json"
    write_json(catalog_path, catalog)
    manifest = {
        "version": "crowd_phase_v2_catalog_manifest",
        "deployable": False,
        "base_catalog": str(base_path), "base_catalog_sha256": file_sha256(base_path),
        "catalog": str(catalog_path), "catalog_sha256": file_sha256(catalog_path),
        "target_slot": TARGET_KEY, "aggregation": "max",
        "candidate_target_graph_keys": [ACTIVE_KEY, AFTERMATH_KEY],
        "scientific_change": (
            "active opposing motion and aftermath damage are alternative phase explanations; "
            "neither is a universal hard prerequisite for all crowd escalation windows"
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
