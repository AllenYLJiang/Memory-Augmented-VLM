#!/usr/bin/env python3
"""Archive schema-valid discovered candidates without activating them.

Kept under the historical filename so old commands fail safe. Activation now belongs only to
``build_active_graph_library.py`` after deterministic held-out validation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from common import write_json
from discover_ot_failures import _validate_graph
from graph_catalog import infer_family
from selection import source_group_id


def _video_id(segment_key: str) -> str:
    return str(segment_key).rsplit("__seg", 1)[0]


def _read_nonempty_lines(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _source_metadata(item: Mapping[str, Any]) -> dict:
    source_video = str(item.get("source_video_id") or "")
    if not source_video and item.get("source_segment_key"):
        source_video = _video_id(str(item["source_segment_key"]))
    return {
        "source_case_id": item.get("source_case_id"),
        "source_segment_key": item.get("source_segment_key"),
        "source_video_id": source_video,
        "source_group_id": str(item.get("source_group_id") or source_group_id(source_video)),
        "proposal_confidence": item.get("proposal_confidence"),
        "signature": item.get("signature"),
    }


def _cue_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _merge_duplicate_graph(target: dict, incoming: Mapping[str, Any], source: dict) -> None:
    """Consolidate a same-key proposal without duplicating a hypothesis in competition."""
    alternatives = target.setdefault("discovery_alternatives", [])
    alternatives.append({
        "title": incoming.get("title"),
        "joint_semantics": incoming.get("joint_semantics"),
        "source": source,
    })
    target.setdefault("discovery_sources", []).append(source)

    target_nodes = target.get("nodes", [])
    for node in incoming.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        match = next((value for value in target_nodes if value.get("key") == node.get("key")), None)
        if match is None:
            same_role_phase = [
                value for value in target_nodes
                if value.get("role") == node.get("role")
                and value.get("phase_hint") == node.get("phase_hint")
            ]
            match = same_role_phase[0] if len(same_role_phase) == 1 else None
        if match is None:
            # An unmatched node is retained as optional evidence so a duplicate proposal
            # cannot silently make the consolidated graph stricter.
            extra = dict(node)
            extra["required"] = False
            extra["anchor"] = False
            extra["weight"] = min(float(extra.get("weight", 0.5)), 0.5)
            target_nodes.append(extra)
            continue
        aliases = match.setdefault("discovery_node_aliases", [])
        if node.get("key") and node.get("key") != match.get("key") and node.get("key") not in aliases:
            aliases.append(node.get("key"))
        cues = _cue_list(match.get("cue_bundle"))
        for cue in _cue_list(node.get("cue_bundle")):
            if cue not in cues:
                cues.append(cue)
        match["cue_bundle"] = cues


def build(base_path: Path, registry_path: Path, out_dir: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "version": "conditional_ot_graph_library_candidate_archive_v4",
        "abnormal": [dict(value) for value in base.get("abnormal", [])],
        "normal": [dict(value) for value in base.get("normal", [])],
        "inactive_hypotheses": [],
    }
    known_keys = {
        str(graph.get("key"))
        for polarity in ("abnormal", "normal")
        for graph in merged[polarity]
    }
    graph_by_key = {
        str(graph.get("key")): graph
        for polarity in ("abnormal", "normal")
        for graph in merged[polarity]
    }
    accepted = []
    accepted_proposals = []
    consolidated = []
    skipped = []
    source_videos = _read_nonempty_lines(base_path.parent / "discovery_source_videos.txt")
    source_groups = {source_group_id(video_id) for video_id in source_videos}
    added_keys = set()
    for item in registry.get("candidate_graphs", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("graph"), Mapping):
            continue
        graph = dict(item["graph"])
        errors = _validate_graph(graph)
        key = str(graph.get("key", ""))
        source = _source_metadata(item)
        source_video = source["source_video_id"]
        if source_video:
            source_videos.add(source_video)
            source_groups.add(source["source_group_id"])
        if errors:
            skipped.append({"key": key, "reason": "schema_invalid", "errors": errors})
            continue
        if key in known_keys:
            if key in added_keys and graph_by_key[key].get("polarity") == graph.get("polarity"):
                _merge_duplicate_graph(graph_by_key[key], graph, source)
                consolidated.append({
                    "key": key,
                    "source_case_id": item.get("source_case_id"),
                    "reason": "same_key_proposals_consolidated",
                })
                accepted_proposals.append({
                    "key": key,
                    "polarity": graph.get("polarity"),
                    "source_case_id": item.get("source_case_id"),
                    "consolidated_into": key,
                })
                continue
            skipped.append({"key": key, "reason": "duplicate_key"})
            continue
        graph["origin"] = "test_derived_v3_discovery"
        graph["family"] = str(graph.get("family") or infer_family(
            graph.get("key", ""), graph.get("title", ""), graph.get("polarity", "normal"),
        ))
        graph["active"] = False
        graph["status"] = "candidate"
        graph["source_case_id"] = item.get("source_case_id")
        graph["requires_independent_validation"] = True
        graph["discovery_sources"] = [source]
        merged["inactive_hypotheses"].append(graph)
        known_keys.add(key)
        added_keys.add(key)
        graph_by_key[key] = graph
        accepted.append({
            "key": key,
            "polarity": graph["polarity"],
            "source_case_id": item.get("source_case_id"),
            "source_segment_key": item.get("source_segment_key"),
        })
        accepted_proposals.append(dict(accepted[-1]))

    merged["abnormal"].sort(key=lambda graph: str(graph.get("key", "")))
    merged["normal"].sort(key=lambda graph: str(graph.get("key", "")))
    catalog_path = out_dir / "graph_catalog_v2.json"
    write_json(catalog_path, merged)
    (out_dir / "discovery_source_videos.txt").write_text(
        "".join(f"{video_id}\n" for video_id in sorted(source_videos)), encoding="utf-8",
    )
    (out_dir / "discovery_source_groups.txt").write_text(
        "".join(f"{group_id}\n" for group_id in sorted(source_groups)), encoding="utf-8",
    )

    counts = Counter(item["polarity"] for item in accepted)
    manifest = {
        "version": "conditional_ot_graph_library_candidate_archive_manifest_v4",
        "base_catalog": str(base_path),
        "discovery_registry": str(registry_path),
        "catalog": str(catalog_path),
        "base_counts": {
            "abnormal": len(base.get("abnormal", [])),
            "normal": len(base.get("normal", [])),
        },
        "discovered_counts": {"abnormal": counts["abnormal"], "normal": counts["normal"]},
        "final_counts": {"abnormal": len(merged["abnormal"]), "normal": len(merged["normal"])},
        "accepted_candidates": accepted,
        "accepted_proposals": accepted_proposals,
        "consolidated_duplicate_proposals": consolidated,
        "skipped_candidates": skipped,
        "excluded_discovery_source_videos": len(source_videos),
        "excluded_discovery_source_groups": len(source_groups),
        "evaluation_rule": (
            "Test-derived candidates are archived inactive. Only held-out validation followed by "
            "build_active_graph_library.py may activate them."
        ),
    }
    write_json(out_dir / "manifest.json", manifest)

    lines = [
        "# Conditional-OT Candidate Graph Archive",
        "",
        f"- Original graphs: {sum(manifest['base_counts'].values())}",
        f"- Archived unique discovered hypotheses: {len(accepted)}",
        f"- Consolidated duplicate proposals: {len(consolidated)}",
        f"- Final abnormal graphs: {manifest['final_counts']['abnormal']}",
        f"- Final normal graphs: {manifest['final_counts']['normal']}",
        "- Legacy explicit edges: not used",
        "- Activation: forbidden in this builder; held-out validation is mandatory",
        "",
    ]
    for polarity in ("abnormal", "normal"):
        lines.extend([f"## {polarity.title()} graphs", ""])
        for graph in merged[polarity]:
            origin = graph.get("origin", "original_partac_graph")
            nodes = ", ".join(f"`{node.get('key')}`" for node in graph.get("nodes", []))
            lines.extend([
                f"### {graph.get('title', graph.get('key'))}",
                "",
                f"- key: `{graph.get('key')}`",
                f"- origin: `{origin}`",
                f"- semantics: {graph.get('joint_semantics', '')}",
                f"- nodes: {nodes}",
                "",
            ])
    (out_dir / "GRAPH_LIBRARY_V2.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--base", type=Path, default=root / "graph_catalog_v2.json")
    parser.add_argument(
        "--registry", type=Path,
        default=root / "graph_library" / "discovered_ot_graphs.json",
    )
    parser.add_argument("--out-dir", type=Path, default=root / "graph_library_v2")
    args = parser.parse_args()
    manifest = build(args.base, args.registry, args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
