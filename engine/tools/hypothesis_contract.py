#!/usr/bin/env python3
"""Deterministic contract for atomic graph-memory mutations."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


GRAPH_MUTATIONS = {"ADD", "UPDATE"}
DIAGNOSTIC_OPERATIONS = {"NOOP", "REPRESENTATION_ONLY", "RETIRE", "LINK_COUNTERFACTUAL"}
COMPOUND_OPERATIONS = {"MERGE", "SPLIT"}


def graph_schema() -> dict:
    return {
        "key": "snake_case_key", "title": "", "polarity": "abnormal|normal", "family": "",
        "canonical_factors": [], "joint_semantics": "", "applicability": [], "falsifiers": [],
        "counterfactual_links": [], "ordered": False,
        "matching_policy": {
            "allow_null": True, "use_conditional_refinement": True, "temporal_mode": "soft_auto",
        },
        "nodes": [{
            "key": "", "title": "directly visible factor", "cue_bundle": ["direct visual cue"],
            "required": True, "anchor": True, "weight": 1.0,
            "role": "event_anchor|confound_anchor|context|effect",
            "phase_hint": "any|prelude|onset|active|aftermath",
        }],
    }


def clean_targets(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable) or isinstance(values, Mapping):
        return []
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def atomic_contract_errors(row: Mapping[str, Any], catalog_keys: set[str] | None = None) -> list[str]:
    """Return errors that would make a catalog mutation ambiguous or partial."""
    operation = str(row.get("operation", "NOOP")).upper()
    targets = clean_targets(row.get("targets", []))
    graph = row.get("graph", {})
    graph = graph if isinstance(graph, Mapping) else {}
    key = str(graph.get("key", "")).strip()
    errors: list[str] = []

    if operation == "ADD":
        if targets:
            errors.append("add_must_not_declare_targets")
        if not key:
            errors.append("add_requires_graph")
        if catalog_keys is not None and key in catalog_keys:
            errors.append(f"add_duplicates_existing_key:{key}")
    elif operation == "UPDATE":
        if len(targets) != 1:
            errors.append("update_requires_exactly_one_target")
        if not key:
            errors.append("update_requires_graph")
        elif len(targets) == 1 and key != targets[0]:
            errors.append(f"update_key_must_equal_target:{key}!={targets[0]}")
        if catalog_keys is not None and len(targets) == 1 and targets[0] not in catalog_keys:
            errors.append(f"update_target_not_in_catalog:{targets[0]}")
    elif operation in COMPOUND_OPERATIONS:
        errors.append(f"{operation.lower()}_requires_expansion_to_atomic_mutations")
    elif operation not in DIAGNOSTIC_OPERATIONS:
        errors.append(f"unsupported_operation:{operation}")

    return sorted(set(errors))


def _bundle_id(row: Mapping[str, Any]) -> str:
    revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
    return str(revision.get("bundle_id") or row.get("bundle_id") or "").strip()


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("id") or f"row_{index}")


def _materialized_mutation(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status", ""))
    return str(row.get("operation", "NOOP")).upper() in GRAPH_MUTATIONS and status not in {
        "superseded_revision_draft", "bundle_contract_rejected",
    }


def bundle_contract_errors(
    rows: Iterable[Mapping[str, Any]],
    catalog_keys: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return row-specific errors for ambiguous mutations emitted in one bundle."""
    values = list(rows)
    errors: dict[str, list[str]] = defaultdict(list)
    bundles: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    global_adds: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for index, row in enumerate(values, 1):
        row_id = _row_id(row, index)
        bundle_id = _bundle_id(row) or f"__unbundled__:{row_id}"
        bundles[bundle_id].append((index, row))
        if _materialized_mutation(row) and str(row.get("operation", "")).upper() == "ADD":
            graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
            key = str(graph.get("key", "")).strip()
            if key:
                global_adds[key].append((row_id, repr(sorted(graph.items()))))

    for key, entries in global_adds.items():
        if len(entries) > 1 and len({payload for _, payload in entries}) > 1:
            for row_id, _ in entries:
                errors[row_id].append(f"conflicting_add_key_across_bundles:{key}")

    for bundle_id, members in bundles.items():
        updates: dict[str, list[str]] = defaultdict(list)
        adds: dict[str, list[str]] = defaultdict(list)
        noops: dict[str, list[str]] = defaultdict(list)
        key_polarities: dict[str, set[str]] = defaultdict(set)
        declared: set[str] = set()
        dispositions: set[str] = set()

        for index, row in members:
            row_id = _row_id(row, index)
            revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
            declared.update(clean_targets(revision.get("bundle_declared_targets", [])))
            operation = str(row.get("operation", "NOOP")).upper()
            targets = clean_targets(row.get("targets", []))
            graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
            key = str(graph.get("key", "")).strip()
            polarity = str(graph.get("polarity", "")).strip()

            if operation == "NOOP":
                for target in targets:
                    noops[target].append(row_id)
                    dispositions.add(target)
            elif _materialized_mutation(row) and operation == "UPDATE":
                for target in targets:
                    updates[target].append(row_id)
                    dispositions.add(target)
            elif _materialized_mutation(row) and operation == "ADD" and key:
                adds[key].append(row_id)

            if _materialized_mutation(row) and key and polarity:
                key_polarities[key].add(polarity)

        for target, row_ids in updates.items():
            if len(row_ids) > 1:
                for row_id in row_ids:
                    errors[row_id].append(
                        f"bundle_duplicate_update_target:{bundle_id}:{target}"
                    )
            if target in noops:
                for row_id in [*row_ids, *noops[target]]:
                    errors[row_id].append(f"bundle_update_and_noop_same_target:{bundle_id}:{target}")
        for key, row_ids in adds.items():
            if len(row_ids) > 1:
                for row_id in row_ids:
                    errors[row_id].append(f"bundle_duplicate_add_key:{bundle_id}:{key}")
        for key, polarities in key_polarities.items():
            if len(polarities) > 1:
                for index, row in members:
                    graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
                    if str(graph.get("key", "")).strip() == key:
                        errors[_row_id(row, index)].append(
                            f"bundle_key_polarity_conflict:{bundle_id}:{key}"
                        )

        missing = sorted(declared - dispositions)
        if missing:
            message = f"bundle_missing_target_disposition:{bundle_id}:{','.join(missing)}"
            for index, row in members:
                errors[_row_id(row, index)].append(message)

    if catalog_keys is not None:
        for index, row in enumerate(values, 1):
            for error in atomic_contract_errors(row, catalog_keys):
                errors[_row_id(row, index)].append(error)
    return {key: sorted(set(value)) for key, value in errors.items() if value}


def validation_action_set_errors(
    rows: Iterable[Mapping[str, Any]],
    catalog_keys: set[str],
) -> list[dict[str, Any]]:
    """Preflight a complete held-out catalog mutation set before applying any row."""
    values = list(rows)
    errors: list[dict[str, Any]] = []
    update_targets: dict[str, list[str]] = defaultdict(list)
    add_keys: dict[str, list[str]] = defaultdict(list)

    for index, row in enumerate(values, 1):
        row_id = _row_id(row, index)
        if row.get("status") != "candidate":
            errors.append({"row_id": row_id, "code": "validation_action_not_candidate"})
            continue
        for error in atomic_contract_errors(row, catalog_keys):
            errors.append({"row_id": row_id, "code": error})
        operation = str(row.get("operation", "NOOP")).upper()
        targets = clean_targets(row.get("targets", []))
        graph = row.get("graph", {}) if isinstance(row.get("graph"), Mapping) else {}
        key = str(graph.get("key", "")).strip()
        if operation == "UPDATE" and len(targets) == 1:
            update_targets[targets[0]].append(row_id)
        elif operation == "ADD" and key:
            add_keys[key].append(row_id)

    for target, row_ids in update_targets.items():
        if len(row_ids) > 1:
            for row_id in row_ids:
                errors.append({"row_id": row_id, "code": f"duplicate_update_action:{target}"})
    for key, row_ids in add_keys.items():
        if len(row_ids) > 1:
            for row_id in row_ids:
                errors.append({"row_id": row_id, "code": f"duplicate_add_action:{key}"})
        if key in update_targets:
            for row_id in [*row_ids, *update_targets[key]]:
                errors.append({"row_id": row_id, "code": f"add_update_key_collision:{key}"})

    for index, row in enumerate(values, 1):
        for error in row.get("bundle_contract_errors", []) or []:
            errors.append({"row_id": _row_id(row, index), "code": str(error)})
    return errors
