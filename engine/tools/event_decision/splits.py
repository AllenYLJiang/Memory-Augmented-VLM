from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .contracts import iter_jsonl, read_json, write_json, write_jsonl


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _assign_balanced(groups: Mapping[str, list[dict[str, Any]]], folds: int, seed: int) -> dict[str, int]:
    counts = [0] * folds
    positives = [0] * folds
    negatives = [0] * folds
    assignment: dict[str, int] = {}
    ordered = sorted(
        groups,
        key=lambda g: (
            -max(sum(int(r["supervision"]["window_target"]) == 1 for r in groups[g]), sum(int(r["supervision"]["window_target"]) == 0 for r in groups[g])),
            -len(groups[g]), _rank(seed, g),
        ),
    )
    for group in ordered:
        group_pos = sum(int(r["supervision"]["window_target"]) == 1 for r in groups[group])
        group_neg = len(groups[group]) - group_pos
        fold = min(range(folds), key=lambda i: (positives[i] if group_pos >= group_neg else negatives[i], counts[i], i))
        assignment[group] = fold
        counts[fold] += len(groups[group])
        positives[fold] += group_pos
        negatives[fold] += group_neg
    return assignment


def make_grouped_split_plan(work_root: Path, config: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role != "development_nested_group_oof":
        raise ValueError("only development_nested_group_oof is registered")
    review = read_json(Path(work_root) / "review_adjudication/supervision_scope_findings.json", {})
    minimum = int(config.get("review_minimum_completed", 20))
    if (
        review.get("completed_unique", 0) < minimum
        or review.get("unresolved_disagreements", 0)
        or review.get("errors", 0)
        or review.get("ready_for_fit") is not True
    ):
        from .contracts import WaitingForHumanReview
        raise WaitingForHumanReview("complete and reconcile the blind review before split generation")
    labels = list(iter_jsonl(Path(work_root) / "labels/label_ledger.jsonl"))
    eligible = [r for r in labels if r.get("supervision", {}).get("window_loss_mask") and r.get("supervision", {}).get("window_target") in (0, 1)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[str(row.get("source_group", ""))].append(row)
    desired = int(config.get("outer_group_folds", 5))
    folds = desired if len(groups) >= desired * 2 else 3 if len(groups) >= 6 else 0
    if not folds:
        from .contracts import MissingLocalData
        raise MissingLocalData(f"insufficient independent source groups for grouped OOF: {len(groups)}")
    seed = int(config.get("seed", 20260907))
    assignment = _assign_balanced(groups, folds, seed)
    rows = [{"window_uid": row["window_uid"], "source_group": row.get("source_group"), "y": row["supervision"]["window_target"], "outer_fold": assignment[str(row.get("source_group", ""))], "label_evidence": row["supervision"]["evidence_level"]} for row in eligible]
    out = Path(work_root) / "splits"
    write_jsonl(out / "historical_source_usage.jsonl", [{"source_group": g, "historical_roles": sorted({r["historical_split"] for r in values}), "v9_role": "development_nested_group_oof"} for g, values in groups.items()])
    plan = {"version": "nested_source_group_plan_v1", "role": role, "outer_folds": folds, "inner_folds": int(config.get("inner_group_folds", 3)), "threshold_pool_fraction": float(config.get("threshold_pool_fraction", .2)), "seed": seed, "rows": rows, "human_review_used_for_selection": False}
    write_json(out / "outer_inner_threshold_plan.json", plan)
    distribution = {str(f): dict(Counter(int(r["y"]) for r in rows if r["outer_fold"] == f)) for f in range(folds)}
    empty_class_folds = [fold for fold, counts in distribution.items() if int(counts.get(0, 0)) == 0 or int(counts.get(1, 0)) == 0]
    if empty_class_folds:
        from .contracts import MissingLocalData
        raise MissingLocalData(f"group split leaves an outer fold without both classes: {empty_class_folds}")
    audit = {"eligible_rows": len(rows), "source_groups": len(groups), "fold_distribution": distribution, "source_leakage": False, "unknown_labels_used": 0, "review_labels_used": 0}
    write_json(out / "split_audit.json", audit)
    return audit
