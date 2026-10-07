from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .adapters import load_baseline_snapshot, load_source_records, load_state_snapshot
from .contracts import iter_jsonl, source_group, write_json, write_jsonl


METHODS = (
    "independent_direct_nodes",
    "shared_unary_rowmax",
    "unary_ot",
    "conditional_rowmax",
    "conditional_ot_no_coherence",
    "conditional_ot_full",
)


def tie_aware_ap(y: Sequence[int], score: Sequence[float]) -> float | None:
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(score, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int((labels == 1).sum())
    if positives == 0 or int((labels == 0).sum()) == 0:
        return None
    total_seen = 0
    positives_seen = 0
    area = 0.0
    for threshold in sorted(set(scores.tolist()), reverse=True):
        group = labels[scores == threshold]
        total_seen += len(group)
        positives_seen += int((group == 1).sum())
        recall_increment = int((group == 1).sum()) / positives
        area += recall_increment * (positives_seen / total_seen)
    return float(area)


def tie_aware_auc(y: Sequence[int], score: Sequence[float]) -> float | None:
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(score, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comparisons / (len(pos) * len(neg)))


def binary_metrics(y: Sequence[int], score: Sequence[float], threshold: float = 0.0) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(score, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    pred = (scores >= threshold).astype(np.int8)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    return {
        "n": int(len(labels)), "positive_n": int((labels == 1).sum()), "negative_n": int((labels == 0).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels) if len(labels) else None,
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "recall": recall, "specificity": specificity, "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if len(labels) else None,
        "ap": tie_aware_ap(labels, scores), "auc": tie_aware_auc(labels, scores), "threshold": threshold,
    }


def metrics_with_predictions(y: Sequence[int], score: Sequence[float], prediction: Sequence[int]) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(score, dtype=float)
    pred = np.asarray(prediction, dtype=np.int8)
    valid = np.isfinite(scores)
    labels, scores, pred = labels[valid], scores[valid], pred[valid]
    tp = int(((pred == 1) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else None; specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    return {"n": len(labels), "positive_n": int((labels == 1).sum()), "negative_n": int((labels == 0).sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(labels) if len(labels) else None, "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None, "recall": recall, "specificity": specificity, "precision": precision, "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if len(labels) else None, "ap": tie_aware_ap(labels, scores), "auc": tie_aware_auc(labels, scores), "threshold": "per_outer_fold_frozen"}


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]], base_key: str, candidate_key: str, repetitions: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_group"])].append(row)
    names = sorted(groups)
    if len(names) < 2:
        return {"status": "insufficient_groups", "source_groups": len(names), "samples": 0}
    rng = np.random.default_rng(seed)
    ap_deltas, ba_deltas = [], []
    for _ in range(repetitions):
        sampled = rng.choice(names, size=len(names), replace=True)
        sample = [r for name in sampled for r in groups[str(name)]]
        y = [int(r["y"]) for r in sample]
        if all("base_pred" in r and "candidate_pred" in r for r in sample):
            base = metrics_with_predictions(y, [float(r[base_key]) for r in sample], [int(r["base_pred"]) for r in sample])
            cand = metrics_with_predictions(y, [float(r[candidate_key]) for r in sample], [int(r["candidate_pred"]) for r in sample])
        else:
            base = binary_metrics(y, [float(r[base_key]) for r in sample], .5)
            cand = binary_metrics(y, [float(r[candidate_key]) for r in sample], .5)
        if base["ap"] is not None and cand["ap"] is not None:
            ap_deltas.append(cand["ap"] - base["ap"])
        if base["balanced_accuracy"] is not None and cand["balanced_accuracy"] is not None:
            ba_deltas.append(cand["balanced_accuracy"] - base["balanced_accuracy"])
    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "ci95": [None, None], "p_positive": None}
        arr = np.asarray(values)
        return {"n": len(values), "mean": float(arr.mean()), "ci95": [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))], "p_positive": float((arr > 0).mean())}
    return {"status": "ok", "source_groups": len(names), "samples": repetitions, "ap_delta": summary(ap_deltas), "balanced_accuracy_delta": summary(ba_deltas)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_snapshot(legacy_run: Path, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows, baseline_conflicts = load_baseline_snapshot(legacy_run)
    state_rows, state_conflicts = load_state_snapshot(legacy_run)
    write_jsonl(out_dir / "lineage_conflicts.jsonl", baseline_conflicts + state_conflicts)
    metric_rows = []
    legacy_rows = []
    for split in ("calibration", "validation"):
        legacy_path = Path(legacy_run) / f"baseline/{split}/method_metrics.csv"
        if legacy_path.is_file():
            with legacy_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("subset") == "all":
                        legacy_rows.append({"split": split, **row, "reported_source": str(legacy_path)})
    for split in ("calibration", "validation"):
        subset = [r for r in baseline_rows if r["historical_split"] == split and r.get("y_true") in (0, 1)]
        for method in METHODS:
            usable = [r for r in subset if isinstance(r.get("competitions", {}).get(method), Mapping)]
            y = [int(r["y_true"]) for r in usable]
            score = [float(r["competitions"][method].get("margin", math.nan)) for r in usable]
            metric = binary_metrics(y, score, threshold=0.03)
            metric_rows.append({"split": split, "method": method, **metric})
    _write_csv(out_dir / "baseline_methods_tie_aware.csv", metric_rows)
    _write_csv(out_dir / "baseline_methods_legacy_reported.csv", legacy_rows)

    parity = []
    for uid, wrapped in state_rows.items():
        record = wrapped["record"]
        baseline = next((r for r in baseline_rows if r["canonical_window_uid"] == uid), None)
        if not baseline:
            continue
        for method in ("conditional_rowmax", "conditional_ot_full"):
            a = baseline.get("competitions", {}).get(method, {}).get("margin")
            b = record.get("base_competitions", {}).get(method, {}).get("margin")
            if a is not None and b is not None:
                parity.append({"window_uid": uid, "split": wrapped["historical_split"], "method": method, "source_margin": float(a), "paired_base_margin": float(b), "abs_delta": abs(float(a) - float(b))})
    write_json(out_dir / "step6_to_step7_base_parity.json", {
        "n": len(parity), "exact_within_1e_12": sum(r["abs_delta"] <= 1e-12 for r in parity),
        "max_abs_delta": max((r["abs_delta"] for r in parity), default=None), "rows": parity[:100],
    })

    calibration = __import__("json").loads((Path(legacy_run) / "calibration/crowd_event_state_v5_calibration.json").read_text(encoding="utf-8"))
    replay_rows = []
    eps = 1e-6
    for uid, wrapped in state_rows.items():
        record, state = wrapped["record"], wrapped["state"]
        complete = bool(state.get("complete"))
        occupancy = min(1.0, max(0.0, float(state.get("current_window_active_occupancy_probability", 0.0) or 0.0)))
        normal = min(1.0, max(0.0, float(state.get("current_window_normal_confound_probability", 0.0) or 0.0)))
        aftermath = min(1.0, max(0.0, float(state.get("aftermath_context_probability", 0.0) or 0.0)))
        support = min(1.0, max(0.0, float(state.get("aftermath_current_window_occupancy_support_probability", 0.0) or 0.0)))
        uncertainty = min(1.0, max(0.0, float(state.get("uncertainty", 1.0) or 0.0)))
        context = state.get("event_context_state_probabilities", {})
        none_probability = min(1.0, max(0.0, float(context.get("none", 0.0) or 0.0))) if isinstance(context, Mapping) else 0.0
        active_logit = math.log((occupancy + eps) / (1.0 - occupancy + eps))
        normal_logit = max(0.0, math.log((normal + eps) / (1.0 - normal + eps)))
        raw = float(calibration.get("beta_active", 0.0)) * active_logit + float(calibration.get("beta_aftermath", 0.0)) * aftermath * support - float(calibration.get("beta_normal_confound", 0.0)) * normal_logit
        bound = float(calibration.get("residual_bound", 0.0))
        # v5_calibration_collect is the pre-fit trace collection and is expected
        # to be a no-op. The frozen coefficients apply only to validation_r1.
        blocked = wrapped["historical_split"] == "calibration" or (not complete) or not bool(record.get("eligible_for_target_effect")) or uncertainty > float(calibration.get("uncertainty_max", 1.0)) or none_probability > float(calibration.get("none_max", 1.0)) or bound <= 0
        residual = 0.0 if blocked else bound * math.tanh(raw / bound)
        for method in ("conditional_rowmax", "conditional_ot_no_coherence", "conditional_ot_full"):
            base_margin = record.get("base_competitions", {}).get(method, {}).get("margin")
            stored = record.get("candidate_competitions", {}).get(method, {}).get("margin")
            if base_margin is None or stored is None:
                continue
            expected = float(base_margin) + residual
            replay_rows.append({"window_uid": uid, "split": wrapped["historical_split"], "method": method, "expected_margin": expected, "stored_margin": float(stored), "abs_delta": abs(expected - float(stored)), "residual": residual})
    replay = {
        "calibration_id": calibration.get("calibration_id"),
        "frozen_coefficients": {key: calibration.get(key) for key in ("beta_active", "beta_aftermath", "beta_normal_confound", "residual_bound")},
        "rule": "reproduce_only_no_refit",
        "state_rows": len(state_rows),
        "method_rows": len(replay_rows),
        "exact_within_1e_12": sum(row["abs_delta"] <= 1e-12 for row in replay_rows),
        "max_abs_delta": max((row["abs_delta"] for row in replay_rows), default=None),
        "sample_mismatches": sorted((row for row in replay_rows if row["abs_delta"] > 1e-12), key=lambda row: row["abs_delta"], reverse=True)[:50],
    }
    write_json(out_dir / "step7_legacy_replay.json", replay)

    reachability = []
    for name in ("m0_margin", "m3a_margin", "m3c_margin", "o_active", "q_direct", "state_uncertainty", "normal_bound_probability", "unexplained_direct_evidence"):
        if name in {"m0_margin", "m3a_margin", "m3c_margin"}:
            n = len(baseline_rows)
        elif name in {"normal_bound_probability", "unexplained_direct_evidence"}:
            n = 0
        else:
            n = len(state_rows)
        reachability.append({"feature": name, "available_rows": n, "total_baseline_rows": len(baseline_rows), "fraction": n / len(baseline_rows) if baseline_rows else 0.0})
    write_json(out_dir / "score_reachability.json", {"features": reachability, "binding_status": "BINDING_NOT_IDENTIFIABLE_FROM_CURRENT_CACHE"})
    write_json(out_dir / "semantic_head_diagnostics.json", {"state_rows": len(state_rows), "baseline_rows": len(baseline_rows), "state_coverage": len(state_rows) / len(baseline_rows) if baseline_rows else 0.0, "provider_unavailable_policy": "missing_not_zero"})

    tail_rows = []
    source_records = load_source_records(legacy_run)
    for split in ("calibration", "validation"):
        subset = [r for r in baseline_rows if r["historical_split"] == split]
        strata = {
            "whole_packet": subset,
            "pure_label_A": [r for r in subset if bool(r.get("gt", {}).get("known_normal_from_video_label"))],
            "legacy_partial_negative": [r for r in subset if source_records.get(r["canonical_window_uid"], {}).get("label_source") == "contains_negative_anchor"],
            "hard_context_normal_legacy": [r for r in subset if bool(source_records.get(r["canonical_window_uid"], {}).get("hard_context_normal"))],
            "post_event_legacy": [r for r in subset if source_records.get(r["canonical_window_uid"], {}).get("event_phase") == "aftermath"],
            "weak_positive": [r for r in subset if source_records.get(r["canonical_window_uid"], {}).get("label_source") == "contains_positive_anchor"],
        }
        for stratum, stratum_rows in strata.items():
            for method in METHODS:
                usable = [r for r in stratum_rows if r.get("y_true") in (0, 1) and r.get("competitions", {}).get(method, {}).get("margin") is not None]
                if usable:
                    y = [int(r["y_true"]) for r in usable]
                    scores = [float(r["competitions"][method]["margin"]) for r in usable]
                    tail_rows.append({"split": split, "stratum": stratum, "method": method, "scope_status": "legacy_diagnostic" if "legacy" in stratum else "mixed", **binary_metrics(y, scores, .03)})
    _write_csv(out_dir / "normal_tail_breakdown.csv", tail_rows)

    cal_groups = {r["source_group"] for r in baseline_rows if r["historical_split"] == "calibration"}
    val_groups = {r["source_group"] for r in baseline_rows if r["historical_split"] == "validation"}
    write_json(out_dir / "source_group_overlap.json", {"calibration_groups": len(cal_groups), "validation_groups": len(val_groups), "overlap": sorted(cal_groups & val_groups), "passed": not bool(cal_groups & val_groups)})
    summary = {"baseline_rows": len(baseline_rows), "state_rows": len(state_rows), "lineage_conflicts": len(baseline_conflicts) + len(state_conflicts), "methods": len(metric_rows), "remote_calls": 0}
    write_json(out_dir / "audit_summary.json", summary)
    return summary


audit_existing_snapshot = audit_snapshot
