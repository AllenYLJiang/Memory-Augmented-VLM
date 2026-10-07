#!/usr/bin/env python3
"""No-API replay of V4 phase weights plus permanent aftermath-negative memory."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from common import iter_jsonl, write_json, write_jsonl
from selection import source_group_id
from validate_graph_candidates import _delta, _metrics


METHOD = "conditional_ot_full"


def _grid(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def _values(rows: list[dict], phase_weight: float, beta: float, bound: float) -> list[tuple[int, int, float, str]]:
    output = []
    for row in rows:
        comp = row.get("base_competitions", {}).get(METHOD, {})
        event = row.get("candidate_event_state", {})
        states = event.get("state_probabilities", {})
        p_active = float(states.get("active_or_ongoing_physical_escalation", 0.0))
        p_after = float(states.get("causally_linked_aftermath", 0.0))
        positive = p_active + phase_weight * p_after
        negative = (
            float(states.get("pre_event_tension_or_flight", 0.0))
            + float(states.get("benign_collective_activity", 0.0))
            + float(states.get("none_or_unobservable", 0.0))
            + (1.0 - phase_weight) * p_after
        )
        noop = not bool(event.get("complete"))
        residual = 0.0 if noop else max(-bound, min(bound, beta * math.log((positive + 1e-6) / (negative + 1e-6))))
        margin = float(comp.get("margin", 0.0)) + residual
        threshold = float(comp.get("decision_margin_threshold", 0.03) or 0.03)
        output.append((int(row.get("y_true", 0)), int(margin > threshold), margin, str(row.get("video_id", ""))))
    return output


def _bootstrap(rows: list[dict], before, phase_weight: float, beta: float, bound: float, samples: int, seed: int) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[source_group_id(str(row.get("video_id", "")))].append(index)
    groups = sorted(grouped)
    rng = random.Random(seed)
    positive = valid = 0
    all_after = _values(rows, phase_weight, beta, bound)
    for _ in range(samples):
        indices = [index for _ in groups for index in grouped[rng.choice(groups)]]
        if not indices:
            continue
        delta = _delta(_metrics([all_after[index] for index in indices]), _metrics([before[index] for index in indices]))
        positive += int(float(delta.get("balanced_accuracy", 0.0)) > 0.0)
        valid += 1
    return positive / valid if valid else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--phase-weights", type=_grid, default=_grid("0,0.25,0.5,0.75,1"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    args = parser.parse_args()
    rows = [
        row for row in iter_jsonl(args.results)
        if bool(row.get("eligible_for_target_effect", True))
    ]
    if not rows:
        raise SystemExit("no target-exposed V4 rows")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    beta = float(calibration.get("beta", 0.0))
    bound = float(calibration.get("residual_bound", 0.0))
    before = [
        (int(row.get("y_true", 0)), int(row.get("base_competitions", {}).get(METHOD, {}).get("y_pred", 0)),
         float(row.get("base_competitions", {}).get(METHOD, {}).get("margin", 0.0)), str(row.get("video_id", "")))
        for row in rows
    ]
    base_metrics = _metrics(before)
    grid_rows = []
    for weight in args.phase_weights:
        after = _values(rows, weight, beta, bound)
        metrics = _metrics(after)
        delta = _delta(metrics, base_metrics)
        helps = sum(b[1] != b[0] and a[1] == a[0] for b, a in zip(before, after))
        hurts = sum(b[1] == b[0] and a[1] != a[0] for b, a in zip(before, after))
        grid_rows.append({
            "aftermath_weight": weight, "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"), "ap": metrics.get("ap"),
            "auc": metrics.get("auc"), "recall": metrics.get("recall"),
            "specificity": metrics.get("specificity"), "ap_delta": delta.get("ap"),
            "balanced_accuracy_delta": delta.get("balanced_accuracy"), "helps": helps, "hurts": hurts,
            "bootstrap_p_ba_delta_positive": _bootstrap(
                rows, before, weight, beta, bound, args.bootstrap_samples, args.bootstrap_seed,
            ),
        })
    current_after = _values(rows, 1.0, beta, bound)
    harmful = [
        row for row, base_value, candidate_value in zip(rows, before, current_after)
        if base_value[1] == base_value[0] and candidate_value[1] != candidate_value[0]
    ]
    memory = []
    for row in harmful:
        event = row.get("candidate_event_state", {})
        memory.append({
            "memory_type": "aftermath_context_not_current_anomaly",
            "segment_key": row.get("segment_key"), "video_id": row.get("video_id"),
            "start_frame": row.get("start_frame"), "end_frame": row.get("end_frame"),
            "y_true": row.get("y_true"), "gt": row.get("gt", {}),
            "state_probabilities": event.get("state_probabilities", {}),
            "visible_evidence": event.get("visible_evidence", {}),
            "lesson": "Aftermath context may identify an earlier episode but is not current-window active anomaly occupancy.",
            "use": "prompt regression and negative-tail audit only; never train or tune a final model on held-out labels",
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "crowd_v4_phase_weight_exploratory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader(); writer.writerows(grid_rows)
    write_jsonl(args.out_dir / "crowd_aftermath_negative_memory.jsonl", memory)
    write_json(args.out_dir / "crowd_v4_phase_weight_exploratory.json", {
        "status": "post_hoc_development_diagnostic_not_deployable", "records": str(args.results),
        "calibration": str(args.calibration), "beta": beta, "residual_bound": bound,
        "target_exposed_n": len(rows), "harmful_current_weight_n": len(harmful), "grid": grid_rows,
    })
    lines = [
        "# Crowd V4 Phase-Weight Exploratory Replay", "",
        "**Status:** post-hoc development diagnostic; these weights must not be deployed.", "",
        f"- Target-exposed windows: {len(rows)}", f"- V4 beta: {beta}", f"- V4 residual bound: {bound}",
        f"- Harmful current-weight cases stored: {len(memory)}", "", "## Replay", "",
        "| Aftermath weight | AP | BA | Recall | Specificity | Helps | Hurts | P(delta BA > 0) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {row['aftermath_weight']:.2f} | {row['ap']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['helps']} | {row['hurts']} | "
            f"{row['bootstrap_p_ba_delta_positive']:.3f} |"
        )
    lines.extend(["", "The replay motivates V5 occupancy separation; it does not select V5 parameters.", ""])
    (args.out_dir / "CROWD_V4_PHASE_WEIGHT_EXPLORATORY_20260831.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "harmful": len(memory), "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
