from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters import load_baseline_snapshot, load_source_records
from .contracts import LABEL_RULE_ID, canonical_window, file_sha256, write_json, write_jsonl


def _load_gather_anchors(pipeline_tools: Path):
    module_path = Path(pipeline_tools) / "train_from_selected_segments.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    sys.path.insert(0, str(pipeline_tools))
    try:
        spec = importlib.util.spec_from_file_location("v9_train_selected_adapter", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.gather_anchors
    finally:
        if sys.path and sys.path[0] == str(pipeline_tools):
            sys.path.pop(0)


def _anchor_provenance(folder: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    if not folder.is_dir():
        return result
    for path in sorted(folder.glob("*/selected_frames.json")):
        stem = path.parent.name[:-4] if path.parent.name.endswith(".mp4") else path.parent.name
        result.setdefault(stem, []).append({"path": str(path), "sha256": file_sha256(path)})
    return result


def _intersections(spans: list[tuple[int, int, float]], start: int, end: int) -> list[tuple[int, int]]:
    # Input spans are inclusive; ledger spans are half-open and clipped to the window.
    out = []
    for a, b, _ in spans:
        lo, hi = max(start, int(a)), min(end, int(b) + 1)
        if hi > lo:
            out.append((lo, hi))
    return sorted(set(out))


def build_label_ledger(
    legacy_run: Path,
    work_root: Path,
    select_root: Path,
    pipeline_tools: Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows, _ = load_baseline_snapshot(legacy_run)
    source_records = load_source_records(legacy_run)
    positive_folder = Path(select_root) / "top_anomalous_frames_72B_positive_segments"
    negative_folder = Path(select_root) / "top_anomalous_frames_72B_negative_segments"
    anchor_error = None
    try:
        gather = _load_gather_anchors(Path(pipeline_tools))
        positive = gather(positive_folder)
        negative = gather(negative_folder)
        spans_available = True
    except Exception as exc:
        positive, negative = {}, {}
        spans_available = False
        anchor_error = f"{type(exc).__name__}: {exc}"
    provenance = {"positive": _anchor_provenance(positive_folder), "negative": _anchor_provenance(negative_folder)}
    width = int(policy.get("window_frames", 96))
    label_rows: list[dict[str, Any]] = []
    frame_labels = np.full((len(baseline_rows), width), -1, dtype=np.int8)
    frame_observed = np.zeros((len(baseline_rows), width), dtype=bool)
    frame_valid = np.zeros((len(baseline_rows), width), dtype=bool)
    window_y = np.full(len(baseline_rows), -1, dtype=np.int8)
    window_mask = np.zeros(len(baseline_rows), dtype=bool)
    reasons: Counter[str] = Counter()

    for index, baseline in enumerate(baseline_rows):
        window = canonical_window(baseline)
        source = source_records.get(window.uid, {})
        video_id = window.video_id
        start, end = window.start_frame, window.end_frame_exclusive
        valid_count = min(width, end - start)
        frame_valid[index, :valid_count] = True
        pos = _intersections(positive.get(video_id, []), start, end) if spans_available else []
        neg = _intersections(negative.get(video_id, []), start, end) if spans_available else []
        local_pos: set[int] = set()
        local_neg: set[int] = set()
        for lo, hi in pos:
            local_pos.update(range(lo - start, min(hi - start, width)))
        for lo, hi in neg:
            local_neg.update(range(lo - start, min(hi - start, width)))
        conflict = local_pos & local_neg
        explicit_normal = bool(source.get("known_normal_from_video_label")) and source.get("label_source") == "explicit_filename_label_A"
        if explicit_normal:
            frame_labels[index, :valid_count] = 0
            frame_observed[index, :valid_count] = True
        else:
            for offset in local_neg - conflict:
                frame_labels[index, offset] = 0
                frame_observed[index, offset] = True
            for offset in local_pos - conflict:
                frame_labels[index, offset] = 1
                frame_observed[index, offset] = True

        target: int | None = None
        loss_mask = False
        if conflict:
            reason = "conflicting_anchor_labels"
            evidence = "conflicting_selected_intervals"
        elif explicit_normal:
            target, loss_mask = 0, True
            reason = "explicit_label_A_known_normal"
            evidence = "explicit_label_A"
        elif source.get("label_source") == "contains_positive_anchor":
            if not spans_available:
                reason, evidence = "positive_anchor_spans_unavailable", "unverified_positive_scope"
            elif not bool(policy.get("require_positive_anchor_semantics_declared", True)):
                reason, evidence = "positive_anchor_semantics_not_declared", "unverified_positive_scope"
            elif max((hi - lo for lo, hi in pos), default=0) >= 8:
                target, loss_mask = 1, True
                reason, evidence = "verified_positive_span_contiguous_overlap_ge8", "weak_positive_anchor"
            else:
                reason, evidence = "positive_span_below_target_duration", "unverified_positive_scope"
        elif source.get("label_source") == "contains_negative_anchor":
            reason, evidence = "partial_negative_anchor_does_not_cover_window", "weak_selected_interval"
        else:
            reason, evidence = "no_scope_aligned_supervision", "unknown"
        if target is not None:
            window_y[index] = target
        window_mask[index] = loss_mask
        reasons[reason] += 1
        covered = len((local_pos | local_neg) - conflict)
        label_rows.append({
            "version": "scope_label_ledger_v1",
            "window_uid": window.uid,
            "row_index": index,
            "window": {"dataset_partition": window.dataset_partition, "video_id": video_id, "start_frame": start, "end_frame_exclusive": end},
            "source_group": source.get("source_group") or baseline.get("source_group"),
            "historical_split": baseline.get("historical_split"),
            "label_rule_id": LABEL_RULE_ID,
            "legacy": {"y_true": baseline.get("y_true"), "label_source": source.get("label_source"), "gt_anom_fraction": source.get("gt_anom_fraction"), "not_frame_ground_truth": True},
            "scope": {
                "type": "interval", "coordinates": "zero_based_half_open", "positive_spans": pos, "negative_spans": neg,
                "conflicting_frame_offsets": sorted(conflict), "unknown_frame_count": valid_count - covered,
                "coverage_fraction": covered / valid_count if valid_count else 0.0, "source_spans_verified": spans_available,
                "source_provenance": {"positive": provenance["positive"].get(video_id, []), "negative": provenance["negative"].get(video_id, [])},
            },
            "supervision": {"window_target": target, "window_loss_mask": loss_mask, "reason": reason, "evidence_level": evidence, "anchor_confidence_is_probability": False},
            "human_review": {"status": "not_reviewed", "use_policy": "audit_only"},
        })

    labels_dir = Path(work_root) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(labels_dir / "label_ledger.jsonl", label_rows)
    np.save(labels_dir / "frame_labels.npy", frame_labels)
    np.save(labels_dir / "frame_observed.npy", frame_observed)
    np.save(labels_dir / "frame_valid.npy", frame_valid)
    np.save(labels_dir / "window_y.npy", window_y)
    np.save(labels_dir / "window_loss_mask.npy", window_mask)
    write_json(labels_dir / "label_policy.json", {
        "version": "event_label_policy_v1", "label_rule_id": LABEL_RULE_ID, "window_coordinates": "zero_based_half_open",
        "legacy_coordinates": "zero_based_inclusive", "positive_anchor_semantics": "selected 8-frame interval is treated as a weak anomalous unit",
        "negative_anchor_window_projection": "forbidden", "human_review_use": "audit_only", **dict(policy),
    })
    summary = {
        "version": "label_scope_summary_v1", "rows": len(label_rows), "supervised": int(window_mask.sum()),
        "positive": int(((window_y == 1) & window_mask).sum()), "known_normal": int(((window_y == 0) & window_mask).sum()),
        "partial_negative_windows_excluded": reasons["partial_negative_anchor_does_not_cover_window"],
        "conflicting_anchor_windows": reasons["conflicting_anchor_labels"], "anchor_spans_available": spans_available,
        "anchor_load_error": anchor_error, "interval_learning_available": False, "review_use": "audit_only", "reason_counts": dict(reasons),
        "hard_normal_validation_status": "INSUFFICIENT_HARD_NORMAL_VALIDATION",
    }
    write_json(labels_dir / "scope_summary.json", summary)
    write_json(Path(work_root) / "audit/label_scope_audit.json", summary)
    availability = []
    for split in ("calibration", "validation"):
        rows = [r for r in label_rows if r["historical_split"] == split]
        availability.append({"split": split, "rows": len(rows), "supervised": sum(bool(r["supervision"]["window_loss_mask"]) for r in rows), "positive": sum(r["supervision"]["window_target"] == 1 and r["supervision"]["window_loss_mask"] for r in rows), "normal": sum(r["supervision"]["window_target"] == 0 and r["supervision"]["window_loss_mask"] for r in rows)})
    import csv
    with (Path(work_root) / "audit/availability_by_stratum.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "rows", "supervised", "positive", "normal"]); writer.writeheader(); writer.writerows(availability)
    return summary

