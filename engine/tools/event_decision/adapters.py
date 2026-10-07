from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import canonical_window, iter_jsonl, source_group


BASELINE_RELATIVE = {
    "calibration": "baseline/calibration/ot_window_results.jsonl",
    "validation": "baseline/validation/ot_window_results.jsonl",
}
STATE_RELATIVE = {
    "calibration": "v5_calibration_collect/frozen_pair_results.jsonl",
    "validation": "validation_r1/frozen_pair_results.jsonl",
}


def keyed_rows(path: Path, *, default_partition: str = "train") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = canonical_window(row, default_partition).uid
        existing = out.get(key)
        if existing is not None and existing != row:
            raise ValueError(f"conflicting duplicate canonical window in {path}: {row.get('segment_key')}")
        out[key] = row
    return out


def load_baseline_snapshot(legacy_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for split, relative in BASELINE_RELATIVE.items():
        for record in iter_jsonl(Path(legacy_run) / relative):
            window = canonical_window(record)
            row = dict(record)
            row["canonical_window_uid"] = window.uid
            row["historical_split"] = split
            row["source_group"] = source_group(row)
            if window.uid in seen:
                conflicts.append({
                    "window_uid": window.uid,
                    "first_split": seen[window.uid]["historical_split"],
                    "second_split": split,
                    "segment_key": row.get("segment_key"),
                })
                continue
            seen[window.uid] = row
            rows.append(row)
    return rows, conflicts


def load_state_snapshot(legacy_run: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for split, relative in STATE_RELATIVE.items():
        for record in iter_jsonl(Path(legacy_run) / relative):
            uid = canonical_window(record).uid
            state = record.get("candidate_event_state")
            if state is None:
                continue
            wrapped = {"record": record, "state": state, "historical_split": split}
            if uid in rows:
                conflicts.append({"window_uid": uid, "reason": "duplicate_state_trace", "splits": [rows[uid]["historical_split"], split]})
                continue
            rows[uid] = wrapped
    return rows, conflicts


def load_source_records(legacy_run: Path) -> dict[str, dict[str, Any]]:
    path = Path(legacy_run) / "source_disjoint_training_anchors/selected_source_records.jsonl"
    return keyed_rows(path)


def count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "<missing>")) for row in rows))


def pointer(record: Mapping[str, Any], *parts: str) -> Any:
    current: Any = record
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current

