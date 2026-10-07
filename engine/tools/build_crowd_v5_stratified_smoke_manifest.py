#!/usr/bin/env python3
"""Build a small V5 smoke packet that covers positives and normal tails."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl, write_json, write_jsonl


def _baseline_records(path: Path) -> dict[str, dict]:
    source = path / "ot_window_results.jsonl" if path.is_dir() else path
    return {str(row.get("segment_key", "")): row for row in iter_jsonl(source)}


def _target_exposed(row: dict | None, target_key: str) -> bool:
    if not row:
        return False
    shortlist = row.get("graph_candidates", {})
    selected = set(shortlist.get("selected_abnormal", [])) | set(shortlist.get("selected_normal", []))
    return target_key in selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--target-key", default="crowd_escalation_chain")
    parser.add_argument("--max-windows", type=int, default=14)
    parser.add_argument(
        "--require-target-exposed-strata",
        nargs="*",
        default=("hard_context_normal_anchor", "post_event_normal_anchor", "pure_normal_anchor"),
    )
    args = parser.parse_args()
    if args.max_windows <= 0:
        raise SystemExit("--max-windows must be positive")

    baseline = _baseline_records(args.baseline_records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for meta in iter_jsonl(args.manifest):
        key = str(meta.get("segment_key", ""))
        if key not in baseline:
            continue
        stratum = str(meta.get("stratum", meta.get("crowd_v4_stratum", "unstratified")))
        enriched = dict(meta)
        enriched["_smoke_target_exposed"] = _target_exposed(baseline.get(key), args.target_key)
        grouped[stratum].append(enriched)

    strata = sorted(grouped)
    if args.max_windows < len(strata):
        raise SystemExit(f"--max-windows={args.max_windows} cannot cover {len(strata)} strata")
    for rows in grouped.values():
        rows.sort(key=lambda row: (not row["_smoke_target_exposed"], str(row.get("segment_key", ""))))

    selected = []
    depth = 0
    while len(selected) < args.max_windows:
        added = False
        for stratum in strata:
            rows = grouped[stratum]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == args.max_windows:
                    break
        if not added:
            break
        depth += 1

    selected_counts = Counter()
    exposed_counts = Counter()
    clean_rows = []
    for row in selected:
        stratum = str(row.get("stratum", row.get("crowd_v4_stratum", "unstratified")))
        selected_counts[stratum] += 1
        if row.pop("_smoke_target_exposed", False):
            exposed_counts[stratum] += 1
        clean_rows.append(row)

    missing_required = [
        stratum for stratum in args.require_target_exposed_strata if exposed_counts.get(stratum, 0) < 1
    ]
    summary = {
        "version": "crowd_v5_stratified_smoke_manifest_v1",
        "source_manifest": str(args.manifest),
        "baseline_records": str(args.baseline_records),
        "target_key": args.target_key,
        "requested_windows": args.max_windows,
        "selected_windows": len(clean_rows),
        "selected_by_stratum": dict(sorted(selected_counts.items())),
        "target_exposed_by_stratum": dict(sorted(exposed_counts.items())),
        "required_target_exposed_strata": list(args.require_target_exposed_strata),
        "missing_required_target_exposed_strata": missing_required,
        "ready_for_smoke": bool(clean_rows) and not missing_required,
    }
    write_jsonl(args.out, clean_rows)
    write_json(args.summary_out or args.out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ready_for_smoke"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
