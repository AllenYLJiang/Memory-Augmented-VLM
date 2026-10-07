#!/usr/bin/env python3
"""Create a no-API audit of V5 occupancy/confound tails in a calibration packet."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, write_json
from selection import is_pure_normal_video
from validate_graph_candidates import _metrics


METHOD = "conditional_ot_full"


def _unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _logit(value: float) -> float:
    eps = 1e-6
    return math.log((value + eps) / (1.0 - value + eps))


def _top_confound(event: Mapping[str, Any]) -> str:
    values = event.get("normal_confound_probabilities", {})
    if not isinstance(values, Mapping):
        return ""
    candidates = [(float(value), str(key)) for key, value in values.items() if key != "none"]
    return max(candidates, default=(0.0, ""))[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--negative-high", type=float, default=0.8)
    parser.add_argument("--positive-low", type=float, default=0.2)
    args = parser.parse_args()

    manifest = {str(row.get("segment_key", "")): row for row in iter_jsonl(args.manifest)}
    rows = []
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in iter_jsonl(args.records):
        if not bool(record.get("eligible_for_target_effect")):
            continue
        event = record.get("candidate_event_state", {})
        if not isinstance(event, Mapping) or not bool(event.get("complete")):
            continue
        key = str(record.get("segment_key", ""))
        meta = manifest.get(key, {})
        occupancy = _unit(event.get("current_window_active_occupancy_probability"))
        normal_confound = _unit(event.get("current_window_normal_confound_probability"))
        signed_score = _logit(occupancy) - max(0.0, _logit(normal_confound))
        base = record.get("base_competitions", {}).get(METHOD, {})
        row = {
            "segment_key": key,
            "video_id": str(record.get("video_id", "")),
            "source_group": str(meta.get("source_group", record.get("source_group", ""))),
            "stratum": str(meta.get("stratum", record.get("evaluation_stratum", ""))),
            "event_phase": str(meta.get("event_phase", record.get("event_phase", ""))),
            "y_true": int(record.get("y_true", 0)),
            "base_margin": float(base.get("margin", 0.0)),
            "base_y_pred": int(base.get("y_pred", 0)),
            "occupancy": occupancy,
            "normal_confound": normal_confound,
            "signed_diagnostic_score": signed_score,
            "top_normal_confound_type": _top_confound(event),
            "uncertainty": _unit(event.get("uncertainty", 1.0)),
            "is_explicit_label_A": is_pure_normal_video(str(record.get("video_id", ""))),
            "visible_active_evidence": str(event.get("visible_evidence", {}).get("active", "")),
            "normal_confound_evidence": json.dumps(
                event.get("normal_confound_evidence", {}), ensure_ascii=False, sort_keys=True,
            ),
            "evidence_images": ";".join(str(path) for path in record.get("evidence", {}).get("image_paths", [])),
        }
        rows.append(row)
        by_stratum[row["stratum"]].append(row)

    if not rows:
        raise SystemExit("no complete target-exposed V5 rows to audit")
    occupancy_values = [
        (row["y_true"], int(row["occupancy"] > 0.5), row["occupancy"], row["video_id"])
        for row in rows
    ]
    signed_values = [
        (row["y_true"], int(row["signed_diagnostic_score"] > 0.0), row["signed_diagnostic_score"], row["video_id"])
        for row in rows
    ]
    extreme_rows = [
        {**row, "audit_reason": "normal_high_occupancy"}
        for row in rows if row["y_true"] == 0 and row["occupancy"] >= args.negative_high
    ] + [
        {**row, "audit_reason": "abnormal_low_occupancy"}
        for row in rows if row["y_true"] == 1 and row["occupancy"] <= args.positive_low
    ]
    extreme_rows.sort(key=lambda row: (
        row["audit_reason"],
        -row["occupancy"] if row["y_true"] == 0 else row["occupancy"],
        row["segment_key"],
    ))

    stratum_summary = {}
    for stratum, selected in sorted(by_stratum.items()):
        stratum_summary[stratum] = {
            "n": len(selected),
            "positive": sum(row["y_true"] == 1 for row in selected),
            "negative": sum(row["y_true"] == 0 for row in selected),
            "explicit_label_A": sum(bool(row["is_explicit_label_A"]) for row in selected),
            "mean_occupancy": sum(row["occupancy"] for row in selected) / len(selected),
            "mean_normal_confound": sum(row["normal_confound"] for row in selected) / len(selected),
        }
    summary = {
        "version": "crowd_v5_signed_confound_extreme_audit_v1",
        "records": str(args.records),
        "manifest": str(args.manifest),
        "target_exposed_complete": len(rows),
        "explicit_label_A": sum(bool(row["is_explicit_label_A"]) for row in rows),
        "thresholds": {"negative_high": args.negative_high, "positive_low": args.positive_low},
        "extreme_counts": dict(Counter(row["audit_reason"] for row in extreme_rows)),
        "occupancy_metrics": _metrics(occupancy_values),
        "signed_diagnostic_metrics": _metrics(signed_values),
        "strata": stratum_summary,
        "note": (
            "signed_diagnostic_score is a no-fit diagnostic only; deployment still requires the "
            "source-group cross-validated calibration and all FP gates"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "occupancy_extreme_audit_summary.json", summary)
    csv_path = args.out_dir / "occupancy_extreme_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(extreme_rows[0]) if extreme_rows else ["audit_reason"])
        writer.writeheader()
        writer.writerows(extreme_rows)
    review = [
        "# Crowd V5 Occupancy/Confound Extreme Audit",
        "",
        f"- Complete target-exposed rows: {len(rows)}",
        f"- Explicit label-A rows: {summary['explicit_label_A']}",
        f"- High-occupancy normals: {summary['extreme_counts'].get('normal_high_occupancy', 0)}",
        f"- Low-occupancy abnormals: {summary['extreme_counts'].get('abnormal_low_occupancy', 0)}",
        f"- Occupancy AP: {summary['occupancy_metrics'].get('ap')}",
        f"- Signed diagnostic AP: {summary['signed_diagnostic_metrics'].get('ap')}",
        "",
        "Review every listed case before treating a weak anchor disagreement as a representation error.",
        "The signed diagnostic is not a deployable result; Step 5 calibration remains authoritative.",
    ]
    (args.out_dir / "OCCUPANCY_EXTREME_AUDIT.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
