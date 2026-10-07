#!/usr/bin/env python3
"""Zero-API decomposition of frozen Event-State V3 crowd results."""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import iter_jsonl, write_json, write_jsonl
from selection import label_codes, source_group_id
from validate_graph_candidates import _metrics


ROUTES = (
    "STATE_RECOGNITION_ERROR",
    "SCORE_ATTENUATION_ERROR",
    "NORMAL_COMPETITOR_OVERCONFIDENCE",
    "OBSERVATION_INADEQUATE",
    "THRESHOLD_ONLY_FLIP",
    "TRUE_DIRECT_HELP",
    "TRUE_DIRECT_HARM",
    "NO_EFFECT",
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _inside(frame: int, intervals: Iterable[Any]) -> bool:
    for interval in intervals:
        if isinstance(interval, (list, tuple)) and len(interval) >= 2:
            if int(interval[0]) <= int(frame) <= int(interval[1]):
                return True
    return False


def observation_audit(row: Mapping[str, Any]) -> dict:
    gt = row.get("gt", {}) if isinstance(row.get("gt"), Mapping) else {}
    evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), Mapping) else {}
    intervals = gt.get("intervals", []) if isinstance(gt.get("intervals"), list) else []
    indices = [int(value) for value in evidence.get("frame_indices", []) if isinstance(value, (int, float))]
    hits = [value for value in indices if _inside(value, intervals)]
    boundaries = sorted({int(value) for pair in intervals if isinstance(pair, (list, tuple)) for value in pair[:2]})
    bracketed = [
        boundary for boundary in boundaries
        if any(left < boundary < right for left, right in zip(indices, indices[1:]))
    ]
    return {
        "sampled_frame_count": len(indices),
        "sampled_anomaly_frame_count": len(hits),
        "sampled_anomaly_frame_indices": hits,
        "sampled_anomaly_fraction": len(hits) / len(indices) if indices else 0.0,
        "any_sampled_frame_in_gt": bool(hits),
        "bracketed_gt_boundaries": bracketed,
        "adjacent_pair_brackets_gt_boundary": bool(bracketed),
        "absolute_source_frame_indices": bool(evidence.get("absolute_source_frame_indices", False)),
    }


def route(row: Mapping[str, Any], state_high: float = 0.55) -> tuple[str, list[str], str]:
    truth = int(row.get("y_true", 0))
    base = row.get("base_competitions", {}).get("conditional_ot_full", {})
    candidate = row.get("candidate_competitions", {}).get("conditional_ot_full", {})
    base_pred, candidate_pred = int(base.get("y_pred", 0)), int(candidate.get("y_pred", 0))
    direct_effect = (
        "TRUE_DIRECT_HELP" if base_pred != truth and candidate_pred == truth else
        "TRUE_DIRECT_HARM" if base_pred == truth and candidate_pred != truth else
        "NO_EFFECT"
    )
    event = row.get("candidate_event_state", {})
    states = event.get("state_probabilities", {}) if isinstance(event.get("state_probabilities"), Mapping) else {}
    score = event.get("score", {}) if isinstance(event.get("score"), Mapping) else {}
    p_active = _float(states.get("active_escalation"))
    p_aftermath = _float(states.get("causally_linked_aftermath"))
    p_benign = _float(states.get("benign_or_pre_event_context"))
    q_state = p_active + p_aftermath
    base_target = _float(row.get("base_target", {}).get("results", {}).get("conditional_ot_full", {}).get("graph_score"))
    candidate_target = _float(score.get("graph_score"))
    normal_score = _float(candidate.get("best_normal_score"))
    margin = _float(candidate.get("margin"))
    threshold = _float(candidate.get("decision_margin_threshold"), 0.03)
    obs = observation_audit(row)
    visible_mechanism = bool(
        str(event.get("observed_transition_evidence", "")).strip()
        or str(event.get("aftermath_causal_evidence", "")).strip()
    )
    evidence: list[str] = []

    if truth == 0 and q_state >= state_high:
        evidence.append(f"normal window has high positive-state mass q={q_state:.4f}")
        return "STATE_RECOGNITION_ERROR", evidence, direct_effect
    if truth == 1 and not obs["any_sampled_frame_in_gt"]:
        evidence.append("none of the sampled frames intersects a GT anomaly interval")
        return "OBSERVATION_INADEQUATE", evidence, direct_effect
    if truth == 1 and q_state < state_high:
        if p_benign >= 0.5 and not visible_mechanism:
            evidence.append(f"benign posterior is high ({p_benign:.4f}) and no mechanism text was returned")
            return "OBSERVATION_INADEQUATE", evidence, direct_effect
        evidence.append(f"positive window has low positive-state mass q={q_state:.4f}")
        return "STATE_RECOGNITION_ERROR", evidence, direct_effect
    if truth == 1 and q_state >= state_high and candidate_target + 1e-9 < base_target:
        evidence.append(
            f"state is positive (q={q_state:.4f}) but target attenuates "
            f"from {base_target:.4f} to {candidate_target:.4f}"
        )
        return "SCORE_ATTENUATION_ERROR", evidence, direct_effect
    if truth == 1 and q_state >= state_high and candidate_pred == 0 and candidate_target >= base_target:
        evidence.append(
            f"target rises to {candidate_target:.4f}, but frozen normal aggregate {normal_score:.4f} still dominates"
        )
        return "NORMAL_COMPETITOR_OVERCONFIDENCE", evidence, direct_effect
    if base_pred != candidate_pred and 0.0 < margin <= threshold:
        evidence.append(f"candidate margin {margin:.4f} is positive but below threshold {threshold:.4f}")
        return "THRESHOLD_ONLY_FLIP", evidence, direct_effect
    return direct_effect, evidence, direct_effect


def flatten(row: Mapping[str, Any]) -> dict:
    event = row.get("candidate_event_state", {})
    states = event.get("state_probabilities", {}) if isinstance(event.get("state_probabilities"), Mapping) else {}
    score = event.get("score", {}) if isinstance(event.get("score"), Mapping) else {}
    base = row.get("base_competitions", {}).get("conditional_ot_full", {})
    candidate = row.get("candidate_competitions", {}).get("conditional_ot_full", {})
    failure_route, failure_evidence, direct_effect = route(row)
    obs = observation_audit(row)
    value = {
        "segment_key": str(row.get("segment_key", "")),
        "video_id": str(row.get("video_id", "")),
        "source_group": source_group_id(str(row.get("video_id", ""))),
        "y_true": int(row.get("y_true", 0)),
        "class_codes": sorted(label_codes(str(row.get("video_id", "")))),
        "event_phase": str(row.get("gt", {}).get("temporal_subset", "")),
        "base_pred": int(base.get("y_pred", 0)),
        "candidate_pred": int(candidate.get("y_pred", 0)),
        "base_margin": _float(base.get("margin")),
        "candidate_margin": _float(candidate.get("margin")),
        "base_target_score": _float(row.get("base_target", {}).get("results", {}).get("conditional_ot_full", {}).get("graph_score")),
        "normal_competitor_score": _float(candidate.get("best_normal_score")),
        "p_active": _float(states.get("active_escalation")),
        "p_aftermath": _float(states.get("causally_linked_aftermath")),
        "p_benign": _float(states.get("benign_or_pre_event_context")),
        "p_none": _float(states.get("none_or_unobservable")),
        "positive_state_mass": _float(states.get("active_escalation")) + _float(states.get("causally_linked_aftermath")),
        "transition_probability": _float(event.get("transition_observed_probability")),
        "causal_link_probability": _float(event.get("aftermath_causal_link_probability")),
        "same_episode_probability": _float(event.get("same_episode_probability")),
        "active_unary_score": _float(score.get("active_unary_ot_score")),
        "aftermath_unary_score": _float(score.get("aftermath_unary_ot_score")),
        "v3_target_score": _float(score.get("graph_score")),
        "failure_route": failure_route,
        "direct_effect": direct_effect,
        "failure_evidence": failure_evidence,
        **obs,
    }
    return value


def _state_metrics(rows: list[dict]) -> dict:
    values = []
    for row in rows:
        score = float(row["positive_state_mass"])
        values.append((int(row["y_true"]), int(score >= 0.5), score, str(row["video_id"])))
    return _metrics(values)


def _coverage_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = "sampled_gt_hit" if row["any_sampled_frame_in_gt"] else "no_sampled_gt_hit"
        grouped[key].append(row)
    output = {}
    for key, values in grouped.items():
        tuples = [
            (int(row["y_true"]), int(row["candidate_pred"]), float(row["candidate_margin"]), str(row["video_id"]))
            for row in values
        ]
        output[key] = _metrics(tuples)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    serial = []
    for row in rows:
        value = dict(row)
        for key in ("class_codes", "failure_evidence", "sampled_anomaly_frame_indices", "bracketed_gt_boundaries"):
            value[key] = json.dumps(value.get(key, []), ensure_ascii=False)
        serial.append(value)
    fields = list(serial[0]) if serial else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serial)


def _write_html(path: Path, rows: list[dict], summary: Mapping[str, Any]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<tr>"
            f"<td><code>{html.escape(row['segment_key'])}</code></td>"
            f"<td>{row['y_true']}</td><td>{row['base_pred']} -&gt; {row['candidate_pred']}</td>"
            f"<td>{html.escape(row['failure_route'])}</td>"
            f"<td>{row['positive_state_mass']:.3f}</td><td>{row['p_benign']:.3f}</td>"
            f"<td>{row['base_target_score']:.3f} -&gt; {row['v3_target_score']:.3f}</td>"
            f"<td>{row['sampled_anomaly_frame_count']}/{row['sampled_frame_count']}</td>"
            f"<td>{html.escape('; '.join(row['failure_evidence']))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Crowd V3 failure audit</title>
<style>body{{font:14px Arial;margin:24px;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dce2;padding:7px;vertical-align:top}}th{{background:#f3f5f7;position:sticky;top:0}}code{{font-size:12px}}.summary{{white-space:pre-wrap;background:#f7f8fa;padding:12px}}</style></head><body>
<h1>Crowd Event-State V3 failure audit</h1><div class="summary">{html.escape(json.dumps(summary, indent=2))}</div>
<table><thead><tr><th>Segment</th><th>GT</th><th>Base -&gt; V3</th><th>Route</th><th>q state</th><th>p benign</th><th>Target</th><th>GT sampled</th><th>Evidence</th></tr></thead><tbody>{''.join(cards)}</tbody></table></body></html>"""
    path.write_text(document, encoding="utf-8")


def analyze(results_path: Path, out_dir: Path) -> dict:
    rows = [flatten(row) for row in iter_jsonl(results_path)]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["failure_route"] for row in rows)
    effects = Counter(row["direct_effect"] for row in rows)
    summary = {
        "version": "crowd_event_state_v3_failure_decomposition_v1",
        "analysis_scope": "development_only_zero_api",
        "n": len(rows),
        "failure_routes": {route: counts.get(route, 0) for route in ROUTES},
        "direct_effects": dict(sorted(effects.items())),
        "state_posterior_metrics": _state_metrics(rows),
        "observation_coverage": {
            "sampled_gt_hit": sum(bool(row["any_sampled_frame_in_gt"]) for row in rows if row["y_true"] == 1),
            "positive_windows": sum(row["y_true"] == 1 for row in rows),
            "metrics_by_coverage": _coverage_metrics(rows),
        },
    }
    write_jsonl(out_dir / "failure_modes.jsonl", rows)
    _write_csv(out_dir / "failure_modes.csv", rows)
    _write_csv(out_dir / "sampled_gt_coverage.csv", rows)
    write_json(out_dir / "failure_mode_summary.json", summary)
    write_json(out_dir / "state_posterior_metrics.json", summary["state_posterior_metrics"])
    write_json(out_dir / "metrics_by_observation_coverage.json", summary["observation_coverage"])
    _write_html(out_dir / "case_audit.html", rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
