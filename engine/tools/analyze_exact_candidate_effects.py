#!/usr/bin/env python3
"""Paired, source-group-aware forensic analysis for one exact graph candidate."""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import iter_jsonl, read_json, write_json, write_jsonl
from selection import label_codes, source_group_id
from validate_graph_candidates import _competition_tuple, _delta, _metrics


METHOD = "conditional_ot_full"


def _records(run: Path) -> dict[str, dict]:
    path = run / "ot_window_results.jsonl"
    if path.is_file():
        return {str(row.get("segment_key")): row for row in iter_jsonl(path)}
    return {
        str(row.get("segment_key")): row
        for path in sorted((run / "records").glob("*.json"))
        if isinstance((row := read_json(path, None)), dict)
    }


def _metric(rows: Iterable[Mapping[str, Any]]) -> dict:
    return _metrics(_competition_tuple(row) for row in rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_exposed(row: Mapping[str, Any], key: str) -> bool:
    selected = row.get("graph_candidates", {})
    return key in selected.get("selected_abnormal", []) or key in selected.get("selected_normal", [])


def _complete(row: Mapping[str, Any], key: str) -> tuple[bool, bool]:
    independent = bool(row.get("completeness", {}).get("independent"))
    joint = bool(row.get("completeness", {}).get("joint"))
    if key:
        joint = joint and key in row.get("joint_graph_calls", {})
    return independent, joint


def _attribution(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> str:
    before_short = before.get("graph_candidates", {})
    after_short = after.get("graph_candidates", {})
    before_set = set(before_short.get("selected_abnormal", [])) | set(before_short.get("selected_normal", []))
    after_set = set(after_short.get("selected_abnormal", [])) | set(after_short.get("selected_normal", []))
    if before_set != after_set:
        return "shortlist_displacement"
    bi, bj = _complete(before, key)
    ai, aj = _complete(after, key)
    if (bi, bj) != (ai, aj):
        return "joint_parse_compliance_change"
    common_nodes = set(before.get("independent_node_calls", {})) & set(after.get("independent_node_calls", {}))
    if any(
        abs(float(before["independent_node_calls"][node].get("presence", 0.0))
            - float(after["independent_node_calls"][node].get("presence", 0.0))) > 1e-9
        for node in common_nodes
    ):
        return "shared_node_prompt_ripple"
    bg = before.get("graph_results", {}).get(METHOD, {}).get(key)
    ag = after.get("graph_results", {}).get(METHOD, {}).get(key)
    if isinstance(bg, Mapping) and isinstance(ag, Mapping):
        if abs(float(bg.get("graph_score", 0.0)) - float(ag.get("graph_score", 0.0))) > 1e-9:
            return "direct_crowd_graph_score_change"
    bm = before.get("competitions", {}).get(METHOD, {})
    am = after.get("competitions", {}).get(METHOD, {})
    if bm.get("y_pred") != am.get("y_pred"):
        return "threshold_or_calibration_flip"
    if before_set & after_set:
        return "counterfactual_completion_change"
    return "remote_vlm_variance_or_unresolved"


def _state(row: Mapping[str, Any], key: str) -> dict:
    competition = row.get("competitions", {}).get(METHOD, {})
    graph = row.get("graph_results", {}).get(METHOD, {}).get(key, {})
    trace = row.get("joint_graph_calls", {}).get(key, {})
    nodes = list((graph.get("node_presence") or {}).values())
    return {
        "shortlist": row.get("graph_candidates", {}),
        "best_abnormal_graph": competition.get("best_abnormal_graph"),
        "best_normal_graph": competition.get("best_normal_graph"),
        "abnormal_score": competition.get("best_abnormal_score"),
        "normal_score": competition.get("best_normal_score"),
        "margin": competition.get("margin"), "decision": competition.get("decision"),
        "joint_parse_complete": bool(row.get("completeness", {}).get("joint")),
        "crowd_graph_selected": _target_exposed(row, key),
        "crowd_graph_score": graph.get("graph_score"),
        "crowd_null_mass": (1.0 - sum(float(value) for value in nodes) / len(nodes)) if nodes else None,
        "crowd_uncertainty": (
            sum(float(value.get("uncertainty", 0.0)) for value in trace.get("nodes", {}).values())
            / len(trace.get("nodes", {})) if trace.get("nodes") else None
        ),
    }


def _subsets(rows: list[tuple[dict, dict]], target_codes: set[str], graph_key: str) -> dict[str, list[tuple[dict, dict]]]:
    result = {"all_matched": rows}
    result["common_independent_parse"] = [pair for pair in rows if _complete(pair[0], graph_key)[0] and _complete(pair[1], graph_key)[0]]
    result["common_joint_parse"] = [pair for pair in rows if _complete(pair[0], graph_key)[1] and _complete(pair[1], graph_key)[1]]
    result["common_shortlist"] = [
        pair for pair in rows
        if bool(pair[0].get("graph_candidates", {}).get("complete"))
        and bool(pair[1].get("graph_candidates", {}).get("complete"))
    ]
    result["common_all"] = [
        pair for pair in rows
        if pair in result["common_independent_parse"] and pair in result["common_joint_parse"] and pair in result["common_shortlist"]
    ]
    result["candidate_exposed"] = [pair for pair in rows if _target_exposed(pair[1], graph_key)]
    result["candidate_not_exposed"] = [pair for pair in rows if not _target_exposed(pair[1], graph_key)]
    for code in ("B4", "B1", "G"):
        result[code] = [pair for pair in rows if code in label_codes(str(pair[0].get("video_id", "")))]
    result["target_labels"] = [pair for pair in rows if target_codes & label_codes(str(pair[0].get("video_id", "")))]
    result["gt_normal"] = [pair for pair in rows if int(pair[0].get("y_true", 0)) == 0]
    result["boundary"] = [
        pair for pair in rows
        if str((pair[0].get("gt") or {}).get("boundary_position", "none")) != "none"
        or "boundary" in str((pair[0].get("gt") or {}).get("temporal_subset", ""))
    ]
    return result


def _bootstrap(pairs: list[tuple[dict, dict]], repetitions: int, seed: int) -> dict:
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        grouped[source_group_id(str(pair[0].get("video_id", "")))].append(pair)
    groups = sorted(grouped)
    metrics = ("ap", "auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "fp", "fn")
    samples = {metric: [] for metric in metrics}
    rng = random.Random(seed)
    for _ in range(max(0, repetitions)):
        sampled = [rng.choice(groups) for _ in groups] if groups else []
        batch = [pair for group in sampled for pair in grouped[group]]
        before, after = _metric(pair[0] for pair in batch), _metric(pair[1] for pair in batch)
        delta = _delta(after, before)
        for metric in metrics:
            samples[metric].append(float(delta[metric]))
    output = {"source_groups": len(groups), "repetitions": repetitions, "seed": seed}
    for metric, values in samples.items():
        ordered = sorted(values)
        if not ordered:
            output[metric] = {"mean": None, "ci95": [None, None], "p_positive": None}
            continue
        lo = ordered[int(0.025 * (len(ordered) - 1))]
        hi = ordered[int(0.975 * (len(ordered) - 1))]
        output[metric] = {
            "mean": sum(ordered) / len(ordered), "ci95": [lo, hi],
            "p_positive": sum(value > 0 for value in ordered) / len(ordered),
            "p_nonpositive": sum(value <= 0 for value in ordered) / len(ordered),
        }
    return output


def analyze(args: argparse.Namespace) -> dict:
    baseline, candidate = _records(args.baseline_run), _records(args.candidate_run)
    shared = sorted(set(baseline) & set(candidate))
    pairs = [(baseline[key], candidate[key]) for key in shared]
    target_codes = {value for value in args.target_label_codes.replace(",", " ").split() if value}
    subsets = _subsets(pairs, target_codes, args.candidate_graph_key)
    paired_rows, subset_summary = [], {}
    attribution_rows = []
    for name, values in subsets.items():
        before, after = _metric(pair[0] for pair in values), _metric(pair[1] for pair in values)
        helps = hurts = 0
        for base, cand in values:
            y = int(base.get("y_true", 0))
            bp = _competition_tuple(base)[1]
            cp = _competition_tuple(cand)[1]
            helps += int(bp != y and cp == y)
            hurts += int(bp == y and cp != y)
        subset_summary[name] = {"n": len(values), "baseline": before, "candidate": after, "delta": _delta(after, before), "helps": helps, "hurts": hurts}
        paired_rows.append({"subset": name, "n": len(values), "helps": helps, "hurts": hurts, **{f"delta_{k}": v for k, v in _delta(after, before).items() if not isinstance(v, dict)}})
    for base, cand in pairs:
        y = int(base.get("y_true", 0))
        bp, cp = _competition_tuple(base)[1], _competition_tuple(cand)[1]
        if bp == cp:
            continue
        mechanism = _attribution(base, cand, args.candidate_graph_key)
        bset = set(base.get("graph_candidates", {}).get("selected_abnormal", [])) | set(base.get("graph_candidates", {}).get("selected_normal", []))
        cset = set(cand.get("graph_candidates", {}).get("selected_abnormal", [])) | set(cand.get("graph_candidates", {}).get("selected_normal", []))
        attribution_rows.append({
            "segment_key": base.get("segment_key"), "video_id": base.get("video_id"),
            "source_group": source_group_id(str(base.get("video_id", ""))), "y_true": y,
            "labels": sorted(label_codes(str(base.get("video_id", "")))),
            "baseline_pred": bp, "candidate_pred": cp,
            "effect": "help" if bp != y and cp == y else "hurt" if bp == y and cp != y else "changed_unresolved",
            "mechanism": mechanism,
            "baseline_margin": _competition_tuple(base)[2], "candidate_margin": _competition_tuple(cand)[2],
            "baseline": _state(base, args.candidate_graph_key),
            "candidate": _state(cand, args.candidate_graph_key),
            "attribution": {
                "candidate_graph_directly_selected": _target_exposed(cand, args.candidate_graph_key),
                "candidate_graph_became_winner": args.candidate_graph_key in {
                    cand.get("competitions", {}).get(METHOD, {}).get("best_abnormal_graph"),
                    cand.get("competitions", {}).get(METHOD, {}).get("best_normal_graph"),
                },
                "shortlist_changed": bset != cset,
                "shared_node_union_changed": set(base.get("independent_node_calls", {})) != set(cand.get("independent_node_calls", {})),
                "parse_completeness_changed": _complete(base, args.candidate_graph_key) != _complete(cand, args.candidate_graph_key),
                "threshold_only_flip": mechanism == "threshold_or_calibration_flip",
                "primary_mechanism": mechanism,
            },
        })
    parse_rows, shortlist_rows = [], []
    for base, cand in pairs:
        bi, bj = _complete(base, args.candidate_graph_key); ai, aj = _complete(cand, args.candidate_graph_key)
        parse_rows.append({"segment_key": base.get("segment_key"), "baseline_independent": bi, "candidate_independent": ai, "baseline_joint": bj, "candidate_joint": aj})
        bsel = set(base.get("graph_candidates", {}).get("selected_abnormal", [])) | set(base.get("graph_candidates", {}).get("selected_normal", []))
        csel = set(cand.get("graph_candidates", {}).get("selected_abnormal", [])) | set(cand.get("graph_candidates", {}).get("selected_normal", []))
        shortlist_rows.append({"segment_key": base.get("segment_key"), "baseline_target": args.candidate_graph_key in bsel, "candidate_target": args.candidate_graph_key in csel, "added": ";".join(sorted(csel-bsel)), "removed": ";".join(sorted(bsel-csel))})
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "effect_summary.json", {"candidate_id": args.candidate_id, "candidate_graph_key": args.candidate_graph_key, "target_label_codes": sorted(target_codes), "subsets": subset_summary})
    _write_csv(out / "paired_subsets.csv", paired_rows)
    bootstrap = _bootstrap(pairs, args.bootstrap_repetitions, args.bootstrap_seed)
    write_json(out / "source_group_bootstrap.json", bootstrap)
    class_bootstrap = {code: _bootstrap(subsets.get(code, []), args.bootstrap_repetitions, args.bootstrap_seed + index + 1) for index, code in enumerate(("B1", "B4", "G"))}
    write_json(out / "class_delta_bootstrap.json", class_bootstrap)
    write_jsonl(out / "help_hurt_attribution.jsonl", attribution_rows)
    counts = Counter((row["effect"], row["mechanism"]) for row in attribution_rows)
    attribution_summary = [{"effect": effect, "mechanism": mechanism, "n": n} for (effect, mechanism), n in sorted(counts.items())]
    _write_csv(out / "help_hurt_summary.csv", attribution_summary)
    _write_csv(out / "parse_completeness_audit.csv", parse_rows)
    _write_csv(out / "shortlist_change_audit.csv", shortlist_rows)
    all_summary = subset_summary["all_matched"]
    rows_html = "".join(f"<tr><td>{html.escape(row['subset'])}</td><td>{row['n']}</td><td>{row['helps']}</td><td>{row['hurts']}</td><td>{row.get('delta_ap', 0):.4f}</td><td>{row.get('delta_balanced_accuracy', 0):.4f}</td></tr>" for row in paired_rows)
    (out / "index.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Exact candidate effects</title><style>body{{font:15px system-ui;max-width:1100px;margin:32px auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}</style><h1>{html.escape(args.candidate_graph_key)}</h1><p>Matched windows: {len(pairs)}. Helps: {all_summary['helps']}; hurts: {all_summary['hurts']}.</p><table><thead><tr><th>Subset</th><th>N</th><th>Helps</th><th>Hurts</th><th>Delta AP</th><th>Delta BA</th></tr></thead><tbody>{rows_html}</tbody></table>", encoding="utf-8")
    return {"matched": len(pairs), "helps": all_summary["helps"], "hurts": all_summary["hurts"], "out_dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-graph-key", required=True)
    parser.add_argument("--target-label-codes", default="")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
