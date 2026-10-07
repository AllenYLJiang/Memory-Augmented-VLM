#!/usr/bin/env python3
"""Export explicit correct-polarity graph competitions with node-level frames.

This is a post-hoc exporter. It does not call the VLM or DeepSeek and does not
change predictions. Each case pairs the winning graph on the GT side with the
closest-scoring graph from the opposite side under the same matching method.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from common import safe_name, write_json, write_jsonl


METHODS = (
    "independent_direct_nodes",
    "shared_unary_rowmax",
    "unary_ot",
    "conditional_rowmax",
    "conditional_ot_no_coherence",
    "conditional_ot_full",
)
METHOD_LABELS = {
    "independent_direct_nodes": "M0 independent nodes",
    "shared_unary_rowmax": "M1 shared unary",
    "unary_ot": "M2 unary OT",
    "conditional_rowmax": "M3a conditional row-max",
    "conditional_ot_no_coherence": "M3b conditional OT",
    "conditional_ot_full": "M3c full conditional OT",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _truth(record: Mapping[str, Any]) -> int:
    return int(record.get("y_true_operational", record.get("y_true", 0)) or 0)


def _catalog(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for polarity in ("abnormal", "normal"):
        for graph in raw.get(polarity, []):
            if isinstance(graph, dict):
                graph = dict(graph)
                graph["polarity"] = polarity
                values.append(graph)
    result = {str(graph["key"]): graph for graph in values}
    if not result:
        raise ValueError(f"empty graph catalog: {path}")
    return result


def select_case(record: Mapping[str, Any], method: str = "conditional_ot_full") -> dict | None:
    """Return winner/nearest-opponent metadata for an explicit correct decision."""
    truth = _truth(record)
    competition = record.get("competitions", {}).get(method, {})
    expected_decision = "abnormal" if truth else "normal"
    if competition.get("decision") != expected_decision:
        return None

    winning_key = str(
        competition.get("best_abnormal_graph" if truth else "best_normal_graph", "")
    )
    if not winning_key or winning_key == "NONE":
        return None
    method_results = record.get("graph_results", {}).get(method, {})
    winning_result = method_results.get(winning_key)
    if not isinstance(winning_result, Mapping):
        return None
    winning_score = _float(winning_result.get("graph_score"))

    opposite_polarity = "normal" if truth else "abnormal"
    candidates = []
    for graph_key, value in method_results.items():
        if not isinstance(value, Mapping):
            continue
        # Candidate polarity can be recovered from which side won the shortlist.
        # The caller fills catalog polarity before exporting; this fallback uses
        # the selected graph lists persisted in every record.
        selected = record.get("graph_candidates", {}).get(
            "selected_normal" if opposite_polarity == "normal" else "selected_abnormal", []
        )
        if graph_key not in selected:
            continue
        score = _float(value.get("graph_score"))
        candidates.append((abs(winning_score - score), -score, str(graph_key), value))
    if not candidates:
        return None
    _, _, losing_key, losing_result = min(candidates)
    losing_score = _float(losing_result.get("graph_score"))

    m0 = record.get("competitions", {}).get("independent_direct_nodes", {})
    m0_pred = int(m0.get("y_pred", 0) or 0)
    graph_pred = int(competition.get("y_pred", 0) or 0)
    return {
        "case_id": str(record.get("case_id", "")),
        "segment_key": str(record.get("segment_key", "")),
        "video_id": str(record.get("video_id", "")),
        "y_true": truth,
        "method": method,
        "winning_graph": winning_key,
        "winning_polarity": expected_decision,
        "winning_graph_score": winning_score,
        "losing_graph": losing_key,
        "losing_polarity": opposite_polarity,
        "losing_graph_score": losing_score,
        "individual_score_gap": winning_score - losing_score,
        "abnormal_aggregate": _float(competition.get("best_abnormal_score")),
        "normal_aggregate": _float(competition.get("best_normal_score")),
        "competition_margin": _float(competition.get("margin")),
        "decision_margin_threshold": _float(competition.get("decision_margin_threshold")),
        "m0_decision": str(m0.get("decision", "")),
        "m0_y_pred": m0_pred,
        "m0_margin": _float(m0.get("margin")),
        "graph_y_pred": graph_pred,
        "graph_advantage": bool(m0_pred != truth and graph_pred == truth),
        "verified_graph_help": bool(record.get("comparison", {}).get("verified_graph_help")),
    }


def _best_bin(trace: Mapping[str, Any] | None) -> int | None:
    if not isinstance(trace, Mapping):
        return None
    value = trace.get("best_bin")
    if isinstance(value, int) and 0 <= value < 8:
        return value
    location = trace.get("location_distribution_given_present", [])
    quality = trace.get("evidence_quality_by_bin", [])
    if not isinstance(location, list) or not location:
        return None
    scores = []
    for index, item in enumerate(location[:8]):
        q = quality[index] if isinstance(quality, list) and index < len(quality) else 1.0
        scores.append(_float(item) * _float(q, 1.0))
    return max(range(len(scores)), key=scores.__getitem__) if scores else None


def _node_bin(record: Mapping[str, Any], graph_key: str, node_key: str, method: str) -> tuple[int, str]:
    graph_result = record.get("graph_results", {}).get(method, {}).get(graph_key, {})
    assignment = next(
        (
            str(item.get("assigned", ""))
            for item in graph_result.get("assignments", [])
            if str(item.get("node_key")) == node_key
        ),
        "",
    )
    match = re.match(r"T(\d+)(?:#\d+)?$", assignment)
    if match:
        return max(0, min(7, int(match.group(1)))), f"OT assignment {assignment}"

    joint = record.get("joint_graph_calls", {}).get(graph_key, {}).get("nodes", {}).get(node_key)
    index = _best_bin(joint)
    if index is not None:
        return index, "joint conditional best bin"
    independent = record.get("independent_node_calls", {}).get(node_key)
    index = _best_bin(independent)
    if index is not None:
        return index, "independent-node best bin fallback"
    return 4, "center-frame fallback"


def _node_rows(
    record: Mapping[str, Any], graph: Mapping[str, Any], graph_key: str, method: str
) -> list[dict]:
    rows = []
    for node in graph.get("nodes", []):
        node_key = str(node.get("key", ""))
        bin_index, frame_basis = _node_bin(record, graph_key, node_key, method)
        scores = {}
        presences = {}
        for candidate_method in METHODS:
            value = record.get("graph_results", {}).get(candidate_method, {}).get(graph_key, {})
            scores[candidate_method] = _float(value.get("node_support", {}).get(node_key))
            presences[candidate_method] = _float(value.get("node_presence", {}).get(node_key))
        rows.append(
            {
                "key": node_key,
                "title": str(node.get("title", node_key)),
                "required": bool(node.get("required", True)),
                "role": str(node.get("role", "")),
                "phase_hint": str(node.get("phase_hint", "")),
                "representative_bin": bin_index,
                "frame_basis": frame_basis,
                "node_support": scores,
                "node_presence": presences,
            }
        )
    return rows


def _copy_node_frames(case_dir: Path, record: Mapping[str, Any], side: str, rows: list[dict]) -> None:
    evidence = record.get("evidence", {})
    paths = [Path(str(value)) for value in evidence.get("image_paths", [])]
    frames = list(evidence.get("frame_indices", []))
    target_dir = case_dir / side
    target_dir.mkdir(parents=True, exist_ok=True)
    for order, row in enumerate(rows, 1):
        index = int(row["representative_bin"])
        row["source_image"] = str(paths[index]) if index < len(paths) else ""
        row["frame_index"] = frames[index] if index < len(frames) else None
        filename = f"{order:02d}_{safe_name(row['key'], 64)}.jpg"
        destination = target_dir / filename
        row["image"] = f"{side}/{filename}"
        if index < len(paths) and paths[index].is_file():
            shutil.copy2(paths[index], destination)
        else:
            row["image"] = ""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _f(value: Any, digits: int = 6) -> str:
    return f"{_float(value):.{digits}f}"


CSS = """
*{box-sizing:border-box}body{margin:0;background:#eef1f3;color:#17202a;font:14px/1.5 Arial,sans-serif}
.shell{max-width:1500px;margin:auto;background:white;min-height:100vh}.band{padding:20px 24px;border-bottom:1px solid #ccd4da}
h1{font-size:25px;margin:0 0 8px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px;overflow-wrap:anywhere}
.muted{color:#5d6871}.good{color:#17663a}.bad{color:#a12d28}.pill{display:inline-block;padding:3px 8px;border:1px solid #aeb8bf;border-radius:4px;margin:3px;background:#fff;font-weight:bold}
.math{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.metric{border:1px solid #ccd4da;padding:10px;background:#f7f9fa}.metric b{display:block;font-size:18px}
.graphs{display:grid;grid-template-columns:1fr 1fr;gap:14px}.graph{border:1px solid #ccd4da;padding:13px}.graph.win{border-left:5px solid #23834b}.graph.lose{border-left:5px solid #bd3e35}
.nodes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.node{border:1px solid #d5dce0;background:#fafbfb;padding:9px}.node img{width:100%;aspect-ratio:16/9;object-fit:cover;background:#111}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ccd4da;padding:6px;text-align:left}th{background:#edf1f3}.table{overflow:auto}
.case{border:1px solid #c8d0d5;padding:13px;margin:10px 0}.case.proof{border-left:6px solid #23834b}a{color:#185f8f;text-decoration:none}
@media(max-width:900px){.graphs,.math,.nodes{grid-template-columns:1fr}.band{padding:14px}}
"""


def _node_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{_f(row['node_support'].get(method), 4)}<br><span class='muted'>P={_f(row['node_presence'].get(method), 4)}</span></td>"
            for method in METHODS
        )
        image = f"<img src='{_e(row['image'])}' alt='{_e(row['key'])}'>" if row.get("image") else "<div class='muted'>frame unavailable</div>"
        body.append(
            f"<article class='node'>{image}<h3>{_e(row['title'])}</h3>"
            f"<p><code>{_e(row['key'])}</code><br>frame={_e(row.get('frame_index'))}, T{row['representative_bin']}<br>"
            f"{_e(row['frame_basis'])}</p><div class='table'><table><tr>"
            + "".join(f"<th>{_e(METHOD_LABELS[m])}</th>" for m in METHODS)
            + f"</tr><tr>{cells}</tr></table></div></article>"
        )
    return "".join(body)


def _case_html(case: Mapping[str, Any]) -> str:
    proof = bool(case.get("graph_advantage"))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{_e(case['case_id'])}</title><style>{CSS}</style></head>
<body><main class='shell'><section class='band'><a href='../../index.html'>Back to all examples</a><h1>{_e(case['segment_key'])}</h1>
<span class='pill'>GT: {_e(case['winning_polarity'].upper())}</span><span class='pill'>{'M0 FAIL -> M3c SUCCESS' if proof else 'M3c correct competition'}</span>
<p>M0: {_e(case['m0_decision'])}, margin={_f(case['m0_margin'])}; M3c: {_e(case['winning_polarity'])}, margin={_f(case['competition_margin'])}.</p></section>
<section class='band'><h2>Direct decision quantities</h2><div class='math'>
<div class='metric'>Abnormal aggregate A<b>{_f(case['abnormal_aggregate'])}</b></div><div class='metric'>Normal aggregate N<b>{_f(case['normal_aggregate'])}</b></div>
<div class='metric'>Margin A - N<b>{_f(case['competition_margin'])}</b></div><div class='metric'>Threshold tau<b>{_f(case['decision_margin_threshold'])}</b></div></div>
<p>The successful and failed graph scores below are individual M3c graph scores. The decision above uses candidate-normalized aggregate scores.</p></section>
<section class='band graphs'><article class='graph win'><h2>Successful graph</h2><h3>{_e(case['winning_graph_title'])}</h3><p><code>{_e(case['winning_graph'])}</code></p><p>score = <b>{_f(case['winning_graph_score'])}</b></p>
<div class='nodes'>{_node_table(case['winning_nodes'])}</div></article>
<article class='graph lose'><h2>Nearest incorrect graph</h2><h3>{_e(case['losing_graph_title'])}</h3><p><code>{_e(case['losing_graph'])}</code></p><p>score = <b>{_f(case['losing_graph_score'])}</b>; gap = {_f(case['individual_score_gap'])}</p>
<div class='nodes'>{_node_table(case['losing_nodes'])}</div></article></section></main></body></html>"""


def _index_html(cases: list[dict], summary: Mapping[str, Any]) -> str:
    cards = []
    for case in cases:
        cls = "case proof" if case["graph_advantage"] else "case"
        cards.append(
            f"<article class='{cls}'><h3><a href='cases/{_e(case['case_id'])}/index.html'>{_e(case['segment_key'])}</a></h3>"
            f"<span class='pill'>GT {_e(case['winning_polarity'].upper())}</span>"
            f"<span class='pill'>{'M0 FAIL -> M3c SUCCESS' if case['graph_advantage'] else 'M3c correct'}</span>"
            f"<p><b>Successful:</b> {_e(case['winning_graph'])} = {_f(case['winning_graph_score'])}<br>"
            f"<b>Nearest incorrect:</b> {_e(case['losing_graph'])} = {_f(case['losing_graph_score'])}<br>"
            f"<b>Aggregate:</b> A={_f(case['abnormal_aggregate'])}, N={_f(case['normal_aggregate'])}, A-N={_f(case['competition_margin'])}</p></article>"
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Winner vs nearest loser</title><style>{CSS}</style></head>
<body><main class='shell'><section class='band'><h1>Correct graph competitions: winner vs nearest opposite graph</h1>
<p>This report is generated offline from stored VLM/OT records. No new model call is made.</p>
<span class='pill'>records {summary['records_seen']}</span><span class='pill'>strict correct competitions {summary['cases_exported']}</span>
<span class='pill'>M0 fail -> M3c success {summary['graph_advantage_cases']}</span><span class='pill'>verified helps {summary['verified_graph_advantage_cases']}</span>
<p class='muted'>Green-left cases are the direct graph-over-independent-nodes subset. Each detail page contains one representative frame per node for both graphs.</p></section>
<section class='band'><h2>Examples</h2>{''.join(cards)}</section></main></body></html>"""


def export(run_dir: Path, graph_catalog: Path, out_dir: Path, method: str) -> dict:
    catalog = _catalog(graph_catalog)
    record_paths = sorted((run_dir / "records").glob("*.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    skipped_missing_catalog = 0
    for path in record_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        selected = select_case(record, method)
        if selected is None:
            continue
        winner = catalog.get(selected["winning_graph"])
        loser = catalog.get(selected["losing_graph"])
        if winner is None or loser is None:
            skipped_missing_catalog += 1
            continue
        selected["winning_graph_title"] = winner.get("title", selected["winning_graph"])
        selected["losing_graph_title"] = loser.get("title", selected["losing_graph"])
        selected["winning_nodes"] = _node_rows(record, winner, selected["winning_graph"], method)
        selected["losing_nodes"] = _node_rows(record, loser, selected["losing_graph"], method)
        case_dir = out_dir / "cases" / selected["case_id"]
        _copy_node_frames(case_dir, record, "successful_graph", selected["winning_nodes"])
        _copy_node_frames(case_dir, record, "nearest_incorrect_graph", selected["losing_nodes"])
        write_json(case_dir / "case.json", selected)
        (case_dir / "index.html").write_text(_case_html(selected), encoding="utf-8")
        cases.append(selected)
        marker = "GRAPH-OVER-NODES" if selected["graph_advantage"] else "CORRECT-COMPETITION"
        print(
            f"[+] {marker} {selected['case_id']} GT={selected['winning_polarity']} "
            f"{selected['winning_graph']}={selected['winning_graph_score']:.4f} vs "
            f"{selected['losing_graph']}={selected['losing_graph_score']:.4f}"
        )

    cases.sort(key=lambda item: (not item["graph_advantage"], abs(item["individual_score_gap"])))
    summary = {
        "version": "winner_nearest_loser_examples_v1",
        "run_dir": str(run_dir),
        "graph_catalog": str(graph_catalog),
        "method": method,
        "records_seen": len(record_paths),
        "cases_exported": len(cases),
        "graph_advantage_cases": sum(bool(case["graph_advantage"]) for case in cases),
        "verified_graph_advantage_cases": sum(
            bool(case["graph_advantage"] and case["verified_graph_help"]) for case in cases
        ),
        "gt_abnormal_cases": sum(case["y_true"] == 1 for case in cases),
        "gt_normal_cases": sum(case["y_true"] == 0 for case in cases),
        "skipped_missing_catalog": skipped_missing_catalog,
        "score_note": "winner/loser scores are individual graph scores; competition uses polarity aggregate scores",
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "examples.jsonl", cases)
    write_jsonl(out_dir / "graph_advantage_examples.jsonl", (case for case in cases if case["graph_advantage"]))
    (out_dir / "index.html").write_text(_index_html(cases, summary), encoding="utf-8")
    print(f"[winner-loser] exported={len(cases)} graph_advantage={summary['graph_advantage_cases']} -> {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--graph-catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--method", default="conditional_ot_full", choices=METHODS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / "winner_loser_examples")
    export(args.run_dir, args.graph_catalog, out_dir, args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
