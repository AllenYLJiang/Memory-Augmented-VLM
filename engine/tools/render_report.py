#!/usr/bin/env python3
"""Case visualizations and aggregate reports for conditional OT v3."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import read_json, write_json, write_jsonl
from live_matching import METHODS, PRIMARY_GRAPH_METHOD

METHOD_LABELS = {
    "independent_direct_nodes": "M0 independent node calls",
    "shared_unary_rowmax": "M1 shared unary row-max",
    "unary_ot": "M2 initial unary OT",
    "conditional_rowmax": "M3a conditional row-max",
    "conditional_ot_no_coherence": "M3b conditional OT, no coherence",
    "conditional_ot_full": "M3c full conditional OT",
}

METHOD_SHORT_LABELS = {
    "independent_direct_nodes": "M0 independent",
    "shared_unary_rowmax": "M1 shared unary",
    "unary_ot": "M2 Unary OT",
    "conditional_rowmax": "M3a conditional row-max",
    "conditional_ot_no_coherence": "M3b conditional OT, no coherence",
    "conditional_ot_full": "M3c full conditional OT",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _heat(value: Any) -> str:
    try:
        number = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        number = 0.0
    red = int(248 - 90 * number)
    green = int(249 - 25 * number)
    blue = int(250 - 160 * number)
    color = "#111" if number < 0.7 else "#071b0c"
    return f"background:rgb({red},{green},{blue});color:{color}"


REPORT_CSS = """
:root {
  --ink: #17202a;
  --muted: #5c6873;
  --line: #cfd6dc;
  --soft: #f4f6f7;
  --paper: #ffffff;
  --good: #17663a;
  --good-bg: #edf7f0;
  --bad: #a12d28;
  --bad-bg: #fff1ef;
  --warn: #855b09;
  --warn-bg: #fff7df;
  --info: #245f87;
  --info-bg: #eef6fb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #e9edf0;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.48;
  letter-spacing: 0;
}
a { color: #185d8d; text-decoration: none; }
a:hover { text-decoration: underline; }
.shell { max-width: 1460px; margin: 0 auto; background: var(--paper); min-height: 100vh; }
.topbar { padding: 14px 24px; border-bottom: 1px solid var(--line); background: #f8fafb; }
.topbar a { font-weight: 650; }
.page-head { padding: 22px 24px 18px; border-bottom: 1px solid var(--line); }
.eyebrow { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
h1 { margin: 5px 0 7px; font-size: 24px; line-height: 1.25; overflow-wrap: anywhere; }
h2 { margin: 0 0 12px; font-size: 18px; }
h3 { margin: 0 0 7px; font-size: 15px; }
p { margin: 6px 0; }
.muted { color: var(--muted); }
.band { padding: 20px 24px; border-bottom: 1px solid var(--line); }
.band.soft { background: var(--soft); }
.status-line { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill { display: inline-flex; align-items: center; min-height: 28px; padding: 4px 9px; border: 1px solid var(--line); border-radius: 4px; background: #fff; font-weight: 650; }
.pill.good { color: var(--good); border-color: #9ac5a7; background: var(--good-bg); }
.pill.bad { color: var(--bad); border-color: #e2aaa5; background: var(--bad-bg); }
.pill.warn { color: var(--warn); border-color: #dbc47f; background: var(--warn-bg); }
.pill.info { color: var(--info); border-color: #a7c7da; background: var(--info-bg); }
.summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.method-panel, .graph-panel, .case-card, .metric-box {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}
.method-panel, .graph-panel { padding: 14px; }
.method-panel.correct, .graph-panel.correct { border-left: 5px solid var(--good); }
.method-panel.wrong, .graph-panel.opposing { border-left: 5px solid var(--bad); }
.method-title { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.method-title strong { font-size: 16px; }
.decision { font-size: 20px; font-weight: 780; }
.decision.good { color: var(--good); }
.decision.bad { color: var(--bad); }
.score-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0 9px; }
.score-box, .metric-box { padding: 10px; }
.score-box { background: var(--soft); border: 1px solid #dbe1e5; border-radius: 4px; }
.score-box b, .metric-box b { display: block; margin-top: 2px; font-size: 18px; }
.metric-box span { color: var(--muted); font-size: 12px; }
.callout { margin-top: 12px; padding: 10px 12px; border-left: 4px solid var(--info); background: var(--info-bg); }
.callout.good { border-left-color: var(--good); background: var(--good-bg); }
.callout.warn { border-left-color: var(--warn); background: var(--warn-bg); }
.timeline { display: block; width: 100%; height: auto; border: 1px solid var(--line); background: #111; }
.graph-heading { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.graph-key { overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.graph-stats { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eef1f3; font-weight: 700; }
tr:nth-child(even) td { background: #fbfcfc; }
.ok { color: var(--good); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
.score-cell { min-width: 104px; }
.score-track { display: block; width: 100%; height: 6px; margin-top: 4px; background: #dfe4e7; border-radius: 3px; overflow: hidden; }
.score-fill { display: block; height: 100%; background: #3379a5; }
.delta-pos { color: var(--good); font-weight: 700; }
.delta-neg { color: var(--bad); font-weight: 700; }
.case-list { display: grid; grid-template-columns: 1fr; gap: 14px; }
.case-card { display: grid; grid-template-columns: minmax(300px, 38%) minmax(0, 1fr); overflow: hidden; }
.case-card > a { align-self: stretch; background: #111; }
.case-card img { display: block; width: 100%; height: 100%; min-height: 230px; max-height: 300px; object-fit: cover; background: #111; }
.case-body { padding: 12px; }
.case-body h3 { overflow-wrap: anywhere; }
.case-detail { grid-column: 1 / -1; border-top: 1px solid var(--line); padding: 12px; }
.case-detail > summary { font-size: 15px; }
.graph-stack { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }
.graph-panel { overflow: hidden; }
.method-score-table th, .method-score-table td { white-space: nowrap; }
.node-method-table th { min-width: 105px; }
.node-method-table th:first-child { min-width: 180px; }
.node-method-table th:last-child { min-width: 210px; }
.case-meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.transition { font-weight: 750; }
.decision-math { margin: 10px 0; }
.decision-math th, .decision-math td { text-align: center; }
.decision-math th:first-child, .decision-math td:first-child { text-align: left; }
.formula { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-weight: 700; }
.arithmetic-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
.arithmetic-card { padding: 12px; border: 1px solid var(--line); border-left: 5px solid var(--line); border-radius: 5px; background: #fff; }
.arithmetic-card.correct { border-left-color: var(--good); background: var(--good-bg); }
.arithmetic-card.wrong { border-left-color: var(--bad); background: var(--bad-bg); }
.arithmetic-card h3 { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }
.arithmetic-equation { margin: 9px 0; padding: 9px 10px; border: 1px solid #cbd3d8; border-radius: 4px; background: #fff; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 16px; font-weight: 750; }
.arithmetic-result { font-weight: 750; }
.aggregate-replay { margin: 10px 0; padding: 10px 12px; border: 1px solid var(--line); border-radius: 5px; background: #fff; }
.aggregate-replay > summary { font-size: 14px; }
.candidate-score-list { margin: 8px 0; padding-left: 22px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.candidate-score-list li { margin: 3px 0; overflow-wrap: anywhere; }
.replay-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
.replay-box { padding: 10px; border: 1px solid var(--line); border-radius: 4px; background: var(--soft); }
.direct-proof { margin: 10px 0; padding: 9px 11px; border-left: 5px solid var(--good); background: var(--good-bg); font-weight: 750; }
.technical details, details.technical { border-top: 1px solid var(--line); padding: 11px 0; }
details > summary { cursor: pointer; font-weight: 700; }
pre { white-space: pre-wrap; word-break: break-word; background: #f5f6f7; border: 1px solid var(--line); padding: 10px; font-size: 12px; }
.table-wrap { overflow-x: auto; }
.legend { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.legend > div { border-left: 4px solid var(--line); padding: 6px 10px; background: #fff; }
.legend .strong { border-color: var(--good); }
.legend .candidate { border-color: var(--warn); }
.legend .boundary { border-color: var(--info); }
@media (max-width: 900px) {
  .summary-grid, .case-list, .legend, .arithmetic-grid, .replay-grid { grid-template-columns: 1fr; }
  .case-card { grid-template-columns: 1fr; }
  .case-detail { grid-column: 1; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .topbar, .page-head, .band { padding-left: 14px; padding-right: 14px; }
  h1 { font-size: 20px; }
}
@media (max-width: 520px) {
  .metric-grid, .score-pair { grid-template-columns: 1fr; }
  .case-card img { height: auto; }
}
"""


def _truth(result: Mapping[str, Any]) -> int:
    value = result.get("y_true_operational", result.get("y_true", 0))
    return int(value or 0)


def _label(value: Any) -> str:
    if value is None:
        return "UNRESOLVED"
    return "ANOMALY" if int(value or 0) == 1 else "NORMAL"


def _score_bar(value: Any) -> str:
    try:
        number = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        number = 0.0
    return (
        f"<div class='score-cell'>{_fmt(number)}"
        f"<span class='score-track'><span class='score-fill' style='width:{number * 100:.1f}%'></span></span></div>"
    )


def _oriented_margin(result: Mapping[str, Any], method: str) -> float:
    margin = float(result.get("competitions", {}).get(method, {}).get("margin", 0.0) or 0.0)
    return margin if _truth(result) == 1 else -margin


def _margin_gain(result: Mapping[str, Any]) -> float:
    return _oriented_margin(result, PRIMARY_GRAPH_METHOD) - _oriented_margin(result, "independent_direct_nodes")


def _method_correct(result: Mapping[str, Any], method: str) -> bool:
    value = result.get("competitions", {}).get(method, {})
    prediction = value.get("y_pred")
    return prediction in {0, 1} and int(prediction) == _truth(result)


def _best_graph(result: Mapping[str, Any], method: str, polarity: str) -> str:
    value = result.get("competitions", {}).get(method, {})
    return str(value.get(f"best_{polarity}_graph", "") or "")


def _ranked_graphs(result: Mapping[str, Any], polarity: str, limit: int | None = None) -> list[str]:
    """Rank a polarity's shortlisted graphs by the final individual graph score."""
    candidates = result.get("graph_candidates", {})
    keys = list(candidates.get(f"selected_{polarity}", []))
    best = _best_graph(result, PRIMARY_GRAPH_METHOD, polarity)
    if best:
        keys.append(best)
    ranking = candidates.get(f"{polarity}_ranking", [])
    keys.extend(
        str(item.get("graph_key"))
        for item in ranking
        if isinstance(item, Mapping) and item.get("graph_key")
    )
    available = result.get("graph_results", {}).get(PRIMARY_GRAPH_METHOD, {})
    unique = []
    for key in keys:
        key = str(key or "")
        if key and key in available and key not in unique:
            unique.append(key)
    unique.sort(
        key=lambda key: float(available.get(key, {}).get("graph_score", float("-inf")) or 0.0),
        reverse=True,
    )
    return unique[:limit] if limit is not None else unique


def _aligned_graph(result: Mapping[str, Any]) -> str:
    polarity = "abnormal" if _truth(result) == 1 else "normal"
    best = _best_graph(result, PRIMARY_GRAPH_METHOD, polarity)
    if best:
        return best
    ranked = _ranked_graphs(result, polarity, limit=1)
    return ranked[0] if ranked else ""


def _opposing_graphs(result: Mapping[str, Any], limit: int = 3) -> list[str]:
    polarity = "normal" if _truth(result) == 1 else "abnormal"
    return _ranked_graphs(result, polarity, limit=limit)


def _case_category(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    verifier = result.get("blind_verifier") or {}
    if comparison.get("verified_graph_help"):
        return "verified correction"
    if comparison.get("graph_helps"):
        return "metric correction; verifier not confirmed"
    if comparison.get("graph_hurts") and verifier.get("preferred_method") == "graph":
        return "better grounding; binary threshold hurt"
    if _method_correct(result, "independent_direct_nodes") and _method_correct(result, PRIMARY_GRAPH_METHOD) and _margin_gain(result) > 0:
        return "both correct; graph margin improves"
    if comparison.get("graph_hurts"):
        return "graph regression"
    return "diagnostic case"


def _timeline(result: Mapping[str, Any], output: Path) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False
    evidence = result.get("evidence", {}) if isinstance(result.get("evidence"), Mapping) else {}
    image_paths = [Path(str(value)) for value in evidence.get("image_paths", [])]
    frame_indices = list(evidence.get("frame_indices", []))
    frames = []
    if image_paths and all(path.is_file() for path in image_paths):
        for index, path in enumerate(image_paths[:8]):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frames.append((index, frame_indices[index] if index < len(frame_indices) else "?", frame))
    else:
        video = Path(str(result.get("video_path", "")))
        if not video.is_file():
            return False
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            return False
        start, end = int(result.get("start_frame", 0)), int(result.get("end_frame", 0))
        try:
            for index in range(8):
                frame_number = round(start + (end - start) * ((index + 0.5) / 8.0))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append((index, frame_number, frame))
        finally:
            capture.release()
    if not frames:
        return False
    rendered = []
    for index, frame_number, frame in frames:
        height, width = frame.shape[:2]
        target_width = 300
        target_height = max(1, round(height * target_width / max(1, width)))
        frame = cv2.resize(frame, (target_width, target_height))
        cv2.rectangle(frame, (0, 0), (145, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"T{index} f{frame_number}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(frame)
    while len(rendered) < 8:
        rendered.append(np.zeros_like(rendered[0]))
    target_height = max(frame.shape[0] for frame in rendered)
    normalized = []
    for frame in rendered:
        if frame.shape[0] < target_height:
            frame = np.vstack([frame, np.zeros((target_height - frame.shape[0], frame.shape[1], 3), dtype=frame.dtype)])
        normalized.append(frame)
    canvas = np.vstack([np.hstack(normalized[:4]), np.hstack(normalized[4:8])])
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88]))


def _competition_table(result: Mapping[str, Any]) -> str:
    rows = []
    for method in METHODS:
        value = result.get("competitions", {}).get(method, {})
        correct = int(value.get("y_pred", 0) == result.get("y_true"))
        rows.append(
            f"<tr><td>{_esc(METHOD_LABELS[method])}</td><td>{_esc(value.get('aggregation'))}</td>"
            f"<td>{_esc(value.get('best_abnormal_graph'))}</td><td>{_fmt(value.get('best_abnormal_graph_score'))}</td>"
            f"<td>{_fmt(value.get('best_abnormal_score'))}</td>"
            f"<td>{_esc(value.get('best_normal_graph'))}</td><td>{_fmt(value.get('best_normal_graph_score'))}</td>"
            f"<td>{_fmt(value.get('best_normal_score'))}</td>"
            f"<td>{_fmt(value.get('margin'))}</td><td>{_esc(value.get('decision'))}</td>"
            f"<td class={'ok' if correct else 'bad'}>{correct}</td></tr>"
        )
    return (
        "<table><thead><tr><th>method</th><th>aggregation</th>"
        "<th>best abnormal graph</th><th>individual score</th><th>abnormal aggregate A</th>"
        "<th>best normal graph</th><th>individual score</th><th>normal aggregate N</th>"
        "<th>raw margin A-N</th><th>decision</th><th>correct</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _shortlist_table(result: Mapping[str, Any]) -> str:
    shortlist = result.get("graph_candidates", {})
    rows = []
    for polarity in ("abnormal", "normal"):
        ranking = {item.get("graph_key"): item for item in shortlist.get(f"{polarity}_ranking", []) if isinstance(item, Mapping)}
        for graph_key in shortlist.get(f"selected_{polarity}", []):
            item = ranking.get(graph_key, {})
            rows.append(
                f"<tr><td>{_esc(polarity)}</td><td>{_esc(graph_key)}</td><td>{_esc(item.get('family'))}</td>"
                f"<td>{_fmt(item.get('visual_support'))}</td><td>{_esc(item.get('visible_reason', ''))}</td></tr>"
            )
    return (
        "<table><thead><tr><th>polarity</th><th>candidate</th><th>family</th><th>selector support</th>"
        "<th>visible reason</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _probability_table(result: Mapping[str, Any]) -> str:
    rows = []
    for item in result.get("probability_flow", []):
        delta = float(item.get("conditional_delta", 0.0) or 0.0)
        rows.append(
            f"<tr><td>{_esc(item.get('graph_key'))}</td><td>{_esc(item.get('node_key'))}</td>"
            f"<td>{_fmt(item.get('independent_probability'))}</td><td>{_fmt(item.get('initial_ot_presence'))}</td>"
            f"<td>{_fmt(item.get('conditional_probability'))}</td>"
            f"<td class={'ok' if delta > 0 else 'bad' if delta < 0 else ''}>{delta:+.3f}</td>"
            f"<td>{_fmt(item.get('conditional_rowmax_presence'))}</td>"
            f"<td>{_fmt(item.get('conditional_ot_no_coherence_presence'))}</td>"
            f"<td>{_fmt(item.get('final_ot_presence'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>graph</th><th>node</th><th>M0 P</th><th>M2 OT P</th><th>conditional P</th>"
        "<th>delta</th><th>M3a P</th><th>M3b P</th><th>M3c P</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _node_heatmaps(result: Mapping[str, Any]) -> str:
    independent = result.get("independent_node_calls", {})
    joint = result.get("joint_graph_calls", {})
    chunks = []
    for graph_key, trace in joint.items():
        rows = []
        for node_key, item in trace.get("nodes", {}).items():
            ind = independent.get(node_key, {})
            ind_location = ind.get("location_distribution_given_present", [0.0] * 8)
            joint_location = item.get("location_distribution_given_present", [0.0] * 8)
            ind_quality = ind.get("evidence_quality_by_bin", [0.0] * 8)
            joint_quality = item.get("evidence_quality_by_bin", [0.0] * 8)
            rows.append(
                f"<tr><td rowspan='4'><b>{_esc(node_key)}</b><br>{_esc(item.get('region'))}</td>"
                f"<td>ind location</td>{''.join(f'<td style=\"{_heat(v)}\">{_fmt(v,2)}</td>' for v in ind_location)}</tr>"
                f"<tr><td>cond location</td>{''.join(f'<td style=\"{_heat(v)}\">{_fmt(v,2)}</td>' for v in joint_location)}</tr>"
                f"<tr><td>ind quality</td>{''.join(f'<td style=\"{_heat(v)}\">{_fmt(v,2)}</td>' for v in ind_quality)}</tr>"
                f"<tr><td>cond quality</td>{''.join(f'<td style=\"{_heat(v)}\">{_fmt(v,2)}</td>' for v in joint_quality)}</tr>"
            )
        chunks.append(
            f"<h3>{_esc(graph_key)}; coherence={_fmt(trace.get('graph_coherence'))}</h3>"
            "<table><thead><tr><th>node</th><th>quantity</th>" + "".join(f"<th>T{i}</th>" for i in range(8))
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    return "".join(chunks)


def _method_panel(result: Mapping[str, Any], method: str, title: str) -> str:
    value = result.get("competitions", {}).get(method, {})
    correct = _method_correct(result, method)
    predicted = _label(value.get("y_pred", 0))
    decision = str(value.get("decision", ""))
    status = "correct" if correct else "wrong"
    policy = value.get("decision_policy", {}) if isinstance(value.get("decision_policy"), Mapping) else {}
    policy_html = ""
    if policy:
        policy_html = (
            f"<p><b>Selected policy:</b> {_esc(policy.get('name'))} &nbsp; "
            f"<b>policy score:</b> {_fmt(policy.get('score'), 6)} &nbsp; "
            f"<b>calibrated P(anomaly):</b> {_fmt(policy.get('calibrated_probability'), 6)} &nbsp; "
            f"<b>policy threshold:</b> {_fmt(policy.get('threshold'), 6)}</p>"
        )
    return f"""
    <section class="method-panel {status}">
      <div class="method-title">
        <div><strong>{_esc(title)}</strong><p class="muted">{_esc(METHOD_LABELS.get(method, method))}</p></div>
        <div class="decision {'good' if correct else 'bad'}">{_esc(predicted)}: {'CORRECT' if correct else 'WRONG'}</div>
      </div>
      <div class="score-pair">
        <div class="score-box"><span>Abnormal aggregate A</span><b>{_fmt(value.get('best_abnormal_score'), 6)}</b><small class="graph-key">top individual: {_esc(value.get('best_abnormal_graph'))} = {_fmt(value.get('best_abnormal_graph_score'), 6)}</small></div>
        <div class="score-box"><span>Normal aggregate N</span><b>{_fmt(value.get('best_normal_score'), 6)}</b><small class="graph-key">top individual: {_esc(value.get('best_normal_graph'))} = {_fmt(value.get('best_normal_graph_score'), 6)}</small></div>
      </div>
      <p><b>Raw margin A - N:</b> {_fmt(value.get('margin'), 6)} &nbsp; <b>Decision:</b> {_esc(decision)} &nbsp; <b>Binary output:</b> {_esc(predicted)}</p>
      {policy_html}
    </section>"""


def _threshold_explanation(value: Mapping[str, Any]) -> str:
    policy = value.get("decision_policy", {}) if isinstance(value.get("decision_policy"), Mapping) else {}
    if policy and policy.get("name") != "legacy_graph":
        probability = float(policy.get("calibrated_probability", 0.0) or 0.0)
        threshold = float(policy.get("threshold", 0.5) or 0.5)
        if policy.get("unresolved"):
            return f"P={probability:.4f} unresolved around {threshold:.3f}"
        return f"P={probability:.4f} {'>=' if probability >= threshold else '<'} {threshold:.3f}"
    margin = float(value.get("margin", 0.0) or 0.0)
    threshold = float(value.get("decision_margin_threshold", 0.0) or 0.0)
    if margin > threshold:
        return f"{margin:.4f} > +{threshold:.3f}"
    if margin < -threshold:
        return f"{margin:.4f} < -{threshold:.3f}"
    return f"|{margin:.4f}| <= {threshold:.3f}"


def _aggregate_decision_table(result: Mapping[str, Any]) -> str:
    truth = _truth(result)
    rows = []
    for method in ("independent_direct_nodes", PRIMARY_GRAPH_METHOD):
        value = result.get("competitions", {}).get(method, {})
        correct = _method_correct(result, method)
        rows.append(
            f"<tr><td><b>{_esc(METHOD_SHORT_LABELS[method])}</b></td>"
            f"<td>{_fmt(value.get('best_abnormal_score'), 4)}</td>"
            f"<td>{_fmt(value.get('best_normal_score'), 4)}</td>"
            f"<td class='formula'>{_fmt(value.get('margin'), 4)}</td>"
            f"<td>{_esc(_threshold_explanation(value))}</td>"
            f"<td>{_esc(value.get('decision'))} / {_esc(_label(value.get('y_pred', 0)))}</td>"
            f"<td class={'ok' if correct else 'bad'}>{'CORRECT' if correct else 'WRONG'}</td></tr>"
        )
    return (
        "<div class='table-wrap decision-math'><table><thead><tr><th>Method</th>"
        "<th>Abnormal aggregate A</th><th>Normal aggregate N</th>"
        "<th>Raw margin A - N</th><th>Threshold test</th><th>Decision / binary</th><th>vs GT</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _signed(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _decision_arithmetic_card(result: Mapping[str, Any], method: str) -> str:
    value = result.get("competitions", {}).get(method, {})
    truth = _truth(result)
    correct = _method_correct(result, method)
    abnormal = float(value.get("best_abnormal_score", 0.0) or 0.0)
    normal = float(value.get("best_normal_score", 0.0) or 0.0)
    margin = float(value.get("margin", abnormal - normal) or 0.0)
    threshold = float(value.get("decision_margin_threshold", 0.0) or 0.0)
    binary = _label(value.get("y_pred", 0))
    outcome = "SUCCEEDS" if correct else "FAILS"
    return f"""
    <section class="arithmetic-card {'correct' if correct else 'wrong'}">
      <h3><span>{_esc(METHOD_SHORT_LABELS.get(method, method))}</span><span class="{'ok' if correct else 'bad'}">{outcome}</span></h3>
      <div class="arithmetic-equation">A {_fmt(abnormal, 6)} - N {_fmt(normal, 6)} = M {_signed(margin, 6)}</div>
      <p><b>Threshold test:</b> <span class="formula">{_esc(_threshold_explanation(value))}</span></p>
      <p class="arithmetic-result">{_esc(value.get('decision'))} &rarr; binary {_esc(binary)}; GT {_esc(_label(truth))} &rarr; {'CORRECT' if correct else 'WRONG'}</p>
    </section>"""


def _decision_arithmetic_cards(result: Mapping[str, Any]) -> str:
    m0_correct = _method_correct(result, "independent_direct_nodes")
    m3_correct = _method_correct(result, PRIMARY_GRAPH_METHOD)
    direct = (
        "<div class='direct-proof'>DIRECT WRONG-TO-RIGHT CORRECTION: M0 independent nodes is wrong, "
        "while M3c full conditional OT is correct.</div>"
        if not m0_correct and m3_correct
        else ""
    )
    return (
        direct
        + "<div class='arithmetic-grid'>"
        + _decision_arithmetic_card(result, "independent_direct_nodes")
        + _decision_arithmetic_card(result, PRIMARY_GRAPH_METHOD)
        + "</div>"
    )


def _candidate_scores(result: Mapping[str, Any], method: str, polarity: str) -> list[tuple[str, float]]:
    candidates = result.get("graph_candidates", {})
    keys = list(candidates.get(f"selected_{polarity}", []))
    method_results = result.get("graph_results", {}).get(method, {})
    output = []
    for key in keys:
        graph_key = str(key or "")
        graph_result = method_results.get(graph_key, {})
        if graph_key and graph_result and graph_result.get("graph_score") is not None:
            output.append((graph_key, float(graph_result.get("graph_score", 0.0) or 0.0)))
    return output


def _replay_aggregate(scores: Sequence[float], aggregation: str, temperature: float) -> float:
    if not scores:
        return 0.0
    if aggregation == "max":
        return max(scores)
    tau = max(float(temperature), 1e-6)
    maximum = max(scores)
    return maximum + tau * math.log(
        sum(math.exp((score - maximum) / tau) for score in scores) / len(scores)
    )


def _candidate_score_list(items: Sequence[tuple[str, float]]) -> str:
    if not items:
        return "<p class='muted'>No candidate graph scores are available.</p>"
    return "<ol class='candidate-score-list'>" + "".join(
        f"<li>{_esc(key)} = {_fmt(score, 8)}</li>" for key, score in items
    ) + "</ol>"


def _aggregation_replay(result: Mapping[str, Any], method: str) -> str:
    value = result.get("competitions", {}).get(method, {})
    aggregation = str(value.get("aggregation", "logmeanexp") or "logmeanexp")
    temperature = float(value.get("temperature", 0.1) or 0.1)
    abnormal = _candidate_scores(result, method, "abnormal")
    normal = _candidate_scores(result, method, "normal")
    replay_a = _replay_aggregate([score for _, score in abnormal], aggregation, temperature)
    replay_n = _replay_aggregate([score for _, score in normal], aggregation, temperature)
    stored_a = float(value.get("best_abnormal_score", 0.0) or 0.0)
    stored_n = float(value.get("best_normal_score", 0.0) or 0.0)
    stored_margin = float(value.get("margin", 0.0) or 0.0)
    return f"""
    <section class="replay-box">
      <h3>{_esc(METHOD_SHORT_LABELS.get(method, method))}</h3>
      <p><b>Abnormal candidate graph scores (k={len(abnormal)})</b></p>
      {_candidate_score_list(abnormal)}
      <p class="formula">replayed A = {_fmt(replay_a, 8)}; stored A = {_fmt(stored_a, 8)}</p>
      <p><b>Normal candidate graph scores (k={len(normal)})</b></p>
      {_candidate_score_list(normal)}
      <p class="formula">replayed N = {_fmt(replay_n, 8)}; stored N = {_fmt(stored_n, 8)}</p>
      <p class="arithmetic-equation">{_fmt(replay_a, 8)} - {_fmt(replay_n, 8)} = {_signed(replay_a - replay_n, 8)}</p>
      <p class="muted">Stored margin: {_signed(stored_margin, 8)}; aggregation={_esc(aggregation)}; temperature tau={_fmt(temperature, 3)}.</p>
    </section>"""


def _aggregation_replay_details(result: Mapping[str, Any], open_details: bool = False) -> str:
    return f"""
    <details class="aggregate-replay" {'open' if open_details else ''}>
      <summary>Recompute A, N and margin from every candidate graph score</summary>
      <p>The exact normalized log-mean-exp formula is <span class="formula">LME_tau(s) = max(s) + tau * ln((1/k) * sum(exp((s_i-max(s))/tau)))</span>. Here tau is 0.1. The division by k prevents a side from winning merely because it has more graph candidates.</p>
      <div class="replay-grid">
        {_aggregation_replay(result, 'independent_direct_nodes')}
        {_aggregation_replay(result, PRIMARY_GRAPH_METHOD)}
      </div>
    </details>"""


def _assignment_map(graph_result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("node_key")): item
        for item in graph_result.get("assignments", [])
        if isinstance(item, Mapping) and item.get("node_key")
    }


def _graph_method_table(result: Mapping[str, Any], graph_key: str) -> str:
    graph_results = result.get("graph_results", {})
    rows = []
    for method in METHODS:
        value = graph_results.get(method, {}).get(graph_key, {})
        if not value:
            rows.append(
                f"<tr><td>{_esc(METHOD_SHORT_LABELS[method])}</td><td colspan='5' class='muted'>not available</td></tr>"
            )
            continue
        rows.append(
            f"<tr><td>{_esc(METHOD_SHORT_LABELS[method])}</td>"
            f"<td>{_fmt(value.get('graph_score'))}</td><td>{_fmt(value.get('node_geometric_score'))}</td>"
            f"<td>{_fmt(value.get('graph_coherence'))}</td><td>{_fmt(value.get('required_coverage'))}</td>"
            f"<td>{_esc(value.get('complete'))}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table class='method-score-table'><thead><tr><th>Setting</th>"
        "<th>Graph score</th><th>Node geometric score</th><th>Coherence</th>"
        "<th>Required coverage</th><th>Complete</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _node_score_table(result: Mapping[str, Any], graph_key: str, include_evidence: bool = True) -> str:
    graph_results = result.get("graph_results", {})
    m3 = graph_results.get(PRIMARY_GRAPH_METHOD, {}).get(graph_key, {})
    joint = result.get("joint_graph_calls", {}).get(graph_key, {})
    joint_nodes = joint.get("nodes", {}) if isinstance(joint.get("nodes"), Mapping) else {}
    order = list(joint_nodes)
    method_presence = {}
    for method in METHODS:
        value = graph_results.get(method, {}).get(graph_key, {})
        presence = value.get("node_presence", {}) if isinstance(value.get("node_presence"), Mapping) else {}
        method_presence[method] = presence
        for node_key in presence:
            if node_key not in order:
                order.append(node_key)
    assignments = _assignment_map(m3)
    rows = []
    for node_key in order:
        assignment = assignments.get(node_key, {})
        node_trace = joint_nodes.get(node_key, {}) if isinstance(joint_nodes.get(node_key), Mapping) else {}
        context = []
        increases = node_trace.get("context_increases_from", [])
        decreases = node_trace.get("context_decreases_from", [])
        if increases:
            context.append("supports from: " + ", ".join(map(str, increases)))
        if decreases:
            context.append("suppressed by: " + ", ".join(map(str, decreases)))
        context_html = f'<br><span class="muted">{_esc("; ".join(context))}</span>' if context else ""
        score_cells = "".join(
            f"<td>{_score_bar(method_presence[method].get(node_key))}</td>"
            if node_key in method_presence[method]
            else "<td class='muted'>-</td>"
            for method in METHODS
        )
        evidence = f"<td>{_esc(node_trace.get('visible_evidence', ''))}{context_html}</td>" if include_evidence else ""
        rows.append(
            f"<tr><td><b class='graph-key'>{_esc(node_key)}</b></td>"
            f"{score_cells}"
            f"<td><b>{_esc(assignment.get('assigned', 'NULL'))}</b><br><span class='muted'>transport mass {_fmt(assignment.get('mass'))}</span></td>"
            f"{evidence}</tr>"
        )
    if not rows:
        return "<p class='muted'>No node-level matching record is available for this graph.</p>"
    return (
        "<div class='table-wrap'><table class='node-method-table'><thead><tr><th>Node</th>"
        + "".join(f"<th title='{_esc(METHOD_LABELS[method])}'>{_esc(METHOD_SHORT_LABELS[method])} node P</th>" for method in METHODS)
        + "<th>Full OT assignment</th>"
        + ("<th>Visible evidence and conditional context</th>" if include_evidence else "")
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _graph_panel(
    result: Mapping[str, Any], graph_key: str, role: str, aligned: bool, include_evidence: bool = True
) -> str:
    graph_results = result.get("graph_results", {})
    m0 = graph_results.get("independent_direct_nodes", {}).get(graph_key, {})
    m3 = graph_results.get(PRIMARY_GRAPH_METHOD, {}).get(graph_key, {})
    joint = result.get("joint_graph_calls", {}).get(graph_key, {})
    diagnostics = m3.get("diagnostics", {}) if isinstance(m3.get("diagnostics"), Mapping) else {}
    collisions = diagnostics.get("collision_details", {}) if isinstance(diagnostics.get("collision_details"), Mapping) else {}
    css_class = "correct" if aligned else "opposing"
    episode = str(joint.get("episode_summary", "") or "")
    return f"""
    <section class="graph-panel {css_class}">
      <div class="graph-heading">
        <div><span class="eyebrow">{_esc(role)}</span><h3 class="graph-key">{_esc(graph_key or 'No graph available')}</h3></div>
        <span class="pill {'good' if aligned else 'bad'}">{'GT-aligned side' if aligned else 'Opposing side'}</span>
      </div>
      {f'<p>{_esc(episode)}</p>' if episode else ''}
      <div class="graph-stats">
        <span class="pill">Node-only graph score: {_fmt(m0.get('graph_score'))}</span>
        <span class="pill info">OT graph score: {_fmt(m3.get('graph_score'))}</span>
        <span class="pill">Coherence: {_fmt(m3.get('graph_coherence'))}</span>
        <span class="pill {'warn' if int(collisions.get('slot_collisions', 0) or 0) else 'good'}">Slot collisions: {_esc(collisions.get('slot_collisions', 0))}</span>
      </div>
      <h3>Graph-level scores under every setting</h3>
      {_graph_method_table(result, graph_key)}
      <h3 style="margin-top:12px">Per-node presence scores under every setting</h3>
      {_node_score_table(result, graph_key, include_evidence=include_evidence)}
    </section>"""


def _case_explanation(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    if comparison.get("verified_graph_help"):
        return "Graph matching changes a wrong node-only decision into the correct decision, and the blind visual verifier also prefers the graph grounding."
    if comparison.get("graph_helps"):
        return "Graph matching changes a wrong node-only decision into the correct binary decision, but the blind verifier does not confirm that its explanation is visually better. Treat this as a candidate example."
    if comparison.get("graph_hurts") and (result.get("blind_verifier") or {}).get("preferred_method") == "graph":
        return "Graph matching is judged better grounded, but its uncertain output maps to binary normal and loses under the operational label. This is a threshold-sensitive boundary case."
    if comparison.get("graph_hurts"):
        return "Graph matching changes a correct node-only decision into an incorrect decision. This is a regression case, not evidence of graph advantage."
    gain = _margin_gain(result)
    if gain > 0 and _method_correct(result, PRIMARY_GRAPH_METHOD):
        return f"Both methods are correct. Graph matching moves the margin {gain:+.3f} toward the ground-truth side, so this is supporting evidence rather than a wrong-to-right correction."
    return "This case is included for diagnosis; it does not provide a direct graph-over-node correctness improvement."


def render_case(result: Mapping[str, Any], run_dir: Path, no_images: bool = False) -> str:
    case_dir = Path(run_dir) / "cases" / str(result.get("case_id"))
    case_dir.mkdir(parents=True, exist_ok=True)
    timeline = case_dir / "timeline.jpg"
    has_timeline = False if no_images else _timeline(result, timeline)
    comparison = result.get("comparison", {})
    verifier = result.get("blind_verifier") or {}
    audit = result.get("conditionality_audit")
    truth = _truth(result)
    aligned_graph = _aligned_graph(result)
    opposing_graphs = _opposing_graphs(result, limit=3)
    gt = result.get("gt", {}) if isinstance(result.get("gt"), Mapping) else {}
    verifier_preference = verifier.get("preferred_method", "not run")
    verifier_class = "good" if verifier_preference == "graph" else "warn"
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(result.get('case_id'))} graph vs nodes</title>
<style>{REPORT_CSS}</style></head><body><main class="shell">
<nav class="topbar"><a href="../../readable_index.html">Back to all graph-vs-node examples</a> &nbsp; <span class="muted">|</span> &nbsp; <a href="record.json">Raw record</a></nav>
<header class="page-head">
  <span class="eyebrow">{_esc(_case_category(result))}</span>
  <h1>{_esc(result.get('video_id'))}</h1>
  <p>Frames {_esc(result.get('start_frame'))}-{_esc(result.get('end_frame'))} &nbsp; Case <span class="graph-key">{_esc(result.get('case_id'))}</span></p>
  <div class="status-line">
    <span class="pill {'bad' if truth else 'good'}">Ground truth: {_label(truth)}</span>
    <span class="pill info">Subset: {_esc(gt.get('temporal_subset'))}</span>
    <span class="pill">Anomaly overlap: {_esc(gt.get('overlap_frames'))}/{_esc(gt.get('window_frames'))} frames</span>
    <span class="pill">Core GT: {_esc(result.get('y_true_core'))}</span>
  </div>
</header>
<section class="band soft">
  <h2>Decision at a glance</h2>
  <div class="summary-grid">
    {_method_panel(result, 'independent_direct_nodes', 'Independent nodes')}
    {_method_panel(result, PRIMARY_GRAPH_METHOD, 'OT graph matching')}
  </div>
  <h3 style="margin-top:14px">Direct decision arithmetic</h3>
  {_decision_arithmetic_cards(result)}
  {_aggregation_replay_details(result, open_details=True)}
  <p class="muted"><span class="formula">raw margin = abnormal aggregate A - normal aggregate N</span>. The aggregates combine every shortlisted graph on each side; they are not the best individual graph scores shown below.</p>
  <div class="callout {'good' if comparison.get('graph_helps') else 'warn' if comparison.get('graph_hurts') else ''}">{_esc(_case_explanation(result))}</div>
  <div class="callout"><b>How to read the scores:</b> the decision cards show abnormal-side versus normal-side competition aggregates. The graph sections below show individual graph scores. M0 and OT graph scores use different constructions, so compare their decisions, node probabilities, assignments and within-method ranking rather than treating the raw graph-score difference as a calibrated probability change.</div>
</section>
<section class="band">
  <h2>What the eight sampled frames show</h2>
  {f'<img class="timeline" src="timeline.jpg" alt="Eight sampled frames T0 to T7">' if has_timeline else '<p>No local timeline image generated.</p>'}
</section>
<section class="band soft">
  <h2>GT-aligned graph candidate versus top 3 opposing graphs</h2>
  <p class="muted">"GT-aligned" means the highest-scoring candidate on the ground-truth polarity. "Opposing" means the graph belongs to the other polarity; the three opposing candidates are ranked by their individual full conditional OT graph score. These labels do not claim that a graph name is the unique semantic truth.</p>
  <div class="graph-stack">
    {_graph_panel(result, aligned_graph, 'GT-aligned graph candidate', True)}
    {''.join(_graph_panel(result, key, f'Opposing graph #{index}', False) for index, key in enumerate(opposing_graphs, 1)) or '<p class="muted">No opposing graph record is available.</p>'}
  </div>
</section>
<section class="band">
  <h2>How to compare node and graph variants</h2>
  <p>Each graph panel reports the graph score and every node's presence score under all six settings: M0, shared unary row-max, Unary OT, conditional row-max, conditional OT without coherence, and full conditional OT. The final column shows the full-OT frame/slot assignment and transport mass.</p>
  <div class="callout {verifier_class}"><b>Blind verifier:</b> prefers {_esc(verifier_preference)} with confidence {_fmt(verifier.get('confidence'))}. {_esc(verifier.get('visual_reason', ''))}</div>
</section>
<section class="band technical">
  <h2>Technical details</h2>
  <details><summary>All six method variants</summary><div class="table-wrap">{_competition_table(result)}</div></details>
  <details><summary>Candidate shortlist and selector reasons</summary><div class="table-wrap">{_shortlist_table(result)}</div></details>
  <details><summary>All node probability flows</summary><div class="table-wrap">{_probability_table(result)}</div></details>
  <details><summary>Independent versus conditional temporal heatmaps</summary>{_node_heatmaps(result)}</details>
  <details><summary>Leave-one-node-out conditionality audit</summary><pre>{_esc(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details>
  <details><summary>Full blind verifier record</summary><pre>{_esc(json.dumps(verifier, ensure_ascii=False, indent=2))}</pre></details>
  <details><summary>Ground-truth audit</summary><pre>{_esc(json.dumps(gt, ensure_ascii=False, indent=2))}</pre></details>
</section>
</main></body></html>"""
    (case_dir / "index.html").write_text(document, encoding="utf-8")
    (case_dir / "readable.html").write_text(document, encoding="utf-8")
    write_json(case_dir / "record.json", result)
    return str(case_dir / "index.html")


def _average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    if not y_true or sum(y_true) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    labels = np.asarray(y_true, dtype=np.int64)[order]
    cumulative = np.cumsum(labels)
    precision = cumulative / (np.arange(len(labels)) + 1)
    return float(np.sum(precision * labels) / max(int(labels.sum()), 1))


def _metrics(rows: Sequence[Mapping[str, Any]], method: str, label_field: str = "y_true") -> dict:
    y_true = [int(row.get(label_field, 0) or 0) for row in rows]
    values = [row.get("competitions", {}).get(method, {}) for row in rows]
    y_pred = [value.get("y_pred") for value in values]
    resolved = [(t, int(p)) for t, p in zip(y_true, y_pred) if p in {0, 1}]
    tp = sum(t == 1 and p == 1 for t, p in resolved)
    tn = sum(t == 0 and p == 0 for t, p in resolved)
    fp = sum(t == 0 and p == 1 for t, p in resolved)
    fn = sum(t == 1 and p == 0 for t, p in resolved)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pure_rows = [
        (int(t), value) for t, value, row in zip(y_true, values, rows)
        if bool((row.get("gt") or {}).get("known_normal"))
    ]
    pure_resolved = [(t, int(value.get("y_pred"))) for t, value in pure_rows if value.get("y_pred") in {0, 1}]
    pure_fp = sum(t == 0 and p == 1 for t, p in pure_resolved)
    pure_tn = sum(t == 0 and p == 0 for t, p in pure_resolved)
    return {
        "method": method,
        "label_field": label_field,
        "n": len(rows), "resolved": len(resolved), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(resolved) if resolved else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "ap": _average_precision(y_true, [float(value.get("margin", 0.0) or 0.0) for value in values]),
        "uncertain": sum(value.get("decision") in {"uncertain", "unresolved"} for value in values),
        "pure_label_A_n": len(pure_rows),
        "pure_label_A_fp": pure_fp,
        "pure_label_A_specificity": pure_tn / (pure_tn + pure_fp) if pure_tn + pure_fp else None,
    }


def _paired(rows, first, second, label_field="y_true"):
    helps = hurts = both_correct = both_wrong = 0
    for row in rows:
        truth = int(row.get(label_field, 0) or 0)
        p1 = row.get("competitions", {}).get(first, {}).get("y_pred")
        p2 = row.get("competitions", {}).get(second, {}).get("y_pred")
        c1 = p1 in {0, 1} and int(p1) == truth
        c2 = p2 in {0, 1} and int(p2) == truth
        helps += int(not c1 and c2)
        hurts += int(c1 and not c2)
        both_correct += int(c1 and c2)
        both_wrong += int(not c1 and not c2)
    return {"first": first, "second": second, "n": len(rows), "helps": helps, "hurts": hurts, "net": helps - hurts, "both_correct": both_correct, "both_wrong": both_wrong}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _video_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    by_video = {}
    for row in rows:
        by_video.setdefault(str(row.get("video_id")), []).append(row)
    output = []
    for video_id, values in sorted(by_video.items()):
        first = _metrics(values, "independent_direct_nodes")
        second = _metrics(values, PRIMARY_GRAPH_METHOD)
        output.append({
            "video_id": video_id,
            "n_windows": len(values),
            "m0_accuracy": first["accuracy"],
            "m3_accuracy": second["accuracy"],
            "delta_accuracy": second["accuracy"] - first["accuracy"],
            "m0_balanced_accuracy": first["balanced_accuracy"],
            "m3_balanced_accuracy": second["balanced_accuracy"],
            "delta_balanced_accuracy": second["balanced_accuracy"] - first["balanced_accuracy"],
            "m0_ap": first["ap"],
            "m3_ap": second["ap"],
            "delta_ap": second["ap"] - first["ap"],
        })
    return output


def _video_bootstrap(rows: Sequence[Mapping[str, Any]], iterations: int = 2000, seed: int = 314) -> dict:
    by_video = {}
    for row in rows:
        by_video.setdefault(str(row.get("video_id")), []).append(row)
    videos = sorted(by_video)
    if not videos:
        return {"n_videos": 0, "iterations": 0}
    rng = np.random.RandomState(seed)
    deltas = {"accuracy": [], "balanced_accuracy": [], "ap": []}
    for _ in range(iterations):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        sample_rows = [row for video in sampled for row in by_video[video]]
        first = _metrics(sample_rows, "independent_direct_nodes")
        second = _metrics(sample_rows, PRIMARY_GRAPH_METHOD)
        for metric in deltas:
            deltas[metric].append(second[metric] - first[metric])
    result = {"n_videos": len(videos), "iterations": iterations, "seed": seed}
    for metric, values_list in deltas.items():
        values = np.asarray(values_list, dtype=np.float64)
        result[f"{metric}_delta_mean"] = float(values.mean())
        result[f"{metric}_delta_ci95"] = [
            float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
        ]
        result[f"{metric}_probability_delta_positive"] = float(np.mean(values > 0))
    effects = _video_effect_rows(rows)
    signs = [np.sign(float(row["delta_accuracy"])) for row in effects]
    result["video_level_sign"] = {
        "graph_better": int(sum(value > 0 for value in signs)),
        "graph_worse": int(sum(value < 0 for value in signs)),
        "tie": int(sum(value == 0 for value in signs)),
    }
    return result


def _example_card(row: Mapping[str, Any], badge: str, badge_class: str) -> str:
    case_id = str(row.get("case_id", ""))
    m0 = row.get("competitions", {}).get("independent_direct_nodes", {})
    m3 = row.get("competitions", {}).get(PRIMARY_GRAPH_METHOD, {})
    truth = _truth(row)
    aligned = "abnormal" if truth == 1 else "normal"
    opposing = "normal" if truth == 1 else "abnormal"
    verifier = row.get("blind_verifier") or {}
    gt = row.get("gt", {}) if isinstance(row.get("gt"), Mapping) else {}
    aligned_graph = _aligned_graph(row)
    opposing_graphs = _opposing_graphs(row, limit=3)
    return f"""
    <article class="case-card">
      <a href="cases/{_esc(case_id)}/readable.html"><img src="cases/{_esc(case_id)}/timeline.jpg" alt="Timeline for case {_esc(case_id)}"></a>
      <div class="case-body">
        <span class="pill {badge_class}">{_esc(badge)}</span>
        <h3><a href="cases/{_esc(case_id)}/readable.html">{_esc(row.get('video_id'))}</a></h3>
        <p class="muted">Frames {_esc(row.get('start_frame'))}-{_esc(row.get('end_frame'))} | GT {_esc(_label(truth))} | {_esc(gt.get('temporal_subset'))} | overlap {_esc(gt.get('overlap_frames'))}/{_esc(gt.get('window_frames'))}</p>
        <p class="transition">Independent: {_esc(_label(m0.get('y_pred')))} ({_esc(m0.get('decision'))}) &rarr; OT graph: {_esc(_label(m3.get('y_pred')))} ({_esc(m3.get('decision'))})</p>
        {_decision_arithmetic_cards(row)}
        {_aggregation_replay_details(row)}
        <p class="muted"><span class="formula">raw margin = abnormal aggregate A - normal aggregate N</span>. A and N are normalized log-mean-exp values over all shortlisted graphs on that side, not the displayed best individual graph score.</p>
        <p><b>GT-aligned graph:</b> <span class="graph-key">{_esc(m3.get(f'best_{aligned}_graph'))}</span><br>
        <b>Highest opposing graph:</b> <span class="graph-key">{_esc(m3.get(f'best_{opposing}_graph'))}</span></p>
        <p><b>Oriented margin:</b> {_fmt(_oriented_margin(row, 'independent_direct_nodes'), 4)} &rarr; {_fmt(_oriented_margin(row, PRIMARY_GRAPH_METHOD), 4)} ({_margin_gain(row):+.4f})</p>
        <p><b>Verifier:</b> {_esc(verifier.get('preferred_method', 'not run'))} {_fmt(verifier.get('confidence'))}</p>
      </div>
      <details class="case-detail">
        <summary>GT-aligned graph + top 3 opposing graphs: all graph and node scores</summary>
        <p class="muted">Opposing graphs are ranked by individual full conditional OT graph score. Expand the tables horizontally if needed; every table contains M0, shared unary, Unary OT, conditional row-max, no-coherence OT, and full conditional OT.</p>
        <div class="graph-stack">
          {_graph_panel(row, aligned_graph, 'GT-aligned graph candidate', True, include_evidence=False)}
          {''.join(_graph_panel(row, key, f'Opposing graph #{index}', False, include_evidence=False) for index, key in enumerate(opposing_graphs, 1)) or '<p class="muted">No opposing graph record is available.</p>'}
        </div>
      </details>
    </article>"""


def _example_section(title: str, intro: str, rows: Sequence[Mapping[str, Any]], badge: str, badge_class: str) -> str:
    if not rows:
        cards = "<p class='muted'>No cases in this category in the current snapshot.</p>"
    else:
        cards = '<div class="case-list">' + "".join(_example_card(row, badge, badge_class) for row in rows) + "</div>"
    return f'<section class="band"><h2>{_esc(title)} <span class="muted">({len(rows)})</span></h2><p>{_esc(intro)}</p>{cards}</section>'


def _find_metric(metric_rows: Sequence[Mapping[str, Any]], method: str, subset: str = "all") -> Mapping[str, Any]:
    return next((row for row in metric_rows if row.get("method") == method and row.get("subset") == subset), {})


def build_report(run_dir: Path, no_images: bool = False, run_signature: str | None = None) -> dict:
    run_dir = Path(run_dir)
    config = read_json(run_dir / "run_config.json", {})
    rows = []
    for path in sorted((run_dir / "records").glob("*.json")):
        value = read_json(path, None)
        if isinstance(value, dict) and (not run_signature or value.get("run_signature") == run_signature):
            rows.append(value)
    rows.sort(key=lambda row: str(row.get("segment_key", "")))
    write_jsonl(run_dir / "ot_window_results.jsonl", rows)
    eligible = [row for row in rows if bool(row.get("metric_eligible", True))]
    proof = [row for row in rows if row.get("comparison", {}).get("verified_graph_help")]
    all_helps = [row for row in rows if row.get("comparison", {}).get("graph_helps")]
    candidate_helps = [row for row in all_helps if not row.get("comparison", {}).get("verified_graph_help")]
    verified_hurts = [row for row in rows if row.get("comparison", {}).get("verified_graph_hurt")]
    regressions = [row for row in rows if row.get("comparison", {}).get("graph_hurts")]
    grounding_threshold_hurts = [
        row for row in regressions
        if (row.get("blind_verifier") or {}).get("preferred_method") == "graph"
    ]
    other_regressions = [row for row in regressions if row not in grounding_threshold_hurts]
    supporting = [
        row for row in rows
        if not row.get("comparison", {}).get("discordant")
        and _method_correct(row, "independent_direct_nodes")
        and _method_correct(row, PRIMARY_GRAPH_METHOD)
        and _margin_gain(row) >= 0.03
    ]
    proof.sort(key=_margin_gain, reverse=True)
    candidate_helps.sort(key=_margin_gain, reverse=True)
    grounding_threshold_hurts.sort(key=lambda row: abs(_margin_gain(row)), reverse=True)
    other_regressions.sort(key=lambda row: abs(_margin_gain(row)), reverse=True)
    supporting.sort(key=_margin_gain, reverse=True)
    failures = [row for row in rows if row.get("comparison", {}).get("conditional_ot_failure")]
    write_jsonl(run_dir / "joint_graph_proof_cases.jsonl", proof)
    write_jsonl(run_dir / "all_graph_help_cases.jsonl", all_helps)
    write_jsonl(run_dir / "supporting_margin_examples.jsonl", supporting)
    write_jsonl(run_dir / "grounding_better_threshold_hurts.jsonl", grounding_threshold_hurts)
    write_jsonl(run_dir / "verified_graph_hurt_cases.jsonl", verified_hurts)
    write_jsonl(run_dir / "joint_graph_regression_cases.jsonl", regressions)
    write_jsonl(run_dir / "conditional_ot_failures.jsonl", failures)
    rendered = set()
    example_rows = proof + candidate_helps + grounding_threshold_hurts + other_regressions + supporting
    for row in example_rows + verified_hurts + failures:
        if row.get("case_id") not in rendered:
            render_case(row, run_dir, no_images=no_images)
            rendered.add(row.get("case_id"))

    metric_rows = []
    for subset_name, subset in (("all", rows), ("metric_eligible", eligible)):
        for method in METHODS:
            metric_rows.append(dict(_metrics(subset, method), subset=subset_name))
    _write_csv(run_dir / "method_metrics.csv", metric_rows)
    core_rows = [row for row in rows if row.get("y_true_core") is not None]
    core_metrics = [dict(_metrics(core_rows, method, "y_true_core"), subset="core_gt") for method in METHODS]
    _write_csv(run_dir / "core_event_metrics.csv", core_metrics)

    decomposition = [
        _paired(rows, "independent_direct_nodes", "shared_unary_rowmax"),
        _paired(rows, "shared_unary_rowmax", "unary_ot"),
        _paired(rows, "shared_unary_rowmax", "conditional_rowmax"),
        _paired(rows, "conditional_rowmax", "conditional_ot_no_coherence"),
        _paired(rows, "conditional_ot_no_coherence", "conditional_ot_full"),
        _paired(rows, "independent_direct_nodes", "conditional_ot_full"),
    ]
    write_json(run_dir / "method_decomposition.json", decomposition)
    flow_rows = [dict(item, segment_key=row.get("segment_key"), y_true=row.get("y_true")) for row in rows for item in row.get("probability_flow", [])]
    _write_csv(run_dir / "node_probability_change.csv", flow_rows)

    paired = {
        "comparison": f"{PRIMARY_GRAPH_METHOD} vs independent_direct_nodes",
        "n": len(rows),
        "graph_helps": sum(bool(row.get("comparison", {}).get("graph_helps")) for row in rows),
        "graph_hurts": sum(bool(row.get("comparison", {}).get("graph_hurts")) for row in rows),
        "verified_graph_helps": len(proof),
        "verified_graph_hurts": len(verified_hurts),
        "both_correct": sum(row.get("comparison", {}).get("independent_correct") and row.get("comparison", {}).get("conditional_ot_correct") for row in rows),
        "both_wrong": sum(not row.get("comparison", {}).get("independent_correct") and not row.get("comparison", {}).get("conditional_ot_correct") for row in rows),
    }
    write_json(run_dir / "paired_effects.json", paired)
    bootstrap_rows = eligible or rows
    bootstrap = _video_bootstrap(bootstrap_rows)
    write_json(run_dir / "video_clustered_bootstrap.json", bootstrap)
    _write_csv(run_dir / "video_level_effect.csv", _video_effect_rows(bootstrap_rows))

    sensitivity_rows = []
    for row in rows:
        for method, values in row.get("top_n_sensitivity", {}).items():
            for value in values:
                sensitivity_rows.append({
                    "segment_key": row.get("segment_key"), "video_id": row.get("video_id"),
                    "y_true": row.get("y_true"), "method": method,
                    "normal_top_n": value.get("normal_top_n"), "margin": value.get("margin"),
                    "decision": value.get("decision"), "y_pred": value.get("y_pred"),
                })
    _write_csv(run_dir / "top_n_sensitivity.csv", sensitivity_rows)

    graph_usage = {
        "shortlisted_abnormal": dict(Counter(key for row in rows for key in row.get("graph_candidates", {}).get("selected_abnormal", []))),
        "shortlisted_normal": dict(Counter(key for row in rows for key in row.get("graph_candidates", {}).get("selected_normal", []))),
        "m0_winning_normal": dict(Counter(row.get("competitions", {}).get("independent_direct_nodes", {}).get("best_normal_graph") for row in rows)),
        "m3_winning_normal": dict(Counter(row.get("competitions", {}).get(PRIMARY_GRAPH_METHOD, {}).get("best_normal_graph") for row in rows)),
    }
    write_json(run_dir / "graph_usage.json", graph_usage)
    scope_keys = (
        "candidate_directed_exact_window_eligible",
        "full_source_video_completion_eligible",
        "natural_distribution_eligible",
        "pure_normal_coverage",
        "dense_temporal_coverage",
    )
    evaluation_scope = {
        key: sum(bool((row.get("evaluation_scope") or {}).get(key)) for row in rows)
        for key in scope_keys
    }
    summary = {
        "version": "multi_candidate_conditional_ot_report_v3",
        "n_windows": len(rows), "n_metric_eligible_windows": len(eligible),
        "n_videos": len({row.get("video_id") for row in rows}),
        "evaluation_scope": evaluation_scope,
        "method_metrics": metric_rows,
        "core_event_metrics": core_metrics,
        "method_decomposition": decomposition,
        "paired_effects": paired,
        "video_clustered_bootstrap": bootstrap,
        "conditional_ot_failures": len(failures),
        "example_groups": {
            "verified_corrections": len(proof),
            "metric_corrections_not_verified": len(candidate_helps),
            "supporting_margin_improvements": len(supporting),
            "grounding_better_threshold_hurts": len(grounding_threshold_hurts),
            "other_regressions": len(other_regressions),
        },
        "parse_completeness": {
            "independent": sum(bool(row.get("completeness", {}).get("independent")) for row in rows) / len(rows) if rows else 0.0,
            "joint": sum(bool(row.get("completeness", {}).get("joint")) for row in rows) / len(rows) if rows else 0.0,
        },
        "graph_usage": graph_usage,
    }
    write_json(run_dir / "summary.json", summary)

    metrics_html = "".join(
        f"<tr><td>{_esc(row['subset'])}</td><td>{_esc(METHOD_LABELS[row['method']])}</td><td>{row['n']}</td>"
        f"<td>{row['accuracy']:.3f}</td><td>{row['balanced_accuracy']:.3f}</td><td>{row['ap']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td><td>{row['specificity']:.3f}</td><td>{row['f1']:.3f}</td></tr>"
        for row in metric_rows
    )
    m0_metric = _find_metric(metric_rows, "independent_direct_nodes")
    m3_metric = _find_metric(metric_rows, PRIMARY_GRAPH_METHOD)
    conditional_rowmax_metric = _find_metric(metric_rows, "conditional_rowmax")
    planned = int(config.get("selected_windows", len(rows)) or len(rows))
    completion = len(rows) / planned if planned else 0.0
    delta_accuracy = float(m3_metric.get("accuracy", 0.0) or 0.0) - float(m0_metric.get("accuracy", 0.0) or 0.0)
    development_note = (
        "The run is complete, but it remains a development evaluation: it uses test-derived active graphs "
        "with discovery-source groups excluded and selects one available window per sampled video. "
        "Do not report it as untouched-test performance."
        if len(rows) >= planned
        else
        "The run is incomplete and remains a development evaluation: it uses test-derived active graphs "
        "with discovery-source groups excluded and selects one available window per sampled video. "
        "Do not report it as final untouched-test performance."
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Active V3 graph matching examples</title>
<style>{REPORT_CSS}</style></head><body><main class="shell">
<header class="page-head">
  <span class="eyebrow">Active V3 development run</span>
  <h1>When does OT graph matching beat independent node matching?</h1>
  <p>This page separates confirmed wrong-to-right corrections, unverified metric corrections, supporting margin improvements, and threshold-sensitive boundary cases.</p>
  <div class="status-line">
    <span class="pill info">Completed: {len(rows)}/{planned} ({completion:.1%})</span>
    <span class="pill good">Verified corrections: {len(proof)}</span>
    <span class="pill">All corrections: {len(all_helps)}</span>
    <span class="pill warn">Raw hurts: {len(regressions)}</span>
  </div>
</header>
<section class="band soft">
  <h2>Current result</h2>
  <div class="metric-grid">
    <div class="metric-box"><span>Independent-node accuracy</span><b>{float(m0_metric.get('accuracy', 0.0)):.3f}</b></div>
    <div class="metric-box"><span>Full OT-graph accuracy</span><b>{float(m3_metric.get('accuracy', 0.0)):.3f}</b></div>
    <div class="metric-box"><span>OT-graph accuracy change</span><b class="{'ok' if delta_accuracy >= 0 else 'bad'}">{delta_accuracy:+.3f}</b></div>
    <div class="metric-box"><span>Conditional row-max accuracy</span><b>{float(conditional_rowmax_metric.get('accuracy', 0.0)):.3f}</b></div>
  </div>
  <div class="callout warn"><b>Development evaluation:</b> {_esc(development_note)}</div>
</section>
<section class="band">
  <h2>What counts as evidence?</h2>
  <div class="legend">
    <div class="strong"><b>Verified correction</b><br>Node-only is wrong, graph is correct, and the blind visual verifier prefers graph grounding.</div>
    <div class="candidate"><b>Candidate correction</b><br>The binary metric changes from wrong to correct, but the verifier disagrees or the explanation has a visible weakness.</div>
    <div class="boundary"><b>Supporting or boundary evidence</b><br>The graph improves confidence/grounding without a clean wrong-to-right binary correction.</div>
  </div>
  <div class="callout"><b>Decision arithmetic:</b> each case card shows abnormal aggregate A, normal aggregate N, and raw margin A-N for M0 and full conditional OT. With threshold 0.03, margin &gt; 0.03 is abnormal, margin &lt; -0.03 is explicitly normal, and |margin| &le; 0.03 is uncertain; uncertain maps to binary normal for the reported metrics. The separate "oriented margin" flips the sign for normal GT only to show movement toward or away from the correct side.</div>
</section>
{_example_section('Strongest graph-over-node examples', 'These are the cleanest current examples: graph matching corrects the decision and the blind verifier prefers its visual grounding.', proof, 'verified correction', 'good')}
{_example_section('Additional wrong-to-right candidates', 'These cases improve the binary metric, but their verifier result or explanation quality prevents treating them as definitive proof.', candidate_helps, 'candidate correction', 'warn')}
{_example_section('Both methods correct, graph margin improves', 'These supporting cases do not change correctness. They move the competition margin by at least 0.03 toward the GT side and are included for broader inspection.', supporting, 'supporting example', 'info')}
{_example_section('Graph grounding preferred, binary threshold loses', 'These are usually partial/boundary anomalies. The verifier prefers graph grounding, but uncertain maps to binary normal and is scored as a hurt.', grounding_threshold_hurts, 'threshold-sensitive', 'info')}
{_example_section('Other graph regressions', 'These cases should be inspected as failures rather than presented as graph advantage.', other_regressions, 'regression', 'bad')}
<section class="band soft technical">
  <h2>Full method metrics and output files</h2>
  <details open><summary>All six method variants</summary><div class="table-wrap"><table><thead><tr><th>subset</th><th>method</th><th>n</th><th>accuracy</th><th>balanced acc</th><th>AP</th><th>recall</th><th>specificity</th><th>F1</th></tr></thead><tbody>{metrics_html}</tbody></table></div></details>
  <p><a href="summary.json">summary.json</a> | <a href="method_metrics.csv">method_metrics.csv</a> | <a href="method_decomposition.json">method_decomposition.json</a> | <a href="all_graph_help_cases.jsonl">all graph helps</a> | <a href="supporting_margin_examples.jsonl">supporting examples</a> | <a href="grounding_better_threshold_hurts.jsonl">threshold-sensitive cases</a> | <a href="conditional_ot_failures.jsonl">all OT failures</a></p>
</section>
</main></body></html>"""
    (run_dir / "index.html").write_text(document, encoding="utf-8")
    (run_dir / "showcase.html").write_text(document, encoding="utf-8")
    (run_dir / "readable_index.html").write_text(document, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    config = read_json(args.run_dir / "run_config.json", {})
    summary = build_report(args.run_dir, no_images=args.no_images, run_signature=str(config.get("run_signature", "") or "") or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
