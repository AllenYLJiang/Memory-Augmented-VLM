from __future__ import annotations

import hashlib
import html
import json
import random
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import load_baseline_snapshot, load_state_snapshot
from .contracts import MissingLocalData, WaitingForHumanReview, iter_jsonl, read_json, write_json, write_jsonl


def _blind_id(seed: int, uid: str) -> str:
    return "C_" + hashlib.sha256(f"{seed}\0{uid}".encode()).hexdigest()[:12]


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def select_blind_review_cases(rows: list[dict[str, Any]], states: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = int(config.get("unique_windows", 32))
    low, high = config.get("allowed_range", [20, 40])
    if not int(low) <= count <= int(high):
        raise ValueError(f"review count must be in [{low}, {high}]")
    seed, max_per_group = int(config.get("seed", 20260907)), int(config.get("max_windows_per_source_group", 2))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = row["canonical_window_uid"]
        comp = row.get("competitions", {})
        m0 = comp.get("independent_direct_nodes", {}).get("margin")
        m3 = comp.get("conditional_ot_full", {}).get("margin")
        y = row.get("y_true")
        state_wrap = states.get(uid, {}) if uid in states else {}
        state = state_wrap.get("state")
        paired = state_wrap.get("record", {})
        if state is not None and y in (0, 1):
            base_pred = paired.get("base_competitions", {}).get("conditional_ot_full", {}).get("y_pred")
            candidate_pred = paired.get("candidate_competitions", {}).get("conditional_ot_full", {}).get("y_pred")
            base_pred = int(float(m3 or 0) >= .03) if base_pred not in (0, 1) else int(base_pred)
            o = float(state.get("current_window_active_occupancy_probability", 0.0) or 0.0)
            n = float(state.get("current_window_normal_confound_probability", 0.0) or 0.0)
            pseudo = float(m3 or 0.0) + .005 * o - .0025 * n
            if candidate_pred in (0, 1) and int(candidate_pred) != base_pred:
                buckets["legacy_flip_or_pseudo_fp"].append(row)
            elif (y == 0 and (int(pseudo >= .03) == 1 or o >= .5)):
                buckets["legacy_flip_or_pseudo_fp"].append(row)
        if m0 is not None and m3 is not None and (int(float(m0) >= .03) != int(float(m3) >= .03) or abs(float(m3) - .03) <= .015):
            buckets["method_disagreement_or_near_threshold"].append(row)
        high_normal = float((state or {}).get("current_window_normal_confound_probability", 0.0) or 0.0) >= .7
        if y == 1 and m3 is not None and (float(m3) < -.02 or high_normal):
            buckets["far_false_negative_or_high_normal_tail"].append(row)
        buckets["stratified_random"].append(row)
    quotas = [("legacy_flip_or_pseudo_fp", 9), ("method_disagreement_or_near_threshold", 7), ("far_false_negative_or_high_normal_tail", 8), ("stratified_random", 8)]
    selected, used, group_counts = [], set(), defaultdict(int)
    for bucket, quota in quotas:
        candidates = sorted(buckets[bucket], key=lambda r: _rank(seed, r["canonical_window_uid"]))
        taken = 0
        for row in candidates:
            uid, group = row["canonical_window_uid"], str(row.get("source_group", ""))
            if uid in used or group_counts[group] >= max_per_group:
                continue
            selected.append({**row, "review_selection_bucket": bucket})
            used.add(uid); group_counts[group] += 1; taken += 1
            if taken >= quota or len(selected) >= count:
                break
    # Sparse diagnostic buckets donate their unused quota to the deterministic
    # random pool; the requested packet size remains fixed and auditable.
    if len(selected) < count:
        for row in sorted(buckets["stratified_random"], key=lambda r: _rank(seed + 1, r["canonical_window_uid"])):
            uid, group = row["canonical_window_uid"], str(row.get("source_group", ""))
            if uid in used or group_counts[group] >= max_per_group:
                continue
            selected.append({**row, "review_selection_bucket": "stratified_random_backfill"})
            used.add(uid); group_counts[group] += 1
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise MissingLocalData(f"only {len(selected)} unique review windows satisfy source-group limits; requested {count}")
    return selected[:count]


def _extract_clip(video: Path, start: int, end_exclusive: int, out: Path) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    expression = f"select=between(n\\,{start}\\,{end_exclusive - 1}),setpts=N/FRAME_RATE/TB"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-map_metadata", "-1", "-vf", expression, "-an", "-vsync", "vfr", "-c:v", "libx264", "-crf", "20", "-preset", "fast", str(out)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        return False, completed.stderr[-1000:]
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(out)], capture_output=True, text=True)
    try:
        frames = int(probe.stdout.strip())
    except ValueError:
        return False, f"ffprobe frame count unavailable: {probe.stderr[-500:]}"
    expected = end_exclusive - start
    return (frames == expected, f"frames={frames} expected={expected}")


def export_blind_review_packet(legacy_run: Path, work_root: Path, config: Mapping[str, Any], media_root: Path | None = None) -> dict[str, Any]:
    rows, _ = load_baseline_snapshot(legacy_run)
    states, _ = load_state_snapshot(legacy_run)
    cases = select_blind_review_cases(rows, states, config)
    private, public = Path(work_root) / "review_private", Path(work_root) / "review_public"
    private.mkdir(parents=True, exist_ok=True); (public / "clips").mkdir(parents=True, exist_ok=True); (public / "forms").mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 20260907))
    private_rows, public_rows, media_errors = [], [], []
    selection_rows, answer_rows, provenance_rows, reviewer_template = [], [], [], []
    for row in cases:
        uid = row["canonical_window_uid"]
        blind = _blind_id(seed, uid)
        start, end = int(row["start_frame"]), int(row["end_frame"]) + 1
        original_video = Path(str(row.get("video_path", "")))
        video = original_video
        if not video.is_file() and media_root:
            matches = list(Path(media_root).rglob(f"{row['video_id']}.mp4"))
            if matches:
                video = matches[0]
        clip = public / "clips" / f"{blind}.mp4"
        ok, detail = _extract_clip(video, start, end, clip) if video.is_file() else (False, f"media missing: {video}")
        if not ok:
            media_errors.append({"blind_id": blind, "error": detail})
        private_rows.append({"blind_id": blind, "window_uid": uid, "segment_key": row.get("segment_key"), "video_id": row.get("video_id"), "video_path": str(video), "start_frame": start, "end_frame_exclusive": end, "source_group": row.get("source_group"), "historical_split": row.get("historical_split"), "selection_reason": row["review_selection_bucket"], "legacy_y_true": row.get("y_true"), "m0_margin": row.get("competitions", {}).get("independent_direct_nodes", {}).get("margin"), "m3c_margin": row.get("competitions", {}).get("conditional_ot_full", {}).get("margin"), "clip_verified": ok, "clip_verification": detail})
        form = {
            "blind_id": blind, "reviewer_id": "", "review_schema_version": "blind_window_review_v1",
            "viewing": {"full_window_viewed": "pending", "audio_used": False, "context_used": False, "media_problem": None},
            "current_window_visual_label": "pending", "direct_mechanism": "", "event_phase": "pending",
            "visible_event_intervals_local": [], "normal_mechanism": "",
            "normal_explains_suspicious_action": "pending", "same_actor_support": "pending", "same_time_support": "pending",
            "unexplained_active_event_remains": "pending", "eight_sampled_frames_sufficient": "not_assessed",
            "confidence": "pending", "notes": "",
            "context_review": {"status": "pending", "context_used": False, "audio_used": False, "revised_visual_label": "pending", "changed_reason": ""},
            "benchmark_comparison": {"status": "pending", "benchmark_window_label": "pending", "label_disagreement_reason": ""},
        }
        write_json(public / "forms" / f"{blind}.json", form)
        reviewer_template.append(form)
        public_rows.append({"blind_id": blind, "clip": f"clips/{blind}.mp4", "form": f"forms/{blind}.json"})
        selection_rows.append({"blind_id": blind, "selection_reason": row["review_selection_bucket"]})
        answer_rows.append({"blind_id": blind, "legacy_y_true": row.get("y_true"), "m0_margin": private_rows[-1]["m0_margin"], "m3c_margin": private_rows[-1]["m3c_margin"]})
        provenance_rows.append({"blind_id": blind, "window_uid": uid, "source_group": row.get("source_group"), "video_path": str(video), "start_frame": start, "end_frame_exclusive": end, "clip_verification": detail})
    write_jsonl(private / "enrollment_manifest.jsonl", private_rows)
    write_json(private / "original_to_blind_id.json", {r["window_uid"]: r["blind_id"] for r in private_rows})
    write_json(private / "blind_mapping.json", {r["blind_id"]: r["window_uid"] for r in private_rows})
    write_jsonl(private / "selection_reasons.jsonl", selection_rows)
    write_jsonl(private / "original_labels_and_predictions.jsonl", answer_rows)
    write_jsonl(private / "source_provenance.jsonl", provenance_rows)
    write_json(public / "manifest.json", {"version": "blind_review_public_v1", "cases": public_rows})
    write_jsonl(public / "reviewer_template.jsonl", reviewer_template)
    write_json(public / "review_schema.json", {
        "current_window_visual_label": sorted(_VISUAL_LABELS),
        "event_phase": sorted(_EVENT_PHASES),
        "normal_explains_suspicious_action": sorted(_TERNARY_FULL),
        "same_actor_support": sorted(_TERNARY_BINDING),
        "same_time_support": sorted(_TERNARY_BINDING),
        "unexplained_active_event_remains": sorted(_TERNARY_BINDING),
        "confidence": sorted(_CONFIDENCE),
        "full_window_viewed_type": "JSON boolean true, not the string 'true'",
        "benchmark_comparison": "leave pending during blind review; import joins the frozen private label afterward",
        "status_default": "pending", "human_label_use": "audit_only",
    })
    (public / "instructions_zh.md").write_text(
        "# 双阶段盲审\n\n"
        "1. 第一遍只观看完整 96 帧 clip，不查看 private 文件、文件名标签、模型分数或图结构结果。\n"
        "2. `viewing.full_window_viewed` 必须写 JSON 布尔值 `true`，不能写字符串 `\"true\"`。\n"
        "3. 填写视觉标签、事件阶段、正常解释、actor/time binding、异常残余和 confidence。"
        "`direct_mechanism` 与 `normal_mechanism` 应写可见机制，不写抽象类别猜测。\n"
        "4. 允许 `uncertain` 或 `unobservable`，不要为了与预期答案一致而强制二值化。\n"
        "5. `benchmark_comparison` 在盲审期间保持 pending；导入器在锁定盲审答案后从 private 冻结标签接回。\n"
        "6. context/audio 第二遍是可选阶段；未实际查看时保持 pending/false。\n",
        encoding="utf-8",
    )
    cards = "\n".join(f'<article><h2>{html.escape(r["blind_id"])}</h2><video controls preload="metadata" src="{html.escape(r["clip"])}"></video><p>Form: {html.escape(r["form"])}</p></article>' for r in public_rows)
    (public / "index.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Blind review</title><style>body{{font-family:Arial;max-width:1100px;margin:auto}}article{{border-bottom:1px solid #bbb;padding:20px 0}}video{{width:min(900px,100%)}}</style><h1>Blind full-window review</h1>{cards}", encoding="utf-8")
    summary = {"version": "blind_review_export_v1", "requested": len(cases), "playable_complete": sum(r["clip_verified"] for r in private_rows), "media_errors": media_errors, "public_contains_answers": False, "status": "WAITING_FOR_HUMAN_REVIEW" if not media_errors else "WAITING_FOR_LOCAL_MEDIA"}
    write_json(private / "export_summary.json", summary)
    if media_errors:
        import csv
        with (private / "media_missing.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["blind_id", "error"]); writer.writeheader(); writer.writerows(media_errors)
        raise MissingLocalData(f"{len(media_errors)} review clips could not be exported; inspect review_private/export_summary.json")
    raise WaitingForHumanReview("blind packet exported; import at least the configured minimum completed reviews before fitting")


_VISUAL_LABELS = {"anomalous", "normal", "uncertain", "unobservable"}
_EVENT_PHASES = {"active", "aftermath", "pre", "benign", "mixed", "unknown"}
_TERNARY_FULL = {"yes", "no", "partly", "unknown", "not_applicable"}
_TERNARY_BINDING = {"yes", "no", "unknown"}
_CONFIDENCE = {"high", "medium", "low"}


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _coerce_viewed(value: Any) -> tuple[bool | None, str | None]:
    if value is True:
        return True, None
    if isinstance(value, str) and value.strip().lower() == "true":
        return True, "string_true_normalized_to_boolean"
    return None, "full_window_viewed_must_be_json_boolean_true"


def _issue(file: Path, blind_id: str, reviewer_id: str, field: str, code: str, detail: str = "") -> dict[str, Any]:
    return {
        "reviewer_file": str(file), "blind_id": blind_id, "reviewer_id": reviewer_id,
        "field": field, "code": code, "detail": detail,
    }


def _validate_review_row(raw: Mapping[str, Any], file: Path, private: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    row = _canonical_copy(raw)
    blind = str(row.get("blind_id", ""))
    reviewer = str(row.get("reviewer_id", "")).strip()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not reviewer:
        errors.append(_issue(file, blind, reviewer, "reviewer_id", "missing_reviewer_id"))
    if row.get("review_schema_version") != "blind_window_review_v1":
        errors.append(_issue(file, blind, reviewer, "review_schema_version", "unsupported_schema_version", str(row.get("review_schema_version"))))

    viewing = row.get("viewing") if isinstance(row.get("viewing"), dict) else {}
    viewed, normalization = _coerce_viewed(viewing.get("full_window_viewed"))
    if viewed is not True:
        errors.append(_issue(file, blind, reviewer, "viewing.full_window_viewed", normalization or "not_fully_viewed"))
    else:
        viewing["full_window_viewed"] = True
        row["viewing"] = viewing
        if normalization:
            warnings.append(_issue(file, blind, reviewer, "viewing.full_window_viewed", normalization))

    required_enums = {
        "current_window_visual_label": _VISUAL_LABELS,
        "event_phase": _EVENT_PHASES,
        "normal_explains_suspicious_action": _TERNARY_FULL,
        "same_actor_support": _TERNARY_BINDING,
        "same_time_support": _TERNARY_BINDING,
        "unexplained_active_event_remains": _TERNARY_BINDING,
        "confidence": _CONFIDENCE,
    }
    for field, allowed in required_enums.items():
        value = row.get(field)
        if value not in allowed:
            errors.append(_issue(file, blind, reviewer, field, "invalid_or_pending_value", f"value={value!r}; allowed={sorted(allowed)}"))

    intervals = row.get("visible_event_intervals_local")
    if not isinstance(intervals, list):
        errors.append(_issue(file, blind, reviewer, "visible_event_intervals_local", "must_be_list"))
    else:
        for interval in intervals:
            valid = (
                isinstance(interval, (list, tuple)) and len(interval) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in interval)
                and 0 <= interval[0] <= interval[1] <= 95
            )
            if not valid:
                errors.append(_issue(file, blind, reviewer, "visible_event_intervals_local", "invalid_local_interval", repr(interval)))

    visual = row.get("current_window_visual_label")
    normal_explains = row.get("normal_explains_suspicious_action")
    remains = row.get("unexplained_active_event_remains")
    if visual == "anomalous" and normal_explains == "yes" and remains == "no":
        errors.append(_issue(
            file, blind, reviewer, "current_window_visual_label",
            "logical_contradiction_fully_normal_explained_anomaly",
            "anomalous conflicts with normal_explains=yes and unexplained_active_event_remains=no",
        ))
    if visual == "normal" and remains == "yes":
        errors.append(_issue(
            file, blind, reviewer, "current_window_visual_label",
            "logical_contradiction_normal_with_active_anomaly_residual",
        ))

    if not str(row.get("direct_mechanism", "")).strip():
        warnings.append(_issue(file, blind, reviewer, "direct_mechanism", "empty_explanatory_text"))
    if not str(row.get("normal_mechanism", "")).strip():
        warnings.append(_issue(file, blind, reviewer, "normal_mechanism", "empty_explanatory_text"))
    if visual == "anomalous" and row.get("event_phase") == "active" and not intervals:
        warnings.append(_issue(file, blind, reviewer, "visible_event_intervals_local", "active_anomaly_has_no_local_interval"))

    # Benchmark data is private during the visual pass. Join it only after the
    # blind response has been captured; never ask a reviewer to guess it.
    legacy_y = private.get("legacy_y_true")
    benchmark_label = "anomalous" if legacy_y == 1 else "normal" if legacy_y == 0 else "unknown"
    supplied = row.get("benchmark_comparison") if isinstance(row.get("benchmark_comparison"), dict) else {}
    supplied_label = supplied.get("benchmark_window_label")
    if supplied_label not in (None, "", "pending", benchmark_label):
        warnings.append(_issue(file, blind, reviewer, "benchmark_comparison.benchmark_window_label", "reviewer_value_replaced_by_frozen_private_label", str(supplied_label)))
    disagreement = visual in {"anomalous", "normal"} and benchmark_label in {"anomalous", "normal"} and visual != benchmark_label
    row["benchmark_comparison"] = {
        "status": "joined_post_blind",
        "benchmark_window_label": benchmark_label,
        "label_disagreement_reason": str(supplied.get("label_disagreement_reason", "")),
        "source": "review_private.enrollment_manifest.legacy_y_true",
        "visual_disagrees_with_benchmark": disagreement,
    }
    return row, errors, warnings


def _agreement_summary(by_reviewer: Mapping[str, Mapping[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    summaries = []
    for left, right in combinations(sorted(by_reviewer), 2):
        common = sorted(set(by_reviewer[left]) & set(by_reviewer[right]))
        if not common:
            continue
        labels_left = [str(by_reviewer[left][blind].get("current_window_visual_label")) for blind in common]
        labels_right = [str(by_reviewer[right][blind].get("current_window_visual_label")) for blind in common]
        observed = sum(a == b for a, b in zip(labels_left, labels_right)) / len(common)
        categories = sorted(set(labels_left) | set(labels_right))
        expected = sum(labels_left.count(value) * labels_right.count(value) for value in categories) / (len(common) ** 2)
        kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
        exact = 0
        fields = (
            "viewing", "current_window_visual_label", "direct_mechanism", "event_phase",
            "visible_event_intervals_local", "normal_mechanism", "normal_explains_suspicious_action",
            "same_actor_support", "same_time_support", "unexplained_active_event_remains",
            "eight_sampled_frames_sufficient", "confidence", "notes", "context_review",
        )
        for blind in common:
            a = {field: by_reviewer[left][blind].get(field) for field in fields}
            b = {field: by_reviewer[right][blind].get(field) for field in fields}
            exact += a == b
        summaries.append({
            "reviewer_left": left, "reviewer_right": right, "common_cases": len(common),
            "primary_label_agreement": observed, "cohen_kappa": kappa,
            "exact_record_agreement": exact / len(common), "exact_records": exact,
        })
    return summaries


def _load_adjudications(path: Path | None, expected: set[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    values: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for raw in iter_jsonl(path):
        blind = str(raw.get("blind_id", ""))
        adjudicator = str(raw.get("adjudicator_id", "")).strip()
        label = raw.get("final_visual_label")
        viewed, _ = _coerce_viewed(raw.get("full_window_viewed"))
        if blind not in expected:
            errors.append(_issue(path, blind, adjudicator, "blind_id", "not_an_unresolved_disagreement")); continue
        if blind in values:
            errors.append(_issue(path, blind, adjudicator, "blind_id", "duplicate_adjudication")); continue
        if not adjudicator:
            errors.append(_issue(path, blind, adjudicator, "adjudicator_id", "missing_adjudicator_id")); continue
        if viewed is not True:
            errors.append(_issue(path, blind, adjudicator, "full_window_viewed", "must_be_json_boolean_true")); continue
        if label not in _VISUAL_LABELS:
            errors.append(_issue(path, blind, adjudicator, "final_visual_label", "invalid_or_pending_value")); continue
        if not str(raw.get("rationale", "")).strip():
            errors.append(_issue(path, blind, adjudicator, "rationale", "missing_adjudication_rationale")); continue
        values[blind] = _canonical_copy(raw)
        values[blind]["full_window_viewed"] = True
    return values, errors


def import_blind_reviews(
    work_root: Path,
    reviewer_files: Sequence[Path],
    use_policy: str,
    minimum_completed: int,
    adjudicator_file: Path | None = None,
) -> dict[str, Any]:
    if use_policy != "audit_only":
        raise ValueError("V9 permits human review labels only as audit_only")
    private = {r["blind_id"]: r for r in iter_jsonl(Path(work_root) / "review_private/enrollment_manifest.jsonl")}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reviewer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_file_ids: dict[str, set[str]] = defaultdict(set)
    for path in reviewer_files:
        for row in iter_jsonl(path):
            blind = str(row.get("blind_id", ""))
            if blind not in private:
                errors.append({"file": str(path), "blind_id": blind, "error": "unknown_blind_id"}); continue
            if blind in seen_file_ids[str(path)]:
                errors.append(_issue(path, blind, str(row.get("reviewer_id", "")), "blind_id", "duplicate_blind_id_in_file")); continue
            seen_file_ids[str(path)].add(blind)
            normalized, row_errors, row_warnings = _validate_review_row(row, path, private[blind])
            errors.extend(row_errors); warnings.extend(row_warnings)
            if row_errors:
                continue
            reviewer = str(normalized["reviewer_id"])
            if blind in by_reviewer[reviewer]:
                errors.append(_issue(path, blind, reviewer, "reviewer_id", "reviewer_submitted_duplicate_case")); continue
            by_case[blind].append(normalized)
            by_reviewer[reviewer][blind] = normalized

    raw_disagreements = []
    for blind, reviews in by_case.items():
        labels = sorted({str(r.get("current_window_visual_label")) for r in reviews})
        if len(labels) > 1:
            raw_disagreements.append({"blind_id": blind, "visual_events": labels, "status": "requires_adjudication"})
    disagreement_ids = {item["blind_id"] for item in raw_disagreements}
    adjudications, adjudication_errors = _load_adjudications(adjudicator_file, disagreement_ids)
    errors.extend(adjudication_errors)

    sidecar, unresolved = [], []
    for blind, reviews in by_case.items():
        visual = {str(r.get("current_window_visual_label")) for r in reviews}
        conflict = len(visual) > 1
        adjudication = adjudications.get(blind)
        if conflict and adjudication is None:
            unresolved.append({"blind_id": blind, "visual_events": sorted(visual), "status": "requires_adjudication"})
        sidecar.append({
            "blind_id": blind, "window_uid": private[blind]["window_uid"], "reviews": reviews,
            "agreement": not conflict, "adjudication": adjudication,
            "resolved_visual_label": adjudication.get("final_visual_label") if adjudication else next(iter(visual)),
            "use_policy": "audit_only", "included_in_model_fit": False,
        })
    out = Path(work_root) / "review_adjudication"
    write_jsonl(out / "human_audit_sidecar.jsonl", sidecar)
    write_jsonl(out / "review_repair_checklist.jsonl", errors)
    write_jsonl(out / "review_warnings.jsonl", warnings)
    write_jsonl(out / "adjudication_template.jsonl", [
        {"blind_id": item["blind_id"], "adjudicator_id": "", "full_window_viewed": "pending", "final_visual_label": "pending", "rationale": ""}
        for item in raw_disagreements
    ])
    agreement = _agreement_summary(by_reviewer)
    independence_warning = any(item["exact_record_agreement"] >= .8 for item in agreement)
    if independence_warning:
        warnings.append({
            "code": "reviewer_independence_requires_external_confirmation",
            "detail": "at least one reviewer pair has >=80% exact multi-field records; treat as single-reviewer audit unless independent completion is confirmed",
        })
        write_jsonl(out / "review_warnings.jsonl", warnings)
    benchmark_counts = Counter(
        review["benchmark_comparison"]["benchmark_window_label"]
        for reviews in by_case.values() for review in reviews
    )
    write_json(out / "disagreement_report.json", {
        "errors": errors, "raw_disagreements": raw_disagreements,
        "unresolved_disagreements": unresolved, "adjudications": list(adjudications.values()),
    })
    summary = {
        "version": "blind_review_import_v2", "completed_unique": len(sidecar),
        "minimum_required": minimum_completed, "reviewer_ids": sorted(by_reviewer),
        "reviewer_file_count": len(reviewer_files), "pairwise_agreement": agreement,
        "independence_status": "requires_external_confirmation" if independence_warning else "not_flagged_by_exact_record_check",
        "benchmark_join": "post_blind_from_private_legacy_y_true",
        "benchmark_label_counts_across_reviews": dict(benchmark_counts),
        "unresolved_disagreements": len(unresolved), "adjudicated_disagreements": len(adjudications),
        "errors": len(errors), "warnings": len(warnings), "use_policy": "audit_only",
        "ready_for_fit": len(sidecar) >= minimum_completed and not unresolved and not errors,
    }
    write_json(out / "review_quality_report.json", summary)
    write_json(out / "supervision_scope_findings.json", summary)
    if not summary["ready_for_fit"]:
        raise WaitingForHumanReview(f"review import not ready: {summary}")
    return summary
