#!/usr/bin/env python3
"""Resumable sampled run for independent nodes versus two-stage conditional OT graphs."""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
from concurrent import futures
from pathlib import Path
from typing import Dict

from common import append_jsonl, case_id, file_sha256, iter_jsonl, read_json, stable_sha1, write_json, write_jsonl
from decision_policy import POLICIES, load_policy_config
from discover_ot_failures import LiveDiscovery
from graph_catalog import read_catalog_json
from gt_annotations import label_window, label_window_from_source_anchor, load_annotations
from live_matching import PRIMARY_GRAPH_METHOD, WindowMatcher
from prompts import PROMPT_VERSION
from render_report import build_report, render_case
from schemas import WindowCase
from selection import (
    discover_prediction_files,
    label_codes,
    is_pure_normal_video,
    load_deduplicated_records,
    sample_video_ids,
    select_windows,
    source_group_id,
    write_selection,
)
from split_provenance import load_split_registry, resolve_split_provenance
from video_completion import audit_video_completion
from vlm_runtime import CachedVideoVLM, RuntimeConfig
from cache_recovery import archive_file
from temporal_contract import CONTRACT_VERSION, window_errors
from prompts import LEGACY_PROMPT_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage conditional OT graph-vs-node experiment")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--graph-catalog", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--video-sample-fraction", type=float, default=0.1)
    parser.add_argument("--video-sample-seed", type=int, default=42)
    parser.add_argument("--max-windows-per-video", type=int, default=1)
    parser.add_argument("--max-total-windows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--evidence-mode", choices=("frames", "video"), default="frames")
    parser.add_argument("--evidence-frames", type=int, default=8)
    parser.add_argument("--evidence-slots-per-bin", type=int, default=2)
    parser.add_argument("--decision-margin", type=float, default=0.03)
    parser.add_argument("--decision-policy", choices=sorted(POLICIES), default="legacy_graph")
    parser.add_argument("--decision-policy-file", type=Path)
    parser.add_argument("--window-manifest", type=Path)
    parser.add_argument(
        "--split-registry", type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "development_splits.json",
    )
    parser.add_argument(
        "--split-role",
        choices=("development", "calibration", "validation", "final_evaluation"),
        default="development",
    )
    parser.add_argument("--coherence-weight", type=float, default=0.25)
    parser.add_argument("--competition-aggregation", choices=("max", "logmeanexp"), default="logmeanexp")
    parser.add_argument("--competition-temperature", type=float, default=0.1)
    parser.add_argument("--top-k-abnormal", type=int, default=2)
    parser.add_argument("--top-k-normal", type=int, default=4)
    parser.add_argument("--shortlist-pool-multiplier", type=float, default=1.0)
    parser.add_argument("--frozen-source-pair", action="store_true")
    parser.add_argument("--exclude-videos-file", type=Path)
    parser.add_argument("--exclude-source-groups", action="store_true")
    parser.add_argument("--gt-min-overlap-frames", type=int, default=8)
    parser.add_argument("--gt-core-overlap-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--gt-label-source",
        choices=("frame_annotations", "source_record_anchor"),
        default="frame_annotations",
        help="Explicit GT contract. source_record_anchor is restricted to provenance-checked training anchors.",
    )
    parser.add_argument("--source-window", type=int, default=96)
    parser.add_argument("--source-stride", type=int, default=16)
    parser.add_argument("--metric-min-video-completion", type=float, default=0.99)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--run-leave-one-out", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--repair-baseline-from", type=Path,
                        help="Explicit archived run_config for validated, targeted legacy-cache reuse.")
    parser.add_argument("--recovery-preflight-only", action="store_true")
    parser.add_argument("--discover-live", action="store_true")
    parser.add_argument("--discovery-out-dir", type=Path)
    parser.add_argument("--discovery-registry", type=Path)
    parser.add_argument("--max-discovery-cases", type=int, default=0)
    parser.add_argument("--discovery-cases-per-video", type=int, default=1)
    parser.add_argument("--teacher-model", default="deepseek-v4-pro")
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    parser.add_argument("--teacher-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--inference-replicate-id", default="",
        help="Explicit cache salt. Reusing the same value resumes; changing it forces fresh VLM calls.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reconcile_error_log(path: Path, out_dir: Path, run_signature: str) -> dict:
    """Keep the active log limited to unresolved windows while archiving history."""
    if not path.is_file():
        return {"historical_rows": 0, "unresolved_rows": 0, "archive": ""}
    historical = list(iter_jsonl(path))
    latest = {str(row.get("segment_key", "")): row for row in historical if row.get("segment_key")}
    completed = set()
    for record_path in (out_dir / "records").glob("*.json"):
        row = read_json(record_path, None)
        if isinstance(row, dict) and row.get("run_signature") == run_signature:
            completed.add(str(row.get("segment_key", "")))
    unresolved = [row for key, row in latest.items() if key not in completed]
    archive = archive_file(path, out_dir, "historical error log before resolved-error reconciliation")
    write_jsonl(path, unresolved)
    return {"historical_rows": len(historical), "unresolved_rows": len(unresolved), "archive": str(archive)}


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    names = (
        "matching.py", "sinkhorn_ot.py", "competition.py", "prompts.py", "live_matching.py",
        "decision_policy.py", "failure_router.py",
        "temporal_contract.py", "vlm_runtime.py", "cache_recovery.py",
    )
    return {name: file_sha256(root / name) for name in names}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog_json(args.graph_catalog)
    graph_catalog_sha256 = file_sha256(args.graph_catalog)
    annotations_sha256 = file_sha256(args.annotations)
    exclusion_sha256 = file_sha256(args.exclude_videos_file) if args.exclude_videos_file and args.exclude_videos_file.is_file() else ""
    decision_policy_sha256 = file_sha256(args.decision_policy_file) if args.decision_policy_file else ""
    window_manifest_sha256 = file_sha256(args.window_manifest) if args.window_manifest else ""
    split_registry = load_split_registry(args.split_registry)
    split_registry_sha256 = file_sha256(args.split_registry)
    split_provenance = resolve_split_provenance(
        args.video_sample_seed, args.split_role, split_registry, args.split_registry,
    )
    decision_policy_config = load_policy_config(
        args.decision_policy_file, args.decision_policy, args.decision_margin,
    )
    prediction_files = discover_prediction_files(args.source_run)
    source_records, input_stats = load_deduplicated_records(prediction_files)
    code_hashes = _code_hashes()

    signature_parts = [
        "multi_candidate_conditional_ot_v3",
        PROMPT_VERSION,
        graph_catalog_sha256,
        annotations_sha256,
        exclusion_sha256,
        decision_policy_sha256,
        window_manifest_sha256,
        split_registry_sha256,
        input_stats.get("prediction_file_sha256"),
        code_hashes,
        args.model,
        args.fps,
        args.evidence_mode,
        args.evidence_frames,
        args.evidence_slots_per_bin,
        args.decision_margin,
        decision_policy_config,
        args.coherence_weight,
        args.competition_aggregation,
        args.competition_temperature,
        args.top_k_abnormal,
        args.top_k_normal,
        args.shortlist_pool_multiplier,
        args.frozen_source_pair,
        args.video_sample_fraction,
        args.video_sample_seed,
        args.split_role,
        args.max_windows_per_video,
        args.max_total_windows,
        args.gt_min_overlap_frames,
        args.gt_core_overlap_threshold,
        args.gt_label_source,
        args.source_window,
        args.source_stride,
        args.metric_min_video_completion,
        args.no_verifier,
        args.run_leave_one_out,
        args.mock,
        args.inference_replicate_id,
    ]
    run_signature = stable_sha1(*signature_parts, size=40)
    repair_signature = None
    if args.repair_baseline_from:
        previous = read_json(args.repair_baseline_from, {})
        repair_signature = validate_recovery_config(signature_parts, previous, code_hashes)
        print(f"[recovery] validated old run signature={repair_signature}; new={run_signature}", flush=True)
    if args.recovery_preflight_only:
        if not repair_signature:
            raise ValueError("recovery preflight requires --repair-baseline-from")
        print("[recovery-preflight] settings/signature compatible; no API calls or result replacement", flush=True)
        return 0

    if args.window_manifest:
        manifest_rows = list(iter_jsonl(args.window_manifest))
        requested_keys = [str(value.get("segment_key", "")) for value in manifest_rows]
        if not requested_keys or any(not value for value in requested_keys):
            raise SystemExit("--window-manifest must contain non-empty segment_key values")
        if len(requested_keys) != len(set(requested_keys)):
            raise SystemExit("--window-manifest contains duplicate segment_key values")
        source_by_key = {str(value.get("segment_key", "")): value for value in source_records}
        missing = [value for value in requested_keys if value not in source_by_key]
        if missing:
            raise SystemExit(
                f"--window-manifest contains {len(missing)} segment keys absent from source records; first={missing[0]}"
            )
        selected = [source_by_key[value] for value in requested_keys]
        selected_video_ids = {str(value.get("video_id", "")) for value in selected}
        sample_manifest = {
            "version": "exact_window_manifest_v1",
            "policy": "exact segment keys supplied by --window-manifest",
            "window_manifest": str(args.window_manifest),
            "window_manifest_sha256": window_manifest_sha256,
            "fraction": None,
            "seed": int(args.video_sample_seed),
            "selected_union_videos": len(selected_video_ids),
            "selected_videos": [
                {"video_id": value, "labels": sorted(label_codes(value)), "selected_by": ["window_manifest"]}
                for value in sorted(selected_video_ids)
            ],
            "class_stats": {},
        }
    else:
        selected_video_ids, sample_manifest = sample_video_ids(
            source_records, args.video_sample_fraction, args.video_sample_seed,
        )
        selected = None
    excluded_video_ids, excluded_groups = set(), set()
    if args.exclude_videos_file and args.exclude_videos_file.is_file():
        excluded_video_ids = {
            line.strip() for line in args.exclude_videos_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        excluded_groups = {source_group_id(value) for value in excluded_video_ids}
        if args.exclude_source_groups:
            selected_video_ids = {
                video_id for video_id in selected_video_ids if source_group_id(video_id) not in excluded_groups
            }
        else:
            selected_video_ids -= excluded_video_ids
        if args.window_manifest:
            excluded_requested = {
                str(value.get("video_id", "")) for value in selected
                if (
                    source_group_id(str(value.get("video_id", ""))) in excluded_groups
                    if args.exclude_source_groups else str(value.get("video_id", "")) in excluded_video_ids
                )
            }
            if excluded_requested:
                raise SystemExit(
                    "--window-manifest conflicts with exclusion policy for videos: "
                    + ", ".join(sorted(excluded_requested)[:5])
                )
    sample_manifest["exclude_videos_file"] = str(args.exclude_videos_file or "")
    sample_manifest["exclude_source_groups"] = bool(args.exclude_source_groups)
    sample_manifest["excluded_source_groups"] = sorted(excluded_groups)
    sample_manifest["selected_videos"] = [
        item for item in sample_manifest.get("selected_videos", [])
        if str(item.get("video_id", "")) in selected_video_ids
    ]
    sample_manifest["selected_union_videos"] = len(selected_video_ids)
    for code, stats in sample_manifest.get("class_stats", {}).items():
        stats["union_videos_containing_class"] = sum(code in label_codes(video_id) for video_id in selected_video_ids)

    if selected is None:
        selected = select_windows(
            source_records,
            selected_video_ids,
            args.video_sample_seed,
            args.max_windows_per_video,
            args.max_total_windows,
            set(catalog),
            require_source_pair=args.frozen_source_pair,
        )
    completion_source_records = (
        [
            value for value in source_records
            if str(value.get("video_id", "")) in selected_video_ids
        ]
        if args.window_manifest else source_records
    )
    completion_rows = audit_video_completion(completion_source_records, args.source_window, args.source_stride)
    completion_by_video = {row["video_id"]: row for row in completion_rows}
    _write_csv(args.out_dir / "video_completion_audit.csv", completion_rows)
    annotations = load_annotations(args.annotations)
    prepared = []
    for source in selected:
        value = dict(source)
        known_normal = is_pure_normal_video(str(value.get("video_id", "")))
        if args.gt_label_source == "source_record_anchor":
            if not args.window_manifest:
                raise SystemExit("source_record_anchor GT requires --window-manifest")
            try:
                gt = label_window_from_source_anchor(
                    value, core_overlap_threshold=args.gt_core_overlap_threshold,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            gt = label_window(
                str(value.get("video_id", "")),
                int(value.get("start_frame", 0)),
                int(value.get("end_frame", 0)),
                annotations,
                min_overlap_frames=args.gt_min_overlap_frames,
                core_overlap_threshold=args.gt_core_overlap_threshold,
                known_normal=known_normal,
            )
            gt["label_available"] = bool(gt["annotation_found"] or gt["known_normal"])
            gt["ground_truth_source"] = (
                "frame_annotations" if gt["annotation_found"] else "pure_normal_video_label"
            )
            if not gt["annotation_found"]:
                if args.window_manifest and not gt["known_normal"]:
                    raise SystemExit(f"manifest segment has no frame annotation: {value.get('segment_key')}")
                if not gt["known_normal"]:
                    continue
        completion = completion_by_video.get(str(value.get("video_id", "")), {})
        full_source_video_completion_eligible = (
            float(args.metric_min_video_completion) <= 0.0
            or (
                completion.get("num_frames_source") == "video_metadata"
                and float(completion.get("coverage", 0.0)) >= float(args.metric_min_video_completion)
            )
        )
        candidate_directed_exact_window_eligible = bool(args.window_manifest)
        natural_distribution_eligible = bool(
            not args.window_manifest
            and float(args.video_sample_fraction) >= 1.0
            and int(args.max_windows_per_video) == 0
            and int(args.max_total_windows) == 0
        )
        evaluation_scope = {
            "candidate_directed_exact_window_eligible": candidate_directed_exact_window_eligible,
            "full_source_video_completion_eligible": bool(full_source_video_completion_eligible),
            "natural_distribution_eligible": natural_distribution_eligible,
            "pure_normal_coverage": bool(known_normal),
            "dense_temporal_coverage": bool(full_source_video_completion_eligible),
        }
        value["_corrected_gt"] = gt
        value["_video_completion"] = completion
        value["_evaluation_scope"] = evaluation_scope
        value["_metric_eligible"] = bool(
            candidate_directed_exact_window_eligible or full_source_video_completion_eligible
        )
        prepared.append(value)

    write_selection(args.out_dir, prepared, sample_manifest, input_stats, require_source_pair=args.frozen_source_pair)
    gt_audit_rows = []
    for value in prepared:
        gt = value.get("_corrected_gt", {})
        source_truth = value.get("y_true")
        gt_audit_rows.append({
            "segment_key": value.get("segment_key"),
            "video_id": value.get("video_id"),
            "start_frame": value.get("start_frame"),
            "end_frame": value.get("end_frame"),
            "source_y_true": source_truth,
            "y_true_operational": gt.get("y_true_operational"),
            "y_true_core": gt.get("y_true_core"),
            "source_vs_operational_mismatch": (
                "" if source_truth is None else int(int(source_truth) != int(gt.get("y_true_operational", 0)))
            ),
            "max_contiguous_overlap_frames": gt.get("max_contiguous_overlap_frames"),
            "union_overlap_frames": gt.get("union_overlap_frames"),
            "overlap_fraction": gt.get("overlap_fraction"),
            "temporal_subset": gt.get("temporal_subset"),
            "event_phase": gt.get("event_phase"),
            "boundary_position": gt.get("boundary_position"),
            "distance_to_event_start_frames": gt.get("distance_to_event_start_frames"),
            "distance_to_event_end_frames": gt.get("distance_to_event_end_frames"),
            "known_normal": gt.get("known_normal"),
            "label_available": gt.get("label_available"),
            "ground_truth_source": gt.get("ground_truth_source"),
            "anchor_status": gt.get("anchor_status"),
            "anchor_confidence": gt.get("anchor_confidence"),
            "metric_eligible": value.get("_metric_eligible"),
            "video_completion": value.get("_video_completion", {}).get("coverage"),
        })
    _write_csv(args.out_dir / "gt_label_audit.csv", gt_audit_rows)
    run_config = {
        "version": "multi_candidate_conditional_ot_v3",
        "prompt_version": PROMPT_VERSION,
        "source_run": str(args.source_run),
        "prediction_files": [str(path) for path in prediction_files],
        "prediction_file_sha256": input_stats.get("prediction_file_sha256", {}),
        "graph_catalog": str(args.graph_catalog),
        "graph_catalog_sha256": graph_catalog_sha256,
        "annotations_sha256": annotations_sha256,
        "exclusion_sha256": exclusion_sha256,
        "code_hashes": code_hashes,
        "run_signature": run_signature,
        "temporal_contract_version": CONTRACT_VERSION,
        "recovery": {"source_config": str(args.repair_baseline_from or ""),
                     "source_run_signature": repair_signature,
                     "policy": "validated_legacy_reuse_plus_targeted_repair" if repair_signature else "fresh_contract"},
        "code_dir": str(args.code_dir),
        "annotations": str(args.annotations),
        "video_sample_fraction": args.video_sample_fraction,
        "video_sample_seed": args.video_sample_seed,
        "max_windows_per_video": args.max_windows_per_video,
        "max_total_windows": args.max_total_windows,
        "workers": args.workers,
        "model": args.model,
        "inference_replicate_id": args.inference_replicate_id,
        "fps": args.fps,
        "evidence_mode": args.evidence_mode,
        "evidence_frames": args.evidence_frames,
        "evidence_slots_per_bin": args.evidence_slots_per_bin,
        "decision_margin": args.decision_margin,
        "decision_policy": decision_policy_config,
        "decision_policy_file": str(args.decision_policy_file or ""),
        "decision_policy_file_sha256": decision_policy_sha256,
        "window_manifest": str(args.window_manifest or ""),
        "window_manifest_sha256": window_manifest_sha256,
        "split_provenance": split_provenance,
        "split_registry_sha256": split_registry_sha256,
        "coherence_weight": args.coherence_weight,
        "competition_aggregation": args.competition_aggregation,
        "competition_temperature": args.competition_temperature,
        "top_k_abnormal": args.top_k_abnormal,
        "top_k_normal": args.top_k_normal,
        "shortlist_pool_multiplier": args.shortlist_pool_multiplier,
        "candidate_mode": "frozen_source_pair" if args.frozen_source_pair else "blind_catalog_shortlist",
        "exclude_videos_file": str(args.exclude_videos_file or ""),
        "exclude_source_groups": args.exclude_source_groups,
        "gt_min_overlap_frames": args.gt_min_overlap_frames,
        "gt_core_overlap_threshold": args.gt_core_overlap_threshold,
        "gt_label_source": args.gt_label_source,
        "source_window": args.source_window,
        "source_stride": args.source_stride,
        "metric_min_video_completion": args.metric_min_video_completion,
        "mock": args.mock,
        "run_leave_one_out": args.run_leave_one_out,
        "discover_live": args.discover_live,
        "max_discovery_cases": args.max_discovery_cases,
        "discovery_cases_per_video": args.discovery_cases_per_video,
        "selected_windows": len(prepared),
        "metric_eligible_windows": sum(bool(value.get("_metric_eligible")) for value in prepared),
        "evaluation_scope": {
            key: sum(bool(value.get("_evaluation_scope", {}).get(key)) for value in prepared)
            for key in (
                "candidate_directed_exact_window_eligible",
                "full_source_video_completion_eligible",
                "natural_distribution_eligible",
                "pure_normal_coverage",
                "dense_temporal_coverage",
            )
        },
    }
    write_json(args.out_dir / "run_config.json", run_config)
    print(
        f"[selection] source_windows={input_stats['unique_segments']} sample_videos={sample_manifest['selected_union_videos']} "
        f"selected_windows={len(prepared)} metric_eligible={run_config['metric_eligible_windows']} "
        f"fraction={args.video_sample_fraction}", flush=True,
    )

    local = threading.local()
    progress_lock = threading.Lock()
    progress = {"done": 0}

    discovery_session = discovery_queue = discovery_thread = None
    discovery_lock = threading.Lock()
    discovery_state = {"enqueued": 0, "videos": {}}
    if args.discover_live:
        if args.discovery_out_dir is None or args.discovery_registry is None:
            raise SystemExit("--discover-live requires --discovery-out-dir and --discovery-registry")
        discovery_session = LiveDiscovery(
            out_dir=args.discovery_out_dir,
            graph_catalog=args.graph_catalog,
            registry_path=args.discovery_registry,
            teacher_model=args.teacher_model,
            teacher_base_url=args.teacher_base_url,
            teacher_key_env=args.teacher_key_env,
            mock=args.mock,
        )
        discovery_queue = queue.Queue()

        def discovery_worker() -> None:
            assert discovery_queue is not None and discovery_session is not None
            while True:
                item = discovery_queue.get()
                try:
                    if item is None:
                        return
                    discovery_session.discover(item, discovery_state["enqueued"], 0)
                except Exception as exc:
                    error = {
                        "case_id": item.get("case_id") if isinstance(item, dict) else None,
                        "segment_key": item.get("segment_key") if isinstance(item, dict) else None,
                        "type": type(exc).__name__, "error": str(exc),
                    }
                    append_jsonl(args.discovery_out_dir / "discovery_worker_errors.jsonl", error)
                    print(f"[discovery-worker-error] case={error['case_id']}: {exc}", flush=True)
                finally:
                    discovery_queue.task_done()

        discovery_thread = threading.Thread(target=discovery_worker, name="ot-graph-discovery", daemon=True)
        discovery_thread.start()

    def enqueue_discovery(result: dict) -> None:
        if discovery_queue is None:
            return
        video_id = str(result.get("video_id", ""))
        with discovery_lock:
            if args.max_discovery_cases > 0 and discovery_state["enqueued"] >= args.max_discovery_cases:
                return
            used = int(discovery_state["videos"].get(video_id, 0))
            if args.discovery_cases_per_video > 0 and used >= args.discovery_cases_per_video:
                return
            discovery_state["videos"][video_id] = used + 1
            discovery_state["enqueued"] += 1
        discovery_queue.put(result)

    def get_matcher() -> WindowMatcher:
        matcher = getattr(local, "matcher", None)
        if matcher is None:
            runtime = CachedVideoVLM(RuntimeConfig(
                code_dir=args.code_dir,
                cache_dir=args.out_dir / "cache",
                model=args.model,
                fps=args.fps,
                key_env=args.key_env,
                mock=args.mock,
                evidence_mode=args.evidence_mode,
                evidence_frames=args.evidence_frames,
                cache_salt=args.inference_replicate_id,
                allow_legacy_temporal_cache=bool(repair_signature),
            ))
            matcher = WindowMatcher(
                runtime,
                catalog,
                args.decision_margin,
                top_k_abnormal=args.top_k_abnormal,
                top_k_normal=args.top_k_normal,
                shortlist_pool_multiplier=args.shortlist_pool_multiplier,
                use_catalog_shortlist=not args.frozen_source_pair,
                evidence_slots_per_bin=args.evidence_slots_per_bin,
                coherence_weight=args.coherence_weight,
                competition_aggregation=args.competition_aggregation,
                competition_temperature=args.competition_temperature,
                run_leave_one_out=args.run_leave_one_out,
                decision_policy_config=decision_policy_config,
            )
            local.matcher = matcher
        return matcher

    def process(source: dict) -> dict:
        segment_key = str(source["segment_key"])
        cid = case_id(segment_key)
        record_path = args.out_dir / "records" / f"{cid}.json"
        cached = read_json(record_path, None)
        accepted_signatures = {run_signature} | ({repair_signature} if repair_signature else set())
        compatible = isinstance(cached, dict) and cached.get("segment_key") == segment_key and cached.get("run_signature") in accepted_signatures
        issues = window_errors(cached) if compatible else []
        if compatible and not issues:
            result, state = cached, "cached"
            if cached.get("run_signature") != run_signature:
                archived = archive_file(record_path, args.out_dir, "validated legacy evidence retained without API or rescoring")
                result = dict(cached)
                result["run_signature"] = run_signature
                result["evidence_recovery"] = {"source_run_signature": cached["run_signature"],
                                              "source_record_archive": str(archived),
                                              "mode": "validated_legacy_window_no_rescore",
                                              "contract": CONTRACT_VERSION}
                write_json(record_path, result)
                state = "validated-legacy"
        else:
            if issues:
                archive_file(record_path, args.out_dir, "; ".join(issues), remove=True)
                print(f"[window-repair] {segment_key}: {issues[0]}", flush=True)
            case = WindowCase(
                segment_key=segment_key,
                video_id=str(source.get("video_id", "")),
                video_path=str(source.get("video_path", "")),
                start_frame=int(source.get("start_frame", 0)),
                end_frame=int(source.get("end_frame", 0)),
                y_true=int(source["_corrected_gt"]["y_true_operational"]),
                source_record=source,
            )
            result = get_matcher().match(case, run_verifier=not args.no_verifier)
            result["run_signature"] = run_signature
            result["temporal_contract_version"] = CONTRACT_VERSION
            write_json(record_path, result)
            state = "new"
        comparison = result.get("comparison", {})
        if comparison.get("verified_graph_help"):
            write_json(args.out_dir / "live_graph_advantage" / f"{result['case_id']}.json", result)
            page = render_case(result, args.out_dir, no_images=args.no_images)
            print(f"+ GRAPH>NODE case={result['case_id']} segment={segment_key} page={page}", flush=True)
        if comparison.get("verified_graph_hurt"):
            print(f"! VERIFIED-GRAPH-HURT case={result['case_id']} segment={segment_key}", flush=True)
        if comparison.get("conditional_ot_failure"):
            write_json(args.out_dir / "live_graph_failures" / f"{result['case_id']}.json", result)
            render_case(result, args.out_dir, no_images=args.no_images)
            print(f"- GRAPH-FAIL case={result['case_id']} segment={segment_key}", flush=True)
            enqueue_discovery(result)
        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
        print(f"[window {done}/{len(prepared)}] {state} {segment_key}", flush=True)
        return result

    errors_path = args.out_dir / "errors.jsonl"
    workers = max(1, int(args.workers))
    if workers == 1:
        for source in prepared:
            try:
                process(source)
            except Exception as exc:
                append_jsonl(errors_path, {"segment_key": source.get("segment_key"), "type": type(exc).__name__, "error": str(exc)})
                print(f"[window-error] {source.get('segment_key')}: {exc}", flush=True)
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {executor.submit(process, source): source for source in prepared}
            for future in futures.as_completed(jobs):
                source = jobs[future]
                try:
                    future.result()
                except Exception as exc:
                    append_jsonl(errors_path, {"segment_key": source.get("segment_key"), "type": type(exc).__name__, "error": str(exc)})
                    print(f"[window-error] {source.get('segment_key')}: {exc}", flush=True)

    if discovery_queue is not None:
        discovery_queue.put(None)
        discovery_queue.join()
        if discovery_thread is not None:
            discovery_thread.join()
        assert discovery_session is not None
        summary = discovery_session.finalize(discovery_state["enqueued"])
        print(f"[live-discovery-complete] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    error_status = reconcile_error_log(errors_path, args.out_dir, run_signature)
    if error_status["historical_rows"]:
        print(f"[errors] unresolved={error_status['unresolved_rows']} history={error_status['archive']}", flush=True)

    summary = build_report(args.out_dir, no_images=args.no_images, run_signature=run_signature)
    print(
        f"[complete] windows={summary['n_windows']} videos={summary['n_videos']} "
        f"verified_graph_helps={summary['paired_effects']['verified_graph_helps']} "
        f"verified_graph_hurts={summary['paired_effects']['verified_graph_hurts']} "
        f"report={args.out_dir / 'index.html'}", flush=True,
    )
    return 0


def validate_recovery_config(parts: list, previous: dict, current_hashes: dict) -> str:
    """Only migrate the audited prompt/parser revision, not changed scoring or data."""
    if previous.get("prompt_version") != LEGACY_PROMPT_VERSION:
        raise ValueError("recovery config is not the supported legacy temporal protocol")
    old_hashes = previous.get("code_hashes", {})
    for name in ("matching.py", "sinkhorn_ot.py", "competition.py", "decision_policy.py", "failure_router.py"):
        if not old_hashes.get(name) or old_hashes[name] != current_hashes.get(name):
            raise ValueError(f"recovery refuses changed scoring code: {name}")
    old_parts = list(parts)
    old_parts[1], old_parts[9] = previous["prompt_version"], old_hashes
    signature = stable_sha1(*old_parts, size=40)
    if signature != previous.get("run_signature"):
        raise ValueError("recovery refuses changed data/configuration: legacy signature mismatch; restore the original Step 2 settings")
    return signature


if __name__ == "__main__":
    raise SystemExit(main())
