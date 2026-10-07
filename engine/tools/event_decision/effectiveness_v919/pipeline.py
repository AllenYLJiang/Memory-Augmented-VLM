"""Three stages with plan/run separation, immutable models, independent evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import shutil

from ..contracts import file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..b1b4_trial.protocol import immutable
from . import VERSION
from .data import build_pilot, build_dense, codes, pure_normal, save_jsonl_new
from .scoring import BASES, LOCAL, features, fit_models, predict, evaluate_pilot
from .metrics import blocks, metrics, paired_bootstrap
from .acquisition import cost_summary, verify_refusals


def implementation(project):
    files = sorted((project / "tools").glob("*.py")) + sorted((project / "tools/event_decision").rglob("*.py"))
    files += sorted((project / "docs").glob("*.py")) + [project / "run_effectiveness_v919.sh"]
    return {str(p.relative_to(project)): file_sha256(p) for p in files}


def freeze(project, out, args):
    existing = read_json(out / "protocol.json")
    if existing:
        verify(project, out)
        for field in ("seed", "pilot_videos", "pilot_windows_per_video", "window", "stride", "workers", "top_k_abnormal", "top_k_normal", "model", "max_output_tokens", "image_max_pixels", "regularization", "bootstrap", "expected_test_videos"):
            if existing[field] != getattr(args, field):
                raise ValueError("Existing TAG has frozen " + field + "; use a new TAG for design changes")
        if file_sha256(Path(args.graph_catalog)) != existing["graph_sha256"]:
            raise ValueError("Existing TAG has another catalog")
        return existing
    catalog_path = Path(args.graph_catalog)
    from graph_catalog import read_catalog_json
    cat = read_catalog_json(catalog_path)
    if sum(g.polarity == "abnormal" for g in cat.values()) < args.top_k_abnormal or sum(g.polarity == "normal" for g in cat.values()) < args.top_k_normal:
        raise ValueError("Candidate counts exceed active catalog")
    if args.top_k_abnormal < 1 or args.top_k_normal < 2 or not 8 <= args.window or not 1 <= args.stride <= args.window:
        raise ValueError("Invalid candidate/window/stride settings")
    if args.pilot_videos < 21 or args.workers < 1 or args.bootstrap < 1 or args.regularization <= 0:
        raise ValueError("Need >=21 pilot sources, positive workers/bootstrap/regularization")
    if not 512 <= args.max_output_tokens <= 8192 or args.image_max_pixels < 4096 or args.expected_test_videos < 1:
        raise ValueError("Invalid output-token, image-pixel, or expected video budget")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "graph_catalog.json"
    if target.exists() and file_sha256(target) != file_sha256(catalog_path):
        raise ValueError("Catalog changed")
    if not target.exists():
        shutil.copyfile(catalog_path, target)
    code = implementation(project)
    immutable(out / "code_inventory.json", code)
    config = {"version": VERSION, "seed": args.seed, "pilot_videos": args.pilot_videos, "pilot_windows_per_video": args.pilot_windows_per_video,
              "window": args.window, "stride": args.stride, "workers": args.workers, "top_k_abnormal": args.top_k_abnormal,
              "top_k_normal": args.top_k_normal, "model": args.model, "temperature": 0., "max_output_tokens": args.max_output_tokens,
              "image_max_pixels": args.image_max_pixels, "coherence_weight": .25, "competition_temperature": .1,
              "regularization": args.regularization, "bootstrap": args.bootstrap, "expected_test_videos": args.expected_test_videos,
              "graph_sha256": file_sha256(target), "code_sha256": semantic_sha256(code),
              "catalog_origin": str(catalog_path.resolve()), "claim_scope": "development_exposed_no_pristine_test_claim",
              "feature_contract": "machine_only_V919_no_human_overlay_no_old_numeric_binding_U",
              "no_discovery": True, "no_verifier": True, "no_C0": True, "no_leave_one_out": True}
    immutable(out / "protocol.json", config)
    return config


def verify(project, out):
    c = read_json(out / "protocol.json")
    if not c:
        raise ValueError("Run stage 1 first")
    current = implementation(project)
    if c["code_sha256"] != semantic_sha256(current):
        from .transport_patch import verify_upgrade
        verify_upgrade(project, out, c, current)
    if file_sha256(out / "graph_catalog.json") != c["graph_sha256"]:
        raise ValueError("Frozen code/catalog changed; preserve this run and use a new TAG")
    for phase in ("pilot", "dense"):
        plan = read_json(out / phase / "plan.json")
        if plan and file_sha256(out / phase / "inputs.jsonl") != plan["manifest_sha256"]:
            raise ValueError("Frozen manifest changed")
        if plan and plan.get("labels_sha256") and file_sha256(out / "private/pilot_labels.jsonl") != plan["labels_sha256"]:
            raise ValueError("Frozen pilot labels changed")
    return c


def plan(out, phase, config):
    from graph_catalog import read_catalog_json
    cat = read_catalog_json(out / "graph_catalog.json")
    counts = [sum(sorted((len(g.nodes) for g in cat.values() if g.polarity == polarity), reverse=True)[:config["top_k_" + polarity]]) for polarity in ("abnormal", "normal")]
    nodes = min(sum(counts), len({n.key for g in cat.values() for n in g.nodes}))
    per = 1 + nodes + config["top_k_abnormal"] + config["top_k_normal"] + 2
    rows = list(iter_jsonl(out / phase / "inputs.jsonl"))
    obj = {"version": VERSION, "phase": phase, "manifest_sha256": file_sha256(out / phase / "inputs.jsonl"),
           "protocol_sha256": file_sha256(out / "protocol.json"), "windows": len(rows), "videos": len({r['video_id'] for r in rows}),
           "logical_calls_upper_per_window": per, "max_unique_nodes_per_window": nodes,
           "physical_attempts_upper_bound": per * len(rows), "output_tokens_upper_bound": per * len(rows) * config["max_output_tokens"],
           "DAG": "1 selector + distinct independent nodes + one joint/candidate + C1 + at most one C2; no repair/verifier/discovery",
           "input_token_cost": None, "cost_limit": "strict physical/output reservations; input token billing only known after provider response",
           "requires_new_explicit_approval": True, "legacy_authorizations_not_used": True,
           "labels_sha256": file_sha256(out / "private/pilot_labels.jsonl") if phase == "pilot" else None}
    immutable(out / phase / "plan.json", obj)
    return obj


def export_features(out, phase):
    rows = list(iter_jsonl(out / phase / "inputs.jsonl"))
    table, missing, refused = [], [], []
    for r in rows:
        p = out / phase / "results" / (r["window_uid"] + ".json")
        result = read_json(p)
        if result is None:
            missing.append(r["window_uid"])
            result = {"window_uid": r["window_uid"]}
        else:
            if result.get("record_sha256") != semantic_sha256({k: v for k, v in result.items() if k != "record_sha256"}):
                raise ValueError("Saved window result changed: " + r["window_uid"])
            if result["input_sha256"] != semantic_sha256(r):
                raise ValueError("Saved window not from this manifest")
            for key, h in result["cache_sha256"].items():
                if file_sha256(out / "cache" / (key + ".json")) != h:
                    raise ValueError("Source request cache changed")
            verify_refusals(out, result)
            if result.get("refusal_sha256"):
                refused.append({"window_uid": r["window_uid"], "request_keys": sorted(result["refusal_sha256"]),
                                "C2_status": result.get("C2_status"), "kept_in_manifest": True})
        table.append(features(result))
    write_jsonl(out / phase / "features.jsonl", table)
    coverage = {"windows": len(rows), "missing_terminal_results": missing,
                "observed_counts": {k: sum(r["values"][k] is not None for r in table) for k in (*BASES, *LOCAL)},
                "local_valid": sum(r["local_valid"] for r in table), "human_features": 0,
                "provider_refused_windows": len(refused), "provider_refusals": refused,
                "not_acquired_features_are_missing_not_normal": True}
    write_json(out / phase / "feature_coverage.json", coverage)
    return table, coverage


def import_exact_cache(out, source, config):
    """Only native request envelopes with identical prompt/media/config identity qualify."""
    if source.resolve() == out.resolve():
        raise ValueError("Cache source must be a separate run")
    copied, rejected = 0, 0
    for path in sorted((source / "cache").glob("*.json")):
        obj = read_json(path)
        body = obj.get("body", {})
        identity = body.get("identity", {})
        if (identity.get("configuration") != config or identity.get("contract") != "v919_exact_request_1" or
            semantic_sha256(body) != obj.get("sha256") or semantic_sha256(identity) != path.stem):
            rejected += 1
            continue
        target = out / "cache" / path.name
        immutable(target, obj)
        copied += 1
    receipt = {"source": str(source.resolve()), "identical_native_envelopes": copied, "rejected": rejected,
               "legacy_prompts_and_human_overlays_not_converted": True, "paid_calls": 0}
    write_json(out / "cache_import.json", receipt)
    return receipt


def prepare(project, out, args):
    config = freeze(project, out, args)
    if getattr(args, "reuse_run", None):
        import_exact_cache(out, Path(args.reuse_run), config)
    if not (out / "pilot/plan.json").exists():
        build_pilot(project, out, Path(args.train_root), Path(args.anchor_root), config)
    table, coverage = export_features(out, "pilot")
    return {"stage": 1, "enrollment": read_json(out / "pilot/enrollment.json"), "budget": plan(out, "pilot", config), "features": coverage,
            "next": "Inspect source/class scope and plan; explicitly approve STAGE=2 ACTION=run to acquire missing machine evidence",
            "new_API_calls": 0, "legacy_C2_automatically_reinterpreted": False}


def fit_and_evaluate(out, config):
    table, coverage = export_features(out, "pilot")
    if coverage["missing_terminal_results"]:
        raise ValueError("Pilot incomplete; resume collection, do not fit on a successful subset")
    rows, labels = list(iter_jsonl(out / "pilot/inputs.jsonl")), list(iter_jsonl(out / "private/pilot_labels.jsonl"))
    model_path = out / "models/frozen.json"
    provenance = {"features_sha256": file_sha256(out / "pilot/features.jsonl"), "labels_sha256": file_sha256(out / "private/pilot_labels.jsonl"),
                  "manifest_sha256": file_sha256(out / "pilot/inputs.jsonl"), "protocol_sha256": file_sha256(out / "protocol.json")}
    old = read_json(model_path)
    if old:
        if old["provenance"] != provenance:
            raise ValueError("Frozen fit data changed")
        models = old
    else:
        models = fit_models(rows, labels, table, config["regularization"])
        models["provenance"] = provenance
        immutable(model_path, models)
    lm = {r["window_uid"]: r for r in labels}
    fm = {r["window_uid"]: r for r in table}
    train_cal = {}
    for role in ("fit", "calibration"):
        rs = [r for r in rows if r["role"] == role and lm[r["window_uid"]]["loss_mask"]]
        ss, ff = predict([fm[r["window_uid"]] for r in rs], models)
        yy = [lm[r["window_uid"]]["target"] for r in rs]
        train_cal[role] = {"windows": len(rs), "not_heldout": True,
            "methods": {k: metrics(yy, v, models["thresholds"][k]) if None not in v else None for k, v in ss.items()}}
    write_json(out / "pilot/fit_calibration_metrics.json", train_cal)
    report = evaluate_pilot(rows, labels, table, models, config["bootstrap"])
    report["six_class_scope_all_roles"] = read_json(out / "pilot/enrollment.json")["six_class_scope_all_roles"]
    report["ready_for_dense_method_test"] &= report["six_class_scope_all_roles"]
    report["cost"] = cost_summary(out, "pilot")
    report["model_sha256"] = file_sha256(model_path)
    write_json(out / "pilot/evaluation.json", report)
    return report


def prepare_test(out, test_root, config, annotations_path=None):
    report = read_json(out / "pilot/evaluation.json")
    if not report or not report.get("ready_for_dense_method_test"):
        raise ValueError("Pilot does not support paid dense expansion; inspect evaluation.json, preserve negative/uncertain results")
    if report["model_sha256"] != file_sha256(out / "models/frozen.json"):
        raise ValueError("Model changed after pilot")
    if not (out / "dense/plan.json").exists():
        build_dense(out, Path(test_root), config, out / "models/frozen.json")
    if annotations_path:
        from gt_annotations import load_annotations
        annotation = load_annotations(Path(annotations_path))
        videos = {r["video_id"]: r for r in iter_jsonl(out / "dense/inputs.jsonl")}
        for vid, r in videos.items():
            key = vid if vid in annotation else vid + ".mp4"
            if key not in annotation and not pure_normal(vid):
                raise ValueError("Annotation preflight missing: " + vid)
            if any(not 0 <= a <= b < r["video_frames"] for a, b in annotation.get(key, [])):
                raise ValueError("Annotation preflight outside exact frame count: " + vid)
        immutable(out / "dense/annotation_identity.json", {"sha256": file_sha256(Path(annotations_path)), "interval_convention": "zero_based_inclusive_in_file; half_open_in_evaluator"})
    return {"stage": 3, "enrollment": read_json(out / "dense/enrollment.json"), "budget": plan(out, "dense", config),
            "next": "Inspect all-video coverage and new budget before STAGE=3 ACTION=run"}


def dense_evaluate(out, annotations_path, config):
    from gt_annotations import load_annotations, label_window
    annotations_path = Path(annotations_path)
    immutable(out / "dense/annotation_identity.json", {"sha256": file_sha256(annotations_path), "interval_convention": "zero_based_inclusive_in_file; half_open_in_evaluator"})
    annotations = load_annotations(annotations_path)
    models = read_json(out / "models/frozen.json")
    if file_sha256(out / "models/frozen.json") != read_json(out / "dense/enrollment.json")["model_sha256"]:
        raise ValueError("Dense scoring requires the frozen pilot model")
    table, coverage = export_features(out, "dense")
    rows = list(iter_jsonl(out / "dense/inputs.jsonl"))
    scores, fallback = predict(table, models)
    missing = {k: [r["window_uid"] for r, s in zip(rows, v) if s is None] for k, v in scores.items()}
    write_json(out / "dense/coverage.json", {"missing_scores": missing, "features": coverage,
        "fallback_counts": {k: dict(Counter(v)) for k, v in fallback.items()}, "all_methods_complete": not any(missing.values())})
    write_jsonl(out / "dense/window_scores.jsonl", [{"window_uid": r["window_uid"], "scores": {k: v[i] for k, v in scores.items()},
        "fallback": {k: v[i] for k, v in fallback.items()}} for i, r in enumerate(rows)])
    if any(missing.values()) or coverage["missing_terminal_results"]:
        raise ValueError("No full-dataset metrics: missing windows/scores remain explicitly recorded in dense/coverage.json")
    by_video = defaultdict(list)
    for i, r in enumerate(rows):
        by_video[r["video_id"]].append(i)
    agg = {k: [] for k in scores}
    group_ids, video_ids, window_gt, boundary = [], [], [], []
    per_video = []
    for vid, ix in by_video.items():
        rs = [rows[i] for i in ix]
        key = vid if vid in annotations else vid + ".mp4"
        if key not in annotations and not pure_normal(vid):
            raise ValueError("Missing annotation for anomalous video: " + vid)
        intervals = [(a, b + 1) for a, b in annotations.get(key, [])]
        current = {"video_id": vid, "source_group": rs[0]["source_group"], "codes": codes(vid), "pure_normal": pure_normal(vid), "methods": {}}
        for k, values in scores.items():
            bs = blocks(rs[0]["video_frames"], rs, [values[i] for i in ix], intervals)
            agg[k].extend(bs)
            yy, ss, ww = zip(*bs)
            current["methods"][k] = metrics(yy, ss, models["thresholds"][k], ww)
            if k == next(iter(scores)):
                group_ids.extend([rs[0]["source_group"]] * len(bs))
                video_ids.extend([vid] * len(bs))
        per_video.append(current)
    # Window labels are separate metrics: contiguous overlap >=8, and >2/3 core.
    for r in rows:
        key = r["video_id"] if r["video_id"] in annotations else r["video_id"] + ".mp4"
        gt = label_window(key, r["start_frame"], r["end_frame_exclusive"] - 1, annotations, known_normal=pure_normal(r["video_id"]))
        window_gt.append(gt)
        boundary.append(gt["boundary_position"] != "none")
    report = {"scope": "XD_test_development_exposed", "videos": len(by_video), "windows": len(rows), "methods": {},
              "aggregation": "exact_overlap_mean_no_unobserved_gap_filling; weighted_constant_frame_blocks", "comparisons": {},
              "threshold_scope": "frozen_weak_window_calibration; not_optimized_on_frame_GT", "per_category": {},
              "cost": cost_summary(out, "dense"), "model_sha256": file_sha256(out / "models/frozen.json")}
    for k, bs in agg.items():
        yy, ss, ww = zip(*bs)
        report["methods"][k] = {"frame": metrics(yy, ss, models["thresholds"][k], ww),
                               "window_operational_overlap_ge8": metrics([g["y_true"] for g in window_gt], scores[k], models["thresholds"][k]),
                               "window_core_overlap_gt_two_thirds": metrics([g["y_true_core"] for g in window_gt], scores[k], models["thresholds"][k])}
        bix = [i for i, v in enumerate(boundary) if v]
        report["methods"][k]["boundary_windows"] = metrics([window_gt[i]["y_true"] for i in bix], [scores[k][i] for i in bix], models["thresholds"][k]) if bix else None
    for code in ["A", "B1", "B2", "B4", "B5", "B6", "G", "all_anomaly_videos"]:
        take = [i for i, vid in enumerate(video_ids) if (pure_normal(vid) if code == "A" else bool(codes(vid)) if code == "all_anomaly_videos" else code in codes(vid))]
        report["per_category"][code] = {k: metrics([bs[i][0] for i in take], [bs[i][1] for i in take], models["thresholds"][k], [bs[i][2] for i in take]) for k, bs in agg.items()} if take else {}
    for a, b in (("C_calibrated", "D_local_OT"), ("D_local_nodes", "D_local_OT"), ("shared_unary_rowmax", "unary_ot")):
        yy, av, ww = zip(*agg[a])
        report["comparisons"][b + "_minus_" + a] = paired_bootstrap(yy, av, [r[1] for r in agg[b]], group_ids, ww, repeats=config["bootstrap"])
    report["mean_calls_per_video"] = report["cost"]["physical_attempts"] / len(by_video)
    report["mean_calls_per_window"] = report["cost"]["physical_attempts"] / len(rows)
    write_jsonl(out / "dense/per_video_metrics.jsonl", per_video)
    write_json(out / "dense/evaluation.json", report)
    return report
