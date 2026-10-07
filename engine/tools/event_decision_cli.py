#!/usr/bin/env python3
"""Governed V9 offline workflow CLI.

The network guard is installed before importing any stage implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from event_decision.contracts import EventDecisionError, MissingLocalData, read_json, resolve_variables, write_json
from event_decision.safety import OfflineGuard, execution_receipt


TOP_KEYS = {"version", "legacy_tag", "new_tag", "paths", "safety", "baseline", "supervision", "review", "features", "fitting", "evaluation", "execution"}
SECTION_KEYS = {
    "paths": {"legacy_run", "work_root", "base_catalog", "source_records", "select_root", "pipeline_tools", "train_videos_root", "seal_registry"},
    "safety": {"offline_only", "allow_new_baseline", "allow_semantic_extraction", "allow_deepseek", "allow_discovery", "allow_grounding", "allow_rescue", "allow_original_result_mutation"},
    "baseline": {"origin", "methods", "expected_top_k_abnormal", "expected_top_k_normal", "expected_temporal_bins", "expected_capacity_slots_per_bin", "incompatible_contract_policy"},
    "supervision": {"label_rule_id", "window_frames", "positive_anchor_requires_verified_span", "negative_anchor_window_projection", "require_positive_anchor_semantics_declared", "allow_full_coverage_weak_negative", "unknown_supervised_weight", "human_review_use", "retain_legacy_proxy_reports"},
    "review": {"unique_windows", "allowed_range", "seed", "max_windows_per_source_group", "reviewers_preferred", "minimum_completed_unique", "full_clip_required", "context_first_pass", "stop_until_review_imported"},
    "features": {"schema_id", "max_model_dimensions", "missing_semantics", "raw_normal_as_primary_suppressor", "binding_without_structured_evidence", "baseline_resampling"},
    "fitting": {"families", "conditional_family", "regularization_grid", "outer_group_folds", "inner_group_folds", "threshold_pool_fraction", "group_weighting", "class_weighting", "max_iter", "tolerance", "select_by", "seed"},
    "evaluation": {"metric_version", "bootstrap_unit", "bootstrap_repetitions", "bootstrap_seed", "include_legacy_proxy", "include_scope_valid", "include_blind_challenge", "include_whole_packet_with_fallback", "candidate_selection_uses_human_audit_labels"},
    "execution": {"stop_after_review_export", "auto_start_validation", "auto_activate_model"},
}


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.config)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EventDecisionError("config must be a mapping")
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise EventDecisionError(f"unknown config keys: {sorted(unknown)}")
    for section, allowed in SECTION_KEYS.items():
        value = data.get(section, {})
        if not isinstance(value, dict):
            raise EventDecisionError(f"config section {section} must be a mapping")
        extra = set(value) - allowed
        if extra:
            raise EventDecisionError(f"unknown {section} keys: {sorted(extra)}")
    if data.get("version") != "governed_v9_scope_aligned_v1":
        raise EventDecisionError(f"unsupported config version: {data.get('version')}")
    if not data.get("safety", {}).get("offline_only") or any(data.get("safety", {}).get(k) for k in ("allow_new_baseline", "allow_semantic_extraction", "allow_deepseek", "allow_discovery", "allow_grounding", "allow_rescue", "allow_original_result_mutation")):
        raise EventDecisionError("governed V9 config must remain offline and must not authorize mutation or remote stages")
    variables = {
        "project_root": str(Path(args.project_root).resolve()), "xd_root": str(Path(args.xd_root).resolve()),
        "legacy_tag": str(data["legacy_tag"]), "new_tag": str(data["new_tag"]),
    }
    variables["legacy_run"] = resolve_variables(data["paths"]["legacy_run"], variables)
    variables["work_root"] = resolve_variables(data["paths"]["work_root"], variables)
    data = resolve_variables(data, variables)
    if getattr(args, "legacy_run", None):
        data["paths"]["legacy_run"] = args.legacy_run
        data["paths"]["source_records"] = str(Path(args.legacy_run) / "source_disjoint_training_anchors/selected_source_records.jsonl")
        data["legacy_tag"] = Path(args.legacy_run).name
    if getattr(args, "work_root", None):
        data["paths"]["work_root"] = args.work_root
        data["new_tag"] = Path(args.work_root).name
    return data


def common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", required=True)
    sub.add_argument("--project-root", required=True)
    sub.add_argument("--xd-root", required=True)
    sub.add_argument("--legacy-run")
    sub.add_argument("--work-root")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Governed V9 scope-aligned offline workflow")
    commands = ap.add_subparsers(dest="command", required=True)
    for name in ("freeze", "inventory", "audit", "build-label-ledger", "export-features", "build-review", "import-review", "make-splits", "fit", "evaluate", "classify-candidate", "report", "verify-seal", "plan-validation", "plan-semantic-completion"):
        sub = commands.add_parser(name); common(sub)
        if name == "export-features":
            sub.add_argument("--baseline-origin", default="source_frozen"); sub.add_argument("--cache-only", action="store_true")
        elif name == "build-review":
            sub.add_argument("--count", type=int); sub.add_argument("--media-root")
        elif name == "import-review":
            sub.add_argument("--reviewer-file", action="append", required=True)
            sub.add_argument("--adjudicator-file")
            sub.add_argument("--use-policy", default="audit_only")
        elif name == "make-splits":
            sub.add_argument("--role", default="development_nested_group_oof")
        elif name == "fit":
            sub.add_argument("--families", nargs="+", default=None)
    guard = commands.add_parser("guard-legacy-write")
    guard.add_argument("--run-root", required=True); guard.add_argument("--requested-stage", required=True); guard.add_argument("--registry", required=True)
    return ap


def _stage_status(work: Path, stage: str, status: str, detail: Any = None) -> None:
    path = work / "stage_status.json"
    state = read_json(path, {"version": "event_decision_stage_status_v1", "stages": {}})
    state["stages"][stage] = {"status": status, "updated_unix": time.time(), "detail": detail}
    state["latest_stage"] = stage
    write_json(path, state)


def execute(args: argparse.Namespace, cfg: dict[str, Any]) -> Any:
    from event_decision import adapters  # noqa: F401 - imported only after OfflineGuard
    from event_decision.evaluation import audit_snapshot
    from event_decision.feature_store import export_features
    from event_decision.fitting import evaluate_fitted_models, fit_registered_models
    from event_decision.inventory import build_inventory
    from event_decision.label_scope import build_label_ledger
    from event_decision.readiness import classify_and_write, write_report
    from event_decision.review import export_blind_review_packet, import_blind_reviews
    from event_decision.safety import create_v8_seal, verify_v8_seal
    from event_decision.semantic_plan import plan_semantic_completion, plan_validation
    from event_decision.splits import make_grouped_split_plan

    paths = cfg["paths"]
    legacy, work, project = Path(paths["legacy_run"]), Path(paths["work_root"]), Path(args.project_root)
    command = args.command
    if command != "freeze":
        verify_v8_seal(work)
    if command == "freeze":
        result = create_v8_seal(legacy, work, project, Path(paths["base_catalog"]), Path(paths["seal_registry"]))
        write_json(work / "config_snapshot.json", cfg)
        return result
    if command == "inventory":
        return build_inventory(legacy, work, Path(paths["base_catalog"]))
    if command == "audit":
        return audit_snapshot(legacy, work / "audit")
    if command == "build-label-ledger":
        return build_label_ledger(legacy, work, Path(paths["select_root"]), Path(paths["pipeline_tools"]), cfg["supervision"])
    if command == "export-features":
        return export_features(legacy, work, args.baseline_origin, args.cache_only)
    if command == "build-review":
        review = dict(cfg["review"])
        if args.count is not None: review["unique_windows"] = args.count
        return export_blind_review_packet(legacy, work, review, Path(args.media_root) if args.media_root else None)
    if command == "import-review":
        return import_blind_reviews(
            work,
            [Path(p) for p in args.reviewer_file],
            args.use_policy,
            int(cfg["review"]["minimum_completed_unique"]),
            Path(args.adjudicator_file) if args.adjudicator_file else None,
        )
    if command == "make-splits":
        fit_cfg = dict(cfg["fitting"]); fit_cfg["review_minimum_completed"] = cfg["review"]["minimum_completed_unique"]
        return make_grouped_split_plan(work, fit_cfg, args.role)
    if command == "fit":
        return fit_registered_models(work, args.families or cfg["fitting"]["families"], cfg["fitting"])
    if command == "evaluate":
        return evaluate_fitted_models(work, cfg["evaluation"])
    if command == "classify-candidate":
        policy = read_json(project / "config/event_claim_policy_v1.json", {})
        return classify_and_write(work, policy)
    if command == "report":
        return {"report": str(write_report(work))}
    if command == "verify-seal":
        return verify_v8_seal(work)
    if command == "plan-validation":
        return plan_validation(work)
    if command == "plan-semantic-completion":
        return plan_semantic_completion(work)
    raise EventDecisionError(f"unimplemented command: {command}")


def main() -> int:
    args = parser().parse_args()
    guard = OfflineGuard(); guard.install()
    if args.command == "guard-legacy-write":
        from event_decision.safety import refuse_sealed_write
        try:
            refuse_sealed_write(Path(args.run_root), args.requested_stage, Path(args.registry))
            return 0
        except EventDecisionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr); return exc.exit_code
    try:
        cfg = load_config(args)
        work = Path(cfg["paths"]["work_root"])
        _stage_status(work, args.command, "running")
        with execution_receipt(work, args.command, guard):
            result = execute(args, cfg)
            guard.assert_no_remote_calls()
        _stage_status(work, args.command, "completed", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except EventDecisionError as exc:
        if "work" in locals(): _stage_status(work, args.command, "stopped", {"error": str(exc), "exit_code": exc.exit_code})
        print(f"ERROR: {exc}", file=sys.stderr); return exc.exit_code
    except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError) as exc:
        if "work" in locals(): _stage_status(work, args.command, "failed", {"error": f"{type(exc).__name__}: {exc}", "exit_code": 2})
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
