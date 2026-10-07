from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import file_sha256, iter_jsonl, read_json, write_json, write_jsonl


def plan_validation(work_root: Path) -> dict[str, Any]:
    root = Path(work_root)
    readiness = read_json(root / "claims/readiness.json", {})
    current = read_json(root / "feature_store/CURRENT.json", {})
    target = root / "plans/validation"
    if readiness.get("status") not in {"ranking_candidate", "binary_candidate"}:
        result = {"status": "NO_CANDIDATE", "remote_execution_authorized": False, "reason": "readiness did not nominate a candidate"}
        write_json(target / "plan_status.json", result)
        return result
    model_path = root / "models" / str(readiness.get("family")) / "frozen_development_model.json"
    model = read_json(model_path, {}) if model_path.is_file() else {}
    write_json(target / "frozen_model.json", {"candidate_id": readiness.get("candidate_id"), "family": readiness.get("family"), "source": str(model_path), "source_sha256": file_sha256(model_path) if model_path.is_file() else None, "model": model, "immutable_before_external_evaluation": True})
    write_json(target / "frozen_threshold.json", {"ranking_only": readiness.get("status") == "ranking_candidate", "threshold": model.get("threshold") if readiness.get("status") != "ranking_candidate" else None, "source_model_sha256": file_sha256(model_path) if model_path.is_file() else None})
    write_json(target / "expected_feature_contract.json", current)
    write_json(target / "requested_claim.json", {"claim": readiness.get("requested_claim"), "population": "new_source_disjoint_scope_aligned_population"})
    usage_path = root / "splits/historical_source_usage.jsonl"
    excluded = sorted({r["source_group"] for r in iter_jsonl(usage_path)}) if usage_path.is_file() else []
    write_json(target / "required_source_exclusions.json", {"source_groups": excluded})
    write_jsonl(target / "validation_population_manifest.jsonl", [])
    write_json(target / "cache_coverage_report.json", {"new_population_rows": 0, "complete_cached_features": 0, "status": "WAITING_FOR_NEW_EVALUATION_DATA"})
    write_jsonl(target / "missing_feature_requests.jsonl", [])
    write_json(target / "estimated_call_budget.json", {"authorized": False, "estimated_remote_calls": 0, "reason": "planning_only_no_new_population"})
    return {"status": "WAITING_FOR_NEW_EVALUATION_DATA", "remote_execution_authorized": False, "path": str(target)}


def plan_semantic_completion(work_root: Path) -> dict[str, Any]:
    root = Path(work_root)
    current = read_json(root / "feature_store/CURRENT.json", {})
    missing_path = Path(current.get("path", "")) / "missing_requests.jsonl"
    requests = list(iter_jsonl(missing_path)) if missing_path.is_file() else []
    target = root / "plans/semantic_completion"
    write_jsonl(target / "missing_feature_requests.jsonl", requests)
    result = {"status": "BINDING_NOT_IDENTIFIABLE_FROM_CURRENT_CACHE", "requests": len(requests), "estimated_remote_calls": 0, "remote_execution_authorized": False, "execution": "not_implemented_by_design"}
    write_json(target / "plan.json", result)
    return result
