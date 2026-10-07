from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import read_json, write_json


def classify_readiness(evaluation: Mapping[str, Any], claim_policy: Mapping[str, Any]) -> dict[str, Any]:
    ranking_min = float(claim_policy.get("ranking_min_ap_delta", 0.0))
    viable = []
    for family, data in evaluation.get("families", {}).items():
        delta = data.get("feature_matched", {}).get("delta", {})
        ap_delta = delta.get("ap")
        if family != "F0_m0_calibrated" and ap_delta is not None and ap_delta > ranking_min:
            viable.append((float(ap_delta), family))
    viable.sort(reverse=True)
    if viable:
        ap_delta, family = viable[0]
        status, allowed = "ranking_candidate", "prepare_frozen_ranking_evaluation_plan"
        requested_claim = "incremental_ranking_over_calibrated_M0"
    else:
        ap_delta, family = 0.0, None
        status, allowed, requested_claim = "NO_CANDIDATE", "collect_scope_aligned_hard_normal_and_binding_evidence", "none"
    return {
        "version": "claim_aware_readiness_v1",
        "candidate_id": None if family is None else hashlib.sha256(f"{family}:{ap_delta}".encode()).hexdigest()[:20],
        "status": status, "family": family, "requested_claim": requested_claim,
        "supported_population": "scope_valid_selected_training_windows",
        "label_evidence": ["weak_positive_anchor", "explicit_label_A"],
        "scope_alignment_passed": True, "evidence_origin": "development_nested_group_oof",
        "binding_claim_supported": False, "hard_normal_risk_estimable": False,
        "allowed_next_action": allowed, "remote_execution_authorized": False, "deployment_authorized": False,
        "checks": {"scope_valid_overall_ap_delta_positive": bool(viable), "human_audit_labels_excluded": True, "binding_columns_observed": False},
        "point_estimates": {"ap_delta": ap_delta},
        "insufficiencies": ["insufficient_independent_scope_valid_hard_normals", "binding_not_identifiable_from_current_cache"],
        "prohibited_claims": ["final_XDViolence_AP", "binary_B4_improvement", "causal_binding_verified", "deployment_ready"],
    }


def classify_and_write(work_root: Path, claim_policy: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = read_json(Path(work_root) / "evaluation/scope_aligned_metrics.json")
    if not evaluation:
        raise FileNotFoundError("evaluation/scope_aligned_metrics.json")
    result = classify_readiness(evaluation, claim_policy)
    claims = Path(work_root) / "claims"
    write_json(claims / "readiness.json", result)
    permitted = "- A development-only ranking candidate improved scope-aligned OOF AP over a same-fold calibrated M0 control.\n" if result["status"] == "ranking_candidate" else "- No registered scorer qualified as a development ranking candidate.\n"
    (claims / "permitted_claims.md").write_text("# Permitted claims\n\n" + permitted + "- Any statement is restricted to weak-positive and explicit-label-A training windows.\n", encoding="utf-8")
    (claims / "prohibited_claims.md").write_text("# Prohibited claims\n\n" + "\n".join(f"- {x}" for x in result["prohibited_claims"]) + "\n", encoding="utf-8")
    write_json(claims / "next_data_request.json", {"priority": ["scope-aligned hard normal windows", "post-event normals", "structured same-event binding evidence", "new source-disjoint evaluation population"], "remote_execution_authorized": False})
    return result


def write_report(work_root: Path) -> Path:
    work_root = Path(work_root)
    scope = read_json(work_root / "labels/scope_summary.json", {})
    evaluation = read_json(work_root / "evaluation/scope_aligned_metrics.json", {})
    readiness = read_json(work_root / "claims/readiness.json", {})
    receipts = []
    receipt_path = work_root / "offline_execution_receipts.jsonl"
    if receipt_path.is_file():
        from .contracts import iter_jsonl
        receipts = list(iter_jsonl(receipt_path))
    lines = [
        "# Governed V9 Scope-Aligned Parallel Scorer Report", "",
        "- Label evidence: weak positive anchors plus explicit label_A normals",
        f"- Scope-aligned denominator: {scope.get('supervised', 0)} windows ({scope.get('positive', 0)} positive, {scope.get('known_normal', 0)} normal)",
        "- Scope: selected training windows; development-only nested source-group OOF",
        "- Data history: all V8 calibration/validation records were previously used for system design; this is not an untouched test",
        f"- API/provider calls in this workflow: {sum(int(r.get('logical_provider_calls', 0)) for r in receipts)}",
        f"- Next-step authorization: remote={readiness.get('remote_execution_authorized', False)}, deployment={readiness.get('deployment_authorized', False)}",
        "", "## Readiness", "", f"Status: `{readiness.get('status', 'not_classified')}`. Candidate: `{readiness.get('family')}`.", "", "## Development Results", "",
    ]
    for family, data in evaluation.get("families", {}).items():
        matched = data.get("feature_matched", {})
        lines.append(f"- `{family}`: n={matched.get('n')}, AP delta={matched.get('delta', {}).get('ap')}, BA delta={matched.get('delta', {}).get('balanced_accuracy')}, fallback={data.get('fallback_n')}")
    lines += ["", "## Interpretation Limits", "", "This experiment cannot validate hard-confound handling or actor/event binding because those labels/features are not identifiable in the frozen cache. Human review is audit-only and was not used to fit, tune, or select a model.", ""]
    path = work_root / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

