#!/usr/bin/env python3
"""Adversarial DeepSeek critic for graph-memory hypotheses."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from common import iter_jsonl, stable_sha1, write_json, write_jsonl
from discover_ot_failures import DeepSeekClient
from event_constitution import load_constitution
from hypothesis_contract import atomic_contract_errors, bundle_contract_errors


PROMPT_VERSION = "adversarial_graph_critic_v3_validation_readiness"
BLOCKING_CODES = {
    "atomic_contract_violation", "bundle_target_collision", "target_key_mismatch",
    "polarity_contradiction", "nuisance_required_anchor", "non_visual_required_node",
    "absence_only_required_node", "missing_positive_mechanism",
    "internal_semantic_contradiction", "duplicate_existing_graph",
    "counterfactual_inversion", "obviously_overbroad_required_cue",
    "obviously_overstrict_required_cue", "title_scope_mismatch",
}
TERMINAL_REJECT_CODES = {"duplicate_existing_graph"}


def _bundle_id(row: Mapping[str, Any]) -> str:
    revision = row.get("revision", {}) if isinstance(row.get("revision"), Mapping) else {}
    return str(revision.get("bundle_id") or row.get("bundle_id") or "")


def _bundle_context(row: Mapping[str, Any], members: list[dict]) -> dict:
    declared_targets: list[str] = []
    target_dispositions: list[str] = []
    reported_missing: list[str] = []
    for member in members:
        revision = member.get("revision", {}) if isinstance(member.get("revision"), Mapping) else {}
        for value in revision.get("bundle_declared_targets", revision.get("declared_targets", [])):
            text = str(value or "").strip()
            if text and text not in declared_targets:
                declared_targets.append(text)
        for value in revision.get("target_dispositions", []):
            text = str(value or "").strip()
            if text and text not in target_dispositions:
                target_dispositions.append(text)
        for value in revision.get("missing_target_dispositions", []):
            text = str(value or "").strip()
            if text and text not in reported_missing:
                reported_missing.append(text)
    missing_targets = sorted((set(declared_targets) - set(target_dispositions)) | set(reported_missing))
    contract_errors = bundle_contract_errors(members)
    return {
        "bundle_id": _bundle_id(row),
        "bundle_declared_targets": declared_targets,
        "target_dispositions": target_dispositions,
        "missing_target_dispositions": missing_targets,
        "bundle_contract_errors": contract_errors,
        "atomic_hypotheses": [{
            "id": value.get("id"), "operation": value.get("operation"),
            "targets": value.get("targets", []), "status": value.get("status"),
            "reason": value.get("reason"), "graph": value.get("graph", {}),
            "schema_errors": value.get("schema_errors", []),
            "theory_checks": value.get("theory_checks", {}),
        } for value in members],
    }


def _request_fingerprint(row: Mapping[str, Any], constitution: Mapping[str, Any],
                         bundle_context: Mapping[str, Any], gate_policy: str,
                         final_semantic_round: bool) -> str:
    return stable_sha1(
        PROMPT_VERSION,
        json.dumps(row, ensure_ascii=False, sort_keys=True),
        json.dumps(bundle_context, ensure_ascii=False, sort_keys=True),
        json.dumps(constitution, ensure_ascii=False, sort_keys=True),
        gate_policy,
        final_semantic_round,
        size=40,
    )


def _prompt(row: dict, constitution: dict, bundle_context: dict | None = None) -> str:
    schema = {
        "gate_stage": "validation_readiness",
        "recommendation": "accept_for_validation|revise|reject|representation_only",
        "blocking_semantic_defects": [{
            "code": "closed_enum_code", "field_or_node": "", "problem": "",
            "concrete_counterexample": "", "required_fix": "", "confidence": 0.0,
        }],
        "held_out_validation_risks": [{
            "code": "fp_risk|fn_risk|crop_sensitivity|phase_coverage|source_shift|calibration_risk",
            "slice_description": "", "expected_failure": "fp|fn|ranking|none",
            "required_measurement": "", "minimum_exposure_groups": 0,
        }],
        "non_blocking_improvements": [],
        "normal_counterexamples": [], "abnormal_counterexamples": [],
        "nuisance_leakage": [], "duplicate_or_rename_of": [],
        "reason": "", "confidence": 0.0,
    }
    return f"""Act as an adversarial critic of this inactive graph hypothesis.
Check normal and abnormal counterexamples, source-style/text/identity leakage, aliases of existing
graphs, directly visible nodes, polarity semantics, and likely cross-class regressions.

You are reviewing readiness for held-out validation, not activation. Acceptance here never
authorizes deployment. A blocker is a deterministic defect in the graph definition: an atomic or
bundle contract violation, nuisance-based required evidence, non-visual required node, polarity
contradiction, internally inconsistent applicability/falsifier logic, duplicate mutation, or an
explicitly overbroad/overstrict required cue demonstrated by a concrete counterexample.

A plausible regression whose truth depends on frequency, model scoring, crop visibility, source
distribution, event phase, or threshold calibration is a held-out validation risk, not a blocker.
Do not require held-out validation as a semantic revision: the next stage exists to measure those
risks. Return accept_for_validation whenever no blocking defect remains, even when validation risks
are non-empty.

The mutation contract is atomic: an UPDATE must contain exactly one target and the revised graph
key must equal it; ADD must have no target. Required graph nodes must be positive, directly visible
evidence. Absence-only conditions belong in falsifiers, not required nodes. Recommend
Use only these blocker codes:
{json.dumps(sorted(BLOCKING_CODES), ensure_ascii=False)}
Use revise only when at least one blocking_semantic_defect cites an exact field or node and a
concrete required fix. Use reject for terminal duplicate/unsupported concepts, and
representation_only when a stable fixed graph cannot express the missing state.

When bundle context is supplied, all listed atomic hypotheses were emitted together. Treat a
companion target as materialized when it appears in bundle_context.atomic_hypotheses or has an
explicit NOOP disposition. Review the current atomic graph on its own merits, but do not claim a
companion is missing merely because it is not embedded in the current graph object.

Event Constitution:
{json.dumps(constitution, ensure_ascii=False, indent=2)}

Hypothesis:
{json.dumps(row, ensure_ascii=False, indent=2)}

Atomic revision bundle context:
{json.dumps(bundle_context or {}, ensure_ascii=False, indent=2)}

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""


def _valid_blocking_defects(response: Mapping[str, Any]) -> tuple[list[dict], list[str]]:
    raw = response.get("blocking_semantic_defects", [])
    if not isinstance(raw, list):
        return [], ["blocking_semantic_defects_not_a_list"]
    defects, errors = [], []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            errors.append(f"blocking_defect_{index}_not_an_object")
            continue
        code = str(value.get("code", "")).strip()
        if code not in BLOCKING_CODES:
            errors.append(f"unknown_blocking_code:{code or '<empty>'}")
            continue
        defects.append(dict(value))
    return defects, errors


def normalize_critic_decision(
    *, raw_response: Mapping[str, Any], row: Mapping[str, Any], final_round: bool,
    bundle_errors: list[str] | None = None, allow_risk_only: bool = True,
) -> tuple[str, dict]:
    """Map critic prose to a deterministic validation-readiness decision."""
    defects, schema_errors = _valid_blocking_defects(raw_response)
    contract_errors = atomic_contract_errors(row)
    theory = row.get("theory_checks", {}) if isinstance(row.get("theory_checks"), Mapping) else {}
    theory_failed = bool(theory) and not bool(theory.get("passes_hard_invariants", False))

    if contract_errors:
        defects.append({
            "code": "atomic_contract_violation", "field_or_node": "operation/targets/graph",
            "problem": "; ".join(contract_errors), "concrete_counterexample": "",
            "required_fix": "satisfy the atomic graph-mutation contract", "confidence": 1.0,
        })
    if bundle_errors:
        defects.append({
            "code": "bundle_target_collision", "field_or_node": "revision bundle",
            "problem": "; ".join(bundle_errors), "concrete_counterexample": "",
            "required_fix": "reconcile the bundle to one canonical mutation per target",
            "confidence": 1.0,
        })
    if theory_failed:
        defects.append({
            "code": "internal_semantic_contradiction", "field_or_node": "theory_checks",
            "problem": "hard Event Constitution invariants did not pass",
            "concrete_counterexample": "", "required_fix": "repair hard theory violations",
            "confidence": 1.0,
        })

    raw_recommendation = str(raw_response.get("recommendation", "reject"))
    blocker_codes = {str(value.get("code", "")) for value in defects}
    schema_complete = "blocking_semantic_defects" in raw_response and not schema_errors
    risks = raw_response.get("held_out_validation_risks", [])
    risks = risks if isinstance(risks, list) else []

    if blocker_codes & TERMINAL_REJECT_CODES:
        decision, reason = "reject", "terminal_blocking_defect"
    elif defects:
        decision = "human_review_required" if final_round else "revise"
        reason = "blocking_semantic_defect_at_final_round" if final_round else "blocking_semantic_defect"
    elif not schema_complete:
        decision = "human_review_required" if final_round else "revise"
        reason = "critic_schema_incomplete"
    elif raw_recommendation == "representation_only":
        decision, reason = "representation_only", "critic_identified_representation_limit"
    elif allow_risk_only:
        decision = "accept_for_validation"
        reason = "risk_only_no_blocking_defect" if risks or raw_recommendation == "revise" else "no_blocking_defect"
    else:
        decision = raw_recommendation if raw_recommendation in {
            "accept_for_validation", "revise", "reject", "representation_only"
        } else "reject"
        reason = "raw_recommendation_risk_promotion_disabled"

    return decision, {
        "raw_recommendation": raw_recommendation,
        "normalized_recommendation": decision,
        "normalization_reason": reason,
        "blocking_semantic_defects": defects,
        "held_out_validation_risks": risks,
        "critic_schema_errors": schema_errors,
    }


def critique(input_path: Path, constitution_path: Path, out_dir: Path, *, model: str,
             base_url: str, key_env: str, retries: int, mock: bool,
             bundle_aware: bool = False, gate_policy: str = "validation_readiness_v3",
             final_semantic_round: bool = False, allow_risk_only: bool = True) -> dict:
    constitution = load_constitution(constitution_path)
    rows = list(iter_jsonl(input_path))
    bundles: dict[str, list[dict]] = defaultdict(list)
    if bundle_aware:
        for row in rows:
            bundle_id = _bundle_id(row)
            if bundle_id:
                bundles[bundle_id].append(row)
    teacher = None if mock else DeepSeekClient(model, base_url, key_env, retries)
    output_rows, errors = [], []
    for index, row in enumerate(rows, 1):
        updated = dict(row)
        hid = str(row.get("id", index))
        if row.get("status") != "proposed":
            updated["critic_skipped"] = {"reason": f"not critic-eligible: {row.get('status')}"}
            output_rows.append(updated)
            continue
        output = out_dir / "critic_cases" / f"{hid}.json"
        bundle_id = _bundle_id(row) if bundle_aware else ""
        context = _bundle_context(row, bundles.get(bundle_id, [row])) if bundle_aware else {}
        request_fingerprint = _request_fingerprint(
            row, constitution, context, gate_policy, final_semantic_round
        )
        try:
            cached = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
            cache_matches = bool(
                cached and cached.get("_critic_request_fingerprint") == request_fingerprint
            )
            if cache_matches:
                response = cached
            elif mock:
                response = {
                    "gate_stage": "validation_readiness",
                    "recommendation": "accept_for_validation", "normal_counterexamples": [],
                    "abnormal_counterexamples": [], "nuisance_leakage": [], "duplicate_or_rename_of": [],
                    "blocking_semantic_defects": [], "held_out_validation_risks": [],
                    "non_blocking_improvements": [],
                    "reason": "mock critic permits validation, not activation", "confidence": 0.5,
                }
            else:
                if cached:
                    print(f"[critic-cache-stale] hypothesis={hid} refreshing v3 review", flush=True)
                response = teacher.call(
                    "You review readiness for held-out validation. Classify blockers and empirical risks separately.",
                    _prompt(row, constitution, context),
                )
            response["_critic_request_fingerprint"] = request_fingerprint
            response["_critic_prompt_version"] = PROMPT_VERSION
            response["_critic_bundle_id"] = bundle_id
            response["_critic_gate_policy"] = gate_policy
            response["_critic_final_semantic_round"] = final_semantic_round
            if not cache_matches:
                write_json(output, response)
            per_row_bundle_errors = context.get("bundle_contract_errors", {}).get(hid, [])
            recommendation, normalization = normalize_critic_decision(
                raw_response=response, row=row, final_round=final_semantic_round,
                bundle_errors=per_row_bundle_errors, allow_risk_only=allow_risk_only,
            )
            updated["critic"] = dict(response, **normalization, recommendation=recommendation)
            updated["status"] = {
                "accept_for_validation": "critic_approved_candidate",
                "revise": "revision_requested",
                "reject": "critic_rejected",
                "representation_only": "diagnostic_only",
                "human_review_required": "human_review_required",
            }[recommendation]
            output_rows.append(updated)
            print(f"+ CRITIQUED [{index}/{len(rows)}] hypothesis={hid} recommendation={recommendation}", flush=True)
        except Exception as exc:
            error = {"hypothesis_id": hid, "type": type(exc).__name__, "error": str(exc)}
            if teacher is not None and teacher.last_trace:
                error["deepseek_trace"] = teacher.last_trace
            errors.append(error)
    write_jsonl(out_dir / "critiqued_hypotheses.jsonl", output_rows)
    write_jsonl(out_dir / "critic_errors.jsonl", errors)
    recommendation_counts = Counter(
        str(row.get("critic", {}).get("recommendation", "not_eligible")) for row in output_rows
    )
    raw_recommendation_counts = Counter(
        str(row.get("critic", {}).get("raw_recommendation", "not_eligible")) for row in output_rows
    )
    summary = {
        "version": PROMPT_VERSION,
        "input": len(rows),
        "eligible": sum(row.get("status") == "proposed" for row in rows),
        "approved_for_validation": sum(row.get("status") == "critic_approved_candidate" for row in output_rows),
        "revision_requested": recommendation_counts.get("revise", 0),
        "terminal_rejected": recommendation_counts.get("reject", 0),
        "representation_only": recommendation_counts.get("representation_only", 0),
        "human_review_required": recommendation_counts.get("human_review_required", 0),
        "risk_only_promoted": sum(
            row.get("critic", {}).get("raw_recommendation") == "revise"
            and row.get("critic", {}).get("recommendation") == "accept_for_validation"
            for row in output_rows
        ),
        "not_eligible": recommendation_counts.get("not_eligible", 0),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "raw_recommendation_counts": dict(sorted(raw_recommendation_counts.items())),
        "bundle_aware": bool(bundle_aware), "prompt_version": PROMPT_VERSION,
        "gate_policy": gate_policy, "final_semantic_round": final_semantic_round,
        "errors": len(errors),
    }
    write_json(out_dir / "critic_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--constitution", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--bundle-aware", action="store_true")
    parser.add_argument("--gate-policy", default="validation_readiness_v3")
    parser.add_argument("--final-semantic-round", action="store_true")
    parser.add_argument("--allow-risk-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(critique(args.input, args.constitution, args.out_dir, model=args.model,
        base_url=args.base_url, key_env=args.key_env, retries=args.retries, mock=args.mock,
        bundle_aware=args.bundle_aware, gate_policy=args.gate_policy,
        final_semantic_round=args.final_semantic_round,
        allow_risk_only=args.allow_risk_only), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
