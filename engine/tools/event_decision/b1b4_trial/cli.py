"""Three operator phases; resume stops precisely at missing approvals/data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import iter_jsonl, read_json, write_json
from ..role_scoped import portable
from .protocol import initialize, run_lock, advance, freeze, verify_frozen, verify_manifest, at_least


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("prepare", "run", "evaluate", "status", "mock"), default="prepare")
    parser.add_argument("--candidates", type=Path, action="append")
    parser.add_argument("--history-audit", type=Path, help="reuse hash-verified inventory; current no-new-use attestation still required")
    parser.add_argument("--media", action="store_true", help="local 96-frame clip decoding after capacity review")
    args = parser.parse_args(argv)
    project, out = args.project.resolve(), args.out.resolve()
    config = read_json(args.config or project / "config/b1b4_minimal_effect_v912.yaml")
    try:
        with run_lock(out):
            if args.phase == "mock":
                from .mock import run_mock
                result = run_mock(out, config)
            elif args.phase == "status":
                result = {"state": read_json(out / "state.json"), "preflight": read_json(out / "enrollment/preflight_report.json"),
                          "models": read_json(out / "models/identity.json"), "commit": read_json(out / "predictions/commit.json")}
            elif args.phase == "prepare":
                from .enrollment import load_candidates, attach_negative_anchor_audit, history_scan, import_history, prepare_enrollment
                from .evidence import prepare_media, make_plan
                from .evaluation import packet
                initialize(project, out, config)
                if at_least(out, "PROTOCOL_LOCKED"):
                    raise ValueError("Protocol already locked: use PHASE=run, not re-enrollment")
                paths = args.candidates or [project / "runs/governed_v91_reenrolled_scope_review_20260911/inputs/candidate_snapshot.jsonl"]
                rows = load_candidates(paths)
                rows = attach_negative_anchor_audit(project, out, rows)
                scope = import_history(out, args.history_audit, rows) if args.history_audit else history_scan(project, out, rows)
                result = prepare_enrollment(out, rows, scope, config, paths)
                if result["ready"]:
                    if args.media or (out / "media/manifest.json").exists():
                        prepare_media(out)
                        packet(out)
                        result["plans"] = {phase: make_plan(project, out, phase, config) for phase in ("smoke", "adaptation", "locked_evaluation")}
                        result["next"] = "Review authorizations/approval.json (all three caps together), then PHASE=run. No API calls have been made."
                    else:
                        result["next"] = "Capacity ready. Rerun PHASE=prepare MEDIA=1 for local clips and budget plan."
                else:
                    result["next"] = "Read enrollment/preflight_report.json; review history/scope_approval.json if requested. No B5 quota exists in this protocol."
            elif args.phase == "run":
                from .evidence import collect, make_plan, role_rows
                from .features import export_store
                from .learning import fit_models, predict_and_commit
                verify_manifest(read_json(out / "seal/legacy_inventory.json"))
                config = read_json(out / "protocol/config.json")
                if at_least(out, "PREDICTIONS_COMMITTED"):
                    result = {"state": "PREDICTIONS_COMMITTED", "next": "Finish two independent blind reviews, then PHASE=evaluate. No more API calls."}
                else:
                    if not at_least(out, "DEV_SMOKE_COMPLETE"):
                        make_plan(project, out, "smoke", config)
                        collect(project, out, "smoke", config)
                    if not at_least(out, "PROTOCOL_LOCKED"):
                        freeze(out)
                    verify_frozen(out)
                    if not at_least(out, "ADAPTATION_FEATURES_FROZEN"):
                        make_plan(project, out, "adaptation", config)
                        collect(project, out, "adaptation", config)
                        export_store(out, "adaptation", role_rows(out, "adaptation"))
                        advance(out, "ADAPTATION_FEATURES_FROZEN")
                    if not at_least(out, "MODELS_FROZEN"):
                        fit_models(out, config)
                    models = read_json(out / "models/frozen_models.json")["models"]
                    if not any(models[k]["status"] == "CERTIFIED" for k in ("T1_DIRECT2", "B1_SAME_EVENT")):
                        raise ValueError("NO_ESTIMABLE_PRIMARY_BRANCH: no locked API calls. Inspect adaptation fit certificates/cohorts; new design requires new TAG.")
                    if not at_least(out, "LOCKED_FEATURES_COMPLETE_OR_DECLARED_MISSING"):
                        make_plan(project, out, "locked_evaluation", config)
                        collect(project, out, "locked_evaluation", config)
                        export_store(out, "locked_evaluation", role_rows(out, "locked_evaluation"))
                        advance(out, "LOCKED_FEATURES_COMPLETE_OR_DECLARED_MISSING")
                    result = predict_and_commit(out)
                    result["next"] = "PHASE=evaluate after two independent blind full-clip reviews. Do not inspect locked per-case predictions beforehand."
            else:
                from .evaluation import evaluate
                verify_frozen(out)
                verify_manifest(read_json(out / "seal/legacy_inventory.json"))
                result = evaluate(out, read_json(out / "protocol/config.json"))
                result = {k: result[k] for k in ("conclusion", "enrolled_locked_n", "primary_human_label_n", "deployment_authorized")}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 3 if result.get("ready") is False else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print("[pause] " + str(exc), flush=True)
        write_json(out / "last_pause.json", {"phase": args.phase, "reason": str(exc), "no_gate_override": True})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
