#!/usr/bin/env python3
"""Explicit three-stage entrypoint; planning/evaluation never call a provider."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))

from event_decision.b1b4_trial.protocol import run_lock
from event_decision.contracts import file_sha256, iter_jsonl, read_json, write_json
from event_decision.effectiveness_v919 import pipeline as p


def parser():
    project = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--action", choices=["plan", "run", "evaluate", "status", "upgrade"], default="plan")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-root", default="/mnt/g/Dataset/XDViolence/train")
    ap.add_argument("--anchor-root", default=str(project.parent / "Transformer_semantic_components_select_anomaly/top_anomalous_frames_72B_positive_segments"))
    ap.add_argument("--test-root", default="/mnt/g/Dataset/XDViolence/videos/videos")
    ap.add_argument("--expected-test-videos", type=int, default=800)
    ap.add_argument("--annotations", default="/mnt/g/Dataset/XDViolence/videos/annotations_uniform_format.txt")
    ap.add_argument("--graph-catalog", default=str(project / "runs/reflective_governed_v5_smoke_20260816/graph_memory/governed_base/active_library/graph_catalog_v2.json"))
    ap.add_argument("--reuse-run", help="Optional exact native V919 request-envelope cache; no legacy schema guessing")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--pilot-videos", type=int, default=200)
    ap.add_argument("--pilot-windows-per-video", type=int, choices=[1, 2], default=2)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--top-k-abnormal", type=int, default=4)
    ap.add_argument("--top-k-normal", type=int, default=6)
    ap.add_argument("--model", default="qwen3.6-plus")
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--image-max-pixels", type=int, default=262144)
    ap.add_argument("--regularization", type=float, default=.1)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--approve-budget", action="store_true")
    ap.add_argument("--approved-by", default="")
    ap.add_argument("--max-attempts", type=int, default=0)
    ap.add_argument("--max-reserved-output-tokens", type=int, default=0)
    ap.add_argument("--retry-transport-failures", action="store_true", help="Explicitly retry confirmed transport failures only; never invalid/undesired responses")
    ap.add_argument("--transport-max-attempts", type=int, default=3, help="Cumulative attempt cap per request, including prior runs; requires --retry-transport-failures")
    ap.add_argument("--retry-base-seconds", type=float, default=10.)
    ap.add_argument("--retry-jitter-seconds", type=float, default=3.)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    project, out = Path(__file__).resolve().parents[1], a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        if a.action == "upgrade":
            from event_decision.effectiveness_v919.transport_patch import accept_upgrade
            result = accept_upgrade(project, out, a.approved_by, p.implementation(project))
        elif a.stage == 1:
            result = p.prepare(project, out, a)
        else:
            config = p.verify(project, out)
            phase = "pilot" if a.stage == 2 else "dense"
            if a.action == "status":
                from event_decision.effectiveness_v919.acquisition import cost_summary
                _, coverage = p.export_features(out, phase)
                result = {"coverage": coverage, "cost": cost_summary(out, phase)}
            elif a.action == "plan":
                result = p.plan(out, phase, config) if a.stage == 2 else p.prepare_test(out, a.test_root, config, a.annotations)
            elif a.action == "evaluate":
                result = p.fit_and_evaluate(out, config) if a.stage == 2 else p.dense_evaluate(out, a.annotations, config)
            else:
                if not a.approve_budget or not a.approved_by or not os.environ.get("DASHSCOPE_API_KEY"):
                    raise ValueError("Explicit APPROVE_BUDGET=1, APPROVED_BY and exported DASHSCOPE_API_KEY required")
                plan_path = out / phase / "plan.json"
                if not plan_path.exists():
                    raise ValueError("Inspect this stage's plan before paid acquisition")
                if a.stage == 2 and not read_json(out / "pilot/enrollment.json")["six_class_scope_all_roles"]:
                    raise ValueError("Pilot lacks A/six-class source scope across roles; repair enrollment in a new TAG before paying")
                if a.stage == 3:
                    p.prepare_test(out, a.test_root, config, a.annotations)
                from graph_catalog import read_catalog_json
                from event_decision.effectiveness_v919.acquisition import Budget, collect
                budget = Budget(out, phase, file_sha256(plan_path), config, a.approved_by, a.max_attempts,
                                a.max_reserved_output_tokens, a.retry_transport_failures, a.transport_max_attempts,
                                a.retry_base_seconds, a.retry_jitter_seconds)
                collect(out, phase, list(iter_jsonl(out / phase / "inputs.jsonl")), read_catalog_json(out / "graph_catalog.json"), config, budget)
                result = p.fit_and_evaluate(out, config) if a.stage == 2 else p.dense_evaluate(out, a.annotations, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[pause] " + str(exc), file=sys.stderr)
        raise SystemExit(2)
