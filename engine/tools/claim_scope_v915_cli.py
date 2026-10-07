#!/usr/bin/env python3
"""Prepare/import three claim reviews; optionally complete one missing C2."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "review", "run", "report"), default="prepare")
    parser.add_argument("--source", type=Path, default=PROJECT / "runs/governed_v914_smoke_offline_audit_20260916")
    parser.add_argument("--out", type=Path, default=PROJECT / "runs/governed_v915_claim_scope_and_missing_C2_20260916")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--approve-budget", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    guard = OfflineGuard()
    if args.phase != "run":
        guard.install()
    from event_decision import claim_scope_v915 as trial
    from event_decision.b1b4_trial.protocol import run_lock
    try:
        if args.approve_budget and args.phase != "run":
            raise ValueError("approval is only accepted in explicit run phase")
        # Reject old/source folders before even creating a lock there.
        if args.phase == "prepare":
            binding = trial.read_json(args.source / "input_binding.json")
            old = trial.portable(binding["source"]).resolve()
            original = trial.portable(trial.read_json(old / "protocol.json")["source_run"]).resolve()
            for root in (args.source.resolve(), old, original):
                trial.audit.check_output(root, args.out.resolve(), root)
        elif not (args.out / "seal.json").exists():
            raise ValueError("prepare this V9.15 TAG first")
        with run_lock(args.out):
            if args.phase == "prepare":
                result = trial.prepare(PROJECT, args.source, args.out)
            elif args.phase == "review":
                result = trial.import_review(args.out, args.review_file or args.out / "review/review.json")
            elif args.phase == "run":
                if args.approve_budget:
                    trial.authorize(args.out, args.approved_by, args.max_attempts)
                result = trial.run(args.out)
            else:
                result = trial.report(args.out)
        if args.phase != "run":
            guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[review] " + str(args.out / "review/index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
