#!/usr/bin/env python3
"""Freeze or explicitly authorize the four-window C2-only diagnostic contrast."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
    parser.add_argument("--source", type=Path, default=PROJECT / "runs/governed_v915_claim_scope_and_missing_C2_20260916")
    parser.add_argument("--out", type=Path, default=PROJECT / "runs/governed_v917_four_C2_contrast_20260917")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--approve-budget", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    guard = OfflineGuard()
    if args.phase != "run":
        guard.install()
    from event_decision import contrast_c2_v917 as trial
    from event_decision.b1b4_trial.protocol import run_lock
    try:
        if args.approve_budget and args.phase != "run":
            raise ValueError("approval is only accepted during explicit run")
        if args.phase == "prepare":
            trial.previous.followup.check_out(args.source.resolve(), args.out.resolve())
        elif not (args.out / "seal.json").exists():
            raise ValueError("prepare this V9.17 TAG first")
        with run_lock(args.out):
            if args.phase == "prepare":
                result = trial.prepare(PROJECT, args.source, args.out, 8192 if args.max_output_tokens is None else args.max_output_tokens)
            elif args.phase == "run":
                if args.max_output_tokens is not None and args.max_output_tokens != trial.read_json(args.out / "config.json")["max_output_tokens"]:
                    raise ValueError("MAX_OUTPUT_TOKENS differs from the frozen budget; use a new TAG")
                if args.approve_budget:
                    trial.authorize(args.out, args.approved_by, args.max_attempts)
                result = trial.run(args.out)
            else:
                result = trial.report(args.out)
        if args.phase != "run":
            guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[report] " + str(args.out / "index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
