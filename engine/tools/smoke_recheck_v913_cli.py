#!/usr/bin/env python3
"""Isolated prepare/approve/run/report entry point for the four-case recheck."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision import smoke_recheck_v913 as trial
from event_decision.b1b4_trial.protocol import run_lock
from event_decision.safety import OfflineGuard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
    p.add_argument("--source", type=Path, default=PROJECT / "runs/governed_v912_b1b4_minimal_effect_20260915")
    p.add_argument("--out", type=Path, default=PROJECT / "runs/governed_v913_smoke_premise_benignity_20260916")
    p.add_argument("--approve-budget", action="store_true")
    p.add_argument("--approved-by")
    p.add_argument("--max-attempts", type=int)
    args = p.parse_args()
    if args.phase != "run":
        OfflineGuard().install()
    try:
        if args.approve_budget and args.phase != "run":
            raise ValueError("budget approval only applies to explicit run")
        with run_lock(args.out):
            if args.phase == "prepare":
                result = trial.prepare(PROJECT, args.source, args.out)
            elif args.phase == "report":
                result = trial.report(args.out)
            else:
                if args.approve_budget:
                    trial.authorize(args.out, args.approved_by, args.max_attempts)
                result = trial.run(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
