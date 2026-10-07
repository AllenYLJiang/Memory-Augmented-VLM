#!/usr/bin/env python3
"""Offline-only snapshots of the imported V9.15 review and missing-C2 outcome."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    guard = OfflineGuard()
    guard.install()
    from event_decision import claim_scope_followup as followup
    from event_decision.b1b4_trial.protocol import run_lock
    try:
        followup.check_out(args.run.resolve(), args.out.resolve())
        with run_lock(args.out):
            result = followup.build(PROJECT, args.run, args.out)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[followup] " + str(args.out / "index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
