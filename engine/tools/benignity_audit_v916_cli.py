#!/usr/bin/env python3
"""Offline four-case benignity evidence audit; never acquires responses."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    guard = OfflineGuard()
    guard.install()
    from event_decision import benignity_audit_v916 as audit
    from event_decision.b1b4_trial.protocol import run_lock
    try:
        audit.followup.check_out(args.source.resolve(), args.out.resolve())
        with run_lock(args.out):
            result = audit.build(PROJECT, args.source, args.out)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[offline] " + str(args.out / "index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
