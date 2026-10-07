#!/usr/bin/env python3
"""Audit all saved V9.13 responses offline. No paid phase or authorization flags."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "runs/governed_v913_smoke_premise_benignity_20260916")
    parser.add_argument("--out", type=Path, default=PROJECT / "runs/governed_v914_smoke_offline_audit_20260916")
    args = parser.parse_args()
    guard = OfflineGuard()
    guard.install()
    from event_decision import smoke_offline_v914 as audit
    from event_decision.b1b4_trial.protocol import run_lock
    try:
        protocol = audit.prior.verify(args.source)
        audit.check_output(args.source.resolve(), args.out.resolve(), audit.portable(protocol["source_run"]).resolve())
        with run_lock(args.out):
            result = audit.build(PROJECT, args.source, args.out)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[report] " + str(args.out / "index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
