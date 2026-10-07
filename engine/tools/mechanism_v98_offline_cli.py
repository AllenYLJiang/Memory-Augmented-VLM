"""Offline-only entry point. No API, native revalidation or review commands."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from event_decision.safety import OfflineGuard


def main():
    guard = OfflineGuard(); guard.install()
    from event_decision.mechanism_v98_offline import DEFAULT_SOURCE, DEFAULT_TAG, execute, lock
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=project/'runs'/DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=project/'runs'/DEFAULT_TAG)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    try:
        with lock(project, args.out):
            result = execute(project, args.source_run, args.out, guard, args.verify_only)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('[hold] Offline replay only. No native revalidation, human review, scoring or training authorized.')
    except (ValueError, OSError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr); return 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
