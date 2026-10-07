"""Read-only development evidence report and noninterfering saved-prediction join."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from event_decision.safety import OfflineGuard


def main():
    guard = OfflineGuard()
    guard.install()
    from event_decision.mechanism_v98_offline import lock
    from event_decision.mechanism_v911_offline import DEFAULT_SOURCE, DEFAULT_TAG, prepare, verify
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify'), nargs='?', default='prepare')
    parser.add_argument('--source-run', type=Path, default=project/'runs'/DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=project/'runs'/DEFAULT_TAG)
    parser.add_argument('--predictions', type=Path, help='Optional exact 36-window saved predictions; default: frozen B5 screen snapshots')
    args = parser.parse_args()
    try:
        with lock(project, args.out):
            if args.phase == 'prepare': result = prepare(project, args.source_run, args.out, guard, args.predictions)
            else: result = verify(project, args.source_run, args.out, args.predictions)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('[report] '+str(args.out/'index.html'))
        print('[scope] Development diagnostics only; no formal review gate bypass or prediction changes.')
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__': raise SystemExit(main())
