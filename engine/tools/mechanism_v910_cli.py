"""Zero-API, source-preserving R1 evidence overlay and full 36-window audit."""
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
    from event_decision.mechanism_v910_offline import DEFAULT_SOURCE, DEFAULT_TAG, prepare, verify
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify'), nargs='?', default='prepare')
    parser.add_argument('--source-run', type=Path, default=project/'runs'/DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=project/'runs'/DEFAULT_TAG)
    parser.add_argument('--review-file', type=Path, default=project/'docs/v99_diagnostic_review_R1.json')
    args = parser.parse_args()
    try:
        with lock(project, args.out):
            if args.phase == 'prepare':
                result = prepare(project, args.source_run, args.out, args.review_file, guard)
            else:
                result = verify(project, args.source_run, args.out, args.review_file)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('[scope] No API, no decoding, no native-evidence overwrite, no score/training authority.')
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
