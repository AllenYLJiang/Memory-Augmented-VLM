"""No API actions; offline encoding repair and diagnostic human review only."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from event_decision.safety import OfflineGuard


def main():
    guard = OfflineGuard(); guard.install()
    from event_decision.mechanism_v98_offline import lock
    from event_decision.mechanism_v99_offline import DEFAULT_TAG, DEFAULT_SOURCE, prepare, verify, import_review
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify', 'import-review'))
    parser.add_argument('--source-run', type=Path, default=project/'runs'/DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=project/'runs'/DEFAULT_TAG)
    parser.add_argument('--review-file', type=Path)
    args = parser.parse_args()
    try:
        with lock(project, args.out):
            if args.phase == 'prepare': result = prepare(project, args.source_run, args.out, guard)
            elif args.phase == 'verify': result = verify(project, args.source_run.resolve(), args.out.resolve())
            else:
                if not args.review_file: raise ValueError('--review-file is required')
                result = import_review(project, args.source_run.resolve(), args.out.resolve(), args.review_file, guard)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('[scope] Diagnostic human answers never automatically patch evidence, scores, or formal gates.')
        return 0
    except (ValueError, OSError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
