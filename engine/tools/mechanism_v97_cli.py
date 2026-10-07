#!/usr/bin/env python3
"""Freeze, collect and audit native observations; human-reviewed diagnostics only."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'docs'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'report', 'review', 'attach'), default='prepare')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--total-attempts', type=int, default=72)
    parser.add_argument('--attempts-per-window', type=int, default=2)
    parser.add_argument('--max-calls', type=int, default=36)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--key-env', default='DASHSCOPE_API_KEY')
    parser.add_argument('--recover-stale-lock', action='store_true')
    parser.add_argument('--review-file', type=Path)
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    guard = None
    if args.phase != 'run' or args.mock:
        from event_decision.safety import OfflineGuard
        guard = OfflineGuard(); guard.install()
    try:
        from event_decision.mechanism_v97_store import prepare, load, run_lock, DEFAULT_SOURCE
        from event_decision.mechanism_v97_runner import acquire, recover_returned_responses
        from event_decision.mechanism_v97_report import report
        with run_lock(PROJECT, args.out, args.recover_stale_lock):
            if args.phase == 'prepare':
                result = prepare(PROJECT, args.source_run or PROJECT/'runs'/DEFAULT_SOURCE, args.out,
                                 mock=args.mock, max_tokens=args.max_tokens,
                                 total_attempts=args.total_attempts, per_window=args.attempts_per_window)
            else:
                protocol, rows = load(args.out)
                if args.mock and not protocol['mock']: raise ValueError('MOCK cannot override a real frozen protocol')
                recover_returned_responses(args.out, rows)
                if args.phase == 'run':
                    result = acquire(args.out, protocol, rows, max_calls=args.max_calls, workers=args.workers, key_env=args.key_env)
                elif args.phase == 'report':
                    result = report(args.out, protocol, rows)
                elif args.phase == 'review':
                    from event_decision.mechanism_v97_review import import_review
                    if not args.review_file: raise ValueError('REVIEW_FILE must point to completed separate review returns')
                    report(args.out, protocol, rows)
                    result = import_review(args.out, protocol, rows, args.review_file)
                else:
                    from event_decision.mechanism_v97_review import attach_diagnostic_trace
                    from event_decision.contracts import write_jsonl
                    if not args.predictions: raise ValueError('PREDICTIONS with exact window_uid required')
                    target = args.out/'integration/annotated_predictions.jsonl'
                    if target.resolve() == args.predictions.resolve() or target.exists(): raise ValueError('Never overwrite source predictions or an existing joined output')
                    predictions = [json.loads(s) for s in args.predictions.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
                    joined = attach_diagnostic_trace(predictions, args.out)
                    write_jsonl(target, joined)
                    result = {'status': 'DIAGNOSTIC_SIDECAR_ATTACHED', 'windows': len(joined), 'output': str(target), 'scores_changed': False}
        if guard: guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print('[run] '+str(args.out), flush=True)
        if args.phase == 'run' and result.get('responded_windows') != 36:
            print('[hold] Incomplete cohort. Completed responses are retained; inspect statuses and remaining attempt budget before resuming.', flush=True)
            return 2
        if result.get('decision') in ('TECHNICAL_HOLD_NO_REVIEW', 'SEMANTIC_REVIEW_HOLD'):
            print('[hold] Do not start human review or integration until the indicated gate is resolved.', flush=True)
            return 3
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, 'ERROR: '+str(exc)+'\n')


if __name__ == '__main__': raise SystemExit(main())
