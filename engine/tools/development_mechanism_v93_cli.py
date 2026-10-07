#!/usr/bin/env python3
"""Prepare, run/resume or report a development-only fixed-image comparison."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'docs'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'report', 'all'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=PROJECT / 'runs/governed_v92_role_scoped_B5_development_20260912_r1')
    parser.add_argument('--feedback', type=Path)
    parser.add_argument('--history-status', type=Path, default=PROJECT / 'runs/governed_v92_history_confirmation_resolution_20260913/current_v92_status.json')
    parser.add_argument('--model')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--max-calls', type=int, default=200)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--attempts-per-window', type=int, default=2)
    parser.add_argument('--max-consecutive-errors', type=int, default=3)
    parser.add_argument('--retry-schema', action='store_true')
    parser.add_argument('--recover-stale-lock', action='store_true')
    parser.add_argument('--key-env', default='DASHSCOPE_API_KEY')
    args = parser.parse_args()
    args.out = args.out.resolve(); args.source_run = args.source_run.resolve()
    feedback = args.feedback or args.source_run / 'review_returns/user_development_feedback_20260913_v1.jsonl'
    try:
        from event_decision.development_mechanism_v93 import prepare, load_run
        from event_decision.development_mechanism_v93_runner import run
        from event_decision.development_mechanism_v93_report import report
        from event_decision.b5_development_screen import run_lock
        if args.phase in {'prepare', 'report'} or args.mock:
            from event_decision.safety import OfflineGuard
            OfflineGuard().install()
        if args.phase == 'prepare' or (args.phase == 'all' and not args.out.exists()):
            result = prepare(PROJECT, args.source_run, args.out, feedback.resolve(), args.history_status.resolve(),
                             mock=args.mock, model=args.model, max_tokens=args.max_tokens)
            print(json.dumps(result, indent=2), flush=True)
            if args.phase == 'prepare':
                report(args.out)
                return 0
        protocol, _ = load_run(args.out)
        if args.mock != protocol['mock']:
            raise ValueError('MOCK/live mismatch; a mock run must never become a live run within the same TAG')
        if args.model and args.model != protocol['model']:
            raise ValueError('model changed; prepare a new TAG')
        if args.max_tokens != protocol['max_tokens']:
            raise ValueError('token limit changed; prepare a new TAG')
        if args.phase == 'all' and Path(protocol['source_run']).resolve() != args.source_run:
            raise ValueError('source changed for an existing TAG')
        with run_lock(args.out, args.recover_stale_lock):
            if args.phase == 'report': result = report(args.out)
            else:
                result = run(args.out, max_calls=args.max_calls, workers=args.workers,
                             attempts_per_window=args.attempts_per_window,
                             max_consecutive_errors=args.max_consecutive_errors,
                             retry_schema=args.retry_schema, key_env=args.key_env)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print('[report] ' + str(args.out / 'index.html'), flush=True)
        return 2 if result.get('counts', {}).get('failed', 0) else 0
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, 'ERROR: ' + str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
