"""Narrow, opt-in API recovery without changing the frozen V919 source/receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def eligible(record):
    return (record.get('status') == 'provider_failed' and record.get('error_type') == 'StopAcquisition'
            and record.get('retryable_transport') is False and record.get('billing_unknown') is True)


def scoped_retry(original, approved):
    def allowed(record):
        return original(record) or (eligible(record) and digest(record) in approved)
    return allowed


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--project', type=Path, required=True)
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--execute', action='store_true')
    a.add_argument('--confirm-provider-restored', action='store_true')
    a.add_argument('--acknowledge-unknown-billing', action='store_true')
    a.add_argument('--approved-by', default='')
    a.add_argument('--max-attempts', type=int, default=14800)
    a.add_argument('--max-reserved-output-tokens', type=int, default=121241600)
    a.add_argument('--transport-max-attempts', type=int, default=5)
    args = a.parse_args()
    sys.path[:0] = [str(args.project / 'tools'), str(args.project / 'docs')]
    from event_decision.effectiveness_v919 import pipeline, acquisition
    from event_decision.b1b4_trial.protocol import run_lock, immutable, now
    from event_decision.contracts import read_json, file_sha256, write_json
    inventory = pipeline.implementation
    pipeline.implementation = lambda project: {k.replace('\\', '/'): v for k, v in inventory(project).items()}
    with run_lock(args.run):
        pipeline.verify(args.project, args.run)
        selected, receipts = [], {}
        for path in sorted((args.run / 'pilot/cost/attempts').glob('*.json')):
            r = read_json(path)
            if eligible(r) and not (args.run / 'cache' / (r['request_key'] + '.json')).exists():
                selected.append(r)
                receipts[path.relative_to(args.run).as_posix()] = file_sha256(path)
        report = {'candidate_failures': selected, 'receipt_hashes': receipts,
            'recovery_scope': 'only the exact StopAcquisition attempt receipts; no refusals or semantic retries',
            'known_root_cause_family': 'missing API key, account/quota/balance/authentication or rate limit',
            'exact_provider_code': 'not recorded by original runtime; cannot reconstruct',
            'new_API_calls': 0, 'original_receipts_modified': False}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.execute:
            return
        if not (args.confirm_provider_restored and args.acknowledge_unknown_billing and args.approved_by.strip()):
            raise ValueError('Confirm account/service restored and acknowledge uncertain prior billing before retry')
        if not 2 <= args.transport_max_attempts <= 5:
            raise ValueError('Recovery requires cumulative per-key attempt cap 2..5')
        if selected:
            approval = {'version': 'v919_exact_provider_stop_ack_1', 'approved_by': args.approved_by,
                'at': now(), 'receipts_sha256': receipts, 'approved_record_digests': [digest(r) for r in selected],
                'provider_restored_attested': True, 'billing_unknown_acknowledged': True,
                'refusal_retries_authorized': False, 'schema_or_semantic_retries_authorized': False,
                'wrapper_sha256': file_sha256(Path(__file__)), 'max_attempts': args.max_attempts,
                'max_reserved_output_tokens': args.max_reserved_output_tokens,
                'request_lifetime_cap': args.transport_max_attempts}
            immutable(args.run / 'provider_recovery' / (digest(approval) + '.json'), approval)
        approved = {digest(r) for r in selected}
    # The original CLI acquires its own lock. On re-entry, hashes and eligible records
    # are checked again by the in-memory predicate; new failures are NOT authorized.
    original = acquisition.retryable_receipt
    acquisition.retryable_receipt = scoped_retry(original, approved)
    from event_decision.b1b4_trial import evidence
    original_provider = evidence.dashscope_once
    def logged_provider(config, media, prompt):
        try:
            return original_provider(config, media, prompt)
        except evidence.StopAcquisition as exc:
            # This exception contains only the provider code or missing-key message,
            # never credentials, raw media, or model output.
            detail = {'at': now(), 'error_type': 'StopAcquisition', 'provider_message': str(exc),
                      'billing_may_be_unknown': True, 'automatic_account_retry': False}
            write_json(args.run / 'provider_recovery/last_provider_stop_detail.json', detail)
            print('[provider-detail] ' + str(exc), file=sys.stderr)
            raise
    evidence.dashscope_once = logged_provider
    import effectiveness_v919_cli
    sys.argv = [str(args.project / 'tools/effectiveness_v919_cli.py'), '--stage', '2', '--action', 'run',
        '--out', str(args.run), '--approve-budget', '--approved-by', args.approved_by,
        '--max-attempts', str(args.max_attempts), '--max-reserved-output-tokens', str(args.max_reserved_output_tokens),
        '--retry-transport-failures', '--transport-max-attempts', str(args.transport_max_attempts),
        '--retry-base-seconds', '10', '--retry-jitter-seconds', '3']
    try:
        raise SystemExit(effectiveness_v919_cli.main())
    finally:
        acquisition.retryable_receipt = original
        evidence.dashscope_once = original_provider


if __name__ == '__main__':
    main()
