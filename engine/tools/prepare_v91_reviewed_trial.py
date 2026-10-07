#!/usr/bin/env python3
"""Reconcile reviewed source history, then screen locally. No paid acquisition."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from event_decision.contracts import file_sha256, iter_jsonl, read_json, write_json, write_jsonl
from event_decision.safety import OfflineGuard


def execute(args):
    import audit_v91_history
    from event_decision.hard_trial import QUOTAS, enroll, history_inventory
    from event_decision.history_reconciliation import reconcile, validate_reviews
    from event_decision.reviewed_screen import screen

    project = Path(__file__).resolve().parents[1]
    receipt = read_json(args.development_run / 'receipts/offline_execution.json', {})
    if receipt.get('status') != 'completed' or (args.development_run / '.v91.lock').exists():
        raise ValueError('complete the existing certified Step 1 first')
    readiness = read_json(args.development_run / 'claims/candidate_readiness.json', {})
    if not readiness or any(r['status'] == 'INVALID_NUMERIC_OR_PROVENANCE' for r in readiness.values()):
        raise ValueError('development numerical/provenance gate is not valid')
    audit_summary = read_json(args.audit_dir / 'audit_summary.json')
    original_trial = audit_v91_history.portable_path(audit_summary['trial']).resolve()
    if file_sha256(original_trial / 'history/history_inventory.json') != audit_summary['inventory_sha256']:
        raise ValueError('original inventory changed since reviewed audit')
    if audit_summary.get('issues') != 0 or not audit_summary.get('supplemental_search_enabled'):
        raise ValueError('review must refer to a complete supplemental local audit')
    reviews = list(iter_jsonl(args.review_file))
    old_rows = list(iter_jsonl(args.audit_dir / 'source_audit.jsonl'))
    requested, errors = validate_reviews(reviews, old_rows)
    if errors:
        raise ValueError('invalid review input: ' + json.dumps(errors))
    sources = {'review_file': args.review_file, 'audit_summary': args.audit_dir / 'audit_summary.json',
               'source_audit': args.audit_dir / 'source_audit.jsonl',
               'step1_receipt': args.development_run / 'receipts/offline_execution.json'}
    contract = {'version': 'v91_reviewed_prepare_v1', 'seed': args.seed,
                'screen_limit': args.screen_limit, 'maximum_windows_per_video': args.max_windows_per_video,
                'source_hashes': {k: file_sha256(p) for k, p in sources.items()},
                'code_hashes': {name: file_sha256(project / 'tools' / name) for name in
                                ('prepare_v91_reviewed_trial.py', 'audit_v91_history.py',
                                 'event_decision/history_reconciliation.py', 'event_decision/reviewed_screen.py',
                                 'event_decision/local_screen.py', 'event_decision/hard_trial.py')},
                'remote_execution_authorized': False}
    for forbidden in (original_trial, args.audit_dir, args.development_run):
        if args.out == forbidden or forbidden in args.out.parents or args.out in forbidden.parents:
            raise ValueError('use a separate new trial TAG, not a historical input directory')
    old_contract = read_json(args.out / 'reviewed_prepare_contract.json')
    if old_contract and old_contract != contract:
        raise ValueError('inputs/configuration/code changed; use a new TAG')
    if not old_contract and args.out.exists() and any(args.out.iterdir()):
        raise ValueError('nonempty output without this workflow contract; use a new TAG')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / '.reviewed_prepare.lock'
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError('trial is locked; verify the writer has stopped before removing a stale lock')
    started, status = time.time(), 'failed'
    try:
        os.write(descriptor, str(os.getpid()).encode())
        write_json(args.out / 'reviewed_prepare_contract.json', contract)
        write_jsonl(args.out / 'history/review_input_snapshot.jsonl', reviews)
        roots = [p / 'runs' for p in sorted(project.parent.iterdir()) if (p / 'runs').is_dir()]
        fresh_trial = args.out / 'current_history'
        inventory = history_inventory(roots, fresh_trial / 'history', excluded_root=args.out)
        fresh_audit = args.out / 'audits' / ('audit_' + uuid.uuid4().hex[:12])
        audit_v91_history.run(SimpleNamespace(trial=fresh_trial, out=fresh_audit, xd_root=project.parent,
                              train_root=args.train_root, max_file_mb=1024, no_supplemental=False))
        fresh_summary = read_json(fresh_audit / 'audit_summary.json')
        if fresh_summary['issues'] or inventory['issues']:
            raise ValueError(f'fresh history audit has unresolved technical issues: {fresh_audit}')
        if any(file_sha256(p) != contract['source_hashes'][k] for k, p in sources.items()):
            raise ValueError('review/development inputs changed during reconciliation')
        history, report = reconcile(reviews, old_rows, list(iter_jsonl(fresh_audit / 'source_audit.jsonl')),
                                    inventory, list(iter_jsonl(fresh_audit / 'scanned_files.jsonl')))
        report.update(fresh_audit=str(fresh_audit), review_sha256=contract['source_hashes']['review_file'])
        write_json(args.out / 'history/reconciliation_report.json', report)
        if report['errors']:
            raise ValueError('review conflicts with fresh evidence; see history/reconciliation_report.json')
        write_json(args.out / 'history/reconciled_history.json', history)
        print(f'[reconcile] released={report["released_sources"]}/{len(requested)} '
              f'excluded={report["remaining_excluded_sources"]}; API=0', flush=True)
        config = {'version': 'hard_normal_same_event_trial_reviewed_v91_v1', 'seed': args.seed,
                  'quotas': QUOTAS, 'window_frames': 96, 'evidence_frames': 8,
                  'disallow_overlapping_windows': True, 'maximum_windows_per_source_group': 2,
                  'primary': 'T1_direct2_vs_T0_m0', 'secondary': 'B1_event_bound4_vs_B0_event_unbound3',
                  'max_event_calls_per_window': 2, 'hard_normal_risk_tolerance': None,
                  'deployment_authorized': False, 'frozen': False}
        write_json(args.out / 'protocol/proposed_config.json', config)
        candidate, plan = screen(project.parent, args.train_root, history, args.out / 'local_screen',
                                 args.screen_limit, args.seed, args.max_windows_per_video, args.reconcile_only)
        if args.reconcile_only:
            status = 'reconciliation_completed_screening_not_run'
            print('[next] History imported. Re-run the same command with RECONCILE_ONLY=0 for local screening.')
            return
        selected, preflight = enroll(list(iter_jsonl(candidate)), history, config)
        ready = not preflight['quota_gaps'] and not history['issues']
        preflight.update(status='READY_FOR_ENROLLMENT_REVIEW' if ready else 'WAITING_FOR_LOCAL_QUOTAS_OR_MEDIA_REVIEW',
                         ready_for_enrollment_review=ready, ready_for_protocol_freeze=False,
                         history_requires_review=False, history_review_scope='requested releases only; not global proof',
                         candidate_input=str(candidate), candidate_input_sha256=file_sha256(candidate),
                         available_pools=plan['pools'], remote_execution_authorized=False,
                         next_stage='Review enrollment, media and label scope; freeze protocol before any API acquisition.')
        write_jsonl(args.out / 'enrollment/private_proposed_windows.jsonl', selected)
        write_json(args.out / 'enrollment/preflight_report.json', preflight)
        status = 'completed'
        print(json.dumps({k: v for k, v in preflight.items() if k not in ('rejected', 'duplicate_candidates')}, indent=2))
    finally:
        write_json(args.out / 'reviewed_prepare_receipt.json', {'status': status,
                   'elapsed_seconds': time.time() - started, 'remote_calls': 0})
        os.close(descriptor)
        lock.unlink()


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--review-file', type=Path, required=True)
    parser.add_argument('--development-run', type=Path, default=project / 'runs/governed_v91_numeric_repair_20260910')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--screen-limit', type=int, default=384)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--max-windows-per-video', type=int, choices=(1, 2), default=2)
    parser.add_argument('--reconcile-only', action='store_true')
    args = parser.parse_args()
    if args.screen_limit < 144 and not args.reconcile_only:
        parser.error('--screen-limit must be >=144 for the fixed trial')
    for field in ('audit_dir', 'review_file', 'development_run', 'out', 'train_root'):
        setattr(args, field, getattr(args, field).resolve())
    guard = OfflineGuard()
    guard.install()
    try:
        execute(args)
        guard.assert_no_remote_calls()
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
