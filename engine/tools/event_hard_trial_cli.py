#!/usr/bin/env python3
"""V9.1 offline prospective-trial preflight. Never authorizes paid calls."""
import argparse
import json
from pathlib import Path

from event_decision.contracts import file_sha256, iter_jsonl, read_json, write_json, write_jsonl
from event_decision.safety import OfflineGuard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','inspect'])
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--development-run',type=Path,required=True)
    p.add_argument('--train-root',type=Path)
    p.add_argument('--candidate-file',type=Path)
    p.add_argument('--screen-limit',type=int,default=384)
    p.add_argument('--inventory-only',action='store_true')
    args=p.parse_args()
    args.project_root=args.project_root.resolve()
    args.trial=args.trial.resolve()
    args.development_run=args.development_run.resolve()
    if (args.development_run/'.v91.lock').exists(): p.error('development replay still running; wait for its receipt')
    guard=OfflineGuard(); guard.install()
    from event_decision.hard_trial import preflight, enroll
    if args.command=='inspect':
        print(json.dumps(read_json(args.trial/'enrollment/preflight_report.json'),indent=2)); return
    gates=read_json(args.development_run/'claims/candidate_readiness.json')
    receipt=read_json(args.development_run/'receipts/offline_execution.json',{})
    if receipt.get('status')!='completed' or not gates or any(r['status']=='INVALID_NUMERIC_OR_PROVENANCE' for r in gates.values()):
        p.error('run certified V9.1 offline replay first')
    report=preflight(args.project_root,args.development_run,args.trial,args.candidate_file)
    if not args.candidate_file and not args.inventory_only:
        if args.train_root is None: p.error('--train-root is required for local screening')
        from event_decision.local_screen import screen
        history=read_json(args.trial/'history/history_inventory.json')
        candidate=screen(args.project_root.parent,args.train_root,history,args.trial/'local_screen',args.screen_limit)
        config=read_json(args.trial/'protocol/proposed_config.json')
        rows,report=enroll(list(iter_jsonl(candidate)),history,config)
        screening=read_json(args.trial/'local_screen/screening_summary.json',{})
        report.update(candidate_input=str(candidate),candidate_input_sha256=file_sha256(candidate),
                      status='WAITING_FOR_QUOTAS_OR_HISTORY_REVIEW',
                      next_stage='review inventory, duplicate exclusions and proposed enrollment before freezing a paid acquisition protocol')
        if screening.get('status')=='WAITING_FOR_NEW_SOURCE_OR_HISTORY_RECONCILIATION':
            report['status']=screening['status']
            report['missing_required_strata']=screening['missing_required_strata']
            report['available_pools']=screening['pools']
        write_jsonl(args.trial/'enrollment/private_proposed_windows.jsonl',rows)
        write_json(args.trial/'enrollment/preflight_report.json',report)
    guard.assert_no_remote_calls()
    print(json.dumps({k:v for k,v in report.items() if k not in ('rejected','duplicate_candidates')},indent=2))


if __name__=='__main__': main()
