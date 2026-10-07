#!/usr/bin/env python3
"""Offline prospective enrollment: prepare, blind review, then protocol freeze."""
import argparse
import json
import os
import time
from pathlib import Path

from event_decision.contracts import EventDecisionError, read_json, write_json
from event_decision.safety import OfflineGuard


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','review','freeze','inspect'])
    parser.add_argument('--source-run',type=Path,default=project/'runs/governed_v91_reviewed_hard_normal_binding_20260911')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--additional-candidates',type=Path,action='append',default=[])
    parser.add_argument('--reviewer-file',type=Path,action='append',default=[])
    parser.add_argument('--adjudication-file',type=Path)
    args=parser.parse_args()
    args.source_run=args.source_run.resolve(); args.out=args.out.resolve()
    args.additional_candidates=[p.resolve() for p in args.additional_candidates]
    args.reviewer_file=[p.resolve() for p in args.reviewer_file]
    if args.adjudication_file: args.adjudication_file=args.adjudication_file.resolve()
    guard=OfflineGuard(); guard.install()
    started=time.time(); lock=None; fd=None
    try:
        from event_decision.reenrollment import prepare,verify_integrity
        from event_decision.enrollment_review import build_packet,freeze
        if args.out==args.source_run or args.source_run in args.out.parents or args.out in args.source_run.parents:
            raise ValueError('use a new independent output/TAG')
        if args.phase!='prepare':
            contract=verify_integrity(project,args.out)
            from audit_v91_history import portable_path
            if portable_path(contract['source_run']).resolve()!=args.source_run: raise ValueError('source-run differs from frozen input contract')
            if args.additional_candidates: raise ValueError('candidate supplementation requires prepare and a new TAG')
            lock=args.out/'.reenrollment.lock'
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.write(fd,str(os.getpid()).encode())
        if args.phase=='prepare':
            result=prepare(project,args.source_run,args.out,args.additional_candidates)
        elif args.phase=='review': result=build_packet(project,args.source_run,args.out)
        elif args.phase=='freeze': result=freeze(project,args.source_run,args.out,args.reviewer_file,args.adjudication_file)
        else:
            result={'preflight':read_json(args.out/'enrollment/preflight_report.json'),
                    'review':read_json(args.out/'review_private/review_summary.json'),
                    'freeze':read_json(args.out/'protocol/freeze_receipt.json')}
        guard.assert_no_remote_calls()
        # Never print private per-case answers or private label distributions.
        print(json.dumps(result,indent=2))
        write_json(args.out/'receipts'/f'{args.phase}_{time.time_ns()}.json',{
            'status':'completed','phase':args.phase,'elapsed_seconds':time.time()-started,
            'remote_calls':0,'decoding_performed':args.phase=='review',
            'remote_execution_authorized':False})
    except (OSError,ValueError,KeyError,EventDecisionError) as exc:
        parser.exit(1,f'ERROR: {exc}\n')
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink()


if __name__=='__main__': main()
