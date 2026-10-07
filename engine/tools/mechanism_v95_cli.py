#!/usr/bin/env python3
"""A: offline freeze; B: bounded new acquisition; C: technical-first review/export."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'docs'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',choices=['A','B','C','1','2','3'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-run',type=Path)
    p.add_argument('--mock',action='store_true')
    p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--max-calls',type=int,default=80)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--attempts-per-window',type=int,default=2)
    p.add_argument('--retry-partial',action='store_true')
    p.add_argument('--recover-stale-lock',action='store_true')
    p.add_argument('--review-file',type=Path)
    args=p.parse_args(); step={'1':'A','2':'B','3':'C'}.get(args.step,args.step); out=args.out.resolve()
    try:
        from event_decision.mechanism_v95_store import prepare,load,DEFAULT_SOURCE
        from event_decision.mechanism_v95_report import report
        from event_decision.b5_development_screen import run_lock
        if step in {'A','C'} or args.mock:
            from event_decision.safety import OfflineGuard
            OfflineGuard().install()
        if step=='A':
            result=prepare(PROJECT,args.source_run or PROJECT/'runs'/DEFAULT_SOURCE,out,mock=args.mock,max_tokens=args.max_tokens)
            protocol,rows=load(out); report(out,protocol,rows)
        else:
            protocol,rows=load(out)
            if args.mock!=protocol['mock']: raise ValueError('MOCK/live mismatch; use a separate TAG')
            if args.max_tokens!=protocol['max_tokens']: raise ValueError('Frozen token limit changed')
            if args.source_run and args.source_run.resolve()!=Path(protocol['source_run']).resolve(): raise ValueError('Explicit V95_SOURCE_RUN conflicts with protocol')
            with run_lock(out,args.recover_stale_lock):
                if step=='B':
                    from event_decision.mechanism_v95_runner import acquire
                    result=acquire(out,protocol,rows,max_calls=args.max_calls,workers=args.workers,
                        attempts_per_window=args.attempts_per_window,retry_partial=args.retry_partial)
                else:
                    from event_decision.mechanism_v95_stage3 import stage3
                    result=stage3(out,protocol,rows,args.review_file); report(out,protocol,rows)
        printable={k:v for k,v in result.items() if k not in {'result_hashes'}}
        print(json.dumps(printable,ensure_ascii=False,indent=2),flush=True)
        print('[report] '+str(out/'index.html'),flush=True)
        if step=='C': print('[gate] '+result['decision'],flush=True)
        return 2 if result.get('counts',{}).get('failed',0) else 0
    except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        p.exit(1,'ERROR: '+str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
