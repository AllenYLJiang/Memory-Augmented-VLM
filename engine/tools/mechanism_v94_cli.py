#!/usr/bin/env python3
"""Three explicit stages; only STEP=2 can make paid calls."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'docs'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',choices=['1','2','3'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-run',type=Path)
    p.add_argument('--mock',action='store_true')
    p.add_argument('--max-tokens',type=int,default=6144)
    p.add_argument('--max-calls',type=int,default=80)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--attempts-per-window',type=int,default=2)
    p.add_argument('--retry-partial',action='store_true')
    p.add_argument('--recover-stale-lock',action='store_true')
    p.add_argument('--review-file',type=Path)
    args=p.parse_args();out=args.out.resolve()
    try:
        from event_decision.mechanism_v94_store import prepare,load,DEFAULT_SOURCE
        from event_decision.mechanism_v94_report import report
        from event_decision.mechanism_v94_stage3 import review_packet,stage3
        from event_decision.b5_development_screen import run_lock
        if args.step in {'1','3'} or args.mock:
            from event_decision.safety import OfflineGuard
            OfflineGuard().install()
        if args.step=='1':
            result=prepare(PROJECT,args.source_run or PROJECT/'runs'/DEFAULT_SOURCE,out,mock=args.mock,max_tokens=args.max_tokens)
            protocol,rows=load(out);report(out,protocol,rows);review_packet(out,protocol,rows)
        else:
            protocol,rows=load(out)
            if args.mock!=protocol['mock']: raise ValueError('MOCK/live mismatch; use separate TAG')
            if args.source_run and args.source_run.resolve()!=Path(protocol['source_run']).resolve(): raise ValueError('Explicit V94_SOURCE_RUN conflicts with frozen source')
            if args.max_tokens!=protocol['max_tokens']: raise ValueError('Frozen token setting changed')
            with run_lock(out,args.recover_stale_lock):
                if args.step=='2':
                    from event_decision.mechanism_v94_runner import acquire
                    result=acquire(out,protocol,rows,max_calls=args.max_calls,workers=args.workers,attempts_per_window=args.attempts_per_window,retry_partial=args.retry_partial)
                else: result=stage3(out,protocol,rows,args.review_file)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        print('[report] '+str(out/'index.html'),flush=True)
        if args.step=='3': print('[gate] '+result['decision'],flush=True)
        return 2 if result.get('counts',{}).get('failed',0) else 0
    except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        p.exit(1,'ERROR: '+str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
