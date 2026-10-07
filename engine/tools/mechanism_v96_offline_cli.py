#!/usr/bin/env python3
"""V9.6 offline only: no provider, reviewer import, scoring or training commands."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'docs'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--recover-stale-lock',action='store_true')
    args=parser.parse_args()
    from event_decision.safety import OfflineGuard
    guard=OfflineGuard(); guard.install()
    try:
        from event_decision.mechanism_v96_offline import execute,replay_lock,DEFAULT_SOURCE
        with replay_lock(PROJECT,args.out,args.recover_stale_lock):
            result=execute(PROJECT,args.source_run or PROJECT/'runs'/DEFAULT_SOURCE,args.out,args.verify_only)
        guard.assert_no_remote_calls()
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        print('[report] '+str(args.out/'index.html'),flush=True)
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,'ERROR: '+str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
