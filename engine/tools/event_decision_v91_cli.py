#!/usr/bin/env python3
"""Offline V9.1 historical replay. Never imports a remote inference provider."""
import argparse
import json
import os
import time
from pathlib import Path

import yaml

from event_decision.safety import OfflineGuard
from event_decision.contracts import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if config.get('version') != 'governed_v91_numeric_repair_v1':
        p.error('wrong config version')
    guard = OfflineGuard()
    guard.install()
    from event_decision.v91 import run
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/'.v91.lock'
    try:
        descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        p.error('output is locked; verify no writer is running before removing a stale .v91.lock')
    started=time.time(); status='failed'
    try:
        os.write(descriptor,str(os.getpid()).encode())
        result = run(args.source, args.out, config)
        from event_decision.audit_challenge import audit
        audit(args.out)
        from event_decision.v91_report import report
        report_path=Path(__file__).resolve().parents[1]/'docs'/f'GOVERNED_V91_{args.out.name}_RESULTS.md'
        report(args.out,report_path)
        print(f'[report] {report_path}')
        guard.assert_no_remote_calls()
        status='completed'
    finally:
        os.close(descriptor); lock.unlink()
        write_json(args.out/'receipts/offline_execution.json',{'status':status,'started_unix':started,
                    'finished_unix':time.time(),'elapsed_seconds':time.time()-started,'remote_calls':0,'blocked_network_attempts':len(guard.attempts)})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
