#!/usr/bin/env python3
"""Independent V9.2 entry; does not alter V9.1 strict enrollment or its hashes."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'docs'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','screen','preview','status'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=PROJECT/'runs/governed_v91_canary_source_audit_20260912')
    p.add_argument('--source-run',type=Path,default=PROJECT/'runs/governed_v91_reviewed_hard_normal_binding_20260911')
    p.add_argument('--train-root',type=Path,default=Path('G:/Dataset/XDViolence/train' if os.name=='nt' else '/mnt/g/Dataset/XDViolence/train'))
    p.add_argument('--anchors',type=Path,default=PROJECT.parent/'Transformer_semantic_components_select_anomaly/top_anomalous_frames_72B_positive_segments')
    p.add_argument('--code-dir',type=Path,default=PROJECT.parent/'Previous_code/structural_vlm_binary_v40_15_qwen36_highres_graph_vs_node_revision')
    p.add_argument('--reserve-source',action='append',default=[])
    p.add_argument('--max-windows-per-video',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--model',default='qwen3.6-plus')
    p.add_argument('--mock',action='store_true')
    p.add_argument('--max-calls',type=int,default=120)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--attempts-per-window',type=int,default=2)
    p.add_argument('--max-consecutive-errors',type=int,default=3)
    p.add_argument('--key-env',default='DASHSCOPE_API_KEY')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--recover-stale-lock',action='store_true')
    a=p.parse_args(); a.out=a.out.resolve()
    try:
        from event_decision.role_scoped import prepare,preview,assert_bound
        from event_decision.b5_development_screen import screen,summarize,run_lock
        if a.phase!='screen':
            from event_decision.safety import OfflineGuard
            guard=OfflineGuard();guard.install()
        if a.phase=='prepare':
            result=prepare(PROJECT,a.out,a.audit.resolve(),a.source_run.resolve(),a.train_root.resolve(),
                a.anchors.resolve(),a.code_dir.resolve(),seed=a.seed,max_windows=a.max_windows_per_video,
                model=a.model,reserved=a.reserve_source,mock=a.mock)
        else:
            protocol=assert_bound(a.out)
            if a.mock and not protocol['mock']: raise ValueError('mock/live cannot be changed within a TAG')
            with run_lock(a.out,a.recover_stale_lock):
                if a.phase=='screen':
                    result=screen(a.out,max_calls=a.max_calls,workers=a.workers,attempts_per_window=a.attempts_per_window,
                        retry_failed=a.retry_failed,key_env=a.key_env,max_consecutive_errors=a.max_consecutive_errors)
                elif a.phase=='preview': result=preview(a.out)
                else: result=summarize(a.out,protocol)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,ImportError) as exc:
        p.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__': main()
