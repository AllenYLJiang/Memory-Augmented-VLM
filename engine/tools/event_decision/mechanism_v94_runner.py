"""Bounded acquisition; retain first core-valid response regardless of label."""
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .contracts import file_sha256
from .role_scoped import atomic_json
from .development_mechanism_v93_runner import provider_factory, provider_error, failure_kind, RequestFailure
from .mechanism_v94_contract import assess, example
from .mechanism_v94_store import raw_json
from .mechanism_v94_report import report, current


def acquire(out,protocol,rows,*,max_calls=80,workers=3,attempts_per_window=2,retry_partial=False,key_env='DASHSCOPE_API_KEY',provider=None):
    if any(type(v) is not int or v<1 for v in (max_calls,workers,attempts_per_window)) or workers>8 or attempts_per_window>3: raise ValueError('Positive budgets; workers<=8, attempts<=3')
    out=Path(out);digest=file_sha256(out/'protocol.json');pending=[]
    for row in rows:
        cached=current(out,row,digest)
        if cached['status'] in {'success','provider_rejected'}: continue
        if cached['status']=='partial' and not retry_partial: continue
        if cached.get('failure_kind') in {'request','runtime'}: raise ValueError('Request/runtime failure needs diagnosis before resume')
        pending.append(row)
    if not pending: return report(out,protocol,rows)
    if provider: request=provider
    elif protocol['mock']: request=lambda images,prompt:json.dumps(example())
    else: request,_=provider_factory(protocol,key_env)
    lock=threading.Lock();output_lock=threading.Lock();stop=threading.Event();used=0;reason=None;consecutive=0
    ident=str(time.time_ns())+'_'+uuid.uuid4().hex[:8]
    inv={'id':ident,'started_unix':time.time(),'status':'running','mock':protocol['mock'],'max_calls':max_calls,'workers':workers,'protocol_sha256':digest}
    atomic_json(out/'invocations'/(ident+'.json'),inv)
    key=os.environ.get(key_env,'')
    def redact(v): return str(v).replace(key,'[REDACTED]') if key else str(v)
    def one(row):
        nonlocal used,reason,consecutive
        uid=row['window_uid'];target=out/'results'/(uid+'.json');repair='';attempted=False
        result={'window_uid':uid,'source_group':row['source_group'],'role':'development_only','protocol_sha256':digest,'mock':protocol['mock'],'model':protocol['model'],'invocation_id':ident}
        for attempt in range(attempts_per_window):
            with lock:
                if stop.is_set() or used>=max_calls: break
                used+=1
            if not attempted and target.exists():
                archive=out/'archived_results'/(uid+'_'+str(time.time_ns())+'.json');archive.parent.mkdir(parents=True,exist_ok=True);target.replace(archive)
            attempted=True
            receipt={**result,'status':'started','started_unix':time.time(),'repair_feedback':repair}
            path=out/'attempts'/uid/(str(time.time_ns())+'_'+uuid.uuid4().hex[:8]+'.json');atomic_json(path,receipt)
            try:
                raw=request([out/'cases'/uid[:20]/f'T{i}.jpg' for i in range(8)],protocol['prompt']+repair)
                receipt['raw']=redact(raw);parsed=raw_json(raw);provider_error(parsed)
                a=assess(parsed)
                result={k:v for k,v in result.items() if k not in {'parsed','assessment','failure_kind','error'}}
                result.update(status='success' if a['core_valid'] else 'partial',parsed=parsed,assessment=a)
                receipt['status']=result['status']
                repair='\nPrevious response failed these structural checks (not label feedback): '+json.dumps(a['issues'][:12])+'. Return the full corrected object; do not invent unseen entities.'
            except Exception as exc:
                kind=failure_kind(exc);result.update(status='provider_rejected' if kind=='provider_policy' else 'failed',failure_kind=kind,error=redact(exc))
                result.pop('parsed',None);result.pop('assessment',None)
                receipt.update(status=result['status'],failure_kind=kind,error=redact(exc))
                if kind in {'account','request','runtime'}:
                    with lock: reason=kind;stop.set()
            receipt['elapsed_seconds']=time.time()-receipt['started_unix'];atomic_json(path,receipt)
            if result['status'] in {'success','provider_rejected'} or stop.is_set(): break
            if attempt+1<attempts_per_window: time.sleep(min(2**attempt,8))
        if not attempted: return
        result['finished_unix']=time.time();atomic_json(target,result)
        with lock:
            consecutive=consecutive+1 if result['status']=='failed' else 0
            if consecutive>=3: reason=reason or 'consecutive_errors';stop.set()
        with output_lock:
            prefix='MOCK-' if protocol['mock'] else ''
            print(f'[{prefix}V94-SAVED] {uid[:20]} status={result["status"]} label={(result.get("parsed") or {}).get("b5",{}).get("label")} -> {out/"cases"/uid[:20]/"index.html"}',flush=True)
            report(out,protocol,rows,changed=uid)
    try:
        report(out,protocol,rows)
        with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(one,pending))
        inv.update(status='finished',stop_reason=reason)
    except BaseException:
        inv['status']='interrupted';raise
    finally:
        inv.update(finished_unix=time.time(),calls=used,budget_exhausted=used>=max_calls);atomic_json(out/'invocations'/(ident+'.json'),inv)
    return report(out,protocol,rows)
