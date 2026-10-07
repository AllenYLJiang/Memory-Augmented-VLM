"""Bounded new requests; never rank responses by agreement with an old label."""
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import file_sha256
from .role_scoped import atomic_json
from .development_mechanism_v93_runner import provider_factory, provider_error, failure_kind
from .mechanism_v94_store import raw_json
from .mechanism_v95_contract import assess, examples
from .mechanism_v95_report import current, report


def acquire(out, protocol, rows, *, max_calls=80, workers=3, attempts_per_window=2,
            retry_partial=False, key_env='DASHSCOPE_API_KEY', provider=None):
    if any(type(v) is not int or v<1 for v in (max_calls,workers,attempts_per_window)) or workers>8 or attempts_per_window>3:
        raise ValueError('Positive budgets required; workers<=8, attempts<=3')
    out=Path(out); digest=file_sha256(out/'protocol.json'); pending=[]
    for row in rows:
        old=current(out,row,protocol)
        if old['status'] in {'success','provider_rejected'}: continue
        if old['status']=='partial' and not retry_partial: continue
        if old.get('failure_kind') in {'request','runtime'} or old.get('last_request_error_kind') in {'request','runtime'}: raise ValueError('Diagnose request/runtime error before resuming')
        pending.append(row)
    if not pending: return report(out,protocol,rows)
    if provider is not None: request=provider
    elif protocol['mock']: request=lambda images,prompt:json.dumps(examples()[0])
    else: request,_=provider_factory(protocol,key_env)
    mutex=threading.Lock(); output_lock=threading.Lock(); stop=threading.Event()
    used=0; streak=0; reason=None
    invocation=str(time.time_ns())+'_'+uuid.uuid4().hex[:8]
    meta={'id':invocation,'started_unix':time.time(),'status':'running','mock':protocol['mock'],
          'max_calls':max_calls,'workers':workers,'protocol_sha256':digest,'attempts_per_window':attempts_per_window}
    atomic_json(out/'invocations'/(invocation+'.json'),meta)
    secret=os.environ.get(key_env,'')
    def redact(v): return str(v).replace(secret,'[REDACTED]') if secret else str(v)
    def run_one(row):
        nonlocal used,reason,streak
        uid=row['window_uid']; target=out/'results'/(uid+'.json'); repair=''; attempted=False
        result={'window_uid':uid,'source_group':row['source_group'],'role':'development_only',
            'protocol_sha256':digest,'mock':protocol['mock'],'model':protocol['model'],'invocation_id':invocation}
        last_partial=None
        for attempt in range(attempts_per_window):
            with mutex:
                if stop.is_set() or used>=max_calls: break
                used+=1
            if not attempted and target.exists():
                archive=out/'archived_results'/(uid+'_'+str(time.time_ns())+'.json'); archive.parent.mkdir(parents=True,exist_ok=True); target.replace(archive)
            attempted=True
            receipt={k:v for k,v in result.items() if k not in {'parsed','assessment'}}
            receipt.update(status='started',started_unix=time.time(),repair_feedback=repair)
            path=out/'attempts'/uid/(str(time.time_ns())+'_'+uuid.uuid4().hex[:8]+'.json'); atomic_json(path,receipt)
            try:
                raw=request([out/'cases'/uid[:20]/f'T{i}.jpg' for i in range(8)],protocol['prompt']+repair)
                receipt['raw']=redact(raw); parsed=raw_json(raw); provider_error(parsed); audit=assess(parsed)
                result={k:v for k,v in result.items() if k not in {'parsed','assessment','failure_kind','error','last_request_error','last_request_error_kind'}}
                result.update(status='success' if audit['core_valid'] else 'partial',parsed=parsed,assessment=audit)
                receipt['status']=result['status']
                if result['status']=='partial': last_partial=dict(result)
                structural=[i for i in audit['issues'] if i['path'].startswith('/frames/') or i['path'] in
                    {'/schema_version','/b5/label','/b5/basis_event_ids','/b5/evidence','/persons','/objects','/events','/links','/frames',
                     '/persons/budget','/objects/budget','/events/budget','/links/budget','/frames/budget'}]
                repair='\nRepair only these core structural defects, not the predicted label: '+json.dumps(structural[:12])+'. Return the complete object; never invent roles or observations.'
            except Exception as exc:
                kind=failure_kind(exc)
                receipt.update(status='provider_rejected' if kind=='provider_policy' else 'failed',failure_kind=kind,error=redact(exc))
                if last_partial is not None and kind!='provider_policy':
                    result=dict(last_partial,last_request_error=redact(exc),last_request_error_kind=kind)
                else:
                    result={k:v for k,v in result.items() if k not in {'parsed','assessment'}}
                    result.update(status=receipt['status'],failure_kind=kind,error=redact(exc))
                if kind in {'account','request','runtime'}:
                    with mutex: reason=kind; stop.set()
            receipt['elapsed_seconds']=time.time()-receipt['started_unix']; atomic_json(path,receipt)
            if result['status'] in {'success','provider_rejected'} or stop.is_set(): break
            if attempt+1<attempts_per_window: time.sleep(min(2**attempt,8))
        if not attempted: return
        result['finished_unix']=time.time(); atomic_json(target,result)
        with mutex:
            streak=streak+1 if result['status']=='failed' else 0
            if streak>=3: reason=reason or 'consecutive_errors'; stop.set()
        with output_lock:
            b5=(result.get('parsed') or {}).get('b5'); b5=b5 if isinstance(b5,dict) else {}
            prefix='MOCK-' if protocol['mock'] else ''
            report(out,protocol,rows,changed=uid)
            print(f'[{prefix}V95-SAVED] {uid[:20]} status={result["status"]} label={b5.get("label")} -> {out/"cases"/uid[:20]/"index.html"}',flush=True)
    try:
        report(out,protocol,rows)
        with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(run_one,pending))
        meta.update(status='finished',stop_reason=reason)
    except BaseException:
        meta['status']='interrupted'; raise
    finally:
        meta.update(finished_unix=time.time(),calls=used,budget_exhausted=used>=max_calls)
        atomic_json(out/'invocations'/(invocation+'.json'),meta)
    return report(out,protocol,rows)
