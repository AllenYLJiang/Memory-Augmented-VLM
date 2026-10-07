"""Bounded, resumable B5 development-only calls; never queries reserved sources."""
from __future__ import annotations

import html
import json
import os
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from .contracts import file_sha256, iter_jsonl, read_json, write_jsonl
from .role_scoped import VERSION, assert_bound, atomic_json, portable, valid_screen


@contextmanager
def run_lock(out, recover=False):
    path=out/'.v92.lock'
    if path.exists() and recover:
        old=read_json(path)
        if old.get('hostname')!=socket.gethostname():
            raise ValueError('lock belongs to another host; verify that process before manual recovery')
        try: os.kill(int(old['pid']),0)
        except ProcessLookupError: path.unlink()
        else: raise ValueError('recorded PID still exists; will not remove lock')
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise ValueError('run locked; stop duplicate launcher or use --recover-stale-lock after checking processes')
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'hostname':socket.gethostname()}).encode())
        os.fsync(fd)
        yield
    finally:
        os.close(fd);path.unlink()


def media_for(row,out):
    import cv2
    from .enrollment_review import extract_clip
    from .local_screen import local_frames
    uid=row['window_uid']; folder=out/'development/cases'/uid[:20]
    receipt=folder/'media.json'; cached=read_json(receipt)
    if cached and cached.get('source_media_sha256')==row['source_media_sha256']:
        if all((folder/name).is_file() and file_sha256(folder/name)==digest
               for name,digest in cached['files'].items()):
            return cached
    folder.mkdir(parents=True,exist_ok=True)
    clip=folder/'clip.mp4'
    extract_clip(portable(row['video_path']),row['start_frame'],clip)
    local=local_frames(clip,0)
    local['sampled_frame_indices']=row['sampled_frame_indices']
    local['screening_method']='complete_96_frame_clip_plus_eight_dhash_frames_v92'
    local['evidence_domain']='decoded_review_clip_crf18_not_original_rgb'
    cap=cv2.VideoCapture(str(clip)); paths=[]
    try:
        wanted={int(i*95/7):i for i in range(8)}; count=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if count in wanted:
                path=folder/f'T{wanted[count]}.jpg'
                if not cv2.imwrite(str(path),frame,[int(cv2.IMWRITE_JPEG_QUALITY),92]):
                    raise ValueError('image write failed')
                paths.append(str(path))
            count+=1
        if count!=96 or len(paths)!=8: raise ValueError('media must have 96 frames and eight exact images')
    finally: cap.release()
    value={'source_media_sha256':row['source_media_sha256'],'decoded_frames':96,
           'image_paths':paths,'local_candidate':local,
           'files':{p.name:file_sha256(p) for p in [clip]+[Path(x) for x in paths]}}
    atomic_json(receipt,value)
    return value


def mock_response():
    return {'b5_presence':'uncertain','b5_probability':0.5,'confidence':'low',
        'visible_evidence':'MOCK plumbing only; not a visual judgment',
        'normal_or_other_explanation':'MOCK unknown','same_actor_same_event':'uncertain',
        'supporting_bins':[], 'per_frame':[{'bin':i,'b5_support':'uncertain','evidence':'MOCK'} for i in range(8)]}


def saved_result(path):
    try:
        value=read_json(path)
        return value if value is None or isinstance(value,dict) else {'status':'invalid_saved_response'}
    except (ValueError,UnicodeError):
        return {'status':'invalid_saved_response'}


def provider_factory(protocol,key_env):
    if protocol['mock']:
        return lambda evidence,prompt: json.dumps(mock_response()), json.loads
    key=os.environ.get(key_env,'').strip()
    if not key: raise ValueError(f'missing {key_env}; export a real key in this shell')
    code_dir=portable(protocol['code_dir'])
    sys.path.insert(0,str(code_dir/'src'))
    from structural_vlm_binary.vlm.dashscope_backend import DashScopeVLM, DashScopeVLMConfig, parse_json_like
    backend=DashScopeVLM(DashScopeVLMConfig(model=protocol['model'],api_key_env=key_env,max_retries=0,json_parse_retries=1))
    def request(evidence,prompt):
        content=[]
        for i,path in enumerate(evidence['image_paths']):
            content.extend([{'text':f'T{i}'},{'image':portable(path).resolve().as_uri()}])
        content.append({'text':prompt})
        # Explicit key mapping; no filename, GT, graph scores or old answers.
        return backend._call_messages([{'role':'user','content':content}],api_key=key,
                                      max_tokens=2048,temperature=0,vl_high_resolution_images=True)
    return request,parse_json_like


def summarize(out, protocol=None):
    protocol=protocol or assert_bound(out)
    windows=list(iter_jsonl(out/'inventory/development_windows.jsonl'))
    outcomes=[]; candidates=[]; counts={k:0 for k in ('yes','no','uncertain','failed','pending')}
    links=[]; confirmed=0
    for row in windows:
        uid=row['window_uid']; folder=out/'development/cases'/uid[:20]
        media=read_json(folder/'media.json'); record=saved_result(out/'development/results'/f'{uid}.json')
        if record and record.get('status')=='success':
            try: valid_screen(record['parsed'])
            except (ValueError,TypeError,KeyError): record={**record,'status':'invalid_saved_response'}
        state=record['parsed']['b5_presence'] if record and record.get('status')=='success' else ('failed' if record else 'pending')
        counts[state]+=1
        if record: outcomes.append(record)
        if media:
            # Never join model decisions into the local enrollment candidate file.
            local={k:v for k,v in row.items() if k not in ('human_window_target','gold_metric_mask')}
            candidates.append({**local,**media['local_candidate']})
        score=record['parsed']['b5_probability'] if record and record.get('status')=='success' else None
        if state=='yes': confirmed+=1
        detail=f'development/cases/{uid[:20]}/index.html'
        if media:
            images=''.join(f'<figure><img src="T{i}.jpg" alt="T{i}"><figcaption>T{i}: frame {frame}</figcaption></figure>'
                           for i,frame in enumerate(row['sampled_frame_indices']))
            payload=html.escape(json.dumps(record or {'status':'pending'},ensure_ascii=False,indent=2))
            page=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B5 development evidence</title><style>body{{font:16px Arial;margin:24px}}img{{max-width:100%}}main{{max-width:1100px;margin:auto}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}figure{{margin:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}video{{width:100%;max-height:500px}}</style><main><a href="../../../index.html">All cases</a><h1>B5 development evidence</h1><p>Model screening, not ground truth; previously exposed source; adaptation only.</p><h2>{html.escape(row["video_id"])}</h2><p>Frames [{row["start_frame"]}, {row["end_frame_exclusive"]}) | {state} | score={score}</p><video controls preload="none" src="clip.mp4"></video><div class="frames">{images}</div><pre>{payload}</pre></main></html>'
            (folder/'index.html').write_text(page,encoding='utf-8')
        links.append(f'<tr><td>{html.escape(row["video_id"])}</td><td>{row["start_frame"]}-{row["end_frame_exclusive"]-1}</td><td>{state}</td><td>{score if score is not None else ""}</td><td>'+(f'<a href="{detail}">Evidence</a>' if media else '')+'</td></tr>')
    write_jsonl(out/'development/predictions.jsonl',outcomes)
    write_jsonl(out/'development/local_candidates.jsonl',candidates)
    attempts=list((out/'development/attempts').glob('*/*.json'))
    result={'version':VERSION,'mock':protocol['mock'],'planned_windows':len(windows),
        'counts':counts,'model_positive_candidates_not_gold':confirmed,'attempt_receipts':len(attempts),
        'remote_attempts_upper_bound':0 if protocol['mock'] else len(attempts),
        'status':'SCREEN_COMPLETE' if not counts['pending'] and not counts['failed'] else 'SCREEN_INCOMPLETE',
        'locked_windows_queried':0,'formal_protocol_frozen':False,'accuracy':None,'AP':None,
        'note':'Sparse-frame B5 labels are model annotations, not continuous frame GT; all outcomes retained.'}
    atomic_json(out/'development/summary.json',result)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B5 development screening</title><style>body{font:15px Arial;margin:24px;color:#222}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}td:first-child{max-width:600px}pre{white-space:pre-wrap}main{overflow-x:auto}</style><h1>B5 development screening</h1><p>Previously exposed sources. Model judgments, not human GT. Reserved sources are not queried.</p><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre><main><table><thead><tr><th>Video</th><th>Frames</th><th>Model judgment</th><th>Probability</th><th>Case</th></tr></thead><tbody>'+''.join(links)+'</tbody></table></main></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    return result


def screen(out, *, max_calls=120, workers=1, attempts_per_window=2, retry_failed=False, max_consecutive_errors=3,
           key_env='DASHSCOPE_API_KEY', provider=None, media_loader=media_for):
    if min(max_calls,workers,attempts_per_window,max_consecutive_errors)<1 or workers>8:
        raise ValueError('positive budgets required; workers must be 1..8')
    protocol=assert_bound(out)
    rows=list(iter_jsonl(out/'inventory/development_windows.jsonl'))
    reserved=set(protocol['reserved_source_groups']); development=set(protocol['development_source_groups'])
    for r in rows:
        if r['source_group'] in reserved or r['source_group'] not in development or r['allowed_roles']!=['adaptation']:
            raise ValueError('development-only source firewall rejected manifest')
    request,parse=provider or provider_factory(protocol,key_env)
    digest=file_sha256(out/'protocol.json'); used=0; streak=0
    budget_lock=threading.Lock(); output_lock=threading.Lock(); stop=threading.Event()
    # Recognized by the existing history scanner, including interrupted calls.
    write_jsonl(out/'development/review_exposure_manifest.jsonl',[
        {'source_group':g,'exposure':'development_pool_committed_not_proof_of_completed_inference',
         'role':'adaptation','protocol_sha256':digest} for g in sorted(development)])
    def redact(value):
        key=os.environ.get(key_env,'')
        return str(value).replace(key,'[REDACTED]') if key else str(value)
    def one(row):
        nonlocal used,streak
        if stop.is_set(): return
        uid=row['window_uid']; path=out/'development/results'/f'{uid}.json'
        old=saved_result(path)
        if old:
            if old.get('protocol_sha256') not in (None,digest): raise ValueError('response protocol mismatch')
            if old.get('status')=='success':
                try:
                    if old.get('protocol_sha256')!=digest: raise ValueError('unbound saved response')
                    valid_screen(old['parsed'])
                    evidence=media_loader(row,out)
                    actual={Path(p).name:file_sha256(portable(p)) for p in evidence['image_paths']}
                    if actual!=old.get('image_sha256'): raise ValueError('response media hashes changed')
                    return
                except (ValueError,TypeError,KeyError): pass
            elif old.get('status')!='invalid_saved_response' and not retry_failed: return
            archive=out/'development/archived_results'/f'{uid}_{time.time_ns()}.json'
            archive.parent.mkdir(parents=True,exist_ok=True); path.replace(archive)
        record={'window_uid':uid,'video_id':row['video_id'],'source_group':row['source_group'],
                'start_frame':row['start_frame'],'end_frame_exclusive':row['end_frame_exclusive'],
                'protocol_sha256':digest,'role':'adaptation','model':protocol['model'],'mock':protocol['mock'],
                'label_evidence_level':'vlm_sparse_frame_screen_not_gold','status':'failed'}
        try:
            with budget_lock:
                if used>=max_calls or stop.is_set(): return
            evidence=media_loader(row,out)
            record['sampled_frame_indices']=row['sampled_frame_indices']
            record['image_sha256']={Path(p).name:file_sha256(portable(p)) for p in evidence['image_paths']}
            attempted=False
            for attempt in range(attempts_per_window):
                with budget_lock:
                    if used>=max_calls or stop.is_set(): break
                    used+=1
                attempted=True
                receipt_path=out/'development/attempts'/uid/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}.json'
                receipt={'protocol_sha256':digest,'window_uid':uid,'source_group':row['source_group'],
                         'started_unix':time.time(),'status':'started','mock':protocol['mock']}
                atomic_json(receipt_path,receipt)
                try:
                    raw=request(evidence,protocol['prompt'])
                    receipt['raw']=redact(raw)
                    parsed=valid_screen(parse(raw))
                    record.update(status='success',parsed=parsed)
                    record.pop('error',None)
                    receipt['status']='success'
                except Exception as exc:
                    receipt.update(status='failed',error=redact(exc))
                    record['error']=redact(exc)
                receipt['elapsed_seconds']=time.time()-receipt['started_unix']
                atomic_json(receipt_path,receipt)
                if record['status']=='success': break
            if not attempted: return
            record['finished_unix']=time.time()
        except Exception as exc:
            record['error']=redact(exc)
        atomic_json(path,record)
        with budget_lock:
            streak=0 if record['status']=='success' else streak+1
            if streak>=max_consecutive_errors: stop.set()
        with output_lock:
            symbol='[B5-CANDIDATE]' if record.get('parsed',{}).get('b5_presence')=='yes' else '[B5-SCREEN]'
            print(f'{symbol} {row["video_id"]} [{row["start_frame"]},{row["end_frame_exclusive"]}) {record.get("parsed",{}).get("b5_presence",record["status"])} -> {path}',flush=True)
            summarize(out,protocol)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one,rows))
    result=summarize(out,protocol)
    result['calls_this_invocation']=used;result['budget_this_invocation']=max_calls
    result['stopped_after_error_streak']=stop.is_set()
    atomic_json(out/'development/summary.json',result)
    return result
