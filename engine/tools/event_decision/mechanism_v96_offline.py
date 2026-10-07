"""Immutable full-cohort offline replay, with no inference or review import path."""
import html
import json
import os
import shutil
import socket
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v94_report import page
from .mechanism_v95_store import load as load_v95
from .mechanism_v95_report import current, technical_metrics
from .mechanism_v96_contract import VERSION, specification, examples, assess_document
from .mechanism_v96_replay import derive_window

DEFAULT_SOURCE='governed_v95_typed_mechanism_development_20260914'
DEFAULT_TAG='governed_v96_mechanism_offline_20260914'


@contextmanager
def replay_lock(project,out,recover=False):
    out=Path(out).resolve(); root=(Path(project)/'runs').resolve()
    if out.parent!=root: raise ValueError('Output must be a direct runs child')
    root.mkdir(exist_ok=True)
    path=root/('.'+out.name+'.v96.lock')
    if path.exists() and recover:
        saved=read_json(path)
        if saved.get('host')!=socket.gethostname(): raise ValueError('Lock belongs to another host; do not force recovery')
        try: os.kill(int(saved['pid']),0)
        except ProcessLookupError: path.unlink()
        else: raise ValueError('Recorded PID still exists; no lock recovery')
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise ValueError('TAG locked; inspect running processes before RECOVER_STALE_LOCK=1')
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname()}).encode()); os.fsync(fd)
        yield
    finally:
        os.close(fd); path.unlink()


def tree_hashes(root):
    return {str(p.relative_to(root)):file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def code_hashes(project):
    own=[project/'tools/event_decision'/('mechanism_v96_'+s+'.py') for s in ('contract','replay','offline')]
    own += [project/'tools/mechanism_v96_offline_cli.py',project/'run_mechanism_v96_offline.sh']
    own += [project/'tools/event_decision'/s for s in ('safety.py','contracts.py','role_scoped.py')]
    return {str(p):file_sha256(p) for p in own}


def verify(out):
    out=Path(out); manifest=read_json(out/'protocol.json')
    if not isinstance(manifest,dict) or manifest.get('version')!=VERSION: raise ValueError('Not a V9.6 offline output')
    if read_json(out/'protocol_integrity.json',{}).get('sha256')!=file_sha256(out/'protocol.json'): raise ValueError('Protocol modified')
    source=portable(manifest['source_run'])
    if tree_hashes(source)!=manifest['source_tree_hashes']: raise ValueError('Source changed; preserve this TAG and inspect before creating a new one')
    for name,digest in manifest['code_hashes'].items():
        if file_sha256(portable(name))!=digest: raise ValueError('Replay code changed; use a new version/TAG')
    receipt=read_json(out/'completion.json')
    if receipt is not None:
        for name,digest in receipt['output_hashes'].items():
            target=(out/name).resolve()
            if out.resolve() not in target.parents or file_sha256(target)!=digest: raise ValueError('Derived output changed: '+name)
        actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'completion.json'}
        if actual!=set(receipt['output_hashes']): raise ValueError('Unexpected derived output files')
    return manifest,receipt


def render_case(out,row,record):
    esc=lambda value:html.escape(str(value))
    uid=row['window_uid'][:20]; body='<a href="../../index.html">全部窗口</a><h1>'+esc(row['video_id'])+'</h1>'
    body+='<p>零 API 派生诊断。原始 B5 判断未修改；不是新推断、人工标签或 graph/OT 分数。</p>'
    b5=record['raw_b5'] if isinstance(record['raw_b5'],dict) else {}
    body+='<p>帧区间 ['+str(row['start_frame'])+','+str(row['end_frame_exclusive'])+')；原 B5='+esc(b5.get('label','unavailable'))+'</p>'
    media='../../source_snapshot/cases/'+uid+'/'
    body+='<div class="frames">'+''.join(f'<figure><a href="{media}T{i}.jpg"><img loading="lazy" src="{media}T{i}.jpg" alt="T{i}"></a><figcaption>T{i}: {f}</figcaption></figure>' for i,f in enumerate(row['sampled_frame_indices']))+'</div>'
    body+='<h2>字段归一化</h2><pre>'+esc(json.dumps(record['mappings'],ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>仍缺什么</h2><table><tr><th>事件</th><th>原因</th><th>证据/字段</th><th>处理边界</th></tr>'
    for issue in record['root_causes']:
        body+='<tr>'+''.join('<td>'+esc(issue[k])+'</td>' for k in ('event_index','code','detail','action'))+'</tr>'
    body+='</table><h2>局部观察与关系</h2>'
    for e in record['observations']:
        body+='<details><summary>'+esc(e['id'])+' / '+esc(e['primary_kind'])+'</summary><pre>'+esc(json.dumps(e,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<details><summary>原始完整响应</summary><pre>'+esc(json.dumps(record['source_raw'],ensure_ascii=False,indent=2))+'</pre></details>'
    folder=out/'cases'/uid; folder.mkdir(parents=True,exist_ok=True)
    (folder/'index.html').write_text(page('V9.6 offline diagnosis',body),encoding='utf-8')


def execute(project,source,out,verify_only=False):
    project,source,out=(Path(p).resolve() for p in (project,source,out))
    if out==source or out in source.parents or source in out.parents: raise ValueError('Use an independent TAG, not a source subdirectory')
    if out.parent != (project/'runs').resolve(): raise ValueError('Output must be one TAG directly under project/runs')
    if out.exists():
        frozen,receipt=verify(out)
        if portable(frozen['source_run']).resolve()!=source: raise ValueError('Different source for existing TAG')
        if receipt:
            result=read_json(out/'summary.json'); return dict(result,execution='verified_existing_no_replay')
        if verify_only: raise ValueError('Output is incomplete; rerun without VERIFY_ONLY')
    elif verify_only: raise ValueError('TAG does not exist')
    if (source/'.v92.lock').exists(): raise ValueError('Source has an active lock')
    source_before=tree_hashes(source)
    protocol,rows=load_v95(source)
    if protocol['mock']: raise ValueError('Use the real V9.5 source, not a MOCK run')
    records=[current(source,row,protocol) for row in rows]
    if not rows or any(r['status'] not in ('success','partial','provider_rejected') for r in records):
        raise ValueError('Source acquisition must be complete; never infer missing responses')
    if len({row['window_uid'] for row in rows})!=len(rows): raise ValueError('Duplicate cohort IDs')
    if len({row['window_uid'][:20] for row in rows})!=len(rows): raise ValueError('Case page ID collision')
    old_metrics=technical_metrics(records,protocol)
    saved_gate=read_json(source/'integration/gate.json',{})
    if any(saved_gate.get(k)!=v for k,v in old_metrics.items()): raise ValueError('Source C gate is missing or stale; inspect source, do not silently replace it')
    example_checks=[assess_document(e) for e in examples()]
    if any(not e['valid'] for e in example_checks): raise ValueError('Prospective contract examples are invalid')
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():
        manifest={'version':VERSION,'operation':'offline_only','source_run':str(source),
                  'source_tree_hashes':source_before,'code_hashes':code_hashes(project),
                  'source_gate_sha256':file_sha256(source/'integration/gate.json'),
                  'source_selection_sha256':file_sha256(source/'selection.json'),
                  'selection_policy':'all_frozen_source_windows_same_order_no_outcome_selection',
                  'thresholds_unchanged':protocol['gates'],'remote_execution_authorized':False,
                  'acquisition_enabled':False,'human_review_requested':False,'training_authorized':False}
        atomic_json(out/'protocol.json',manifest)
        atomic_json(out/'protocol_integrity.json',{'sha256':file_sha256(out/'protocol.json')})
    snapshot=out/'source_snapshot'
    for name,digest in source_before.items():
        target=snapshot/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if file_sha256(target)!=digest: raise ValueError('Snapshot changed; do not overwrite it')
        else: shutil.copyfile(source/name,target)
    if tree_hashes(snapshot)!=source_before: raise ValueError('Snapshot copy mismatch')
    derived=[]; canonical_records=[]; root_counts=Counter(); mapping_counts=Counter(); table=[]
    for row,result in zip(rows,records):
        uid=row['window_uid']
        if result['status']=='provider_rejected':
            record={'window_uid':uid,'status':'excluded_provider_policy','root_causes':[], 'new_visual_observation':False}
            canonical_records.append({'status':'provider_rejected'}); derived.append(record); continue
        record=derive_window(result['parsed'],result['assessment'])
        record.update(window_uid=uid,video_id=row['video_id'],source_group=row['source_group'],
                      selection_reasons=row['selection_reasons'],source_result_sha256=file_sha256(source/'results'/(uid+'.json')),
                      source_raw=result['parsed'],status='offline_diagnostic_only')
        root_counts.update(record['root_counts']); mapping_counts.update(m['rule'] for m in record['mappings'])
        derived.append(record)
        canonical_records.append({'status':result['status'],'assessment':record['alias_replay_assessment']})
        render_case(out,row,record)
        old=sum(e['valid'] for e in record['source_assessment']['events'])
        new=sum(e['valid'] for e in record['alias_replay_assessment']['events'])
        b5=record['raw_b5'] if isinstance(record['raw_b5'],dict) else {}
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (row['video_id'],row['start_frame'],b5.get('label','unavailable'),old,new,len(record['mappings'])))+
                     '<td><a href="cases/'+uid[:20]+'/index.html">查看原图和诊断</a></td></tr>')
        print('[OFFLINE-CASE] '+uid[:20]+' mappings='+str(len(record['mappings']))+' -> '+str(out/'cases'/uid[:20]/'index.html'),flush=True)
    alias_metrics=technical_metrics(canonical_records,protocol)
    all_events=[e for r in derived for e in r.get('observations',[])]
    labels=Counter((r['raw_b5'].get('label') if isinstance(r.get('raw_b5'),dict) else 'unavailable') for r in derived if 'raw_b5' in r)
    summary={'version':VERSION,'execution':'replayed_all_windows','source_windows':len(rows),
        'videos':len({r['video_id'] for r in rows}),'source_groups':len({r['source_group'] for r in rows}),
        'source_counts':dict(Counter(r['status'] for r in records)),
        'raw_B5_labels_unchanged':dict(labels),'source_metrics':old_metrics,
        'alias_only_same_v95_validator_metrics_not_new_inference':alias_metrics,
        'mapping_counts':dict(mapping_counts),'windows_with_mapping':sum(bool(r.get('mappings')) for r in derived),
        'event_alias_only_format_recoveries':sum(e['alias_only_format_recovery'] for e in all_events),
        'root_cause_counts':dict(root_counts),'same_cohort':True,'native_contract_examples_validated':len(example_checks),
        'remote_calls':0,'new_media_decodes':0,'new_vlm_responses':0,'original_run_files_unchanged':len(source_before),
        'decision':'OFFLINE_REPLAY_COMPLETE_NOT_DEPLOYABLE',
        'new_requests_decided':False,'remote_execution_authorized':False,'human_review_requested':False,
        'scoring_authorized':False,'training_authorized':False,'ready_for_shadow_integration':False,
        'formal_accuracy':None,'formal_AP':None,
        'claim_limit':'encoding availability and evidence obligations only; no new category or visual truth'}
    write_jsonl(out/'records.jsonl',derived)
    write_jsonl(out/'mapping_audit.jsonl',[dict(window_uid=r['window_uid'],**m) for r in derived for m in r.get('mappings',[])])
    write_jsonl(out/'root_causes.jsonl',[dict(window_uid=r['window_uid'],**e) for r in derived for e in r['root_causes']])
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'prospective_contract.json',specification())
    atomic_json(out/'prospective_examples.json',examples())
    atomic_json(out/'gate.json',{'decision':summary['decision'],'source_gate_sha256':file_sha256(source/'integration/gate.json'),
        'source_technical_ready':old_metrics['technical_ready'],'source_thresholds':protocol['gates'],
        'no_gate_promotion_from_alias_replay':True,'human_review_requested':False,'remote_execution_authorized':False,
        'ready_for_shadow_integration':False,'training_authorized':False,'benchmark_evaluation_authorized':False})
    atomic_json(out/'next_request_plan.json',{'status':'DRAFT_NOT_AUTHORIZED','remote_execution_authorized':False,
        'decision':'review offline obligations before deciding whether any new visual request is necessary',
        'required_before_new_calls':['settle one primary vocabulary and observation-only records',
            'explicit relation arguments/instruments/context partition; no automatic legacy reassignment',
            'identity and edit-inference evidence scope; no B5 promotion by alias mapping',
            'freeze new version and a complete, outcome-independent paired cohort'],
        'complete_control_cohort':[{k:r[k] for k in ('window_uid','video_id','source_group','selection_reasons')} for r in rows],
        'source_selection_sha256':file_sha256(source/'selection.json'),
        'rules':['no cherry-picked retries','retain no/unknown and residual cases','no automatic legacy human-answer import']})
    body='<h1>V9.6 零 API 契约重放</h1><p>全部固定名单；保留原始响应。归一化统计不是新推断或准确率。</p>'
    body+='<p>目前不请求人工复核，不启用 API、不训练、不改变 graph/OT 分数。</p>'
    body+='<table><tr><th>视频</th><th>开始帧</th><th>原 B5</th><th>原有效事件数</th><th>仅别名重放有效事件数</th><th>映射数</th><th>证据</th></tr>'+''.join(table)+'</table>'
    body+='<details><summary>完整诊断统计</summary><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></details>'
    (out/'index.html').write_text(page('V9.6 offline contract replay',body),encoding='utf-8')
    if tree_hashes(source)!=source_before: raise ValueError('Source changed while replaying; no completion approval')
    outputs={k:v for k,v in tree_hashes(out).items() if k!='completion.json'}
    atomic_json(out/'completion.json',{'remote_calls':0,'source_unchanged':True,'output_hashes':outputs})
    return summary
