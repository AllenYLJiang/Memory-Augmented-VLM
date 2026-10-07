"""Immutable V9.3 snapshot and outcome-independent-within-V9.4 selection."""
import json
import shutil
from collections import Counter
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .development_mechanism_v93 import load_run as load_v93, validate as validate_v93
from .mechanism_v94_contract import VERSION, PROMPT, RULES, CATEGORIES, FEATURES

DEFAULT_SOURCE = 'governed_v93_b5_mechanism_development_20260913'
DEFAULT_TAG = 'governed_v94_scoped_mechanism_development_20260913'
GATES = {'minimum_core_fraction': .95, 'minimum_event_valid_fraction': .95,
         'minimum_frame_valid_fraction': .95, 'minimum_link_valid_fraction': .95, 'minimum_consistency_fraction': .95, 'minimum_resolved_review_fraction': .8,
         'minimum_boundary_agreement': .9, 'minimum_supported_claims_fraction': .9,
         'minimum_b5_positive_reviews': 4, 'minimum_b5_negative_reviews': 4,
         'minimum_sources_each_label': 2, 'require_all_selected_reviewed': True}


def raw_json(raw):
    if isinstance(raw, dict): return raw
    if not isinstance(raw, str): return None
    try: return json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
    except (ValueError, TypeError): return None


def selection(rows, results):
    selected=[]
    for r in rows:
        if r['exclusion']: continue
        result=results.get(r['window_uid'],{}); reasons=[]; parsed=result.get('parsed',{})
        if r['design_feedback']: reasons.append('design_case')
        if result.get('status')=='failed': reasons.append('v93_failed_'+result.get('failure_kind','unknown'))
        if not r['design_feedback'] and result.get('status')=='success' and r['baseline']['parsed']['b5_presence']=='no' and parsed.get('b5_presence')=='yes': reasons.append('additional_no_to_yes')
        if result.get('status')=='success' and parsed.get('phase')=='residual_only': reasons.append('residual_only')
        if reasons: selected.append({**r,'selection_reasons':reasons})
    return selected


def code_paths(project, protocol):
    names=['mechanism_v94_contract.py','mechanism_v94_store.py','mechanism_v94_runner.py','mechanism_v94_report.py','mechanism_v94_stage3.py']
    paths=[project/'tools/event_decision'/n for n in names]
    paths += [project/'tools/mechanism_v94_cli.py',project/'run_mechanism_v94.sh']
    paths += [portable(p) for p in protocol['code_hashes']]
    paths += [project/'tools/event_decision/safety.py']
    return sorted(set(p.resolve() for p in paths))


def legacy_audit(snapshot):
    """Salvage observations for inspection only, never accept legacy failures."""
    rows=[]
    for path in sorted((snapshot/'attempts').glob('*/*.json')):
        receipt=read_json(path); raw=raw_json(receipt.get('raw')); issue=None
        if raw is not None:
            try: validate_v93(raw)
            except (TypeError,ValueError,KeyError) as exc: issue=str(exc)
        else: issue='no decodable model object'
        observations=[]
        if isinstance(raw,dict):
            frames=raw.get('per_frame',[])
            if isinstance(frames,list):
                for i,frame in enumerate(frames):
                    if not isinstance(frame,dict): continue
                    fields={}
                    for k in ('b5_support','injury_trace'):
                        v=frame.get(k); valid=isinstance(v,str) and v in {'yes','no','uncertain'}
                        fields[k]={'raw':v,'valid':valid,'observed':valid and v!='uncertain'}
                    fields['bin']={'raw':frame.get('bin'),'valid':type(frame.get('bin')) is int and frame['bin']==i and i<8}
                    fields['evidence']={'raw':frame.get('evidence'),'valid':isinstance(frame.get('evidence'),str) and bool(frame['evidence'].strip())}
                    observations.append(fields)
        rows.append({'window_uid':receipt['window_uid'],'receipt':str(path.relative_to(snapshot)),
                     'receipt_sha256':file_sha256(path),'original_status':receipt['status'],
                     'legacy_contract_valid':issue is None,'legacy_error':issue,
                     'raw_model_object':raw,'frame_field_audit':observations,
                     'promoted_to_success':False,'usable_as_training_labels':False,
                     'category_codes_semantically_unverified':True,
                     'legacy_context_is_global_not_convertible_to_local_scope':True})
    return rows


def prepare(project, source, out, *, mock=False, max_tokens=6144):
    project,source,out=map(lambda p:Path(p).resolve(),(project,source,out))
    if out.exists(): raise ValueError('Use a new TAG for STEP=1; no overwriting prepared runs')
    if source==out or source in out.parents or out in source.parents: raise ValueError('New TAG must be independent from V9.3')
    if type(max_tokens) is not int or not 2048<=max_tokens<=8192: raise ValueError('max_tokens must be 2048..8192')
    original,rows=load_v93(source)
    if original['mock']: raise ValueError('Source must be real V9.3, not MOCK')
    if (source/'.v92.lock').exists(): raise ValueError('V9.3 appears to be running; stop it before snapshot')
    results={p.stem:read_json(p) for p in (source/'results').glob('*.json')}
    expected={r['window_uid'] for r in rows if not r['exclusion']}
    if set(results)!=expected: raise ValueError('Need a terminal V9.3 receipt for every eligible window')
    digest=file_sha256(source/'protocol.json')
    for uid,r in results.items():
        if r.get('window_uid')!=uid or r.get('protocol_sha256')!=digest or r.get('mock') is not False or r.get('status') not in {'success','failed','provider_rejected'}:
            raise ValueError('Unbound/nonterminal V9.3 result')
        if r['status']=='success': validate_v93(r['parsed'])
    chosen=selection(rows,results)
    if not chosen: raise ValueError('Empty predeclared development subset')
    # New policy rejections, if any, remain excluded from new calls too.
    if any(results[r['window_uid']]['status']=='provider_rejected' for r in chosen): raise ValueError('Never select provider-policy rejected input for re-query')
    files=[source/n for n in ('protocol.json','integrity.json','windows.json','summary.json','feedback_input.jsonl')]
    files+=sorted((source/'results').glob('*.json'))+sorted((source/'attempts').glob('*/*.json'))
    source_hashes={str(p.relative_to(source)):file_sha256(p) for p in files}
    out.mkdir(parents=True); snapshot=out/'source_snapshot'; snapshot.mkdir()
    for p in files:
        dest=snapshot/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for row in chosen:
        folder=out/'cases'/row['window_uid'][:20];folder.mkdir(parents=True)
        for name,digest in row['media_hashes'].items():
            shutil.copyfile(source/'cases'/row['window_uid'][:20]/name,folder/name)
            if file_sha256(folder/name)!=digest: raise ValueError('Media changed during snapshot')
    for rel,digest in source_hashes.items():
        if file_sha256(source/rel)!=digest or file_sha256(snapshot/rel)!=digest: raise ValueError('Source changed during snapshot; preserve incomplete TAG')
    audit=legacy_audit(snapshot)
    write_jsonl(out/'offline/legacy_field_audit.jsonl',audit)
    atomic_json(out/'offline/audit_summary.json',{'receipts':len(audit),'legacy_valid':sum(r['legacy_contract_valid'] for r in audit),
        'original_status':dict(Counter(r['original_status'] for r in audit)), 'promoted_failures':0,'remote_calls':0,
        'source_hashes':source_hashes,'not_a_new_v94_prediction':True})
    atomic_json(out/'selection.json',chosen)
    atomic_json(out/'selection_summary.json',{'selected':len(chosen),'design_cases':sum(bool(r['design_feedback']) for r in chosen),
        'reason_counts':dict(Counter(reason for r in chosen for reason in r['selection_reasons'])),
        'policy':'union_all_design_all_terminal_failures_all_extra_no_to_yes_all_residual; no V94 outcomes used',
        'development_only':True,'requires_no_history_reentry':True})
    atomic_json(out/'boundary_definitions.json',{'categories':CATEGORIES,'rules':RULES,'thresholds':GATES})
    protocol={'version':VERSION,'mock':bool(mock),'source_run':str(source),'source_protocol_sha256':file_sha256(source/'protocol.json'),
              'role':'design_exposed_development_only','locked_authorized':False,'model':original['model'],
              'code_dir':original['code_dir'],'prompt':PROMPT,'temperature':0,'max_tokens':max_tokens,
              'development_sources':original['development_sources'],'reserved_sources':original['reserved_sources'],
              'gates':GATES,'features':FEATURES,'retention':'first core-valid response regardless of label; otherwise last partial, all attempts retained',
              'code_hashes':{str(p):file_sha256(p) for p in code_paths(project,original)},
              'data_hashes':{str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file()}}
    atomic_json(out/'protocol.json',protocol);atomic_json(out/'integrity.json',{'protocol_sha256':file_sha256(out/'protocol.json')})
    return {'status':'PREPARED_ZERO_API','selected':len(chosen),'mock':bool(mock),'remote_calls':0}


def load(out):
    out=Path(out);p=read_json(out/'protocol.json')
    if not isinstance(p,dict) or p.get('version')!=VERSION or p.get('role')!='design_exposed_development_only' or p.get('locked_authorized') is not False:
        raise ValueError('Not a V9.4 development protocol')
    if read_json(out/'integrity.json',{}).get('protocol_sha256')!=file_sha256(out/'protocol.json'): raise ValueError('Protocol changed; use new TAG')
    for path,digest in p['code_hashes'].items():
        if file_sha256(portable(path))!=digest: raise ValueError('Frozen code changed: '+path)
    for rel,digest in p['data_hashes'].items():
        path=(out/rel).resolve()
        if out.resolve() not in path.parents or file_sha256(path)!=digest: raise ValueError('Snapshot/media changed: '+rel)
    rows=read_json(out/'selection.json')
    for r in rows:
        if r['exclusion'] or r['source_group'] not in p['development_sources'] or r['source_group'] in p['reserved_sources']:
            raise ValueError('Development source firewall failed')
    return p,rows
