"""Same-cohort snapshot, descriptive legacy audit, and immutable new protocol."""
import json
import shutil
from collections import Counter
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v94_store import load as load_v94, GATES as V94_GATES
from .mechanism_v94_contract import assess as assess_v94
from .mechanism_v95_contract import VERSION, PROMPT, RULES, CATEGORIES, LIMITS, FEATURES, assess, examples

DEFAULT_SOURCE = 'governed_v94_scoped_mechanism_development_20260913'
DEFAULT_TAG = 'governed_v95_typed_mechanism_development_20260914'
GATES = dict(V94_GATES, minimum_numeric_valid_fraction=.95, minimum_event_consistency_fraction=.95)


def offline_audit(snapshot, rows):
    audits = []; reason_counts = Counter(); frames = 0
    for row in rows:
        old = read_json(snapshot/'results'/(row['window_uid']+'.json'))
        raw = old.get('parsed')
        assessment = assess_v94(raw)
        if old['status'] in {'success','partial'} and assessment != old['assessment']:
            raise ValueError('V9.4 assessment changed; do not replay with a changed validator')
        raw = raw if isinstance(raw, dict) else {}
        valid_frames = sum(all(assessment['fields'].get(f'/frames/{i}/{k}',{}).get('valid')
            for k in ('action','constraint','injury','quality','evidence')) for i in range(8)) if assessment['fields'].get('/frames/order',{}).get('valid') else 0
        frames += valid_frames
        actions = []
        for i, e in enumerate(raw.get('events', [])):
            if not isinstance(e, dict): continue
            reasons = []
            if e.get('type') in {'injury_trace','other_event'} and e.get('binding',{}).get('state') == 'observed': reasons.append('needs_typed_non_interpersonal_relationship')
            if e.get('boundary',{}).get('basis') == 'mutual_combat': reasons.append('review_reciprocal_participants_not_permanent_attacker')
            if e.get('context_scope',{}).get('category_required') is True: reasons.append('legacy_context_cannot_be_auto_resolved')
            if reasons: actions.append({'event_index':i, 'reasons':reasons})
            reason_counts.update(reasons)
        audits.append({'window_uid':row['window_uid'],'original_status':old['status'],
            'original_b5':raw.get('b5'), 'original_assessment':assessment,
            'numeric_available': assessment['fields'].get('/b5/probability',{}).get('value') is not None,
            'frame_valid_independent_of_core':valid_frames,'typed_design_followup':actions,
            'v95_prediction':None,'promoted_to_success':False,'used_as_gold':False})
    return audits, {'windows':len(rows),'independent_valid_frame_records':frames,'frame_denominator':8*len(rows),
                    'design_issue_counts':dict(reason_counts),'remote_calls':0,'promoted_results':0,
                    'identity_fabrication':False,'legacy_context_auto_flipped':False}


def prepare(project, source, out, *, mock=False, max_tokens=8192):
    project, source, out = (Path(p).resolve() for p in (project,source,out))
    if out.exists(): raise ValueError('STEP=A requires a new TAG; use B/C to resume')
    if source == out or source in out.parents or out in source.parents: raise ValueError('Use an independent destination')
    if type(max_tokens) is not int or not 2048 <= max_tokens <= 16384: raise ValueError('max_tokens must be 2048..16384')
    original, rows = load_v94(source)
    if original['mock']: raise ValueError('Use the real completed V9.4 source')
    if (source/'.v92.lock').exists(): raise ValueError('Source has a run lock; finish it before snapshot')
    selected = read_json(source/'selection_summary.json')
    expected = {r['window_uid'] for r in rows}
    if len(expected) != len(rows) or len({u[:20] for u in expected}) != len(rows): raise ValueError('Duplicate selected IDs')
    if {p.stem for p in (source/'results').glob('*.json')} != expected: raise ValueError('Source does not cover its full cohort')
    digest = file_sha256(source/'protocol.json')
    for uid in expected:
        result = read_json(source/'results'/(uid+'.json'))
        if result.get('window_uid') != uid or result.get('protocol_sha256') != digest or result.get('mock') is not False:
            raise ValueError('Unbound V9.4 result')
        if result.get('status') not in {'success','partial'}: raise ValueError('Source must be complete and non-policy-rejected; never re-query blocked media')
    checked_examples = [assess(e) for e in examples()]
    if any(a['issues'] for a in checked_examples): raise ValueError('Prompt examples fail new contract')
    files = [source/n for n in ('protocol.json','integrity.json','selection.json','selection_summary.json','summary.json','boundary_definitions.json')]
    for folder in ('results','attempts','invocations','integration'):
        files += sorted((source/folder).rglob('*.json'))
    source_hashes = {str(p.relative_to(source)):file_sha256(p) for p in files}
    out.mkdir(parents=True); snapshot=out/'source_snapshot'; snapshot.mkdir()
    for p in files:
        target=snapshot/p.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
    # Cohort and media are unchanged, including all negative flips and residual cases.
    shutil.copyfile(source/'selection.json',out/'selection.json')
    for row in rows:
        folder=out/'cases'/row['window_uid'][:20]; folder.mkdir(parents=True)
        for name, sha in row['media_hashes'].items():
            target=folder/name; shutil.copyfile(source/'cases'/row['window_uid'][:20]/name,target)
            if file_sha256(target) != sha: raise ValueError('Media changed while copying')
    for rel, sha in source_hashes.items():
        if file_sha256(source/rel) != sha or file_sha256(snapshot/rel) != sha: raise ValueError('Source changed during snapshot')
    audit, summary=offline_audit(snapshot,rows)
    summary['source_hashes']=source_hashes; summary['valid_complete_prompt_examples']=len(checked_examples)
    write_jsonl(out/'offline/legacy_audit.jsonl',audit); atomic_json(out/'offline/audit_summary.json',summary)
    atomic_json(out/'selection_summary.json',dict(selected, policy='exact_V94_cohort_same_order_no_label_based_subselection', source_selection_sha256=file_sha256(source/'selection.json')))
    atomic_json(out/'definitions.json',{'categories':CATEGORIES,'rules':RULES,'limits':LIMITS,'examples':examples(),'gates':GATES})
    own = [project/'tools/event_decision'/('mechanism_v95_'+n+'.py') for n in ('contract','store','runner','report','stage3')]
    own += [project/'tools/mechanism_v95_cli.py', project/'run_mechanism_v95.sh']
    paths=sorted(set(p.resolve() for p in own+[portable(p) for p in original['code_hashes']]))
    p={'version':VERSION,'role':'design_exposed_development_only','mock':bool(mock),'locked_authorized':False,
       'source_run':str(source),'source_protocol_sha256':digest,'model':original['model'],'code_dir':original['code_dir'],
       'temperature':0,'max_tokens':max_tokens,'prompt':PROMPT,'features':FEATURES,'gates':GATES,
       'development_sources':original['development_sources'],'reserved_sources':original['reserved_sources'],
       'retention':'first core-valid response, regardless of label; partials retained; no semantic retry',
       'code_hashes':{str(path):file_sha256(path) for path in paths},
       'data_hashes':{str(path.relative_to(out)):file_sha256(path) for path in out.rglob('*') if path.is_file()}}
    atomic_json(out/'protocol.json',p); atomic_json(out/'integrity.json',{'protocol_sha256':file_sha256(out/'protocol.json')})
    return {'status':'PREPARED_ZERO_API','selected':len(rows),'same_cohort':True,'remote_calls':0,'mock':bool(mock)}


def load(out):
    out=Path(out); p=read_json(out/'protocol.json',{})
    if p.get('version') != VERSION or p.get('role') != 'design_exposed_development_only' or p.get('locked_authorized') is not False:
        raise ValueError('Not a V9.5 development protocol')
    if read_json(out/'integrity.json',{}).get('protocol_sha256') != file_sha256(out/'protocol.json'): raise ValueError('Protocol changed')
    for path,sha in p['code_hashes'].items():
        if file_sha256(portable(path)) != sha: raise ValueError('Frozen code changed: '+path)
    for rel,sha in p['data_hashes'].items():
        path=(out/rel).resolve()
        if out.resolve() not in path.parents or file_sha256(path) != sha: raise ValueError('Frozen inputs changed: '+rel)
    rows=read_json(out/'selection.json')
    for r in rows:
        if r['exclusion'] or r['source_group'] not in p['development_sources'] or r['source_group'] in p['reserved_sources']:
            raise ValueError('Source role firewall failed')
    return p,rows
