"""Technical-first review scheduling and masked, opt-in shadow features."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import file_sha256, read_json, write_jsonl, semantic_sha256
from .role_scoped import atomic_json
from .mechanism_v94_stage3 import review_packet
from .mechanism_v94_report import page
from .mechanism_v95_contract import VERSION, FEATURES, assess, feature_row
from .mechanism_v95_report import current, technical_metrics


def import_reviews(path, out, rows, results, result_hashes):
    accepted={}; errors=[]; digest=file_sha256(out/'protocol.json')
    by_uid={r['window_uid']:r for r in rows}
    if path.resolve()==(out/'review/template.jsonl').resolve(): raise ValueError('Use a separate returns.jsonl, not the regenerated template')
    for n,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        try:
            v=json.loads(line); uid=v['window_uid']
            if uid not in by_uid or uid in accepted: raise ValueError('unknown or duplicate case')
            if v.get('protocol_sha256')!=digest or not result_hashes[uid] or v.get('result_sha256')!=result_hashes[uid]: raise ValueError('stale protocol or result hash')
            if v.get('completed') is not True or not isinstance(v.get('reviewer_id'),str) or not v['reviewer_id'].strip(): raise ValueError('incomplete review')
            if v.get('evidence_scope')!='eight_frames': raise ValueError('Review scope must match the eight images')
            for key in ('b5_label','category_mapping_correct','local_claims_supported','cross_event_claims_supported'):
                if v.get(key) not in {'yes','no','unknown'}: raise ValueError('invalid review value: '+key)
            if type(v.get('boundary_resolved')) is not bool or not isinstance(v.get('notes'),str) or not v['notes'].strip(): raise ValueError('boundary flag and evidence notes required')
            if v['boundary_resolved'] and v['b5_label']=='unknown': raise ValueError('unknown boundary cannot be resolved')
            accepted[uid]=v
        except (ValueError,TypeError,KeyError) as exc: errors.append({'line':n,'error':str(exc)})
    resolved={k:v for k,v in accepted.items() if v['boundary_resolved'] and v['b5_label']!='unknown'}
    agreement=0
    for uid,v in resolved.items():
        b5=(results[uid].get('parsed') or {}).get('b5'); b5=b5 if isinstance(b5,dict) else {}
        agreement+=b5.get('label')==v['b5_label']
    sources={label:len({by_uid[uid]['source_group'] for uid,v in resolved.items() if v['b5_label']==label}) for label in ('yes','no')}
    return accepted,errors,resolved,agreement,sources


def stage3(out, protocol, rows, review_file=None):
    out=Path(out); gate_dir=out/'integration'; gate_dir.mkdir(exist_ok=True)
    # Revoke an earlier pointer before doing any new export or import that may fail.
    atomic_json(gate_dir/'CURRENT.json',{'ready':False,'reason':'rechecking','bundle_hashes':{}})
    results={r['window_uid']:current(out,r,protocol) for r in rows}
    hashes={uid:file_sha256(out/'results'/(uid+'.json')) if (out/'results'/(uid+'.json')).exists() else None for uid in results}
    technical=technical_metrics(list(results.values()),protocol); digest=file_sha256(out/'protocol.json')
    traces=[]; values=[]; masks=[]
    for row in rows:
        uid=row['window_uid']; r=results[uid]; raw=r.get('parsed'); a=r.get('assessment') or assess(None)
        f=feature_row(raw,a)
        values.append([f[n]['value'] if f[n]['observed'] else np.nan for n in FEATURES]); masks.append([f[n]['observed'] for n in FEATURES])
        traces.append({'window_uid':uid,'video_id':row['video_id'],'source_group':row['source_group'],
            'start_frame':row['start_frame'],'end_frame_exclusive':row['end_frame_exclusive'],
            'sampled_frame_indices':row['sampled_frame_indices'],'media_hashes':row['media_hashes'],
            'status':r['status'],'result_sha256':hashes[uid],'protocol_sha256':digest,
            'features':f,'field_validity':a['fields'],'event_validity':a['events'],'link_validity':a['links'],
            'raw_observations':raw,'issues':a['issues'],'window_target':None,
            'training_loss_mask':False,'evaluation_loss_mask':False,'scope':'design_exposed_development_only'})
    features=out/'features'; features.mkdir(exist_ok=True)
    write_jsonl(features/'trace.jsonl',traces)
    np.save(features/'values.npy',np.asarray(values,dtype=np.float32)); np.save(features/'observed.npy',np.asarray(masks,dtype=bool))
    atomic_json(features/'schema.json',{'version':VERSION,'features':FEATURES,'shape':[len(rows),len(FEATURES)],
        'unknown':'null/NaN with observed=false; never normal','fraction_scope':'eight sampled frames, not dense occupancy',
        'coverage_scope':'count of known, valid sample observations divided by 8',
        'mask_meaning':'availability/structural validity, not independent semantic truth','training_authorized':False})
    diagnostics={name:{'observed':int(sum(row[i] for row in masks)),
                      'distinct_observed_values':sorted(set(float(row[i]) for row,mask in zip(values,masks) if mask[i]))}
                 for i,name in enumerate(FEATURES)}
    atomic_json(features/'availability_summary.json',diagnostics)
    accepted={}; errors=[]; resolved={}; agreement=0; sources={'yes':0,'no':0}; review_sha=None
    review_root=out/'review'; review_root.mkdir(exist_ok=True)
    review_enabled=technical['technical_ready']
    if review_enabled:
        review_packet(out,protocol,rows)
        if protocol['mock']:
            (review_root/'index.html').write_text(page('MOCK review packet',
                '<h1>MOCK packet for software verification only</h1><p>No human review is requested for these synthetic observations.</p>'),encoding='utf-8')
        if review_file:
            review_file=Path(review_file); review_sha=file_sha256(review_file)
            accepted,errors,resolved,agreement,sources=import_reviews(review_file,out,rows,results,hashes)
    else:
        (review_root/'index.html').write_text(page('Review held',
            '<h1>Technical checks have not passed</h1><p>No new human review is requested. Saved observations remain diagnostic only.</p>'
            '<p><a href="../index.html">Current results</a></p>'),encoding='utf-8')
        if review_file: errors.append({'error':'Review not imported: technical checks failed; do not spend reviewer time yet'})
    atomic_json(review_root/'packet_status.json',{'active':review_enabled and not protocol['mock'],'synthetic_test_packet':protocol['mock'],'protocol_sha256':digest,
        'result_hashes':hashes,'reason':'ready_for_development_review' if review_enabled else 'technical_checks_failed',
        'existing_returns_preserved':True})
    n=len(rows); thresholds=protocol['gates']; counts=Counter(v['b5_label'] for v in resolved.values())
    local=sum(v['local_claims_supported']=='yes' for v in accepted.values())/max(1,n)
    cross=sum(v['cross_event_claims_supported']=='yes' for v in accepted.values())/max(1,n)
    categories=bool(accepted) and all(v['category_mapping_correct']=='yes' for v in accepted.values())
    semantic=(review_enabled and not errors and len(accepted)==n and len(resolved)/max(1,n)>=thresholds['minimum_resolved_review_fraction'] and
        agreement/max(1,len(resolved))>=thresholds['minimum_boundary_agreement'] and local>=thresholds['minimum_supported_claims_fraction'] and
        cross>=thresholds['minimum_supported_claims_fraction'] and categories and counts['yes']>=thresholds['minimum_b5_positive_reviews'] and
        counts['no']>=thresholds['minimum_b5_negative_reviews'] and min(sources.values())>=thresholds['minimum_sources_each_label'])
    ready=bool(semantic and not protocol['mock'])
    decision='SHADOW_FEATURES_ONLY_NO_TRAINING' if ready else ('REVIEW_REQUIRED' if review_enabled else 'TECHNICAL_HOLD_NO_REVIEW_REQUESTED')
    if protocol['mock']: decision='MOCK_DIAGNOSTIC_ONLY'
    gate={'version':VERSION,'protocol_sha256':digest,'result_hashes':hashes,'review_input_sha256':review_sha,'mock':protocol['mock'],
        **technical,'human_review_requested':review_enabled and not protocol['mock'],'review_completed':len(accepted),'review_errors':errors,
        'resolved_reviews':len(resolved),'resolved_b5_label_counts':dict(counts),'source_groups_each_label':sources,
        'development_boundary_agreement_not_benchmark_accuracy':agreement/len(resolved) if resolved else None,
        'local_claims_supported_fraction':local,'cross_event_claims_supported_fraction':cross,'category_codes_human_confirmed':categories,
        'semantic_ready':bool(semantic),'ready_for_shadow_integration':ready,'ready_for_training':False,
        'locked_evaluation_authorized':False,'formal_accuracy':None,'formal_AP':None,'decision':decision,'thresholds':thresholds}
    atomic_json(gate_dir/'gate.json',gate)
    atomic_json(gate_dir/'learning_requirements.json',{'training_authorized':False,'evaluation_authorized':False,
        'requirements':['scope-correct supervision, not model judgments','real normal/hard-normal negatives, not only non-B5 anomalies',
                        'unknown labels masked','source-separated evaluation with exposure audit','freeze learner and scoring policy before evaluation'],
        'current_scope':'same 36 development windows from the 93 design-exposed cohort; never final generalization proof'})
    if review_file and review_enabled: atomic_json(gate_dir/'review_import.json',{'input_sha256':review_sha,'accepted':list(accepted.values()),'errors':errors,'use_policy':'development_gate_only'})
    files={str(p.relative_to(out)):file_sha256(p) for p in features.iterdir() if p.is_file()}
    atomic_json(gate_dir/'CURRENT.json',{'ready':ready,'protocol_sha256':digest,'gate_sha256':file_sha256(gate_dir/'gate.json'),
        'bundle_hashes':files if ready else {},'contract_id':semantic_sha256(files) if ready else None})
    return gate


def load_shadow_bundle(out):
    from .mechanism_v95_store import load
    out=Path(out); load(out)
    pointer=read_json(out/'integration/CURRENT.json',{}); gate=read_json(out/'integration/gate.json',{})
    if not pointer.get('ready') or gate.get('ready_for_shadow_integration') is not True or gate.get('mock'):
        raise ValueError('Shadow integration held; no training or score override is authorized')
    if pointer.get('gate_sha256')!=file_sha256(out/'integration/gate.json') or pointer.get('protocol_sha256')!=file_sha256(out/'protocol.json'): raise ValueError('Gate/protocol changed')
    for uid,sha in gate['result_hashes'].items():
        if file_sha256(out/'results'/(uid+'.json'))!=sha: raise ValueError('Response changed after review')
    for rel,sha in pointer['bundle_hashes'].items():
        path=(out/rel).resolve()
        if out.resolve() not in path.parents or file_sha256(path)!=sha: raise ValueError('Feature bundle changed')
    return np.load(out/'features/values.npy',allow_pickle=False),np.load(out/'features/observed.npy',allow_pickle=False)
