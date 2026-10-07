"""Masked trace export and a human-supported, development-only shadow gate."""
import html
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .contracts import file_sha256, read_json, write_jsonl, semantic_sha256
from .role_scoped import atomic_json
from .mechanism_v94_contract import FEATURES, VERSION, assess, feature_row
from .mechanism_v94_report import current, page


def review_packet(out,protocol,rows):
    out=Path(out);digest=file_sha256(out/'protocol.json');folder=out/'review';folder.mkdir(exist_ok=True)
    template=[];links=[]
    for row in rows:
        uid=row['window_uid'];path=out/'results'/(uid+'.json')
        template.append({'window_uid':uid,'protocol_sha256':digest,'result_sha256':file_sha256(path) if path.exists() else None,
            'reviewer_id':'','completed':False,'evidence_scope':'eight_frames',
            'b5_label':'unknown','category_mapping_correct':'unknown','local_claims_supported':'unknown',
            'cross_event_claims_supported':'unknown',
            'boundary_resolved':False,'notes':''})
        links.append('<li><a href="../cases/'+uid[:20]+'/index.html">'+html.escape(row['video_id'])+' ['+str(row['start_frame'])+','+str(row['end_frame_exclusive'])+')</a></li>')
    write_jsonl(folder/'template.jsonl',template)
    body='<h1>Development evidence review</h1><p>This is NOT blind evaluation. Inspect the eight frames and new claims in each case. Prior design feedback is visible and is not ground truth.</p>'
    body+='<p>Complete a separate returns.jsonl, not template.jsonl. B5=no does not mean normal. Use unknown where needed. Mark category_mapping_correct, local_claims_supported and cross_event_claims_supported yes/no/unknown. For cross-event review, yes means no unsupported asserted connection, including when links are absent or explicitly unknown. Mark boundary_resolved only when the B1/B5, B6/B5 or constraint boundary is actually clear. Do not infer identities from plot.</p><ul>'+''.join(links)+'</ul>'
    (folder/'index.html').write_text(page('Development review',body),encoding='utf-8')


def stage3(out,protocol,rows,review_file=None):
    out=Path(out);digest=file_sha256(out/'protocol.json');review_packet(out,protocol,rows)
    trace=[];array=[];mask=[];result_hashes={};by_uid={};events=[];links=[];frame_valid=[]
    for row in rows:
        uid=row['window_uid'];r=current(out,row,digest);path=out/'results'/(uid+'.json')
        result_hashes[uid]=file_sha256(path) if path.exists() else None
        raw=r.get('parsed');a=r.get('assessment') or assess(None);f=feature_row(raw,a)
        array.append([f[n]['value'] if f[n]['observed'] else np.nan for n in FEATURES]);mask.append([f[n]['observed'] for n in FEATURES])
        item={'window_uid':uid,'video_id':row['video_id'],'source_group':row['source_group'],
              'start_frame':row['start_frame'],'end_frame_exclusive':row['end_frame_exclusive'],
              'sampled_frame_indices':row['sampled_frame_indices'],'image_sha256':{n:h for n,h in row['media_hashes'].items() if n.endswith('.jpg')},
              'status':r['status'],'result_sha256':result_hashes[uid],'protocol_sha256':digest,
              'features':f,'field_validity':a['fields'],'typed_events':(raw or {}).get('events',[]) if isinstance(raw,dict) else [],
              'event_links':(raw or {}).get('links',[]) if isinstance(raw,dict) else [],
              'issues':a['issues'],'raw_b5':(raw or {}).get('b5') if isinstance(raw,dict) else None,
              'window_target':None,'training_loss_mask':False,'evaluation_loss_mask':False,
              'scope':'design_exposed_development_only','labels_are_not_derived_from_model':True}
        trace.append(item);by_uid[uid]=(row,r,a)
        events+=a.get('events',[]);links+=a.get('links',[])
        if a.get('core_valid'):
            frame_valid += [all(a['fields'].get(f'/frames/{i}/{k}',{}).get('valid') for k in ('action','constraint','injury','quality','evidence')) for i in range(8)]
    feature_root=out/'features';feature_root.mkdir(exist_ok=True)
    write_jsonl(feature_root/'trace.jsonl',trace)
    np.save(feature_root/'values.npy',np.asarray(array,dtype=np.float32));np.save(feature_root/'observed.npy',np.asarray(mask,dtype=bool))
    atomic_json(feature_root/'schema.json',{'version':VERSION,'features':FEATURES,'shape':[len(rows),len(FEATURES)],
        'unknown':'NaN + observed=false in numpy; null + observed=false in JSON',
        'masks_mean':'parser/observation availability, NOT independently verified semantic truth',
        'lineage':'each trace row includes source, evidence and result hashes','training_authorized':False})
    checks=protocol['gates'];errors=[];reviews={};review_sha=None
    if review_file:
        review_file=Path(review_file)
        if review_file.resolve()==(out/'review/template.jsonl').resolve(): raise ValueError('Use separate review returns, not the regenerated template')
        review_sha=file_sha256(review_file)
        for lineno,line in enumerate(review_file.read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip(): continue
            try:
                v=json.loads(line);uid=v['window_uid']
                if uid not in by_uid or uid in reviews: raise ValueError('unknown/duplicate case')
                if v.get('protocol_sha256')!=digest or not result_hashes[uid] or v.get('result_sha256')!=result_hashes[uid]: raise ValueError('stale protocol/result binding')
                if v.get('completed') is not True or not isinstance(v.get('reviewer_id'),str) or not v['reviewer_id'].strip(): raise ValueError('incomplete reviewer attestation')
                if v.get('evidence_scope')!='eight_frames': raise ValueError('scope must match model evidence, no extra context as gold')
                if any(v.get(k) not in {'yes','no','unknown'} for k in ('b5_label','category_mapping_correct','local_claims_supported','cross_event_claims_supported')): raise ValueError('invalid review enum')
                if type(v.get('boundary_resolved')) is not bool or not isinstance(v.get('notes'),str) or not v['notes'].strip(): raise ValueError('missing boundary flag/notes')
                if v['boundary_resolved'] and v['b5_label']=='unknown': raise ValueError('unknown label cannot attest resolved boundary')
                reviews[uid]=v
            except (ValueError,TypeError,KeyError) as exc: errors.append({'line':lineno,'error':str(exc)})
    resolved={uid:v for uid,v in reviews.items() if v['boundary_resolved'] and v['b5_label']!='unknown'}
    agreement=sum((by_uid[uid][1].get('parsed') or {}).get('b5',{}).get('label')==v['b5_label'] for uid,v in resolved.items())
    n=len(rows);core=sum(a['core_valid'] for _,_,a in by_uid.values())
    core_fraction=core/n if n else 0
    event_fraction=sum(e['valid'] for e in events)/len(events) if events else 0
    link_fraction=sum(e['valid'] for e in links)/len(links) if links else 1
    frame_fraction=sum(frame_valid)/(8*n) if n else 0
    consistency_fraction=sum(a['fields'].get('/b5/consistency',{}).get('valid',False) for _,_,a in by_uid.values())/n if n else 0
    technical=(core_fraction>=checks['minimum_core_fraction'] and event_fraction>=checks['minimum_event_valid_fraction'] and
        frame_fraction>=checks['minimum_frame_valid_fraction'] and link_fraction>=checks['minimum_link_valid_fraction'] and
        consistency_fraction>=checks['minimum_consistency_fraction'] and
        all(r['status'] in {'success','partial','provider_rejected'} for _,r,_ in by_uid.values()))
    label_counts=Counter(v['b5_label'] for v in resolved.values())
    sources={label:len({by_uid[uid][0]['source_group'] for uid,v in resolved.items() if v['b5_label']==label}) for label in ('yes','no')}
    claims=sum(v['local_claims_supported']=='yes' for v in reviews.values())/n if n else 0
    cross_claims=sum(v['cross_event_claims_supported']=='yes' for v in reviews.values())/n if n else 0
    categories=bool(reviews) and all(v['category_mapping_correct']=='yes' for v in reviews.values())
    semantic=not errors and len(reviews)==n and len(resolved)/n>=checks['minimum_resolved_review_fraction'] and agreement/max(1,len(resolved))>=checks['minimum_boundary_agreement'] and claims>=checks['minimum_supported_claims_fraction'] and cross_claims>=checks['minimum_supported_claims_fraction'] and categories and label_counts['yes']>=checks['minimum_b5_positive_reviews'] and label_counts['no']>=checks['minimum_b5_negative_reviews'] and min(sources.values())>=checks['minimum_sources_each_label']
    ready=technical and semantic and not protocol['mock']
    gate={'version':VERSION,'protocol_sha256':digest,'result_hashes':result_hashes,'review_input_sha256':review_sha,'mock':protocol['mock'],
        'technical_ready':bool(technical),'core_valid_fraction':core_fraction,'event_valid_fraction':event_fraction,
        'link_valid_fraction':link_fraction,'frame_valid_fraction':frame_fraction,
        'judgment_consistency_fraction':consistency_fraction,
        'review_completed':len(reviews),'selected':n,'review_errors':errors,'resolved_reviews':len(resolved),
        'resolved_b5_label_counts':dict(label_counts),'source_groups_each_label':sources,
        'development_boundary_agreement_not_benchmark_accuracy':agreement/max(1,len(resolved)) if resolved else None,
        'local_claims_supported_fraction':claims,'cross_event_claims_supported_fraction':cross_claims,
        'category_codes_human_confirmed':categories,'semantic_ready':bool(semantic),
        'ready_for_shadow_integration':bool(ready),'ready_for_training':False,'locked_evaluation_authorized':False,
        'decision':'SHADOW_FEATURES_ONLY_NO_TRAINING' if ready else 'TRACE_EXPORTED_INTEGRATION_HELD',
        'thresholds':checks,'formal_accuracy':None,'formal_AP':None}
    gate_dir=out/'integration';gate_dir.mkdir(exist_ok=True);atomic_json(gate_dir/'gate.json',gate)
    atomic_json(gate_dir/'learning_requirements.json',{'training_authorized':False,'evaluation_authorized':False,
        'requirements':['explicit human/annotation label scope, not model pseudo-gold','unknown labels masked',
                        'adequate real normal/negative sources, not merely non-B5 anomalies','source-disjoint train/validation/evaluation with exposure audit',
                        'freeze learner and score policy before independent evaluation'],
        'current_shortfall':'all current windows are development/design exposed; no independent evaluation enrollment or valid training labels supplied'})
    # Downstream readers use CURRENT, never discover stale bundles by globbing.
    bundle_hashes={str(p.relative_to(out)):file_sha256(p) for p in feature_root.iterdir() if p.is_file()}
    atomic_json(gate_dir/'CURRENT.json',{'ready':bool(ready),'protocol_sha256':digest,'gate_sha256':file_sha256(gate_dir/'gate.json'),
        'bundle_hashes':bundle_hashes if ready else {},'contract_id':semantic_sha256(bundle_hashes) if ready else None})
    if review_file:
        atomic_json(gate_dir/'review_import.json',{'input_sha256':review_sha,'accepted':list(reviews.values()),'errors':errors,'use_policy':'development_gate_only_not_training'})
    return gate


def load_shadow_bundle(out):
    """Explicit opt-in adapter for later graph/OT or learner development."""
    from .mechanism_v94_store import load
    out=Path(out);load(out)
    pointer=read_json(out/'integration/CURRENT.json',{});gate=read_json(out/'integration/gate.json',{})
    if not pointer.get('ready') or gate.get('ready_for_shadow_integration') is not True or gate.get('mock'):
        raise ValueError('Shadow integration held; inspect gate.json. Trace files are diagnostic only.')
    if pointer['gate_sha256']!=file_sha256(out/'integration/gate.json') or pointer['protocol_sha256']!=file_sha256(out/'protocol.json'):
        raise ValueError('Shadow gate or protocol changed')
    for uid,digest in gate['result_hashes'].items():
        path=out/'results'/(uid+'.json')
        if not path.is_file() or file_sha256(path)!=digest: raise ValueError('Results changed after gate')
    for rel,digest in pointer['bundle_hashes'].items():
        path=(out/rel).resolve()
        if out.resolve() not in path.parents or file_sha256(path)!=digest: raise ValueError('Feature bundle changed')
    return np.load(out/'features/values.npy',allow_pickle=False),np.load(out/'features/observed.npy',allow_pickle=False)
