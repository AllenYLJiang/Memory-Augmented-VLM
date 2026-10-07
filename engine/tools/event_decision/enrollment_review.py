"""Full-window blind review and fail-closed prospective protocol sealing."""
import html
import json
import subprocess
from collections import Counter
from pathlib import Path

from .contracts import file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from .hard_trial import continuous_target
from .reenrollment import (CANARY_CODES, audit_selection, fresh_history, scope_report,
                           validate_plan, verify_integrity, verify_anchor_sources)

SEMANTIC_FIELDS = ('current_window_visual_label','visible_event_intervals_local','event_category',
                   'event_phase','suspicious_action_resemblance','normal_explains_suspicious_action',
                   'same_actor_support','same_time_support','unexplained_active_event_remains')
CHOICES = {'current_window_visual_label':['normal','anomalous','uncertain'],
           'event_category':['B1','B2','B4','B5','B6','G','none','uncertain'],
           'event_phase':['prelude','active','aftermath','normal','mixed','uncertain'],
           'confidence':['low','medium','high']}
for _field in SEMANTIC_FIELDS[4:]: CHOICES[_field]=['yes','no','uncertain','not_applicable']


def template(row, enrollment_hash, clip_hash):
    return {'schema':'v91_complete_window_review_v1','blind_id':row['blind_id'],
            'enrollment_sha256':enrollment_hash,'clip_sha256':clip_hash,
            'reviewer_id':'','independent_review_completed':False,
            'viewing':{'full_window_viewed':False,'context_used':False,'audio_used':False,'media_problem':''},
            **{k:'pending' for k in CHOICES}, 'visible_event_intervals_local':[],
            'direct_mechanism':'','normal_mechanism':'','notes':''}


def verify_media_sources(out):
    from audit_v91_history import portable_path
    errors=[]
    for r in iter_jsonl(out/'enrollment/media_signatures.jsonl'):
        p=portable_path(r['path'])
        if not p.is_file() or r['size'] is None or (p.stat().st_size,p.stat().st_mtime_ns)!=(r['size'],r['mtime_ns']):
            errors.append({'window_uid':r['window_uid'],'error':'SOURCE_MEDIA_MISSING_OR_CHANGED'})
    return errors


def extract_clip(source,start,out):
    expression=f'select=between(n\\,{start}\\,{start+95}),setpts=N/FRAME_RATE/TB'
    temp=out.with_name(out.stem+'.partial.mp4')
    proc=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),
                         '-map','0:v:0','-map_metadata','-1','-map_chapters','-1','-an',
                         '-vf',expression,'-frames:v','96','-fps_mode','passthrough',
                         '-c:v','libx264','-crf','18','-preset','fast',str(temp)],
                        capture_output=True,text=True,timeout=900)
    if proc.returncode: raise ValueError('clip extraction failed: '+proc.stderr[-500:])
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                          '-show_entries','stream=nb_read_frames','-of','json',str(temp)],
                         capture_output=True,text=True,timeout=120)
    streams=json.loads(probe.stdout).get('streams',[]) if probe.returncode==0 else []
    if len(streams)!=1 or int(streams[0].get('nb_read_frames',0))!=96:
        raise ValueError('review clip must contain exactly 96 decoded frames')
    temp.replace(out)


def build_packet(project,source,out):
    from audit_v91_history import portable_path
    verify_integrity(project,out)
    plan=read_json(out/'protocol/scientific_plan.json',{})
    errors=validate_plan(plan)
    rows=list(iter_jsonl(out/'enrollment/private_proposed_windows.jsonl'))
    scope=scope_report(rows)
    if not scope['adequate_filename_category_scope']: errors.append('CANARY_CATEGORY_SCOPE_INCOMPLETE')
    if errors: raise ValueError('review not started: '+json.dumps(errors))
    current_plan_hash=file_sha256(out/'protocol/scientific_plan.json')
    locked=read_json(out/'protocol/review_plan_lock.json')
    if locked and locked['plan_sha256']!=current_plan_hash: raise ValueError('plan changed after blind review began; new TAG required')
    candidates=list(iter_jsonl(out/'inputs/candidate_snapshot.jsonl'))
    history,_=fresh_history(project,source,out,candidates)
    if not audit_selection(rows,history)['structurally_ready']: raise ValueError('fresh enrollment/history check failed')
    if verify_anchor_sources(rows): raise ValueError('anchor provenance changed or missing')
    if verify_media_sources(out): raise ValueError('source media changed or missing; inspect media signatures')
    verify_integrity(project,out)
    enrollment_hash=file_sha256(out/'enrollment/private_proposed_windows.jsonl')
    if not locked:
        write_json(out/'protocol/review_plan_lock.json',{'plan_sha256':current_plan_hash,
                   'enrollment_sha256':enrollment_hash,'labels_seen_at_lock':False,
                   'remote_execution_authorized':False})
    public,private=out/'review_public',out/'review_private'
    (public/'clips').mkdir(parents=True,exist_ok=True)
    private.mkdir(parents=True,exist_ok=True)
    (out/'review_returns').mkdir(exist_ok=True)
    # Once a blind packet is issued, future trials must not silently reuse its
    # sources, even if reviewers have not yet returned their answers.
    write_jsonl(private/'review_exposure_manifest.jsonl',[
        {'source_group':r['source_group'],'video_id':r['video_id'],'window_uid':r['window_uid'],
         'exposure':'prospective_full_window_review_packet_issued'} for r in rows])
    entries=[]
    # Shuffling by opaque IDs hides role, stratum and allocation order.
    for i,r in enumerate(sorted(rows,key=lambda r:r['blind_id']),1):
        clip=public/'clips'/(r['blind_id']+'.mp4')
        receipt=private/'media_receipts'/(r['blind_id']+'.json')
        old=read_json(receipt)
        if not(old and clip.is_file() and old['clip_sha256']==file_sha256(clip) and old['enrollment_sha256']==enrollment_hash):
            extract_clip(portable_path(r['video_path']),r['start_frame'],clip)
            old={'blind_id':r['blind_id'],'clip_sha256':file_sha256(clip),'enrollment_sha256':enrollment_hash,'decoded_frames':96}
            write_json(receipt,old)
        entries.append({**old,'clip':'clips/'+clip.name})
        print(f'[review-media] {i}/{len(rows)} {r["blind_id"]}',flush=True)
    write_json(public/'manifest.json',{'enrollment_sha256':enrollment_hash,'cases':entries})
    forms=[template(r,enrollment_hash,r['clip_sha256']) for r in entries]
    if not(public/'reviewer_template.jsonl').exists(): write_jsonl(public/'reviewer_template.jsonl',forms)
    write_json(public/'schema.json',{'choices':CHOICES,'intervals':'inclusive local indices 0..95',
               'rule':'anomaly target requires at least 8 contiguous frames; uncertain is not normal'})
    (public/'README.md').write_text(
        '# Independent full-window review\n\n'
        'Only share this review_public folder with each reviewer. Do not inspect private files, original titles, '
        'dataset labels, selection strata, source roles, graph results or another reviewer\'s answers.\n\n'
        'Watch every complete 96-frame silent clip. Each reviewer independently returns a separate JSONL based '
        'on reviewer_template.jsonl, preserving blind_id and both hashes. Never fill a viewed flag without watching. '
        'The form uses JSON booleans, not strings. Return files to the coordinator outside this public folder.\n\n'
        '- current_window_visual_label: normal, anomalous or uncertain, based only on this clip.\n'
        '- visible_event_intervals_local: inclusive anomaly intervals from 0 to 95; [] if none. Do not guess hidden frames.\n'
        '- event_category: B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, G explosion; '
        'none for normal, uncertain if not identifiable.\n'
        '- direct_mechanism: concrete visible action explaining an anomalous judgment.\n'
        '- normal_mechanism: concrete visible non-anomalous explanation, required for a normal judgment.\n'
        '- suspicious_action_resemblance: whether a normal-looking explanation could be confused with an anomalous action.\n'
        '- normal_explains_suspicious_action: does that normal mechanism explain the suspicious action itself?\n'
        '- same_actor_support / same_time_support: does the explanation concern the same actors and time?\n'
        '- unexplained_active_event_remains: is any active anomalous action still unexplained?\n'
        '- event_phase and confidence: use schema.json choices; do not force normal when visibility is inadequate.\n'
        '- independent_review_completed: true only after working independently; reviewer_id must identify you.\n\n'
        'uncertain answers are valid and excluded from binary metrics, with coverage reported. Motion alone is not '
        'evidence of a hard normal. Only a full-window normal with concrete, same-actor/same-time explanation and '
        'suspicious-action resemblance meets the semantic hard-normal definition.\n',encoding='utf-8')
    body=''.join(f'<section><h2>{html.escape(r["blind_id"])}</h2><video controls preload="none" src="{r["clip"]}"></video></section>' for r in entries)
    (public/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>V9.1 Blind Window Review</title><style>body{font:16px Arial;margin:24px;color:#222}main{max-width:960px;margin:auto}'
        'section{border-top:1px solid #ccc;padding:16px 0}h2{font-size:18px}video{width:100%;max-height:540px;background:#111}</style>'
        '<main><h1>Blind Window Review</h1>'+body+'</main></html>',encoding='utf-8')
    write_json(private/'packet_integrity.json',{'manifest_sha256':file_sha256(public/'manifest.json'),
               'enrollment_sha256':enrollment_hash,'plan_sha256':current_plan_hash})
    return {'status':'WAITING_FOR_TWO_INDEPENDENT_HUMAN_REVIEWS','windows':len(rows),'remote_calls':0,
            'public_packet':str(public/'index.html'),'returns':str(out/'review_returns')}


def validate_answer(r,expected):
    errors=[]
    if r.get('schema')!='v91_complete_window_review_v1': errors.append('schema')
    for k in ('blind_id','enrollment_sha256','clip_sha256'):
        if r.get(k)!=expected[k]: errors.append('identity:'+k)
    if not isinstance(r.get('reviewer_id'),str) or not r['reviewer_id'].strip(): errors.append('reviewer_id')
    if r.get('independent_review_completed') is not True: errors.append('independence_attestation')
    v=r.get('viewing',{})
    if not isinstance(v,dict) or v.get('full_window_viewed') is not True or v.get('context_used') is not False or v.get('audio_used') is not False or v.get('media_problem')!='':
        errors.append('complete_visual_only_window_required')
    for k,options in CHOICES.items():
        if r.get(k) not in options: errors.append('field:'+k)
    intervals=r.get('visible_event_intervals_local')
    try:
        if not isinstance(intervals,list): raise ValueError('intervals must be list')
        target=continuous_target(intervals)
    except ValueError:
        errors.append('invalid_local_intervals'); target=None
    label=r.get('current_window_visual_label')
    if label=='normal' and (intervals or r.get('event_category')!='none'): errors.append('normal_with_event')
    if label=='anomalous' and (not intervals or r.get('event_category') not in ['B1','B2','B4','B5','B6','G']):
        errors.append('anomaly_requires_event_and_intervals')
    for field in ('direct_mechanism','normal_mechanism','notes'):
        if not isinstance(r.get(field),str): errors.append('text:'+field)
    if label=='anomalous' and not str(r.get('direct_mechanism','')).strip(): errors.append('direct_mechanism_required')
    if label=='normal' and not str(r.get('normal_mechanism','')).strip(): errors.append('normal_mechanism_required')
    return errors, None if label=='uncertain' else target


def semantic_answer(r):
    # Merge equivalent interval unions, without changing the public inclusive convention.
    intervals=sorted({i for a,b in r['visible_event_intervals_local'] for i in range(a,b+1)})
    return {**{k:r[k] for k in SEMANTIC_FIELDS if k!='visible_event_intervals_local'},'visible_frames':intervals}


def is_hard_normal(r):
    return (r['current_window_visual_label']=='normal' and r['suspicious_action_resemblance']=='yes'
            and bool(r['normal_mechanism'].strip()) and r['normal_explains_suspicious_action']=='yes'
            and r['same_actor_support']=='yes' and r['same_time_support']=='yes'
            and r['unexplained_active_event_remains']=='no')


def import_reviews(out,reviewer_files,adjudication_file=None):
    public,private=out/'review_public',out/'review_private'
    integrity=read_json(private/'packet_integrity.json')
    if not integrity or file_sha256(public/'manifest.json')!=integrity['manifest_sha256']:
        raise ValueError('complete, unchanged review packet required')
    if file_sha256(out/'protocol/scientific_plan.json')!=integrity['plan_sha256']:
        raise ValueError('scientific plan changed after packet creation')
    if integrity.get('enrollment_sha256') and file_sha256(out/'enrollment/private_proposed_windows.jsonl')!=integrity['enrollment_sha256']:
        raise ValueError('review packet belongs to a different enrollment')
    cases={r['blind_id']:r for r in read_json(public/'manifest.json')['cases']}
    for r in cases.values():
        if file_sha256(public/r['clip'])!=r['clip_sha256']: raise ValueError('review media changed')
    if len(reviewer_files)!=2: raise ValueError('exactly two independent reviewer files are required')
    sources=[{'path':str(p),'sha256':file_sha256(p)} for p in reviewer_files]
    if sources[0]['sha256']==sources[1]['sha256']: raise ValueError('identical reviewer files are not independent returns')
    returns,ids,errors={},[],[]
    for path in reviewer_files:
        records=list(iter_jsonl(path)); current={}; raw_ids=[r.get('reviewer_id') for r in records]
        reviewer_ids={x for x in raw_ids if isinstance(x,str)}
        if len(reviewer_ids)!=1 or not reviewer_ids or not next(iter(reviewer_ids)) or any(not isinstance(x,str) for x in raw_ids):
            errors.append({'file':str(path),'error':'single_named_reviewer_required'})
        ids.extend(reviewer_ids)
        for r in records:
            bid=r.get('blind_id')
            if bid not in cases or bid in current:
                errors.append({'blind_id':bid,'error':'unknown_or_duplicate_case'}); continue
            issues,_=validate_answer(r,cases[bid])
            errors.extend({'blind_id':bid,'error':x} for x in issues)
            current[bid]=r
        if set(current)!=set(cases): errors.append({'file':str(path),'error':'all_enrolled_windows_required'})
        returns[str(path)]=current
    if len(ids)!=2 or len(set(ids))!=2: errors.append({'error':'reviewer_ids_must_differ'})
    write_json(private/'review_import_errors.json',errors)
    if errors:
        return {'ready':False,'status':'WAITING_FOR_VALID_HUMAN_REVIEWS','errors':len(errors),'remote_execution_authorized':False}
    left,right=returns.values()
    disagreements={bid for bid in cases if semantic_answer(left[bid])!=semantic_answer(right[bid])}
    adjudications={}
    if adjudication_file:
        sources.append({'path':str(adjudication_file),'sha256':file_sha256(adjudication_file)})
        for r in iter_jsonl(adjudication_file):
            bid=r.get('blind_id')
            if bid not in disagreements or bid in adjudications: raise ValueError('unrequested/duplicate adjudication')
            issues,_=validate_answer(r,cases[bid])
            if issues or r['reviewer_id'] in ids or not r.get('adjudication_reason','').strip():
                raise ValueError('adjudication requires a third reviewer, complete answer and rationale')
            adjudications[bid]=r
    pending=sorted(disagreements-set(adjudications))
    write_jsonl(private/'adjudication_template.jsonl',[
        {**template(cases[bid],cases[bid]['enrollment_sha256'],cases[bid]['clip_sha256']),
         'adjudication_reason':''} for bid in pending])
    if pending:
        return {'ready':False,'status':'WAITING_FOR_ADJUDICATION','unresolved_disagreements':len(pending),
                'remote_execution_authorized':False}
    rows=list(iter_jsonl(out/'enrollment/private_proposed_windows.jsonl'))
    ledger=[]; adaptation_conflicts=0; hard=Counter(); canary={role:{c:set() for c in CANARY_CODES} for role in ('adaptation','locked_evaluation')}
    for row in rows:
        bid=row['blind_id']; answer=adjudications.get(bid,left[bid]); _,target=validate_answer(answer,cases[bid])
        role=row['role']; is_hard=is_hard_normal(answer)
        if row['stratum']=='hard_label_A' and is_hard: hard[role]+=1
        if row['stratum']=='other_class_canary' and target==1 and answer['event_category'] in CANARY_CODES:
            canary[role][answer['event_category']].add(row['source_group'])
        conflict=role=='adaptation' and target is not None and row['weak_training_target'] is not None and target!=row['weak_training_target']
        adaptation_conflicts+=int(conflict)
        ledger.append({'blind_id':bid,'window_uid':row['window_uid'],'source_group':row['source_group'],'role':role,
                       'dataset_weak_target':row['weak_training_target'],'training_loss_mask':row['training_loss_mask'],
                       'human_window_target':target,'gold_metric_mask':role=='locked_evaluation' and target is not None,
                       'label_evidence_level':'two_reviewer_full_window_with_adjudication_as_needed',
                       'human_label_use':'audit_only' if role=='adaptation' else 'evaluation_only',
                       'semantic_hard_normal':is_hard,'adaptation_label_conflict':conflict,
                       'reviewer_answers':[left[bid],right[bid]],'adjudication':adjudications.get(bid)})
    # Keep all answers outside the feature/training manifest, including uncertain and conflicting cases.
    write_jsonl(private/'human_label_ledger.jsonl',ledger)
    write_json(private/'review_sources.json',sources)
    plan=read_json(out/'protocol/scientific_plan.json')
    blockers=[]
    for role in canary:
        if hard[role]<plan['min_human_hard_normal_per_role'][role]: blockers.append('HUMAN_HARD_NORMAL_COVERAGE:'+role)
        for code,groups in canary[role].items():
            if not groups: blockers.append('HUMAN_CANARY_CATEGORY_NOT_CONFIRMED:'+role+':'+code)
    locked=[r for r in ledger if r['role']=='locked_evaluation']
    evaluable=sum(r['gold_metric_mask'] for r in locked)
    if evaluable/len(locked)<plan['minimum_binary_evaluation_coverage']: blockers.append('BINARY_LABEL_COVERAGE_INSUFFICIENT')
    if adaptation_conflicts: blockers.append('ADAPTATION_LABEL_CONFLICT_REQUIRES_NEW_DATA_AUDIT')
    return {'ready':not blockers,'status':'REVIEW_COMPLETE' if not blockers else 'WAITING_FOR_SEMANTIC_SCOPE_REVIEW',
            'completed_unique':len(ledger),'reviewer_ids':ids,'adjudicated_disagreements':len(adjudications),
            'blockers':blockers,'binary_evaluable_locked_windows':evaluable,
            'human_label_ledger_sha256':file_sha256(private/'human_label_ledger.jsonl'),
            'independence':'attested; file and ID checks cannot prove psychological independence',
            'remote_execution_authorized':False}


def freeze(project,source,out,reviewer_files,adjudication_file=None):
    verify_integrity(project,out)
    if (out/'protocol/freeze_receipt.json').exists(): raise ValueError('protocol already frozen; no overwrite')
    plan=read_json(out/'protocol/scientific_plan.json',{})
    if validate_plan(plan): raise ValueError('complete the predeclared scientific plan first')
    summary=import_reviews(out,reviewer_files,adjudication_file)
    write_json(out/'review_private/review_summary.json',summary)
    if not summary['ready']: return summary
    rows=list(iter_jsonl(out/'enrollment/private_proposed_windows.jsonl'))
    history,_=fresh_history(project,source,out,list(iter_jsonl(out/'inputs/candidate_snapshot.jsonl')))
    checks=audit_selection(rows,history)
    if not checks['structurally_ready'] or not scope_report(rows)['adequate_filename_category_scope']:
        raise ValueError('fresh history/structural/category check failed')
    if verify_media_sources(out): raise ValueError('source media missing or changed')
    if verify_anchor_sources(rows): raise ValueError('anchor provenance changed or missing')
    verify_integrity(project,out)
    if file_sha256(out/'protocol/scientific_plan.json')!=read_json(out/'protocol/review_plan_lock.json')['plan_sha256']:
        raise ValueError('scientific plan changed during freeze')
    for r in read_json(out/'review_private/review_sources.json'):
        if file_sha256(Path(r['path']))!=r['sha256']: raise ValueError('review input changed during freeze')
    bound=['enrollment/private_proposed_windows.jsonl','protocol/scientific_plan.json',
           'protocol/review_plan_lock.json','review_private/human_label_ledger.jsonl',
           'review_private/review_summary.json','review_private/review_sources.json','review_public/manifest.json']
    receipt={'version':'v91_reviewed_protocol_freeze_v1','frozen':True,
             'files':{p:file_sha256(out/p) for p in bound},
             'ready_for_acquisition_plan':True,'remote_execution_authorized':False,
             'deployment_authorized':False,'accuracy_measured':False,'AP':None,
             'next':'Explicit acquisition plan, feature/model freeze, predictions before label unblinding; no API implemented here.'}
    receipt['freeze_id']=semantic_sha256(receipt)
    write_json(out/'protocol/freeze_receipt.json',receipt)
    return receipt
