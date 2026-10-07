"""Outcome-blind re-enrollment, history refresh and scientific scope gates."""
from collections import Counter
from pathlib import Path
import os
import uuid

from .contracts import WindowKey, file_sha256, iter_jsonl, read_json, write_json, write_jsonl
from .hard_trial import HISTORY_NAMES, PRUNE, QUOTAS, anomaly_codes, enroll, group_id, weak_label

CANARY_CODES = ['B2', 'B5', 'B6', 'G']
POLICY = 'canary_both_roles_first_v1'
INPUTS = ('local_screen/candidate_windows.jsonl', 'history/reconciled_history.json',
          'history/reconciliation_report.json', 'history/review_input_snapshot.jsonl',
          'protocol/proposed_config.json', 'enrollment/preflight_report.json')


def fresh_history(project, source, out, candidates):
    """Refresh recognized local use; reuse attestations, never silently broaden them."""
    from audit_v91_history import SUPPLEMENTAL, portable_path, scan_file
    base = read_json(source / 'history/reconciled_history.json')
    release = set(base.get('released_source_groups', []))
    targets = release | {group_id(r['video_id']) for r in candidates}
    previous = read_json(source / 'history/reconciliation_report.json')
    old_audit = portable_path(previous['fresh_audit'])
    old_files = {portable_path(r['path']).resolve(): r['sha256']
                 for r in iter_jsonl(old_audit / 'scanned_files.jsonl')}
    files = set(old_files)
    for sibling in sorted(project.parent.iterdir()):
        root = sibling / 'runs'
        if not root.is_dir(): continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and
                             (Path(directory)/d).resolve() != out.resolve())
            files.update((Path(directory)/n).resolve() for n in set(names) & (HISTORY_NAMES | SUPPLEMENTAL))
    audit = out / 'history_checks' / ('check_' + uuid.uuid4().hex[:12])
    evidence, errors, scanned, kinds = [], [], [], {}
    registered = set(base['source_groups']) | release
    for i, path in enumerate(sorted(files), 1):
        before = path.stat() if path.is_file() else None
        rows, issues, meta = scan_file(path, targets, 1024**3)
        if before and path.is_file() and (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            issues.append({'path': str(path), 'issue': 'HISTORY_CHANGED_DURING_SCAN'})
        errors.extend(issues); evidence.extend(rows)
        if meta:
            scanned.append(meta)
            registered.update(meta['source_groups_found'])
        for r in rows: kinds.setdefault(r['source_group'], set()).add(r['kind'])
        if i % 20 == 0 or i == len(files):
            print(f'[history-refresh] {i}/{len(files)} files; errors={len(errors)}; API=0', flush=True)
    safe_releases = {g for g in release if kinds.get(g) == {'planned_registration'}}
    for g in sorted(release-safe_releases):
        errors.append({'source_group': g, 'issue': 'RELEASE_CONFLICT_OR_MISSING_PLANNED_EVIDENCE',
                       'current_evidence_kinds': sorted(kinds.get(g, []))})
    report = {'files_scanned': len(scanned), 'errors': errors,
              'changed_or_new_files': [r['path'] for r in scanned if old_files.get(Path(r['path']).resolve()) != r['sha256']],
              'released_sources_retained': len(safe_releases), 'remote_calls': 0,
              'scope': 'recognized local records only; external-use attestation still required before review/freeze'}
    history = {**base, 'source_groups': sorted(registered-safe_releases),
               'issues': list(base.get('issues', []))+errors}
    write_jsonl(audit/'scanned_files.jsonl', scanned)
    write_jsonl(audit/'source_evidence.jsonl', evidence)
    write_json(audit/'report.json', report)
    write_json(audit/'reconciled_history.json', history)
    write_json(out/'history_checks/latest.json', {'path': str(audit), 'report_sha256': file_sha256(audit/'report.json')})
    return history, report


def scope_report(rows):
    counts, missing = {}, []
    for role in QUOTAS:
        canaries = [r for r in rows if r['role']==role and r['stratum']=='other_class_canary']
        counts[role] = {}
        for code in CANARY_CODES:
            match = [r for r in canaries if code in anomaly_codes(r['video_id'])]
            counts[role][code] = {'windows': len(match), 'source_groups': len({r['source_group'] for r in match})}
            if not match: missing.append({'role': role, 'category': code, 'minimum_source_groups': 1})
    return {'adequate_filename_category_scope': not missing, 'coverage': counts, 'missing': missing,
            'not_semantic_verification': True,
            'claim_limit': 'small prospective train-source canary study, not full six-class benchmark accuracy'}


def verify_anchor_sources(rows):
    from audit_v91_history import portable_path
    checked, errors = {}, []
    for row in rows:
        if row['weak_training_target'] != 1: continue
        for item in row.get('positive_span_provenance', []):
            path=portable_path(item.get('path',''))
            if path not in checked: checked[path]=file_sha256(path) if path.is_file() else None
            if not item.get('sha256') or checked[path]!=item['sha256']:
                errors.append({'window_uid':row['window_uid'],'error':'ANCHOR_PROVENANCE_MISSING_OR_CHANGED'})
    return errors


def validate_candidates(rows):
    for row in rows:
        if any(type(row.get(k)) is not int for k in ('start_frame','end_frame_exclusive')):
            raise ValueError('candidate frame coordinates must be integers')
        start=row['start_frame']
        if row.get('sampled_frame_indices')!=[start+int(i*95/7) for i in range(8)]:
            raise ValueError('candidate must preserve the eight-frame local screening contract')
        code=anomaly_codes(row.get('video_id','')); stratum=row.get('stratum')
        if stratum=='B1_weak_positive' and 'B1' not in code or stratum=='B4_weak_positive' and 'B4' not in code:
            raise ValueError('stratum and video category mismatch')
        if stratum=='other_class_canary' and not code.intersection(CANARY_CODES):
            raise ValueError('canary must be a non-B1/B4 anomaly category')


def audit_selection(rows, history):
    errors, counts, roles, uids = [], Counter(), {}, set()
    for i, r in enumerate(rows):
        uid = WindowKey('train', r['video_id'], r['start_frame'], r['end_frame_exclusive']).uid
        g, role, s = group_id(r['video_id']), r['role'], r['stratum']
        if uid in uids: errors.append('duplicate_window_uid')
        uids.add(uid)
        if r['window_uid']!=uid or r['source_group']!=g: errors.append('identity_mismatch')
        if g in roles and roles[g]!=role: errors.append('source_role_overlap')
        roles[g] = role; counts[g]+=1; counts[(role,s)]+=1
        if g in history['source_groups']: errors.append('historical_source_exposure')
        target = weak_label(r)
        if r['weak_training_target']!=target or r['training_loss_mask'] is not (role=='adaptation' and target is not None):
            errors.append('invalid_target_or_training_loss_mask')
        if r['dataset_partition']!='train' or r['end_frame_exclusive']-r['start_frame']!=96:
            errors.append('partition_or_window_contract')
        for old in rows[:i]:
            if old['video_id']==r['video_id'] and max(old['start_frame'],r['start_frame'])<min(old['end_frame_exclusive'],r['end_frame_exclusive']):
                errors.append('same_video_window_overlap')
    if any(counts[g]>2 for g in roles): errors.append('source_capacity_exceeded')
    gaps = [{'role':role,'stratum':s,'requested':n,'available':counts[(role,s)]}
            for role,q in QUOTAS.items() for s,n in q.items() if counts[(role,s)]!=n]
    return {'errors': sorted(set(errors)), 'quota_gaps': gaps, 'selected_n':len(rows),
            'source_groups':len(roles), 'structurally_ready':not errors and not gaps and not history['issues']}


def scientific_plan():
    return {'version':'v91_prospective_scope_and_review_v1', 'approved_by':'', 'rationale':'',
            'external_history_checked':False, 'source_aliases_checked':False,
            'no_outside_source_use_since_audit':False,
            'required_canary_codes':CANARY_CODES, 'minimum_canary_source_groups_per_role_per_code':1,
            'weak_labels_are_not_gold':True, 'unverified_context_training_mask':False,
            'human_adaptation_labels_use':'audit_only', 'human_locked_labels_use':'evaluation_only',
            'uncertain_binary_metric_policy':'exclude_and_report_coverage',
            'hard_normal_semantics':'full-window normal AND suspicious-action resemblance AND concrete normal mechanism',
            'min_human_hard_normal_per_role':{'adaptation':None,'locked_evaluation':None},
            'hard_normal_fp_increase_tolerance':None, 'risk_confidence_level':None,
            'minimum_binary_evaluation_coverage':None,
            'primary':'T1_direct2_vs_T0_m0', 'secondary':'B1_event_bound4_vs_B0_event_unbound3',
            'evaluation_population':'new_training_sources_not_official_test',
            'remote_execution_authorized':False, 'deployment_authorized':False}


def validate_plan(plan):
    default = scientific_plan()
    editable = {'approved_by','rationale','external_history_checked','source_aliases_checked',
                'no_outside_source_use_since_audit','min_human_hard_normal_per_role',
                'hard_normal_fp_increase_tolerance','risk_confidence_level','minimum_binary_evaluation_coverage'}
    errors = [f'immutable_plan_field:{k}' for k,v in default.items() if k not in editable and plan.get(k)!=v]
    if set(plan)!=set(default): errors.append('unexpected_or_missing_plan_fields')
    for k in ('approved_by','rationale'):
        if not isinstance(plan.get(k),str) or not plan[k].strip(): errors.append('required:'+k)
    for k in ('external_history_checked','source_aliases_checked','no_outside_source_use_since_audit'):
        if plan.get(k) is not True: errors.append('required_attestation:'+k)
    for k in ('hard_normal_fp_increase_tolerance','risk_confidence_level','minimum_binary_evaluation_coverage'):
        v=plan.get(k)
        if type(v) not in (int,float) or not 0<=v<=1 or (k!='hard_normal_fp_increase_tolerance' and not 0<v<1.000000001):
            errors.append('invalid_predeclared_risk_setting:'+k)
    if plan.get('risk_confidence_level') == 1: errors.append('confidence_must_be_less_than_one')
    minima=plan.get('min_human_hard_normal_per_role',{})
    if not isinstance(minima,dict) or set(minima)!=set(QUOTAS): errors.append('hard_normal_minima_roles')
    else:
        for role in QUOTAS:
            if type(minima[role]) is not int or not 1<=minima[role]<=QUOTAS[role]['hard_label_A']:
                errors.append('invalid_hard_normal_minimum:'+role)
    return errors


def prepare(project, source, out, additional=()):
    from audit_v91_history import portable_path
    paths = [source/rel for rel in INPUTS] + list(additional)
    if out==source or source in out.parents or out in source.parents:
        raise ValueError('new independent output/TAG required')
    if any(out==p or out in p.parents for p in paths): raise ValueError('input is inside output')
    if out.exists() and any(out.iterdir()): raise ValueError('output exists; preserve it and use a new TAG')
    hashes = {str(p):file_sha256(p) for p in paths}
    old_report=read_json(source/'enrollment/preflight_report.json')
    if old_report['candidate_input_sha256']!=hashes[str(source/INPUTS[0])]:
        raise ValueError('source candidates changed since original enrollment')
    code_paths = ['tools/event_decision/reenrollment.py','tools/event_decision/enrollment_review.py',
                  'tools/event_decision/contracts.py','tools/event_decision/safety.py',
                  'tools/event_decision/hard_trial.py','tools/reenroll_v91_trial.py','tools/audit_v91_history.py']
    contract={'version':'v91_reenrollment_v1','source_run':str(source),'input_hashes':hashes,
              'allocation_policy':POLICY,'code_hashes':{n:file_sha256(project/n) for n in code_paths},
              'remote_execution_authorized':False}
    candidates=[]
    for p in [source/INPUTS[0]]+list(additional): candidates.extend(iter_jsonl(p))
    forbidden={'competitions','y_pred','predictions','model_score','current_window_visual_label','role','weak_training_target'}
    if any(forbidden & r.keys() for r in candidates): raise ValueError('candidates must be pre-outcome local screening rows')
    validate_candidates(candidates)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'reenrollment_contract.json',contract)
    write_jsonl(out/'inputs/candidate_snapshot.jsonl',candidates)
    history, hreport=fresh_history(project,source,out,candidates)
    config={**read_json(source/'protocol/proposed_config.json'),'allocation_policy':POLICY,
            'quotas':QUOTAS,'required_canary_codes':CANARY_CODES,'frozen':False}
    selected,report=enroll(candidates,history,config)
    for r in selected:
        r['label_evidence_level'] = ('unverified_context' if r['weak_training_target'] is None else
                                     'weak_positive_anchor' if r['weak_training_target']==1 else 'dataset_video_label_A')
        r['human_window_target']=None
        r['gold_metric_mask']=False
    checks=audit_selection(selected,history); scope=scope_report(selected)
    anchor_errors=verify_anchor_sources(selected)
    if anchor_errors:
        checks['errors'].append('ANCHOR_PROVENANCE_MISSING_OR_CHANGED')
        checks['structurally_ready']=False
    write_json(out/'enrollment/anchor_provenance_check.json',{'errors':anchor_errors})
    report.update(checks, allocation_policy=POLICY, candidate_input_sha256=hashes[str(source/INPUTS[0])],
                  combined_candidate_sha256=file_sha256(out/'inputs/candidate_snapshot.jsonl'),
                  adequate_category_scope=scope['adequate_filename_category_scope'],
                  ready_for_enrollment_review=checks['structurally_ready'] and scope['adequate_filename_category_scope'],
                  ready_for_protocol_freeze=False,remote_execution_authorized=False,
                  status='WAITING_FOR_CATEGORY_COVERAGE' if not scope['adequate_filename_category_scope'] else 'WAITING_FOR_SCIENTIFIC_PLAN_AND_HUMAN_REVIEW',
                  history_requires_review=bool(history['issues']))
    if not checks['structurally_ready']: report['status']='WAITING_FOR_QUOTAS_OR_HISTORY'
    write_json(out/'protocol/proposed_config.json',config)
    write_json(out/'protocol/scientific_plan.json',scientific_plan())
    write_jsonl(out/'enrollment/private_proposed_windows.jsonl',selected)
    write_json(out/'enrollment/preflight_report.json',report)
    write_json(out/'enrollment/category_scope_report.json',scope)
    write_json(out/'enrollment/category_gap_plan.json',{
        'missing':scope['missing'],'acceptable_input':'--additional-candidates verified local screening JSONL in a new TAG',
        'requirements':['unexposed training source groups','explicit positive anchor provenance',
                        '96-frame windows and eight-frame fingerprints','no model outcomes or human answers used to select'],
        'do_not':['borrow old validation sources','relabel G as another category','lower quotas or edit gates'],
        'new_decoding_performed':False})
    signatures=[]
    for r in selected:
        p=portable_path(r['video_path']); stat=p.stat() if p.is_file() else None
        signatures.append({'window_uid':r['window_uid'],'path':str(p),
                           'size':stat.st_size if stat else None,'mtime_ns':stat.st_mtime_ns if stat else None})
    write_jsonl(out/'enrollment/media_signatures.jsonl',signatures)
    write_json(out/'evaluation/measurement_status.json',{
        'status':'NOT_MEASURED','accuracy':None,'AP':None,'predicted_windows':0,
        'reason':'Enrollment is not inference. Future metrics require frozen models/predictions and independent human labels.',
        'context_unverified_is_not_normal':True,'official_test_metric':False})
    for p in paths:
        if file_sha256(p)!=hashes[str(p)]: raise ValueError('source input changed during re-enrollment')
    bound=['enrollment/private_proposed_windows.jsonl','enrollment/preflight_report.json',
           'enrollment/category_scope_report.json','enrollment/media_signatures.jsonl',
           'protocol/proposed_config.json','inputs/candidate_snapshot.jsonl']
    write_json(out/'enrollment/integrity.json',{'files':{p:file_sha256(out/p) for p in bound}})
    return report


def verify_integrity(project,out):
    from audit_v91_history import portable_path
    c=read_json(out/'reenrollment_contract.json')
    if not c: raise ValueError('not a re-enrollment run')
    for path,digest in c['input_hashes'].items():
        if file_sha256(portable_path(path))!=digest: raise ValueError('source input changed: '+path)
    for path,digest in c['code_hashes'].items():
        if file_sha256(project/path)!=digest: raise ValueError('workflow code changed; use a new TAG: '+path)
    integrity=read_json(out/'enrollment/integrity.json')
    if not integrity: raise ValueError('prepare incomplete; use a new TAG')
    for path,digest in integrity['files'].items():
        if file_sha256(out/path)!=digest: raise ValueError('enrollment input changed: '+path)
    return c
