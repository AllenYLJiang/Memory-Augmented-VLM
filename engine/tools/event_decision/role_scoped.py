"""V9.2 opt-in development acquisition, isolated from hash-bound V9.1 runs.

The inventory is NOT a history release. The allocator produces a preview, never
a reviewed/frozen protocol or an authorization to infer locked-evaluation data.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from .contracts import WindowKey, file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from .hard_trial import QUOTAS, anomaly_codes, group_id, weak_label
from .reenrollment import CANARY_CODES, scope_report, validate_candidates, verify_anchor_sources

VERSION = 'v92_exposed_development_independent_locked_v1'
PROMPT = '''Inspect the eight chronological images T0..T7 from one short silent clip.
Assess visible abuse: sustained one-sided coercion, restraint, assault on a
defenseless person, or prevention of escape. Distinguish mutual fighting,
assistance/rescue, sports/play, crowd activity and unclear interaction. Do not
infer abuse just from proximity, distress, a held object, or a filename.
Return only JSON with exactly these fields:
{"b5_presence":"yes|no|uncertain", "b5_probability":0.0,
 "confidence":"low|medium|high", "visible_evidence":"concrete observations",
 "normal_or_other_explanation":"alternative explanation or none observed",
 "same_actor_same_event":"yes|no|uncertain",
 "supporting_bins":[0], "per_frame":[
 {"bin":0,"b5_support":"yes|no|uncertain","evidence":"visible observation"}]}
per_frame must contain exactly eight ordered entries, bin 0 through 7.
supporting_bins must list precisely the bins whose b5_support is yes. They are
sampled images, NOT continuous event intervals. Do not invent activity between
images or claim that eight sparse samples establish eight consecutive frames.
No GT, movie name, filename category, or prior model result is supplied.
'''


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def portable(value):
    from audit_v91_history import portable_path
    return portable_path(value)


def code_hashes(project, code_dir):
    paths = [project / x for x in (
        'tools/event_decision/role_scoped.py', 'tools/event_decision/b5_development_screen.py',
        'tools/role_scoped_v92.py', 'tools/event_decision/contracts.py',
        'tools/event_decision/hard_trial.py', 'tools/event_decision/reenrollment.py',
        'tools/event_decision/local_screen.py', 'tools/event_decision/enrollment_review.py',
        'tools/audit_v91_history.py', 'docs/audit_v91_canary_sources.py')]
    paths.append(code_dir / 'src/structural_vlm_binary/vlm/dashscope_backend.py')
    return {str(p.resolve()): file_sha256(p) for p in paths}


def assert_bound(out):
    protocol = read_json(out / 'protocol.json')
    if not protocol or protocol.get('version') != VERSION:
        raise ValueError('not a V9.2 role-scoped run')
    integrity = read_json(out / 'integrity.json')
    if not integrity or file_sha256(out / 'protocol.json') != integrity['protocol_sha256']:
        raise ValueError('protocol changed or prepare incomplete; preserve run and use a new TAG')
    for path, digest in protocol['input_hashes'].items():
        if file_sha256(portable(path)) != digest:
            raise ValueError('input/code changed: ' + path)
    for rel, digest in integrity['files'].items():
        if file_sha256(out / rel) != digest:
            raise ValueError('fixed inventory changed: ' + rel)
    return protocol


def role_assignment(source, audit_row, reserved):
    # Unknown/planned sources remain reserved, rather than consuming the only
    # potential fresh evaluation source through speculative development calls.
    if source in reserved:
        return 'reserved_pending_review'
    if audit_row and audit_row.get('primary_status') == 'RETAIN_EXCLUSION_EXECUTION_EVIDENCE':
        return 'previously_exposed_development'
    return 'reserved_pending_review'


def choose_starts(video, spans, nframes, limit, seed):
    if nframes < 96:
        return []
    if spans:
        starts = {min(max(0, (a+b)//2 - 48), nframes-96) for a,b in spans}
    else:
        # Filename-driven exploration only; it does not create positive labels.
        starts = {int(i*(nframes-96)/max(1,limit-1)) for i in range(limit)}
    chosen = []
    for start in sorted(starts, key=lambda s: semantic_sha256([seed,video,s])):
        if all(abs(start-other) >= 96 for other in chosen):
            chosen.append(start)
        if len(chosen) == limit:
            break
    return sorted(chosen)


def prepare(project, out, audit, source, train_root, anchors, code_dir, *,
            seed=20260912, max_windows=2, model='qwen3.6-plus', reserved=(), mock=False):
    from audit_v91_canary_sources import list_videos, anchor_audit
    from .local_screen import frame_probe
    if max_windows < 1:
        raise ValueError('max_windows must be positive')
    if out.exists():
        raise ValueError('new output/TAG required; use screen/status to resume an existing run')
    if not train_root.is_dir():
        raise ValueError('training media root is unavailable')
    for root in (source,audit,train_root,anchors):
        if root == out or root in out.parents or out in root.parents:
            raise ValueError('output must be independent of input roots')
    group_path = audit / 'canary_source_audit.jsonl'
    groups = {r['source_group']:r for r in iter_jsonl(group_path)}
    candidate_path = source / 'local_screen/candidate_windows.jsonl'
    history_path = source / 'history/reconciled_history.json'
    hashes = code_hashes(project,code_dir)
    for p in (group_path,candidate_path,history_path): hashes[str(p)] = file_sha256(p)
    issues = []
    videos, total = list_videos(train_root, {'B5'}, issues)
    if issues: raise ValueError('media inventory incomplete: '+json.dumps(issues))
    eligibility, windows, errors, reservation = [], [], [], set(reserved)
    for vid in sorted(videos):
        g = group_id(vid)
        if role_assignment(g,groups.get(g),reservation) != 'previously_exposed_development':
            reservation.add(g)
    out.mkdir(parents=True)
    anchor_inputs = {}
    for i, (vid, paths) in enumerate(sorted(videos.items()),1):
        g = group_id(vid); role = role_assignment(g,groups.get(g),reservation)
        entry = {'video_id':vid,'source_group':g,'filename_categories':sorted(anomaly_codes(vid)),
                 'media_paths':[str(p) for p in paths], 'assigned_pool':role,
                 'history_status':groups.get(g,{}).get('primary_status','NOT_IN_SNAPSHOT'),
                 'eligible_for_locked_evaluation':False}
        eligibility.append(entry)
        if role != 'previously_exposed_development': continue
        try:
            # Duplicate paths must really be copies, not silently chosen aliases.
            digests = {}
            for p in paths:
                before = (p.stat().st_size,p.stat().st_mtime_ns)
                digests[str(p)] = file_sha256(p)
                if before != (p.stat().st_size,p.stat().st_mtime_ns):
                    raise ValueError('media changed during hashing')
            hashes.update(digests)
            if len(set(digests.values())) != 1:
                raise ValueError('duplicate video IDs have different file contents')
            path = paths[0]; n = frame_probe(path)
            info = anchor_audit(vid, anchors, anchor_inputs)
            spans = info['positive_spans_inclusive'] if info['structurally_traceable_anchor'] else []
            starts = choose_starts(vid,spans,n,max_windows,seed)
            if not starts: raise ValueError('video shorter than 96 frames')
            entry['anchor_issues'] = info['problems']
            for start in starts:
                uid = WindowKey('train',vid,start,start+96).uid
                windows.append({'window_uid':uid,'dataset_partition':'train','video_id':vid,
                    'video_path':str(path),'source_group':g,'source_media_sha256':digests[str(path)],
                    'start_frame':start,'end_frame_exclusive':start+96,
                    'sampled_frame_indices':[start+int(i*95/7) for i in range(8)],
                    'assigned_pool':role,'allowed_roles':['adaptation'],
                    'stratum':'other_class_canary',
                    'label_source':'verified_positive_anchor' if spans else 'unverified_development_window',
                    'positive_spans_half_open':[[a,b+1] for a,b in spans],
                    'positive_span_provenance':[{'path':f['path'],'sha256':f['sha256']}
                                                for f in info['files'] if spans],
                    'selection_rule':'hash_selected_existing_anchors_or_uniform_unlabeled_v1',
                    'human_window_target':None,'gold_metric_mask':False})
        except (OSError,ValueError,KeyError) as exc:
            errors.append({'video_id':vid,'error':str(exc)})
        print(f'[prepare-B5] {i}/{len(videos)} windows={len(windows)} errors={len(errors)} API=0',flush=True)
    hashes.update({p:d['sha256'] for p,d in anchor_inputs.items()})
    protocol = {'version':VERSION, 'policy_basis':'user_explicitly_approved_role_scoped_protocol',
        'source_run':str(source),'history_audit':str(audit),'train_root':str(train_root),
        'code_dir':str(code_dir),'model':model,'mock':mock,'seed':seed,'max_windows_per_video':max_windows,
        'role_policy':{'adaptation':'declared historical exposure allowed',
                       'locked_evaluation':'requires fresh history clearance, no development source overlap'},
        'reserved_source_groups':sorted(reservation),
        'development_source_groups':sorted({r['source_group'] for r in windows}),
        'input_hashes':hashes,'prompt':PROMPT,'quotas':QUOTAS,
        'locked_inference_authorized':False,'formal_protocol_frozen':False,
        'development_screening_only':True,'discovery_enabled':False,
        'vlm_screen_is_gold':False,'automatic_label_replacement':False,
        'history_scope':'saved inventory plus new local B5 enumeration; NOT a full fresh history release',
        'evaluation_claim':'new-source small canary study only after separate fresh-history/review/freeze'}
    write_json(out/'protocol.json',protocol)
    write_json(out/'run_config.json',{'mock':mock,'version':VERSION,'model':model,'purpose':'development_B5_screen'})
    write_jsonl(out/'inventory/b5_videos.jsonl',eligibility)
    write_jsonl(out/'inventory/development_windows.jsonl',windows)
    write_jsonl(out/'inventory/prepare_errors.jsonl',errors)
    write_jsonl(out/'inventory/reserved_sources.jsonl',[
        {'source_group':g,'status':'RESERVED_NOT_APPROVED','inference_allowed':False} for g in sorted(reservation)])
    summary = {'status':'DEVELOPMENT_POOL_READY' if windows else 'NO_DEVELOPMENT_WINDOWS',
        'all_train_video_paths':total,'b5_paths':sum(map(len,videos.values())),
        'b5_video_ids':len(videos),'b5_source_groups':len({group_id(v) for v in videos}),
        'development_videos':len({r['video_id'] for r in windows}),
        'development_sources':len(protocol['development_source_groups']), 'development_windows':len(windows),
        'reserved_sources':sorted(reservation),'media_or_anchor_errors':len(errors),
        'logical_calls_without_retries':len(windows),'ready_for_formal_enrollment':False,'remote_calls':0}
    write_json(out/'inventory/summary.json',summary)
    bound = ['inventory/b5_videos.jsonl','inventory/development_windows.jsonl',
             'inventory/prepare_errors.jsonl','inventory/reserved_sources.jsonl','run_config.json']
    write_json(out/'integrity.json',{'protocol_sha256':file_sha256(out/'protocol.json'),
                                    'files':{p:file_sha256(out/p) for p in bound}})
    assert_bound(out)
    return summary


def valid_screen(value):
    required = {'b5_presence','b5_probability','confidence','visible_evidence',
                'normal_or_other_explanation','same_actor_same_event','supporting_bins','per_frame'}
    if not isinstance(value,dict) or set(value) != required:
        raise ValueError('B5 response must have exactly the requested fields, not an API error envelope')
    choices = {'yes','no','uncertain'}
    if value['b5_presence'] not in choices or value['same_actor_same_event'] not in choices:
        raise ValueError('invalid categorical state')
    p = value['b5_probability']
    if type(p) not in (int,float) or not 0 <= p <= 1:
        raise ValueError('invalid finite probability')
    if value['confidence'] not in {'low','medium','high'}:
        raise ValueError('invalid confidence')
    for k in ('visible_evidence','normal_or_other_explanation'):
        if not isinstance(value[k],str) or not value[k].strip(): raise ValueError('missing evidence text')
    frames = value['per_frame']
    if not isinstance(frames,list) or len(frames) != 8: raise ValueError('exactly eight frame records required')
    for i,r in enumerate(frames):
        if not isinstance(r,dict) or set(r) != {'bin','b5_support','evidence'}:
            raise ValueError('invalid frame record')
        if type(r['bin']) is not int or r['bin'] != i or r['b5_support'] not in choices:
            raise ValueError('frame bins must be ordered integers 0..7')
        if not isinstance(r['evidence'],str) or not r['evidence'].strip(): raise ValueError('missing frame evidence')
    bins = value['supporting_bins']
    if not isinstance(bins,list) or any(type(x) is not int for x in bins): raise ValueError('invalid supporting bins')
    if bins != [r['bin'] for r in frames if r['b5_support']=='yes']: raise ValueError('supporting bins inconsistent')
    if value['b5_presence']=='yes' and not bins: raise ValueError('positive claim without supporting sample')
    return value


def role_allowed(group, role, historical, reserved, pending, development):
    if role == 'adaptation': return group not in reserved and group not in pending
    if role == 'locked_evaluation': return group not in (historical | pending | development)
    return False


def reserve_canaries(rows, quotas, historical, reserved, pending, development):
    """Joint source-level DP: meet both role quotas before maximizing diversity.

    A state records counts and category masks in the two roles. A source may
    contribute at most two non-overlapping windows, to exactly one role. Keeping
    only the best assignment per state avoids enumerating every allocation.
    """
    roles=('adaptation','locked_evaluation')
    targets=tuple(quotas[r].get('other_class_canary',0) for r in roles)
    grouped={}
    for i,row in enumerate(rows): grouped.setdefault(row['source_group'],[]).append(i)
    def mask(indices):
        codes=set().union(*(anomaly_codes(rows[i]['video_id']) for i in indices))
        return sum(1<<i for i,c in enumerate(CANARY_CODES) if c in codes)
    # value: distinct sources, negative fresh-development use, chosen assignments
    states={(0,0,0,0):(0,0,())}
    for group,indices in grouped.items():
        options={}
        for role_index,role in enumerate(roles):
            if not role_allowed(group,role,historical,reserved,pending,development): continue
            for n in (1,2):
                for combo in combinations(indices,n):
                    if n==2:
                        a,b=(rows[i] for i in combo)
                        if a['video_id']==b['video_id'] and max(a['start_frame'],b['start_frame'])<min(a['end_frame_exclusive'],b['end_frame_exclusive']): continue
                    options.setdefault((role_index,n,mask(combo)),combo)
        updated=dict(states)
        for state,value in states.items():
            for (role_index,n,categories),combo in options.items():
                if state[role_index]+n>targets[role_index]: continue
                new=list(state);new[role_index]+=n;new[role_index+2]|=categories;new=tuple(new)
                fresh=int(role_index==0 and group not in historical and group not in development)
                proposed=(value[0]+1,value[1]-fresh,value[2]+((role_index,combo),))
                if new not in updated or proposed[:2]>updated[new][:2]: updated[new]=proposed
        states=updated
    best=max(states,key=lambda s:(s[0]+s[1],s[2].bit_count()+s[3].bit_count(),states[s][0],states[s][1]))
    return [(roles[role_index],rows[i]) for role_index,combo in states[best][2] for i in combo]


def allocate_preview(candidates, historical, reserved=(), pending=(), development=(), seed=20260912, quotas=None):
    """Role-constrained allocation, not authority to freeze or acquire evaluation."""
    quotas = quotas or QUOTAS
    historical,reserved,pending,development = map(set,(historical,reserved,pending,development))
    eligible,errors,seen = [],[],set()
    for raw in candidates:
        try:
            row = dict(raw); validate_candidates([row])
            start,end = row['start_frame'],row['end_frame_exclusive']
            if end-start != 96 or row.get('dataset_partition')!='train': raise ValueError('invalid window/partition')
            if row.get('media_verified') is not True: raise ValueError('unverified media')
            fp=row.get('perceptual_fingerprint',[])
            if len(fp)!=8 or any(not re.fullmatch('[0-9a-f]{16}',str(x)) for x in fp): raise ValueError('invalid fingerprints')
            if not re.fullmatch('[0-9a-f]{64}',str(row.get('evidence_sha256',''))): raise ValueError('invalid evidence digest')
            label=weak_label(row); s=row['stratum']
            if s not in QUOTAS['adaptation']: raise ValueError('unknown stratum')
            if s in ('B1_weak_positive','B4_weak_positive','other_class_canary') and label!=1: raise ValueError('no positive anchor')
            if s in ('hard_label_A','easy_label_A') and label!=0: raise ValueError('no explicit normal')
            if s=='hard_postevent_unverified' and label is not None: raise ValueError('context is not supervised normal')
            uid=WindowKey('train',row['video_id'],start,end).uid
            if uid in seen: continue
            seen.add(uid)
            row.update(window_uid=uid,source_group=group_id(row['video_id']),weak_training_target=label)
            eligible.append(row)
        except (ValueError,KeyError,TypeError) as exc:
            errors.append({'video_id':raw.get('video_id'),'reason':str(exc)})
    eligible.sort(key=lambda r:semantic_sha256([seed,r['window_uid']]))
    blocked=set(); duplicates=[]
    for i,r in enumerate(eligible):
        for q in eligible[:i]:
            if r['source_group']==q['source_group']: continue
            close=sum((int(a,16)^int(b,16)).bit_count() for a,b in zip(r['perceptual_fingerprint'],q['perceptual_fingerprint']))<=32
            if close or r['evidence_sha256']==q['evidence_sha256']:
                blocked.update((r['source_group'],q['source_group']))
                duplicates.append([r['window_uid'],q['window_uid']])
    selected=[]; roles={}; counts=Counter(); used=set()
    def take(role,stratum,target,pool):
        for cap in (1,2):
            for r in pool:
                if counts[role,stratum]>=target: return
                g=r['source_group']
                if g in blocked or r['window_uid'] in used or counts[g]>=cap: continue
                if g in roles and roles[g]!=role: continue
                if not role_allowed(g,role,historical,reserved,pending,development): continue
                if any(q['video_id']==r['video_id'] and max(q['start_frame'],r['start_frame'])<min(q['end_frame_exclusive'],r['end_frame_exclusive']) for q in selected): continue
                selected.append({**r,'role':role,'training_loss_mask':role=='adaptation' and r['weak_training_target'] is not None,
                                 'exposure_policy':VERSION,'previously_exposed':g in historical,
                                 'gold_metric_mask':False,'human_window_target':None})
                roles[g]=role;used.add(r['window_uid']);counts[g]+=1;counts[role,stratum]+=1
    canary=[r for r in eligible if r['stratum']=='other_class_canary' and r['source_group'] not in blocked]
    for role,row in reserve_canaries(canary,quotas,historical,reserved,pending,development):
        take(role,'other_class_canary',counts[role,'other_class_canary']+1,[row])
    for role in ('locked_evaluation','adaptation'):
        for s,n in quotas[role].items():
            if s!='other_class_canary': take(role,s,n,[r for r in eligible if r['stratum']==s])
    gaps=[{'role':role,'stratum':s,'requested':n,'available':counts[role,s]} for role,qs in quotas.items() for s,n in qs.items() if counts[role,s]!=n]
    scope=scope_report(selected)
    return selected,{'status':'PREVIEW_ONLY_REQUIRES_FRESH_HISTORY_AND_REVIEW','selected_n':len(selected),
        'quota_gaps':gaps,'rejected':errors,'duplicate_pairs':duplicates,'scope':scope,
        'reserved_sources':sorted(reserved),'pending_sources':sorted(pending),
        'allocation_policy':'joint_canary_role_capacity_category_dp_v1_then_context',
        'source_role_overlap':False,'ready_for_protocol_freeze':False,'remote_execution_authorized':False}


def preview(out):
    protocol=assert_bound(out); source=portable(protocol['source_run'])
    rows=list(iter_jsonl(source/'local_screen/candidate_windows.jsonl'))
    local=out/'development/local_candidates.jsonl'
    if local.is_file(): rows.extend(iter_jsonl(local))
    history=read_json(source/'history/reconciled_history.json')
    groups=list(iter_jsonl(portable(protocol['history_audit'])/'canary_source_audit.jsonl'))
    historical=set(history['source_groups'])
    historical.update(r['source_group'] for r in groups if r['primary_status'].startswith('RETAIN_EXCLUSION'))
    pending={r['source_group'] for r in groups if r['primary_status'] not in ('RETAIN_EXCLUSION_EXECUTION_EVIDENCE','RETAIN_EXCLUSION_UNVERIFIED_EXECUTION') and not r.get('prior_release_in_input_snapshot')}
    pending.update(protocol['reserved_source_groups'])
    selected,report=allocate_preview(rows,historical,protocol['reserved_source_groups'],pending,
                                     protocol['development_source_groups'],protocol['seed'])
    # No screen probabilities/results are read; only independent anchor/media data.
    anchor_errors=verify_anchor_sources(selected)
    report.update(anchor_errors=anchor_errors,selection_used_vlm_outcomes=False,
                  history_refreshed=False,formal_metrics=None)
    write_jsonl(out/'allocation_preview/selected_windows_preview.jsonl',selected)
    write_json(out/'allocation_preview/report.json',report)
    return report
