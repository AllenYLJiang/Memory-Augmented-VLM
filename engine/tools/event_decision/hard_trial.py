"""Offline enrollment preflight for a fixed, source-disjoint 48/96-window trial.

History is evidence of use, not mere presence in a directory. This preflight cannot
authorize API calls, claim complete history, or silently repair quota deficits.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from .contracts import WindowKey, file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl


QUOTAS = {
    'adaptation': {'B1_weak_positive': 8, 'B4_weak_positive': 8, 'hard_label_A': 18,
                   'hard_postevent_unverified': 6, 'easy_label_A': 4, 'other_class_canary': 4},
    'locked_evaluation': {'B1_weak_positive': 16, 'B4_weak_positive': 16, 'hard_label_A': 30,
                          'hard_postevent_unverified': 10, 'easy_label_A': 12, 'other_class_canary': 12}}

HISTORY_NAMES = {'selected_source_records.jsonl', 'selected_windows.jsonl', 'selection.jsonl',
                 'enrollment_manifest.jsonl', 'label_ledger.jsonl', 'source_records.jsonl',
                 'split_manifest.json', 'split_registry.json', 'development_splits.json',
                 'sample_manifest.json', 'graph_proposals.jsonl', 'graph_hypotheses.jsonl',
                 'frozen_development_model.json', 'ot_window_results.jsonl', 'frozen_pair_results.jsonl',
                 'failure_descriptions.jsonl', 'graph_catalog_v2.json'}
PRUNE = {'code_backups','backups','.cache','cache','vlm_cache','segments','images','frames','cases',
         'node_modules','__pycache__','.git','source_reference','.v91_testdeps'}


def group_id(video):
    video = str(video).replace('\\', '/').rsplit('/', 1)[-1]
    if video.lower().endswith('.mp4'): video = video[:-4]
    video = video.split('__seg', 1)[0]
    # Do not Path.stem a movie identifier such as Before.Sunset.2004.
    return video.split('__#', 1)[0].split('_label_', 1)[0]


def anomaly_codes(video):
    suffix = str(video).rsplit('_label_', 1)[-1].removesuffix('.mp4')
    return set(suffix.split('-')) & {'B1', 'B2', 'B4', 'B5', 'B6', 'G'}


def groups_in(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'source_group','source_group_id','video_id','segment_key','video_path'} and isinstance(item,str):
                if item.strip(): result.add(group_id(item))
            elif key in {'source_groups','fit_source_groups','threshold_source_groups','model_pool_groups','test_groups','excluded_source_groups','source_video_ids','video_ids','selected_video_ids'} and isinstance(item, list):
                result.update(group_id(x) for x in item if isinstance(x, str) and x)
            elif isinstance(item, (dict,list)):
                result.update(groups_in(item))
    elif isinstance(value,list):
        for item in value:
            if isinstance(item,(dict,list)): result.update(groups_in(item))
    return result


def history_inventory(roots, out, excluded_root=None):
    files, groups, issues, exposure_evidence = [], set(), [], []
    excluded_root = Path(excluded_root).resolve() if excluded_root else None
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            issues.append({'path': str(root), 'issue': 'MISSING_HISTORY_ROOT'}); continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and (excluded_root is None or (Path(directory)/d).resolve()!=excluded_root))
            for name in sorted(set(names) & HISTORY_NAMES):
                path = Path(directory)/name
                found = set()
                try:
                    if path.suffix == '.jsonl':
                        for row in iter_jsonl(path): found.update(groups_in(row))
                    else: found = groups_in(read_json(path))
                except (ValueError, OSError) as exc:
                    issues.append({'path': str(path), 'issue': type(exc).__name__}); continue
                files.append({'path': str(path.resolve()), 'sha256': file_sha256(path), 'recognized_groups': len(found)})
                exposure_evidence.extend({'source_group':g,'path':str(path.resolve()),'sha256':files[-1]['sha256']} for g in sorted(found))
                groups.update(found)
    result = {'version': 'history_exposure_inventory_v1', 'files': files, 'source_groups': sorted(groups),
              'issues': issues, 'history_completeness': 'REQUIRES_HUMAN_INVENTORY_REVIEW',
              'not_scanned': 'unregistered external runs, caches and arbitrary prompt logs',
              'remote_calls': 0}
    write_json(Path(out)/'history_inventory.json', result)
    write_jsonl(Path(out)/'source_exposure_evidence.jsonl',exposure_evidence)
    return result


def continuous_target(intervals, width=96):
    """Reviewer intervals are local inclusive; merge once into a half-open union."""
    points = set()
    for pair in intervals:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair):
            raise ValueError('inclusive local interval must contain two integer indices')
        a,b = pair
        if not 0 <= a <= b < width: raise ValueError('local interval outside clip')
        points.update(range(a,b+1))
    longest, run, previous = 0, 0, -2
    for point in sorted(points):
        run = run+1 if point == previous+1 else 1
        longest, previous = max(longest,run), point
    return int(longest >= 8)


def weak_label(row):
    if row.get('dataset_partition') != 'train':
        raise ValueError('unknown partition cannot default to train')
    if row.get('label_source') == 'explicit_filename_label_A' and re.search(r'_label_A(?:-0-0)?(?:\.mp4)?$', row['video_id']):
        return 0
    if row.get('label_source') == 'verified_positive_anchor' and row.get('positive_span_provenance'):
        intervals = [[max(0,int(a)-row['start_frame']), min(95,int(b)-row['start_frame']-1)]
                     for a,b in row.get('positive_spans_half_open',[]) if min(row['end_frame_exclusive'],b)>max(row['start_frame'],a)]
        return 1 if continuous_target(intervals) else None
    return None


def enroll(candidates, history, config):
    quotas = config.get('quotas', QUOTAS)
    seed = config.get('seed', 20260910)
    used_history = set(history['source_groups'])
    eligible, rejected = [], []
    for raw in candidates:
        row = dict(raw)
        try:
            if row.get('dataset_partition') != 'train': raise ValueError('PARTITION_NOT_EXPLICIT_TRAIN')
            key = WindowKey('train',row['video_id'],row['start_frame'],row['end_frame_exclusive'])
            if key.end_frame_exclusive-key.start_frame != 96: raise ValueError('NOT_A_COMPLETE_96_FRAME_WINDOW')
            group = group_id(row['video_id'])
            if group in used_history: raise ValueError('HISTORICAL_SOURCE_EXPOSURE')
            if row.get('media_verified') is not True: raise ValueError('MEDIA_NOT_VERIFIED')
            fp=row.get('perceptual_fingerprint')
            if not re.fullmatch('[0-9a-f]{64}',str(row.get('evidence_sha256',''))) or not isinstance(fp,list) or len(fp)!=8 or any(not re.fullmatch('[0-9a-f]{16}',str(x)) for x in fp):
                raise ValueError('DUPLICATE_AUDIT_UNAVAILABLE')
            label = weak_label(row)
            if row['stratum'] in ('B1_weak_positive','B4_weak_positive','other_class_canary') and label != 1: raise ValueError('POSITIVE_SCOPE_NOT_VERIFIED')
            if row['stratum'] in ('hard_label_A','easy_label_A') and label != 0: raise ValueError('NORMAL_NOT_EXPLICIT_LABEL_A')
            if row['stratum'] == 'hard_postevent_unverified' and label is not None: raise ValueError('POSTEVENT_IS_NOT_A_TRAINING_LABEL')
            row.update(window_uid=key.uid, source_group=group, weak_training_target=label)
            eligible.append(row)
        except (ValueError, KeyError) as exc:
            rejected.append({'video_id': row.get('video_id'), 'reason': str(exc)})
    eligible.sort(key=lambda r: semantic_sha256([seed, r['window_uid']]))
    roles, selected, used, digest_groups, gaps, duplicates = {}, [], set(), {}, [], []
    # Exact and near-duplicate groups are conservatively excluded pending review.
    blocked = set()
    for i, row in enumerate(eligible):
        for other in eligible[:i]:
            if row['source_group'] == other['source_group']: continue
            a,b = row['perceptual_fingerprint'],other['perceptual_fingerprint']
            close = len(a)==len(b)==8 and sum((int(x,16)^int(y,16)).bit_count() for x,y in zip(a,b)) <= 32
            if row['evidence_sha256'] == other['evidence_sha256'] or close:
                blocked.update([row['source_group'],other['source_group']])
                duplicates.append({'left':row['window_uid'],'right':other['window_uid'],'status':'HUMAN_MERGE_OR_EXCLUDE_REQUIRED'})
    counts = Counter()
    pools = {s: [r for r in eligible if r['stratum']==s and r['source_group'] not in blocked]
             for strata in quotas.values() for s in strata}

    def take(role, stratum, target, pool):
        # Prefer one per source; only use a second if the fixed quota needs it.
        for max_per_group in (1, 2):
            for row in pool:
                g, uid = row['source_group'], row['window_uid']
                if counts[(role, stratum)] >= target: break
                if uid in used or counts[g]>=max_per_group or (g in roles and roles[g]!=role): continue
                if config.get('disallow_overlapping_windows') and any(
                    previous['video_id']==row['video_id'] and
                    max(previous['start_frame'],row['start_frame']) < min(previous['end_frame_exclusive'],row['end_frame_exclusive'])
                    for previous in selected): continue
                roles[g]=role; used.add(uid); counts[g]+=1; counts[(role,stratum)]+=1
                selected.append({**row,'role':role,'blind_id':'H_'+semantic_sha256([seed,'blind',uid])[:16],
                                 'training_loss_mask': role=='adaptation' and row['weak_training_target'] is not None,
                                 'human_label_use':'audit_only' if role=='adaptation' else 'evaluation_only'})

    policy = config.get('allocation_policy', 'legacy_role_first')
    if policy not in ('legacy_role_first', 'canary_both_roles_first_v1'):
        raise ValueError('unknown allocation policy')
    canary = 'other_class_canary'
    if policy == 'canary_both_roles_first_v1':
        pool = pools.get(canary, [])
        codes = sorted(config.get('required_canary_codes', ['B2','B5','B6','G']),
                       key=lambda c: (len({r['source_group'] for r in pool if c in anomaly_codes(r['video_id'])}), c))
        # Reserve scarce category/source slots across BOTH roles before filling
        # bulk canary quotas. No outcome/model score participates in selection.
        for code in codes:
            for role, strata in quotas.items():
                target = strata.get(canary, 0)
                if counts[(role, canary)] >= target or any(
                    r['role']==role and r['stratum']==canary and code in anomaly_codes(r['video_id']) for r in selected):
                    continue
                take(role, canary, counts[(role, canary)]+1,
                     [r for r in pool if code in anomaly_codes(r['video_id'])])
        for role, strata in quotas.items():
            take(role, canary, strata.get(canary, 0), pool)
    for role, strata in quotas.items():
        for stratum, target in strata.items():
            if policy == 'legacy_role_first' or stratum != canary:
                take(role, stratum, target, pools[stratum])
            got = counts[(role,stratum)]
            if got != target: gaps.append({'role':role,'stratum':stratum,'requested':target,'available':got})
    return selected, {'quota_gaps': gaps, 'rejected': rejected, 'duplicate_candidates': duplicates,
                      'selected_n':len(selected),'counts':{role:{s:counts[(role,s)] for s in q} for role,q in quotas.items()},
                      'ready_for_protocol_freeze': not gaps and not history['issues'],
                      'history_requires_review': True, 'remote_execution_authorized':False}


def preflight(project, source, out, candidate_file=None, history_roots=None):
    project, out = Path(project).resolve(), Path(out).resolve()
    config = {'version':'hard_normal_same_event_trial_v1','seed':20260910,'quotas':QUOTAS,
              'window_frames':96,'evidence_frames':8,'primary':'T1_direct2_vs_T0_m0',
              'secondary':'B1_event_bound4_vs_B0_event_unbound3','max_event_calls_per_window':2,
              'hard_normal_risk_tolerance':None,'deployment_authorized':False}
    old = read_json(out/'protocol/proposed_config.json')
    if old and old != config: raise ValueError('configuration changed; use a new trial TAG')
    write_json(out/'protocol/proposed_config.json',config)
    roots = history_roots or [p/'runs' for p in project.parent.iterdir()
                              if p.is_dir() and (p/'runs').is_dir() and ('backup' not in p.name.lower() or p == project)]
    # The Part-A/Part-C backup-named pipeline is an actual historical experiment.
    for name in ('pipeline_backup_PartA_PartC_OK','pipeline_backup_PartA_PartC_OK_graph_vs_node_revision'):
        p=project.parent/name/'runs'
        if p.is_dir() and p not in roots: roots.append(p)
    history=history_inventory(roots,out/'history',excluded_root=out)
    candidates=list(iter_jsonl(candidate_file)) if candidate_file else []
    selected, report=enroll(candidates,history,config)
    report['candidate_input']=str(candidate_file) if candidate_file else None
    report['candidate_input_sha256']=file_sha256(candidate_file) if candidate_file else None
    report['status']='WAITING_FOR_CANDIDATE_SCREENING' if not candidate_file else 'WAITING_FOR_QUOTAS_OR_HISTORY_REVIEW'
    report['readiness_source']=str(source)
    write_jsonl(out/'enrollment/private_proposed_windows.jsonl',selected)
    write_json(out/'enrollment/preflight_report.json',report)
    write_json(out/'enrollment/candidate_schema.json',{
        'required':['dataset_partition','video_id','video_path','start_frame','end_frame_exclusive','stratum',
                    'label_source','positive_spans_half_open','positive_span_provenance','media_verified','evidence_sha256','perceptual_fingerprint'],
        'coordinates':'zero_based_half_open','perceptual_fingerprint':'eight 64-bit dHash hex strings; local frames only',
        'screening':'local motion or contract-compatible cached M0; no new VLM calls for screening',
        'negative_anchor_policy':'unverified; never a normal training target'})
    return report
