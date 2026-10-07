"""Fixed-evidence development experiment, isolated from locked evaluation."""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from .contracts import WindowKey, file_sha256, iter_jsonl, read_json
from .role_scoped import assert_bound, atomic_json, portable, valid_screen

VERSION = "v93_b5_mechanism_development_v1"
STATES = {"observed", "not_observed", "uncertain"}
PHASES = {"active", "ongoing_constraint", "residual_only", "no_b5", "unclear"}
PROMPT = '''Inspect only the eight chronological silent images T0..T7.
Assess visible B5 abuse: sustained one-sided coercion, restraint, assault on a
defenseless person, or prevention of escape. Other anomalies are not normal.
Distinguish an active mechanism or ongoing constraint from residual injury.
Blood, a prone person, a weapon, distress or darkness alone is not proof of
current abuse. Immobility alone is not proof of death. Do not guess liquid type.
Different actors/shots can belong to one incident. Keep adult, child, victim,
attacker and intervenor identities distinct. Bind actors to targets within each
subevent; separately explain links BETWEEN subevents. A changed actor is not
by itself evidence of a changed incident. Intent or causality requiring unseen
context must be marked needs_context, not asserted from narrative familiarity.
Do not use movie recognition, inferred plot, filenames or unseen audio as proof.
Bins are sparse observations, NOT continuous event intervals. Report uncertainty
instead of forcing unclear or aftermath-only evidence into a normal label.
Return one JSON object, no markdown, with exactly these top-level keys:
{
 "b5_presence":"yes|no|uncertain", "b5_probability":0.5,
 "phase":"active|ongoing_constraint|residual_only|no_b5|unclear",
 "observation_quality":"good|limited|insufficient", "context_needed":true,
 "actors":[{"id":"p1","description":"visible identity/role","bins":[0]}],
 "subevents":[{"id":"e1","actor_ids":["p1"],"target_ids":["p2"],
   "bins":[0],"phase":"active","binding":"observed",
   "mechanism":"observed","evidence_scope":"current_frames",
   "b5_support":0.8,"evidence":"concrete visible mechanism and binding"}],
 "event_links":[{"source":"e1","target":"e2",
   "kind":"intervention|continuation|shared_incident|unclear",
   "relation_state":"observed|not_observed|uncertain",
   "evidence_scope":"current_frames|needs_context","bins":[0],
   "evidence":"what supports this link and what cannot be observed"}],
 "per_frame":[{"bin":0,"b5_support":"yes|no|uncertain",
   "injury_trace":"yes|no|uncertain","evidence":"visible observation"}],
 "other_event":{"category":"none|B1|B2|B4|B6|G|uncertain",
   "evidence":"alternative event or no supported alternative"},
 "uncertainties":["missing visual/context evidence"]
}
Use actual enum values, not the pipe-separated alternatives. actors: at most 8;
subevents: at most 4; event_links: at most 4; uncertainties: at most 5.
per_frame MUST contain exactly 8 entries in order with integer bin 0..7.
IDs must be unique. References must exist. Distinct people need distinct IDs;
use empty actor/target lists if identity is unobservable, with uncertain binding.
Every actor/subevent needs nonempty sorted unique integer bins 0..7. Link bins
may be empty when the relation requires unseen context. A subevent bin must
include its referenced actor/target observations. Use only directly observed
current_frames links; never fill unseen chronological gaps from sparse bins.
mechanism and binding use observed/not_observed/uncertain. Probabilities are
finite numbers in [0,1]. Empty subevents/links are allowed. Keep evidence brief.
'''


def exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"{name}: unexpected or missing fields")


def choice(value, allowed, name):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name}: invalid enum")


def text_field(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2400:
        raise ValueError("evidence must be a nonempty short string")


def probability(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite numeric [0,1], not Boolean")


def bins(value, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError("missing bins")
    if any(type(x) is not int or not 0 <= x < 8 for x in value) or value != sorted(set(value)):
        raise ValueError("bins must be sorted unique integers 0..7")


def bounded_list(value, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("invalid list or too many entries")


def validate(value):
    exact(value, "b5_presence b5_probability phase observation_quality context_needed actors subevents event_links per_frame other_event uncertainties", "response")
    choice(value['b5_presence'], {'yes', 'no', 'uncertain'}, 'b5_presence')
    probability(value['b5_probability'])
    choice(value['phase'], PHASES, 'phase')
    choice(value['observation_quality'], {'good', 'limited', 'insufficient'}, 'quality')
    if type(value['context_needed']) is not bool:
        raise ValueError("context_needed must be Boolean")
    actors = {}
    bounded_list(value['actors'], 8)
    for actor in value['actors']:
        exact(actor, 'id description bins', 'actor')
        ident = actor['id']
        if not isinstance(ident, str) or not re.fullmatch(r'p[1-9][0-9]*', ident) or ident in actors:
            raise ValueError("invalid or duplicate actor ID")
        text_field(actor['description']); bins(actor['bins'])
        actors[ident] = actor
    events = {}
    bounded_list(value['subevents'], 4)
    for event in value['subevents']:
        exact(event, 'id actor_ids target_ids bins phase binding mechanism evidence_scope b5_support evidence', 'subevent')
        ident = event['id']
        if not isinstance(ident, str) or not re.fullmatch(r'e[1-9][0-9]*', ident) or ident in events:
            raise ValueError("invalid or duplicate subevent ID")
        bins(event['bins']); probability(event['b5_support']); text_field(event['evidence'])
        choice(event['phase'], PHASES, 'subevent phase')
        choice(event['binding'], STATES, 'binding'); choice(event['mechanism'], STATES, 'mechanism')
        choice(event['evidence_scope'], {'current_frames', 'needs_context'}, 'scope')
        for key in ('actor_ids', 'target_ids'):
            refs = event[key]
            if not isinstance(refs, list) or any(not isinstance(x, str) or x not in actors for x in refs) or len(set(refs)) != len(refs):
                raise ValueError("invalid actor reference")
            if any(not (set(event['bins']) & set(actors[x]['bins'])) for x in refs):
                raise ValueError("actor reference has no observation in subevent bins")
        if set(event['actor_ids']) & set(event['target_ids']):
            raise ValueError("B5 subevent cannot conflate actor and target identities")
        if event['binding'] == 'observed' and not (event['actor_ids'] and event['target_ids']):
            raise ValueError("observed binding needs an actor and target")
        events[ident] = event
    bounded_list(value['event_links'], 4)
    seen_links = set()
    for link in value['event_links']:
        exact(link, 'source target kind relation_state evidence_scope bins evidence', 'link')
        if any(not isinstance(link[k], str) or link[k] not in events for k in ('source', 'target')) or link['source'] == link['target']:
            raise ValueError("invalid subevent reference")
        choice(link['kind'], {'intervention', 'continuation', 'shared_incident', 'unclear'}, 'link kind')
        pair = (link['source'], link['target'], link['kind'])
        if pair in seen_links: raise ValueError("duplicate event link")
        seen_links.add(pair)
        choice(link['relation_state'], STATES, 'link state')
        choice(link['evidence_scope'], {'current_frames', 'needs_context'}, 'link scope')
        bins(link['bins'], empty=True); text_field(link['evidence'])
        if not set(link['bins']) <= set(events[link['source']]['bins'] + events[link['target']]['bins']):
            raise ValueError("link bins outside its subevents")
        if link['relation_state'] == 'observed' and (link['evidence_scope'] != 'current_frames' or not link['bins']):
            raise ValueError("observed relation requires current-frame support")
    frames = value['per_frame']
    bounded_list(frames, 8)
    if len(frames) != 8: raise ValueError("exactly eight frame records required")
    for i, frame in enumerate(frames):
        exact(frame, 'bin b5_support injury_trace evidence', 'frame')
        if type(frame['bin']) is not int or frame['bin'] != i: raise ValueError("frame order must be 0..7")
        for key in ('b5_support', 'injury_trace'): choice(frame[key], {'yes', 'no', 'uncertain'}, key)
        text_field(frame['evidence'])
    exact(value['other_event'], 'category evidence', 'other_event')
    choice(value['other_event']['category'], {'none', 'B1', 'B2', 'B4', 'B6', 'G', 'uncertain'}, 'other category')
    text_field(value['other_event']['evidence'])
    bounded_list(value['uncertainties'], 5)
    for item in value['uncertainties']: text_field(item)
    return value


def evidence_state(value):
    """Ternary diagnostic, not an anomaly-score override or trained decision rule."""
    validate(value)
    supported = [e['id'] for e in value['subevents'] if e['phase'] in {'active', 'ongoing_constraint'}
                 and e['mechanism'] == 'observed' and e['binding'] == 'observed'
                 and e['evidence_scope'] == 'current_frames' and e['b5_support'] >= .5]
    reasons = []
    if value['phase'] == 'residual_only': reasons.append('residual_not_current_mechanism')
    if value['context_needed']: reasons.append('additional_context_requested')
    if value['observation_quality'] == 'insufficient': reasons.append('insufficient_visual_evidence')
    if value['b5_presence'] == 'yes' and not supported: reasons.append('positive_without_supported_subevent')
    if value['b5_presence'] == 'no' and supported: reasons.append('negative_with_supported_subevent')
    if value['b5_presence'] == 'no' and any(f['b5_support'] == 'yes' for f in value['per_frame']):
        reasons.append('negative_with_supporting_frame')
    if value['b5_presence'] == 'yes' and value['b5_probability'] < .5: reasons.append('categorical_probability_disagreement')
    if value['b5_presence'] == 'no' and value['b5_probability'] >= .5: reasons.append('categorical_probability_disagreement')
    if not reasons and value['b5_presence'] == 'yes' and value['phase'] in {'active', 'ongoing_constraint'} and supported:
        state = 'supported_b5'
    elif not reasons and value['b5_presence'] == 'no' and value['phase'] == 'no_b5':
        state = 'no_b5_observed_not_binary_normal'
    else:
        state = 'unresolved'
    return {'state': state, 'supported_subevents': supported, 'reasons': reasons,
            'score_override': False, 'binary_anomaly_target': None}


def mock_response():
    return {'b5_presence': 'uncertain', 'b5_probability': .5, 'phase': 'unclear',
            'observation_quality': 'insufficient', 'context_needed': True, 'actors': [],
            'subevents': [], 'event_links': [], 'per_frame': [
                {'bin': i, 'b5_support': 'uncertain', 'injury_trace': 'uncertain', 'evidence': 'MOCK: no visual judgment'} for i in range(8)],
            'other_event': {'category': 'uncertain', 'evidence': 'MOCK'}, 'uncertainties': ['MOCK only']}


def code_files(project, code_dir):
    paths = [project / 'tools/event_decision' / name for name in (
        'development_mechanism_v93.py', 'development_mechanism_v93_runner.py',
        'development_mechanism_v93_report.py', 'contracts.py', 'role_scoped.py',
        'b5_development_screen.py', 'hard_trial.py', 'reenrollment.py')]
    return paths + [project / 'tools/development_mechanism_v93_cli.py',
                    project / 'run_development_mechanism_v93.sh',
                    code_dir / 'src/structural_vlm_binary/vlm/dashscope_backend.py']


def prepare(project, source, out, feedback, history, *, mock=False, model=None, max_tokens=4096):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists(): raise ValueError('prepare needs a new TAG; use run/report to resume')
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('new run must be separate from original run')
    original = assert_bound(source)
    if original['mock']: raise ValueError('source must be a real previously completed V9.2 run')
    if type(max_tokens) is not int or not 512 <= max_tokens <= 8192:
        raise ValueError('max_tokens must be 512..8192')
    rows = list(iter_jsonl(source / 'inventory/development_windows.jsonl'))
    feedback_hash = file_sha256(feedback)
    history_hash = file_sha256(history)
    feedback_rows = list(iter_jsonl(feedback))
    by_uid = {r['window_uid']: r for r in rows}
    if len(rows) != len(by_uid): raise ValueError('duplicate source windows')
    reviewed = {}
    for r in feedback_rows:
        uid = r['window_uid']
        if uid not in by_uid or uid in reviewed: raise ValueError('unknown/duplicate feedback window')
        if r.get('role_scope') != 'adaptation_development_only' or r.get('gold_metric_mask') is not False:
            raise ValueError('feedback must be non-gold development only')
        text_field(r.get('human_note_verbatim'))
        interpretation = r.get('normalized_interpretation', {})
        text_field(interpretation.get('assessment'))
        target = interpretation.get('proposed_development_b5_target')
        if target is not None and (type(target) is not int or target not in (0, 1)):
            raise ValueError('development interpretation must be null/0/1, not a gold target')
        if r['source_result_sha256'] != file_sha256(source / 'development/results' / (uid + '.json')):
            raise ValueError('feedback references different model output')
        reviewed[uid] = r
    history_data = read_json(history)
    if history_data.get('ready_for_locked_acquisition') is not False:
        raise ValueError('this entry is development only, not a locked acquisition entry')
    development = set(original['development_source_groups'])
    reserved = set(original['reserved_source_groups'])
    planned = []
    for row in rows:
        uid = row['window_uid']
        if row.get('dataset_partition') != 'train' or row['allowed_roles'] != ['adaptation'] or row['source_group'] not in development or row['source_group'] in reserved:
            raise ValueError('source role firewall rejected input')
        if WindowKey('train', row['video_id'], row['start_frame'], row['end_frame_exclusive']).uid != uid:
            raise ValueError('source window identity mismatch')
        if row['end_frame_exclusive'] - row['start_frame'] != 96 or row['sampled_frame_indices'] != [row['start_frame'] + int(i * 95 / 7) for i in range(8)]:
            raise ValueError('source must have exact 96-frame/eight-image evidence')
        old = read_json(source / 'development/results' / (uid + '.json'))
        if not old or old.get('window_uid') != uid or old.get('protocol_sha256') != file_sha256(source / 'protocol.json'):
            raise ValueError('missing or unbound old result')
        if old.get('mock') is not False or old.get('role') != 'adaptation' or old.get('model') != original['model']:
            raise ValueError('baseline is not a real matching adaptation result')
        exclusion = None
        if old['status'] == 'success': valid_screen(old['parsed'])
        else:
            receipts = [read_json(p) for p in (source / 'development/attempts' / uid).glob('*.json')]
            # Only actual provider error envelopes justify a policy exclusion.
            envelopes = []
            for receipt in receipts:
                try: envelopes.append(json.loads(receipt.get('raw', '{}')))
                except (ValueError, TypeError): pass
            if any(isinstance(v, dict) and v.get('code') == 'DataInspectionFailed' for v in envelopes):
                exclusion = 'prior_provider_policy_rejection_do_not_retry'
            else:
                raise ValueError('source has unresolved non-policy failure; preserve/recover it first: ' + uid)
        folder = source / 'development/cases' / uid[:20]
        media = read_json(folder / 'media.json')
        expected_names = {'clip.mp4'} | {f'T{i}.jpg' for i in range(8)}
        if not media or media.get('decoded_frames') != 96 or set(media.get('files', {})) != expected_names:
            raise ValueError('incomplete cached media')
        for name, digest in media['files'].items():
            if file_sha256(folder / name) != digest: raise ValueError('source media changed')
            if name != 'clip.mp4' and old.get('image_sha256', {}).get(name) != digest:
                raise ValueError('old inference/image mismatch')
        planned.append({**row, 'role': 'development', 'baseline': old,
                        'media_hashes': media['files'], 'exclusion': exclusion,
                        'design_feedback': reviewed.get(uid), 'gold_metric_mask': False})
    order = {uid: i for i, uid in enumerate(reviewed)}
    planned.sort(key=lambda r: (order.get(r['window_uid'], len(order)), r['video_id'], r['start_frame']))
    out.mkdir(parents=True)
    for row in planned:
        folder = out / 'cases' / row['window_uid'][:20]
        folder.mkdir(parents=True)
        for name in row['media_hashes']:
            shutil.copyfile(source / 'development/cases' / row['window_uid'][:20] / name, folder / name)
            if file_sha256(folder / name) != row['media_hashes'][name]:
                raise ValueError('source media changed during snapshot; preserve incomplete run')
        atomic_json(folder / 'baseline.json', row['baseline'])
    for src, name in ((feedback, 'feedback_input.jsonl'), (history, 'history_scope_input.json')):
        shutil.copyfile(src, out / name)
    if file_sha256(out / 'feedback_input.jsonl') != feedback_hash or file_sha256(out / 'history_scope_input.json') != history_hash:
        raise ValueError('feedback/history changed during preparation; preserve incomplete run')
    atomic_json(out / 'windows.json', planned)
    code_dir = portable(original['code_dir'])
    protocol = {'version': VERSION, 'mock': bool(mock), 'role': 'development_only',
                'source_run': str(source), 'source_protocol_sha256': file_sha256(source / 'protocol.json'),
                'model': model or original['model'], 'max_tokens': max_tokens, 'temperature': 0,
                'prompt': PROMPT, 'code_dir': str(code_dir), 'development_sources': sorted(development),
                'reserved_sources': sorted(reserved), 'locked_authorized': False,
                'experiment': 'same_images_different_prompts_not_an_OT_or_node_ablation',
                'code_hashes': {str(p.resolve()): file_sha256(p) for p in code_files(project, code_dir)},
                'data_hashes': {str(p.relative_to(out)): file_sha256(p) for p in out.rglob('*') if p.is_file()}}
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'integrity.json', {'protocol_sha256': file_sha256(out / 'protocol.json')})
    return {'status': 'PREPARED_DEVELOPMENT_ONLY', 'windows': len(planned),
            'eligible_calls': sum(not r['exclusion'] for r in planned),
            'policy_exclusions': sum(bool(r['exclusion']) for r in planned),
            'design_feedback_cases': len(reviewed), 'mock': bool(mock), 'remote_calls': 0}


def load_run(out):
    out = Path(out)
    protocol = read_json(out / 'protocol.json')
    if not protocol or protocol.get('version') != VERSION or protocol.get('locked_authorized') is not False or protocol.get('role') != 'development_only':
        raise ValueError('not an authorized V9.3 development run')
    if read_json(out / 'integrity.json', {}).get('protocol_sha256') != file_sha256(out / 'protocol.json'):
        raise ValueError('protocol changed; use a new TAG')
    for path, digest in protocol['code_hashes'].items():
        if file_sha256(portable(path)) != digest: raise ValueError('bound code changed: ' + path)
    for rel, digest in protocol['data_hashes'].items():
        target = (out / rel).resolve()
        if out.resolve() not in target.parents or file_sha256(target) != digest:
            raise ValueError('bound snapshot/media changed: ' + rel)
    rows = read_json(out / 'windows.json')
    for r in rows:
        if r['role'] != 'development' or r['source_group'] not in protocol['development_sources'] or r['source_group'] in protocol['reserved_sources']:
            raise ValueError('development firewall rejected stored window')
    return protocol, rows
