"""Versioned same-event evidence contracts; unknown is never a measured zero."""
from __future__ import annotations

import math

from .contracts import semantic_sha256


FRAMES = {f'T{i}' for i in range(8)}
RELATIONS = {'supported', 'contradicted', 'unknown'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probability(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, 'probability must be finite [0,1], not bool')
    return float(value)


def ids(rows, key, maximum):
    require(isinstance(rows, list) and len(rows) <= maximum, f'invalid {key} list/overflow')
    require(all(isinstance(r,dict) for r in rows), f'{key} entries must be objects')
    result = [r.get(key) for r in rows]
    require(all(isinstance(x, str) and x for x in result) and len(set(result)) == len(result), f'duplicate/missing {key}')
    return set(result)


def refs(values, allowed, name, nonempty=False):
    require(isinstance(values, list) and all(isinstance(v, str) for v in values), f'invalid {name}')
    require(len(set(values)) == len(values) and set(values) <= allowed and (bool(values) or not nonempty), f'unknown/duplicate/empty {name}')


def validate_proposal(c1, window_id=None, evidence_signature=None):
    require(c1.get('schema_version') == 'event_candidates_v1', 'wrong C1 schema')
    for name, value in [('window_id', window_id), ('evidence_signature', evidence_signature)]:
        require(isinstance(c1.get(name), str) and bool(c1[name]) and (value is None or c1[name] == value), f'C1 {name} mismatch')
    for flag in ('scan_complete', 'overflow', 'observation_sufficient'):
        require(type(c1.get(flag)) is bool, f'C1 {flag} must be boolean')
    entities = ids(c1.get('entities'), 'entity_id', 12)
    evidence = ids(c1.get('evidence'), 'evidence_id', 64)
    ids(c1.get('events'), 'event_id', 4)
    for e in c1['evidence']:
        refs(e.get('frame_ids'), FRAMES, 'evidence frames', True)
        require(bool(e.get('description')), 'evidence description missing')
    for e in c1['entities']:
        require(bool(e.get('kind')) and bool(e.get('visual_descriptor')), 'entity description missing')
        refs(e.get('observed_frame_ids'), FRAMES, 'entity frames', True)
    for e in c1['events']:
        refs(e.get('participant_ids'), entities, 'participant IDs', True)
        refs(e.get('observed_frame_ids'), FRAMES, 'event frames', True)
        refs(e.get('evidence_ids'), evidence, 'direct evidence IDs', True)
        frame_union={f for item in c1['evidence'] if item['evidence_id'] in e['evidence_ids'] for f in item['frame_ids']}
        require(set(e['observed_frame_ids']) <= frame_union, 'event frames lack referenced direct evidence')
        for field in ('direct_mechanism_probability', 'direct_evidence_quality', 'visibility'):
            probability(e.get(field))
        require(type(e.get('time_relation_unknown')) is bool and bool(e.get('observed_action')), 'event observation fields missing')
    probability(c1.get('uncertainty'))
    return c1


def validate_binding(c1, c2):
    validate_proposal(c1)
    require(c2.get('schema_version') == 'same_event_normal_binding_v1', 'wrong C2 schema')
    require(c2.get('proposal_sha256') == semantic_sha256(c1), 'C2 changed or did not reference frozen C1')
    for name in ('window_id', 'evidence_signature'):
        require(c2.get(name) == c1[name], f'C2 {name} mismatch')
    require(type(c2.get('complete')) is bool, 'C2 completeness missing')
    normal_ids = ids(c2.get('normal_evidence'), 'evidence_id', 64)
    for row in c2['normal_evidence']:
        refs(row.get('frame_ids'), FRAMES, 'normal evidence frames', True)
        require(bool(row.get('description')), 'normal evidence description missing')
    events = {e['event_id']: e for e in c1['events']}
    returned = ids(c2.get('event_bindings'), 'event_id', 4)
    require(returned == set(events), 'C2 must return each C1 event exactly once')
    refs(c2.get('unmapped_event_ids'), set(events), 'unmapped event IDs')
    probability(c2.get('uncertainty'))
    for row in c2['event_bindings']:
        event = events[row['event_id']]
        require(type(row.get('assessment_complete')) is bool, 'assessment completeness missing')
        ids(row.get('explanations'), 'explanation_id', 3)
        for ex in row['explanations']:
            probability(ex.get('bound_support_score'))
            require(ex.get('explanation_coverage') in {'full','partial','none','unknown'}, 'invalid coverage')
            for name in ('same_participant', 'same_time', 'same_event'):
                require(ex.get(name) in RELATIONS, f'invalid {name}')
            require(isinstance(ex.get('mechanism'), str) and bool(ex.get('type')) and bool(ex.get('reason')), 'normal mechanism fields missing')
            require(type(ex.get('visible_normal_mechanism')) is bool, 'visible mechanism boolean required')
            refs(ex.get('observed_frame_ids'), FRAMES, 'explanation frames')
            refs(ex.get('normal_evidence_ids'), normal_ids, 'normal evidence IDs')
            frame_union={f for item in c2['normal_evidence'] if item['evidence_id'] in ex['normal_evidence_ids'] for f in item['frame_ids']}
            require(set(ex['observed_frame_ids']) <= frame_union, 'normal explanation frames lack referenced evidence')
            refs(ex.get('unexplained_direct_evidence_ids'), set(event['evidence_ids']), 'residual direct evidence IDs')
            mapping = ex.get('participant_correspondence')
            require(isinstance(mapping, list), 'participant correspondence missing')
            mapped = ids(mapping, 'event_entity_id', 12)
            require(mapped <= set(event['participant_ids']) and all(bool(m.get('normal_role')) for m in mapping), 'cross-event participant mapping')
            if ex['same_participant'] == 'supported':
                require(mapped == set(event['participant_ids']), 'same-participant support must cover event participants')
            if ex['same_time'] == 'supported':
                require(bool(set(ex['observed_frame_ids']) & set(event['observed_frame_ids'])), 'supported time lacks shared observed frame')
    return c2


def binding_features(c1, c2):
    validate_binding(c1, c2)
    events = {e['event_id']: e for e in c1['events']}
    traces = []
    for row in c2['event_bindings']:
        event = events[row['event_id']]
        d = probability(event['direct_mechanism_probability']) * probability(event['direct_evidence_quality'])
        known = row['assessment_complete'] and row['event_id'] not in c2['unmapped_event_ids']
        eligible = []
        for ex in row['explanations']:
            relation = [ex[n] for n in ('same_participant','same_time','same_event')]
            if 'contradicted' in relation or ex['explanation_coverage'] == 'none':
                continue
            if 'unknown' in relation or ex['explanation_coverage'] in ('unknown','partial') or ex['unexplained_direct_evidence_ids']:
                known = False; continue
            if (all(v == 'supported' for v in relation) and ex['visible_normal_mechanism'] and
                    ex['mechanism'].strip() and ex['normal_evidence_ids'] and ex['observed_frame_ids']):
                eligible.append(ex['bound_support_score'])
        bound = max(eligible, default=0.) if known else None
        traces.append({'event_id': row['event_id'], 'direct': d, 'bound': bound, 'known': known,
                       'residual': d*(1-bound) if known else None,
                       'residual_interval_audit_only': [d*(1-bound), d*(1-bound)] if known else [0., d]})
    complete = c1['scan_complete'] and not c1['overflow'] and c1['observation_sufficient'] and c2['complete']
    q = max((r['direct'] for r in traces), default=0.) if complete else None
    u = max((r['residual'] for r in traces), default=0.) if complete and all(r['known'] for r in traces) else None
    return {'schema_version': 'event_binding_evidence_v1', 'window_id': c1['window_id'],
            'proposal_sha256': semantic_sha256(c1), 'binding_sha256': semantic_sha256(c2),
            'Q': q, 'U': u, 'Q_observed': q is not None, 'U_observed': u is not None, 'events': traces,
            'core8_substitution_forbidden': True}


def proposal_prompt(window_id, evidence_signature):
    return f'''Inspect only these eight chronological frames T0..T7. No filename, label,
graph score, benchmark answer or event category is supplied. Find up to four directly
observed suspicious physical events, not arbitrary co-occurring objects. Assign stable
entity and evidence IDs. Never invent temporal continuity from unobserved frames.
Return JSON with schema_version=event_candidates_v1, window_id={window_id},
evidence_signature={evidence_signature}, scan_complete (bool), overflow (bool),
observation_sufficient (bool), uncertainty [0,1], entities (max12):
[{{entity_id,kind,visual_descriptor,observed_frame_ids:[T0..T7]}}],
evidence:[{{evidence_id,frame_ids,description}}], events (max4):
[{{event_id,participant_ids,observed_frame_ids,observed_action,evidence_ids,
direct_mechanism_probability:[0,1],direct_evidence_quality:[0,1],visibility:[0,1],
time_relation_unknown:bool}}]. Empty events is allowed; insufficient visibility is
not proof of no event. Set overflow if additional plausible events cannot be listed.'''


def binding_prompt(c1):
    import json
    return '''Use the SAME eight frames. The following C1 record is frozen; do not
change its events, participants or direct probabilities. For each event, look for a
visible normal mechanism explaining THAT event, not a normal scene elsewhere.
Unknown relations remain unknown. Return all events exactly once, including no
applicable normal explanation. Return JSON: schema_version=same_event_normal_binding_v1,
window_id and evidence_signature copied from C1, proposal_sha256=''' + semantic_sha256(c1) + ''',
complete:bool, uncertainty:[0,1], unmapped_event_ids:[],
normal_evidence:[{evidence_id,frame_ids:[T0..T7],description}], event_bindings:[{
event_id, assessment_complete:bool, explanations (max3):[{explanation_id,mechanism,type,
visible_normal_mechanism:bool,participant_correspondence:[{event_entity_id,normal_role}],
observed_frame_ids,normal_evidence_ids,same_participant:supported|contradicted|unknown,
same_time:supported|contradicted|unknown,same_event:supported|contradicted|unknown,
explanation_coverage:full|partial|none|unknown,bound_support_score:[0,1],
unexplained_direct_evidence_ids,reason}]}]. A full explanation needs evidence for all
participants in the same event/time and no unexplained direct evidence.
Frozen C1:\n''' + json.dumps(c1, ensure_ascii=False)
