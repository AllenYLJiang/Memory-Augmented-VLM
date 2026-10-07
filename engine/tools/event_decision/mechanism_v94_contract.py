"""Field-scoped observation contract. Validation is not visual verification."""
import json
import math
from collections import Counter

VERSION = 'v94_scoped_mechanism_1'
CATEGORIES = {'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot', 'B5': 'Abuse',
              'B6': 'Car accident', 'G': 'Explosion'}
STATES = {'observed', 'not_observed', 'unknown'}
EVENT_TYPES = {'interpersonal_action', 'object_constraint', 'injury_trace', 'other_event', 'unknown'}
PHASES = {'active', 'ongoing_constraint', 'residual', 'other', 'unknown'}
FEATURES = ['b5_probability', 'interpersonal_observation', 'object_constraint_observation',
            'injury_observation', 'local_person_binding', 'local_object_binding',
            'local_b5_supported', 'cross_event_link_support', 'local_context_needed',
            'cross_event_context_needed', 'story_context_needed', 'frame_action_fraction',
            'frame_constraint_fraction', 'frame_injury_fraction']
RULES = {
    'b5_boundary': 'B5 needs unilateral coercion/credible person-directed threat or assault on a defenseless person. '
        'One contact in a reciprocal fight is not sufficient. Defensive resistance does not by itself make abuse mutual.',
    'vehicle_boundary': 'Pursuit, interception and collision are not automatically B5 prevention of escape. '
        'Record the vehicle event (B6) separately; require distinct person-directed abuse evidence for B5.',
    'constraint_boundary': 'A visible person-object attachment may support ongoing constraint without a visible perpetrator. '
        'Constraint alone is not automatically B5: medical/supportive restraint and unknown purpose remain alternatives.',
    'injury_boundary': 'Injury traces are not an observed current assault. No visible attacker is not proof of no ongoing constraint.',
    'context_boundary': 'Local mechanism, local binding, category context, cross-event continuity and global plot are separate. '
        'Unknown links do not erase supported local observations. Do not invent identities, death, intent, or causal continuity.',
    'evidence_boundary': 'Eight silent samples are not continuous intervals. No filename, movie knowledge, old model answer or review is evidence.',
}


def example():
    return {
        'schema_version': VERSION,
        'b5': {'label': 'unknown', 'probability': None, 'basis_event_ids': [], 'evidence': 'Purpose of constraint is unobservable.'},
        'persons': [{'id': 'p1', 'description': 'Visible wrist and hand of a person', 'bins': [0]}],
        'objects': [{'id': 'o1', 'kind': 'restraint', 'description': 'A strap visibly attached around the wrist', 'bins': [0]}],
        'events': [{'id': 'e1', 'type': 'object_constraint', 'bins': [0], 'actor_ids': [], 'target_ids': ['p1'],
                    'object_ids': ['o1'], 'phase': 'ongoing_constraint', 'observation': 'observed',
                    'binding': {'state': 'observed', 'mode': 'same_frame', 'evidence': 'Strap surrounds the visible wrist in T0.'},
                    'boundary': {'b5_state': 'unknown', 'alternative_classes': [], 'basis': 'object_restraint_only',
                                 'evidence': 'Visible attachment; coercive purpose is not established.'},
                    'context_scope': {'local_mechanism_required': False, 'local_binding_required': False, 'category_required': True},
                    'evidence': 'Current attachment, with no visible perpetrator.'}],
        'links': [],
        'frames': [{'bin': i, 'action': 'unknown', 'constraint': 'yes' if i == 0 else 'unknown',
                    'injury': 'unknown', 'quality': 'limited', 'evidence': 'Attachment visible.' if i == 0 else 'Unclear sample.'} for i in range(8)],
        'context': {'cross_event_required': False, 'global_story_required': True, 'evidence': 'Purpose is outside these samples.'},
    }


PROMPT = '\n'.join([
    'Inspect ONLY the eight chronological silent images T0..T7. Return JSON only.',
    'Category dictionary: ' + json.dumps(CATEGORIES), *RULES.values(),
    'Do not turn unknown into normal. B5=no is not binary normal. Report multi-class alternatives using the dictionary; [] means no committed alternative, not normal.',
    'Use these exact fields; all lists may be empty except frames, which must contain exactly bins 0..7 in order.',
    'Maximum 8 persons, 4 objects, 4 events, 4 links. Keep every evidence string brief.',
    'A fully valid example (illustrative ONLY, not a description of the supplied images):', json.dumps(example()),
    'Allowed: b5.label=yes/no/unknown; probability=null for unknown or a finite number 0..1. basis_event_ids reference existing events.',
    'Person IDs p1,p2,...; object IDs o1,o2,...; event IDs e1,e2,...; unique within each list. All observed entities/events have nonempty sorted unique bins 0..7.',
    'objects.kind=restraint/container/support/weapon/vehicle/other/unknown.',
    'events.type=interpersonal_action/object_constraint/injury_trace/other_event/unknown; phase=active/ongoing_constraint/residual/other/unknown.',
    'observation and binding.state=observed/not_observed/unknown. binding.mode=same_frame/tracked_cross_frame/inferred_edit/unknown.',
    'Observed interpersonal binding requires distinct visible actor and target references. Observed object constraint binding needs a target person and an object, but actor_ids may be empty.',
    'Every referenced entity has observations inside its event bins. same_frame requires all referenced entities co-visible in at least one such bin.',
    'inferred_edit is an inference, not observed binding: use binding.state=unknown. Never fabricate off-screen actor IDs.',
    'boundary.b5_state=supported/not_supported/unknown; basis=unilateral_control/defenseless_assault/credible_person_threat/object_restraint_only/mutual_combat/vehicle_event/injury_only/other/unknown.',
    'object_restraint_only, mutual_combat, vehicle_event and injury_only cannot by themselves yield b5_state=supported.',
    'context_scope booleans concern ONLY that event. Uncertain links/story must not set local context booleans automatically.',
    'links entries: {"source":"e1","target":"e2","relation":"shared_incident","state":"unknown","scope":"needs_context","bins":[],"evidence":"Unconfirmed continuity"}.',
    'Only include links whose distinct source and target events exist. relation=shared_incident/continuation/intervention/unknown; state=observed/not_observed/unknown; scope=current_frames/needs_context.',
    'Observed links need current_frames support and nonempty bins within their two events. Sparse chronology alone does not establish causation.',
    'frames action/constraint/injury=yes/no/unknown; quality=good/limited/insufficient. Local frame observations, not filled from later plot.',
    'The b5 top-level judgment must be consistent with its basis events and category boundary. Keep raw uncertainty; do not force yes to make a narrative coherent.',
])


def assess(raw):
    """Keep each original field, mask invalid components, never repair identities."""
    fields, issues = {}, []
    def check(path, value, valid, reason='invalid field'):
        valid = bool(valid)
        fields[path] = {'valid': valid, 'value': value if valid else None}
        if not valid: issues.append({'path': path, 'reason': reason})
        return valid
    def enum(path, v, allowed): return check(path, v, isinstance(v, str) and v in allowed, 'invalid enum')
    def txt(v): return isinstance(v, str) and 0 < len(v.strip()) <= 2000
    def valid_bins(v):
        return isinstance(v, list) and bool(v) and all(type(x) is int and 0 <= x < 8 for x in v) and v == sorted(set(v))
    def number(v): return type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
    if not isinstance(raw, dict):
        return {'core_valid': False, 'fields': {}, 'issues': [{'path': '/', 'reason': 'root is not an object'}],
                'events': [], 'links': [], 'local_supported_events': [], 'diagnostic': 'unknown', 'schema_version': VERSION}
    check('/schema_version', raw.get('schema_version'), raw.get('schema_version') == VERSION, 'wrong schema version')
    def listing(name, limit):
        v = raw.get(name)
        ok = check('/' + name, None, isinstance(v, list) and len(v) <= limit, 'missing or oversized list')
        return v if ok else []
    people, objects, events, links, frames = (listing(n, limit) for n, limit in [('persons',8),('objects',4),('events',4),('links',4),('frames',8)])
    entities = {}
    for name, values, prefix in [('persons', people, 'p'), ('objects', objects, 'o')]:
        ids = Counter(v.get('id') for v in values if isinstance(v, dict) and isinstance(v.get('id'), str))
        for i, value in enumerate(values):
            path = f'/{name}/{i}'; v = value if isinstance(value, dict) else {}
            ident = v.get('id'); id_ok = isinstance(ident, str) and ident.startswith(prefix) and ident[1:].isdigit() and not ident[1:].startswith('0') and ids[ident] == 1
            check(path+'/id', ident, id_ok, 'invalid or duplicate ID')
            b_ok = check(path+'/bins', v.get('bins'), valid_bins(v.get('bins')), 'invalid observation bins')
            check(path+'/description', v.get('description'), txt(v.get('description')))
            if name == 'objects': enum(path+'/kind', v.get('kind'), {'restraint','container','support','weapon','vehicle','other','unknown'})
            if id_ok and b_ok: entities[ident] = v
    event_ids = Counter(v.get('id') for v in events if isinstance(v, dict) and isinstance(v.get('id'), str))
    event_map, event_checks = {}, []
    for i, value in enumerate(events):
        path = f'/events/{i}'; v = value if isinstance(value, dict) else {}; ident = v.get('id')
        id_ok = isinstance(ident, str) and ident.startswith('e') and ident[1:].isdigit() and not ident[1:].startswith('0') and event_ids[ident] == 1
        check(path+'/id', ident, id_ok, 'invalid or duplicate event ID')
        b_ok = check(path+'/bins', v.get('bins'), valid_bins(v.get('bins')), 'invalid event bins')
        type_ok = enum(path+'/type', v.get('type'), EVENT_TYPES)
        phase_ok = enum(path+'/phase', v.get('phase'), PHASES)
        obs_ok = enum(path+'/observation', v.get('observation'), STATES)
        evidence_ok = check(path+'/evidence', v.get('evidence'), txt(v.get('evidence')))
        refs_ok, refs = True, []
        for key, prefix in [('actor_ids','p'),('target_ids','p'),('object_ids','o')]:
            values = v.get(key)
            ok = isinstance(values, list) and all(isinstance(x,str) and x.startswith(prefix) and x in entities for x in values)
            ok = ok and len(values) == len(set(values)) and b_ok and all(set(entities[x]['bins']) & set(v['bins']) for x in values)
            check(path+'/'+key, values, ok, 'unknown/repeated/unobserved entity reference')
            refs_ok &= ok
            if ok: refs.extend(values)
        distinct = refs_ok and not (set(v['actor_ids']) & set(v['target_ids']))
        binding = v.get('binding') if isinstance(v.get('binding'), dict) else {}
        bind_ok = enum(path+'/binding/state', binding.get('state'), STATES)
        bind_ok &= enum(path+'/binding/mode', binding.get('mode'), {'same_frame','tracked_cross_frame','inferred_edit','unknown'})
        bind_ok &= check(path+'/binding/evidence', binding.get('evidence'), txt(binding.get('evidence')))
        if binding.get('state') == 'observed':
            needed = distinct
            if v.get('type') == 'interpersonal_action': needed &= bool(v.get('actor_ids')) and bool(v.get('target_ids'))
            elif v.get('type') == 'object_constraint': needed &= bool(v.get('target_ids')) and bool(v.get('object_ids'))
            else: needed = False
            needed &= binding.get('mode') in {'same_frame','tracked_cross_frame'}
            if needed and binding['mode'] == 'same_frame':
                common = set(v['bins'])
                for ref in refs: common &= set(entities[ref]['bins'])
                needed = bool(common)
            bind_ok &= check(path+'/binding/structural_support', True, needed, 'binding lacks typed entity/temporal support')
        else:
            bind_ok &= check(path+'/binding/structural_support', True, distinct, 'inconsistent entity references')
        context = v.get('context_scope') if isinstance(v.get('context_scope'), dict) else {}
        ctx_ok = True
        for key in ('local_mechanism_required','local_binding_required','category_required'):
            ctx_ok &= check(path+'/context_scope/'+key, context.get(key), type(context.get(key)) is bool)
        boundary = v.get('boundary') if isinstance(v.get('boundary'), dict) else {}
        cat_ok = enum(path+'/boundary/b5_state', boundary.get('b5_state'), {'supported','not_supported','unknown'})
        cat_ok &= enum(path+'/boundary/basis', boundary.get('basis'), {'unilateral_control','defenseless_assault','credible_person_threat','object_restraint_only','mutual_combat','vehicle_event','injury_only','other','unknown'})
        classes = boundary.get('alternative_classes')
        cat_ok &= check(path+'/boundary/alternative_classes', classes, isinstance(classes,list) and all(isinstance(x,str) and x in CATEGORIES for x in classes) and len(classes)==len(set(classes)), 'unknown/repeated category code')
        cat_ok &= check(path+'/boundary/evidence', boundary.get('evidence'), txt(boundary.get('evidence')))
        if boundary.get('b5_state') == 'supported':
            cat_ok &= check(path+'/boundary/specific_mechanism', True,
                            v.get('type') in {'interpersonal_action','object_constraint'} and boundary.get('basis') in {'unilateral_control','defenseless_assault','credible_person_threat'}, 'generic injury/vehicle/constraint is not B5 evidence')
        local_ok = bool(id_ok and b_ok and type_ok and phase_ok and obs_ok and evidence_ok and refs_ok and bind_ok and cat_ok and ctx_ok)
        supported = local_ok and v.get('observation')=='observed' and binding.get('state')=='observed' and boundary.get('b5_state')=='supported' and v.get('phase') in {'active','ongoing_constraint'} and not any(context.values())
        event_checks.append({'id': ident if id_ok else None, 'index': i, 'valid': local_ok,
                             'observation_valid': bool(id_ok and b_ok and type_ok and obs_ok and evidence_ok),
                             'binding_valid': bool(id_ok and b_ok and refs_ok and bind_ok),
                             'category_valid': bool(id_ok and cat_ok), 'context_valid': bool(ctx_ok), 'supported': bool(supported)})
        if id_ok and b_ok: event_map[ident] = v
    link_checks = []
    for i, value in enumerate(links):
        path=f'/links/{i}'; v=value if isinstance(value,dict) else {}
        ok=check(path+'/references', None, isinstance(v.get('source'),str) and isinstance(v.get('target'),str) and v['source'] in event_map and v['target'] in event_map and v['source']!=v['target'], 'unknown or self event link')
        ok &= enum(path+'/relation',v.get('relation'),{'shared_incident','continuation','intervention','unknown'})
        ok &= enum(path+'/state',v.get('state'),STATES)
        ok &= enum(path+'/scope',v.get('scope'),{'current_frames','needs_context'})
        bs=v.get('bins'); bins_ok = bs==[] or valid_bins(bs)
        if ok and bins_ok: bins_ok=set(bs)<=set(event_map[v['source']]['bins']+event_map[v['target']]['bins'])
        ok &= check(path+'/bins',bs,bins_ok,'invalid link support bins')
        ok &= check(path+'/evidence',v.get('evidence'),txt(v.get('evidence')))
        if v.get('state')=='observed': ok &= check(path+'/visible_support',True,v.get('scope')=='current_frames' and bool(bs),'observed link needs current-frame evidence')
        link_checks.append({'index':i,'valid':bool(ok)})
    ordered = len(frames)==8 and all(isinstance(v,dict) and type(v.get('bin')) is int and v['bin']==i for i,v in enumerate(frames))
    check('/frames/order',None,ordered,'exactly eight ordered frame records required')
    for i, value in enumerate(frames):
        v=value if isinstance(value,dict) else {}; path=f'/frames/{i}'
        for key in ('action','constraint','injury'): enum(path+'/'+key,v.get(key),{'yes','no','unknown'})
        enum(path+'/quality',v.get('quality'),{'good','limited','insufficient'})
        check(path+'/evidence',v.get('evidence'),txt(v.get('evidence')))
    context=raw.get('context') if isinstance(raw.get('context'),dict) else {}
    for key in ('cross_event_required','global_story_required'): check('/context/'+key,context.get(key),type(context.get(key)) is bool)
    check('/context/evidence',context.get('evidence'),txt(context.get('evidence')))
    b5=raw.get('b5') if isinstance(raw.get('b5'),dict) else {}
    enum('/b5/label',b5.get('label'),{'yes','no','unknown'})
    check('/b5/probability',b5.get('probability'),number(b5.get('probability')) or (b5.get('label')=='unknown' and b5.get('probability') is None),'invalid probability')
    refs=b5.get('basis_event_ids')
    check('/b5/basis_event_ids',refs,isinstance(refs,list) and all(isinstance(x,str) and x in event_map for x in refs) and len(refs)==len(set(refs)),'unknown/repeated basis events')
    check('/b5/evidence',b5.get('evidence'),txt(b5.get('evidence')))
    supported=[e['id'] for e in event_checks if e['supported']]
    core_paths=['/schema_version','/persons','/objects','/events','/links','/frames','/frames/order','/b5/label','/b5/probability','/b5/basis_event_ids','/b5/evidence']
    core=all(fields.get(p,{}).get('valid') for p in core_paths)
    core &= ordered and all(fields.get(f'/frames/{i}/{k}',{}).get('valid') for i in range(8) for k in ('action','constraint','injury','quality','evidence'))
    if core and b5['label']=='yes':
        check('/b5/consistency',True,bool(set(refs)&set(supported)),'positive label without a supported local basis')
    elif core and b5['label']=='no': check('/b5/consistency',True,not supported,'negative label with positive local evidence')
    else: check('/b5/consistency',None,core,'invalid core judgment')
    return {'schema_version':VERSION,'core_valid':bool(core),'fields':fields,'issues':issues,'events':event_checks,
            'links':link_checks,'local_supported_events':supported,
            'diagnostic':'local_b5_support' if supported else 'unknown_or_no_supported_b5',
            'raw_label_not_overridden':b5.get('label'),'score_override':False}


def feature_row(raw, audit):
    """JSON null + observed=false; absence of an event list is not a negative."""
    result={name:{'value':None,'observed':False} for name in FEATURES}
    def put(name,value):
        if value is not None: result[name]={'value':float(value),'observed':True}
    if not isinstance(raw,dict): return result
    fields=audit['fields']
    def field(path):
        f=fields.get(path,{})
        return f.get('value') if f.get('valid') else None
    def state(v): return {'observed':1,'not_observed':0,'yes':1,'no':0}.get(v)
    if field('/b5/label') in {'yes','no'}: put('b5_probability',field('/b5/probability'))
    for event_type,name in [('interpersonal_action','interpersonal_observation'),('object_constraint','object_constraint_observation'),('injury_trace','injury_observation')]:
        relevant=[e for e in audit['events'] if field(f'/events/{e["index"]}/type')==event_type]
        values=[state(field(f'/events/{e["index"]}/observation')) if e['observation_valid'] else None for e in relevant]
        if values and all(v is not None for v in values): put(name,max(values))
    for event_type,name in [('interpersonal_action','local_person_binding'),('object_constraint','local_object_binding')]:
        relevant=[e for e in audit['events'] if field(f'/events/{e["index"]}/type')==event_type]
        values=[state(field(f'/events/{e["index"]}/binding/state')) if e['binding_valid'] else None for e in relevant]
        if values and all(v is not None for v in values): put(name,max(values))
    if audit['local_supported_events']: put('local_b5_supported',1)
    # A zero here means explicitly observed non-support, not a binary normal label.
    elif audit['events'] and all(e['valid'] and field(f'/events/{e["index"]}/boundary/b5_state')=='not_supported' for e in audit['events']): put('local_b5_supported',0)
    values=[state(field(f'/links/{e["index"]}/state')) if e['valid'] else None for e in audit['links']]
    if values and all(v is not None for v in values): put('cross_event_link_support',max(values))
    vals=[field(f'/events/{e["index"]}/context_scope/{k}') for e in audit['events'] for k in ('local_mechanism_required','local_binding_required','category_required')]
    if vals and all(type(v) is bool for v in vals): put('local_context_needed',any(vals))
    for key,name in [('cross_event_required','cross_event_context_needed'),('global_story_required','story_context_needed')]:
        v=field('/context/'+key)
        if type(v) is bool: put(name,v)
    if fields.get('/frames/order',{}).get('valid'):
        for key,name in [('action','frame_action_fraction'),('constraint','frame_constraint_fraction'),('injury','frame_injury_fraction')]:
            vals=[state(field(f'/frames/{i}/{key}')) if field(f'/frames/{i}/quality') in {'good','limited'} else None for i in range(8)]
            if all(v is not None for v in vals): put(name,sum(vals)/8)
    return result
