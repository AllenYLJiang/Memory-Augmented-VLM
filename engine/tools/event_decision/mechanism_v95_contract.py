"""Typed relationships and independent availability, without inferred identity repair."""
import copy
import json
import math
from collections import Counter

from .mechanism_v94_contract import CATEGORIES, RULES

VERSION = 'v95_typed_mechanism_1'
STATES = {'observed', 'not_observed', 'unknown'}
ANSWERS = {'yes', 'no', 'unknown'}
TYPES = {'interpersonal_action', 'object_constraint', 'injury_trace', 'other_event', 'unknown'}
KINDS = {'directed_interpersonal', 'reciprocal_interpersonal', 'person_object_constraint',
         'injury_attribution', 'person_object_interaction', 'object_interaction', 'co_presence', 'unresolved'}
LIMITS = {'persons': 8, 'objects': 8, 'events': 6, 'links': 6, 'frames': 8}
FEATURES = ['b5_probability', 'interpersonal_observation', 'object_constraint_observation',
            'injury_observation', 'directed_binding', 'reciprocal_binding', 'constraint_binding',
            'injury_binding', 'object_binding', 'local_b5_supported', 'cross_event_link_support',
            'mechanism_needs_more', 'binding_needs_more', 'category_needs_more',
            'cross_event_needs_more', 'story_context_known',
            'frame_action_fraction', 'frame_constraint_fraction', 'frame_injury_fraction',
            'frame_action_coverage', 'frame_constraint_coverage', 'frame_injury_coverage']


def examples():
    """Complete, validated examples with both polarities and honest uncertainty."""
    def person(i): return {'id': f'p{i}', 'description': f'Visible person {i}', 'bins': [0, 1]}
    def obj(i, kind): return {'id': f'o{i}', 'kind': kind, 'description': f'Visible {kind} {i}', 'bins': [0, 1]}
    base = {'schema_version': VERSION, 'persons': [person(1), person(2)], 'objects': [],
        'b5': {'label': 'yes', 'probability': .9, 'basis_event_ids': ['e1'], 'evidence': 'Unilateral assault on a defenseless person.'},
        'events': [{'id': 'e1', 'type': 'interpersonal_action', 'bins': [0, 1],
            'actor_ids': ['p1'], 'target_ids': ['p2'], 'participant_ids': [], 'object_ids': [],
            'phase': 'active', 'observation': 'observed', 'evidence': 'p1 chokes p2; p2 tries to shield the neck.',
            'binding': {'kind': 'directed_interpersonal', 'state': 'observed', 'mode': 'same_frame',
                        'evidence_bins': [0, 1], 'evidence': 'Hands and target neck are co-visible.'},
            'boundary': {'b5_state': 'supported', 'alternative_classes': [], 'basis': 'defenseless_assault', 'evidence': 'Direct assault, not reciprocal engagement.'},
            'additional_evidence_needed': {'mechanism': 'no', 'binding': 'no', 'b5_category': 'no'}}],
        'links': [], 'frames': [{'bin': i, 'action_presence': 'yes' if i < 2 else 'unknown',
            'constraint_presence': 'no' if i < 2 else 'unknown', 'injury_presence': 'unknown',
            'action_description': 'Choking' if i < 2 else 'Unclear', 'quality': 'limited',
            'evidence': 'Contact visible' if i < 2 else 'Insufficient detail'} for i in range(8)],
        'context': {'cross_event_link_needs_more_evidence': 'unknown', 'story_context_known': 'no',
                    'evidence': 'Plot unknown; local assault is sufficient.'}}
    reciprocal = copy.deepcopy(base)
    reciprocal['b5'].update(label='no', probability=None, evidence='Reciprocal combat, not established unilateral abuse.')
    e = reciprocal['events'][0]
    e.update(actor_ids=[], target_ids=[], participant_ids=['p1', 'p2'], evidence='Both people visibly trade blows.')
    e['binding'].update(kind='reciprocal_interpersonal', evidence='The two participants trade blows in view.')
    e['boundary'].update(b5_state='not_supported', alternative_classes=['B1'], basis='mutual_combat', evidence='Both initiate combat; not merely defensive resistance.')
    reciprocal['context']['evidence'] = 'Story unknown does not prevent describing reciprocal combat.'
    vehicle = copy.deepcopy(base)
    vehicle.update(persons=[], objects=[obj(1, 'vehicle'), obj(2, 'vehicle')])
    vehicle['b5'].update(label='no', probability=None, evidence='Vehicle contact without person-directed abuse evidence.')
    e = vehicle['events'][0]
    e.update(type='other_event', actor_ids=[], target_ids=[], object_ids=['o1', 'o2'], evidence='Two vehicles collide.')
    e['binding'].update(kind='object_interaction', evidence='The vehicles contact each other.')
    e['boundary'].update(b5_state='not_supported', alternative_classes=['B6'], basis='vehicle_event', evidence='Collision is B6, not automatically B5.')
    for f in vehicle['frames'][:2]: f.update(action_presence='no', action_description='Vehicle collision', evidence='Vehicle contact, no interpersonal action visible.')
    vehicle['context']['evidence'] = 'No inference about pursuit intent.'
    constraint = copy.deepcopy(base)
    constraint.update(persons=[person(1)], objects=[obj(1, 'restraint')])
    constraint['b5'].update(label='unknown', probability=None, evidence='Attachment is visible, coercive purpose unresolved.')
    e = constraint['events'][0]
    e.update(type='object_constraint', actor_ids=[], target_ids=['p1'], object_ids=['o1'], phase='ongoing_constraint', evidence='Strap surrounds a wrist, no perpetrator visible.')
    e['binding'].update(kind='person_object_constraint', evidence='Visible strap around wrist.')
    e['boundary'].update(b5_state='unknown', basis='object_restraint_only', evidence='Medical, supportive or coercive purpose is not distinguished.')
    e['additional_evidence_needed']['b5_category'] = 'yes'
    for f in constraint['frames'][:2]: f.update(action_presence='no', constraint_presence='yes', action_description='Visible attachment', evidence='Strap around wrist.')
    constraint['context']['evidence'] = 'Constraint is observed but abuse category needs more evidence.'
    return [base, reciprocal, vehicle, constraint]


PROMPT = '\n'.join([
    'Return JSON only for the eight supplied silent images T0..T7. Do not use film/actor identity, filename, plot or prior answers.',
    'Fixed categories: ' + json.dumps(CATEGORIES), *RULES.values(),
    'No B5 is NOT binary normal. Null probability is allowed for any label; do not fabricate numeric confidence.',
    'All observations below refer to these images. Keep evidence strings short. Lists may be empty except exactly eight ordered frames.',
    'Budgets: ' + json.dumps(LIMITS) + '. Do not omit required fields from individual records.',
    'persons: id p1...; objects: id o1..., kind restraint/container/support/weapon/vehicle/other/unknown; description and nonempty observation bins 0..7.',
    'Every event defines id e1..., type interpersonal_action/object_constraint/injury_trace/other_event/unknown, bins, '
    'actor_ids,target_ids,participant_ids,object_ids (all lists), phase active/ongoing_constraint/residual/other/unknown, observation observed/not_observed/unknown, evidence.',
    'Roles reference existing entities seen somewhere in this window. Their personal observation bins are not the same thing as relationship evidence_bins.',
    'binding fields: kind, state observed/not_observed/unknown, mode same_frame/tracked_cross_frame/inferred_edit/unknown, evidence_bins, evidence.',
    'Kinds: directed_interpersonal needs distinct actors and targets; reciprocal_interpersonal uses participant_ids (>=2) and empty actor_ids/target_ids; '
    'person_object_constraint needs target+object but no invented actor; injury_attribution needs target; person_object_interaction needs participant+object; '
    'object_interaction needs >=2 objects; co_presence needs >=2 distinct entities but proves no mechanism; unresolved cannot be observed.',
    'For observed relationships, evidence_bins must be in event bins and each relationship entity must be seen there. same_frame requires co-visibility. '
    'tracked_cross_frame needs visible tracking evidence, not a story assumption. Unknown candidate targets may appear outside event bins; '
    'use inferred_edit/unknown and do not expand bins or invent identity to make the relation observed.',
    'boundary: b5_state supported/not_supported/unknown; alternative_classes from dictionary; basis unilateral_control/defenseless_assault/credible_person_threat/'
    'object_restraint_only/mutual_combat/vehicle_event/injury_only/other/unknown; evidence. Co-presence, injury or restraint alone is not B5.',
    'additional_evidence_needed contains mechanism,binding,b5_category, each yes/no/unknown. It asks whether evidence OUTSIDE THESE EIGHT IMAGES '
    'is needed to decide that specific question. Unknown plot does NOT imply yes. Supported local abuse with adequate visible evidence has all three no. '
    'If category really needs more evidence, boundary.b5_state must be unknown, not supported. Do not copy example flags.',
    'links: source and target are existing distinct event IDs; relation shared_incident/continuation/intervention/unknown; state observed/not_observed/unknown; '
    'scope current_frames/needs_context; bins; evidence. Unknown links do not erase local events. Observed links need visible support from both events, not temporal order alone.',
    'frames: bin 0..7, action_presence,constraint_presence,injury_presence each yes/no/unknown, action_description free text, quality good/limited/insufficient, evidence. '
    'action_presence means visible interpersonal action, NOT whether any motion exists. Never put fighting or dragging in a presence field.',
    'b5: label yes/no/unknown, probability null or finite 0..1, basis_event_ids, evidence. Label and numeric availability are independent. '
    'An unknown role link or unknown B5 category must not become a confident yes. Preserve uncertainty and alternative mechanisms.',
    'context: cross_event_link_needs_more_evidence yes/no/unknown; story_context_known yes/no/unknown; evidence. These global answers do not veto independently supported local events.',
    'Complete illustrative examples, NOT descriptions of the supplied images. They show positive, reciprocal non-B5, vehicle non-B5, and unknown-purpose constraint:',
    *[json.dumps(e, separators=(',', ':')) for e in examples()],
])


def assess(raw):
    fields, issues, event_checks, link_checks = {}, [], [], []
    def check(path, value, valid, reason='invalid field'):
        fields[path] = {'valid': bool(valid), 'value': value if valid else None}
        if not valid: issues.append({'path': path, 'reason': reason})
        return bool(valid)
    def enum(path, value, options): return check(path, value, isinstance(value, str) and value in options, 'invalid enum')
    def text(value): return isinstance(value, str) and 0 < len(value.strip()) <= 1500
    def bins(value, empty=False):
        return isinstance(value, list) and (empty or bool(value)) and all(type(i) is int and 0 <= i < 8 for i in value) and value == sorted(set(value))
    def obj(value): return value if isinstance(value, dict) else {}
    def ident(value, prefix): return isinstance(value, str) and value.startswith(prefix) and value[1:].isascii() and value[1:].isdigit() and not value[1:].startswith('0')
    raw = obj(raw)
    check('/schema_version', raw.get('schema_version'), raw.get('schema_version') == VERSION)
    lists = {}
    for name, budget in LIMITS.items():
        value = raw.get(name)
        shape = check('/' + name, None, isinstance(value, list), 'missing list')
        check('/' + name + '/budget', len(value) if shape else None, shape and len(value) <= budget, 'list exceeds declared budget')
        # Over-budget elements are audited too; a sibling is never silently removed.
        lists[name] = value if shape else []
    entities, entity_bins, entity_types = {}, {}, {}
    for name, prefix in [('persons', 'p'), ('objects', 'o')]:
        ids = Counter(v.get('id') for v in lists[name] if isinstance(v, dict) and isinstance(v.get('id'), str))
        for i, value in enumerate(lists[name]):
            v = obj(value); path = f'/{name}/{i}'; key = v.get('id')
            id_ok = check(path + '/id', key, ident(key, prefix) and ids[key] == 1, 'invalid or duplicate entity ID')
            b_ok = check(path + '/bins', v.get('bins'), bins(v.get('bins')), 'invalid entity observation bins')
            desc_ok = check(path + '/description', v.get('description'), text(v.get('description')))
            kind_ok = True
            if name == 'objects': kind_ok = enum(path + '/kind', v.get('kind'), {'restraint','container','support','weapon','vehicle','other','unknown'})
            if id_ok and b_ok:
                entities[key] = bool(desc_ok and kind_ok); entity_bins[key] = set(v['bins']); entity_types[key] = prefix
    eids = Counter(v.get('id') for v in lists['events'] if isinstance(v, dict) and isinstance(v.get('id'), str))
    event_bins = {}
    for i, value in enumerate(lists['events']):
        v = obj(value); path = f'/events/{i}'; key = v.get('id')
        id_ok = check(path+'/id', key, ident(key, 'e') and eids[key] == 1, 'invalid or duplicate event ID')
        b_ok = check(path+'/bins', v.get('bins'), bins(v.get('bins')), 'invalid event bins')
        type_ok = enum(path+'/type', v.get('type'), TYPES)
        phase_ok = enum(path+'/phase', v.get('phase'), {'active','ongoing_constraint','residual','other','unknown'})
        obs_ok = enum(path+'/observation', v.get('observation'), STATES)
        evidence_ok = check(path+'/evidence', v.get('evidence'), text(v.get('evidence')))
        refs_ok = True; roles = {}
        for role, prefix in [('actor_ids','p'),('target_ids','p'),('participant_ids','p'),('object_ids','o')]:
            refs = v.get(role)
            ok = isinstance(refs, list) and all(isinstance(r, str) and entity_types.get(r) == prefix and entities.get(r) for r in refs)
            ok = ok and len(refs) == len(set(refs))
            refs_ok &= check(path+'/'+role, refs, ok, 'invalid entity reference')
            roles[role] = refs if ok else []
        a, t, p, o = (roles[n] for n in ('actor_ids','target_ids','participant_ids','object_ids'))
        all_refs = set(a+t+p+o)
        binding = obj(v.get('binding')); kind = binding.get('kind'); state = binding.get('state'); mode = binding.get('mode')
        bind_ok = enum(path+'/binding/kind', kind, KINDS)
        bind_ok &= enum(path+'/binding/state', state, STATES)
        bind_ok &= enum(path+'/binding/mode', mode, {'same_frame','tracked_cross_frame','inferred_edit','unknown'})
        bind_ok &= check(path+'/binding/evidence', binding.get('evidence'), text(binding.get('evidence')))
        support = binding.get('evidence_bins')
        support_ok = bins(support, empty=True) and b_ok and set(support) <= set(v['bins'])
        bind_ok &= check(path+'/binding/evidence_bins', support, support_ok, 'invalid relationship evidence bins')
        kinds_ok = {
            'directed_interpersonal': v.get('type') == 'interpersonal_action' and bool(a) and bool(t) and not set(a)&set(t) and not p,
            'reciprocal_interpersonal': v.get('type') == 'interpersonal_action' and len(p) >= 2 and not a and not t,
            'person_object_constraint': v.get('type') == 'object_constraint' and bool(t) and bool(o) and not set(a)&set(t),
            'injury_attribution': v.get('type') == 'injury_trace' and bool(t) and not a,
            'person_object_interaction': v.get('type') == 'other_event' and bool(p) and bool(o) and not a and not t,
            'object_interaction': v.get('type') == 'other_event' and len(o) >= 2 and not a and not t,
            'co_presence': len(all_refs) >= 2,
            'unresolved': state != 'observed',
        }
        typed_ok = refs_ok and isinstance(kind, str) and bool(kinds_ok.get(kind, False))
        bind_ok &= check(path+'/binding/typed_support', True, typed_ok, 'relationship kind/roles do not agree')
        if state == 'observed':
            visible = typed_ok and support_ok and bool(support) and bool(all_refs) and mode in {'same_frame','tracked_cross_frame'}
            if visible:
                visible = all(entity_bins[r] & set(support) for r in all_refs)
                if visible and mode == 'same_frame': visible = bool(set(support).intersection(*(entity_bins[r] for r in all_refs)))
            bind_ok &= check(path+'/binding/visible_support', True, visible, 'observed relation lacks typed/co-visible or tracking support')
        need = obj(v.get('additional_evidence_needed')); ctx_ok = True
        for name in ('mechanism','binding','b5_category'):
            ctx_ok &= enum(path+'/additional_evidence_needed/'+name, need.get(name), ANSWERS)
        boundary = obj(v.get('boundary'))
        cat_ok = enum(path+'/boundary/b5_state', boundary.get('b5_state'), {'supported','not_supported','unknown'})
        cat_ok &= enum(path+'/boundary/basis', boundary.get('basis'), {'unilateral_control','defenseless_assault','credible_person_threat','object_restraint_only','mutual_combat','vehicle_event','injury_only','other','unknown'})
        classes = boundary.get('alternative_classes')
        classes_ok = isinstance(classes, list) and all(isinstance(c, str) and c in CATEGORIES for c in classes) and len(classes) == len(set(classes))
        cat_ok &= check(path+'/boundary/alternative_classes', classes, classes_ok, 'invalid category code')
        cat_ok &= check(path+'/boundary/evidence', boundary.get('evidence'), text(boundary.get('evidence')))
        if boundary.get('b5_state') == 'supported':
            specific = kind in ('directed_interpersonal','person_object_constraint') and boundary.get('basis') in ('unilateral_control','defenseless_assault','credible_person_threat')
            cat_ok &= check(path+'/boundary/specific_mechanism', True, specific, 'co-occurrence/reciprocal combat/injury/constraint alone is not B5')
        observation_valid = bool(id_ok and b_ok and type_ok and phase_ok and obs_ok and evidence_ok)
        valid = bool(observation_valid and refs_ok and bind_ok and cat_ok and ctx_ok)
        locally_sufficient = valid and v.get('observation') == 'observed' and state == 'observed' and v.get('phase') in ('active','ongoing_constraint') and all(need.get(n) == 'no' for n in ('mechanism','binding','b5_category'))
        supported = locally_sufficient and boundary.get('b5_state') == 'supported'
        consistent = boundary.get('b5_state') != 'supported' or supported
        check(path+'/boundary/support_consistency', True, consistent, 'supported category conflicts with local evidence/uncertainty')
        event_checks.append({'id': key if id_ok else None, 'index': i, 'valid': valid, 'observation_valid': observation_valid,
            'binding_valid': bool(id_ok and b_ok and bind_ok and refs_ok), 'category_valid': bool(cat_ok and ctx_ok),
            'support_consistent': bool(consistent), 'supported': bool(supported),
            'role_observation_bins': {r: sorted(entity_bins[r]) for r in all_refs if r in entity_bins}})
        if id_ok and b_ok: event_bins[key] = set(v['bins'])
    for i, value in enumerate(lists['links']):
        v = obj(value); path = f'/links/{i}'
        refs = isinstance(v.get('source'), str) and isinstance(v.get('target'), str) and v['source'] in event_bins and v['target'] in event_bins and v['source'] != v['target']
        ok = check(path+'/references', None, refs, 'invalid event link reference')
        ok &= enum(path+'/relation', v.get('relation'), {'shared_incident','continuation','intervention','unknown'})
        ok &= enum(path+'/state', v.get('state'), STATES)
        ok &= enum(path+'/scope', v.get('scope'), {'current_frames','needs_context'})
        bs = v.get('bins'); bs_ok = bins(bs, empty=True)
        if bs_ok and refs: bs_ok = set(bs) <= event_bins[v['source']] | event_bins[v['target']]
        ok &= check(path+'/bins', bs, bs_ok, 'invalid link support bins')
        ok &= check(path+'/evidence', v.get('evidence'), text(v.get('evidence')))
        if v.get('state') == 'observed':
            visible = refs and bs_ok and bool(bs) and v.get('scope') == 'current_frames'
            if visible: visible = bool(set(bs)&event_bins[v['source']]) and bool(set(bs)&event_bins[v['target']])
            ok &= check(path+'/visible_support', True, visible, 'observed link lacks support from both events')
        link_checks.append({'index': i, 'valid': bool(ok)})
    frames = lists['frames']
    ordered = len(frames) == 8 and all(isinstance(f, dict) and type(f.get('bin')) is int and f['bin'] == i for i, f in enumerate(frames))
    check('/frames/order', None, ordered, 'exactly eight ordered frame records required')
    frame_checks = []
    for i in range(8):
        v = obj(frames[i]) if i < len(frames) else {}; path = f'/frames/{i}'; ok = ordered
        for name in ('action_presence','constraint_presence','injury_presence'):
            ok &= enum(path+'/'+name, v.get(name), ANSWERS)
        ok &= enum(path+'/quality', v.get('quality'), {'good','limited','insufficient'})
        for name in ('action_description','evidence'): ok &= check(path+'/'+name, v.get(name), text(v.get(name)))
        frame_checks.append(bool(ok))
    context = obj(raw.get('context'))
    for name in ('cross_event_link_needs_more_evidence','story_context_known'):
        enum('/context/'+name, context.get(name), ANSWERS)
    check('/context/evidence', context.get('evidence'), text(context.get('evidence')))
    b5 = obj(raw.get('b5'))
    label_ok = enum('/b5/label', b5.get('label'), ANSWERS)
    p = b5.get('probability')
    numeric = type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
    check('/b5/probability', p, isinstance(raw.get('b5'),dict) and (p is None or numeric), 'invalid numeric probability; label retained independently')
    refs = b5.get('basis_event_ids')
    refs_ok = isinstance(refs, list) and all(isinstance(r, str) and r in event_bins for r in refs) and len(refs) == len(set(refs))
    label_ok &= check('/b5/basis_event_ids', refs, refs_ok, 'invalid label basis reference')
    label_ok &= check('/b5/evidence', b5.get('evidence'), text(b5.get('evidence')))
    supported = [e['id'] for e in event_checks if e['supported']]
    consistent = bool(label_ok)
    if label_ok and b5['label'] == 'yes': consistent = bool(set(refs)&set(supported))
    if label_ok and b5['label'] == 'no': consistent = not supported
    check('/b5/consistency', True, consistent, 'label and supported local basis disagree')
    core = label_ok and fields['/schema_version']['valid'] and all(fields['/'+n]['valid'] and fields['/'+n+'/budget']['valid'] for n in LIMITS) and all(frame_checks)
    return {'version': VERSION, 'core_valid': bool(core), 'label_valid': bool(label_ok),
            'probability_observed': bool(numeric and b5.get('label') in ('yes','no')), 'fields': fields, 'issues': issues,
            'events': event_checks, 'links': link_checks, 'frame_valid': frame_checks,
            'local_supported_events': supported, 'raw_label_not_overridden': b5.get('label')}


def feature_row(raw, audit):
    output = {n: {'value': None, 'observed': False} for n in FEATURES}
    def field(path):
        f = audit['fields'].get(path, {})
        return f.get('value') if f.get('valid') else None
    def numeric(v): return {'yes': 1, 'no': 0, 'observed': 1, 'not_observed': 0}.get(v)
    def put(name, v):
        if v is not None: output[name] = {'value': float(v), 'observed': True}
    def aggregate(name, vals):
        if vals and all(v is not None for v in vals): put(name, max(vals))
    if audit['probability_observed']: put('b5_probability', field('/b5/probability'))
    for name, kind in [('interpersonal_observation','interpersonal_action'),('object_constraint_observation','object_constraint'),('injury_observation','injury_trace')]:
        subset = [e for e in audit['events'] if field(f'/events/{e["index"]}/type') == kind]
        aggregate(name, [numeric(field(f'/events/{e["index"]}/observation')) if e['observation_valid'] else None for e in subset])
    for name, kinds in [('directed_binding',{'directed_interpersonal'}),('reciprocal_binding',{'reciprocal_interpersonal'}),
                        ('constraint_binding',{'person_object_constraint'}),('injury_binding',{'injury_attribution'}),
                        ('object_binding',{'object_interaction','person_object_interaction'})]:
        subset = [e for e in audit['events'] if field(f'/events/{e["index"]}/binding/kind') in kinds]
        aggregate(name, [numeric(field(f'/events/{e["index"]}/binding/state')) if e['binding_valid'] else None for e in subset])
    if audit['local_supported_events']: put('local_b5_supported', 1)
    elif audit['events'] and all(e['valid'] and field(f'/events/{e["index"]}/boundary/b5_state') == 'not_supported' for e in audit['events']): put('local_b5_supported', 0)
    aggregate('cross_event_link_support', [numeric(field(f'/links/{l["index"]}/state')) if l['valid'] else None for l in audit['links']])
    for key, name in [('mechanism','mechanism_needs_more'),('binding','binding_needs_more'),('b5_category','category_needs_more')]:
        aggregate(name, [numeric(field(f'/events/{e["index"]}/additional_evidence_needed/{key}')) for e in audit['events']])
    for key, name in [('cross_event_link_needs_more_evidence','cross_event_needs_more'),('story_context_known','story_context_known')]:
        put(name, numeric(field('/context/'+key)))
    if audit['fields']['/frames/order']['valid']:
        for kind in ('action','constraint','injury'):
            vals = [numeric(field(f'/frames/{i}/{kind}_presence')) if field(f'/frames/{i}/quality') in ('good','limited') else None for i in range(8)]
            put('frame_'+kind+'_coverage', sum(v is not None for v in vals)/8)
            if all(v is not None for v in vals): put('frame_'+kind+'_fraction', sum(vals)/8)
    return output
