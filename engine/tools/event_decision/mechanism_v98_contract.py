"""Payload-only contract and dependency-local validation; no acquisition authority."""
import copy
import json

from . import mechanism_v97_contract as v97
from .mechanism_v96_contract import ROLES, STATES, STRENGTHS

VERSION = 'v98_payload_and_local_validity_1'
GATES = dict(v97.GATES)
KINDS = dict(v97.KINDS)
TOP_FIELDS = {'version', 'entities', 'observations'}
META_FIELDS = {'acquisition_enabled', 'status', 'primary_kind_to_derived_coarse_type',
               'states', 'evidence_strengths', 'future_event_links', 'limits'}
CRITICAL = (v97.CRITICAL - {'FORBIDDEN_ROLE_OR_TOP_FIELD'}) | {'TOP_LEVEL_FIELDS'}

# Bounds follow the frozen V9.7 semantics. None means no extra maximum.
# Columns: actor, target, person participant, object, instrument.
ROLE_BOUNDS = {
    'directed_interpersonal': ((1, 1), (1, 1), (0, 0), (0, 0), (0, None)),
    'reciprocal_interpersonal': ((0, 0), (0, 0), (2, None), (0, 0), (0, None)),
    'object_constraint': ((0, 1), (1, 1), (0, 0), (1, 1), (0, None)),
    'injury_trace': ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)),
    'person_object_interaction': ((0, 0), (0, 0), (1, 1), (1, 1), (0, None)),
    'object_interaction': ((0, 0), (0, 0), (0, 0), (2, 2), (0, None)),
    'co_presence': ((0, 0), (0, 0), (0, None), (0, None), (0, 0)),
}


def wire_schema():
    """Document shape only. Cross-reference/evidence rules remain in assess()."""
    def obj(properties):
        return {'type': 'object', 'additionalProperties': False,
                'required': list(properties), 'properties': properties}
    def array(item, minimum=0, maximum=None):
        value = {'type': 'array', 'items': item, 'minItems': minimum, 'uniqueItems': True}
        if maximum is not None: value['maxItems'] = maximum
        return value
    def enum(values): return {'type': 'string', 'enum': list(values)}
    def ids(prefix): return array({'type': 'string', 'pattern': '^'+prefix+'[1-9][0-9]*$'})
    text = {'type': 'string', 'pattern': r'\S'}
    bins = array({'type': 'integer', 'minimum': 0, 'maximum': 7}, 1, 8)
    relation = obj({**{r: ids('o' if r in ('object_ids', 'instrument_ids') else 'p') for r in ROLES},
                    'state': enum(STATES), 'strength': enum(STRENGTHS),
                    'bins': array({'type': 'integer', 'minimum': 0, 'maximum': 7}, 0, 8),
                    'evidence': text})
    entity = obj({'id': {'type': 'string', 'pattern': '^[po][1-9][0-9]*$'},
                  'description': text, 'bins': bins, 'identity_scope': {'const': 'local_instance'}})
    observation = obj({'id': {'type': 'string', 'pattern': '^e[1-9][0-9]*$'},
                       'kind': enum(KINDS), 'bins': bins,
                       'phase': enum(('active', 'ongoing_constraint', 'residual', 'other', 'unknown')),
                       'presence': enum(STATES), 'evidence': text,
                       'relation': {'anyOf': [{'type': 'null'}, {'$ref': '#/$defs/relation'}]},
                       'context_entity_ids': ids('[po]'),
                       'context_needed_for': array(enum(v97.CONTEXT))})
    result = obj({'version': {'const': VERSION},
                  'entities': array({'$ref': '#/$defs/entity'}, 0, v97.LIMITS['entities']),
                  'observations': array({'$ref': '#/$defs/observation'}, 1, v97.LIMITS['observations'])})
    result.update({'$schema': 'https://json-schema.org/draft/2020-12/schema',
                   '$defs': {'entity': entity, 'observation': observation, 'relation': relation}})
    return result


def assess(raw):
    """Never fix roles, entities, frames or evidence. Validate dependencies locally."""
    legacy_input = copy.deepcopy(raw)
    if isinstance(legacy_input, dict): legacy_input['version'] = v97.VERSION
    previous = v97.assess(legacy_input)
    issues = []
    seen = set()
    def add(path, code):
        if (path, code) not in seen:
            issues.append({'path': path, 'code': code}); seen.add((path, code))
    for item in previous['issues']:
        code = item['code']
        if code in ('UNEXPECTED_OR_MISSING_TOP_LEVEL_FIELD', 'FORBIDDEN_ROLE_OR_TOP_FIELD'):
            code = 'TOP_LEVEL_FIELDS'
        add(item['path'], code)
    if not isinstance(raw, dict):
        return {'version': VERSION, 'valid': False, 'issues': issues,
                'entities': [], 'observations': [], 'entities_valid': False}
    if raw.get('version') != VERSION: add('/version', 'WRONG_VERSION')
    entities = raw.get('entities') if isinstance(raw.get('entities'), list) else []
    obs = raw.get('observations') if isinstance(raw.get('observations'), list) else []
    def at(issue, path): return issue['path'] == path or issue['path'].startswith(path+'/')
    entity_checks = []
    valid_ids = set()
    for i, entity in enumerate(entities):
        local = [x for x in issues if at(x, f'/entities/{i}')]
        uid = entity.get('id') if isinstance(entity, dict) else None
        valid = isinstance(uid, str) and not local
        if valid: valid_ids.add(uid)
        entity_checks.append({'index': i, 'id': uid, 'valid': valid, 'issues': local})
    checks = []
    for i, event in enumerate(obs):
        path = f'/observations/{i}'
        local = [x for x in issues if at(x, path)]
        check = copy.deepcopy(previous['observations'][i])
        relation = event.get('relation') if isinstance(event, dict) else None
        refs = sorted({x for role in ROLES for x in
                       (relation.get(role, []) if isinstance(relation, dict) and isinstance(relation.get(role), list) else [])
                       if isinstance(x, str)})
        invalid_refs = [x for x in refs if x not in valid_ids]
        # The inherited validator already rejects missing/duplicate/malformed referenced
        # entities. No unrelated /entities issue is allowed to poison this relation.
        relation_valid = bool(relation is not None and not invalid_refs and
                              check['observation_valid'] and check['context_valid'] and not local)
        check.update(relation_valid=relation_valid, referenced_entity_ids=refs,
                     presence=event.get('presence') if isinstance(event, dict) else None,
                     invalid_referenced_entity_ids=invalid_refs,
                     local_issues=local,
                     local_relation_structurally_supported=bool(relation_valid and relation.get('state') == 'observed'))
        checks.append(check)
    return {'version': VERSION, 'valid': not issues, 'issues': issues,
            'entities': entity_checks, 'observations': checks,
            'entities_valid': not any(at(e, '/entities') for e in issues)}


def replay_payload(raw, *, project_metadata=False):
    """Explicit version projection, plus optional exact inert metadata removal only."""
    projected = copy.deepcopy(raw)
    edits = []
    if not isinstance(raw, dict) or raw.get('version') != v97.VERSION:
        raise ValueError('Only a saved V9.7 payload may enter this replay')
    projected['version'] = VERSION
    edits.append({'path': '/version', 'before': v97.VERSION, 'after': VERSION,
                  'rule': 'explicit_version_projection_not_native_inference'})
    if project_metadata:
        spec = json.loads(json.dumps(v97.specification()))
        for key in sorted(set(raw)-TOP_FIELDS):
            # JSON equality keeps true distinct from 1, and lists distinct from scalars.
            if key in META_FIELDS and json.dumps(raw[key], sort_keys=True) == json.dumps(spec[key], sort_keys=True):
                del projected[key]
                edits.append({'path': '/'+key, 'before': raw[key], 'rule': 'exact_documentation_echo_only'})
    for key in ('entities', 'observations'):
        if projected.get(key) != raw.get(key): raise AssertionError('Evidence changed during projection')
    return projected, edits


def examples():
    """Synthetic syntax fixtures for every kind, including the actual error families."""
    def entity(uid, bs=(0,)):
        return {'id': uid, 'description': 'Visible local instance '+uid, 'bins': list(bs), 'identity_scope': 'local_instance'}
    def make(kind, roles=None, phase='active', scope=()):
        relation = None if roles is None else dict(
            {r: [] for r in ROLES}, state='observed', strength='directly_visible', bins=[0],
            evidence='The explicitly listed arguments are visible together.', **roles)
        refs = sorted({x for r in ROLES for x in (relation[r] if relation else [])})
        return {'version': VERSION, 'entities': [entity(x) for x in refs], 'observations': [{
            'id': 'e1', 'kind': kind, 'bins': [0], 'phase': phase, 'presence': 'observed',
            'evidence': 'Synthetic local observation for '+kind+'.', 'relation': relation,
            'context_entity_ids': [], 'context_needed_for': list(scope)}]}
    rows = [
        make('directed_interpersonal', {'actor_ids': ['p1'], 'target_ids': ['p2'], 'instrument_ids': ['o1']}),
        make('reciprocal_interpersonal', {'participant_ids': ['p1', 'p2']}),
        make('object_constraint', {'target_ids': ['p1'], 'object_ids': ['o1']}, 'ongoing_constraint'),
        make('injury_trace', {'target_ids': ['p1']}, 'residual'),
        make('person_object_interaction', {'participant_ids': ['p1'], 'object_ids': ['o1']}),
        make('object_interaction', {'object_ids': ['o1', 'o2']}),
        make('co_presence', {'participant_ids': ['p1'], 'object_ids': ['o1']}),
    ]
    for kind in KINDS:
        if kind not in ROLE_BOUNDS: rows.append(make(kind, scope=['relation'] if kind == v97.GROUP else []))
    rows[-2]['observations'][0]['evidence'] = 'An observation, not a verified event category.'
    inferred = make('directed_interpersonal', {'actor_ids': ['p1'], 'target_ids': ['p2']}, scope=['identity', 'relation'])
    inferred['entities'][1]['bins'] = [7]
    inferred['observations'][0]['bins'] = [0, 7]
    inferred['observations'][0]['relation'].update(state='unknown', strength='edit_inferred', bins=[0, 7],
                                                 evidence='Separate shots do not establish local binding.')
    rows.append(inferred)
    bodypart = make('directed_interpersonal', {'actor_ids': ['p1'], 'target_ids': ['p2']}, scope=['identity'])
    bodypart['entities'][0]['description'] = 'A visible hand, owner not identified across shots.'
    rows.append(bodypart)
    return rows


def prompt():
    # No runtime configuration or specification() serialization enters the request.
    instructions = '''Inspect exactly eight chronological silent images T0..T7.
Return ONE JSON object with EXACTLY these three top-level keys: version, entities, observations.
Return an instance, NOT the schema, instructions, examples, configuration or category dictionary.
All object keys in the schema are mandatory; no extra keys. Bins are sorted unique integers 0..7.
Use p1,p2,... for visible local people/body parts and o1,o2,... for objects (never p0 or env_fire).
Never infer the unseen owner of a hand or identity continuity across a cut.
participant_ids accepts PEOPLE ONLY. object_ids and instrument_ids accept OBJECTS ONLY.
For person_object_interaction put the person in participant_ids, never actor_ids.
For object_constraint put the restraint in object_ids, not only instrument_ids.
For co_presence put people in participant_ids and objects in object_ids; at least two total.
Instruments are necessary evidence. context_entity_ids are ancillary and cannot duplicate arguments.
No automatically inferred targets, actors, object roles, identities, frames, categories or links.
Preserve current action, ongoing restraint, residual injury and other-event observations separately.
An injury does not establish a current attacker. A weapon does not establish its target.
Use relation=null for observation-only kinds and unresolved group pairings. Null/unknown is NOT normal.
Observed relations require all necessary arguments within support bins; directly_visible requires a common bin.
Use unknown+edit_inferred for a cutaway hypothesis, never observed+edit_inferred.
tracked requires visible continuity; reusing an ID across shots does not verify continuity.
context_needed_for identifies unresolved identity/relation/phase/intent, not general scene background.
If relation needs context it cannot be observed; if phase needs context use phase=unknown.
No B5 verdict, anomaly score, graph score or cross-event link is requested.
Role bounds below apply only when relation is non-null. Omitted roles must be empty arrays.
Actor and target must be distinct. Object and instrument lists must not overlap.
'''
    rules = []
    for kind, bounds in ROLE_BOUNDS.items():
        values = [f'{r}={lo}..{hi if hi is not None else "many"}' for r, (lo, hi) in zip(ROLES, bounds)]
        rules.append(kind+': '+', '.join(values))
    rules.append('Observation-only kinds (relation=null): '+', '.join(k for k in KINDS if k not in ROLE_BOUNDS))
    # The full example suite is stored offline; four compact examples cover recurring wire ambiguities.
    selected = [examples()[i] for i in (2, 4, 6, 12)]
    return (instructions+'\n'+'\n'.join(rules)+'\nOUTPUT JSON SCHEMA (do not echo):\n'+
            json.dumps(wire_schema(), ensure_ascii=True, separators=(',', ':'))+
            '\nSYNTHETIC VALID ANSWER EXAMPLES (not target answers):\n'+
            json.dumps(selected, ensure_ascii=True, separators=(',', ':')))


def metrics(assessments):
    n = len(assessments)
    observations = [o for a in assessments for o in a['observations']]
    relations = [o for o in observations if o['relation_claim_present']]
    from collections import Counter
    issues = Counter(e['code'] for a in assessments for e in a['issues'])
    ratio = lambda numerator, denominator: numerator/denominator if denominator else 0.0
    result = {'windows': n, 'valid_documents': sum(a['valid'] for a in assessments),
              'observations': len(observations), 'valid_observations': sum(o['observation_valid'] for o in observations),
              'relations': len(relations), 'valid_relations': sum(o['relation_valid'] for o in relations),
              'windows_with_observed_observation': sum(any(o['observation_valid'] and o.get('presence') == 'observed' for o in a['observations']) for a in assessments),
              'issue_counts': dict(issues), 'critical_issues': {k: v for k, v in issues.items() if k in CRITICAL}}
    for field, numerator, denominator in (
        ('document_valid_fraction', result['valid_documents'], n),
        ('observation_valid_fraction', result['valid_observations'], len(observations)),
        ('relation_valid_fraction', result['valid_relations'], len(relations)),
        ('windows_with_observed_observation_fraction', result['windows_with_observed_observation'], n)):
        result[field] = ratio(numerator, denominator)
    failed = [k for k in ('document_valid_fraction', 'observation_valid_fraction', 'relation_valid_fraction',
                          'windows_with_observed_observation_fraction') if result[k] < GATES['minimum_'+k]]
    if result['critical_issues']: failed.append('critical_reference_or_evidence_contradictions')
    if n != 36: failed.append('complete_fixed_36_cohort')
    result.update(failed_checks=failed, technical_proportions_and_critical_checks_pass=not failed)
    return result


def offline_gate(replay_metrics):
    return {'decision': 'OFFLINE_COMPLETE_REVIEW_DISABLED',
            'technical_proportions_and_critical_checks_pass': replay_metrics['technical_proportions_and_critical_checks_pass'],
            'native_v98_revalidation_completed': False, 'new_requests_decided': False,
            'remote_execution_authorized': False, 'human_review_requested': False,
            'ready_for_shadow_integration': False, 'scoring_authorized': False, 'training_authorized': False,
            'reason': 'Saved-response replay is not a native V9.8 acquisition or human review authorization.'}
