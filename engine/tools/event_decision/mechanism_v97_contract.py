"""Native local observations; no category prediction or scoring authority."""
import copy
import json

from . import mechanism_v96_contract as legacy

VERSION = 'v97_native_local_observation_1'
GROUP = 'group_interpersonal_observation'
KINDS = dict(legacy.KINDS, **{GROUP: 'interpersonal_action'})
CONTEXT = ('identity', 'relation', 'phase', 'intent')
LIMITS = {'entities': 24, 'observations': 16}
GATES = {'minimum_document_valid_fraction': .95,
         'minimum_observation_valid_fraction': .95,
         'minimum_relation_valid_fraction': .95,
         'minimum_windows_with_observed_observation_fraction': .5,
         'minimum_review_supported_fraction': .9,
         'minimum_review_controls': 8}
CRITICAL = {'ENTITY_ID', 'ENTITY_BINS', 'EVENT_ID', 'EVENT_BINS', 'CONTEXT_REFERENCES',
            'ROLE_REFERENCES', 'RELATION_BINS', 'NECESSARY_ARGUMENT_OUTSIDE_SUPPORT',
            'NO_COMMON_LOCAL_FRAME', 'INFERRED_IS_NOT_DIRECT', 'PRESENCE_RELATION_CONTRADICTION',
            'CONTEXT_ARGUMENT_OVERLAP', 'OBJECT_INSTRUMENT_OVERLAP', 'RELATION_NEEDS_CONTEXT',
            'OBSERVATION_CONTEXT_REQUIRED', 'WRONG_VERSION', 'FORBIDDEN_ROLE_OR_TOP_FIELD'}


def specification():
    spec = copy.deepcopy(legacy.specification())
    spec.update(version=VERSION, acquisition_enabled=True, status='development_native_frozen',
                primary_kind_to_derived_coarse_type=KINDS, limits=LIMITS)
    spec['entity']['description'] = 'visible local person/body part or object; never assign an unseen owner'
    spec['observation']['context_needed_for'] = list(CONTEXT)
    spec['role_cardinality'][GROUP] = 'observation only, relation=null; group activity without fabricated pairings'
    spec['invariants'] += [
        'Eight silent frames only; bins denote samples, not dense temporal intervals.',
        'Split directed pairs only when each pair is visibly supported; never expand a group into all pairs.',
        'A visible unidentified hand can be a local p-ID, not a claim about its owner in another shot.',
        'If relation needs external context, it cannot be declared observed in this input.',
        'If phase needs external context, phase must be unknown.',
        'Unknown/null are valid representations, not normal or verified negative labels.',
        'Not-observed arguments need not co-occur; observed direct relations require a common sample.',
        'Tracking strength is an assertion, not a cross-shot identity verifier.',
    ]
    spec['future_event_links'] = 'No generated cross-event links or category head in this experiment.'
    return spec


def assess(raw):
    transformed = copy.deepcopy(raw)
    if isinstance(transformed, dict):
        transformed['version'] = legacy.VERSION
        if isinstance(transformed.get('observations'), list):
            for event in transformed['observations']:
                if isinstance(event, dict):
                    event.pop('context_needed_for', None)
                    if event.get('kind') == GROUP:
                        event['kind'] = 'person_observation'
    old = legacy.assess_document(transformed)
    issues = list(old['issues'])
    def add(path, code): issues.append({'path': path, 'code': code})
    if not isinstance(raw, dict):
        return {'version': VERSION, 'valid': False, 'issues': issues, 'observations': [], 'entities_valid': False}
    if raw.get('version') != VERSION: add('/version', 'WRONG_VERSION')
    if set(raw) != {'version', 'entities', 'observations'}: add('/', 'FORBIDDEN_ROLE_OR_TOP_FIELD')
    obs = raw.get('observations') if isinstance(raw.get('observations'), list) else []
    entities = raw.get('entities') if isinstance(raw.get('entities'), list) else []
    if not 1 <= len(obs) <= LIMITS['observations']: add('/observations', 'OBSERVATION_BUDGET_OR_EMPTY')
    if len(entities) > LIMITS['entities']: add('/entities', 'ENTITY_BUDGET')
    checks = []
    for i, event in enumerate(obs):
        path = f'/observations/{i}'
        if not isinstance(event, dict):
            checks.append({'id': None, 'observation_valid': False, 'context_valid': False,
                           'relation_claim_present': False, 'relation_valid': False,
                           'local_relation_structurally_supported': False})
            continue
        scope = event.get('context_needed_for')
        scope_ok = (isinstance(scope, list) and all(isinstance(x, str) and x in CONTEXT for x in scope)
                    and len(scope) == len(set(scope)))
        if not scope_ok: add(path+'/context_needed_for', 'CONTEXT_SCOPE')
        relation = event.get('relation')
        if isinstance(relation, dict):
            o, ins = relation.get('object_ids'), relation.get('instrument_ids')
            if isinstance(o, list) and isinstance(ins, list) and all(isinstance(x, str) for x in o+ins) and set(o)&set(ins):
                add(path+'/relation', 'OBJECT_INSTRUMENT_OVERLAP')
            if scope_ok and 'relation' in scope and relation.get('state') == 'observed':
                add(path+'/relation', 'RELATION_NEEDS_CONTEXT')
        if scope_ok and 'phase' in scope and event.get('phase') != 'unknown':
            add(path+'/phase', 'OBSERVATION_CONTEXT_REQUIRED')
        local = [e for e in issues if e['path'] == path or e['path'].startswith(path+'/')]
        entity_valid = not any(e['path'].startswith('/entities') for e in issues)
        observation_valid = not any('/relation' not in e['path'] and '/context_' not in e['path'] for e in local)
        context_valid = not any('/context_' in e['path'] for e in local)
        relation_valid = relation is not None and entity_valid and context_valid and observation_valid and not local
        checks.append({'id': event.get('id'), 'derived_coarse_type': KINDS.get(event.get('kind')) if isinstance(event.get('kind'), str) else None,
                       'observation_valid': observation_valid, 'context_valid': context_valid,
                       'relation_claim_present': relation is not None, 'relation_valid': relation_valid,
                       'local_relation_structurally_supported': bool(relation_valid and relation.get('state') == 'observed'),
                       'independently_verified': False, 'cross_event_identity_verified': False})
    return {'version': VERSION, 'valid': not issues, 'issues': issues, 'observations': checks,
            'entities_valid': not any(e['path'].startswith('/entities') for e in issues)}


def examples():
    rows = copy.deepcopy(legacy.examples())
    for row in rows:
        row['version'] = VERSION
        for event in row['observations']: event['context_needed_for'] = []
    rows[4]['observations'][0]['context_needed_for'] = ['identity', 'relation']
    # Synthetic cases, not transcripts or preferred answers for the target cohort.
    group = copy.deepcopy(rows[1])
    group['observations'][0].update(kind=GROUP, evidence='Several people jostle; individual directed pairs are unclear.',
                                    context_needed_for=['relation'])
    group['entities'] = []
    rows.append(group)
    return rows


def prompt():
    return '''Inspect exactly eight chronological silent images T0..T7. Return one JSON object only.
Describe visible local observations, not an abuse verdict, anomaly score or graph match.
Never use movie knowledge or guess invisible identities. Preserve uncertainty and other-event observations.
Distinguish current interpersonal action, ongoing object constraint, residual injury, and object/environment activity.
A visible injury does not establish a current attacker. A held weapon alone does not establish its target.
Directed relations are local pairs. Do not require the whole clip to contain the same people.
For visible group activity with unclear pairings use group_interpersonal_observation with relation=null.
Do not invent one pair per possible actor. For an unidentified visible hand use a local person/body-part ID,
without assigning the hand to a named person in another image. Instruments are required evidence;
context_entity_ids are explicitly ancillary, never automatically reassigned to make a relation pass.
Shot/reverse-shot inference is unknown+edit_inferred, not observed. Use tracked only with visible continuity;
an ID repeated across edits is not evidence of that continuity. No cross-event links are requested.
context_needed_for lists only unresolved judgments requiring input outside these eight frames.
Null means no relation claimed, not normal. Unknown is not a negative label. Never force a relation
for an isolated person, damaged object or fire. Include at least one observation, even if unspecified/unknown.
Use at most 24 entities and 16 observations. All bins are sorted unique integers 0..7.
The exact interface and examples follow. Examples are synthetic syntax demonstrations, not target answers.
INTERFACE:
''' + json.dumps(specification(), ensure_ascii=True) + '\nEXAMPLES:\n' + json.dumps(examples(), ensure_ascii=True)
