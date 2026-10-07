"""One primary vocabulary for a prospective, local-observation interface.

This draft is validated offline. It does not enable acquisition or scoring.
"""
import copy

VERSION = 'v96_single_kind_local_observation_draft_1'
STATES = ('observed', 'not_observed', 'unknown')
STRENGTHS = ('directly_visible', 'tracked', 'edit_inferred', 'unresolved')
KINDS = {
    'directed_interpersonal': 'interpersonal_action',
    'reciprocal_interpersonal': 'interpersonal_action',
    'object_constraint': 'object_constraint',
    'injury_trace': 'injury_trace',
    'person_object_interaction': 'other_event',
    'object_interaction': 'other_event',
    'co_presence': 'other_event',
    'person_observation': 'other_event',
    'object_observation': 'other_event',
    'environment_observation': 'other_event',
    'unspecified_observation': 'unknown',
}
ROLES = ('actor_ids', 'target_ids', 'participant_ids', 'object_ids', 'instrument_ids')
OBSERVATION_ONLY = {'person_observation', 'object_observation', 'environment_observation', 'unspecified_observation'}


def specification():
    return {
        'version': VERSION, 'acquisition_enabled': False, 'status': 'offline_contract_draft',
        'primary_kind_to_derived_coarse_type': KINDS,
        'states': STATES, 'evidence_strengths': STRENGTHS,
        'entity': {'id': 'pN or oN', 'description': 'visible description', 'bins': 'observed T0..T7 bins',
                   'identity_scope': 'local_instance'},
        'observation': {
            'id': 'eN', 'kind': 'one key from the primary vocabulary', 'bins': 'T0..T7 bins',
            'phase': 'active/ongoing_constraint/residual/other/unknown', 'presence': STATES,
            'evidence': 'visible observation, not a narrative explanation',
            'relation': 'null if not claimed; otherwise explicit role lists, state, strength, bins, evidence',
            'context_entity_ids': 'explicit ancillary entities, excluded from relation arguments',
        },
        'relation': {**{r: 'explicit IDs; never automatically partition legacy lists' for r in ROLES},
                     'state': STATES, 'strength': STRENGTHS, 'bins': 'support bins within observation',
                     'evidence': 'evidence for this particular relation'},
        'role_cardinality': {
            'directed_interpersonal': 'one actor, one distinct target; optional explicit instruments',
            'reciprocal_interpersonal': 'at least two participants; no actor/target assignment',
            'object_constraint': 'one target, one constraint object; optional observed actor',
            'injury_trace': 'one attributed target; attacker not required',
            'person_object_interaction': 'one participant, one object',
            'object_interaction': 'two objects',
            'co_presence': 'at least two participants/objects; not a mechanism',
            'observation_only': 'relation must be null; a visible observation needs no invented counterpart',
        },
        'invariants': [
            'No separately generated event.type or relation.kind. Coarse type and role family are derived locally.',
            'Null relation means no relationship claim, not a negative observation or normal label.',
            'Instruments are necessary relation evidence; context entities are not instruments.',
            'Do not drop an out-of-bin object merely to obtain common visibility.',
            'Do not bundle independent targets into one single-frame relationship.',
            'A repeated entity ID is local bookkeeping, not proof of identity across edits or events.',
            'Edit-inferred claims remain weaker traces; they never become directly observed claims.',
            'No B5, normal, anomaly score, coercion verdict or probability is derived by this contract.',
            'Textual evidence and tracking declarations are model assertions, not independent semantic verification.',
        ],
        'future_event_links': 'Separate explicit identity/link evidence is required before any cross-event integration. Not authorized by this draft.',
    }


def assess_document(raw):
    """Check a prospective payload without deciding B5 or inferring hidden roles."""
    issues = []; event_checks = []
    def issue(path, code): issues.append({'path': path, 'code': code})
    def text(value): return isinstance(value, str) and bool(value.strip())
    def bins(value, empty=False):
        return isinstance(value, list) and (empty or bool(value)) and all(type(i) is int and 0 <= i < 8 for i in value) and value == sorted(set(value))
    def key(value, prefix): return isinstance(value, str) and value.startswith(prefix) and value[1:].isascii() and value[1:].isdigit() and not value[1:].startswith('0')
    def choice(value, options): return isinstance(value, str) and value in options
    if not isinstance(raw, dict): return {'valid': False, 'issues': [{'path': '/', 'code': 'OBJECT_REQUIRED'}], 'observations': []}
    if set(raw) != {'version', 'entities', 'observations'}: issue('/', 'UNEXPECTED_OR_MISSING_TOP_LEVEL_FIELD')
    if raw.get('version') != VERSION: issue('/version', 'WRONG_VERSION')
    entities = raw.get('entities'); entities = entities if isinstance(entities, list) else []
    if not isinstance(raw.get('entities'), list): issue('/entities', 'LIST_REQUIRED')
    ids = [e.get('id') for e in entities if isinstance(e, dict) and isinstance(e.get('id'), str)]
    table = {}
    for i, entity in enumerate(entities):
        path = f'/entities/{i}'
        if not isinstance(entity, dict): issue(path, 'OBJECT_REQUIRED'); continue
        uid = entity.get('id'); start = len(issues)
        if set(entity) != {'id', 'description', 'bins', 'identity_scope'}: issue(path, 'ENTITY_FIELDS')
        if not (key(uid, 'p') or key(uid, 'o')) or ids.count(uid) != 1: issue(path+'/id', 'ENTITY_ID')
        if not bins(entity.get('bins')): issue(path+'/bins', 'ENTITY_BINS')
        if not text(entity.get('description')): issue(path+'/description', 'EVIDENCE_REQUIRED')
        if entity.get('identity_scope') != 'local_instance': issue(path+'/identity_scope', 'NO_CROSS_EVENT_IDENTITY_AUTHORIZATION')
        if len(issues) == start: table[uid] = set(entity['bins'])
    observations = raw.get('observations'); observations = observations if isinstance(observations, list) else []
    if not isinstance(raw.get('observations'), list): issue('/observations', 'LIST_REQUIRED')
    eids = [e.get('id') for e in observations if isinstance(e, dict) and isinstance(e.get('id'), str)]
    for i, event in enumerate(observations):
        path = f'/observations/{i}'; start = len(issues)
        if not isinstance(event, dict): issue(path, 'OBJECT_REQUIRED'); continue
        if set(event) != {'id','kind','bins','phase','presence','evidence','relation','context_entity_ids'}: issue(path, 'OBSERVATION_FIELDS')
        kind = event.get('kind'); eb = event.get('bins')
        if not key(event.get('id'), 'e') or eids.count(event.get('id')) != 1: issue(path+'/id', 'EVENT_ID')
        if not choice(kind, KINDS): issue(path+'/kind', 'PRIMARY_KIND')
        if not bins(eb): issue(path+'/bins', 'EVENT_BINS')
        if not choice(event.get('phase'), {'active','ongoing_constraint','residual','other','unknown'}): issue(path+'/phase', 'PHASE')
        if not choice(event.get('presence'), STATES): issue(path+'/presence', 'PRESENCE')
        if not text(event.get('evidence')): issue(path+'/evidence', 'EVIDENCE_REQUIRED')
        observation_valid = len(issues) == start
        context = event.get('context_entity_ids')
        if not isinstance(context, list) or not all(isinstance(x,str) and x in table for x in context) or len(context) != len(set(context)):
            issue(path+'/context_entity_ids', 'CONTEXT_REFERENCES')
        relation = event.get('relation'); direct = False
        if relation is not None:
            rp = path+'/relation'; rs = len(issues)
            if not isinstance(relation, dict): issue(rp, 'OBJECT_OR_NULL_REQUIRED')
            else:
                if set(relation) != set(ROLES) | {'state','strength','bins','evidence'}: issue(rp, 'RELATION_FIELDS')
                if isinstance(kind,str) and kind in OBSERVATION_ONLY: issue(rp, 'OBSERVATION_NEEDS_NO_RELATION')
                role_values = {}
                for role in ROLES:
                    values = relation.get(role)
                    prefix = 'o' if role in ('object_ids','instrument_ids') else 'p'
                    valid = isinstance(values,list) and all(isinstance(x,str) and x in table and x.startswith(prefix) for x in values)
                    valid = valid and len(values) == len(set(values))
                    if not valid: issue(rp+'/'+role, 'ROLE_REFERENCES')
                    role_values[role] = values if valid else []
                a,t,p,o,instruments = (role_values[n] for n in ROLES)
                allowed = {
                    'directed_interpersonal': len(a)==1 and len(t)==1 and a!=t and not p and not o,
                    'reciprocal_interpersonal': len(p)>=2 and not a and not t and not o,
                    'object_constraint': len(t)==1 and len(o)==1 and len(a)<=1 and not set(a)&set(t) and not p,
                    'injury_trace': len(t)==1 and not a and not p and not o and not instruments,
                    'person_object_interaction': len(p)==1 and len(o)==1 and not a and not t,
                    'object_interaction': len(o)==2 and not a and not t and not p,
                    'co_presence': len(set(p+o))>=2 and not a and not t and not instruments,
                }
                if not isinstance(kind,str) or not allowed.get(kind, False): issue(rp, 'ROLE_CARDINALITY')
                refs = set(a+t+p+o+instruments)
                if isinstance(context,list) and all(isinstance(x,str) for x in context) and refs&set(context): issue(rp, 'CONTEXT_ARGUMENT_OVERLAP')
                bs = relation.get('bins')
                bounds = bins(bs, empty=True) and bins(eb) and set(bs)<=set(eb)
                if not bounds: issue(rp+'/bins', 'RELATION_BINS')
                state, strength = relation.get('state'), relation.get('strength')
                if not choice(state, STATES): issue(rp+'/state', 'RELATION_STATE')
                if not choice(strength, STRENGTHS): issue(rp+'/strength', 'EVIDENCE_STRENGTH')
                if not text(relation.get('evidence')): issue(rp+'/evidence', 'EVIDENCE_REQUIRED')
                if state == 'observed':
                    if strength not in ('directly_visible','tracked'): issue(rp+'/strength', 'INFERRED_IS_NOT_DIRECT')
                    if not bounds or not bs or not refs: issue(rp+'/bins', 'OBSERVED_SUPPORT_REQUIRED')
                    elif any(not table[x]&set(bs) for x in refs): issue(rp, 'NECESSARY_ARGUMENT_OUTSIDE_SUPPORT')
                    elif strength == 'directly_visible' and not set(bs).intersection(*(table[x] for x in refs)):
                        issue(rp, 'NO_COMMON_LOCAL_FRAME')
                    if event.get('presence') != 'observed': issue(rp, 'PRESENCE_RELATION_CONTRADICTION')
                direct = len(issues)==rs and observation_valid and state=='observed' and strength in ('directly_visible','tracked')
        event_checks.append({'id':event.get('id'),'observation_valid':observation_valid,
            'derived_coarse_type':KINDS.get(kind) if isinstance(kind,str) else None,
            'relation_claim_present':relation is not None,'local_relation_structurally_supported':direct,
            'independently_verified':False,'cross_event_identity_verified':False,'category_support':None})
    return {'version':VERSION,'valid':not issues,'issues':issues,'observations':event_checks,
            'scoring_authorized':False,'human_review_requested':False}


def examples():
    def entity(uid, bs): return {'id':uid,'bins':bs,'description':'Visible local instance '+uid,'identity_scope':'local_instance'}
    def relation(**roles):
        return {**{r:[] for r in ROLES},'state':'observed','strength':'directly_visible','bins':[4,5],
                'evidence':'Required people and pistol are visible together.', **roles}
    event = {'id':'e1','kind':'directed_interpersonal','bins':[3,4,5],'phase':'active','presence':'observed',
        'evidence':'One person points a pistol towards another person.','context_entity_ids':['o2'],
        'relation':relation(actor_ids=['p1'],target_ids=['p2'],instrument_ids=['o1'])}
    direct={'version':VERSION,'entities':[entity('p1',[3,4,5]),entity('p2',[4,5]),entity('o1',[4,5]),entity('o2',[3])],
            'observations':[event]}
    personal=copy.deepcopy(direct); personal.update(entities=[entity('p1',[0,1])])
    personal['observations']=[dict(event,kind='person_observation',bins=[0,1],evidence='A person looks aside.',relation=None,context_entity_ids=[])]
    wreck=copy.deepcopy(personal); wreck['entities']=[entity('o1',[0,1])]
    wreck['observations'][0].update(kind='object_observation',phase='residual',evidence='A damaged vehicle is visible.')
    blast=copy.deepcopy(personal); blast['entities']=[]
    blast['observations'][0].update(kind='environment_observation',evidence='Fire and smoke are visible; the source is unresolved.')
    inferred=copy.deepcopy(direct); inferred['entities']=[entity('p1',[0]),entity('p2',[7])]
    inferred['observations'][0].update(bins=[0,7],context_entity_ids=[],relation=relation(actor_ids=['p1'],target_ids=['p2'],
        state='unknown',strength='edit_inferred',bins=[0,7],evidence='Cutaway suggests a relation, but local binding is not established.'))
    injury=copy.deepcopy(personal); injury['observations'][0].update(kind='injury_trace',phase='residual',
        evidence='A blood trace is visible on a hand.',relation=relation(target_ids=['p1'],bins=[0,1],evidence='Trace and hand are visible together.'))
    return [direct,personal,wreck,blast,inferred,injury]
