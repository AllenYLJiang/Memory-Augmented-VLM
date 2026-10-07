"""Conservative legacy replay: normalize symbols, never invent observations."""
import copy
from collections import Counter

from .mechanism_v95_contract import assess as assess_v95
from .mechanism_v96_contract import KINDS

MODE_MAP = {'same_frame':'directly_visible', 'tracked_cross_frame':'tracked', 'inferred_edit':'edit_inferred', 'unknown':'unresolved'}
BINDING_TO_KIND = {k:k for k in KINDS}
BINDING_TO_KIND.update(person_object_constraint='object_constraint', injury_attribution='injury_trace')
LEGACY_COARSE = {'interpersonal_action','object_constraint','injury_trace','other_event','unknown'}


def replayable_shape(raw):
    if not isinstance(raw,dict) or not all(isinstance(raw.get(k),list) for k in ('persons','objects','events','links')):
        return False
    def bs(value): return isinstance(value,list) and all(type(x) is int for x in value)
    for item in raw['persons']+raw['objects']:
        if not isinstance(item,dict) or not isinstance(item.get('id'),str) or not bs(item.get('bins')): return False
    for event in raw['events']:
        if not isinstance(event,dict) or not isinstance(event.get('id'),str) or not isinstance(event.get('type'),str) or not bs(event.get('bins')): return False
        for key in ('binding','boundary','additional_evidence_needed'):
            if not isinstance(event.get(key),dict): return False
        if not all(isinstance(event['binding'].get(k),str) for k in ('kind','state','mode')) or not bs(event['binding'].get('evidence_bins')): return False
        for key in ('actor_ids','target_ids','participant_ids','object_ids'):
            if not isinstance(event.get(key),list) or not all(isinstance(v,str) for v in event[key]): return False
    return all(isinstance(link,dict) for link in raw['links'])


def symbol_replay(raw):
    """Only explicit vocabulary aliases. Raw and normalized payloads stay separate."""
    canonical = copy.deepcopy(raw); mappings=[]
    if not replayable_shape(raw): return canonical,mappings
    for i, event in enumerate(canonical.get('events', [])):
        kind = event.get('binding', {}).get('kind'); type_ = event.get('type')
        if type_ in ('person_object_interaction','co_presence') and kind == type_:
            event['type'] = 'other_event'
            mappings.append({'path':f'/events/{i}/type','raw_value':type_,'canonical_value':'other_event',
                'rule':'matching_fine_type_is_subtype_of_other_event','effect':'vocabulary_only_no_roles_or_evidence_changed'})
        binding = event.get('binding', {})
        if binding.get('kind') == 'unknown' and binding.get('state') in ('unknown','not_observed'):
            binding['kind'] = 'unresolved'
            mappings.append({'path':f'/events/{i}/binding/kind','raw_value':'unknown','canonical_value':'unresolved',
                'rule':'unknown_kind_with_nonobserved_state','effect':'unresolved_relation_not_observed_or_negative_category'})
    for i, link in enumerate(canonical.get('links', [])):
        if link.get('state') == 'inferred_edit':
            link['state'] = 'unknown'
            mappings.append({'path':f'/links/{i}/state','raw_value':'inferred_edit','canonical_value':'unknown',
                'rule':'misplaced_strength_cannot_establish_observed_state','canonical_strength':'edit_inferred',
                'effect':'conservative_unknown_state_strength_retained_in_trace'})
    # No label-conditioned mapping or role/identity repair is permitted.
    assert canonical.get('b5') == raw.get('b5')
    assert canonical.get('persons') == raw.get('persons') and canonical.get('objects') == raw.get('objects')
    for old, new in zip(raw.get('events', []), canonical.get('events', [])):
        for key in ('bins','actor_ids','target_ids','participant_ids','object_ids','boundary','additional_evidence_needed','evidence','phase','observation'):
            assert old.get(key) == new.get(key)
        assert old.get('binding', {}).get('evidence_bins') == new.get('binding', {}).get('evidence_bins')
    return canonical, mappings


def derive_window(raw, source_assessment=None):
    original = copy.deepcopy(raw)
    source = source_assessment or assess_v95(raw)
    if not replayable_shape(raw):
        reason={'event_index':None,'code':'LEGACY_COMPONENT_SHAPE_UNSUPPORTED',
                'detail':'original raw/field audit retained without attempting symbol conversion',
                'action':'inspect saved field masks offline; no deletion, identity repair or new request', 'severity':'unresolved'}
        return {'raw_b5':copy.deepcopy(raw.get('b5')) if isinstance(raw,dict) else None,
                'normalized_legacy_response':copy.deepcopy(raw),'mappings':[],
                'source_assessment':source,'alias_replay_assessment':copy.deepcopy(source),'observations':[],
                'links':[],'root_causes':[reason],'root_counts':{reason['code']:1},
                'new_visual_observation':False,'new_category_label':None,'unknown_is_normal':False,
                'training_loss_mask':False,'evaluation_loss_mask':False,'scoring_authorized':False}
    normalized, mappings = symbol_replay(raw)
    replay = assess_v95(normalized)
    entities = {e['id']:e for name in ('persons','objects') for e in raw.get(name, [])}
    events=[]; roots=[]
    def root(index, code, detail, action, severity='unresolved'):
        entry={'event_index':index,'code':code,'detail':detail,'action':action,'severity':severity}
        roots.append(entry)
        return entry
    for i, (old, new, old_check, new_check) in enumerate(zip(raw['events'],normalized['events'],source['events'],replay['events'])):
        start=len(roots); b=old['binding']; nb=new['binding']; coarse=new['type']; kind=nb['kind']
        primary=BINDING_TO_KIND.get(kind)
        if primary and KINDS[primary] != coarse:
            root(i,'TYPE_KIND_CONFLICT',{'type':coarse,'kind':kind},'keep coarse observation; do not choose a convenient relationship')
            primary=None
        if kind=='unresolved':
            primary={'injury_trace':'injury_trace','object_constraint':'object_constraint','other_event':'unspecified_observation','unknown':'unspecified_observation'}.get(coarse)
        if coarse not in LEGACY_COARSE:
            root(i,'UNMAPPED_PRIMARY_TYPE',{'raw_type':old['type'],'raw_kind':b['kind']},'new schema or explicit clarification; no alias guessing')
        if b['kind']=='unknown' and b['state']=='observed':
            root(i,'UNKNOWN_KIND_CANNOT_BE_OBSERVED',{},'preserve claim; no automatic upgrade to a relationship')
        local_mappings=[m for m in mappings if m['path'].startswith(f'/events/{i}/')]
        if local_mappings:
            root(i,'SYMBOL_ALIAS',{'mappings':local_mappings},'separate normalized view only','encoding_only')
        role_names=('actor_ids','target_ids','participant_ids','object_ids')
        roles={k:copy.deepcopy(old[k]) for k in role_names}
        refs=set(x for values in roles.values() for x in values)
        support=set(b['evidence_bins'])
        per_entity={uid:sorted(set(entities[uid]['bins']) & support) for uid in refs if uid in entities}
        absent=[uid for uid in refs if not per_entity.get(uid)]
        common=sorted(support.intersection(*(set(entities[uid]['bins']) for uid in refs))) if refs and all(uid in entities for uid in refs) else []
        people=set(roles['actor_ids']+roles['target_ids']+roles['participant_ids'])
        people_common=sorted(support.intersection(*(set(entities[uid]['bins']) for uid in people))) if people and all(uid in entities for uid in people) else []
        if roles['object_ids'] and kind in ('directed_interpersonal','reciprocal_interpersonal','co_presence'):
            root(i,'LEGACY_OBJECT_ROLES_UNPARTITIONED',{'objects':roles['object_ids']},
                'instruments versus ancillary objects were not declared; preserve all IDs; do not auto-assign')
        if absent:
            root(i,'ARGUMENT_OUTSIDE_SUPPORT',{'ids':sorted(absent),'per_entity_support':per_entity},'do not expand bins or remove a referenced object to pass')
        if b['state']=='observed' and b['mode']=='inferred_edit':
            root(i,'EDIT_INFERENCE_DECLARED_OBSERVED',{'state':b['state'],'mode':b['mode']},'retain weaker edit trace; no direct binding authorization')
        if b['state']=='observed' and b['mode']=='same_frame' and refs and not common:
            root(i,'NO_COMMON_FRAME_FOR_ALL_ARGUMENTS',{'per_entity_support':per_entity},'check separate relation instances, not a union of all objects/people')
        if len(roles['target_ids'])>1 and kind in ('person_object_constraint','injury_attribution','directed_interpersonal'):
            root(i,'MULTI_TARGET_PARTITION_UNDECLARED',{'targets':roles['target_ids'],'candidate_bins_not_relations':per_entity},
                'no automatic relationship splitting; candidates are only intersections of existing bins')
        if kind=='co_presence' and len(refs)<2:
            root(i,'SINGLE_ENTITY_IS_AN_OBSERVATION',{'references':sorted(refs)},'future payload may use observation-only kind; do not invent another entity')
        if kind=='object_interaction' and len(roles['object_ids'])<2:
            root(i,'OBJECT_STATE_OR_ENVIRONMENT_WITHOUT_PAIR',{'objects':roles['object_ids']},'preserve visible observation; no fabricated second object')
        if kind=='directed_interpersonal' and (not roles['actor_ids'] or not roles['target_ids']):
            root(i,'MISSING_DIRECTED_ROLE',roles,'keep local action observation; unknown counterpart stays unknown')
        if not new_check['binding_valid']:
            root(i,'RELATION_STILL_INVALID_AFTER_ALIASES',{'paths':[v['path'] for v in replay['issues'] if v['path'].startswith(f'/events/{i}/binding/') or v['path'].startswith(f'/events/{i}/actor_ids') or v['path'].startswith(f'/events/{i}/target_ids')]},
                'diagnose role and strength evidence; do not retry by label preference')
        if not old_check['support_consistent']:
            root(i,'CATEGORY_SUPPORT_UNRESOLVED',{'boundary':old['boundary'],'needed':old['additional_evidence_needed']},
                'category claim is not repaired by vocabulary normalization')
        event_alias_only=not old_check['valid'] and new_check['valid']
        events.append({
            'index':i,'id':old['id'],'raw':old,'primary_kind':primary,
            'coarse_observation_type':coarse if coarse in LEGACY_COARSE else None,
            'coarse_type_is_locally_derived_for_new_payloads':True,
            'symbol_mappings':local_mappings,
            'observation_fields_available':new_check['observation_valid'],
            'source_observation_fields_available':old_check['observation_valid'],
            'source_event_valid':old_check['valid'],'alias_replay_event_valid':new_check['valid'],
            'alias_only_format_recovery':event_alias_only,
            'relation':{'raw_claim':b,'canonical_strength':MODE_MAP.get(b['mode']),
                'source_roles':roles,'instruments':None,'ancillary_context_objects':None,
                'object_role_partition_known':not bool(roles['object_ids']),
                'all_references_common_bins':common,'per_entity_support_bins':per_entity,
                'people_only_common_bins_diagnostic_not_proof':people_common,
                'source_binding_valid':old_check['binding_valid'],'alias_replay_binding_valid':new_check['binding_valid'],
                'identity_scope':'legacy_local_claim_only','cross_event_identity_verified':False,
                'independently_verified':False,'new_native_payload_ready':False},
            'category':{'raw':old['boundary'],'source_local_support':old_check['supported'],
                        'alias_replay_local_support':new_check['supported'],
                        'new_category_evidence':None,'deployment_support':None},
            'root_causes':roots[start:],
        })
    links=[]
    for i, link in enumerate(raw['links']):
        strength='edit_inferred' if link.get('state')=='inferred_edit' else 'unresolved'
        links.append({'index':i,'raw':link,'normalized':normalized['links'][i],
            'canonical_strength':strength,'strength_inferred_from_state':strength=='edit_inferred',
            'source_valid':source['links'][i]['valid'],'alias_replay_valid':replay['links'][i]['valid'],
            'source_issue_paths':[x['path'] for x in source['issues'] if x['path'].startswith(f'/links/{i}/')],
            'identity_verified':False,'new_observed_link_authorized':False})
    if not source['fields']['/b5/consistency']['valid']:
        root(None,'WINDOW_LABEL_WITHOUT_ACCEPTED_LOCAL_BASIS',raw['b5'],'do not modify label, probability or basis to satisfy gate')
    assert raw == original
    return {'raw_b5':copy.deepcopy(raw['b5']),'normalized_legacy_response':normalized,'mappings':mappings,
        'source_assessment':source,'alias_replay_assessment':replay,'observations':events,'links':links,'root_causes':roots,
        'root_counts':dict(Counter(x['code'] for x in roots)),
        'new_visual_observation':False,'new_category_label':None,'unknown_is_normal':False,
        'training_loss_mask':False,'evaluation_loss_mask':False,'scoring_authorized':False}
