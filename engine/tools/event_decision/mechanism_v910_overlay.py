"""A bounded R1 evidence overlay, never a replacement for native model evidence."""
import copy

from .contracts import semantic_sha256
from .mechanism_v98_contract import assess, metrics
from .mechanism_v99_review import validate_return

VERSION = 'v910_r1_scoped_human_overlay_1'
REVIEW_SHA256 = '714475ffea23a9b81f55494f32dd661c0ab5d69c5b7a6eae9aad6d0a1f0c988a'
PACKET_ID = '7092b188c2953a7ad210249410786a78078d26db9d7aeb608535602a1a7ff32c'
PLAN = {
    ('ced3cdd508be', 'q6_identity'): ('e6', 'same_people_visible'),
    ('eca2847f1c43', 'q3_object_role'): ('e3', 'contact_object'),
    ('53d4dc5131fe', 'q1_reaction'): ('e1', 'reaction_only'),
    ('53d4dc5131fe', 'q4_directness'): ('e4', 'edit_inference_only'),
}


def policy():
    return {
        'version': VERSION, 'review_file_sha256': REVIEW_SHA256, 'packet_id': PACKET_ID,
        'plan': [dict(case_id=k[0], question_id=k[1], observation_id=v[0], choice=v[1])
                 for k, v in PLAN.items()],
        'identity_rule': 'R1 support for e6/T7 only; never expand the shared entity table.',
        'object_rule': 'Remove the reviewed object from ancillary context for e3 only.',
        'inference_rule': 'Retain the event hypothesis as unknown + edit_inferred, not normal.',
        'raw_text_rule': 'Original wording is retained verbatim, not relabeled as human-verified prose.',
        'scope_rule': 'Review bins are consulted evidence, not joint visibility of all arguments.',
        'authorization': 'User requested zero-API provenance overlay and all-36 offline audit.',
        'scoring_authorized': False, 'training_authorized': False,
        'ready_for_shadow_integration': False, 'new_review_task_authorized': False,
    }


def pointer_get(obj, path):
    for part in path.strip('/').split('/'):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def pointer_set(obj, path, value):
    parts = path.strip('/').split('/')
    parent = obj
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
    parent[key] = copy.deepcopy(value)


def replay_patches(payload, changes, reverse=False):
    result = copy.deepcopy(payload)
    for change in reversed(changes) if reverse else changes:
        before, after = ('after', 'before') if reverse else ('before', 'after')
        if pointer_get(result, change['path']) != change[before]:
            raise ValueError('Patch precondition mismatch: '+change['path'])
        pointer_set(result, change['path'], change[after])
    return result


def scoped_assessment(payload, supports):
    """Revalidate only the reviewed observation against its local human support view."""
    result = assess(payload)
    seen = set()
    for support in supports:
        oid = support['observation_id']
        if oid in seen:
            raise ValueError('Duplicate scoped support')
        seen.add(oid)
        indices = [i for i, e in enumerate(payload['observations']) if e['id'] == oid]
        if len(indices) != 1:
            raise ValueError('Scoped observation is not unique')
        i = indices[0]
        relation = payload['observations'][i]['relation']
        if support['bins'] != relation['bins'] or not set(support['bins']) <= set(support['review_bins']):
            raise ValueError('Scoped support must match the reviewed relation bins')
        local = copy.deepcopy(payload)
        entity = next(e for e in local['entities'] if e['id'] == support['entity_id'])
        if entity['bins'] != support['model_entity_bins']:
            raise ValueError('Model entity support changed')
        entity['bins'] = sorted(set(entity['bins']) | set(support['bins']))
        local_check = assess(local)
        prefix = f'/observations/{i}'
        belongs = lambda issue: issue['path'] == prefix or issue['path'].startswith(prefix+'/')
        # Entity support in this temporary view cannot repair any other observation.
        result['issues'] = [x for x in result['issues'] if not belongs(x)] + [
            x for x in local_check['issues'] if belongs(x)]
        result['observations'][i] = local_check['observations'][i]
    result['valid'] = not result['issues']
    return result


def apply_overlay(records, packet, review, private_map, provenance):
    validate_return(packet, review)
    if packet['packet_id'] != PACKET_ID or review['reviewer_id'] != 'R1':
        raise ValueError('This version is bound to the audited R1 packet')
    if len(records) != 36 or len({r['window_uid'] for r in records}) != 36:
        raise ValueError('All 36 unique source windows are required')
    if len(private_map) != 3 or set(private_map) != {k[0] for k in PLAN} or len(set(private_map.values())) != 3:
        raise ValueError('Private case binding changed')
    answers = {(a['case_id'], a['question_id']): a for a in review['answers']}
    questions = {(c['case_id'], q['question_id']): (c, q) for c in packet['cases'] for q in c['questions']}
    if set(questions) != set(PLAN) or set(answers) != set(PLAN):
        raise ValueError('The complete fixed four-question plan is required')
    original_hash = semantic_sha256(records)
    output = copy.deepcopy(records)
    by_uid = {r['window_uid']: r for r in output}
    for row in output:
        if assess(row['compatible_payload']) != row['compatible_assessment']:
            raise ValueError('Saved V99 assessment disagrees with replay')
        row.update(overlay_version=VERSION, review_payload=copy.deepcopy(row['compatible_payload']),
                   review_assertions=[], field_changes=[], scoped_entity_support=[],
                   payload_sha256_before=semantic_sha256(row['compatible_payload']))
    for key, (oid, choice) in PLAN.items():
        answer = answers[key]
        case, question = questions[key]
        if answer['choice'] != choice or question['observation_id'] != oid:
            raise ValueError('Changed answer requires a new explicit repair plan')
        row = by_uid[private_map[key[0]]]
        payload = row['review_payload']
        indices = [i for i, o in enumerate(payload['observations']) if o['id'] == oid]
        if len(indices) != 1:
            raise ValueError('Question observation is not unique')
        i = indices[0]
        original_event = row['compatible_payload']['observations'][i]
        if question['observation'] != original_event or question['entities'] != payload['entities']:
            raise ValueError('Question does not match the frozen source payload')
        if key[1] in ('q6_identity', 'q3_object_role') and not set(original_event['bins']) <= set(answer['evidence_bins']):
            raise ValueError('Positive local support must include the question support bins')
        assertion_id = key[0]+'/'+key[1]
        assertion = {
            'assertion_id': assertion_id, 'case_id': key[0], 'question_id': key[1],
            'observation_id': oid, 'reviewer_id': review['reviewer_id'], 'choice': choice,
            'review_bins': answer['evidence_bins'],
            'review_frame_indices': [case['frame_indices'][b] for b in answer['evidence_bins']],
            'question_bins': original_event['bins'], 'notes_verbatim': answer['notes'],
            'original_observation_sha256': semantic_sha256(original_event),
            'source_compatible_payload_sha256': row['payload_sha256_before'],
            'source_record_sha256': semantic_sha256(records[[r['window_uid'] for r in records].index(row['window_uid'])]),
            'provenance': copy.deepcopy(provenance),
            'evidence_scope': 'eight_frames', 'numeric_event_probability': None,
            'anomaly_label': None, 'new_vlm_evidence': False,
        }
        row['review_assertions'].append(assertion)

        def change(suffix, before, after, rule):
            path = f'/observations/{i}/'+suffix
            if pointer_get(payload, path) != before:
                raise ValueError('Repair precondition mismatch: '+path)
            row['field_changes'].append(dict(path=path, before=copy.deepcopy(before),
                after=copy.deepcopy(after), rule=rule, assertion_id=assertion_id))
            pointer_set(payload, path, after)

        if key[1] == 'q6_identity':
            rel = original_event['relation']
            if (original_event['kind'] != 'reciprocal_interpersonal' or rel['participant_ids'] != ['p3', 'p4']
                    or rel['bins'] != [7]):
                raise ValueError('Identity plan is only for the frozen p3/p4 e6 T7 claim')
            entity = next(e for e in payload['entities'] if e['id'] == 'p3')
            if entity['bins'] != [2, 5, 6]:
                raise ValueError('Unexpected original p3 visibility')
            row['scoped_entity_support'].append({
                'observation_id': oid, 'entity_id': 'p3', 'bins': [7],
                'model_entity_bins': [2, 5, 6], 'review_bins': answer['evidence_bins'],
                'assertion_id': assertion_id, 'authority': 'R1_answer_to_e6_T7_identity_question',
                'applies_to_shared_entity_table': False, 'cross_event_identity_verified': False,
            })
            assertion['supports'] = {'local_identity_and_contact': True}
            assertion['unverified'] = ['reciprocal_action_semantics', 'cross_event_identity', 'anomaly_category']
        elif key[1] == 'q3_object_role':
            rel = original_event['relation']
            if (original_event['kind'] != 'person_object_interaction' or rel['object_ids'] != ['o2']
                    or rel['participant_ids'] != ['p4'] or original_event['bins'] != [3]):
                raise ValueError('Object role plan is only for p4/o2 e3 T3')
            change('context_entity_ids', ['o2'], [], 'R1_contact_object_not_ancillary_in_this_observation')
            assertion['supports'] = {'contact_object': True}
            assertion['unverified'] = ['voluntary_gripping', 'shooter_victim_causal_chain', 'anomaly_category']
        else:
            strength = 'directly_visible' if key[1] == 'q1_reaction' else 'edit_inferred'
            change('relation/state', 'observed', 'unknown', 'R1_inference_not_direct_contact')
            if strength == 'directly_visible':
                change('relation/strength', strength, 'edit_inferred', 'R1_reaction_based_event_hypothesis')
            change('presence', 'observed', 'unknown', 'Event_mechanism_is_inferred_not_directly_observed')
            assertion['supports'] = {'direct_contact_evidenced': False, 'event_inference': True}
            assertion['unverified'] = ['exact_contact_frame', 'anomaly_category']
            assertion['interpretation'] = 'Unknown direct evidence is not event absence or a normal label.'

    for row in output:
        before, after = row['compatible_payload'], row['review_payload']
        if replay_patches(before, row['field_changes']) != after or replay_patches(after, row['field_changes'], True) != before:
            raise ValueError('Non-reversible or undeclared field changes')
        if before['entities'] != after['entities']:
            raise ValueError('Human support must never expand the shared entity table')
        for a, b in zip(before['observations'], after['observations']):
            for field in ('id', 'kind', 'bins', 'phase', 'evidence', 'context_needed_for'):
                if a[field] != b[field]: raise ValueError('Unreviewed observation field changed')
            if a['relation'] is not None:
                for field in set(a['relation']) - {'state', 'strength'}:
                    if a['relation'][field] != b['relation'][field]: raise ValueError('Unreviewed relation field changed')
        row['payload_sha256_after'] = semantic_sha256(after)
        row['review_projection_assessment'] = assess(after)
        row['human_scoped_assessment'] = scoped_assessment(after, row['scoped_entity_support'])
        row['round_trip_verified'] = True
        row['shared_entities_unchanged'] = True
        row['unreviewed_payload_unchanged'] = bool(row['review_assertions']) or before == after
        row['new_inference'] = False
    if semantic_sha256(records) != original_hash:
        raise ValueError('Source records mutated')
    return output


def summarize(rows):
    before = metrics([r['compatible_assessment'] for r in rows])
    projected = metrics([r['review_projection_assessment'] for r in rows])
    scoped = metrics([r['human_scoped_assessment'] for r in rows])
    regressed = [r['window_uid'] for r in rows if r['compatible_assessment']['valid'] and not r['human_scoped_assessment']['valid']]
    if regressed: raise ValueError('Human overlay regressed a valid document')
    assertions = [a for r in rows for a in r['review_assertions']]
    return {
        'version': VERSION, 'execution': 'all_36_saved_windows_human_overlay_audited',
        'windows': len(rows), 'videos': len({r['video_id'] for r in rows}),
        'source_groups': len({r['source_group'] for r in rows}),
        'v99_original_compatibility_metrics_unchanged': before,
        'review_field_projection_metrics': projected,
        'human_scoped_contract_metrics_not_native_performance': scoped,
        'reviewed_windows': sum(bool(r['review_assertions']) for r in rows),
        'reviewed_questions': len(assertions),
        'unreviewed_windows_unchanged': sum(not r['review_assertions'] and r['compatible_payload'] == r['review_payload'] for r in rows),
        'field_changes': sum(len(r['field_changes']) for r in rows),
        'scoped_support_assertions': sum(len(r['scoped_entity_support']) for r in rows),
        'all_36_round_trip_verified': all(r['round_trip_verified'] for r in rows),
        'all_shared_entity_tables_unchanged': all(r['shared_entities_unchanged'] for r in rows),
        'regressed_documents': regressed,
        'remaining_contract_failures': [r['window_uid'] for r in rows if not r['human_scoped_assessment']['valid']],
        'review_limitations': [{'assertion_id': a['assertion_id'], 'unverified': a['unverified']} for a in assertions],
        'human_scoped_technical_checks_pass': scoped['technical_proportions_and_critical_checks_pass'],
        'formal_gate_overridden': False, 'formal_review_active': False,
        'human_review_requested': False, 'ready_for_shadow_integration': False,
        'scoring_authorized': False, 'training_authorized': False,
        'remote_execution_authorized': False, 'remote_calls': 0, 'new_media_decodes': 0,
        'new_model_responses': 0, 'formal_accuracy': None, 'formal_AP': None,
        'decision': 'OFFLINE_HUMAN_OVERLAY_AUDITED_NO_INTEGRATION_AUTHORITY',
        'claim_limit': 'Single-reviewer development evidence overlay; contract validity is not visual truth or accuracy.',
    }
