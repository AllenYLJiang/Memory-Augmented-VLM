"""Read-only development sidecars; model claims and human support remain separate."""
import copy
from collections import Counter

from .contracts import semantic_sha256

VERSION = 'v911_readonly_development_trace_1'
TRACE_KEY = 'v911_development_diagnostic_trace'
IDENTITY = ('window_uid', 'video_id', 'source_group', 'start_frame',
            'end_frame_exclusive', 'sampled_frame_indices')
SUPPORT_FIELDS = ('local_identity', 'direct_contact_evidenced', 'contact_object',
                  'event_inference', 'kind_semantics', 'phase_semantics',
                  'context_scope', 'cross_event_identity', 'anomaly_category')


def claim_mode(event):
    rel = event.get('relation')
    if rel is None: return 'observation_only'
    if rel['strength'] == 'edit_inferred': return 'inferred_relation'
    if rel['state'] == 'observed':
        return {'directly_visible': 'direct_relation_claim', 'tracked': 'tracked_relation_claim'}.get(
            rel['strength'], 'observed_relation_other_strength')
    return 'unresolved_or_nonobserved_relation'


def field_support(assertions):
    fields = {key: {'value': None, 'reviewed': False, 'assertion_ids': []} for key in SUPPORT_FIELDS}
    aliases = {'local_identity_and_contact': ('local_identity', 'direct_contact_evidenced'),
               'contact_object': ('contact_object', 'direct_contact_evidenced'),
               'direct_contact_evidenced': ('direct_contact_evidenced',),
               'event_inference': ('event_inference',)}
    for assertion in assertions:
        for key, value in assertion['supports'].items():
            if key not in aliases or type(value) is not bool:
                raise ValueError('Unknown semantic support assertion')
            for target in aliases[key]:
                field = fields[target]
                if field['reviewed'] and field['value'] != value:
                    raise ValueError('Conflicting human support must not be resolved automatically')
                field.update(value=value, reviewed=True)
                field['assertion_ids'].append(assertion['assertion_id'])
    return fields


def observation_view(event, check):
    return {'kind': event['kind'], 'phase': event['phase'], 'presence': event['presence'],
            'bins': event['bins'], 'evidence': event['evidence'], 'relation': event['relation'],
            'context_entity_ids': event['context_entity_ids'], 'context_needed_for': event['context_needed_for'],
            'claim_mode': claim_mode(event), 'structural_check': check}


def build_traces(rows, selection, source_provenance):
    if len(rows) != 36 or len({r['window_uid'] for r in rows}) != 36:
        raise ValueError('Full unique 36-window development cohort required')
    if [r['window_uid'] for r in rows] != [r['window_uid'] for r in selection]:
        raise ValueError('Selection/order differs from the frozen overlay')
    traces = []
    for row, selected in zip(rows, selection):
        for key in IDENTITY:
            if row[key] != selected[key]: raise ValueError('Selection identity mismatch: '+key)
        old = row['compatible_payload']['observations']
        new = row['review_payload']['observations']
        if len(old) != len(new) or [e['id'] for e in old] != [e['id'] for e in new]:
            raise ValueError('Observations removed/reordered by overlay')
        native_obs = row['original_model_payload']['observations']
        if [e['id'] for e in native_obs] != [e['id'] for e in old]:
            raise ValueError('Native observation identities changed')
        observations = []
        for i, (before, after) in enumerate(zip(old, new)):
            oid = after['id']
            assertions = [a for a in row['review_assertions'] if a['observation_id'] == oid]
            support = field_support(assertions)
            check = row['human_scoped_assessment']['observations'][i]
            if any(a['id'] != oid for a in (check, row['compatible_assessment']['observations'][i],
                                            row['original_model_assessment']['observations'][i])):
                raise ValueError('Assessment/observation join mismatch')
            flags = []
            if assertions: flags.append('human_reviewed_field')
            if claim_mode(after) == 'inferred_relation': flags.append('event_inference_not_direct_contact')
            if after['context_needed_for']: flags.append('context_dependent_judgment')
            if after['presence'] == 'unknown' or (after['relation'] and after['relation']['state'] == 'unknown'):
                flags.append('unknown_not_normal')
            if after['kind'] == 'injury_trace' or after['phase'] == 'residual': flags.append('residual_not_current_mechanism')
            if after['kind'] == 'object_constraint': flags.append('ongoing_constraint_not_necessarily_current_actor')
            if not support['kind_semantics']['reviewed']: flags.append('kind_not_human_verified')
            observations.append({
                'id': oid,
                'native_model': observation_view(native_obs[i], row['original_model_assessment']['observations'][i]),
                'v99_compatible': observation_view(before, row['compatible_assessment']['observations'][i]),
                'review_view': observation_view(after, check),
                'semantic_support': support,
                'human_assertions': copy.deepcopy(assertions),
                'scoped_entity_support': [s for s in row['scoped_entity_support'] if s['observation_id'] == oid],
                'field_changes': [c for c in row['field_changes'] if c['path'].startswith(f'/observations/{i}/')],
                'diagnostic_flags': flags,
                'observation_semantic_verified': False, 'relation_semantic_verified': False,
                'training_loss_mask': False, 'evaluation_loss_mask': False,
            })
        traces.append({
            'version': VERSION, **{k: copy.deepcopy(row[k]) for k in IDENTITY},
            'source_provenance': source_provenance,
            'source_overlay_record_sha256': semantic_sha256(row),
            'original_model_payload_sha256': row['original_model_payload_sha256'],
            'scope': 'previously_exposed_development_diagnostic_only',
            'human_reviewed_questions': len(row['review_assertions']),
            'whole_window_semantically_reviewed': False,
            'observations': observations,
            'shared_entities': copy.deepcopy(row['compatible_payload']['entities']),
            'native_entities': copy.deepcopy(row['original_model_payload']['entities']),
            'selection_reasons': selected['selection_reasons'],
            'category_label': None, 'anomaly_score': None,
            'training_loss_mask': False, 'evaluation_loss_mask': False,
            'cross_event_identity_verified': False,
        })
    return traces


def join_rows(predictions, traces):
    """Append one namespaced sidecar; every original field must survive round-trip."""
    by_uid = {t['window_uid']: t for t in traces}
    if len(by_uid) != len(traces): raise ValueError('Duplicate trace windows')
    if len(predictions) != len(traces): raise ValueError('Exact full-cohort prediction join required')
    before_hash = semantic_sha256(predictions)
    seen = set()
    joined = []
    for pred in predictions:
        uid = pred.get('window_uid')
        if uid not in by_uid or uid in seen: raise ValueError('Unknown/duplicate prediction window')
        seen.add(uid)
        if TRACE_KEY in pred: raise ValueError('Existing diagnostic sidecar will not be overwritten')
        trace = by_uid[uid]
        for key in IDENTITY:
            if key not in pred or semantic_sha256(pred[key]) != semantic_sha256(trace[key]):
                raise ValueError('Exact prediction identity required: '+key)
        value = copy.deepcopy(pred)
        value[TRACE_KEY] = copy.deepcopy(trace)
        joined.append(value)
    restored = [{k: v for k, v in p.items() if k != TRACE_KEY} for p in joined]
    if semantic_sha256(restored) != before_hash or semantic_sha256(predictions) != before_hash:
        raise ValueError('Prediction content changed during sidecar join')
    return joined, {
        'predictions': len(predictions), 'input_semantic_sha256': before_hash,
        'restored_semantic_sha256': semantic_sha256(restored), 'all_original_fields_preserved': True,
        'changed_existing_fields': 0, 'duplicate_windows': 0, 'missing_windows': 0,
        'comparison': 'All original keys, values and row order, not only a score allow-list.',
    }


def inventory(traces):
    obs = [o for t in traces for o in t['observations']]
    distribution = {}
    for layer in ('native_model', 'v99_compatible', 'review_view'):
        distribution[layer] = {
            'kind_counts': dict(Counter(o[layer]['kind'] for o in obs)),
            'phase_counts': dict(Counter(o[layer]['phase'] for o in obs)),
            'presence_counts': dict(Counter(o[layer]['presence'] for o in obs)),
            'claim_mode_counts': dict(Counter(o[layer]['claim_mode'] for o in obs)),
        }
    review_counts = {f: {'supported': sum(o['semantic_support'][f]['value'] is True for o in obs),
                         'not_supported': sum(o['semantic_support'][f]['value'] is False for o in obs),
                         'not_reviewed': sum(o['semantic_support'][f]['value'] is None for o in obs)} for f in SUPPORT_FIELDS}
    families = ('human_reviewed_field', 'event_inference_not_direct_contact', 'context_dependent_judgment',
                'unknown_not_normal', 'residual_not_current_mechanism',
                'ongoing_constraint_not_necessarily_current_actor', 'kind_not_human_verified')
    coverage = {}
    for flag in families:
        windows = [t for t in traces if any(flag in o['diagnostic_flags'] for o in t['observations'])]
        coverage[flag] = {'observations': sum(flag in o['diagnostic_flags'] for o in obs),
                          'windows': len(windows), 'source_groups': len({t['source_group'] for t in windows}),
                          'window_uids': [t['window_uid'] for t in windows]}
    return {
        'version': VERSION, 'windows': len(traces), 'videos': len({t['video_id'] for t in traces}),
        'source_groups': len({t['source_group'] for t in traces}), 'observations': len(obs),
        'distribution_of_claims_not_visual_truth': distribution,
        'field_review_coverage': review_counts, 'diagnostic_family_coverage': coverage,
        'whole_window_semantic_reviews': 0,
        'purpose': 'All-cohort diagnostic availability, not error rates or candidate selection for accuracy.',
    }
