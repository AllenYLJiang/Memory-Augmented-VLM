"""Build a standalone development reader, without changing any formal integration gate."""
import copy
import re
import shutil
from pathlib import Path

from .contracts import file_sha256, semantic_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v98_offline import tree_hashes
from .mechanism_v910_offline import verify as verify_overlay, jsonl
from .mechanism_v911_trace import VERSION, IDENTITY, TRACE_KEY, build_traces, inventory, join_rows
from .mechanism_v911_report import case_page, index_page

DEFAULT_SOURCE = 'governed_v910_human_evidence_overlay_20260915'
DEFAULT_TAG = 'governed_v911_readonly_diagnostic_trace_20260915'


def owned(project):
    return [project/'tools/event_decision'/('mechanism_v911_'+s+'.py') for s in ('trace', 'offline', 'report')] + [
        project/'tools/mechanism_v911_cli.py', project/'run_mechanism_v911_offline.sh',
        project/'tests/test_mechanism_v911.py']


def check_source(project, source):
    protocol = read_json(source/'protocol.json')
    v99 = portable(protocol['source_run']).resolve()
    review = portable(protocol['review_file']).resolve()
    summary = verify_overlay(project, v99, source, review)
    if summary.get('human_scoped_technical_checks_pass') is not True:
        raise ValueError('Human-overlay technical checks must pass before diagnostic export')
    # This is not the V97 formal semantic-review gate and does not modify it.
    v98 = portable(read_json(v99/'protocol.json')['source_run']).resolve()
    v97 = portable(read_json(v98/'protocol.json')['source_run']).resolve()
    return v97, summary


def hash_outputs(out):
    return {k: v for k, v in tree_hashes(out).items() if k != 'completion.json'}


def paths(project, source, out, predictions):
    if (out.parent != (project/'runs').resolve() or source == out or source in out.parents
            or out in source.parents or (predictions and out in predictions.parents)):
        raise ValueError('Independent direct runs TAG required; no source/input overwrite')


def input_predictions(selection, predictions_path):
    if predictions_path is not None:
        import json
        def unique(items):
            result = {}
            for key, value in items:
                if key in result: raise ValueError('Duplicate prediction JSON key: '+key)
                result[key] = value
            return result
        def nonfinite(value): raise ValueError('Non-finite prediction JSON value: '+value)
        records = [json.loads(line, object_pairs_hook=unique, parse_constant=nonfinite)
                   for line in predictions_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        return records, {'kind': 'explicit_saved_predictions', 'path': str(predictions_path),
                                         'sha256': file_sha256(predictions_path)}
    if any(not isinstance(s.get('baseline'), dict) for s in selection):
        raise ValueError('No saved baseline for every selected window')
    return [copy.deepcopy(s['baseline']) for s in selection], {
        'kind': 'historical_B5_sparse_frame_screening_snapshots_not_graph_scores_or_GT',
        'path': None, 'sha256': semantic_sha256([s['baseline'] for s in selection]),
    }


def verify(project, source, out, predictions=None):
    project, source, out = (Path(p).resolve() for p in (project, source, out))
    predictions = Path(predictions).resolve() if predictions else None
    paths(project, source, out, predictions)
    protocol = read_json(out/'protocol.json', {})
    if protocol.get('version') != VERSION or protocol.get('operation') != 'readonly_development_sidecar_all_36':
        raise ValueError('Not a V9.11 diagnostic TAG')
    if read_json(out/'integrity.json', {}).get('protocol_sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('Diagnostic protocol changed')
    if portable(protocol['source_run']).resolve() != source or tree_hashes(source) != protocol['source_tree_hashes']:
        raise ValueError('Source changed for existing diagnostic TAG')
    check_source(project, source)
    _, manifest = input_predictions(read_json(source/'selection.json'), predictions)
    if manifest != protocol['prediction_input']:
        raise ValueError('Prediction input changed for existing TAG')
    for path, digest in protocol['code_hashes'].items():
        if file_sha256(portable(path)) != digest: raise ValueError('Frozen diagnostic code changed: '+path)
    if hash_outputs(out) != read_json(out/'completion.json', {}).get('output_hashes'):
        raise ValueError('Diagnostic output incomplete/changed; do not overwrite')
    return read_json(out/'summary.json')


def prepare(project, source, out, guard, predictions=None):
    project, source, out = (Path(p).resolve() for p in (project, source, out))
    predictions = Path(predictions).resolve() if predictions else None
    paths(project, source, out, predictions)
    if out.exists(): return dict(verify(project, source, out, predictions), execution='verified_existing_no_replay')
    before = tree_hashes(source)
    frame_root, prior = check_source(project, source)
    rows, selection = jsonl(source/'records.jsonl'), read_json(source/'selection.json')
    provenance = {'overlay_protocol_sha256': file_sha256(source/'protocol.json'),
                  'overlay_records_sha256': file_sha256(source/'records.jsonl'),
                  'overlay_review_sha256': prior['review_sha256'], 'source_tag': source.name}
    traces = build_traces(rows, selection, provenance)
    if len({t['window_uid'][:20] for t in traces}) != len(traces):
        raise ValueError('Case directory UID prefixes collide')
    predictions_rows, prediction_manifest = input_predictions(selection, predictions)
    joined, join_audit = join_rows(predictions_rows, traces)
    join_audit['prediction_input'] = prediction_manifest
    stats = inventory(traces)
    media = []
    for trace, selected in zip(traces, selection):
        uid = trace['window_uid']
        if not re.fullmatch('[0-9a-f]{64}', uid): raise ValueError('Unsafe or invalid frozen window UID')
        for i in range(8):
            image = frame_root/'cases'/uid[:20]/f'T{i}.jpg'
            expected = selected['media_hashes'][f'T{i}.jpg']
            if file_sha256(image) != expected: raise ValueError('Existing frame fingerprint changed')
            media.append({'source': str(image), 'target': f'cases/{uid[:20]}/T{i}.jpg', 'sha256': expected})
    observation_features = []
    for trace in traces:
        for obs in trace['observations']:
            view = obs['review_view']
            observation_features.append({
                'window_uid': trace['window_uid'], 'observation_id': obs['id'],
                'kind_claim': view['kind'], 'phase_claim': view['phase'],
                'presence_claim': view['presence'], 'evidence_mode_claim': view['claim_mode'],
                'claim_contract_valid': view['structural_check']['observation_valid'],
                'relation_contract_valid': view['structural_check']['relation_valid'],
                'semantic_support': obs['semantic_support'],
                'context_needed_for': view['context_needed_for'],
                'diagnostic_only': True, 'training_loss_mask': False, 'evaluation_loss_mask': False,
            })
    summary = {
        'version': VERSION, 'execution': 'all_36_readonly_diagnostic_sidecars_built',
        'windows': len(traces), 'observations': stats['observations'],
        'human_reviewed_windows': sum(t['human_reviewed_questions'] > 0 for t in traces),
        'human_reviewed_questions': sum(t['human_reviewed_questions'] for t in traces),
        'whole_window_semantic_reviews': 0,
        'all_original_prediction_fields_preserved': join_audit['all_original_fields_preserved'],
        'changed_existing_prediction_fields': 0, 'evidence_images_reused': len(media),
        'readonly_development_sidecar_available': True, 'formal_shadow_gate_overridden': False,
        'new_human_review_requested': False, 'scoring_authorized': False, 'training_authorized': False,
        'remote_calls': 0, 'new_media_decodes': 0, 'new_model_responses': 0,
        'formal_accuracy': None, 'formal_AP': None,
        'prediction_input': prediction_manifest,
        'claim_limit': 'Single-reviewer field-level development diagnostics, not new accuracy or gold supervision.',
    }
    out.mkdir()
    atomic_json(out/'protocol.json', {'version': VERSION, 'operation': 'readonly_development_sidecar_all_36',
        'source_run': str(source), 'source_tree_hashes': before,
        'code_hashes': {str(p): file_sha256(p) for p in owned(project)},
        'prediction_input': prediction_manifest, 'authorization': 'User requested next improvements; diagnostic only.',
        'formal_gate_override': False, 'network_authorized': False, 'scoring_authorized': False, 'training_authorized': False})
    atomic_json(out/'integrity.json', {'protocol_sha256': file_sha256(out/'protocol.json')})
    write_jsonl(out/'diagnostic_trace.jsonl', traces)
    write_jsonl(out/'diagnostic_observation_features.jsonl', observation_features)
    write_jsonl(out/'predictions_with_diagnostic_trace.jsonl', joined)
    atomic_json(out/'noninterference_audit.json', join_audit)
    atomic_json(out/'evidence_inventory.json', stats)
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'media_manifest.json', media)
    baselines = [s['baseline'] for s in selection]
    for trace, baseline in zip(traces, baselines):
        case = out/'cases'/trace['window_uid'][:20]
        case.mkdir(parents=True)
        (case/'index.html').write_text(case_page(trace, baseline), encoding='utf-8')
    for item in media:
        dest = out/item['target']
        shutil.copyfile(portable(item['source']), dest)
        if file_sha256(dest) != item['sha256']: raise ValueError('Frame copy mismatch')
    (out/'index.html').write_text(index_page(traces, baselines, stats, join_audit), encoding='utf-8')
    atomic_json(out/'reader_manifest.json', {'version': VERSION, 'development_reader_available': True,
        'full_semantic_review_gate_passed': False, 'score_override_allowed': False, 'training_allowed': False,
        'trace_key': TRACE_KEY, 'join_identity_fields': list(IDENTITY)})
    if tree_hashes(source) != before: raise ValueError('Source overlay changed during export')
    _, after_predictions = input_predictions(selection, predictions)
    if after_predictions != prediction_manifest: raise ValueError('Prediction input changed during export')
    guard.assert_no_remote_calls()
    atomic_json(out/'completion.json', {'output_hashes': hash_outputs(out), 'source_unchanged': True,
        'remote_calls': 0, 'all_original_prediction_fields_preserved': True})
    verify(project, source, out, predictions)
    return summary


def attach_readonly(predictions, project, bundle):
    """Public opt-in reader: a sealed development bundle, never the formal V97 loader."""
    bundle = Path(bundle).resolve()
    protocol = read_json(bundle/'protocol.json')
    saved_predictions = protocol['prediction_input']['path']
    verify(Path(project), portable(protocol['source_run']), bundle,
           portable(saved_predictions) if saved_predictions else None)
    return join_rows(predictions, jsonl(bundle/'diagnostic_trace.jsonl'))
