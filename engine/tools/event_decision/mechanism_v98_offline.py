"""Immutable all-cohort replay of saved responses, without a provider or review path."""
import copy
import json
import os
import re
import socket
from contextlib import contextmanager
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v97_store import load as load_v97, current
from .mechanism_v97_contract import assess as assess_v97
from .mechanism_v98_contract import (VERSION, GATES, assess, examples, prompt, wire_schema,
                                   replay_payload, metrics, offline_gate)

DEFAULT_SOURCE = 'governed_v97_native_mechanism_development_20260914'
DEFAULT_TAG = 'governed_v98_contract_local_validity_offline_20260915'


def tree_hashes(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


@contextmanager
def lock(project, out):
    out = Path(out).resolve(); runs = (Path(project)/'runs').resolve()
    if out.parent != runs or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', out.name):
        raise ValueError('Output must be a new direct runs TAG')
    runs.mkdir(exist_ok=True)
    path = runs/('.'+out.name+'.v98.lock')
    try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError: raise ValueError('Offline TAG locked; inspect the recorded PID, do not force takeover')
    try:
        os.write(fd, json.dumps({'pid': os.getpid(), 'host': socket.gethostname()}).encode()); os.fsync(fd)
        yield
    finally:
        os.close(fd); path.unlink()


def owned_code(project):
    return [project/'tools/event_decision'/('mechanism_v98_'+s+'.py') for s in ('contract', 'offline')] + [
        project/'tools/mechanism_v98_offline_cli.py', project/'run_mechanism_v98_offline.sh',
        project/'tests/test_mechanism_v98.py', project/'tools/event_decision/safety.py']


def verify(project, source, out):
    protocol = read_json(out/'protocol.json', {})
    if protocol.get('version') != VERSION or protocol.get('operation') != 'offline_saved_response_replay':
        raise ValueError('Not a V9.8 offline TAG')
    if read_json(out/'integrity.json', {}).get('sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('Offline protocol changed')
    if portable(protocol['source_run']).resolve() != source:
        raise ValueError('Source differs for existing TAG')
    if tree_hashes(source) != protocol['source_tree_hashes']:
        raise ValueError('Original run changed; inspect without overwriting this TAG')
    load_v97(source)
    for name, digest in protocol['code_hashes'].items():
        if file_sha256(portable(name)) != digest: raise ValueError('Frozen replay code changed: '+name)
    completion = read_json(out/'completion.json')
    if not completion: raise ValueError('Incomplete offline output; inspect it and use a new TAG, never overwrite')
    actual = tree_hashes(out); actual.pop('completion.json', None)
    if actual != completion['output_hashes']: raise ValueError('Offline output changed or unexpected files added')
    return read_json(out/'summary.json')


def execute(project, source, out, guard, verify_only=False):
    project, source, out = (Path(p).resolve() for p in (project, source, out))
    if out.parent != (project/'runs').resolve() or source == out or source in out.parents or out in source.parents:
        raise ValueError('Source and output must be independent; output must be a direct runs child')
    if out.exists(): return dict(verify(project, source, out), execution='verified_existing_no_replay')
    if verify_only: raise ValueError('Offline output does not exist')
    if (source.parent/('.'+source.name+'.v97.lock')).exists(): raise ValueError('Source V9.7 run is locked')
    before = tree_hashes(source)
    old_protocol, rows = load_v97(source)
    if old_protocol['mock']: raise ValueError('Real completed V9.7 source required')
    if len(rows) != 36 or len({r['window_uid'] for r in rows}) != 36:
        raise ValueError('Must replay all 36 unique frozen windows')
    answers = [current(source, row) for row in rows]
    if any(a['status'] != 'response' or a.get('mock') is not False for a in answers):
        raise ValueError('All original responses must be complete; no selective replay')
    saved = read_json(source/'summary.json')
    official = [assess_v97(a['parsed']) for a in answers]
    original_metrics = metrics(official)
    # V9.7 assessments do not carry presence, so obtain it without rewriting them.
    observed_windows = sum(any(c['observation_valid'] and o['presence'] == 'observed'
                              for c, o in zip(a['assessment']['observations'], a['parsed']['observations'])) for a in answers)
    original_metrics['windows_with_observed_observation'] = observed_windows
    original_metrics['windows_with_observed_observation_fraction'] = observed_windows/36
    for name, value in saved['metrics'].items():
        if original_metrics[name] != value: raise ValueError('Source summary disagrees with original receipts: '+name)
    if any(o != a['assessment'] for o, a in zip(official, answers)): raise ValueError('Original assessment changed')
    example_values = examples()
    if not all(assess(e)['valid'] for e in example_values): raise ValueError('New payload examples fail validation')
    derived = []; wire_checks = []; projected_checks = []; changes = []
    for row, answer in zip(rows, answers):
        raw = answer['parsed']; original_copy = copy.deepcopy(raw)
        strict, version_edits = replay_payload(raw)
        projected, projection_edits = replay_payload(raw, project_metadata=True)
        wire, final = assess(strict), assess(projected)
        wire_checks.append(wire); projected_checks.append(final)
        recovered = []
        for old, new in zip(answer['assessment']['observations'], final['observations']):
            if old['relation_valid'] != new['relation_valid']:
                recovered.append({'id': new['id'], 'before': old['relation_valid'], 'after': new['relation_valid'],
                                  'reason': 'dependency_local_entity_validity_only',
                                  'referenced_entity_ids': new['referenced_entity_ids'],
                                  'local_issues': new['local_issues']})
        record = {'window_uid': row['window_uid'], 'video_id': row['video_id'],
                  'source_group': row['source_group'], 'selection_reasons': row['selection_reasons'],
                  'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
                  'sampled_frame_indices': row['sampled_frame_indices'],
                  'original_receipt_attempt': answer['attempt'], 'original_payload': raw,
                  'official_v97_assessment': answer['assessment'],
                  'version_projection_edits': version_edits, 'version_only_assessment': wire,
                  'metadata_projection_edits': projection_edits, 'diagnostic_payload': projected,
                  'metadata_projection_assessment': final, 'local_relation_validity_changes': recovered,
                  'new_inference': False, 'evidence_unchanged': raw == original_copy}
        if not record['evidence_unchanged']: raise AssertionError('Original evidence mutated')
        derived.append(record)
        changes.extend(dict(window_uid=row['window_uid'], **x) for x in recovered)
    replay_metrics = metrics(projected_checks)
    gate = offline_gate(replay_metrics)
    summary = {'version': VERSION, 'execution': 'all_36_saved_responses_replayed', 'source_run': str(source),
               'windows': len(rows), 'videos': len({r['video_id'] for r in rows}),
               'source_groups': len({r['source_group'] for r in rows}),
               'official_v97_metrics': saved['metrics'], 'official_v97_technical_ready': saved['technical_ready'],
               'v98_version_only_replay': metrics(wire_checks), 'v98_metadata_projection_replay': replay_metrics,
               'metadata_projection_windows': sum(len(r['metadata_projection_edits']) > 1 for r in derived),
               'local_relation_validity_changes': changes,
               'remaining_failed_windows': [r['window_uid'] for r in derived if not r['metadata_projection_assessment']['valid']],
               'synthetic_examples_validated': len(example_values), 'source_files_unchanged': len(before),
               'thresholds_unchanged': GATES, 'remote_calls': 0, 'new_media_decodes': 0, 'new_vlm_responses': 0,
               'formal_accuracy': None, 'formal_AP': None, **gate,
               'claim_limit': 'Encoding and validator changes only; no new visual truth, labels or score gains.'}
    out.mkdir()
    protocol = {'version': VERSION, 'operation': 'offline_saved_response_replay', 'source_run': str(source),
                'source_tree_hashes': before, 'source_selection_sha256': file_sha256(source/'selection.json'),
                'code_hashes': {str(p): file_sha256(p) for p in owned_code(project)},
                'selection': 'all_36_same_order_same_frames_no_outcome_filter', 'gates': GATES,
                'remote_execution_authorized': False, 'human_review_requested': False,
                'scoring_authorized': False, 'training_authorized': False}
    atomic_json(out/'protocol.json', protocol)
    atomic_json(out/'integrity.json', {'sha256': file_sha256(out/'protocol.json')})
    atomic_json(out/'selection.json', rows)
    atomic_json(out/'wire_schema.json', wire_schema())
    atomic_json(out/'synthetic_examples.json', example_values)
    (out/'prospective_prompt.txt').write_text(prompt(), encoding='utf-8')
    write_jsonl(out/'records.jsonl', derived)
    write_jsonl(out/'field_changes.jsonl', [dict(window_uid=r['window_uid'], **e) for r in derived for e in r['metadata_projection_edits']])
    write_jsonl(out/'remaining_issues.jsonl', [dict(window_uid=r['window_uid'], video_id=r['video_id'], **e)
                                             for r in derived for e in r['metadata_projection_assessment']['issues']])
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'gate.json', gate)
    atomic_json(out/'review/packet.json', {'active': False, 'reason': gate['reason'], 'human_review_requested': False})
    atomic_json(out/'integration/CURRENT.json', {'ready': False, 'scoring_authorized': False, 'training_authorized': False})
    atomic_json(out/'next_request_decision.json', {'status': 'NOT_DECIDED_NO_API_AUTHORITY',
        'remote_execution_authorized': False, 'review_requested': False,
        'remaining_failed_windows': summary['remaining_failed_windows'],
        'requirements_before_any_new_request': [
            'Review full-cohort offline differences and remaining evidence obligations.',
            'Decide whether requests are necessary; freeze a separate native acquisition protocol before calling.',
            'Predeclare cohort and controls; retain unknown/residual/other-event cases without preferred-answer retries.',
            'Human review requires technical proportions AND critical reference checks; replay alone never activates it.']})
    if tree_hashes(source) != before: raise ValueError('Original run changed during replay; no completion receipt')
    guard.assert_no_remote_calls()
    atomic_json(out/'completion.json', {'source_unchanged': True, 'remote_calls': 0, 'output_hashes': tree_hashes(out)})
    verify(project, source, out)
    return summary
