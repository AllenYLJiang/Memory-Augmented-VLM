"""Versioned typed repairs and a separate, user-authorized diagnostic review."""
import json
import shutil
from pathlib import Path

from .contracts import read_json, file_sha256, semantic_sha256, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v98_contract import assess, metrics, GATES
from .mechanism_v98_offline import verify as verify_v98, tree_hashes
from .mechanism_v99_repair import VERSION, normalize, assert_evidence_unchanged
from .mechanism_v99_review import build_packet, render, import_return

DEFAULT_SOURCE = 'governed_v98_contract_local_validity_offline_20260915'
DEFAULT_TAG = 'governed_v99_typed_repair_diagnostic_review_20260915'


def owned(project):
    return [project/'tools/event_decision'/('mechanism_v99_'+s+'.py') for s in ('repair', 'review', 'offline')] + [
        project/'tools/mechanism_v99_cli.py', project/'run_mechanism_v99.sh', project/'tests/test_mechanism_v99.py']


def protected_hashes(out):
    return {name: digest for name, digest in tree_hashes(out).items()
            if not name.replace('\\', '/').startswith('review_imports/') and name != 'completion.json'}


def verify(project, source, out):
    protocol = read_json(out/'protocol.json', {})
    if protocol.get('version') != VERSION or protocol.get('operation') != 'typed_compatibility_and_diagnostic_review':
        raise ValueError('Not a V9.9 compatibility run')
    if read_json(out/'integrity.json', {}).get('sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('V9.9 protocol changed')
    if portable(protocol['source_run']).resolve() != source:
        raise ValueError('Source changed for existing TAG')
    if tree_hashes(source) != protocol['source_tree_hashes']: raise ValueError('V9.8 source changed')
    old_source = portable(read_json(source/'protocol.json')['source_run'])
    verify_v98(project, old_source, source)
    for path, digest in protocol['code_hashes'].items():
        if file_sha256(portable(path)) != digest: raise ValueError('V9.9 code changed: '+path)
    if protected_hashes(out) != read_json(out/'completion.json', {}).get('output_hashes'):
        raise ValueError('V9.9 output changed or incomplete')
    return read_json(out/'summary.json')


def prepare(project, source, out, guard):
    project, source, out = (Path(x).resolve() for x in (project, source, out))
    if out.parent != (project/'runs').resolve() or out == source or source in out.parents or out in source.parents:
        raise ValueError('Use an independent direct runs TAG')
    if out.exists(): return dict(verify(project, source, out), execution='verified_existing_no_replay')
    before = tree_hashes(source)
    original = portable(read_json(source/'protocol.json')['source_run'])
    prior = verify_v98(project, original, source)
    records = [json.loads(line) for line in (source/'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = read_json(source/'selection.json')
    if len(records) != 36 or [r['window_uid'] for r in records] != [r['window_uid'] for r in rows]:
        raise ValueError('Full fixed cohort/order required')
    output = []
    for row in records:
        payload, changes = normalize(row['diagnostic_payload'])
        preserved = assert_evidence_unchanged(row['diagnostic_payload'], payload, changes)
        output.append({'window_uid': row['window_uid'], 'video_id': row['video_id'],
                       'source_group': row['source_group'], 'selection_reasons': row['selection_reasons'],
                       'source_v98_payload': row['diagnostic_payload'], 'source_v98_assessment': row['metadata_projection_assessment'],
                       'compatible_payload': payload, 'compatible_assessment': assess(payload),
                       'encoding_changes': changes, 'round_trip_and_evidence_preserved': preserved,
                       'new_visual_truth': False, 'new_inference': False})
    after_metrics = metrics([r['compatible_assessment'] for r in output])
    fixed = [r['window_uid'] for r in output if not r['source_v98_assessment']['valid'] and r['compatible_assessment']['valid']]
    regressed = [r['window_uid'] for r in output if r['source_v98_assessment']['valid'] and not r['compatible_assessment']['valid']]
    if regressed: raise ValueError('Encoding compatibility regressed previously valid documents')
    run_binding = semantic_sha256({'source_protocol': file_sha256(source/'protocol.json'), 'tag': out.name,
                                   'code': {str(p.relative_to(project)): file_sha256(p) for p in owned(project)}})
    packet, private = build_packet(output, {r['window_uid']: r for r in rows}, run_binding)
    summary = {'version': VERSION, 'execution': 'all_36_reversible_compatibility_replayed',
               'windows': 36, 'videos': prior['videos'], 'source_groups': prior['source_groups'],
               'v98_metadata_projection_metrics': prior['v98_metadata_projection_replay'],
               'v99_compatibility_metrics': after_metrics,
               'newly_format_compatible_windows': fixed, 'regressed_documents': regressed,
               'remaining_failed_windows': [r['window_uid'] for r in output if not r['compatible_assessment']['valid']],
               'encoding_changed_windows': sum(bool(r['encoding_changes']) for r in output),
               'all_36_evidence_round_trip_verified': True,
               'diagnostic_review_active': packet['active'], 'diagnostic_review_cases': len(packet['cases']),
               'diagnostic_review_questions': sum(len(c['questions']) for c in packet['cases']),
               'formal_review_active': False, 'technical_gate_overridden': False,
               'automatic_human_answer_patch': False, 'ready_for_shadow_integration': False,
               'scoring_authorized': False, 'training_authorized': False, 'remote_execution_authorized': False,
               'remote_calls': 0, 'new_media_decodes': 0, 'new_model_responses': 0,
               'formal_accuracy': None, 'formal_AP': None,
               'decision': 'DIAGNOSTIC_REVIEW_ONLY' if packet['active'] else 'OFFLINE_COMPLETE_NO_FORMAL_REVIEW',
               'claim_limit': 'Versioned encoding compatibility, not native model compliance or visual accuracy.'}
    out.mkdir()
    atomic_json(out/'protocol.json', {'version': VERSION, 'operation': 'typed_compatibility_and_diagnostic_review',
        'source_run': str(source), 'source_tree_hashes': before,
        'code_hashes': {str(p): file_sha256(p) for p in owned(project)},
        'selection': 'all_36_saved_responses_no_outcome_selection', 'gates': GATES,
        'authorization': 'user_requested_code_repairs_and_diagnostic_review_of_remaining_evidence',
        'remote_execution_authorized': False, 'formal_review_authorized': False,
        'scoring_authorized': False, 'training_authorized': False})
    atomic_json(out/'integrity.json', {'sha256': file_sha256(out/'protocol.json')})
    write_jsonl(out/'records.jsonl', output)
    write_jsonl(out/'encoding_changes.jsonl', [dict(window_uid=r['window_uid'], **c) for r in output for c in r['encoding_changes']])
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'diagnostic_review/packet.json', packet)
    atomic_json(out/'private_case_map.json', private)
    atomic_json(out/'integration/CURRENT.json', {'ready': False, 'reason': 'Troubleshooting review is not deployment approval'})
    source_rows = {r['window_uid']: r for r in rows}
    for case in packet['cases']:
        uid = private[case['case_id']]; row = source_rows[uid]
        for i in range(8):
            src = original/'cases'/uid[:20]/f'T{i}.jpg'
            # V9.7 load() verified all frozen media. Recheck each copied asset too.
            dest = out/'diagnostic_review/frames'/case['case_id']/f'T{i}.jpg'
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dest)
            if file_sha256(src) != file_sha256(dest): raise ValueError('Review media copy mismatch')
    (out/'diagnostic_review/index.html').write_text(render(packet), encoding='utf-8')
    if tree_hashes(source) != before: raise ValueError('V9.8 source changed during replay')
    guard.assert_no_remote_calls()
    atomic_json(out/'completion.json', {'output_hashes': protected_hashes(out), 'remote_calls': 0, 'source_unchanged': True})
    verify(project, source, out)
    return summary


def import_review(project, source, out, path, guard):
    verify(project, source, out)
    result = import_return(out, path)
    guard.assert_no_remote_calls()
    verify(project, source, out)
    return result
