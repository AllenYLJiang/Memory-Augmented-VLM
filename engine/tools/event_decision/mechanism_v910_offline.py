"""Immutable all-cohort human overlay with imported-review and source seals."""
import json
from pathlib import Path

from .contracts import file_sha256, semantic_sha256, read_json, write_jsonl
from .role_scoped import atomic_json, portable
from .mechanism_v98_offline import tree_hashes
from .mechanism_v99_offline import verify as verify_v99
from .mechanism_v99_review import validate_return
from .mechanism_v910_overlay import VERSION, REVIEW_SHA256, policy, apply_overlay, summarize

DEFAULT_SOURCE = 'governed_v99_typed_repair_diagnostic_review_20260915'
DEFAULT_TAG = 'governed_v910_human_evidence_overlay_20260915'


def strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def owned(project):
    paths = [project/'tools/event_decision'/('mechanism_v910_'+s+'.py') for s in ('overlay', 'offline')]
    paths += [project/'tools/mechanism_v910_cli.py', project/'run_mechanism_v910_offline.sh',
              project/'tests/test_mechanism_v910.py']
    paths += [project/'tools/event_decision'/s for s in ('contracts.py', 'role_scoped.py', 'safety.py',
              'mechanism_v96_contract.py', 'mechanism_v97_contract.py', 'mechanism_v98_contract.py',
              'mechanism_v98_offline.py', 'mechanism_v99_offline.py', 'mechanism_v99_review.py')]
    return paths


def inputs(project, source, review_path):
    review_bytes_hash = file_sha256(review_path)
    if review_bytes_hash != REVIEW_SHA256:
        raise ValueError('Review differs from the audited R1 file; use a new versioned repair plan')
    ancestor = portable(read_json(source/'protocol.json')['source_run']).resolve()
    verify_v99(project, ancestor, source)
    packet = read_json(source/'diagnostic_review/packet.json')
    review = strict_json(review_path)
    result = validate_return(packet, review)
    receipt_path = source/'review_imports'/(review_bytes_hash+'.json')
    if not receipt_path.is_file():
        raise ValueError('R1 review must be successfully imported into V99 first')
    receipt = strict_json(receipt_path)
    if receipt != dict(input_sha256=review_bytes_hash, packet_id=packet['packet_id'], review=review, result=result):
        raise ValueError('Imported review receipt does not match the actual R1 file')
    records = jsonl(source/'records.jsonl')
    original = jsonl(ancestor/'records.jsonl')
    selection = read_json(ancestor/'selection.json')
    uids = [r['window_uid'] for r in records]
    if len(uids) != 36 or len(set(uids)) != 36 or uids != [r['window_uid'] for r in original] or uids != [r['window_uid'] for r in selection]:
        raise ValueError('All 36 frozen windows in their original order are required')
    for r, old in zip(records, original):
        if r['source_v98_payload'] != old['diagnostic_payload']:
            raise ValueError('Original evidence layer does not match V99 source')
    provenance = {
        'review_file_sha256': review_bytes_hash, 'packet_id': packet['packet_id'],
        'packet_file_sha256': file_sha256(source/'diagnostic_review/packet.json'),
        'import_receipt_relative_path': str(receipt_path.relative_to(source)),
        'import_receipt_sha256': file_sha256(receipt_path),
        'source_v99_records_sha256': file_sha256(source/'records.jsonl'),
        'source_v98_records_sha256': file_sha256(ancestor/'records.jsonl'),
        'source_v99_protocol_sha256': file_sha256(source/'protocol.json'),
    }
    return records, original, selection, packet, review, receipt, provenance


def output_hashes(out):
    return {k: v for k, v in tree_hashes(out).items() if k != 'completion.json'}


def check_paths(project, source, out, review):
    runs = (project/'runs').resolve()
    if (out.parent != runs or out == source or out in source.parents or source in out.parents
            or out in review.parents):
        raise ValueError('Use a new independent direct runs TAG, not a source/review folder')


def verify(project, source, out, review_path):
    project, source, out, review_path = (Path(p).resolve() for p in (project, source, out, review_path))
    check_paths(project, source, out, review_path)
    protocol = read_json(out/'protocol.json', {})
    if protocol.get('version') != VERSION or protocol.get('operation') != 'r1_scoped_human_overlay_all_36':
        raise ValueError('Not a complete V9.10 human overlay TAG')
    if read_json(out/'integrity.json', {}).get('protocol_sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('Overlay protocol changed')
    if portable(protocol['source_run']).resolve() != source or protocol['review_sha256'] != file_sha256(review_path):
        raise ValueError('Source or reviewer file changed for existing TAG')
    if tree_hashes(source) != protocol['source_tree_hashes']:
        raise ValueError('Source changed since overlay freeze; do not overwrite this TAG')
    inputs(project, source, review_path)
    for path, digest in protocol['code_hashes'].items():
        if file_sha256(portable(path)) != digest:
            raise ValueError('Frozen overlay code changed: '+path)
    if output_hashes(out) != read_json(out/'completion.json', {}).get('output_hashes'):
        raise ValueError('Overlay output changed or incomplete; inspect and use a new TAG')
    return read_json(out/'summary.json')


def prepare(project, source, out, review_path, guard):
    project, source, out, review_path = (Path(p).resolve() for p in (project, source, out, review_path))
    check_paths(project, source, out, review_path)
    if out.exists():
        return dict(verify(project, source, out, review_path), execution='verified_existing_no_replay')
    before = tree_hashes(source)
    records, original, selection, packet, review, receipt, provenance = inputs(project, source, review_path)
    private = read_json(source/'private_case_map.json')
    rows = apply_overlay(records, packet, review, private, provenance)
    for row, old, selected in zip(rows, original, selection):
        row['original_model_payload'] = old['original_payload']
        row['original_model_assessment'] = old['official_v97_assessment']
        row['original_receipt_attempt'] = old['original_receipt_attempt']
        row['original_model_payload_sha256'] = semantic_sha256(old['original_payload'])
        row['sampled_frame_indices'] = selected['sampled_frame_indices']
        row['start_frame'] = selected['start_frame']
        row['end_frame_exclusive'] = selected['end_frame_exclusive']
    summary = summarize(rows)
    native = read_json(portable(read_json(source/'protocol.json')['source_run'])/'summary.json')['official_v97_metrics']
    summary.update(original_native_v97_metrics_unchanged=native,
                   source_files_verified_unchanged=len(before), review_sha256=REVIEW_SHA256,
                   assessment_rule='Frozen V98 validator; scoped R1 entity support affects only e6, never all events.')
    changes = [dict(window_uid=r['window_uid'], **c) for r in rows for c in r['field_changes']]
    assertions = [dict(window_uid=r['window_uid'], **a) for r in rows for a in r['review_assertions']]
    supports = [dict(window_uid=r['window_uid'], **s) for r in rows for s in r['scoped_entity_support']]
    audit = [dict(window_uid=r['window_uid'], video_id=r['video_id'], source_group=r['source_group'],
                  human_reviewed=bool(r['review_assertions']),
                  native_model_valid=r['original_model_assessment']['valid'],
                  v99_valid=r['compatible_assessment']['valid'],
                  field_projection_valid=r['review_projection_assessment']['valid'],
                  human_scoped_valid=r['human_scoped_assessment']['valid'],
                  remaining_issues=r['human_scoped_assessment']['issues'],
                  field_changes=len(r['field_changes']), scoped_support=len(r['scoped_entity_support']),
                  round_trip_verified=r['round_trip_verified'],
                  entity_table_unchanged=r['shared_entities_unchanged'],
                  before_sha256=r['payload_sha256_before'], after_sha256=r['payload_sha256_after']) for r in rows]
    out.mkdir()
    atomic_json(out/'protocol.json', {
        'version': VERSION, 'operation': 'r1_scoped_human_overlay_all_36',
        'source_run': str(source), 'review_file': str(review_path), 'review_sha256': REVIEW_SHA256,
        'source_tree_hashes': before, 'code_hashes': {str(p): file_sha256(p) for p in owned(project)},
        'selection': 'all_36_original_order_no_outcome_filter', 'policy': policy(),
        'source_provenance': provenance,
    })
    atomic_json(out/'integrity.json', {'protocol_sha256': file_sha256(out/'protocol.json')})
    atomic_json(out/'policy.json', policy())
    # Preserve exact review bytes as well as a machine-readable imported receipt.
    (out/'review_return.json').write_bytes(review_path.read_bytes())
    atomic_json(out/'review_import_receipt.json', receipt)
    atomic_json(out/'review_packet.json', packet)
    atomic_json(out/'selection.json', selection)
    write_jsonl(out/'records.jsonl', rows)
    write_jsonl(out/'overlay_assertions.jsonl', assertions)
    write_jsonl(out/'field_changes.jsonl', changes)
    write_jsonl(out/'scoped_entity_support.jsonl', supports)
    write_jsonl(out/'all_36_audit.jsonl', audit)
    atomic_json(out/'summary.json', summary)
    atomic_json(out/'integration/CURRENT.json', {
        'ready': False, 'scoring_authorized': False, 'training_authorized': False,
        'reason': 'Human-assisted contract audit only; not native compliance or a full semantic review.',
    })
    if tree_hashes(source) != before or file_sha256(review_path) != REVIEW_SHA256:
        raise ValueError('Source or human review changed during replay')
    guard.assert_no_remote_calls()
    atomic_json(out/'completion.json', {'output_hashes': output_hashes(out), 'remote_calls': 0,
                                      'source_unchanged': True, 'all_36_replayed': True})
    verify(project, source, out, review_path)
    return summary
