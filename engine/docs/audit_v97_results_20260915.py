"""Read-only V9.7 result audit. Writes only to a new docs subdirectory."""
import argparse
import copy
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'tools'))
sys.path.insert(0, str(PROJECT/'docs'))
from event_decision.safety import OfflineGuard
guard = OfflineGuard(); guard.install()
from event_decision.contracts import read_json, file_sha256, semantic_sha256
from event_decision.mechanism_v97_store import load, current, receipts
from event_decision.mechanism_v97_contract import assess, specification

META = {'acquisition_enabled', 'status', 'primary_kind_to_derived_coarse_type',
        'states', 'evidence_strengths', 'future_event_links', 'limits'}
WIRE_SPEC = json.loads(json.dumps(specification()))


def snapshot(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT/'runs/governed_v97_native_mechanism_development_20260914')
    parser.add_argument('--out', type=Path, default=PROJECT/'docs/v97_result_audit_20260915_v2')
    args = parser.parse_args()
    out = args.out.resolve(); root = args.run.resolve()
    if out.parent != (PROJECT/'docs').resolve() or out.exists():
        raise ValueError('Use a new direct docs child; do not overwrite prior analysis')
    before = snapshot(root)
    protocol, rows = load(root)
    saved = read_json(root/'summary.json')
    digest = file_sha256(root/'protocol.json')
    items = []; all_attempts = []; invalid_relations = []; all_obs = []
    field_echoes = Counter(); issue_counts = Counter(); gate_issues = Counter()
    metadata_only = []; other_failures = []; echoes = []; no_local_relation_issue = []
    native_valid = projected_valid = observed_count = relation_count = relation_valid_count = 0
    for row in rows:
        uid = row['window_uid']; r = current(root, row)
        rr = receipts(root, uid, digest); all_attempts.extend(rr)
        assert len(rr) == 1 and r['status'] == 'response' and r['mock'] is False
        assert json.loads(r['raw'].strip().removeprefix('```json').removesuffix('```').strip()) == r['parsed']
        raw, a = r['parsed'], r['assessment']
        native_valid += a['valid']; extras = set(raw)-{'version', 'entities', 'observations'}
        exact_echo = bool(extras) and extras <= META and all(raw[k] == WIRE_SPEC[k] for k in extras)
        projection = copy.deepcopy(raw)
        if exact_echo:
            echoes.append(uid)
            for k in extras: projection.pop(k); field_echoes[k] += 1
        projected = assess(projection); projected_valid += projected['valid']
        assert projection['entities'] == raw['entities'] and projection['observations'] == raw['observations']
        if not a['valid'] and projected['valid']: metadata_only.append(uid)
        if not projected['valid']: other_failures.append(uid)
        local = [x for x in a['issues'] if x['path'] != '/']
        if not local: no_local_relation_issue.append(uid)
        issue_counts.update(x['code'] for x in a['issues']); gate_issues.update(x['code'] for x in projected['issues'])
        old = read_json(root/'source_results'/(uid+'.json'))['parsed']
        observations = []
        for i, (obs, check) in enumerate(zip(raw['observations'], a['observations'])):
            observed_count += check['observation_valid']
            record = dict(window_uid=uid, index=i, observation=obs, check=check)
            all_obs.append(record); observations.append(record)
            if check['relation_claim_present']:
                relation_count += 1; relation_valid_count += check['relation_valid']
                if not check['relation_valid']:
                    invalid_relations.append(dict(record, entities=raw['entities'],
                        local_issues=[x for x in a['issues'] if x['path'].startswith(f'/observations/{i}/')],
                        entity_issues=[x for x in a['issues'] if x['path'].startswith('/entities')]))
        items.append({'window_uid': uid, 'video_id': row['video_id'], 'source_group': row['source_group'],
                      'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
                      'sampled_frame_indices': row['sampled_frame_indices'], 'selection_reasons': row['selection_reasons'],
                      'raw_top_level_keys': list(raw), 'exact_spec_metadata_echo': exact_echo,
                      'official_valid': a['valid'], 'metadata_projection_valid_diagnostic_only': projected['valid'],
                      'issues': a['issues'], 'projection_issues': projected['issues'],
                      'old_b5_not_gold': old.get('b5'), 'old_event_count': len(old.get('events', [])),
                      'observations': observations, 'usage': r.get('usage'), 'elapsed_seconds': r['elapsed_seconds']})
    n = len(rows)
    assert saved['metrics']['document_valid_fraction'] == native_valid/n
    assert saved['metrics']['observation_valid_fraction'] == observed_count/len(all_obs)
    assert saved['metrics']['relation_valid_fraction'] == relation_valid_count/relation_count
    assert saved['issue_counts'] == dict(issue_counts)
    assert saved['observations'] == len(all_obs) and saved['relations'] == relation_count
    summaries = read_json(root/'review/packet.json'); pointer = read_json(root/'integration/CURRENT.json')
    assert summaries['active'] is False and pointer['ready'] is False
    totals = {k: sum(a['usage'][k] for a in all_attempts) for k in ('input_tokens', 'output_tokens', 'total_tokens', 'image_tokens')}
    totals['text_input_tokens'] = sum(a['usage']['input_tokens_details']['text_tokens'] for a in all_attempts)
    totals['reasoning_tokens'] = sum(a['usage']['output_tokens_details']['reasoning_tokens'] for a in all_attempts)
    totals['answer_text_tokens'] = sum(a['usage']['output_tokens_details']['text_tokens'] for a in all_attempts)
    invocations = [read_json(p) for p in (root/'invocations').glob('*.json')]
    lo = min(x['started_unix'] for x in invocations); hi = max(x['finished_unix'] for x in invocations)
    durations = [a['elapsed_seconds'] for a in all_attempts]
    format_time = lambda t: datetime.fromtimestamp(t, timezone(timedelta(hours=8))).isoformat()
    # Reproduce a dependency-mask bug on an in-memory copy. No response is normalized or rewritten.
    election = next(x for x in items if x['window_uid'].startswith('7b4d0e68a5aaa10527e7'))
    original = copy.deepcopy(current(root, next(r for r in rows if r['window_uid']==election['window_uid']))['parsed'])
    id_demo = copy.deepcopy(original)
    for entity in id_demo['entities']:
        if entity['id'] == 'p0': entity['id'] = 'p99'
    assert original['observations'] == id_demo['observations']
    demo_checks = assess(id_demo)['observations']
    mask_demo = [{'id': o['id'], 'original_relation_valid': c['relation_valid'],
                  'with_unreferenced_entity_id_renamed_only': d['relation_valid']}
                 for o, c, d in zip(original['observations'], assess(original)['observations'], demo_checks)
                 if c['relation_claim_present']]
    result = {'auditor_version': 2, 'source_run': str(root), 'read_only': True, 'windows': n,
              'videos': len({r['video_id'] for r in rows}), 'groups': len({r['source_group'] for r in rows}),
              'summary_recomputed_and_matches': True, 'official_metrics': saved['metrics'],
              'official_valid_documents': native_valid, 'spec_metadata_echo_windows': len(echoes),
              'metadata_only_failed_windows': len(metadata_only), 'field_echo_counts': dict(field_echoes),
              'metadata_projection_valid_documents_diagnostic_only': projected_valid,
              'remaining_failed_windows_after_metadata_projection': [x[:20] for x in other_failures],
              'relations': relation_count, 'valid_relations': relation_valid_count,
              'invalid_relation_windows': len({x['window_uid'] for x in invalid_relations}),
              'invalid_relations_without_local_issue': sum(not x['local_issues'] for x in invalid_relations),
              'metadata_projection_issue_counts': dict(gate_issues),
              'unrelated_entity_id_mask_reproduction': mask_demo,
              'relation_state_counts': dict(Counter(x['observation']['relation']['state'] for x in all_obs if x['observation']['relation'])),
              'relation_kind_counts': dict(Counter(x['observation']['kind'] for x in all_obs if x['observation']['relation'])),
              'context_need_counts': dict(Counter(k for x in all_obs for k in x['observation']['context_needed_for'])),
              'windows_with_context_need': len({x['window_uid'] for x in all_obs if x['observation']['context_needed_for']}),
              'token_totals': totals, 'average_total_tokens_per_window': totals['total_tokens']/n,
              'reasoning_fraction_of_output_tokens': totals['reasoning_tokens']/totals['output_tokens'],
              'started_local': format_time(lo), 'finished_local': format_time(hi), 'wall_seconds': hi-lo,
              'wall_seconds_per_window': (hi-lo)/n, 'invocations': invocations,
              'request_latency_seconds': {'mean': statistics.mean(durations), 'median': statistics.median(durations), 'min': min(durations), 'max': max(durations), 'sum': sum(durations)},
              'finish_reasons': dict(Counter(a.get('finish_reason') for a in all_attempts)),
              'thinking_request_explicitly_configured': False,
              'human_review_active': summaries['active'], 'integration_ready': pointer['ready'],
              'formal_accuracy': None, 'formal_AP': None, 'new_api_calls': 0}
    after = snapshot(root)
    assert before == after
    result['source_files_unchanged'] = len(before)
    guard.assert_no_remote_calls()
    out.mkdir()
    def write(name, data): (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write('audit.json', result); write('all_cases.json', items); write('all_observations.json', all_obs)
    write('invalid_relations.json', invalid_relations); write('source_hashes.json', before)
    text = ['# V9.7 All 36 Windows: Saved-Response Audit', '',
            'Diagnostic metadata projection does not change the official gate or establish visual truth.', '',
            '| ID | Video | Frames [start,end) | Official valid | Metadata projection valid | Obs | Valid relations / claims | Residual issues |',
            '|---|---|---|---|---|---:|---:|---|']
    for x in items:
        checks = [e['check'] for e in x['observations']]
        valid = sum(e['relation_valid'] for e in checks); claimed = sum(e['relation_claim_present'] for e in checks)
        text.append(f'| {x["window_uid"][:20]} | {x["video_id"]} | [{x["start_frame"]},{x["end_frame_exclusive"]}) | {x["official_valid"]} | {x["metadata_projection_valid_diagnostic_only"]} | {len(checks)} | {valid}/{claimed} | '+', '.join(sorted({e['code'] for e in x['projection_issues']}))+' |')
    for x in items:
        text += ['', '## '+x['window_uid'][:20], '', x['video_id'], '']
        for e in x['observations']:
            o = e['observation']; rel = o['relation']
            text += [f'- {o["id"]}, {o["kind"]}, {o["phase"]}, bins={o["bins"]}: {o["evidence"]}',
                     '  Relation: '+json.dumps(rel, ensure_ascii=False)+'; context_needed_for='+str(o['context_needed_for'])]
    (out/'ALL_36_WINDOWS.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
