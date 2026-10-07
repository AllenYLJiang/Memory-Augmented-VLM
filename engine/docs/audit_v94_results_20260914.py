"""Read-only V9.4 result audit; writes derived reports only under docs."""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fraction(n, d):
    return {'n': n, 'denominator': d, 'fraction': n / d if d else None}


def at(obj, path):
    try:
        for key in path.strip('/').split('/'):
            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
        return obj
    except (ValueError, TypeError, KeyError, IndexError):
        return '<derived_or_missing>'


def distribution(values):
    return dict(sorted(Counter(values).items(), key=lambda x: (-x[1], str(x[0]))))


def desktop_link(path):
    value = path.as_posix()
    if value.startswith('/mnt/c/'):
        return 'C:/' + value[len('/mnt/c/'):]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tag', default='governed_v94_scoped_mechanism_development_20260913')
    parser.add_argument('--output-name', default='v94_result_audit_20260914')
    args = parser.parse_args()
    project = args.project.resolve()
    if any(Path(v).name != v for v in (args.tag, args.output_name)):
        parser.error('Use directory names, not paths')
    sys.path.insert(0, str(project / 'tools'))
    from event_decision.safety import OfflineGuard
    from event_decision.contracts import file_sha256
    from event_decision.mechanism_v94_store import load
    from event_decision.mechanism_v94_contract import assess, example, FEATURES
    OfflineGuard().install()
    root = project / 'runs' / args.tag
    protocol, rows = load(root)
    before = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}
    gate = read(root / 'integration/gate.json')
    result_rows, events, links, receipts = [], [], [], []
    issues, actual_frame_valid, ordered_frames = [], 0, 0
    for row in rows:
        uid = row['window_uid']
        result = read(root / 'results' / (uid + '.json'))
        assert file_sha256(root / 'results' / (uid + '.json')) == gate['result_hashes'][uid]
        assert result['protocol_sha256'] == file_sha256(root / 'protocol.json')
        raw = result.get('parsed') or {}
        audit = assess(raw)
        assert audit == result['assessment']
        old = read(root / 'source_snapshot/results' / (uid + '.json'))
        old_label = (old.get('parsed') or {}).get('b5_presence', 'unavailable')
        per_case_issues = []
        for issue in audit['issues']:
            item = dict(issue, value=at(raw, issue['path']), window_uid=uid)
            issues.append(item); per_case_issues.append(item)
        if audit['fields'].get('/frames/order', {}).get('valid'):
            ordered_frames += 1
            actual_frame_valid += sum(all(audit['fields'].get(f'/frames/{i}/{k}', {}).get('valid')
                for k in ('action', 'constraint', 'injury', 'quality', 'evidence')) for i in range(8))
        for i, event in enumerate(raw.get('events', [])):
            event_audit = audit['events'][i]
            prefix = f'/events/{i}/'
            event_issues = [v for v in per_case_issues if v['path'].startswith(prefix)]
            generic_reject = (event['type'] in {'injury_trace', 'other_event'} and
                              event.get('binding', {}).get('state') == 'observed')
            before_context = (event_audit['valid'] and event.get('observation') == 'observed' and
                event.get('binding', {}).get('state') == 'observed' and
                event.get('boundary', {}).get('b5_state') == 'supported' and
                event.get('phase') in {'active', 'ongoing_constraint'})
            events.append({'window_uid': uid, 'video_id': row['video_id'], 'index': i, 'event': event,
                           'audit': event_audit, 'issues': event_issues,
                           'type_based_binding_rejection': generic_reject,
                           'sole_issue_is_type_based_binding_rejection': generic_reject and len(event_issues) == 1,
                           'local_support_before_context_check': before_context})
        for i, link in enumerate(raw.get('links', [])):
            links.append({'window_uid': uid, 'link': link, 'audit': audit['links'][i]})
        attempts = []
        for path in sorted((root / 'attempts' / uid).glob('*.json')):
            receipt = read(path)
            assert receipt['protocol_sha256'] == result['protocol_sha256']
            receipts.append(receipt); attempts.append(receipt)
        result_rows.append({'uid': uid, 'video': row['video_id'], 'source_group': row['source_group'],
            'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
            'reasons': row['selection_reasons'], 'old_status': old['status'], 'old_label': old_label,
            'v92_label': row['baseline']['parsed']['b5_presence'], 'status': result['status'],
            'b5': raw.get('b5'), 'context': raw.get('context'), 'supported_events': audit['local_supported_events'],
            'event_count': len(audit['events']), 'valid_events': sum(e['valid'] for e in audit['events']),
            'attempts': len(attempts), 'request_seconds': sum(r['elapsed_seconds'] for r in attempts),
            'issues': per_case_issues, 'sampled_indices': row['sampled_frame_indices']})
    invocations = [read(p) for p in sorted((root / 'invocations').glob('*.json'))]
    seconds = [r['elapsed_seconds'] for r in receipts]
    wall = sum(r['finished_unix'] - r['started_unix'] for r in invocations)
    by_reason = {}
    for reason in sorted({v for r in result_rows for v in r['reasons']}):
        group = [r for r in result_rows if reason in r['reasons']]
        by_reason[reason] = {'n': len(group), 'statuses': distribution(r['status'] for r in group),
                            'raw_b5': distribution(r['b5']['label'] for r in group)}
    values = np.load(root / 'features/values.npy', allow_pickle=False)
    masks = np.load(root / 'features/observed.npy', allow_pickle=False)
    assert values.shape == masks.shape == (len(rows), len(FEATURES))
    assert np.isnan(values[~masks]).all() and np.isfinite(values[masks]).all()
    feature_summary = {name: {'observed': int(masks[:, i].sum()),
                             'unobserved': int((~masks[:, i]).sum()),
                             'values': distribution(float(x) for x in values[masks[:, i], i])}
                       for i, name in enumerate(FEATURES)}
    invalid_probability = [i for i in issues if i['path'] == '/b5/probability']
    summary = {
        'run': str(root), 'protocol_sha256': file_sha256(root / 'protocol.json'),
        'frozen_integrity_verified': True, 'selected': len(rows),
        'videos': len({r['video'] for r in result_rows}), 'source_groups': len({r['source_group'] for r in result_rows}),
        'status': distribution(r['status'] for r in result_rows),
        'old_status_same_subset': distribution(r['old_status'] for r in result_rows),
        'old_raw_labels_same_subset': distribution(r['old_label'] for r in result_rows),
        'raw_b5_all_including_partial': distribution(r['b5']['label'] for r in result_rows),
        'raw_b5_core_valid_only': distribution(r['b5']['label'] for r in result_rows if r['status'] == 'success'),
        'old_new_transitions': distribution(f'{r["old_label"]}->{r["b5"]["label"]}' for r in result_rows),
        'by_selection_reason': by_reason, 'events': len(events),
        'event_types': distribution(e['event']['type'] for e in events),
        'event_phases': distribution(e['event']['phase'] for e in events),
        'event_validity': fraction(sum(e['audit']['valid'] for e in events), len(events)),
        'observation_validity': fraction(sum(e['audit']['observation_valid'] for e in events), len(events)),
        'event_valid_by_type': {t: fraction(sum(e['audit']['valid'] for e in events if e['event']['type'] == t),
                                               sum(e['event']['type'] == t for e in events))
                                for t in sorted({e['event']['type'] for e in events})},
        'binding_rejection_types': distribution(e['event']['type'] for e in events if any(i['reason'] == 'binding lacks typed entity/temporal support' for i in e['issues'])),
        'type_based_binding_rejection': sum(e['type_based_binding_rejection'] for e in events),
        'sole_issue_type_based_binding_rejection': sum(e['sole_issue_is_type_based_binding_rejection'] for e in events),
        'event_context': {key: distribution(str(e['event'].get('context_scope', {}).get(key)) for e in events)
                          for key in ('local_mechanism_required', 'local_binding_required', 'category_required')},
        'global_context': {key: distribution(str(r['context'].get(key)) for r in result_rows)
                           for key in ('cross_event_required', 'global_story_required')},
        'boundary_states': distribution(e['event']['boundary']['b5_state'] for e in events),
        'boundary_bases': distribution(e['event']['boundary']['basis'] for e in events),
        'category_alternatives': distribution(c for e in events for c in e['event']['boundary']['alternative_classes']),
        'local_support_before_context_check': sum(e['local_support_before_context_check'] for e in events),
        'local_support_after_context_check': sum(e['audit']['supported'] for e in events),
        'windows_with_support_before_context_check': len({e['window_uid'] for e in events if e['local_support_before_context_check']}),
        'links': len(links), 'link_states': distribution(l['link']['state'] for l in links),
        'link_validity': fraction(sum(l['audit']['valid'] for l in links), len(links)),
        'eight_ordered_frames': ordered_frames,
        'actual_frame_record_validity_independent_of_core': fraction(actual_frame_valid, 8 * len(rows)),
        'reported_frame_valid_fraction': gate['frame_valid_fraction'],
        'invalid_frame_values': distribution(str(i['value']) for i in issues if i['path'].startswith('/frames/')),
        'invalid_probabilities': invalid_probability,
        'issue_counts': distribution(i['reason'] for i in issues),
        'attempts': len(receipts), 'attempt_statuses': distribution(r['status'] for r in receipts),
        'attempts_per_window': distribution(r['attempts'] for r in result_rows),
        'invocations': invocations,
        'runtime': {'wall_seconds': wall, 'request_seconds_total': sum(seconds),
            'request_seconds_mean': statistics.mean(seconds), 'request_seconds_median': statistics.median(seconds),
            'request_seconds_min': min(seconds), 'request_seconds_max': max(seconds),
            'wall_seconds_per_window_throughput': wall / len(rows),
            'request_seconds_per_window_including_retries': sum(seconds) / len(rows)},
        'features': feature_summary, 'gate': {k: v for k, v in gate.items() if k != 'result_hashes'},
        'example_context_triple': example()['events'][0]['context_scope'],
        'events_equal_example_context_triple': sum(e['event'].get('context_scope') == example()['events'][0]['context_scope'] for e in events),
        'raw_issues_are_diagnostic_not_repaired_predictions': True,
    }
    after = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*') if p.is_file()}
    assert before == after, 'Run changed while auditing'
    summary['input_files_verified_unchanged'] = len(before)
    out = project / 'docs' / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in [('audit.json', summary), ('all_windows.json', result_rows), ('all_events.json', events), ('all_links.json', links)]:
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# All 36 V9.4 Development Windows', '',
             'No prediction or annotation was changed. Labels below include raw partial responses, not gold labels.', '',
             '| ID | Video / frames | Reasons | V9.3 status/label | V9.4 status/label | Valid events | Local support | Calls |',
             '|---|---|---|---|---|---:|---|---:|']
    for r in result_rows:
        case = root / 'cases' / r['uid'][:20] / 'index.html'
        lines.append(f'| [{r["uid"][:20]}]({desktop_link(case)}) | {r["video"]} [{r["start_frame"]},{r["end_frame_exclusive"]}) | '
            f'{", ".join(r["reasons"])} | {r["old_status"]}/{r["old_label"]} | {r["status"]}/{r["b5"]["label"]} | '
            f'{r["valid_events"]}/{r["event_count"]} | {r["supported_events"]} | {r["attempts"]} |')
    for r in result_rows:
        lines += ['', '## ' + r['uid'][:20], '', r['video'], '', '```json', json.dumps(r, ensure_ascii=False, indent=2), '```']
    (out / 'ALL_36_WINDOWS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
