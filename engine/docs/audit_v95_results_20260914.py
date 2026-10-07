"""Read-only V9.5 run audit; writes derived diagnostics under docs, never runs APIs."""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT / 'tools'), str(PROJECT / 'docs')]
from event_decision.safety import OfflineGuard
from event_decision.contracts import file_sha256
from event_decision.role_scoped import portable
from event_decision.mechanism_v95_store import load
from event_decision.mechanism_v95_report import current, technical_metrics
from event_decision.mechanism_v95_contract import FEATURES, feature_row


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(folder):
    return {str(p.relative_to(folder)): file_sha256(p) for p in folder.rglob('*') if p.is_file()}


def counts(values):
    return dict(Counter(values))


def get_path(data, path):
    for token in path.strip('/').split('/'):
        try:
            data = data[int(token)] if isinstance(data, list) else data[token]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    return data


def local_link(path):
    value = str(path).replace('\\', '/')
    if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        return value[5].upper() + ':/' + value[7:]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT/'runs/governed_v95_typed_mechanism_development_20260914')
    parser.add_argument('--out', type=Path, default=PROJECT/'docs/v95_result_audit_20260914')
    args = parser.parse_args()
    OfflineGuard().install()
    run, out = args.run.resolve(), args.out.resolve()
    if (PROJECT/'docs').resolve() not in out.parents:
        raise ValueError('Derived audit output must be under docs')
    before = fingerprint(run)
    protocol, rows = load(run)
    records = [current(run, row, protocol) for row in rows]
    assert not protocol['mock'] and len(records) == 36
    gate = read(run/'integration/gate.json')
    measured = technical_metrics(records, protocol)
    assert all(gate[k] == v for k, v in measured.items())
    assert all(read(run/'summary.json')[k] == v for k, v in measured.items())
    old_records = [read(run/'source_snapshot/results'/(row['window_uid']+'.json')) for row in rows]
    assert all(gate['result_hashes'][row['window_uid']] == file_sha256(run/'results'/(row['window_uid']+'.json')) for row in rows)
    cases, events, links, enum_issues = [], [], [], []
    groups = defaultdict(list)
    for row, new, old in zip(rows, records, old_records):
        uid = row['window_uid']; a = new['assessment']; raw = new['parsed']
        old_raw = old['parsed']
        case = {'id': uid[:20], 'window_uid': uid, 'video_id': row['video_id'], 'source_group': row['source_group'],
                'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
                'selection_reasons': row['selection_reasons'], 'old_status': old['status'], 'status': new['status'],
                'old_b5': old_raw['b5'], 'new_b5': raw['b5'], 'local_supported': a['local_supported_events'],
                'events_valid': sum(e['valid'] for e in a['events']), 'events_total': len(a['events']),
                'judgment_consistent': a['fields']['/b5/consistency']['valid'], 'issues': a['issues'],
                'context': raw['context'], 'raw_new': raw,
                'case_page': local_link(run/'cases'/uid[:20]/'index.html')}
        cases.append(case)
        for reason in row['selection_reasons']:
            groups[reason].append(case)
        for issue in a['issues']:
            if issue['reason'] == 'invalid enum':
                enum_issues.append({'id': uid[:20], **issue, 'raw_value': get_path(raw, issue['path'])})
        for i, (event, checked) in enumerate(zip(raw['events'], a['events'])):
            prefix = f'/events/{i}/'
            own_issues = [issue for issue in a['issues'] if issue['path'].startswith(prefix)]
            events.append({'case_id': uid[:20], 'video_id': row['video_id'], 'index': i,
                           'event': event, 'check': checked, 'issues': own_issues,
                           'entities': raw['persons'] + raw['objects']})
        for i, (link, checked) in enumerate(zip(raw['links'], a['links'])):
            links.append({'case_id': uid[:20], 'index': i, 'link': link, 'check': checked,
                          'issues': [x for x in a['issues'] if x['path'].startswith(f'/links/{i}/')]})
    attempts = [read(p) for p in (run/'attempts').rglob('*.json')]
    invocations = [read(p) for p in (run/'invocations').glob('*.json')]
    elapsed = [a['elapsed_seconds'] for a in attempts]
    start = min(i['started_unix'] for i in invocations)
    end = max(i['finished_unix'] for i in invocations)
    duration = sum(i['finished_unix'] - i['started_unix'] for i in invocations)
    tz = timezone(timedelta(hours=8))
    values = np.load(run/'features/values.npy', allow_pickle=False)
    masks = np.load(run/'features/observed.npy', allow_pickle=False)
    assert values.shape == masks.shape == (36, len(FEATURES))
    assert np.isnan(values[~masks]).all() and np.isfinite(values[masks]).all()
    for i, result in enumerate(records):
        replay = feature_row(result['parsed'], result['assessment'])
        for j, name in enumerate(FEATURES):
            assert masks[i, j] == replay[name]['observed']
            if masks[i, j]: assert np.isclose(values[i, j], replay[name]['value'])
    traces = [json.loads(line) for line in (run/'features/trace.jsonl').read_text().splitlines()]
    assert all(t['window_target'] is None and not t['training_loss_mask'] and not t['evaluation_loss_mask'] for t in traces)
    source = portable(protocol['source_run'])
    source_hashes = read(run/'offline/audit_summary.json')['source_hashes']
    assert all(file_sha256(source/name) == digest for name, digest in source_hashes.items())
    assert file_sha256(run/'selection.json') == file_sha256(source/'selection.json')
    visible = [e for e in events if any(x['path'].endswith('/binding/visible_support') for x in e['issues'])]
    typed = [e for e in events if any(x['path'].endswith('/binding/typed_support') for x in e['issues'])]
    typed_visible = [e for e in visible if e in typed]
    metrics = {
        'scope': 'same development cohort; no gold accuracy or AP', 'run': str(run),
        'windows': len(rows), 'videos': len({r['video_id'] for r in rows}), 'sources': len({r['source_group'] for r in rows}),
        'technical': measured, 'decision': gate['decision'], 'old_status': counts(r['status'] for r in old_records),
        'new_status': counts(r['status'] for r in records),
        'old_labels': counts(r['parsed']['b5']['label'] for r in old_records),
        'new_labels': counts(r['parsed']['b5']['label'] for r in records),
        'transitions': counts(c['old_b5']['label']+' -> '+c['new_b5']['label'] for c in cases),
        'groups': {name: {'n': len(cs), 'old_labels': counts(c['old_b5']['label'] for c in cs),
                         'new_labels': counts(c['new_b5']['label'] for c in cs),
                         'windows_with_local_support': sum(bool(c['local_supported']) for c in cs),
                         'transitions': counts(c['old_b5']['label']+' -> '+c['new_b5']['label'] for c in cs)} for name, cs in groups.items()},
        'windows_with_any_issue': sum(bool(c['issues']) for c in cases),
        'clean_windows': [c['id'] for c in cases if not c['issues']],
        'yes_without_local_support': [c['id'] for c in cases if c['new_b5']['label']=='yes' and not c['local_supported']],
        'windows_with_local_support': sum(bool(c['local_supported']) for c in cases),
        'events': {'total': len(events), 'valid': sum(e['check']['valid'] for e in events),
                   'observation_valid': sum(e['check']['observation_valid'] for e in events),
                   'binding_valid': sum(e['check']['binding_valid'] for e in events),
                   'supported': sum(e['check']['supported'] for e in events),
                   'types': counts(e['event']['type'] for e in events), 'phases': counts(e['event']['phase'] for e in events),
                   'kinds': counts(e['event']['binding']['kind'] for e in events),
                   'boundary': counts(e['event']['boundary']['b5_state'] for e in events),
                   'needs_more': {k: counts(e['event']['additional_evidence_needed'][k] for e in events) for k in ('mechanism','binding','b5_category')},
                   'visible_failures':len(visible), 'typed_failures':len(typed), 'visible_failures_with_typed_failure':len(typed_visible),
                   'visible_failures_without_typed_support_failure': [{'id':e['case_id'],'event_id':e['event']['id']} for e in visible if e not in typed]},
        'enum_issues': enum_issues, 'link_counts': counts(l['link']['state'] for l in links),
        'context_counts': {key: counts(c['context'][key] for c in cases) for key in ('cross_event_link_needs_more_evidence','story_context_known')},
        'local_support_despite_global_context_need': [c['id'] for c in cases if c['local_supported'] and c['context']['cross_event_link_needs_more_evidence'] in ('yes','unknown')],
        'numeric_by_label': {label: {'n':sum(c['new_b5']['label']==label for c in cases),
                                   'numeric':sum(c['new_b5']['label']==label and c['new_b5'].get('probability') is not None for c in cases)} for label in ('yes','no','unknown')},
        'runtime': {'attempts':len(attempts), 'attempt_status':counts(a['status'] for a in attempts),
                    'attempts_per_window_distribution':counts(Counter(a['window_uid'] for a in attempts).values()),
                    'invocations':len(invocations), 'calls_recorded':sum(i['calls'] for i in invocations),
                    'start_utc8':datetime.fromtimestamp(start,tz).isoformat(), 'end_utc8':datetime.fromtimestamp(end,tz).isoformat(),
                    'wall_seconds':duration, 'throughput_seconds_per_window':duration/len(rows),
                    'mean_request_seconds':statistics.mean(elapsed), 'median_request_seconds':statistics.median(elapsed),
                    'min_request_seconds':min(elapsed),'max_request_seconds':max(elapsed),'sum_request_seconds':sum(elapsed)},
        'features': {'shape':list(values.shape),'masked':int((~masks).sum()),'observed':int(masks.sum()),
                     'availability':read(run/'features/availability_summary.json')},
        'review_files': [str(p.relative_to(run)) for p in (run/'review').iterdir()],
        'integrity': {'frozen_load_passed':True,'summary_gate_replayed':True,'feature_replayed':True,
                      'source_files_unchanged':len(source_hashes),'same_selection':True,'run_file_count':len(before)},
    }
    after = fingerprint(run)
    assert before == after, 'Source run changed during audit'
    metrics['integrity']['audit_read_only'] = True
    out.mkdir(parents=True,exist_ok=True)
    for name, obj in [('audit.json',metrics),('all_cases.json',cases),('all_events.json',events),('all_links.json',links)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table=['# All 36 V9.5 Development Windows', '', 'Model labels only; yes/no refer to B5, not binary anomaly GT. No new human adjudication.', '',
           '| Case / Video | Frames [start,end) | V9.4 -> V9.5 | Local support | Valid events | Issues |',
           '|---|---|---|---|---|---|']
    for c in cases:
        win=f"[{c['start_frame']},{c['end_frame_exclusive']})"
        table.append(f"| [{c['id']}]({c['case_page']}) {c['video_id']} | {win} | {c['old_b5']['label']} -> {c['new_b5']['label']} | {','.join(c['local_supported']) or 'none'} | {c['events_valid']}/{c['events_total']} | {len(c['issues'])} |")
    (out/'ALL_36_WINDOWS.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in {'features','enum_issues','groups'}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
