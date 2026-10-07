"""Read-only source audit; derived JSON and tables go under docs, never runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tools'))
from event_decision.development_mechanism_v93 import load_run, validate, evidence_state
from event_decision.safety import OfflineGuard


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def counts(values):
    return dict(Counter(values))


def raw_object(raw):
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else None
    try:
        return json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT / 'runs/governed_v93_b5_mechanism_development_20260913')
    parser.add_argument('--out', type=Path, default=PROJECT / 'docs/v93_result_audit_20260913')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if not out.is_relative_to(PROJECT / 'docs'):
        raise ValueError('Derived output must stay under project/docs')
    OfflineGuard().install()
    protocol, windows = load_run(run)
    source_files = [run / 'summary.json', run / 'protocol.json', run / 'windows.json']
    source_files += sorted((run / 'results').glob('*.json'))
    source_files += sorted((run / 'attempts').glob('*/*.json'))
    source_files += sorted((run / 'invocations').glob('*.json'))
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    records = []
    for row in windows:
        path = run / 'results' / (row['window_uid'] + '.json')
        result = read(path) if path.exists() else {}
        if result.get('status') == 'success':
            validate(result['parsed'])
            assert evidence_state(result['parsed']) == result['evidence_diagnostic']
        records.append({'row': row, 'result': result, 'status': 'excluded' if row['exclusion'] else result.get('status', 'pending')})
    attempts = []
    for path in sorted((run / 'attempts').glob('*/*.json')):
        item = read(path)
        item['path'] = str(path.relative_to(run))
        item['parsed_raw_for_audit_only'] = raw_object(item.get('raw'))
        attempts.append(item)
    invocations = [read(p) for p in sorted((run / 'invocations').glob('*.json'))]

    def subset(items):
        ok = [x for x in items if x['status'] == 'success']
        parsed = [x['result']['parsed'] for x in ok]
        diagnostics = [x['result']['evidence_diagnostic'] for x in ok]
        events = [e for v in parsed for e in v['subevents']]
        links = [e for v in parsed for e in v['event_links']]
        return {
            'n': len(items), 'videos': len({x['row']['video_id'] for x in items}),
            'sources': len({x['row']['source_group'] for x in items}),
            'status': counts(x['status'] for x in items),
            'old_all': counts(x['row']['baseline'].get('parsed', {}).get('b5_presence') for x in items),
            'old_on_success': counts(x['row']['baseline']['parsed']['b5_presence'] for x in ok),
            'new_on_success': counts(v['b5_presence'] for v in parsed),
            'transitions': counts(x['row']['baseline']['parsed']['b5_presence'] + ' -> ' + x['result']['parsed']['b5_presence'] for x in ok),
            'phases': counts(v['phase'] for v in parsed),
            'quality': counts(v['observation_quality'] for v in parsed),
            'context_needed': counts(str(v['context_needed']) for v in parsed),
            'diagnostics': counts(v['state'] for v in diagnostics),
            'reason_incidence': counts(r for v in diagnostics for r in v['reasons']),
            'context_only_unresolved': sum(v['state'] == 'unresolved' and v['reasons'] == ['additional_context_requested'] for v in diagnostics),
            'supported_subevents_but_context_only_unresolved': sum(bool(v['supported_subevents']) and v['reasons'] == ['additional_context_requested'] for v in diagnostics),
            'supported_subevents_windows': sum(bool(v['supported_subevents']) for v in diagnostics),
            'subevent_count': len(events), 'subevent_phase': counts(e['phase'] for e in events),
            'subevent_mechanism': counts(e['mechanism'] for e in events), 'subevent_binding': counts(e['binding'] for e in events),
            'subevent_scope': counts(e['evidence_scope'] for e in events),
            'event_link_count': len(links), 'event_link_kind': counts(e['kind'] for e in links),
            'event_link_state': counts(e['relation_state'] for e in links),
            'other_category_not_semantically_validated': counts(v['other_event']['category'] for v in parsed),
            'probability_mean_old_on_success': statistics.mean(x['row']['baseline']['parsed']['b5_probability'] for x in ok) if ok else None,
            'probability_mean_new_on_success': statistics.mean(v['b5_probability'] for v in parsed) if ok else None,
        }

    eligible = [x for x in records if not x['row']['exclusion']]
    design = [x for x in eligible if x['row']['design_feedback']]
    remaining = [x for x in eligible if not x['row']['design_feedback']]
    failed = [x for x in eligible if x['status'] == 'failed']
    attempt_groups = {}
    for a in attempts:
        attempt_groups.setdefault(a['window_uid'], []).append(a)
    failed_audit = []
    for x in failed:
        calls = sorted(attempt_groups.get(x['row']['window_uid'], []), key=lambda a: a['started_unix'])
        failed_audit.append({
            'uid': x['row']['window_uid'], 'video': x['row']['video_id'], 'start': x['row']['start_frame'],
            'error': x['result']['error'], 'failure_kind': x['result']['failure_kind'],
            'attempts': [{k: a.get(k) for k in ('path', 'failure_kind', 'error', 'elapsed_seconds', 'parsed_raw_for_audit_only')} for a in calls],
        })
    successes = [x for x in eligible if x['status'] == 'success']
    successes_after_retry = sum(len(attempt_groups[x['row']['window_uid']]) > 1 for x in successes)
    recovered_errors = [a for x in successes for a in attempt_groups[x['row']['window_uid']] if a['status'] != 'success']
    elapsed = [a['elapsed_seconds'] for a in attempts]
    tz = timezone(timedelta(hours=8))
    audit = {
        'source_run': str(run), 'integrity_passed': True, 'source_hashes': hashes,
        'overall_planned': subset(records), 'eligible_91': subset(eligible),
        'design_8': subset(design), 'remaining_83': subset(remaining),
        'terminal_error_counts': counts(x['result']['error'] for x in failed),
        'terminal_failure_kinds': counts(x['result']['failure_kind'] for x in failed),
        'attempt_status': counts(a['status'] for a in attempts),
        'attempt_error_counts': counts(a.get('error') for a in attempts if a['status'] != 'success'),
        'attempt_count_distribution': counts(str(len(v)) for v in attempt_groups.values()),
        'successful_after_retry': successes_after_retry,
        'recovered_failure_attempts': len(recovered_errors),
        'failed_raw_phase_counts_attempt_level_not_results': counts(a['parsed_raw_for_audit_only'].get('phase') for a in attempts if a['status'] != 'success' and a['parsed_raw_for_audit_only']),
        'raw_json_decodable_attempts': sum(a['parsed_raw_for_audit_only'] is not None for a in attempts),
        'timing': {
            'attempts': len(attempts), 'sum_sdk_seconds': sum(elapsed), 'mean_sdk_seconds': statistics.mean(elapsed),
            'median_sdk_seconds': statistics.median(elapsed), 'min_sdk_seconds': min(elapsed), 'max_sdk_seconds': max(elapsed),
            'active_invocation_wall_seconds': sum(i['finished_unix'] - i['started_unix'] for i in invocations),
            'calendar_span_seconds': max(i['finished_unix'] for i in invocations) - min(i['started_unix'] for i in invocations),
            'invocations': [{**i, 'started_local': datetime.fromtimestamp(i['started_unix'], tz).isoformat(),
                             'finished_local': datetime.fromtimestamp(i['finished_unix'], tz).isoformat(),
                             'wall_seconds': i['finished_unix'] - i['started_unix']} for i in invocations],
        },
        'per_source': {s: subset([x for x in eligible if x['row']['source_group'] == s]) for s in sorted({x['row']['source_group'] for x in eligible})},
        'failed_raw_diagnostic_only': failed_audit,
        'note': 'Read-only audit. No cache repairs, no API, no formal metrics, no promotion of failed responses.',
    }
    for path in source_files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path.relative_to(run))], 'source changed during audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['case', 'video / [start,end)', 'design', 'status', 'old B5 / p', 'new B5 / p', 'phase', 'diagnostic / reason']
    table = ['# All 93 Windows: Frozen Result Audit', '', 'Failed and excluded rows are not normal predictions. No new accuracy/AP is computed.', '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for x in records:
        r, result = x['row'], x['result']; old = r['baseline'].get('parsed', {}); new = result.get('parsed', {})
        d = result.get('evidence_diagnostic', {})
        url = (run / 'cases' / r['window_uid'][:20] / 'index.html').as_posix()
        if url.startswith('/mnt/c/'): url = 'C:/' + url[7:]
        cells = [f'[{r["window_uid"][:20]}]({url})', f'{r["video_id"]} [{r["start_frame"]},{r["end_frame_exclusive"]})',
                 str(bool(r['design_feedback'])), x['status'], str(old.get('b5_presence')) + ' / ' + str(old.get('b5_probability')),
                 str(new.get('b5_presence')) + ' / ' + str(new.get('b5_probability')), str(new.get('phase')),
                 str(d.get('state') or result.get('error') or r['exclusion']) + ' ' + ', '.join(d.get('reasons', []))]
        table.append('| ' + ' | '.join(c.replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |')
    (out / 'ALL_93_WINDOWS.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k not in {'source_hashes', 'per_source', 'failed_raw_diagnostic_only'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
