"""Read-only V919 progress/cost/validity audit, without fitting on partial data."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.project / 'tools'), str(args.project / 'docs')]
    from event_decision.contracts import read_json, iter_jsonl, file_sha256, semantic_sha256, write_json
    from event_decision.effectiveness_v919 import pipeline, acquisition
    from event_decision.effectiveness_v919.scoring import features, BASES
    from graph_catalog import read_catalog_json
    inventory = pipeline.implementation
    pipeline.implementation = lambda project: {k.replace('\\', '/'): v for k, v in inventory(project).items()}
    config = pipeline.verify(args.project, args.run)
    rows = list(iter_jsonl(args.run / 'pilot/inputs.jsonl'))
    labels = {r['window_uid']: r for r in iter_jsonl(args.run / 'private/pilot_labels.jsonl')}
    results = {p.stem: read_json(p) for p in (args.run / 'pilot/results').glob('*.json')}
    mismatches = []
    cache = {}
    for row in rows:
        r = results.get(row['window_uid'])
        if r is None:
            continue
        if r['record_sha256'] != semantic_sha256({k: v for k, v in r.items() if k != 'record_sha256'}):
            mismatches.append(row['window_uid'] + ': result hash')
        if r['input_sha256'] != semantic_sha256(row) or r['config_sha256'] != semantic_sha256(config):
            mismatches.append(row['window_uid'] + ': input/config hash')
        for key, digest in r['cache_sha256'].items():
            if key not in cache:
                cache[key] = file_sha256(args.run / 'cache' / (key + '.json'))
            if cache[key] != digest:
                mismatches.append(key + ': cache hash')
        acquisition.verify_refusals(args.run, r)
    ff = {k: features(v) for k, v in results.items()}
    video = defaultdict(list)
    for r in rows:
        video[r['video_id']].append(r['window_uid'])
    roles = {}
    for role in ('fit', 'calibration', 'validation'):
        rr = [r for r in rows if r['role'] == role]
        supervised = [r for r in rr if labels[r['window_uid']]['loss_mask']]
        done = [r for r in supervised if r['window_uid'] in ff]
        valid = sum(ff[r['window_uid']]['local_valid'] for r in done)
        roles[role] = {'planned_windows': len(rr), 'completed_windows': sum(r['window_uid'] in ff for r in rr),
            'supervised_planned': len(supervised), 'supervised_completed': len(done), 'local_valid_supervised': valid,
            'local_valid_best_possible_if_all_remaining_valid': (valid + len(supervised) - len(done)) / len(supervised)}
    cost = acquisition.cost_summary(args.run, 'pilot')
    cat = read_catalog_json(args.run / 'graph_catalog.json')
    missing = [r for r in rows if r['window_uid'] not in results]
    report = {'snapshot_latest_pause': read_json(args.run / 'pilot/last_pause.json'),
        'protocol': config, 'results': len(results), 'planned_windows': len(rows),
        'complete_videos': sum(all(x in results for x in xx) for xx in video.values()), 'planned_videos': len(video),
        'roles': roles, 'missing_windows': missing,
        'catalog': {'graphs': len(cat), 'polarities': dict(Counter(g.polarity for g in cat.values()))},
        'result_integrity_mismatches': mismatches, 'verified_response_cache_files': len(cache),
        'baseline_present': {k: sum(f['values'][k] is not None for f in ff.values()) for k in BASES},
        'C1_present': sum('C1' in r for r in results.values()), 'C2_present': sum('C2' in r for r in results.values()),
        'C2_empty_parent_skips': sum('C2_skipped' in r for r in results.values()),
        'local_valid': sum(f['local_valid'] for f in ff.values()),
        'result_errors': dict(Counter(e.get('error', e.get('error_type')) for r in results.values() for e in r.get('errors', []))),
        'cost': cost, 'remaining_attempts_under_14800': 14800 - cost['physical_attempts'],
        'formal_accuracy': None, 'formal_AP': None, 'partial_fit_performed': False, 'new_API_calls': 0,
        'dense_gate_can_reach_local_valid_095': roles['validation']['local_valid_best_possible_if_all_remaining_valid'] >= .95,
        'fit_exists': (args.run / 'models/frozen.json').exists(),
        'pilot_evaluation_exists': (args.run / 'pilot/evaluation.json').exists()}
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing_windows', 'protocol', 'cost', 'result_errors')}, indent=2))


if __name__ == '__main__':
    main()
