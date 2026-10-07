#!/usr/bin/env python3
"""Read-only B1 exposure audit. Produces review inputs, never an allow-list."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from event_decision.contracts import read_json, write_json, write_jsonl
from event_decision.hard_trial import PRUNE, group_id, groups_in
from event_decision.label_scope import _load_gather_anchors
from event_decision.safety import OfflineGuard


PLANNED = {'selected_source_records.jsonl', 'selected_windows.jsonl', 'selection.jsonl',
           'source_records.jsonl', 'sample_manifest.json', 'split_manifest.json',
           'split_registry.json', 'development_splits.json'}
SUPPLEMENTAL = {'predictions.jsonl', 'baseline_predictions.jsonl', 'train_predictions.jsonl',
                'closed_loop_log.jsonl', 'pot_log.jsonl', 'feature_dump.jsonl',
                'adjudicated_reviews.jsonl', 'human_reviews.jsonl', 'human_label_ledger.jsonl',
                'review_exposure_manifest.jsonl'}
REGISTRATIONS = {'label_ledger.jsonl', 'enrollment_manifest.jsonl'}


def portable_path(value):
    value = str(value).replace('\\', '/')
    if os.name == 'nt' and value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        return Path(value[5].upper() + ':/' + value[7:])
    if os.name != 'nt' and len(value) > 2 and value[1:3] == ':/':
        return Path('/mnt/' + value[0].lower() + '/' + value[3:])
    return Path(value)


def record_groups(row):
    result = groups_in(row)
    # Old Part-A/Part-B logs use "videos", not "video_ids".
    if isinstance(row, dict) and isinstance(row.get('videos'), list):
        result.update(group_id(v) for v in row['videos'] if isinstance(v, str))
    return result


def mock_context(path):
    """Trust an explicit config flag, not a directory containing 'mock'."""
    for parent in path.parents:
        if parent.name == 'runs':
            break
        for name in ('run_config.json', 'frozen_pair_summary.json'):
            config = parent / name
            if not config.is_file():
                continue
            try:
                data = read_json(config)
            except (ValueError, OSError):
                return None, str(config), 'UNREADABLE_MOCK_CONFIG'
            if isinstance(data, dict) and type(data.get('mock')) is bool:
                return data['mock'], str(config), ''
    return None, '', ''


def synthetic_record(row):
    if not isinstance(row, dict):
        return False
    if row.get('mock') is True:
        return True
    calls = row.get('independent_node_calls', {})
    if not isinstance(calls, dict) or not calls:
        return False
    return all(isinstance(x, dict) and str(x.get('visible_evidence', '')).startswith(
        'mock independent evidence for ') for x in calls.values())


def classify(name, mock, row):
    if name in PLANNED:
        return 'planned_registration'
    if name in REGISTRATIONS:
        return 'label_or_review_registration'
    if name == 'frozen_development_model.json':
        return 'model_registration'
    if name == 'graph_catalog_v2.json':
        return 'catalog_provenance_requires_review'
    if mock is True:
        return 'mock_config_execution'
    if synthetic_record(row):
        return 'mock_marker_execution' if mock is None else 'conflicting_mock_evidence'
    return 'execution_record' if mock is False else 'execution_record_unverified_mode'


def priority(kinds):
    if kinds and kinds <= {'planned_registration'}:
        return 'REVIEW_PLANNED_ONLY_NOT_CLEARED'
    if kinds and kinds <= {'planned_registration', 'mock_config_execution', 'mock_marker_execution'}:
        return 'REVIEW_MOCK_OR_PLANNED_ONLY_NOT_CLEARED'
    if not kinds:
        return 'UNRESOLVED_NO_VERIFIED_RECORD_NOT_CLEARED'
    return 'RETAIN_EXCLUSION_PENDING_PROVENANCE_REVIEW'


def supplemental_files(xd_root):
    for project in sorted(xd_root.iterdir()):
        root = project / 'runs'
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE)
            for name in sorted(set(files) & SUPPLEMENTAL):
                yield Path(directory) / name


def scan_file(path, target_groups, max_bytes, expected_hash=None):
    evidence, issues = [], []
    if not path.is_file():
        return [], [{'path': str(path), 'issue': 'MISSING_FILE'}], {}
    if path.stat().st_size > max_bytes:
        return [], [{'path': str(path), 'issue': 'FILE_OVER_BUDGET',
                     'bytes': path.stat().st_size}], {}
    mock, config, config_error = mock_context(path)
    if config_error:
        issues.append({'path': config, 'issue': config_error})
    digest = hashlib.sha256()
    count = 0
    all_groups = set()
    found = defaultdict(lambda: {'count': 0, 'first_lines': []})

    def consume(row, line):
        kind = classify(path.name, mock, row)
        row_groups = record_groups(row)
        all_groups.update(row_groups)
        for group in row_groups & target_groups:
            item = found[(group, kind)]
            item['count'] += 1
            if len(item['first_lines']) < 3:
                item['first_lines'].append(line)

    try:
        with path.open('rb') as handle:
            if path.suffix == '.jsonl':
                for line, raw in enumerate(handle, 1):
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    try:
                        consume(json.loads(raw), line)
                        count += 1
                    except (ValueError, TypeError) as exc:
                        issues.append({'path': str(path), 'line': line,
                                       'issue': 'INVALID_RECORD_' + type(exc).__name__})
            else:
                raw = handle.read()
                digest.update(raw)
                consume(json.loads(raw), 1)
                count = 1
    except (OSError, ValueError, TypeError) as exc:
        issues.append({'path': str(path), 'issue': type(exc).__name__})
    actual_hash = digest.hexdigest()
    if expected_hash and actual_hash != expected_hash:
        issues.append({'path': str(path), 'issue': 'CHANGED_SINCE_INVENTORY',
                       'expected_sha256': expected_hash, 'current_sha256': actual_hash})
    for (group, kind), value in sorted(found.items()):
        evidence.append({'source_group': group, 'kind': kind, 'path': str(path.resolve()),
                         'sha256': actual_hash, 'mock_config': config, 'mock_flag': mock, **value})
    meta = {'path': str(path.resolve()), 'sha256': actual_hash, 'records': count,
            'bytes': path.stat().st_size, 'target_groups_found': len({g for g, _ in found}),
            'source_groups_found': sorted(all_groups)}
    return evidence, issues, meta


def run(args):
    start = time.time()
    trial, out = args.trial.resolve(), args.out.resolve()
    if out == trial or trial in out.parents or out in trial.parents:
        raise ValueError('audit output must be separate from the existing trial')
    if out.exists():
        raise ValueError('output already exists; use a new audit TAG to preserve prior review files')
    inventory_path = trial / 'history/history_inventory.json'
    inventory = read_json(inventory_path)
    if not inventory or not args.train_root.is_dir():
        raise ValueError('existing history inventory and a local train root are required')
    gather = _load_gather_anchors(args.xd_root / 'pipeline/tools')
    anchors = gather(args.xd_root / 'Transformer_semantic_components_select_anomaly/'
                     'top_anomalous_frames_72B_positive_segments')
    videos = {p.stem for p in args.train_root.rglob('*.mp4')
              if 'B1' in p.stem.split('_label_')[-1].split('-') and anchors.get(p.stem)}
    groups = {group_id(v) for v in videos} & set(inventory['source_groups'])
    files = {portable_path(x['path']).resolve(): x['sha256'] for x in inventory['files']}
    original_paths = set(files)
    if not args.no_supplemental:
        for path in supplemental_files(args.xd_root):
            files.setdefault(path.resolve(), None)
    out.mkdir(parents=True)
    evidence, issues, scanned = [], [], []
    for i, (path, expected) in enumerate(sorted(files.items()), 1):
        rows, errors, meta = scan_file(path, groups, args.max_file_mb * 1024 ** 2, expected)
        for row in rows:
            row['supplemental'] = path not in original_paths
        evidence.extend(rows)
        issues.extend(errors)
        if meta:
            scanned.append(meta)
        if i % 10 == 0 or i == len(files):
            print(f'[history-audit] {i}/{len(files)} files; issues={len(issues)}; API=0', flush=True)
    by_group = defaultdict(list)
    for row in evidence:
        by_group[row['source_group']].append(row)
    reports, reviews = [], []
    for group in sorted(groups):
        rows = by_group[group]
        original_kinds = {r['kind'] for r in rows if not r['supplemental']}
        kinds = {r['kind'] for r in rows}
        initial = priority(original_kinds)
        final = priority(kinds)
        candidate = initial.startswith('REVIEW_') or final.startswith('REVIEW_')
        reports.append({'source_group': group, 'video_ids': sorted(v for v in videos if group_id(v) == group),
                        'inventory_only_priority': initial, 'audit_priority': final,
                        'evidence_kinds': sorted(kinds), 'supplemental_hits': sum(r['supplemental'] for r in rows),
                        'eligible_for_enrollment': False, 'evidence': rows})
        if candidate:
            reviews.append({'source_group': group, 'audit_priority': final, 'disposition': 'pending',
                            'reviewer_id': '', 'evidence_notes': '',
                            'checked_fit_threshold_evaluation': False,
                            'checked_discovery_prompt_design_human_review': False,
                            'checked_external_history_and_source_aliases': False,
                            'eligible_for_enrollment': False})
    summary = {'version': 'v91_read_only_history_audit_v1', 'status': 'REQUIRES_HISTORY_REVIEW',
               'trial': str(trial), 'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
               'anchored_B1_videos': len(videos), 'excluded_B1_source_groups': len(groups),
               'files_requested': len(files), 'files_scanned': len(scanned), 'issues': len(issues),
               'inventory_only_priority_counts': dict(Counter(x['inventory_only_priority'] for x in reports)),
               'audit_priority_counts': dict(Counter(x['audit_priority'] for x in reports)),
               'review_rows': len(reviews), 'supplemental_search_enabled': not args.no_supplemental,
               'supplemental_names': sorted(SUPPLEMENTAL), 'elapsed_seconds': time.time() - start,
               'remote_calls': 0, 'released_sources': 0, 'changes_existing_exclusions': False,
               'remote_execution_authorized': False,
               'limitations': ['not an exhaustive external-history or visual-duplicate audit',
                              'mock may still involve human viewing or label-aware design',
                              'absence of a result file is not proof of non-use',
                              'review template has no import or automatic release effect']}
    write_jsonl(out / 'source_audit.jsonl', reports)
    write_jsonl(out / 'history_review_template.jsonl', reviews)
    write_jsonl(out / 'audit_errors.jsonl', issues)
    write_jsonl(out / 'scanned_files.jsonl', scanned)
    write_json(out / 'audit_summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--xd-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--max-file-mb', type=int, default=1024)
    parser.add_argument('--no-supplemental', action='store_true')
    args = parser.parse_args()
    if args.max_file_mb <= 0:
        parser.error('--max-file-mb must be positive')
    guard = OfflineGuard()
    guard.install()
    try:
        run(args)
        guard.assert_no_remote_calls()
    except (OSError, ValueError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
