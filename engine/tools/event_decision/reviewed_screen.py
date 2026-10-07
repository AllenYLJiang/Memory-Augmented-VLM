"""Source-reviewed local candidate screen; no inference or label adjudication."""
import subprocess
from collections import deque
from pathlib import Path

from .contracts import file_sha256, read_json, semantic_sha256, write_json, write_jsonl
from .hard_trial import group_id, weak_label
from .label_scope import _load_gather_anchors
from .local_screen import frame_probe, local_frames


def anchor_starts(spans, maximum=2):
    starts = sorted({max(0, (int(a) + int(b)) // 2 - 48) for a, b, _ in spans if b - a + 1 >= 8})
    selected = []
    for start in starts:
        if not selected or start >= selected[-1] + 96:
            selected.append(start)
        if len(selected) == maximum:
            break
    return selected


def pool_index(xd_root, train_root, history, maximum=2):
    gather = _load_gather_anchors(xd_root / 'pipeline/tools')
    folder = xd_root / 'Transformer_semantic_components_select_anomaly'
    positive = gather(folder / 'top_anomalous_frames_72B_positive_segments')
    negative = gather(folder / 'top_anomalous_frames_72B_negative_segments')
    used = set(history['source_groups'])
    unique = {}
    paths_n = 0
    for path in sorted(train_root.rglob('*.mp4')):
        paths_n += 1
        unique.setdefault(path.stem, path)
    pools = {k: [] for k in ('B1', 'B4', 'A', 'post', 'canary')}
    for video, path in unique.items():
        if group_id(video) in used:
            continue
        codes = video.split('_label_')[-1].split('-')
        if codes[0] == 'A':
            pools['A'].append((path, video, None))
            continue
        if positive.get(video) and set(codes) & {'B1', 'B2', 'B4', 'B5', 'B6', 'G'}:
            kind = 'B1' if 'B1' in codes else 'B4' if 'B4' in codes else 'canary'
            for start in anchor_starts(positive[video], maximum):
                pools[kind].append((path, video, start))
        outside = [s for s in negative.get(video, [])
                   if all(s[1] < a or b < s[0] for a, b, _ in positive.get(video, []))]
        for start in anchor_starts(outside, maximum):
            pools['post'].append((path, video, start))
    counts = {}
    for kind, items in pools.items():
        counts[kind] = {'videos': len({v for _, v, _ in items}),
                        'source_groups': len({group_id(v) for _, v, _ in items}),
                        'candidate_start_upper_bound_before_media': len(items)}
    return pools, positive, counts, {'video_paths': paths_n, 'unique_video_ids': len(unique),
                                   'duplicate_paths_not_resampled': paths_n - len(unique)}


def screen(xd_root, train_root, history, out, limit=384, seed=20260911, maximum=2, inventory_only=False):
    out = Path(out)
    if not Path(train_root).is_dir():
        raise ValueError('local training video root is absent')
    pools, positive, counts, paths = pool_index(Path(xd_root), Path(train_root), history, maximum)
    missing = [kind for kind, rows in pools.items() if not rows]
    requirements = {'B1': 24, 'B4': 24, 'A': 64, 'post': 16, 'canary': 16}
    shortages = {k: {'required': n, 'upper_bound': min(len(pools[k]), counts[k]['source_groups'] * 2)}
                 for k, n in requirements.items()
                 if min(len(pools[k]), counts[k]['source_groups'] * 2) < n}
    plan = {'pools': counts, 'path_deduplication': paths, 'missing_required_strata': missing,
            'obvious_capacity_shortages': shortages, 'media_verified': False,
            'source_disjoint_assignment_still_required': True}
    write_json(out / 'available_pool_before_decoding.json', plan)
    if inventory_only:
        return None, plan
    if missing or shortages:
        write_jsonl(out / 'candidate_windows.jsonl', [])
        write_jsonl(out / 'screening_errors.jsonl', [])
        write_json(out / 'screening_summary.json', {'status': 'WAITING_FOR_POOL_CAPACITY',
                   'screened': 0, 'usable': 0, 'remote_calls': 0, **plan})
        return out / 'candidate_windows.jsonl', plan
    queues = {}
    for kind, items in pools.items():
        # First window of each video before its second; ample label_A cannot starve other strata.
        ranks = {}
        for _, video, start in sorted(items, key=lambda x: (x[1], x[2] or 0)):
            ranks[(video, start)] = sum(v == video for v, _ in ranks)
        queues[kind] = deque(sorted(items, key=lambda x: (ranks[(x[1], x[2])], semantic_sha256([seed, x[1]]))))
    jobs = []
    while any(queues.values()) and len(jobs) < limit:
        for kind, queue in queues.items():
            if queue and len(jobs) < limit:
                jobs.append((kind, *queue.popleft()))
    rows, errors, probes, seen = [], [], {}, set()
    select = Path(xd_root) / 'Transformer_semantic_components_select_anomaly/top_anomalous_frames_72B_positive_segments'
    for ordinal, (kind, path, video, suggested) in enumerate(jobs, 1):
        try:
            if video not in probes:
                probes[video] = frame_probe(path)
            n = probes[video]
            if n < 96:
                raise ValueError('shorter than 96 frames')
            start = min(max(0, n // 2 - 48 if suggested is None else suggested), n - 96)
            if (video, start) in seen:
                continue
            seen.add((video, start))
            provenance = [{'path': str(p.resolve()), 'sha256': file_sha256(p)} for p in
                          (select / video / 'selected_frames.json', select / (video + '.mp4') / 'selected_frames.json')
                          if p.is_file()]
            key = semantic_sha256(['reviewed_screen_v1', str(path.resolve()), path.stat().st_size,
                                   path.stat().st_mtime_ns, start, kind, positive.get(video, []), provenance])
            cache = out / 'cache' / (key + '.json')
            row = read_json(cache)
            if row is None:
                row = {'dataset_partition': 'train', 'video_id': video, 'video_path': str(path.resolve()),
                       'start_frame': start, 'end_frame_exclusive': start + 96,
                       'stratum': {'A': 'label_A_pool', 'post': 'hard_postevent_unverified',
                                   'B1': 'B1_weak_positive', 'B4': 'B4_weak_positive', 'canary': 'other_class_canary'}[kind],
                       'label_source': 'explicit_filename_label_A' if kind == 'A' else
                                       'negative_anchor_unverified' if kind == 'post' else 'verified_positive_anchor',
                       'positive_spans_half_open': [[int(a), int(b) + 1] for a, b, _ in positive.get(video, [])],
                       'positive_span_provenance': provenance if kind not in ('A', 'post') else []}
                if kind in ('B1', 'B4', 'canary') and weak_label(row) != 1:
                    raise ValueError('no verified >=8-frame positive overlap')
                row.update(local_frames(path, start))
                write_json(cache, row)
            rows.append(row)
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
            errors.append({'video_id': video, 'error': str(exc)[:500]})
        if ordinal % 10 == 0 or ordinal == len(jobs):
            print(f'[reviewed-screen] {ordinal}/{len(jobs)} usable={len(rows)} errors={len(errors)} API=0', flush=True)
    normals = sorted((r for r in rows if r['stratum'] == 'label_A_pool'),
                     key=lambda r: (r['local_motion_score'], r['video_id']))
    easy = {r['video_id'] for r in normals[:max(16, len(normals) // 4)]}
    for row in normals:
        row['stratum'] = 'easy_label_A' if row['video_id'] in easy else 'hard_label_A'
    write_jsonl(out / 'candidate_windows.jsonl', rows)
    write_jsonl(out / 'screening_errors.jsonl', errors)
    write_json(out / 'screening_summary.json', {'status': 'LOCAL_SCREEN_COMPLETED', 'screened': len(jobs),
               'usable': len(rows), 'errors': len(errors), 'seed': seed, 'budget': limit,
               'maximum_positive_windows_per_video': maximum, 'remote_calls': 0,
               'postevent_labels': 'unverified and masked', 'motion_rank_is_not_human_hard_normal_validation': True})
    return out / 'candidate_windows.jsonl', plan
