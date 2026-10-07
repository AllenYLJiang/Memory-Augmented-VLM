"""Portable V919, fresh model-specific run, identical frozen pilot enrollment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'
sys.path[:0] = [str(ENGINE / 'tools'), str(ENGINE / 'docs')]
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from local_backend import LocalVLM, model_identity, network_disabled, sha
from event_decision.contracts import read_json, write_json, write_jsonl, iter_jsonl, semantic_sha256
from event_decision.b1b4_trial.protocol import immutable, run_lock
from event_decision.effectiveness_v919 import pipeline as p, acquisition as acq

_inventory = p.implementation
p.implementation = lambda project: {k.replace('\\', '/'): v for k, v in _inventory(project).items()}


def adapter_inventory():
    return {f.name: sha(f) for f in sorted(ROOT.glob('*.py'))}


def prepare(args, out):
    if (out / 'protocol.json').exists():
        raise ValueError('Already prepared. Resume with stage 2; do not re-enroll an existing TAG')
    fixture = ROOT / 'frozen_cohort'
    seal = read_json(fixture / 'inventory.json')
    for name, digest in seal.items():
        if sha(fixture / name) != digest:
            raise ValueError('Transferred cohort changed: ' + name)
    source = read_json(fixture / 'protocol.json')
    model = model_identity(args.vlm_dir)
    if (model['model_type'] != 'qwen3_vl' or model['text_shape'] !=
            {'hidden_size': 4096, 'num_hidden_layers': 36, 'intermediate_size': 12288}):
        raise ValueError('Expected Qwen3-VL-Instruct model directory')
    config = {**source, 'version': 'v919_local_qwen8b_paired_1', 'model': 'Qwen/Qwen3-VL-8B-Instruct',
              'workers': 1, 'code_sha256': semantic_sha256(p.implementation(ENGINE)),
              'catalog_origin': str(fixture / 'graph_catalog.json'),
              'local_backend': {'identity': model, 'device': args.device, 'dtype': args.dtype,
                  'attention': args.attention, 'decoding': 'greedy', 'adapter_sha256': adapter_inventory(),
                  'versions': {x: importlib.metadata.version(x) for x in ('torch', 'transformers', 'numpy', 'scikit-learn', 'Pillow')},
                  'LLM_used_in_effectiveness': False}}
    train = Path(args.train_root).resolve()
    files = {}
    for path in train.rglob('*.mp4'):
        files.setdefault(path.stem, []).append(path)
    rows = list(iter_jsonl(fixture / 'inputs.jsonl'))
    expected = read_json(fixture / 'media_expectations.json')
    identities, remapped = {}, []
    for row in rows:
        vid = row['video_id']
        matches = files.get(vid, [])
        if len(matches) != 1:
            raise ValueError(f'Expected one exact video for {vid}, found {len(matches)}')
        path = matches[0]
        if path.stat().st_size != row['video_size']:
            raise ValueError('Transferred video size differs: ' + vid)
        if vid not in identities:
            digest = sha(path)
            if vid in expected and expected[vid]['source_sha256'] != digest:
                raise ValueError('Transferred video hash differs: ' + vid)
            identities[vid] = {'source_sha256': digest, 'path': str(path), 'compared_to_API_video_hash': vid in expected}
        remapped.append({**row, 'video_path': str(path), 'video_mtime_ns': path.stat().st_mtime_ns})
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture / 'graph_catalog.json', out / 'graph_catalog.json')
    immutable(out / 'code_inventory.json', p.implementation(ENGINE))
    immutable(out / 'protocol.json', config)
    write_jsonl(out / 'pilot/inputs.jsonl', remapped)
    (out / 'private').mkdir(exist_ok=True)
    shutil.copyfile(fixture / 'pilot_labels.jsonl', out / 'private/pilot_labels.jsonl')
    shutil.copyfile(fixture / 'enrollment.json', out / 'pilot/enrollment.json')
    immutable(out / 'media_expectations.json', expected)
    immutable(out / 'portable_media_identity.json', identities)
    immutable(out / 'transfer_receipt.json', {'source_inventory': seal, 'same_window_uids': True,
        'same_frames_and_roles_and_labels': True, 'API_response_caches_imported': 0,
        'changed_fields_in_inputs_only': ['video_path', 'video_mtime_ns'],
        'source_protocol_sha256': seal['protocol.json'], 'source_input_sha256': seal['inputs.jsonl'],
        'model_size_alone_not_isolated': True})
    return p.plan(out, 'pilot', config)


def verify(out):
    config = p.verify(ENGINE, out)
    local = config.get('local_backend')
    if not local or config['model'] != 'Qwen/Qwen3-VL-8B-Instruct':
        raise ValueError('Not a local run; never resume the API TAG with another model')
    if local['adapter_sha256'] != adapter_inventory():
        raise ValueError('Frozen local adapter changed; new TAG required')
    if local['versions'] != {x: importlib.metadata.version(x) for x in local['versions']}:
        raise ValueError('Dependency versions changed; preserve environment or use a new TAG')
    return config


def bind_local_media(out):
    original = acq.prepare_video_media
    expected = read_json(out / 'media_expectations.json', {})
    identities = read_json(out / 'portable_media_identity.json', {})
    def checked(root, rows):
        result = original(root, rows)
        vid = rows[0]['video_id']
        folder = root / 'media' / semantic_sha256(vid)[:24]
        receipt = read_json(folder / 'receipt.json')
        if vid in identities and receipt['source_sha256'] != identities[vid]['source_sha256']:
            raise ValueError('Prepared video bytes changed')
        if vid in expected:
            # Eight frame indices alone do not guarantee pixel equivalence across decoders.
            for name, digest in expected[vid]['files'].items():
                if receipt['files'].get(name) != digest:
                    raise ValueError('Exact API frame JPEG differs: ' + vid + '/' + name + '; transfer original media cache')
        return result
    acq.prepare_video_media = checked


def local_runtime_summary(out, phase):
    import numpy as np
    attempts = [read_json(f) for f in (out / phase / 'cost/attempts').glob('*.json')]
    usage = [r['usage'] for r in attempts if r.get('usage')]
    seconds = [r.get('local_generation_seconds', 0.) for r in usage]
    allocated = [r.get('cuda_peak_allocated_bytes', 0) for r in usage]
    reserved = [r.get('cuda_peak_reserved_bytes', 0) for r in usage]
    report = {'remote_API_calls': 0, 'local_attempts': len(attempts), 'generations_with_usage': len(usage),
              'completed_windows': len(list((out / phase / 'results').glob('*.json'))),
              'generation_seconds_sum': sum(seconds),
              'generation_latency_p50_p95_seconds': np.quantile(seconds, [.5, .95]).tolist() if seconds else None,
              'max_GPU_allocated_GiB': max(allocated, default=0) / 2**30,
              'mean_request_peak_allocated_GiB': float(np.mean(allocated)) / 2**30 if allocated else None,
              'max_GPU_reserved_GiB': max(reserved, default=0) / 2**30,
              'memory_note': 'request peaks include resident model, not additive per-window memory',
              'generations_hitting_output_cap': sum(bool(r.get('output_cap_reached')) for r in usage)}
    write_json(out / phase / 'local_runtime_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['1', '2', '3'], default='1')
    parser.add_argument('--action', choices=['plan', 'status', 'run', 'evaluate'], default='plan')
    parser.add_argument('--tag', default='v919_local_qwen8b_seed20260917')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--train-root', default=os.getenv('TRAIN_ROOT', 'G:/Dataset/XDViolence/train'))
    parser.add_argument('--test-root', default=os.getenv('TEST_ROOT', 'G:/Dataset/XDViolence/videos/videos'))
    parser.add_argument('--annotations', default=os.getenv('ANNOTATIONS', 'G:/Dataset/XDViolence/videos/annotations_uniform_format.txt'))
    parser.add_argument('--vlm-dir', default=os.getenv('VLM_MODEL_DIR', 'G:/Qwen3-VL-8B'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--attention', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    parser.add_argument('--max-attempts', type=int, default=14800)
    parser.add_argument('--max-reserved-output-tokens', type=int, default=121241600)
    parser.add_argument('--approve-local-compute', action='store_true')
    parser.add_argument('--approved-by', default=os.getenv('USERNAME', os.getenv('USER', '')))
    args = parser.parse_args()
    out = args.out or ROOT / 'runs' / args.tag
    with run_lock(out), network_disabled():
        if args.stage == '1':
            report = prepare(args, out)
        else:
            config = verify(out)
            phase = 'pilot' if args.stage == '2' else 'dense'
            if args.action == 'status':
                _, coverage = p.export_features(out, phase)
                report = {'coverage': coverage, 'cost': acq.cost_summary(out, phase)}
            elif args.action == 'plan':
                report = p.plan(out, phase, config) if phase == 'pilot' else p.prepare_test(out, args.test_root, config, args.annotations)
            else:
                if args.action == 'run':
                    if not args.approve_local_compute or not args.approved_by:
                        raise ValueError('Explicit --approve-local-compute and --approved-by required')
                    if phase == 'dense':
                        p.prepare_test(out, args.test_root, config, args.annotations)
                    local = config['local_backend']
                    if model_identity(local['identity']['directory']) != local['identity']:
                        raise ValueError('Frozen model files changed')
                    # Reuse the original validated request DAG and accounting, replace only its provider.
                    from event_decision.b1b4_trial import evidence
                    evidence.dashscope_once = LocalVLM(local['identity']['directory'], local['device'], local['dtype'], local['attention'])
                    bind_local_media(out)
                    from graph_catalog import read_catalog_json
                    budget = acq.Budget(out, phase, config, args.approved_by, args.max_attempts, args.max_reserved_output_tokens)
                    acq.collect(out, phase, list(iter_jsonl(out / phase / 'inputs.jsonl')),
                                read_catalog_json(out / 'graph_catalog.json'), config, budget)
                report = p.fit_and_evaluate(out, config) if phase == 'pilot' else p.dense_evaluate(out, args.annotations, config)
        if args.stage != '1':
            report['local_runtime'] = local_runtime_summary(out, phase)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('[stop] ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
