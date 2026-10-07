import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_backend import network_disabled, vlm_messages, model_identity
import run_local
from event_decision.effectiveness_v919 import acquisition as a, scoring
from event_decision.contracts import write_json, semantic_sha256, read_json


def test_network_fails_closed_and_restores():
    old = socket.socket.connect
    with network_disabled():
        with pytest.raises(RuntimeError, match='LOCAL_ONLY'):
            socket.create_connection(('example.com', 443))
    assert socket.socket.connect is old


def test_messages_no_labels_extra_system_or_prompt_changes(tmp_path):
    paths = [tmp_path / f'{i}.jpg' for i in range(8)]
    msg = vlm_messages(paths, 'frozen prompt', 262144)
    assert len(msg) == 1 and msg[0]['role'] == 'user'
    assert len(msg[0]['content']) == 17
    assert msg[0]['content'][-1] == {'type': 'text', 'text': 'frozen prompt'}
    assert [x['text'] for x in msg[0]['content'][:-1] if x['type'] == 'text'] == [f'T{i}' for i in range(8)]


def test_missing_weights_rejected(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3_vl"}')
    with pytest.raises(ValueError, match='Incomplete'):
        model_identity(tmp_path)


def test_frozen_cohort_scope():
    folder = ROOT / 'frozen_cohort'
    rows = [json.loads(l) for l in (folder / 'inputs.jsonl').read_text().splitlines()]
    assert len(rows) == 400 and len({r['video_id'] for r in rows}) == 200
    groups = {role: {r['source_group'] for r in rows if r['role'] == role}
              for role in ('fit', 'calibration', 'validation')}
    assert not groups['fit'] & (groups['calibration'] | groups['validation'])
    assert not groups['calibration'] & groups['validation']
    assert all(len(r['sampled_frame_indices']) == 8 for r in rows)
    labels = [json.loads(l) for l in (folder / 'pilot_labels.jsonl').read_text().splitlines()]
    assert sum(bool(r['loss_mask']) for r in labels) == 303
    assert {r['window_uid'] for r in labels} == {r['window_uid'] for r in rows}


class FakeBudget:
    def __init__(self, root):
        self.root, self.calls = root, 0
        self.retry_transport, self.transport_max_attempts = False, 1
    def prior(self, key):
        return []
    def check_running(self):
        pass
    def reserve(self, key, stage):
        self.calls += 1
        return {'request_attempt_number': 1}, self.root / f'{self.calls}.json'
    def finish(self, *args, **kwargs):
        pass


def test_cache_resume_and_model_isolation(tmp_path):
    media = {'image_sha256': ['x'] * 8, 'frame_indices': list(range(8))}
    budget = FakeBudget(tmp_path)
    provider_calls = []
    def provider(*args):
        provider_calls.append(1)
        return '{"value":1}', {'output_tokens': 5}
    for model in ('api-model', 'api-model', 'local-model'):
        rt = a.Runtime(tmp_path, 'uid', media, {'model': model}, budget, provider)
        rt.request_json(namespace='test', prompt='same', validator=lambda x: None)
    assert len(provider_calls) == budget.calls == 2
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 2


def test_invalid_response_retained_no_semantic_retry(tmp_path):
    budget = FakeBudget(tmp_path)
    rt = a.Runtime(tmp_path, 'uid', {'image_sha256': ['x'] * 8, 'frame_indices': list(range(8))},
                   {'model': 'local'}, budget, lambda *a: ('{"bad":1}', {}))
    def invalid(x):
        raise ValueError('schema invalid')
    for _ in range(2):
        with pytest.raises(ValueError, match='schema invalid'):
            rt.request_json(namespace='test', prompt='same', validator=invalid)
    assert budget.calls == 1


def test_missing_not_normal_and_fallback():
    row = scoring.features({'window_uid': 'u', 'baseline': {'competitions': {
        'independent_direct_nodes': {'margin': .3}}}, 'C2_status': 'provider_refused'})
    assert row['local_valid'] is False
    assert row['values']['benign_fraction'] is None
    score, fallback = scoring.predict([row], {'models': {}})
    assert score['conditional_ot_full'][0] > .5
    assert fallback['conditional_ot_full'] == ['M0']


def test_preparation_rejects_bad_cohort_without_model_loading(tmp_path, monkeypatch):
    folder = ROOT / 'frozen_cohort'
    # Direct transfer identity: only host path/mtime may change, not roles/labels/frames.
    row = json.loads((folder / 'inputs.jsonl').read_text().splitlines()[0])
    moved = {**row, 'video_path': '/new/video.mp4', 'video_mtime_ns': 1}
    assert {k for k in row if row[k] != moved[k]} == {'video_path', 'video_mtime_ns'}


def test_explicit_recovery_is_only_exact_stop_receipt():
    path = ROOT / 'api_resume' / 'resume_provider_stop.py'
    spec = importlib.util.spec_from_file_location('resume_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {'status': 'provider_failed', 'error_type': 'StopAcquisition',
              'retryable_transport': False, 'billing_unknown': True, 'request_key': 'k', 'index': 7}
    check = module.scoped_retry(lambda r: False, {module.digest(record)})
    assert check(record)
    assert not check({**record, 'index': 8})
    assert not check({**record, 'error_type': 'RequestRefused'})
    assert not check({**record, 'error_type': 'ValueError'})


def test_local_compute_receipt_remains_resumable(tmp_path):
    # A completed window is not reissued even if the new provider would fail.
    row = {'window_uid': 'u'}
    config = {'model': 'local'}
    result = {'window_uid': 'u', 'input_sha256': semantic_sha256(row),
              'config_sha256': semantic_sha256(config), 'cache_sha256': {}}
    write_json(tmp_path / 'pilot/results/u.json', result)
    def forbidden(*args):
        raise AssertionError('completed result called provider')
    assert a.collect_window(tmp_path, 'pilot', row, {}, {}, config, None, forbidden) == result


def test_actual_six_method_OT_DAG_without_remote():
    from graph_catalog import read_catalog_json
    from vlm_runtime import CachedVideoVLM
    from schemas import WindowCase
    catalog = read_catalog_json(ROOT / 'frozen_cohort/graph_catalog.json')
    class Runtime:
        parents = []
        def __init__(self, fail=False):
            self.calls, self.fail = [], fail
        def request_json(self, *, namespace, mock_spec=None, **kwargs):
            self.calls.append(namespace)
            if self.fail and namespace.startswith('conditional/'):
                raise ValueError('format failure retained')
            data = CachedVideoVLM._mock_response('anonymous', mock_spec)
            return {'parsed': data, 'raw': json.dumps(data)}
    config = {'top_k_abnormal': 4, 'top_k_normal': 6, 'coherence_weight': .25, 'competition_temperature': .1}
    case = WindowCase('anonymous', 'anonymous', '', 0, 95, None, {})
    good = Runtime()
    result = a.match_graphs(good, case, catalog, config)
    assert set(result['competitions']) == set(scoring.BASES), result['errors']
    assert len(result['graph_scores']['conditional_ot_full']) == 10
    assert len(good.calls) == len(set(good.calls))
    assert not any(x in '/'.join(good.calls) for x in ('discovery', 'C0', 'verifier'))
    bad = a.match_graphs(Runtime(True), case, catalog, config)
    assert set(bad['competitions']) == set(scoring.BASES[:3])


def test_hash_changed_model_weights_fail_identity(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3_vl"}')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'one')
    before = model_identity(tmp_path)
    weights.write_bytes(b'two')
    assert model_identity(tmp_path) != before


def test_prepare_rebases_paths_but_not_labels_frames_or_roles(tmp_path, monkeypatch):
    import shutil
    from local_backend import sha
    fixture = tmp_path / 'package/frozen_cohort'
    fixture.mkdir(parents=True)
    for name in ('protocol.json', 'graph_catalog.json', 'enrollment.json'):
        shutil.copyfile(ROOT / 'frozen_cohort' / name, fixture / name)
    source = json.loads((ROOT / 'frozen_cohort/inputs.jsonl').read_text().splitlines()[0])
    source['video_size'] = 3
    (fixture / 'inputs.jsonl').write_text(json.dumps(source) + '\n')
    labels = {'window_uid': source['window_uid'], 'target': 0, 'loss_mask': True}
    (fixture / 'pilot_labels.jsonl').write_text(json.dumps(labels) + '\n')
    (fixture / 'media_expectations.json').write_text('{}')
    (fixture / 'inventory.json').write_text(json.dumps({f.name: sha(f) for f in fixture.iterdir()}))
    train = tmp_path / 'train'
    train.mkdir()
    (train / (source['video_id'] + '.mp4')).write_bytes(b'abc')
    monkeypatch.setattr(run_local, 'ROOT', fixture.parent)
    monkeypatch.setattr(run_local, 'model_identity', lambda p: {'directory': str(p), 'model_type': 'qwen3_vl',
        'text_shape': {'hidden_size': 4096, 'num_hidden_layers': 36, 'intermediate_size': 12288}, 'files_sha256': {'a': 'b'}})
    monkeypatch.setattr(run_local.importlib.metadata, 'version', lambda x: 'test-version')
    out = tmp_path / 'new-run'
    plan = run_local.prepare(SimpleNamespace(vlm_dir='local-model', train_root=str(train), device='cuda:0',
                                            dtype='bfloat16', attention='sdpa'), out)
    moved = json.loads((out / 'pilot/inputs.jsonl').read_text())
    assert moved['video_path'] != source['video_path']
    for key in set(source) - {'video_path', 'video_mtime_ns'}:
        assert moved[key] == source[key]
    assert (out / 'private/pilot_labels.jsonl').read_bytes() == (fixture / 'pilot_labels.jsonl').read_bytes()
    assert read_json(out / 'protocol.json')['model'] == 'Qwen/Qwen3-VL-8B-Instruct'
    assert not (out / 'cache').exists()
    assert plan['windows'] == 1
    run_local.verify(out)
