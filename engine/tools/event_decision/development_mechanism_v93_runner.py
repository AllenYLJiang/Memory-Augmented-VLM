"""Account-aware retries and atomic per-window results for the V9.3 experiment."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import file_sha256, read_json
from .role_scoped import atomic_json, portable
from .development_mechanism_v93 import load_run, mock_response, validate, evidence_state


class RequestFailure(RuntimeError):
    def __init__(self, kind, detail):
        super().__init__(str(detail))
        self.kind = kind


def provider_error(value):
    if not isinstance(value, dict): return
    code = str(value.get('code', ''))
    status = value.get('status_code')
    try: status = int(status) if status is not None else 0
    except (ValueError, TypeError): status = 0
    if code == 'DataInspectionFailed':
        raise RequestFailure('provider_policy', code)
    if status < 400 and not code: return
    detail = f"HTTP {status} {code}: {value.get('message', '')}"
    if status in (401, 403) or any(x in code.lower() for x in ('balance', 'arrears', 'apikey', 'account', 'quotaexhausted')):
        raise RequestFailure('account', detail)
    if status in (408, 429) or status >= 500:
        raise RequestFailure('transient', detail)
    if status >= 400:
        raise RequestFailure('request', detail)


def failure_kind(exc):
    if isinstance(exc, RequestFailure): return exc.kind
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(s in str(exc).lower() for s in (
            'timed out', 'timeout', 'connection', 'rate limit', '429', '502', '503', '504')):
        return 'transient'
    # Unexpected programming/runtime errors stop the run instead of becoming model data.
    return 'runtime'


def read_result(out, row, protocol_sha):
    path = Path(out) / 'results' / (row['window_uid'] + '.json')
    try: result = read_json(path)
    except (ValueError, UnicodeError): return {'status': 'invalid_cache'}
    if result is None: return None
    if not isinstance(result, dict): return {'status': 'invalid_cache'}
    if result.get('protocol_sha256') not in (None, protocol_sha):
        raise ValueError('cached response belongs to another protocol')
    if result.get('window_uid') not in (None, row['window_uid']):
        raise ValueError('cached response belongs to another window')
    if result.get('status') == 'success':
        try:
            if result.get('protocol_sha256') != protocol_sha or result.get('window_uid') != row['window_uid']:
                raise ValueError('unbound successful cache')
            validate(result['parsed'])
        except (ValueError, TypeError, KeyError): return {'status': 'invalid_cache'}
    elif result.get('status') not in {'failed', 'provider_rejected'} or result.get('protocol_sha256') != protocol_sha:
        return {'status': 'invalid_cache'}
    return result


def provider_factory(protocol, key_env):
    if protocol['mock']:
        return lambda images, prompt: json.dumps(mock_response()), json.loads
    key = os.environ.get(key_env, '').strip()
    if not key: raise ValueError(f'missing {key_env}; no key values are logged')
    sys.path.insert(0, str(portable(protocol['code_dir']) / 'src'))
    from structural_vlm_binary.vlm.dashscope_backend import DashScopeVLM, DashScopeVLMConfig, parse_json_like
    backend = DashScopeVLM(DashScopeVLMConfig(model=protocol['model'], api_key_env=key_env, max_retries=0, json_parse_retries=1))
    def request(images, prompt):
        content = []
        for i, image in enumerate(images):
            content.extend([{'text': f'T{i}'}, {'image': image.resolve().as_uri()}])
        content.append({'text': prompt})
        return backend._call_messages([{'role': 'user', 'content': content}], api_key=key,
                                      max_tokens=protocol['max_tokens'], temperature=protocol['temperature'],
                                      vl_high_resolution_images=True)
    return request, parse_json_like


def run(out, *, max_calls=200, workers=3, attempts_per_window=2, max_consecutive_errors=3,
        retry_schema=False, key_env='DASHSCOPE_API_KEY', provider=None):
    from .development_mechanism_v93_report import report
    out = Path(out)
    if any(type(x) is not int or x < 1 for x in (max_calls, workers, attempts_per_window, max_consecutive_errors)) or workers > 8:
        raise ValueError('positive integer budgets required; workers must be 1..8')
    protocol, rows = load_run(out)
    digest = file_sha256(out / 'protocol.json')
    pending = []
    for row in rows:
        if row['exclusion']: continue
        old = read_result(out, row, digest)
        if old and old['status'] in {'success', 'provider_rejected'}: continue
        if old and old.get('failure_kind') in {'request', 'runtime'}:
            raise ValueError('non-retryable request/runtime failure needs diagnosis: ' + row['window_uid'])
        if old and old.get('failure_kind') == 'schema' and not retry_schema: continue
        pending.append(row)
    if not pending:
        return report(out, protocol, rows)
    request, parse = provider or provider_factory(protocol, key_env)
    report(out, protocol, rows)
    key = os.environ.get(key_env, '')
    def redacted(value):
        text = str(value)
        return text.replace(key, '[REDACTED]') if key else text
    invocation_id = str(time.time_ns()) + '_' + uuid.uuid4().hex[:8]
    invocation = {'id': invocation_id, 'protocol_sha256': digest, 'started_unix': time.time(),
                  'mock': protocol['mock'], 'workers': workers, 'max_calls': max_calls,
                  'attempts_per_window': attempts_per_window, 'status': 'running'}
    invocation_path = out / 'invocations' / (invocation_id + '.json')
    atomic_json(invocation_path, invocation)
    atomic_json(out / 'development_exposure.json', {
        'role': 'development', 'source_groups': protocol['development_sources'],
        'protocol_sha256': digest, 'scope': 'committed_development_not_proof_of_all_calls_completed'})
    lock = threading.Lock(); output_lock = threading.Lock(); stop = threading.Event()
    used = 0; streak = 0; stop_reason = None

    def one(row):
        nonlocal used, streak, stop_reason
        uid = row['window_uid']; path = out / 'results' / (uid + '.json')
        current = {'window_uid': uid, 'source_group': row['source_group'], 'role': 'development',
                   'protocol_sha256': digest, 'mock': protocol['mock'], 'model': protocol['model'],
                   'status': 'failed', 'invocation_id': invocation_id}
        attempted = False
        images = [out / 'cases' / uid[:20] / f'T{i}.jpg' for i in range(8)]
        for attempt in range(attempts_per_window):
            with lock:
                if stop.is_set() or used >= max_calls: break
                used += 1
            if not attempted and path.exists():
                archive = out / 'archived_results' / (uid + '_' + str(time.time_ns()) + '.json')
                archive.parent.mkdir(parents=True, exist_ok=True)
                path.replace(archive)
            attempted = True
            receipt = {'window_uid': uid, 'source_group': row['source_group'], 'role': 'development',
                       'protocol_sha256': digest, 'invocation_id': invocation_id,
                       'started_unix': time.time(), 'mock': protocol['mock'], 'status': 'started'}
            receipt_path = out / 'attempts' / uid / (str(time.time_ns()) + '_' + uuid.uuid4().hex[:8] + '.json')
            atomic_json(receipt_path, receipt)
            try:
                raw = request(images, protocol['prompt'])
                receipt['raw'] = redacted(raw)
                # Inspect envelopes before applying the model schema.
                try: envelope = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError): envelope = None
                provider_error(envelope)
                try:
                    parsed = parse(raw)
                    provider_error(parsed)
                    validate(parsed)
                except RequestFailure: raise
                except (ValueError, TypeError, KeyError) as exc:
                    raise RequestFailure('schema', str(exc)) from exc
                current.update(status='success', parsed=parsed, evidence_diagnostic=evidence_state(parsed))
                current.pop('failure_kind', None); current.pop('error', None)
                receipt['status'] = 'success'
            except Exception as exc:
                kind = failure_kind(exc)
                current.update(status='provider_rejected' if kind == 'provider_policy' else 'failed',
                               failure_kind=kind, error=redacted(exc))
                receipt.update(status='failed', failure_kind=kind, error=redacted(exc))
            receipt['elapsed_seconds'] = time.time() - receipt['started_unix']
            atomic_json(receipt_path, receipt)
            if current['status'] in {'success', 'provider_rejected'}: break
            if current['failure_kind'] in {'account', 'request', 'runtime'}:
                with lock:
                    stop_reason = current['failure_kind']; stop.set()
                break
            if attempt + 1 < attempts_per_window and not stop.is_set():
                time.sleep(min(2 ** attempt, 8))
        if not attempted: return
        current['finished_unix'] = time.time()
        atomic_json(path, current)
        with lock:
            if current['status'] in {'success', 'provider_rejected'}:
                streak = 0
            else:
                streak += 1
                if streak >= max_consecutive_errors:
                    stop_reason = stop_reason or 'consecutive_errors'; stop.set()
        with output_lock:
            if current['status'] == 'success':
                old = row['baseline'].get('parsed', {}).get('b5_presence')
                new = current['parsed']['b5_presence']
                tag = 'DEV-CHANGE' if old != new else 'DEV-RESULT'
                if protocol['mock']: tag = 'MOCK-' + tag
                print(f'[{tag}] {uid[:20]} old={old} new={new} phase={current["parsed"]["phase"]} '
                      f'evidence={current["evidence_diagnostic"]["state"]} -> {out / "cases" / uid[:20] / "index.html"}', flush=True)
            else:
                print(f'[DEV-UNAVAILABLE] {uid[:20]} {current.get("failure_kind")} -> {path}', flush=True)
            report(out, protocol, rows, changed_uid=uid)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, pending))
        invocation.update(status='finished', finished_unix=time.time(), calls_this_invocation=used,
                          stop_reason=stop_reason, budget_exhausted=used >= max_calls)
    except BaseException:
        invocation.update(status='interrupted', finished_unix=time.time(), calls_this_invocation=used)
        raise
    finally:
        atomic_json(invocation_path, invocation)
    return report(out, protocol, rows, changed_uid='')
