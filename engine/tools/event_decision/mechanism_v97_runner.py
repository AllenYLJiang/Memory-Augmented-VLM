"""One immutable native response per window, with globally bounded transport retries."""
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import file_sha256
from .role_scoped import atomic_json
from .development_mechanism_v93_runner import provider_error, failure_kind, RequestFailure
from .mechanism_v97_contract import assess, examples
from .mechanism_v97_store import current, receipts, seal, unseal


def parse_response(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key: '+key)
            out[key] = value
        return out
    def invalid(value): raise ValueError('Non-finite JSON number: '+value)
    if not isinstance(text, str): raise ValueError('Text response required')
    # A single wrapping code fence is syntax, not an excuse to extract a convenient nested object.
    clean = text.strip()
    if clean.startswith('```json\n') and clean.endswith('```'): clean = clean[8:-3].strip()
    elif clean.startswith('```\n') and clean.endswith('```'): clean = clean[4:-3].strip()
    return json.loads(clean, object_pairs_hook=pairs, parse_constant=invalid)


def provider_factory(protocol, key_env):
    key = os.environ.get(key_env, '').strip()
    if not key: raise ValueError('Missing '+key_env+'; credentials are never printed')
    import dashscope
    from dashscope import MultiModalConversation
    dashscope.base_http_api_url = protocol['endpoint']
    def request(images, prompt):
        content = []
        for i, image in enumerate(images):
            content += [{'text': f'T{i}'}, {'image': image.resolve().as_uri()}]
        content.append({'text': prompt})
        resp = MultiModalConversation.call(model=protocol['model'], messages=[{'role': 'user', 'content': content}],
            api_key=key, max_tokens=protocol['max_tokens'], temperature=protocol['temperature'],
            vl_high_resolution_images=True)
        envelope = {'status_code': getattr(resp, 'status_code', None), 'code': getattr(resp, 'code', None),
                    'message': getattr(resp, 'message', '')}
        # A successful SDK response may use code=None; the shared classifier expects an empty code.
        envelope['code'] = envelope['code'] or ''
        provider_error(envelope)
        try:
            choice = resp.output.choices[0]
            blocks = choice.message.content
            text = blocks if isinstance(blocks, str) else '\n'.join(b['text'] for b in blocks if isinstance(b, dict) and 'text' in b)
            usage = getattr(resp, 'usage', None)
            usage = dict(usage) if usage is not None else None
            finish = choice.get('finish_reason')
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            raise RequestFailure('request', 'Unexpected provider envelope; inspect SDK, do not reinterpret as model output') from exc
        return {'text': text, 'usage': usage, 'finish_reason': finish}
    return request


def acquire(out, protocol, rows, *, max_calls=36, workers=3, key_env='DASHSCOPE_API_KEY', provider=None):
    from .mechanism_v97_report import report
    if type(max_calls) is not int or max_calls < 1 or type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('max_calls>=1 and 1<=workers<=8 required')
    out = Path(out); digest = file_sha256(out/'protocol.json')
    allowed = {r['window_uid'] for r in rows}
    if any(p.name not in allowed for p in (out/'attempts').glob('*')): raise ValueError('Unexpected receipt window')
    all_receipts = {r['window_uid']: receipts(out, r['window_uid'], digest) for r in rows}
    pending = []
    for row in rows:
        saved = current(out, row)
        if saved['status'] in ('response', 'provider_rejected'): continue
        if saved.get('failure_kind') in ('request', 'runtime'):
            raise ValueError('Diagnose non-retryable '+saved['failure_kind']+' for '+row['window_uid'][:20])
        if len(all_receipts[row['window_uid']]) < protocol['attempts_per_window']: pending.append(row)
    spent = sum(map(len, all_receipts.values()))
    if not pending or spent >= protocol['total_attempts']: return report(out, protocol, rows)
    if provider is not None:
        if not protocol['mock']: raise ValueError('Injected provider requires MOCK protocol')
        request = provider
    elif protocol['mock']:
        request = lambda images, prompt: {'text': json.dumps(examples()[0]), 'usage': None, 'finish_reason': 'mock'}
    else:
        if not protocol['remote_execution_authorized']: raise ValueError('Acquisition not authorized by this protocol')
        request = provider_factory(protocol, key_env)
    mutex = threading.Lock(); output_lock = threading.Lock(); stop = threading.Event()
    used = 0; streak = 0
    invocation = uuid.uuid4().hex
    meta = {'id': invocation, 'protocol_sha256': digest, 'started_unix': time.time(), 'status': 'running',
            'max_calls_this_invocation': max_calls, 'workers': workers, 'mock': protocol['mock']}
    path = out/'invocations'/(invocation+'.json'); atomic_json(path, meta)
    key = os.environ.get(key_env, '')
    def redact(value): return str(value).replace(key, '[REDACTED]') if key else str(value)
    def one(row):
        nonlocal used, streak
        with mutex:
            if stop.is_set() or used >= max_calls or spent+used >= protocol['total_attempts']: return
            used += 1
        uid = row['window_uid']; attempt = len(all_receipts[uid])+1
        receipt_path = out/'attempts'/uid/f'{attempt:03d}.json'
        if receipt_path.exists(): raise ValueError('Attempt already exists')
        r = {'window_uid': uid, 'protocol_sha256': digest, 'invocation': invocation, 'attempt': attempt,
             'status': 'started', 'started_unix': time.time(), 'mock': protocol['mock'],
             'prior_indeterminate_attempt': bool(all_receipts[uid] and all_receipts[uid][-1]['status'] == 'started')}
        seal(receipt_path, r)
        try:
            reply = request([out/'cases'/uid[:20]/f'T{i}.jpg' for i in range(8)], protocol['prompt'])
            if not isinstance(reply, dict) or not isinstance(reply.get('text'), str):
                raise RequestFailure('runtime', 'Provider adapter must return text and optional usage')
            r.update(status='response', raw=redact(reply['text']), usage=reply.get('usage'), finish_reason=reply.get('finish_reason'))
            # Persist the actual response before parsing. A crash must not cause a new semantic sample.
            seal(receipt_path, r)
            finish_response(r)
        except Exception as exc:
            if r['status'] == 'response':
                seal(receipt_path, r)
                raise  # Fail closed on a validator bug, retaining the returned response.
            kind = failure_kind(exc)
            r.update(status='provider_rejected' if kind == 'provider_policy' else 'failed', failure_kind=kind, error=redact(exc))
            if kind in ('account', 'request', 'runtime'): stop.set()
        r['elapsed_seconds'] = time.time()-r['started_unix']; seal(receipt_path, r)
        with mutex:
            streak = streak+1 if r['status'] == 'failed' else 0
            if streak >= 3: stop.set()
        with output_lock:
            summary = report(out, protocol, rows, build_review=False)
            valid = r.get('assessment', {}).get('valid')
            print(f'[V97-SAVED] {summary["responded_windows"]}/{len(rows)} {uid[:20]} status={r["status"]} valid={valid} -> {out/"cases"/uid[:20]/"index.html"}', flush=True)
    try:
        report(out, protocol, rows, build_review=False)
        with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(one, pending))
        meta['status'] = 'finished'
    except BaseException:
        meta['status'] = 'interrupted'; raise
    finally:
        meta.update(finished_unix=time.time(), calls_reserved=used, stopped_after_errors=stop.is_set())
        atomic_json(path, meta)
    return report(out, protocol, rows)


def finish_response(receipt):
    try:
        parsed = parse_response(receipt['raw'])
        receipt['parsed'] = parsed
    except (ValueError, TypeError) as exc:
        receipt.update(parsed=None, parse_error=str(exc))
    receipt['assessment'] = assess(receipt.get('parsed'))


def recover_returned_responses(out, rows):
    """Offline recovery of the durable raw response, never a provider retry."""
    for row in rows:
        # Verify provenance and sequence before touching any crash-interrupted receipt.
        receipts(out, row['window_uid'], file_sha256(Path(out)/'protocol.json'))
        for path in (Path(out)/'attempts'/row['window_uid']).glob('*.json'):
            r = unseal(path)
            if r['status'] == 'response' and 'assessment' not in r:
                finish_response(r)
                r['offline_parse_recovered'] = True
                seal(path, r)
