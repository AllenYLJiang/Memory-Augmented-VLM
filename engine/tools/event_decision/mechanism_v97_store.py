"""New-TAG protocol, fixed media, and hash-bound append-only evidence receipts."""
import os
import re
import shutil
import socket
from contextlib import contextmanager
from pathlib import Path

from .contracts import file_sha256, read_json, semantic_sha256
from .role_scoped import atomic_json, portable
from .mechanism_v96_offline import verify as verify_v96
from .mechanism_v97_contract import VERSION, GATES, assess, examples, prompt, specification

DEFAULT_SOURCE = 'governed_v96_mechanism_offline_20260914'
DEFAULT_TAG = 'governed_v97_native_mechanism_development_20260914'


def child(root, name):
    result = (Path(root)/name).resolve()
    if Path(root).resolve() not in result.parents: raise ValueError('Path escapes frozen run')
    return result


@contextmanager
def run_lock(project, out, recover=False):
    out = Path(out).resolve(); root = (Path(project)/'runs').resolve()
    if out.parent != root or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', out.name):
        raise ValueError('Output must be one TAG directly below project/runs')
    root.mkdir(exist_ok=True)
    path = root/('.'+out.name+'.v97.lock')
    if path.exists() and recover:
        saved = read_json(path)
        if saved.get('host') != socket.gethostname(): raise ValueError('Lock is from another host; inspect manually')
        pid = saved.get('pid')
        if type(pid) is not int or pid <= 0: raise ValueError('Invalid lock PID')
        try: os.kill(pid, 0)
        except ProcessLookupError: path.unlink()
        else: raise ValueError('Lock PID still exists; refusing takeover')
    try: fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError: raise ValueError('TAG locked. Inspect processes before RECOVER_STALE_LOCK=1')
    try:
        import json
        os.write(fd, json.dumps({'pid': os.getpid(), 'host': socket.gethostname()}).encode()); os.fsync(fd)
        yield
    finally:
        os.close(fd); path.unlink()


def seal(path, content):
    atomic_json(path, {'content': content, 'sha256': semantic_sha256(content)})


def unseal(path):
    value = read_json(path)
    if not isinstance(value, dict) or semantic_sha256(value.get('content')) != value.get('sha256'):
        raise ValueError('Corrupt evidence receipt: '+str(path))
    return value['content']


def prepare(project, source, out, *, mock=False, max_tokens=8192, total_attempts=72, per_window=2):
    project, source, out = (Path(x).resolve() for x in (project, source, out))
    if out.exists():
        p, rows = load(out)
        if portable(p['source_run']).resolve() != source or p['mock'] != mock or p['max_tokens'] != max_tokens or p['total_attempts'] != total_attempts or p['attempts_per_window'] != per_window:
            raise ValueError('Existing TAG has different settings; preserve it and use a new TAG')
        return {'status': 'VERIFIED_EXISTING_PROTOCOL', 'windows': len(rows), 'remote_calls': 0}
    if out.parent != (project/'runs').resolve() or source == out or source in out.parents or out in source.parents:
        raise ValueError('Independent direct runs child required')
    if not 2048 <= max_tokens <= 16384 or not 36 <= total_attempts <= 108 or not 1 <= per_window <= 3:
        raise ValueError('Frozen limits: tokens 2048..16384, total attempts 36..108, per window 1..3')
    prior, completion = verify_v96(source)
    if not completion: raise ValueError('V9.6 source must be complete')
    snapshot = source/'source_snapshot'
    rows = read_json(snapshot/'selection.json')
    plan = read_json(source/'next_request_plan.json')
    if len(rows) != 36 or [r['window_uid'] for r in rows] != [r['window_uid'] for r in plan['complete_control_cohort']]:
        raise ValueError('Must preserve all 36 windows in frozen order')
    if any(not re.fullmatch('[a-f0-9]{64}', r['window_uid']) for r in rows) or len({r['window_uid'][:20] for r in rows}) != 36:
        raise ValueError('Invalid or duplicate window identity')
    old = read_json(snapshot/'protocol.json')
    if old['mock']: raise ValueError('Input must be real completed observations')
    checks = [assess(x) for x in examples()]
    if any(not x['valid'] for x in checks): raise ValueError('Native examples fail validator')
    out.mkdir(parents=True)
    # This is a private local snapshot. Only neutral frame names and the common prompt go to the provider.
    atomic_json(out/'selection.json', rows)
    diagnostics = []
    import json
    for line in (source/'records.jsonl').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            record = json.loads(line)
            diagnostics.append({'window_uid': record['window_uid'], 'root_causes': record.get('root_causes', [])})
    atomic_json(out/'source_diagnostics.json', diagnostics)
    for row in rows:
        uid = row['window_uid']
        if row['exclusion'] or row['source_group'] in old['reserved_sources'] or row['source_group'] not in old['development_sources']:
            raise ValueError('Development source firewall failed')
        if len(row['sampled_frame_indices']) != 8 or len(set(row['sampled_frame_indices'])) != 8:
            raise ValueError('Eight distinct frozen frames required')
        result = read_json(snapshot/'results'/(uid+'.json'))
        if result.get('status') != 'success' or result.get('mock') is not False:
            raise ValueError('Only the complete non-policy-rejected source is eligible')
        atomic_json(out/'source_results'/(uid+'.json'), result)
        for i in range(8):
            name = f'T{i}.jpg'; src = snapshot/'cases'/uid[:20]/name
            if file_sha256(src) != row['media_hashes'][name]: raise ValueError('Source frame changed')
            dst = out/'cases'/uid[:20]/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    atomic_json(out/'contract.json', specification())
    atomic_json(out/'examples.json', examples())
    owned = list((project/'tools/event_decision').glob('mechanism_v97_*.py'))
    owned += [project/'tools/mechanism_v97_cli.py', project/'run_mechanism_v97.sh']
    owned += [project/'tests/test_mechanism_v97.py']
    inherited = [portable(x) for x in old['code_hashes']]
    inherited += [project/'tools/event_decision'/x for x in ('mechanism_v96_contract.py', 'mechanism_v96_offline.py')]
    inherited += [project/'tools/event_decision'/x for x in ('development_mechanism_v93_runner.py', 'mechanism_v94_report.py', 'contracts.py', 'role_scoped.py')]
    inputs = {str(f.relative_to(out)): file_sha256(f) for f in out.rglob('*') if f.is_file()}
    protocol = {'version': VERSION, 'role': 'design_exposed_development_only', 'mock': bool(mock),
                'source_run': str(source), 'source_tree_hashes': prior['source_tree_hashes'],
                'source_selection_sha256': prior['source_selection_sha256'],
                'source_integrity': {str(source/n): file_sha256(source/n) for n in ('protocol.json', 'completion.json', 'next_request_plan.json')},
                'code_hashes': {str(f.resolve()): file_sha256(f) for f in sorted(set(owned+inherited))},
                'data_hashes': inputs, 'model': old['model'], 'code_dir': old['code_dir'],
                'endpoint': 'https://dashscope.aliyuncs.com/api/v1', 'temperature': 0,
                'max_tokens': max_tokens, 'prompt': prompt(), 'gates': GATES,
                'total_attempts': total_attempts, 'attempts_per_window': per_window,
                'retention': 'first completed model response including invalid schema; no semantic or schema retries',
                'retry': 'only transient/account/indeterminate transport receipts; bounded across all invocations',
                'selection': 'all 36 same frames same order, no outcome-based removal',
                'remote_execution_authorized': not mock, 'scoring_authorized': False,
                'training_authorized': False, 'review_policy': 'technical gate then critical/changed cases plus eight deterministic controls',
                'integration_policy': 'human-reviewed, individually masked diagnostic trace only; no score override'}
    verify_v96(source)  # Detect source changes during the copy, before issuing a frozen acquisition protocol.
    atomic_json(out/'protocol.json', protocol)
    atomic_json(out/'integrity.json', {'protocol_sha256': file_sha256(out/'protocol.json')})
    load(out)
    return {'status': 'PREPARED_ZERO_API', 'windows': len(rows), 'remote_calls': 0,
            'mock': mock, 'first_response_requests': 36, 'frozen_total_attempt_limit': total_attempts}


def load(out):
    out = Path(out); p = read_json(out/'protocol.json', {})
    if p.get('version') != VERSION or p.get('role') != 'design_exposed_development_only' or p.get('training_authorized') is not False or p.get('scoring_authorized') is not False:
        raise ValueError('Not a V9.7 diagnostic-only protocol')
    if read_json(out/'integrity.json', {}).get('protocol_sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('Frozen protocol changed')
    for path, digest in dict(p['code_hashes'], **p['source_integrity']).items():
        if file_sha256(portable(path)) != digest: raise ValueError('Frozen dependency changed: '+path)
    for rel, digest in p['data_hashes'].items():
        if file_sha256(child(out, rel)) != digest: raise ValueError('Frozen input changed: '+rel)
    rows = read_json(out/'selection.json')
    if len(rows) != 36: raise ValueError('Incomplete cohort')
    return p, rows


def receipts(out, uid, protocol):
    result = []
    for path in sorted((Path(out)/'attempts'/uid).glob('*.json')):
        r = unseal(path)
        if r.get('window_uid') != uid or r.get('protocol_sha256') != protocol:
            raise ValueError('Receipt belongs to another window/protocol')
        if r.get('status') not in ('started', 'response', 'provider_rejected', 'failed'):
            raise ValueError('Unknown receipt status')
        if type(r.get('attempt')) is not int or path.name != f'{r["attempt"]:03d}.json' or r['attempt'] != len(result)+1:
            raise ValueError('Receipt sequence changed or has gaps')
        result.append(r)
    return result


def current(out, row):
    digest = file_sha256(Path(out)/'protocol.json')
    rr = receipts(out, row['window_uid'], digest)
    mock = read_json(Path(out)/'protocol.json')['mock']
    if any(r.get('mock') is not mock for r in rr): raise ValueError('Receipt MOCK identity differs from protocol')
    completed = [r for r in rr if r['status'] in ('response', 'provider_rejected')]
    if len(completed) > 1: raise ValueError('Multiple completed responses; never select a preferred answer')
    r = completed[0] if completed else (rr[-1] if rr else {'status': 'pending'})
    if r['status'] == 'response':
        if 'assessment' not in r: return dict(r, status='response_pending_parse')
        audit = assess(r.get('parsed'))
        if audit != r.get('assessment'): raise ValueError('Saved assessment changed')
    return r
