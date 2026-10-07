"""Technical-first review and opt-in masked diagnostic sidecars, never scores."""
import copy
import json
from pathlib import Path

from .contracts import file_sha256, read_json, semantic_sha256, write_jsonl
from .role_scoped import atomic_json
from .mechanism_v94_report import page
from .mechanism_v97_store import load, current, seal, unseal, child
from .mechanism_v97_report import esc, frames

QUESTIONS = {
    'observations_supported': 'Are the stated visible observations and phases supported (including appropriately stated uncertainty)?',
    'role_partition_supported': 'Are relation participants, required instruments and ancillary objects correctly separated?',
    'identity_scope_supported': 'Are local roles supported without guessing identities across edits?',
    'context_scope_supported': 'Are judgments needing unseen context left unresolved, rather than asserted as visible?',
    'coverage_sufficient': 'Does the response preserve the important visible observations, rather than omit difficult facts?',
}


def revoke(out, reason):
    atomic_json(Path(out)/'integration/CURRENT.json', {'ready': False, 'reason': reason, 'training_authorized': False, 'scoring_authorized': False})


def export_review(out, protocol, rows, results, summary, selection):
    root = Path(out)/'review'; root.mkdir(exist_ok=True)
    digest = file_sha256(Path(out)/'protocol.json')
    hashes = {uid: semantic_sha256(r) for uid, r in results.items()}
    active = summary['technical_ready'] and not protocol['mock']
    packet = {'version': 'v97_native_review_1', 'active': active, 'protocol_sha256': digest,
              'result_hashes': hashes, 'selected_reasons': selection if active else {},
              'scope': 'eight silent frames, development evidence audit; not independent benchmark labels'}
    packet['packet_id'] = semantic_sha256(packet)
    old = read_json(root/'packet.json', {})
    if packet != old: revoke(out, 'new_or_inactive_review_packet')
    if old == packet and (not active or (root/'template.jsonl').exists()): return packet
    atomic_json(root/'packet.json', packet)
    if not active:
        (root/'index.html').write_text(page('V9.7 review held', '<h1>No human review requested</h1><p>'+esc(summary['decision'])+'</p>'), encoding='utf-8')
        return packet
    templates = []; links = []
    for row in rows:
        uid = row['window_uid']
        if uid not in selection: continue
        r = results[uid]
        template = {'window_uid': uid, 'packet_id': packet['packet_id'], 'protocol_sha256': digest,
                    'result_sha256': hashes[uid], 'reviewer_id': '', 'completed': False,
                    'evidence_scope': 'eight_frames', **{key: 'pending' for key in QUESTIONS},
                    'evidence_bins': [], 'notes': ''}
        templates.append(template)
        body = '<h1>Local evidence audit '+uid[:20]+'</h1>'+frames(row, '../../cases/'+uid[:20]+'/')
        body += '<h2>Native observations</h2><pre>'+esc(json.dumps(r.get('parsed'), ensure_ascii=False, indent=2))+'</pre>'
        body += '<h2>Review fields</h2><table>'+''.join('<tr><th>'+key+'</th><td>'+esc(q)+'</td></tr>' for key, q in QUESTIONS.items())+'</table>'
        body += '<p>Values: yes / no / unknown. not_applicable is allowed only for role_partition_supported and identity_scope_supported when no relation is claimed.</p>'
        body += '<p>Bins 0..7 refer to these eight samples. Unknown is not normal. No B5 label or score is requested.</p>'
        path = root/'cases'/(uid[:20]+'.html'); path.parent.mkdir(exist_ok=True)
        path.write_text(page('V9.7 local review', body), encoding='utf-8')
        links.append(f'<li><a href="cases/{uid[:20]}.html">{uid[:20]}</a></li>')
    write_jsonl(root/'template.jsonl', templates)
    atomic_json(root/'field_definitions.json', QUESTIONS)
    (root/'index.html').write_text(page('V9.7 evidence review', '<h1>Native local evidence review</h1><p>'+str(len(templates))+' selected windows; same eight-frame scope.</p><ul>'+''.join(links)+'</ul>'), encoding='utf-8')
    return packet


def import_review(out, protocol, rows, review_file):
    out = Path(out); revoke(out, 'review_import_in_progress')
    packet = read_json(out/'review/packet.json', {})
    if protocol['mock'] or not packet.get('active'): raise ValueError('Technical review is not active; no import permitted')
    if Path(review_file).resolve() == (out/'review/template.jsonl').resolve(): raise ValueError('Preserve the template; use a separate review_returns.jsonl')
    results = {r['window_uid']: current(out, r) for r in rows}
    hashes = {uid: semantic_sha256(r) for uid, r in results.items()}
    if hashes != packet['result_hashes'] or packet['protocol_sha256'] != file_sha256(out/'protocol.json'):
        raise ValueError('Review packet is stale')
    selected = set(packet['selected_reasons']); accepted = {}; errors = []
    source_bytes = Path(review_file).read_bytes()
    for line_no, line in enumerate(source_bytes.decode('utf-8-sig').splitlines(), 1):
        if not line.strip(): continue
        try:
            r = json.loads(line); uid = r['window_uid']
            if uid not in selected or uid in accepted: raise ValueError('Unknown, unselected or duplicate review window')
            if r.get('packet_id') != packet['packet_id'] or r.get('protocol_sha256') != packet['protocol_sha256'] or r.get('result_sha256') != hashes[uid]:
                raise ValueError('Stale packet/protocol/result hash')
            if r.get('completed') is not True or not isinstance(r.get('reviewer_id'), str) or not r['reviewer_id'].strip(): raise ValueError('Reviewer ID and completed=true required')
            if r.get('evidence_scope') != 'eight_frames': raise ValueError('Extra context must be evaluated separately')
            claims = any(c.get('relation_claim_present') for c in results[uid].get('assessment', {}).get('observations', []))
            for name in QUESTIONS:
                allowed = {'yes', 'no', 'unknown'}
                if name in ('role_partition_supported', 'identity_scope_supported') and not claims: allowed.add('not_applicable')
                if not isinstance(r.get(name), str) or r[name] not in allowed: raise ValueError('Invalid answer: '+name)
            bins = r.get('evidence_bins')
            if not isinstance(bins, list) or not bins or not all(type(b) is int and 0 <= b < 8 for b in bins) or bins != sorted(set(bins)):
                raise ValueError('Evidence bins must be nonempty unique sorted sample indices 0..7')
            if not isinstance(r.get('notes'), str) or not r['notes'].strip(): raise ValueError('Brief evidence notes required')
            accepted[uid] = r
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({'line': line_no, 'error': str(exc)})
    missing = sorted(selected-set(accepted))
    supported = {uid for uid, r in accepted.items() if all(r[k] in ('yes', 'not_applicable') for k in QUESTIONS)}
    fraction = len(supported)/len(selected) if selected else 0
    ready = not errors and not missing and fraction >= protocol['gates']['minimum_review_supported_fraction']
    review_digest = file_sha256(review_file)
    receipt_path = out/'review/imports'/(review_digest+'.json')
    receipt = {'input_sha256': review_digest, 'packet': packet, 'accepted': accepted, 'errors': errors,
               'missing': missing, 'supported_fraction': fraction, 'scope': 'development_human_audit_only'}
    seal(receipt_path, receipt)
    gate = {'technical_ready': True, 'review_completed': len(accepted), 'review_required': len(selected),
            'review_errors': errors, 'missing_windows': missing, 'supported_fraction': fraction,
            'ready_for_shadow_integration': ready, 'training_authorized': False, 'scoring_authorized': False,
            'formal_accuracy': None, 'formal_AP': None,
            'decision': 'REVIEWED_DIAGNOSTIC_TRACE_ONLY' if ready else 'SEMANTIC_REVIEW_HOLD',
            'review_receipt': str(receipt_path.relative_to(out)), 'result_hashes': hashes,
            'protocol_sha256': packet['protocol_sha256']}
    atomic_json(out/'integration/gate.json', gate)
    if not ready: return gate
    traces = []
    by_uid = {row['window_uid']: row for row in rows}
    for uid, result in results.items():
        row = by_uid[uid]; raw = result.get('parsed'); raw = raw if isinstance(raw, dict) else {}
        obs = raw.get('observations'); obs = obs if isinstance(obs, list) else []
        review = accepted.get(uid); reviewed = review is not None; fields = []
        for e, check in zip(obs, result.get('assessment', {}).get('observations', [])):
            if not isinstance(e, dict): continue
            observation_mask = bool(check['observation_valid'] and review and review['observations_supported'] == 'yes')
            relation_mask = bool(observation_mask and check['relation_valid'] and review and
                                 all(review[k] == 'yes' for k in ('role_partition_supported', 'identity_scope_supported', 'context_scope_supported')))
            context_mask = bool(review and review['context_scope_supported'] == 'yes' and check['context_valid'])
            fields.append({'id': e.get('id'), 'kind': e.get('kind'), 'phase': e.get('phase'), 'presence': e.get('presence'),
                           'bins': e.get('bins'), 'context_needed_for': e.get('context_needed_for'),
                           'observation_semantic_mask': observation_mask, 'relation_semantic_mask': relation_mask,
                           'context_scope_mask': context_mask,
                           'phase_observed_mask': bool(observation_mask and context_mask and e.get('phase') != 'unknown' and 'phase' not in e.get('context_needed_for', [])),
                           'relation_observed_mask': bool(relation_mask and isinstance(e.get('relation'), dict) and e['relation'].get('state') == 'observed'),
                           'structural_validity': check, 'relation': e.get('relation'), 'evidence': e.get('evidence')})
        traces.append({'window_uid': uid, 'source_group': row['source_group'], 'video_id': row['video_id'],
                       'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
                       'sampled_frame_indices': row['sampled_frame_indices'], 'result_sha256': hashes[uid],
                       'human_reviewed': reviewed, 'whole_window_supported': uid in supported,
                       'observations': fields, 'raw_observations': raw, 'category_label': None, 'anomaly_score': None,
                       'training_loss_mask': False, 'evaluation_loss_mask': False, 'cross_event_identity_verified': False,
                       'purpose': 'diagnostic_only'})
    write_jsonl(out/'integration/diagnostic_trace.jsonl', traces)
    files = ('integration/diagnostic_trace.jsonl', 'integration/gate.json', str(receipt_path.relative_to(out)), 'review/packet.json')
    atomic_json(out/'integration/CURRENT.json', {'ready': True, 'protocol_sha256': packet['protocol_sha256'],
                'bundle_hashes': {f: file_sha256(out/f) for f in files}, 'training_authorized': False, 'scoring_authorized': False})
    return gate


def load_diagnostic_trace(out):
    out = Path(out); protocol, rows = load(out)
    pointer = read_json(out/'integration/CURRENT.json', {})
    if protocol['mock'] or not pointer.get('ready') or pointer.get('protocol_sha256') != file_sha256(out/'protocol.json'):
        raise ValueError('No reviewed diagnostic trace is available')
    for rel, digest in pointer['bundle_hashes'].items():
        if file_sha256(child(out, rel)) != digest: raise ValueError('Reviewed bundle changed')
    gate = read_json(out/'integration/gate.json')
    if gate.get('ready_for_shadow_integration') is not True or gate.get('scoring_authorized') is not False or gate.get('training_authorized') is not False:
        raise ValueError('Diagnostic gate is not ready')
    hashes = {r['window_uid']: semantic_sha256(current(out, r)) for r in rows}
    if hashes != gate['result_hashes']: raise ValueError('Results changed since review')
    review = unseal(out/gate['review_receipt'])
    if review['errors'] or review['missing']: raise ValueError('Incomplete human review')
    return [json.loads(line) for line in (out/'integration/diagnostic_trace.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def attach_diagnostic_trace(predictions, out):
    """Non-mutating sidecar join. Existing graph scores and predictions are untouched."""
    traces = {r['window_uid']: r for r in load_diagnostic_trace(out)}
    joined = []
    for pred in predictions:
        uid = pred.get('window_uid')
        if uid not in traces: raise ValueError('Diagnostic join requires an exact frozen window_uid')
        if 'v97_diagnostic_trace' in pred: raise ValueError('Do not overwrite an existing trace')
        trace = traces[uid]
        for key in ('video_id', 'start_frame', 'end_frame_exclusive', 'sampled_frame_indices'):
            if key in pred and pred[key] != trace[key]: raise ValueError('Prediction/frame identity mismatch')
        value = copy.deepcopy(pred); value['v97_diagnostic_trace'] = copy.deepcopy(trace); joined.append(value)
    return joined
