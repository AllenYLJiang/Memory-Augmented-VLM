"""All-window technical report; incomplete or unstable evidence cannot request review."""
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl, semantic_sha256
from .role_scoped import atomic_json
from .mechanism_v94_report import page
from .mechanism_v97_contract import CRITICAL, VERSION
from .mechanism_v97_store import current, receipts


def esc(value): return html.escape(str(value))


def review_selection(out, rows, results):
    prior = {r['window_uid']: r for r in read_json(Path(out)/'source_diagnostics.json')}
    reasons = defaultdict(list)
    for row in rows:
        uid = row['window_uid']; r = results[uid]; raw = r.get('parsed') or {}
        raw = raw if isinstance(raw, dict) else {}
        if 'design_case' in row['selection_reasons']: reasons[uid].append('fixed_design_case')
        root = {x['code'] for x in prior[uid]['root_causes']}
        if root-{'SYMBOL_ALIAS', 'LEGACY_OBJECT_ROLES_UNPARTITIONED'}: reasons[uid].append('prior_evidence_obligation')
        if not r.get('assessment', {}).get('valid'): reasons[uid].append('native_technical_issue')
        obs = raw.get('observations') if isinstance(raw.get('observations'), list) else []
        if any(isinstance(o, dict) and (o.get('context_needed_for') or o.get('presence') != 'observed') for o in obs):
            reasons[uid].append('native_uncertainty_or_context')
        old = read_json(Path(out)/'source_results'/(uid+'.json'))['parsed']
        old_n = len(old.get('events', []))
        if len(obs) != old_n: reasons[uid].append('observation_count_changed_not_semantic_improvement')
    # Controls come from the remaining cohort first, not from already flagged examples.
    controls = sorted(rows, key=lambda row: hashlib.sha256(('v97-control:'+row['window_uid']).encode()).hexdigest())
    nonflagged = [r for r in controls if not reasons[r['window_uid']]]
    chosen = (nonflagged+[r for r in controls if reasons[r['window_uid']]])[:8]
    for row in chosen: reasons[row['window_uid']].append('deterministic_control')
    return {k: v for k, v in reasons.items() if v}


def frames(row, prefix=''):
    return '<div class="frames">'+''.join(
        f'<figure><a href="{prefix}T{i}.jpg"><img loading="lazy" src="{prefix}T{i}.jpg" alt="T{i}"></a><figcaption>T{i} / frame {n}</figcaption></figure>'
        for i, n in enumerate(row['sampled_frame_indices']))+'</div>'


def render_case(out, row, result):
    uid = row['window_uid']; a = result.get('assessment', {})
    body = '<a href="../../index.html">All windows</a><h1>'+esc(row['video_id'])+'</h1>'
    body += f'<p>Frames [{row["start_frame"]}, {row["end_frame_exclusive"]}); {esc(result["status"])}. Development observations only.</p>'
    body += frames(row)
    raw = result.get('parsed')
    if isinstance(raw, dict) and isinstance(raw.get('observations'), list):
        body += '<table><tr><th>Observation</th><th>Kind / phase</th><th>Visible evidence</th><th>Relation</th><th>Context needed</th></tr>'
        for e in raw['observations']:
            if not isinstance(e, dict): continue
            body += '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in (e.get('id'), f'{e.get("kind")} / {e.get("phase")}',
                e.get('evidence'), json.dumps(e.get('relation'), ensure_ascii=False), e.get('context_needed_for')))+'</tr>'
        body += '</table>'
    body += '<h2>Field diagnostics</h2><pre>'+esc(json.dumps(a, ensure_ascii=False, indent=2))+'</pre>'
    body += '<details><summary>Immutable response</summary><pre>'+esc(result.get('raw', 'No completed response'))+'</pre></details>'
    (Path(out)/'cases'/uid[:20]/'index.html').write_text(page('V9.7 local observations', body), encoding='utf-8')


def report(out, protocol, rows, build_review=True):
    out = Path(out); digest = file_sha256(out/'protocol.json')
    results = {r['window_uid']: current(out, r) for r in rows}
    issues = Counter(); kind = Counter(); phase = Counter(); presence = Counter(); strength = Counter()
    group = defaultdict(Counter); records = []; comparisons = []; observations = []; relations = []
    valid_docs = observed_windows = 0
    for row in rows:
        uid = row['window_uid']; r = results[uid]; a = r.get('assessment', {}); raw = r.get('parsed')
        raw = raw if isinstance(raw, dict) else {}
        valid_docs += bool(a.get('valid')); issues.update(x['code'] for x in a.get('issues', []))
        obs = raw.get('observations') if isinstance(raw.get('observations'), list) else []
        ac = a.get('observations', [])
        observed = any(c.get('observation_valid') and isinstance(e, dict) and e.get('presence') == 'observed' for e, c in zip(obs, ac))
        observed_windows += observed
        for e, c in zip(obs, ac):
            observations.append(c)
            if c.get('relation_claim_present'): relations.append(c)
            if not isinstance(e, dict): continue
            for name, counter in (('kind', kind), ('phase', phase), ('presence', presence)):
                counter[str(e.get(name))] += 1
            rel = e.get('relation')
            strength[str(rel.get('strength')) if isinstance(rel, dict) else 'null'] += 1
        for label in row['selection_reasons']:
            group[label].update(windows=1, responses=int(r['status']=='response'), valid_documents=int(bool(a.get('valid'))),
                                observed_windows=int(observed), observations=len(obs))
        old = read_json(out/'source_results'/(uid+'.json'))
        old_raw = old['parsed']
        comparisons.append({'window_uid': uid, 'source_group': row['source_group'], 'selection_reasons': row['selection_reasons'],
            'old_b5_not_gold': old_raw.get('b5'), 'old_events': old_raw.get('events'), 'native_observations': obs,
            'old_event_count': len(old_raw.get('events', [])), 'native_observation_count': len(obs),
            'no_automatic_semantic_alignment': True, 'new_b5': None, 'new_anomaly_score': None})
        records.append({'window_uid': uid, 'status': r['status'], 'raw_observations': r.get('parsed'), 'assessment': a,
                        'result_sha256': semantic_sha256(r), 'training_loss_mask': False, 'evaluation_loss_mask': False})
        render_case(out, row, r)
    statuses = Counter(r['status'] for r in results.values()); n = len(rows)
    fraction = lambda num, den: num/den if den else None
    metrics = {'document_valid_fraction': valid_docs/n,
               'observation_valid_fraction': fraction(sum(c.get('observation_valid', False) for c in observations), len(observations)),
               'relation_valid_fraction': fraction(sum(c.get('relation_valid', False) for c in relations), len(relations)),
               'windows_with_observed_observation_fraction': observed_windows/n}
    failed = []
    if statuses['response'] != n: failed.append('complete_all_36_responses')
    for name in metrics:
        if metrics[name] is None or metrics[name] < protocol['gates']['minimum_'+name]: failed.append(name)
    critical = {k: v for k, v in issues.items() if k in CRITICAL}
    if critical: failed.append('critical_reference_or_evidence_contradictions')
    if len(relations) == 0: failed.append('no_relation_information_to_audit')
    ready = not failed
    attempts = [a for row in rows for a in receipts(out, row['window_uid'], digest)]
    latencies = [a['elapsed_seconds'] for a in attempts if 'elapsed_seconds' in a]
    summary = {'version': VERSION, 'mock': protocol['mock'], 'windows': n, 'responded_windows': statuses['response'],
        'videos': len({r['video_id'] for r in rows}), 'source_groups': len({r['source_group'] for r in rows}),
        'status_counts': dict(statuses), 'metrics': metrics, 'observations': len(observations), 'relations': len(relations),
        'issue_counts': dict(issues), 'critical_issues': critical, 'failed_checks': failed,
        'technical_ready': ready, 'human_review_requested': ready and not protocol['mock'],
        'decision': 'MOCK_DIAGNOSTIC_ONLY' if protocol['mock'] else ('REVIEW_REQUIRED' if ready else 'TECHNICAL_HOLD_NO_REVIEW'),
        'kind_counts': dict(kind), 'phase_counts': dict(phase), 'presence_counts': dict(presence), 'strength_counts': dict(strength),
        'selection_groups_overlapping': {k: dict(v) for k, v in group.items()}, 'attempts_reserved': len(attempts),
        'indeterminate_attempts': sum(a['status']=='started' for a in attempts),
        'remote_calls_upper_bound': 0 if protocol['mock'] else len(attempts),
        'average_recorded_attempt_seconds': sum(latencies)/len(latencies) if latencies else None,
        'usage_by_attempt': [dict(window_uid=a['window_uid'], attempt=a['attempt'], usage=a.get('usage')) for a in attempts],
        'tokens_when_unreported': None, 'formal_accuracy': None, 'formal_AP': None,
        'training_authorized': False, 'scoring_authorized': False, 'ready_for_shadow_integration': False,
        'claim_limit': 'native contract reliability, not visual truth or anomaly accuracy'}
    write_jsonl(out/'records.jsonl', records); write_jsonl(out/'paired_observations.jsonl', comparisons)
    atomic_json(out/'summary.json', summary)
    links = ''.join(f'<tr><td><a href="cases/{row["window_uid"][:20]}/index.html">{esc(row["video_id"])}</a></td><td>{esc(results[row["window_uid"]]["status"])}</td><td>{results[row["window_uid"]].get("assessment",{}).get("valid")}</td></tr>' for row in rows)
    body = '<h1>V9.7 native local observations</h1><p>'+esc(summary['decision'])+'</p>'
    body += '<table><tr><th>Video / evidence</th><th>Response</th><th>Contract valid</th></tr>'+links+'</table>'
    body += '<h2>Complete-cohort statistics</h2><pre>'+esc(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'
    (out/'index.html').write_text(page('V9.7 complete cohort', body), encoding='utf-8')
    if build_review:
        from .mechanism_v97_review import export_review
        export_review(out, protocol, rows, results, summary, review_selection(out, rows, results))
    return summary
