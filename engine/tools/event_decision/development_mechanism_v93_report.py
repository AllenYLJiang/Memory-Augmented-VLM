"""Local paired-prompt report. All comparisons are development diagnostics."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from .contracts import file_sha256, read_json
from .role_scoped import atomic_json
from .development_mechanism_v93 import load_run, evidence_state
from .development_mechanism_v93_runner import read_result

CSS = '''body{font:15px/1.5 Arial,sans-serif;color:#222;background:#fff;margin:0;letter-spacing:0}
main{max-width:1280px;margin:auto;padding:20px}h1{font-size:24px}h2{font-size:19px;margin-top:28px}
a{color:#075e9f}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#f1f3f4}td{max-width:400px}.scroll{overflow-x:auto}.notice{border-left:4px solid #bb3947;padding:8px 12px;background:#fff1f3}
.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#eee}
video{width:100%;max-height:420px;background:#111}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:12px}
select,input{font:inherit;padding:7px;max-width:100%;box-sizing:border-box}section{margin:20px 0}summary{cursor:pointer}small{color:#555}
@media(max-width:680px){main{padding:12px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:21px}}
'''


def esc(value):
    return html.escape(str(value))


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8', newline='\n')
    temp.replace(path)


def document(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows) + '</tbody></table></div>'


NOTICE = '<p class="notice">Development only. Same images, different prompts; not an OT/node ablation. No ground-truth accuracy or AP. Non-B5 does not mean normal. Human feedback was used for design and is not blind test data.</p>'


def case_page(row, result, state):
    old = row['baseline'].get('parsed', {})
    new = result.get('parsed', {}) if result else {}
    diag = evidence_state(new) if new else {}
    body = '<a href="../../index.html">All windows</a><h1>'+esc(row['video_id'])+'</h1>'+NOTICE
    body += f'<p>Original frames [{row["start_frame"]}, {row["end_frame_exclusive"]}) | {esc(state)}</p><video controls preload="none" src="clip.mp4"></video>'
    body += '<h2>Paired observations</h2>' + table(['Method', 'B5 decision', 'Self-reported probability', 'Scope'], [
        ['V9.2 cached', old.get('b5_presence', 'unavailable'), old.get('b5_probability', ''), 'Old compound actor/event prompt'],
        ['V9.3 raw model', new.get('b5_presence', 'unavailable'), new.get('b5_probability', ''), 'Mechanism, phase, roles, subevents'],
        ['V9.3 evidence diagnostic', diag.get('state', ''), 'No replacement score', ', '.join(diag.get('reasons', []))]])
    feedback = row.get('design_feedback')
    if feedback:
        body += '<h2>Previously supplied development feedback</h2><p>'+esc(feedback['human_note_verbatim'])+'</p><p><small>Assistant-normalized interpretation: '+esc(feedback['normalized_interpretation']['assessment'])+'; no automatic training or gold label export.</small></p>'
    body += '<h2>Exactly the supplied images</h2><div class="frames">'+''.join(
        f'<figure><a href="T{i}.jpg"><img loading="lazy" src="T{i}.jpg" alt="Sample T{i}"></a><figcaption>T{i}: original frame {frame}</figcaption></figure>'
        for i, frame in enumerate(row['sampled_frame_indices']))+'</div>'
    if new:
        body += '<h2>Phase and observability</h2>' + table(['Phase', 'Quality', 'Needs context', 'Other event'], [[
            new['phase'], new['observation_quality'], new['context_needed'], new['other_event']['category']+': '+new['other_event']['evidence']]])
        body += '<h2>Participants</h2>'+table(['ID', 'Visual identity / role', 'Observed bins'], [[a['id'], a['description'], a['bins']] for a in new['actors']])
        body += '<h2>Within-subevent binding</h2>'+table(['Event', 'Actor -> target', 'Bins / phase', 'Binding / mechanism', 'Scope / B5 support', 'Evidence'], [[
            e['id'], ', '.join(e['actor_ids'])+' -> '+', '.join(e['target_ids']), str(e['bins'])+' / '+e['phase'],
            e['binding']+' / '+e['mechanism'], e['evidence_scope']+' / '+str(e['b5_support']), e['evidence']] for e in new['subevents']])
        body += '<h2>Between-subevent relations</h2>'+table(['Link', 'Relation', 'State / scope', 'Bins', 'Evidence'], [[
            e['source']+' -> '+e['target'], e['kind'], e['relation_state']+' / '+e['evidence_scope'], e['bins'], e['evidence']] for e in new['event_links']])
        body += '<h2>Per-frame evidence</h2>'+table(['Bin', 'B5 support', 'Injury trace', 'Evidence'], [[
            f['bin'], f['b5_support'], f['injury_trace'], f['evidence']] for f in new['per_frame']])
        body += '<h2>Uncertainties</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in new['uncertainties'])+'</ul>'
    body += '<details><summary>Saved model records</summary><pre>'+esc(json.dumps({'baseline':row['baseline'], 'new_result':result},ensure_ascii=False,indent=2))+'</pre></details>'
    return document(row['video_id'], body)


def report(out, protocol=None, rows=None, changed_uid=None):
    out = Path(out)
    if protocol is None or rows is None: protocol, rows = load_run(out)
    digest = file_sha256(out / 'protocol.json')
    counts = Counter(); changes = Counter(); phases = Counter(); diagnostics = Counter()
    comparison = []; design_rows = []; table_rows = []
    for row in rows:
        uid = row['window_uid']; result = read_result(out, row, digest)
        state = 'excluded_prior_provider_policy' if row['exclusion'] else result['status'] if result else 'pending'
        counts[state] += 1
        new = result['parsed'] if result and result.get('status') == 'success' else None
        old = row['baseline'].get('parsed', {})
        diag = evidence_state(new) if new else None
        item = {'window_uid': uid, 'video_id': row['video_id'], 'source_group': row['source_group'],
                'start_frame': row['start_frame'], 'end_frame_exclusive': row['end_frame_exclusive'],
                'sampled_frame_indices': row['sampled_frame_indices'], 'status': state,
                'old_b5': old.get('b5_presence'), 'old_probability': old.get('b5_probability'),
                'new_b5': new['b5_presence'] if new else None,
                'new_probability': new['b5_probability'] if new else None,
                'phase': new['phase'] if new else None, 'evidence_diagnostic': diag,
                'context_needed': new['context_needed'] if new else None,
                'design_feedback_case': bool(row['design_feedback']), 'gold_metric_mask': False,
                'human_interval_target': None, 'formal_accuracy': None, 'formal_AP': None}
        comparison.append(item)
        changed = bool(new and old.get('b5_presence') != new['b5_presence'])
        if new:
            changes[str(old.get('b5_presence'))+' -> '+new['b5_presence']] += 1
            phases[new['phase']] += 1; diagnostics[diag['state']] += 1
        feedback = row.get('design_feedback')
        if feedback:
            target = feedback['normalized_interpretation']['proposed_development_b5_target']
            design_rows.append({**item, 'human_note': feedback['human_note_verbatim'],
                               'assistant_proposed_target_not_gold': target,
                               'old_agrees_with_proposed_target': old.get('b5_presence') == ('yes' if target == 1 else 'no') if target is not None else None,
                               'new_agrees_with_proposed_target': new['b5_presence'] == ('yes' if target == 1 else 'no') if target is not None and new else None})
        page = out / 'cases' / uid[:20] / 'index.html'
        if changed_uid is None or changed_uid == uid or not page.exists():
            atomic_text(page, case_page(row, result, state))
        attrs = ' '.join(['review' if feedback else '', 'change' if changed else '', 'unresolved' if diag and diag['state']=='unresolved' else ''])
        columns = [f'<a href="cases/{uid[:20]}/index.html">{esc(row["video_id"])}</a>',
                   esc(f'{row["start_frame"]}-{row["end_frame_exclusive"]-1}'), esc(state),
                   esc(str(old.get('b5_presence',''))+' / '+str(old.get('b5_probability',''))),
                   esc((new['b5_presence']+' / '+str(new['b5_probability'])) if new else ''),
                   esc(new['phase'] if new else ''), esc(diag['state'] if diag else ''), 'reviewed for design' if feedback else '']
        table_rows.append('<tr data-tags="'+attrs+'">'+''.join('<td>'+x+'</td>' for x in columns)+'</tr>')
    attempts = [read_json(p) for p in (out / 'attempts').glob('*/*.json')]
    elapsed = [a['elapsed_seconds'] for a in attempts if 'elapsed_seconds' in a]
    failure_counts = Counter(a.get('failure_kind', a['status']) for a in attempts if a['status'] != 'success')
    unfinished = sum(counts[k] for k in ('pending', 'invalid_cache', 'failed'))
    complete = 'DEVELOPMENT_COMPLETE_WITH_EXCLUSIONS' if counts['excluded_prior_provider_policy'] or counts['provider_rejected'] else 'DEVELOPMENT_COMPLETE'
    summary = {'version': protocol['version'], 'mock': protocol['mock'],
               'status': 'DEVELOPMENT_INCOMPLETE' if unfinished else complete,
               'planned_windows': len(rows), 'counts': dict(counts), 'transitions': dict(changes),
               'phases': dict(phases), 'evidence_diagnostics': dict(diagnostics),
               'attempt_receipts': len(attempts), 'remote_attempts_upper_bound': 0 if protocol['mock'] else len(attempts),
               'attempt_failures': dict(failure_counts), 'sdk_seconds_mean': sum(elapsed)/len(elapsed) if elapsed else None,
               'sdk_seconds_sum': sum(elapsed), 'design_cases': len(design_rows),
               'paired_design_cases_with_proposed_binary_interpretation': sum(x['assistant_proposed_target_not_gold'] is not None and x['new_b5'] is not None for x in design_rows),
               'old_matches_proposed_design_interpretation_on_paired_cases': sum(x['old_agrees_with_proposed_target'] is True and x['new_b5'] is not None for x in design_rows),
               'new_matches_proposed_design_interpretation_on_paired_cases': sum(x['new_agrees_with_proposed_target'] is True for x in design_rows),
               'formal_accuracy': None, 'formal_AP': None, 'locked_authorized': False,
               'score_rule': 'raw VLM probability retained; evidence state is separate ternary diagnostic, not fitted',
               'study_scope': 'design-exposed development only; same images but different prompts; not controlled OT/node evidence'}
    atomic_json(out / 'summary.json', summary)
    for name, content in (('comparison.jsonl', comparison), ('design_case_comparison.jsonl', design_rows)):
        atomic_text(out / name, ''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in content))
    head = ['Video / case', 'Frames', 'Status', 'Old B5 / p', 'New B5 / p', 'Phase', 'Evidence diagnostic', 'Review scope']
    body = '<h1>B5 mechanism development comparison</h1>'+NOTICE+'<pre>'+esc(json.dumps(summary,indent=2))+'</pre>'
    body += '<label>Window filter <select id="scope"><option value="">All windows</option><option value="review">Design feedback</option><option value="change">Changed B5 judgment</option><option value="unresolved">Unresolved evidence</option></select></label> <label>Video <input id="query" type="search"></label>'
    body += '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody id="windows">'+''.join(table_rows)+'</tbody></table></div>'
    body += '''<script>function filter(){const scope=document.getElementById('scope').value,q=document.getElementById('query').value.toLowerCase();document.querySelectorAll('#windows tr').forEach(r=>r.hidden=!(r.dataset.tags.includes(scope)&&r.textContent.toLowerCase().includes(q)));}document.getElementById('scope').addEventListener('change',filter);document.getElementById('query').addEventListener('input',filter);</script>'''
    atomic_text(out / 'index.html', document('B5 mechanism development comparison', body))
    return summary
