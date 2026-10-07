"""Local, read-only evidence inspection. No review form or score recomputation."""
import html
import json


def esc(value):
    return html.escape(str(value), quote=True)


CSS = '''
*{box-sizing:border-box;letter-spacing:0}body{margin:0;color:#222;background:#fafbfc;font:15px/1.5 system-ui,sans-serif}
header,main,footer{max-width:1320px;margin:auto;padding:20px 28px}header{border-bottom:1px solid #cdd4da}
h1{font-size:24px;margin:8px 0;overflow-wrap:anywhere}h2{font-size:18px;margin:20px 0 12px}h3{font-size:15px;margin:4px 0}
p{margin:6px 0 12px}a{color:#066c78}small,.muted{color:#606970}code{overflow-wrap:anywhere}
.stats{display:flex;gap:24px;flex-wrap:wrap;border-bottom:1px solid #cdd4da;padding:12px 0;margin:0 0 20px}
.stats strong{font-size:22px;display:block}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
input,select{font:inherit;padding:8px;border:1px solid #adb7bf;border-radius:4px;min-width:0;max-width:100%}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #dbe1e5;vertical-align:top}
th{font-size:13px;color:#46515a}td.video{overflow-wrap:anywhere;min-width:200px}tr[hidden]{display:none}
.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0;min-width:0}
figure img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#121212;display:block}figcaption{font-size:12px;padding:3px 0;color:#46515a}
details{border-top:1px solid #cdd4da;padding:12px 0}summary{cursor:pointer;overflow-wrap:anywhere;font-weight:600}
.columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;padding:14px 0}.columns>section{min-width:0}
.columns pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace;margin:10px 0}
.status{color:#0c7461}.unknown{color:#8e4c00}.human{color:#813447}.note{border-left:3px solid #813447;padding-left:12px;margin:12px 0}
.support{font-size:13px;overflow-wrap:anywhere}.support span{display:inline-block;margin:3px 14px 3px 0}
.banner{font-size:13px;color:#46515a}.wordwrap{overflow-wrap:anywhere;white-space:pre-wrap}.provenance{font-size:12px}
footer{font-size:12px;border-top:1px solid #cdd4da;color:#606970}
@media(max-width:700px){header,main,footer{padding:16px}h1{font-size:20px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}.columns{grid-template-columns:1fr;gap:12px}.stats{gap:16px}.controls>*{flex:1 1 100%}}
'''


def page(title, body):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style></head><body>'+body+'</body></html>'


def baseline_text(baseline):
    parsed = baseline.get('parsed') or {}
    return esc(parsed.get('b5_presence', 'unavailable'))+' / '+esc(parsed.get('b5_probability', 'unavailable'))


def case_page(trace, baseline):
    uid = trace['window_uid']
    body = '<header><a href="../../index.html">All 36 windows</a><h1>'+esc(trace['video_id'])+'</h1>'
    body += '<p class="banner">Development only | Historical B5 presence / probability: '+baseline_text(baseline)+' | Not ground truth</p>'
    body += '<p class="muted">Frames ['+str(trace['start_frame'])+', '+str(trace['end_frame_exclusive'])+') | R1 questions: '+str(trace['human_reviewed_questions'])+'</p></header><main>'
    body += '<div class="frames">'+''.join('<figure><img src="T'+str(i)+'.jpg" alt="Sample T'+str(i)+' frame '+str(frame)+'"><figcaption>T'+str(i)+' · frame '+str(frame)+'</figcaption></figure>' for i, frame in enumerate(trace['sampled_frame_indices']))+'</div>'
    body += '<h2>Observation evidence</h2>'
    for obs in trace['observations']:
        reviewed = bool(obs['human_assertions'])
        body += '<details'+(' open' if reviewed else '')+'><summary>'+esc(obs['id'])+' · '+esc(obs['review_view']['kind'])+' · '+esc(obs['review_view']['claim_mode'])+(' <span class="human">R1 field review</span>' if reviewed else ' <span class="muted">Semantics unreviewed</span>')+'</summary>'
        body += '<div class="columns">'
        for key, title in (('native_model', 'Original model'), ('v99_compatible', 'V99 compatible'), ('review_view', 'R1-scoped view')):
            view = obs[key]
            check = view['structural_check']
            body += '<section><h3>'+title+'</h3><p>Presence: <strong>'+esc(view['presence'])+'</strong> · Phase: '+esc(view['phase'])+'</p>'
            body += '<p>Observation format: '+esc(check['observation_valid'])+' · Relation contract: '+esc(check['relation_valid'])+'</p>'
            body += '<p>'+esc(view['evidence'])+'</p><pre>'+esc(json.dumps(view['relation'], ensure_ascii=False, indent=2))+'</pre>'
            body += '<p class="muted">Context objects: '+esc(view['context_entity_ids'])+'<br>Unresolved context: '+esc(view['context_needed_for'])+'</p></section>'
        body += '</div><div class="support">'
        for field, item in obs['semantic_support'].items():
            value = 'unreviewed' if item['value'] is None else ('supported' if item['value'] else 'not supported')
            body += '<span class="'+('muted' if item['value'] is None else 'human')+'">'+esc(field)+': <strong>'+value+'</strong></span>'
        body += '</div>'
        for assertion in obs['human_assertions']:
            body += '<div class="note"><strong>'+esc(assertion['assertion_id'])+' · '+esc(assertion['choice'])+'</strong><p>'+esc(assertion['notes_verbatim'])+'</p><small>R1 evidence bins: '+esc(assertion['review_bins'])+'; unverified: '+esc(', '.join(assertion['unverified']))+'</small></div>'
        if obs['field_changes']:
            body += '<h3>Versioned field changes</h3><pre class="wordwrap">'+esc('\n'.join(c['path']+': '+json.dumps(c['before'])+' -> '+json.dumps(c['after']) for c in obs['field_changes']))+'</pre>'
        if obs['scoped_entity_support']:
            body += '<p class="human">Local human support for this observation only; shared entity visibility unchanged.</p>'
        body += '</details>'
    body += '<details><summary>Shared entity table and source provenance</summary><pre class="wordwrap provenance">'+esc(json.dumps({'entities': trace['shared_entities'], 'provenance': trace['source_provenance'], 'record_hash': trace['source_overlay_record_sha256']}, indent=2))+'</pre></details>'
    body += '</main><footer>Contract validity is not visual truth. Unknown direct evidence is not a normal label. No scores or training targets are changed.</footer>'
    return page('Development evidence '+uid[:12], body)


def index_page(traces, baselines, stats, join_audit):
    baseline_by_uid = {p['window_uid']: p for p in baselines}
    body = '<header><h1>Local Mechanism Evidence</h1><p class="banner">V9.11 · Fixed development cohort · Read-only diagnostic sidecars</p></header><main>'
    body += '<div class="stats">'+''.join('<div><strong>'+str(value)+'</strong>'+label+'</div>' for value, label in (
        (len(traces), 'windows'), (stats['observations'], 'observations'), (sum(t['human_reviewed_questions'] for t in traces), 'R1 questions'),
        (join_audit['changed_existing_fields'], 'changed baseline fields')))+'</div>'
    body += '<p class="banner">R1 supports specific fields, not complete window semantics. Historical B5 probabilities are not GT or graph/OT scores.</p>'
    body += '<div class="controls"><input id="search" type="search" aria-label="Video or source group" placeholder="Video or source group"><select id="family" aria-label="Evidence family"><option value="">All evidence families</option>'
    for flag in stats['diagnostic_family_coverage']:
        body += '<option value="'+esc(flag)+'">'+esc(flag.replace('_', ' '))+'</option>'
    body += '</select><span id="visible-count" aria-live="polite">36 / 36 windows</span></div>'
    body += '<div class="table-wrap"><table><thead><tr><th>Video / source</th><th>Frames</th><th>Historical B5</th><th>Observations</th><th>R1 questions</th><th>Evidence families</th></tr></thead><tbody>'
    for trace in sorted(traces, key=lambda t: not bool(t['human_reviewed_questions'])):
        flags = sorted({f for o in trace['observations'] for f in o['diagnostic_flags']})
        body += '<tr data-search="'+esc((trace['video_id']+' '+trace['source_group']).lower())+'" data-flags="'+esc(' '.join(flags))+'"><td class="video"><a href="cases/'+trace['window_uid'][:20]+'/index.html">'+esc(trace['video_id'])+'</a><br><small>'+esc(trace['source_group'])+'</small></td>'
        body += '<td>'+str(trace['start_frame'])+'–'+str(trace['end_frame_exclusive']-1)+'</td><td>'+baseline_text(baseline_by_uid[trace['window_uid']])+'</td><td>'+str(len(trace['observations']))+'</td><td>'+str(trace['human_reviewed_questions'])+'</td><td>'+esc(', '.join(flags))+'</td></tr>'
    body += '</tbody></table></div><h2>Whole-cohort evidence inventory</h2><div class="table-wrap"><table><tr><th>Evidence family</th><th>Observations</th><th>Windows</th><th>Source groups</th></tr>'
    for flag, counts in stats['diagnostic_family_coverage'].items():
        body += '<tr><td>'+esc(flag)+'</td>'+''.join('<td>'+str(counts[k])+'</td>' for k in ('observations', 'windows', 'source_groups'))+'</tr>'
    body += '</table></div><p><a href="summary.json">Summary</a> · <a href="evidence_inventory.json">Full inventory</a> · <a href="noninterference_audit.json">Baseline preservation audit</a></p></main>'
    body += '<footer>No API calls or new decoding. No formal accuracy/AP measured. No human review form or score override is active.</footer>'
    body += '''<script>
const search=document.getElementById('search'),family=document.getElementById('family');
function filter(){let n=0;for(const row of document.querySelectorAll('tr[data-search]')){
row.hidden=!(row.dataset.search.includes(search.value.toLowerCase())&&(!family.value||row.dataset.flags.split(' ').includes(family.value)));if(!row.hidden)n++;}
document.getElementById('visible-count').textContent=n+' / 36 windows';}
search.addEventListener('input',filter);family.addEventListener('change',filter);
</script>'''
    return page('V9.11 Local Mechanism Evidence', body)
