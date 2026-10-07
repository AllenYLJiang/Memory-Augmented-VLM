"""Local reports for partially valid observations, not benchmark scores."""
import html
from collections import Counter
from pathlib import Path
from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json
from .mechanism_v94_contract import assess, VERSION

CSS='''body{font:15px system-ui;margin:24px;color:#17221d;background:#f6f7f9;max-width:1500px}h1{font-size:24px}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #cbd2d0;padding:8px;text-align:left;vertical-align:top}th{background:#e5ece8}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frames img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}figure{margin:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:12px}a{color:#0a6573}video{max-width:720px;width:100%}@media(max-width:700px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}body{margin:12px}table{font-size:12px}}'''


def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><body>'+body+'</body></html>'


def esc(v): return html.escape(str(v))


def current(out,row,digest):
    path=Path(out)/'results'/(row['window_uid']+'.json')
    try: r=read_json(path)
    except (ValueError,UnicodeError): return {'status':'invalid_cache'}
    if r is None: return {'status':'pending'}
    if not isinstance(r,dict) or r.get('window_uid')!=row['window_uid'] or r.get('protocol_sha256')!=digest:
        raise ValueError('Mismatched V9.4 result identity')
    if r.get('status') in {'success','partial'}:
        a=assess(r.get('parsed'))
        if a!=r.get('assessment'): return {'status':'invalid_cache'}
    return r


def report(out,protocol,rows,changed=None):
    out=Path(out);digest=file_sha256(out/'protocol.json');lines=[];comparisons=[];statuses=Counter();core=0;issues=Counter()
    for row in rows:
        uid=row['window_uid'];r=current(out,row,digest);statuses[r['status']]+=1
        a=r.get('assessment',{});p=r.get('parsed') if isinstance(r.get('parsed'),dict) else {};old=read_json(out/'source_snapshot/results'/(uid+'.json'),{})
        oldp=old.get('parsed',{});core+=bool(a.get('core_valid'))
        issues.update(i['reason'] for i in a.get('issues',[]))
        comparisons.append({'window_uid':uid,'source_group':row['source_group'],'video_id':row['video_id'],
             'selection_reasons':row['selection_reasons'],'status':r['status'],
             'v93_raw_b5':oldp.get('b5_presence'),'v94_raw_b5':p.get('b5'),
             'assessment':a,'formal_accuracy':None,'formal_AP':None,'training_loss_mask':False})
        url='cases/'+uid[:20]+'/index.html'
        lines.append('<tr>'+''.join('<td>'+x+'</td>' for x in [f'<a href="{url}">{esc(row["video_id"])}</a>',esc(row['start_frame']),esc(r['status']),esc(oldp.get('b5_presence')),esc(p.get('b5',{}).get('label')),esc(a.get('local_supported_events')),esc(len(a.get('issues',[])))])+'</tr>')
        if changed is not None and changed!=uid: continue
        folder=out/'cases'/uid[:20]
        body='<a href="../../index.html">All selected windows</a><h1>'+esc(row['video_id'])+'</h1><p>Development only. No gold labels, accuracy, or graph-over-node proof. Raw judgment is not overwritten by diagnostics.</p>'
        body+='<p>Original frames ['+str(row['start_frame'])+', '+str(row['end_frame_exclusive'])+')</p><video controls preload="none" src="clip.mp4"></video><div class="frames">'
        body+=''.join(f'<figure><a href="T{i}.jpg"><img loading="lazy" src="T{i}.jpg" alt="T{i}"></a><figcaption>T{i}: frame {frame}</figcaption></figure>' for i,frame in enumerate(row['sampled_frame_indices']))+'</div>'
        body+='<h2>Status: '+esc(r['status'])+'</h2><p>V9.3: '+esc(oldp.get('b5_presence'))+' | V9.4: '+esc(p.get('b5'))+'</p>'
        body+='<p>Supported local events: '+esc(a.get('local_supported_events'))+'; unknown links do not erase observations.</p>'
        import json
        for title,value in [('Typed observations',p),('Per-field validity and issues',a),('Failed request',r.get('error')),('Original V9.3 result',old)]:
            body+='<details><summary>'+title+'</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
        if row['design_feedback']: body+='<p>Prior development feedback, not blind gold: '+esc(row['design_feedback']['human_note_verbatim'])+'</p>'
        (folder/'index.html').write_text(page(row['video_id'],body),encoding='utf-8')
    summary={'version':VERSION,'mock':protocol['mock'],'selected':len(rows),'counts':dict(statuses),
        'core_valid':core,'field_issue_counts':dict(issues),'complete_acquisition':all(k in {'success','partial','provider_rejected'} for k in statuses),
        'formal_accuracy':None,'formal_AP':None,'locked_authorized':False,'training_authorized':False}
    atomic_json(out/'summary.json',summary);write_jsonl(out/'comparison.jsonl',comparisons)
    body='<h1>V9.4 Scoped Mechanism Development</h1><p>Same saved frames; new contract. No graph discovery, no training, no independent evaluation.</p>'
    body+='<p>'+esc(dict(statuses))+' | Core valid: '+str(core)+'/'+str(len(rows))+'</p><p><a href="review/index.html">Evidence review packet</a></p>'
    body+='<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in ['Video','Start','Status','Old B5','New B5','Supported local events','Issues'])+'</tr></thead><tbody>'+''.join(lines)+'</tbody></table>'
    (out/'index.html').write_text(page('V9.4 development',body),encoding='utf-8')
    return summary
