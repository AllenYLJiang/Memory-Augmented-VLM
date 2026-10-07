"""Incremental same-cohort reports and field-independent technical metrics."""
import html
import json
from collections import Counter
from pathlib import Path

from .contracts import file_sha256, read_json, write_jsonl
from .role_scoped import atomic_json
from .mechanism_v94_report import page
from .mechanism_v95_contract import VERSION, assess


def current(out, row, protocol):
    path=Path(out)/'results'/(row['window_uid']+'.json')
    try: result=read_json(path)
    except (ValueError,UnicodeError): return {'status':'invalid_cache'}
    if result is None: return {'status':'pending'}
    if not isinstance(result,dict) or result.get('window_uid') != row['window_uid'] or result.get('protocol_sha256') != file_sha256(Path(out)/'protocol.json') or result.get('mock') is not protocol['mock']:
        raise ValueError('Result identity/protocol/MOCK mismatch')
    if result.get('status') not in {'success','partial','failed','provider_rejected'}: return {'status':'invalid_cache'}
    if result['status'] in {'success','partial'} and assess(result.get('parsed')) != result.get('assessment'): return {'status':'invalid_cache'}
    return result


def technical_metrics(results, protocol):
    n=len(results); audits=[r.get('assessment') or assess(None) for r in results]
    events=[e for a in audits for e in a['events']]; links=[e for a in audits for e in a['links']]
    def ratio(a,b,empty=0): return a/b if b else empty
    metrics={
        'core_valid_fraction':ratio(sum(a['core_valid'] for a in audits),n),
        'label_valid_fraction':ratio(sum(a['label_valid'] for a in audits),n),
        'numeric_valid_fraction':ratio(sum(a['fields'].get('/b5/probability',{}).get('valid',False) for a in audits),n),
        'numeric_observed_fraction':ratio(sum(a['probability_observed'] for a in audits),n),
        'event_valid_fraction':ratio(sum(e['valid'] for e in events),len(events)),
        'observation_valid_fraction':ratio(sum(e['observation_valid'] for e in events),len(events)),
        'event_consistency_fraction':ratio(sum(e['support_consistent'] for e in events),len(events)),
        'link_valid_fraction':ratio(sum(e['valid'] for e in links),len(links),1),
        # Every window contributes its independently checked frames, even with a bad score or event list.
        'frame_valid_fraction':ratio(sum(sum(a['frame_valid']) for a in audits),8*n),
        'judgment_consistency_fraction':ratio(sum(a['fields'].get('/b5/consistency',{}).get('valid',False) for a in audits),n),
        'complete_acquisition':bool(n) and all(r['status'] in {'success','partial','provider_rejected'} for r in results),
        'event_count':len(events),'link_count':len(links),'window_count':n}
    names={'core_valid_fraction':'minimum_core_fraction','event_valid_fraction':'minimum_event_valid_fraction',
           'event_consistency_fraction':'minimum_event_consistency_fraction','numeric_valid_fraction':'minimum_numeric_valid_fraction',
           'link_valid_fraction':'minimum_link_valid_fraction','frame_valid_fraction':'minimum_frame_valid_fraction',
           'judgment_consistency_fraction':'minimum_consistency_fraction'}
    metrics['failed_checks']=[name for name,key in names.items() if metrics[name] < protocol['gates'][key]]
    if not metrics['complete_acquisition']: metrics['failed_checks'].append('incomplete_acquisition')
    metrics['technical_ready']=not metrics['failed_checks']
    return metrics


def report(out, protocol, rows, changed=None):
    out=Path(out); records=[current(out,r,protocol) for r in rows]
    metrics=technical_metrics(records,protocol); statuses=Counter(r['status'] for r in records)
    issue_counts=Counter(i['reason'] for r in records for i in r.get('assessment',{}).get('issues',[]))
    comparisons=[]; table=[]
    esc=lambda v:html.escape(str(v))
    for row,r in zip(rows,records):
        uid=row['window_uid']; old=read_json(out/'source_snapshot/results'/(uid+'.json'),{})
        raw=r.get('parsed') if isinstance(r.get('parsed'),dict) else {}; oldraw=old.get('parsed') or {}
        b5=raw.get('b5') if isinstance(raw.get('b5'),dict) else {}
        a=r.get('assessment') or assess(None); url='cases/'+uid[:20]+'/index.html'
        old_label=oldraw.get('b5',{}).get('label'); new_label=b5.get('label')
        comparisons.append({'window_uid':uid,'source_group':row['source_group'],'video_id':row['video_id'],
            'selection_reasons':row['selection_reasons'],'old_status':old.get('status'),'old_b5':oldraw.get('b5'),
            'status':r['status'],'new_b5':raw.get('b5'),'assessment':a,'formal_accuracy':None,'formal_AP':None})
        table.append('<tr>'+''.join('<td>'+v+'</td>' for v in [f'<a href="{url}">{esc(row["video_id"])}</a>',
            esc(row['start_frame']),esc(r['status']),esc(old_label),esc(new_label),esc(b5.get('probability')),
            esc(a['local_supported_events']),esc(len(a['issues']))])+'</tr>')
        if changed is not None and changed != uid: continue
        body='<a href="../../index.html">All 36 development cases</a><h1>'+esc(row['video_id'])+'</h1>'
        if protocol['mock']: body+='<p><strong>MOCK: synthetic responses, not new VLM observations. Do not review these as real results.</strong></p>'
        body+='<p>Frames ['+str(row['start_frame'])+','+str(row['end_frame_exclusive'])+'). Model claims, not gold labels.</p>'
        body+='<video controls preload="none" src="clip.mp4"></video><div class="frames">'
        body+=''.join(f'<figure><a href="T{i}.jpg"><img loading="lazy" src="T{i}.jpg" alt="T{i}"></a><figcaption>T{i}: {f}</figcaption></figure>' for i,f in enumerate(row['sampled_frame_indices']))+'</div>'
        body+='<h2>'+esc(r['status'])+'</h2><p>V9.4 B5: '+esc(old_label)+'; V9.5 B5: '+esc(new_label)+'; probability: '+esc(b5.get('probability'))+'</p>'
        body+='<p>Supported local events: '+esc(a['local_supported_events'])+'. Unknown is not normal.</p>'
        body+='<table><tr><th>Event</th><th>Observation / phase</th><th>Typed relationship</th><th>Role observations / relation bins</th><th>Extra evidence needed</th></tr>'
        raw_events=raw.get('events') if isinstance(raw.get('events'),list) else []
        for e,ea in zip(raw_events,a['events']):
            if not isinstance(e,dict):continue
            b=e.get('binding') if isinstance(e.get('binding'),dict) else {}
            body+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [e.get('id'),str(e.get('type'))+' / '+str(e.get('phase')),b,
                {'roles':ea['role_observation_bins'],'evidence_bins':b.get('evidence_bins')},e.get('additional_evidence_needed')])+'</tr>'
        body+='</table>'
        for title,data in [('Issues and field masks',a),('Raw new response',raw),('Original V9.4 response',old),('Request error',r.get('error'))]:
            body+='<details><summary>'+title+'</summary><pre>'+esc(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>'
        (out/'cases'/uid[:20]/'index.html').write_text(page('V9.5 development case',body),encoding='utf-8')
    summary={'version':VERSION,'mock':protocol['mock'],'counts':dict(statuses),'field_issue_counts':dict(issue_counts),
        **metrics,'formal_accuracy':None,'formal_AP':None,'training_authorized':False,'locked_authorized':False}
    atomic_json(out/'summary.json',summary); write_jsonl(out/'comparison.jsonl',comparisons)
    body='<h1>V9.5 Typed Mechanism Development</h1><p>'+esc(dict(statuses))+'</p><p>Technical readiness: '+esc(metrics['technical_ready'])+'; failed checks: '+esc(metrics['failed_checks'])+'</p>'
    if protocol['mock']: body+='<p><strong>MOCK: software verification only. No new model result or human review is requested.</strong></p>'
    body+='<p>Diagnostic same-cohort comparison. No benchmark accuracy, no training authorization.</p>'
    if (out/'review/index.html').exists(): body+='<p><a href="review/index.html">Review packet (check current technical gate first)</a></p>'
    body+='<table><tr>'+''.join('<th>'+v+'</th>' for v in ('Video','Start','Status','V9.4 B5','V9.5 B5','Raw probability','Local support','Issues'))+'</tr>'+''.join(table)+'</table>'
    (out/'index.html').write_text(page('V9.5 development',body),encoding='utf-8')
    return summary
