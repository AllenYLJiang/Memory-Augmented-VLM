"""Small human troubleshooting packet, explicitly NOT the formal integration review."""
import html
import json
import re
from pathlib import Path

from .contracts import file_sha256, read_json, semantic_sha256
from .role_scoped import atomic_json

REVIEW_VERSION = 'v99_diagnostic_review_1'


def questions(record):
    raw = record['compatible_payload']; checks = record['compatible_assessment']
    result = []
    def add(index, code, title, options):
        event = raw['observations'][index]
        result.append({'question_id': f'q{index+1}_{code}', 'observation_id': event['id'],
                       'title': title, 'options': [{'value': value, 'label': label} for value, label in options],
                       'observation': event, 'entities': raw['entities']})
    for index, event in enumerate(raw['observations']):
        local = checks['observations'][index]['local_issues']
        codes = {e['code'] for e in local}
        if 'NECESSARY_ARGUMENT_OUTSIDE_SUPPORT' in codes:
            add(index, 'identity', '本条关系所列的人物，在声称的支持帧中能否被确认？请区分局部接触与跨镜头身份。', [
                ('same_people_visible', '能确认这些人物，并能看到所述局部接触'),
                ('contact_identity_unknown', '可见局部接触，但不能确认是同一人物'),
                ('contact_not_visible', '看不到所述接触'), ('unclear', '画面不足，不能判断')])
        if 'CONTEXT_ARGUMENT_OVERLAP' in codes:
            add(index, 'object_role', '本条所述物体究竟是人物直接接触的动作对象，还是只在背景出现？', [
                ('contact_object', '可见人物直接接触该物体，它属于动作对象'),
                ('background_only', '该物体只在背景，不能确认直接接触'), ('unclear', '遮挡或画面不足，无法区分')])
        if 'INFERRED_IS_NOT_DIRECT' in codes:
            add(index, 'directness', '本条双人接触，在同一采样帧中直接可见，还是需要通过镜头剪辑推断？', [
                ('direct_local_contact', '同一采样帧中可直接看到所述双人接触'),
                ('edit_inference_only', '只靠剪辑顺序或反应推断，不能直接确认'),
                ('not_visible', '没有所述关系'), ('unclear', '无法判断')])
        relation = event.get('relation')
        evidence = event.get('evidence', '')+' '+(relation.get('evidence', '') if isinstance(relation, dict) else '')
        if (isinstance(relation, dict) and relation.get('state') == 'observed' and
            relation.get('strength') == 'directly_visible' and re.search(r'\bsuggest(?:s|ing)?\b', evidence, re.I)):
            add(index, 'reaction', '这条动作能直接看到接触/打击，还是只看到人物反应后推测发生了打击？', [
                ('direct_strike_visible', '直接看到所述接触或打击'),
                ('reaction_only', '只看到反应，不能直接确认打击'),
                ('not_visible', '看不到所述动作'), ('unclear', '无法判断')])
    if not result and not checks['valid']:
        add(0, 'other', '编码兼容处理后仍有未分类错误。仅判断这条观察是否由八帧支持，并在备注说明。', [
            ('supported', '可见证据支持'), ('unsupported', '可见证据不支持'), ('unclear', '无法判断')])
    return result


def build_packet(records, original_rows, source_digest):
    cases = []
    for record in records:
        if record['compatible_assessment']['valid']: continue
        uid = record['window_uid']; case_id = semantic_sha256({'uid': uid, 'scope': REVIEW_VERSION})[:12]
        row = original_rows[uid]
        cases.append({'case_id': case_id, 'frame_indices': row['sampled_frame_indices'],
                      'questions': questions(record), 'private_window_uid': uid})
    private = {c['case_id']: c.pop('private_window_uid') for c in cases}
    packet = {'version': REVIEW_VERSION, 'scope': 'diagnostic_troubleshooting_only', 'active': bool(cases),
              'source_digest': source_digest, 'evidence_scope': 'eight_frames', 'cases': cases,
              'formal_review_active': False, 'integration_authorized': False, 'automatic_patch_authorized': False}
    packet['packet_id'] = semantic_sha256(packet)
    return packet, private


STYLE = '''
*{box-sizing:border-box}body{margin:0;background:#fff;color:#202329;font:16px/1.5 system-ui,sans-serif;letter-spacing:0}
main{max-width:1320px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}h2{font-size:21px}h3{font-size:17px}
.scope{border-left:4px solid #b47b10;padding:8px 14px;background:#fff8e8}.status{color:#50605b}
header{border-bottom:1px solid #cdd2d4;padding-bottom:18px}.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.case{padding:16px 0 28px;border-bottom:2px solid #d8dedc}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
figure{margin:0;min-width:0}img{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#16191d}
figcaption{font-size:13px;color:#4d565d;padding:4px 0}fieldset{border:0;border-top:1px solid #dce1e3;padding:16px 0;margin-top:14px;min-width:0}
legend{font-size:16px;font-weight:650;padding:0 12px 0 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f2f4f5;padding:12px;max-height:260px;overflow:auto}
label{display:block;margin:8px 0}input[type=text],select,textarea{font:inherit;border:1px solid #9ba9b0;border-radius:4px;padding:8px;max-width:100%}
select,textarea{width:100%}textarea{min-height:70px}.bins{display:flex;gap:14px;flex-wrap:wrap}.bins label{white-space:nowrap}
button{font:inherit;background:#087b62;color:white;border:0;padding:10px 20px;border-radius:4px;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #bc7a14;outline-offset:3px}
a{color:#14598c}footer{padding:22px 0}.error{color:#a32124}.small{font-size:14px;color:#56616a}
@media(max-width:700px){main{padding:16px}.frames{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:23px}.bar{align-items:flex-start}}
'''


def render(packet):
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>V9.9 局部证据复核</title><style>'+STYLE+'</style><main><header>',
             '<h1>V9.9 局部证据复核</h1><p class="scope">仅排查剩余证据问题，不是异常分类打分或正式接入审批。unknown 不等于 normal。请只依据下方八帧，不借用影片情节。</p>',
             '<div class="bar"><label>复核者编号 <input type="text" id="reviewer" placeholder="R1" maxlength="64"></label>',
             '<span id="progress" class="status"></span></div></header><form id="review-form">']
    for number, case in enumerate(packet['cases'], 1):
        cid = case['case_id']
        parts += [f'<section class="case" id="case-{cid}"><h2>窗口 {number} · {cid}</h2><div class="frames">']
        for i, frame in enumerate(case['frame_indices']):
            src = f'frames/{cid}/T{i}.jpg'
            parts.append(f'<figure><a href="{src}" target="_blank"><img src="{src}" alt="T{i}，帧 {frame}"></a><figcaption>T{i} · 帧 {frame}</figcaption></figure>')
        parts.append('</div>')
        for q in case['questions']:
            key = cid+'__'+q['question_id']
            relation = q['observation'].get('relation') or {}
            refs = {uid for role in ('actor_ids', 'target_ids', 'participant_ids', 'object_ids', 'instrument_ids')
                    for uid in relation.get(role, [])}
            support = ', '.join('T'+str(b) for b in relation.get('bins', q['observation']['bins']))
            brief = ('<p><strong>待核查描述：</strong>'+esc(q['observation']['evidence'])+'</p>'+
                     '<p class="small">模型声称的支持帧：'+esc(support)+'</p><ul class="small">'+
                     ''.join('<li>'+esc(e['id']+': '+e['description'])+'；模型记录可见帧 '+
                             esc(', '.join('T'+str(b) for b in e['bins']))+'</li>' for e in q['entities'] if e['id'] in refs)+'</ul>')
            parts += [f'<fieldset data-key="{key}"><legend>{esc(q["observation_id"])} · {esc(q["title"])}</legend>',
                      brief,
                      '<details><summary>待核查的模型观察与人物/物体描述（不是标准答案）</summary><pre>'+esc(json.dumps({'observation': q['observation'], 'entities': q['entities']}, ensure_ascii=False, indent=2))+'</pre></details>',
                      '<label>判断<select name="choice" required><option value="">尚未填写</option>'+''.join(f'<option value="{esc(o["value"])}">{esc(o["label"])}</option>' for o in q['options'])+'</select></label>',
                      '<div>据此作答的帧（可多选）</div><div class="bins">'+''.join(f'<label><input type="checkbox" name="bin" value="{i}">T{i}</label>' for i in range(8))+'</div>',
                      '<label>简短依据<textarea name="notes" required placeholder="指出可见部位、接触或无法判断的原因"></textarea></label></fieldset>']
        parts.append('</section>')
    parts.append('<footer><button type="submit">导出复核结果</button><p id="message" role="status"></p></footer></form></main>')
    data = json.dumps(packet, ensure_ascii=False).replace('<', '\\u003c')
    parts.append('<script>const packet='+data+';\n'+SCRIPT+'</script></html>')
    return ''.join(parts)


SCRIPT = r'''
const form=document.getElementById('review-form');
const reviewer=document.getElementById('reviewer');
const fields=[...document.querySelectorAll('fieldset[data-key]')];
const storageKey='v99_review_'+packet.packet_id;
function snapshot(){return {reviewer:reviewer.value,answers:Object.fromEntries(fields.map(f=>[f.dataset.key,{choice:f.querySelector('select').value,notes:f.querySelector('textarea').value,bins:[...f.querySelectorAll('input[name=bin]:checked')].map(x=>Number(x.value))}]))};}
function update(){const s=snapshot();let n=Object.values(s.answers).filter(a=>a.choice&&a.notes.trim()&&a.bins.length).length;document.getElementById('progress').textContent=`${n} / ${fields.length} 项已填写`;try{localStorage.setItem(storageKey,JSON.stringify(s));}catch(e){}}
try{const s=JSON.parse(localStorage.getItem(storageKey)||'null');if(s){reviewer.value=s.reviewer||'';for(const f of fields){const a=s.answers?.[f.dataset.key];if(a){f.querySelector('select').value=a.choice||'';f.querySelector('textarea').value=a.notes||'';for(const box of f.querySelectorAll('input[name=bin]'))box.checked=(a.bins||[]).includes(Number(box.value));}}}}catch(e){}
form.addEventListener('input',update);reviewer.addEventListener('input',update);update();
form.addEventListener('submit',event=>{event.preventDefault();const message=document.getElementById('message');message.className='error';
if(!/^[A-Za-z0-9_.-]{1,64}$/.test(reviewer.value.trim())){message.textContent='请填写复核者编号，例如 R1。';reviewer.focus();return;}
const answers=[];for(const c of packet.cases)for(const q of c.questions){const f=fields.find(x=>x.dataset.key===c.case_id+'__'+q.question_id);const a=snapshot().answers[f.dataset.key];if(!a.choice||!a.notes.trim()||!a.bins.length){message.textContent='请为每项填写判断、所依据的帧及简短依据。';f.scrollIntoView({block:'center'});return;}answers.push({case_id:c.case_id,question_id:q.question_id,choice:a.choice,evidence_bins:a.bins,notes:a.notes.trim()});}
const result={version:packet.version,packet_id:packet.packet_id,reviewer_id:reviewer.value.trim(),evidence_scope:'eight_frames',completed:true,answers};
const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='v99_diagnostic_review_'+result.reviewer_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message.className='status';message.textContent='复核文件已导出；原始结果和门禁未改变。';});
'''


def validate_return(packet, value):
    if not isinstance(value, dict) or set(value) != {'version', 'packet_id', 'reviewer_id', 'evidence_scope', 'completed', 'answers'}:
        raise ValueError('Invalid diagnostic review envelope')
    if value['version'] != REVIEW_VERSION or value['packet_id'] != packet['packet_id'] or not packet['active']:
        raise ValueError('Stale or inactive diagnostic packet')
    if value['evidence_scope'] != 'eight_frames' or value['completed'] is not True:
        raise ValueError('Only completed eight-frame diagnostic review can be imported')
    if not isinstance(value['reviewer_id'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', value['reviewer_id']):
        raise ValueError('Reviewer ID required, for example R1')
    expected = {(c['case_id'], q['question_id']): q for c in packet['cases'] for q in c['questions']}
    if not isinstance(value['answers'], list): raise ValueError('Answers must be a list')
    seen = set()
    for answer in value['answers']:
        if not isinstance(answer, dict) or set(answer) != {'case_id', 'question_id', 'choice', 'evidence_bins', 'notes'}:
            raise ValueError('Unexpected answer fields')
        if not all(isinstance(answer[k], str) for k in ('case_id', 'question_id', 'choice', 'notes')):
            raise ValueError('Invalid text fields')
        key = (answer['case_id'], answer['question_id'])
        if key in seen or key not in expected: raise ValueError('Duplicate or unexpected question')
        seen.add(key)
        if answer['choice'] not in {o['value'] for o in expected[key]['options']}:
            raise ValueError('Invalid choice; unknown/unclear is allowed, pending is not')
        bins = answer['evidence_bins']
        if (not isinstance(bins, list) or not bins or any(type(i) is not int or not 0 <= i < 8 for i in bins)
                or bins != sorted(set(bins))): raise ValueError('Evidence bins must be unique sorted integers 0..7')
        if not answer['notes'].strip(): raise ValueError('Brief evidence notes required')
    if seen != set(expected): raise ValueError('Complete all selected diagnostic questions')
    return {'diagnostic_review_complete': True, 'questions': len(seen),
            'unclear_answers': sum(a['choice'] == 'unclear' for a in value['answers']),
            'formal_review_active': False, 'technical_gate_overridden': False,
            'automatic_patch_authorized': False, 'scoring_authorized': False,
            'training_authorized': False, 'ready_for_shadow_integration': False,
            'next': 'Inspect human evidence before any explicitly versioned evidence correction; do not auto-upgrade scores.'}


def import_return(out, path):
    packet = read_json(Path(out)/'diagnostic_review/packet.json')
    data = Path(path).read_bytes()
    def no_duplicates(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('Duplicate JSON field')
            value[key] = item
        return value
    value = json.loads(data.decode('utf-8-sig'), object_pairs_hook=no_duplicates)
    result = validate_return(packet, value)
    digest = file_sha256(path)
    target = Path(out)/'review_imports'/(digest+'.json')
    receipt = {'input_sha256': digest, 'packet_id': packet['packet_id'], 'review': value, 'result': result}
    if target.exists():
        if read_json(target) != receipt: raise ValueError('Immutable review receipt changed')
    else: atomic_json(target, receipt)
    return result
