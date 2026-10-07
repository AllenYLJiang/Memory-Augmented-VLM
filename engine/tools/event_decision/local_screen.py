"""Bounded, resumable local-only candidate screening. No VLM imports."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .contracts import file_sha256, read_json, semantic_sha256, write_json, write_jsonl
from .hard_trial import group_id
from .label_scope import _load_gather_anchors


def frame_probe(path):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                             'stream=nb_frames,duration,r_frame_rate','-of','json',str(path)],
                            capture_output=True, check=True, timeout=90)
    stream = json.loads(result.stdout)['streams'][0]
    # Estimated duration*fps cannot establish a valid last frame.
    n = str(stream.get('nb_frames',''))
    if not n.isdigit(): raise ValueError('exact nb_frames unavailable')
    return int(n)


def local_frames(path, start):
    indices = [start+int(i) for i in np.linspace(0,95,8)]
    expr = '+'.join(f'eq(n\\,{i})' for i in indices)
    command=['ffmpeg','-v','error','-i',str(path),'-vf',f'select={expr},scale=32:32,format=gray',
             '-frames:v','8','-fps_mode','passthrough','-f','rawvideo','pipe:1']
    result=subprocess.run(command,capture_output=True,check=True,timeout=180)
    if len(result.stdout)!=8*32*32: raise ValueError('eight exact frames unavailable')
    images=np.frombuffer(result.stdout,dtype=np.uint8).reshape(8,32,32)
    hashes=[]
    for image in images:
        tiny=image[np.ix_(np.linspace(0,31,8).astype(int),np.linspace(0,31,9).astype(int))]
        bits=(tiny[:,1:]>tiny[:,:-1]).flatten()
        value=0
        for bit in bits: value=(value<<1)|int(bit)
        hashes.append(f'{value:016x}')
    return {'evidence_sha256':hashlib.sha256(result.stdout).hexdigest(), 'perceptual_fingerprint':hashes,
            'local_motion_score':float(np.abs(np.diff(images.astype(float),axis=0)).mean()/255),
            'sampled_frame_indices':indices,'media_verified':True,
            'screening_method':'mean_gray_frame_difference_includes_camera_motion_v1',
            'not_inference_media':'32px fingerprints are screening only; inference must use original eight-frame contract'}


def screen(xd_root, train_root, history, out, limit=384, seed=20260910):
    xd_root,train_root,out=Path(xd_root),Path(train_root),Path(out)
    if not train_root.is_dir(): raise ValueError('WAITING_FOR_LOCAL_MEDIA: train root absent')
    gather=_load_gather_anchors(xd_root/'pipeline/tools')
    select=xd_root/'Transformer_semantic_components_select_anomaly'
    positive=gather(select/'top_anomalous_frames_72B_positive_segments')
    negative=gather(select/'top_anomalous_frames_72B_negative_segments')
    used=set(history['source_groups']); buckets={k:[] for k in ('B1','B4','A','post','canary')}
    for path in sorted(train_root.rglob('*.mp4')):
        video=path.name[:-4]
        if group_id(video) in used: continue
        suffix=video.split('_label_',1)[-1].split('-')
        if suffix[0]=='A':
            buckets['A'].append((path,video)); continue
        if positive.get(video):
            kind='B1' if 'B1' in suffix else 'B4' if 'B4' in suffix else 'canary'
            buckets[kind].append((path,video))
        # A negative/context anchor may coexist with a positive anchor elsewhere
        # in the same video. It is still NOT a verified normal/event-phase label.
        outside=[span for span in negative.get(video,[]) if all(span[1]<a or b<span[0] for a,b,_ in positive.get(video,[]))]
        if outside: buckets['post'].append((path,video))
    pool_counts={kind:{'videos':len({video for _,video in values}), 'source_groups':len({group_id(video) for _,video in values})}
                 for kind,values in buckets.items()}
    missing=[kind for kind in ('B1','B4','A','post','canary') if not buckets[kind]]
    write_json(out/'available_pool_before_decoding.json',{'pools':pool_counts,'missing_required_strata':missing})
    if missing:
        write_jsonl(out/'candidate_windows.jsonl',[])
        write_jsonl(out/'screening_errors.jsonl',[])
        write_json(out/'screening_summary.json',{'status':'WAITING_FOR_NEW_SOURCE_OR_HISTORY_RECONCILIATION',
                    'missing_required_strata':missing,'pools':pool_counts,'screened':0,'usable':0,'errors':0,
                    'remote_calls':0,'reason':'do not spend decode/API budget when a required pool is empty'})
        return out/'candidate_windows.jsonl'
    for values in buckets.values(): values.sort(key=lambda x:semantic_sha256([seed,x[1]]))
    # Round-robin local screening prevents abundant normals consuming the budget.
    order=[]
    while any(buckets.values()) and len(order)<limit:
        for kind in buckets:
            if buckets[kind] and len(order)<limit:
                path,video=buckets[kind].pop(0); order.append((kind,path,video))
    rows,errors=[],[]
    for ordinal,(kind,path,video) in enumerate(order,1):
        cache_key=semantic_sha256([str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns,seed,kind,
                                  positive.get(video,[]),negative.get(video,[]),'screen_v2'])
        cache=out/'cache'/f'{cache_key}.json'
        cached=read_json(cache)
        if cached is not None:
            rows.append(cached); continue
        try:
            n=frame_probe(path)
            if n<96: raise ValueError('shorter than96 frames')
            spans=positive.get(video,[]) if kind not in ('A','post') else negative.get(video,[])
            if kind=='post': spans=[span for span in spans if all(span[1]<a or b<span[0] for a,b,_ in positive.get(video,[]))]
            center=int((spans[0][0]+spans[0][1])/2) if spans else int(n*.5)
            start=min(max(0,center-48),n-96)
            row={'dataset_partition':'train','video_id':video,'video_path':str(path.resolve()),
                 'start_frame':start,'end_frame_exclusive':start+96,**local_frames(path,start),
                 'stratum': {'A':'label_A_pool','post':'hard_postevent_unverified','B1':'B1_weak_positive','B4':'B4_weak_positive','canary':'other_class_canary'}[kind],
                 'label_source':'explicit_filename_label_A' if kind=='A' else 'negative_anchor_unverified' if kind=='post' else 'verified_positive_anchor',
                 'positive_spans_half_open':[[int(a),int(b)+1] for a,b,_ in positive.get(video,[])],
                 'positive_span_provenance':[]}
            if row['label_source']=='verified_positive_anchor':
                folder=select/'top_anomalous_frames_72B_positive_segments'
                candidates=[folder/video/'selected_frames.json',folder/(video+'.mp4')/'selected_frames.json']
                row['positive_span_provenance']=[{'path':str(p),'sha256':file_sha256(p)} for p in candidates if p.is_file()]
            write_json(cache,row); rows.append(row)
        except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
            errors.append({'video_id':video,'error':str(exc)[:500]})
        if ordinal%20==0: print(f'[local-screen] {ordinal}/{len(order)} candidates={len(rows)} errors={len(errors)} API=0',flush=True)
    normals=sorted([r for r in rows if r['stratum']=='label_A_pool'],key=lambda r:(r['local_motion_score'],r['video_id']))
    easy={r['video_id'] for r in normals[:max(16,len(normals)//4)]}
    for r in normals: r['stratum']='easy_label_A' if r['video_id'] in easy else 'hard_label_A'
    write_jsonl(out/'candidate_windows.jsonl',rows)
    write_jsonl(out/'screening_errors.jsonl',errors)
    write_json(out/'screening_summary.json',{'screened':len(order),'usable':len(rows),'errors':len(errors),'budget':limit,
                                          'seed':seed,'sampling':'bounded center-window screen, not full-dataset enumeration',
                                          'postevent_label_status':'unverified candidate only; negative anchor does not establish event phase',
                                          'remote_calls':0})
    return out/'candidate_windows.jsonl'
