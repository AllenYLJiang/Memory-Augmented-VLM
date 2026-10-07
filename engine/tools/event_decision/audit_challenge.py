"""Human audit diagnostics only. This module deliberately has no fitting imports."""
from pathlib import Path

import numpy as np

from .contracts import iter_jsonl, read_json, write_json, write_jsonl
from .evaluation import metrics_with_predictions
from .models import sigmoid


def audit(out):
    out = Path(out); ref = out / 'source_reference'
    current = read_json(ref / 'feature_store/CURRENT.json')
    store = ref / 'feature_store' / current['contract_id']
    X, mask = np.load(store / 'values.npy', allow_pickle=False), np.load(store / 'observed.npy', allow_pickle=False)
    features = {r['window_uid']: r for r in iter_jsonl(store / 'rows.jsonl')}
    enrollment = {r['window_uid']: r for r in iter_jsonl(ref / 'review_private/enrollment_manifest.jsonl')}
    sidecar = list(iter_jsonl(ref / 'review_adjudication/human_audit_sidecar.jsonl'))
    results = []
    for r in sidecar:
        uid = r['window_uid']; feature = features.get(uid); meta = enrollment[uid]
        label = {'normal': 0, 'anomalous': 1}.get(r.get('resolved_visual_label'))
        for family_dir in sorted(p for p in (out / 'models').iterdir() if p.is_dir()):
            eligible = []
            for path in sorted(family_dir.glob('outer_fold_*/models.json')):
                bundle = read_json(path)
                models = [bundle[k] for k in ('candidate', 'F0_matched', 'F0_operational')]
                if all(m.get('status') == 'CERTIFIED' and meta['source_group'] not in set(m['fit_source_groups'] + m['threshold_source_groups']) for m in models):
                    eligible.append((path, bundle))
            result = {"window_uid": uid, "blind_id": r['blind_id'], "family": family_dir.name,
                      "source_group": meta['source_group'], "human_visual_label": r.get('resolved_visual_label'),
                      "y": label, "legacy_proxy_label": meta.get('legacy_y_true'), "use_policy": "audit_only",
                      "chosen_without_looking_at_predictions": True, "included_in_fit": False}
            if not eligible or not feature:
                result['status'] = 'NO_SOURCE_EXCLUDED_MODEL_OR_FEATURE'
            else:
                path, bundle = eligible[0]
                result.update(status='scored', model_path=str(path.relative_to(out)))
                for name in ('candidate', 'F0_matched', 'F0_operational'):
                    m = bundle[name]; cols = m['feature_indices']; i = feature['row_index']
                    seen = mask[i, cols].all() and np.isfinite(X[i, cols]).all()
                    p = float(sigmoid(np.asarray(m['model']['intercept'] + ((X[i, cols]-m['standardizer']['mean'])/m['standardizer']['scale']) @ m['model']['coef']))) if seen else None
                    result[name] = p
                    result[name+'_pred'] = int(p >= m['threshold']) if p is not None else None
                historical=read_json(ref/path.relative_to(out).parent/'model.json')
                if historical and result.get('candidate') is not None:
                    s,m=historical['standardizer'],historical['model']; cols=historical['feature_indices']; i=feature['row_index']
                    hp=float(sigmoid(np.asarray(m['intercept']+((X[i,cols]-s['mean'])/s['scale'])@m['coef'])))
                    result['historical_same_fold_candidate_probability']=hp
                    result['historical_same_fold_candidate_pred']=int(hp>=historical['threshold'])
                    result['numerical_repair_changed_binary_decision']=result['candidate_pred']!=int(hp>=historical['threshold'])
            results.append(result)
    summary = {}
    for family in sorted({r['family'] for r in results}):
        all_rows = [r for r in results if r['family'] == family]
        common = [r for r in all_rows if r['y'] is not None and r.get('candidate') is not None and r.get('F0_matched') is not None]
        summary[family] = {'requested': len(all_rows), 'binary_feature_complete_source_excluded': len(common),
                           'uncertain': sum(r['y'] is None for r in all_rows),
                           'metrics': {name: metrics_with_predictions([r['y'] for r in common], [r[name] for r in common], [r[name+'_pred'] for r in common])
                                       for name in ('candidate', 'F0_matched', 'F0_operational')}}
    write_jsonl(out / 'audit_only/human_challenge_predictions.jsonl', results)
    write_json(out / 'audit_only/human_challenge_metrics.json', {"use_policy": "audit_only", "selection_allowed": False, "families": summary})
    return summary
