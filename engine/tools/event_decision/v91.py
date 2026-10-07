"""Read-only historical replay and certified V9.1 development experiment."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .contracts import FEATURE_NAMES, file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from .evaluation import _write_csv, grouped_bootstrap_delta, metrics_with_predictions, tie_aware_ap
from .fitting import _choose_threshold, _inner_folds, _split_model_threshold
from .models import Standardizer, group_weights, weighted_bce
from .numerics import CertifiedRidgeLogisticScorer, certificate, named_bounds


SPECS = {
    "F0_m0_calibrated": {"names": ["m0_margin"], "lower": {"m0_margin": 1e-6}},
    "F1_graph_optional": {"names": list(FEATURE_NAMES[:3]), "lower": {"m0_margin": 0}},
    "F2_event_active": {"names": list(FEATURE_NAMES[:6]), "lower": {"m0_margin": 0, "o_active": 0, "q_direct": 0}},
    "T1_direct2": {"names": ["o_active", "q_direct"], "lower": {"o_active": 0, "q_direct": 0}},
}


def inventory(source):
    source = Path(source).resolve()
    paths = []
    for directory in ('labels', 'feature_store', 'splits', 'models', 'evaluation', 'review_private',
                      'review_returns', 'review_adjudication', 'review_imported', 'reviews', 'archive', 'seal'):
        root = source / directory
        if root.exists():
            paths.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.npy', '.csv'))
    paths.extend(source.glob('*.json'))
    return [{"path": p.relative_to(source).as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in sorted(set(paths))]


def snapshot(source, out, config):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("source and output must be disjoint sibling runs")
    if any(source.glob('*.lock')):
        raise ValueError("source has an active lock; stop its writer first")
    files = inventory(source)
    required = ['labels/label_ledger.jsonl', 'feature_store/CURRENT.json', 'splits/outer_inner_threshold_plan.json']
    if not all(any(r['path'] == name for r in files) for name in required):
        raise ValueError("source labels, features or frozen split missing")
    manifest = {"source": str(source), "files": files, "configuration": config}
    old = read_json(out / 'archive/source_run_manifest.json')
    if old is not None:
        old_tag=old['source'].replace('\\','/').rstrip('/').rsplit('/',1)[-1]
        if old.get('files')!=files or old.get('configuration')!=config or old_tag!=source.name:
            raise ValueError("source/configuration changed; use a NEW TAG")
    for row in files:
        target = out / 'source_reference' / row['path']
        if target.exists():
            if file_sha256(target) != row['sha256']:
                raise ValueError(f"snapshot altered: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row['path'], target)
    # Keep archived bytes unchanged when the same files are mounted via /mnt/c.
    if old is None:
        write_json(out / 'archive/source_run_manifest.json', manifest)
        write_jsonl(out / 'archive/source_files.sha256.jsonl', files)
        write_json(out / 'archive/config_snapshot.json', config)
    return files


def arrays(reference):
    reference = Path(reference)
    current = read_json(reference / 'feature_store/CURRENT.json')
    # CURRENT may name /mnt/c on Windows. Resolve its immutable contract locally.
    store = reference / 'feature_store' / current['contract_id']
    schema = read_json(store / 'schema.json', {})
    X, mask = np.load(store / 'values.npy', allow_pickle=False), np.load(store / 'observed.npy', allow_pickle=False)
    feature_rows = list(iter_jsonl(store / 'rows.jsonl'))
    lookup = {r['window_uid']: r['row_index'] for r in feature_rows}
    rows = read_json(reference / 'splits/outer_inner_threshold_plan.json')['rows']
    if len(lookup) != len(feature_rows) or len({r['window_uid'] for r in rows}) != len(rows):
        raise ValueError('duplicate feature/split UID')
    missing = [r['window_uid'] for r in rows if r['window_uid'] not in lookup]
    if missing:
        raise ValueError(f'frozen split contains {len(missing)} absent feature rows')
    index = [lookup[r['window_uid']] for r in rows]
    return X[index], mask[index].astype(bool), rows


def verify_external_seal(reference, project):
    seal=read_json(reference/'archive/v8_seal.json')
    if not seal: raise ValueError('historical V8 seal missing')
    root=project/'runs'/seal['legacy_tag']
    manifest=reference/'archive/v8_files.sha256.jsonl'
    errors=[]
    sealed_rows=list(iter_jsonl(manifest))
    if semantic_sha256(sealed_rows)!=seal['files_manifest_sha256']: errors.append('seal file manifest hash changed')
    for row in sealed_rows:
        path=(root/row['relative_path']).resolve()
        if root.resolve() not in path.parents or not path.is_file() or file_sha256(path)!=row['sha256']:
            errors.append(row['relative_path'])
    raw=seal['base_catalog'].replace('\\','/')
    catalog=project/'runs'/raw.split('/runs/',1)[1]
    if not catalog.is_file() or file_sha256(catalog)!=seal['base_catalog_sha256']: errors.append('governed13 graph catalog changed')
    stop=root/'gate_r1/STOPPED_BY_GATE'
    if not stop.is_file() or file_sha256(stop)!=seal['stop_marker_sha256']: errors.append('historical stop marker changed')
    code=read_json(reference/'archive/code_manifest.json',{})
    for row in code.get('files',[]):
        path=project/row['path'].replace('\\','/')
        if not path.is_file() or file_sha256(path)!=row['sha256']: errors.append('sealed_code:'+row['path'])
    return {'verified':not errors,'errors':errors,'catalog_path':str(catalog),'catalog_sha256':seal['base_catalog_sha256'],
            'declared_graph_count':seal['active_graph_total']}


def fit_one(X, y, groups, uids, indices, spec, regularization, config):
    cols = [FEATURE_NAMES.index(n) for n in spec['names']]
    scaler = Standardizer.fit(X[np.ix_(indices, cols)])
    matrix = scaler.transform(X[np.ix_(indices, cols)])
    context = {"ordered_window_uids": [uids[i] for i in indices], "scaler": scaler.to_json(), "spec": spec}
    model = CertifiedRidgeLogisticScorer(regularization, config.get('max_iter', 1000), config.get('tolerance', 1e-7),
                                        named_bounds(spec['names'], spec.get('lower', {})),
                                        named_bounds(spec['names'], spec.get('upper', {})))
    model.fit(matrix, y[indices], group_weights([groups[i] for i in indices]), fit_context=context)
    return {"spec": spec, "feature_indices": cols, "standardizer": scaler.to_json(), "model": model.to_json(),
            "fit_window_uids": context['ordered_window_uids'], "fit_source_groups": sorted({groups[i] for i in indices})}


def predict(model, X):
    from .models import sigmoid
    s, m = model['standardizer'], model['model']
    matrix = (X[:, model['feature_indices']] - s['mean']) / s['scale']
    return sigmoid(m['intercept'] + matrix @ m['coef'])


def fit_selected(X, y, groups, uids, train, threshold, spec, config, seed):
    if len(set(y[train])) != 2 or len(set(y[threshold])) != 2:
        return {'status': 'INSUFFICIENT_CLASSES', 'inner_cv': [], 'reason': 'fit/threshold pool lacks both classes'}
    heldouts = _inner_folds(train, groups, 3, seed)
    cv = []
    for reg in config.get('regularization_grid', [.01, .1, 1.]):
        record = {"regularization": reg, "folds": [], "valid": True}
        for heldout in heldouts:
            subset = [i for i in train if i not in set(heldout)]
            if not heldout or len(set(y[subset])) != 2 or len(set(y[heldout])) != 2:
                record['valid'] = False
                record['folds'].append({"status": "INSUFFICIENT_CLASSES"}); continue
            fitted = fit_one(X, y, groups, uids, subset, spec, reg, config)
            cert = fitted['model']['numeric_certificate']
            record['valid'] &= cert['certified']
            record['folds'].append({"certificate": cert, "fit_window_uids": [uids[i] for i in subset],
                                    "heldout_window_uids": [uids[i] for i in heldout],
                                    "loss": weighted_bce(y[heldout], predict(fitted, X[heldout]), group_weights([groups[i] for i in heldout]))})
        losses = [r['loss'] for r in record['folds'] if 'loss' in r]
        record['mean_bce'] = float(np.mean(losses)) if record['valid'] else None
        record['se_bce'] = float(np.std(losses, ddof=1) / np.sqrt(3)) if record['valid'] else None
        cv.append(record)
    eligible = [r for r in cv if r['valid']]
    if not eligible:
        return {"status": "NO_CERTIFIED_LAMBDA", "inner_cv": cv}
    best = min(eligible, key=lambda r: r['mean_bce'])
    reg = max(r['regularization'] for r in eligible if r['mean_bce'] <= best['mean_bce'] + best['se_bce'])
    fitted = fit_one(X, y, groups, uids, train, spec, reg, config)
    fitted.update(inner_cv=cv, status='CERTIFIED' if fitted['model']['numeric_certificate']['certified'] else 'INCOMPLETE_NUMERIC_EVALUATION')
    if fitted['status'] == 'CERTIFIED':
        threshold_value, metric = _choose_threshold(y[threshold], predict(fitted, X[threshold]))
        fitted.update(threshold=threshold_value, threshold_metric=metric,
                      threshold_window_uids=[uids[i] for i in threshold], threshold_source_groups=sorted({groups[i] for i in threshold}))
    return fitted


def reliability(y, probability):
    y, p = np.asarray(y), np.asarray(probability)
    if not len(y):
        return {"n": 0}
    bins = np.minimum((p * 10).astype(int), 9)
    cells = [{"bin": b, "n": int((bins == b).sum()), "mean_p": float(p[bins == b].mean()),
              "fraction_positive": float(y[bins == b].mean())} for b in range(10) if np.any(bins == b)]
    return {"n": len(y), "brier": float(np.mean((p-y)**2)),
            "ece_10_equal_width": sum(c['n'] * abs(c['mean_p']-c['fraction_positive']) for c in cells) / len(y),
            "bins": cells, "interpretation": "descriptive; not a separate calibration experiment"}


def metric_rows(rows, key):
    use = [r for r in rows if r.get(key) is not None]
    return metrics_with_predictions([r['y'] for r in use], [r[key] for r in use], [r[key+'_pred'] for r in use])


def run(source, out, config):
    source, out = Path(source), Path(out)
    before = snapshot(source, out, config)
    project=Path(__file__).resolve().parents[2]
    policy=read_json(project/'config/event_claim_policy_v2.json')
    if not policy or policy.get('version')!='event_claim_policy_v2' or policy.get('api_authorized') is not False or policy.get('deployment_authorized') is not False:
        raise ValueError('invalid offline claim policy')
    minimum_coverage=float(policy['minimum_expected_cohort_coverage'])
    if not .99<=minimum_coverage<=1: raise ValueError('claim policy cannot relax coverage below .99')
    old_policy=read_json(out/'archive/claim_policy_snapshot.json')
    if old_policy is not None and old_policy!=policy: raise ValueError('claim policy changed; use a new TAG')
    write_json(out/'archive/claim_policy_snapshot.json',policy)
    code_paths=[Path(__file__), Path(__file__).with_name('numerics.py'), Path(__file__).with_name('models.py'),
                Path(__file__).with_name('fitting.py'),Path(__file__).with_name('evaluation.py'),Path(__file__).with_name('contracts.py')]
    code_manifest=[{'path':p.relative_to(project).as_posix(),'sha256':file_sha256(p)} for p in code_paths]
    write_json(out/'archive/runtime_code_manifest.json',code_manifest)
    ref = out / 'source_reference'
    external=verify_external_seal(ref,source.resolve().parents[1])
    write_json(out/'receipts/external_seal_integrity.json',external)
    if not external['verified']: raise ValueError(f'historical external seal failed: {external["errors"]}')
    X, observed, rows = arrays(ref)
    ledger = {r['window_uid']: r for r in iter_jsonl(ref / 'labels/label_ledger.jsonl')}
    scope_errors = [r['window_uid'] for r in rows if r['window_uid'] not in ledger or
                    not ledger[r['window_uid']]['supervision']['window_loss_mask'] or
                    ledger[r['window_uid']]['supervision']['window_target'] != r['y'] or
                    ledger[r['window_uid']]['source_group'] != r['source_group']]
    write_json(out / 'receipts/scope_integrity.json', {'rows': len(rows), 'mismatches': scope_errors,
                                                     'ledger_sha256': file_sha256(ref / 'labels/label_ledger.jsonl'),
                                                     'passed': not scope_errors, 'human_labels_used': False})
    if scope_errors:
        raise ValueError('frozen split labels disagree with scope-valid source ledger')
    y = np.array([r['y'] for r in rows]); groups = [r['source_group'] for r in rows]
    uids = [r['window_uid'] for r in rows]; folds = np.array([r['outer_fold'] for r in rows])
    seed = config.get('seed', 20260907)
    outputs, numeric_audit, family_status = [], [], {}
    for family in config.get('families', list(SPECS)[:3]):
        spec = SPECS[family]; cols = [FEATURE_NAMES.index(n) for n in spec['names']]
        available = observed[:, cols].all(1) & np.isfinite(X[:, cols]).all(1)
        family_status[family] = []
        for fold in sorted(set(folds.tolist())):
            old = read_json(ref / f'models/{family}/outer_fold_{fold}/model.json')
            if old is None:
                raise ValueError(f'no frozen historical groups for {family}/{fold}')
            train = [i for i in range(len(rows)) if groups[i] in old['model_pool_groups']]
            threshold = [i for i in range(len(rows)) if groups[i] in old['threshold_pool_groups']]
            test = [i for i in range(len(rows)) if folds[i] == fold]
            if set(groups[i] for i in test) != set(old['test_groups']) or set(train) & set(threshold) or set(train+threshold) & set(test):
                raise ValueError('historical split provenance mismatch')
            ct, ch = [i for i in train if available[i]], [i for i in threshold if available[i]]
            bundle = {}
            for name, fit_ids, threshold_ids, use_spec in (
                ('candidate', ct, ch, spec), ('F0_matched', ct, ch, SPECS['F0_m0_calibrated']),
                ('F0_operational', train, threshold, SPECS['F0_m0_calibrated'])):
                bundle[name] = fit_selected(X, y, groups, uids, fit_ids, threshold_ids, use_spec, config, seed + fold * 17)
            bundle['test_window_uids'] = [uids[i] for i in test]
            bundle['test_source_groups'] = sorted({groups[i] for i in test})
            write_json(out / f'models/{family}/outer_fold_{fold}/models.json', bundle)
            statuses = {name: bundle[name]['status'] for name in ('candidate', 'F0_matched', 'F0_operational')}
            family_status[family].append(statuses)
            # Re-certify stored coefficients without modifying the source or calling it a refit.
            scaler, om = old['standardizer'], old['model']
            ox = (X[np.ix_(ct, cols)] - scaler['mean']) / scaler['scale']
            lower, upper = np.full(len(cols)+1, -np.inf), np.full(len(cols)+1, np.inf)
            for key, value in om.get('lower_bounds', {}).items(): lower[int(key)+1] = value
            for key, value in om.get('upper_bounds', {}).items(): upper[int(key)+1] = value
            w = group_weights([groups[i] for i in ct]); w /= w.sum()
            numeric_audit.append({"family": family, "fold": fold, "old_converged_flag": om['converged'],
                                  "posthoc_old_coefficients": certificate(np.r_[om['intercept'], om['coef']], ox, y[ct], w, om['regularization'], lower, upper,
                                                                          fit_context={"ordered_uids": [uids[i] for i in ct], "scaler": scaler})})
            for i in test:
                result = {"family": family, "fold": fold, "window_uid": uids[i], "source_group": groups[i], "y": int(y[i]),
                          "candidate_available": bool(available[i]), "raw_m0": float(X[i, 0])}
                for name in statuses:
                    usable = statuses[name] == 'CERTIFIED' and (available[i] or name == 'F0_operational')
                    p = float(predict(bundle[name], X[[i]])[0]) if usable else None
                    result[name] = p
                    result[name+'_pred'] = int(p >= bundle[name]['threshold']) if p is not None else None
                result['fallback_used'] = result['candidate'] is None
                result['whole_packet'] = result['F0_operational'] if result['fallback_used'] else result['candidate']
                result['whole_packet_pred'] = result['F0_operational_pred'] if result['fallback_used'] else result['candidate_pred']
                outputs.append(result)
            print(f'[v9.1] {family} fold={fold}: {statuses}', flush=True)
        old_final = read_json(ref / f'models/{family}/frozen_development_model.json')
        ft = [i for i in range(len(rows)) if available[i] and groups[i] in old_final['fit_source_groups']]
        fh = [i for i in range(len(rows)) if available[i] and groups[i] in old_final['threshold_source_groups']]
        final = fit_selected(X, y, groups, uids, ft, fh, spec, config, seed + 9002)
        final.update(development_only=True, human_labels_used=False)
        write_json(out / f'models/{family}/frozen_development_model.json', final)
        family_status[family].append({'frozen': final['status']})
        om, s = old_final['model'], old_final['standardizer']
        ox = (X[np.ix_(ft, cols)] - s['mean']) / s['scale']
        lo, hi = np.full(len(cols)+1, -np.inf), np.full(len(cols)+1, np.inf)
        for key,value in om.get('lower_bounds',{}).items(): lo[int(key)+1]=value
        for key,value in om.get('upper_bounds',{}).items(): hi[int(key)+1]=value
        w=group_weights([groups[i] for i in ft]); w/=w.sum()
        numeric_audit.append({'family':family,'fold':'frozen','old_converged_flag':om['converged'],
                              'posthoc_old_coefficients':certificate(np.r_[om['intercept'],om['coef']],ox,y[ft],w,om['regularization'],lo,hi,
                                                                      fit_context={'ordered_uids':[uids[i] for i in ft],'scaler':s})})
    write_jsonl(out / 'numeric_audit/old_coefficients_posthoc.jsonl', numeric_audit)
    write_jsonl(out / 'evaluation/oof_predictions.jsonl', outputs)
    reports, per_fold, paired = {}, [], []
    for family in family_status:
        subset = [r for r in outputs if r['family'] == family]
        common = [r for r in subset if r['candidate'] is not None and r['F0_matched'] is not None]
        metric = {name: metric_rows(common, name) for name in ('candidate', 'F0_matched', 'F0_operational')}
        valid = all(v == 'CERTIFIED' for fold in family_status[family] for v in fold.values())
        bs = [{"source_group": r['source_group'], "y": r['y'], "base": r['F0_matched'], "candidate": r['candidate'],
               "base_pred": r['F0_matched_pred'], "candidate_pred": r['candidate_pred']} for r in common]
        bootstrap = grouped_bootstrap_delta(bs, 'base', 'candidate', config.get('bootstrap_repetitions', 5000), seed)
        reports[family] = {"status": 'COMPLETE' if valid and len(subset) == len(rows) else 'INCOMPLETE_NUMERIC_EVALUATION',
                           "requested_n": len(rows), "available_n": sum(r['candidate_available'] for r in subset), "scored_common_n": len(common),
                           "pooled_oof": metric, "whole_packet_with_flagged_F0_fallback": metric_rows(subset, 'whole_packet'),
                           "fallback_n": sum(r['fallback_used'] for r in subset), "source_group_bootstrap": bootstrap,
                           "reliability": {name: reliability([r['y'] for r in common], [r[name] for r in common]) for name in metric}}
        for fold in sorted(set(folds.tolist())):
            fs = [r for r in common if r['fold'] == fold]
            ms = {name: metric_rows(fs, name) for name in metric}
            for name, value in ms.items(): per_fold.append({"family": family, "fold": fold, "method": name, **value})
            raw_ap = tie_aware_ap([r['y'] for r in fs], [r['raw_m0'] for r in fs])
            paired.append({"family": family, "fold": fold, "raw_m0_ap": raw_ap, "matched_f0_ap": ms['F0_matched']['ap'],
                           "ap_delta": ms['candidate']['ap']-ms['F0_matched']['ap'] if ms['candidate']['ap'] is not None and ms['F0_matched']['ap'] is not None else None})
        reports[family]['macro_fold_mean'] = {}
        for name in metric:
            reports[family]['macro_fold_mean'][name] = {}
            for key in ('ap','balanced_accuracy'):
                values=[r[key] for r in per_fold if r['family']==family and r['method']==name and r[key] is not None]
                reports[family]['macro_fold_mean'][name][key]=float(np.mean(values)) if values else None
    _write_csv(out / 'evaluation/per_fold_metrics.csv', per_fold)
    _write_csv(out / 'evaluation/paired_delta_by_fold.csv', paired)
    write_json(out / 'evaluation/scope_aligned_metrics.json', reports)
    write_json(out / 'models/fit_summary.json', family_status)
    unchanged = inventory(source) == before
    write_json(out / 'receipts/source_integrity.json', {"unchanged": unchanged, "source_files": len(before), "remote_calls": 0})
    if not unchanged:
        raise ValueError('source changed during replay; results invalid')
    gates = {}
    for family, report in reports.items():
        cert_ok = report['status'] == 'COMPLETE'
        coverage = report['scored_common_n']/report['available_n'] if report['available_n'] else 0
        ap_ci = report['source_group_bootstrap'].get('ap_delta', {}).get('ci95', [None, None])
        positive = ap_ci[0] is not None and ap_ci[0] > 0
        checks = [{"check": "numeric_certificates", "status": 'pass' if cert_ok else 'fail'},
                  {"check": "source_provenance", "status": 'pass' if unchanged else 'fail'},
                  {"check": "scope_alignment", "status": 'pass' if not scope_errors else 'fail', 'value': len(scope_errors)},
                  {"check": "feature_complete_cohort_coverage", "status": 'pass' if coverage>=minimum_coverage else 'insufficient', 'value':coverage, 'criterion':f'>={minimum_coverage}'},
                  {"check": "hard_normal_prospective_evidence", "status": 'insufficient', "criterion": "new source-disjoint trial required"},
                  {"check": "matched_AP_CI", "status": 'pass' if positive else 'fail', "value": ap_ci}]
        for c in checks:
            c['evidence_path'] = 'evaluation/scope_aligned_metrics.json' if c['check'] != 'source_provenance' else 'receipts/source_integrity.json'
            if c['check']=='scope_alignment': c['evidence_path']='receipts/scope_integrity.json'
            c['evidence_sha256'] = file_sha256(out / c['evidence_path'])
        gates[family] = {"status": 'INVALID_NUMERIC_OR_PROVENANCE' if not cert_ok or not unchanged else 'DEVELOPMENT_RANKING_CANDIDATE' if positive else 'NO_CANDIDATE',
                         "checks": checks, "deployment_authorized": False, "api_authorized": False}
        if cert_ok and unchanged and coverage<minimum_coverage:
            gates[family]['status']='INSUFFICIENT_COVERAGE'
        files=[{'path':p.relative_to(out).as_posix(),'sha256':file_sha256(p)} for p in sorted((out/'models'/family).rglob('*.json'))]
        evidence={'models_scalers_thresholds_certificates':files,'spec':SPECS[family], 'protocol':config,
                  'claim_policy':policy,'runtime_code_manifest':code_manifest,
                  'source_manifest_sha256':file_sha256(out/'archive/source_run_manifest.json'),
                  'label_sha256':file_sha256(ref/'labels/label_ledger.jsonl'),
                  'features':[r for r in before if r['path'].startswith('feature_store/')],
                  'split_sha256':file_sha256(ref/'splits/outer_inner_threshold_plan.json'),
                  'evaluation_sha256':file_sha256(out/'evaluation/scope_aligned_metrics.json'),
                  'oof_sha256':file_sha256(out/'evaluation/oof_predictions.jsonl')}
        write_json(out/f'claims/{family}_evidence_manifest.json',evidence)
        gates[family]['candidate_id']=semantic_sha256(evidence)
    write_json(out / 'claims/candidate_readiness.json', gates)
    return gates
