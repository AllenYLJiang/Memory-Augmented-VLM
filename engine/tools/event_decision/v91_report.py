"""Generate factual results in docs, never mix human audit into model selection."""
from pathlib import Path

from .contracts import file_sha256, iter_jsonl, read_json


def report(out, destination):
    out,destination=Path(out),Path(destination)
    evaluation=read_json(out/'evaluation/scope_aligned_metrics.json')
    audit=read_json(out/'audit_only/human_challenge_metrics.json')
    checks=read_json(out/'claims/candidate_readiness.json')
    posthoc=list(iter_jsonl(out/'numeric_audit/old_coefficients_posthoc.jsonl'))
    lines=['# Governed V9.1 Numeric Replay and Audit Results','',
           'Scope: unchanged, selected training windows; nested source-group development OOF. Not full XD-Violence test AP.',
           f'Run: `{out.resolve()}`','',
           '## Development Metrics','']
    for family,record in evaluation.items():
        lines += [f'### {family}', '', '| Family/cohort | Method | N | AP | AUC | BA | FP | FN |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for name,m in record['pooled_oof'].items():
            lines.append(f"| {family} | {name} | {m['n']} | {m['ap']:.8f} | {m['auc']:.8f} | {m['balanced_accuracy']:.8f} | {m['fp']} | {m['fn']} |")
        whole=record['whole_packet_with_flagged_F0_fallback']
        lines+=['',f"{family}: whole packet AP={whole['ap']:.8f}, BA={whole['balanced_accuracy']:.8f}, FP={whole['fp']}, FN={whole['fn']}; flagged fallback={record['fallback_n']}.",
                f"Matched-control AP-delta bootstrap 95% CI: {record['source_group_bootstrap'].get('ap_delta',{}).get('ci95')}. Gate: `{checks[family]['status']}`.",'']
    lines+=['## Numerical Audit','',
            'Stable logaddexp objective, normalized inverse-source weights, unregularized intercept, unchanged sign bounds and lambda grid.',
            'The certificate requires projected-gradient and KKT infinity norms <= 1e-7 and primal violation <= 1e-10.',
            'All inner lambda/fold certificates and exact fitting UIDs are stored in each model bundle. Invalid fits cannot win lambda selection.','',
            '| Family | Stored fit | Old converged | Posthoc PG infinity norm | Certified |',
            '|---|---|---|---:|---|']
    for r in posthoc:
        c=r['posthoc_old_coefficients']
        lines.append(f"| {r['family']} | {r['fold']} | {r['old_converged_flag']} | {c['projected_gradient_inf']:.10g} | {c['certified']} |")
    lines+=['','## Human Audit Only','',
            'Choose the lexicographically first outer model whose fitting AND threshold groups exclude the reviewed source. Missing features and uncertain labels remain excluded with their counts retained.',
            'These labels are never used for fitting, regularization, thresholding, family selection or readiness.',
            '| Family | Method | Binary common N | AP | BA | FP | FN |',
            '|---|---|---:|---:|---:|---:|---:|']
    for family,record in audit['families'].items():
        for name,m in record['metrics'].items():
            lines.append(f"| {family} | {name} | {m['n']} | {m['ap']:.8f} | {m['balanced_accuracy']:.8f} | {m['fp']} | {m['fn']} |")
    records=list(iter_jsonl(out/'audit_only/human_challenge_predictions.jsonl'))
    changed=[r for r in records if r.get('numerical_repair_changed_binary_decision')]
    lines+=['',f'Binary changes from the historical coefficients on identical source-excluded audit folds: {len(changed)}.',
            'A high weak-label development AP does not establish hard-normal specificity. Matched F0 and operational F0 use different fitting populations and thresholds; both must be reported.',
            'Per-fold ranking comparisons are meaningful within a single calibrated M0 model. Pooled OOF probabilities from different fold calibrators need not preserve global raw-M0 ordering.',
            '', '## Provenance and Status','',
            f"Source integrity: {read_json(out/'receipts/source_integrity.json')}",
            f"Historical external seal: {read_json(out/'receipts/external_seal_integrity.json')}",
            'No DashScope/DeepSeek calls, graph discovery, graph activation, new human answers, or locked-trial evaluation were performed.',
            'Next: local-only fixed-trial preflight, followed by inventory/duplicate/quota review. Paid acquisition and locked evaluation are separate implementation stages, not authorized by this report.',
            '', '## Result Files','']
    for relative in ('evaluation/scope_aligned_metrics.json','evaluation/per_fold_metrics.csv','evaluation/paired_delta_by_fold.csv',
                     'evaluation/oof_predictions.jsonl','audit_only/human_challenge_predictions.jsonl',
                     'numeric_audit/old_coefficients_posthoc.jsonl','claims/candidate_readiness.json','archive/source_files.sha256.jsonl'):
        path=out/relative
        lines.append(f'- [{relative}]({path.resolve().as_posix()}), SHA-256 `{file_sha256(path)}`')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return destination
