"""Import explicit source-use attestations without changing the historical inventory."""
from collections import Counter


CHECKS = ('checked_fit_threshold_evaluation',
          'checked_discovery_prompt_design_human_review',
          'checked_external_history_and_source_aliases')


def validate_reviews(reviews, audited_sources):
    audited = {r['source_group']: r for r in audited_sources}
    expected = {g for g, row in audited.items() if row['audit_priority'].startswith('REVIEW_')}
    seen, releases, errors = set(), set(), []
    dispositions = {'pending', 'confirmed_used', 'not_used_after_audit', 'unresolved'}
    for row in reviews:
        group = row.get('source_group')
        if group in seen or group not in expected:
            errors.append({'source_group': group, 'error': 'DUPLICATE_OR_UNREQUESTED_GROUP'})
            continue
        seen.add(group)
        if row.get('audit_priority') != audited[group]['audit_priority']:
            errors.append({'source_group': group, 'error': 'AUDIT_IDENTITY_CHANGED'})
        if row.get('disposition') not in dispositions or row.get('eligible_for_enrollment') is not False:
            errors.append({'source_group': group, 'error': 'INVALID_DISPOSITION_OR_MANUAL_ENROLLMENT_OVERRIDE'})
        if row.get('disposition') == 'not_used_after_audit':
            if not all(row.get(k) is True for k in CHECKS) or not all(
                    isinstance(row.get(k), str) and row[k].strip() for k in ('reviewer_id', 'evidence_notes')):
                errors.append({'source_group': group, 'error': 'INCOMPLETE_NOT_USED_ATTESTATION'})
            else:
                releases.add(group)
    for group in sorted(expected - seen):
        errors.append({'source_group': group, 'error': 'MISSING_REVIEW'})
    if errors:
        return set(), errors
    return releases, []


def reconcile(reviews, old_audit_rows, fresh_audit_rows, inventory, scanned_files):
    requested, errors = validate_reviews(reviews, old_audit_rows)
    fresh = {r['source_group']: r for r in fresh_audit_rows}
    for group in sorted(requested):
        row = fresh.get(group)
        # This release path is intentionally narrower than arbitrary human overrides.
        if not row or set(row['evidence_kinds']) != {'planned_registration'}:
            errors.append({'source_group': group, 'error': 'FRESH_NONPLANNED_OR_MISSING_EVIDENCE'})
    released = requested if not errors else set()
    registered = set(inventory['source_groups'])
    for record in scanned_files:
        if 'source_groups_found' not in record:
            errors.append({'path': record.get('path'), 'error': 'FRESH_AUDIT_REQUIRED_FOR_ALL_CLASS_EXPOSURE'})
        registered.update(record.get('source_groups_found', []))
    if errors:
        released = set()
    history = {**inventory, 'version': 'history_exclusions_reviewed_v91_v1',
               'source_groups': sorted(registered - released),
               'released_source_groups': sorted(released),
               'history_completeness': 'LOCAL_FILES_PLUS_EXPLICIT_ATTESTATIONS_NOT_GLOBAL_PROOF',
               'issues': list(inventory.get('issues', [])) + errors,
               'remote_execution_authorized': False}
    report = {'version': 'v91_history_review_import_v1',
              'status': 'REVIEW_IMPORT_BLOCKED' if history['issues'] else 'READY_FOR_LOCAL_SCREENING',
              'reviews': len(reviews), 'dispositions': dict(Counter(r.get('disposition') for r in reviews)),
              'reviewer_ids': sorted({r.get('reviewer_id', '') for r in reviews}),
              'requested_releases': len(requested), 'released_sources': len(released),
              'registered_sources_before_release': len(registered),
              'remaining_excluded_sources': len(history['source_groups']),
              'errors': history['issues'], 'remote_calls': 0, 'remote_execution_authorized': False,
              'interpretation': 'Release from historical exclusion only, not enrollment, labels or deployment.'}
    return history, report
