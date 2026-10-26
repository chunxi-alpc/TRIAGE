"""Pre-test method/seed freeze. This module never opens test questions."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SEEDS = [0, 1, 2]
DOMAINS = {'data_construction', 'projection', 'head_selection', 'option_order', 'generation'}
REVISIONS = {'MEGA-CDP': '8f058c88ce1ed43693785b435ad0ff99907a2890',
             'MedPIC-Bench': '9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983'}
COUNTS = {'MEGA-CDP': 7458, 'MedPIC-Bench': 467}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def payload_digest(lock):
    payload = {k: v for k, v in lock.items() if k != 'lock_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_shape(lock):
    if lock.get('schema_version') != 'aster.confirmation.v1' or lock.get('trial_seeds') != SEEDS:
        raise ValueError('Declare the fixed confirmation trials 0/1/2')
    domains = lock.get('seed_domains', {})
    if set(domains) != DOMAINS:
        raise ValueError('Freeze data/projection/head/option/generation seed domains separately')
    for name, policy in domains.items():
        if set(policy.get('values_by_trial', {})) != {'0', '1', '2'} or not policy.get('meaning'):
            raise ValueError('Each seed domain requires all trials and its actual meaning: ' + name)
        if any(v is not None and (type(v) is not int or v < 0) for v in policy['values_by_trial'].values()):
            raise ValueError('Seed is a nonnegative integer, or null for an unused RNG')
    if not lock.get('methods') or len(set(lock['methods'])) != len(lock['methods']):
        raise ValueError('Freeze distinct methods and all intended comparisons')
    if not lock.get('primary_comparisons') or not lock.get('thresholds'):
        raise ValueError('Freeze primary comparisons and effect/noninferiority thresholds before testing')
    if (not lock.get('model_fingerprint') or not lock.get('model_files_sha256')
            or not lock.get('method_config') or not lock.get('source_files_sha256')):
        raise ValueError('Freeze actual checkpoint, method parameters and executable source files')
    if not lock.get('projection_files_sha256'):
        raise ValueError('Bind actual attention projection artifacts')
    for name, revision in REVISIONS.items():
        data = lock.get('datasets', {}).get(name, {})
        if data.get('revision') != revision or data.get('rows') != COUNTS[name] or not data.get('input_sha256'):
            raise ValueError('Bind the complete native dataset and file fingerprint: ' + name)
    if lock.get('seed_interpretation') != 'Report all trials; greedy duplicates are not independent replications':
        raise ValueError('Do not claim independent replication from duplicate greedy outputs')


def verify_frozen(lock, seed, dataset):
    # Pure validation precedes all dataset path resolution, opening and hashing.
    if lock.get('frozen') is not True or not lock.get('frozen_at_utc'):
        raise ValueError('Final method has not been frozen; do not open test questions')
    validate_shape(lock)
    if lock.get('lock_sha256') != payload_digest(lock):
        raise ValueError('Frozen method/configuration changed')
    if seed not in SEEDS or dataset not in REVISIONS:
        raise ValueError('Undeclared confirmation seed or dataset')


def verify_executable_files(lock):
    for field in ('source_files_sha256', 'projection_files_sha256'):
        for path, expected in lock[field].items():
            if digest_file(path) != expected:
                raise ValueError('File changed after method freeze: ' + str(path))


def freeze(spec, output):
    """Called by the coordinator only after development evaluation/power analysis.

    This is not a user-approval gate and never starts evaluation or changes seeds.
    """
    lock = deepcopy(spec)
    if lock.get('frozen'):
        raise ValueError('Use the existing freeze; never replace a frozen configuration')
    validate_shape(lock)
    verify_executable_files(lock)
    evidence_model = lock.get('model_manifest_evidence', {})
    if (not evidence_model.get('path')
            or digest_file(evidence_model['path']) != evidence_model.get('sha256')):
        raise ValueError('Bind the actual completed model-run manifest')
    model_run = json.loads(Path(evidence_model['path']).read_text())
    if (model_run.get('process_state') != 'exited' or model_run.get('exit_code') != 0
            or model_run.get('model_files_sha256') != lock['model_files_sha256']):
        raise ValueError('Frozen checkpoint files differ from the actual completed model run')
    evidence = lock.get('development_evidence', {})
    for key in ('summary', 'power_analysis'):
        item = evidence.get(key, {})
        if not item.get('path') or digest_file(item['path']) != item.get('sha256'):
            raise ValueError('Bind actual development evidence before testing: ' + key)
        report = json.loads(Path(item['path']).read_text())
        if report.get('status') != 'complete_verified':
            raise ValueError('Development evidence is incomplete: ' + key)
        if key == 'summary' and (not report.get('methods') or not report.get('comparisons') or not report.get('n_fixed_cases')):
            raise ValueError('An input audit or synthetic test cannot replace clinical development results')
        if key == 'power_analysis' and (
                report.get('report_type') != 'development_power_analysis'
                or report.get('source_development_summary_sha256') != evidence['summary']['sha256']
                or report.get('thresholds') != lock['thresholds'] or not report.get('assumptions')):
            raise ValueError('Power analysis must bind actual development scores, assumptions and frozen thresholds')
    lock.update(frozen=True, frozen_at_utc=datetime.now(timezone.utc).isoformat())
    lock['lock_sha256'] = payload_digest(lock)
    with Path(output).open('x') as stream:
        json.dump(lock, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return lock


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--draft', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = freeze(json.loads(args.draft.read_text()), args.output)
    print(json.dumps({'frozen':result['frozen'], 'lock_sha256':result['lock_sha256'],
                      'test_data_opened':False}))
