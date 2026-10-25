#!/usr/bin/env python3
"""Server CPU summary of completed native MEGA development evaluation artifacts.

No inference, judge requests, dataset text or reference-label reads. A JSON spec
names actual artifacts and comparisons; this script never selects a winning seed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(base, path):
    path = Path(path)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def measurement(record):
    """Do not add nested adapter elapsed time to the enclosing measured call."""
    completed = record.get('forward_completed', record.get('actual_forward_calls'))
    attempts = record.get('forward_attempts')
    seconds = record.get('wall_seconds', record.get('elapsed_seconds'))
    fields = {'forward_completed': completed, 'forward_attempts': attempts,
              'seconds': seconds, 'routing_forward_completed': record.get('routing_forward_completed'),
              'routing_forward_attempts': record.get('routing_forward_attempts'),
              'peak_allocated_gib': record.get('peak_allocated_gib'),
              'peak_reserved_gib': record.get('peak_reserved_gib')}
    for key, value in fields.items():
        require(value is None or number(value), 'Invalid recorded cost: ' + key)
    return fields


def total(events):
    values = list(events.values())
    result = {'unique_events': len(values)}
    for key in ('forward_completed', 'forward_attempts', 'seconds',
                'routing_forward_completed', 'routing_forward_attempts'):
        known = [event[key] for event in values if event[key] is not None]
        result[key + '_known'] = math.fsum(known)
        result[key + '_unknown_events'] = len(values) - len(known)
        result[key + '_total'] = math.fsum(known) if len(known) == len(values) else None
    for key in ('peak_allocated_gib', 'peak_reserved_gib'):
        known = [event[key] for event in values if event[key] is not None]
        result[key + '_max_recorded'] = max(known) if known else None
    result['memory_scope'] = 'Maximum recorded call peak; model-load peak may be absent after inner resets'
    return result


def event(path, index, record):
    return str(Path(path).resolve()) + '#' + str(index), measurement(record)


def cluster_mean(values, clusters, samples, seed=0):
    from paired_statistics import quantile
    sums, counts = defaultdict(list), Counter()
    for sid, value in values.items():
        sums[clusters[sid]].append(value)
        counts[clusters[sid]] += 1
    group_ids = sorted(sums)
    sums = {key: math.fsum(value) for key, value in sums.items()}
    result = {'mean': math.fsum(values.values()) / len(values), 'ci95': None,
              'n_cases': len(values), 'n_guideline_groups': len(group_ids),
              'bootstrap_samples': samples, 'bootstrap_seed': seed,
              'weighting': 'case mean; resample whole guideline groups with all their cases'}
    if len(group_ids) < 2:
        result.update(status='point_estimate_only', reason='Fewer than two guideline groups')
        return result
    rng, boot = random.Random(seed), []
    for _ in range(samples):
        chosen = rng.choices(group_ids, k=len(group_ids))
        boot.append(math.fsum(sums[group] for group in chosen) / sum(counts[group] for group in chosen))
    result.update(status='complete_verified', ci95=[quantile(boot, .025), quantile(boot, .975)])
    return result


def numerical(rows_for_method):
    diagnostics = [r.get('generation', {}).get('numerical_control') for r in rows_for_method]
    found = [d for d in diagnostics if d is not None]
    maxima = defaultdict(list)
    for item in found:
        for key, value in item.items():
            if key.startswith('max_abs_'):
                require(number(value), 'Invalid recorded numerical diagnostic')
                maxima[key].append(value)
    return {'rows': len(diagnostics), 'diagnostics_recorded': len(found),
            'diagnostics_missing': len(diagnostics) - len(found),
            'argmax_changed': sum(d.get('argmax_changed') is True for d in found),
            'same_full_input_true': sum(d.get('same_full_input') is True for d in found),
            'same_full_input_false': sum(d.get('same_full_input') is False for d in found),
            'same_full_input_not_recorded': sum('same_full_input' not in d for d in found),
            'maximum_recorded_differences': {key: max(value) for key, value in maxima.items()},
            'scope': 'Audit of saved scalar diagnostics only; full vocabulary scores were not retained for independent recomputation'}


class CostLedger:
    def __init__(self, primary, frozen_ids):
        self.primary, self.ids = primary, set(frozen_ids)
        self.stage_path = primary / 'stage_predictions.private.jsonl'
        self.stages = rows(self.stage_path)
        self.extractions = {}
        self.conditions = defaultdict(dict)
        self.condition_rules = defaultdict(set)
        self.execution_cache = {}
        self.actual = {}
        self.missing = []
        for index, row in enumerate(self.stages):
            require(row['sample_id'] in self.ids, 'Extra case in primary condition ledger')
            label = row['stage_configuration']
            if label == 'guideline_extraction':
                require(row['sample_id'] not in self.extractions, 'Duplicate case extraction reference')
                self.extractions[row['sample_id']] = row['cache_key']
            else:
                key, cost = event(self.stage_path, index, row.get('generation', row))
                self.actual[key] = cost
                if label in {'A0', 'A1'}:
                    rule_id = row.get('rule_id', row.get('prediction', {}).get('rule_id'))
                    require(rule_id not in self.condition_rules[(row['sample_id'], label)], 'Duplicate measured condition rule')
                    self.condition_rules[(row['sample_id'], label)].add(rule_id)
                    self.conditions[(row['sample_id'], label)][key] = cost
        for cache_key in set(self.extractions.values()):
            path = primary / 'private_guideline_extractions' / (cache_key + '.json')
            value = read(path)
            require(value['cache_key'] == cache_key, 'Wrong extraction cache')
            key, cost = event(path, 0, value['generation'])
            self.actual[key] = cost

    def load_execution(self, path):
        path = Path(path).resolve()
        if path not in self.execution_cache:
            lookup = {}
            for index, row in enumerate(rows(path)):
                sid = row['sample_id']
                require(sid in self.ids and sid not in lookup, 'Extra/duplicate generation case')
                lookup[sid] = (index, row)
                key, cost = event(path, index, row.get('generation', {}))
                self.actual[key] = cost
            self.execution_cache[path] = lookup
        return self.execution_cache[path]

    def execution(self, path, sid, active=None):
        path = Path(path).resolve()
        active = set() if active is None else active
        marker = (path, sid)
        require(marker not in active, 'Cyclic output reuse reference')
        active.add(marker)
        row_item = self.load_execution(path).get(sid)
        if row_item is None:
            return {}, [{'sample_id': sid, 'reason': 'missing_prediction_cost'}]
        index, row = row_item
        value = row.get('generation')
        if value is None:
            return {}, [{'sample_id': sid, 'reason': 'preparation_failure_or_missing_cost'}]
        reused = value.get('reused_from')
        if reused:
            if reused == 'primary A0B1':
                source = self.primary / 'A0B1.predictions.private.jsonl'
            elif reused in {'A1B0', 'A0B0'}:
                source = self.primary / (reused + '.predictions.private.jsonl')
            else:
                source = resolve(path.parent, reused)
            expected = value.get('reused_source_sha256', value.get('source_sha256'))
            require(expected is None or sha(source) == expected, 'Reused output source changed')
            return self.execution(source, sid, active)
        key, cost = event(path, index, value)
        return {key: cost}, []

    def attributed(self, path, condition):
        execution, conditions, extraction, missing = {}, {}, {}, []
        for sid in sorted(self.ids):
            events, failures = self.execution(path, sid)
            execution.update(events)
            missing.extend(failures)
            if condition is None:
                continue
            cache_key = self.extractions.get(sid)
            if cache_key is None:
                missing.append({'sample_id': sid, 'reason': 'missing_shared_extraction_reference'})
                continue
            cache_path = self.primary / 'private_guideline_extractions' / (cache_key + '.json')
            cached = read(cache_path)
            key, cost = event(cache_path, 0, cached['generation'])
            extraction[key] = cost
            conditions.update(self.conditions[(sid, condition)])
            expected_rules = {rule['rule_id'] for rule in cached.get('rules', [])}
            if expected_rules != self.condition_rules[(sid, condition)]:
                missing.append({'sample_id': sid, 'reason': 'incomplete_shared_condition_rule_costs'})
        combined = {**execution, **conditions, **extraction}
        # For all fixed cases, one cache cost per unique guideline is exactly the
        # sum of that cost amortized over all cases sharing the cache.
        return {'execution_including_reused_original': total(execution),
                'shared_condition': total(conditions), 'shared_extraction': total(extraction),
                'method_inference_total': total(combined), 'unresolved_costs': missing,
                'status': 'complete_from_recorded_events' if not missing else 'incomplete_cost_evidence',
                'amortization': 'Each extraction cache once per method over the full batch; condition once per case/rule',
                'method_totals_must_not_be_added_across_methods': True}


def condition_for(name, protocol, primary):
    if primary:
        if name == 'full_guideline':
            return None
        if name.startswith('A0') or name in {'text_emphasis', 'predicted_crop'}:
            return 'A0'
        if name.startswith('A1'):
            return 'A1'
        raise ValueError('Unknown primary method condition: ' + name)
    return protocol.get('condition_variant', 'A0')


def summarize(spec, spec_dir, bootstrap_samples=10000):
    require(bootstrap_samples >= 100, 'Use at least 100 bootstrap resamples')
    frozen_path = resolve(spec_dir, spec['freeze_manifest'])
    frozen = read(frozen_path)
    require(frozen['split'] == 'train_val', 'This summary is for train_val development only')
    clusters = {row['sample_id']: row['source_cluster'] for row in frozen['selected']}
    require(len(clusters) == len(frozen['selected']) > 0 and all(clusters.values()), 'Invalid frozen group mapping')
    primary = resolve(spec_dir, spec['primary_run'])
    primary_protocol = read(primary / 'protocol.json')
    require(primary_protocol['input_manifest_sha256'] == sha(frozen_path), 'Primary frozen input changed')
    require(primary_protocol['seed'] == 0, 'This development summary is for the actual seed 0 only')
    ledger = CostLedger(primary, clusters)
    methods, metric_rows, evaluation_jobs, generation_dirs = {}, [], {}, {primary}
    fingerprints = {str(frozen_path): sha(frozen_path), str(primary / 'protocol.json'): sha(primary / 'protocol.json'),
                    str(ledger.stage_path): sha(ledger.stage_path)}
    for item in spec['methods']:
        name = item['name']
        require(name not in methods, 'Duplicate public method name')
        path = resolve(spec_dir, item['prediction'])
        protocol_path = path.parent / 'protocol.json'
        protocol = read(protocol_path)
        generation_dirs.add(path.parent)
        require(protocol['input_manifest_sha256'] == sha(frozen_path), 'Generation used a different frozen input')
        require(protocol['model_fingerprint'] == primary_protocol['model_fingerprint'], 'Generation checkpoint differs')
        require(protocol.get('seed') == 0, 'Unexpected method seed')
        internal = item.get('generation_method', name)
        condition = condition_for(internal, protocol, path.parent == primary)
        require(item['condition'] == condition, 'Declared A0/A1 group disagrees with generation source')
        if path.parent != primary:
            require(protocol['primary_protocol_sha256'] == sha(primary / 'protocol.json'), 'Wrong primary cost/TFU source')
            stage_hash = protocol.get('primary_stages_sha256', protocol.get('primary_stage_sha256'))
            require(stage_hash == sha(ledger.stage_path), 'Primary condition output changed')
        raw_lookup = ledger.load_execution(path)
        raw = [value[1] for value in raw_lookup.values()]
        require(all(row['method'] == internal for row in raw), 'Generation method/file mapping differs')
        evaluation_dir = resolve(spec_dir, item['evaluation_dir'])
        evaluation_name = item.get('evaluation_method', name)
        config = read(evaluation_dir / 'evaluation_config.json')
        report = read(evaluation_dir / 'evaluation_report.json')
        require(report['status'] == 'complete_verified', 'Judge evaluation not completed; retain failures, do not synthesize a report')
        require(config['freeze_manifest_sha256'] == sha(frozen_path), 'Evaluation frozen input differs')
        require(config['prediction_sha256'][evaluation_name] == sha(path), 'Evaluated predictions changed')
        metric_path = evaluation_dir / 'case_metrics.jsonl'
        current = [row for row in rows(metric_path) if row['method'] == evaluation_name]
        require(len(current) == len(clusters) and {r['sample_id'] for r in current} == set(clusters), 'Incomplete/duplicate scored denominator')
        for row in current:
            require(row['tree_cluster'] == clusters[row['sample_id']] and row['method_seed'] == 0, 'Evaluation group/seed differs')
            for metric, valid, worst in [('acc_normalized', 'accuracy_valid', 0.), ('pcc_dtw_cost', 'pathway_valid', 1.)]:
                require(number(row[metric]) and row[metric] <= 1, 'Invalid native continuous metric')
                require(row[valid] or row[metric] == worst, 'Failed native metric not retained at fixed worst score')
            metric_rows.append({**row, 'method': name, 'seed': 0})
        evaluation_jobs[str(evaluation_dir)] = {'judge': config['judge'],
            'official_code_commit': config['official_code_commit'], 'official_source_files_sha256': config['official_source_files_sha256'],
            'judge_request_costs': report.get('judge_request_costs'), 'judge_process_costs': report.get('judge_process_costs'),
            'elapsed_seconds': report.get('elapsed_seconds'), 'scope': 'Evaluation costs only; not added to method inference'}
        metric_summary = {}
        for metric in ('acc_normalized', 'pcc_dtw_cost'):
            metric_summary[metric] = cluster_mean({r['sample_id']: r[metric] for r in current}, clusters, bootstrap_samples)
            metric_summary[metric]['higher_is_better'] = metric == 'acc_normalized'
        new_events = dict(event(path, index, row['generation']) for index, row in raw_lookup.values() if 'generation' in row)
        methods[name] = {'condition_outputs': condition, 'generation_method': internal,
            'n_fixed_cases': len(clusters), 'prediction_rows_present': len(raw), 'prediction_rows_missing': len(clusters) - len(raw),
            'generation_status_counts': dict(Counter(r.get('status', 'unrecorded') for r in raw)),
            'parse_failures': sum(not r['parsed_success'] for r in current),
            'accuracy_failures_scored_zero': sum(not r['accuracy_valid'] for r in current),
            'pathway_failures_scored_one': sum(not r['pathway_valid'] for r in current),
            'metrics': metric_summary, 'cost': ledger.attributed(path, condition),
            'new_execution_in_this_file': total(new_events), 'numerical_control': numerical(raw),
            'evaluation_directory': str(evaluation_dir), 'calibration_jobs': item['calibration_jobs'],
            'prediction_sha256': sha(path), 'protocol_sha256': sha(protocol_path)}
        dependencies = set()
        if condition == 'A1' or (path.parent == primary and internal in {'A0B1'} or internal == 'seka_execution_reuse'):
            dependencies.add('seka')
        if internal.startswith('pasta') and internal != 'pasta_no_hook':
            dependencies.add('pasta')
        if internal.startswith('adaseka') and internal != 'adaseka_no_hook':
            dependencies.add('adaseka')
        if internal.startswith('fusion'):
            dependencies.add('fusion')
        methods[name]['calibration_families_to_account_for'] = sorted(dependencies)
        for file in (path, protocol_path, metric_path, evaluation_dir / 'evaluation_config.json', evaluation_dir / 'evaluation_report.json'):
            fingerprints[str(file)] = sha(file)
    # Actual observed inference ledger includes controls and extra probes, not just
    # the methods chosen for comparison. Reuse rows contribute their true zero cost.
    for directory in generation_dirs:
        protocol = read(directory / 'protocol.json')
        for method in protocol.get('arms', protocol.get('methods', [])):
            path = directory / (method + '.predictions.private.jsonl')
            if path.exists():
                ledger.load_execution(path)
                fingerprints[str(path)] = sha(path)
            else:
                ledger.missing.append(str(path))
        for filename in ('numeric_probes.private.jsonl', 'additional_references.private.jsonl'):
            path = directory / filename
            if path.exists():
                fingerprints[str(path)] = sha(path)
                for index, row in enumerate(rows(path)):
                    key, cost = event(path, index, row.get('generation', row))
                    ledger.actual[key] = cost
    for file in {key.rsplit('#', 1)[0] for key in ledger.actual}:
        fingerprints[file] = sha(file)
    calibration, seen_calibration_manifests = {}, set()
    for name, definition in spec.get('calibration_jobs', {}).items():
        path = resolve(spec_dir, definition['run_manifest'])
        run = read(path)
        require(name not in calibration, 'Duplicate calibration job')
        require(path not in seen_calibration_manifests, 'Same physical calibration manifest registered twice')
        seen_calibration_manifests.add(path)
        calibration[name] = {'run_manifest_sha256': sha(path), 'process_state': run.get('process_state'),
            'exit_code': run.get('exit_code'), 'forward_attempts': run.get('forward_attempts'),
            'forward_completed': run.get('forward_completed'), 'elapsed_seconds': run.get('elapsed_seconds'),
            'gpu_memory': {k: v for k, v in run.items() if 'peak' in k and ('gib' in k or 'bytes' in k)},
            'scope': definition.get('scope', 'projection/head/calibration construction; once per artifact'),
            'family': definition['family'],
            'model_files_sha256': run.get('model_files_sha256'), 'source_commit': run.get('source_commit')}
        fingerprints[str(path)] = sha(path)
    for method in methods.values():
        require(isinstance(method['calibration_jobs'], list), 'Explicit calibration job attribution required, including [] when none')
        require(len(method['calibration_jobs']) == len(set(method['calibration_jobs'])), 'Duplicate attributed calibration job')
        for key in method['calibration_jobs']:
            require(key in calibration, 'Referenced calibration cost missing: ' + key)
        method['cost']['calibration'] = {key: calibration[key] for key in method['calibration_jobs']}
        method['cost']['calibration_scope'] = 'Separate one-time build costs; not added again to shared inference or across methods'
        available = {calibration[key]['family'] for key in method['calibration_jobs']}
        method['cost']['calibration_families_without_recorded_cost'] = sorted(set(method['calibration_families_to_account_for']) - available)
        if method['cost']['calibration_families_without_recorded_cost']:
            method['cost']['status'] = 'incomplete_cost_evidence'
    generation_processes = {}
    for item in spec.get('generation_processes', []):
        path = resolve(spec_dir, item['run_manifest'])
        require(str(path) not in generation_processes, 'Duplicate generation process manifest')
        run = read(path)
        generation_processes[str(path)] = {key: run.get(key) for key in (
            'process_state', 'exit_code', 'forward_attempts', 'forward_completed', 'elapsed_seconds',
            'gpu_lock_wall_seconds', 'forward_input_hashing_seconds', 'peak_allocated_gib', 'peak_reserved_gib')}
        fingerprints[str(path)] = sha(path)
    from paired_statistics import paired_cluster_comparison
    comparisons = []
    for pair in spec.get('comparisons', []):
        name, reference = pair['method'], pair['reference']
        require(name in methods and reference in methods and name != reference, 'Unknown/self comparison')
        current = {r['sample_id']: r for r in metric_rows if r['method'] == name}
        prior = {r['sample_id']: r for r in metric_rows if r['method'] == reference}
        judge_a = evaluation_jobs[methods[name]['evaluation_directory']]
        judge_b = evaluation_jobs[methods[reference]['evaluation_directory']]
        judge_match = all(judge_a[k] == judge_b[k] for k in ('judge', 'official_code_commit', 'official_source_files_sha256'))
        for metric in ('acc_normalized', 'pcc_dtw_cost'):
            result = paired_cluster_comparison(metric_rows, name, reference, metric, clusters, [0],
                                               bootstrap_samples=bootstrap_samples, random_seed=0)
            # These are descriptive development comparisons, not a preregistered
            # confirmatory family. Do not imply a significance claim from p-values.
            for key in ('p_value', 'test', 'permutations', 'alternative'):
                result.pop(key, None)
            benefits = [(current[sid][metric] - prior[sid][metric]) * (1 if metric == 'acc_normalized' else -1) for sid in clusters]
            result.update(higher_is_better=metric == 'acc_normalized',
                score_improved=sum(d > 1e-12 for d in benefits), score_degraded=sum(d < -1e-12 for d in benefits),
                score_unchanged=sum(abs(d) <= 1e-12 for d in benefits), binary_corrected=None, binary_regressed=None,
                same_condition_outputs=methods[name]['condition_outputs'] == methods[reference]['condition_outputs'],
                method_condition=methods[name]['condition_outputs'], reference_condition=methods[reference]['condition_outputs'],
                same_recorded_judge_configuration=judge_match,
                interpretation='Continuous native scores; marginal group-bootstrap CI; no confirmatory significance or independent-model claim')
            if not judge_match:
                result['comparison_limitation'] = 'Different recorded judge settings; descriptive difference is not a matched judge comparison'
            comparisons.append(result)
    stage_numerics = defaultdict(list)
    for row in ledger.stages:
        if row.get('generation', {}).get('numerical_control'):
            stage_numerics[row['stage_configuration']].append(row)
    return {'status': 'complete_verified', 'status_scope': 'Independent CPU aggregation of saved continuous scores and cost records only',
        'scope': 'MEGA train_val development, actual method seed 0; not frozen final seeds 0/1/2 evaluation',
        'n_fixed_cases': len(clusters), 'n_guideline_groups': len(set(clusters.values())),
        'independent_model_claim': False, 'judge_limitation': 'Local Qwen judge differs from paper GPT-5.4 Mini; may share generation checkpoint',
        'no_new_inference': True, 'no_test_or_reference_text_read': True, 'raw_text_published': False,
        'methods': methods, 'comparisons': comparisons, 'condition_zero_diagnostics': {k: numerical(v) for k, v in stage_numerics.items()},
        'actual_observed_generation_ledger': total(ledger.actual), 'generation_files_missing': ledger.missing,
        'calibration_jobs_counted_once': calibration, 'evaluation_jobs_counted_once': evaluation_jobs,
        'generation_process_manifests': generation_processes,
        'generation_process_scope': 'Optional complete-process wall times/counters; includes loading/hash overhead, overlaps call ledger and must not be added to it',
        'cost_note': 'Method attributed totals include reused originals and shared work; actual ledger counts physical calls once. These are alternative views, never add them. Unknown attempts/costs remain null; process overhead and model load are not inferred.',
        'input_files_sha256': fingerprints, 'statistical_source_sha256': sha(Path(sys.modules['paired_statistics'].__file__)),
        'checker_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--statistics-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data_audit')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.statistics_dir.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = summarize(read(args.spec), args.spec.resolve().parent, args.bootstrap_samples)
        report['spec_sha256'] = sha(args.spec)
        code = 0
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                  'no_new_inference': True, 'checker_sha256': sha(__file__), 'spec_sha256': sha(args.spec)}
        code = 1
    with (args.output_dir / 'summary.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'no_new_inference': True}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
