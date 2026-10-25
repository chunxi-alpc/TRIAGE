#!/usr/bin/env python3
"""Server CPU audit of actual FP32 AdaSEKA standard CounterFact runs.

No model loads or inference. Efficacy: paper3/default1/zero, all pairs.
Paraphrase: paper3/default1 only; no unrun zero/nohook comparison is invented.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


NAMES = ('adaseka_paper3', 'adaseka_default1', 'adaseka_zero')
AMPLITUDES = dict(zip(NAMES, (3., 1., 0.)))
PIN = 'aab9cfed80dc012fb9e83f8bdefb8675c33f40e7'


def helpers(audit_dir=None, statistics_dir=None):
    root = Path(__file__).resolve().parent.parent
    for path in (audit_dir or root / 'independent_audit', statistics_dir or root / 'data_audit'):
        sys.path.insert(0, str(Path(path).resolve()))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def benchmark_order(arguments):
    for index, value in enumerate(arguments):
        if value in {'--benchmarks', '-b'}:
            result = []
            for candidate in arguments[index + 1:]:
                if candidate.startswith('-'):
                    break
                result.append(candidate)
            return result
        if value.startswith('--benchmarks='):
            return [value.split('=', 1)[1]]
    return ['efficacy', 'paraphrase']


def checked_export(folder, benchmark, expected_cases):
    from compare_standard import load_export, sha
    data = load_export(folder, benchmark, expected_cases)
    audit = read(Path(folder) / 'audit.json')
    by_case = defaultdict(list)
    for key, row in data['prompts'].items():
        target, comparator = float(row['target_score']), float(row['comparator_score'])
        require(math.isfinite(target) and math.isfinite(comparator) and max(target, comparator) <= 1e-6,
                'Nonfinite/non-log-probability exported score')
        win, margin = math.exp(target) > math.exp(comparator), math.exp(target) - math.exp(comparator)
        require(row['target_wins'] == win and math.isclose(row['probability_margin'], margin, abs_tol=1e-12),
                'Saved per-prompt efficacy differs from log probabilities')
        by_case[key[0]].append((float(win), margin))
    for sid, row in data['cases'].items():
        values = by_case[sid]
        require(len(values) == row['prompt_count'], 'Native prompt count differs')
        require(math.isclose(math.fsum(v[0] for v in values) / len(values), row['score'], abs_tol=1e-12),
                'Native per-case score differs')
        require(math.isclose(math.fsum(v[1] for v in values) / len(values), row['magnitude'], abs_tol=1e-12),
                'Native per-case probability margin differs')
    return data, audit


def load_run(name, folder, manifest_path, data_path, data_sha, expected_cases, trace_override=None):
    from compare_standard import read_trace, sha, tensor_key
    from compare_adaptive_counterfact import option
    manifest_path, folder = Path(manifest_path), Path(folder)
    manifest = read(manifest_path)
    require(manifest.get('process_state') == 'exited' and manifest.get('exit_code') == 0, name + ': run did not exit successfully')
    require(manifest.get('first_forward_parameter_dtype') == 'torch.float32', name + ': not observed FP32')
    arguments = manifest['module_args']
    require('--adaptive-seka' in arguments and not any(flag in arguments for flag in ('--seka', '--pasta', '--anchor')),
            name + ': this is not a standalone official AdaSEKA run')
    require(int(option(arguments, '--batch_size', '16')) == 1, name + ': official routing requires batch size 1')
    require(float(option(arguments, '--adaptive_amplify_factor', '1.0')) == AMPLITUDES[name], name + ': amplitude label differs')
    require(data_sha in manifest.get('input_files_sha256', {}).values(), name + ': entity data hash absent from actual run')
    require(manifest.get('source_commit') == PIN and manifest.get('source_files_sha256'), name + ': official source identity unavailable')
    require(manifest.get('model_files_sha256'), name + ': checkpoint fingerprints missing')
    order = benchmark_order(arguments)
    expected_order_set = {'efficacy'} if name == 'adaseka_zero' else {'efficacy', 'paraphrase'}
    require(len(order) == len(set(order)) and set(order) == expected_order_set, name + ': actual benchmark list differs')
    exports, audits = {}, {}
    for benchmark in order:
        exports[benchmark], audits[benchmark] = checked_export(folder / benchmark, benchmark, expected_cases)
        require(audits[benchmark]['run_manifest_sha256'] == sha(manifest_path), name + ': export/run manifest binding differs')
    expected_calls = 2 * sum(len(data['prompts']) for data in exports.values())
    require(manifest['forward_attempts'] == manifest['forward_completed'] == expected_calls, name + ': actual forward total differs')
    trace, trace_info = read_trace(str(manifest_path), str(trace_override) if trace_override else None)
    calls, patterns, offset = {}, {}, 0
    for benchmark in order:
        prompts = exports[benchmark]['prompts']
        assigned, issues, missing = {}, [], []
        for position, key in enumerate(prompts):
            if trace is None:
                assigned[key] = {'routing': None, 'generation': None}
                missing.append({'id': key[0], 'prompt_sha256': key[1], 'reason': 'Trace unavailable'})
                continue
            routed, generated = trace[offset + 2 * position:offset + 2 * position + 2]
            assigned[key] = {'routing': routed, 'generation': generated}
            if tensor_key(routed) is None or tensor_key(generated) is None:
                missing.append({'id': key[0], 'prompt_sha256': key[1], 'reason': 'Missing actual tensor fingerprint'})
                continue
            if routed['attention_mask'] is not None or generated['attention_mask'] is None:
                issues.append({'id': key[0], 'prompt_sha256': key[1], 'reason': 'Unexpected routing/generation attention-mask pattern'})
            if routed['input_ids'] != generated['input_ids']:
                issues.append({'id': key[0], 'prompt_sha256': key[1], 'reason': 'Route/generation input IDs differ'})
        calls[benchmark] = assigned
        patterns[benchmark] = {'status': 'failed' if issues else 'blocked' if missing else 'complete_verified',
            'issues': issues, 'missing_evidence': missing, 'routing_forwards': len(prompts), 'generation_forwards': len(prompts),
            'first_call_1based': offset + 1, 'last_call_1based': offset + 2 * len(prompts),
            'assignment': 'Native export prompt order, each prompt route then one-token generation; paraphrase has multiple prompts per case'}
        offset += 2 * len(prompts)
    source_root = Path(option(manifest['command'], '--source', '.')).resolve()
    expert_raw = option(arguments, '--adaptive-expert-path')
    require(expert_raw is not None, name + ': four-expert configuration absent')
    expert_config = (source_root / expert_raw).resolve()
    inputs = manifest['input_files_sha256']
    require(inputs.get(str(expert_config)) == sha(expert_config), name + ': actual expert configuration changed')
    experts = read(expert_config)
    require(isinstance(experts, dict) and len(experts) == 4, name + ': not four actual experts')
    expert_hashes = {}
    for expert, path in experts.items():
        actual_path = (source_root / path).resolve()
        actual_hash = sha(actual_path)
        require(inputs.get(str(actual_path)) == actual_hash, name + ': projection fingerprint absent or changed')
        expert_hashes[expert] = actual_hash
    settings = {key: option(arguments, flag, default) for key, flag, default in [
        ('layers', '--layers', 'last10'), ('top_k_singular', '--top_k_singular', '5'),
        ('combination_method', '--combination_method', 'weighted_top_k'),
        ('max_length', '--max_length', None), ('add_unmediated_fact', '--add_unmediated_fact', 'True'),
        ('marker_start', '--marker_start', '**'), ('marker_end', '--marker_end', None)]}
    settings.update(chat='--chat' in arguments, add_marker='--add_marker' in arguments,
                    attribute_no_entity='--attribute-no-entity' in arguments, effective_max_new_tokens=1)
    if settings['marker_end'] is None:
        settings['marker_end'] = settings['marker_start']
    return {'name': name, 'exports': exports, 'audits': audits, 'manifest': manifest,
            'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'calls': calls, 'patterns': patterns, 'trace_info': trace_info, 'settings': settings,
            'requested_max_new_tokens': option(arguments, '--max_new_tokens'),
            'requested_subset': option(arguments, '--example_subset'),
            'expert_hashes': expert_hashes, 'expert_config_sha256': sha(expert_config)}


def compare_prompt_calls(candidate, reference, benchmark):
    from compare_adaptive_counterfact import input_match
    from compare_standard import tensor_key
    current, prior = candidate['exports'][benchmark]['prompts'], reference['exports'][benchmark]['prompts']
    # Efficacy has one native prompt per case. For paraphrase, align the original
    # prompt SHA within each case, not its unstable set iteration index.
    current_keys = {key[0]: key for key in current} if benchmark == 'efficacy' else {key: key for key in current}
    prior_keys = {key[0]: key for key in prior} if benchmark == 'efficacy' else {key: key for key in prior}
    aligned, records = set(current_keys) == set(prior_keys), []
    for identity in sorted(set(current_keys) | set(prior_keys)):
        akey, bkey = current_keys.get(identity), prior_keys.get(identity)
        if akey is None or bkey is None:
            records.append({'id': (akey or bkey)[0], 'prompt_sha256': (akey or bkey)[1],
                            'status': 'blocked', 'reason': 'No exact native prompt partner'})
            continue
        a, b = current[akey], prior[bkey]
        ac, bc = candidate['calls'][benchmark][akey], reference['calls'][benchmark][bkey]
        route, generation = input_match(ac['routing'], bc['routing']), input_match(ac['generation'], bc['generation'])
        states = [route['status'], generation['status']]
        records.append({'id': akey[0], 'candidate_prompt_sha256': akey[1], 'reference_prompt_sha256': bkey[1],
            'status': 'failed' if 'failed' in states else 'blocked' if 'blocked' in states else 'complete_verified',
            'routing_input': route, 'generation_input': generation,
            'target_logp_difference': float(a['target_score']) - float(b['target_score']),
            'comparator_logp_difference': float(a['comparator_score']) - float(b['comparator_score']),
            'candidate_target_wins': a['target_wins'], 'reference_target_wins': b['target_wins']})
    states = [row['status'] for row in records]
    def ordered_keys(run):
        return [tensor_key(call) if call is not None else None for item in run['calls'][benchmark].values()
                for call in (item['routing'], item['generation'])]
    akeys, bkeys = ordered_keys(candidate), ordered_keys(reference)
    ordered_equal = None not in akeys and None not in bkeys and akeys == bkeys
    return {'status': 'failed' if 'failed' in states else 'blocked' if 'blocked' in states or not aligned else 'complete_verified',
        'native_prompt_keys_align': aligned, 'prompt_pairs': len(records),
        'matched_route_and_generation_pairs': states.count('complete_verified'),
        'ordered_full_segment_inputs_identical': ordered_equal,
        'alignment_scope': 'Matched actual tensor inputs per source-case prompt. Batch size 1; changed paraphrase prompt order is separately disclosed.',
        'full_vocabulary_logits_independently_recomputed': False}, records


def analyze(loaded, entity_mapping, output_dir, bootstrap_samples=10000):
    from compare_standard import compare_cases, sha
    from counterfact_entity_ci import bootstrap_effects
    base = loaded['adaseka_zero']
    for name, run in loaded.items():
        for field in ('model_files_sha256', 'source_commit', 'source_files_sha256'):
            require(run['manifest'][field] == base['manifest'][field], name + ': checkpoint/source differs')
        require(run['settings'] == base['settings'], name + ': frozen input or generation/routing setting differs')
        require(run['expert_hashes'] == base['expert_hashes'], name + ': expert bank differs')
        for data in run['exports'].values():
            require(set(data['cases']) == set(entity_mapping), name + ': standard 5000-case source coverage differs')
    ordered = sorted(entity_mapping, key=lambda sid: entity_mapping[sid]['absolute_processed_row_index'])
    entities = [entity_mapping[sid]['entity_cluster'] for sid in ordered]
    comparisons, all_states = {}, []
    for benchmark, pairs in [('efficacy', [('adaseka_paper3', 'adaseka_zero'), ('adaseka_default1', 'adaseka_zero'),
                                         ('adaseka_paper3', 'adaseka_default1')]),
                             ('paraphrase', [('adaseka_paper3', 'adaseka_default1')])]:
        columns, pending = [], []
        for candidate, reference in pairs:
            a, b = loaded[candidate]['exports'][benchmark], loaded[reference]['exports'][benchmark]
            summary, case_changes, _ = compare_cases(a, b, benchmark)
            inputs, prompt_changes = compare_prompt_calls(loaded[candidate], loaded[reference], benchmark)
            all_states.append(inputs['status'])
            delta = np.asarray([float(a['cases'][sid]['score']) - float(b['cases'][sid]['score']) for sid in ordered])
            improved, degraded = (delta > 0).astype(float), (delta < 0).astype(float)
            columns.extend([delta, improved, degraded])
            name = benchmark + '_' + candidate + '_vs_' + reference
            case_path, prompt_path = output_dir / (name + '_cases.jsonl'), output_dir / (name + '_inputs.jsonl')
            with case_path.open('x') as handle:
                for row in case_changes:
                    handle.write(json.dumps({**entity_mapping[str(row['id'])], **row}, sort_keys=True) + '\n')
            with prompt_path.open('x') as handle:
                for row in prompt_changes:
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
            summary.update(candidate=candidate, reference=reference, benchmark=benchmark,
                input_matching=inputs, per_case_sha256=sha(case_path), per_prompt_inputs_sha256=sha(prompt_path),
                native_score_definition='One probability-win indicator per efficacy case; paraphrase mean within case, then mean over cases',
                score_scope='Saved target/comparator log probabilities only; no full-vocabulary zero-drift claim')
            if benchmark == 'paraphrase':
                summary.update(corrected=None, regressed=None)
            comparisons[name] = summary
            pending.append(name)
        intervals = bootstrap_effects(np.asarray(columns).T, entities, bootstrap_samples, 0)
        for index, name in enumerate(pending):
            comparisons[name].update(ci95_score_difference=intervals[3 * index],
                ci95_score_difference_percentage_points=[100 * v for v in intervals[3 * index]],
                ci95_improvement_rate=intervals[3 * index + 1], ci95_degradation_rate=intervals[3 * index + 2])
    methods = {}
    for name, run in loaded.items():
        all_states.extend(item['status'] for item in run['patterns'].values())
        methods[name] = {'label': 'AdaSEKA author adaptive implementation; FP32; amplitude=' + str(AMPLITUDES[name]),
            'preset_scope': 'Paper strength 3/default CLI strength 1/zero strength 0; not a claim of whole-paper reproduction',
            'benchmarks': {k: v['summary'] for k, v in run['exports'].items()}, 'call_patterns': run['patterns'],
            'run_manifest_sha256': run['manifest_sha256'], 'trace': run['trace_info'], 'settings': run['settings'],
            'recorded_environment': {key: run['manifest'].get(key) for key in (
                'torch', 'transformers', 'datasets', 'tokenizers', 'requested_seed', 'python_hash_seed')},
            'requested_max_new_tokens': run.get('requested_max_new_tokens'), 'requested_subset': run.get('requested_subset'),
            'budget_note': 'Author efficacy wrapper forces one generated token, including paraphrase; full source-ID coverage verifies the actual standard slice',
            'expert_hashes': run['expert_hashes'], 'expert_config_sha256': run['expert_config_sha256'],
            'run_cost': {key: run['manifest'].get(key) for key in ('forward_attempts', 'forward_completed', 'elapsed_seconds',
                'peak_allocated_gib', 'peak_reserved_gib', 'forward_input_hashing_seconds', 'gpu_lock_wall_seconds')},
            'cost_scope': 'Whole run includes all listed benchmarks. Per-benchmark routing/generation counts are exact; time/memory cannot be partitioned from whole-run totals.'}
    sizes = Counter(entities)
    return {'status': 'failed' if 'failed' in all_states else 'blocked' if 'blocked' in all_states else 'complete_verified',
        'status_scope': 'Independent native score aggregation, entity intervals and actual route/generation input matching',
        'new_gpu_forwards': 0, 'clinical_sota_claim': False, 'all_negative_results_retained': True,
        'methods': methods, 'comparisons': comparisons, 'n_cases': len(ordered), 'n_entity_clusters': len(sizes),
        'repeated_entity_clusters': sum(size > 1 for size in sizes.values()), 'largest_entity_cluster': max(sizes.values()),
        'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': 0, 'numpy_version': np.__version__,
        'ci_scope': 'Marginal 95% whole-entity percentile bootstrap; case weighting; conditional on existing fixed runs; no multiplicity/significance or clinical claim',
        'checkpoint_files_sha256': base['manifest']['model_files_sha256'], 'source_commit': base['manifest']['source_commit'],
        'nohook_scope': 'FP32 nohook was verified only on 16 cases elsewhere; no 5000-case nohook or full-vocabulary zero-drift evidence is claimed here',
        'unavailable_comparisons': ['5000-case FP32 nohook', 'paraphrase versus matched zero'],
        'generation_scope': 'One existing greedy run per method; no best-seed selection or final multi-seed claim'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    for name in ('paper3', 'default1', 'zero'):
        parser.add_argument('--' + name + '-export-root', type=Path, required=True)
        parser.add_argument('--' + name + '-manifest', type=Path, required=True)
        parser.add_argument('--' + name + '-trace', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--statistics-dir', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    helpers(args.audit_dir, args.statistics_dir)
    from counterfact_entity_ci import source_clusters
    from compare_standard import sha
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        entities, source_sha = source_clusters(args.data)
        loaded = {name: load_run(name, getattr(args, suffix + '_export_root'), getattr(args, suffix + '_manifest'),
                                args.data, source_sha, 5000, getattr(args, suffix + '_trace'))
                  for name, suffix in zip(NAMES, ('paper3', 'default1', 'zero'))}
        report = analyze(loaded, entities, args.output_dir, args.bootstrap_samples)
        report.update(source_data_sha256=source_sha, standard_slice='train[5000:10000]')
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'new_gpu_forwards': 0}
    report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=sha(__file__),
        helper_sources_sha256={name: sha(sys.modules[name].__file__) for name in (
            'compare_standard', 'counterfact_entity_ci', 'compare_adaptive_counterfact') if name in sys.modules})
    with (args.output_dir / 'comparison.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'new_gpu_forwards': 0}))
    return 0 if report['status'] == 'complete_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
