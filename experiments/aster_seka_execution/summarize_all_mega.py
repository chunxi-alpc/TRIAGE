#!/usr/bin/env python3
"""One CPU entrypoint for all 43 frozen MEGA development methods.

This discovers method names from generation protocols and judge aliases from
prediction SHA mappings. It never runs a judge or reads raw evaluation text.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from summarize_mega import (calibration_families, condition_for, read, require,
                            resolve, sha, summarize)


GROUPS = (('primary', 'primary_run', 11), ('extra_A0', 'additional_a0_run', 18),
          ('extra_A1', 'additional_a1_run', 6), ('attention_A0', 'attention_run', 8))


def catalog(args):
    frozen = read(args.freeze_manifest)
    require(frozen.get('split') == 'train_val' and len(frozen['selected']) == 16,
            'The unified entrypoint requires the fixed 16-case train_val batch')
    result = []
    for prefix, attribute, expected_count in GROUPS:
        directory = Path(getattr(args, attribute)).resolve()
        protocol = read(directory / 'protocol.json')
        require(protocol.get('input_manifest_sha256') == sha(args.freeze_manifest), prefix + ': frozen input differs')
        require(protocol.get('expected_cases') == 16, prefix + ': declared denominator differs')
        require(protocol.get('execution_status') in {'finished', 'failed'},
                prefix + ': generation is still running; retain partial failure artifacts')
        methods = protocol.get('arms') if prefix == 'primary' else protocol.get('methods')
        require(isinstance(methods, list) and len(methods) == len(set(methods)) == expected_count,
                prefix + ': expected method count differs; do not silently drop an arm')
        if prefix.startswith('extra_'):
            require(protocol.get('condition_variant', 'A0') == prefix.removeprefix('extra_'), prefix + ': condition branch differs')
        for method in methods:
            path = directory / (method + '.predictions.private.jsonl')
            require(path.is_file(), prefix + ': missing prediction file ' + method)
            condition = condition_for(method, protocol, prefix == 'primary')
            result.append({'name': prefix + '__' + method, 'generation_method': method,
                           'prediction': str(path), 'condition': condition,
                           'source_group': prefix, 'prediction_sha256': sha(path)})
    require(len(result) == 43, 'Expected all 43 method arms')
    return result


def comparison_plan(methods):
    names = {item['name'] for item in methods}
    pairs = []
    def add(candidate, reference):
        if candidate in names and reference in names and candidate != reference:
            pair = {'method': candidate, 'reference': reference}
            if pair not in pairs:
                pairs.append(pair)
    # Every arm is retained relative to its full-information, same-TFU B0.
    for item in methods:
        condition = item['condition']
        reference = 'primary__' + condition + 'B0' if condition else 'primary__A0B0'
        add(item['name'], reference)
        if item['source_group'].startswith('extra_'):
            add('primary__' + condition + 'B1', item['name'])
    for arm in ('A0B0', 'A0B1', 'A1B0', 'A1B1'):
        add('primary__' + arm, 'primary__full_guideline')
    add('primary__A1B0', 'primary__A0B0')
    add('primary__A1B1', 'primary__A0B1')
    for method in ('pasta', 'pasta_zero'):
        add('attention_A0__' + method, 'attention_A0__pasta_no_hook')
    add('attention_A0__pasta', 'attention_A0__pasta_zero')
    for method in ('adaseka_zero', 'adaseka_05', 'adaseka_3'):
        add('attention_A0__' + method, 'attention_A0__adaseka_no_hook')
    for method in ('adaseka_05', 'adaseka_3'):
        add('attention_A0__' + method, 'attention_A0__adaseka_zero')
    add('attention_A0__seka_execution_reuse', 'primary__A0B1')
    return pairs


def build_spec(args, methods):
    available = defaultdict(list)
    for directory in args.evaluation_dir:
        directory = Path(directory).resolve()
        config = read(directory / 'evaluation_config.json')
        require(config['freeze_manifest_sha256'] == sha(args.freeze_manifest), 'Judge input manifest differs')
        require(read(directory / 'evaluation_report.json')['status'] == 'complete_verified',
                'A supplied evaluation directory is incomplete; do not treat missing judged cases as success')
        for alias, fingerprint in config['prediction_sha256'].items():
            value = {'evaluation_dir': str(directory), 'evaluation_method': alias}
            if value not in available[fingerprint]:
                available[fingerprint].append(value)
    explicit = read(args.evaluation_map) if args.evaluation_map else {}
    costs = read(args.cost_spec) if args.cost_spec else {}
    cost_base = args.cost_spec.resolve().parent if args.cost_spec else Path.cwd()
    calibration = {}
    for name, item in costs.get('calibration_jobs', {}).items():
        calibration[name] = {**item, 'run_manifest': str(resolve(cost_base, item['run_manifest']))}
    selected = []
    for item in methods:
        matches = available[item['prediction_sha256']]
        if item['name'] in explicit:
            selection = explicit[item['name']]
            directory = resolve(args.evaluation_map.resolve().parent, selection['evaluation_dir'])
            selected_eval = {'evaluation_dir': str(directory), 'evaluation_method': selection['evaluation_method']}
            require(selected_eval in matches, 'Explicit judge mapping is not an exact evaluated prediction hash: ' + item['name'])
        else:
            named = [match for match in matches if match['evaluation_method'] == item['name']]
            require(len(matches) == 1 or len(named) == 1,
                    'Need one exact evaluated prediction mapping (or explicit --evaluation-map): ' + item['name'])
            selected_eval = named[0] if len(named) == 1 else matches[0]
        families = calibration_families(item['generation_method'], item['condition'], item['source_group'] == 'primary')
        attributed = costs.get('method_calibration_jobs', {}).get(item['name'])
        if attributed is None:
            attributed = list(dict.fromkeys(job for family in families for job in costs.get('family_jobs', {}).get(family, [])))
        selected.append({key: value for key, value in {**item, **selected_eval, 'calibration_jobs': attributed}.items()
                         if key not in {'source_group', 'prediction_sha256'}})
    return {'freeze_manifest': str(args.freeze_manifest.resolve()), 'primary_run': str(args.primary_run.resolve()),
            'methods': selected, 'comparisons': comparison_plan(methods), 'calibration_jobs': calibration,
            'generation_processes': [{**item, 'run_manifest': str(resolve(cost_base, item['run_manifest']))}
                                     for item in costs.get('generation_processes', [])]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze-manifest', type=Path, required=True)
    for _, name, _ in GROUPS:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, action='append', default=[])
    parser.add_argument('--evaluation-map', type=Path,
                        help='Optional explicit choice of existing judge alias, before reading any scores')
    parser.add_argument('--cost-spec', type=Path,
                        help='Actual calibration jobs, family_jobs/method_calibration_jobs and generation manifests')
    parser.add_argument('--prepare-evaluation-map-only', action='store_true',
                        help='Write all 43 evaluator prediction arguments; never execute evaluation')
    parser.add_argument('--statistics-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data_audit')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.statistics_dir.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        methods = catalog(args)
        mapping = {item['name']: item['prediction'] for item in methods}
        with (args.output_dir / 'evaluation_predictions.json').open('x') as handle:
            json.dump({'prediction_mapping': mapping, 'argument_vector': [part for name, path in mapping.items()
                for part in ('--prediction', name + '=' + path)], 'reference': 'primary__full_guideline',
                'expected_methods': 43, 'fixed_cases_per_method': 16, 'no_judge_started': True}, handle, indent=2)
            handle.write('\n')
        if args.prepare_evaluation_map_only:
            report = {'status': 'planned', 'expected_methods': 43, 'fixed_cases_per_method': 16,
                      'new_gpu_forwards': 0, 'no_judge_started': True,
                      'scope': 'Input mapping prepared only; no clinical scores read or computed'}
        else:
            require(args.evaluation_dir, 'Supply completed native evaluator directories')
            spec = build_spec(args, methods)
            spec_path = args.output_dir / 'summary_spec.json'
            with spec_path.open('x') as handle:
                json.dump(spec, handle, indent=2)
                handle.write('\n')
            report = summarize(spec, args.output_dir.resolve(), args.bootstrap_samples)
            report.update(spec_sha256=sha(spec_path), expected_methods=43, fixed_method_case_pairs=688,
                source_groups={prefix: count for prefix, _, count in GROUPS},
                no_evaluation_or_method_rerun=True,
                repeated_output_note='SEKA execution reuse is an existing A0B1 output, not an additional independent method result; separately sampled judge scores can differ from judge randomness')
        code = 0
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                  'new_gpu_forwards': 0, 'no_judge_started': True}
        code = 1
    report['entrypoint_sha256'] = sha(Path(__file__))
    with (args.output_dir / 'summary.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'new_gpu_forwards': 0}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
