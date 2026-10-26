#!/usr/bin/env python3
"""Prepare a GPU0 queue, or execute the unchanged native judge after dependencies.

Only the coordinator invokes execute through run_official_job. prepare never
starts a process. Existing scores are reused only for byte-identical files.
"""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

from summarize_all_mega import catalog
from summarize_mega import read, require, sha


RUN_DIRS = {'primary_run': 'mega_dev16_seed0_gpu01',
            'additional_a0_run': 'mega_dev16_additional_gpu01_A0',
            'additional_a1_run': 'mega_dev16_additional_gpu016_A1',
            'attention_run': 'mega_dev16_attention_gpu01'}
PRIMARY_EVALUATION = 'mega_dev16_primary_gpu016_evaluation'


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def cost_budget(unique_methods, primary_report=None, judge_max_tokens=4096):
    primary_report = primary_report or {}
    elapsed = primary_report.get('elapsed_seconds')
    process = primary_report.get('judge_process_costs', {})
    peak = process.get('peak_allocated_gib')
    requests = unique_methods * 16 * 2 * 5
    return {'gpu_physical': 0, 'gpu_count': 1, 'planned_card': 'A100 80 GiB',
        'unique_new_method_files': unique_methods, 'cases_per_method': 16,
        'native_judge_samples': {'PCC': 5, 'Acc': 5},
        'successful_request_budget_before_parse_failures': requests,
        'transport_attempt_upper_bound': requests * 3,
        'max_generated_tokens_per_request': judge_max_tokens,
        'completion_token_upper_bound_without_retries': requests * judge_max_tokens,
        'completion_token_upper_bound_with_transport_retries': requests * 3 * judge_max_tokens,
        'forward_budget_note': 'Sequential local generation: at most one model forward per generated token per attempt; failures may stop earlier. This is a cap, not predicted actual work.',
        'planning_peak_allocated_gib_range': [20, 40], 'completed_primary_peak_allocated_gib': peak,
        'memory_note': '20–40 GiB is a planning allowance for one BF16 8B judge including KV/prefill; actual primary measurement is retained separately, not a worst-case proof.',
        'rough_wall_seconds_from_primary': elapsed * unique_methods / 11 if elapsed is not None else None,
        'rough_wall_formula': 'completed primary 11-method wall time * actual new unique methods / 11',
        'time_note': 'Planning extrapolation only; different path lengths, parse failures, retries and fixed loading costs affect runtime. No invented elapsed measurement.'}


def select_pending(methods, primary_config, primary_directory, native_directory):
    by_hash = defaultdict(list)
    hashes = primary_config['prediction_sha256']
    order = primary_config.get('method_order', list(hashes))
    require(len(hashes) == 11 and set(order) == set(hashes), 'Expected the actual completed primary 11-arm evaluation')
    for alias in order:
        by_hash[hashes[alias]].append(alias)
    mapping, unique, new_by_hash, reuse = {}, [], {}, []
    for item in methods:
        fingerprint = item['prediction_sha256']
        if fingerprint in by_hash:
            aliases = by_hash[fingerprint]
            alias = next((a for a in (item['name'], item['generation_method']) if a in aliases), aliases[0])
            target = {'evaluation_dir': str(primary_directory), 'evaluation_method': alias}
            origin = 'existing_primary_exact_file_hash'
        else:
            require(item['source_group'] != 'primary', 'A primary prediction changed after its completed evaluation')
            if fingerprint not in new_by_hash:
                new_by_hash[fingerprint] = item['name']
                unique.append(item)
                origin = 'new_native_evaluation'
            else:
                origin = 'same_exact_file_hash_as_new_representative'
            target = {'evaluation_dir': str(native_directory), 'evaluation_method': new_by_hash[fingerprint]}
        mapping[item['name']] = target
        reuse.append({'method': item['name'], 'prediction_sha256': fingerprint, **target, 'relationship': origin,
                      'fixed_denominator': 16,
                      'new_judge_requests': 'actual requests recorded by native evaluator' if origin == 'new_native_evaluation' else 0})
    require(len(mapping) == 43, 'Every frozen method must remain represented')
    return unique, mapping, reuse


def evaluator_arguments(args, primary_config, pending, output):
    judge = primary_config['judge']
    require(judge['backend'] == 'local', 'Continue the existing local Qwen judge track')
    require(judge['expected_results_per_metric'] == 5 and judge['temperature'] == .7 and judge['top_p'] == .8
            and judge['top_k'] == 20 and judge['thinking'] is False, 'Primary judge parameters differ from the native client contract')
    command = [str(args.reader_dir / 'mega_development_evaluation.py'), '--train-val-root', str(args.train_val_root),
        '--freeze-manifest', str(args.freeze_manifest), '--official-parser', str(args.official_parser),
        '--output-dir', str(output), '--backend', 'local', '--model', str(args.model),
        '--served-model-name', judge['served_name'], '--judge-max-tokens', str(judge['max_new_tokens']),
        '--judge-context-length', str(judge['context_length']), '--judge-seed', str(judge['local_rng_seed']),
        '--bootstrap-samples', str(primary_config['statistics']['bootstrap_samples'])]
    for item in pending:
        command += ['--prediction', item['name'] + '=' + item['prediction']]
    if len(pending) > 1:
        # Fixed catalog order, not chosen from scores. Final cross-run comparisons
        # are made later by summarize_all_mega with the original primary scores.
        command += ['--reference', pending[0]['name']]
    return command


def execute(args):
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=True)  # Outer runner owns run_manifest.json here.
    report_path = args.output_dir / 'remaining_evaluation_report.json'
    require(not report_path.exists(), 'Use a new wrapper output directory; preserve previous failures')
    started = time.monotonic()
    report = {'status': 'running', 'primary_evaluation_recomputed': False, 'new_scoring_rule': False,
              'gpu_physical': 0, 'fixed_methods': 43, 'fixed_cases_per_method': 16,
              'independent_model_claim': False, 'raw_clinical_text_published': False}
    code = 1
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Execute only inside the coordinator GPU0 runner')
        primary_dir = args.primary_evaluation_config.parent
        primary_config = read(args.primary_evaluation_config)
        primary_report = read(primary_dir / 'evaluation_report.json')
        require(primary_report['status'] == 'complete_verified', 'Primary native evaluation is not complete')
        require(primary_config['freeze_manifest_sha256'] == sha(args.freeze_manifest), 'Primary evaluation used another input batch')
        previous = read(args.primary_evaluation_manifest)
        current = read(args.output_dir / 'run_manifest.json')
        require(previous.get('process_state') == 'exited' and previous.get('exit_code') == 0, 'Primary judge process did not finish successfully')
        require(previous.get('model_files_sha256') and previous['model_files_sha256'] == current.get('model_files_sha256'),
                'Actual outer-run judge checkpoint fingerprints differ from primary')
        require(sha(args.model / 'config.json') == primary_config['judge']['model_config_sha256'], 'Judge model config differs')
        methods = catalog(args)
        native_output = args.output_dir / 'native_remaining'
        pending, mapping, reuse = select_pending(methods, primary_config, primary_dir, native_output)
        budget = cost_budget(len(pending), primary_report, primary_config['judge']['max_new_tokens'])
        write(args.output_dir / 'resource_budget.json', budget)
        write(args.output_dir / 'evaluation_map.json', mapping)
        write(args.output_dir / 'reuse_manifest.json', {'methods': reuse,
            'primary_config_sha256': sha(args.primary_evaluation_config),
            'primary_report_sha256': sha(primary_dir / 'evaluation_report.json'),
            'primary_run_manifest_sha256': sha(args.primary_evaluation_manifest),
            'rule': 'Only complete prediction-file byte SHA equality; no score, text similarity or favorable outcome selection',
            'cost_rule': 'Existing primary process cost remains counted once; each new unique file evaluated once; all 43 methods retain 16 cases'})
        report.update(unique_new_methods=len(pending), reused_method_slots=43 - len(pending), resource_budget=budget)
        if pending:
            command = evaluator_arguments(args, primary_config, pending, native_output)
            write(args.output_dir / 'native_evaluator_invocation.json', {'argv': command,
                'evaluator_sha256': sha(Path(command[0])),
                'client_sha256': sha(args.reader_dir / 'mega_local_qwen_client.py'),
                'reference_note': 'Within-new-output comparison reference is the first fixed catalog method; final comparisons reuse primary scores'})
            original_argv = sys.argv
            sys.path.insert(0, str(args.reader_dir))
            try:
                sys.argv = command
                try:
                    runpy.run_path(command[0], run_name='__main__')
                    result = 0
                except SystemExit as finished:
                    result = finished.code or 0
            finally:
                sys.argv = original_argv
            report['native_exit_code'] = result
            require(result == 0, 'Native evaluator failed; its full failure report and requests are preserved')
            native_report = read(native_output / 'evaluation_report.json')
            require(native_report['status'] == 'complete_verified', 'Native evaluator did not finish all selected methods')
            require(read(native_output / 'evaluation_config.json')['prediction_sha256'] ==
                    {item['name']: item['prediction_sha256'] for item in pending}, 'Native evaluated prediction fingerprints differ')
            report.update(native_report_sha256=sha(native_output / 'evaluation_report.json'),
                          native_process_costs=native_report.get('judge_process_costs'),
                          native_request_costs=native_report.get('judge_request_costs'))
        summary = [str(args.reader_dir / 'summarize_all_mega.py'), '--statistics-dir', str(args.reader_dir),
            '--freeze-manifest', str(args.freeze_manifest), '--primary-run', str(args.primary_run),
            '--additional-a0-run', str(args.additional_a0_run), '--additional-a1-run', str(args.additional_a1_run),
            '--attention-run', str(args.attention_run), '--evaluation-dir', str(primary_dir),
            '--evaluation-map', str(args.output_dir / 'evaluation_map.json'),
            '--cost-spec', str(args.reader_dir / 'clinical_all43_cost_spec.json'),
            '--output-dir', str(args.output_dir / 'all43_summary')]
        if pending:
            summary += ['--evaluation-dir', str(native_output)]
        write(args.output_dir / 'summary_invocation.json', {'python_argv': summary,
            'status': 'planned_CPU_only', 'note': 'Coordinator can run this after the native evaluation; not executed by this wrapper'})
        report.update(status='complete_verified', status_scope='Native remaining evaluation and exact-file reuse binding completed; final 43-method statistics remain a separate CPU step')
        code = 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=sha(Path(__file__)))
    write(report_path, report)
    print(json.dumps({'status': report['status'], 'primary_evaluation_recomputed': False}))
    return code


def prepare(args):
    root, runtime = args.experiment_root.resolve(), args.runtime_root.resolve()
    output = args.output_dir.resolve()
    require(args.gpu == 0, 'This follow-up queue is assigned to GPU0 only')
    worker_args = ['execute', '--reader-dir', str(root), '--freeze-manifest', str(args.freeze_manifest.resolve()),
        '--train-val-root', str(args.train_val_root.resolve()), '--official-parser', str(args.official_parser.resolve()),
        '--model', str(args.model.resolve()), '--primary-evaluation-config', str(runtime / PRIMARY_EVALUATION / 'evaluation_config.json'),
        '--primary-evaluation-manifest', str(args.primary_evaluation_manifest.resolve()), '--output-dir', str(output)]
    for attribute, folder in RUN_DIRS.items():
        worker_args += ['--' + attribute.replace('_', '-'), str(runtime / folder)]
    python_path = str(root) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    command = ['env', 'PYTHONPATH=' + python_path, str(args.python.resolve()), str(root / 'run_official_job.py'),
        '--gpu', '0', '--source', str(args.source.resolve()), '--model', str(args.model.resolve()),
        '--output', str(output), '--seed', '0', '--module', 'evaluate_remaining_mega', '--', *worker_args]
    primary_config_path = runtime / PRIMARY_EVALUATION / 'evaluation_config.json'
    primary_report_path = runtime / PRIMARY_EVALUATION / 'evaluation_report.json'
    primary_costs, maximum = None, 4096
    if primary_config_path.is_file():
        maximum = read(primary_config_path)['judge']['max_new_tokens']
    if primary_report_path.is_file():
        candidate = read(primary_report_path)
        if candidate.get('status') == 'complete_verified':
            primary_costs = {key: candidate.get(key) for key in ('elapsed_seconds', 'judge_process_costs')}
    plan = {'status': 'planned', 'gpu': 0, 'wait_for_pids': args.wait_for_pids,
        'scope': 'Remaining native evaluation only; wait for primary evaluation and attention generation parents',
        'resources': cost_budget(32, primary_costs, maximum), 'primary_evaluation_directory': str(runtime / PRIMARY_EVALUATION),
        'budget_parameter_source': 'Existing primary configuration when available; otherwise native default 4096, refreshed from the actual primary config before judging',
        'parameter_policy': 'At execution, copy actual primary native judge parameters exactly; no requested setting is independently changed',
        'jobs': [{'name': 'mega_dev16_remaining_native_evaluation_gpu0', 'command': command,
                  'log': str(output.with_name(output.name + '.log')), 'status': 'planned'}],
        'launch_command': [str(args.python.resolve()), str(root / 'run_baseline_queue.py'), str(args.plan.resolve())],
        'no_process_started': True, 'checker_sha256': sha(Path(__file__))}
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    write(args.plan, plan)
    print(json.dumps({'status': 'planned', 'gpu': 0, 'maximum_new_method_files': 32, 'no_process_started': True}))
    return 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    planning = sub.add_parser('prepare')
    for name in ('experiment-root', 'runtime-root', 'python', 'source', 'plan'):
        planning.add_argument('--' + name, type=Path, required=True)
    planning.add_argument('--gpu', type=int, choices=[0], default=0)
    planning.add_argument('--wait-for-pids', type=int, nargs='+', default=[428868, 428866])
    running = sub.add_parser('execute')
    for name in ('reader-dir', 'primary-run', 'additional-a0-run', 'additional-a1-run', 'attention-run', 'primary-evaluation-config'):
        running.add_argument('--' + name, type=Path, required=True)
    for child in (planning, running):
        for name in ('freeze-manifest', 'train-val-root', 'official-parser', 'model', 'primary-evaluation-manifest', 'output-dir'):
            child.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    return prepare(args) if args.mode == 'prepare' else execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
