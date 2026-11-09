#!/usr/bin/env python3
"""Strict 5-backbone x 3-dataset x 16-baseline evidence and execution harness.
No model or baseline is implicitly simulated; unimplemented providers are blocked.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPEC = BASE / 'matrix_spec.json'
STATUS = BASE / 'matrix_status_285.csv'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def make_jobs(spec):
    return [dict(model_id=m['key'], dataset_id=d['id'], method_id=b['id'],
                 family=b['family'], seed=11,
                 run_id='__'.join([m['key'], d['id'], b['id'], 'seed11']))
            for m in spec['models'] for d in spec['datasets'] for b in spec['methods']]


def verify_grid(spec, path=STATUS):
    jobs = make_jobs(spec)
    expected = {x['run_id'] for x in jobs}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    observed = [x['run_id'] for x in rows]
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError('Status grid must contain every distinct model/dataset/method exactly once')
    if len([x for x in jobs if x['family'] not in ('P',)]) != 240 or len(jobs) != 285:
        raise ValueError('Invalid factorial cardinality')
    if not all(x['status'] == 'not_completed_in_common_protocol' for x in rows):
        raise ValueError('Committed status inventory is historical and not a results database')
    return jobs


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def verify_predictions(record, expected, primary_correct):
    """Recount official cases from archived per-case predictions, never from a claimed scalar alone."""
    required = ('predictions_file', 'predictions_sha256')
    if any(not record.get(k) for k in required):
        raise ValueError('Per-case predictions file and SHA-256 are required')
    p = Path(record['predictions_file'])
    if not p.is_file() or file_digest(p) != record['predictions_sha256']:
        raise ValueError('Per-case predictions missing or SHA-256 mismatch')
    seen = set()
    n = correct = 0
    with p.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if 'case_id' not in row or 'gold' not in row or 'prediction' not in row:
                raise ValueError('Per-case records require case_id/gold/prediction')
            if row['case_id'] in seen:
                raise ValueError('Duplicate case_id in predictions')
            seen.add(row['case_id']); n += 1
            prediction, gold = row['prediction'], row['gold']
            if isinstance(prediction, list) and isinstance(gold, list):
                if record['dataset_id'] == 'medpic':
                    equal = sorted(prediction) == sorted(gold)
                else:
                    equal = prediction == gold
            else:
                equal = type(prediction) is type(gold) and prediction == gold
            correct += int(equal)
    if n != expected or correct != primary_correct:
        raise ValueError(f'Per-case recount disagrees: n={n}, correct={correct}')


def validate_result(record, spec, job):
    """Validate a newly produced actual run; no invented defaults for missing metrics."""
    for key in ('model_id', 'dataset_id', 'method_id', 'run_id'):
        if record.get(key) != job[key]:
            raise ValueError(f'{key}: mismatching run identity')
    if record.get('status') != 'completed' or record.get('provenance') != 'real_model_execution':
        raise ValueError('A real executed run and explicit provenance are required')
    for field in ('model_revision', 'code_revision', 'input_manifest_sha256',
                  'fit_source_id', 'dev_source_id', 'score_protocol', 'metrics'):
        if not record.get(field):
            raise ValueError(f'Missing reproducibility field {field}')
    if not isinstance(record.get('seed'), int):
        raise ValueError('Seed must be an integer')
    data = next(d for d in spec['datasets'] if d['id'] == job['dataset_id'])
    metrics = record['metrics']
    for key in data['required_metrics']:
        if key not in metrics or metrics[key] is None:
            raise ValueError(f'Missing required {data["id"]} metric {key}')
    n = data['n']
    if metrics['primary_total'] != n:
        raise ValueError(f'Wrong official denominator: expected {n}')
    for k in ('primary_correct', 'primary_total', 'option_correct', 'option_total',
              'flipping_pair_both_correct', 'flipping_pair_total',
              'preserving_consistency_correct', 'preserving_consistency_total',
              'conditional_faithful_correct', 'conditional_faithful_eligible',
              'evidence_pair_both_correct', 'evidence_pair_total'):
        if k in metrics and (type(metrics[k]) is not int or metrics[k] < 0):
            raise ValueError(f'{k} must be a nonnegative integer')
    if not 0 <= metrics['primary_correct'] <= n:
        raise ValueError('Invalid primary numerator')
    verify_predictions(record, n, metrics['primary_correct'])
    for k in ('interference_rms', 'evidence_correction_rms'):
        if k in metrics and (not isinstance(metrics[k], (float, int)) or metrics[k] < 0):
            raise ValueError(f'Invalid nonnegative RMS: {k}')
    if data['id'] == 'mega_cdp':
        if metrics['evidence_pair_total'] != 40 or metrics['evidence_pair_both_correct'] > 40:
            raise ValueError('MEGA controlled evidence denominator must be 40')
    elif data['id'] == 'nli4ct_2024':
        if metrics['preserving_consistency_total'] != 4136 or metrics['flipping_pair_total'] != 864:
            raise ValueError('NLI native relation denominator mismatch')
        if not 0 <= metrics['original_macro_f1'] <= 1:
            raise ValueError('Original macro-F1 must be a fraction')
        if not 0 <= metrics['conditional_faithful_correct'] <= metrics['conditional_faithful_eligible'] <= 864:
            raise ValueError('Conditional faithfulness counts invalid')
    elif data['id'] == 'medpic':
        if metrics['option_total'] != 2335 or metrics['option_correct'] > 2335:
            raise ValueError('MedPIC option denominator mismatch')
        if metrics['flipping_pair_total'] != 244 or metrics['flipping_pair_both_correct'] > 244:
            raise ValueError('MedPIC historical relation mapping must first be audited or amended')
    return True


def collect(spec, jobs, root: Path, output: Path):
    rows = []
    for job in jobs:
        result_path = root / job['model_id'] / job['dataset_id'] / job['method_id'] / 'seed11' / 'metrics.json'
        state, message = 'not_run', ''
        metric = {}
        if result_path.exists():
            try:
                record = read_json(result_path)
                validate_result(record, spec, job)
                state = 'verified_real_run'
                metric = record['metrics']
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                state, message = 'invalid_result', str(exc)
        rows.append(dict(job, status=state, reason=message, result_path=str(result_path),
                         primary_correct=metric.get('primary_correct', ''),
                         primary_total=metric.get('primary_total', ''),
                         original_macro_f1=metric.get('original_macro_f1', ''),
                         interference_rms=metric.get('interference_rms', ''),
                         evidence_correction_rms=metric.get('evidence_correction_rms', '')))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    complete = sum(r['status'] == 'verified_real_run' for r in rows)
    invalid = sum(r['status'] == 'invalid_result' for r in rows)
    return {'verified': complete, 'invalid': invalid, 'pending': len(rows)-complete-invalid,
            'total': len(rows), 'summary': str(output)}


def doctor(spec, providers: Path | None, model_root: Path | None, data_root: Path | None):
    try:
        import torch
        gpu = bool(torch.cuda.is_available()); count = int(torch.cuda.device_count())
    except ImportError:
        gpu, count = False, 0
    entries = {'cuda_available': gpu, 'visible_cuda_devices': count,
               'models': {}, 'datasets': {}, 'providers': {}}
    for m in spec['models']:
        path = model_root / m['key'] if model_root else None
        entries['models'][m['key']] = bool(path and (path / 'config.json').is_file())
    for d in spec['datasets']:
        path = data_root / f"{d['id']}_official_test.jsonl" if data_root else None
        entries['datasets'][d['id']] = bool(path and path.is_file())
    configured = read_json(providers) if providers and providers.exists() else {}
    for b in spec['methods']:
        cmd = configured.get(b['id'])
        entries['providers'][b['id']] = bool(isinstance(cmd, list) and cmd and all(isinstance(x,str) for x in cmd))
    return entries


def run_selected(spec, jobs, args):
    """Execute explicit real adapters; no speculative default implementation."""
    providers = read_json(args.providers) if args.providers and args.providers.exists() else {}
    selected = [j for j in jobs if (not args.model or j['model_id'] == args.model)
                and (not args.dataset or j['dataset_id'] == args.dataset)
                and (not args.method or j['method_id'] == args.method)]
    if not selected:
        raise ValueError('No jobs matched filters')
    if not args.model_root or not args.data_root:
        raise ValueError('--model-root and --data-root required for execution')
    for job in selected:
        template = providers.get(job['method_id'])
        if not isinstance(template,list) or not template:
            raise RuntimeError(f'No validated provider command registered for {job["method_id"]}; no fake test will run')
        context = dict(job,model_path=str(args.model_root / job['model_id']),
                       input_path=str(args.data_root / (job['dataset_id'] + '_official_test.jsonl')),
                       output_dir=str(args.results / job['model_id'] / job['dataset_id'] / job['method_id'] / 'seed11'),
                       seed='11')
        if not (Path(context['model_path']) / 'config.json').exists():
            raise FileNotFoundError('Model checkpoint missing: ' + context['model_path'])
        if not Path(context['input_path']).exists():
            raise FileNotFoundError('Official dataset manifest missing: ' + context['input_path'])
        command = [piece.format_map(context) for piece in template]
        print('EXEC',shlex.join(command),flush=True)
        if not args.dry_run:
            Path(context['output_dir']).mkdir(parents=True,exist_ok=True)
            subprocess.run(command, check=True)
            result = read_json(Path(context['output_dir']) / 'metrics.json')
            validate_result(result,spec,job)
    return {'matched_jobs':len(selected),'mode':'dry_run' if args.dry_run else 'validated_execution'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'doctor', 'collect', 'run'])
    p.add_argument('--spec', type=Path, default=SPEC)
    p.add_argument('--status', type=Path, default=STATUS)
    p.add_argument('--results', type=Path, default=BASE / 'results')
    p.add_argument('--summary', type=Path, default=BASE / 'summary_actual_runs.csv')
    p.add_argument('--providers', type=Path)
    p.add_argument('--model-root', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--model');p.add_argument('--dataset');p.add_argument('--method')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    spec=read_json(args.spec)
    jobs=verify_grid(spec,args.status)
    if args.action=='check': answer={'expected_core_baseline_cells':240,'expected_all_cells':285,'matrix_validated':True}
    elif args.action=='doctor':answer=doctor(spec,args.providers,args.model_root,args.data_root)
    elif args.action=='collect':answer=collect(spec,jobs,args.results,args.summary)
    else:answer=run_selected(spec,jobs,args)
    print(json.dumps(answer,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
