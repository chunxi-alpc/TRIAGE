#!/usr/bin/env python3
"""Run an explicit, development-only Qwen3 applicability representation pilot.

No fallback model, pseudo labels, test-set selection, or fabricated activations.
This entry point is independent of the historical three-module pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.module2.applicability import (bootstrap_stability, evaluate_heads,
                                      fit_subspaces, save_fit, select_heads)
from src.module2.minimal_pairs import audit_triplets


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=json_value, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(path):
    path = Path(path)
    files = sorted(path.glob('*.safetensors')) + sorted(path.glob('*.json'))
    return {file.name: {'sha256': sha256(file), 'bytes': file.stat().st_size}
            for file in files if file.is_file()}


def fit_and_select(extracted, rows, out, args):
    partitions = {name: np.array([r['split'] == name for r in rows])
                  for name in ('fit', 'select', 'confirm')}
    groups = np.array([r['group_id'] for r in rows])
    fit_mask = partitions['fit']
    keys, queries = extracted['keys'], extracted['effective_queries']
    fit = fit_subspaces(*(keys[v][fit_mask] for v in ('positive', 'negative', 'neutral')),
        groups[fit_mask], rank=args.rank,
        baseline_score_mean=np.stack([extracted['baseline_score_mean'][v][fit_mask]
                                      for v in ('positive', 'negative')]),
        baseline_score_std=np.stack([extracted['baseline_score_std'][v][fit_mask]
                                     for v in ('positive', 'negative')]))
    evaluations = {}
    def evaluate(name):
        mask = partitions[name]
        return evaluate_heads(fit, *(keys[v][mask] for v in ('positive', 'negative', 'neutral')),
            queries['positive'][mask], queries['negative'][mask], groups[mask],
            draws=args.bootstrap, seed=args.seed)
    evaluations['select'] = evaluate('select')
    ev = evaluations['select']
    ranked = sorted([(l, h) for l in range(fit['delta'].shape[0])
                     for h in range(fit['delta'].shape[1])],
                    key=lambda head: (-float(ev['score'][head]), head))
    # Bootstrap only the leading predeclared candidate count. Unchecked heads
    # cannot enter the accepted set. Confirmation has not been consulted.
    candidates = ranked[:min(12, len(ranked))]
    stability = bootstrap_stability(keys['positive'][fit_mask], keys['negative'][fit_mask],
        groups[fit_mask], fit, candidates, draws=args.stability_bootstrap, seed=args.seed)
    selected = select_heads(fit, ev, max_heads=args.max_heads, stability=stability)
    diagnostic_head = selected[0] if selected else ranked[0]
    selection = {'schema_version': 'module2.selection.v2',
        'selected_heads': [{'layer': l, 'kv_head': h} for l, h in selected],
        'status': 'eligible_heads' if selected else 'no_eligible_heads',
        'diagnostic_head': {'layer': diagnostic_head[0], 'kv_head': diagnostic_head[1]},
        'diagnostic_head_accepted': bool(selected),
        'diagnostic_policy': 'first eligible head; otherwise top select-score candidate for falsification only',
        'rank': args.rank, 'max_heads': args.max_heads,
        'fit_groups': fit['fit_group_ids'], 'selection_groups': ev['group_ids'],
        'confirmation_not_used': True,
        'thresholds': {'min_selection_groups': 3, 'read_positive_lower_95_gt': 0,
            'read_negative_lower_95_gt': 0, 'paired_sign_consistency_min': .6,
            'fit_coherence_min': .01, 'fit_bootstrap_overlap_lower_min': .5},
        'thresholds_status': 'prespecified exploratory screens; no multiplicity-corrected discovery claim',
        'stability': [{'layer': l, 'kv_head': h, **v} for (l, h), v in stability.items()]}
    write_json(out / 'selection.json', selection)
    save_fit(out / 'operators.npz', fit)
    write_json(out / 'operator_metadata.json', {k: v for k, v in fit.items()
                                               if not isinstance(v, np.ndarray)})
    # Only after selection has been persisted may confirmation be evaluated.
    evaluations['confirm'] = evaluate('confirm')
    numeric = {f'{split}_{name}': value for split, results in evaluations.items()
               for name, value in results.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(out / 'head_evaluation.npz', **numeric)
    records = []
    for split, evaluation in evaluations.items():
        for layer, head in ranked:
            row = {'split': split, 'layer': layer, 'kv_head': head,
                   'selected': (layer, head) in selected,
                   'diagnostic_head': (layer, head) == diagnostic_head,
                   'fit_mean_paired_l2': float(fit['mean_paired_l2'][layer, head]),
                   'fit_coherence': float(fit['coherence'][layer, head]),
                   'fit_presence_ratio': float(fit['presence_ratio'][layer, head]),
                   'fit_presence_cosine': float(fit['presence_cosine'][layer, head]),
                   'delta_norm': float(np.linalg.norm(fit['delta'][layer, head])),
                   'fit_sigma': float(fit['sigma'][layer, head])}
            for metric, value in evaluation.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    row[metric] = float(value[layer, head])
            records.append(row)
    with (out / 'head_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Frozen training-difference axes; show confirmation pairs in those axes.
    # Never fit a second visualization PCA on confirmation data.
    visual = []
    for layer, head in candidates[:4]:
        u = fit['basis'][layer, head, :, :min(2, args.rank)]
        for i in np.flatnonzero(partitions['confirm']):
            for variant in ('positive', 'negative', 'neutral'):
                point = (keys[variant][i, layer, head] - fit['mu_neutral'][layer, head]) @ u
                visual.append({'id': rows[i]['id'], 'group_id': rows[i]['group_id'],
                    'layer': layer, 'kv_head': head, 'variant': variant,
                    'x': float(point[0]), 'y': float(point[1]) if len(point) > 1 else 0.})
    with (out / 'paired_projection.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(visual[0]))
        writer.writeheader()
        writer.writerows(visual)
    return fit, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triplets', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True,
                        help='Explicit local pretrained Qwen3 snapshot; never substituted')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--max-heads', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--stability-bootstrap', type=int, default=64)
    parser.add_argument('--seed', type=int, default=261027)
    parser.add_argument('--reuse-extraction', action='store_true',
                        help='Reuse an already completed, fingerprint-matched extraction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output
    if (out / 'status.json').exists() and not args.reuse_extraction:
        raise ValueError('Use a fresh output directory; completed runs are not overwritten')
    rows = [json.loads(line) for line in args.triplets.read_text().splitlines() if line.strip()]
    data_audit = audit_triplets(rows)
    started = time.time()
    write_json(out / 'status.json', {'status': 'running', 'stage': 'preflight'})
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.module2.extraction import extract_triplets
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no device/model fallback')
        fingerprint = {'triplets_sha256': sha256(args.triplets), 'model_id': args.model_id,
            'revision': args.revision, 'dtype': args.dtype, 'device': args.device,
            'rank': args.rank, 'max_length': args.max_length, 'roles': ['action'],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'threads': args.threads, 'source_files': {
                str(path.relative_to(ROOT)): sha256(path) for path in [
                    ROOT / 'src/module2/minimal_pairs.py', ROOT / 'src/module2/extraction.py',
                    ROOT / 'src/module2/applicability.py', Path(__file__)]}}
        if args.reuse_extraction:
            prior = json.loads((out / 'extraction_fingerprint.json').read_text())
            for key in ('triplets_sha256', 'model_id', 'revision', 'dtype', 'device', 'max_length'):
                if prior[key] != fingerprint[key]:
                    raise ValueError(f'Extraction provenance mismatch: {key}')
            if prior['source_files']['src/module2/extraction.py'] != fingerprint['source_files']['src/module2/extraction.py']:
                raise ValueError('Extraction implementation changed; extract afresh')
            metadata = json.loads((out / 'extraction_metadata.json').read_text())
            arrays = np.load(out / 'extraction.npz', allow_pickle=False)
            extracted = {**metadata, **{name: {variant: arrays[f'{name}_{variant}']
                for variant in ('positive', 'negative', 'neutral')}
                for name in ('keys', 'effective_queries', 'baseline_score_mean',
                             'baseline_score_std', 'candidate_logits')}}
        else:
            fingerprint['weight_files'] = model_manifest(args.model)
            fingerprint['repository_base_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            write_json(out / 'extraction_fingerprint.json', fingerprint)
            write_json(out / 'data_audit.json', data_audit)
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            print('Loading explicit pretrained snapshot', args.model_id, args.revision, flush=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=getattr(torch, args.dtype), attn_implementation='eager').to(args.device).eval()
            def progress(done, total, forwards):
                if done % 4 == 0 or done == total:
                    print(json.dumps({'triplets': done, 'total': total, 'unique_forwards': forwards,
                                      'elapsed_seconds': round(time.time() - started, 1)}), flush=True)
            extracted = extract_triplets(model, tokenizer, rows, max_length=args.max_length,
                                          progress=progress)
            names = ('keys', 'effective_queries', 'baseline_score_mean',
                     'baseline_score_std', 'candidate_logits')
            np.savez_compressed(out / 'extraction.npz', **{
                f'{name}_{variant}': values for name in names
                for variant, values in extracted[name].items()})
            write_json(out / 'extraction_metadata.json', {k: v for k, v in extracted.items() if k not in names})
            del model
        print('Fitting offline subspaces and freezing selection', flush=True)
        fit, selection = fit_and_select(extracted, rows, out, args)
        first_layer_max = float(np.max(np.abs(extracted['keys']['positive'][:, 0] -
                                              extracted['keys']['negative'][:, 0])))
        write_json(out / 'status.json', {'status': 'completed', 'stage': 'representation_pilot',
            'elapsed_seconds': round(time.time() - started, 3),
            'unique_model_forwards': extracted['forward_count'],
            'triplets': len(rows), 'source_guidelines': data_audit['source_guidelines'],
            'first_layer_max_paired_key_difference': first_layer_max,
            'selected_head_count': len(selection['selected_heads']),
            'clinical_case_evaluation': False, 'module1_evaluation': False,
            'confirmation_status': 'development-only exploratory; not a benchmark test set'})
        print(json.dumps(selection, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(out / 'status.json', {'status': 'failed', 'error_type': type(error).__name__,
            'error': str(error), 'elapsed_seconds': round(time.time() - started, 3)})
        raise


if __name__ == '__main__':
    main()
