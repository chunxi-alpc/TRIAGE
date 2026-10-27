#!/usr/bin/env python3
"""Fixed-head causal dose response, after offline selection is frozen.

This is an oracle branch-choice diagnostic. It neither evaluates module-one
predictions nor claims treatment/path execution quality. Every positive dose is
a target fixed-query score RMS, including the random-direction control. A
single fixed-head group-sign control is not a whole-pipeline permutation test.
All doses and blocked conditions are reported; confirmation never chooses a dose.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from run_module2_preexperiment import model_manifest, sha256, write_json
from src.module2.applicability import bootstrap_interval, group_weights


METHODS = ('projected_mean', 'raw_mean', 'random_direction', 'reversed_direction',
           'label_shuffled', 'direct_score_bias')
METHOD_DESCRIPTIONS = {
    'projected_mean': 'projected paired mean; fixed-query score RMS matched',
    'raw_mean': 'raw paired mean; fixed-query score RMS matched',
    'random_direction': 'one random direction; score RMS matched, not final key-norm matched',
    'reversed_direction': 'negative projected mean; fixed-query score RMS matched',
    'label_shuffled': 'one source-group sign draw at the true-label-selected fixed head; no reselection',
    'direct_score_bias': 'constant signed bias on identical decision-query/key support',
}
INTERVAL_SCOPE = 'source-guideline bootstrap conditional on frozen operator; unadjusted exploratory interval'


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _estimate(values, groups, seed):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('An estimand needs nonempty finite observations')
    low, high = bootstrap_interval(values, groups, draws=2000, seed=seed)
    return {'mean': float(np.sum(values * group_weights(groups))),
            'ci_low': float(low), 'ci_high': float(high),
            'observations': len(values), 'source_guidelines': len(set(groups)),
            'interval_scope': INTERVAL_SCOPE}


def _metric_samples(index):
    """Keep condition identities so pairing never depends on journal order."""
    metrics = {name: {} for name in ('case_accuracy', 'pair_accuracy',
        'gold_logit_margin', 'gold_attention_mass', 'target_attention_mass',
        'harm_rate', 'correction_rate')}
    pairs = defaultdict(list)
    for key, row in sorted(index.items()):
        group = row['group_id']
        correct, baseline = bool(row['correct']), bool(row['baseline_correct'])
        values = {'case_accuracy': float(correct),
                  'gold_logit_margin': float(row['gold_logit_margin']),
                  'gold_attention_mass': float(row['gold_attention_mass']),
                  'target_attention_mass': float(row['target_attention_mass']),
                  'harm_rate': float(baseline and not correct),
                  'correction_rate': float(not baseline and correct)}
        for metric, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite completed outcome: {metric}')
            metrics[metric][key] = (value, group)
        pairs[row['id']].append(row)
    for identifier, rows in sorted(pairs.items()):
        if sorted(r['variant'] for r in rows) != ['negative', 'positive']:
            raise ValueError('Pair accuracy requires both variants of every triplet')
        if len({r['group_id'] for r in rows}) != 1:
            raise ValueError('Members of a triplet have different source groups')
        metrics['pair_accuracy'][(identifier, 'paired')] = (
            float(all(r['correct'] for r in rows)), rows[0]['group_id'])
    return metrics


def summarize(records, output, seed=261027, *, expected_conditions=None,
              expected_methods=None, expected_taus=None):
    """Report complete estimands, explicit coverage, and matched paired deltas.

    One blocked or missing condition invalidates that entire method/dose's main
    estimand. It remains visible in coverage. Comparisons require the full same
    set of (triplet id, variant) conditions, with identical source-group IDs.
    """
    grouped = defaultdict(dict)
    discovered = {}
    for row in records:
        if row['variant'] not in ('positive', 'negative'):
            raise ValueError('Causal summaries only accept positive/negative variants')
        condition = (row['id'], row['variant'])
        group = row['group_id']
        if condition in discovered and discovered[condition] != group:
            raise ValueError('A paired condition changed its source group')
        discovered[condition] = group
        key = (row['method'], float(row['tau']))
        if condition in grouped[key]:
            raise ValueError('Duplicate method/dose/condition journal record')
        grouped[key][condition] = row
    expected = discovered if expected_conditions is None else dict(expected_conditions)
    if not expected or any(k not in expected or expected[k] != g for k, g in discovered.items()):
        raise ValueError('Unexpected condition or source group in causal journal')
    pairs = defaultdict(list)
    for (identifier, variant), group in expected.items():
        pairs[identifier].append((variant, group))
    if any(sorted(v for v, _ in p) != ['negative', 'positive'] or
           len({g for _, g in p}) != 1 for p in pairs.values()):
        raise ValueError('Expected conditions must contain complete source-consistent pairs')
    methods = sorted(set(expected_methods) if expected_methods is not None else {k[0] for k in grouped})
    taus = sorted(set(expected_taus) if expected_taus is not None else {k[1] for k in grouped})
    if not methods or not taus or 0. not in taus:
        raise ValueError('Causal summaries require methods and a zero-dose baseline')
    if any(m not in methods or t not in taus for m, t in grouped):
        raise ValueError('Journal contains an unplanned method or dose')

    coverage, complete = [], {}
    for method in methods:
        for tau in taus:
            index = grouped.get((method, tau), {})
            counts = Counter(r['status'] for r in index.values())
            missing = len(expected.keys() - index.keys())
            blocked = len(index) - counts['completed']
            valid = not missing and not blocked
            if valid:
                complete[(method, tau)] = index
            reasons = {k: v for k, v in sorted(counts.items()) if k != 'completed'}
            if missing:
                reasons['missing_records'] = missing
            coverage.append({'method': method, 'tau': tau,
                'status': 'completed' if valid else 'partial',
                'expected_observations': len(expected), 'observed': len(index),
                'completed': counts['completed'], 'blocked': blocked, 'missing': missing,
                'coverage_fraction': counts['completed'] / len(expected),
                'source_guidelines': len(set(expected.values())),
                'reasons': json.dumps(reasons, sort_keys=True)})

    baseline_method = 'direct_score_bias' if 'direct_score_bias' in methods else methods[0]
    baseline_key = (baseline_method, 0.)
    baseline_index = complete.get(baseline_key)
    if baseline_index is not None:
        for (method, tau), index in complete.items():
            for key, row in index.items():
                if bool(row['baseline_correct']) != bool(baseline_index[key]['correct']):
                    raise ValueError('Stored baseline correctness disagrees across matched conditions')
                if tau == 0:
                    reference = baseline_index[key]
                    if bool(row['correct']) != bool(reference['correct']) or any(
                        float(row[name]) != float(reference[name]) for name in
                        ('gold_logit_margin', 'gold_attention_mass', 'target_attention_mass')):
                        raise ValueError('Zero-dose outcomes are not identical across methods')

    result, samples = [], {}
    for (method, tau), index in sorted(complete.items()):
        samples[(method, tau)] = _metric_samples(index)
        for metric, observations in samples[(method, tau)].items():
            ordered = sorted(observations)
            stats = _estimate([observations[k][0] for k in ordered],
                              [observations[k][1] for k in ordered], seed)
            result.append({'method': method, 'tau': tau, 'metric': metric, **stats})

    comparisons, comparison_coverage = [], []
    for method in methods:
        for tau in taus:
            references = [('baseline', baseline_key)]
            if tau != 0 and method != 'direct_score_bias' and 'direct_score_bias' in methods:
                references.append(('direct_score_bias', ('direct_score_bias', tau)))
            for reference_name, reference_key in references:
                target_key = (method, tau)
                valid = target_key in complete and reference_key in complete
                comparison_coverage.append({'method': method, 'tau': tau,
                    'reference': reference_name, 'reference_method': reference_key[0],
                    'reference_tau': reference_key[1],
                    'status': 'completed' if valid else 'blocked_incomplete_method_or_reference',
                    'method_complete': target_key in complete, 'reference_complete': reference_key in complete,
                    'expected_observations': len(expected), 'source_guidelines': len(set(expected.values()))})
                if not valid:
                    continue
                for metric, target in samples[target_key].items():
                    reference = samples[reference_key][metric]
                    if target.keys() != reference.keys() or any(target[k][1] != reference[k][1] for k in target):
                        raise ValueError('Paired comparison requires identical condition and group IDs')
                    ordered = sorted(target)
                    stats = _estimate([target[k][0] - reference[k][0] for k in ordered],
                                      [target[k][1] for k in ordered], seed)
                    comparisons.append({'method': method, 'tau': tau,
                        'reference': reference_name, 'reference_method': reference_key[0],
                        'reference_tau': reference_key[1], 'metric': metric,
                        'delta_method_minus_reference': stats.pop('mean'), **stats,
                        'pairing': 'triplet_id+variant; pair_accuracy aggregates both variants by triplet_id'})

    output = Path(output)
    estimate_fields = ['ci_low', 'ci_high', 'observations', 'source_guidelines', 'interval_scope']
    _write_csv(output, result, ['method', 'tau', 'metric', 'mean', *estimate_fields])
    _write_csv(output.parent / 'causal_coverage.csv', coverage,
        ['method', 'tau', 'status', 'expected_observations', 'observed', 'completed',
         'blocked', 'missing', 'coverage_fraction', 'source_guidelines', 'reasons'])
    _write_csv(output.parent / 'paired_comparisons.csv', comparisons,
        ['method', 'tau', 'reference', 'reference_method', 'reference_tau', 'metric',
         'delta_method_minus_reference', *estimate_fields, 'pairing'])
    _write_csv(output.parent / 'paired_coverage.csv', comparison_coverage,
        ['method', 'tau', 'reference', 'reference_method', 'reference_tau', 'status',
         'method_complete', 'reference_complete', 'expected_observations', 'source_guidelines'])
    return {'metrics': result, 'coverage': coverage, 'paired_comparisons': comparisons,
            'paired_coverage': comparison_coverage,
            'status': 'completed' if all(r['status'] == 'completed' for r in coverage) else 'partial'}


def verify_model_files(model_path, fingerprint):
    """Compare actual weights, fast-tokenizer JSON, and configuration bytes."""
    frozen = fingerprint.get('weight_files', {})
    if ('tokenizer.json' not in frozen or 'config.json' not in frozen or
            not any(name.endswith('.safetensors') for name in frozen)):
        raise ValueError('The representation fingerprint lacks weights or tokenizer/config hashes')
    actual = model_manifest(model_path)
    if actual.keys() != frozen.keys():
        raise ValueError('Local snapshot file names differ from the frozen representation manifest')
    for name, expected in frozen.items():
        if actual[name] != expected:
            raise ValueError(f'Actual model/tokenizer SHA-256 or size differs: {name}')
    return actual


def max_abs_error(reference, actual):
    reference, actual = np.asarray(reference, float), np.asarray(actual, float)
    if reference.shape != actual.shape or not reference.size:
        raise ValueError('Zero-check arrays must have the same nonempty shape')
    if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
        raise ValueError('Nonfinite model output in zero-intervention validation')
    return float(np.max(np.abs(reference - actual)))


def enforce_zero_identity(full_prefix_error, cache_zero_error, *, full_prefix_atol=1e-4):
    if not all(math.isfinite(x) and x >= 0 for x in (full_prefix_error, cache_zero_error)):
        raise ValueError('Zero-intervention errors must be finite and nonnegative')
    if full_prefix_error > full_prefix_atol:
        raise RuntimeError(f'Full/prefix zero-intervention error {full_prefix_error:g} exceeds {full_prefix_atol:g}')
    if cache_zero_error != 0.:
        raise RuntimeError(f'Cached zero interventions are not exact identities: error={cache_zero_error:g}')


def calibration_status(actual_rms, target_rms, tolerance=.05):
    if not math.isfinite(actual_rms) or actual_rms < 0 or not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError('Score RMS calibration needs finite nonnegative actual and positive target values')
    error = abs(actual_rms - target_rms) / target_rms
    return ('completed' if error <= tolerance + 1e-12 else 'blocked_score_rms_mismatch'), error


def run(args, started):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.module3.applicability_shift import (decision_logits, measure_fixed_query_shifts,
                                               prepare_prefix)

    representation_status = json.loads((args.representation_results / 'status.json').read_text())
    if representation_status.get('status') != 'completed':
        raise ValueError('The representation run must complete before causal confirmation')
    selection_file = args.representation_results / 'selection.json'
    selection = json.loads(selection_file.read_text())
    fingerprint = json.loads((args.representation_results / 'extraction_fingerprint.json').read_text())
    if fingerprint['triplets_sha256'] != sha256(args.triplets):
        raise ValueError('Triplets differ from the representation run')
    if sha256(ROOT / 'src/module2/extraction.py') != fingerprint['source_files']['src/module2/extraction.py']:
        raise ValueError('Extraction adapter differs from the frozen representation run')
    runtime = {'torch': str(torch.__version__), 'transformers': str(transformers.__version__)}
    for package, version in runtime.items():
        if version != fingerprint[package]:
            raise ValueError(f'{package} version differs from the representation run: {version}')
    actual_files = verify_model_files(args.model, fingerprint)
    provenance = args.model / 'download_provenance.json'
    if provenance.exists():
        source = json.loads(provenance.read_text())
        if source['repo_id'] != fingerprint['model_id'] or source['revision'] != fingerprint['revision']:
            raise ValueError('Causal model does not match the extracted model')
    write_json(args.output / 'model_verification.json', {
        'status': 'verified_actual_files_and_runtime', 'model_id': fingerprint['model_id'],
        'revision': fingerprint['revision'], 'files': actual_files, 'runtime': runtime,
        'fingerprint_sha256': sha256(args.representation_results / 'extraction_fingerprint.json')})
    operators = np.load(args.representation_results / 'operators.npz', allow_pickle=False)
    extracted = np.load(args.representation_results / 'extraction.npz', allow_pickle=False)
    metadata = json.loads((args.representation_results / 'extraction_metadata.json').read_text())
    all_rows = [json.loads(line) for line in args.triplets.read_text().splitlines() if line.strip()]
    if [r['id'] for r in metadata['metadata']] != [r['id'] for r in all_rows]:
        raise ValueError('Extraction row order/identities differ from the frozen triplets')
    rows = [r for r in all_rows if r['split'] == 'confirm']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Confirmation requires nonempty unique triplet IDs')
    confirmation_groups = {r['group_id'] for r in rows}
    if confirmation_groups & (set(selection['fit_groups']) | set(selection['selection_groups'])):
        raise ValueError('Confirmation reuses fitting or selection source guidelines')
    layer, head = (selection['diagnostic_head'][k] for k in ('layer', 'kv_head'))
    rng = np.random.default_rng(args.seed)
    projected = operators['delta'][layer, head]
    random = rng.standard_normal(projected.shape)
    random *= np.linalg.norm(projected) / max(np.linalg.norm(random), 1e-12)
    fit_indices = [i for i, r in enumerate(all_rows) if r['split'] == 'fit']
    fit_groups = [all_rows[i]['group_id'] for i in fit_indices]
    # Shuffle signs at source-group level; all repeated variants keep one sign.
    signs = dict(zip(sorted(set(fit_groups)), rng.choice([-1., 1.], len(set(fit_groups)))))
    diff = extracted['keys_positive'][fit_indices, layer, head] - extracted['keys_negative'][fit_indices, layer, head]
    shuffled = np.einsum('n,nd->d', group_weights(fit_groups) * np.array([signs[g] for g in fit_groups]), diff)
    basis = operators['basis'][layer, head]
    shuffled = basis @ (basis.T @ shuffled)
    vectors = {'projected_mean': projected, 'raw_mean': operators['raw_delta'][layer, head],
               'random_direction': random, 'reversed_direction': -projected, 'label_shuffled': shuffled}
    plan = {'schema_version': 'module2.causal_plan.v2', 'model_id': fingerprint['model_id'],
        'revision': fingerprint['revision'], 'selection_sha256': sha256(selection_file),
        'operators_sha256': sha256(args.representation_results / 'operators.npz'),
        'triplets_sha256': sha256(args.triplets), 'head': {'layer': layer, 'kv_head': head},
        'head_accepted_by_offline_screen': selection['diagnostic_head_accepted'],
        'taus': args.taus, 'methods': METHODS, 'method_descriptions': METHOD_DESCRIPTIONS,
        'routing': 'oracle sign of target rule applicability',
        'query_scope': 'first-answer query only; edit prefix cache before final prompt token',
        'target_scope': 'one rule action span; all GQA queries sharing one KV head',
        'matching': 'per-input per-direction fixed-query QK score RMS; zero or excessive-error responses blocked',
        'score_rms_relative_error_tolerance': args.score_rms_tolerance,
        'full_prefix_candidate_logit_atol': args.full_prefix_atol,
        'cached_zero_logit_and_attention_atol': 0.,
        'seed': args.seed, 'shuffled_group_signs': signs,
        'confirmation_groups': sorted(confirmation_groups), 'runtime': runtime,
        'dtype': fingerprint['dtype'], 'device': fingerprint['device'], 'threads': args.threads,
        'outcome': 'single-token A/B branch choice; not action execution',
        'summary_policy': 'complete method/dose only; all blocked/missing coverage remains visible',
        'paired_comparisons': 'matched triplet_id+variant vs zero baseline and same-dose direct bias',
        'source_files': {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__), ROOT / 'src/module3/applicability_shift.py']}}
    write_json(args.output / 'plan.json', plan)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=getattr(torch, fingerprint['dtype']), attn_implementation='eager').to(fingerprint['device']).eval()
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ('A', 'B')]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError('This diagnostic requires single-token A/B labels')
    label_ids = [ids[0] for ids in label_ids]
    if label_ids != metadata['candidate_token_ids'] or metadata['candidate_labels'] != ['A', 'B']:
        raise ValueError('Actual tokenizer candidate IDs differ from the frozen extraction')
    by_prompt = defaultdict(list)
    index_by_id = {r['id']: i for i, r in enumerate(all_rows)}
    for row in rows:
        for variant in ('positive', 'negative'):
            by_prompt[row['prompts'][variant]].append((row, variant))
    records, zero_errors, cache_zero_errors, calibration_errors = [], [], [], []
    journal = args.output / 'predictions.jsonl'
    with journal.open('w') as stream, (args.output / 'zero_validation.jsonl').open('w') as validation:
        for prompt_index, (prompt, requests) in enumerate(by_prompt.items()):
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            first, first_variant = requests[0]
            state = prepare_prefix(model, tokenizer, prompt, first['spans'][first_variant],
                                   max_length=fingerprint['max_length'])
            target_diagnostics = [{'layer': layer, 'kv_head': head,
                'positions': positions, 'rule_id': rule_id}
                for rule_id, positions in state.rule_positions.items()]
            baseline = decision_logits(model, state, return_diagnostics=True,
                                       diagnostic_targets=target_diagnostics)
            base_scores = baseline['logits'][label_ids].numpy()
            prompt_errors = [max_abs_error(
                extracted[f'candidate_logits_{variant}'][index_by_id[row['id']]], base_scores)
                for row, variant in requests]
            zero_shift = decision_logits(model, state,
                shifts=[{**entry, 'delta': projected, 'coefficient': 0.} for entry in target_diagnostics],
                return_diagnostics=True, diagnostic_targets=target_diagnostics)
            zero_bias = decision_logits(model, state,
                score_bias=[{**entry, 'bias': 0.} for entry in target_diagnostics],
                return_diagnostics=True, diagnostic_targets=target_diagnostics)
            base_mass = [r['per_query_mass'] for r in baseline['attention_mass']]
            identity_errors = {
                'zero_shift_full_vocabulary_logit_error': max_abs_error(baseline['logits'].numpy(), zero_shift['logits'].numpy()),
                'zero_bias_full_vocabulary_logit_error': max_abs_error(baseline['logits'].numpy(), zero_bias['logits'].numpy()),
                'zero_shift_attention_error': max_abs_error(base_mass, [r['per_query_mass'] for r in zero_shift['attention_mass']]),
                'zero_bias_attention_error': max_abs_error(base_mass, [r['per_query_mass'] for r in zero_bias['attention_mass']]),
            }
            full_error, identity_error = max(prompt_errors), max(identity_errors.values())
            zero_errors.extend(prompt_errors)
            cache_zero_errors.append(identity_error)
            validation.write(json.dumps({'prompt_sha256': prompt_hash,
                'max_split_vs_full_candidate_logit_error': full_error, **identity_errors,
                'full_prefix_atol': args.full_prefix_atol, 'cached_zero_atol': 0.,
                'passed': full_error <= args.full_prefix_atol and identity_error == 0.}, allow_nan=False) + '\n')
            validation.flush()
            enforce_zero_identity(full_error, identity_error, full_prefix_atol=args.full_prefix_atol)
            if state.cache.get_seq_length() != state.prefix_length:
                raise RuntimeError('Zero-intervention checks mutated the reusable prefix length')

            for row, variant in requests:
                target_id = row['target_rule_id']
                target_index = ('A', 'B').index(target_id)
                gold_index = target_index if variant == 'positive' else 1 - target_index
                gold_id = ('A', 'B')[gold_index]
                sign = 1. if variant == 'positive' else -1.
                target_positions = state.rule_positions[target_id]
                shift_rows = [{'layer': layer, 'kv_head': head, 'positions': target_positions,
                               'delta': vector, 'coefficient': 1.} for vector in vectors.values()]
                measurements = measure_fixed_query_shifts(model, state, shift_rows)
                enforce_zero_identity(0., max_abs_error(baseline['logits'].numpy(),
                    measurements['baseline_logits'].numpy()), full_prefix_atol=args.full_prefix_atol)
                units = dict(zip(vectors, measurements['shifts']))
                for method in METHODS:
                    for tau in args.taus:
                        record = {'id': row['id'], 'group_id': row['group_id'],
                            'variant': variant, 'target_rule_id': target_id, 'gold_rule_id': gold_id,
                            'method': method, 'tau': tau, 'status': 'completed',
                            'prompt_sha256': prompt_hash,
                            'baseline_correct': int(np.argmax(base_scores)) == gold_index,
                            'layer': layer, 'kv_head': head, 'head_accepted': selection['diagnostic_head_accepted']}
                        if tau == 0:
                            outcome = baseline
                            record.update({'matched_score_rms': 0., 'coefficient': 0.,
                                           'key_shift_l2': 0., 'fixed_query_mean': 0.,
                                           'score_rms_relative_error': 0.})
                        elif method == 'direct_score_bias':
                            actual_bias = float(torch.tensor(sign * tau,
                                dtype=model.get_input_embeddings().weight.dtype).item())
                            status, relative_error = calibration_status(abs(actual_bias), tau, args.score_rms_tolerance)
                            calibration_errors.append(relative_error)
                            record.update({'status': status, 'matched_score_rms': abs(actual_bias),
                                'coefficient': actual_bias, 'key_shift_l2': 0.,
                                'fixed_query_mean': actual_bias, 'score_rms_relative_error': relative_error,
                                'fixed_query_positive_fraction': float(actual_bias > 0)})
                            if status == 'completed':
                                outcome = decision_logits(model, state, score_bias=[{'layer': layer,
                                    'kv_head': head, 'positions': target_positions, 'bias': actual_bias}],
                                    return_diagnostics=True, diagnostic_targets=target_diagnostics)
                        elif units[method]['rms'] <= 1e-10:
                            record.update({'status': 'blocked_zero_fixed_query_response',
                                           'unit_score_rms': units[method]['rms']})
                        else:
                            coefficient = sign * tau / units[method]['rms']
                            shift = {'layer': layer, 'kv_head': head, 'positions': target_positions,
                                     'delta': vectors[method], 'coefficient': coefficient}
                            check = measure_fixed_query_shifts(model, state, [shift])['shifts'][0]
                            status, relative_error = calibration_status(check['rms'], tau, args.score_rms_tolerance)
                            calibration_errors.append(relative_error)
                            record.update({'status': status, 'coefficient': coefficient,
                                'matched_score_rms': check['rms'], 'fixed_query_mean': check['mean'],
                                'fixed_query_positive_fraction': check['positive_fraction'],
                                'score_rms_relative_error': relative_error,
                                'unit_score_rms': units[method]['rms'],
                                'key_shift_l2': abs(coefficient) * float(np.linalg.norm(vectors[method]))})
                            if status == 'completed':
                                outcome = decision_logits(model, state, shifts=[shift],
                                    return_diagnostics=True, diagnostic_targets=target_diagnostics)
                        if record['status'] == 'completed':
                            scores = outcome['logits'][label_ids].numpy()
                            if not np.isfinite(scores).all():
                                raise RuntimeError('Nonfinite candidate logits')
                            masses = {r['rule_id']: r['target_attention_mass'] for r in outcome['attention_mass']}
                            record.update({'candidate_logits': scores.tolist(),
                                'correct': int(np.argmax(scores)) == gold_index,
                                'gold_logit_margin': float(scores[gold_index] - scores[1 - gold_index]),
                                'target_attention_mass': masses[target_id], 'gold_attention_mass': masses[gold_id]})
                        records.append(record)
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
            print(json.dumps({'unique_prompt': prompt_index + 1, 'total': len(by_prompt),
                              'elapsed_seconds': round(time.time() - started, 1)}), flush=True)
    summary = summarize(records, args.output / 'causal_metrics.csv', args.seed,
        expected_conditions={(r['id'], v): r['group_id'] for r in rows for v in ('positive', 'negative')},
        expected_methods=METHODS, expected_taus=args.taus)
    counts = Counter(r['status'] for r in records)
    write_json(args.output / 'status.json', {'status': summary['status'],
        'elapsed_seconds': round(time.time() - started, 3), 'journal_records': len(records),
        'record_status_counts': dict(counts), 'blocked_records': len(records) - counts['completed'],
        'incomplete_method_doses': sum(r['status'] != 'completed' for r in summary['coverage']),
        'unique_prompts': len(by_prompt), 'triplets': len(rows),
        'source_guidelines': len(confirmation_groups),
        'zero_validation_passed': True,
        'max_split_vs_full_forward_logit_error': max(zero_errors),
        'max_cached_zero_intervention_error': max(cache_zero_errors),
        'max_score_rms_relative_error': max(calibration_errors, default=None),
        'score_rms_relative_error_tolerance': args.score_rms_tolerance,
        'head_accepted_by_offline_screen': selection['diagnostic_head_accepted'],
        'coverage_file': 'causal_coverage.csv', 'paired_comparisons_file': 'paired_comparisons.csv',
        'paired_coverage_file': 'paired_coverage.csv',
        'warning': 'Exploratory development confirmation with few source guidelines; no efficacy claim'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triplets', type=Path, required=True)
    parser.add_argument('--representation-results', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--taus', nargs='+', type=float, default=[0., .01, .05, .1])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=261027)
    parser.add_argument('--full-prefix-atol', type=float, default=1e-4,
                        help='Absolute A/B logit tolerance; cached zero edits must be exactly identical')
    parser.add_argument('--score-rms-tolerance', type=float, default=.05,
                        help='Block an observation when relative fixed-query RMS mismatch exceeds this value')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty causal output directory')
    if (not args.taus or any(not math.isfinite(t) or t < 0 for t in args.taus)
            or 0. not in args.taus or len(set(args.taus)) != len(args.taus)):
        raise ValueError('Use unique finite nonnegative doses including zero')
    if (not math.isfinite(args.full_prefix_atol) or args.full_prefix_atol < 0
            or not math.isfinite(args.score_rms_tolerance) or not 0 <= args.score_rms_tolerance < 1):
        raise ValueError('Invalid zero-check or RMS tolerance')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(args.output / 'status.json', {'status': 'running', 'stage': 'preflight'})
    try:
        run(args, started)
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'failed', 'error_type': type(error).__name__,
            'error': str(error), 'elapsed_seconds': round(time.time() - started, 3)})
        raise


if __name__ == '__main__':
    main()
