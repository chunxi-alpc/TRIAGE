#!/usr/bin/env python3
"""Read-only, post-run stratification of the frozen module-two causal pilot.

This script never loads a model or chooses a head, direction, or dose. Running
and interrupted journals are rejected before predictions are read. The one
complete run is split by original/swapped action mapping and source guideline.
Every planned method/dose remains visible. Single-guideline estimates are
descriptive and have no cluster-bootstrap interval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module2.applicability import bootstrap_interval, group_weights


METRICS = ('case_accuracy', 'pair_accuracy', 'gold_logit_margin',
           'gold_attention_mass', 'target_attention_mass', 'harm_rate', 'correction_rate')
METHOD_NAMES = {'projected_mean': '投影均值', 'raw_mean': '原始均值',
                'random_direction': '随机方向', 'reversed_direction': '反向方向',
                'label_shuffled': '组符号对照', 'direct_score_bias': '直接偏置'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def samples(index):
    result = {metric: {} for metric in METRICS}
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
                raise ValueError('Nonfinite completed outcome')
            result[metric][key] = (value, group)
        pairs[row['id']].append(row)
    for identifier, pair in sorted(pairs.items()):
        if sorted(row['variant'] for row in pair) != ['negative', 'positive']:
            raise ValueError('A stratum must retain both variants of each target-defined pair')
        if len({row['group_id'] for row in pair}) != 1:
            raise ValueError('Paired variants have different source guidelines')
        result['pair_accuracy'][(identifier, 'paired')] = (
            float(all(row['correct'] for row in pair)), pair[0]['group_id'])
    return result


def estimate(observations, seed):
    ordered = sorted(observations)
    values = np.asarray([observations[key][0] for key in ordered], dtype=float)
    groups = [observations[key][1] for key in ordered]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite estimand')
    independent = len(set(groups))
    low = high = None
    if independent > 1:
        low, high = map(float, bootstrap_interval(values, groups, draws=2000, seed=seed))
    return {'mean': float(np.sum(values * group_weights(groups))),
            'ci_low': low, 'ci_high': high, 'observations': len(values),
            'source_guidelines': independent,
            'interval_scope': ('source-guideline percentile bootstrap; frozen operator; exploratory; unadjusted'
                               if independent > 1 else 'single guideline; descriptive only; no cluster interval')}


def validate_records(records, confirm, plan):
    expected = {(identifier, variant): row['group_id'] for identifier, row in confirm.items()
                for variant in ('positive', 'negative')}
    grouped = defaultdict(dict)
    for row in records:
        condition = (row['id'], row['variant'])
        if condition not in expected or row['group_id'] != expected[condition]:
            raise ValueError('Prediction identity/group differs from confirmation data')
        source = confirm[row['id']]
        if row['target_rule_id'] != source['target_rule_id']:
            raise ValueError('Prediction target differs from its paired source rule')
        gold = source['target_rule_id'] if row['variant'] == 'positive' else (
            'B' if source['target_rule_id'] == 'A' else 'A')
        prompt_hash = hashlib.sha256(source['prompts'][row['variant']].encode()).hexdigest()
        if row['gold_rule_id'] != gold or row['prompt_sha256'] != prompt_hash:
            raise ValueError('Prediction gold or prompt differs from the frozen source')
        if any(row[name] != plan['head'][name] for name in ('layer', 'kv_head')):
            raise ValueError('Prediction head differs from the frozen diagnostic head')
        method, tau = row['method'], float(row['tau'])
        if method not in plan['methods'] or tau not in plan['taus']:
            raise ValueError('Unplanned method or dose')
        if condition in grouped[(method, tau)]:
            raise ValueError('Duplicate method/dose/condition observation')
        if row['status'] == 'completed':
            logits = np.asarray(row['candidate_logits'], dtype=float)
            if logits.shape != (2,) or not np.isfinite(logits).all():
                raise ValueError('Invalid A/B scores')
            if bool(row['correct']) != (int(np.argmax(logits)) == ('A', 'B').index(gold)):
                raise ValueError('Saved correctness disagrees with its candidate scores')
        grouped[(method, tau)][condition] = row
    return expected, grouped


def check_overall_against_canonical(rows, canonical_path, *, paired=False):
    own = {}
    for row in rows:
        if row['stratum'] != 'all':
            continue
        key = (row['method'], float(row['tau']), row['metric'])
        if paired:
            key += (row['reference'],)
        own[key] = row
    with Path(canonical_path).open() as stream:
        canonical = list(csv.DictReader(stream))
    canonical_keys = set()
    value_field = 'delta_method_minus_reference' if paired else 'mean'
    for row in canonical:
        key = (row['method'], float(row['tau']), row['metric'])
        if paired:
            key += (row['reference'],)
        if key in canonical_keys or key not in own:
            raise ValueError('Canonical summary has duplicate or unexpected estimands')
        canonical_keys.add(key)
        for field in (value_field, 'ci_low', 'ci_high'):
            if not math.isclose(float(row[field]), own[key][field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Overall recomputation disagrees with canonical {field}: {key}')
        for field in ('observations', 'source_guidelines'):
            if int(row[field]) != own[key][field]:
                raise ValueError('Canonical denominator mismatch')
    if canonical_keys != own.keys():
        raise ValueError('Canonical summary omits a recomputed overall estimand')


def run(args):
    # This gate precedes reading the data or prediction journal.
    status = json.loads((args.causal / 'status.json').read_text())
    if status.get('status') not in ('completed', 'partial'):
        raise ValueError('Only a terminal completed/partial causal run can be summarized; no live or interrupted effects are read')
    own_outputs = {'metrics.csv', 'paired_comparisons.csv', 'coverage.csv', 'paired_coverage.csv',
                   'routing_diagnostics.csv', 'calibration_diagnostics.csv', 'STRATA.zh.md', 'manifest.json'}
    if args.output.resolve() == args.causal.resolve():
        raise ValueError('Use a separate analysis directory; frozen causal artifacts must not be overwritten')
    if args.output.exists() and any(path.name not in own_outputs for path in args.output.iterdir()):
        raise ValueError('The analysis directory contains artifacts not owned by this script')
    plan = json.loads((args.causal / 'plan.json').read_text())
    if digest(args.triplets) != plan['triplets_sha256']:
        raise ValueError('Triplets differ from the frozen causal plan')
    data = read_jsonl(args.triplets)
    if len({row['id'] for row in data}) != len(data):
        raise ValueError('Duplicate source triplet identities')
    confirm = {row['id']: row for row in data if row['split'] == 'confirm'}
    if sorted({row['group_id'] for row in confirm.values()}) != plan['confirmation_groups']:
        raise ValueError('Confirmation source groups differ from the plan')
    if any(row['controls']['action_permutation'] not in (0, 1) for row in confirm.values()):
        raise ValueError('Expected original/swapped binary action mappings')
    records = read_jsonl(args.causal / 'predictions.jsonl')
    if len(records) != status['journal_records']:
        raise ValueError('Journal length differs from its terminal status')
    expected, grouped = validate_records(records, confirm, plan)
    methods, taus, seed = plan['methods'], plan['taus'], plan['seed']
    if 'direct_score_bias' not in methods or 0. not in taus:
        raise ValueError('Expected the frozen direct-bias and zero-dose references')
    strata = [('all', '全部确认观测', set(confirm))]
    strata += [('action_original', '原始动作映射', {i for i, r in confirm.items() if r['controls']['action_permutation'] == 0}),
               ('action_swapped', '置换动作映射', {i for i, r in confirm.items() if r['controls']['action_permutation'] == 1})]
    strata += [(group, group, {i for i, r in confirm.items() if r['group_id'] == group})
               for group in plan['confirmation_groups']]
    metrics, comparisons, coverage, paired_coverage, completed_samples = [], [], [], [], {}
    for stratum, label, identifiers in strata:
        conditions = {key: group for key, group in expected.items() if key[0] in identifiers}
        if not conditions:
            raise ValueError('An expected descriptive stratum is empty')
        for method in methods:
            for tau in taus:
                full = grouped.get((method, tau), {})
                global_complete = (full.keys() == expected.keys() and
                                   all(row['status'] == 'completed' for row in full.values()))
                index = {key: row for key, row in full.items() if key in conditions}
                counts = Counter(row['status'] for row in index.values())
                missing = len(conditions.keys() - index.keys())
                row_status = 'completed' if global_complete else 'blocked_incomplete_global_method_dose'
                coverage.append({'stratum': stratum, 'label': label, 'method': method, 'tau': tau,
                    'status': row_status, 'global_method_dose_complete': global_complete,
                    'expected_observations': len(conditions), 'expected_pairs': len(identifiers),
                    'expected_unique_prompts': len({confirm[i]['prompts'][v] for i, v in conditions}),
                    'source_guidelines': len(set(conditions.values())), 'observed': len(index),
                    'completed': counts['completed'], 'blocked': len(index) - counts['completed'],
                    'missing': missing, 'coverage_fraction': counts['completed'] / len(conditions)})
                if not global_complete:
                    continue
                current = samples(index)
                completed_samples[(stratum, method, tau)] = current
                for metric, values in current.items():
                    metrics.append({'stratum': stratum, 'label': label, 'method': method,
                        'tau': tau, 'metric': metric, **estimate(values, seed)})
    for stratum, label, _ in strata:
        for method in methods:
            for tau in taus:
                current = completed_samples.get((stratum, method, tau))
                references = [('baseline', 'direct_score_bias', 0.)]
                if tau != 0 and method != 'direct_score_bias':
                    references.append(('direct_score_bias', 'direct_score_bias', tau))
                for reference_name, reference_method, reference_tau in references:
                    reference = completed_samples.get((stratum, reference_method, reference_tau))
                    paired_coverage.append({'stratum': stratum, 'label': label,
                        'method': method, 'tau': tau, 'reference': reference_name,
                        'reference_method': reference_method, 'reference_tau': reference_tau,
                        'method_complete': current is not None, 'reference_complete': reference is not None,
                        'status': 'completed' if current is not None and reference is not None
                                  else 'blocked_incomplete_method_or_reference'})
                    if current is None or reference is None:
                        continue
                    for metric, target in current.items():
                        other = reference[metric]
                        if target.keys() != other.keys() or any(target[k][1] != other[k][1] for k in target):
                            raise ValueError('Reference comparison is not source-consistent and condition-paired')
                        delta = {k: (target[k][0] - other[k][0], target[k][1]) for k in target}
                        stats = estimate(delta, seed)
                        comparisons.append({'stratum': stratum, 'label': label, 'method': method,
                            'tau': tau, 'reference': reference_name, 'reference_method': reference_method,
                            'reference_tau': reference_tau, 'metric': metric,
                            'delta_method_minus_reference': stats.pop('mean'), **stats})
    check_overall_against_canonical(metrics, args.causal / 'causal_metrics.csv')
    check_overall_against_canonical(comparisons, args.causal / 'paired_comparisons.csv', paired=True)
    routing, calibration = [], []
    for stratum, label, identifiers in strata:
        baseline = grouped.get(('direct_score_bias', 0.), {})
        for method in methods:
            pooled_calibration = {}
            for tau in taus:
                if (stratum, method, tau) not in completed_samples:
                    continue
                index = {key: row for key, row in grouped[(method, tau)].items() if key[0] in identifiers}
                if (stratum, 'direct_score_bias', 0.) in completed_samples:
                    for variant in ('all', 'positive', 'negative'):
                        subset = {key: row for key, row in index.items()
                                  if variant == 'all' or row['variant'] == variant}
                        for metric in ('signed_target_mass_change', 'raw_target_mass_change', 'gold_mass_change'):
                            values = {}
                            for key, row in subset.items():
                                base, sign = baseline[key], 1. if row['variant'] == 'positive' else -1.
                                field = 'gold_attention_mass' if metric == 'gold_mass_change' else 'target_attention_mass'
                                value = float(row[field]) - float(base[field])
                                values[key] = (sign * value if metric == 'signed_target_mass_change' else value,
                                               row['group_id'])
                            routing.append({'stratum': stratum, 'label': label, 'method': method, 'tau': tau,
                                'variant': variant, 'metric': metric, **estimate(values, seed),
                                'reference': 'matched direct_score_bias zero dose',
                                'scope': 'one diagnostic layer/KV head; average over its shared GQA query heads'})
                if tau == 0:
                    continue
                ratio_values = {}
                for key, row in index.items():
                    sign = 1. if row['variant'] == 'positive' else -1.
                    rms = float(row['matched_score_rms'])
                    ratio = sign * float(row['fixed_query_mean']) / rms if rms > 0 else float('nan')
                    if not math.isfinite(ratio) or abs(ratio) > 1.0001:
                        raise ValueError('Invalid signed mean/RMS calibration diagnostic')
                    ratio_values[key] = (ratio, row['group_id'])
                    pooled_calibration[(*key, tau)] = (ratio, row['group_id'])
                stats = estimate(ratio_values, seed)
                calibration.append({'stratum': stratum, 'label': label, 'method': method,
                    'dose_scope': 'per_dose', 'tau': tau, 'metric': 'oracle_sign_times_score_mean_over_score_rms',
                    **stats, 'min': min(x[0] for x in ratio_values.values()),
                    'max': max(x[0] for x in ratio_values.values())})
            # Pool only when every frozen nonzero dose is complete.
            if pooled_calibration and all((stratum, method, tau) in completed_samples for tau in taus if tau > 0):
                stats = estimate(pooled_calibration, seed)
                calibration.append({'stratum': stratum, 'label': label, 'method': method,
                    'dose_scope': 'all_nonzero', 'tau': '', 'metric': 'oracle_sign_times_score_mean_over_score_rms',
                    **stats, 'min': min(x[0] for x in pooled_calibration.values()),
                    'max': max(x[0] for x in pooled_calibration.values())})
    if not metrics or not comparisons or not coverage:
        raise ValueError('No complete zero-dose estimates and paired references are available for this report')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('metrics.csv', metrics), ('paired_comparisons.csv', comparisons),
                       ('coverage.csv', coverage), ('paired_coverage.csv', paired_coverage),
                       ('routing_diagnostics.csv', routing), ('calibration_diagnostics.csv', calibration)]:
        if rows:
            write_csv(args.output / name, rows, list(rows[0]))
    summary = {(r['stratum'], r['method'], r['tau'], r['metric']): r for r in metrics}
    text = ['# 模块二确认结果的预定分层', '',
            '只读完整运行。保留全部冻结方法与剂量，不选择最优配置。准确率以指定目标观测计数；配对全对要求同一目标正负两侧均正确。', '',
            '总体及两个动作映射分层按原始指南等权。聚类区间与配对差见 CSV；只有三个指南，区间是探索性描述。单指南不生成聚类区间。', '']
    for stratum, label, identifiers in strata:
        text += [f'## {label}', '', '| 方法 | 剂量 | 正确观测 | 配对全对 | 正确规则平均分数间隔 | 正确规则注意力质量 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for method in methods:
            for tau in taus:
                case = summary.get((stratum, method, tau, 'case_accuracy'))
                if case is None:
                    text.append(f'| {METHOD_NAMES.get(method, method)} | {tau:g} | 不完整，见覆盖表 | — | — | — |')
                    continue
                index = {key: row for key, row in grouped[(method, tau)].items() if key[0] in identifiers}
                correct = sum(bool(row['correct']) for row in index.values())
                pair_correct = sum(all(index[(i, v)]['correct'] for v in ('positive', 'negative')) for i in identifiers)
                margin = summary[(stratum, method, tau, 'gold_logit_margin')]['mean']
                mass = summary[(stratum, method, tau, 'gold_attention_mass')]['mean']
                text.append(f'| {METHOD_NAMES.get(method, method)} | {tau:g} | {correct}/{len(index)} | {pair_correct}/{len(identifiers)} | {margin:.9f} | {mass:.9f} |')
        text.append('')
    (args.output / 'STRATA.zh.md').write_text('\n'.join(text) + '\n')
    manifest = {'schema_version': 'module2.causal_strata.v1', 'status': 'completed',
        'source_run_status': status['status'], 'model_run': str(args.causal), 'methods': methods, 'taus': taus,
        'seed': seed, 'bootstrap_draws': 2000, 'pairing': 'triplet_id+variant; pair accuracy by triplet_id',
        'weighting': 'equal source-guideline means within stratum',
        'single_guideline_intervals': 'omitted; descriptive only',
        'routing_diagnostic': 'oracle_sign * (target attention mass minus matched zero-dose target mass); also positive/negative strata',
        'calibration_diagnostic': 'oracle_sign * fixed-query score mean / actual score RMS; cosine to the signed constant score pattern',
        'calibration_pooling': 'all frozen nonzero doses only when all complete; repeated doses do not add independent inputs',
        'incomplete_policy': 'any incomplete global method/dose suppresses every corresponding stratum estimand',
        'head_or_dose_reselection': False, 'canonical_overall_recomputation': 'matched within 1e-12',
        'strata': [{'stratum': key, 'label': label, 'triplets': len(ids)} for key, label, ids in strata],
        'source_sha256': {str(path): digest(path) for path in [args.triplets, args.causal / 'status.json',
            args.causal / 'plan.json', args.causal / 'predictions.jsonl', args.causal / 'causal_metrics.csv',
            args.causal / 'paired_comparisons.csv', Path(__file__)]}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'strata': len(strata), 'metrics': len(metrics),
                      'paired_comparisons': len(comparisons), 'output': str(args.output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal', type=Path, required=True)
    parser.add_argument('--triplets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
