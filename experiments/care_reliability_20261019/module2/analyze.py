#!/usr/bin/env python3
"""Analyze only completed real inference; the two dataset modes cannot be pooled."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run import METHODS, read, rows, save, digest

CORE_METRICS = ('paired_accuracy', 'accuracy', 'app_accuracy', 'nonapp_accuracy',
                'h_accuracy', 'misuse', 'gold_logit_margin', 'gold_probability_margin')
LEGACY_METRICS = ('eligible_accuracy', 'eligible_pair_accuracy', 'no_conflict_accuracy', 'no_conflict_retention')


def ratio(num, den):
    return num / den if den else None


def nd(records, metric, baseline):
    if metric in ('paired_accuracy', 'eligible_pair_accuracy'):
        groups = defaultdict(list)
        for record in records:
            groups[record['pair_id']].append(record)
        pairs = [pair for pair in groups.values() if len(pair) == 2
                 and (metric == 'paired_accuracy' or all(row.get('eligible') is True for row in pair))]
        return sum(all(row['correct'] for row in pair) for pair in pairs), len(pairs)
    if metric == 'all_variant_pair_accuracy':
        groups = defaultdict(list)
        for record in records:
            if record.get('variant_group_id') is None:
                raise ValueError('Sealed robustness metric requires variant_group_id')
            groups[record['variant_group_id']].append(record)
        # Do not silently drop incomplete variants.
        if any(len(group) != 6 or len({r['pair_id'] for r in group}) != 3 for group in groups.values()):
            raise ValueError('Each canonical base pair must retain all three two-member variants')
        return sum(all(row['correct'] for row in group) for group in groups.values()), len(groups)
    subset = records
    if metric == 'app_accuracy':
        subset = [row for row in records if row['applicable']]
    elif metric in ('nonapp_accuracy', 'misuse'):
        subset = [row for row in records if not row['applicable']]
    elif metric == 'eligible_accuracy':
        subset = [row for row in records if row.get('eligible') is True]
    elif metric == 'no_conflict_accuracy':
        subset = [row for row in records if row.get('no_conflict') is True]
    elif metric == 'no_conflict_retention':
        subset = [row for row in records if row.get('no_conflict') is True and baseline[row['case_id']]['correct']]
    key = metric if metric in ('misuse', 'gold_logit_margin', 'gold_probability_margin') else 'correct'
    return sum(row[key] for row in subset), len(subset)


def point(records, metric, baseline):
    if metric == 'h_accuracy':
        a, b = [ratio(*nd(records, m, baseline)) for m in ('app_accuracy', 'nonapp_accuracy')]
        if a is None or b is None:
            return None
        return 2*a*b/(a+b) if a+b else 0.0
    return ratio(*nd(records, metric, baseline))


def resampled(records, metric, families, weights, baseline):
    if metric == 'h_accuracy':
        a = resampled(records, 'app_accuracy', families, weights, baseline)
        b = resampled(records, 'nonapp_accuracy', families, weights, baseline)
        return np.divide(2*a*b, a+b, out=np.zeros_like(a), where=(a+b) > 0)
    counts = np.asarray([nd([row for row in records if row['group_id'] == family], metric, baseline)
                         for family in families], dtype=float)
    counts = weights @ counts
    return np.divide(counts[:, 0], counts[:, 1], out=np.full(len(counts), np.nan), where=counts[:, 1] > 0)


def interval(values, confidence=0.95):
    values = np.asarray(values)
    if not np.isfinite(values).any():
        return [None, None]
    tail = (1-confidence)*50
    return [float(x) for x in np.nanpercentile(values, [tail, 100-tail])]


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def validate_zero_checks(zero, fold_records, protocol):
    """Require every input's mask-expansion zero and the ten first-case key zeros."""
    if zero.get('status') != 'passed_real_model_checks' or zero.get('policy') != protocol['zero_checks']:
        raise AssertionError('Missing real-model zero checks or frozen zero policy')
    base_records = [row for row in fold_records if row['method'] == 'baseline']
    baseline = {row['case_id']: row for row in base_records}
    if not baseline or len(baseline) != len(base_records):
        raise AssertionError('Missing/duplicate baseline cases in zero validation')
    first = min(baseline)
    expected = {(case_id, 'direct_bias') for case_id in baseline}
    expected.update((first, method) for method in METHODS if method != 'direct_bias')
    checks = zero.get('checks', [])
    observed = {(entry['case_id'], entry['method']) for entry in checks}
    if (len(checks) != len(expected) or observed != expected
            or zero.get('expected_direct_bias_cases') != len(baseline)):
        raise AssertionError('Incomplete/duplicate per-case direct-bias zero coverage')
    for check in checks:
        base = baseline[check['case_id']]
        if (check.get('identical_candidate_logits') is not True
                or check.get('identical_candidate_probabilities') is not True
                or check.get('candidate_logits') != base['candidate_logits']
                or check.get('candidate_probabilities') != base['candidate_probabilities']
                or check.get('tau') != 0.0):
            raise AssertionError('A real-model zero check failed exact candidate equality')
        n = check.get('prompt_tokens')
        if type(n) is not int or not 0 < n <= protocol['maximum_prompt_tokens'] or n != base.get('prompt_tokens'):
            raise AssertionError('Zero check did not record the same input sequence length')
        if check.get('mask_head_expansion_checked') is not (check['method'] == 'direct_bias'):
            raise AssertionError('Zero check mask-expansion scope is inconsistent')
    return len(checks)


def validate_journals(run_root):
    manifests, data, protocols = [], [], []
    for fold in (0, 1, 2):
        folder = run_root / f'fold{fold}'
        manifest = read(folder / 'run_manifest.json')
        if manifest.get('status') != 'completed_real_inference' or manifest.get('real_model_inference_completed') is not True:
            raise RuntimeError(f'{folder} has not completed real inference; no performance table will be produced')
        protocol = read(folder / 'protocol_snapshot.json')
        fold_records = rows(folder / 'predictions.jsonl')
        if len(fold_records) != manifest['records']:
            raise AssertionError('Journal row count mismatch')
        zero_count = validate_zero_checks(read(folder / 'zero_checks.json'), fold_records, protocol)
        if (manifest.get('zero_check_forwards') != zero_count
                or manifest.get('actual_model_forwards') !=
                manifest['training_cases'] + len(fold_records) + zero_count - manifest['no_eligible_no_op_records']):
            raise AssertionError('Model forward accounting omitted or duplicated zero checks')
        compatibility = read(folder / 'model_compatibility.json')
        if (compatibility != manifest.get('model_compatibility')
                or compatibility.get('status') != 'passed_configuration_and_environment_gates'
                or compatibility.get('original_manifest_sha256') != protocol['model_manifest_sha256']
                or compatibility.get('loaded_attention') != protocol['attention']
                or compatibility.get('loaded_precision') != 'torch.' + protocol['precision']
                or compatibility['environment'].get('compatible') is not True
                or compatibility['environment'].get('blocking_differences') != {}
                or any(compatibility[stage].get('compatible') is not True
                       or compatibility[stage].get('differences') != {}
                       for stage in ('resolved_config', 'loaded_config'))):
            raise AssertionError('Frozen model configuration/environment gate did not pass')
        manifests.append(manifest)
        data.extend(fold_records)
        protocols.append(protocol)
    if not all(p == protocols[0] for p in protocols):
        raise AssertionError('Protocol changed between folds')
    if len({m['mode'] for m in manifests}) != 1 or len({m['data_sha256'] for m in manifests}) != 1:
        raise AssertionError('Dataset modes or frozen source files were mixed')
    if len({m['model_revision'] for m in manifests}) != 1:
        raise AssertionError('Model revision changed between folds')
    mode = manifests[0]['mode']
    method_rows = {method: [r for r in data if r['method'] == method] for method in ('baseline',) + METHODS}
    baseline = {r['case_id']: r for r in method_rows['baseline']}
    expected = set(baseline)
    if len(baseline) != len(method_rows['baseline']):
        raise AssertionError('Duplicate baseline case IDs')
    if mode == 'legacy384_regression' and len(expected) != 384:
        raise AssertionError('The historical regression must retain all 384 cases')
    for method, records in method_rows.items():
        if len(records) != len(expected) or {r['case_id'] for r in records} != expected:
            raise AssertionError(f'Incomplete/duplicate comparison condition: {method}')
        pairs = defaultdict(list)
        for r in records:
            b = baseline[r['case_id']]
            if any(r[key] != b[key] for key in ('pair_id', 'group_id', 'applicable', 'gold_answer', 'mode', 'fold', 'dataset_id', 'variant', 'variant_group_id', 'prompt_tokens')):
                raise AssertionError('Case identity/labels differ across methods')
            if not np.isfinite(r['candidate_logits']).all() or not np.isfinite(r['candidate_probabilities']).all():
                raise AssertionError('Nonfinite candidate scores')
            pairs[r['pair_id']].append(r)
            if method != 'baseline':
                audit = r['disturbance']
                if method.endswith('_calibrated') and not audit['anchor_group_all_nonnegative']:
                    raise AssertionError('Realized anchor sign certificate failed')
                if audit['calibration_status'] == 'no_eligible_direction':
                    if r['candidate_logits'] != b['candidate_logits'] or audit['actual_rms'] != 0:
                        raise AssertionError('A policy no-op changed the model')
                elif audit['relative_error'] > protocols[0]['maximum_rms_relative_error']:
                    raise AssertionError('A nonzero run exceeded its fixed RMS budget tolerance')
        if any(len(pair) != 2 or {r['applicable'] for r in pair} != {True, False} for pair in pairs.values()):
            raise AssertionError('An incomplete applicability pair entered analysis')
    return mode, method_rows, baseline, protocols[0], manifests


def analyze(run_root, output):
    mode, all_methods, baseline, protocol, manifests = validate_journals(run_root)
    methods = all_methods
    if mode == 'sealed_mechanism':
        methods = {m: [r for r in records if r.get('variant') == 'canonical']
                   for m, records in all_methods.items()}
        if not methods['baseline'] or len(methods['baseline']) * 3 != len(all_methods['baseline']):
            raise AssertionError('Sealed primary analysis requires one canonical variant out of three')
    output.mkdir(parents=True, exist_ok=False)
    families = sorted({row['group_id'] for row in methods['baseline']})
    if set(families) != {row['group_id'] for row in all_methods['baseline']}:
        raise AssertionError('A family is missing its canonical primary variant')
    metrics = list(CORE_METRICS) + list(LEGACY_METRICS if mode == 'legacy384_regression' else ())
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    weights = rng.multinomial(len(families), np.full(len(families), 1/len(families)),
                              size=protocol['bootstrap_replicates'])
    draws = {(method, metric): resampled(records, metric, families, weights, baseline)
             for method, records in methods.items() for metric in metrics}
    statistics, family_rows = [], []
    for method, records in methods.items():
        for metric in metrics:
            value = point(records, metric, baseline)
            n = len(records) if metric == 'h_accuracy' else nd(records, metric, baseline)[1]
            ci = interval(draws[method, metric])
            statistics.append(dict(method=method, metric=metric, value=value, n=n, ci_low=ci[0], ci_high=ci[1]))
        for family in families:
            subset = [row for row in records if row['group_id'] == family]
            family_rows.append(dict(family=family, method=method,
                                    **{metric: point(subset, metric, baseline) for metric in metrics}))
    comparisons = []
    primary_ordered = [tuple(x) for x in protocol['co_primary_comparisons']]
    matched_ordered = [tuple(x) for x in protocol['matched_direction_comparisons']]
    # Co-primary contrasts must always be learned minus direct, independent
    # of the generic combinations' insertion order.
    pairs = list(itertools.combinations(methods, 2))
    explicit = primary_ordered + matched_ordered
    pairs = [(a, b) for a, b in pairs if (a, b) not in explicit and (b, a) not in explicit] + explicit
    for a, b in pairs:
        for metric in metrics:
            ap, bp = point(methods[a], metric, baseline), point(methods[b], metric, baseline)
            delta = None if ap is None or bp is None else bp-ap
            distribution = draws[b, metric] - draws[a, metric]
            ci95, ci975 = interval(distribution), interval(distribution, 0.975)
            primary = (a, b) in primary_ordered and metric == protocol['primary_metric']
            comparisons.append(dict(method_a=a, method_b=b, metric=metric, delta_b_minus_a=delta,
                                    ci95_low=ci95[0], ci95_high=ci95[1], co_primary=primary,
                                    matched_direction_comparison=(a, b) in matched_ordered,
                                    co_primary_ci975_low=ci975[0] if primary else None,
                                    co_primary_ci975_high=ci975[1] if primary else None))
    calibration = []
    for method, records in all_methods.items():
        if method == 'baseline':
            continue
        audits = [r['disturbance'] for r in records]
        matched = [a for a in audits if a['calibration_status'] == 'matched']
        entry = dict(method=method, analysis_scope='all scored cases, including secondary variants', n=len(records), matched_nonzero_n=len(matched),
                     no_eligible_no_op_n=sum(a['calibration_status'] == 'no_eligible_direction' for a in audits),
                     max_rms_relative_error=max((a['relative_error'] for a in matched), default=None),
                     p95_rms_relative_error=float(np.quantile([a['relative_error'] for a in matched], .95)) if matched else None)
        for key in ('anchor_nonnegative_fraction', 'anchor_strict_correct_fraction', 'anchor_nonzero_fraction',
                    'anchor_wrong_fraction', 'anchor_signed_mean', 'anchor_rms', 'anchor_absolute_max',
                    'retained_token_fraction', 'gqa_ideal_consensus_fraction', 'full_support_negative_fraction'):
            entry['mean_'+key] = float(np.mean([a[key] for a in audits]))
        calibration.append(entry)
    win_loss = []
    for a, b in primary_ordered:
        aa = {r['case_id']: r for r in methods[a]}
        bb = {r['case_id']: r for r in methods[b]}
        win_loss.append(dict(method_a=a, method_b=b,
                             cases_corrected=sum(bb[k]['correct'] and not aa[k]['correct'] for k in aa),
                             cases_broken=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa),
                             answer_changes=sum(aa[k]['prediction'] != bb[k]['prediction'] for k in aa)))
    all_variants = None
    variant_statistics, variant_comparisons = [], []
    if mode == 'sealed_mechanism':
        variant_metrics = CORE_METRICS + ('all_variant_pair_accuracy',)
        all_variants = {m: {metric: point(rs, metric, baseline) for metric in variant_metrics}
                        for m, rs in all_methods.items()}
        variant_draws = {(m, metric): resampled(rs, metric, families, weights, baseline)
                         for m, rs in all_methods.items() for metric in variant_metrics}
        for m, rs in all_methods.items():
            for metric in variant_metrics:
                ci = interval(variant_draws[m, metric])
                variant_statistics.append(dict(method=m, metric=metric, value=all_variants[m][metric],
                                               n=len(rs) if metric == 'h_accuracy' else nd(rs, metric, baseline)[1],
                                               ci95_low=ci[0], ci95_high=ci[1], analysis_scope='secondary_all_variants'))
        for a, b in primary_ordered:
            for metric in variant_metrics:
                ci = interval(variant_draws[b, metric]-variant_draws[a, metric])
                variant_comparisons.append(dict(method_a=a, method_b=b, metric=metric,
                                                delta_b_minus_a=all_variants[b][metric]-all_variants[a][metric],
                                                ci95_low=ci[0], ci95_high=ci[1], co_primary=False,
                                                analysis_scope='secondary_all_variants'))
    summary = dict(status='analysis_of_completed_real_inference', mode=mode,
                   clinical_efficacy_claim=False, cases=len(baseline),
                   pairs=len({r['pair_id'] for r in baseline.values()}), families=len(families),
                   primary_cases=len(methods['baseline']),
                   primary_pairs=len({r['pair_id'] for r in methods['baseline']}),
                   primary_population=protocol['primary_population'][mode],
                   canonical_base_pairs=(len({r['variant_group_id'] for r in baseline.values()})
                                         if mode == 'sealed_mechanism' else None),
                   oracle_applicability=True, oracle_action_span=True,
                   primary=[c for c in comparisons if c['co_primary']],
                   matched_direction_comparisons=protocol['matched_direction_comparisons'],
                   primary_interpretation='Two frozen learned-minus-direct contrasts; 97.5% marginal intervals implement Bonferroni alpha allocation. Small-family percentile intervals are approximate.',
                   confidence_interval_scope='Family bootstrap conditional on fitted old training folds; no refitting uncertainty. All other comparisons exploratory.',
                   selection='No new head, rank, strength, seed or test subset selection',
                   all_variants_secondary=all_variants,
                   source_journal_sha256={f'fold{f}': digest(run_root/f'fold{f}/predictions.jsonl') for f in range(3)},
                   analysis_source_sha256=digest(Path(__file__)),
                   source_data_sha256=manifests[0]['data_sha256'])
    save(output/'summary.json', summary)
    write_csv(output/'metrics_ci.csv', statistics)
    write_csv(output/'paired_comparisons.csv', comparisons)
    write_csv(output/'per_family.csv', family_rows)
    write_csv(output/'calibration.csv', calibration)
    write_csv(output/'primary_case_changes.csv', win_loss)
    write_csv(output/'all_variant_metrics_ci.csv', variant_statistics)
    write_csv(output/'all_variant_paired_comparisons.csv', variant_comparisons)
    text = ['# 模块二固定方向与查询校准实验', '',
            f"本报告仅来自已完成的真实模型推断。模式：`{mode}`。{summary['cases']} 例、{summary['pairs']} 个表面形式配对、{len(families)} 个家族。", '',
            f"主要终点人群：{summary['primary_cases']} 例、{summary['primary_pairs']} 个配对。新机制集只用基础形式作主要分析，全部表面变体为次要分析。", '',
            '主要比较为学习方向减直接分数偏置。两个预先固定的主要比较使用 97.5% 边际家族自助法区间；其余 95% 区间仅作探索性分析。', '',
            '| 方法 | 适用准确率 | 不适用准确率 | H | 成对准确率 |',
            '|---|---:|---:|---:|---:|']
    for method, records in methods.items():
        values = [point(records, metric, baseline) for metric in ('app_accuracy', 'nonapp_accuracy', 'h_accuracy', 'paired_accuracy')]
        text.append('| '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
    text += ['', '## 证据边界', '',
             '真实适用性和动作定位均为诊断输入，不能解释为模块一预测后的端到端效果。旧病例回归与新合成机制确认分别报告。', '',
             '全因果位置上的分数均方根匹配不构成决策锚点的方向保证。校准条件的锚点实际分数已在 BF16 计算路径后逐组核验；零覆盖仍可能得到无操作，必须结合 calibration.csv 中的有效覆盖和无操作比例判断。', '',
             '不同方向的 GQA 有效词元覆盖率可能不同。学习方向相对直接偏置的差异同时包含方向、校准与覆盖策略的影响，不能全部归因于子空间。', '',
             '本实验没有证据预算，也没有自由生成或 KV 缓存检验。家族自助法不包含重新拟合不确定性，少量家族不支持临床效果或总体等效结论。']
    if mode == 'sealed_mechanism':
        text += ['', f"新集的 {summary['pairs']} 个配对来自 {summary['canonical_base_pairs']} 个基础配对的三种表面形式。基础形式用于主要终点；全部表面形式和三种形式全部正确率分别保存在 all_variant_metrics_ci.csv。变体不视为独立家族。"]
    (output/'README.zh.md').write_text('\n'.join(text)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True, help='One mode root containing fold0/fold1/fold2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
