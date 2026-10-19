#!/usr/bin/env python3
"""Evaluate frozen V8.1 predictions; selection reads only dev96 outputs."""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = HERE.parent / 'care_module1_semantic_v8'
ARMS = ('A1', 'B1', 'B2')
spec = importlib.util.spec_from_file_location('v8_metrics', OLD / 'analyze.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
read, write = legacy.read, legacy.write


def dataset(split):
    if split in ('dev', 'confirmation'):
        rows = [r for r in read(REPO / 'experiments/care_reliability_20261019/data/mechanism_cases.jsonl') if r['split'] == split]
        inputs = {r['case_id']: r for r in read(OLD / 'data' / f'semantic_{split}_inputs.jsonl')}
        return [dict(r, rules=inputs[r['case_id']]['rules']) for r in rows]
    root = OLD if split == 'new72' else HERE
    rows = read(root / 'data' / f'{split}_cases.jsonl')
    gold = {r['case_id']: r for r in read(root / 'data' / f'{split}_gold.jsonl')}
    return [dict(r, **{k: v for k, v in gold[r['case_id']].items() if k != 'case_id'}) for r in rows]


def source(split, arm):
    path = HERE / 'results' / f'{split}_{arm}' / 'predictions.jsonl'
    if arm == 'A1' and split != 'natural72' and not path.exists():
        path = OLD / 'results' / f'{split}_{arm}' / 'predictions.jsonl'
    return path


def evaluate(cases, rows, historical_reuse=False):
    m, records = legacy.evaluate(cases, rows)
    m['F_to_T_rate'] = m['F_to_T'] / m['F_to_T_denominator'] if m['F_to_T_denominator'] else None
    m['U_recall'] = m['per_state']['U']['recall']
    m['all_rules_case_accuracy'] = m['all_rules_correct_cases'] / m['cases']
    for prefix in ('target_pair', 'flipped_rule_pair', 'all_rules_pair'):
        m[prefix + '_accuracy'] = m[prefix + '_correct'] / m[prefix + '_total'] if m[prefix + '_total'] else None
    m['input_tokens'] = sum(sum(a.get('input_tokens', 0) for a in r.get('attempts', [])) for r in rows)
    m['output_tokens'] = sum(sum(a.get('output_tokens', 0) for a in r.get('attempts', [])) for r in rows)
    m['total_tokens'] = m['input_tokens'] + m['output_tokens']
    for field in ('input_tokens', 'output_tokens'):
        m['added_' + field] = sum(r.get('added_' + field, sum(a.get(field, 0) for a in r.get('attempts', []) if not a.get('reused', False))) for r in rows)
    m['added_total_tokens'] = m['added_input_tokens'] + m['added_output_tokens']
    m['mean_tokens_per_case'] = m['total_tokens'] / m['cases']
    m['mean_calls_per_case'] = m['model_calls'] / m['cases']
    m['added_model_calls'] = sum(r.get('added_model_call_count', sum(not a.get('reused', False) for a in r.get('attempts', []))) for r in rows)
    m['added_inference_seconds'] = sum(r.get('added_inference_seconds', sum(a.get('generation_seconds', 0) for a in r.get('attempts', []) if not a.get('reused', False))) for r in rows)
    if historical_reuse:
        for field in ('added_model_calls', 'added_inference_seconds', 'added_input_tokens', 'added_output_tokens', 'added_total_tokens'):
            m[field] = 0
    byid = {c['case_id']: c for c in cases}
    for r in records:
        c = byid[r['case_id']]
        r.update(patient=c['patient'], rule_text=next((x['text'] for x in c['rules'] if x['rule_id'] == r['rule_id']), ''), pair_id=c.get('pair_id'))
        if not r['correct']:
            r['error_type'] = f"{r['gold']}→{r['predicted']}"
    return m, records


def select():
    metrics, hashes = {}, {}
    cases = dataset('dev')
    for arm in ARMS:
        path = source('dev', arm)
        rows = read(path)
        assert len(rows) == len(cases) and {r['case_id'] for r in rows} == {c['case_id'] for c in cases}, f'Incomplete dev {arm}'
        metrics[arm], _ = evaluate(cases, rows, historical_reuse=arm == 'A1')
        hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    order = sorted(ARMS, key=lambda a: (-metrics[a]['accuracy'], metrics[a]['F_to_T_rate'], -metrics[a]['U_recall'], metrics[a]['mean_calls_per_case'], metrics[a]['mean_tokens_per_case'], a))
    result = dict(selected=order[0], ranked=order, based_only_on='dev96', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), selection_order=['highest rule accuracy', 'lowest F→T rate', 'highest U recall', 'lowest calls per case', 'lowest tokens per case', 'arm name'], metrics=metrics, prediction_sha256=hashes)
    path = HERE / 'selection.json'
    if path.exists():
        frozen = json.loads(path.read_text())
        assert frozen['prediction_sha256'] == hashes and frozen['selected'] == result['selected'], 'Frozen dev selection differs'
        return frozen
    write(path, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', action='store_true', help='Evaluate dev only and freeze selection before holdout inference')
    args = ap.parse_args()
    if args.select:
        result = select()
        print(json.dumps({'selected': result['selected'], 'based_only_on': 'dev96'}))
        return
    frozen = json.loads((HERE / 'selection.json').read_text())
    summaries, details, family, case_details = {}, {}, [], []
    incomplete = []
    for split in ('dev', 'confirmation', 'new72', 'natural72'):
        cases = dataset(split)
        for arm in ARMS:
            path = source(split, arm)
            if not path.exists():
                incomplete.append(f'{split}_{arm}'); continue
            rows = read(path)
            if len(rows) != len(cases) or {r['case_id'] for r in rows} != {c['case_id'] for c in cases}:
                incomplete.append(f'{split}_{arm}'); continue
            key = f'{split}_{arm}'
            m, rs = evaluate(cases, rows, historical_reuse=arm == 'A1' and split != 'natural72')
            m.update(split=split, arm=arm, prediction_file=str(path.relative_to(REPO)), prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            m['historical_reuse'] = arm == 'A1' and split != 'natural72'
            if m['historical_reuse']:
                m['added_model_calls'] = 0; m['added_inference_seconds'] = 0
                m['added_input_tokens'] = m['added_output_tokens'] = m['added_total_tokens'] = 0
            manifest_path = path.parent / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                m['run_elapsed_seconds'] = manifest.get('elapsed_seconds')
                m['peak_allocated_gib'] = manifest.get('peak_allocated_gib')
            summaries[key], details[key] = m, rs
            grouped = collections.defaultdict(list)
            for r in rs: grouped[r['case_id']].append(r)
            for cid, rr in grouped.items():
                case_details.append(dict(split=split, arm=arm, case_id=cid, all_rules_correct=all(r['correct'] for r in rr), rules=rr))
            for fid in sorted({c['family_id'] for c in cases}):
                subset = [c for c in cases if c['family_id'] == fid]
                ids = {c['case_id'] for c in subset}
                fm, _ = evaluate(subset, [r for r in rows if r['case_id'] in ids], historical_reuse=m['historical_reuse'])
                family.append(dict(split=split, arm=arm, family_id=fid, **fm))
    out = HERE / 'analysis'; out.mkdir(exist_ok=True)
    write(out / 'metrics.json', summaries)
    write(out / 'family_metrics.json', family)
    write(out / 'completion.json', dict(complete=not incomplete, missing=incomplete, selected=frozen['selected']))
    for name, values in [('rule_predictions', [dict(experiment=k, **r) for k, rs in details.items() for r in rs]), ('case_predictions', case_details), ('errors', [dict(experiment=k, **r) for k, rs in details.items() for r in rs if not r['correct']])]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
    comparisons = {}
    for split in ('dev', 'confirmation', 'new72', 'natural72'):
        for arm in ('B1', 'B2'):
            left, right = f'{split}_A1', f'{split}_{arm}'
            if left not in details or right not in details: continue
            baseline = {(r['case_id'], r['rule_id']): r for r in details[left]}
            counts, changes = collections.Counter(), []
            for r in details[right]:
                a = baseline[r['case_id'], r['rule_id']]
                category = 'both_correct' if a['correct'] and r['correct'] else 'A1_only_correct' if a['correct'] else f'{arm}_only_correct' if r['correct'] else 'both_wrong'
                counts[category] += 1
                if a['predicted'] != r['predicted']: changes.append(dict(**r, A1_predicted=a['predicted'], A1_correct=a['correct']))
            comparisons[right + '_vs_A1'] = dict(counts=counts, changes=changes)
    write(out / 'paired_comparison.json', comparisons)
    transitions = {}
    parser_spec = importlib.util.spec_from_file_location('v8_parser', OLD / 'prompts.py')
    parser_module = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser_module)
    for split in ('dev', 'confirmation', 'new72', 'natural72'):
        key = f'{split}_B2'
        if key not in details: continue
        rows = {r['case_id']: r for r in read(source(split, 'B2'))}
        initial = {}
        for cid, row in rows.items():
            attempts = [a for a in row['attempts'] if a.get('stage') == 'initial']
            if not attempts: raise ValueError(f'Missing initial stage for {cid}')
            parsed, _ = parser_module.parse_output(attempts[-1]['raw_output'], [r['rule_id'] for r in row['rules']])
            initial[cid] = {r['rule_id']: r['state'] for r in (parsed or [])}
        counts = dict(corrected=0, broken=0, stayed_wrong=0, stayed_correct=0, F_to_T_added=0, F_to_T_removed=0)
        records = []
        for r in details[key]:
            before = initial[r['case_id']].get(r['rule_id'], 'ERROR')
            was_ok = before == r['gold']
            category = 'stayed_correct' if was_ok and r['correct'] else 'broken' if was_ok else 'corrected' if r['correct'] else 'stayed_wrong'
            counts[category] += 1
            added = r['gold'] == 'F' and before != 'T' and r['predicted'] == 'T'
            removed = r['gold'] == 'F' and before == 'T' and r['predicted'] != 'T'
            counts['F_to_T_added'] += added; counts['F_to_T_removed'] += removed
            records.append(dict(case_id=r['case_id'], rule_id=r['rule_id'], gold=r['gold'], initial=before, final=r['predicted'], transition=category, F_to_T_added=added, F_to_T_removed=removed))
        transitions[split] = dict(counts=counts, rules=records)
    write(out / 'review_transitions.json', transitions)
    ratios = {}
    for key, m in summaries.items():
        baseline = summaries.get(m['split'] + '_A1')
        if baseline is None: continue
        values = {field: m[field] / baseline[field] if baseline[field] else None for field in ('model_calls', 'input_tokens', 'output_tokens', 'total_tokens', 'summed_case_inference_seconds')}
        m['cost_ratio_to_A1'] = values
        ratios[key] = values
    write(out / 'cost_ratios.json', ratios)
    write(out / 'metrics.json', summaries)

    fields = ['split','arm','cases','rules','correct','accuracy','macro_f1','F_to_T','F_to_T_denominator','F_to_T_rate','U_recall','target_pair_correct','target_pair_total','target_pair_accuracy','all_rules_case_accuracy','model_calls','added_model_calls','input_tokens','output_tokens','total_tokens','added_input_tokens','added_output_tokens','added_total_tokens','summed_case_inference_seconds','added_inference_seconds']
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n'); writer.writeheader(); writer.writerows(summaries.values())
    lines = ['# V8.1 错误分析', '', f"开发集冻结选择：{frozen['selected']}。其余数据仅用于评价。", '', '|数据|方案|正确规则|F→T|U 召回|反事实配对|调用|输入/输出 token|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in summaries.values():
        lines.append(f"|{m['split']}|{m['arm']}|{m['correct']}/{m['rules']}|{m['F_to_T']}/{m['F_to_T_denominator']}|{m['U_recall']:.1%}|{m['target_pair_correct']}/{m['target_pair_total']}|{m['model_calls']}|{m['input_tokens']}/{m['output_tokens']}|")
    lines += ['', '耗时为实际批次 GPU 生成耗时分摊到病例后的总和；包含语义判断、复核和格式重试，不含加载模型。完整逻辑成本包括 B2 复用的 A1 首次判断，本次新增成本单独保存。历史与本次耗时来自不同运行，仅作描述性对照。', '', '## 错误分布', '']
    for key, rs in details.items():
        errors = [r for r in rs if not r['correct']]
        kinds = dict(collections.Counter(r['error_type'] for r in errors))
        fam = dict(collections.Counter(r['family_id'] for r in errors).most_common())
        lines += [f'### {key}', '', f'错误类型：{json.dumps(kinds, ensure_ascii=False)}', '', f'各病例组错误数：{json.dumps(fam, ensure_ascii=False)}', '']
        for r in errors[:5]:
            lines += [f"- {r['case_id']} / {r['rule_id']}：{r['gold']} → {r['predicted']}；模型理由：{r.get('reason_short') or '无'}"]
        lines.append('')
    lines += ['完整病例文本、规则文本、标签、预测和模型理由保存在 errors.jsonl；逐病例结果保存在 case_predictions.jsonl；所有规则结果保存在 rule_predictions.jsonl。', '这些病例属于人工编写的开发/验证数据，不能直接估计真实临床部署效果。']
    (out / 'ERROR_ANALYSIS.zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'selected': frozen['selected'], 'completed_arms': len(summaries), 'missing': incomplete}))


if __name__ == '__main__':
    main()
