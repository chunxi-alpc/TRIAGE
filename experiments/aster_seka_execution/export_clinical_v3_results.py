"""CPU recomputation and public export of the completed v3 clinical comparison."""
from collections import Counter
from clinical_v3_identity import index_predictions
import gzip
import json
import math
from pathlib import Path
import shutil

from mega_development_evaluation import digest, load_official, native_metrics, parse_with_official, read_jsonl

P = Path(__file__).resolve().parent
R = P / 'runtime'


def main():
    plan = json.loads((P / 'clinical_v3_plan.json').read_text())
    gen = R / 'clinical_v3_generation'
    evaluation = R / 'clinical_v3_evaluation'
    protocol = json.loads((gen / 'protocol.json').read_text())
    report = json.loads((evaluation / 'evaluation_report.json').read_text())
    cfg = json.loads((evaluation / 'evaluation_config.json').read_text())
    assert protocol['status'] == 'finished_pending_evaluation'
    assert report['status'] == 'complete_verified'
    assert protocol['completed_rows'] == 112
    frozen = json.loads(Path(plan['input_manifest']).read_text())
    expected = {row['sample_id'] for row in frozen['selected']}
    official, hashes = load_official(Path(plan['official_root']) / 'mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py')
    assert hashes == cfg['official_source_files_sha256']
    scores = read_jsonl(evaluation / 'case_metrics.jsonl')
    by_key = {(row['method'], row['sample_id']): row for row in scores}
    assert len(scores) == len(by_key) == 112
    predictions, generation_summary = [], {}
    for method in plan['methods']:
        path = gen / (method + '.predictions.private.jsonl')
        assert digest(path) == cfg['prediction_sha256'][method]
        rows = index_predictions(read_jsonl(path), frozen)
        raw = read_jsonl(evaluation / 'private' / (method + '_official.jsonl'))
        assert set(rows) == expected and {row['sample_id'] for row in raw} == expected and len(raw) == 16
        for item in raw:
            sid = item['sample_id']
            parsed, _ = parse_with_official(official['parser'], rows[sid])
            assert parsed == item['parsed']
            calculated = native_metrics(item['pathway'], item['accuracy'], official['summary'], official['metrics'])
            for metric in ['acc_normalized', 'pcc_dtw_cost']:
                assert math.isclose(calculated[metric], by_key[method, sid][metric], abs_tol=1e-12)
            original = rows[sid]; measured = original['generation']
            predictions.append({
                'sample_id': sid, 'method': method, 'status': original['status'],
                'prediction_text_sha256': __import__('hashlib').sha256(original['llm_output'].encode()).hexdigest(),
                'base_prompt_sha256': original['base_prompt_sha256'],
                'support_prompt_sha256': original['support_prompt_sha256'],
                'mapping_sha256': original['mapping_sha256'],
                'selected_span_count': original['selected_span_count'],
                **{k: measured.get(k) for k in ['output_tokens', 'actual_forward_calls', 'wall_seconds',
                                               'peak_allocated_gib', 'peak_reserved_gib', 'numerical_control']}})
        generation_summary[method] = {
            'cases': 16, 'status_counts': dict(Counter(row['status'] for row in rows.values())),
            'at_output_budget': sum(row['generation'].get('output_tokens') == 1024 for row in rows.values()),
            'forward_calls': sum(row['generation']['actual_forward_calls'] for row in rows.values()),
            'elapsed_seconds': sum(row['generation']['wall_seconds'] for row in rows.values()),
            'peak_reserved_gib': max(row['generation']['peak_reserved_gib'] for row in rows.values())}
    zero = [row for row in predictions if row['method'] == 'gain_0']
    zero_pass = all(row['numerical_control'] and row['numerical_control']['same_full_answer']
                    and row['numerical_control']['max_abs_first_token_score_difference'] == 0 for row in zero)
    nohook = {row['sample_id']: row for row in predictions if row['method'] == 'no_hook'}
    for row in zero:
        if row['prediction_text_sha256'] == nohook[row['sample_id']]['prediction_text_sha256']:
            for metric in ['acc_normalized', 'pcc_dtw_cost']:
                assert by_key['gain_0', row['sample_id']][metric] == by_key['no_hook', row['sample_id']][metric]
    out = P / 'clinical_v3_results_verified'
    out.mkdir(exist_ok=False)
    for name in ['case_metrics.jsonl', 'evaluation_config.json', 'evaluation_report.json',
                 'judge_cache_events.jsonl', 'judge_attempts.jsonl']:
        shutil.copyfile(evaluation / name, out / name)
    (out / 'predictions_metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    records = {}
    for name in ['clinical_v3_generation_job', 'clinical_v3_evaluation_job']:
        source = R / name
        manifest = json.loads((source / 'run_manifest.json').read_text())
        assert manifest['process_state'] == 'exited' and manifest['exit_code'] == 0
        folder = out / name; folder.mkdir()
        shutil.copyfile(source / 'run_manifest.json', folder / 'run_manifest.json')
        trace = source / 'forward_inputs.jsonl'
        if trace.exists():
            with trace.open('rb') as src, (folder / 'forward_inputs.jsonl.gz').open('wb') as dst:
                with gzip.GzipFile(fileobj=dst, mode='wb', mtime=0) as compressed:
                    shutil.copyfileobj(src, compressed)
            records[name] = {'trace_sha256_uncompressed': digest(trace), 'trace_bytes': trace.stat().st_size}
    audit = {'status': 'complete_verified', 'scope': '112 saved predictions and native metrics recomputed on CPU',
             'cases': 16, 'new_gpu_forwards': 0, 'zero_control_passed': zero_pass,
             'generation_summary': generation_summary, 'trace_archives': records,
             'plan_sha256': digest(P / 'clinical_v3_plan.json'), 'exporter_sha256': digest(Path(__file__)),
             'identity_resolver_sha256': digest(P / 'clinical_v3_identity.py'),
             'shared_preparation_cost_note': 'Include prior A0 extraction/condition costs referenced in each private prediction',
             'independent_confirmation': False, 'raw_text_published': False}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# 临床强度对照 v3：已完成开发结果', '',
             '16 例，seed=0，7 组，112 条预测。以下为连续评分，不是正确率或独立测试。', '',
             '| 方法 | 路径成本↓ | 答案评分↑ | 解析失败 |', '|---|---:|---:|---:|']
    for method, values in report['methods'].items():
        lines.append(f"| {method} | {values['PCC_DTW_cost_lower_better']:.6f} | {values['Acc_normalized_1_to_5_higher_better']:.6f} | {values['parse_failures']} |")
    lines += ['', f'零强度与无干预一致性：{zero_pass}。相同请求按评分序号复用相同评分；原始历史结果保留。',
              '分组置信区间、逐例评分、运行成本和指纹见 clinical_v3_results_verified/。',
              '本轮固定旧抽取和 A0 判断；覆盖不足没有被修复，不能将结果推广到完整数据集。',
              '模型同时承担生成与评分，仍有评分偏差；置信区间未完整纳入评分器不确定性。',
              '比较相同输入信息与输出长度上限；概率融合每步两次前向，计算预算并不完全相同，成本已单列。']
    (P / 'CLINICAL_V3_RESULTS.zh.md').write_text('\n'.join(lines) + '\n')
    ledger = P / 'task_ledger.json'; data = json.loads(ledger.read_text())
    for task in data['subtasks']:
        if task['id'] == 'clinical_strength_v3':
            task.update(status='complete_verified', evidence='clinical_v3_results_verified/audit.json',
                        execution_state='completed', completed_predictions=112, expected_predictions=112,
                        native_evaluation_status='complete_verified',
                        scope='112 fixed development predictions and scores; not independent test')
    ledger.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
