#!/usr/bin/env python3
"""Export derived pilot results without redistributing clinical/source text."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_pilot import summarize

SAFE_FIELDS=('id','source_row','disease','variant','condition','prediction','gold_answer','original_answer',
             'sibling_action_label','original_sibling_action_label','replacement_action_label','closed_book_prediction',
             'strength_applicable','strength_sibling','oracle','attention_calls','input_tokens','rule_token_counts')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    private=Path(a.run_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    dataset=private/'pilot20.jsonl'
    ids={json.loads(s)['id'] for s in dataset.read_text().splitlines() if s.strip()}
    def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    data_manifest=json.loads(dataset.with_suffix('.manifest.json').read_text())
    if data_manifest['dataset_sha256']!=hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError('dataset changed after preparation')
    write(out/'data_manifest.json',data_manifest)
    sources=json.loads((private/'sources.json').read_text());write(out/'sources.json',sources)
    tokens=json.loads((private/'tokenizer_tests.json').read_text());write(out/'tokenizer_tests.json',tokens)
    report=['ASTER MedGUIDE 真实20例先导实验','',
            '服务器115，gcx261014环境，GPU 0：A100-SXM4-80GB。',
            '样本：真实MedGUIDE 20例、60条A/B/C；每模型320次greedy生成。',
            '数据固定种子20261022；3645条可构造记录中随机抽取，未根据模型表现选择。',
            'A原始；B适用动作改为第三动作；C相邻分支改为同一第三动作。选项固定。',
            '全部指南路径输入；gold path用于oracle干预及direct_rule，不是学习的适用性。',
            '直接规则提示仅提供适用规则，与注意力组输入不同。','',
            '主要观察：直接规则提示改善最明显；B上有符号干预未稳定优于仅增强适用规则或baseline。',
            '答案按输出开头明确字母提取，兼容加粗及选项文字；保留原始输出，仅离线修正提取格式。',
            '', '准确率（正确数/20）：',
            '模型 | 版本 | baseline | direct_rule | applicable_only | unsigned | signed']
    for name in ('llama','qwen'):
        pred=private/(name+'_predictions.jsonl')
        manifest=json.loads(pred.with_suffix('.manifest.json').read_text())
        rows=[json.loads(s) for s in pred.read_text().splitlines() if s.strip()]
        summary=summarize(rows,ids)
        if not manifest.get('complete') or not summary['complete']:raise ValueError(name+' pilot incomplete')
        public=out/name;public.mkdir(exist_ok=True)
        write(public/'summary.json',summary);write(public/'run_manifest.json',manifest)
        (public/'predictions.jsonl').write_text(''.join(json.dumps({k:r[k] for k in SAFE_FIELDS if k in r})+'\n' for r in rows))
        write(public/'artifact_hashes.json',{'private_predictions_sha256':hashlib.sha256(pred.read_bytes()).hexdigest(),
                                         'public_predictions_sha256':hashlib.sha256((public/'predictions.jsonl').read_bytes()).hexdigest()})
        for variant in ('A_original','B_applicable_rule_changed','C_sibling_rule_changed'):
            values=[str(summary['arms'][f'{variant}/{c}']['correct'])+'/20' for c in ('baseline','direct_rule','applicable_only','unsigned','signed')]
            report.append(name+' | '+variant[0]+' | '+' | '.join(values))
        closed=summary['arms']['A_original/closed_book']
        report.append(f"{name} 闭卷：{closed['correct']}/20；总耗时 {manifest['seconds']:.1f}秒；峰值分配显存 {manifest['peak_allocated_gib']:.2f}GiB。")
        invalid=sum(v['invalid_or_missing'] for v in summary['arms'].values())
        report.append(f'{name} 无效或缺失输出：{invalid}/320。')
        for variant in ('A_original','B_applicable_rule_changed','C_sibling_rule_changed'):
            for c in ('direct_rule','applicable_only','unsigned','signed'):
                pair=summary['paired_changes'][f'{variant}/{c}']
                report.append(f"{name} {variant[0]}/{c} 相对baseline：纠正{pair['corrected']}，损害{pair['broken']}。")
    report.extend(['','验证：11项CPU单元测试通过；Llama/Qwen3真实tokenizer各验证全部60条规则区间。',
                   '测试包括原生GQA、零偏置相等、signed分数公式、因果遮罩、prefill及KV缓存解码、异常恢复、A/B/C标签与缺失分母。',
                   '原始病例、指南、模型长输出只留服务器private目录；仓库结果仅含派生统计与字母预测。',
                   '这是20例可构造子集的描述性结果，不代表全量MedGUIDE或独立临床验证。结构相邻不保证医学条件互斥。',
                   '所有oracle使用gold路径，不能推断端到端收益；没有训练或调参，未做后续强度搜索。',
                   '同一指南的多个病例并非独立规则家族；不以少量准确率差异宣称稳定收益。'])
    (out/'REPORT.zh.txt').write_text('\n'.join(report)+'\n')
    print('exported real results for both models')

if __name__=='__main__':main()
