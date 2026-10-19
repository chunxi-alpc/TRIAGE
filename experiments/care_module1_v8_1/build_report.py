#!/usr/bin/env python3
"""Render descriptive V8.1 results; the final adoption decision is authored separately."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'

def load(name):
    return json.loads((OUT / name).read_text())

def pct(x):
    return '—' if x is None else f'{x:.2%}'

def num(x):
    return '—' if x is None else f'{x:.2f}'

def cell(x):
    return str(x).replace('|', '\\|').replace('\n', ' ')

def main():
    metrics = load('metrics.json')
    selection = json.loads((HERE / 'selection.json').read_text())
    completion = load('completion.json')
    transitions = load('review_transitions.json')
    errors = [json.loads(s) for s in (OUT / 'errors.jsonl').read_text().splitlines() if s.strip()]
    lines = ['# CARE 模块一 V8.1 实验结果', '', '报告生成时间（Asia/Shanghai）：' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), '', f"开发集选择已冻结为 **{selection['selected']}**，冻结时间（UTC）：{selection['created_utc']}。", '', '本报告只陈述结果；是否采用 V8.1 的最终结论单独撰写。', '', '## 实验设置', '', '基于 PR #12 的 Phi-4＋TrialMatchAI LoRA。A1 使用原始语义判断；B1 在单次判断中要求检查所有条件、例外、否定和未知信息；B2 复用 A1 首次判断，再进行一次语义复核。没有加入 RAG、置信度或复杂规则门控。每阶段仅保留既有的一次格式重试。', '', '96 例开发集用于选择；历史 192 例、原新增 72 例和本次新写自然语言 72 例仅用于评价。每例含 3 条规则。选择按规则准确率、F→T 比率、U 召回率、完整调用成本与 token 成本依次排序，评价集不反向影响选择。历史 A1 结果直接复用；B2 复用对应 A1 输出作为首次判断，新增 GPU 工作为复核。自然语言 72 例的 A1、B1、B2 复核均实际执行新推理。', '']
    if not completion['complete']:
        lines += ['**结果尚未齐全：** ' + ', '.join(completion['missing']), '']
    lines += ['## 主要指标', '', 'T/F/U 是规则适用、不适用和信息不足。准确率按全部规则计算；格式错误计为错误，不记为 U。F→T 的分母为金标准 F 规则数。反事实配对要求同一对中目标规则的 T 与 F 均预测正确。', '', '|数据|方案|正确规则|准确率|宏平均 F1|F→T|F→T 比率|U 召回|目标配对|病例全部正确|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics.values():
        lines.append(f"|{m['split']}|{m['arm']}|{m['correct']}/{m['rules']}|{pct(m['accuracy'])}|{num(m['macro_f1'])}|{m['F_to_T']}/{m['F_to_T_denominator']}|{pct(m['F_to_T_rate'])}|{pct(m['U_recall'])}|{m['target_pair_correct']}/{m['target_pair_total']}|{pct(m['all_rules_case_accuracy'])}|")
    lines += ['', '## 各类别表现', '', '|数据|方案|类别|支持数|精确率|召回率|F1|', '|---|---|---|---:|---:|---:|---:|']
    for m in metrics.values():
        for state, v in m['per_state'].items():
            lines.append(f"|{m['split']}|{m['arm']}|{state}|{v['support']}|{pct(v['precision'])}|{pct(v['recall'])}|{num(v['f1'])}|")
    lines += ['', '## 混淆矩阵', '', '|数据|方案|金标准|预测 T|预测 F|预测 U|格式错误|', '|---|---|---|---:|---:|---:|---:|']
    for m in metrics.values():
        for state, v in m['confusion'].items():
            lines.append(f"|{m['split']}|{m['arm']}|{state}|{v['T']}|{v['F']}|{v['U']}|{v['ERROR']}|")
    lines += ['', '## 反事实配对', '', '|数据|方案|目标规则配对|全部规则配对|发生 T/F 翻转的规则配对|', '|---|---|---:|---:|---:|']
    for m in metrics.values():
        lines.append(f"|{m['split']}|{m['arm']}|{m['target_pair_correct']}/{m['target_pair_total']}|{m['all_rules_pair_correct']}/{m['all_rules_pair_total']}|{m['flipped_rule_pair_correct']}/{m['flipped_rule_pair_total']}|")
    lines += ['', '## 推理成本', '', '完整成本表示从头运行该方案所需的全部判断和格式重试；新增成本表示本次实际执行的工作。token 为模型 tokenizer 计数。时间为真实 GPU 批次生成时间分摊后的合计，不含模型加载。历史与本次批次、并发和硬件运行条件可能不同，因此时间比只作描述性比较。', '', '|数据|方案|完整/新增调用|完整输入/输出 token|新增输入/输出 token|完整/新增生成秒|调用比 A1|token 比 A1|时间比 A1|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics.values():
        r = m.get('cost_ratio_to_A1', {})
        lines.append(f"|{m['split']}|{m['arm']}|{m['model_calls']}/{m['added_model_calls']}|{m['input_tokens']}/{m['output_tokens']}|{m['added_input_tokens']}/{m['added_output_tokens']}|{num(m['summed_case_inference_seconds'])}/{num(m['added_inference_seconds'])}|{num(r.get('model_calls'))}|{num(r.get('total_tokens'))}|{num(r.get('summed_case_inference_seconds'))}|")
    lines += ['', '## B2 复核改变了什么', '', '直接比较 B2 自身首次判断与复核后的规则状态。纠正为错→对，改错为对→错，保持错包含错→另一个错。F→T 修正表示原来的 F→T 不再出现，可能改成 F 或 U；具体状态保存在 review_transitions.json。', '', '|数据|纠正|改错|保持错|保持对|新增 F→T|消除 F→T|净增加正确|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, t in transitions.items():
        c = t['counts']
        lines.append(f"|{split}|{c['corrected']}|{c['broken']}|{c['stayed_wrong']}|{c['stayed_correct']}|{c['F_to_T_added']}|{c['F_to_T_removed']}|{c['corrected']-c['broken']}|")
    lines += ['', '## 错误案例', '', '以下按每个数据集和方案选取至多三个不同病例的错误，优先显示 F→T，其次显示 U 误识别；不是随机样本。完整错误在 analysis/errors.jsonl。', '']
    for key in metrics:
        candidates = sorted((r for r in errors if r['experiment'] == key), key=lambda r: (0 if r['gold'] == 'F' and r['predicted'] == 'T' else 1 if r['gold'] == 'U' else 2, r['case_id'], r['rule_id']))
        seen = set()
        for r in candidates:
            if r['case_id'] in seen: continue
            seen.add(r['case_id'])
            lines += [f"### {key}：{r['case_id']} / {r['rule_id']}", '', f"金标准 **{r['gold']}**，预测 **{r['predicted']}**。", '', '**病例：** ' + cell(r['patient']), '', '**规则：** ' + cell(r['rule_text']), '', '**模型理由：** ' + cell(r.get('reason_short') or '无'), '']
            if len(seen) >= 3: break
    lines += ['## 数据与解释限制', '', '历史 192 例及原新增 72 例已用于前序实验，属于复用对照，不能称为全新盲测。本次自然语言 72 例在推理前冻结，标签由人工设定的观察事实和既有三值逻辑生成，不由被测模型生成；它们仍是合成病例和虚构分流规则，不代表真实临床指南或临床患者分布。', '', '病例包含反事实配对和重复主题，同一主题内的结果相关，不能把每条规则视为独立抽样来夸大统计证据。自然语言病例与原实验使用相同规则语义体系，因此结果说明叙述变化下的表现，不能单独证明跨任务泛化。', '', '复核可能纠正错误，也可能把正确答案改错；成本和 F→T/U 表现需要与总准确率一起解释。开发集选出的方案保持冻结，不因后续某个数据集较好而重新挑选。', '', '## 复现及结果文件', '', '- `selection.json`：开发集选择及输入预测哈希。', '- `analysis/metrics.json`、`summary.csv`、`family_metrics.json`：完整指标和病例组指标。', '- `analysis/case_predictions.jsonl`、`rule_predictions.jsonl`、`errors.jsonl`：逐例、逐规则及错误明细。', '- `analysis/paired_comparison.json`：B1/B2 相对 A1 的逐规则变化。', '- `analysis/review_transitions.json`：B2 首次判断到复核的变化。', '- `analysis/cost_ratios.json`：各数据集相对 A1 的完整成本比。', '- `results/*/predictions.jsonl`：模型原始输出、提示、token 和实际 GPU 时间。', '', '所有推理完成后在实验环境运行 `python experiments/care_module1_v8_1/analyze.py`，随后运行 `python experiments/care_module1_v8_1/build_report.py`。开发集选择命令为 `python experiments/care_module1_v8_1/analyze.py --select`，应在其他数据集推理开始前执行。', '']
    (HERE / 'RESULTS.zh.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(HERE / 'RESULTS.zh.md'), 'complete': completion['complete']}))

if __name__ == '__main__':
    main()
