"""Write Chinese results only from recorded status, metrics and error diagnostics.

This reporting tool never reads model predictions to choose prompts, labels,
thresholds or methods. Absent or incomplete runs remain explicitly unreported.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
METHODS = (
    ('trialmatch_model_only', 'A：TrialMatchAI + 三值逻辑'),
    ('trialmatch', 'A+数值：TrialMatchAI + 程序比较'),
    ('combined', 'B：TrialMatchAI + BioLinkBERT + 程序比较'),
    ('qwen_v4', 'C：V4 已保存 Qwen3 直接判定'),
)
VARIANTS = (
    ('original', '完整原病例'), ('counterfactual', '条件翻转'),
    ('partial_missing', '信息缺失'), ('multiple_values', '多值与不同主体'),
    ('negation', '否定表达'), ('contradiction', '同次评估矛盾'),
)
CATEGORIES = (
    ('known_to_unknown', '已知 T/F 判为 U'), ('unknown_to_known', '未知 U 判为 T/F'),
    ('false_rejection_T_to_F', '错误拒绝 T→F'), ('false_acceptance_F_to_T', '错误接受 F→T'),
    ('evidence_error_known', '已知样本证据定位错误'),
)
ERROR_NAMES = dict(CATEGORIES) | {
    'evidence_localization_error': '判定可能正确，证据定位不符合参考',
    'known_polarity_error': '已知标签极性错误',
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_errors(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def number(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else '未采集'


def pct(value):
    return f'{100 * value:.2f}%' if isinstance(value, (int, float)) else '未计算'


def safe(value):
    return str(value).replace('|', '/').replace('\n', ' ').replace('\r', ' ')


def complete(row):
    return row.get('status') == 'complete'


def ratio(correct, count):
    if not isinstance(correct, (int, float)) or not isinstance(count, (int, float)) or count <= 0:
        return '未计算'
    return f'{correct}/{count} ({pct(correct / count)})'


def delta(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return '未计算'
    return f'{(new - old) * 100:+.2f} 个百分点'


def method_table(methods, keys=None):
    lines = ['| 方法 | 判定正确/准确率 | 宏 F1 | 已知证据定位 | 判定与证据同时正确 | 反事实成对正确 | 推理墙钟秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key, name in METHODS:
        if keys is not None and key not in keys:
            continue
        row = methods.get(key, {})
        if not complete(row):
            lines.append(f'| {name} | 未完成，不报告指标 | — | — | — | — | — |')
            continue
        evidence = ratio(row.get('known_evidence_correct'), row.get('known_evidence_n')) if row.get('evidence_available') else '不适用：无句级证据'
        joint = ratio(row.get('joint_correct'), row.get('n')) if row.get('joint_accuracy') is not None else '不适用'
        lines.append(f"| {name} | {ratio(row.get('correct'), row.get('n'))} | {number(row.get('macro_f1'), 4)} | {evidence} | {joint} | {ratio(row.get('counterfactual_pairs_correct'), row.get('counterfactual_pair_n'))} | {number(row.get('total_inference_seconds'))} |")
    return lines


def scenario_table(methods, expected=32):
    lines = ['| 场景 | A：TrialMatchAI | A+数值 | B：加 BioLinkBERT | C：历史 Qwen3 |',
             '|---|---:|---:|---:|---:|']
    for variant, name in VARIANTS:
        cells = []
        for key, _ in METHODS:
            method = methods.get(key, {})
            row = method.get('by_variant', {}).get(variant, {}) if complete(method) else {}
            cells.append(f"{ratio(row.get('correct'), row.get('n'))}；预测 U={row['unknown_predictions']}" if row else '未计算')
        lines.append(f"| {name}（{expected} 条） | " + ' | '.join(cells) + ' |')
    return lines


def category_table(methods):
    lines = ['| 错误类别 | A：TrialMatchAI | A+数值 | B：加 BioLinkBERT | C：历史 Qwen3 |',
             '|---|---:|---:|---:|---:|']
    for category, name in CATEGORIES:
        cells = []
        for key, _ in METHODS:
            row = methods.get(key, {})
            if not complete(row):
                cell = '未计算'
            elif category == 'evidence_error_known' and not row.get('evidence_available'):
                cell = '不适用'
            else:
                cell = str(row.get('error_categories', {}).get(category, 0))
            cells.append(cell)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return lines


def recorded_diagnosis(error):
    details = []
    for row in error.get('condition_diagnostics', []):
        if row.get('trialmatch_parse_error') or row.get('trialmatch_truncated') or row.get('conflict') or row.get('state') == 'U':
            details.append(row)
    if not details:
        details = error.get('condition_diagnostics', [])[:1]
    parts = []
    for row in details[:2]:
        part = f"{row.get('condition_id', '?')}: TrialMatchAI={row.get('trialmatch_state', '?')}, 数值={row.get('numeric_status', '?')}, NLI={row.get('nli_evidence_state', '?')}, 最终={row.get('state', '?')}"
        if row.get('trialmatch_parse_error'):
            part += f", 解析错误={row['trialmatch_parse_error']}"
        if row.get('trialmatch_truncated'):
            part += ', 达生成长度上限'
        if row.get('conflict'):
            part += ', 记录有模型冲突'
        if row.get('resolution'):
            part += f", resolution={row['resolution']}"
        parts.append(part)
    if not parts:
        parts = ['未保存条件级诊断']
    return safe('；'.join(parts))


def failure_examples(errors, limit=10):
    selected = []
    seen = set()
    # Prefer final-method decision failures; then different methods/scenarios.
    ordered = sorted(errors, key=lambda x: (x.get('gold') == x.get('prediction'), x.get('method') != 'combined', x.get('variant', ''), x.get('sample_id', '')))
    variants = set()
    for row in ordered:
        sample = row.get('sample_id')
        if sample in seen or row.get('variant') in variants:
            continue
        selected.append(row); seen.add(sample); variants.add(row.get('variant'))
        if len(selected) >= limit:
            return selected
    for row in ordered:
        if row.get('sample_id') not in seen:
            selected.append(row); seen.add(row.get('sample_id'))
        if len(selected) >= limit:
            break
    return selected


def conclusion_lines(methods, paired, exploratory, original_changes=None):
    raw = methods.get('trialmatch_model_only', {})
    numeric = methods.get('trialmatch', {})
    combined = methods.get('combined', {})
    qwen = methods.get('qwen_v4', {})
    lines = ['## 四个问题的回答', '', '**第一，TrialMatchAI 单独判断的效果如何？**']
    if complete(raw):
        cats = raw.get('error_categories', {})
        lines += [f"A 的判定正确为 {ratio(raw.get('correct'), raw.get('n'))}，宏 F1 {number(raw.get('macro_f1'), 4)}，反事实成对正确 {ratio(raw.get('counterfactual_pairs_correct'), raw.get('counterfactual_pair_n'))}。未知误判已知 {cats.get('unknown_to_known', 0)} 条、已知误判未知 {cats.get('known_to_unknown', 0)} 条、错误拒绝 {cats.get('false_rejection_T_to_F', 0)} 条。"]
        original = raw.get('by_variant', {}).get('original', {})
        if original:
            lines += [f"在保留完整原病例的 32 条样本上，正确 {ratio(original.get('correct'), original.get('n'))}；这 32 条仍是数据集的合成病例记录。"]
        if complete(numeric):
            lines += [f"增加程序数值比较后正确 {ratio(numeric.get('correct'), numeric.get('n'))}，相对纯模型准确率变化 {delta(numeric.get('accuracy'), raw.get('accuracy'))}；因此需分别评价专业模型和数值程序的贡献。"]
    else:
        lines += ['TrialMatchAI 尚未得到完整的已评价输出，当前不能判断其准确率或相对优势。']
    lines += ['', '**第二，加入 BioLinkBERT 后，证据与最终判断是否改善？**']
    if complete(combined) and complete(numeric):
        lines += [f"B 的准确率为 {pct(combined.get('accuracy'))}，相对 A+数值变化 {delta(combined.get('accuracy'), numeric.get('accuracy'))}；宏 F1 从 {number(numeric.get('macro_f1'), 4)} 变为 {number(combined.get('macro_f1'), 4)}。"]
        if paired:
            lines += [f"逐例比较纠正 {paired.get('combined_rescued', '未计算')} 条、损害原正确判断 {paired.get('combined_regressed', '未计算')} 条，共改变 {paired.get('prediction_changed', '未计算')} 条判断。"]
        lines += [f"B 的已知样本证据定位为 {ratio(combined.get('known_evidence_correct'), combined.get('known_evidence_n'))}，判定与证据同时正确为 {ratio(combined.get('joint_correct'), combined.get('n'))}。A+数值只提供数值句证据，其证据定位为 {ratio(numeric.get('known_evidence_correct'), numeric.get('known_evidence_n'))}；纯 TrialMatchAI 不生成句级证据，不能把该组的缺失定位指标写成 0%。"]
        for category, label in CATEGORIES[:4]:
            lines += [f"{label}：A+数值 {numeric.get('error_categories', {}).get(category, 0)} 条 → B {combined.get('error_categories', {}).get(category, 0)} 条。"]
        a_original = numeric.get('by_variant', {}).get('original', {})
        b_original = combined.get('by_variant', {}).get('original', {})
        if a_original and b_original:
            lines += [f"完整病例组正确数由 {a_original.get('correct')}/{a_original.get('n')} 变为 {b_original.get('correct')}/{b_original.get('n')}。"]
            if original_changes is not None:
                lines += [f"根据已完成逐例错误记录，原来正确的 {a_original.get('correct')} 条中保留正确 {original_changes['retained_correct']} 条、损害 {original_changes['regressed']} 条；另纠正原错误 {original_changes['rescued']} 条。"]
            else:
                lines += ['尚无完整逐例错误文件可计算原正确病例的保留交集，不能仅用两个正确总数断言所有原正确病例均保留。']
    else:
        lines += ['组合组或 A+数值组未完成评价，不能给出改善或损害结论。']
    lines += ['', '**第三，相对历史 Qwen3，准确率、证据和效率有什么变化？**']
    if complete(qwen) and complete(raw) and complete(combined):
        lines += [f"历史 Qwen3 直接判定准确率 {pct(qwen.get('accuracy'))}、宏 F1 {number(qwen.get('macro_f1'), 4)}。A 相对它的准确率变化为 {delta(raw.get('accuracy'), qwen.get('accuracy'))}；B 相对它变化为 {delta(combined.get('accuracy'), qwen.get('accuracy'))}。"]
        lines += [f"已知样本证据定位：Qwen3 {ratio(qwen.get('known_evidence_correct'), qwen.get('known_evidence_n'))}，B {ratio(combined.get('known_evidence_correct'), combined.get('known_evidence_n'))}，差 {delta(combined.get('known_evidence_accuracy'), qwen.get('known_evidence_accuracy'))}。联合正确率差 {delta(combined.get('joint_accuracy'), qwen.get('joint_accuracy'))}；反事实成对正确率差 {delta(combined.get('counterfactual_pair_accuracy'), qwen.get('counterfactual_pair_accuracy'))}。"]
        lines += [f"已记录推理墙钟：Qwen3 {number(qwen.get('total_inference_seconds'))} 秒，A {number(raw.get('total_inference_seconds'))} 秒，B {number(combined.get('total_inference_seconds'))} 秒。模型大小、量化、输出长度、条件请求数量和批量协议不同；这些是各自流程的实际时间，不能据此推断同条件下的纯模型或硬件速度优势。"]
    else:
        lines += ['V5 完整评价不足，不能相对历史 Qwen3 给出准确率、证据或效率的优劣结论；历史输出未重新运行。']
    lines += ['', '**第四，是否可以作为 CARE 正式模块一？**', '当前适合作为可复现的研究实现，尚不能据此批准正式替换。V4 的 16 条研究规则缺少完整原始指南动作，192 条样本已用于设计；新增探索集也只是明确合成事实，没有独立临床专家金标。离线手工结构准备也是正式部署需要解决的来源与维护负担。虽然接口提供适用性、原文字符位置和模块二/三兼容结构，仍需完整指南动作、未见过的真实完整病例和独立标注验证。']
    if any(complete(x) for x in exploratory.get('methods', {}).values()):
        eraw = exploratory.get('methods', {}).get('trialmatch_model_only', {})
        eb = exploratory.get('methods', {}).get('combined', {})
        if complete(eraw) and complete(eb):
            lines += [f"新增合成探索集 A 正确 {ratio(eraw.get('correct'), eraw.get('n'))}，B 正确 {ratio(eb.get('correct'), eb.get('n'))}；该结果支持的范围是这些新规则上的明确事实判断，不能升级为临床可靠性结论。"]
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-status', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--errors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predictions', type=Path, help='Optional saved public predictions used only to aggregate recorded timing fields')
    p.add_argument('--exploratory-metrics', type=Path, default=EXP/'exploratory/results/metrics.json')
    p.add_argument('--exploratory-status', type=Path, default=EXP/'exploratory/runs/frozen_v1/run_status.json')
    a = p.parse_args()
    status = read(a.run_status) or {}
    metrics = read(a.metrics) or {}
    methods = {key: dict(row) for key, row in metrics.get('methods', {}).items()}
    if status.get('status') != 'completed':
        for key, _ in METHODS[:3]:
            if key in methods:
                methods[key]['status'] = 'incomplete'
    exploratory = read(a.exploratory_metrics) or {}
    exploratory_status = read(a.exploratory_status) or {}
    exploratory['methods'] = {key: dict(row) for key, row in exploratory.get('methods', {}).items()}
    if exploratory_status.get('status') != 'completed':
        for row in exploratory['methods'].values():
            row['status'] = 'incomplete'
    errors = read_errors(a.errors)
    historic = read(EXP/'data/v4_reference_metrics.json') or {}
    run_dir = a.run_status.resolve().parent
    report_date = datetime.now(timezone.utc).date().isoformat()
    predictions_path = a.predictions or run_dir/'predictions_public.jsonl'
    recorded_rows = read_errors(predictions_path)
    timings = {}
    if status.get('status') == 'completed' and len(recorded_rows) == status.get('sample_count_completed'):
        for field in ('numeric_seconds', 'evidence_selection_seconds', 'mednli_seconds', 'postprocess_seconds'):
            values = [row.get(field) for row in recorded_rows]
            if values and all(isinstance(value, (int, float)) for value in values):
                timings[field] = sum(values)
    original_changes = None
    if a.errors.exists() and complete(methods.get('trialmatch', {})) and complete(methods.get('combined', {})):
        wrong = {key: {row['sample_id'] for row in errors if row.get('method') == key and row.get('variant') == 'original' and row.get('gold') != row.get('prediction')} for key in ('trialmatch', 'combined')}
        original_n = methods['trialmatch'].get('by_variant', {}).get('original', {}).get('correct')
        if isinstance(original_n, int):
            regressed = len(wrong['combined'] - wrong['trialmatch'])
            original_changes = {'regressed': regressed, 'rescued': len(wrong['trialmatch'] - wrong['combined']), 'retained_correct': original_n - regressed}
    lines = ['# CARE 模块一 V5 实验报告', '', f'报告生成日期（UTC）：{report_date}', '',
             '新模块一使用 Phi-4 基础模型及 TrialMatchAI LoRA 条件判断、BioLinkBERT-MedNLI 证据推断和确定性三值逻辑。Qwen3 只读取 V4 已保存对照，新模块一不调用它。', '',
             '## 实际运行状态', '',
             f"服务器运行状态：`{safe(status.get('status', 'not_started'))}`。已完成 {status.get('sample_count_completed', '未采集')}/{status.get('sample_count', 192)} 条。GPU `{safe(status.get('gpu_name', '未加载'))}`，设备 `{safe(status.get('device', '未采集'))}`，设备容量 {number(status.get('gpu_total_memory_gib'))} GiB。开始 `{status.get('start_utc', '未采集')}`，结束 `{status.get('end_utc', '未结束')}`。", '',
             '192 条回归样本包括 32 条原始合成病例和 160 条控制变体，涉及 16 条研究规则，每类场景各 32 条。主规则共 41 个原子条件、0 条例外，共 492 个逐样本条件请求。该集合已用于 V5 方法设计，属于探索性回归。', '',
             f"TrialMatchAI 已记录适配器：`{safe(status.get('trialmatch_active_adapters', '未采集'))}`；LoRA 参数数 {status.get('trialmatch_lora_parameter_count', '未采集')}。BioLinkBERT 标签编号由实际配置读取：`{safe(status.get('mednli_id2label', '未采集'))}`。", '',
             '## V5 与历史直接判定的统一评价', '']
    lines += method_table(methods)
    lines += ['', 'A 只使用 TrialMatchAI 条件标签和三值逻辑。A+数值另加入可识别的数值、区间及单位比较。B 在同一 TrialMatchAI 输出上增加 BioLinkBERT 证据判断。纯 A 没有句级证据；A+数值的证据指标仅来自程序数字句；B 才纳入 NLI 证据。V5 复用 V4 手工结构规则，历史 Qwen3 直接组接受自由文本研究规则；这是系统方案对比，不能把结构规则的额外信息与全部性能变化都归因于专业模型本身。', '',
              '## 六类场景（每类 32 条）', '']
    lines += scenario_table(methods)
    lines += ['', '完整原病例指数据集原有合成病例文字全部保留，不表示真实患者。每格同时列正确数和预测未知数量，避免只用总体准确率掩盖未知误判。', '', '## 错误类别', '']
    lines += category_table(methods)
    paired = metrics.get('paired_trialmatch_vs_combined', {}) if complete(methods.get('trialmatch', {})) and complete(methods.get('combined', {})) else {}
    if paired:
        lines += ['', f"B 相对 A+数值：纠正 {paired.get('combined_rescued', '未计算')} 条，损害原正确判断 {paired.get('combined_regressed', '未计算')} 条；判定改变 {paired.get('prediction_changed', '未计算')} 条，其中 U→T/F {paired.get('unknown_to_known', '未计算')} 条、T/F→U {paired.get('known_to_unknown', '未计算')} 条。该比较覆盖 {paired.get('n', '未采集')} 条，运行未完整时只能视为中间状态。"]
    parse = status.get('trialmatch_parse_errors')
    parse_total = sum(parse.values()) if isinstance(parse, dict) and all(isinstance(v, int) for v in parse.values()) else None
    lines += ['', f"TrialMatchAI 条件请求 {status.get('trialmatch_condition_count', '未采集')} 条，完成 {status.get('trialmatch_conditions_completed', '未采集')} 条；解析错误 {parse_total if parse_total is not None else '未采集'} 条，达到生成长度上限 {status.get('trialmatch_truncated_conditions', '未采集')} 条；生成 token {status.get('trialmatch_output_tokens', '未采集')}。解析错误细分：`{safe(parse if parse is not None else '未采集')}`。截断不自动视为已得到有效标签。"]
    if status.get('error'):
        lines += ['', f"实际异常：`{safe(status.get('error_type', 'unknown'))}`：{safe(status['error'])}。"]
    lines += ['', '## 失败样本与已有诊断', '']
    examples = failure_examples(errors)
    if examples:
        lines += ['下表只摘录已保存的错误类别与条件诊断，不据此虚构病历语义原因。完整记录包含预测证据编号、足够参考证据组合、冲突编号和全部条件状态。', '',
                  '| 样本 ID | 方法 | 场景 | 金标→预测 | 已保存错误类别 | 已保存条件诊断 |',
                  '|---|---|---|---|---|---|']
        for row in examples:
            name = dict(METHODS).get(row.get('method'), row.get('method', '?'))
            lines.append(f"| `{safe(row.get('sample_id'))}` | {name} | {safe(row.get('variant'))} | {safe(row.get('gold'))}→{safe(row.get('prediction'))} | {safe(ERROR_NAMES.get(row.get('error_category'), row.get('error_category', '未归类')))} | {recorded_diagnosis(row)} |")
    elif methods and all(complete(methods.get(k, {})) for k, _ in METHODS[:3]) and a.errors.exists():
        lines += ['已完成评价的错误文件为空；没有可列出的失败样本。']
    else:
        lines += ['尚无可用的逐例错误记录，未填充失败样本或原因。']
    combined = methods.get('combined', {})
    numeric = methods.get('trialmatch', {})
    raw = methods.get('trialmatch_model_only', {})
    numeric_seconds = numeric.get('total_inference_seconds') - raw.get('total_inference_seconds') if complete(numeric) and complete(raw) and all(isinstance(x.get('total_inference_seconds'), (int, float)) for x in (numeric, raw)) else None
    lines += ['', '## 实际耗时与显存', '',
              '| 项目 | 已保存实际值 |', '|---|---:|',
              f"| 整体墙钟（含模型加载及后处理） | {number(status.get('elapsed_seconds'))} 秒 |",
              f"| Phi-4 + TrialMatchAI 加载墙钟 | {number(status.get('trialmatch_load_seconds'))} 秒 |",
              f"| BioLinkBERT 加载墙钟 | {number(status.get('mednli_load_seconds'))} 秒 |",
              f"| TrialMatchAI 条件批量预计算墙钟 | {number(status.get('trialmatch_precompute_seconds'))} 秒 |",
              f"| TrialMatchAI 批量生成计时之和 | {number(status.get('trialmatch_shared_generation_seconds'))} 秒 |",
              f"| 数值比较墙钟之和 | {number(timings.get('numeric_seconds', numeric_seconds))} 秒 |",
              f"| 候选句筛选墙钟之和 | {number(timings.get('evidence_selection_seconds'))} 秒 |",
              f"| BioLinkBERT 调用墙钟之和 | {number(timings.get('mednli_seconds'))} 秒 |",
              f"| 全部证据处理墙钟之和 | {number(timings.get('postprocess_seconds'))} 秒 |",
              f"| B 推理与后处理分摊墙钟之和 | {number(combined.get('total_inference_seconds')) if complete(combined) else '未计算'} 秒 |",
              f"| CPU 用户进程时间 | {number(status.get('cpu_user_seconds'))} 秒 |",
              f"| CPU 系统进程时间 | {number(status.get('cpu_system_seconds'))} 秒 |",
              f"| CUDA 内核纯设备时间 | {number(status.get('gpu_kernel_seconds'))} 秒 |",
              f"| GPU 峰值 allocated | {number(status.get('peak_allocated_gib'))} GiB |",
              f"| GPU 峰值 reserved | {number(status.get('peak_reserved_gib'))} GiB |", '',
              f"TrialMatchAI 实际 batch={status.get('trialmatch_batch_size', '未采集')}，每条件生成上限 {status.get('trialmatch_max_new_tokens', '未采集')} token，计时方式 `{safe(status.get('trialmatch_timing_method', '未采集'))}`。批量墙钟按该批大小分摊到条件，再汇总到样本；缓存查找耗时不冒充新一次生成。", '',
              'CPU 用户/系统时间和 CUDA 内核计时若未保存则列“未采集”；整体墙钟及模型调用墙钟包含主机开销，不能当作纯 GPU 计算时间。Qwen3 的历史直接判定为一次规则级判定，V5 为逐条件长输出且使用不同量化和批量协议；本报告不给出硬件等价的速度结论。', '',
              '## 未使用规则上的新合成探索集', '',
              f"运行状态：`{safe(exploratory_status.get('status', 'not_started'))}`。冻结数据为 4 条完整树规则、24 条自编病例、11 个原子条件和 66 个条件请求。来源指南 ID 为 42、44、80、53，AND/OR/NOT、原始条件局部位置及动作位置已在运行前验证；未读取病历正确路径或答案作为输入。", '']
    lines += method_table(exploratory.get('methods', {}), keys={'trialmatch_model_only', 'trialmatch', 'combined'})
    if exploratory_status:
        lines += ['', f"该集实际整体墙钟 {number(exploratory_status.get('elapsed_seconds'))} 秒，GPU 峰值 allocated/reserved {number(exploratory_status.get('peak_allocated_gib'))}/{number(exploratory_status.get('peak_reserved_gib'))} GiB。"]
    lines += ['', '新集没有数值阈值，A+数值不会产生程序数值证据；因此其已知样本证据得分为 0 或联合得分只来自未知样本协议时，反映该组缺少定位输出，不能作为独立证据模型性能，也不能与 B 的联合正确率作等价能力比较。纯 A 的定位与联合指标不适用。新集是模型运行前固定的明确合成事实，标签由规则逻辑确定，未经临床专家复核。它检验新规则上的基础条件与逻辑判断，不能作为真实病历独立临床验证。源树许可证待明确，第三方树和动作仅留服务器；公开自编病例、标签、行号、SHA256 和重建脚本。', '']
    lines += conclusion_lines(methods, paired, exploratory, original_changes)
    lines += ['', '## 方法与适用范围', '',
              'TrialMatchAI 是 Phi-4 的生成式 LoRA 适配器，不是现成医学三分类接口。入组标签 Met→T、Not Met→F、Unclear/Irrelevant→U；排除侧 Not Violated 不证明条件为假，保守映射 U。主 16 条规则没有 exclusion criterion，新合成集的 NOT 也不是 TrialMatchAI 官方排除标签；排除侧映射目前只有单元测试，尚无真实排除标准的系统实验。适配器加载失败不回退普通 Phi-4 或 Qwen3。', '',
              '患者按原文句子保留 Python 字符区间 [start,end)，先筛候选句，再由 BioLinkBERT 推断支持、反驳或中立。标签编号读取模型配置；中立不表示阴性。模型冲突保留原始意见和证据，无法解决时设 U。', '',
              '数值逻辑使用固定字段的数值、区间、单位及明确撤回标记。它没有通用时间线解析或完整的主体消歧能力；复杂旧记录、新记录与其他患者信息仍可能处理错误。无法提取的数值保留 U，不把缺失默认为阴性。程序已经确定的数值比较不因 NLI 意见不同而改成相反结论，分歧仍保存；具体范围以冻结的 care_v5.py 为准。', '',
              'V4 研究规则是条件改写，16 条均缺少可独立核验的完整原始指南动作；其规则字符位置属于预备改写文本，不冒充原始指南位置。原数据 path 列是病例正确路径，未作为在线指南动作输入。新增探索集才含来源完整树的可重建动作片段。推理冻结文件完整性记录见 `experiments/care_module1_v5/provenance/freeze_integrity_check.json`。模块二/三兼容结构通过接口测试并不等于完成真实临床三模块效果验证。', '',
              '模块三原有先验冲突分支依赖模块一的闭卷/指南动作比较。V5 不再调用 Qwen3 生成这两种意见，因此该信号明确不可用，prior_competition_weight 为 0。适用性和证据的上下文接口保留，但不能声称原先的全部先验抑制功能已由新模型替代；模块二、三代码没有修改。', '', '## V4 其他已保存参考结果', '',
              '| 历史方法 | 正确数/准确率 | 宏 F1 | 已知证据定位 | 联合正确 | 反事实成对正确 | 历史生成秒数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    historic_names = {'qwen3_direct': 'Qwen3 直接判定', 'qwen3_structured_no_guard': 'Qwen3 结构化不核验', 'qwen3_structured_with_guard': 'Qwen3 结构化加核验'}
    for key, name in historic_names.items():
        row = historic.get('methods', {}).get(key, {})
        n = historic.get('n')
        lines.append(f"| {name} | {ratio(row.get('correct'), n)} | {number(row.get('macro_f1'), 4)} | {ratio(row.get('known_evidence_correct'), row.get('known_evidence_n'))} | {ratio(row.get('joint_correct'), n)} | {ratio(row.get('counterfactual_pair_correct'), row.get('counterfactual_pair_n'))} | {number(row.get('generation_seconds'))} |")
    lines += ['', f"历史来源：`{safe(historic.get('source', '参考文件缺失'))}`，仅读取保存结果，没有重新调用 Qwen3。", '',
              '## 对应文件位置', '',
              f"- 主运行状态：`{a.run_status.resolve()}`。",
              f"- 主指标：`{a.metrics.resolve()}`；逐例错误：`{a.errors.resolve()}`。",
              f"- 逐例公开预测：`{predictions_path.resolve()}`，报告仅汇总其中已记录的耗时字段。",
              f"- 原始模型输出与原文证据：服务器 `{run_dir/'private/predictions_private.jsonl'}`；条件原始生成 `{run_dir/'private/trialmatch_precomputed.jsonl'}`。",
              f"- 新探索指标：`{a.exploratory_metrics.resolve()}`；新探索状态：`{a.exploratory_status.resolve()}`。",
              f"- 新探索逐例错误：`{a.exploratory_metrics.resolve().parent/'errors.jsonl'}`；冻结记录：`{EXP/'exploratory/data/FREEZE.json'}`。",
              f"- V4 标签和配对：`{EXP/'data/labels_public.jsonl'}`、`{EXP/'data/pairs_public.jsonl'}`；离线规则 `{EXP/'data/rules_v5.jsonl'}`。",
              '- V4 原文输入和全部历史结果保留 `/home/gcx/care_module1_20261018_v4/`，没有覆盖。']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines) + '\n')
    print(str(a.output))


if __name__ == '__main__':
    main()
