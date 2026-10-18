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
    lines = ['| 方法 | 判定正确/准确率 | 宏 F1 | 已知证据定位 | 判定与证据同时正确 | 反事实成对正确 | 生成/后处理累计分摊工作秒 |',
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


def conclusion_lines(methods, paired, exploratory):
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
            lines += ['以上为正确数边际比较；现有汇总指标不含样本交集，不能据此计算原正确病例的保留或损害数量。']
    else:
        lines += ['组合组或 A+数值组未完成评价，不能给出改善或损害结论。']
    lines += ['', '**第三，相对历史 Qwen3，准确率、证据和效率有什么变化？**']
    if complete(qwen) and complete(raw) and complete(combined):
        lines += [f"历史 Qwen3 直接判定准确率 {pct(qwen.get('accuracy'))}、宏 F1 {number(qwen.get('macro_f1'), 4)}。A 相对它的准确率变化为 {delta(raw.get('accuracy'), qwen.get('accuracy'))}；B 相对它变化为 {delta(combined.get('accuracy'), qwen.get('accuracy'))}。"]
        lines += [f"已知样本证据定位：Qwen3 {ratio(qwen.get('known_evidence_correct'), qwen.get('known_evidence_n'))}，B {ratio(combined.get('known_evidence_correct'), combined.get('known_evidence_n'))}，差 {delta(combined.get('known_evidence_accuracy'), qwen.get('known_evidence_accuracy'))}。联合正确率差 {delta(combined.get('joint_accuracy'), qwen.get('joint_accuracy'))}；反事实成对正确率差 {delta(combined.get('counterfactual_pair_accuracy'), qwen.get('counterfactual_pair_accuracy'))}。"]
        lines += [f"已记录累计工作计时：Qwen3 历史单卡生成 {number(qwen.get('total_inference_seconds'))} 秒，A {number(raw.get('total_inference_seconds'))} 秒，B {number(combined.get('total_inference_seconds'))} 秒。模型大小、量化、输出长度、条件请求数量和批量协议不同；V5 此处是原先单卡与随后四卡条件生成按批量分摊的累计工作秒，B 再加证据后处理墙钟；不是用户等待的端到端墙钟，也不是 CUDA 内核设备秒。Qwen3 是历史单卡规则级生成墙钟，两者口径与批量不同，不能据此推断纯模型、硬件速度或四卡加速比。"]
    else:
        lines += ['V5 完整评价不足，不能相对历史 Qwen3 给出准确率、证据或效率的优劣结论；历史输出未重新运行。']
    lines += ['', '**第四，是否可以作为 CARE 正式模块一？**', '当前适合作为可复现的研究实现，不能作为正式模块一替换旧方案。主要问题是固定接口接受率低、已知条件被大量判为未知，BioLinkBERT 补回部分未知时也引入错误拒绝；组合方案的准确率、证据定位和完整病例表现仍低于历史对照。V4 的 16 条研究规则缺少完整原始指南动作，192 条样本已用于设计；新增探索集也只是明确合成事实，没有独立临床专家金标。离线手工结构准备也是部署所需的来源与维护负担。接口兼容已验证，正式应用仍需先修复格式接口并用重新冻结的新数据评价，同时补齐原始指南动作和独立临床标注。']
    if any(complete(x) for x in exploratory.get('methods', {}).values()):
        eraw = exploratory.get('methods', {}).get('trialmatch_model_only', {})
        eb = exploratory.get('methods', {}).get('combined', {})
        if complete(eraw) and complete(eb):
            lines += [f"新增合成探索集 A 正确 {ratio(eraw.get('correct'), eraw.get('n'))}，B 正确 {ratio(eb.get('correct'), eb.get('n'))}；该结果支持的范围是这些新规则上的明确事实判断，不能升级为临床可靠性结论。"]
    return lines


def elapsed_between(start, end):
    if not start or not end:
        return None
    try:
        value = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        return value if value >= 0 else None
    except (ValueError, TypeError):
        return None


def timing_report(status, initial_serial, initial_path):
    """Keep process wall clocks, allocated generation work and stage memory apart."""
    replay = status.get('trialmatch_loaded_in_replay') is False
    manifest = status.get('parallel_manifest') or {}
    workers = manifest.get('parallel_workers') or status.get('parallel_worker_active_adapter_checks') or []
    sources = status.get('generator_status_sources') or []
    if manifest.get('reused_count') == 0:
        initial_serial = {}
    elif not initial_serial:
        archived = next((x.get('status', {}) for x in sources if x.get('status', {}).get('trialmatch_condition_count') and not x.get('status', {}).get('rank') and x.get('status', {}).get('trialmatch_active_adapters')), {})
        initial_serial = archived
    generated = status.get('trialmatch_generation_gpu_seconds', status.get('trialmatch_shared_generation_seconds'))
    rows = ['| 阶段或资源 | 实际记录 |', '|---|---|']
    if replay:
        rows += [f"| 当前证据重放墙钟（仅 BioLinkBERT 加载和证据处理） | {number(status.get('replay_wall_seconds', status.get('elapsed_seconds')))} 秒 |",
                 f"| 证据重放 GPU / device | {safe(status.get('gpu_name', '未采集'))} / {safe(status.get('device', '未采集'))} |",
                 f"| 证据重放 BioLinkBERT 加载墙钟 | {number(status.get('mednli_load_seconds'))} 秒 |",
                 f"| 仅证据重放峰值 allocated / reserved | {number(status.get('peak_allocated_gib'))} / {number(status.get('peak_reserved_gib'))} GiB |",
                 f"| TrialMatchAI 累计生成分摊工作秒（{'复用单卡 + 四卡' if manifest.get('reused_count') else '四卡生成'}） | {number(generated)} 秒 |",
                 f"| 四卡阶段墙钟（最早 worker 开始至最晚结束，含加载） | {number(status.get('parallel_workers_wallclock_seconds'))} 秒 |"]
    else:
        rows += [f"| 单进程整体墙钟（本次加载和推理） | {number(status.get('elapsed_seconds'))} 秒 |",
                 f"| 单进程 Phi-4 + TrialMatchAI 加载墙钟 | {number(status.get('trialmatch_load_seconds'))} 秒 |",
                 f"| 单进程 BioLinkBERT 加载墙钟 | {number(status.get('mednli_load_seconds'))} 秒 |",
                 f"| 单进程条件预计算墙钟 | {number(status.get('trialmatch_precompute_seconds'))} 秒 |",
                 f"| 累计批量生成分摊工作秒 | {number(generated)} 秒 |",
                 f"| 单进程全部已加载模型峰值 allocated / reserved | {number(status.get('peak_allocated_gib'))} / {number(status.get('peak_reserved_gib'))} GiB |"]
    serial_end = initial_serial.get('interrupted_utc') or initial_serial.get('interruption_utc') or initial_serial.get('end_utc')
    serial_start = initial_serial.get('start_utc')
    serial_wall = elapsed_between(serial_start, serial_end)
    if initial_serial:
        saved = initial_serial.get('trialmatch_conditions_completed', initial_serial.get('saved_conditions', manifest.get('reused_count')))
        rows += [f"| 最初单卡设备 / 状态 | {safe(initial_serial.get('device', '未采集'))} / {safe(initial_serial.get('status', '未采集'))} |",
                 f"| 最初单卡开始 / 中断记录（UTC） | {safe(serial_start or '未采集')} / {safe(serial_end or '未采集')} |",
                 f"| 最初单卡墙钟（已记录开始至中断） | {number(serial_wall)} 秒 |",
                 f"| 最初单卡已刷盘条件 / 合并复用条件 | {saved if saved is not None else '未采集'} / {manifest.get('reused_count', '未采集')} |",
                 f"| 最初单卡 Phi-4 + LoRA 加载墙钟 | {number(initial_serial.get('trialmatch_load_seconds'))} 秒 |"]
        serial_peak = initial_serial.get('peak_allocated_gib')
        serial_reserved = initial_serial.get('peak_reserved_gib')
        rows += [f"| 最初单卡所有模型峰值 allocated / reserved | {number(serial_peak)} / {number(serial_reserved)} GiB；未记录时不估计 |"]
    parallel_devices = [str(x.get('device', '?')) for x in workers]
    if workers:
        peaks = [x['peak_allocated_gib'] for x in workers if isinstance(x.get('peak_allocated_gib'), (int, float))]
        reserved = [x['peak_reserved_gib'] for x in workers if isinstance(x.get('peak_reserved_gib'), (int, float))]
        max_device = max(workers, key=lambda x: x.get('peak_allocated_gib') or -1).get('device', '?') if peaks else '未采集'
        rows += [f"| 四卡设备 | {safe(', '.join(parallel_devices))} |",
                 f"| 四卡各自生成阶段最大单卡 allocated 峰值 | {number(max(peaks) if peaks else None)} GiB（{safe(max_device)}） |",
                 f"| 四卡各自生成阶段最大单卡 reserved 峰值 | {number(max(reserved) if reserved else None)} GiB |"]
    end_to_end = elapsed_between(serial_start, status.get('end_utc')) if replay and initial_serial else None
    if replay and initial_serial:
        rows += [f"| 最初单卡开始到证据重放结束墙钟（含中断、调度、加载和间隔） | {number(end_to_end)} 秒；不可视为稳定吞吐基准 |"]
    rows += [f"| CPU 用户 / 系统时间 | {number(status.get('cpu_user_seconds'))} / {number(status.get('cpu_system_seconds'))} 秒 |",
             f"| CUDA 内核纯设备计时 | {number(status.get('gpu_kernel_seconds'))} 秒 |"]
    if workers:
        rows += ['', '| worker | GPU / 设备 | 条件数 | 加载墙钟秒 | 进程墙钟秒 | allocated 峰值 GiB | reserved 峰值 GiB |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for worker in workers:
            rows.append(f"| {worker.get('rank', '?')} | {safe(worker.get('gpu_name', '未采集'))} / {safe(worker.get('device', '未采集'))} | {worker.get('completed_conditions', worker.get('assigned_conditions', '未采集'))} | {number(worker.get('load_seconds'))} | {number(worker.get('elapsed_seconds'))} | {number(worker.get('peak_allocated_gib'))} | {number(worker.get('peak_reserved_gib'))} |")
    for source, value in sorted(status.get('generation_by_source', {}).items()):
        rows += ['', f"生成来源 `{safe(source)}`：{value.get('conditions', '未采集')} 个条件，累计分摊工作 {number(value.get('allocated_gpu_seconds'))} 秒。"]
    if replay:
        rows += ['', '证据重放不加载 Phi-4 或 TrialMatchAI；适配器身份从已记录的原生成进程继承并验证。原生成与四卡各自的加载和显存不能从 NLI-only 状态推断。累计分摊工作秒是同步生成调用的墙钟工作量，虽然状态字段名含 gpu_seconds，也不是 CUDA 内核设备时间。']
        if initial_path:
            rows += ['', f"最初单卡原始状态档案：`{initial_path}`。"]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-status', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--errors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predictions', type=Path, help='Optional saved public predictions used only to aggregate recorded timing fields')
    p.add_argument('--initial-serial-status', type=Path, default=EXP/'runs/v5_serial_initial/run_status.json')
    p.add_argument('--exploratory-metrics', type=Path, default=EXP/'exploratory/results/metrics.json')
    p.add_argument('--exploratory-status', type=Path, default=EXP/'exploratory/runs/frozen_v1/run_status.json')
    a = p.parse_args()
    status = read(a.run_status) or {}
    initial_serial = read(a.initial_serial_status) or {}
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
    resource_lines = timing_report(status, initial_serial, a.initial_serial_status)
    predictions_path = a.predictions or run_dir/'predictions_public.jsonl'
    recorded_rows = read_errors(predictions_path)
    timings = {}
    if status.get('status') == 'completed' and len(recorded_rows) == status.get('sample_count_completed'):
        for field in ('numeric_seconds', 'evidence_selection_seconds', 'mednli_seconds', 'postprocess_seconds'):
            values = [row.get(field) for row in recorded_rows]
            if values and all(isinstance(value, (int, float)) for value in values):
                timings[field] = sum(values)
    lines = ['# CARE 模块一 V5 实验报告', '', f'报告生成日期（UTC）：{report_date}', '',
             '新模块一使用 Phi-4 基础模型及 TrialMatchAI LoRA 条件判断、BioLinkBERT-MedNLI 证据推断和确定性三值逻辑。Qwen3 只读取 V4 已保存对照，新模块一不调用它。', '',
             '## 实际运行状态', '',
             f"服务器运行状态：`{safe(status.get('status', 'not_started'))}`。已完成 {status.get('sample_count_completed', '未采集')}/{status.get('sample_count', 192)} 条。GPU `{safe(status.get('gpu_name', '未加载'))}`，设备 `{safe(status.get('device', '未采集'))}`，设备容量 {number(status.get('gpu_total_memory_gib'))} GiB。开始 `{status.get('start_utc', '未采集')}`，结束 `{status.get('end_utc', '未结束')}`。", '',
             '192 条回归样本包括 32 条原始合成病例和 160 条控制变体，涉及 16 条研究规则，每类场景各 32 条。主规则共 41 个原子条件、0 条例外，共 492 个逐样本条件请求。该集合已用于 V5 方法设计，属于探索性回归。', '',
             f"实际资源口径：{'本状态只描述 BioLinkBERT 证据重放阶段；TrialMatchAI 原始生成身份与资源另见下文单卡档案和四卡清单。' if status.get('trialmatch_loaded_in_replay') is False else '本状态描述单进程模型加载与推理。'}", '',
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
    if isinstance(parse_total, int) and isinstance(status.get('trialmatch_conditions_completed'), int):
        count = status['trialmatch_conditions_completed']
        lines += ['', f"固定接口接受 {count - parse_total}/{count} 条，拒绝 {parse_total}/{count} 条。当前准确率是固定协议下的端到端结果，不能单独估计模型的医学判断能力；不用诊断性提取的标签重新评分。"]
    if (EXP/'ANALYSIS_PARSE.zh.md').exists():
        lines += ['', '本次已完成运行的结构诊断与真实失败样本见 [解析与失败诊断](ANALYSIS_PARSE.zh.md)。主要格式问题包括非目标空占位、JSON 围栏及长生成截断。']
    if (EXP/'ANALYSIS_ERRORS.zh.md').exists():
        lines += ['', '汇总错误比较见 [错误分类分析](ANALYSIS_ERRORS.zh.md)；该补充报告只使用既有汇总指标。']
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
    lines += ['', '## 实际耗时与显存', '']
    lines += resource_lines
    lines += ['', '| 证据处理项目 | 已保存实际值 |', '|---|---:|',
              f"| 数值比较累计墙钟 | {number(timings.get('numeric_seconds', numeric_seconds))} 秒 |",
              f"| 候选句筛选累计墙钟 | {number(timings.get('evidence_selection_seconds'))} 秒 |",
              f"| BioLinkBERT 调用累计墙钟 | {number(timings.get('mednli_seconds'))} 秒 |",
              f"| 全部证据处理累计墙钟 | {number(timings.get('postprocess_seconds'))} 秒 |",
              f"| B：原生成工作秒 + 后处理墙钟 | {number(combined.get('total_inference_seconds')) if complete(combined) else '未计算'} 秒 |", '',
              'CPU 用户/系统时间和 CUDA 内核计时未保存时列“未采集”；模型调用墙钟包含主机开销。原先单卡先完成的输出保持不变，剩余条件按原 batch=8 分配四卡；每个原始批内样本顺序、提示和 token 上限不变。中断且未刷盘的批次不纳入结果，也不估计其无效计算时长。', '',
              '四卡累计生成工作秒不能当作四卡墙钟，证据重放墙钟不能当作全实验运行时间，NLI 重放阶段峰值也不能当作 Phi-4 + LoRA 生成峰值。Qwen3 历史单卡直接判定与 V5 的模型规模、量化、生成长度、逐条件请求和批量协议不同，不报告等价硬件速度或加速比。', '',
              '## 未使用规则上的新合成探索集', '',
              f"运行状态：`{safe(exploratory_status.get('status', 'not_started'))}`。冻结数据为 4 条完整树规则、24 条自编病例、11 个原子条件和 66 个条件请求。来源指南 ID 为 42、44、80、53，AND/OR/NOT、原始条件局部位置及动作位置已在运行前验证；未读取病历正确路径或答案作为输入。", '']
    lines += method_table(exploratory.get('methods', {}), keys={'trialmatch_model_only', 'trialmatch', 'combined'})
    if exploratory_status:
        exploratory_parse = exploratory_status.get('trialmatch_parse_errors') or {}
        lines += ['', f"新集条件请求 {exploratory_status.get('trialmatch_condition_count', '未采集')} 条；解析失败 {sum(exploratory_parse.values())} 条，达到生成长度上限 {exploratory_status.get('trialmatch_truncated_conditions', '未采集')} 条。细分：`{safe(exploratory_parse)}`。"]
        lines += ['', '该集资源使用单独记录：', ''] + timing_report(exploratory_status, {}, None)
    lines += ['', '新集没有数值阈值，A+数值不会产生程序数值证据；因此其已知样本证据得分为 0 或联合得分只来自未知样本协议时，反映该组缺少定位输出，不能作为独立证据模型性能，也不能与 B 的联合正确率作等价能力比较。纯 A 的定位与联合指标不适用。新集是模型运行前固定的明确合成事实，标签由规则逻辑确定，未经临床专家复核。它检验新规则上的基础条件与逻辑判断，不能作为真实病历独立临床验证。源树许可证待明确，第三方树和动作仅留服务器；公开自编病例、标签、行号、SHA256 和重建脚本。', '']
    lines += conclusion_lines(methods, paired, exploratory)
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
              f"- 主运行状态：`{a.run_status.resolve()}`；最初单卡状态：`{a.initial_serial_status.resolve()}`。",
              f"- 主指标：`{a.metrics.resolve()}`；逐例错误：`{a.errors.resolve()}`。",
              f"- 逐例公开预测：`{predictions_path.resolve()}`，报告仅汇总其中已记录的耗时字段。",
              f"- 原始模型输出与原文证据：服务器 `{run_dir/'private/predictions_private.jsonl'}`；合并条件原始生成 `{status.get('merged_file', str(run_dir/'private/trialmatch_precomputed.jsonl'))}`；合并计时清单 `{status.get('parallel_manifest_path', '未使用并行清单')}`。",
              f"- 新探索指标：`{a.exploratory_metrics.resolve()}`；新探索状态：`{a.exploratory_status.resolve()}`。",
              f"- 新探索逐例错误：`{a.exploratory_metrics.resolve().parent/'errors.jsonl'}`；冻结记录：`{EXP/'exploratory/data/FREEZE.json'}`。",
              f"- V4 标签和配对：`{EXP/'data/labels_public.jsonl'}`、`{EXP/'data/pairs_public.jsonl'}`；离线规则 `{EXP/'data/rules_v5.jsonl'}`。",
              '- 解析诊断：`ANALYSIS_PARSE.zh.md`、`provenance/parse_failure_diagnostics.json`；公开完整性检查：`provenance/completed_artifact_integrity.json`。',
              '- V4 原文输入和全部历史结果保留 `/home/gcx/care_module1_20261018_v4/`，没有覆盖。']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines) + '\n')
    print(str(a.output))


if __name__ == '__main__':
    main()
