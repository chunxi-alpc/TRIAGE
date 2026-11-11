"""Idempotent result interpretation; no fitting, GPU work, or result-table edits."""
from __future__ import annotations
import csv
import json
import math
import os
import re
from pathlib import Path
from .common import OUT

BEGIN = '<!-- ASTER_CR_FINAL_INTERPRETATION_BEGIN -->'
END = '<!-- ASTER_CR_FINAL_INTERPRETATION_END -->'
METRICS = (
    ('ap_score_change_rms', 'A/P分数变化RMS', False),
    ('ap_vr', 'VR', True),
    ('cf_correction_rms', 'C/F修正RMS', False),
    ('cf_er', 'ER', True),
    ('cf_direction_degradation_count', '证据方向退化数', False),
    ('cf_over_budget_rate', 'C/F超预算比例', True),
    ('natural_node_accuracy', '自然节点正确率', True),
    ('natural_path_accuracy', '自然完整路径正确率', True),
    ('interference_path_accuracy', '干扰完整路径正确率', True),
)
NATIVE = (
    ('natural_native_accuracy', '原生正确率', True),
    ('native_preserving_paired_accuracy', '原生保持配对全对率', True),
    ('native_changing_paired_accuracy', '原生翻转配对全对率', True),
    ('native_preserving_relation_accuracy', '原生保持关系正确率', True),
    ('native_changing_relation_accuracy', '原生翻转关系正确率', True),
)

def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def _number(value):
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

def _fmt(value, percentage=False, difference=False):
    value = _number(value)
    if value is None:
        return '未提供'
    value = value * (100 if percentage else 1)
    if abs(value) < 1e-12:
        value = 0.0
    text = f'{value:+.3f}' if difference else f'{value:.3f}'
    return text + ('个百分点' if percentage and difference else '%' if percentage else '')

def _count(value):
    value = _number(value)
    return '未提供' if value is None else str(int(value)) if value.is_integer() else str(value)

def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)

class Results:
    def __init__(self, out):
        self.out = Path(out)
        self.scopes, self.intervals = {}, {}
        for scope in ('dev', 'test', 'new60', 'nli4ct', 'medpic'):
            result = _read(self.out / 'evaluation' / scope / 'metrics.json')
            self.scopes[scope] = result
            for row in result.get('bootstrap', []):
                self.intervals[(scope, row.get('comparison'), row.get('metric'))] = row
        path = self.out / 'source_bootstrap.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    self.intervals.setdefault((row.get('scope'), row.get('comparison'), row.get('metric')), row)

    def model(self, scope, group):
        return self.scopes.get(scope, {}).get('models', {}).get(group, {})

    def ci(self, scope, lhs, rhs, metric):
        row = self.intervals.get((scope, lhs + ' minus ' + rhs, metric))
        sign = 1
        if row is None:
            row = self.intervals.get((scope, rhs + ' minus ' + lhs, metric))
            sign = -1
        if row is None:
            return None
        estimate, low, high = (_number(row.get(k)) for k in ('estimate', 'low', 'high'))
        if low is None or high is None:
            return None
        if sign < 0:
            estimate, low, high = (-estimate if estimate is not None else None), -high, -low
        return dict(estimate=estimate, low=low, high=high)

    def comparison(self, scope, lhs, rhs, metric, percentage):
        ci = self.ci(scope, lhs, rhs, metric)
        if ci is None:
            return '差值及95%区间未提供'
        return (_fmt(ci['estimate'], percentage, True) + '，95%区间['
                + _fmt(ci['low'], percentage, True) + '，'
                + _fmt(ci['high'], percentage, True) + ']')

    def item(self, scope, group, reference, spec):
        metric, label, percentage = spec
        value = self.model(scope, group).get(metric)
        rendered = _count(value) if metric.endswith('_count') else _fmt(value, percentage)
        return label + ' ' + rendered + '（' + self.comparison(scope, group, reference, metric, percentage) + '）'

def _path_count(model):
    a, n = (_number(model.get(k)) for k in ('natural_path_correct', 'natural_path_count'))
    return '未提供' if a is None or n is None else f'{int(a)}/{int(n)}'

def _outcome_statement(results, scope, group):
    categories = {'positive': [], 'negative': [], 'uncertain': [], 'missing': []}
    for key, label, _ in METRICS[6:]:
        ci = results.ci(scope, group, 'E0', key)
        bucket = 'missing' if ci is None else 'positive' if ci['low'] > 0 else 'negative' if ci['high'] < 0 else 'uncertain'
        categories[bucket].append(label)
    text = []
    for bucket, suffix in [('positive', '的差值区间高于0'), ('negative', '的差值区间低于0'),
                           ('uncertain', '的差值区间含0'), ('missing', '的差值区间未提供')]:
        if categories[bucket]:
            text.append('、'.join(categories[bucket]) + suffix)
    return '较E0，' + '；'.join(text) + '。'

def _native_interpretation(results, task, group):
    statements = []
    accuracy = results.ci(task, group, 'raw', 'natural_native_accuracy')
    if accuracy is not None and accuracy['estimate'] is not None and accuracy['estimate'] < 0:
        statements.append('原生正确率下降，差值95%区间完全低于0。' if accuracy['high'] < 0 else
                          '原生正确率点估计下降，差值95%区间含0；未能确认跨任务准确率改善。')
    preserving = results.ci(task, group, 'raw', 'native_preserving_paired_accuracy')
    changing = results.ci(task, group, 'raw', 'native_changing_paired_accuracy')
    def brief(ci):
        estimate = '未提供' if ci['estimate'] is None else f"{100*ci['estimate']:+.2f}"
        return estimate + f"个百分点，95%区间[{100*ci['low']:+.2f}, {100*ci['high']:+.2f}]"
    if preserving is not None and preserving['high'] < 0:
        statements.append('原生保持对全对率显著下降（' + brief(preserving) + '），表明保持关系的配对表现出现退化。')
        if changing is not None and changing['low'] > 0:
            statements.append('同时原生翻转对全对率提高（' + brief(changing) + '），两者存在明确取舍；不能仅据翻转改善声称跨任务整体提升。')
    return ''.join(statements)

def _next_steps(selected, completed):
    return ('# 下一阶段执行计划\n\n本轮冻结方案：' + (selected or '未提供') +
            '；最终评价汇总已列范围：' + ('、'.join(completed) if completed else '未提供') + '。\n\n'
            '1. 根据九项指标、来源差值区间及原生任务结果整理需要补充验证的问题；保留冻结配置与逐样本结果。开发路径多对一条不能证明E1稳定优于E2。\n'
            '2. 优先解释正确率、证据超预算和来源差异，仅使用已有结果整理问题，不重复生成样本或进行AI审核。新增实验另列问题、数据、评价来源、GPU、显存、规模与时长。\n'
            '3. 后续基线另行提交执行清单和资源计划，本轮不自动启动285组全量基线。历史40和60来源仍标为已有使用记录的评价，新独立验证需另行确定尚未参与既有选择的数据。\n')

def finalize_report(out=OUT):
    """Read final results and replace one marked interpretation section."""
    out = Path(out)
    results = Results(out)
    selection = _read(out / 'development_selection.json')
    final_summary = _read(out / 'evaluation_summary.json')
    selected = selection.get('frozen_group') or final_summary.get('frozen_group')
    completed = final_summary.get('completed_scopes', [])
    text = ['## 结果解读与后续安排', '',
            '本轮按开发集规则冻结 ' + selected + '；留出集和目标任务仅用于评价，未用于重新选择方案。'
            if selected else '开发集冻结方案未提供，不据缺失字段推测主方案。']
    e1, e2 = results.model('dev', 'E1'), results.model('dev', 'E2')
    text += ['', '开发集自然完整路径：E1为 ' + _path_count(e1) + '，E2为 ' + _path_count(e2) +
             '。E1−E2为' + results.comparison('dev', 'E1', 'E2', 'natural_path_accuracy', True) + '。']
    dev_ci = results.ci('dev', 'E1', 'E2', 'natural_path_accuracy')
    if dev_ci is None:
        text.append('该差值区间未提供，不能据单条路径差异确认E1稳定优于E2。')
    elif dev_ci['low'] <= 0 <= dev_ci['high']:
        text.append('差值95%区间含0。385/420对384/420仅用于本轮选择，不能夸大为E1稳定优于E2。'
                    if _path_count(e1) == '385/420' and _path_count(e2) == '384/420' else
                    '差值95%区间含0，该开发路径差异不能证明E1稳定优于E2。')
    else:
        text.append('以上仅为开发集选择证据，能否推广须结合留出集及原生任务实际结果。')
    text += ['', '以下比例差值用百分点，其余差值沿用原单位；九项指标共同用于判断，不能仅以平均VR认定成功。']
    for scope, label in [('test', '历史40来源留出集'), ('new60', '已有60来源评价集')]:
        text += ['', '**' + label + '**']
        if not results.scopes.get(scope) or not selected or not results.model(scope, selected):
            text.append('该范围冻结方案的最终九项指标及差值区间未提供。')
            continue
        text.append('括号均为' + selected + '−E0。' + _outcome_statement(results, scope, selected))
        for indices in [(6, 7, 8), (0, 1), (2, 3, 4, 5)]:
            text.append('- ' + '；'.join(results.item(scope, selected, 'E0', METRICS[i]) for i in indices) + '。')
        for lhs, rhs, label in [('E2', 'E1', '主数据比较'), ('E2', 'E3', '权重消融')]:
            text.append(label + '（' + lhs + '−' + rhs + '）：自然路径' +
                        results.comparison(scope, lhs, rhs, 'natural_path_accuracy', True) + '；干扰路径' +
                        results.comparison(scope, lhs, rhs, 'interference_path_accuracy', True) + '；C/F修正RMS' +
                        results.comparison(scope, lhs, rhs, 'cf_correction_rms', False) + '。')
    for task, label in [('nli4ct', 'NLI4CT-2024'), ('medpic', 'MedPIC')]:
        text += ['', '**' + label + '原生任务**']
        model = results.model(task, selected) if selected else {}
        if not model:
            text.append('该任务冻结方案的原生指标及差值区间未提供。')
            continue
        counts = [model.get(k) for k in ('native_preserving_pairs', 'native_changing_pairs')]
        total = sum(_number(x) for x in counts) if all(_number(x) is not None for x in counts) else None
        text.append('原生保持配对数 ' + _count(counts[0]) + '，原生翻转配对数 ' + _count(counts[1]) + '，原生配对合计 ' + _count(total) + '。')
        for subset in [NATIVE[:1], NATIVE[1:3], NATIVE[3:]]:
            text.append('- ' + '；'.join(results.item(task, selected, 'raw', spec) for spec in subset) + '。')
        if task == 'medpic':
            text.append('额外背景对单列synthetic_preserving，数量 ' + _count(model.get('synthetic_preserving_pairs')) +
                        '，不计入原生保持或翻转配对。背景配对全对率 ' +
                        _fmt(model.get('synthetic_preserving_paired_accuracy'), True) + '，保持关系正确率 ' +
                        _fmt(model.get('synthetic_preserving_relation_accuracy'), True) + '。')
        interpretation = _native_interpretation(results, task, selected)
        if interpretation:
            text.append(interpretation)
        text.append('保持和翻转仅表示原生标签关系，未经语义核验不解释为A/P或C/F；差值均为' + selected + '−原模型。')
    text += ['', '历史40测试来源和已有60来源均有使用记录，本轮结果不能称为全新独立确认。',
             '最终评价汇总列出的已完成范围：' + '、'.join(completed) + '。' if completed else
             '最终评价汇总尚未提供，以上仅解释已经落盘的结果。',
             '', '下一阶段依据实际变化及区间整理需补充验证的问题和资源计划，不自动启动285组全量基线。详见 NEXT_STEPS.zh.md。']
    report = out / 'REPORT.zh.md'
    existing = report.read_text(encoding='utf-8') if report.exists() else '# ASTER-CR新配对数据训练与验证报告\n'
    pattern = re.compile(re.escape(BEGIN) + r'.*?' + re.escape(END), flags=re.S)
    block = BEGIN + '\n' + '\n'.join(text) + '\n' + END
    _atomic_text(report, pattern.sub('', existing).rstrip() + '\n\n' + block + '\n')
    next_steps = out / 'NEXT_STEPS.zh.md'
    _atomic_text(next_steps, _next_steps(selected, completed))
    return dict(report=str(report), next_steps=str(next_steps), selected_group=selected,
                completed_scopes=completed, interpretation_block='replaced idempotently', experiments_started=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    print(json.dumps(finalize_report(parser.parse_args().out), ensure_ascii=False))
