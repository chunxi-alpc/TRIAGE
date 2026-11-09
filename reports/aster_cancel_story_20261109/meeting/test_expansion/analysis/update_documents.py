
"""Update existing manuscript files after official-test metrics are complete.

No experiments, git commands, or network publishing are performed.
Default call:
  python update_documents.py --metrics /home/gcx/aster_next_20261109/results/summary/metrics.json
All targets can be changed with command-line arguments below.
"""
from pathlib import Path
import argparse
import gzip
import json
import re
import shutil
import sys

# Configurable destinations. The current publish worktree is the text source.
DEFAULT_RUN_ROOT = Path('/home/gcx/aster_next_20261109')
DEFAULT_PUBLISH_WORKTREE = Path('/home/gcx/aster-real-completion-publish-20261109')
DEFAULT_STANDALONE_ROOT = Path('/home/gcx/aster_cancel_story_20261109')
REPORT_RELATIVE_ROOT = Path('reports/aster_cancel_story_20261109')
PACKET_RELATIVE_ROOT = Path('meeting/test_expansion')
PAPER_FILE = Path('paper_story_latest.zh.md')
BRIEF_FILE = Path('meeting/meeting_brief.zh.md')
README_FILE = Path('README.md')

METHOD_LABELS = {
    'raw': '原模型',
    'fit_bias': '仅 fit 校准',
    'fit_bias_only': '仅 fit 校准',
    'bias_fit': '仅 fit 校准',
    'fitbias': '仅 fit 校准',
    'old_linear': '原 16 维线性方法',
    'oldlinear': '原 16 维线性方法',
    'strict': '较强保护配置（fit 校准）',
    'strict_fit_ce': '较强保护配置（fit 校准）',
    'optimized_strict_fit_calibration': '较强保护配置（fit 校准）',
    'balanced': '平衡配置（fit 校准）',
    'balanced_fit_ce': '平衡配置（fit 校准）',
    'optimized_balanced_fit_calibration': '平衡配置（fit 校准）',
    'traceonly': '仅痕迹平衡配置（fit 校准）',
    'trace_only': '仅痕迹平衡配置（fit 校准）',
    'trace_only_balanced_fit_calibration': '仅痕迹平衡配置（fit 校准）',
    'prompt_only': '证据对齐提示（对照）',
    'prompt_control': '证据对齐提示（对照）',
    'evidence_aligned_prompt': '证据对齐提示（对照）',
    'evidence_alignment_prompt': '证据对齐提示（对照）',
}
GROUP_LABELS = {
    'all': '官方 test 全部',
    'no_recorded_input_use': '未见已盘点历史输入记录',
    'earlier_recorded_source': '已有更早输入记录',
    'current_fit_dev_sources': '当前 fit/dev 同源',
    'outside_current_fit_dev_sources': '当前 fit/dev 之外来源',
    'old_test_cases': '原 806 条测试同源/病例组',
    'additional_cases': '新增病例组',
    'current_fit': '当前 fit 同源',
    'current_dev': '当前 dev 同源',
    'current_test': '原测试同源',
}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def is_prompt_method(name):
    lowered = name.lower()
    return ('prompt' in lowered or lowered.startswith('evidence_alignment')
            or lowered.startswith('condition_only') or lowered.startswith('condition_prompt'))


def label(name):
    lowered = name.lower()
    if lowered.startswith(('condition_only', 'condition_prompt')):
        return '当前条件提示（对照）' + ('＋fit 校准' if 'fit' in lowered else '')
    if lowered.startswith('evidence_alignment') and name not in METHOD_LABELS:
        return '证据对齐提示（对照）' + ('＋fit 校准' if 'fit' in lowered else '')
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    if is_prompt_method(name):
        return name + '（提示对照）'
    return name


def ordered_names(methods):
    priorities = list(METHOD_LABELS)
    return sorted(methods, key=lambda name: (
        priorities.index(name) if name in priorities else len(priorities), name))


def number(value, digits=2, scale=1.0):
    return '—' if value is None else f'{float(value) * scale:.{digits}f}'


def percent(value):
    return number(value, scale=100)


def count(value):
    return '—' if value is None else f'{int(value):,}'


def ci_text(value, scale=100):
    if value is None or len(value) != 2:
        return '—'
    return '[' + ', '.join(number(x, scale=scale) for x in value) + ']'


def path_ci(item):
    value = item.get('source_bootstrap_95ci')
    if value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|', '／').replace('\n', ' ') for x in row) + ' |')
    return '\n'.join(lines) + '\n'


def marked(tag, body):
    return f'<!-- {tag}:start -->\n{body.rstrip()}\n<!-- {tag}:end -->'


def replace_or_insert(text, tag, body, anchor=None, after=False):
    block = marked(tag, body)
    pattern = re.compile(r'<!-- ' + re.escape(tag) + r':start -->.*?<!-- ' +
                         re.escape(tag) + r':end -->', re.S)
    if pattern.search(text):
        return pattern.sub(lambda match: block, text)
    if anchor is None:
        return text.rstrip() + '\n\n' + block + '\n'
    match = re.search(anchor, text, re.M)
    if match is None:
        raise ValueError(f'Cannot find insertion anchor for {tag}: {anchor}')
    pos = match.end() if after else match.start()
    return text[:pos] + ('\n\n' if after else '') + block + '\n\n' + text[pos:]


def load_diagnosis(run_root, requested=None):
    paths = ([Path(requested)] if requested else []) + [
        run_root / 'diagnosis/summary.json',
        run_root / 'diagnosis/diagnosis.json',
    ]
    for path in paths:
        if path.exists():
            return read_json(path), path
    return {}, None


def diagnosis_text(diagnosis):
    summary = diagnosis.get('summary', {})
    if not summary and isinstance(diagnosis.get('methods'), dict):
        summary = diagnosis['methods']
    rows = []
    for name in ordered_names(summary):
        item = summary[name]
        if not isinstance(item, dict):
            continue
        correct = item.get('correct_paths', item.get('path_correct'))
        total = item.get('paths', item.get('evaluated'))
        if correct is None or total is None:
            continue
        rows.append([label(name), f'{count(correct)}/{count(total)}',
                     count(item.get('error_paths', int(total) - int(correct))),
                     count(item.get('unique_first_error_source_nodes')),
                     count(item.get('first_error_sources'))])
    if not rows:
        return ''
    paragraphs = [
        '**开发集错误分布。** 分析使用既有开发集节点分数和参考路径，定位首次错误及重复来源。',
        table(['方法', '正确路径', '错误路径', '首次错误的来源-节点数', '首次错误来源数'], rows),
    ]
    raw = summary.get('raw', {})
    error_paths = raw.get('error_paths')
    margin = raw.get('first_error_margin_bins', {}).get('abs_raw_gt_5')
    repeated = raw.get('first_error_repeated_nodes', [])
    if error_paths is not None and margin is not None:
        text = (f'原模型 {count(error_paths)} 条错误路径中，{count(margin)} 条首次错误的原分数绝对值超过 '
                '5 logit。')
        if repeated:
            most = max(repeated, key=lambda item: item.get('paths', 0))
            text += f'其中一个来源-节点在 {count(most.get("paths"))} 条路径中重复出现。'
        text += '错误集中程度和较大的原分数幅度，有助于解释小幅对消为何未必改变整条路径的正确数。'
        paragraphs.append(text)
    pma = next((item for item in repeated
                if 'pma' in str(item.get('condition', '')).lower()
                or 'postmenstrual' in str(item.get('condition', '')).lower()), None)
    if pma is not None and error_paths:
        paragraphs.append(
            f'PMA 时间条件在原模型错误路径中占 {count(pma.get("paths"))}/{count(error_paths)}。'
            '这一集中现象与个别病例文字、条件及参考标签可能存在不一致的线索，均作为待核对观察；'
            '本轮保留原参考标签，不将其解释为已确认的数据错误或临床原因。')
    observed = diagnosis.get('observations', {})
    common = observed.get('all_four_wrong_gold_instances')
    common_unique = observed.get('all_four_wrong_unique_source_nodes')
    if common is not None and common_unique is not None:
        paragraphs.append(f'四种方法共同判断错误的 gold 节点实例为 {count(common)} 个，'
                          f'涉及 {count(common_unique)} 个来源-节点。实例数与独立节点数分别报告。')
    if diagnosis.get('fit_path_cache', {}).get('available') is False:
        paragraphs.append('既有 fit 路径索引为空，因此本次不推断 fit 完整路径正确率。')
    return '\n\n'.join(paragraphs) + '\n\n'


def dataset_text(inventory, metrics):
    totals = inventory.get('totals', {})
    train = totals.get('train_val', {})
    test = totals.get('test', {})
    current = inventory.get('current_paths', {})
    official = inventory.get('official_manifest', {})
    parts = [
        ('本研究使用 **MEGA-CDP**。发布数据包含 '
         f'{count(inventory.get("all_trees", 2274))} 棵指南决策树、'
         f'{count(inventory.get("all_cases", 42353))} 个生成病例及对应参考路径；'
         '这些病例为根据指南路径生成的 vignette。'),
        (f'原始 train_val 含 {count(train.get("trees", 1834))} 棵树、'
         f'{count(train.get("cases", 34895))} 条路径；官方 test 含 '
         f'{count(test.get("trees", 440))} 棵树、{count(test.get("cases", 7458))} 条路径。'
         '发布树的全部根到叶路径数与现存参考路径数一致。'),
        ('repair-v2 原实验从 train_val 中按来源选取子集，fit/dev/原留出测试分别为 '
         f'{count(current.get("fit", 1712))}/'
         f'{count(current.get("dev", 420))}/'
         f'{count(current.get("test", 806))} 条病例路径。'
         '下表的配对数量和第 8.5–8.10 节的 806 条路径结果继续使用这一子集口径。'),
        ('本轮另将固定方法用于官方 test 的全部 '
         f'{count(official.get("cases", test.get("cases", 7458)))} 条路径，'
         f'涉及 {count(official.get("trees", test.get("trees", 440)))} 棵树和 '
         f'{count(official.get("source_clusters", test.get("source_clusters", 418)))} 个既有同源簇；'
         '扩展结果见第 8.11 节。官方 test 扩展与原 train_val 留出测试分别报告。'),
    ]
    return '\n\n'.join(parts)


def official_main_table(group, compact=False):
    rows = []
    methods = group['methods']
    for name in ordered_names(methods):
        item = methods[name]
        evaluated = item.get('evaluated', 0)
        paired = item.get('paired_vs_raw', {})
        path_count = f'{count(item.get("path_correct"))}/{count(evaluated)}'
        if compact:
            rows.append([label(name), path_count, percent(item.get('path_accuracy')),
                         number(paired.get('accuracy_difference'), scale=100)])
        else:
            changes = ('—' if not paired else
                       f'{count(paired.get("fixed"))}/{count(paired.get("broken"))}')
            rows.append([label(name), path_count, percent(item.get('path_accuracy')),
                         ci_text(path_ci(item)), percent(item.get('leaf_accuracy')),
                         changes, number(paired.get('accuracy_difference'), scale=100),
                         ci_text(paired.get('source_bootstrap_95ci'))])
    if compact:
        return table(['方法', '正确路径／已评价', '路径正确率%', '相对原模型差（百分点）'], rows)
    return table(['方法', '正确路径／已评价', '路径正确率%', '来源95%区间（%）',
                  '叶节点正确率%', '纠正／新增错误', '相对原模型差（百分点）',
                  '差值95%区间（百分点）'], rows)



def paired_result_interpretation(name, item):
    paired = item.get('paired_vs_raw', {})
    difference = paired.get('accuracy_difference')
    interval = paired.get('source_bootstrap_95ci')
    if difference is None or interval is None or len(interval) != 2:
        return f'{label(name)}的路径配对差值或来源区间尚不可用。'
    text = (f'{label(name)}相对原模型的路径正确率差为 '
            f'{number(difference, scale=100)} 个百分点，来源 95% 区间为 '
            f'{ci_text(interval)} 个百分点。')
    low, high = (float(interval[0]), float(interval[1]))
    if low > 0:
        return text + '该区间全为正，结果支持本组路径正确率提高。'
    if high < 0:
        return text + '该区间全为负，结果支持本组路径正确率降低。'
    return text + '该区间包含 0，路径提升尚无充分证据。'


def selected_paired_interpretations(methods):
    ordered = ordered_names(methods)
    balanced = next((name for name in ordered
                     if 'balanced' in name.lower()
                     and 'trace' not in name.lower()), None)
    prompts = [name for name in ordered if is_prompt_method(name)]
    chosen = ([balanced] if balanced is not None else []) + prompts
    return '\n\n'.join(paired_result_interpretation(name, methods[name]) for name in chosen)


def load_prompt_development(run_root):
    records = []
    for family, title in [('alignment', '证据对齐提示'),
                          ('condition_only', '当前条件提示')]:
        folder = run_root / 'diagnosis'
        summary = read_json(folder / (family + '_summary.json'), {})
        pairs = read_json(folder / (family + '_pair_metrics.json'), {})
        if not pairs and isinstance(summary, dict):
            pairs = summary.get('development_pair_metrics', {})
        quick = read_json(folder / (family + '_quick_summary.json'), {})
        if summary or pairs or quick:
            records.append({'family': family, 'title': title, 'summary': summary,
                            'pairs': pairs, 'quick': quick})
    return records


def prompt_variant_label(title, variant):
    if variant == 'raw':
        return '原模型'
    if variant == 'old_linear':
        return '原 16 维线性方法'
    if 'fit' in variant:
        return title + '（fit 校准）'
    if 'zero' in variant or 'raw' in variant:
        return title + '（未校准）'
    return title + '（' + variant + '）'


def item_correct_fraction(item):
    if not isinstance(item, dict) or item.get('n') is None:
        return '—'
    return count(item.get('correct')) + '/' + count(item.get('n'))


def prompt_development_text(records):
    if not records:
        return ''
    path_rows = []
    pair_rows = []
    seen_baselines = set()
    pair_counts = None
    interpretations = []
    quick_notes = []
    for record in records:
        title = record['title']
        data = record['summary']
        summary = data.get('summary', {})
        bootstrap = data.get('source_bootstrap', {})
        changes = data.get('paired_changes_vs_raw', {})
        for variant, item in summary.items():
            if not isinstance(item, dict) or not any(
                    key in item for key in ('dev_paths', 'dev_natural')):
                continue
            if variant == 'raw' and any(row[0] == '原模型' for row in path_rows):
                continue
            transitions = changes.get(variant, {}).get('paths', {})
            fixed = transitions.get('fixed_count')
            broken = transitions.get('broken_count')
            changed = '—' if fixed is None else count(fixed) + '/' + count(broken)
            difference_ci = bootstrap.get(variant, {}).get('path', {}).get('difference_vs_raw_ci')
            path_rows.append([
                prompt_variant_label(title, variant),
                item_correct_fraction(item.get('dev_paths')),
                item_correct_fraction(item.get('fit_natural')),
                item_correct_fraction(item.get('dev_natural')),
                item_correct_fraction(item.get('dev_PMA_diagnostic_family')),
                changed, ci_text(difference_ci)])
        if not summary and record['quick']:
            quick = record['quick'].get('summary', {})
            natural = quick.get('dev_natural', {})
            focus = quick.get('dev_focus', {})
            if natural.get('n') is not None:
                path_rows.append([
                    title + '（快速节点观察）', '—（未评价完整路径）', '—',
                    count(natural.get('candidate_correct')) + '/' + count(natural.get('n')),
                    count(focus.get('candidate_correct')) + '/' + count(focus.get('n')),
                    '—', '—'])
                quick_notes.append(
                    title + '当前仅有 ' + count(natural.get('n')) + ' 个自然节点及 ' +
                    count(focus.get('n')) + ' 个既有诊断子组节点的快速观察，未推算完整路径结果。')
        chosen = data.get('recommendation', {}).get('variant')
        pair_methods = record['pairs'].get('methods', {})
        variants = [key for key in pair_methods if key not in ('raw', 'old_linear')]
        pair_chosen = chosen if chosen in variants else (
            'prompt_zero' if 'prompt_zero' in variants else
            ('prompt_fit_ce' if 'prompt_fit_ce' in variants else
             (variants[0] if variants else None)))
        for variant in ['raw', 'old_linear', pair_chosen]:
            if variant is None or variant not in pair_methods:
                continue
            if variant in ('raw', 'old_linear'):
                if variant in seen_baselines:
                    continue
                seen_baselines.add(variant)
            item = pair_methods[variant]
            ap = item.get('AP', {})
            cf = item.get('CF', {})
            f0 = item.get('F0', {})
            pair_counts = pair_counts or (ap.get('n'), cf.get('n'), f0.get('n'))
            pair_rows.append([
                prompt_variant_label(title, variant),
                number(ap.get('difference_rms'), 4),
                percent(ap.get('variance_reduction')),
                percent(cf.get('mean_evidence_retention')),
                number(cf.get('correction_rms'), 4),
                percent(cf.get('correction_rms_over_ap_rms')),
                count(cf.get('response_direction_degraded')),
                number(f0.get('difference_rms'), 4),
                percent(f0.get('variance_reduction'))])
        if chosen in bootstrap:
            path = bootstrap[chosen].get('path', {})
            if path.get('difference_vs_raw') is not None:
                display_name = ('evidence_alignment_' if record['family'] == 'alignment'
                                else 'condition_only_') + chosen
                interpretations.append(paired_result_interpretation(
                    display_name, {'paired_vs_raw': {
                        'accuracy_difference': path.get('difference_vs_raw'),
                        'source_bootstrap_95ci': path.get('difference_vs_raw_ci')}}))
    if not path_rows and not pair_rows:
        return ''
    text = ('**开发集输入提示对照。** 证据对齐提示保留原输入，并补充变量、时间点、单位与阈值的判断要求；'
            '当前条件提示保留完整病例、定义和当前条件，移除其他树节点与动作。'
            '以下仅为开发集观察，两个候选分别报告。\n\n')
    if path_rows:
        text += table(['输入／配置', '开发完整路径', 'fit 自然节点', '开发自然节点',
                       'PMA 节点（诊断子组）', '纠正／新增路径错误',
                       '路径差95%区间（百分点）'], path_rows) + '\n'
    if interpretations:
        text += '\n\n'.join(interpretations) + '\n\n'
    if quick_notes:
        text += '\n\n'.join(quick_notes) + '\n\n'
    if pair_rows:
        text += ('下表汇总已完成的 A/P、C/F 与 F0 配对评价。差分指标不随常数 b 改变，'
                 '提示候选各列一行；不同 b 的路径和自然节点正确数见上表。\n\n')
        text += table(['方法', 'A/P RMS', 'A/P VR%', 'C/F ER%',
                       'C/F 修正 RMS', '修正／干扰%', '方向退化数',
                       'F0 RMS', 'F0 VR%'], pair_rows) + '\n'
        if pair_counts:
            text += ('对应 A/P、C/F、F0 配对数分别为 ' +
                     '/'.join(count(value) for value in pair_counts) +
                     '；A、P、C、F 的逐项指标及来源区间保存在对应 pair_metrics.json。\n\n')
        text += ('输入提示对照未施加原方法的逐对 fit C/F 硬约束。'
                 '表中的证据改动是开发集实测，表示不同保护条件，不能当作相同保护预算下的比较；'
                 '本表不据开发集数值推断 fit 预算是否满足。\n\n')
    return text


def expansion_section(metrics, inventory, diagnosis, prompt_records=None):
    groups = metrics['groups']
    overall = groups['all']
    expected = overall.get('case_count', inventory.get('official_manifest', {}).get('cases', 7458))
    source_count = overall.get('source_count', inventory.get('official_manifest', {}).get('source_clusters'))
    text = ('### 8.11 开发集错因与官方测试扩展\n\n' +
            diagnosis_text(diagnosis) +
            prompt_development_text(prompt_records or []) +
            '**官方 test 扩展。** 为扩大完整路径评价，本轮使用 '
            f'{count(expected)} 个原始官方 test 病例和参考路径，覆盖 {count(source_count)} 个指南同源簇。'
            '各配置在其扩展评分前固定；本轮扩大路径评价，未用官方 test 重新拟合或选择配置。'
            '完整路径要求决策序列与目标叶节点同时正确，叶节点正确率另列。'
            '配对干扰 VR、证据 ER 与证据修正幅度继续见第 8.10 节。\n\n')
    text += official_main_table(overall) + '\n'
    interpretations = selected_paired_interpretations(overall['methods'])
    if interpretations:
        text += interpretations + '\n\n'
    missing_rows = []
    for name in ordered_names(overall['methods']):
        item = overall['methods'][name]
        missing_rows.append([label(name), count(item.get('expected', expected)),
                             count(item.get('available')), count(item.get('evaluated')),
                             count(item.get('failed')), count(item.get('missing'))])
    text += table(['方法', '预期病例', '有记录', '已评价', '失败', '缺项'], missing_rows) + '\n'
    text += ('来源区间按指南同源簇进行 2,000 次配对重采样，种子为 11。'
             '纠正／新增错误和差值只在与原模型共有的已评价病例上计算。\n\n')
    official = inventory.get('official_manifest', {})
    roles = official.get('current_role_sources', {})
    if roles and set(roles) == {'none'}:
        text += ('按既有同源归并，官方 test 的来源与当前 fit、dev 和原 806 条测试子集的来源无交集。')
    history = official.get('history_cases', {})
    if 'earlier_recorded_source' in history:
        text += (f'其中 {count(history["earlier_recorded_source"])} 条路径的来源出现在更早输入记录中，'
                 f'{count(history.get("no_recorded_input_use"))} 条未见已盘点历史输入记录。')
    text += ('来源标记依据已有划分和历史记录，'
             '“未见记录”表示当前盘点范围，不作为此前从未使用的保证。\n\n')
    other_groups = [name for name in groups if name != 'all']
    if other_groups:
        rows = []
        for group_name in sorted(other_groups):
            group = groups[group_name]
            for name in ordered_names(group.get('methods', {})):
                item = group['methods'][name]
                paired = item.get('paired_vs_raw', {})
                rows.append([GROUP_LABELS.get(group_name, group_name),
                             count(group.get('source_count')), count(group.get('case_count')),
                             label(name),
                             f'{count(item.get("path_correct"))}/{count(item.get("evaluated"))}',
                             percent(item.get('path_accuracy')),
                             number(paired.get('accuracy_difference'), scale=100)])
        text += '**按使用记录分组。** 各字段形成的分组可能重叠，路径数不直接相加。\n\n'
        text += table(['分组', '来源簇', '预期路径', '方法', '正确／已评价',
                       '路径正确率%', '相对原模型差（百分点）'], rows) + '\n'
    if any(is_prompt_method(name) for name in overall['methods']):
        text += ('输入提示对照使用修改后的模型输入和额外前向，其指标单列；'
                 '官方表仅包含本轮实际评分的方法，候选开发观察见本节前表。\n\n')
    text += ('原模型、仅 fit 校准、原线性方法和固定优化配置共同报告；'
             '方法间的路径正确率差与来源区间按实际结果解释。'
             '本节保持旧 806 条测试、开发集、合成实验和五模型预设模拟的原有结果。\n\n'
             '完整扩展指标见 [test_expansion/metrics.json](meeting/test_expansion/metrics.json)，'
             '来源分组见同目录 summary.csv，开发集错因与复算代码一并保存。\n')
    return text


def brief_text(metrics, inventory):
    overall = metrics['groups']['all']
    expected = overall.get('case_count', inventory.get('official_manifest', {}).get('cases', 7458))
    return ('### 官方 test 扩展与开发集错因\n\n'
            f'MEGA-CDP 全集包含 42,353 个生成病例／路径；本轮另评价官方 test 的 '
            f'{count(expected)} 条路径，与原 806 条 train_val 留出测试分别报告。'
            '各配置在新增评分前固定。\n\n' +
            official_main_table(overall, compact=True) +
            '\n开发集错误分布、历史来源分组和按来源计算的不确定性见正文第 8.11 节。'
            '输入提示对照与缓存分数对消分别报告。\n')


def readme_text(metrics):
    overall = metrics['groups']['all']
    return (f'官方 test 扩展见正文第 8.11 节，覆盖 {count(overall.get("case_count"))} 条路径。'
            '开发集错因、所有固定方法的路径及叶节点结果、来源分组和配对差值区间'
            '见 [test_expansion](meeting/test_expansion/)。原 806 条留出测试与模拟结果保留原口径。')



def metadata_without_prompt_inputs(value):
    raw_keys = {
        'prompt', 'prompt_text', 'rendered_prompt', 'original_prompt',
        'input', 'raw_input', 'input_text', 'patient', 'patient_text',
        'guideline', 'guideline_text', 'case_text', 'vignette',
        'vignette_text', 'messages',
    }
    record_keys = {'cases', 'inputs', 'views', 'nodes', 'records'}
    if isinstance(value, list):
        return [metadata_without_prompt_inputs(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: metadata_without_prompt_inputs(item)
        for key, item in value.items()
        if key.lower() not in raw_keys
        and not (key.lower() in record_keys and isinstance(item, (list, dict)))
    }

def copy_packet(run_root, metric_path, packet_root):
    packet_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metric_path, packet_root / 'metrics.json')
    csv_path = metric_path.with_name('summary.csv')
    if csv_path.exists():
        shutil.copy2(csv_path, packet_root / 'summary.csv')
    # Only code and compact analysis/diagnostic metadata are included.
    # Case manifests, original guideline text, JSONL node caches and binary caches are excluded.
    for folder in ['analysis', 'diagnosis', 'inventory']:
        source = run_root / folder
        if not source.exists():
            continue
        dest = packet_root / folder
        dest.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.suffix in {'.py', '.json', '.csv'}:
                shutil.copy2(path, dest / path.name)
    inference = run_root / 'inference'
    if inference.exists():
        dest = packet_root / 'inference'
        dest.mkdir(exist_ok=True)
        for path in inference.glob('*.py'):
            shutil.copy2(path, dest / path.name)
    sources = run_root / 'inventory/sources.jsonl'
    if sources.exists():
        dest = packet_root / 'inventory'
        dest.mkdir(exist_ok=True)
        shutil.copy2(sources, dest / sources.name)
    for shard in [0, 1]:
        path = run_root / f'results/shard{shard}/results.jsonl'
        if path.exists():
            with path.open('rb') as source, gzip.open(
                    packet_root / f'path_results_shard{shard}.jsonl.gz',
                    'wb', compresslevel=6) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    # Prompt-run metadata is copied without rendered prompts or raw case inputs.
    metadata_names = {
        'metadata.json', 'prompt_metadata.json', 'metrics.json', 'summary.json',
        'status.json', 'configuration.json', 'resource_plan.json',
        'frozen_prompt.json', 'frozen.json', 'freeze.json', 'launch.json', 'done.json', 'failed.json',
    }
    for folder in ['prompt_results', 'condition_only_results']:
        prompt_root = run_root / folder
        if prompt_root.exists():
            for path in prompt_root.rglob('*.json'):
                if (path.name in metadata_names
                        or path.name.endswith(('.done.json', '.status.json', '.failed.json'))):
                    target = packet_root / folder / path.relative_to(prompt_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    value = metadata_without_prompt_inputs(read_json(path))
                    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                                      encoding='utf-8')
    for path in run_root.iterdir():
        if path.is_file() and path.suffix == '.py':
            shutil.copy2(path, packet_root / path.name)
    frozen = run_root / 'frozen_methods.json'
    if frozen.exists():
        shutil.copy2(frozen, packet_root / frozen.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path,
                        default=DEFAULT_RUN_ROOT / 'results/summary/metrics.json')
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--publish-worktree', type=Path, default=DEFAULT_PUBLISH_WORKTREE)
    parser.add_argument('--standalone-root', type=Path, default=DEFAULT_STANDALONE_ROOT)
    parser.add_argument('--diagnosis-summary', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    metric_path = args.metrics.resolve(strict=True)
    metrics = read_json(metric_path)
    if not isinstance(metrics, dict) or not metrics.get('groups', {}).get('all', {}).get('methods'):
        raise ValueError('Metrics must contain nonempty groups.all.methods')
    inventory = read_json(args.run_root / 'inventory/inventory.json', {})
    diagnosis, diagnosis_path = load_diagnosis(args.run_root, args.diagnosis_summary)
    prompt_records = load_prompt_development(args.run_root)
    report_root = args.publish_worktree / REPORT_RELATIVE_ROOT
    originals = {rel: (report_root / rel).read_text(encoding='utf-8')
                 for rel in [PAPER_FILE, BRIEF_FILE, README_FILE]}
    paper = replace_or_insert(originals[PAPER_FILE], 'aster-official-expansion-dataset',
                              dataset_text(inventory, metrics),
                              anchor=r'^### 7\.1\b[^\n]*', after=True)
    paper = replace_or_insert(paper, 'aster-official-expansion-section',
                              expansion_section(metrics, inventory, diagnosis, prompt_records),
                              anchor=r'^## 9\.[^\n]*')
    brief = replace_or_insert(originals[BRIEF_FILE], 'aster-official-expansion-brief',
                              brief_text(metrics, inventory),
                              anchor=r'^## 4\.[^\n]*')
    readme = replace_or_insert(originals[README_FILE], 'aster-official-expansion-readme',
                               readme_text(metrics))
    changed = {PAPER_FILE: paper, BRIEF_FILE: brief, README_FILE: readme}
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'metrics': str(metric_path),
                          'diagnosis': str(diagnosis_path) if diagnosis_path else None,
                          'targets': [str(report_root / rel) for rel in changed]},
                         ensure_ascii=False))
        return
    # All text is prepared before existing documents are changed.
    for rel, text in changed.items():
        target = report_root / rel
        target.write_text(text, encoding='utf-8')
        standalone = args.standalone_root / rel
        standalone.parent.mkdir(parents=True, exist_ok=True)
        standalone.write_text(text, encoding='utf-8')
    copy_packet(args.run_root, metric_path, report_root / PACKET_RELATIVE_ROOT)
    print(json.dumps({'state': 'complete',
                      'updated': [str(report_root / rel) for rel in changed],
                      'standalone_root': str(args.standalone_root),
                      'results_packet': str(report_root / PACKET_RELATIVE_ROOT),
                      'git_actions': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()

