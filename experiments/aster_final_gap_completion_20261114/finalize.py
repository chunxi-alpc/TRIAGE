"""Summarize completed outputs, render measured results, archive, and fast-forward PR 69."""
import csv
import gzip
import shutil
import subprocess
import time
import sys
from common import *

def table_csv(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

def percent(x):
    return f'{100 * x:.2f}'

def interval(values):
    return '[' + ', '.join(percent(x) for x in values) + ']'

def official_table():
    result = []
    labels = dict(qwen='Qwen3-8B-IT', llama31='Llama-3.1-8B-Instruct', gemma2='Gemma-2-9B-IT',
                  gemma3='Gemma-3-27B-IT', medgemma='MedGemma-27B-Text-IT')
    for model, label in labels.items():
        if model == 'llama31':
            old = read(R / 'models/llama31/evaluation/mega_official/metrics.json')
            for value in old['metrics']:
                if value['metric'] == 'dynamic_path':
                    result.append(dict(model=model, label=label, method=value['method'], n=value['n'], correct=value['correct'],
                        accuracy=value['accuracy'], delta_accuracy=value['delta_accuracy'],
                        delta_low=value['paired_source_ci_low'], delta_high=value['paired_source_ci_high'],
                        sources=value['sources'], origin='existing_complete'))
        else:
            value = read(OUT / 'e6' / model / 'evaluation/mega_official/paired_statistics.json')
            for method, stat in value['paths'].items():
                result.append(dict(model=model, label=label, method=method, n=stat['n'], correct=stat['correct'],
                    accuracy=stat['accuracy'], delta_accuracy=stat['difference_vs_raw'],
                    delta_low=stat['difference_vs_raw_ci'][0], delta_high=stat['difference_vs_raw_ci'][1],
                    sources=stat['sources'], origin='completed_gap'))
    return result

def e5_table():
    results = []
    names = [kind + '_seed' + str(seed) for kind in ['lora', 'lofit', 'loreft'] for seed in [11, 22, 33]]
    names += ['leace_pair', 'iti_style', 'sea_adaptation', 'strong_evidence_prompt', 'CC']
    for name in names:
        metric = read(OUT / 'e5' / name / 'metrics.json')
        for scope, stat in metric['paths'].items():
            results.append(dict(method=name, scope=scope, n=stat['n'], correct=stat['correct'], accuracy=stat['accuracy'],
                accuracy_low=stat['accuracy_ci'][0], accuracy_high=stat['accuracy_ci'][1],
                delta=stat['difference_vs_raw'], delta_low=stat['difference_vs_raw_ci'][0], delta_high=stat['difference_vs_raw_ci'][1],
                corrected=stat['corrected_vs_raw'], new_errors=stat['new_errors_vs_raw'], sources=stat['sources']))
    ext = read(OUT / 'e5/external_paired_ci/metrics.json')['methods']
    for name, entry in ext.items():
        for scope, stat in entry['paths'].items():
            results.append(dict(method=name, scope=scope, n=stat['n'], correct=stat['correct'], accuracy=stat['accuracy'],
                accuracy_low=stat['accuracy_ci'][0], accuracy_high=stat['accuracy_ci'][1],
                delta=stat['difference_vs_raw'], delta_low=stat['difference_vs_raw_ci'][0], delta_high=stat['difference_vs_raw_ci'][1],
                corrected=stat['corrected_vs_raw'], new_errors=stat['new_errors_vs_raw'], sources=stat['sources']))
    return results

def transfer_table():
    results = []
    for short, scope, key in [('nli500', 'nli4ct', 'original_statements'),
                               ('nli5500', 'nli4ct_official5500', 'all_statements'),
                               ('medpic467', 'medpic_official467', None)]:
        metric = read(OUT / 'e7/qwen' / short / 'evaluation' / scope / 'metrics.json')
        native = metric['native_population_statistics']
        if key:
            native = native[key]
        for method, stat in native.items():
            results.append(dict(population=short, method=method, n=stat['n'], correct=stat['correct'], accuracy=stat['accuracy'],
                low=stat['accuracy_ci'][0], high=stat['accuracy_ci'][1], delta=stat['difference_vs_raw'],
                delta_low=stat['difference_vs_raw_ci'][0], delta_high=stat['difference_vs_raw_ci'][1],
                sources=stat['sources'], clustering=metric['clustering']))
    return results

def plots(official, e5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    dest = OUT / 'figures'
    dest.mkdir(exist_ok=True)
    models = list(dict.fromkeys(r['label'] for r in official))
    fig, ax = plt.subplots(figsize=(9.6, 4.5), layout='constrained')
    methods = [('raw', 'Raw', '#8d99ae'), ('bias_only', 'Source intercept', '#2a9d8f'), ('aster_cr', 'ASTER-CR', '#3659a2')]
    for i, (method, label, color) in enumerate(methods):
        values = [100 * next(r['accuracy'] for r in official if r['label'] == model and r['method'] == method) for model in models]
        bars = ax.bar(np.arange(len(models)) + (i - 1) * .25, values, width=.24, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set_xticks(np.arange(len(models)), [s.replace('-Instruct', '').replace('-Text-IT', '').replace('-IT', '') for s in models])
    ax.set_ylabel('Complete-path accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Official MEGA: 7,458 paths, 418 sources per model')
    for extension in ['svg', 'png']:
        fig.savefig(dest / ('e6_official_accuracy.' + extension), dpi=220)
    plt.close(fig)
    names = list(dict.fromkeys(r['method'] for r in e5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 11), sharey=True, layout='constrained')
    for ax, scope, title in zip(axes, ['natural', 'interference'], ['Natural paths (n=806)', 'Interference paths (n=2,418)']):
        records = [next(r for r in e5 if r['method'] == name and r['scope'] == scope) for name in names]
        ys = np.arange(len(names))
        xs = np.array([100 * r['accuracy'] for r in records])
        err = np.array([[100 * (r['accuracy'] - r['accuracy_low']) for r in records],
                        [100 * (r['accuracy_high'] - r['accuracy']) for r in records]])
        ax.errorbar(xs, ys, xerr=np.maximum(err, 0), fmt='o', color='#3659a2', capsize=2, markersize=4)
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel('Path accuracy with source-cluster 95% CI (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    for extension in ['svg', 'png']:
        fig.savefig(dest / ('e5_paths.' + extension), dpi=220)
    plt.close(fig)

def report(official, e5, transfer):
    text = ['# ASTER-CR 最小补实验结果', '',
            '当前E1协议的四个官方MEGA缺口、两款27B受控评价、E5十三行干扰评价和批准的CPU统计已完成。全部结果使用既有冻结参数，新增训练和模型下载均为0。', '',
            '## E6 官方MEGA', '',
            '| 模型 | raw正确/7458 | 独立截距正确/7458 | E1正确/7458 | E1相对raw差值及95%区间（百分点） |',
            '| --- | ---: | ---: | ---: | --- |']
    for model in list(dict.fromkeys(r['model'] for r in official)):
        records = {r['method']: r for r in official if r['model'] == model}
        e = records['aster_cr']
        text.append(f"| {e['label']} | {records['raw']['correct']} | {records['bias_only']['correct']} | {e['correct']} | {percent(e['delta_accuracy'])} {interval([e['delta_low'], e['delta_high']])} |")
    text += ['', '每模型均覆盖7,458条唯一路径、418来源；三方法共享当前检查点前向，保留45,627个金标准自然节点。Llama完整结果沿用既有归档。Gemma2保留原2,379条，接续5,079条。', '',
             '![官方MEGA结果](figures/e6_official_accuracy.png)', '', '## 两款27B受控评价', '',
             '| 模型 | raw正确/806 | 独立截距正确/806 | E1正确/806 | 新前向 |', '| --- | ---: | ---: | ---: | ---: |']
    for model in ['gemma3', 'medgemma']:
        m = read(OUT / 'e6' / model / 'evaluation/test/metrics.json')['dynamic_paths']
        text.append(f"| {model} | {m['raw']['natural']['correct']} | {m['calibration']['natural']['correct']} | {m['E1']['natural']['correct']} | 0 |")
    text += ['', '受控路径由已审计全头缓存和当前E1参数重建，覆盖40来源。A/P、C/F、纠正和新增错误及来源配对区间保存在对应指标文件。', '',
             '## E5 自然与干扰路径', '',
             '| 方法/种子 | 自然正确/806 | 干扰正确/2418 | 干扰相对raw差值及95%区间（百分点） |', '| --- | ---: | ---: | --- |']
    for method in list(dict.fromkeys(r['method'] for r in e5)):
        records = {r['scope']: r for r in e5 if r['method'] == method}
        s = records['interference']
        text.append(f"| {method} | {records['natural']['correct']} | {s['correct']} | {percent(s['delta'])} {interval([s['delta_low'], s['delta_high']])} |")
    text += ['', '每行使用同一组40来源抽样，同时计算方法和raw的差值；2,000次配对来源重采样。A/P为2,293对、C/F为40对、F0为6对另列。所有随机种子均保留。', '',
             '![E5路径结果](figures/e5_paths.png)', '', '## Qwen迁移评价', '',
             '| 总体 | 方法 | 正确/分母 | 准确率95%区间（%） | 来源簇 |', '| --- | --- | ---: | --- | ---: |']
    for r in transfer:
        text.append(f"| {r['population']} | {r['method']} | {r['correct']}/{r['n']} | {interval([r['low'], r['high']])} | {r['sources']} |")
    text += ['', 'NLI500按113个试验或试验组合聚类；NLI5500按500个原始声明及其变体聚类。两种口径已经用户确认并分别标注。MedPIC467包含开发题，作为描述性评价，按376来源聚类。独立截距来自MEGA的761个拟合节点、80个来源；未使用目标任务标签拟合或选参。', '',
             '## 复用与文件入口', '',
             '- E5保留已修复LoReFT适配器、全部原随机种子和训练参数；PSR seed22训练及既有11行前向均未重跑。',
             '- 同卡LoRA/LoFiT任务连续复用主干；LoReFT三个种子及LEACE/ITI/SEA分别共享合法的原模型提示前缀，修正分数独立保存。',
             '- SEA新增最终层原始残差保存在服务器private_cache目录，公开归档保存数值预测和缓存位置。',
             '- E7其他四模型前向与完整统计、Qwen MedPIC188及E8比较沿用既有结果。模型版本追溯见provenance/model_identity.json。',
             '- 完整表格：e6_official_summary.csv、e5_path_summary.csv、e7_qwen_summary.csv；逐例及参数文件见artifact_manifest.json。',
             '- 代码位于experiments/aster_final_gap_completion_20261114；授权和协议选择见authorization.json，调度及性能见queue_status.json和各worker记录。', '']
    (OUT / 'REPORT.zh.md').write_text('\n'.join(text))

def archive_and_publish():
    from reconcile_status import reconcile
    reconciled_files = reconcile()
    provenance = OUT / 'provenance'
    provenance.mkdir(exist_ok=True)
    for name in ['MINIMAL_EXECUTION_PLAN.json', 'model_identity.json', 'protocol_current.json',
                 'qwen_transfer_provenance.json', 'adapter_effectiveness_audit.json', 'unarchived_extra_parameters.json',
                 'publication_prediction_mapping.json', 'artifact_sha256_registry.json', 'e8_comparisons.json']:
        shutil.copy2(AUDIT / name, provenance / name)
    # Publish prompt-free results only. Activations, raw inputs, and operational logs stay on the server.
    files = []
    manifest = []
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(OUT)
        if any(part in ['private_cache', 'logs', 'features', '.git'] for part in rel.parts):
            continue
        if p.name in ['publication_status.json', 'artifact_manifest.json'] or '.tmp' in p.name:
            continue
        if any(part.startswith('shard') for part in rel.parts) and p.suffix == '.jsonl' and p.name != 'nodes.jsonl':
            continue
        if p.suffix == '.npz' and '/models/E1/full/' not in str(p):
            continue
        if p.suffix == '.jsonl':
            dest = p.with_name(p.name + '.gz')
            with p.open('rb') as source, gzip.open(dest, 'wb', compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            files.append(dest)
            manifest.append(dict(server_source=str(p), repository_file=str(dest.relative_to(PROJECT)),
                                 original_bytes=p.stat().st_size, archive_bytes=dest.stat().st_size))
        elif p.suffix in ['.json', '.csv', '.md', '.png', '.svg', '.npz']:
            files.append(p)
            manifest.append(dict(server_source=str(p), repository_file=str(p.relative_to(PROJECT)), bytes=p.stat().st_size))
    write(OUT / 'artifact_manifest.json', dict(complete=True, files=manifest, raw_clinical_text_published=False,
          activation_cache_root=str(OUT / 'private_cache'), original_artifacts_preserved=True))
    files.append(OUT / 'artifact_manifest.json')
    files += list(CODE.glob('*.py'))
    files += reconciled_files
    doc = PROJECT / 'docs/ASTER/final-gap-completion-20261114.zh.md'
    doc.write_text('# ASTER-CR 最小补实验结果\n\n当前E1剩余评价已完成，包含E6官方路径、E5干扰路径及E7独立截距统计。\n\n'
                   '[完整结果、图表与统计](../../reports/aster_final_gap_completion_20261114/REPORT.zh.md)\n')
    files.append(doc)
    plan = read(OUT / 'publication_plan.json')
    paths = [str(p.relative_to(PROJECT)) for p in files]
    for start in range(0, len(paths), 80):
        subprocess.run(['git', '-C', str(PROJECT), 'add', '-f', '--', *paths[start:start+80]], check=True)
    staged = subprocess.check_output(['git', '-C', str(PROJECT), 'diff', '--cached', '--name-only'], text=True)
    if staged.strip():
        subprocess.run(['git', '-C', str(PROJECT), 'commit', '-m', 'Complete ASTER E5 interference, E6 official paths, and frozen transfer statistics'], check=True)
    commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    push = subprocess.run(['git', '-C', str(PROJECT), 'push', plan['remote'], 'HEAD:refs/heads/' + plan['branch']],
                          capture_output=True, text=True, timeout=300)
    write(OUT / 'publication_status.json', dict(state='published' if push.returncode == 0 else 'commit_ready_push_failed',
          commit=commit, pull_request=69, branch=plan['branch'], exit_code=push.returncode,
          error=push.stderr[-2000:] if push.returncode else None, force=False, time=time.time()))
    if push.returncode:
        raise RuntimeError('Results are committed locally; non-force GitHub push failed')

def main():
    queue = read(OUT / 'queue_status.json', {})
    cpu = read(OUT / 'cpu_status.json', {})
    if queue.get('state') != 'complete' or cpu.get('state') != 'complete':
        (OUT / 'REPORT.zh.md').write_text('# ASTER-CR 补实验状态\n\n任务尚未全部完成。\n\n'
                '具体失败或未完成任务见queue_status.json和cpu_status.json；已完成逐例结果保留，未发布完整完成声明。\n')
        raise RuntimeError('Incomplete execution; results remain on server for continuation')
    official = official_table()
    e5 = e5_table()
    transfer = transfer_table()
    table_csv(OUT / 'e6_official_summary.csv', official)
    table_csv(OUT / 'e5_path_summary.csv', e5)
    table_csv(OUT / 'e7_qwen_summary.csv', transfer)
    plots(official, e5)
    report(official, e5, transfer)
    archive_and_publish()
    write(OUT / 'completion.json', dict(complete=True, cpu_complete=True, gpu_units_complete=17,
          experiments_complete=True, archived=True, github_submitted=True, time=time.time()))

if __name__ == '__main__':
    main()
