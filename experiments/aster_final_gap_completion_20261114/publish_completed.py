"""Archive completed CPU results and existing artifacts while the GPU queue continues."""
import csv
import gzip
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from common import *
from reconcile_status import AUDIT_COPY, reconcile


def compressed(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, dest.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as out:
            shutil.copyfileobj(src, out)
    return dest


def verify_cpu():
    report = {}
    for model in ['gemma3', 'medgemma']:
        root = OUT / 'e6' / model / 'evaluation/test'
        predictions = rows(root / 'path_predictions.jsonl')
        methods = sorted({r['method'] for r in predictions})
        assert set(methods) == {'raw', 'calibration', 'E1'}
        details = {}
        for method in methods:
            part = [r for r in predictions if r['method'] == method]
            assert len(part) == len({r['case_id'] for r in part}) == 806
            assert len({r['source_id'] for r in part}) == 40
            assert not any(r.get('error') for r in part)
            details[method] = dict(paths=806, unique_cases=806, sources=40,
                                   correct=sum(r['correct'] for r in part), duplicates=0)
        report[model] = dict(methods=details, new_model_forwards=0,
                             predictions=str((root / 'path_predictions.jsonl.gz').relative_to(PROJECT)))
    metric = read(OUT / 'e5/CC/metrics.json')
    cc = rows(OUT / 'e5/CC/path_predictions.jsonl')
    for scope, expected in [('natural', 806), ('interference', 2418)]:
        part = [r for r in cc if bool(r['interference']) == (scope == 'interference')]
        assert len(part) == len({(r['case_id'], r['variant']) for r in part}) == expected
        assert len({r['source_id'] for r in part}) == 40
        assert metric['paths'][scope]['n'] == expected
        assert sum(r['correct'] for r in part) == metric['paths'][scope]['correct']
    report['CC'] = metric['paths']
    external = read(OUT / 'e5/external_paired_ci/metrics.json')['methods']
    assert len(external) == 11
    for item in external.values():
        for scope, expected in [('natural', 806), ('interference', 2418)]:
            stat = item['paths'][scope]
            assert stat['n'] == expected and stat['sources'] == 40 and stat['bootstrap']['paired']
            assert len(stat['difference_vs_raw_ci']) == 2
    report['external_paired_intervals'] = dict(method_rows=11, populations_per_method=2, paired_source_bootstrap=True)
    for short, scope, n, view_count in [('nli500', 'nli4ct', 500, 2864),
                                      ('nli5500', 'nli4ct_official5500', 5500, 5500),
                                      ('medpic467', 'medpic_official467', 467, 7005)]:
        root = OUT / 'e7/qwen' / short / 'evaluation' / scope
        data = rows(root / 'predictions.jsonl')
        assert len(data) == len({r['view_id'] for r in data}) == view_count
        assert all(set(r['predictions']) == {'raw', 'calibration', 'E1'} for r in data)
        if short == 'medpic467':
            q = rows(root / 'question_predictions.jsonl')
            natural = [r for r in q if r['variant'] in ['original', 'full']]
            assert Counter(r['method'] for r in natural) == dict(raw=n, calibration=n, E1=n)
            assert len({r['item_id'] for r in natural}) == n
        else:
            natural = [r for r in data if r['original']] if short == 'nli500' else data
            assert len(natural) == n
        report[short] = dict(population_per_method=n, saved_views=view_count, methods=['raw', 'calibration', 'E1'],
                             sources=len({r['source_id'] for r in natural}), new_model_forwards=0)
    report.update(complete=True, new_model_forwards=0, models_loaded=0,
                  updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    write(OUT / 'cpu_acceptance.json', report)
    return report


def cpu_report(acceptance):
    from finalize import transfer_table, table_csv
    transfer = transfer_table()
    table_csv(OUT / 'e7_qwen_summary.csv', transfer)
    e5 = []
    cc = read(OUT / 'e5/CC/metrics.json')['paths']
    external = read(OUT / 'e5/external_paired_ci/metrics.json')['methods']
    methods = dict(CC=dict(paths=cc), **external)
    for name, value in methods.items():
        for scope, v in value['paths'].items():
            e5.append(dict(method=name, scope=scope, n=v['n'], sources=v['sources'], correct=v['correct'],
                           accuracy=v['accuracy'], delta=v['difference_vs_raw'],
                           low=v['difference_vs_raw_ci'][0], high=v['difference_vs_raw_ci'][1],
                           corrected=v['corrected_vs_raw'], new_errors=v['new_errors_vs_raw']))
    table_csv(OUT / 'e5_cpu_paired_summary.csv', e5)
    text = ['# ASTER-CR 已完成CPU补算结果', '',
            '**C01–C07七项独立CPU任务全部完成。** 使用现有参数、节点分数和激活缓存，模型加载、新前向和训练均为0。'
            'C08（E5十三行接续统计）和C09（E6四个官方单元合并统计）依赖正在执行的GPU任务，随产物完成。', '',
            '[当前实验状态](../../docs/ASTER/experiments.zh.md) · [逐例验收](cpu_acceptance.json) · '
            '[归档清单](cpu_archive_manifest.json)', '',
            '## 两款27B受控路径', '',
            '| 模型 | raw正确/806 | 独立截距正确/806 | E1正确/806 | 来源 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for model in ['gemma3', 'medgemma']:
        m = acceptance[model]['methods']
        text.append(f"| {model} | {m['raw']['correct']} | {m['calibration']['correct']} | {m['E1']['correct']} | 40 |")
    text += ['', '三方法均保存806个唯一case，重复和执行错误为0；对应A/P、C/F、纠正/新增错误及来源区间见e6各模型evaluation/test/metrics.json。', '',
             '## CC与11行新增方法的配对区间', '',
             '| 方法 | 自然正确/806 | 干扰正确/2418 | 干扰相对raw差值及95%区间（百分点） |',
             '| --- | ---: | ---: | --- |']
    for name in methods:
        natural, interference = methods[name]['paths']['natural'], methods[name]['paths']['interference']
        lo, hi = interference['difference_vs_raw_ci']
        text.append(f"| {name} | {natural['correct']} | {interference['correct']} | {100*interference['difference_vs_raw']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] |")
    text += ['', '每种方法与raw共享同一组40来源抽样，2,000次来源配对重采样；保留各随机种子。'
             '自然及干扰的差值区间、纠正与新增错误见e5_cpu_paired_summary.csv和各metrics.json。', '',
             '## Qwen冻结迁移', '',
             '| 总体 | 方法 | 正确/分母 | 准确率95%区间（%） | 来源簇 |',
             '| --- | --- | ---: | --- | ---: |']
    for r in transfer:
        text.append(f"| {r['population']} | {r['method']} | {r['correct']}/{r['n']} | [{100*r['low']:.2f}, {100*r['high']:.2f}] | {r['sources']} |")
    text += ['', '按用户确认保留两种NLI口径：NLI500按113个试验或试验组合聚类，NLI5500按500个原始声明及其变体聚类。'
             'MedPIC467按376来源聚类，包含开发题，作描述性评价。', '',
             '独立截距b₀=3.0993948961894406来自MEGA源任务761个拟合节点、80来源。目标任务标签用于评价，未用于截距拟合。'
             'Qwen MedPIC188沿用已完成结果，本次补充模型与参数版本追溯；其他四个模型的四个总体前向和统计沿用既有归档。', '',
             '## 本次归档及旧状态更新', '',
             '- 七项CPU统计、脱敏逐例、代码、配置、第二轮审计及证据附件一并提交。',
             '- Qwen NLI5500和MedPIC467原有raw/E1逐例及指标补充归档，原文件中的E0/E2/E3/E4保留原身份；当前三方法结果单独保存在本目录。',
             '- 补充SEA冻结算子、DoRA三个适配器、INLP选中投影及Qwen Base下载清单；大文件分片方式见frozen_artifacts_manifest.json。',
             '- Llama官方、Qwen Base官方、两27B冻结/迁移及受控CPU重建的完成状态已按产物更新；原控制器记录存入provenance/legacy_status。',
             '- LoReFT使用修复后的非零旋转适配器；PSR seed22已有真实参数更新。已完成训练、E5自然路径、11行前向、E7前向和E8拟合继续复用。',
             '- 未确认的上游不可变revision继续保留NEEDS_VALIDATION，具体缺项见审计model_identity.json；完成计算的单元保持复用。', '',
             '所有公开逐例只包含ID、标签、数值分数、路径及统计；原始临床输入、模型主干权重和激活缓存留在服务器。', '']
    (OUT / 'CPU_RESULTS.zh.md').write_text('\n'.join(text))


def archive():
    files, records = [], []
    AUDIT_COPY.mkdir(parents=True, exist_ok=True)
    # The existing evidence bundle contains metadata and IDs only, already reviewed during the audit.
    for source in [AUDIT / 'ASTER_FINAL_MISSING_EXPERIMENTS_AUDIT.md', AUDIT / 'AUDIT_EVIDENCE.zip', *AUDIT.glob('*.json')]:
        dest = AUDIT_COPY / source.name
        if not dest.exists():
            shutil.copy2(source, dest)
        files.append(dest)
    verification = verify_cpu()
    cpu_report(verification)
    roots = [OUT / 'e6' / m / 'evaluation/test' for m in ['gemma3', 'medgemma']]
    roots += [OUT / 'e5/CC', OUT / 'e5/external_paired_ci', OUT / 'e7/qwen']
    for root in roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ['.jsonl', '.json', '.csv']:
                continue
            dest = compressed(p, p.with_name(p.name + '.gz')) if p.suffix == '.jsonl' else p
            files.append(dest)
            records.append(dict(server_source=str(p), repository_file=str(dest.relative_to(PROJECT)),
                                 original_bytes=p.stat().st_size, archive_bytes=dest.stat().st_size, scope='completed_CPU'))
    for model in ['gemma3', 'medgemma']:
        files.append(OUT / 'e6' / model / 'test_done.json')
    # Existing numerical source predictions are retained independently from the new three-method outputs.
    for scope in ['nli4ct_official5500', 'medpic_official467']:
        for name in ['predictions.jsonl', 'pair_predictions.jsonl', 'question_predictions.jsonl', 'metrics.json']:
            p = F / 'evaluation' / scope / name
            if not p.exists() or p.stat().st_size == 0:
                continue
            dest = OUT / 'provenance/existing_qwen_transfer' / scope / name
            if p.suffix == '.jsonl':
                dest = compressed(p, dest.with_name(dest.name + '.gz'))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            files.append(dest)
            records.append(dict(server_source=str(p), repository_file=str(dest.relative_to(PROJECT)),
                                 original_bytes=p.stat().st_size, archive_bytes=dest.stat().st_size,
                                 scope='existing_predictions_original_method_names_preserved'))
    provenance = OUT / 'provenance'
    for name in ['model_identity.json', 'protocol_current.json', 'qwen_transfer_provenance.json',
                 'adapter_effectiveness_audit.json', 'unarchived_extra_parameters.json', 'e8_comparisons.json']:
        shutil.copy2(AUDIT / name, provenance / name)
        files.append(provenance / name)
    shutil.copy2(D / 'configs/independent_intercepts_reused.json', provenance / 'independent_intercepts_reused.json')
    files.append(provenance / 'independent_intercepts_reused.json')
    parameter_records = []
    for item in read(AUDIT / 'unarchived_extra_parameters.json'):
        p = Path(item['path'])
        rel = Path('reports') / p.relative_to(EXT.parent)
        dest = PROJECT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        files.append(dest)
        parameter_records.append(dict(**item, repository_file=str(rel), sha256_source='existing_audit_record'))
    sea = E5 / 'representation_e1_plan/sea_adaptation/operator.pt'
    # GitHub per-file limit: keep reversible ordered gzip chunks below 48 MiB.
    sea_dir = PROJECT / 'reports/aster_e5_gaps_20261112/representation_e1_plan/sea_adaptation'
    packed = OUT / 'private_cache/sea_operator.pt.gz'
    compressed(sea, packed)
    chunks = []
    with packed.open('rb') as src:
        index = 1
        while chunk := src.read(48 * 1024 * 1024):
            dest = sea_dir / f'operator.pt.gz.part{index:02}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(chunk)
            files.append(dest)
            chunks.append(str(dest.relative_to(PROJECT)))
            index += 1
    parameter_records.append(dict(path=str(sea), size=sea.stat().st_size,
                                  sha256='dc423b9d09b9587ad802742063e0071f46e905122312df4d801d95a6d456cb91',
                                  sha256_source='existing_audit_record', chunks=chunks,
                                  reconstruction='Concatenate chunks in listed order, then gzip-decompress to operator.pt'))
    download = Path('/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/qwen_base_download_manifest_v2.json')
    dest = provenance / download.name
    shutil.copy2(download, dest)
    files.append(dest)
    parameter_records.append(dict(path=str(download), repository_file=str(dest.relative_to(PROJECT))))
    write(OUT / 'frozen_artifacts_manifest.json', dict(files=parameter_records, newly_computed_weight_hashes=0))
    files.append(OUT / 'frozen_artifacts_manifest.json')
    files += reconcile()
    plan = read(OUT / 'publication_plan.json')
    plan.update(completed_CPU_publication='immediate_user_requested', only_after_all_tasks_complete=True,
                only_after_all_tasks_complete_scope='Final GPU completion archive; CPU archive is published separately')
    write(OUT / 'publication_plan.json', plan)
    for name in ['cpu_status.json', 'cpu_acceptance.json', 'CPU_RESULTS.zh.md', 'e7_qwen_summary.csv',
                 'e5_cpu_paired_summary.csv', 'authorization.json', 'execution_provenance.json',
                 'queue_manifest.json', 'publication_plan.json', 'e5_ready.json']:
        files.append(OUT / name)
    files += list(CODE.glob('*.py'))
    files = list(dict.fromkeys(files))
    write(OUT / 'cpu_archive_manifest.json', dict(scope='Completed CPU and previously unarchived frozen artifacts',
          complete=True, gpu_completion_claimed=False, raw_clinical_text_published=False, records=records,
          repository_files=[str(p.relative_to(PROJECT)) for p in files]))
    files.append(OUT / 'cpu_archive_manifest.json')
    # Stage only the explicit completed-output list; active GPU files continue in place.
    for i in range(0, len(files), 70):
        subprocess.run(['git', '-C', str(PROJECT), 'add', '-f', '--',
                        *[str(p.relative_to(PROJECT)) for p in files[i:i+70]]], check=True)
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files), cpu_tasks_complete=7,
                deferred_cpu_tasks=['C08', 'C09'], frozen_artifacts=len(parameter_records))


if __name__ == '__main__':
    print(json.dumps(archive()))
