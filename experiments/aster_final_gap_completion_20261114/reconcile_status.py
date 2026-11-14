"""Refresh the existing experiment tracker from completed artifacts and queue state."""
from datetime import datetime, timezone
import re
import shutil
from common import *

AUDIT_COPY = PROJECT / 'reports/aster_final_missing_experiments_audit_20261114'
STATUS_REL = 'reports/aster_final_gap_completion_20261114/execution_status.json'
TRACKER_REL = 'docs/ASTER/experiments.zh.md'
METHODS = [f'{kind}_seed{seed}' for kind in ['lora', 'lofit', 'loreft'] for seed in [11, 22, 33]]
METHODS += ['leace_pair', 'iti_style', 'sea_adaptation', 'strong_evidence_prompt']


def saved_status(path, value, changed, sync_source=False):
    old = OUT / 'provenance/legacy_status' / path.relative_to(PROJECT)
    if path.exists() and not old.exists():
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, old)
    value['previous_snapshot'] = str(old.relative_to(PROJECT)) if old.exists() else None
    value['current_status'] = STATUS_REL
    value['current_status_server'] = str(OUT / 'execution_status.json')
    write(path, value)
    changed.append(path)
    if old.exists():
        changed.append(old)
    if sync_source:
        rel = path.relative_to(PROJECT)
        base = REPO6 if rel.parts[1] == 'aster_e678_20261112' else REPO5
        source = base / rel
        if source.exists():
            write(source, value)


def block(text, key, body):
    start, end = f'<!-- {key}-START -->', f'<!-- {key}-END -->'
    value = start + '\n' + body.strip() + '\n' + end
    if start in text:
        return re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: value, text, flags=re.S)
    first, rest = text.split('\n', 1)
    return first + '\n\n' + value + '\n' + rest


def reconcile():
    changed = []
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    queue = read(OUT / 'queue_status.json', {})
    manifest = read(OUT / 'queue_manifest.json', {})
    cpu = read(OUT / 'cpu_status.json', {})
    completed = [k for k, v in cpu.get('tasks', {}).items() if v.get('state') == 'complete']
    assert set(completed) == {f'C{i:02}' for i in range(1, 8)}
    official = {}
    for model, job in [('qwen', 'G01'), ('gemma2', 'G02'), ('gemma3', 'G03'), ('medgemma', 'G04')]:
        done = (OUT / 'e6' / model / 'evaluation/mega_official/completed.json').exists()
        task = queue.get('jobs', {}).get(job, {})
        state = 'COMPLETE_CURRENT' if done else ('RUNNING' if task.get('state') == 'running' else
                'FAILED' if task.get('state') == 'failed' else 'PARTIAL_CURRENT')
        official[model] = dict(state=state, calculation_complete=done, expected_paths_per_method=7458,
                               expected_sources=418, job=job, queue_state=task.get('state', 'pending'),
                               output=str((OUT / 'e6' / model / 'evaluation/mega_official').relative_to(PROJECT)),
                               identity_status='NEEDS_VALIDATION', identity_reason='Upstream immutable revision unavailable')
    all_e6 = all(x['calculation_complete'] for x in official.values())
    e5 = {}
    for name in METHODS:
        done = (OUT / 'e5' / name / 'done.json').exists()
        e5[name] = dict(state='COMPLETE_CURRENT' if done else 'PARTIAL_CURRENT', natural_paths=806,
                        interference_complete=done, expected_interference_paths=2418,
                        output=str((OUT / 'e5' / name).relative_to(PROJECT)))
    all_e5 = all(x['interference_complete'] for x in e5.values())
    state = dict(updated_at=now, scope='E5-E8 current E1 completion; existing fits reused',
                 canonical_tracker=TRACKER_REL, queue_state=queue.get('state'),
                 independent_cpu=dict(state='COMPLETE_CURRENT', completed=completed, total=7,
                                      evidence='reports/aster_final_gap_completion_20261114/cpu_acceptance.json'),
                 dependent_cpu=dict(C08=dict(state='COMPLETE_CURRENT' if all_e5 else 'PARTIAL_CURRENT',
                                            dependency='E5 thirteen interference evaluations', waiting_for_gpu=not all_e5),
                                    C09=dict(state='COMPLETE_CURRENT' if all_e6 else 'PARTIAL_CURRENT',
                                             dependency='E6 four official evaluations', waiting_for_gpu=not all_e6)),
                 E5=dict(state='COMPLETE_CURRENT' if all_e5 else 'PARTIAL_CURRENT', CC_complete=True,
                         external_eleven_paired_intervals_complete=True, methods=e5),
                 E6=dict(controlled_five_models_complete=True, controlled_paths_per_method=806,
                         controlled_sources=40, official=official, llama_official_complete=True),
                 E7=dict(calculation_state='COMPLETE_CURRENT', all_twenty_model_populations_complete=True,
                         qwen_nli500_clustering='113 trials or trial combinations',
                         qwen_nli5500_clustering='500 original statements and variants',
                         clustering_confirmed_by_user=True, medpic467_includes_development=True,
                         medpic467_role='descriptive evaluation', target_task_fit=False),
                 E8=dict(comparisons_complete=True, state='NEEDS_VALIDATION',
                         missing='Immutable revisions unavailable for some checkpoints; retained as unknown',
                         new_fitting_required=False),
                 queue_snapshot=queue, training_tasks=0, model_download_tasks=0,
                 artifact_status_note='Calculation completion and immutable model identity are recorded separately.')
    write(OUT / 'execution_status.json', state)
    changed.append(OUT / 'execution_status.json')
    # Preserve the original audit plan; its public copy now records actual authorization and outputs.
    plan = read(AUDIT / 'MINIMAL_EXECUTION_PLAN.json')
    original = OUT / 'provenance/MINIMAL_EXECUTION_PLAN.audit_snapshot.json'
    if not original.exists():
        write(original, plan)
    changed.append(original)
    plan.update(status='COMPLETE' if all_e6 and all_e5 else 'APPROVED_EXECUTING',
                updated_at=now, active_scheduler_manifest=False, current_status=STATUS_REL,
                active_queue_manifest='reports/aster_final_gap_completion_20261114/queue_manifest.json',
                open_decision=None, decision=dict(NLI='Retain and label existing two cluster definitions', confirmed=True),
                independent_cpu_complete=True, dependent_cpu=state['dependent_cpu'])
    cpu_outputs = {'C01': 'e6/gemma3/evaluation/test', 'C02': 'e6/medgemma/evaluation/test',
                   'C03': 'e5/CC', 'C04': 'e5/external_paired_ci', 'C05': 'e7/qwen/nli500',
                   'C06': 'e7/qwen/nli5500', 'C07': 'e7/qwen/medpic467'}
    for task in plan['cpu_tasks']:
        task['audit_state'] = task['state']
        task['state'] = 'COMPLETE_CURRENT' if task['id'] in completed else state['dependent_cpu'][task['id']]['state']
        if task['id'] in cpu_outputs:
            task['output'] = str((OUT / cpu_outputs[task['id']]).relative_to(PROJECT))
            task['implementation'] = 'experiments/aster_final_gap_completion_20261114/cpu_tasks.py'
        else:
            task['waiting_for_gpu'] = state['dependent_cpu'][task['id']]['waiting_for_gpu']
    for task in plan['gpu_tasks']:
        task['audit_state'] = task['state']
        if task['id'] in ['G01', 'G02', 'G03', 'G04']:
            entry = next(v for v in official.values() if v['job'] == task['id'])
            task.update(state=entry['state'], output=entry['output'])
        else:
            name = task['unit'].removesuffix('_interference')
            task['state'] = e5[name]['state']
    write(AUDIT_COPY / 'MINIMAL_EXECUTION_PLAN.json', plan)
    changed.append(AUDIT_COPY / 'MINIMAL_EXECUTION_PLAN.json')
    # Replace obsolete aggregate statuses. Historical process snapshots remain in provenance.
    populations = ['nli4ct', 'nli4ct_official5500', 'medpic', 'medpic_official467']
    legacy_models = {}
    for model in ['qwen', 'llama31', 'gemma2', 'gemma3', 'medgemma', 'qwen_base', 'gemma2_base']:
        current = official.get(model, dict(state='COMPLETE_CURRENT', calculation_complete=True))
        value = dict(updated_at=now, state=current['state'], stage='complete' if current['calculation_complete'] else 'official_meGA_only',
                     modules_fitted=True, parameters_frozen=True, controlled_test_complete=True,
                     controlled_test_paths=806, transfer_complete=True,
                     populations={scope: dict(state='COMPLETE_CURRENT') for scope in populations},
                     official_mega=current, full_pipeline_resume=False,
                     identity_status='COMPLETE_CURRENT' if model == 'qwen_base' else 'NEEDS_VALIDATION')
        legacy_models[model] = value.copy()
        p = PROJECT / 'reports/aster_e678_20261112/models' / model / 'run_status.json'
        if p.exists():
            saved_status(p, value, changed, sync_source=True)
    aggregate = dict(updated_at=now, state='COMPLETE_CURRENT' if all_e6 else 'RUNNING',
                     scope='Current E1 artifact status; completed later stages supersede old controller snapshots',
                     models=legacy_models, E8=state['E8'], queue=STATUS_REL,
                     training_required=False, transfer_forward_required=False)
    for name in ['status.json', 'status_only_missing.json']:
        saved_status(PROJECT / 'reports/aster_e678_20261112' / name, aggregate.copy(), changed, sync_source=True)
    saved_status(PROJECT / 'reports/aster_e678_20261112/handoff_status.json',
                 dict(updated_at=now, state='COMPLETE_CURRENT' if all_e6 else 'RUNNING',
                      queue=str(OUT), allowed_gpus=[0, 1, 5, 6], standalone_dispatch=False,
                      previous_queue_scope_complete=True), changed, sync_source=True)
    for name in ['delivery_status.json', 'delivery_status_current.json']:
        saved_status(PROJECT / 'reports/aster_e5_gaps_20261112' / name,
                     dict(updated_at=now, state=state['E5']['state'], scope='Current E1 required E5 natural and interference evaluations',
                          training_complete=True, natural_paths_complete=True, CC_complete=True,
                          external_eleven_forwards_complete=True, external_eleven_paired_intervals_complete=True,
                          interference_methods=e5, loreft_repaired_adapter=True, psr_seed22_update_complete=True,
                          old_queue_selected_scope_complete=True), changed, sync_source=True)
    # Update the sole human experiment tracker, retaining unrelated manuscript evidence.
    tracker = PROJECT / TRACKER_REL
    text = tracker.read_text()
    e6word = '四个IT官方评价已完成' if all_e6 else '四个IT官方评价正在统一队列执行，Gemma2从既有2,379条接续'
    e5word = '十三行干扰路径及接续统计已完成' if all_e5 else '十三行干扰路径已入统一队列，尚待完成'
    dependent_text = 'C08/C09接续统计也已完成。' if all_e6 and all_e5 else '后续CPU任务C08/C09随对应GPU产物完成。'
    summary = ('更新时间：' + now + '。独立CPU任务C01–C07已全部完成；五个IT检查点的806条受控评价齐全。'
               + e6word + '；' + e5word + '。CC和11行配对区间、Qwen三项独立截距统计已完成。'
               + dependent_text + '\n\n'
               '[CPU结果及验收](../../reports/aster_final_gap_completion_20261114/CPU_RESULTS.zh.md) · '
               '[执行状态](../../reports/aster_final_gap_completion_20261114/execution_status.json) · '
               '[第二轮审计](../../reports/aster_final_missing_experiments_audit_20261114/ASTER_FINAL_MISSING_EXPERIMENTS_AUDIT.md)。')
    text = block(text, 'ASTER-CURRENT-EXECUTION', summary)
    text = re.sub(r'(?:E6（跨模型泛化）与E8（Base/IT匹配）按同一检查点台账管理。|E6与E8共用检查点台账。)[^\n]+',
                  'E6与E8共用检查点台账。五个IT模型当前E1受控806路径评价已齐全，其中Gemma3和MedGemma由既有缓存完成CPU重建。'
                  + e6word + '。Llama及两个Base官方评价已完成。E8两组头集合与子空间比较已完成；不可变revision缺项保留追溯状态。', text)
    replacements = {
        '| E6：跨模型泛化 |': '| E6：跨模型泛化 | 五个IT检查点 | E1受控806路径及官方7,458路径 | 五模型受控评价齐全；Llama官方完成；' + e6word + '。各自M1/M2/M3均已冻结。 |',
        '| E8：Gemma匹配对 |': '| E8：Gemma匹配对 | Gemma-2-9B-Base/IT | 头集合、子空间与Base官方评价 | 比较已完成，共享36/51头；IT官方评价共用E6。 |',
        '| E8：Qwen匹配对 |': '| E8：Qwen匹配对 | Qwen3-8B-Base/IT | 头集合、子空间与Base官方评价 | 比较已完成，共享20/51头；IT官方评价共用E6。 |',
        '| E5 |': '| E5 | Qwen3-8B外部基线比较 | **CC及11行配对统计完成；' + ('十三行补测完成' if all_e5 else '十三行干扰补测待完成') + '** | 全部原训练与自然路径复用；LoReFT修复及PSR seed22更新已完成 | ' + e5word + '；各方法保留独立修正分数和原种子 |',
        '| E6 |': '| E6 | 五模型跨架构泛化 | **五模型受控评价齐全；' + ('官方评价齐全' if all_e6 else '四个官方单元执行中') + '** | 每模型受控三方法各806路径、40来源；两27B由CPU重建 | ' + e6word + '；Llama官方7,458路径复用 |',
        '| E7 |': '| E7 | 冻结迁移评价 | **全部前向与所需CPU统计完成** | Qwen NLI500/NLI5500/MedPIC467已补独立截距、逐例和95%区间；其他四模型直接复用 | NLI500按113试验/组合簇，NLI5500按500声明簇；MedPIC467含开发题，作描述性评价 |',
        '| E8 |': '| E8 | Base/IT匹配检查点 | **两组比较完成；部分版本追溯待确认** | Qwen共享20/51头，Gemma2共享36/51头；既有子空间和Base官方结果复用 | 缺失上游revision保留未知；不重拟合，不作后训练阶段因果归因 |',
        '| 是否优于匹配强基线？ |': '| 是否优于匹配强基线？ | E5 | 固定E1配置比较强基线 | 训练与自然路径完成，CC和11行配对区间完成；' + e5word + ' |',
        '| 是否依赖模型家族或后训练？ |': '| 是否依赖模型家族或后训练？ | E6、E8 | 五IT及两组Base/IT | 五IT受控完成；' + e6word + '；E8比较完成 |',
    }
    text = '\n'.join(next((v for k, v in replacements.items() if line.startswith(k)), line) for line in text.splitlines()) + '\n'
    text = text.replace('IT侧官方MEGA仍是E6缺项', 'IT侧官方MEGA共用E6的统一队列产物')
    text = text.replace('LoReFT旧缓存与原模型完全相同，必须核查适配器加载、干预层与位置、参数变化和实际logit变化。',
                        'LoReFT已完成非零旋转参数修复，三个种子复用修复后的适配器；PSR seed22已完成真实参数更新。')
    text = text.replace('### 2026-11-14 本轮补缺结果已归档', '### 2026-11-14 前批次结果已归档')
    text = text.replace('本轮保留范围已完成。E5兼容矩阵', '前批次选定范围已归档。当前E5干扰补测与E6官方评价状态见第6节。E5兼容矩阵')
    tracker.write_text(text)
    changed.append(tracker)
    # Date historical reports and point readers to the existing tracker.
    historic = ['docs/ASTER/results-20261114.zh.md', 'docs/ASTER/external-method-results-20261114.zh.md',
                'reports/aster_completion_20261114/REPORT.zh.md', 'reports/aster_remaining_experiments_audit_20261114/REPORT.zh.md',
                'reports/aster_final_missing_experiments_audit_20261114/ASTER_FINAL_MISSING_EXPERIMENTS_AUDIT.md']
    for directory in ['reports/aster_e5_gaps_20261112', 'reports/aster_e678_20261112']:
        historic += [str(p.relative_to(PROJECT)) for p in (PROJECT / directory).glob('*.md')]
    for relative in historic:
        p = PROJECT / relative
        if not p.exists():
            continue
        link = 'experiments.zh.md' if relative.startswith('docs/') else '../../docs/ASTER/experiments.zh.md'
        note = ('> **历史批次/审计快照。** 以下完成或缺失状态对应当时范围；后续CPU补算已完成，GPU接续按统一队列执行。'
                f'当前逐项状态见[实验框架]({link})及其执行状态附件。原始数值和审计证据保留。')
        if all_e6 and all_e5:
            note = note.replace('GPU接续按统一队列执行', '本轮GPU接续及统计已完成')
        p.write_text(block(p.read_text(), 'HISTORICAL-SCOPE', note))
        changed.append(p)
    audit_report = AUDIT_COPY / 'ASTER_FINAL_MISSING_EXPERIMENTS_AUDIT.md'
    audit_text = audit_report.read_text()
    audit_text = audit_text.replace('## 11. 最终答复与批准事项', '## 11. 审计结束时的结论与后续授权')
    audit_text = audit_text.replace('**审计完成后停止。等待用户批准`MINIMAL_EXECUTION_PLAN.json`，并确认NLI聚类口径。** 批准后依次处理CPU缺项、E6官方前向、E5干扰前向、逐例复算/来源配对区间和归档提交。',
                                   '**后续授权已完成。** 用户已批准开始实验，并确认保留NLI两种聚类口径。C01–C07已完成；执行进展与C08/C09状态见[实验框架](../../docs/ASTER/experiments.zh.md)。下表保留审计结束时的缺口判断，新增产物见CPU结果报告。')
    audit_report.write_text(audit_text)
    if all_e6 and all_e5:
        cpu_report = OUT / 'CPU_RESULTS.zh.md'
        cpu_report.write_text(cpu_report.read_text().replace(
            'C08（E5十三行接续统计）和C09（E6四个官方单元合并统计）依赖正在执行的GPU任务，随产物完成。',
            'C08（E5十三行接续统计）和C09（E6四个官方单元合并统计）已随GPU产物完成，见完整结果报告。'))
        changed.append(cpu_report)
    # Correct the launch note's old encoding without making it a live completion statement.
    launch = OUT / 'RUN_STARTED.zh.md'
    launch.write_text('# ASTER-CR 启动记录\n\n'
                      '启动时已完成C01–C07七项CPU任务，并按用户批准的两种NLI聚类口径统计。'
                      'GPU 0、1、5、6分别启动Qwen、Gemma2、Gemma3、MedGemma官方评价；E5十三行由同一队列接续。'
                      '启动时预计32–46 GPU小时、墙钟12–18小时。\n\n'
                      '本文件记录启动时状态；[当前实验状态](../../docs/ASTER/experiments.zh.md)与'
                      '[CPU结果](CPU_RESULTS.zh.md)分别维护完成情况和实测数据。\n')
    changed.append(launch)
    index = PROJECT / 'docs/ASTER/final-gap-completion-20261114.zh.md'
    index.write_text('# ASTER-CR 补实验产物\n\n[当前状态](experiments.zh.md) · '
                     '[已完成CPU结果](../../reports/aster_final_gap_completion_20261114/CPU_RESULTS.zh.md) · '
                     '[审计报告](../../reports/aster_final_missing_experiments_audit_20261114/ASTER_FINAL_MISSING_EXPERIMENTS_AUDIT.md)\n')
    changed.append(index)
    return list(dict.fromkeys(changed))


if __name__ == '__main__':
    print(json.dumps({'updated_files': len(reconcile())}))
