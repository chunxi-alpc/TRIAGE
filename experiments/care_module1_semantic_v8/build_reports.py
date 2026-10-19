#!/usr/bin/env python3
"""Build Chinese reports from completed, saved measurements."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
H=Path(__file__).resolve().parent

def load(p):return json.loads((H/p).read_text())
def pct(x):return str(Decimal(str(100*x)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))+'%'
def frac(n,d):return f'{n}/{d}（{pct(n/d)}）' if d else '不适用'
def main():
    ms=load('analysis/metrics.json');selection=load('selection.json')
    assert len(ms)==8, 'All six semantic runs and both A2 comparisons must be complete'
    selected=selection['selected'];ca=ms['confirmation_'+selected];na=ms['new72_'+selected]
    q=load('results/qwen_'+selected+'/summary.json');qc=load('results/qwen_'+selected+'/completion.json')
    assert qc['status']=='completed'
    families=load('analysis/family_metrics.json');comp=load('analysis/paired_comparison.json')
    report=['# CARE 模块一 V8：轻量语义适用性识别真实实验报告','',
        '日期：2026-10-19。基于 PR #11 `f0c3c28a30096bdda37c021f6db9de85754cb8e7` 的独立 worktree 与实验分支；所有推理均在原服务器完成，旧成果未覆写。','',
        f'**本轮已经完成。开发集选定默认 {selected}，但当前语义版仍不足以替代已有 A2。** '
        f'默认版在历史 192 例上为 {frac(ca["correct"],ca["rules"])}，A2 为 560/576（97.22%）；'
        f'新 72 例上为 {frac(na["correct"],na["rules"])}，A2 为 210/216（97.22%）。'
        f'默认版 F→T 分别为 {ca["F_to_T"]}/{ca["F_to_T_denominator"]} 和 {na["F_to_T"]}/{na["F_to_T_denominator"]}。'
        '语义直接判断确实简化了模块一的输入处理，但这一实现的判断正确性和信息不足识别尚未达到强基线。','',
        '## 1. 实际完成的范围','',
        '- A0、A1 分别完成 96 例 dev、192 例历史 confirmation、72 例新病例：共 720 次病例预测、2160 次规则判断。',
        '- A2 复用全部 192 例 PR #11 修复结果；新 72 例另做 207 个去重原子请求并聚合为 216 条规则判断。没有为 dev 补跑 A2。',
        '- Qwen3-4B 完成固定 24 例四臂比较及一次零操作测试，共 97 次前向；没有新一轮参数搜索。',
        '- 保存全部实际提示、原始输出、模型来源、调用参数、重试/失败、耗时、逐规则评价及冻结数据。','',
        '## 2. 方法、选型与输入公平性','',
        'A0 和 A1 使用同一 Phi-4 本体、TrialMatchAI LoRA、NF4 4bit、BF16 计算；batch=8、greedy、最多384新token。一次输入问题Q、患者X、三条完整自然语言规则及相邻分支，一次返回全部规则的T/F/U、原文摘录和一句依据。A1增加三个固定训练演示，其中T/F对含两个患者叙述，因此共四个训练患者示例。例外/否定及U演示均来自训练家族，未使用dev或测试病例。','',
        '预测输入没有金标、患者结构化事实或正确动作。历史指南从既有规则结构离线展开成完整自然语言；新数据使用作者撰写的自然语言规则。动作位置由原始规则ID确定，模型不生成动作。证据定位是软诊断；找不到摘录不会改写或阻断状态。解析失败最多一次短重试，再失败独立记为ERROR/null，不伪装成U。','',
        f'默认版按预先固定的dev准确率→F→T较少→宏F1→A0顺序选为 **{selected}**，选择文件在运行后续数据前写入。后续仍同时跑完整A0/A1，未按测试结果再选版本。','',
        'A2有作者预解析的原子命题及逻辑树，历史A2还保留数值处理优先逻辑。语义版获得同等规则含义和相邻分支，但没有该机器可执行结构。新A2复用PR11紧凑单token原子接口及三值聚合，所有原子均实际由模型判断，无数值覆盖；历史A2主要复用长输出缓存，包含17条紧凑接口修复。故比较同时涉及算法、输入结构及输出接口差异，不能说差值纯粹证明某一模型更强。','',
        '## 3. 规则判断结果','',
        '| 数据 | 方法 | 正确规则 | 宏F1 | F→T/真实F | U正确/真实U | 目标规则T/F配对全对 | 有效病例 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    splitlabel={'dev':'dev 96','confirmation':'历史192','new72':'新72'}
    for key,m in ms.items():
        u=m['confusion']['U'];report.append(f'| {splitlabel[m["split"]]} | {m["arm"]} | {frac(m["correct"],m["rules"])} | {m["macro_f1"]:.4f} | {m["F_to_T"]}/{m["F_to_T_denominator"]} | {u["U"]}/{sum(u.values())} | {m["target_pair_correct"]}/{m["target_pair_total"]} | {m["valid_cases"]}/{m["cases"]} |')
    report += ['', '配对指标要求同一目标规则在最小对比的两侧分别正确为T和F。历史全配对包含改写版本，dev为36对、历史192为72对；另存canonical标准配对及所有发生翻转的规则指标，不能与PR11的下游动作配对数直接混为一谈。', '',
       '### 完整混淆矩阵','', '| 数据/方法 | 参考状态 | 预测T | 预测F | 预测U | 技术失败 |','|---|---|---:|---:|---:|---:|']
    for key,m in ms.items():
        for s,v in m['confusion'].items():report.append(f'| {key} | {s} | {v["T"]} | {v["F"]} | {v["U"]} | {v["ERROR"]} |')
    report += ['', '### 同病例与A2逐规则比较','', '| 数据/语义版 | 两者均对 | 仅语义版对 | 仅A2对 | 两者均错 |','|---|---:|---:|---:|---:|']
    for k,v in comp.items():
        c=v['counts'];report.append(f'| {k} | {c.get("both_correct",0)} | {c.get("semantic_only_correct",0)} | {c.get("A2_only_correct",0)} | {c.get("both_wrong",0)} |')
    report += ['', '所有改变逐例保存在 `analysis/paired_comparison.json`；完整分母与失败保留，不依据表现排除病例。','',
        '## 4. 新家族结果与数据范围','',
        '新数据是虚构管理分流情境，覆盖标本时序、回电许可、设备材料、出院后药单、联系渠道、设备归属、口译偏好、签署撤回、签到离场、玻片借出、交通安排和日志更新。每家族6条病例记录：两对T/F最小对比与两个U场景。自然文本包含时序、否定、例外、患者与家属区别。','',
        '参考标签来自作者指定事实状态和可执行三值逻辑，没有用被评模型定金标。预测前冻结，保留预测前措辞修订的v1/v2记录；预测后不修改数据。作者/Codex做简短文本阅读检查，没有独立人类专家审核。新的是语义主题，布尔运算定义沿用已有三值逻辑。','',
        '**72条记录含63个唯一患者文本**：9个家族的两个对比共用相同正例。两对均保留，A0/A1/A2同批评价；这些记录不能视为72个完全独立样本，也不构成临床验证。','',
        '| 新家族 | A0正确/18 | A1正确/18 | A2正确/18 |','|---|---:|---:|---:|']
    ids=sorted({x['family_id'] for x in families if x['split']=='new72'})
    for fid in ids:
        vals={x['arm']:x['correct'] for x in families if x['split']=='new72' and x['family_id']==fid}
        report.append(f'| {fid} | {vals["A0"]} | {vals["A1"]} | {vals["A2"]} |')
    report += ['', '家族完整T/F/U、配对和混淆矩阵保存在 `analysis/family_metrics.json`。新A2的6个错误全部是将U判成确定状态，发生于标本时序、回电许可和交通安排三个家族；其F→T为0不代表所有不确定性问题已解决。','',
       '## 5. 原文证据与推断质量','', '| 数据/方法 | 非空证据 | 可原文定位 | 定位率（非空分母） |','|---|---:|---:|---:|']
    for key,m in ms.items():
        if m['arm']=='A2':continue
        r=m['evidence_exact_rate_of_nonempty'];report.append(f'| {key} | {m["evidence_nonempty"]} | {m["evidence_exact_located"]} | {pct(r) if r is not None else "不适用"} |')
    report += ['', '定位率只判断摘录是否出现在患者原文，**不等于摘录能支持整条规则的结论**。简短作者检查已发现：引用了真实的年龄或授权语句，却忽略OR另一支、误解嵌套否定，或把非当前的年龄记录当成判F依据。原始例子、全部规则和说明在 `analysis/evidence_review.json`；这不是专家标注或证据蕴含准确率。','',
       'A2新数据的紧凑接口不生成证据摘录，证据定位指标记为不适用；历史A2的NLI片段仅作为旧产物保存，未伪装成相同的生成证据任务。','',
       '## 6. Qwen接口与决策影响','',
       '统一S接口保留全部规则和动作，动作字符位置按rule_id取得，证据位置来自原文匹配；先构造完整Qwen提示再分词映射token。固定24例在查看V8结果前冻结，八家族各一对T/F及一个U。沿用C配置18–23层、强度1、最后一个查询位置，不重新选层头强度。','',
       '| 方案 | V8默认版正确/24 | PR11修复版同例正确/24 |','|---|---:|---:|']
    labels={'baseline':'无结构无干预','structure_only':'仅结构提示','attention_only':'仅干预','combined':'结构提示＋干预'}
    for a,label in labels.items():report.append(f'| {label} | {q["v8"][a]["correct"]} | {q["pr11_fixed_historical_same_cases"][a]["correct"]} |')
    report += ['', f'实际前向 {qc["forward_calls"]} 次，零操作检查通过：{q["noop_check_passed"]}。解析、字符/token位置及零操作各作必要轻量检查，没有额外多轮筛选。', '',
        '历史baseline的24例完整提示、预测及候选logit均完全复现，所有臂无技术失败。实际干预改变了24例的注意力分布，但仅干预的最终选项与baseline全部相同，组合与仅结构提示的最终选项也全部相同。仅结构提示相对baseline修正10例、损坏5例；相比历史A2结构提示损坏7例、修正0例。','',
        '这只检验接入与决策影响。Qwen仍收到历史预解析指南；各组保留相同输入，仅改变状态提示或固定干预。结构提示收益不能归因于注意力，24例结果不能替代完整临床验证。原始四组、历史同例和实际注意力诊断均保存。','',
        '## 7. 资源、调用预算、耗时与失败','',
        '正式启动前已报告GPU 0/1用于语义版、GPU2用于A2、GPU3用于Qwen。未使用占用中的GPU6，GPU7也未启用。所有运行在服务器，稳定后停止持续监控，在阶段完成时收集结果。','',
        '| 数据/方法 | 病例/原子请求 | 实际请求 | GPU | 含加载总秒数 | 生成秒数 | 峰值allocated GiB | 首轮/重试成功/最终解析失败 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for key,m in ms.items():
        if m['arm']=='A2':continue
        r=m['run_manifest'];c=r['counts']
        report.append(f'| {key} | {m["cases"]} | {c["case_requests"]} | {r["cuda_visible_devices"]} | {r["elapsed_seconds"]:.2f} | {r["generation_seconds"]:.2f} | {r["peak_allocated_gib"]:.2f} | {c["accepted_first"]}/{c["accepted_retry"]}/{c["parse_failure"]} |')
    a2=load('results/new72_A2/manifest.json')
    report.append(f'| new72_A2 | {a2["unique_atomic_requests"]}原子 | {a2["actual_atomic_calls"]} | 2 | {a2["elapsed_seconds"]:.2f} | {a2["inference_seconds"]:.2f} | {a2["peak_allocated_gib"]:.2f} | 不适用/0/{a2["parse_failures"]} |')
    report += ['', f'Qwen用时{qc["elapsed_seconds"]:.2f}秒，峰值allocated {qc["peak_allocated_gib"]:.2f} GiB，reserved {qc["peak_reserved_gib"]:.2f} GiB。', '',
        '每个语义版本的主请求预算为360，最多360次格式重试，token上限为每次384；两版主预算720、最大含重试1440次病例请求。A2新数据207个去重原子请求、26个批次、每原子最多1个token，复用相同X+原子输入；其无需生成一句解释与证据，因此时间不是等任务成本比较。历史A2本轮新增请求为0。逐例时间是批处理实测时间分摊，不是在线单请求延迟。完整元数据见各manifest。','',
        '## 8. 结论与适用范围','',
        '语义版已实现真正的完整自然语言判断，并成功接入原Qwen联合代码；没有RAG，没有患者事实解析器，没有把规则适用性改造成硬规则执行器。接口可运行与格式有效，不代表语义识别已经可靠。当前同批结果支持保留A2作为强基线，V8作为已完成的语义简化尝试，而不直接替换。','',
        '主要风险是F被误判为T，以及把缺失、时序或作用对象不明确的信息判成确定状态。少量固定示例有开发集收益，但不能据此保证所有家族或独立数据的收益。证据能够引用原文，也可能不能支持完整推断。','',
        '历史192例是回归探索；新72例是本轮冻结的合成语义主题，非临床病例且部分配对共用正例。历史参考还采用“仅当前观测”的人工约定，连仅历史年龄也赋U；这不同于真实年龄具有单调性的常识。语义提示要求理解时序，A2原子提示更明确限制只采用当前记录。因此历史指标衡量与既有合成参考的一致性，不能直接推广成真实医学适用性，也不能完全分离提示差异。没有置信度研究、额外检索、按结果删样本或新增参数搜索。结果可以为阴性，本轮结论按实际输出保留。','',
        '复现入口、字段定义和产物索引见 `README.md`；精简结论见 `TEACHER_REPORT.zh.md`。','']
    (H/'REPORT.zh.md').write_text('\n'.join(report))
    teacher=['# 给老师：CARE 模块一 V8 实验结果','',
        '**已完成真实GPU实验。当前直接语义判断可以简化实现，但准确性仍不足以替代原子方案。**','',
        f'使用与原方案相同的Phi-4＋TrialMatchAI LoRA。A0直接判断完整规则；A1只增加三个固定训练演示。开发集A1为220/288（76.39%），高于A0的212/288（73.61%），因此预先固定A1为默认。','',
        '| 数据 | A0 | A1（默认） | A2原子方案 |','|---|---:|---:|---:|']
    for s,label in [('confirmation','历史192例/576规则'),('new72','新72例/216规则')]:
        vals=[ms[s+'_'+a] for a in ['A0','A1','A2']]
        teacher.append('| '+label+' | '+' | '.join(f'{m["correct"]}/{m["rules"]}（{pct(m["accuracy"])}）' for m in vals)+' |')
    teacher += ['', f'误用风险：默认A1在历史集F→T为{ca["F_to_T"]}/{ca["F_to_T_denominator"]}，新集为{na["F_to_T"]}/{na["F_to_T_denominator"]}；A2分别为14/352和0/120。原文摘录通常能定位，但仍可伴随分支、否定或信息不足判断错误。','',
        f'Qwen固定24例接入已完成：仅结构提示{q["v8"]["structure_only"]["correct"]}/24、仅干预{q["v8"]["attention_only"]["correct"]}/24、组合{q["v8"]["combined"]["correct"]}/24；基线{q["v8"]["baseline"]["correct"]}/24。这仅说明接入和决策影响，不能把结构提示的效果归因于注意力。','',
        '需要区分：A2有预解析原子及逻辑结构优势；历史192例已用于开发，只作回归探索。新72例来自12个新合成语义主题，含63个唯一患者文本，参考由逻辑设计生成，没有独立临床专家标注，不是临床验证。','',
        '本轮结论：保留A2强基线；V8的直接语义简化方向可实现，但本次效果尚不支持替换。所有结果、原始输出和失败均已保存，旧成果未覆盖。','']
    # Use the measured F support, rather than assuming a historical count.
    teacher=[x.replace('14/352',f'14/{ms["confirmation_A2"]["F_to_T_denominator"]}') for x in teacher]
    (H/'TEACHER_REPORT.zh.md').write_text('\n'.join(teacher))
    manifest=dict(status='completed',selected=selected,base_commit='f0c3c28a30096bdda37c021f6db9de85754cb8e7',
        gpu_devices_used=[0,1,2,3],semantic_case_predictions=720,semantic_rule_predictions=2160,
        a2_new_unique_atomic_requests=a2['actual_atomic_calls'],a2_historical_reused_cases=192,qwen_forward_calls=qc['forward_calls'],
        runs={k:m.get('run_manifest',{'status':'reused_historical'}) for k,m in ms.items()},
        qwen=qc,scope='synthetic study; historical confirmation is regression exploration')
    (H/'execution_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('REPORT.zh.md, TEACHER_REPORT.zh.md, execution_manifest.json written from completed measurements')

if __name__=='__main__':main()
