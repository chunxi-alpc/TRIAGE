"""Render final report and tracker from completed publication tables."""
from pathlib import Path
import csv, json, shutil
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/aster_completion_20261114"
def read(name):
    return list(csv.DictReader((OUT/name).open(encoding="utf-8-sig")))
def pct(x): return f"{float(x)*100:.2f}"
def pp(x): return f"{float(x)*100:+.2f}"
def ci(r,p): return f"[{pp(r[p+'_ci_low'])}, {pp(r[p+'_ci_high'])}]"
def put(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(s.strip()+"\n",encoding="utf-8")
names={"qwen_base":"Qwen3-8B-Base","gemma2_base":"Gemma-2-9B-Base","llama31":"Llama-3.1-8B-Instruct","gemma2":"Gemma-2-9B-IT","gemma3":"Gemma-3-27B-IT","medgemma":"MedGemma-27B-Text-IT"}
lines=["""# ASTER-CR 本轮补缺实验：结果与结论（2026-11-14）

本轮保留范围全部完成：统一队列 **46个单元完成、0个失败、3个重复流程取消**。本报告汇总Codex02的E5补缺，以及共享队列的两个Base、已完成跨模型源任务、缺失迁移和Base/IT比较。历史报告与论文正文保持原样。

**两个Base检查点在官方MEGA完整路径上均获得准确率收益，相对独立截距仍有额外提升；冻结跨任务迁移的效果随模型与任务变化。** E5分别报告响应抑制、证据响应改动、任务准确率和概率校准，完整保留各随机种子。

## 1. 官方MEGA：7,458条路径

每模型覆盖418来源、7,458条动态路径、45,627个金标准路径节点，三个方法合计22,374条路径预测。Qwen Base复用已完成的2,380条路径，GPU0/1/4/5/6接续其余5,078条，五卡阶段约35分钟完成。沿用既有E1参数、开发集冻结配置及原评价器。

| 模型 | 原模型 | 仅截距 | ASTER-CR E1 | E1－原模型，百分点 [95% CI] | E1－仅截距，百分点 [95% CI] |
|---|---:|---:|---:|---:|---:|"""]
official=read("official_MEGA_7458.csv")
for model in ["qwen_base","gemma2_base","llama31"]:
    rs={r["method"]:r for r in official if r["model"]==model}; a=rs["aster_cr"]
    cells=[f"{rs[m]['correct']}/{rs[m]['n']} ({pct(rs[m]['accuracy'])}%)" for m in ["raw","bias_only","aster_cr"]]
    raw=f"{pp(a['delta_vs_raw'])} [{pp(a['reported_delta_vs_raw_ci_low'])}, {pp(a['reported_delta_vs_raw_ci_high'])}]"
    lines.append(f"| {names[model]} | "+" | ".join(cells)+f" | {raw} | {pp(a['delta_vs_bias'])} {ci(a,'delta_vs_bias')} |")
lines.append("""
Qwen Base完整路径正确数从521增至1,087；仅截距达到1,009，输入相关修正在此基础上增加78条。Gemma2 Base相对仅截距再增加412条。两项相对截距的差值区间均高于零，支持在这两个检查点分别报告校准收益与额外修正收益。

Llama的E1比原模型增加210条，较仅截距增加1条；相对截距的区间包含零，该项主要体现校准收益。

![官方MEGA完整路径正确率；误差线为95%来源聚类区间](figures/official_MEGA_accuracy.png)

[完整指标](official_MEGA_7458.csv)；[逐例预测目录](official_prediction_exports.json)。官方集合保留逐例既往来源与输入使用记录，不表述为全新独立测试。

## 2. E5基线补缺与第4.5节数据

本轮完成12种方法、18行实际运行结果，另列原模型、固定0.95缩放、冻结ASTER-CR E1三个参照。训练使用E1 / Dataset A 2,540对，A/P/C/F各635；监督端点共4,637个。LoRA/LoFiT保留完整词表交叉熵目标，LoReFT二元监督以及LEACE-pair、ITI-style、SEA的任务适配配置单独标注。

LoReFT旧适配器失效已确认，修复通过参数更新、分数变化及保存重载测试后才补跑。超参数按开发集冻结，三个随机种子完整列出。

| 方法/运行 | A/P响应RMS | C/F修正RMS | 自然路径全对 | 路径正确率95% CI |
|---|---:|---:|---:|---:|""")
labels={"raw":"原模型（复用）","scale_095":"固定0.95缩放","ASTER-CR E1 frozen":"ASTER-CR E1（冻结复用）","temperature_e1_endpoint_nll":"Temperature Scaling","intercept_e1_endpoint_nll":"仅截距校准","affine_e1_endpoint_nll":"仿射校准","linear_readout_e1_last_attention_ce":"监督线性读出","leace_pair_E1":"LEACE-pair（任务适配）","iti_style_E1":"ITI-style（任务适配）","sea_adaptation_E1":"SEA（任务适配）","CC":"CC","strong_evidence_prompt":"强证据提示"}
for r in read("section_4_5_E1_with_95ci.csv"):
    m=r["method"]; label=labels.get(m,m.replace("lora_SFT_E1_seed","LoRA SFT / seed ").replace("lofit_SFT_E1_seed","LoFiT SFT / seed ").replace("LoReFT_binary_task_adaptation_E1_seed","LoReFT二元适配 / seed "))
    lines.append(f"| {label} | {float(r['AP_response_rms']):.4f} | {float(r['CF_correction_rms']):.4f} | {r['path_correct']}/{r['path_n']} | [{pct(r['path_accuracy_ci_low'])}, {pct(r['path_accuracy_ci_high'])}]% |")
lines.append("""
A/P响应RMS衡量标签保持配对的分数变化；C/F修正RMS衡量方法对标签翻转配对原有分数差的改动。两者结合路径正确率解释。

温度缩放将A/P响应RMS从2.6282降至0.3827，同时C/F修正RMS为31.0113，自然路径维持768/806。其自然节点Brier由0.010661降至0.007842，log loss由0.084546降至0.043146，支持概率校准收益。冻结E1的C/F修正RMS为0.0950，保留较接近原模型的证据响应。

LoRA三种子路径正确率为91.44%—96.03%，均值93.76%；LoFiT为95.29%—95.41%，均值95.37%；LoReFT为80.65%—95.78%，均值90.41%。完整种子结果均保留，汇总不按测试表现挑选种子。ITI任务适配为321/806，须结合其具体训练目标和适配配置解释。

评价总体为有历史使用记录的40来源、2,293个A/P对、40个C/F对、806条自然路径。输出控制另有2,418条干扰路径，未计算该项的方法留空。[第4.5节完整表及A/P、C/F、路径95%区间](section_4_5_E1_with_95ci.csv)；[种子变异](E5_seed_variation.csv)。

Codex03的RePS、PSR-MSE、CAA、INLP、DoRA共11次冻结测试已在主分支单独交付，见[外部候选报告](../aster_codex03_external_20261112/REPORT.zh.md)。

## 3. 冻结跨任务迁移

下表使用冻结源任务E1参数及源任务独立截距。MedPIC报告整题选项集合全对率，本组未加入目标任务M4集合校准。指定Qwen NLI500、MedPIC188结果直接复用。

| 模型 | 评价总体 | 原模型 | 仅截距 | E1 | E1－原模型，百分点 [95% CI] |
|---|---|---:|---:|---:|---:|""")
transfer=read("frozen_transfer_results_95ci.csv")
pops={"nli4ct":"NLI原声明500","nli4ct_official5500":"NLI正式5500","medpic":"MedPIC冻结188","medpic_official467":"MedPIC全题467"}
for model in ["llama31","gemma2","gemma3","medgemma"]:
    for pop in pops:
        rs={r["method"]:r for r in transfer if r["model"]==model and r["population"]==pop}; e=rs["E1"]
        cells=[f"{rs[m]['correct']}/{rs[m]['n']} ({pct(rs[m]['accuracy'])}%)" for m in ["raw","calibration","E1"]]
        lines.append(f"| {names[model]} | {pops[pop]} | "+" | ".join(cells)+f" | {pp(e['delta_vs_raw'])} {ci(e,'delta_vs_raw')} |")
lines.append("""
NLI正式5,500条中，四模型E1相对原模型的差值区间均包含零。MedPIC188中，Gemma3从61题降至48题，差值−6.91个百分点，区间低于零；其他三个模型差值区间包含或接触零。源任务完整路径收益与目标任务结果分别报告，跨任务效应按模型和任务判断。

Gemma3、MedGemma复用已拟合参数，补完缺失开发集计算、冻结与迁移；NLI500的本轮统计直接使用已有缓存。MedPIC467包含既有开发题，作为全题描述性结果，188题冻结测试单列。[完整迁移数值及95%区间](frozen_transfer_results_95ci.csv)。

## 4. Base/IT检查点比较

Qwen与Gemma2的Base和IT均各自选择51个头。Qwen共享20个头，Jaccard为0.2439；Gemma2共享36个头，Jaccard为0.5455。两组比较记录各自子空间谱、有效秩和修正行为，属于检查点描述性比较。[比较表](Base_IT_head_comparison.csv)；[详细结果](../aster_e678_20261112/base_it)。

## 5. 协议和产物

- 正式训练固定E1的2,540对；旧A/P 4,887对、C/F 135对结果另存历史层。CAD、SEA、温度的既有完整评价保留原协议记录。
- 各方法原生目标与任务适配目标逐项记录。没有新增Qwen IT的ASTER-CR训练，也未重复指定Qwen迁移评价。
- E5复用各方法已有2,000次来源聚类百分位区间，本次补齐统一表中的已有区间。
- 上方MEGA相对原模型区间复用原评价文件；绝对正确率、相对截距区间及迁移汇总区间使用既有逐例预测，在CPU按来源联合重采样2,000次，随机种子11，各方法共享采样索引。CSV同时保留原MEGA差值区间。
- 本轮保留范围完成。Gemma3/MedGemma的E1源任务受控测试及官方MEGA、Gemma2 IT未完成的本轮官方MEGA保留缺项状态；旧协议结果另列。未运行方法或总体没有填入数值表。

| 交付项 | 文件 |
|---|---|
| E5兼容性矩阵 | [当前矩阵](../aster_e5_gaps_20261112/compatibility_matrix_current.csv) |
| 复用结果 | [清单](../aster_e5_gaps_20261112/reused_results.json)、[完整索引](../aster_e5_gaps_20261112/existing_results_inventory.json) |
| 新增运行 | [运行清单](../aster_e5_gaps_20261112/new_runs_current.json)、[统一队列完成记录](queue/completion.json) |
| 逐例预测 | [发布文件索引](artifact_manifest.csv)、[官方MEGA合并预测](official_prediction_exports.json) |
| 共同指标和95%区间 | [E5](section_4_5_E1_with_95ci.csv)、[官方MEGA](official_MEGA_7458.csv)、[迁移](frozen_transfer_results_95ci.csv) |
| 协议差异 | [说明](../aster_e5_gaps_20261112/PROTOCOL_DIFFERENCES.zh.md)、[CSV](../aster_e5_gaps_20261112/protocol_differences.csv) |
| E6受控测试 | [数据](E6_controlled_source_test.csv) |
| 代码和配置 | [E5](../../experiments/aster_e5_gaps_20261112)、[E6/E7/E8](../../experiments/aster_e678)、[CPU汇总](../../experiments/aster_completion_20261114) |
| 完成核对 | [记录](publication_validation.json) |

逐例JSONL以gzip保存原始字节；文件索引映射服务器原文件与GitHub文件。大体积激活缓存和预训练权重保留在服务器，见[服务器产物记录](server_artifacts.json)。历史报告按原内容归档，其中的阶段状态以本报告及最终完成记录为准。
""")
put(OUT/"REPORT.zh.md","\n".join(lines))
put(OUT/"README.md","# Completed ASTER experiment artifacts\n\n[结果与结论](REPORT.zh.md) · [实验管理入口](../../docs/ASTER/experiments.zh.md) · [文件索引](artifact_manifest.csv)\n\nCPU-only reproduction: python experiments/aster_completion_20261114/summarize_results.py --root <repository>. Published predictions are sufficient; no model inference. render_summary.py renders the report and tracker update. package_results.py exports original server artifacts.")
put(ROOT/"docs/ASTER/results-20261114.zh.md","""# ASTER-CR 补缺实验结果汇总（2026-11-14）

本轮保留范围全部完成：E5补缺、两个Base的官方MEGA、缺失冻结迁移和两组Base/IT比较。统一队列46个单元完成、0个失败，3个重复流程按用户要求取消。

官方MEGA的7,458条路径中，Qwen3-8B-Base从521/7,458（6.99%）提高到1,087/7,458（14.57%），差值+7.59个百分点，95% CI [6.68, 8.51]；相对仅截距再提高1.05个百分点，[0.53, 1.58]。Gemma-2-9B-Base从31.76%提高到55.35%，相对仅截距提高5.52个百分点，[4.45, 6.62]。

E5交付12种方法的18行实际结果、3个冻结或解析参照，以及逐例预测、共同指标、95%区间和协议差异。LoRA、LoFiT、LoReFT三个随机种子均保留。冻结迁移按模型和任务分别报告，包含Gemma3在MedPIC188上的负向结果。

- [完整结果与结论](../../reports/aster_completion_20261114/REPORT.zh.md)
- [第4.5节E1数据与95%区间](../../reports/aster_completion_20261114/section_4_5_E1_with_95ci.csv)
- [官方MEGA数据](../../reports/aster_completion_20261114/official_MEGA_7458.csv)
- [冻结迁移数据](../../reports/aster_completion_20261114/frozen_transfer_results_95ci.csv)
- [逐例预测索引](../../reports/aster_completion_20261114/artifact_manifest.csv)
- [E5兼容性矩阵](../../reports/aster_e5_gaps_20261112/compatibility_matrix_current.csv)

所有40来源/806路径结果标注历史使用；官方7,458路径保留既往来源与输入使用记录。论文正文与历史报告保持原样。
""")
tracker=ROOT/"docs/ASTER/experiments.zh.md"; text=tracker.read_text(encoding="utf-8")
rows={
"E5":"| E5 | Qwen3-8B外部基线比较 | **本轮基线补缺完成；历史结果按协议分层** | Codex02新增12种方法18行结果，Codex03新增5种方法11次冻结测试；逐例、兼容矩阵、共同指标及95%区间已交付 | 见[本轮结果汇总](results-20261114.zh.md)；完整保留三个种子，历史40来源806路径单列 |",
"E6":"| E6 | 五模型跨架构泛化 | **本轮选定范围完成，按模型分项记录** | Llama、Gemma2 IT及两个Base已有E1受控测试；两个Base和Llama完成官方7458路径 | Gemma3/MedGemma源任务受控测试及官方MEGA、Gemma2 IT本轮官方MEGA保留缺项；历史协议另报 |",
"E7":"| E7 | 三个主数据集的冻结迁移 | **本轮缺失迁移评价完成** | Llama、Gemma2、Gemma3、MedGemma均完成NLI500/5500及MedPIC188/467；Qwen既有结果复用 | [完整数值与95%区间](../../reports/aster_completion_20261114/frozen_transfer_results_95ci.csv)；MedPIC467为含开发题的全题描述性评价 |",
"E8":"| E8 | Base/IT匹配检查点 | **Gemma2与Qwen两组比较完成** | 各检查点独立拟合；Qwen共享20/51个头，Gemma2共享36/51个头 | [比较数据](../../reports/aster_completion_20261114/Base_IT_head_comparison.csv)；描述检查点差异 |"}
updated=[]; found=set()
for line in text.splitlines():
    key=next((k for k in rows if line.startswith("| "+k+" |")),None)
    if key: line=rows[key]; found.add(key)
    updated.append(line)
assert found==set(rows)
marker="<!-- ASTER-FINAL-RESULTS-20261114 -->"
if marker not in text:
    updated += ["",marker,"### 2026-11-14 本轮补缺结果已归档","","本轮保留范围已完成。E5兼容矩阵、复用/新增清单、逐例预测、共同指标、95%区间及第4.5节数据，连同共享队列Base与迁移结果，统一见[结果与结论](results-20261114.zh.md)。所有数值来自已完成评价，历史报告和论文正文保持原样。"]
put(tracker,"\n".join(updated))
for name in ["baseline-audit-20261112.zh.md","baseline-compatibility-20261112.json"]:
    target=ROOT/"docs/ASTER"/name
    source=Path("/home/gcx/aster-e678-gaps-20261112/docs/ASTER")/name
    if not target.exists() and source.exists(): shutil.copy2(source,target)
print(json.dumps({"report_ready":True,"tracker_rows":sorted(found),"manuscript_edited":False}))
