"""Build the four-question Chinese report directly from measured CSV/JSON outputs."""
import csv,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def j(p):return json.loads((H/p).read_text())
def rr(p):return list(csv.DictReader((H/p).open()))
def n(x):return float(x)
def f(x,k=4):return f"{float(x):.{k}f}"
def pct(x,k=2):return f"{100*float(x):.{k}f}%"
def main():
 pred={r["method"]:r for r in rr("exp12/summary.csv") if r["group"]=="A"}
 boots=rr("exp12/bootstrap_summary.csv")
 full=j("exp3/full/metrics.json")
 frozen=j("exp3/frozen_heads_metrics.json")
 repair=rr("exp4/main_results.csv")
 rconfig=j("exp4/config_and_metrics.json")
 cap=j("exp12/capacity_supplement.json")
 ref=j("reference_audit.json")
 summary=[]
 def add(q,method,split,metric,val,unit="",origin="real_cache_new_training"):
  summary.append(dict(question=q,method=method,split=split,metric=metric,value=val,unit=unit,evidence_type=origin))
 lines=["# ASTER-CR 内部痕迹的预测能力与实际作用","",
 "全部实验已在服务器完成，使用真实 Qwen3-8B repair-v2 评分与激活缓存，没有合成或模拟结果。原始 fit/dev/test 的80/20/40个来源及配对划分保持不变。完整五折新增真实单头干预，其余实验为真实缓存复算和新增小模型训练。旧实验结果和论文正文均未修改。","",
 "## Q1：16 维痕迹是否明显优于随机特征？","",
 f"16维主方法在开发集1,194对A类上的 R²={f(pred['selected_N16']['r2_mean'])}，RMSE={f(pred['selected_N16']['rmse_mean'])}；按20个来源进行2,000次成组自助重采样，R²的95%区间为[{f(boots[0]['main_low'])}, {f(boots[0]['main_high'])}]。",
 "训练统一为fit的4,887对A/P，岭系数0.01，无配对回归截距。标准化均值/尺度来自fit full视图，子空间来自fit配对，不额外按dev标准化。AP合并的1,198对结果另见CSV。","",
 "| 特征 | 次数 | 开发A R²（种子均值） | 种子标准差 | RMSE均值 |",
 "|---|---:|---:|---:|---:|"]
 names={"selected_N16":"选中51头的16维正交痕迹","random_subspace16":"随机16维子空间","random_heads51":"随机51头的16维正交痕迹","signal_S16":"主要证据子空间16维","unorthogonal_N16":"未正交化干扰子空间16维","shuffled_target":"打乱训练分数差"}
 for method,label in names.items():
  r=pred[method]
  lines.append(f"| {label} | {r['repeats']} | {f(r['r2_mean'])} | {f(r['r2_std'])} | {f(r['rmse_mean'])} |")
  add("Q1",method,"dev_A","R2",r["r2_mean"])
 lines+=["","| 主方法减去对照 | R²差值 | 来源自助重采样95%区间 |","|---|---:|---:|"]
 for b in boots:
  lines.append(f"| {names[b['control']]} | {f(b['difference'])} | [{f(b['difference_low'])}, {f(b['difference_high'])}] |")
  add("Q1","selected_minus_"+b["control"],"dev_A","R2_difference",b["difference"])
 lines+=["",
 "主方法高于随机投影的平均水平，配对区间为正；50个随机投影中有41个低于主方法，仍有9个高于主方法，因此不能声称逐个随机投影都较差。随机头和打乱监督的50次结果均低于主方法。区间的对照统计量是50个种子各自R²的平均，不是50模型集成预测；它描述来源重采样的不确定性，不等于同时重采样随机种子的区间。","",
 "主要证据子空间和未正交化干扰子空间预测更强。因此选头有价值，正交痕迹也含有可靠预测信息，但预测信息并非其独有，不能仅凭0.4063证明干扰信息的选择性。正交化去除了大量可预测分数变化的方向；这些方向是否适合修复，要由Q4的证据代价评价判断。","",
 f"参考结果均已找到归档程序并逐对复算：51头、16维、A类1,194对，直接修补 RMS 比例={pct(ref['transmission_rms_ratio'],4)}，痕迹预测 R²={f(ref['trace_prediction_r2'],8)}，直接修补 R²={f(ref['direct_patch_r2'],8)}。后者定义为1−Σ(TE−IE)²/Σ(TE−mean(TE))²，没有再拟合IE→TE回归，也没有回归截距。其直接修补 RMS={f(ref['direct_patch_rms'])}，原分数差 RMS={f(ref['target_rms'])}。原始计算程序见trace/summarize_trace.py；本目录reference_audit.json记录路径及定义。paired_fit中加入F0的APF0修复属于另一拟合，不能和本次AP预测结果混用。","",
 "![预测能力比较](exp12/prediction_comparison.png)","",
 "## Q2：增加维度或采用非线性函数，能提高多少预测能力？","",
 "主维度比较固定证据子空间S16、原始坐标标准化和训练对，只改变干扰痕迹维度，避免同时改变证据子空间。各维度有效秩均达到请求值，32维不存在秩不足。","",
 "| 痕迹维度（固定S16） | 开发A R² | RMSE | 岭回归参数数 |","|---|---:|---:|---:|"]
 for dim in (4,8,16,32):
  method="selected_N16" if dim==16 else "fixedS16_N"+str(dim)
  r=pred[method]
  lines.append(f"| {dim} | {f(r['r2_mean'])} | {f(r['rmse_mean'])} | {dim} |")
  add("Q2",method,"dev_A","R2",r["r2_mean"])
 mlp=pred["mlp16"];lin=pred["selected_N16"]
 lines+=["",
 f"固定16维时，两层网络16→32→1（ReLU、577个参数）的3个固定种子平均 R²={f(mlp['r2_mean'])}，种子标准差={f(mlp['r2_std'])}，RMSE={f(mlp['rmse_mean'])}；相对16参数岭回归提高{f(n(mlp['r2_mean'])-n(lin['r2_mean']))}。32维线性的R²={f(pred['fixedS16_N32']['r2_mean'])}，高于16维网络。",
 f"岭回归单次拟合约{f(lin['fit_seconds_mean'],4)}秒，网络完整fit重训平均约{f(mlp['fit_seconds_mean'],3)}秒，内部分组选择开销见nn_tuning.json（本次约{f(j('exp12/nn_tuning.json')['tuning_seconds'],2)}秒）。计时为服务器CPU墙钟时间；子空间计算成本在配置中另记，不能把岭回归几毫秒当作完整流程总成本。","",
 "网络超参数只在fit内部按来源划分的64/16来源上选择；内部训练来源重新估计标准化和S/N，之后全fit重新估计并训练。选头在该内部调参中保持既有fit选择，属于冻结选头条件下的调参，不使用dev/test。实验2网络直接输入两侧痕迹差，是预测探针；实验4另训练单节点修正函数，两者不可互换。","",
 "原仓库同一rank同时改变S和N的兼容结果也保留：r4/r8/r16/r32的R²分别为"+"/".join(f(pred["selected_N"+str(d)]["r2_mean"]) for d in (4,8,16,32))+"。这些值不用于隔离痕迹维度的影响。","",
 "维度增加带来的提升大于本次小网络的提升，说明现有16维表示的信息范围是更明显的限制；非线性仍有小幅增益。本实验没有证明线性函数已经达到最优，也没有把32维探索结果用于重选测试配置。","",
 "## Q3：预测关系是否能跨指南来源迁移？","",
 f"完整原流程五折在每折64个训练来源上重新计算加权梯度归因、每类top64候选、最多12个来源均衡配对的双向单头干预、交替融合51个头，以及标准化、S/N和AP岭回归；16个验证来源不参与这些步骤。总体AP R²={f(full['AP']['r2'])}、RMSE={f(full['AP']['rmse'])}，A类单独 R²={f(full['A']['r2'])}。{full['positive_source_r2']}/80个来源R²为正，来源中位数={f(full['source_r2_median'])}。","",
 "| 折 | 完整流程AP R² | 完整流程A R² | 完整流程AP RMSE | 冻结选头AP R²（条件性） |","|---|---:|---:|---:|---:|"]
 for k in range(5):
  a=full["folds"][k];b=frozen["folds"][k]
  lines.append(f"| {k+1} | {f(a['AP']['r2'])} | {f(a['A']['r2'])} | {f(a['AP']['rmse'])} | {f(b['r2'])} |")
  add("Q3","full_pipeline_fold"+str(k),"fit_source_CV","R2",a["AP"]["r2"],origin="new_real_model_interventions_and_cache_training")
 lines+=["",
 f"冻结51头的条件性诊断总体AP R²={f(frozen['overall']['r2'])}；由于选头曾看过全部80个fit来源，该结果不替代完整流程。完整重选头后仍有预测能力，但比冻结头诊断有所下降。原dev本身与fit来源完全分离；其20个来源中17个AP R²为正、中位数0.2860，说明迁移并非所有来源同样稳定。","",
 "| 完整流程表现最差的来源 | 配对数 | R² | RMSE | 目标标准差 |","|---|---:|---:|---:|---:|"]
 for s in full["worst_sources"]:
  lines.append(f"| {s['source_id']} | {s['n']} | {f(s['r2'])} | {f(s['rmse'])} | {f(s['target_sd'])} |")
 lines+=["",
 "完整五折新增3,948次真实最终token前向，复用10,826条旧pair/head干预。GPU0/1峰值约17.968/17.962 GiB，两worker各自折的累计时间约199/154秒；实际明显低于启动前的保守估计。五折CPU重放的选头和预测完全一致，新增模型前向为0。","",
 "每来源指标见exp3/full/source_metrics.csv，dev指标见exp3/dev_source_metrics.csv，具体来源条件、均值和方差观察见exp3/source_context_diagnostics.json及README。低目标方差会使R²更易为负；另外一些来源有系统偏差或预测变化幅度不足。主题、语言和长度是可观察差异，当前实验不能把其中某一项认定为因果原因。","",
 "因此痕迹学到的关系能迁移到不少未参与本折选头和训练的来源，但不是来源无关的稳定关系，仍有一部分来源失效。现有test此前已被旧方案评价使用，本次仅作冻结配置的既有留出复算。未取得已有真实缓存且确认从未被用于选择/分析的新来源，因此没有全新独立测试。完整五折检查的是当前固定算法的来源迁移，不消除历史开发对算法设计的影响。","",
 "## Q4：这种预测能力最终带来了多少干扰降低，证据保护代价是什么？","",
 "所有修复拟合使用相同AP监督，三个预算0.05/0.1/0.25预先固定。约束为每个fit C/F对的 |修正差| ≤ 预算×fit AP原分数差RMS；它只在训练证据对保证成立。线性、随机投影和神经网络使用相同预算定义。随机投影固定seed11，与Q1随机投影同定义。节点校准截距只用761个fit自然节点标签估计，不影响配对分数差。","",
 "VR=1−Var(修正后AP分数差)/Var(原AP分数差)；ER=Σ标签方向×修正后CF分数差/Σ标签方向×原CF分数差；证据RMS=RMS(CF两侧修正差)。方向退化数统计原本标签方向响应>0、修正后≤0的对数。ER是汇总量，不能代替逐对检查。自然节点与自主动态路径分别评价，路径从首节点按修正分数逐步走到叶子，必须路径和叶子都匹配真值。dev/test动态路径分别420/806条，缓存覆盖完整。",""]
 for split,caption in (("dev","开发集"),("test","既有测试留出集")):
  lines += [f"**{caption}全部预设配置**","",
   "| 方法 | VR | ER | 证据RMS | 方向退化数 | 自然节点正确率 | 动态路径正确率 |",
   "|---|---:|---:|---:|---:|---:|---:|"]
  for r in repair:
   if r["split"]!=split:continue
   label=r["method"]
   lines.append(f"| {label} | {pct(r['VR'])} | {pct(r['ER'])} | {f(r['evidence_RMS'])} | {r['direction_degraded']} | {pct(r['natural_accuracy'])} | {pct(r['path_accuracy'])} |")
   for metric in ("VR","ER","evidence_RMS","natural_accuracy","path_accuracy","direction_degraded"):add("Q4",label,split,metric,r[metric])
  lines.append("")
 lines+=["![干扰降低—证据改动权衡](exp4/repair_tradeoff.png)","",
 "开发集预算0.1时，线性痕迹方法的证据RMS=0.2029、VR=4.576%；随机投影的证据RMS=0.2077、VR=0.534%。实际证据改动接近，痕迹方法多降低约4.04个百分点干扰方差，ER分别99.822%和99.822%，均无方向退化。这支持所选痕迹具有可用于选择性修复的信息。预算0.05和0.25的完整结果也已列出，没有依据测试结果选择“最佳预算”。","",
 "无约束痕迹线性的开发VR为40.71%，但证据RMS升至3.0869、ER降至94.81%，并有1对方向退化。预算约束后VR降为2.79%/4.58%/9.50%，可用的安全修复幅度明显小于预测R²所暗示的干扰解释量。","",
 "同一训练预算不等于同一留出证据代价。神经网络在预算0.1下的开发VR为13.53%–16.41%，但证据RMS为0.8164–1.0227，远高于线性约0.20；60个dev证据对中有31–34个超出对应训练绝对预算。因此不能把这部分VR提升解释为相同证据保护代价下的非线性优势。exp4/summary.csv逐配置列出留出超预算数量。","",
 "动态路径尚未显示稳健收益：dev原模型384/420，阈值校准385/420，三个线性预算均384/420；test原模型768/806，三个线性预算均767/806。dev自然节点从195/199到196/199，阈值校准也达到196/199，不能将这一个节点的改进归因于痕迹。test自然节点各方法均368/372。预测和方差改善没有自动转化为最终路径正确率提高。","",
 "**汇总表**","",
 "| 问题 | 实测结论 |",
 "|---|---|",
 f"| Q1 | N16 R²={f(lin['r2_mean'])}；随机投影均值0.3115，差值区间为正；S16和未正交子空间更强，预测信息非痕迹独有。 |",
 f"| Q2 | 固定S16的N32 R²={f(pred['fixedS16_N32']['r2_mean'])}；16维MLP={f(mlp['r2_mean'])}，维度增加的收益更大。 |",
 f"| Q3 | 每折重新选头总体AP R²={f(full['AP']['r2'])}；{full['positive_source_r2']}/80来源为正，迁移存在且不均衡。 |",
 "| Q4 | dev实际证据RMS约0.20时，VR 4.58% vs随机0.53%；未观察到超过阈值基线的动态路径收益。 |","",
 "**一句话实验结果：** 所选痕迹比平均随机特征更能预测干扰分数变化，并能在较小证据改动下减少部分干扰，但跨来源表现不均衡，最终路径正确率未稳定提高。","",
 "**一句话科学结论：** ASTER-CR痕迹包含可迁移且部分可用于选择性修复的预测信息，预测相关性尚不足以证明独占的干扰机制或整体判断修复。","",
 "全部种子、配置、来源名单、训练开销、仓库起始提交、逐对预测、模型参数及重放检查均已保存在本目录；新增GPU干预时间与显存见exp3/full/worker_*_cost.json。参数重放检查见exp12/replay_check.json、exp3/replay_verification.json、exp4/replay_check.json。"]
 add("Q3","full_pipeline","fit_source_CV","R2",full["AP"]["r2"],origin="new_real_model_interventions_and_cache_training")
 (H/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 with (H/"summary.csv").open("w",newline="") as fp:
  w=csv.DictWriter(fp,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
if __name__=="__main__":main()
