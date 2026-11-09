
import json,re,shutil
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if (OUT/'document_update.json').exists():raise SystemExit('Documents already updated')
BASE=OUT.parents[1]
REPO=Path('/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
m=json.loads((OUT/'metrics.json').read_text())
sm=json.loads((OUT/'seed_summary.json').read_text())['methods']
mm=m['methods']
names={'raw':'原模型','no_w':'去掉 w／仅校准 b','linear_output':'线性输出对消（主方法）','no_b':'去掉 b','no_CF':'去掉 C/F 约束','leace':'LEACE','iti':'ITI','closed_form':'闭式擦除','ordinary_r8':'普通读出（冻结 r8）','ordinary_matched':'普通读出（匹配系数）','last_layer':'末层偏移','loreft':'LoReFT','lora':'LoRA','lofit':'LoFiT','r8':'8 维痕迹','unorthogonal_U':'未正交化 U','signal_only_S':'只用证据子空间 S','random_subspace':'随机子空间（种子 11）','r32_budget_0.1':'32 维 t＋S','mlp_output':'MLP 输出对消','linear_representation':'线性末层表示编辑','mlp_representation':'MLP 末层表示编辑'}
def f(x,d=2):return f'{x:.{d}f}'
def pct(x):return f(100*x)
def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rs])+'\n'
def count(r,key):return f"{r[key]}/{r['n']}"
def interval(z,scale=100):return '['+', '.join(f(scale*v) for v in z)+']'
def group(n,g):
 if n in sm:
  ns=[n+'_seed'+str(s) for s in (11,22,33)]
  keys=mm[ns[0]]['groups'][g]
  return {k:float(np.mean([mm[x]['groups'][g][k] for x in ns])) for k,v in keys.items() if isinstance(v,(int,float))}
 return mm[n]['groups'][g]
primary=['raw','no_w','linear_output','leace','iti','closed_form','ordinary_r8','ordinary_matched','last_layer','loreft','lora','lofit']
detail=primary+['no_b','no_CF','r8','unorthogonal_U','signal_only_S','random_subspace','r32_budget_0.1','mlp_output','linear_representation','mlp_representation']
mainrows=[]
for n in primary:
 g=mm[n]['groups'];p=mm[n]['path']
 mainrows.append([names[n],f(g['AP']['difference_rms'],4),pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),pct(g['CF']['correction_rms_over_ap_rms']),g['CF']['response_direction_degraded'],count(g['AP'],'paired_correct'),count(g['CF'],'paired_correct'),count(g['natural'],'correct'),count(p,'correct')])
main_table=table(['方法','A/P RMS','VR%','C/F ER%','C/F 修正 RMS','修正／干扰%','方向退化','A/P 全对','C/F 全对','自然节点','路径'],mainrows)
section='''### 8.5 Qwen 留出测试与冻结基线

本轮完成 3,100 个测试输入和 806 例真实动态路径评价。该集合此前已用于旧版方案评价，因此称为“留出测试集”。当前主方法直接读取既有 r=16、λ=0.01、τ=0.1 的 w 与 b；既有开发集结果沿用。新增拟合仅使用原 fit 配对，测试用于评价。fit/dev/test 分别有 80/20/40 个来源。

所有旧基线使用冻结配置和现有逐例分数：普通读出为 ordinary_r8，匹配系数读出另列；LoRA、LoFiT、LoReFT、末层偏移分别沿用学习率 0.001、0.001、0.01、0.0001。没有重新搜索学习率、训练基线或运行主干。不同监督和作用位置按各自实现解释，基线结果用于共同任务指标比较。

下表的证据修正是“方法引起的配对分数差变化”，单位 logit；“修正／干扰”用测试原始 A/P RMS 作分母。方向退化只计原本沿真值方向响应、修正后变为非正的配对。自然节点为 original=True 且 variant=full 的 372 例。

'''+main_table+'''
主方法的 A/P 方差降低为 4.10%，平均 C/F ER 为 99.60%，证据修正 RMS 为 0.2539 logit，40 对证据配对没有新增方向退化；其中 11/40 超过训练预算，最大修正 0.5569 logit。训练逐对约束仍需留出评价。主方法自然节点为 368/372，与原模型、仅校准一致；路径为 767/806，原模型为 768/806，仅校准为 767/806。A/P 全对数比原模型多 8 对，但比仅校准少 1 对，因此分类收益不能归因于输入相关修正。

ITI 和普通读出虽大幅降低干扰变化，也大幅压低证据响应；无约束 32 维对消同样出现明显退化。LEACE、闭式擦除和参数高效方法的保护条件与主方法不同，不能只按 VR 排名。旧 LoReFT 缓存的测试分数与原模型一致，按实际零变化结果报告。

### 8.6 A/P、C/F 分项与 F0

A 有 2,283 对、40 个来源；P 有 10 对、5 个来源；C 有 33 对、11 个来源；F 有 7 对、2 个来源；F0 有 6 对、2 个来源。P、F 和 F0 的来源覆盖较少。下面逐项列出 RMS 和保留率，避免由 A 主导的汇总掩盖其他类型。新增三种子模型列均值；逐种子值、全部配对全对率及置信区间见 [summary.csv](meeting/real_completion/summary.csv) 与 [metrics.json](meeting/real_completion/metrics.json)。

'''
rs=[]
for n in detail:
 A=group(n,'A');P=group(n,'P');F0=group(n,'F0')
 rs.append([names[n],f(A['difference_rms'],4),pct(A['variance_reduction']),f(P['difference_rms'],4),pct(P['variance_reduction']),f(F0['difference_rms'],4),pct(F0['variance_reduction'])])
section+=table(['方法','A RMS','A VR%','P RMS','P VR%','F0 RMS','F0 VR%'],rs)+'\n'
for kind in ('C','F'):
 rs=[]
 for n in detail:
  g=group(n,kind)
  rs.append([names[n],pct(g['mean_evidence_retention']),f(g['correction_rms'],4),pct(g['correction_rms_over_ap_rms']),f(g['response_direction_degraded'],0),pct(g['paired_accuracy'])])
 section+=f'**{kind} 证据配对。**\n\n'+table(['方法','ER%','证据修正 RMS','修正／干扰%','方向退化数','配对全对率%'],rs)+'\n'
section+='''F0 只作单列评价，不进入本轮 A/P 拟合或 C/F 约束。ER 是方向加权平均响应之比，可能略高于 100%；它不表示逐对响应完全保留。

### 8.7 核心消融与预算权衡

除明确去掉 C/F 的对照外，新增线性消融使用同一 fit A/P、λ=0.01、逐对预算 δ=0.2923349 logit，再单独拟合 b。未正交化 U 使用原 fit A/P 标准化差分的前 16 个方向；S 和正交痕迹沿用既有子空间；随机子空间固定种子 11。8 维沿用既有支持空间，32 维是既有 t＋S 扩展，不将其称为 32 维纯干扰子空间。去掉 w、去掉 b、无约束及 32 维结果复用已有系数。

'''
rs=[]
for n in ['no_w','linear_output','no_b','no_CF','unorthogonal_U','signal_only_S','random_subspace','r8','r32_budget_0.1','r32_no_CF']:
 label=names.get(n,'32 维去掉 C/F 约束')
 g=mm[n]['groups']
 rs.append([label,pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),count(g['natural'],'correct'),count(mm[n]['path'],'correct')])
section+=table(['消融','A/P VR%','C/F ER%','C/F 修正 RMS','自然节点','路径'],rs)+'\n'
section+='''在相同训练预算下，16 维正交痕迹优于本轮 U、S 和随机子空间的干扰降低率，8 维及 32 维扩展没有在测试上超过 16 维主方法。去掉 b 不改变差分指标，却将路径数从 767 改为 769；该测试现象只作消融解释，不据此重选配置。

**既有预算曲线的留出评价。** 不同预算逐点列出，允许的保护条件随 τ 改变；无约束单独标明。

'''
rs=[]
for n in ['budget_0.0','budget_0.05','linear_output','budget_0.25','budget_0.5','budget_1.0','budget_2.0','no_CF','r32_budget_0.0','r32_budget_0.1','r32_budget_0.25','r32_no_CF']:
 r=mm[n];g=r['groups'];tau=r['settings'].get('tau')
 rs.append(['32 维' if n.startswith('r32') else '16 维','无约束' if tau is None else str(tau),pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),g['CF']['response_direction_degraded']])
section+=table(['特征','τ','A/P VR%','C/F ER%','C/F 修正 RMS','方向退化数'],rs)+'\n'
section+='''### 8.8 真实非线性对照：函数与修正位置

为比较函数表达能力与作用位置，三个新增模型都读取相同 16 维痕迹。MLP 为 16→32→1 或 16→32→16，隐藏层用 GELU 和偏置，末层不加偏置；线性末层表示编辑为无偏置 16→16。它们分别有 576、1,056、256 个训练系数，另各有单独校准的 b。

末层表示编辑采用 h′=h+Vg(t)，V 是原 fit A/P 末层残差差分的前 16 个方向。G 使用真实模型冻结的最终 RMSNorm 权重、原 epsilon 和“是／否”输出行进行 FP32 计算。分数按 ℓ_cache+[G(h′)−G(h)] 锚定原缓存；零编辑时修正严格为 0。该实现是末层表示编辑，评价当前输出行为，不作为完整概念擦除的证明；既有主干分数重建检查沿用。

每个新增模型使用种子 11、22、33，各训练 600 步；Adam 学习率 0.001，每步随机抽取 64 对 fit A/P，并使用全部 135 对 fit C/F。目标为 A/P 差分均方误差，正则为 0.01×系数平方和／4,887，与线性求和损失中的 λ 对齐；超预算平方惩罚权重为 100。所有末层权重零初始化。训练后仅按 fit C/F 检查修正幅度，每次减半，最多 20 次，不满足则使用零修正；随后单独校准 b，没有配置搜索。三类模型均满足原训练预算。

下表为真实缓存测试，均值±样本标准差来自三个训练种子；线性输出主方法为既有单次确定性拟合。各次训练的尺度、最大 fit 证据修正和实际成本保存在同目录 JSON 中。

'''
rs=[]
g=mm['linear_output']['groups']
rs.append(['线性输出对消','16＋b',pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),pct(g['CF']['correction_rms_over_ap_rms']),pct(g['natural']['accuracy']),pct(mm['linear_output']['path']['accuracy'])])
for n,params in [('mlp_output','576＋b'),('linear_representation','256＋b'),('mlp_representation','1056＋b')]:
 def val(g,k,scale=100,d=2):
  x=sm[n][g][k];return f(x['mean']*scale,d)+'±'+f(x['seed_sample_sd']*scale,d)
 ratios=[mm[n+'_seed'+str(s)]['groups']['CF']['correction_rms_over_ap_rms'] for s in (11,22,33)]
 rs.append([names[n],params,val('AP','variance_reduction'),val('CF','mean_evidence_retention'),val('CF','correction_rms',1,4),pct(float(np.mean(ratios)))+'±'+pct(float(np.std(ratios,ddof=1))),val('natural','accuracy'),val('path','accuracy')])
nonlinear_table=table(['方法','系数','A/P VR%','C/F ER%','C/F 修正 RMS','修正／干扰%','自然正确率%','路径正确率%'],rs)
section+=nonlinear_table+'''
MLP 输出和 MLP 末层表示编辑的平均 VR 分别为 14.59% 和 13.12%，均高于线性输出的 4.10%；它们的平均证据修正 RMS 也更大。相同 fit 预算没有使留出集的保护程度相同，因此这些结果说明表达能力与保护迁移的权衡，不能据此声称某一修正位置全面更优。三个新增模型的全部测试证据配对均未新增方向退化。

线性末层表示编辑的平均 VR 为 3.48%，路径均为 767/806；MLP 输出的路径分别为 771、771、770/806，MLP 末层表示编辑为 769、768、769/806。MLP 输出相对于线性输出的平均路径正确率差为 0.455 个百分点，按来源的 95% 区间为 [0.000, 0.982] 个百分点；测试集覆盖下的改变量较小。

### 8.9 来源不确定性、复算与成本

按 40 个测试来源成组抽样 2,000 次，种子 11；同一次来源抽样同时用于所有方法，直接计算方法差的百分位 95% 区间。下表的新增模型区间先对固定的三个拟合种子取指标均值，再对来源抽样；它与三次训练的标准差分别解释。第 9 节合成生成种子波动保持原口径，第 10 节预设模拟不并入真实不确定性。

'''
rs=[]
r=mm['linear_output']
rs.append(['线性输出',interval(r['groups']['AP']['variance_reduction_ci']),interval(r['groups']['CF']['mean_evidence_retention_ci']),interval(r['path']['accuracy_ci'])])
for n in ('mlp_output','linear_representation','mlp_representation'):
 r=sm[n];rs.append([names[n],interval(r['AP']['variance_reduction']['source_ci']),interval(r['CF']['mean_evidence_retention']['source_ci']),interval(r['path']['accuracy']['source_ci'])])
section+=table(['方法','A/P VR 95% 区间（%）','C/F ER 95% 区间（%）','路径正确率 95% 区间（%）'],rs)+'\n'
section+='''主方法路径相对原模型的差为 −0.124 个百分点，来源区间为 [−0.932, 0.655] 个百分点，当前结果不支持准确率提升或 −0.5 个百分点界限下的非劣性结论。P、F、F0 少量来源的部分重采样没有该类配对，未定义指标从对应区间计算中排除；分项样本数与来源数始终列出。

路径由每个方法从根节点按自身分数动态选择分支；只有决策序列与目标叶节点均正确才计整条路径正确。所有可达节点来自既有缓存。新评估器复算的 806 条原模型路径与旧记录逐例一致，零编辑最大分数误差为 0，所有数据及选定基线的配对、节点和路径结果均无缺项。新增三类训练的 fit C/F 均满足 δ；测试超预算另报。

CPU 共读取 27,425 个不同特征文件，整理 28,186 条输入及路径节点记录，准备约 34.6 秒。GPU 0、1 仅运行小模型，未加载 8B 主干；实际峰值分配显存最高约 0.43 GiB，各次 600 步训练约 2 秒，完整计时及训练缩放记录随结果保存。此成本不含历史主干前向、选头、子空间估计和基线训练，也不作为在线端到端延迟。

复算入口为 [prepare.py](meeting/real_completion/prepare.py)、[train.py](meeting/real_completion/train.py)、[evaluate.py](meeting/real_completion/evaluate.py)、[comparisons.py](meeting/real_completion/comparisons.py)。完整指标见 [metrics.json](meeting/real_completion/metrics.json)，来源配对比较见 [comparisons.json](meeting/real_completion/comparisons.json)，训练种子波动见 [seed_summary.json](meeting/real_completion/seed_summary.json)，覆盖和预算核对见 [verification.json](meeting/real_completion/verification.json)。旧缓存、权重及冻结配置保持原版本，新增结果独立保存。

'''
paper=BASE/'paper_story_latest.zh.md'
s=paper.read_text()
oldsim=s[s.index('## 10. 五模型'):s.index('## 11. 讨论')]
s=s.replace('独立测试','留出测试')
s=s.replace('第 8 节报告真实开发集主实验与消融','第 8 节报告真实开发集、留出测试、消融和非线性对照')
s=s.replace('同监督的输出与表示编辑比较、留出测试及路径评价继续完成。','同监督的输出与末层表示编辑比较、留出测试及路径评价见第 8.5–8.9 节。')
s=s.replace('第 9.1 节实际合成训练，第 7.4 节真实对照设计 | 非线性场景中 MLP 有效；真实模型比较待完成','第 9.1 节实际合成训练，第 8.8 节真实对照 | MLP 提高干扰降低率，也增加留出证据改动')
s=s.replace('留出测试待完成。','留出测试结果见第 8.5–8.9 节。')
s=s.replace('当前主方法使用 r=16、固定证据改动预算，并在原开发集上评价。训练配对的硬约束具有直接保证，开发集检验其迁移；部署选择和留出测试仍需按事先确定的规则执行。','当前主方法固定 r=16 和证据改动预算，开发集沿用旧结果，留出测试只评价既有系数。训练配对的硬约束具有直接保证，开发集与留出测试分别检验其迁移。')
s=s.replace('测试集没有参与本轮拟合或评价。','测试集仅用于本轮评价，未参与拟合或配置选择；该集合此前已用于旧版方案评价。')
s=s.replace('**表：repair-v2 配对数量。test 是数据规模，不表示已完成相应测试评分。**','**表：repair-v2 配对数量。fit、dev、test 缓存均完整，测试评分见第 8.5–8.9 节。**')
s=s.replace('所有本轮结果均来自 CPU 重用缓存，没有新增模型前向，也未完成留出测试或动态路径。','本轮复用 CPU 缓存，并在 GPU 0、1 训练三个新增小模型；没有新增主干前向。留出测试与 806 例动态路径已完成。')
s=s.replace('下表按七类列出拟进行实测比较的对照及本文方法。','下表概述相关对照类别及本文方法；本轮实际完成的冻结基线列于第 8.5 节。')
s=s.replace('训练式方法拟采用 LoReFT 秩 1–8、LoFiT 头数 4–16、LoRA 秩 4–16；实际参数量、显存及全部训练阶段耗时应由真实运行导出。当前不填未测量的成本，也不要求更强表达能力的训练方法压缩为仅 17 个系数。','训练式方法沿用旧流程冻结的 LoReFT、LoFiT、LoRA 和末层偏移配置，不重复训练或搜索。其历史训练成本保留原记录，本轮仅复算逐例评价指标。')
start=s.index('### 7.4 用同一目标')
end=s.index('### 7.5 方法设计',start)
s=s[:start]+'''### 7.4 用同一目标比较线性与非线性方法

为区分函数表达能力与修正位置，真实比较使用既有线性输出对消、MLP 输出对消、线性末层表示编辑及 MLP 末层表示编辑。三类新增模型共享 16 维痕迹、fit A/P 与 C/F 配对、证据预算和单独校准流程，具体训练、计分与参数量见第 8.8 节。

表示编辑只改缓存末层残差，再经真实冻结的最终 RMSNorm 和“是／否”输出权重计算修正。它检验这一位置的输出行为，不代称完整概念擦除。所有方法共同报告 A/P、C/F、F0、自然节点与动态路径。来源是置信区间的抽样单位，新增训练种子波动另报。

'''+s[end:]
s=s.replace('第 10.5 节旧版预设；新方法的真实消融待做','第 8.7 节真实消融；随机子空间种子 11')
s=s.replace('第 9.1 节合成；真实模型待做','第 8.8 节真实模型；第 9.1 节合成')
s=s.replace('差分指标代数上不变；第 10.5 节预设分类结果','差分指标不变；第 8.7 节真实分类结果')
s=s.replace('当前真实比较覆盖原模型、校准、无约束回归、证据约束与特征扩展；完整基线排名由实际模型比较补齐。','当前真实比较覆盖原模型、校准、冻结基线、核心消融与末层非线性对照，结果见第 8 节。')
s=s.replace('## 8. 真实开发集主实验与消融','## 8. 真实开发集、留出测试与消融')
s=s.replace('仍需留出测试检验迁移。','留出测试的迁移结果见第 8.5 节。')
s=s.replace('## 9. 合成机制实验',section+'## 9. 合成机制实验',1)
s=s.replace('本轮已确认 fit/dev 的 source_id 无交集；下一阶段完成留出测试、动态路径和同监督的输出/表示编辑对照，并检查冻结修正函数在条件、事实与处置交叉变化下的预测关系。历史几何与多层写回结果统一到 repair-v2，之后按模型安排扩展。','本轮复用 3,100 个留出测试输入和 806 条动态路径完成评价，并补齐核心消融与三种子输出/末层表示编辑对照。当前结果支持小幅干扰降低和较高平均证据保留，分类提升与仅校准的区别仍有限；更灵活的函数在留出证据保护上仍有成本。')
# Add the actual held-out finding to abstract and conclusion without changing old dev results.
key='方法直接优化任务所需的分数不变性，并明确控制证据改动。'
s=s.replace(key,'留出测试上，固定主方法的干扰 VR 为 4.10%、ER 为 99.60%，路径为 767/806，原模型为 768/806；新增 MLP 输出的平均 VR 为 14.59%，同时带来更大证据改动。'+key,1)
key='方法把“可预测干扰”和“保留证据”落实为同一个修正函数的拟合目标与约束。'
s=s.replace(key,'留出测试复现了干扰降低与证据保护的权衡；主方法路径未优于原模型，MLP 输出提高干扰降低率但增加留出证据改动。'+key)
assert s[s.index('## 10. 五模型'):s.index('## 11. 讨论')]==oldsim
paper.write_text(s,encoding='utf-8')
brief=BASE/'meeting/meeting_brief.zh.md';b=brief.read_text().replace('独立测试','留出测试')
b=b.replace('留出测试与动态路径尚未完成。','留出测试与动态路径已完成，结果如下。')
briefrows=[]
for n in ['raw','no_w','linear_output','closed_form','lora','lofit']:
 g=mm[n]['groups'];briefrows.append([names[n],pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),count(mm[n]['path'],'correct')])
insert='''### 留出测试：真实 Qwen 缓存

3,100 个输入、806 条动态路径；该集合此前用于旧版方案评价，称为留出测试集。本轮复用固定系数与冻结基线，只补指标。

'''+table(['方法','A/P VR%','C/F ER%','证据修正 RMS','路径'],briefrows)+'''
主方法自然节点为 368/372，与原模型、仅校准一致；A/P 全对 2266/2293，低于仅校准的 2267/2293。40 对 C/F 无新增方向退化，但 11 对超过训练预算。主方法 VR 的来源 95% 区间为 [2.02%, 6.77%]；路径相对原模型差 −0.124 个百分点，区间 [−0.932, 0.655]，不能声称正确率提升。

### 真实非线性比较

以下三类新增模型各用三个种子、600 步；均满足同一 fit 证据预算，留出证据改动仍有差异。

'''+nonlinear_table+'''
MLP 输出与 MLP 末层表示编辑提高了干扰降低率，也增加留出证据改动，不能只按 VR 判断优劣。末层表示编辑经过真实冻结的 RMSNorm 与输出权重计分，不称为完整概念擦除。主方法、8/16/32 维、去掉 w/b/C/F 约束、U、S 和随机子空间的完整真实消融见正文第 8.7 节。按来源的区间与三次训练的标准差分别报告。

'''
b=b.replace('## 4. 合成机制实验',insert+'## 4. 合成机制实验',1)
b=b.replace('另完成留出测试和 62 个节点的医生核验。','本轮已完成留出测试与真实对照，62 个节点的医生核验仍待完成。')
b=b.replace('下一步在相同证据保护要求下比较不同方法，并用留出测试确认效果。','留出测试显示该权衡，MLP 提高干扰降低率，也带来更大证据改动；路径正确率提升仍需更多证据。')
brief.write_text(b,encoding='utf-8')
readme=BASE/'README.md';r=readme.read_text().replace('第 8 节报告真实开发集结果','第 8 节报告真实开发集、留出测试、核心消融与非线性对照')
r=r.replace('独立测试、完整路径和医生抽样核验待完成。','3,100 个留出测试输入、806 条真实路径、核心消融和三种子非线性对照已完成；测试集此前用于旧版方案评价。医生抽样核验仍待完成。')
r=r.replace('真实开发集 ER=99.82%、干扰 VR=4.58%。','真实开发集 ER=99.82%、干扰 VR=4.58%；留出测试 ER=99.60%、VR=4.10%，路径 767/806。')
readme.write_text(r,encoding='utf-8')
# Synchronize only the existing story and the new, independent result directory.
target=REPO/'reports/aster_cancel_story_20261109'
for rel in ['paper_story_latest.zh.md','meeting/meeting_brief.zh.md','README.md']:
 shutil.copy2(BASE/rel,target/rel)
dest=target/'meeting/real_completion';dest.mkdir(exist_ok=True)
for p in OUT.iterdir():
 if p.suffix in ('.py','.json','.jsonl','.csv','.pt') or (p.suffix=='.npz' and p.name!='coordinates.npz'):shutil.copy2(p,dest/p.name)
dump(OUT/'document_update.json',dict(updated=['paper_story_latest.zh.md','meeting/meeting_brief.zh.md','README.md'],simulation_section_preserved=True,new_report_documents=0))
if __name__=='__main__':print('updated existing manuscript and meeting brief')

