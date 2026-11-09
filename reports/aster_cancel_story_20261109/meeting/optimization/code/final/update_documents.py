
import json,shutil,re,time,copy
from pathlib import Path
import numpy as np
B=Path('/home/gcx/aster_optimization_20261109')
R=Path('/home/gcx/aster-real-completion-publish-20261109/reports/aster_cancel_story_20261109')
DEST=R/'meeting/optimization'
DEST.mkdir(exist_ok=True);(DEST/'models').mkdir(exist_ok=True)
sel=json.loads((B/'final/selected_calibration_ablation.json').read_text())
metrics=json.loads((B/'final/results_calibration_ablation/metrics.json').read_text())
cmp=json.loads((B/'final/results_calibration_ablation/comparisons.json').read_text())
# Independent results and code, without copying large input caches or every search checkpoint.
for folder in ('shared','score_conditioned','neural','linear','diagnosis','selection','final'):
 source=B/folder;target=DEST/'code'/folder;target.mkdir(parents=True,exist_ok=True)
 for p in source.glob('*.py'):shutil.copy2(p,target/p.name)
 for p in source.glob('*configs.json'):shutil.copy2(p,target/p.name)
for name in ('selected.json','selected_calibration_ablation.json'):
 shutil.copy2(B/'final'/name,DEST/name)
for p in (B/'final/results_calibration_ablation').iterdir():
 if p.is_file():shutil.copy2(p,DEST/p.name)
shutil.copy2(B/'final/scores_calibration_ablation.npz',DEST/'scores.npz')
shutil.copy2(B/'final/results/metrics.json',DEST/'dev_path_calibration_metrics.json')
for source,name in [(B/'shared/residual_basis.npz','residual_basis.npz'),(B/'linear/feature_bases.npz','head_bases.npz'),(Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/supports_r16.npz'),'supports_r16.npz')]:shutil.copy2(source,DEST/name)
models={}
def save_model(member):
 path=member['model_file']
 if path not in models:
  q=Path(path);name='model_'+str(len(models)).zfill(2)+q.suffix;shutil.copy2(q,DEST/'models'/name);models[path]='models/'+name
 return models[path]
portable=copy.deepcopy(sel)
for name,entry in portable['methods'].items():
 if entry['kind']=='ensemble':
  for member in entry['members']:member['original_model_file']=member['model_file'];member['model_file']=save_model(member)
 else:entry['original_model_file']=entry['model_file'];entry['model_file']=save_model(entry)
(DEST/'selected_portable.json').write_text(json.dumps(portable,ensure_ascii=False,indent=2)+'\n')
# Compact records keep every searched training/checkpoint config and its fit/dev outcome.
catalog=[]
for rr in json.loads((B/'linear/candidates.json').read_text()):
 cfg=rr['config'];d=rr['metrics']['dev']
 catalog.append(dict(kind='linear',config=cfg,dev_AP_VR=d['AP']['variance_reduction'],dev_CF_ER=d['CF']['mean_evidence_retention'],dev_CF_RMS=d['CF']['correction_rms'],dev_natural=d['natural']['correct']))
for rr in json.loads((B/'selection/candidates.json').read_text()):
 catalog.append(dict(kind=rr['kind'],name=rr['name'],tier=rr['tier'],config=rr['config'],seed=rr['seed'],gain=rr['gain'],b=rr['b'],dev_AP_VR=rr['dev']['AP']['variance_reduction'],dev_CF_ER=rr['dev']['CF']['mean_evidence_retention'],dev_CF_RMS=rr['dev']['CF']['correction_rms'],dev_natural=rr['dev']['natural']['correct'],dev_path=rr['path']['correct']))
(DEST/'search_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
shutil.copy2(B/'linear/ensemble_candidates.json',DEST/'ensemble_candidates.json')
summary=dict(training_configurations=874,linear_fits=540,neural_training_runs=334,neural_checkpoints=470,
 linear_gain_and_bias_configurations=2008,neural_operating_points=1868,ensemble_configurations=64,
 best_main='optimized_balanced_fit_calibration',model_files=len(models),
 test_used_for_training=False,post_initial_test_calibration_ablation=sel['calibration_ablation_note'])
(DEST/'experiment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
M=metrics['methods']
labels={'raw':'原模型','no_w':'仅 fit 校准','linear_output':'原 16 维线性对消','optimized_strict_fit_calibration':'扩展：较强证据保护','optimized_balanced_fit_calibration':'扩展：平衡配置','optimized_high_vr_fit_calibration':'扩展：较高干扰降低','optimized_aggressive_fit_calibration':'扩展：更大修正','optimized_max_vr_fit_calibration':'扩展：最高 dev VR 配置','optimized_path_fit_calibration':'扩展：dev 路径优先','trace_only_balanced_fit_calibration':'仅头痕迹 MLP','score_only_balanced_fit_calibration':'仅原分数 MLP','linear_expanded_balanced_fit_calibration':'扩充子空间线性对消'}
def f(x,n=2):return f'{x:.{n}f}'
def pct(x):return f(100*x)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']+['---:']*(len(h)-1))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
rows=[]
order=list(labels)
for name in order:
 r=M[name];g=r['groups']
 rows.append([labels[name],pct(g['AP']['variance_reduction']),pct(g['CF']['mean_evidence_retention']),f(g['CF']['correction_rms'],4),pct(g['CF']['correction_rms_over_ap_rms']),g['CF']['response_direction_degraded'],str(g['AP']['paired_correct'])+'/2293',str(g['natural']['correct'])+'/372',str(r['path']['correct'])+'/806'])
main_table=table(['方法','A/P VR%','C/F ER%','证据修正 RMS','修正／干扰%','方向退化','A/P 全对','自然节点','路径'],rows)
section='''### 8.10 缓存优化扩展：更丰富的特征与证据保护

为提高实际效果，我们在固定线性实例之后开展缓存级开发搜索。CPU 完成 540 个线性拟合，GPU 0、1 完成 334 次小网络训练，共 874 个拟合或训练配置；神经网络保存 470 个检查点。随后评价 2,008 个线性幅度／阈值配置、1,868 个神经网络工作点和 64 个有限集成。所有训练、标准化、降维与配置选择使用 fit/dev，没有新增主干前向、其他模型或其他数据集。测试仍是此前用于旧版评价的同一留出集合。

**扩展模型。** 平衡配置读取既有 N、S 各 16 维的头坐标、在 761 个自然 fit 节点上标准化并估计的末层残差 PCA 64 维，以及原分数的 ℓ/10、|ℓ|/10、tanh(ℓ/4) 三个特征，共 99 维。网络为 99→128→1，GELU 隐藏层带偏置、输出层无偏置，每个网络 12,928 个系数。种子 11、22、33 的三个网络等权平均修正，共 38,784 个训练系数，另有一个截距。该扩展直接在输出端修正，加入了末层残差信息和原分数，与原 16 维线性实例分别报告。

为提高保护的迁移性，C/F 两侧加入相同的标准化输入高斯扰动，标准差 0.1，再惩罚共享扰动下的修正差。平衡配置每个种子训练 1,600 步，AdamW 学习率 0.001、weight decay 0.001；每步 128 对 A/P，其中固定抽取 12 对 P，其余从全部 A/P 抽取。使用全部 135 对 C/F 和 761 个自然 fit 节点。损失包含 A/P 差分均方误差、权重 10,000 的 C/F 修正均方误差、权重 1 的自然节点分类损失，以及权重 0.00001 的自然节点修正平方均值。末层零初始化。

原线性实例采用逐对硬预算；扩展采用证据均方惩罚，并按开发集“证据修正 RMS／原始 A/P RMS”的 5%、10%、20%、40% 档选择修正幅度。两种保护条件分别记录，不将其视为同一个训练约束。平衡配置的开发集为 VR=16.99%、ER=99.61%，使用路径校准截距时为 388/420；较强保护配置为 VR=8.92%。

**校准对照。** 开发集路径选择截距使原模型从 384/420 提高到 388/420，但该截距在测试上仅为 744/806；平衡扩展对应为 754/806。修正的差分指标不受常数截距影响，因此我们统一补报所有搜索时已保存的 fit 自然节点交叉熵截距。以下表格使用这些预存截距；网络、幅度和截距均未用测试标签拟合。补报发生在首次扩展测试之后，按统一规则覆盖全部入选配置，作为同一留出集上的校准消融报告，不称为一次新的独立测试。完整 dev 路径截距结果同样保留。

'''+main_table+'''
**实际改进。** 平衡扩展的测试 VR 从原线性实例的 4.10% 提高至 14.59%，证据修正 RMS 从 0.2539 略降至 0.2403 logit，ER 从 99.60% 提高到 99.77%，40 对 C/F 没有新增方向退化。按来源配对抽样，VR 增量为 10.49 个百分点，95% 区间 [6.94, 13.92] 个百分点。与原三种子 16 维 MLP 输出的平均结果相比，干扰降低率接近，证据修正 RMS 从约 0.6491 降至 0.2403 logit。

较强保护配置将证据修正 RMS 降至 0.1201 logit，VR 为 8.02%；相对原线性实例，证据改动约减半，干扰降低率约翻倍。平衡配置的自然节点为 368/372、路径为 768/806，原线性实例为 368/372、767/806，原模型为 368/372、768/806。平衡配置相对原线性实例的路径差区间为 [−0.443, 0.806] 个百分点，当前改进集中在干扰与证据的权衡，不能据此声称分类正确率显著提高。A/P 全对数也须单列，平衡配置为 2262/2293，原线性实例为 2266/2293。

**分项测试。** 以下为原线性实例、较强保护与平衡配置。A/P、C/F 分别报告，F0 只评价，不作拟合监督。

'''
rs=[]
for name in ['linear_output','optimized_strict_fit_calibration','optimized_balanced_fit_calibration','optimized_aggressive_fit_calibration']:
 g=M[name]['groups']
 rs.append([labels[name],pct(g['A']['variance_reduction']),pct(g['P']['variance_reduction']),pct(g['C']['mean_evidence_retention']),f(g['C']['correction_rms'],4),pct(g['F']['mean_evidence_retention']),f(g['F']['correction_rms'],4),pct(g['F0']['variance_reduction'])])
section+=table(['方法','A VR%','P VR%','C ER%','C 修正 RMS','F ER%','F 修正 RMS','F0 VR%'],rs)+'\n'
rms=[M['balanced_seed'+str(s)+'_fit_calibration']['groups']['CF']['correction_rms'] for s in (11,22,33)]
vr=[M['balanced_seed'+str(s)+'_fit_calibration']['groups']['AP']['variance_reduction']*100 for s in (11,22,33)]
section+=f'''平衡架构三个独立种子的测试 VR 为 {f(vr[0])}%、{f(vr[1])}%、{f(vr[2])}%，均值±样本标准差为 {f(float(np.mean(vr)))}±{f(float(np.std(vr,ddof=1)))} 个百分点；它们的证据修正 RMS 为 {f(rms[0],4)}、{f(rms[1],4)}、{f(rms[2],4)}。集成先平均修正，再采用预存的统一幅度和截距，因而其指标不等于逐种子指标均值。来源抽样与训练种子波动分别保存。

配置、训练脚本、入选模型和逐例结果见 [optimization](meeting/optimization/)。[selected_calibration_ablation.json](meeting/optimization/selected_calibration_ablation.json) 保存冻结的模型、幅度及两个预存截距，[metrics.json](meeting/optimization/metrics.json) 和 [comparisons.json](meeting/optimization/comparisons.json) 保存全部分项、来源区间与配对差。原始逐例缓存和未入选搜索结果保留在服务器 /home/gcx/aster_optimization_20261109；原方法、开发集结果和五模型模拟表均保留。

'''
paper=R/'paper_story_latest.zh.md';s=paper.read_text()
sim=s[s.index('## 10. 五模型'):s.index('## 11. 讨论')]
assert '### 8.10 ' not in s
s=s.replace('## 9. 合成机制实验',section+'## 9. 合成机制实验',1)
s=s.replace('当前线性实例仍为 16 个系数加截距','线性实例为 16 个系数加截距')
key='方法直接优化任务所需的分数不变性，并明确控制证据改动。'
s=s.replace(key,'进一步的缓存优化扩展在同一留出集上达到 VR=14.59%、ER=99.77%、证据修正 RMS=0.2403 logit，路径为 768/806；较强保护配置以 0.1201 logit 证据改动获得 VR=8.02%。'+key,1)
s=s.replace('## 11. 讨论\n','## 11. 讨论\n\n缓存优化扩展说明，更多可用特征、输入标准化、共享扰动下的证据损失和集成能改善实际干扰／证据权衡。该扩展使用了末层残差和原分数，系数数量高于 16 维线性实例，应按实际实现与成本解释。开发集路径截距的迁移失败也说明，差分修正与分类校准需要分别检验；本轮统一补报搜索时已保存的 fit 校准，结果作为同一留出集上的扩展探索报告。\n',1)
key='方法把“可预测干扰”和“保留证据”落实为同一个修正函数的拟合目标与约束。'
s=s.replace(key,'进一步的缓存优化扩展将测试 VR 提高到 14.59%，证据修正 RMS 为 0.2403 logit、路径为 768/806；更强保护配置以 0.1201 logit 改动获得 VR=8.02%。'+key)
assert s[s.index('## 10. 五模型'):s.index('## 11. 讨论')]==sim
paper.write_text(s,encoding='utf-8')
brief=R/'meeting/meeting_brief.zh.md';s=brief.read_text()
brief_table=table(['配置','测试 VR%','ER%','证据修正 RMS','路径'],[[labels[n],pct(M[n]['groups']['AP']['variance_reduction']),pct(M[n]['groups']['CF']['mean_evidence_retention']),f(M[n]['groups']['CF']['correction_rms'],4),str(M[n]['path']['correct'])+'/806'] for n in ['linear_output','optimized_strict_fit_calibration','optimized_balanced_fit_calibration','optimized_aggressive_fit_calibration']])
brieftext='''### 缓存优化后的结果

本轮又尝试 874 个拟合或训练配置，并做开发集幅度、校准和有限集成比较。扩展平衡配置使用头坐标、末层残差 PCA 和原分数，三个种子的小网络平均修正；完整实现见正文第 8.10 节。

'''+brief_table+'''
平衡配置将 VR 从 4.10% 提高到 14.59%，证据改动略降；较强保护配置使证据改动约减半，同时将 VR 提高到 8.02%。自然节点均为 368/372，准确率提升尚无明确证据。开发集按路径选择截距迁移较差，表中统一使用搜索时已保存的 fit 校准；这一系统性校准对照在首次扩展测试后补报，完整失败结果同样保留。

'''
s=s.replace('## 4. 合成机制实验',brieftext+'## 4. 合成机制实验',1);brief.write_text(s,encoding='utf-8')
p=R/'README.md';s=p.read_text()
s+='\n缓存优化扩展见正文第 8.10 节：平衡配置测试 VR=14.59%、ER=99.77%、证据修正 RMS=0.2403、路径768/806；较强保护配置 VR=8.02%、证据修正 RMS=0.1201。完整搜索、模型和校准消融见 [optimization](meeting/optimization/)。\n'
p.write_text(s,encoding='utf-8')
# Keep the standalone server manuscript aligned with the same published version.
for rel in ['paper_story_latest.zh.md','meeting/meeting_brief.zh.md','README.md']:
 shutil.copy2(R/rel,Path('/home/gcx/aster_cancel_story_20261109')/rel)
print('Existing manuscript and brief updated; selected experiment package prepared')

