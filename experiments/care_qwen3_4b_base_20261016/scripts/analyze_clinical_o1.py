#!/usr/bin/env python3
"""Saved-data clinical O1 analysis; bootstrap clinical rule families, not vignettes."""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
R=ROOT/'results/obs1_clinical'; F=ROOT/'figures'
METHODS=['baseline','seka','pasta']; METRICS=['A_app','A_non','M_non','P_revert','H_Acc']
COLORS=['#333333','#0072B2','#D55E00']; RNG=np.random.default_rng(261016)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
def dump(p,x): pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str))
def save(fig,name):
    for ext in ['png','pdf','svg']: fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
def interval(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return [float(np.quantile(x,.025)),float(np.quantile(x,.975))] if len(x) else [None,None]
def ratio(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)
def compute_metrics(df,families,ix):
    app=df.subset.eq('app'); non=~app
    specs={'A_app':(df.correct*app,app),'A_non':(df.correct*non,non),
           'M_non':(df.misapplied*non,non),'P_revert':(df.prior_reversion,np.ones(len(df)))}
    values={}; boots={}; numerators={}
    for metric,(num,den) in specs.items():
        temp=pd.DataFrame({'group_id':df.group_id,'num':np.asarray(num),'den':np.asarray(den)}).groupby('group_id').sum().reindex(families,fill_value=0)
        a=temp.num.to_numpy(float); b=temp.den.to_numpy(float)
        values[metric]=float(ratio(a.sum(),b.sum())) if b.sum() else None
        boots[metric]=ratio(a[ix].sum(1),b[ix].sum(1))
        numerators[metric]=a
    a=values['A_app']; b=values['A_non']
    values['H_Acc']=None if a is None or b is None else 2*a*b/(a+b) if a+b else 0.
    aa=boots['A_app']; bb=boots['A_non']
    boots['H_Acc']=ratio(2*aa*bb,aa+bb)
    boots['H_Acc'][(aa+bb)==0]=0.
    return values,boots,numerators
def permutation_p(nums):
    nums=np.asarray(nums,dtype=np.float64); n=len(nums)
    if n<=18:
        patterns=np.arange(2**n,dtype=np.uint32)[:,None]
        signs=((patterns>>np.arange(n))&1)*2.-1
    else: signs=RNG.choice([-1,1],size=(100000,n))
    return float(np.mean(abs(signs@nums)>=abs(nums.sum())-1e-12))
def subset_analysis(df,name):
    families=sorted(df.group_id.unique()); n=len(families)
    if not n: return [],[],{'status':'not_testable','n_cases':0,'n_families':0}
    ix=RNG.integers(0,n,(10000,n))
    metrics=[]; contrasts=[]; cache={}
    for method in METHODS:
        d=df[df.method==method]; vals,boot,nums=compute_metrics(d,families,ix); cache[method]=(vals,boot,nums)
        for metric in METRICS:
            lo,hi=interval(boot[metric]); count=len(d[d.subset==('app' if metric=='A_app' else 'non')]) if metric in ['A_app','A_non','M_non'] else len(d)
            metrics.append({'analysis_subset':name,'method':method,'metric':metric,'estimate':vals[metric],
                'ci_low':lo,'ci_high':hi,'n_cases':count,'n_families':n,
                'bootstrap_valid_reps':int(np.isfinite(boot[metric]).sum())})
    tradeoff={}
    for method in METHODS[1:]:
        for metric in METRICS:
            point=cache[method][0][metric]; base=cache['baseline'][0][metric]
            delta=point-base if point is not None and base is not None else None
            lo,hi=interval(cache[method][1][metric]-cache['baseline'][1][metric])
            p=permutation_p(cache[method][2][metric]-cache['baseline'][2][metric]) if metric in cache[method][2] else None
            contrasts.append({'analysis_subset':name,'method':method,'metric':metric,'estimate':delta,'ci_low':lo,'ci_high':hi,
                'p_family_swap_two_sided':p,'n_families':n})
        a=next(x for x in contrasts if x['method']==method and x['metric']=='A_app')
        m=next(x for x in contrasts if x['method']==method and x['metric']=='M_non')
        point_positive=a['estimate'] is not None and m['estimate'] is not None and a['estimate']>0 and m['estimate']>0
        supported=a['ci_low'] is not None and m['ci_low'] is not None and a['ci_low']>0 and m['ci_low']>0
        tradeoff[method]='supported' if supported else 'inconclusive' if point_positive else 'not_observed'
    return metrics,contrasts,{'status':'testable','n_cases':int(df[df.method=='baseline'].shape[0]),'n_families':n,'tradeoff':tradeoff}
def frame_table(df):
    lines=['|方法|指标|估计|95% CI|n|','|---|---|---:|---|---:|']
    for r in df.itertuples():
        if pd.isna(r.estimate): point='NA'; span='NA'
        else: point=f'{r.estimate:.3f}'; span=f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'
        lines.append(f'|{r.method}|{r.metric}|{point}|{span}|{r.n_cases}|')
    return '\n'.join(lines)
def plot_main(metrics,contrasts,summary):
    m=metrics[metrics.analysis_subset=='eligible_cases']; c=contrasts[contrasts.analysis_subset=='eligible_cases']
    fig,axs=plt.subplots(1,2,figsize=(10,3.9))
    for j,method in enumerate(METHODS):
        a=m[(m.method==method)&(m.metric=='A_app')].iloc[0]; b=m[(m.method==method)&(m.metric=='A_non')].iloc[0]
        axs[0].errorbar(a.estimate,b.estimate,
            xerr=[[max(0,a.estimate-a.ci_low)],[max(0,a.ci_high-a.estimate)]],
            yerr=[[max(0,b.estimate-b.ci_low)],[max(0,b.ci_high-b.estimate)]],
            fmt=['o','s','^'][j],ms=7,color=COLORS[j],capsize=3,label=method.upper())
    axs[0].set(xlabel='Applicable-rule accuracy',ylabel='Nonapplicable-rule accuracy',xlim=(-.04,1.04),ylim=(-.04,1.04),title='(a) Source-grounded clinical conflicts')
    axs[0].legend(frameon=False,fontsize=8)
    for j,method in enumerate(METHODS[1:]):
        d=c[(c.method==method)&c.metric.isin(['A_app','A_non','M_non'])].set_index('metric').loc[['A_app','A_non','M_non']]
        yy=np.arange(3)+(j-.5)*.17
        axs[1].errorbar(d.estimate,yy,xerr=[np.maximum(0,d.estimate-d.ci_low),np.maximum(0,d.ci_high-d.estimate)],
            fmt=['s','^'][j],color=COLORS[j+1],capsize=3,label=method.upper())
    axs[1].axvline(0,color='.6',lw=.8); axs[1].invert_yaxis()
    axs[1].set(yticks=np.arange(3),yticklabels=['Applicable accuracy','Nonapplicable accuracy','Highlighted-rule misuse'],
        xlabel='Change from baseline',title='(b) Paired family-bootstrap 95% CI')
    axs[1].legend(frameon=False,fontsize=8)
    n=summary['screening']
    fig.suptitle(f'Authentic clinical rules; simulated patients | {n["eligible"]} eligible cases ({n["eligible_app"]} app / {n["eligible_non"]} non) | {summary["subsets"]["eligible_cases"]["n_families"]} families',fontsize=9.5)
    fig.tight_layout(); save(fig,'O1_clinical_faithfulness_misuse')
def main():
    df=pd.read_csv(R/'predictions.csv'); screen=pd.read_csv(R/'prior_screen.csv'); manifest=json.loads((R/'manifest.json').read_text())
    frozen=json.loads((R/'screening_frozen.json').read_text())
    assert hashlib.sha256((R/'prior_screen.csv').read_bytes()).hexdigest()==frozen['screen_sha256']
    test=df[df.role=='test'].copy()
    assert len(test)==384*3 and test.groupby(['case_id','method']).size().eq(1).all()
    primary=test[test.prior_eligible]; pairs=test[test.complete_pair_eligible]
    data=[('all_cases',test),('eligible_cases',primary),('complete_eligible_pairs',pairs)]
    metrics=[]; contrasts=[]; summaries={}
    for name,d in data:
        a,b,s=subset_analysis(d,name); metrics.extend(a); contrasts.extend(b); summaries[name]=s
    m=pd.DataFrame(metrics); c=pd.DataFrame(contrasts)
    m.to_csv(R/'metrics.csv',index=False); c.to_csv(R/'contrasts.csv',index=False)
    counts=screen.groupby('group_id').agg(candidates=('case_id','size'),stable=('stable','sum'),eligible=('eligible','sum'),eligible_pair_cases=('complete_pair_eligible','sum')).reset_index()
    counts['complete_eligible_pairs']=counts.eligible_pair_cases//2; counts.to_csv(R/'screening_by_family.csv',index=False)
    summary={'scope':'source-grounded clinical rules with simulated patients; original factual rules, not fabricated counterfactual guidelines',
        'screening':frozen,'stable_cases':int(screen.stable.sum()),'stable_correct_excluded':int((screen.stable&~screen.eligible).sum()),
        'unstable_excluded':int((~screen.stable).sum()),'rules':24,'families':12,'subsets':summaries,
        'bootstrap':'10000 clinical-rule-family cluster resamples; paired methods and patients retained; fixed out-of-fold fits',
        'model':'Qwen/Qwen3-4B-Base','model_revision':manifest['model_revision'],'seka_commit':manifest['seka_commit'],
        'independent_clinician_adjudication':False,'high_confidence_cutoff':None}
    dump(R/'summary.json',summary)
    dump(R/'checks.json',{'screen_frozen_before_interventions':True,'source_verified_all':bool(test.source_verified.all()),
        'outer_test_family_disjoint':all(not(set(f['test_families'])&(set(f['train_families'])|set(f['dev_families']))) for f in manifest['folds']),
        'test_predictions_per_case':3,'conditional_candidate_scoring_identical':True,'independent_clinician_review':False})
    if frozen['eligible_app'] and frozen['eligible_non']: plot_main(m,c,summary)
    # Preserve every family, including those with zero eligible observations.
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.1))
    stage_names=['Source-verified','Stable closed-book','Conflict eligible','Both-side pair cases']
    stage_values=[len(screen),int(screen.stable.sum()),int(screen.eligible.sum()),int(screen.complete_pair_eligible.sum())]
    axs[0].bar(range(4),stage_values,color=['.6','.45','#0072B2','#009E73'])
    for i,v in enumerate(stage_values): axs[0].text(i,v+5,str(v),ha='center',fontsize=8)
    axs[0].set(xticks=range(4),xticklabels=['Verified','Stable','Eligible','Paired'],ylabel='Simulated patient cases',ylim=(0,len(screen)*1.15),
        title='(a) Screening without a confidence cutoff')
    yy=np.arange(len(counts)); axs[1].barh(yy,counts.candidates,color='.85',label='Candidate')
    axs[1].barh(yy,counts.eligible,color='#0072B2',label='Eligible')
    axs[1].set(yticks=yy,yticklabels=counts.group_id.str.replace('_',' '),xlabel='Cases',title='(b) All 12 clinical rule families')
    axs[1].tick_params(axis='y',labelsize=7); axs[1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); save(fig,'O1_clinical_screening')
    # Selection and stability sensitivity are both reported, without pooling their denominators.
    fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
    for ax,subset in zip(axs,['all_cases','eligible_cases','complete_eligible_pairs']):
        d=m[(m.analysis_subset==subset)&(m.metric=='H_Acc')]
        if len(d):
            d=d.set_index('method').loc[METHODS]
            ax.errorbar(range(3),d.estimate,yerr=[np.maximum(0,d.estimate-d.ci_low),np.maximum(0,d.ci_high-d.estimate)],
                fmt='o',capsize=4,color='#0072B2')
            ax.set(xticks=range(3),xticklabels=['Base','SEKA','PASTA'],ylim=(-.03,1.03))
        ax.set_title(subset.replace('_',' ')+'\n'+f'n={summaries[subset]["n_cases"]}, families={summaries[subset]["n_families"]}',fontsize=9)
    axs[0].set_ylabel('Harmonic app/non accuracy, 95% CI')
    fig.suptitle('Family-held-out clinical O1 | screening sensitivity',fontsize=10)
    fig.tight_layout(); save(fig,'O1_clinical_subset_sensitivity')
    # Per-family outcomes and developer-only strength tables remain available.
    per=test.groupby(['group_id','method','subset']).agg(n=('case_id','size'),accuracy=('correct','mean'),misuse=('misapplied','mean'),reversion=('prior_reversion','mean')).reset_index()
    per.to_csv(R/'per_family_results.csv',index=False)
    dev=df[df.role=='dev'].groupby(['outer_fold','method','strength','subset']).agg(n=('case_id','size'),accuracy=('correct','mean'),misuse=('misapplied','mean')).reset_index()
    dev.to_csv(R/'dev_strength_results.csv',index=False)
    pri=screen[screen.eligible]
    report=f"""# 临床规则先验冲突：观察一补充实测报告

本报告补齐先前缺失的临床规则观察一。模型输出均来自服务器上真实 Qwen/Qwen3-4B-Base 前向计算。规则来自公开临床参考资料，病例为模拟病例；未使用真实患者记录，未人为改错指南或交换临床推荐制造冲突。

## 数据与核验

12个临床规则家族、24条规则、192个适用/不适用配对，共384个候选病例。涉及血糖分类、标准成人BMI、CKD GFR、年龄相关血压目标、PE/DVT Wells、CRB65/CURB65、Centor/FeverPAIN、房颤抗凝推荐措辞、骨密度、贫血、COPD气流受限及低血糖。
每条规则在 data/clinical/rule_cards.json 记录权威来源URL、表格/推荐编号、适用人群、单位、必要例外、忠实转述及可执行金标。已检查主要阈值边界、两侧患者、词元span和选项映射；无独立临床专家复核。
这里的“适用/不适用”指同一高亮分支对患者是否适用，两侧都属于该临床规则的有效范围。许多评分已给定，只评估分支分类/推荐措辞，不声称能从完整病历正确计算评分。

## 宽松且明确的闭卷筛选

主分析条件仅三项：来源核验通过；同一组选项下两个闭卷表述的最大概率选项相同；该闭卷选项与真实规则金标不一致。不设置概率、概率间隔、临床专家审核或方法收益门槛。
384例均完成两次闭卷前向。稳定 {int(screen.stable.sum())} 例；稳定但与金标相同、因无冲突排除 {int((screen.stable&~screen.eligible).sum())} 例；表述间不稳定排除 {int((~screen.stable).sum())} 例。
**合格 {frozen['eligible']} 例，其中适用 {frozen['eligible_app']}、不适用 {frozen['eligible_non']}；覆盖 {summaries['eligible_cases']['n_families']} 个临床规则家族。**
**两侧都合格的完整配对 {frozen['complete_eligible_pairs']} 对、{2*frozen['complete_eligible_pairs']} 例。** 单侧合格病例保留在主分析，完整配对作为单独的敏感性分析；没有强制双侧同时通过而把其他合格病例全部丢弃。
筛选CSV在任何干预结果产生前冻结，文件哈希写入screening_frozen.json。完整选项概率、稳定性与排除原因均保存。
“稳定闭卷错误与正确上下文冲突”是本实验对先验冲突的操作性定义，不能证明模型曾记忆并确信某个错误医学事实。

## 主结果：全部合格病例

每个病例都有原始模型、官方SEKA和官方PASTA的独立真实输出；每个临床家族的测试数据都未用于该折投影构建或强度选择。
状态：{summaries['eligible_cases'].get('tradeoff')}。

{frame_table(m[m.analysis_subset=='eligible_cases'])}

A_app/A_non分别是适用/不适用合格病例准确率；M_non是非适用病例选择高亮动作的比例；P_revert是回到闭卷答案的比例；H_Acc为两个准确率的调和平均。分母分别报告，不把答错其他动作自动算成高亮规则误用。
95% CI使用10000次完整临床规则家族bootstrap，同一次抽样保持患者配对和方法比较。并报告家族内方法交换的双侧检验；这些区间固定本次三折拟合，不包括重新拟合投影的不确定性。
权衡只有在ΔA_app和ΔM_non都为正且各自95%区间下限均大于0时才标记supported；只点估计为正时为inconclusive，不强行认定成功。配对变化见contrasts.csv。

## 严格完整配对敏感性分析

状态：{summaries['complete_eligible_pairs'].get('tradeoff')}。

{frame_table(m[m.analysis_subset=='complete_eligible_pairs'])}

## 未筛选全体病例

{frame_table(m[m.analysis_subset=='all_cases'])}

## 方法与资源

固定模型revision {manifest['model_revision']}；官方SEKA/PASTA commit {manifest['seka_commit']}，源码未修改。
三折按临床规则家族划分。每折4个测试家族；其余8个中6个用于谱投影，2个用于强度选择。每个家族只有一次正式测试。24条规则的相关变体始终同属一个家族。
SEKA复用官方传统谱投影计算与k_norm hook；只改变临床提示组装和流式Key提取。投影取最后10层、8个KV头；PASTA对应全部32个Q头，GQA比例4:1。
SEKA强度网格0.25/0.5/1/2/4；PASTA log-bias网格0.25/0.5/1/2/log(100)。开发集使用全部病例的H_Acc最大值选强度，同分取较小值；不看测试表现选强度。
选项顺序按pair预先固定并在三种方法和两个闭卷表述间一致；两侧病例除数值外保持文本、选项与规则相同。用空格+A/B/C/D的单词元条件概率做一致判定。
BF16、GPU {manifest['physical_gpu']} ({manifest['gpu']})，模型运行 {manifest['seconds']/60:.2f} 分钟；峰值显存 {manifest['peak_memory_allocated_GiB']:.2f} GiB。
方法确实改变概率：{manifest['intervention_effects']}。详细参数、折分、来源、软件和原始概率见manifest和predictions.csv。

## 局限

- 真实规则、模拟病例；无患者队列或独立医生复核，不能外推为临床疗效或安全性。
- 只有12个规则家族，评分值及选项变体不等于384个独立临床规则。所有区间按家族估计，但家族间仍可能共享疾病知识。
- 以自然发生的稳定闭卷错误定义冲突；受控选择也可能反映选项偏好，不能等同于高置信度参数知识。未设置高置信度门槛是有意采用的宽松标准。
- 来源规则被忠实压缩为可执行片段，保留本任务必要人群限制；不覆盖完整指南和所有临床例外。
- 筛选后适用与不适用分母可能不平衡；完整双侧配对另报。样本少时区间宽或经验bootstrap退化，不意味着真实误差概率为零。
- 仅一个Base模型、最后10层、固定强度网格和候选选择；观察不到权衡不等于所有干预强度或自然生成都不存在权衡。
- O2/O3仍是上一轮虚构规则机制实验，本轮未把它们改称临床结果。

## 复现

source /home/gcx/gcx261014/bin/activate
python scripts/build_clinical_pairs.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_clinical_o1.py
python scripts/analyze_clinical_o1.py
python -m pytest -q

完整临床结果目录：results/obs1_clinical。主图：figures/O1_clinical_faithfulness_misuse、O1_clinical_screening和O1_clinical_subset_sensitivity，各有PNG/PDF/SVG。
"""
    (R/'clinical_report.md').write_text(report)
    print(json.dumps({'screening':frozen,'subsets':summaries},ensure_ascii=False),flush=True)
if __name__=='__main__': main()

