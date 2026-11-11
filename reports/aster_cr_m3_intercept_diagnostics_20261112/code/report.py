import csv,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
def rd(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def js(n):return json.loads((OUT/n).read_text())
M=rd('metrics.csv');B=rd('source_bootstrap.csv');T=rd('transitions.csv');cal=js('configs/intercepts.json');dc=js('configs/medpic_decoder_calibration.json');selected=dc['selected_decoder']
def m(s,n):return next(r for r in M if r['scope']==s and r['method']==n)
def b(s,c,k):return next(r for r in B if r['scope']==s and r['comparison']==c and r['metric']==k)
def pct(v):return '—' if v in [None,''] else f'{float(v)*100:.2f}%'
def num(v):return '—' if v in [None,''] else f'{float(v):.4f}'
def ci(s,c,k,percent=True):
 r=b(s,c,k);mult=100 if percent else 1;unit='个百分点' if percent else '';return f"{float(r['estimate'])*mult:+.3f}{unit} [{float(r['low'])*mult:+.3f}, {float(r['high'])*mult:+.3f}]（有效{r['defined_draws']}次）"
def table(head,body):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in body])
def transition(s,c,k):
 r=next(r for r in T if r['scope']==s and r['comparison']==c and r['level']==k);return f"{r['wrong_to_right']} / {r['right_to_wrong']}"
parts=['# ASTER-CR M3 截距校准与多选答案解码诊断','## 主要结论']
parts.append('本轮冻结 E1，仅改变截距或解码，不替换论文主方法。所有区间均按来源配对自助重采样2,000次计算。')
parts.append('1. **配对修正尚未提供超过简单截距的稳定正确率收益。** 五折合并 full−independent_intercept 的自然节点差值为 '+ci('cv_pooled_common','full minus independent_intercept','natural_node_accuracy')+'，自然路径为 '+ci('cv_pooled_common','full minus independent_intercept','natural_path_accuracy')+'，干扰路径为 '+ci('cv_pooled_common','full minus independent_intercept','interference_path_accuracy')+'。这些区间没有严格排除零。')
parts.append('2. **MedPIC 保持退化主要发生在加入截距时。** 固定配对修正，只加入原截距，保持配对全对率变化为 '+ci('medpic_test','full minus pair_only','native_preserving_paired_correct')+'；保持关系正确率变化为 '+ci('medpic_test','full minus pair_only','native_preserving_relation_correct')+'。候选集合分数增加 k·b，直接解释了不同大小集合排序变化。')
parts.append('3. **开发集选中的集合级校准减轻保持损失，但没有稳定整题收益。** 选中方案为 '+selected+'，其测试保持全对率为74.12%，相对原full变化为 '+ci('medpic_test',selected+' minus full','native_preserving_paired_correct')+'；整题正确率变化为 '+ci('medpic_test',selected+' minus full','question_accuracy/all')+'。翻转配对全对率变化为 '+ci('medpic_test',selected+' minus full','native_changing_paired_correct')+'，存在明显取舍。')
parts.append('4. **未证明同时满足干扰抑制、证据保护与自然任务非劣性。** 配对分数稳定性有改善，但验证来源仍有C/F超预算，解码校准也未得到稳定自然任务收益。计划未给定非劣界值，本轮没有事后补设，不把区间含零写成非劣。')
parts += ['## 固定条件与校准','E1 / Dataset A 2,540对；Qwen3-8B BF16；原提示、分词和最终token评分不变；51个头、16维子空间、lambda=0.01、delta=0.2923349。复用原头、子空间、配对参数、分数和缓存特征。没有模型训练或GPU推理。', '独立b0复用前轮自然节点独立拟合结果：全量761节点/80来源，每折仅64个训练来源。缩放与b_dev只在原20来源开发集上估计，每折使用自己的冻结模型计算开发分数。MedPIC保持原93题开发集、188题测试集以及原来源分组。所有校准与选择均先写入配置，再评价测试。', '历史40来源、已有60来源、MedPIC测试均已有使用记录，本轮为冻结数据诊断复评，不是新独立确认。单选/多选按金标准答案大小分层；原题允许多选，没有改变合法答案约束。']
parts.append(table(['模型','原b','独立b0','开发选中缩放','开发重拟合b'],[[f,num(c['b']),num(c['b0']),c['selected_scale'],num(c['b_dev'])] for f,c in cal.items()]))
parts += ['## 实验A：收益分离','raw=l；pair_only=l+c；original_intercept_only=l+b；independent_intercept=l+b0；full=l+c+b。']
for scope,title in [('dev','20来源开发集'),('test','历史40来源测试'),('new60','已有60来源'),('cv_pooled_common','五折合并80来源')]:
 parts.append('### '+title)
 parts.append(table(['方法','自然节点','自然路径','干扰路径','A/P RMS','VR','C/F修正RMS','ER','方向退化','超预算'],[[n,pct(m(scope,n).get('natural_node_accuracy')),pct(m(scope,n).get('natural_path_accuracy')),pct(m(scope,n).get('interference_path_accuracy')),num(m(scope,n).get('ap_score_change_rms')),pct(m(scope,n).get('ap_vr')),num(m(scope,n).get('cf_correction_rms')),pct(m(scope,n).get('cf_er')),m(scope,n).get('cf_direction_degradation_count'),pct(m(scope,n).get('cf_over_budget_rate'))] for n in ['raw','pair_only','original_intercept_only','independent_intercept','full']]))
parts.append('### 五折主要差值与错误转移')
parts.append(table(['比较','指标','差值与95%区间','改对/改错'],[[c,k,ci('cv_pooled_common',c,k),transition('cv_pooled_common',c,k)] for c in ['pair_only minus raw','pair_only minus independent_intercept','full minus independent_intercept','full minus pair_only'] for k in ['natural_node_accuracy','natural_path_accuracy','interference_path_accuracy']]))
parts.append('五折pair_only−raw的A/P RMS变化为 '+ci('cv_pooled_common','pair_only minus raw','ap_score_change_rms',False)+'，VR为 '+ci('cv_pooled_common','pair_only minus raw','ap_vr')+'。C/F超预算 '+m('cv_pooled_common','pair_only')['cf_over_budget_count']+'/'+m('cv_pooled_common','pair_only')['cf_pairs']+'，方向退化 '+m('cv_pooled_common','pair_only')['cf_direction_degradation_count']+'。完整比较区间见source_bootstrap.csv。')
parts += ['## 实验B：截距敏感性','缩放固定为0/0.25/0.5/0.75/1/1.25/1.5，选择目标是开发自然节点平均逻辑损失；b_dev最小化同一目标。表中报告全部预设候选，不根据测试选择。pair_plus_b0在固定配对修正上加入独立原分数截距，仅作消融。']
for scope in ['dev','test','cv_pooled_common']:
 parts.append('### '+scope)
 names=['pair_only','full','pair_plus_b0']+['scale_'+str(s) for s in [0.25,0.5,0.75,1.25,1.5]]+['dev_selected_scale','dev_refit']
 parts.append(table(['方法','自然节点','自然路径','干扰路径','自然路径改对/改错','干扰路径改对/改错'],[[n,pct(m(scope,n)['natural_node_accuracy']),pct(m(scope,n)['natural_path_accuracy']),pct(m(scope,n)['interference_path_accuracy']),transition(scope,n+' minus raw','natural_path_accuracy'),transition(scope,n+' minus raw','interference_path_accuracy')] for n in names]))
parts.append('错误转移均相对raw。对配对两端加入同一截距，分数差严格不变；因此固定c的所有截距版本有相同A/P RMS、VR、C/F修正RMS、ER、方向退化与超预算。证据指标完整存表，数值与实验A的pair_only相同。仅调整截距不会改善这些证据指标。每折使用其自身参数，五折合并后重新计算，不平均比率。')
parts += ['## 实验C：MedPIC解码','存在“以上均非”时，原解码比较该单独选项与其他选项的所有非空子集；不存在时沿用逐选项正分选择，等价于所有子集（含空集）求和最大化。保留原候选顺序。',table(['方案','候选得分','参数来源'],[['A/full','sum(l+c)+k*b','原E1'],['B/pair_only','sum(l+c)','无截距'],['C/C_mean_full','sum(l+c+b)/k；空集0','固定归一化'],['D/D_none_calibrated','sum(l+c)+eta*I(以上均非)','开发集合交叉熵'],['E/E_size_calibrated','sum(l+c)+tau*k','开发集合交叉熵']]), '向所有候选集合加同一个常数不改变排序，因此D采用一次集合类别偏移，而不是逐选项重复加截距。D/E各只有一个参数：eta='+num(dc['eta'])+'，tau='+num(dc['tau'])+'。开发精确集合正确率选择方案，同分按B/C/D/E/A顺序。']
parts.append(table(['方案','开发整题正确率','开发集合NLL','选中'],[[n,pct(dc['development_accuracy'][n]),num(dc['development_set_nll'][n]),str(n==selected)] for n in dc['development_accuracy']]))
parts.append('### 冻结测试结果')
names=['raw','pair_only','original_intercept_only','independent_intercept','full','C_mean_full','D_none_calibrated','E_size_calibrated']
parts.append(table(['方法','整题正确率','保持全对率','保持关系正确率','翻转全对率','翻转关系正确率'],[[n,pct(m('medpic_test',n)['question_accuracy/all']),pct(m('medpic_test',n)['native_preserving_paired_correct']),pct(m('medpic_test',n)['native_preserving_relation_correct']),pct(m('medpic_test',n)['native_changing_paired_correct']),pct(m('medpic_test',n)['native_changing_relation_correct'])] for n in names]))
parts.append('85/110是原选项级保持/翻转配对数，不是整题数。全对率要求两端选项判断都正确；关系正确率只要求保持/翻转关系正确。历史报告开头83.53%/90.59%/69.41%属于关系正确率，本计划77.65%/78.82%/63.53%属于配对全对率，现已按口径分别复现。')
parts.append(table(['比较','指标','差值与95%区间','改对/改错'],[[c,k,ci('medpic_test',c,k),transition('medpic_test',c,k)] for c in ['full minus pair_only',selected+' minus full',selected+' minus raw',selected+' minus pair_only'] for k in ['question_accuracy/all','native_preserving_paired_correct','native_changing_paired_correct']]))
parts.append('数量归一化C的保持全对率为84.71%，但任何集合平均分都不超过其中最高单选项分数，而该单选项也是合法候选。这会使C选择单元素集合（没有以上均非时还可能选空集）。本次46道金标准多选题正确率为0/46。C未被开发集选中，不能依据测试保持分数将其改为选中方案。')
parts.append('### 题型分层')
strata=['single','multiple','gold_none','gold_substantive','has_none','no_none']
parts.append(table(['分层','题数']+names,[[s,m('medpic_test','raw').get('question_accuracy/'+s+'_n','0')]+[pct(m('medpic_test',n).get('question_accuracy/'+s)) for n in names] for s in strata]))
parts.append('single/multiple按金标准答案大小定义；gold_none表示答案为以上均非；has_none表示候选中有以上均非。本测试所有题均有以上均非，因此no_none分层不可估计。按金标准大小、题型、以上均非分层的全部配对来源区间与错误转移见source_bootstrap.csv和transitions.csv；原选项配对另按两端题型与以上均非分层。候选与预测大小分布见size_distribution_test.csv。')
parts.append('### 整题配对补充')
parts.append(table(['方法','保持整题配对数','保持整题全对率','翻转整题配对数','翻转整题全对率'],[[n,m('medpic_test',n).get('question_preserving_paired_correct_n'),pct(m('medpic_test',n).get('question_preserving_paired_correct')),m('medpic_test',n).get('question_changing_paired_correct_n'),pct(m('medpic_test',n).get('question_changing_paired_correct'))] for n in names]))
parts.append('### 截距与集合排序')
rank=rd('medpic_candidate_scores/ranking_changes_test.csv')
parts.append(table(['范围','题目版本数','答案改变','集合大小改变','以上均非状态改变'],[[label,len(rr),sum(r['changed']=='True' for r in rr),sum(r['k_before']!=r['k_after'] for r in rr),sum(r['none_before']!=r['none_after'] for r in rr)] for label,rr in [('自然题',[r for r in rank if r['variant']=='full']),('含背景变体',rank)]]))
parts.append('原b='+num(cal['full']['b'])+'。每个集合full−pair_only=k*b，保存9,024条测试候选记录，浮点最大误差约5.3e−15。同大小集合的相对排序不变。以上均非只增加b，其他k元素集合增加k*b，正截距使大集合获得相对优势。全部候选得分和排名见test.jsonl.gz，以上均非与最优其他集合的边际变化见none_competition_test.csv。')
changes=rd('medpic_candidate_scores/native_pair_changes_test.csv')
parts.append(table(['选项关系','配对数','至少一端整题答案改变','至少一端选项判断改变'],[[g,len(rr),sum(r['left_answer_changed']=='True' or r['right_answer_changed']=='True' for r in rr),sum(r['left_option_changed']=='True' or r['right_option_changed']=='True' for r in rr)] for g in ['preserving','changing'] for rr in [[r for r in changes if r['option_relation']==g]]]))
parts += ['## 指标与统计口径','A/P RMS为调整后配对分数差的均方根；VR=1−Var(调整后差)/Var(原差)。C/F修正RMS只度量修正项之差。ER=mean((y右−y左)*调整后差)/mean((y右−y左)*原差)。方向退化为原方向正确而修正后不正确；超预算为修正差绝对值大于delta+1e−9。准确率分母只含有标签记录；完整路径按预测自主分支回放。', '每个评价范围按来源抽样，同一组权重用于所有方法和差值；2,000次百分位区间。稀少分层中没有分母的抽样标为无效，defined_draws逐行保存。五折合并逐例数据再估计。全部预设消融均报告，区间未作多重比较校正，不从测试结果重新挑选方案。', 'MedPIC解码C/D/E的节点scores记录底层选项分数，真正集合得分在medpic_candidate_scores。源任务c没有因解码改动而改变。原生保持/翻转不等同A/P、C/F，也不能以解码改善宣称证据约束改善。', '## 交付与复算','全部151行方法结果、31,632行来源重采样结果和6,076行错误转移已从保存逐例预测重新计算。历史解码的2,820条方法×题目版本预测全部复现。主要分析实际用时约144秒；CPU执行，GPU用量为0。README提供复算入口，figures提供PNG/PDF图。原E1参数与历史报告未改写。']
(OUT/'REPORT.zh.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

# Standard exportable figures.

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(14,4))
plotnames=['raw','pair_only','full','C_mean_full','D_none_calibrated','E_size_calibrated'];labels=['Raw','Pair only','Full','Mean','None cal.','Size cal.']
for ax,k,title in zip(axs,['question_accuracy/all','native_preserving_paired_correct','native_changing_paired_correct'],['Natural exact-set accuracy','Preserving option pairs: both correct','Changing option pairs: both correct']):
 ys=[100*float(m('medpic_test',n)[k]) for n in plotnames];ax.bar(range(len(ys)),ys,color=['#888888','#3B7EA1','#C45A49','#A888B5','#3D987A','#D4A341']);ax.set_xticks(range(len(ys)),labels,rotation=35,ha='right');ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,100)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('medpic_decoders.'+ext),dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));ks=['natural_node_accuracy','natural_path_accuracy','interference_path_accuracy'];xx=np.arange(3)
for i,c in enumerate(['pair_only minus raw','full minus independent_intercept','full minus pair_only']):
 rr=[b('cv_pooled_common',c,k) for k in ks];yy=np.array([float(z['estimate'])*100 for z in rr]);lo=np.array([float(z['low'])*100 for z in rr]);hi=np.array([float(z['high'])*100 for z in rr]);ax.errorbar(xx+(i-1)*.14,yy,yerr=np.vstack([yy-lo,hi-yy]),fmt='o',capsize=4,label=c)
ax.axhline(0,color='gray',lw=1);ax.set_xticks(xx,['Natural node','Natural path','Interference path']);ax.set_ylabel('Difference (percentage points), 95% source CI');ax.legend(fontsize=8);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('source_effects.'+ext),dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4));scales=[0,0.25,0.5,0.75,1,1.25,1.5]
for ax,scope in zip(axs,['dev','test']):
 for k in ['natural_node_accuracy','natural_path_accuracy','interference_path_accuracy']:ax.plot(scales,[100*float(m(scope,'scale_'+str(s))[k]) for s in scales],marker='o',label=k.replace('_accuracy',''))
 ax.axvline(cal['full']['selected_scale'],color='gray',ls='--');ax.set_title(scope);ax.set_xlabel('Original intercept scale');ax.set_ylabel('Accuracy (%)');ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('intercept_sensitivity.'+ext),dpi=180)
plt.close(fig)
print('Report and figures created.')

# Rebuild none-versus-other margins from saved candidate predictions.
from collections import defaultdict
with gzip.open(OUT/'medpic_candidate_scores/test.jsonl.gz','rt',encoding='utf-8') as h:cc=[json.loads(x) for x in h]
groups=defaultdict(list)
for c in cc:groups[(c['item_id'],c['variant'])].append(c)
records=[]
for key,cs in groups.items():
 nn=[c for c in cs if c['none']];other=[c for c in cs if not c['none']]
 if not nn or not other:continue
 n=nn[0];a=max(other,key=lambda c:c['scores']['pair_only']);b0=max(other,key=lambda c:c['scores']['full'])
 records.append(dict(item_id=key[0],variant=key[1],source_id=n['source_id'],none_candidate=n['candidate'],best_other_before=a['candidate'],best_other_after=b0['candidate'],margin_before=n['scores']['pair_only']-a['scores']['pair_only'],margin_after=n['scores']['full']-b0['scores']['full'],before_k=a['k'],after_k=b0['k']))
with (OUT/'medpic_candidate_scores/none_competition_test.csv').open('w',encoding='utf-8-sig',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
