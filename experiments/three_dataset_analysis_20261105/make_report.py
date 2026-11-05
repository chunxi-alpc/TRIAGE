from common import *
def get(name):return json.loads((ROOT/'results'/name).read_text())
def pct(m,counts=False):
 if m['value'] is None:return '— (0/0)'
 s=f"{100*m['value']:.2f}%"
 if counts:s+=f" ({m['numerator']:g}/{m['denominator']:g})"
 return s
def diff(m,scale=100):
 s=f"{scale*m['value']:+.2f}"
 if m.get('ci95') is not None:s+=f" [{scale*m['ci95'][0]:+.2f}, {scale*m['ci95'][1]:+.2f}]"
 else:s+=' [区间不可估计]'
 return s
def table(headers,rr):
 return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rr)+'\n'
def flat(r):
 out={}
 for k,v in r.items():
  if isinstance(v,dict) and 'value' in v:
   for sub,val in v.items():out[k+'_'+sub]=val
  else:out[k]=v
 return out
def main():
 old=get('medpair_deletion.json');new=get('medpair_new_accuracy.json');comp=get('medpair_new_comparisons.json')
 guide=get('medguide_accuracy.json');swaps=get('medguide_swaps.json');einst=get('medeinst_m3a.json');cs=get('medguide_threshold.json')
 fusion=get('medpair_fusion.json')+get('medguide_fusion.json')
 for name,data in [('table1_medpair_deletion',old),('table1_medpair_addback',comp),('table1_medpair_new_accuracy',new),('table2_medguide_accuracy',guide),('table2_medguide_swaps',swaps),('table3_medeinst_fact_response',einst),('table3_medguide_threshold',[cs]),('table4_fusion',fusion)]:
  csv_table(ROOT/'results'/(name+'.csv'),[flat(r) for r in data])
 rr=[]
 for r in old:
  if r['backbone']!='base_final' or r['split']!='test' or r['candidate'] not in (None,'TOP1'):continue
  if r['policy'] not in ('FULL','TOP1','NO_TOP1','LENGTH_MATCHED_RANDOM_K'):continue
  name={'FULL':'Full','TOP1':'High','NO_TOP1':'Low','LENGTH_MATCHED_RANDOM_K':'句数/长度匹配随机子集（题内先平均）'}[r['policy']]
  rr.append([name,r['n'],pct(r['accuracy']),diff(r['A']) if 'A' in r else '—',diff(r['gold_logp_change'],1) if 'gold_logp_change'in r else '—',pct(r['correct_to_wrong'],True) if 'correct_to_wrong'in r else '—'])
 t1='原有 Qwen3-8B base 删除对照；A=配对 Full 准确率−当前条件准确率，单位为百分点。正确答案分数变化为当前条件−Full。随机对照每题先对现有种子平均，其错误计数是题内平均后的有效计数。\n\n'
 t1+=table(['测试条件','题数','准确率','A [95% CI]','正确答案 log p 变化 [95% CI]','Full 对、当前错 / 配对题数'],rr)
 t1+='\n本轮 Full/High 完整选项评分与历史输出分别保存。历史 High 为 56/89，本轮为 55/89；Full 均为 60/89。旧文件没有完整选项分数，无法据此确定差异原因。删除对照沿用历史结果，融合统一使用本轮完整分数。\n\n'
 t1+=table(['本轮条件','测试题数','准确率'],[[r['view'],r['n'],pct(r['accuracy'],True)] for r in new if r['split']=='test'])
 t1+='\n补回均位于两个被删除句子中较早的原位置，其他句子顺序和原文不变。\n\n'
 t1+=table(['比较（前者−后者）','题数','准确率差/百分点 [95% CI]','正确答案 log p 差 [95% CI]'],[[r['first']+' − '+r['second'],r['n'],diff(r['accuracy_difference']),diff(r['gold_logp_difference'],1)] for r in comp if r['split']=='test' and r['first'].startswith('C')])
 q=next(r for r in get('medpair_addback_lengths.json') if r['split']=='test')
 t1+=f"\n测试补回句子的 token 长度绝对差：中位数 {q['absolute_token_gap_median']:g}，均值 {q['absolute_token_gap_mean']:.2f}，范围 {q['absolute_token_gap_min']}–{q['absolute_token_gap_max']}。逐题实际提示长度差已保存。\n"
 t2='方案交换均相对原始 A。开发集 149 例、测试集 651 例，按指南树分组。原始 A 与 B/C/D 的遮挡输入及分数完全一致，遮挡预测改变率均为 0；基础错误率只计 A 一次。\n\n'
 for split in ('test','all'):
  rr=[]
  for protocol in ('all','PR23_strict','extended_offpath'):
   if split=='all' and protocol=='all':continue
   for v in ('A','B','C','D'):
    a=next(r for r in guide if r['split']==split and r['protocol']==protocol and r['variant']==v and r['view']=='visible')
    b=next(r for r in guide if r['split']==split and r['protocol']==protocol and r['variant']==v and r['view']=='hidden')
    s=next((r for r in swaps if r['split']==split and r['protocol']==protocol and r['variant']==v and r['view']=='visible'),None)
    rr.append([protocol,v,a['n'],pct(a['accuracy'],True),pct(b['accuracy'],True),pct(s['changed'],True) if s else '—',pct(s['correct_to_wrong'],True) if s else '—',pct(s['wrong_to_correct'],True) if s else '—'])
  t2+=('测试集：\n\n' if split=='test' else '\n全体原协议 304 例、扩展构造 496 例的描述性结果（不用于测试集选参）：\n\n')
  t2+=table(['来源','变体','n','可见准确率','遮挡准确率','可见预测改变','可见正确→错误','可见错误→正确'],rr)
 t2+='\n95% 配对区间按指南树重采样，见 table2_medguide_swaps.csv。自由生成答案的正确率另存于 table2_medguide_accuracy.csv 的 generated_accuracy 列，不与最高分预测混用。开发集的 3 个缺失 D 变体不补造。\n'
 er=[r for r in einst if r['split']=='fresh_test' and r['score']=='logprob_sum']
 rr=[[r['model'],r['n'],pct(r['accuracy_a'],True),pct(r['accuracy_b'],True),pct(r['P_b'],True),pct(r['unchanged'],True),pct(r['delta_positive'],True),f"{r['delta_median']:+.3f}"] for r in er]
 rr.append(['MedGUIDE 规则阈值变化（遮挡）',84,pct(cs['accuracy_a'],True),pct(cs['accuracy_b'],True),pct(cs['P_b'],True),pct(cs['unchanged'],True),pct(cs['delta_positive'],True),f"{cs['delta_median']:+.3f}"])
 t3='MedEinst 使用每侧独立的 46 候选序列总 log p 最高分预测，不使用强制两侧诊断不同的 predicted_transition。P_b 为任一侧错误的病例对比例；Δ 为变化后“新正确诊断相对原诊断”的分数支持增量。\n\n'
 t3+=table(['模型/实验','病例对','原侧准确率','变化侧准确率','P_b','预测不变','Δ>0','Δ 中位数'],rr)
 t3+='\n'+table(['模型/实验','全部单例错误','原侧对、变化侧坚持原诊断 / 所有对','方向改善但变化侧仍错','未向正确方向变化'],[[r.get('model','MedGUIDE 阈值变化'),pct(r['single_error'],True),pct(r['correct_then_persist_old'],True),pct(r['toward_but_changed_case_wrong'],True),pct(r['not_toward'],True)] for r in er+[cs]])
 t3+='\n84 组规则阈值变化全部来自 1 棵指南树，患者文本不变、规则阈值及正确分支改变。跨指南树的置信区间无法估计，以空值保存，不报告零宽区间。三个模型均保留原 512/256/256 划分。M1、Work3 和候选 token 均值评分单列补充。\n'
 wc=get('medguide_threshold_within_tree_supplement.json')
 t3+='\n同一来源内按 84 个病例对重采样的补充区间，仅反映该指南树内的不确定性：\n\n'
 t3+=table(['比较（变化后−变化前）','差值 [95% CI]'],[['准确率 / 百分点',diff(wc['accuracy_b_minus_a'])],['新正确分支 log p',diff(wc['new_gold_logp_change'],1)],['原正确分支 log p',diff(wc['old_gold_logp_change'],1)]])
 t4='仅在开发集选参并固定。MedGUIDE 先对同病例的变体平均，再对病例平均；两个数据集分别拟合。损害率分母为至少一路正确的样本；双错纠正率分母为两路均错误的样本。分母为零时记为 —。\n\n'
 rr=[]
 for r in fusion:
  if r['method'] in ('Full','Masked'):continue
  param=r['parameter'];param='/'.join(f'{v:.4f}' for v in param) if isinstance(param,list) else str(param)
  rr.append([r['dataset']+'/'+r['variant'],r['method'],param,r['n'],pct(r['accuracy'],True),pct(r['B'],True),pct(r['corrected'],True),pct(r['same_wrong_corrected'],True),pct(r['different_wrong_corrected'],True),pct(r['harm'],True)])
 t4+=table(['数据/测试变体','方法','参数','n','融合准确率','共同错误 B','双错纠正','同答案双错纠正','异答案双错纠正','损害率'],rr)
 t4+='\n概率平均参数为 Full 权重；CAD 参数为 α；学习式参数依次为 Full、Masked 的共享权重，允许为负。Full、Masked 基线、准确率相对两路的配对差值区间、同/异错误的占比，以及损害数/全部测试样本，见 table4_fusion.csv。CS 只有一棵来源树，跨树区间留空。\n'
 for i,s in enumerate([t1,t2,t3,t4],1):(ROOT/f'results/table{i}.md').write_text(s,encoding='utf-8')
 ah=next(r for r in old if r['backbone']=='base_final' and r['split']=='test' and r['policy']=='TOP1')
 al=next(r for r in old if r['backbone']=='base_final' and r['split']=='test' and r['policy']=='NO_TOP1')
 rec=next(r for r in comp if r['split']=='test' and r['first']=='C+h' and r['second']=='C')
 adv=next(r for r in comp if r['split']=='test' and r['first']=='C+h' and r['second']=='C+u')
 lf=next(r for r in fusion if r['dataset']=='MedGUIDE' and r['variant']=='A' and r['method']=='learned')
 pc=next(r for r in fusion if r['dataset']=='MedGUIDE' and r['variant']=='C' and r['method']=='probability_average')
 timings=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('scoring_*_status.json'))]
 s='# 三数据集补充分析\n\n自动预实验，复用已有标注和历史结果；零新增标注。本轮为对既有数据的补充分析，不声称测试集此前未使用。真实指南 Q% 暂不填。\n\n'
 s+=f"MedPAIR 测试删除对照：A_High={diff(ah['A'])} 个百分点（n=89），A_Low={diff(al['A'])} 个百分点（n=87）。高相关句补回恢复量为 {diff(rec['accuracy_difference'])} 个百分点；相对不相关句补回差为 {diff(adv['accuracy_difference'])} 个百分点（n=29）。补回收益较小、区间包含零，尚不足以支持稳定优势。相关性标签不等同于语义依赖标签。\n\n"
 s+='MedGUIDE 方案交换的影响随变体而异：B、D 降低可见条件的准确率，C 提高准确率；遮挡后对应输入与分数不变。因此可支持方案文字影响条件判断，不能概括为所有方案文字都造成损害。MedEinst 三个模型的测试配对失误率为 96.88%–99.61%，预测不变率为 62.89%–66.80%；部分病例分数已经向正确方向变化，但仍未选对。这些结果不用于归因到特定训练阶段。\n\n'
 s+=f"融合可以纠正部分共同错误。MedGUIDE 原始 A 的学习式融合准确率为 {pct(lf['accuracy'],True)}，纠正双错 {pct(lf['corrected'],True)}，同时对至少一路正确的样本造成错误 {pct(lf['harm'],True)}；相对完整输入准确率变化为 {diff(lf['accuracy_minus_full'])} 个百分点。C 变体的概率平均也纠正了 {pc['corrected']['numerator']:g}/{pc['corrected']['denominator']:g} 个共同错误。MedPAIR 本轮四种选定融合均未纠正 28 个共同错误；融合效果依数据和变体而变。\n\n"
 s+='历史 MedPAIR High 为 56/89，本轮完整选项重评分为 55/89；Full 均为 60/89。两批结果分别呈现：历史删除对照不变，融合统一使用本轮分数；不推测差异原因。84 组规则阈值变化均来自同一棵指南树，只报告该来源结果，不估计跨树区间。\n\n'
 s+='所有实验均在 115 的 gcx261014 环境运行，使用 GPU 0、1。MedPAIR 共 540 条条件记录、538 个唯一输入；20 道开发题的 55 次试跑评分已计入。其余分析完全离线完成，GPU 进程已正常退出。\n\n'
 s+=table(['阶段','GPU','完成评分','含加载秒数','峰值预留显存/GiB'],[[r['stage'],r['gpu'],r['done'],f"{r.get('elapsed_seconds',0):.2f}",f"{r.get('peak_reserved_gib',0):.2f}"] for r in timings])
 s+='\n四张主表：\n\n- [表1：MedPAIR 删除与补回](table1.md)\n- [表2：MedGUIDE 方案交换](table2.md)\n- [表3：事实响应与规则阈值变化](table3.md)\n- [表4：输出融合](table4.md)\n\n'
 s+='服务器结果目录：/home/gcx/three_dataset_analysis_20261105/results。代码入口：/home/gcx/work3-migration-20261104/target/experiments/three_dataset_analysis_20261105。数据入口：/home/gcx/experiment_data_reuse_20261105/data_inventory.json。\n\n'
 s+='逐条输入、来源、预测和完整分数保存在 medpair_complete_views.jsonl、../data/medguide_scores_and_inputs.jsonl、medeinst_independent_predictions.jsonl；固定划分见 ../data/medguide_split.json 和各样本 split 字段；融合参数见 medpair_fusion_parameters.json、medguide_fusion_parameters.json。\n\n'
 s+='统计：种子 42、2,000 次配对 bootstrap。MedPAIR 按题、MedEinst 按病例对、MedGUIDE 按指南树重采样。随机删除对照在题内先平均。概率只在有效选项内归一化。四项融合测试及样本配对、划分和概率检查已通过，见 verification.json。\n\n'
 s+='旧内部干预、PCD、CEBU、自回归 CAD、M1 与 Work3 结果分别保存为补充文件，未训练内部模块，未合并不同协议。\n'
 (ROOT/'results/REPORT.zh.md').write_text(s,encoding='utf-8')
 write_json(ROOT/'results/completion.json',dict(status='complete',automatic_preexperiment=True,new_annotations=0,unique_gpu_inputs=538,view_records=540,bootstrap_seed=42,bootstrap_samples=2000,report=str(ROOT/'results/REPORT.zh.md')))
if __name__=='__main__':main()
