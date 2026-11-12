"""Final report from frozen per-record results, encoded as UTF-8."""
import csv,json,datetime
import run as r
from run import ROOT,REPO,read,write
def finish(table,boot,trans):
 def val(s,m,k):
  return next((x['value'] for x in table if x['scope']==s and x['method']==m and x['metric']==k),None)
 def fmt(v,p=False):
  return '不可估计' if v is None else ('%.2f%%'%(v*100) if p else '%.6g'%v)
 def diff(s,a,b,k):
  return next((x for x in boot if x['scope']==s and x['comparison']==a+' minus '+b and x['metric']==k),{})
 def ci(s,a,b,k):
  d=diff(s,a,b,k)
  return '不可估计' if d.get('low') is None else '%+.3f 个百分点，95%% CI [%+.3f, %+.3f]'%(100*d['estimate'],100*d['low'],100*d['high'])
 def sig(s,a,b,k):return (diff(s,a,b,k).get('low') or 0)>0
 scope='cv_pooled_common'
 di=diff(scope,'E4','E1','interference_path_accuracy');dn=diff(scope,'E4','E1','natural_path_accuracy')
 passed=di.get('low') is not None and di['low']>0 and dn.get('low') is not None and dn['low']>=-.005
 out=['# ASTER-CR：10,006 对最终方法实验报告','',
 '本轮完成冻结数据复用、五折独立拟合、M3 中心化、M4 校准、消融及三个数据集评价。完整 E4 '+('满足' if passed else '未同时满足')+'两项预先指定任务标准。训练和校准数据、逐例预测、全部不利结果均保留。',
 '','## 主要结论','',
 '1. **多来源均衡训练尚未证明整体优于单来源 E1。** 五折 E2−E1 干扰路径：'+ci(scope,'E2','E1','interference_path_accuracy')+'；自然路径：'+ci(scope,'E2','E1','natural_path_accuracy')+'。A/P 方差降低率从 '+fmt(val(scope,'E1','ap_vr'),True)+' 升至 '+fmt(val(scope,'E2','ap_vr'),True)+'，但 C/F 超预算比例也从 '+fmt(val(scope,'E1','cf_over_budget_rate'),True)+' 升至 '+fmt(val(scope,'E2','cf_over_budget_rate'),True)+'。分数稳定性改善不等于任务准确率提高。',
 '2. **去掉全局截距改善了部分决策，但没有新增配对抑制能力。** E2/E3 的数据、M1、M2、受约束最优 w 完全相同。五折 E3−E2 干扰路径：'+ci(scope,'E3','E2','interference_path_accuracy')+'。旧实现先拟合 w、再单独拟合 b；删除 b 仅改变绝对分数零点，A/P、C/F 差分指标保持不变。',
 '3. **中心化不改变配对差分和训练约束。** 完整模型和五折逐对不变性、无可学习全局截距、训练 C/F 约束以及训练统计来源检查均通过。训练配对约束通过不代表留出来源也全部满足预算。',
 '4. **MEGA M4 改善概率质量，任务效用收益未建立。** 五折 E3→E4：Brier '+fmt(val(scope,'E3','brier'))+'→'+fmt(val(scope,'E4','brier'))+'，log loss '+fmt(val(scope,'E3','log_loss'))+'→'+fmt(val(scope,'E4','log_loss'))+'。干扰路径差值 '+ci(scope,'E4','E3','interference_path_accuracy')+'；自然路径差值 '+ci(scope,'E4','E3','natural_path_accuracy')+'。',
 '5. **MedPIC 集合概率模型获得点估计改善，但冻结测试区间包含零。** 官方 467 题 E3→E4：'+fmt(val('medpic_official467','E3','question_accuracy'),True)+'→'+fmt(val('medpic_official467','E4','question_accuracy'),True)+'；冻结 188 题测试：'+fmt(val('medpic_test','E3','question_accuracy'),True)+'→'+fmt(val('medpic_test','E4','question_accuracy'),True)+'，差值 '+ci('medpic_test','E4','E3','question_accuracy')+'。467 题包含 93 题校准开发集，只作全量描述性评价。',
 '6. **跨任务迁移分别报告，不合并准确率。** NLI4CT 官方 5,500 条测试和 500 条原声明使用冻结 M1–M3，不做目标校准。E1/E3 原声明宏 F1 为 '+fmt(val('nli4ct_official5500','E1','original_macro_f1'))+'/'+fmt(val('nli4ct_official5500','E3','original_macro_f1'))+'；正式准确率差值 '+ci('nli4ct_official5500','E3','E1','formal_test_accuracy')+'。MedPIC 冻结测试 E3−E1 整题差值 '+ci('medpic_test','E3','E1','question_accuracy')+'。',
 '7. **完整新版尚不能作为整体成功的方法替代 E1。** 五折 E4−E1 干扰路径 '+ci(scope,'E4','E1','interference_path_accuracy')+'；自然路径 '+ci(scope,'E4','E1','natural_path_accuracy')+'。前者区间下限须 >0，后者须 ≥−0.5 个百分点。留出 C/F 方向退化 '+fmt(val(scope,'E3','cf_direction_degradation_count'))+'，超预算 '+fmt(val(scope,'E3','cf_over_budget_rate'),True)+'，双端全对率 '+fmt(val(scope,'E3','cf_paired_accuracy'),True)+'。是否采用新版以这些结果共同判断。',
 '','## MEGA-CDP','',
 '|范围|方法|自然节点|自然路径|干扰路径|A/P RMS|VR|C/F修正RMS|ER|方向退化|超预算|',
 '|---|---|---|---|---|---|---|---|---|---|---|']
 ks=['natural_node_accuracy','natural_path_accuracy','interference_path_accuracy','ap_score_change_rms','ap_vr','cf_correction_rms','cf_er','cf_direction_degradation_count','cf_over_budget_rate']
 percent=set([ks[i] for i in [0,1,2,4,6,8]])
 for sc in ['test','new60',scope]:
  for m in ['E0','E1','E2','E3','E4']:out.append('|'+sc+'|'+m+'|'+'|'.join(fmt(val(sc,m,k),k in percent) for k in ks)+'|')
 out+=['','## NLI4CT-2024 官方冻结迁移','',
 '|方法|原声明宏F1（500）|测试准确率（5,500）|保持一致率（4,136对）|条件翻转忠实性|条件成功数/合格数|翻转双端全对（864对）|','|---|---|---|---|---|---|---|']
 sc='nli4ct_official5500'
 for m in ['E0','E1','E2','E3','E4']:
  out.append('|'+m+'|'+'|'.join(fmt(val(sc,m,k),True) for k in ['original_macro_f1','formal_test_accuracy','native_preserving_relation_accuracy','conditional_flip_faithfulness'])+'|'+fmt(val(sc,m,'conditional_flip_success'))+'/'+fmt(val(sc,m,'conditional_flip_eligible'))+'|'+fmt(val(sc,m,'native_changing_paired_accuracy'),True)+'|')
 out+=['','条件翻转忠实性仅在原声明预测正确的翻转对中计算成功比例；全部翻转对双端全对率固定以 864 对为分母。NLI4CT 以原声明家族为重采样单位，沿用原生标签关系。原 2,864 条迁移子集另存 nli4ct_test，不能替代上述正式评价。E4 在本任务与 E3 相同。',
 '','## MedPIC','',
 '|范围|方法|整题集合正确率|单选|多选|None|集合大小MAE|保持双端全对|翻转双端全对|','|---|---|---|---|---|---|---|---|---|']
 for sc in ['medpic_test','medpic_official467']:
  for m in ['E0','E1','E2','E3','E4']:
   keys=['question_accuracy','single_choice_accuracy','multiple_choice_accuracy','none_accuracy','answer_set_size_mae','native_preserving_paired_accuracy','native_changing_paired_accuracy']
   out.append('|'+sc+'|'+m+'|'+'|'.join(fmt(val(sc,m,k),k!='answer_set_size_mae') for k in keys)+'|')
 out+=['','## MEGA 校准与必要消融','',
 '|配置|开发自然节点|开发自然路径|开发干扰路径|A/P RMS|C/F修正RMS|超预算|','|---|---|---|---|---|---|---|']
 for m in ['E3','rank8','rank32','without_M1','without_M2_projection','without_M3','without_constraints','without_centering']:
  out.append('|'+m+'|'+'|'.join(fmt(val('ablation_dev',m,k),k not in ['ap_score_change_rms','cf_correction_rms']) for k in ['natural_node_accuracy','natural_path_accuracy','interference_path_accuracy','ap_score_change_rms','cf_correction_rms','cf_over_budget_rate'])+'|')
 out+=['','|校准配置（五折）|Brier|Log loss|ECE|自然路径|干扰路径|','|---|---|---|---|---|---|']
 for m in ['M4-0','M4-1','M4-2','M4-3']:
  out.append('|'+m+'|'+'|'.join(fmt(val(scope,m,k),k.endswith('accuracy')) for k in ['brier','log_loss','ece','natural_path_accuracy','interference_path_accuracy'])+'|')
 out+=['',
 'M4-3 为预先指定主配置，正则化固定为 0.01，开发集拟合后阈值固定 0.5。评价数据上的校准截距和斜率只作诊断，不用于预测。MedPIC 温度、None 偏置及集合大小先验由 93 题开发来源五折 NLL 选择，候选得分保存在 medpic_candidate_scores/。',
 'rank=8/16/32 只比较开发集；无 M1 使用固定随机 51 头；无 M2 对照去掉证据正交投影、保留干扰基 U；其余消融定义及测试结果见 configs/ 和 metrics.csv。没有按测试结果重新选择配置。',
 '','## 可复算性与运行记录','',
 'MEGA 以指南来源为重采样单位进行 2,000 次配对自助重采样，保留差值、95% CI、有效抽样次数及错误转移。三个数据集的评价分母分别保存。历史评价集不称为全新独立确认。',
 '主实验耗时约 60.5 分钟，缺失激活修补使用 GPU 0、1，峰值分别为 17.08、17.20 GiB；主实验其余特征均已有缓存，因此当时未启用预备的 5、6 卡。正式 NLI 补齐使用 0、1、5、6，实际时间和显存见 status/nli_gpu_*.json。',
 '已修复报告中文编码，以及 new60 自然节点 original 标记的序列化错误。后者只影响汇总，不改变任何模型分数或判断，原始错误文件保留在 logs/。补齐正式 NLI 仅新增缺失推理，没有重训或目标校准。',
 '所有检查见 status/tests.json，逐例复算入口 code/analysis.py，完整收尾入口 code/finalize.py。代码、配置、来源划分、种子、版本及数据 SHA256 均保留。']
 text='\n'.join(out)+'\n';(ROOT/'REPORT.zh.md').write_text(text,encoding='utf-8')
 summary='本轮完整 E4 未同时满足两项主要任务标准。五折 E4−E1 干扰路径 '+ci(scope,'E4','E1','interference_path_accuracy')+'；自然路径 '+ci(scope,'E4','E1','natural_path_accuracy')+'。E3→E4 的 Brier 为 '+fmt(val(scope,'E3','brier'))+'→'+fmt(val(scope,'E4','brier'))+'，log loss 为 '+fmt(val(scope,'E3','log_loss'))+'→'+fmt(val(scope,'E4','log_loss'))+'。'
 docs=REPO/'docs/ASTER'
 p=docs/'experiments.zh.md';s=p.read_text();start='<!-- FINAL10006-STATUS-START -->';end='<!-- FINAL10006-STATUS-END -->'
 block=start+'\n'+summary+'\n[最终报告](../../reports/aster_cr_final_method_10006_20261112/REPORT.zh.md)。正式 NLI 已覆盖 5,500 条，MedPIC 已覆盖 467 题并单列 188 题冻结测试。\n'+end+'\n\n'
 if start in s:
  a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+block+s[b:]
 else:s=s.replace('## 1. 当前研究范围',block+'## 1. 当前研究范围')
 s=s.replace('通过三个模块定位、估计并修正干扰','通过四个模块定位、估计并修正干扰')
 s=s.replace('| 线性/非线性修正函数、预算记录及独立截距 |','| 无全局截距的修正函数、训练中心化统计量及预算记录 |\n| M4：任务感知校准与决策 | 固定 M1–M3 后估计概率和任务决策 | MEGA 逻辑再校准、MedPIC 合法集合概率 |')
 p.write_text(s,encoding='utf-8')
 p=docs/'manuscript.zh.md';s=p.read_text()
 subsection='### 4.10 冻结 10,006 对最终方法比较（2026-11-12）\n\n'+summary+'\n\nE2/E3 共用同一受约束 w，删除截距与训练中心化保持所有配对差分不变；五折 E3−E2 干扰路径 '+ci(scope,'E3','E2','interference_path_accuracy')+'，属于绝对决策零点变化。E2 相对 E1 的方差抑制提高，但 C/F 超预算比例从 '+fmt(val(scope,'E1','cf_over_budget_rate'),True)+' 增至 '+fmt(val(scope,'E2','cf_over_budget_rate'),True)+'。\n\nMedPIC 冻结 188 题 E3→E4 整题正确率 '+fmt(val('medpic_test','E3','question_accuracy'),True)+'→'+fmt(val('medpic_test','E4','question_accuracy'),True)+'，差值区间包含零。官方 467 题包含开发校准题，只作完整覆盖的描述性评价。NLI4CT 官方 5,500 条结果及全部消融、置信区间和错误转移见[最终报告](../../reports/aster_cr_final_method_10006_20261112/REPORT.zh.md)。\n\n'
 if '### 4.10 ' in s:
  a=s.index('### 4.10 ');b=s.index('## 5 ',a);s=s[:a]+subsection+s[b:]
 else:s=s.replace('## 5 机制分析与消融',subsection+'## 5 机制分析与消融')
 s=s.replace('<!-- FINAL10006-RESULTS -->','')
 p.write_text(s,encoding='utf-8')
 write(ROOT/'completion.json',dict(complete=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_success_criteria=passed,test_results='status/tests.json',report='REPORT.zh.md',metrics=len(table),bootstrap_rows=len(boot),official_nli_count=5500,medpic_count=467,main_run_seconds=3631.339109182358))
