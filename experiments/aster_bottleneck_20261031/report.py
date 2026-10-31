"""Generate aggregate, paired results without publishing private case text."""
import json,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from .run import OUT,read,write
ROOT=Path(__file__).resolve().parents[2]
PUBLIC=Path(__file__).parent/'public'
def load(name,default=None):
 p=OUT/name;return json.loads(p.read_text()) if p.exists() else default
def stage_rows(stage):
 rs=[]
 for i in range(4):
  p=OUT/f'{stage}_{i}.jsonl'
  if p.exists():rs+=read(p)
 return list({(r.get('case_id'),r.get('arm'),r.get('order'),r.get('state'),r.get('pair_id'),r.get('candidate')):r for r in rs}.values())
def paired(rows,baseline_rows,metric='exact_path'):
 baseline={r['case_id']:r for r in baseline_rows};groups=defaultdict(list)
 for r in rows:
  if r['case_id'] not in baseline:continue
  a=bool(r.get('score',{}).get(metric,False));b=bool(baseline[r['case_id']].get('score',{}).get(metric,False))
  groups[r['source_id']].append((int(a)-int(b),a,b))
 flat=[v for vs in groups.values() for v in vs];n=len(flat);ci=None
 if len(groups)>=10:
  rng=np.random.default_rng(20261031);means=np.array([np.mean([x[0] for x in vs]) for vs in groups.values()])
  boot=means[rng.integers(0,len(means),size=(2000,len(means)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
 return {'cases':n,'sources':len(groups),'corrected':sum(a and not b for d,a,b in flat),'regressed':sum(b and not a for d,a,b in flat),
  'case_delta':sum(d for d,a,b in flat)/n if n else None,'source_equal_delta':float(np.mean([np.mean([x[0] for x in vs]) for vs in groups.values()])) if groups else None,
  'source_equal_95ci':ci}
def table(rows):
 lines=['| 条件 | 例数 | 完整路径正确 | 终点正确 | 失败 |','|---|---:|---:|---:|---:|']
 if isinstance(rows,dict):rows=[{'arm':k,**v} for k,v in rows.items()]
 for r in rows:lines.append(f'| {r["arm"]} | {r["n"]} | {r["path"]} | {r["leaf"]} | {r["failed"]} |')
 return '\n'.join(lines)
def build_report():
 PUBLIC.mkdir(exist_ok=True)
 manifest=load('manifest.json');core=load('core_summary.json');validation=load('validation_summary.json');advanced=load('advanced_status.json',{'status':'pending'})
 report={'manifest':manifest,'initial_status':load('execution_status.json'),'advanced_status':advanced,'core':core,'validation':validation}
 text=['# ASTER 三模块瓶颈实验报告','',f'初轮状态：{report["initial_status"]["status"]}；后续阶段状态：{advanced["status"]}。','',
 '43例、2来源为已暴露开发诊断；120例、12新来源用于验证，根分支74∶46。所有金标路线结果仅用于算子诊断。',
 '主线保留方向学习、处置盲评估和键编辑；均值差默认，谱为对照。']
 if core:text+=['','## 1. 开发集算子对照','',table(core['arms'])]
 if validation:text+=['','## 2. 新来源算子验证','',table(validation['arms'])]
 initial_rows=stage_rows('core');zero={(r['case_id'],t['layer'],t['kv_head']):t for r in initial_rows if r['arm']=='zero' for t in r.get('result',{}).get('trace',[])}
 p0={}
 for arm in ['spectral025','mean025']:
  ts=[(r,t) for r in initial_rows if r['arm']==arm for t in r.get('result',{}).get('trace',[]) if 'delta_sign_fraction' in t]
  ratios=[.25*t['query_norm_over_sqrt_dim']/max(zero[r['case_id'],t['layer'],t['kv_head']]['score_std'],1e-12) for r,t in ts]
  p0[arm]={'case_head_observations':len(ts),'median_bound_over_zero_score_std':float(np.median(ratios)) if ratios else None,
   'mean_sign_consistency':float(np.mean([t['delta_sign_fraction'] for _,t in ts])) if ts else None,
   'mean_abs_actual_delta':float(np.mean([t['delta_abs_mean'] for _,t in ts])) if ts else None}
 report['p0']=p0;text+=['','## 3. P0连续诊断','', '下列比值仅比较同一注意力打分空间，不能与输出logit边际直接比较。词元和头不作为独立样本。','',
 '| 条件 | 位移上界／零编辑打分标准差中位数 | 预期符号比例均值 | 实际绝对位移均值 |','|---|---:|---:|---:|']
 for arm,v in p0.items():
  if v['median_bound_over_zero_score_std'] is not None:text.append(f'| {arm} | {v["median_bound_over_zero_score_std"]:.4f} | {v["mean_sign_consistency"]:.4f} | {v["mean_abs_actual_delta"]:.6f} |')
 for stage,title in [('windows','层窗口与选择性重算'),('subsets','头覆盖'),('key_controls','开发集匹配位移键对照'),('key_validation','新来源匹配位移键对照'),('behavior_validation','新来源行为目标方向')]:
  result=load(stage+'_summary.json')
  if result:report[stage]=result;text+=['',f'## {title}','',table(result)]
 routing=load('routing_summary.json')
 if routing:
  report['routing']=routing;text+=['','## 模块二','',f'新100例开发集选择：{routing["selected"]}。新来源不参与选择。','',
   '| 方法 | 新100例开发完整路径 | 新来源完整路径 | 新来源接受节点 | 新来源错误接受节点 |','|---|---:|---:|---:|---:|']
  for name,d in routing.get('development100',routing['development']).items():
   v=routing['validation'][name];text.append(f'| {name} | {d["path"]}/{d["cases"]} | {v["path"]}/{v["cases"]} | {v["accepted"]} | {v["wrong_accepted"]} |')
  text+=['', '模块二直接读取原始病历与规则条件，不抽取结构化事实、不拆原子条件。只比较原分类器、NONE校正及直接T/F/U判断；停止门控后保留已接受祖先。']
 for stage,title in [('predicted_dev','开发预测路径执行'),('predicted_validation','新来源预测路径执行')]:
  result=load(stage+'_summary.json')
  if result:report[stage]=result;text+=['',f'## {title}','',table(result)]
 pairings={}
 valzero=[r for r in stage_rows('validation') if r['arm']=='zero']
 for stage in ['validation','key_validation','behavior_validation','predicted_validation']:
  rr=stage_rows(stage)
  for arm in sorted({r['arm'] for r in rr}):
   pairings[stage+'/'+arm]=paired([r for r in rr if r['arm']==arm],valzero)
 report['paired_new_validation']=pairings
 text+=['','## 新来源配对变化','', '| 条件 | 纠正 | 退化 | 来源等权差 | 来源重采样95%区间 |','|---|---:|---:|---:|---|']
 for arm,v in pairings.items():
  if v['source_equal_delta'] is not None:text.append(f'| {arm} | {v["corrected"]} | {v["regressed"]} | {v["source_equal_delta"]:.4f} | {v["source_equal_95ci"]} |')
 proof=load('proof_summary.json')
 if proof:
  report['proof']=proof;text+=['','## 充分性与必要性','', '仅为两个已有年龄来源的机制检验；不是一般临床机制证明。LEACE使用原拟合来源的加权token协方差，随机擦除匹配秩。','',
   '| 顺序 | 状态 | 干预 | 例数 | 根判断正确 | 目标分支被选 | 目标边际 |','|---|---|---:|---:|---:|---:|']
  for r in proof['rows']:text.append(f'| {r["order"]} | {r["state"]} | {r["arm"]} | {r["n"]} | {r["root_correct"]} | {r["target_selected"]} | {r["mean_target_margin"]:.4f} |')
 behavior=load('behavior_summary.json')
 if behavior:
  report['behavior']=behavior;text+=['','行为方向只在原拟合来源上，在均值、低频均值和固定随机方向的七种组合中优化双向决策目标，并加初始输出位置KL。它是有限搜索实验，不称为全向量梯度训练或BiPO复现。']
 curves={}
 for stage in ['errors_dev','errors_validation']:
  rs=stage_rows(stage);groups=defaultdict(list)
  for r in rs:groups[(r['spec']['epsilon'],r['spec']['error_type'],r['spec']['kind'])].append(r)
  curve=[]
  for (eps,error,kind),vs in sorted(groups.items()):
   curve.append({'epsilon':eps,'error':error,'operator':kind,'records':len(vs),'cases':len({r['case_id'] for r in vs}),
    'path_accuracy':sum(r['score']['exact_path'] for r in vs)/len(vs),'leaf_accuracy':sum(r['score']['leaf_correct'] for r in vs)/len(vs)})
  if curve:curves[stage]=curve
 report['error_curves']=curves
 if curves:
  text+=['','## 错误门控曲线','', 'ε为每个参考决策位置的扰动概率，每条路径最多在首次扰动处改错兄弟或停止；错误兄弟后保留合法但未确认的子树。开发3个种子，新来源2个非零剂量、1个种子；共享扰动不视为新增独立病例；同一病例、算子和路径的确定性重复复用已生成结果，并记录来源。','',
   '| 集合 | 类型 | ε | 算子 | 完整路径准确率 |','|---|---|---:|---|---:|']
  for stage,rs in curves.items():
   for r in rs:text.append(f'| {stage} | {r["error"]} | {r["epsilon"]} | {r["operator"]} | {r["path_accuracy"]:.4f} |')
 m1_path=OUT/'module_one'/'summary.json'
 if m1_path.exists():
  report['module_one']=json.loads(m1_path.read_text())
  text+=['','## 模块一修订实验','', '模块一以新100例／10来源选参，旧43例仅提议候选。结果详见 [模块一报告](module-one-results-20261031.zh.md)。七种组合搜索和LEACE不在本轮队列。']
 resources=[]
 for p in OUT.glob('*_status.json'):
  obj=json.loads(p.read_text())
  if 'peak_gb' in obj:resources.append({'file':p.name,**obj})
 report['resources']=resources
 report['deterministic_reuse']={stage:sum('reused_from_arm' in r for r in stage_rows(stage)) for stage in ['predicted_dev','predicted_validation','errors_dev','errors_validation']}
 if resources:text+=['','## 资源与解释','',f'使用GPU 0、1、5、6；记录到的最大峰值显存为{max(r["peak_gb"] for r in resources):.2f}GB。各阶段时长详见汇总JSON。初轮计时包含诊断，不据此声称部署速度优势。']
 text+=['','没有独立祖先治疗历史时不伪造历史条件化对照。模型三值判断不作为医生金标。未覆盖的临床语义和额外训练方法保留为后续工作。',
  '','所有已有结果和失败记录保留；新来源数据仅用于验证，未用于参数选择。']
 write(PUBLIC/'summary.json',report);write(OUT/'final_summary.json',report)
 content='\n'.join(text)+'\n'
 (PUBLIC/'REPORT.zh.md').write_text(content);(OUT/'REPORT.zh.md').write_text(content)
 (ROOT/'docs/ASTER/bottleneck-results-20261031.zh.md').write_text(content)
 # Exportable scientific figure; plotting failure does not invalidate model runs.
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axs=plt.subplots(1,2,figsize=(12,4))
  for ax,summary_obj,title in zip(axs,[core,validation],['Development (43 cases, 2 sources)','New sources (120 cases, 12 sources)']):
   if summary_obj:
    arms=summary_obj['arms'];ax.bar(range(len(arms)),[a['path']/a['n'] for a in arms],color='#3478a5')
    ax.set_xticks(range(len(arms)),[a['arm'] for a in arms],rotation=65,ha='right');ax.set_ylim(0,1);ax.set_title(title);ax.set_ylabel('Exact-path accuracy')
  fig.tight_layout();fig.savefig(PUBLIC/'operator_results.png',dpi=180);plt.close(fig)
  if curves:
   fig,axs=plt.subplots(1,2,figsize=(10,4))
   for ax,error in zip(axs,['wrong','stop']):
    for kind in ['clip','hard','bias']:
     rs=sorted([r for r in curves.get('errors_dev',[]) if r['error']==error and r['operator']==kind],key=lambda x:x['epsilon'])
     if rs:ax.plot([r['epsilon'] for r in rs],[r['path_accuracy'] for r in rs],marker='o',label=kind)
    ax.set_title(error);ax.set_xlabel('Per-decision corruption probability');ax.set_ylabel('Exact-path accuracy');ax.legend()
   fig.tight_layout();fig.savefig(PUBLIC/'error_curves.png',dpi=180);plt.close(fig)
 except Exception as e:write(OUT/'plot_error.json',{'error':str(e)})
 return report
if __name__=='__main__':build_report()
