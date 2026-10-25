"""Dev-only complete-matrix report; no test artifacts or private text exported."""
import argparse,collections,gzip,hashlib,json,random,statistics
from pathlib import Path
CORE=('NC','AC','IC','SW')
def metric(rows,method):
 n=len(rows);correct=sum(r['answers'][method]==r['gold_option'] for r in rows)
 harmed=sum(r['answers']['original']==r['gold_option'] and r['answers'][method]!=r['gold_option'] for r in rows)
 fixed=sum(r['answers']['original']!=r['gold_option'] and r['answers'][method]==r['gold_option'] for r in rows)
 baseline_correct=sum(r['answers']['original']==r['gold_option'] for r in rows)
 use=0
 for r in rows:
  source=r.get('oof_sources',{}).get(method,method)
  use+=max(r['S'])>=.6 if 'conf0.6' in source else source not in ('original','mass_zero_control')
 return dict(n=n,correct=correct,accuracy=correct/n,harmed=harmed,harm=harmed/n,harm_among_original_correct=harmed/baseline_correct if baseline_correct else None,corrected=fixed,changed=sum(r['answers'][method]!=r['answers']['original'] for r in rows),intervention_coverage=use/n,mapping_coverage=sum(r['covered'] for r in rows)/n)
def paired(rows,a,b):
 by=collections.defaultdict(list)
 for r in rows:by[r['tree']].append(int(r['answers'][a]==r['gold_option'])-int(r['answers'][b]==r['gold_option']))
 totals=[(len(v),sum(v)) for v in by.values()];rng=random.Random(0);ds=[]
 for _ in range(2000):
  x=rng.choices(totals,k=len(totals));ds.append(sum(v for n,v in x)/sum(n for n,v in x))
 ds.sort()
 return {'accuracy_delta':sum(v for n,v in totals)/len(rows),'tree_bootstrap_95':[ds[50],ds[1950]],'trees':len(totals),'replicates':2000}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/home/gcx/aster_action_mass_seed0_20261025'));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--models',nargs='+',choices=['qwen','llama'],default=['qwen','llama']);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);report={};text=['# ASTER 动作级质量守恒：seed=0 开发集实测','', '本报告只评价原有 200 病例 × NC/AC/IC/SW。未重新运行或使用独立测试标签选参。所有候选完整报告；开发集最优成绩不代表独立测试效果。','']
 for model in a.models:
  folder=a.root/model;status=json.loads((folder/'status.json').read_text());assert status['complete']
  rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
  assert len(rows)==800 and len({(r['id'],r['variant']) for r in rows})==800
  assert len({r['id'] for r in rows})==200 and all(r['split']=='dev' and r['seed']==0 for r in rows)
  methods=list(rows[0]['answers']);assert all(set(r['answers'])==set(methods) for r in rows)
  groups={'all':rows,**{v:[r for r in rows if r['variant']==v] for v in CORE}}
  metrics={m:{v:metric(rr,m) for v,rr in groups.items()} for m in methods}
  simple=[m for m in methods if m=='direct_map' or m.startswith('fusion_')]
  mass=[m for m in methods if m.startswith('mass_') and 'zero' not in m]
  best_simple=max(simple,key=lambda m:metrics[m]['all']['accuracy']);best_mass=max(mass,key=lambda m:metrics[m]['all']['accuracy'])
  choices={}
  for t in sorted({r['tree'] for r in rows}):
   train=[r for r in rows if r['tree']!=t]
   def select(ms):return max(ms,key=lambda m:sum(r['answers'][m]==r['gold_option'] for r in train))
   choices[t]={'simple':select(simple),'mass':select(mass)}
  for r in rows:
   r['oof_sources']={'oof_'+f:choices[r['tree']][f] for f in ('simple','mass')}
   for family in ('simple','mass'):r['answers']['oof_'+family]=r['answers'][choices[r['tree']][family]]
  oof={m:{v:metric(rr,m) for v,rr in groups.items()} for m in ('oof_simple','oof_mass')}
  payload={'status':status,'metrics':metrics,'best_simple_dev':best_simple,'best_mass_dev':best_mass,'dev_best_gap':paired(rows,best_mass,best_simple),'oof_metrics':oof,'oof_gap':paired(rows,'oof_mass','oof_simple'),'oof_choices':choices,'data_coverage':{'records':len(rows),'cases':200,'excluded':0,'orphan_option_records':sum(r['orphan_options']>0 for r in rows),'multi_rule_action_records':sum(r['rules']>r['actions'] for r in rows),'represented_mass_mean':statistics.mean(r['represented_mass'] for r in rows),'max_tokens':max(r['tokens'] for r in rows)},'condition_gpu_seconds':sum(json.loads(l)['seconds'] for l in (folder/'condition_cache.jsonl').read_text().splitlines()),'runtime_seconds_by_method':{m:sum(r['seconds'].get(m,0) for r in rows) for m in methods},'fresh_condition_seconds':sum(r.get('condition_seconds',0) for r in rows),'raw_sha256':hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest(),'manifest':json.loads((folder/'manifest.json').read_text()),'gpu_tests':json.loads((folder/'gpu_tests.json').read_text())}
  payload['zero_control']={'prediction_changes':sum(r['answers']['original']!=r['answers']['mass_zero_control'] for r in rows),'max_logit_abs_error':max(abs(x-y) for r in rows for x,y in zip(r['scores']['original'],r['scores']['mass_zero_control'])),'best_mass_vs_zero':paired(rows,best_mass,'mass_zero_control')}
  report[model]=payload
  with gzip.open(a.output/f'{model}_predictions.jsonl.gz','wt') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
  text += [f'## {model}',f"实际耗时 {status['elapsed_seconds']/3600:.3f} 小时；峰值显存 {status['peak_GiB']:.2f} GiB；800/800 条完成。",'', '| 配置 | NC准确率 | NC伤害 | IC准确率 | IC伤害 | AC准确率 | SW准确率 | 总准确率 | 覆盖率 | 纠正/退化 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for m in methods:
   z=metrics[m]
   vals=[z['NC']['accuracy'],z['NC']['harm'],z['IC']['accuracy'],z['IC']['harm'],z['AC']['accuracy'],z['SW']['accuracy'],z['all']['accuracy'],z['all']['intervention_coverage']]
   text.append('| '+m+' | '+' | '.join(f'{x*100:.2f}%' for x in vals)+f" | {z['all']['corrected']}/{z['all']['harmed']} |")
  text += ['',f"零强度数值对照改变 {payload['zero_control']['prediction_changes']}/800 个答案；最大选项logit误差 {payload['zero_control']['max_logit_abs_error']:.4f}。BF16实现并非逐位相同，小幅差异不能自动归因为干预。",'']
  gap=payload['dev_best_gap'];og=payload['oof_gap']
  text += ['',f"开发集最强简单基线：{best_simple}；最强质量守恒配置：{best_mass}。差距 {gap['accuracy_delta']*100:+.2f} 个百分点（按树配对区间 {[round(x*100,2) for x in gap['tree_bootstrap_95']]}；同集选择后的描述性区间）。",f"按树留一选参后的质量守恒相对简单基线差距：{og['accuracy_delta']*100:+.2f} 个百分点。该结果仍是开发集交叉验证，未作独立测试声明。",'质量守恒没有超过最强简单基线。下一步最有价值的改进是检查条件判断的校准与原答案信号互补性，优先改进简单融合，而非增加注意力强度。' if gap['accuracy_delta']<=0 else '质量守恒在本开发集有正向差距；须在未来预先冻结的新独立数据上验证，不能用已公布 P3 标签继续适应。','']
 text += ['伤害＝原答案正确而新方法错误的条数/全部记录；这是答题退化，不是临床伤害。完整 JSON 另列相对原答对样本的退化率。覆盖率＝启用方法的比例，另报动作到选项的映射覆盖率。孤儿选项保留原注意力；直接映射的孤儿选项质量为零，融合仍保留原答案概率。','', '所有条件判断来自本次真实 GPU 调用；完全相同的条件输入在本次运行内缓存，缓存键和使用情况逐例记录。选项、病例、条件和规则集合统一；动作裁剪仅将非胜出动作文本替为固定占位符，保留条件及完整选项。旧分支门控仍按原规则级 max 方式，作为明确的历史对照。','', '限制：单随机种子；10 个开发树；只评分答案字母，未进行长文本生成；不共享跨配置 KV 缓存；未运行独立测试。树区间未作多重比较校正。实际成本按 GPU 小时列出，未假定货币单价。']
 (a.output/'summary.json').write_text(json.dumps(report,indent=2));(a.output/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
