"""Module-one selection on 100 new cases, followed by frozen validation or diagnosis."""
import argparse,copy,json,math,os,subprocess,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from .run import OUT,BASE,GPUS,read,write
from .module_one_data import M1,prepare

def rows(stage):
 result=[]
 for i in range(4):
  p=M1/f'{stage}_{i}.jsonl'
  if p.exists():result+=read(p)
 return list({(r['case_id'],r['arm']):r for r in result}.values())

def summaries(stage):
 groups=defaultdict(list)
 for r in rows(stage):groups[r['arm']].append(r)
 result={}
 for arm,rs in groups.items():
  margins=[r['result']['decision_margin'] for r in rs if r.get('result',{}).get('decision_margin') is not None]
  result[arm]={'n':len(rs),'path':sum(r.get('score',{}).get('exact_path',False) for r in rs),
   'leaf':sum(r.get('score',{}).get('leaf_correct',False) for r in rs),
   'invalid':sum(r.get('score',{}).get('invalid',False) for r in rs),
   'failed':sum(r['status']!='complete' for r in rs),'margin':float(np.mean(margins)) if margins else None}
 write(M1/f'{stage}_summary.json',result);return result

def register(stage,case_ids,specs,**extra):
 config={'stage':stage,'cases':case_ids,'specs':specs,**extra}
 p=M1/f'{stage}_config.json'
 if p.exists() and json.loads(p.read_text())!=config:raise ValueError('Existing stage has different settings: '+stage)
 write(p,config)

def make_engine():
 import torch
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from src.aster.manuscript_core import load_expert
 torch.set_num_threads(4);torch.manual_seed(20261031)
 tok,model,_=model_and_classifier()
 return ClinicalExecutor(model,tok,load_expert(BASE/'analysis/expert.json'),json.loads((BASE/'frozen.json').read_text()))

def worker(stage,shard):
 if stage=='diagnostic':
  from .module_one_diagnostic import worker as diagnostic_worker
  return diagnostic_worker(shard)
 import torch
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
 from .runtime import execute
 engine=make_engine();cfg=json.loads((M1/f'{stage}_config.json').read_text())
 allcases={c['case_id']:c for c in read(M1/'development100.jsonl')+read(OUT/'validation120.jsonl')}
 cases=[allcases[x] for x in cfg['cases']]
 depths=json.loads((M1/'decision_depths.json').read_text()) if (M1/'decision_depths.json').exists() else {}
 if any(s.get('deeper') for s in cfg['specs']):
  basefile=M1/f'baseline_{shard}.jsonl'
  if basefile.exists():
   own={r['case_id']:r for r in read(basefile) if r['status']=='complete'}
   for c in cases:
    if c['case_id'] not in depths and c['case_id'] in own:depths[c['case_id']]=first_error_depth(c,own[c['case_id']])

 routings={}
 if cfg.get('predicted'):
  from .advanced import records
  routings={r['case_id']:r['result']['outcomes'][cfg['route_variant']]['path'] for r in records('routing') if r['status']=='complete'}
 output=M1/f'{stage}_{shard}.jsonl'
 done={(r['case_id'],r['arm']) for r in read(output) if r['status']=='complete'} if output.exists() else set()
 start=time.time();count=0
 with query_chunked_attention(engine.model,256),output.open('a') as f:
  for i,case in enumerate(cases):
   if i%4!=shard:continue
   for original in cfg['specs']:
    if (case['case_id'],original['name']) in done:continue
    spec=copy.deepcopy(original)
    if spec.pop('deeper',False):spec['decision_depth']=depths[case['case_id']]
    selected_route=routings.get(case['case_id']) if cfg.get('predicted') else None
    row={'case_id':case['case_id'],'source_id':case['source_id'],'arm':spec['name'],'spec':spec,
     'stage':stage,'applied_route':selected_route,'gold_route_diagnostic':not bool(cfg.get('predicted'))}
    try:
     result=execute(engine,case,spec,route=selected_route,capture=bool(spec.get('calibrate_only')))
     score=strict_score(case,result['text']) if 'text' in result else {}
     row.update(status='complete',result=result,score=score)
    except Exception as e:
     traceback.print_exc();row.update(status='failed',error=str(e),score={'exact_path':False,'leaf_correct':False,'invalid':True})
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
    write(M1/f'{stage}_{shard}_status.json',{'status':'running','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
    if row['status']=='failed':raise RuntimeError(row['error'])
 write(M1/f'{stage}_{shard}_status.json',{'status':'complete','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})

def launch(stage):
 write(M1/'status.json',{'status':'running','stage':stage,'physical_gpus':GPUS})
 children=[]
 for shard,gpu in enumerate(GPUS):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
  with (M1/f'{stage}_{shard}.log').open('a') as f:
   children.append(subprocess.Popen([sys.executable,'-m',__package__+'.module_one','--worker',stage,'--shard',str(shard)],env=env,stdout=f,stderr=subprocess.STDOUT))
 codes=[p.wait() for p in children]
 if any(codes):raise RuntimeError(f'Module one {stage} failed: {codes}')
 return summaries(stage) if stage!='diagnostic' else None

def first_error_depth(case,baseline):
 gold=case['gold_rules'];pred=baseline['score']['path']
 return next((i for i,r in enumerate(gold) if i>=len(pred) or pred[i]!=r),len(gold)-1)

def reference_gains(calibration,headsets,target_rms):
 result={}
 for name,heads in headsets.items():
  hs=set(map(tuple,heads));per_case=[]
  for r in calibration:
   ts=[t['theory_unit_square_mean'] for t in r['result']['trace'] if (t['layer'],t['kv_head']) in hs and 'theory_unit_square_mean' in t]
   if len(ts)!=len(hs):raise ValueError('Missing calibration heads')
   per_case.append(sum(ts)/len(ts))
  square=float(np.mean(per_case))
  if not math.isfinite(square) or square<=0:raise ValueError('No measurable key response: '+name)
  result[name]={'gain':target_rms/math.sqrt(square),'unit_square':square,'cases':len(per_case),'heads':len(hs)}
 return result

def choose_config(results,specs):
 return max(specs,key=lambda s:(results[s['name']]['path'],results[s['name']]['margin'],-abs(s['gain']),s['name']))

def validation_specs(selected):
 common={k:v for k,v in selected.items() if k not in ['name','deeper','probe_only']}
 return [
  {'name':'zero','kind':'zero'},
  {**common,'name':'mean','kind':'mean'},
  {**common,'name':'spectral','kind':'spectral'},
  {**common,'name':'reverse_mean','kind':'mean','gain':-selected['gain']},
  *[{**common,'name':'random'+str(seed),'kind':'random','random_seed':seed} for seed in [13,37,73]]]

def _run_pipeline():
 prepare()
 cases=read(M1/'development100.jsonl');ids=[c['case_id'] for c in cases]
 expert=json.loads((BASE/'analysis/expert.json').read_text())
 original=[[int(x) for x in h.split(':')] for h in expert['selected_heads']]
 b4=json.loads((OUT/'b4_selected_head_subset.json').read_text())['heads']
 b4=[h for h in b4 if h[0]>0]
 b4_note=None
 if not b4:
  ranking=json.loads((OUT/'causal_head_ranking.json').read_text())['heads']
  requested=len(json.loads((OUT/'b4_selected_head_subset.json').read_text())['heads'])
  b4=[r['head'] for r in ranking if r['head'][0]>0][:requested]
  b4_note='B4 subset was layer zero only; same-window ranked learnable heads replace unavailable directions'
  if not b4:raise ValueError('B4 window has no learnable heads')
 headsets={'original4':original,'b4':b4,'all280':[[l,h] for l in range(1,36) for h in range(8)]}
 write(M1/'head_candidates.json',{'headsets':headsets,'source':'historical43 only proposes B4 candidate; development100 selects final',
  'target_rms':2.,'gain_multipliers':[.5,1.,2.],'b4_note':b4_note})
 register('baseline',ids,[{'name':'zero','kind':'zero'}]);base_summary=launch('baseline')
 baselines={r['case_id']:r for r in rows('baseline')}
 write(M1/'decision_depths.json',{c['case_id']:first_error_depth(c,baselines[c['case_id']]) for c in cases})
 unit={'name':'unit','kind':'mean','gain':1.,'direction_heads':headsets['all280'],'unit_directions':True,
  'full_rule_keys':True,'calibrate_only':True,'probe_only':True,'deeper':True}
 register('calibration',ids,[unit]);launch('calibration')
 gains=reference_gains(rows('calibration'),headsets,2.);write(M1/'reference_gains.json',gains)
 specs=[]
 for name,heads in headsets.items():
  for factor in [.5,1.,2.]:
   specs.append({'name':name+'_x'+str(factor),'kind':'mean','gain':gains[name]['gain']*factor,
    'direction_heads':heads,'unit_directions':True,'full_rule_keys':True,'probe_only':True,'deeper':True})
 register('screen',ids,specs);screen=launch('screen')
 top=sorted(specs,key=lambda s:(-screen[s['name']]['margin'],abs(s['gain']),s['name']))[:2]
 full=[{k:v for k,v in s.items() if k!='probe_only'} for s in top]
 register('finalists',ids,full);result=launch('finalists');selected=choose_config(result,full)
 zero_margin=float(np.mean([r['result']['decision_margin'] for r in rows('calibration')]))
 margin_change=result[selected['name']]['margin']-zero_margin
 net=result[selected['name']]['path']-base_summary['zero']['path']
 passes=net>0 and margin_change>0
 frozen={'selected':selected,'development_cases':100,'development_sources':10,'net_path_gain':net,'mean_margin_change':margin_change,
  'proceed_to_validation':passes,'rule':'strictly positive net exact-path gain and mean decision-margin gain; exploratory continuation criterion, not significance',
  'baseline':base_summary['zero'],'selected_summary':result[selected['name']],'frozen_before_validation':True}
 write(M1/'frozen.json',frozen)
 if passes:
  # No validation outputs are read before this configuration has been saved.
  val_ids=[c['case_id'] for c in read(OUT/'validation120.jsonl')]
  arms=validation_specs(selected)
  register('oracle_validation',val_ids,arms);launch('oracle_validation')
  routing=json.loads((OUT/'routing_summary.json').read_text())
  predictive=[s for s in arms if s['name'] not in ['spectral','reverse_mean']]
  register('predicted_validation',val_ids,predictive,predicted=True,route_variant=routing['selected_direct']);launch('predicted_validation')
  branch='frozen_validation'
 else:
  launch('diagnostic')
  from .module_one_diagnostic import summarize as summarize_diagnostic
  summarize_diagnostic();branch='paired_diagnostic'
 write(M1/'status.json',{'status':'complete','branch':branch,'physical_gpus':GPUS})
 build_report()


def run_pipeline():
 try:return _run_pipeline()
 except Exception as e:
  write(M1/'status.json',{'status':'failed','error':str(e)})
  build_report()
  raise


def paired(rows_in,zero):
 baseline={r['case_id']:r for r in zero};groups=defaultdict(list)
 for r in rows_in:
  if r['case_id'] not in baseline:continue
  a=bool(r['score'].get('exact_path'));b=bool(baseline[r['case_id']]['score'].get('exact_path'))
  groups[r['source_id']].append((int(a)-int(b),a,b))
 flat=[x for xs in groups.values() for x in xs]
 effects=np.array([np.mean([x[0] for x in xs]) for xs in groups.values()])
 ci=None
 if len(effects)>=10:
  rng=np.random.default_rng(20261031);boot=effects[rng.integers(len(effects),size=(2000,len(effects)))].mean(1)
  ci=np.quantile(boot,[.025,.975]).tolist()
 return {'cases':len(flat),'sources':len(groups),'corrected':sum(a and not b for d,a,b in flat),
  'regressed':sum(b and not a for d,a,b in flat),'source_equal_delta':float(effects.mean()) if len(effects) else None,'source_bootstrap_ci':ci}

def build_report():
 def load(name,default=None):
  p=M1/name;return json.loads(p.read_text()) if p.exists() else default
 report={'status':load('status.json'),'manifest':load('manifest.json'),'frozen':load('frozen.json'),'stages':{}}
 text=['# 模块一：新开发集选参与冻结验证','',
  '原16对／8来源拟合方向；旧43例只作历史对照和候选提议；新100例／10来源选参；120例／12来源验证。',
  '方向仍由年龄最小对学习，新来源实验检验迁移，不预设通用临床适用性。','',
  '状态：'+str(report['status'])]
 for stage in ['baseline','calibration','screen','finalists','oracle_validation','predicted_validation']:
  obj=load(stage+'_summary.json')
  if not obj:continue
  report['stages'][stage]=obj
  text+=['','## '+stage,'','| 条件 | 例数 | 完整路径正确 | 终点正确 | 无效输出 | 决策边际 |','|---|---:|---:|---:|---:|---:|']
  for name,v in obj.items():
   if stage in ['calibration','screen']:text.append(f'| {name} | {v["n"]} | — | — | — | {v["margin"]} |')
   else:text.append(f'| {name} | {v["n"]} | {v["path"]} | {v["leaf"]} | {v["invalid"]} | {v["margin"]} |')
  if stage in ['calibration','screen']:text+=['','本阶段只评分，不生成；路径与终点列不适用，不能解释为准确率零。']
 if report['frozen']:
  f=report['frozen'];text+=['','## 继续条件','',f'开发完整路径净变化：{f["net_path_gain"]}；同决策位置平均边际变化：{f["mean_margin_change"]:.6f}。进入验证：{f["proceed_to_validation"]}。',
   '继续条件仅是开发判断，不是统计显著或临床有效性证明。']
 pairings={}
 for stage in ['oracle_validation','predicted_validation']:
  rs=rows(stage);zero=[r for r in rs if r['arm']=='zero']
  for arm in sorted({r['arm'] for r in rs}):pairings[stage+'/'+arm]=paired([r for r in rs if r['arm']==arm],zero)
 report['paired']=pairings
 if pairings:
  text+=['','## 来源配对结果','','| 条件 | 纠正 | 退化 | 来源等权差 | 95%来源重采样区间 |','|---|---:|---:|---:|---|']
  for arm,v in pairings.items():text.append(f'| {arm} | {v["corrected"]} | {v["regressed"]} | {v["source_equal_delta"]} | {v["source_bootstrap_ci"]} |')
 if (M1/'diagnostic_summary.json').exists():
  report['diagnostic']=load('diagnostic_summary.json')
  text+=['','## 失败定位','', '4对来自2个已暴露来源，不作为独立确认。只编辑目标规则编号和处置，患者前置比较K、V、KV替换；指南前置移植RoPE前差分。']
  for r in report['diagnostic']['rows']:text.append('- '+json.dumps(r,ensure_ascii=False))
 text+=['','开发集边际读取零编辑首错位置，全对病例读取末个决策。验证边际固定为根决策辅助指标；完整路径由自由生成评分。词元、头和随机种子不作为独立病例。']
 resources=[]
 for p in M1.glob('*_status.json'):
  x=json.loads(p.read_text())
  if 'peak_gb' in x:resources.append({'file':p.name,**x})
 report['resources']=resources
 write(M1/'summary.json',report);(M1/'REPORT.zh.md').write_text('\n'.join(text)+'\n')
 root=Path(__file__).resolve().parents[2]
 (root/'docs/ASTER/module-one-results-20261031.zh.md').write_text('\n'.join(text)+'\n')
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--shard',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 try:
  if a.worker:worker(a.worker,a.shard)
  elif a.report:build_report()
  else:run_pipeline()
 except Exception as e:
  if not a.worker:write(M1/'status.json',{'status':'failed','error':str(e)});build_report()
  raise
