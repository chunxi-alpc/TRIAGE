"""Sequential layer/head tests, matched key controls and shared-route experiments."""
import argparse,copy,json,math,os,random,select,subprocess,sys,time,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from .prepare import OUT
from .run import BASE,GPUS,read,write
from .routing import route,decision,beam_routes
def records(stage):
 rows=[]
 for i in range(4):
  p=OUT/f'{stage}_{i}.jsonl'
  if p.exists():rows+=read(p)
 return list({(r.get('case_id'),r.get('arm'),r.get('order'),r.get('state'),r.get('pair_id'),r.get('candidate')):r for r in rows}.values())
def summary(stage):
 rows=records(stage);groups=defaultdict(list)
 for r in rows:groups[r['arm']].append(r)
 result={}
 for name,rs in groups.items():
  margins=[r.get('result',{}).get('decision_margin',r.get('result',{}).get('root_margin')) for r in rs]
  margins=[m for m in margins if m is not None]
  result[name]={'n':len(rs),'path':sum(r.get('score',{}).get('exact_path',False) for r in rs),
   'leaf':sum(r.get('score',{}).get('leaf_correct',False) for r in rs),'failed':sum(r['status']!='complete' for r in rs),
   'decision_margin':sum(margins)/len(margins) if margins else None}
 write(OUT/f'{stage}_summary.json',result);return result
def launch(stage):
 children=[]
 for shard,gpu in enumerate(GPUS):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
  log=(OUT/f'{stage}_{shard}.log').open('a')
  proc=subprocess.Popen([sys.executable,'-m',__package__+'.advanced','--worker',stage,'--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT)
  children.append(proc);log.close()
 codes=[p.wait() for p in children]
 if any(codes):raise RuntimeError(f'{stage} failed: {codes}')
 return summary(stage) if stage not in ['routing','proof','behavior'] else None
def register(stage,cases,specs):
 write(OUT/f'{stage}_config.json',{'stage':stage,'cases':cases,'specs':specs,'seed':20261031})
def worker(stage,shard):
 if stage=='behavior':
  from .behavior import worker as behavior_worker
  return behavior_worker(shard)
 if stage=='proof':
  from .proof import worker as proof_worker
  return proof_worker(shard)
 if stage=='routing':
  from .routing import worker as routing_worker
  return routing_worker(shard)
 import torch
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
 from src.aster.manuscript_core import load_expert
 from .runtime import execute,marginal_execute
 torch.set_num_threads(4);torch.manual_seed(20261031)
 tok,model,_=model_and_classifier();engine=ClinicalExecutor(model,tok,load_expert(BASE/'analysis/expert.json'),json.loads((BASE/'frozen.json').read_text()))
 config=json.loads((OUT/f'{stage}_config.json').read_text())
 allcases={c['case_id']:c for c in read(OUT/'development43.jsonl')+read(OUT/'validation120.jsonl')}
 cases=[allcases[x] for x in config['cases']];specs=config['specs']
 depths=json.loads((OUT/'decision_depths.json').read_text())
 routings={r['case_id']:r for r in records('routing')} if stage.startswith('predicted') else {}
 output=OUT/f'{stage}_{shard}.jsonl';done={(r['case_id'],r['arm']) for r in read(output) if r['status']=='complete'} if output.exists() else set()
 start=time.time();count=0
 with query_chunked_attention(model,256),output.open('a') as f:
  for i,case in enumerate(cases):
   if i%4!=shard:continue
   repeated={}
   for original in specs:
    spec=copy.deepcopy(original)
    if spec.get('deeper'):spec['decision_depth']=depths.get(case['case_id'],0)
    if (case['case_id'],spec['name']) in done:continue
    selected_route=None
    if stage.startswith('predicted'):
     selected_route=routings[case['case_id']]['result']['outcomes'][spec.pop('route_variant')]['path']
    if 'epsilon' in spec:selected_route=corrupt_route(case,spec['epsilon'],spec['error_type'],spec['seed'])
    row={'case_id':case['case_id'],'source_id':case['source_id'],'arm':spec['name'],'spec':spec,'stage':stage,'applied_route':selected_route}
    reusable=stage.startswith('errors') or (stage.startswith('predicted') and spec['kind']!='marginal')
    equivalent=(tuple(selected_route or []),json.dumps({k:v for k,v in spec.items() if k not in ['name','epsilon','error_type','seed']},sort_keys=True))
    try:
     if reusable and equivalent in repeated:
      result,origin=repeated[equivalent];row['reused_from_arm']=origin;row['compute_seconds']=0.
     elif spec['kind']=='marginal':
      rs=beam_routes(case,routings[case['case_id']]['result']['decisions']['calibrated'],3)
      result=marginal_execute(engine,case,rs,spec['beta'])
     else:result=execute(engine,case,spec,route=selected_route,capture=bool(spec.get('capture') or spec.get('calibrate_only')))
     if reusable and equivalent not in repeated:repeated[equivalent]=(result,spec['name'])
     if 'compute_seconds' not in row:row['compute_seconds']=result.get('seconds',0.)
     score=strict_score(case,result['text']) if 'text' in result else {}
     row.update(status='complete',result=result,score=score)
    except Exception as e:
     traceback.print_exc();row.update(status='failed',error=str(e),score={'leaf_correct':False,'exact_path':False,'invalid':True})
     torch.cuda.empty_cache()
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
    write(OUT/f'{stage}_{shard}_status.json',{'status':'running','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
    if row['status']=='failed':raise RuntimeError(row['error'])
 write(OUT/f'{stage}_{shard}_status.json',{'status':'complete','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
def corrupt_route(case,epsilon,error_type,seed):
 # One possible corruption per reference path, chosen independently of method.
 rng=random.Random(str(seed)+':'+case['case_id']);gold=case['gold_rules'];chosen=[]
 for rid in gold:
  if rng.random()<epsilon:
   if error_type=='stop':return chosen
   wrong=rid[:-1]+('N' if rid[-1]=='Y' else 'Y');return chosen+[wrong]
  chosen.append(rid)
 return chosen
def best(summary,specs):
 return max(specs,key=lambda x:(summary[x['name']]['path'],summary[x['name']]['decision_margin'] or -1e9,-len(x.get('heads',[]))))
def run_pipeline(wait_pid=None):
 if wait_pid is not None:
  try:
   fd=os.pidfd_open(wait_pid);select.select([fd],[],[]);os.close(fd)
  except ProcessLookupError:pass
 initial=json.loads((OUT/'execution_status.json').read_text())
 if initial['status']!='complete':raise RuntimeError('Initial core/validation did not complete')
 write(OUT/'advanced_status.json',{'status':'running','stage':'windows','physical_gpus':GPUS})
 dev=read(OUT/'development43.jsonl');val=read(OUT/'validation120.jsonl');dev_ids=[c['case_id'] for c in dev];val_ids=[c['case_id'] for c in val]
 zeros={r['case_id']:r for r in records('core') if r['arm']=='zero'}
 depths={}
 for c in dev:
  predicted=zeros[c['case_id']]['score']['path'];gold=c['gold_rules']
  depths[c['case_id']]=next((i for i,r in enumerate(gold) if i>=len(predicted) or predicted[i]!=r),len(gold)-1)
 for c in val:depths[c['case_id']]=0
 write(OUT/'decision_depths.json',depths)
 beta=json.loads((OUT/'selected_soft.json').read_text())['beta']
 windows=[{'name':'window'+str(start),'kind':'bias','beta':beta,'heads':[[l,h] for l in range(start,start+6) for h in range(8)],'deeper':True} for start in range(0,36,6)]
 core={s['arm']:s for s in json.loads((OUT/'core_summary.json').read_text())['arms']}
 repair=[]
 if core['hard']['path']<core['clip']['path']:
  repair=[{'name':'repair'+str(int(x*100)),'kind':'repair','fraction':x,'deeper':True} for x in [.1,.2]]
 register('windows',dev_ids,windows+repair);result=launch('windows');chosen=best(result,windows)
 # Actual single-head interventions, measured at each case's first baseline error.
 probe_ids=dev_ids[::4]
 heads=chosen['heads']
 hs=[{'name':'probe_zero','kind':'zero','probe_only':True,'deeper':True}]
 hs += [{'name':'head'+str(l)+'_'+str(h),'kind':'bias','beta':beta,'heads':[[l,h]],'probe_only':True,'deeper':True} for l,h in heads]
 register('headscan',probe_ids,hs);write(OUT/'advanced_status.json',{'status':'running','stage':'headscan'});launch('headscan')
 rr=records('headscan');zero={r['case_id']:r['result']['decision_margin'] for r in rr if r['arm']=='probe_zero'}
 scored=[]
 for spec in hs[1:]:
  pairs=[r['result']['decision_margin']-zero[r['case_id']] for r in rr if r['arm']==spec['name']]
  scored.append({'head':spec['heads'][0],'effect':sum(pairs)/len(pairs)})
 scored.sort(key=lambda x:(-x['effect'],x['head']))
 write(OUT/'causal_head_ranking.json',{'window':chosen,'selection_cases':probe_ids,'heads':scored})
 subsets=[{'name':'heads'+str(n),'kind':'bias','beta':beta,'heads':[r['head'] for r in scored[:n]],'deeper':True} for n in [4,8,16,32,len(scored)]]
 register('subsets',dev_ids,subsets);result=launch('subsets');selected=best(result,subsets)
 write(OUT/'b4_selected_head_subset.json',selected)
 # If no window subset approaches all-layer soft bias, test the full available
 # direction pool; layer zero has no patient-conditioned key difference.
 if result[selected['name']]['path']<core['bias'+str(beta)]['path']:
  selected={'name':'all_candidate_heads','kind':'bias','beta':beta,'heads':[[l,h] for l in range(1,36) for h in range(8)],'deeper':True}
 selected['heads']=[h for h in selected['heads'] if h[0]>0]
 if not selected['heads']:selected['heads']=[[l,h] for l in range(1,36) for h in range(8)];selected['selection_note']='No learned direction at layer zero; use full available candidate pool'
 write(OUT/'selected_head_control.json',selected)
 cal_specs=[{'name':kind+'_unit','kind':kind,'gain':1.,'direction_heads':selected['heads'],'full_rule_keys':True,'calibrate_only':True,'probe_only':True,'deeper':True} for kind in ['mean','spectral','lowfreq']]
 register('key_calibration',probe_ids,cal_specs);launch('key_calibration')
 rr=records('key_calibration');key_specs=[]
 fits={}
 for kind in ['mean','spectral','lowfreq']:
  ts=[t for r in rr if r['arm']==kind+'_unit' for t in r['result']['trace'] if 'theory_unit_square_mean' in t]
  sq=sum(t['theory_unit_square_mean'] for t in ts)/len(ts);mean=sum(t['theory_unit_aligned_mean'] for t in ts)/len(ts)
  rms=beta/math.sqrt(max(sq,1e-12));ls=max(0.,beta*mean/max(sq,1e-12))
  fits[kind]={'unit_mean':mean,'unit_square_mean':sq,'rms_gain':rms,'least_squares_gain':ls,'sample_heads':len(ts)}
  key_specs.append({'name':'matched_'+kind,'kind':kind,'gain':rms,'direction_heads':selected['heads'],'full_rule_keys':True,'deeper':True,'capture':True})
  if kind=='mean':key_specs.append({'name':'least_squares_mean','kind':kind,'gain':ls,'direction_heads':selected['heads'],'full_rule_keys':True,'deeper':True,'capture':True})
 write(OUT/'key_gain_fits.json',fits)
 comparison={**selected,'name':'matched_bias','capture':True}
 register('key_controls',dev_ids,[comparison]+key_specs);write(OUT/'advanced_status.json',{'status':'running','stage':'key_controls'});launch('key_controls')
 val_specs=[{**s,'deeper':False} for s in [comparison]+key_specs]
 # Pending old module-one validation is superseded by development100 selection.
 # Module-two estimators are frozen independently of operator selection.
 write(OUT/'advanced_status.json',{'status':'running','stage':'routing'})
 launch('routing')
 from .routing import summarize as summarize_routes
 routing_summary=summarize_routes();variant=routing_summary['selected']
 write(OUT/'advanced_status.json',{'status':'running','stage':'module_one_new_development'})
 from .module_one import run_pipeline as run_module_one
 run_module_one()
 pred_specs=[]
 for rv in list(dict.fromkeys(['legacy:0.0','calibrated:0.0',routing_summary['selected_direct']])):
  for op in [{'kind':'clip'},{'kind':'hard'},{'kind':'bias','beta':beta},{'kind':'mean','gain':.25}]:
   name=rv.replace(':','_')+'_'+op['kind']
   if not any(s['name']==name for s in pred_specs):pred_specs.append({**op,'name':name,'route_variant':rv})
 pred_specs.append({'name':'calibrated_marginal3','kind':'marginal','beta':beta,'route_variant':'calibrated:0.0'})
 register('predicted_dev',dev_ids,pred_specs);launch('predicted_dev')
 register('predicted_validation',val_ids,pred_specs);launch('predicted_validation')
 # Same deterministic perturbations for every operator; repeated zero-error
 # generations are reused from the preceding oracle stage.
 curve=[]
 for eps in [.1,.2,.3,.5]:
  for error in ['wrong','stop']:
   for seed in [13,37,73]:
    for op in [{'kind':'clip'},{'kind':'hard'},{'kind':'bias','beta':beta}]:
     curve.append({**op,'name':f'eps{eps}_{error}_{seed}_{op["kind"]}','epsilon':eps,'error_type':error,'seed':seed})
 register('errors_dev',dev_ids,curve);write(OUT/'advanced_status.json',{'status':'running','stage':'errors_dev'});launch('errors_dev')
 valid_curve=[s for s in curve if s['epsilon'] in [.2,.5] and s['seed']==13]
 register('errors_validation',val_ids,valid_curve);launch('errors_validation')
 write(OUT/'advanced_status.json',{'status':'complete','stages':['windows','headscan','subsets','key_calibration','key_controls','module_one_new_development','routing','predicted_dev','predicted_validation','errors_dev','errors_validation']})
 from .report import build_report
 build_report()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--shard',type=int);p.add_argument('--wait-pid',type=int);a=p.parse_args()
 try:
  if a.worker:worker(a.worker,a.shard)
  else:run_pipeline(a.wait_pid)
 except Exception as e:
  if not a.worker:
   write(OUT/'advanced_status.json',{'status':'failed','error':str(e)})
   try:
    from .report import build_report
    build_report()
   except Exception:pass
  raise
