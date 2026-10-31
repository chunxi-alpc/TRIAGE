import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from collections import defaultdict
from .prepare import OUT
BASE=Path('/home/gcx/aster_manuscript_20261030/clinical01')
GPUS=[0,1,5,6]
ARMS=[
 {'name':'zero','kind':'zero'},
 {'name':'spectral025','kind':'spectral','gain':.25},
 {'name':'mean025','kind':'mean','gain':.25},
 {'name':'clip','kind':'clip'},
 {'name':'neutral','kind':'neutral'},
 {'name':'hard','kind':'hard'},
 {'name':'hard_siblings','kind':'hard','siblings_only':True},
 *[{'name':'bias'+str(b),'kind':'bias','beta':b} for b in [1,2,4,8]]]
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def summarize(stage):
 rows=[]
 for shard in range(4):
  p=OUT/f'{stage}_{shard}.jsonl'
  if p.exists():rows+=read(p)
 rows=list({(r['case_id'],r['arm']):r for r in rows}.values())
 groups=defaultdict(list)
 for r in rows:groups[r['arm']].append(r)
 summaries=[]
 for arm,rs in groups.items():
  margins=[r['result']['root_margin'] for r in rs if r.get('result',{}).get('root_margin') is not None]
  per={}
  for source in sorted({r['source_id'] for r in rs}):
   sub=[r for r in rs if r['source_id']==source]
   per[source]={'n':len(sub),'path':sum(r['score']['exact_path'] for r in sub),'leaf':sum(r['score']['leaf_correct'] for r in sub)}
  summaries.append({'arm':arm,'n':len(rs),'path':sum(r['score']['exact_path'] for r in rs),
   'leaf':sum(r['score']['leaf_correct'] for r in rs),'failed':sum(r['status']!='complete' for r in rs),
   'invalid':sum(r['score']['invalid'] for r in rs),'root_margin_mean':sum(margins)/len(margins) if margins else None,
   'mean_seconds':sum(r.get('result',{}).get('seconds',0) for r in rs)/len(rs),'per_source':per})
 out={'stage':stage,'records':len(rows),'arms':summaries}
 write(OUT/f'{stage}_summary.json',out)
 return out
def worker(stage,shard,smoke=False):
 import torch
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from src.aster.manuscript_core import load_expert
 from experiments.aster_abc_diagnostics_20261030.c.audit import strict_score
 from .runtime import execute
 torch.set_num_threads(4);torch.manual_seed(20261031)
 tok,model,_=model_and_classifier()
 freeze=json.loads((BASE/'frozen.json').read_text());expert=load_expert(BASE/'analysis/expert.json')
 engine=ClinicalExecutor(model,tok,expert,freeze)
 cases=read(OUT/('development43.jsonl' if stage=='core' else 'validation120.jsonl'))
 arms=ARMS if stage=='core' else [ARMS[0],ARMS[3],ARMS[5],ARMS[2],json.loads((OUT/'selected_soft.json').read_text())]
 if smoke:cases=cases[:1];arms=[ARMS[0],ARMS[5]]
 output=OUT/(f'smoke_{shard}.jsonl' if smoke else f'{stage}_{shard}.jsonl')
 completed=set()
 if output.exists():completed={(r['case_id'],r['arm']) for r in read(output) if r['status']=='complete'}
 start=time.time();count=0
 with query_chunked_attention(model,256),output.open('a') as f:
  for i,case in enumerate(cases):
   if i%4!=shard:continue
   for spec in arms:
    if (case['case_id'],spec['name']) in completed:continue
    row={'case_id':case['case_id'],'source_id':case['source_id'],'arm':spec['name'],'stage':stage,
      'gold_route_diagnostic':True,'gold_depth':len(case['gold_rules'])}
    try:
     result=execute(engine,case,spec,capture=stage=='core' and spec['name'] in ['zero','spectral025','mean025','hard','bias4'])
     row.update(status='complete',result=result,score=strict_score(case,result['text']))
    except Exception as e:
     traceback.print_exc();row.update(status='failed',error=str(e),score={'leaf_correct':False,'exact_path':False,'invalid':True})
     torch.cuda.empty_cache()
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
    write(OUT/f'{stage}_{shard}_status.json',{'status':'running','records_this_run':count,'last_arm':spec['name'],
     'last_case':case['case_id'],'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
    if row['status']=='failed':raise RuntimeError(row['error'])
 write(OUT/f'{stage}_{shard}_status.json',{'status':'complete','records_this_run':count,'elapsed':time.time()-start,
  'peak_gb':torch.cuda.max_memory_allocated()/1e9})
def launch_stage(stage):
 children=[]
 for shard,gpu in enumerate(GPUS):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
  log=(OUT/f'{stage}_{shard}.log').open('a')
  child=subprocess.Popen([sys.executable,'-m',__package__+'.run','--stage',stage,'--shard',str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT)
  children.append(child);log.close()
 failures=[p.wait() for p in children]
 if any(failures):raise RuntimeError(f'{stage} workers failed: {failures}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['core','validation']);p.add_argument('--shard',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.stage:return worker(a.stage,a.shard,a.smoke)
 write(OUT/'execution_status.json',{'status':'running','stage':'core','gpus':GPUS})
 try:
  launch_stage('core');summary=summarize('core')
  candidates=[s for s in summary['arms'] if s['arm'].startswith('bias')]
  best=max(candidates,key=lambda s:(s['path'],s['root_margin_mean'],-int(s['arm'][4:])))
  spec=next(x for x in ARMS if x['name']==best['arm']);write(OUT/'selected_soft.json',spec)
  write(OUT/'execution_status.json',{'status':'running','stage':'validation','selected_soft':spec})
  launch_stage('validation');summarize('validation')
  write(OUT/'execution_status.json',{'status':'complete','stages':['core','validation'],'gpus':GPUS})
 except Exception as e:
  write(OUT/'execution_status.json',{'status':'failed','error':str(e)});raise
if __name__=='__main__':main()
