"""Resource pilot; development-only inputs, never used to select a configuration."""
import argparse,json,math,os,select,time
import numpy as np
import torch
from .run import BASE,read,write
from .module_one_data import M1
from .module_one import make_engine
from .runtime import execute
from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
def main(wait_pid=None):
 if wait_pid:
  try:
   fd=os.pidfd_open(wait_pid);select.select([fd],[],[]);os.close(fd)
  except ProcessLookupError:pass
 from .module_one_diagnostic import run_pair
 write(M1/'pilot_status.json',{'status':'running','physical_gpu':6})
 engine=make_engine();cases=read(M1/'development100.jsonl')
 lengths=[]
 for c in cases:
  p=engine.builder.prepare(c['guideline'],c['patient'],c['question'],online=True)
  lengths.append((p.input_ids.shape[1],c['case_id'],c))
 lengths.sort(key=lambda x:(x[0],x[1]));chosen=[lengths[49][2],lengths[89][2]]
 heads=[[l,h] for l in range(1,36) for h in range(8)];records=[];start=time.time()
 with query_chunked_attention(engine.model,256),torch.inference_mode(),(M1/'pilot_records.jsonl').open('w') as f:
  for i,c in enumerate(chosen):
   specs=[{'name':'zero','kind':'zero'},{'name':'unit','kind':'mean','gain':1.,'direction_heads':heads,
    'full_rule_keys':True,'unit_directions':True,'calibrate_only':True,'probe_only':True}]
   gain=None
   for spec in specs:
    result=execute(engine,c,spec,capture=spec['name']=='unit')
    if spec['name']=='unit':
     sq=np.mean([t['theory_unit_square_mean'] for t in result['trace'] if 'theory_unit_square_mean' in t]);gain=2./math.sqrt(sq)
    row={'case_id':c['case_id'],'arm':spec['name'],'result':result};records.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   common={'kind':'mean','gain':gain,'direction_heads':heads,'unit_directions':True,'full_rule_keys':True}
   more=[{**common,'name':'mean_probe','probe_only':True},{**common,'name':'mean_generation'}]
   if i==0:more += [{**common,'name':'random_probe','kind':'random','random_seed':13,'probe_only':True},
     {**common,'name':'spectral_probe','kind':'spectral','probe_only':True},
     {**common,'name':'reverse_probe','gain':-gain,'probe_only':True}]
   for spec in more:
    result=execute(engine,c,spec)
    row={'case_id':c['case_id'],'arm':spec['name'],'result':result};records.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
  pair=next(r for r in read(BASE/'review/accepted_pairs.jsonl') if r['split']=='confirm')
  for row in run_pair(engine,pair):
   records.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
 sample=lambda arm:float(np.mean([r['result']['seconds'] for r in records if r['arm']==arm]))
 selection=(100*sample('zero')+100*sample('unit')+900*sample('mean_probe')+200*sample('mean_generation'))/4
 validation=1440*sample('mean_generation')/4
 diag=sum(r['result']['seconds'] for r in records if r.get('pair_id'))
 obj={'status':'complete','physical_gpu':6,'records':len(records),'elapsed_seconds':time.time()-start,
  'peak_gb':torch.cuda.max_memory_allocated()/1e9,'sample_cases':[c['case_id'] for c in chosen],
  'timing_seconds':{arm:sample(arm) for arm in ['zero','unit','mean_probe','mean_generation']},
  'estimated_module_one_hours':{'selection':selection/3600,'validation_if_pass':validation/3600,'diagnostic_if_fail':diag/3600},
  'formal_units':{'development_generations':300,'development_probes':1000,'validation_if_pass':1440,'diagnostic_if_fail':64},
  'note':'median/90th-percentile input length development cases; estimate excludes pending B4 and module-two jobs; timings include instrumentation'}
 write(M1/'pilot_status.json',obj)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
 try:main(a.wait_pid)
 except Exception as e:write(M1/'pilot_status.json',{'status':'failed','error':str(e)});raise
