"""Shard fit pairs across GPUs; sum sufficient statistics before eigendecomposition."""
import argparse,json,time,traceback
from pathlib import Path
from collections import Counter
import torch
from fullscale_engine import Engine
from fullscale_run import rows
from fullscale_data import write

def catalog(run):
 seen=set();tasks=[]
 for c in rows(run,'mega',{'fit'}):
  if c['invalid_source']:continue
  for r in c['rules']:
   key=(c['group'],r['id'])
   if key in seen:continue
   seen.add(key);tasks.append({'group':c['group'],'node':r['id'],'condition':r['condition']})
 dest=Path(run)/'direction_tasks.jsonl'
 with dest.open('x') as f:
  for task in tasks:f.write(json.dumps(task,ensure_ascii=False)+'\n')
 write(Path(run)/'direction_catalog.json',{'tasks':len(tasks),'source_families':len({x['group'] for x in tasks}),'split':'mega:fit','synthetic_logical_pairs':True,'clinical_gold_pairs':0})

def worker(a):
 root=Path(a.run)/'parallel_fit';root.mkdir(exist_ok=True)
 out=root/('shard_'+str(a.rank));out.mkdir(exist_ok=True)
 cp=out/'checkpoint.pt';tasks=[json.loads(x) for x in (Path(a.run)/'direction_tasks.jsonl').open()][a.rank::a.world]
 state=torch.load(cp,weights_only=False) if cp.exists() else {'cursor':0,'n':0,'sums':{},'cov':{},'counts':Counter()}
 if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete':return
 start=time.time();write(out/'status.json',{'status':'running','assigned_pairs':len(tasks),'rank':a.rank,'world':a.world})
 def save(status):
  temp=out/'checkpoint.tmp';torch.save(state,temp);temp.replace(cp)
  write(out/'status.json',{'status':status,'rank':a.rank,'world':a.world,'assigned_pairs':len(tasks),'processed':state['cursor'],'accepted':state['n'],'counts':dict(state['counts']),'seconds':time.time()-start})
 try:
  e=Engine(a.model);layers=[e.model.config.num_hidden_layers//3,e.model.config.num_hidden_layers//2,e.model.config.num_hidden_layers*2//3]
  with (out/'pairs.jsonl').open('a') as log:
   for i,task in enumerate(tasks):
    if i<state['cursor']:continue
    cond=task['condition'];status='accepted'
    try:
     if not cond.strip():raise ValueError('empty_condition')
     prompts=[e.prompt('Rule: If condition C is true, choose A; if condition C is false, choose B.\nCondition C: '+cond+'\nPatient fact: condition C is '+truth+'.\nReturn A or B only. Answer:') for truth in ('true','false')]
     lengths=[e.encode(p)[0]['input_ids'].shape[1] for p in prompts]
     if lengths[0]!=lengths[1]:raise ValueError('token_length_unaligned')
     positive,negative=[e.query(p,layers) for p in prompts]
     for layer in layers:
      delta=(positive[layer]-negative[layer]).double()
      state['sums'][layer]=state['sums'].get(layer,torch.zeros_like(delta))+delta
      outer=torch.einsum('hi,hj->hij',delta,delta)
      state['cov'][layer]=state['cov'].get(layer,torch.zeros_like(outer))+outer
     state['n']+=1
    except Exception as ex:status=str(ex);torch.cuda.empty_cache()
    state['counts'][status]+=1;state['cursor']=i+1
    log.write(json.dumps({'group':task['group'],'node':task['node'],'status':status})+'\n');log.flush()
    if state['cursor']%64==0:save('running')
   save('complete')
 except Exception:
  save('failed');raise

def merge(a):
 root=Path(a.run)/'parallel_fit';sums={};cov={};n=0;counts=Counter();processed=0
 for rank in range(a.world):
  out=root/('shard_'+str(rank));status=json.loads((out/'status.json').read_text())
  if status['status']!='complete':raise RuntimeError('incomplete direction shard '+str(rank))
  s=torch.load(out/'checkpoint.pt',weights_only=False);n+=s['n'];processed+=s['cursor'];counts.update(s['counts'])
  for layer in s['sums']:
   sums[layer]=sums.get(layer,torch.zeros_like(s['sums'][layer]))+s['sums'][layer]
   cov[layer]=cov.get(layer,torch.zeros_like(s['cov'][layer]))+s['cov'][layer]
 if not n:raise RuntimeError('no accepted fit pairs')
 expected=json.loads((Path(a.run)/'direction_catalog.json').read_text())['tasks']
 if processed!=expected:raise RuntimeError('direction task denominator mismatch')
 candidates=[]
 for layer in sorted(cov):
  vals,vec=torch.linalg.eigh(cov[layer]/n)
  for head in range(len(vals)):candidates.append((float(vals[head,-1]),layer,head,vec[head,:,-1].float(),(sums[layer][head]/n).float()))
 _,layer,head,d,mean=max(candidates,key=lambda x:x[0]);d/=d.norm();mean/=mean.norm().clamp_min(1e-8)
 generator=torch.Generator().manual_seed(20261029);random=torch.randn(d.shape,generator=generator);random/=random.norm()
 artifact={'selection':{'layer':layer,'head':head,'strength':.5},'directions':{'learned':d,'mean':mean,'random':random,'reverse':d},'training_pairs':n,'training_split':'mega:fit','method':'leading eigenvector of summed paired final-query second moments; raw paired mean control','parallel_world':a.world}
 torch.save(artifact,root/'directions.pt');write(root/'summary.json',{'processed_pairs':processed,'accepted_pairs':n,'counts':dict(counts),'world':a.world,'selected_layer':layer,'selected_head':head,'clinical_gold_pairs':0})

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['catalog','worker','merge']);p.add_argument('--run',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=5);p.add_argument('--model',default='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B');a=p.parse_args()
 if a.stage=='catalog':catalog(a.run)
 elif a.stage=='worker':worker(a)
 else:merge(a)
