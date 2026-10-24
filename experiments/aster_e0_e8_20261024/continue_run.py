"""Resumable E2/E3 -> development lock -> E5/E6/E8. All inference on one allowed GPU."""
import argparse,collections,hashlib,json,os,random,time,traceback
from pathlib import Path
import torch
from engine import Engine
ROOT=Path('/home/gcx/aster_e0_e8_20261024/private')
CORE=('NC','AC','IC','SW')
def digest(x):return hashlib.sha256(x).hexdigest()
def key(c):return f"{c['id']}|{c['seed']}|{c['variant']}"
def safe(c):return {k:c[k] for k in ('patient','rules','options')}
def argmax(p):return max(range(len(p)),key=lambda i:p[i])
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def read_lines(path):
 if not path.exists():return []
 data=path.read_bytes();lines=data.splitlines(keepends=True);good=[];size=0
 for i,line in enumerate(lines):
  try:r=json.loads(line)
  except json.JSONDecodeError:
   if i!=len(lines)-1:raise
   with path.open('r+b') as f:f.truncate(size)
   break
  good.append(r);size+=len(line)
 return good
class Journal:
 def __init__(self,out):self.out=out;self.tables={};self.handles={}
 def table(self,stage,method):
  name=f'{stage}/{method}'
  if name not in self.tables:
   path=self.out/(name+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
   self.tables[name]={r['key']:r for r in read_lines(path)};self.handles[name]=path.open('a')
  return self.tables[name]
 def get(self,stage,method,c):return self.table(stage,method).get(key(c))
 def put(self,stage,method,c,**values):
  table=self.table(stage,method);r={k:c[k] for k in ('id','tree','variant','seed','split')};r.update(key=key(c),**values)
  self.handles[f'{stage}/{method}'].write(json.dumps(r)+'\n');self.handles[f'{stage}/{method}'].flush();table[key(c)]=r;return r
class Runner:
 def __init__(self,a):
  self.a=a;self.out=a.out;self.out.mkdir(parents=True,exist_ok=True);self.started=time.time()
  gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ('0','1','6'),gpu
  self.rows=[json.loads(x) for x in (ROOT/'cases.jsonl').read_text().splitlines()]
  signature={'model':a.model,'gpu':gpu,'data_sha256':digest((ROOT/'cases.jsonl').read_bytes()),'engine_sha256':digest(Path(__file__).with_name('engine.py').read_bytes()),'runner_sha256':digest(Path(__file__).read_bytes()),'role':a.role,'dev_seed':0,'test_seeds':[0,1,2],'test_cases':1000,'core':CORE,'shard_size':50}
  manifest=self.out/'manifest.json'
  if manifest.exists():assert json.loads(manifest.read_text())==json.loads(json.dumps(signature)),'Run signature changed; use a new output directory'
  else:atomic(manifest,signature)
  self.j=Journal(self.out);self.e=Engine(a.model);self.evidence={};self.evidence_offset=0;self.calls=0
 def probability(self,c,name,**kw):
  r=self.j.get('branches',name,c)
  if r:return r['p']
  t=time.time();p=self.e.branch(safe(c),**kw);self.j.put('branches',name,c,p=p,seconds=time.time()-t);return p
 def external(self,c):
  path=self.a.evidence/'branches'/'hidden.jsonl'
  deadline=time.time()+7200
  while key(c) not in self.evidence:
   if path.exists():
    with path.open() as f:
     f.seek(self.evidence_offset)
     while True:
      pos=f.tell();line=f.readline()
      if not line or not line.endswith('\n'):break
      r=json.loads(line);self.evidence[r['key']]=r['p'];self.evidence_offset=f.tell()
   if key(c) in self.evidence:break
   if time.time()>deadline:raise RuntimeError('Evidence worker has not produced required row in two hours')
   if (self.a.evidence/'FAILED.json').exists():raise RuntimeError('Evidence worker failed; inspect its FAILED.json')
   time.sleep(2)
  return self.evidence[key(c)]
 def answer(self,c,stage,name,oracle=False,direct=None,**kw):
  old=self.j.get(stage,name,c)
  if old:return old
  signature=json.dumps({'direct':direct,'kw':kw},sort_keys=True)
  t=time.time();cached=signature in self.cache
  if cached:pred,scores=self.cache[signature]
  else:
   pred,scores=self.e.direct_map(safe(c),direct) if direct is not None else self.e.answer(safe(c),**kw)
   self.cache[signature]=(pred,scores);self.calls+=1
  return self.j.put(stage,name,c,prediction=pred,correct=pred==c['gold_option'],gold_option=c['gold_option'],oracle=oracle,scores=scores,seconds=time.time()-t,reused=cached,config=kw if direct is None else {'direct_map':True})
 def status(self,stage,completed,total):
  torch.cuda.synchronize();atomic(self.out/'status.json',{'stage':stage,'completed_rows':completed,'total_rows':total,'elapsed_seconds':time.time()-self.started,'gpu_hours_this_process':(time.time()-self.started)/3600,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'updated_unix':time.time(),'unique_answer_calls':self.calls,'complete':stage=='complete'})
 def ordered(self,split,seed=None):
  rows=[r for r in self.rows if r['split']==split and (seed is None or r['seed']==seed)]
  return sorted(rows,key=lambda c:(c['seed'],digest(c['id'].encode()),CORE.index(c['variant']) if c['variant'] in CORE else 9,c['variant']))
 def oracle_arms(self,c):
  selected=c['gold_rule'];p=[.01/(len(c['rules'])-1)]*len(c['rules']);p[selected]=.99
  return {
   'none':{},'gold_crop':dict(form='crop',selected=selected),'gold_action_crop':dict(form='crop_action',selected=selected),
   'gold_prompt':dict(selected=selected,write_rule=True),
   'stage_hard':dict(mode='hard',p=p),
   'stage_hard_blind_options':dict(mode='hard',p=p,blind=True,gate_options=True),
   'all_action_hard':dict(mode='hard',p=p,gate_phase='all'),
   'all_rule_hard':dict(mode='hard',p=p,gate_phase='all',gate_scope='rule'),
   'all_action_options_hard':dict(mode='hard',p=p,gate_phase='all',gate_options=True),
   **{f'all_action_soft_{lam}':dict(mode='gate',p=p,lam=lam,gate_phase='all') for lam in (1,2,4)}}
 def develop(self):
  rows=self.ordered('dev',0);minimal=set(c['id'] for c in rows[:200]);lams=(1,2,4)
  for i,c in enumerate(rows):
   self.cache={}
   for name,kw in self.oracle_arms(c).items():self.answer(c,'E2',name,oracle=name!='none',**kw)
   visible=self.probability(c,'visible');blind=self.probability(c,'blind',blind=True);ext=self.external(c);full=[(a+b)/2 for a,b in zip(blind,ext)]
   for name,p in [('self',blind),('full',full)]:
    for lam in lams:
     for layers in ('all','last_half'):
      self.answer(c,'dev_select',f'{name}_soft_{lam}_{layers}',mode='gate',p=p,lam=lam,layers=layers,gate_phase='all')
   self.answer(c,'dev_select','none');self.answer(c,'dev_select','crop',form='crop',selected=argmax(blind))
   self.answer(c,'dev_select','explicit',selected=argmax(visible),write_rule=True)
   self.answer(c,'dev_select','direct',direct=blind)
   if c['id'] in minimal:
    self.probability(c,'hidden',hidden=True);self.probability(c,'patient_first',order='patient_first');self.probability(c,'repeat3',repeat=3)
   if (i+1)%20==0:self.status('development',i+1,len(rows))
  # Only development correctness is accessed during parameter selection.
  means={name:sum(r['correct'] for r in table.values())/len(table) for name,table in self.j.tables.items() if name.startswith('dev_select/')}
  lock={'created_unix':time.time(),'split':'dev','seed':0,'rows':len(rows),'selection_metric':'equal-weight mean over NC/AC/IC/SW; ties prefer lower lambda and all layers','candidate_accuracy':means,'manifest_sha256':digest((self.out/'manifest.json').read_bytes())}
  for source in ('self','full'):
   candidates=[(lam,layers) for lam in lams for layers in ('all','last_half')]
   lam,layers=max(candidates,key=lambda z:means[f'dev_select/{source}_soft_{z[0]}_{z[1]}'])
   lock[source]={'lam':lam,'layers':layers,'mode':'gate','gate_phase':'all','gate_scope':'action','gate_options':False,'blind':False}
  threshold_scores={}
  for threshold in (.5,.7,.9):
   threshold_scores[str(threshold)]=sum(self.j.get('dev_select','crop' if max(self.j.get('branches','blind',c)['p'])>=threshold else 'none',c)['correct'] for c in rows)/len(rows)
  lock['confidence_threshold']=max((.5,.7,.9),key=lambda t:threshold_scores[str(t)]);lock['confidence_accuracy']=threshold_scores
  path=self.out/'locked.json'
  if path.exists():
   old=json.loads(path.read_text());assert old['candidate_accuracy']==lock['candidate_accuracy'];return old
  atomic(path,lock);return lock
 def evaluate(self,lock):
  rows=self.ordered('test');subset=set(c['id'] for c in rows if c['seed']==0 and c['variant']=='NC');subset=set(sorted(subset,key=lambda x:digest(x.encode()))[:100])
  for i,c in enumerate(rows):
   self.cache={};blind=self.probability(c,'blind',blind=True);visible=self.probability(c,'visible');external=self.external(c);full=[(a+b)/2 for a,b in zip(blind,external)];selected=argmax(blind)
   self.answer(c,'E5','none');self.answer(c,'E5','explicit_branch',selected=argmax(visible),write_rule=True)
   self.answer(c,'E5','two_stage_crop',form='crop',selected=selected)
   kw=dict(form='crop',selected=selected) if max(blind)>=lock['confidence_threshold'] else {}
   self.answer(c,'E5','confidence_crop',**kw);self.answer(c,'E5','direct_map',direct=blind)
   self.answer(c,'E5','ASTER_self',p=blind,**lock['self']);self.answer(c,'E5','ASTER_full',p=full,**lock['full'])
   self.answer(c,'E5','gold_crop',oracle=True,form='crop',selected=c['gold_rule'])
   if c['seed']==0 and c['id'] in subset and c['variant'] in CORE:
    self.robustness(c,blind,external,lock)
    self.answer(c,'E8','visible_gate',p=visible,**lock['self'])
    self.answer(c,'E8','hard_gate',p=blind,mode='hard',gate_phase='all')
    self.answer(c,'E8','with_options',p=blind,**dict(lock['self'],gate_options=True))
    self.answer(c,'E8','with_blind',p=blind,**dict(lock['self'],blind=True))
   if (i+1)%20==0:self.status('test',i+1,len(rows))
  self.status('complete',len(rows),len(rows))
 def robustness(self,c,blind,external,lock):
  # Controlled oracle perturbations are diagnostics, explicitly tagged oracle.
  # Predictions in E5 never use these probabilities or test labels.
  n=len(blind);rng=random.Random(int(digest(key(c).encode())[:16],16));u=rng.random();wrong=rng.choice([j for j in range(n) if j!=c['gold_rule']])
  for kind,levels in [('noise',(0.,.1,.2,.4)),('coverage',(1.,.75,.5))]:
   for level in levels:
    covered=(u<level) if kind=='coverage' else True;chosen=wrong if kind=='noise' and u<level else c['gold_rule'];p=[.01/(n-1)]*n;p[chosen]=.99
    suffix=f'{kind}_{level:g}'
    for method in ('crop','gate','direct'):
     if not covered:self.answer(c,'E6',method+'_'+suffix,oracle=True)
     elif method=='crop':self.answer(c,'E6',method+'_'+suffix,oracle=True,form='crop',selected=chosen)
     elif method=='direct':self.answer(c,'E6',method+'_'+suffix,oracle=True,direct=p)
     else:self.answer(c,'E6',method+'_'+suffix,oracle=True,p=p,**lock['self'])
 def evidence_run(self):
  rows=self.ordered('dev',0)+self.ordered('test')
  for i,c in enumerate(rows):
   self.probability(c,'hidden',hidden=True)
   if (i+1)%50==0:self.status('external_evidence',i+1,len(rows))
  # Additional-model generalization, separate development choice and seed-0 test.
  dev=self.ordered('dev',0)
  for c in dev:
   self.cache={};p=self.probability(c,'hidden',hidden=True)
   for lam in (1,2,4):self.answer(c,'E7_dev',f'soft_{lam}',mode='gate',p=p,lam=lam,gate_phase='all')
  lam=max((1,2,4),key=lambda l:sum(r['correct'] for r in self.j.table('E7_dev',f'soft_{l}').values()))
  atomic(self.out/'locked.json',{'split':'dev','seed':0,'lam':lam,'created_unix':time.time()})
  tests=self.ordered('test',0)
  for i,c in enumerate(tests):
   self.cache={};p=self.probability(c,'hidden',hidden=True)
   self.answer(c,'E7','none');self.answer(c,'E7','direct_map',direct=p);self.answer(c,'E7','crop',form='crop',selected=argmax(p));self.answer(c,'E7','hidden_soft',mode='gate',p=p,lam=lam,gate_phase='all');self.answer(c,'E7','gold_crop',oracle=True,form='crop',selected=c['gold_rule'])
   if (i+1)%50==0:self.status('E7',i+1,len(tests))
  self.status('complete',len(tests),len(tests))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--role',choices=['main','evidence'],default='main');ap.add_argument('--evidence',type=Path);a=ap.parse_args()
 try:
  r=Runner(a)
  if a.role=='evidence':r.evidence_run()
  else:r.evaluate(r.develop())
 except BaseException as exc:
  a.out.mkdir(parents=True,exist_ok=True);atomic(a.out/'FAILED.json',{'error':repr(exc),'time':time.time()});traceback.print_exc();raise
if __name__=='__main__':main()
