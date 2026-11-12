"""Frozen 10006 experiment: server-only, resumable, historical inputs read-only."""
import os,sys,json,time,shutil,traceback,subprocess,collections,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
OLDREPO=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111')
B=OLDREPO/'reports/aster_cr_multisource_20261112/balanced_10006'
H=OLDREPO/'reports/aster_cr_retrain_paired_v2_20261111'
sys.path.insert(0,str(OLDREPO))
sys.path.append(str(B/'code'))
import prepare
from prepare import rows,write,lines
import worker
import training_impl as tr
import m1_impl as m1
import evaluation_impl as ev
from functools import lru_cache
CFG=json.loads((ROOT/'configs/preregistered.json').read_text())
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else ({} if default is None else default)
def state(stage,**kw):write(ROOT/'status/current.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def link(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src,target_is_directory=Path(src).is_dir())
class Store(worker.CombinedStore):
 def __init__(self):
  super().__init__()
  self.extra_records={r['view_id']:r for p in (ROOT/'cache').glob('scores_*.jsonl') for r in rows(p)}
 def ell(self,k):return float(self.extra_records[k]['ell']) if k in self.extra_records else super().ell(k)
 @lru_cache(maxsize=1500)
 def get(self,k):
  if k not in self.extra_records:return super().get(k)
  r=dict(self.extra_records[k])
  with np.load(r['feature']) as z:
   r.update({n:z[n].astype(np.float32) for n in z.files})
  return r
 def heads(self,k,selected=None):
  h=self.get(k)['heads'];return h if selected is None else h[np.asarray(selected,int)].reshape(-1)
class Patcher(worker.Patcher):
 def path(self,p):return ROOT/'cache/exact'/(p['left']+'__'+p['right']+'.json')
 def cached(self,p):
  c=super().cached(p);c.update(read(self.path(p)));return c
def train_pairs():return rows(B/'data/fit_pairs.jsonl')
def folds():return read(ROOT/'configs/folds.json')['folds']
def fdir(g,f='full'):return ROOT/'models'/g/f
def endweights(pairs):
 out=collections.defaultdict(float)
 for p,w in zip(pairs,tr.pair_weights(pairs,'E-Multi-Balanced')):
  out[p['left']]+=w/2;out[p['right']]+=w/2
 return out
def center(old,new,pairs,store,unconstrained=False):
 new.mkdir(parents=True,exist_ok=True)
 for f in ['m1.json','m1_plan.json','m2.npz']:
  if (old/f).exists():link(old/f,new/f)
 support=dict(np.load(old/'m2.npz'));prior=dict(np.load(old/'m3.npz'))
 weights=endweights(pairs)
 coord={k:tr.coordinates(k,store,support) for k in weights}
 w=prior['w'].copy()
 if unconstrained:
  ap=[p for p in pairs if p['kind'] in ('A','P')]
  X=np.stack([coord[p['right']]-coord[p['left']] for p in ap])
  y=np.array([store.ell(p['right'])-store.ell(p['left']) for p in ap])
  ww=tr.pair_weights(ap,'E-Multi-Balanced',total=4887)
  w=-np.linalg.solve(X.T@(ww[:,None]*X)+.01*np.eye(X.shape[1]),X.T@(ww*y))
 c={k:float(t@w) for k,t in coord.items()}
 mu=sum(weights[k]*c[k] for k in weights)
 err={k:0. for k in ['A','P','C','F']}
 cf=[]
 for p in pairs:
  a,b=c[p['left']],c[p['right']]
  err[p['kind']]=max(err[p['kind']],abs((b-a)-((b-mu)-(a-mu))))
  if p['kind'] in ('C','F'):cf.append(b-a)
 assert max(err.values())<1e-12
 violation=max(0.,max(abs(np.array(cf)))-tr.DELTA)
 if not unconstrained:assert violation<=1e-7,violation
 np.savez_compressed(new/'m3.npz',w=w,mu_c=np.array(mu),delta=prior['delta'],regularization=prior['regularization'],
   centering_ids=np.array(list(weights)),centering_weights=np.array(list(weights.values())))
 meta=read(old/'model.json')
 meta.update(group=new.parent.name,formula='ell + w.T @ t - mu_c',global_intercept=False,
  training_pair_count=len(pairs),training_sources=sorted({p['source_id'] for p in pairs}),
  centering_source='weighted frozen training pair endpoints',mu_c=mu,
  validation=dict(pair_difference_max_error=err,cf_max_violation=violation,cf_count=len(cf),centering_mean=sum(weights[k]*(c[k]-mu) for k in weights)))
 # Remove historical intercept field from the new metadata.
 if 'M3' in meta:meta['M3']={k:v for k,v in meta['M3'].items() if k not in ['b','calibration_nodes']}
 write(new/'model.json',meta)
 return meta
def model(g,f='full'):
 p=fdir(g,f);z=dict(np.load(p/'m2.npz'));q=dict(np.load(p/'m3.npz'));a=(z['N'].astype(float)@q['w'])/z['scale']
 off=-float(z['mu']@a)+(float(q['b']) if 'b' in q else -float(q['mu_c']))
 return dict(group=g,fold=f,heads=z['heads'],alpha=a,offset=off,directory=str(p),metadata=read(p/'model.json'))
def setup():
 assert read(ROOT/'dataset_manifest.json')['matches_frozen']
 link(H/'models/E1',ROOT/'models/E1')
 link(B/'training/E-Multi-Balanced/full',fdir('E2'))
 store=Store();ps=train_pairs()
 if not (fdir('E3')/'m3.npz').exists():center(fdir('E2'),fdir('E3'),ps,store)
 write(ROOT/'configs/model_revision.json',dict(base_model=CFG['base_model'],config=read(Path(CFG['base_model'])/'config.json'),repo_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
 # Reuse attribution vectors across fold aggregations, never selected heads or fitted subspaces.
 attr=ROOT/'cache/pair_attribution.npy'
 if not attr.exists():
  aa=np.empty((len(ps),1152),np.float64)
  for i,p in enumerate(ps):
   l,r=store.get(p['left']),store.get(p['right'])
   aa[i]=np.abs(((r['heads']-l['heads'])*l['grads']).sum(-1))
  np.save(attr,aa)
 aa=np.load(attr);pi={p['pair_id']:i for i,p in enumerate(ps)}
 plans=[]
 full=read(fdir('E2')/'m1_plan.json')
 for fold in folds():
  name='fold'+str(fold['fold']);dest=fdir('E2',name)
  if (dest/'m1_plan.json').exists():plans.append(read(dest/'m1_plan.json'));continue
  allowed=set(fold['train_sources'])|{'MedEinst/DDXPlus'}
  pp=[p for p in ps if p['source_id'] in allowed]
  plan=dict(full,training_sources=sorted(allowed),train_sources=sorted(allowed),validation_sources=fold['validation_sources'],fit_pair_counts=dict(collections.Counter(p['kind'] for p in pp)),kinds={})
  for k in tr.KINDS:
   kk=[p for p in pp if p['kind']==k];ww=tr.pair_weights(kk,'E-Multi-Balanced')
   av=np.sum(aa[[pi[p['pair_id']] for p in kk]]*ww[:,None],axis=0)
   groups=collections.defaultdict(list);mass=collections.defaultdict(float)
   for p,w in zip(kk,ww):
    key=(p['dataset_id'],p.get('diagnosis_before') if p['dataset_id']=='MedEinst' else p['source_id'])
    groups[key].append(p);mass[key]+=w
   reps=[dict(min(groups[key],key=lambda p:p['pair_id']),weight=float(mass[key])) for key in sorted(groups)]
   plan['kinds'][k]=dict(attribution=av.tolist(),candidates=np.argsort(-av)[:64].tolist(),representatives=reps,fit_pair_count=len(kk))
  write(dest/'m1_plan.json',plan);plans.append(plan)
 jobs={}
 patcher=Patcher(store)
 for plan in plans:
  for block in plan['kinds'].values():
   for p in block['representatives']:
    key=(p['left'],p['right']);need={int(h) for h in block['candidates'] if str(h) not in patcher.cached(p)}
    if need:
     if key not in jobs:jobs[key]=dict(p,heads=[])
     jobs[key]['heads']=sorted(set(jobs[key]['heads'])|need)
 lines(ROOT/'configs/patch_jobs.jsonl',jobs.values())
 state('prepared',patch_pairs=len(jobs),patch_pair_heads=sum(len(p['heads']) for p in jobs.values()))
def gpu(mode,shard):
 # Compatibility with the already-running two-coordinator parent: each coordinator
 # now waits for two workers, so evaluation starts only after all four finish.
 if mode=='extract' and len(CFG['gpus'])==4:
  workers=[]
  for idx in [shard,shard+2]:
   gpu_id=CFG['gpus'][idx];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu_id))
   log=(ROOT/'logs'/f'extract_gpu_{gpu_id}.log').open('a')
   proc=subprocess.Popen([sys.executable,'-u',__file__,'extract_leaf',str(idx)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   workers.append((proc,log))
  write(ROOT/f'status/extraction_coordinator_{shard}.json',dict(workers=[dict(pid=p.pid,gpu=CFG['gpus'][idx]) for idx,(p,l) in zip([shard,shard+2],workers)]))
  codes=[p.wait() for p,l in workers]
  for p,l in workers:l.close()
  if any(codes):raise RuntimeError('four GPU extraction failed: '+str(codes))
  return
 import torch
 torch.set_num_threads(4);start=time.time();store=Store()
 if mode=='patch':
  jobs=rows(ROOT/'configs/patch_jobs.jsonl')[shard::2];patcher=Patcher(store)
  for i,p in enumerate(jobs):
   patcher(p,p['heads'])
   write(ROOT/f'status/patch_{shard}.json',dict(completed=i+1,total=len(jobs),seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
 else:
  jobs=rows(ROOT/'configs/extract_jobs.jsonl')[shard::len(CFG['gpus'])]
  if not jobs:return
  from experiments.aster_cr.engine import Engine
  eng=Engine('qwen');dest=ROOT/'cache/features';dest.mkdir(exist_ok=True)
  done=set(store.extra_records)
  for i,v in enumerate(jobs):
   key=v['view_id']
   if key in done:continue
   context=eng.prepare(store.input(key)['prompt']);r=eng.run(context,capture=True,grad=False)
   heads=r.pop('heads').numpy().astype(np.float16).reshape(1152,128);path=dest/(key+'.npz')
   np.savez(path,heads=heads)
   worker.append(ROOT/f'cache/scores_{shard}.jsonl',dict(r,view_id=key,feature=str(path),status='ok',available_heads='all',has_grads=False))
   del context
   if i<3 or (i+1)%25==0:write(ROOT/f'status/extract_{shard}.json',dict(completed=i+1,total=len(jobs),seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
 write(ROOT/f'status/{mode}_{shard}.json',dict(complete=True,total=len(jobs),seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
def launch_gpu(mode):
 jobs=rows(ROOT/'configs'/('patch_jobs.jsonl' if mode=='patch' else 'extract_jobs.jsonl'))
 if not jobs:return
 pp=[]
 for shard in [0,1]:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard))
  f=(ROOT/'logs'/f'{mode}_{shard}.log').open('a')
  p=subprocess.Popen([sys.executable,'-u',__file__,mode,str(shard)],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  pp.append((p,f))
 write(ROOT/'status/gpu_workers.json',dict(mode=mode,workers=[dict(pid=p.pid,gpu=i) for i,(p,f) in enumerate(pp)]))
 codes=[p.wait() for p,f in pp]
 for p,f in pp:f.close()
 if any(codes):raise RuntimeError(mode+' worker failure: '+str(codes))
def fit_folds():
 ps=train_pairs();store=Store();nv=rows(H/'data/natural_views.jsonl');patcher=Patcher(store)
 for fold in folds():
  f='fold'+str(fold['fold']);dest=fdir('E2',f);plan=read(dest/'m1_plan.json')
  pp=[p for p in ps if p['source_id'] in plan['training_sources']]
  natural=[v for v in nv if v['source_id'] in fold['train_sources'] and v.get('split')=='fit']
  assert not {p['source_id'] for p in pp}&set(fold['validation_sources'])
  if not (dest/'m3.npz').exists():
   state('fitting_'+f)
   heads,mm=m1.finish_plan(plan,store,lambda p,h:patcher(p,h,read_only=True),dest)
   sup,m2s=tr._m2('E-Multi-Balanced',heads,pp,store,[],dest)
   m3s=tr.refit_m3('E-Multi-Balanced',pp,store,natural,sup,dest)
   write(dest/'model.json',dict(complete=True,fold=f,training_sources=plan['training_sources'],validation_sources=fold['validation_sources'],M1=mm,M2=m2s,M3=m3s,delta=tr.DELTA))
  if not (fdir('E3',f)/'m3.npz').exists():center(dest,fdir('E3',f),pp,store)
def fit_ablations():
 store=Store();ps=train_pairs();nv=[v for v in rows(H/'data/natural_views.jsonl') if v.get('split')=='fit']
 full=fdir('E2');heads=dict(np.load(full/'m2.npz'))['heads'];basis=tr._basis
 for name,rank in [('rank8',8),('rank32',32),('without_M1',16),('without_M2_projection',16)]:
  dest=ROOT/'models/ablation_fit'/name;target=fdir(name)
  if (target/'m3.npz').exists():continue
  state('fitting_'+name);dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'m3.npz').exists():
   if name=='without_M2_projection':
    sup=dict(np.load(full/'m2.npz'));sup['N']=sup['U'].copy();np.savez_compressed(dest/'m2.npz',**sup)
   else:
    hh=np.sort(np.random.default_rng(CFG['seed']).choice(1152,51,replace=False)) if name=='without_M1' else heads
    tr._basis=lambda matrix: basis(matrix,rank=rank)
    sup,_=tr._m2('E-Multi-Balanced',hh,ps,store,[],dest);tr._basis=basis
   tr.refit_m3('E-Multi-Balanced',ps,store,nv,sup,dest)
   write(dest/'model.json',dict(ablation=name,rank=rank,delta=tr.DELTA))
  center(dest,target,ps,store)
 if not (fdir('without_constraints')/'m3.npz').exists():center(full,fdir('without_constraints'),ps,store,True)
def mega_specs(split,sources=None):
 vv=[v for v in ev.data_rows(H,'historical_views.jsonl') if v.get('split')==split]
 pp=[p for p in ev.data_rows(H,'historical_pairs.jsonl') if p.get('split')==split]
 pv=[v for n in ['path_features.jsonl','path_interference_views.jsonl'] for v in ev.data_rows(H,n) if v.get('split')==split]
 cc=[c for c in ev.case_rows(ev.HIST/'data/path_cases.jsonl') if c['split']==split]
 if sources is not None:
  sources=set(sources);vv=[v for v in vv if v['source_id'] in sources];pp=[p for p in pp if p['source_id'] in sources];pv=[v for v in pv if v['source_id'] in sources];cc=[c for c in cc if c['source_id'] in sources]
 return dict(views=vv,pairs=pp,path_views=pv,cases=cc)
def cv_specs(fold):
 sources=set(fold['validation_sources']);spec=mega_specs('fit',sources)
 spec['views']=[dict(v,original=False) for v in ev.data_rows(H,'E2_views.jsonl') if v['source_id'] in sources]+[v for v in ev.data_rows(H,'natural_views.jsonl') if v['source_id'] in sources]
 spec['pairs']=[p for p in ev.data_rows(H,'E2_pairs.jsonl') if p['source_id'] in sources]
 return spec
def target_specs(task,split=None):
 return dict(views=[v for v in ev.data_rows(H,task+'_views.jsonl') if split is None or v.get('split')==split],
 pairs=[p for p in ev.data_rows(H,task+'_pairs.jsonl') if split is None or p.get('split')==split],native_task=task)
def extraction_plan():
 store=Store();needs=collections.defaultdict(set);meta={}
 def add(spec,models):
  hh=set(int(h) for m in models for h in m['heads'])
  for v in spec['views']+spec.get('path_views',[]):needs[ev.uid(v)].update(hh);meta[ev.uid(v)]=v
 full=[model(g) for g in ['E1','E2','E3']]
 ab=[model(g) for g in ['rank8','rank32','without_M1','without_M2_projection','without_constraints']]
 for sp in ['dev','test']:add(mega_specs(sp),full+ab)
 add(dict(views=ev.data_rows(H,'new60_views.jsonl')),full)
 for task in ['nli4ct','medpic']:add(target_specs(task),full)
 for fold in folds():
  ff='fold'+str(fold['fold']);mm=[model(g,ff) for g in ['E1','E2','E3']]
  add(cv_specs(fold),mm);add(mega_specs('dev'),mm)
 jobs=[]
 for k,hh in needs.items():
  try:ok=np.isfinite(store.heads(k,sorted(hh))).all()
  except (KeyError,IndexError,ValueError):ok=False
  if not ok:jobs.append(dict(view_id=k))
 lines(ROOT/'configs/extract_jobs.jsonl',jobs)
 write(ROOT/'status/extraction_plan.json',dict(total_inputs=len(needs),missing=len(jobs),gpus=read(ROOT/'configs/preregistered.json')['gpus']))
def run():
 start=time.time()
 for stage,fn in [('setup',setup),('patch',lambda:launch_gpu('patch')),('fit_folds',fit_folds),('ablations',fit_ablations),('extraction_plan',extraction_plan),('extract',lambda:launch_gpu('extract'))]:
  state(stage);fn()
 import analysis
 analysis.run()
 state('complete',seconds=time.time()-start)
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1] in ['patch','extract','extract_leaf']:gpu(sys.argv[1],int(sys.argv[2]))
  elif len(sys.argv)>1 and sys.argv[1]=='setup':setup()
  else:run()
 except Exception as e:
  write(ROOT/'status/error.json',dict(error=str(e),traceback=traceback.format_exc(),time=time.time()))
  raise
