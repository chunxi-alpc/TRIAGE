"""Source-separated, label-supervised score repair. CPU fitting; immutable real caches."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("OMP_NUM_THREADS","4")
import argparse,csv,gzip,importlib.util,json,time,sys,shutil
from pathlib import Path
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize,brentq,LinearConstraint
from scipy.special import expit
torch.set_num_threads(int(os.environ.get("TRACE_TRAIN_THREADS","4")))
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=Path(os.environ.get("TRACE_REPAIR_CACHE","/home/gcx/aster_trace_repair_20261110"))
OUT=HERE/"training"
LEGACY=Path("/home/gcx/aster_cr_repair_v2_run_20261108")
FOLDS=REPO/"reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full"
BUDGETS=[.05,.1,.25]
SEEDS=[11,22,33]
FAMILIES=["label_linear","trace_nn","score_nn","random_nn"]
REG=.01
def rows(p):
 p=Path(p)
 if not p.exists():return []
 with p.open() as f:return [json.loads(s) for s in f if s.strip()]
def iterrows(p):
 p=Path(p)
 if p.exists():
  with p.open() as f:
   for s in f:
    if s.strip():yield json.loads(s)
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvout(p,rr):
 if not rr:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 opener=gzip.open if p.suffix==".gz" else open
 with opener(p,"wt",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
@lru_cache(maxsize=12)
def support(k="main"):
 file=LEGACY/"qwen/supports_r16.npz" if k=="main" else FOLDS/("fold"+str(k))/"support_and_ridge.npz"
 with np.load(file) as a:return {x:a[x] for x in ("heads","mu","scale","N","S")}
def select_raw(heads,head_ids,wanted):
 h=np.asarray(heads)
 if h.ndim==1:h=h.reshape(-1,128)
 elif h.ndim>2:h=h.reshape(-1,128)
 ids=np.asarray(head_ids).astype(int) if head_ids is not None else np.arange(len(h))
 if len(h)==51 and head_ids is None:ids=support()["heads"].astype(int)
 mapping={int(v):i for i,v in enumerate(ids)}
 missing=set(map(int,wanted))-set(mapping)
 if missing:raise ValueError("Missing required attention heads: "+str(sorted(missing)))
 return h[[mapping[int(v)] for v in wanted]].reshape(-1).astype(np.float32)
def feature_read(file):
 with np.load(file) as a:
  if "heads" in a:return a["heads"].copy(),a["head_ids"].copy() if "head_ids" in a else None
  return a["selected_heads"].copy(),None
def source_weights(sources,indices):
 indices=np.asarray(indices,int)
 if not len(indices):return np.empty(0,float)
 cc=Counter(sources[indices])
 w=np.array([1./cc[s] for s in sources[indices]])
 return w/w.sum()
def hierarchical_weights(sources,cases,variants):
 """Equal sources, cases within source, variants within case, then nodes."""
 groups={}
 for i,(s,c,v) in enumerate(zip(sources,cases,variants)):groups.setdefault(s,{}).setdefault(c,{}).setdefault(v,[]).append(i)
 w=np.zeros(len(sources),float)
 for cases0 in groups.values():
  for vv in cases0.values():
   for ii in vv.values():w[ii]=1./len(groups)/len(cases0)/len(vv)/len(ii)
 return w
def task_weights(d,indices):
 indices=np.asarray(indices,int)
 return hierarchical_weights(d.sources[indices],
  [d.tasks[i].get("case_id",d.ids[i]) for i in indices],
  [d.tasks[i].get("variant","original") for i in indices]) if len(indices) else np.empty(0,float)
def path_accuracy(d,scores,indices):
 groups={}
 for i in indices:
  task=d.tasks[i];source=task["source_id"];case=task.get("case_id",d.ids[i]);variant=task.get("variant","original")
  groups.setdefault(source,{}).setdefault(case,{}).setdefault(variant,[]).append(bool((scores[i]>0)==d.labels[i]))
 value=float(np.mean([np.mean([np.mean([all(nodes) for nodes in variants.values()]) for variants in cases.values()]) for cases in groups.values()]))
 return value,sum(len(cases) for cases in groups.values())
def tagged_config(family,budget,ap,clean):
 return dict(family=family,budget=float(budget),lambda_AP=float(ap),lambda_clean=float(clean),
  regularization=REG,source_equal_loss=True,AP_clean_normalization="training AP raw-difference RMS squared",
  evidence_penalty=100.,lr=.001,steps=1000,early_stopping_every=50)
def candidate_name(c):
 return c["family"]+"_b"+str(c["budget"])+"_ap"+str(c["lambda_AP"])+"_clean"+str(c["lambda_clean"])
class Dataset:
 def __init__(self,split,allow_partial=False):
  self.split=split
  if split=="fit" and not allow_partial and not (ROOT/"scoring/fit/done.json").exists():
   raise RuntimeError("Fit scoring is incomplete: need scoring/fit/done.json")
  keys=("id","task_id","case_id","source_id","gid","variant","split","label","kind","original")
  tasks=[]
  for v in [*iterrows(ROOT/"data"/("tasks_"+split+".jsonl")),*iterrows(ROOT/"data"/("tasks_diagnostic_"+split+".jsonl"))]:
   t={k:v[k] for k in keys if k in v};t["id"]=str(t.get("id",t.get("task_id")));tasks.append(t)
  score={}
  for file in sorted((ROOT/"scoring"/split).glob("shard*/scores.jsonl")):
   for r in iterrows(file):
    if r.get("status")=="ok":score[str(r.get("task_id",r.get("id")))]=r
  missing=[t["id"] for t in tasks if t["id"] not in score]
  if missing and not allow_partial:raise RuntimeError(f"{split}: {len(missing)} tasks lack successful scores")
  self.tasks=[t for t in tasks if t["id"] in score]
  self.pairs=rows(ROOT/"data"/("pairs_"+split+".jsonl"))+rows(ROOT/"data"/("pairs_diagnostic_"+split+".jsonl"))
  if split=="fit":
   # Old fit C/F constraints remain real observations and stay in their source fold.
   pv=rows(LEGACY/"data/pairs.jsonl")
   oldcf=[p for p in pv if p["split"]=="fit" and p["kind"] in ("C","F")]
   want={p[k] for p in oldcf for k in ("left","right")}
   oldmeta={v["view_id"]:v for v in iterrows(LEGACY/"data/views.jsonl") if v["view_id"] in want}
   olds={}
   for file in sorted((LEGACY/"qwen/scores").glob("fit_*.jsonl")):
    for r in iterrows(file):
     if r.get("status")=="ok" and r["view_id"] in want:olds[r["view_id"]]=r
   for vid in sorted(want):
    if vid not in olds:raise RuntimeError("Missing legacy fit evidence view "+vid)
    v=oldmeta[vid];key="legacy_cf|"+vid
    self.tasks.append(dict(id=key,source_id=v["source_id"],split="fit",label=v["label"],kind="evidence_constraint",variant="legacy_evidence",original=False))
    score[key]=olds[vid]
   self.pairs.extend([{**p,"left":"legacy_cf|"+p["left"],"right":"legacy_cf|"+p["right"]} for p in oldcf])
  self.ids=[t["id"] for t in self.tasks];self.ix={v:i for i,v in enumerate(self.ids)}
  self.scores=score
  self.pairs=[p for p in self.pairs if p["left"] in self.ix and p["right"] in self.ix]
  self.ell=np.array([score[v]["ell"] for v in self.ids],float)
  self.labels=np.array([v["label"] for v in self.tasks],float)
  self.sources=np.array([v["source_id"] for v in self.tasks])
  self.natural=np.array([v.get("kind")=="original" or v.get("variant")=="original" for v in self.tasks])
  self.interference=np.array([v.get("kind") in ("A","P") for v in self.tasks])
  self.coordinates={}
 def pair_idx(self,kinds,sources=None):
  ps=[p for p in self.pairs if p["kind"] in kinds and (sources is None or p["source_id"] in sources)]
  return np.array([self.ix[p["left"]] for p in ps],int),np.array([self.ix[p["right"]] for p in ps],int),np.array([p["source_id"] for p in ps])
 def prepare(self):
  dest=ROOT/"training_cache"/(self.split+"_coordinates.npz");meta=dest.with_suffix(".json")
  if dest.exists():
   with np.load(dest) as a:
    if a["ids"].tolist()==self.ids:
     self.coordinates={k:a[k] for k in a.files if k!="ids"};return
  sup={str(i):support(i) for i in range(5)} if self.split=="fit" else {}
  sup["main"]=support()
  q={k:np.linalg.qr(np.random.default_rng(11).normal(size=(len(s["mu"]),16)),mode="reduced")[0].astype(np.float32) for k,s in sup.items()}
  coord={k:np.empty((len(self.ids),32),np.float32) for k in sup}
  files=[self.scores[v]["feature"] for v in self.ids]
  with ThreadPoolExecutor(max_workers=8) as pool:
   for lo in range(0,len(files),256):
    hi=min(lo+256,len(files));batch=list(pool.map(feature_read,files[lo:hi]))
    for k,s in sup.items():
     raw=np.stack([select_raw(h,hh,s["heads"]) for h,hh in batch])
     z=(raw-s["mu"])/s["scale"]
     coord[k][lo:hi]=z@np.concatenate([s["N"],q[k]],axis=1)
    if lo%4096==0:dump(OUT/"prepare_status.json",dict(split=self.split,completed=hi,total=len(files)))
  self.coordinates=coord;dest.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(dest,ids=np.array(self.ids),**coord)
  dump(meta,dict(real_cache=True,n=len(self.ids),split=self.split,source_ids=sorted(set(self.sources)),projection_keys=list(sup)))
 def input(self,family,key):
  t=self.coordinates[str(key)]
  if family=="label_linear":return t[:,:16].astype(float)
  tr=t[:,16:] if family=="random_nn" else np.zeros_like(t[:,:16]) if family=="score_nn" else t[:,:16]
  return np.column_stack([tr,self.ell]).astype(float)
def index_groups(d,sources):
 mask=np.isin(d.sources,list(sources))
 ni=np.flatnonzero(mask&d.natural);ii=np.flatnonzero(mask&d.interference)
 clean=ni[((d.ell[ni]>0)==d.labels[ni])]
 al,ar,aps=d.pair_idx(("A","P"),sources);cl,cr,cfs=d.pair_idx(("C","F"),sources)
 if not len(ni) or not len(ii) or not len(al):raise ValueError("Training/validation groups require original, interference and AP rows")
 scale=max(float(np.sqrt(np.mean((d.ell[ar]-d.ell[al])**2))),1e-6)
 return dict(ni=ni,ii=ii,clean=clean,al=al,ar=ar,cl=cl,cr=cr,
  wn=task_weights(d,ni),wi=task_weights(d,ii),wc=task_weights(d,clean),
  wa=task_weights(d,ar),we=source_weights(cfs,np.arange(len(cfs))),scale=scale)
def calibrate(raw,labels,weights):
 def f(b):return float(np.sum(weights*(expit(raw+b)-labels)))
 lo,hi=-1.,1.
 while f(lo)>0:lo*=2
 while f(hi)<0:hi*=2
 return float(brentq(f,lo,hi,xtol=1e-10))
def xscale(x,g):
 w=g["wn"];mu=w@x[g["ni"]];sd=np.sqrt(w@((x[g["ni"]]-mu)**2));return mu,np.maximum(sd,1e-5)
def shrink_to_fit(delta,g,budget):
 observed=float(np.max(np.abs(delta[g["cr"]]-delta[g["cl"]]))) if len(g["cl"]) else 0.
 return min(1.,budget*g["scale"]/max(observed,1e-12))
def scalar_ce(scores,labels,w):return float(np.sum(w*(np.logaddexp(0,scores)-labels*scores)))
def measure(d,s,g):
 nat=float(np.sum(g["wn"]*((s[g["ni"]]>0)==d.labels[g["ni"]])))
 inter=float(np.sum(g["wi"]*((s[g["ii"]]>0)==d.labels[g["ii"]])))
 ce=.5*scalar_ce(s[g["ni"]],d.labels[g["ni"]],g["wn"])+.5*scalar_ce(s[g["ii"]],d.labels[g["ii"]],g["wi"])
 raw=d.ell[g["ar"]]-d.ell[g["al"]];new=s[g["ar"]]-s[g["al"]]
 delta=s-d.ell;ed=delta[g["cr"]]-delta[g["cl"]]
 natural_path,ncases=path_accuracy(d,s,g["ni"]);interference_path,icases=path_accuracy(d,s,g["ii"])
 return dict(natural_accuracy=natural_path,interference_accuracy=interference_path,natural_path_accuracy=natural_path,interference_path_accuracy=interference_path,
  natural_node_accuracy=nat,interference_node_accuracy=inter,natural_cases=ncases,interference_cases=icases,CE=ce,AP_RMS=float(np.sqrt(g["wa"]@(new*new))),
  AP_VR=float(1-(g["wa"]@((new-g["wa"]@new)**2))/(g["wa"]@((raw-g["wa"]@raw)**2))),
  evidence_RMS=float(np.sqrt(g["we"]@(ed*ed))) if len(ed) else None,
  evidence_max=float(np.max(np.abs(ed))) if len(ed) else None,natural_count=len(g["ni"]),interference_count=len(g["ii"]))
def fit_linear(d,x,g,c):
 mu,sd=xscale(x,g);z=(x-mu)/sd
 E=z[g["cr"]]-z[g["cl"]];X=z[g["ar"]]-z[g["al"]];y=d.ell[g["ar"]]-d.ell[g["al"]]
 scale2=g["scale"]**2;nc=z[g["clean"]]
 def fun(w):
  dc=z@w;s=d.ell+dc
  loss=0.;grad=np.zeros(len(w))
  for inds,weights in [(g["ni"],g["wn"]),(g["ii"],g["wi"])]:
   loss+=.5*scalar_ce(s[inds],d.labels[inds],weights)
   grad+=.5*z[inds].T@(weights*(expit(s[inds])-d.labels[inds]))
  ap=y+X@w
  loss+=c["lambda_AP"]*float(g["wa"]@(ap*ap))/scale2
  grad+=2*c["lambda_AP"]*(X.T@(g["wa"]*ap))/scale2
  if len(nc):
   clean=nc@w;loss+=c["lambda_clean"]*float(g["wc"]@(clean*clean))/scale2
   grad+=2*c["lambda_clean"]*(nc.T@(g["wc"]*clean))/scale2
  loss+=REG*float(np.mean(w*w));grad+=2*REG*w/len(w)
  return loss,grad
 constraints=[LinearConstraint(E,-c["budget"]*g["scale"],c["budget"]*g["scale"])] if len(E) else []
 opt=minimize(fun,np.zeros(x.shape[1]),jac=True,method="SLSQP",constraints=constraints,options=dict(maxiter=500,ftol=1e-10))
 if not opt.success:raise RuntimeError("Label linear SLSQP: "+opt.message)
 delta=z@opt.x
 sh=shrink_to_fit(delta,g,c["budget"]);delta*=sh
 b=calibrate(d.ell[g["ni"]]+delta[g["ni"]],d.labels[g["ni"]],g["wn"])
 state=dict(type="linear",w=opt.x,mu=mu,sd=sd,shrink=sh,b=b,feature=c["family"])
 return state,d.ell+delta+b,dict(success=True,iterations=int(opt.nit),steps=None,shrink=sh)
class SmallNet(nn.Module):
 def __init__(self,dim=17):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x).squeeze(-1)
def fit_nn(d,x,g,c,seed,vg=None,epochs=None):
 torch.manual_seed(seed)
 mu,sd=xscale(x,g);z=torch.tensor((x-mu)/sd,dtype=torch.float32)
 e=torch.tensor(d.ell,dtype=torch.float32);y=torch.tensor(d.labels,dtype=torch.float32)
 model=SmallNet(x.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=.001)
 tg={k:torch.tensor(v,dtype=torch.long if k in ("ni","ii","clean","al","ar","cl","cr") else torch.float32) for k,v in g.items() if k!="scale"}
 def loss(delta,group,tgroup):
  val=.5*(tgroup["wn"]*nn.functional.binary_cross_entropy_with_logits(e[tgroup["ni"]]+delta[tgroup["ni"]],y[tgroup["ni"]],reduction="none")).sum()
  val+=.5*(tgroup["wi"]*nn.functional.binary_cross_entropy_with_logits(e[tgroup["ii"]]+delta[tgroup["ii"]],y[tgroup["ii"]],reduction="none")).sum()
  val+=c["lambda_AP"]*(tgroup["wa"]*(e[tgroup["ar"]]-e[tgroup["al"]]+delta[tgroup["ar"]]-delta[tgroup["al"]]).square()).sum()/(g["scale"]**2)
  if len(group["clean"]):val+=c["lambda_clean"]*(tgroup["wc"]*delta[tgroup["clean"]].square()).sum()/(g["scale"]**2)
  if len(group["cl"]):
   ed=delta[tgroup["cr"]]-delta[tgroup["cl"]]
   val+=100.*(tgroup["we"]*torch.relu(ed.abs()-c["budget"]*g["scale"]).square()).sum()/(g["scale"]**2)
  return val
 tvg={k:torch.tensor(v,dtype=torch.long if k in ("ni","ii","clean","al","ar","cl","cr") else torch.float32) for k,v in vg.items() if k!="scale"} if vg is not None else None
 # Compute only train rows during optimizer steps, retaining full indexing through a compact remap.
 active=np.unique(np.concatenate([g[k] for k in ("ni","ii","clean","al","ar","cl","cr")]))
 local=np.full(len(x),-1,int);local[active]=np.arange(len(active));za=z[active];ea=e[active];ya=y[active]
 orig_e,orig_y=e,y
 train_tg={k:torch.tensor(local[g[k]],dtype=torch.long) if k in ("ni","ii","clean","al","ar","cl","cr") else v for k,v in tg.items()}
 bestloss=float("inf");best=None;beststep=epochs or c["steps"]
 patience=0
 for step in range(1,(epochs or c["steps"])+1):
  opt.zero_grad();delta=model(za)
  e,y=ea,ya
  v=loss(delta,g,train_tg)+REG*sum(p.square().sum() for p in model.parameters())/sum(p.numel() for p in model.parameters())
  v.backward();opt.step()
  e,y=orig_e,orig_y
  if vg is not None and step%50==0:
   model.eval()
   with torch.no_grad():
    full=model(z);gain=shrink_to_fit(full.numpy().astype(float),g,c["budget"])
    vl=float(loss(full*gain,vg,tvg))
   if vl<bestloss-1e-7:
    bestloss=vl;beststep=step;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
   else:patience+=1
   model.train()
   if patience>=5:break
 if best is not None:model.load_state_dict(best)
 model.eval()
 with torch.no_grad():delta=model(z).numpy().astype(float)
 sh=shrink_to_fit(delta,g,c["budget"]);delta*=sh
 b=calibrate(d.ell[g["ni"]]+delta[g["ni"]],d.labels[g["ni"]],g["wn"])
 state=dict(type="nn",state_dict=model.state_dict(),mu=mu,sd=sd,shrink=sh,b=b,feature=c["family"])
 return state,d.ell+delta+b,dict(steps=int(beststep),shrink=sh,validation_objective=None if not np.isfinite(bestloss) else bestloss)
def save_model(path,state):
 if state["type"]=="nn":torch.save(state,path)
 else:np.savez_compressed(path,**state)
def cv(args):
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 d=Dataset("fit",args.allow_partial);d.prepare()
 allrows=[];foldstates={}
 predfile=gzip.open(OUT/"cv_predictions.csv.gz","wt",newline="")
 pw=csv.DictWriter(predfile,fieldnames=["id","method","fold","score"]);pw.writeheader()
 for fold in range(5):
  cfg=json.loads((FOLDS/("fold"+str(fold))/"configuration.json").read_text())
  train=set(cfg["train_sources"]);val=set(cfg["validation_sources"])
  g=index_groups(d,train);vg=index_groups(d,val)
  rawm=measure(d,d.ell,vg)
  for family in FAMILIES:
   x=d.input(family,fold)
   for budget in BUDGETS:
    for ap in (0.,1.):
     for clean in (0.,.1):
      c=tagged_config(family,budget,ap,clean);name=candidate_name(c)
      resultfile=OUT/"cv"/("fold"+str(fold))/(name+".json")
      predcache=ROOT/"training_cache/cv"/("fold"+str(fold))/(name+".npz")
      if resultfile.exists() and predcache.exists():
       rec=json.loads(resultfile.read_text())
       with np.load(predcache) as a:scores=a["score"]
      else:
       tt=time.time()
       state,scores,info=fit_linear(d,x,g,c) if family=="label_linear" else fit_nn(d,x,g,c,11,vg)
       rec=dict(config=c,name=name,fold=fold,seconds=time.time()-tt,metrics=measure(d,scores,vg),raw_metrics=rawm,fit_info=info,
        training_sources=sorted(train),validation_sources=sorted(val),data_origin="new_real_training_expanded_fit_sources")
       dump(resultfile,rec);predcache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(predcache,score=scores)
      allrows.append(rec)
      for i in np.flatnonzero(np.isin(d.sources,list(val))):
       pw.writerow(dict(id=d.ids[i],method=name,fold=fold,score=float(scores[i])))
      dump(OUT/"cv_status.json",dict(stage="cross_validation",completed=len(allrows),total=5*48,seconds=time.time()-started,last=name,fold=fold))
 predfile.close()
 chosen=[];summary=[]
 for family in FAMILIES:
  for budget in BUDGETS:
   candidates=[]
   for ap in (0.,1.):
    for clean in (0.,.1):
     rr=[r for r in allrows if r["config"]["family"]==family and r["config"]["budget"]==budget and r["config"]["lambda_AP"]==ap and r["config"]["lambda_clean"]==clean]
     m={k:float(np.mean([r["metrics"][k] for r in rr])) for k in ("natural_accuracy","interference_accuracy","CE","AP_VR")}
     baseline=float(np.mean([r["raw_metrics"]["natural_accuracy"] for r in rr]))
     entry=dict(config=rr[0]["config"],name=rr[0]["name"],metrics=m,raw_natural_accuracy=baseline,
       eligible=m["natural_accuracy"]>=baseline-.005,epochs=int(np.median([r["fit_info"]["steps"] for r in rr])) if family!="label_linear" else None)
     candidates.append(entry);summary.append(entry)
   # Same source folds, explicit accuracy protection; ties use held-out CE, no external dev labels.
   best=max(candidates,key=lambda r:(r["eligible"],r["metrics"]["interference_accuracy"],-r["metrics"]["CE"],r["metrics"]["AP_VR"]))
   chosen.append(best)
 dump(OUT/"cv_selection.json",dict(chosen=chosen,all_candidates=summary,seeds=[11],
  rule="Mean of five source-fold natural exact-path success drop <=.005 preferred; then interference exact-path success (variant mean then case mean then source mean), then lower balanced CE, then AP VR.",
  early_stopping="Every50 steps, training-config validation objective; patience5 checks, max1000; final refit median best steps.",
  source_fold_configurations=[str(FOLDS/("fold"+str(i))/"configuration.json") for i in range(5)],
  evidence="Real expanded fit gold/A examples plus old fit C/F135 constraints. All preprocessing per fold uses its existing training-source support and new training-source input standardization.",
  seconds=time.time()-started))
 csvout(OUT/"cv_summary.csv",[dict(method=r["name"],**r["config"],**r["metrics"],eligible=r["eligible"],epochs=r["epochs"]) for r in summary])
 dump(OUT/"cv_done.json",dict(status="complete",seconds=time.time()-started,candidates=48,fold_fits=len(allrows)))
def ridge_state(d,x,g,budget=None):
 mu=np.zeros(x.shape[1]);sd=np.ones(x.shape[1]);X=x[g["ar"]]-x[g["al"]];y=d.ell[g["ar"]]-d.ell[g["al"]]
 # Original AP-only ridge with source-equal loss on expanded fit.
 Q=X.T@(g["wa"][:,None]*X)+REG*np.eye(x.shape[1]);q=X.T@(g["wa"]*y)
 w=-np.linalg.solve(Q,q)
 if budget is not None:
  E=x[g["cr"]]-x[g["cl"]];delta=budget*g["scale"]
  opt=minimize(lambda v:(float(.5*v@Q@v+q@v),Q@v+q),np.zeros_like(w),jac=True,method="SLSQP",
   constraints=[LinearConstraint(E,-delta,delta)] if len(E) else [],options=dict(maxiter=500,ftol=1e-10))
  if not opt.success:raise RuntimeError("AP ridge constrained: "+opt.message)
  w=opt.x
 corr=x@w;b=calibrate(d.ell[g["ni"]]+corr[g["ni"]],d.labels[g["ni"]],g["wn"])
 return dict(type="linear",feature="label_linear",w=w,mu=mu,sd=sd,shrink=1.,b=b)
def fit(args):
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 selection=json.loads((OUT/"cv_selection.json").read_text())
 d=Dataset("fit");d.prepare();g=index_groups(d,set(d.sources));records=[]
 sup=support();np.savez_compressed(OUT/"support.npz",**sup)
 rq=np.linalg.qr(np.random.default_rng(11).normal(size=(len(sup["mu"]),16)),mode="reduced")[0].astype(np.float32)
 np.savez_compressed(OUT/"random_basis.npz",Q=rq)
 def entry(name,family,architecture,file=None,budget=None,parameters=0,**extra):
  r=dict(name=name,family=family,architecture=architecture,model_file=str(file) if file else None,budget=budget,parameters=parameters,main=True,
   evidence_budget_absolute=None if budget is None or architecture=="frozen_optimized" else float(budget*g["scale"]),
   evidence_budget_kind="historical_dev_RMS_tier" if architecture=="frozen_optimized" else "max_fit_CF_difference" if budget is not None else "none")
  r.update(extra);records.append(r);return r
 entry("raw","baseline","raw",parameters=0)
 b=calibrate(d.ell[g["ni"]],d.labels[g["ni"]],g["wn"])
 file=OUT/"bias.npz";np.savez_compressed(file,type="bias",b=b);entry("expanded_fit_bias","baseline","bias",file,parameters=1)
 xx=d.ell[g["ni"]];yy=d.labels[g["ni"]];ww=g["wn"]
 def afun(v):
  s=v[0]*xx+v[1];ee=expit(s)-yy
  return scalar_ce(s,yy,ww),np.array([np.sum(ww*ee*xx),np.sum(ww*ee)])
 opt=minimize(afun,[1.,0.],jac=True,method="BFGS",options=dict(gtol=1e-9))
 file=OUT/"affine.npz";np.savez_compressed(file,type="affine",a=opt.x[0],b=opt.x[1])
 entry("expanded_fit_affine","baseline","affine",file,parameters=2)
 for budget in [None]+BUDGETS:
  state=ridge_state(d,d.input("label_linear","main"),g,budget)
  name="AP_ridge_"+("unbounded" if budget is None else str(budget));file=OUT/(name+".npz");save_model(file,state)
  entry(name,"baseline","AP_ridge",file,budget,17,objective="source-equal AP residual MSE + .01 L2; no classification labels in coefficients; b fit natural only")
 for pick in selection["chosen"]:
  c=pick["config"];family=c["family"];budget=c["budget"];x=d.input(family,"main")
  if family=="label_linear":
   state,scores,info=fit_linear(d,x,g,c);name="label_linear_b"+str(budget);file=OUT/(name+".npz");save_model(file,state)
   entry(name,"trace",family,file,budget,17,config=c,fit_info=info,seeds=[])
  else:
   members=[]
   for seed in SEEDS:
    tt=time.time();state,scores,info=fit_nn(d,x,g,c,seed,epochs=pick["epochs"])
    name=family+"_b"+str(budget)+"_seed"+str(seed);file=OUT/(name+".pt");save_model(file,state)
    members.append(dict(file=str(file),seed=seed,coefficient=1/3))
    entry(name,"trace" if family=="trace_nn" else "baseline",family,file,budget,610,main=False,config=c,fit_info=info,seed=seed,
     effective_parameters=98 if family=="score_nn" else 610,seconds=time.time()-tt)
   # Independent natural-label calibration on the equal-weight mean correction.
   modelnet=SmallNet();all_delta=[]
   for m in members:
    state=torch.load(m["file"],map_location="cpu",weights_only=False);modelnet.load_state_dict(state["state_dict"]);modelnet.eval()
    with torch.no_grad():v=modelnet(torch.tensor((x-state["mu"])/state["sd"],dtype=torch.float32)).numpy().astype(float)*state["shrink"]
    all_delta.append(v)
   dc=np.mean(all_delta,axis=0);gain=shrink_to_fit(dc,g,budget);dc*=gain
   bb=calibrate(d.ell[g["ni"]]+dc[g["ni"]],d.labels[g["ni"]],g["wn"])
   name=family+"_ensemble_b"+str(budget);file=OUT/(name+".json")
   dump(file,dict(type="ensemble",members=members,b=bb,shrink=gain,feature=family))
   entry(name,"trace" if family=="trace_nn" else "baseline",family+"_ensemble",file,budget,3*609+1,
    config=c,seeds=SEEDS,effective_parameters=3*97+1 if family=="score_nn" else 3*609+1,
    calibration="Mean of three seed corrections; ensemble b fit only expanded fit natural labels")
 # Preserve the old strict/balanced correction exactly. They require residual64 inputs.
 for name in ("optimized_strict_fit_calibration","optimized_balanced_fit_calibration"):
  entry("frozen_"+("strict" if "strict" in name else "balanced"),"baseline","frozen_optimized",parameters=38786,new_training_parameters=0,
   frozen_name=name,manifest="/home/gcx/aster_next_20261109/frozen_methods.json",budget=.05 if "strict" in name else .1,
   provenance="Existing frozen historical model, not trained or selected by this run")
 dump(OUT/"models.json",dict(models=records,fit_rows=len(d.ids),natural_rows=int(d.natural.sum()),interference_rows=int(d.interference.sum()),
  fit_AP_pairs=len(g["al"]),legacy_fit_CF_pairs=len(g["cl"]),fit_sources=sorted(set(d.sources)),fit_AP_RMS=g["scale"],
  seeds=SEEDS,loss=".5 hierarchical-equal natural CE + .5 hierarchical-equal A/P CE + lambda_AP hierarchical-equal corrected AP MSE / fit_AP_RMS^2 + lambda_clean hierarchical-equal original-basecorrect correction MSE / fit_AP_RMS^2 + .01 mean parameter square",
  loss_averaging="sources equal; cases within source equal; variants within case equal; then nodes. Legacy C/F constraints source-equal.",
  evidence_constraint="Linear hard SLSQP |delta_CF|<=budget*fit_AP_RMS; NN penalty100 for violations then max-fit shrink; final b cancels in pair differences.",
  source_support="Final frozen old selected51/S16/N16 and fit-only original coordinate normalization; fold-specific precomputed source-train support during CV.",
  model_selection="Lambdas/epochs chosen only five expanded-fit source folds; no dev/final tasks read during cv/fit.",seconds=time.time()-started))
 dump(OUT/"fit_done.json",dict(status="complete",seconds=time.time()-started,models=len(records)))
class Models:
 def __init__(self,directory):
  self.directory=Path(directory);self.registry=json.loads((self.directory/"models.json").read_text());self.entries={r["name"]:r for r in self.registry["models"]}
  self.names=list(self.entries);self.support=dict(np.load(self.directory/"support.npz"));self.q=np.load(self.directory/"random_basis.npz")["Q"]
  self.states={};self.frozen=None
  for r in self.entries.values():
   file=r.get("model_file")
   if not file:continue
   p=Path(file)
   if not p.exists():p=self.directory/p.name
   if p.suffix==".pt":
    state=torch.load(p,map_location="cpu",weights_only=False);net=SmallNet();net.load_state_dict(state["state_dict"]);net.eval();state["net"]=net;self.states[r["name"]]=state
   elif p.suffix==".npz":
    with np.load(p) as a:self.states[r["name"]]={k:a[k] for k in a.files}
   else:self.states[r["name"]]=json.loads(p.read_text())
  if any(r["architecture"]=="frozen_optimized" for r in self.entries.values()):
   file=REPO/"reports/aster_cancel_story_20261109/meeting/test_expansion/inference/frozen.py"
   spec=importlib.util.spec_from_file_location("trace_frozen_adapter",file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   self.frozen=mod.FrozenMethods()
 def _features(self,ell,heads,head_ids):
  scalar=np.ndim(ell)==0;e=np.atleast_1d(np.asarray(ell,float));h=np.asarray(heads)
  if scalar:h=h[None,...]
  if len(h)!=len(e):raise ValueError("Head batch and score batch lengths differ")
  raw=np.stack([select_raw(one,head_ids,self.support["heads"]) for one in h])
  std=(raw-self.support["mu"])/self.support["scale"]
  t=std@self.support["N"];rt=std@self.q
  return scalar,e,raw,std,t,rt
 def predict_all(self,ell,heads,head_ids=None,residual=None,names=None):
  scalar,e,raw,std,t,rt=self._features(ell,heads,head_ids);out={};cache={}
  def corrected(name):
   if name in cache:return cache[name]
   entry=self.entries[name];arch=entry["architecture"];s=self.states.get(name)
   if arch=="raw":v=e.copy()
   elif arch=="bias":v=e+float(s["b"])
   elif arch=="affine":v=e*float(s["a"])+float(s["b"])
   elif arch=="frozen_optimized":
    if residual is None:raise ValueError("Historical frozen optimized baselines require residual activations")
    res=np.asarray(residual,float)
    if res.ndim==1:res=res[None,:]
    fs=self.frozen;basis=fs.residual_basis
    ts=np.concatenate([t,std@fs.support["S"]],axis=1).astype(np.float32)
    hp=((res-basis["mu"])/basis["scale"])@basis["R"]
    q=np.column_stack([e/10,np.abs(e)/10,np.tanh(e/4)]).astype(np.float32)
    av=dict(r16=t.astype(np.float32),r32=ts,residual64=hp.astype(np.float32),q=q)
    ens=fs.manifest["methods"][entry["frozen_name"]];dc=np.zeros(len(e))
    for member in ens.get("members",[ens]):
     key="frozen|"+member["model_file"]
     if key not in cache:
      obj=fs.models[member["model_file"]];cfg=obj["cfg"]
      if obj["kind"]=="neural":x=av[cfg["feature"]]
      elif cfg["family"]=="combined_score":x=np.concatenate([av["r32"],av["residual64"],av["q"]],axis=1)
      elif cfg["family"]=="score_only":x=av["q"]
      else:raise ValueError("Unsupported frozen input "+cfg["family"])
      with torch.no_grad():cache[key]=obj["model"](torch.tensor((x-obj["mu"])/obj["scale"],dtype=torch.float32)).numpy().astype(float)
     dc+=member.get("coefficient",1.)*cache[key]
    v=e+ens.get("gain",1.)*dc+ens["b"]
   elif s.get("type")=="ensemble":
    dc=np.zeros(len(e))
    for member in s["members"]:
     target=Path(member["file"]).stem
     member_s=self.states[target];x=np.column_stack([rt if member_s["feature"]=="random_nn" else np.zeros_like(t) if member_s["feature"]=="score_nn" else t,e])
     with torch.no_grad():delta=member_s["net"](torch.tensor((x-member_s["mu"])/member_s["sd"],dtype=torch.float32)).numpy().astype(float)*member_s["shrink"]
     dc+=member["coefficient"]*delta
    v=e+dc*s["shrink"]+s["b"]
   else:
    family=str(s["feature"])
    x=t if family=="label_linear" else np.column_stack([rt if family=="random_nn" else np.zeros_like(t) if family=="score_nn" else t,e])
    z=(x-s["mu"])/s["sd"]
    if str(s["type"])=="linear":dc=z@s["w"]
    else:
     with torch.no_grad():dc=s["net"](torch.tensor(z,dtype=torch.float32)).numpy().astype(float)
    v=e+dc*float(s["shrink"])+float(s["b"])
   cache[name]=v;return v
  for name in names or self.names:
   v=corrected(name);out[name]=float(v[0]) if scalar else v
  return out
 def correct(self,name,ell,heads,head_ids=None,residual=None):
  return self.predict_all(ell,heads,head_ids,residual,names=[name])[name]
def load_models(model_dir=OUT):return Models(model_dir)
def predict_to_jsonl(models,tasks,scores,output):
 """Write gzip CSV id/method/score without touching other splits."""
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 opener=gzip.open if output.suffix==".gz" else open
 with opener(output,"wt",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["id","method","score"]);w.writeheader()
  for lo in range(0,len(tasks),128):
   tt=tasks[lo:lo+128];rr=[scores[str(t.get("id",t.get("task_id")))] for t in tt]
   loaded=[]
   for r in rr:
    with np.load(r["feature"]) as a:
     h=a["heads"].copy() if "heads" in a else a["selected_heads"].copy()
     hi=a["head_ids"].copy() if "head_ids" in a else None
     selected=select_raw(h,hi,models.support["heads"]).reshape(51,128)
     loaded.append((selected,a["residual"].copy()))
   h=np.stack([v[0] for v in loaded]);res=np.stack([v[1] for v in loaded])
   out=models.predict_all(np.array([r["ell"] for r in rr]),h,models.support["heads"],res)
   for j,t in enumerate(tt):
    for name,value in out.items():w.writerow(dict(id=str(t.get("id",t.get("task_id"))),method=name,score=float(value[j])))
def predict(args):
 d=Dataset(args.split,args.allow_partial);models=load_models(args.model_dir)
 output=args.output or OUT/(args.split+"_predictions.csv.gz")
 predict_to_jsonl(models,d.tasks,d.scores,output)
 dump(OUT/(args.split+"_prediction_done.json"),dict(status="complete",rows=len(d.tasks),models=len(models.names),output=str(output)))
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=["cv","fit","predict"],required=True)
 p.add_argument("--split",choices=["fit","dev","final"],default="dev");p.add_argument("--allow-partial",action="store_true")
 p.add_argument("--model-dir",type=Path,default=OUT);p.add_argument("--output",type=Path)
 args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.stage=="cv":cv(args)
 elif args.stage=="fit":fit(args)
 else:predict(args)
if __name__=="__main__":main()
