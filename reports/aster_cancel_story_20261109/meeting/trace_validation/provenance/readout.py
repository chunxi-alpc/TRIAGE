"""Fit-only projections and binary-score correction; frozen features, no hidden edits."""
import argparse,pickle,time
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as TF
from .common import *
from .features import Features,basis,orth_nuisance
from .mechanisms import supports
from .statistics import pair_metrics
TRAIN_KINDS=TASK_KINDS
def project(raw,support,component="both"):
 x=(raw-support["mu"])/support["scale"]
 if component=="ordinary":return (x@support["O"]).astype(np.float32)
 parts=[]
 if component in ("signal","both"):parts.append(x@support["S"])
 if component in ("nuisance","both"):parts.append(x@support["N"])
 return np.concatenate(parts,axis=-1).astype(np.float32) if parts else np.zeros((len(raw),0),np.float32)
def optimize(ell,x,labels,pairs,correct,reg,weights,mode):
 torch.set_num_threads(4)
 e=torch.tensor(ell,dtype=torch.float32);z=torch.tensor(x,dtype=torch.float32);y=torch.tensor(labels,dtype=torch.float32)
 w=torch.zeros(z.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True)
 temp=torch.zeros((),requires_grad=True)
 params=[temp] if mode=="temperature" else [b,temp] if mode=="logistic" else [b,w] if z.shape[1] else [b]
 opt=torch.optim.LBFGS(params,max_iter=90,line_search_fn="strong_wolfe")
 groups={}
 for k in ("A","P","C","F","F0"):
  group=[(a,c,sgn) for kind,a,c,sgn in pairs if kind==k]
  if group:groups[k]=(torch.tensor([q[0] for q in group]),torch.tensor([q[1] for q in group]),torch.tensor([q[2] for q in group]))
 mask=torch.tensor(correct,dtype=torch.bool);base=TF.sigmoid(e).clamp(1e-6,1-1e-6)
 def logits():
  if mode=="temperature":return e*torch.exp(temp.clamp(-5,5))
  if mode=="logistic":return e*(1+temp)+b
  return e+b+(z@w if z.shape[1] else 0)
 def closure():
  opt.zero_grad();s=logits();loss=TF.binary_cross_entropy_with_logits(s,y)
  if mode not in ("bias","temperature","logistic"):
   if w.numel():loss=loss+reg*w.square().mean()
   for k,(a,c,sgn) in groups.items():
    if k in ("A","P","F0"):
     weight=weights["invariance"] if k=="F0" or weights.get("AP",True) else 0
     loss=loss+weight*((s[c]-s[a]).square().mean())
    elif weights.get("CF",True):loss=loss+weights["sensitivity"]*TF.relu(1-sgn*(s[c]-s[a])).square().mean()
   if mask.any():
    p=base[mask];logp=TF.logsigmoid(s[mask]);logq=TF.logsigmoid(-s[mask])
    kl=p*(p.log()-logp)+(1-p)*((1-p).log()-logq)
    loss=loss+weights["preservation"]*kl.mean()
  loss.backward();return loss
 opt.step(closure)
 return dict(w=w.detach().numpy(),b=float(b.detach()),temperature=float(temp.detach()),mode=mode,parameters=sum(p.numel() for p in params))
def predict(model,ell,raw=None):
 ell=np.asarray(ell)
 mode=model["fit"]["mode"];fit=model["fit"]
 if mode=="temperature":return ell*np.exp(np.clip(fit["temperature"],-5,5))
 if mode=="logistic":return ell*(1+fit["temperature"])+fit["b"]
 if mode=="full":return ell
 if raw is None or len(fit["w"])==0:return ell+fit["b"]
 z=project(raw,model["support"],model["component"])
 return ell+fit["b"]+z@fit["w"]
def train(model):
 torch.set_num_threads(4);started=time.time();f=Features(model);sel=read(ROOT/model/"selection.json")
 h2=read(ROOT/model/"h2.json");h3=read(ROOT/model/"h3.json")
 train=[v for v in f.views if v["split"]=="fit" and v["kind"] in TRAIN_KINDS]
 dev=[v for v in f.views if v["split"]=="dev" and v["kind"] in TRAIN_KINDS]
 index={v["view_id"]:i for i,v in enumerate(train)};ell=np.array([f.ell(v["view_id"]) for v in train],np.float32)
 yd=np.array([v["label"] for v in train]);correct=(ell>0)==yd
 ps=[(p["kind"],index[p["left"]],index[p["right"]],p["right_label"]-p["left_label"]) for p in f.pairs if p["left"] in index and p["right"] in index]
 devpairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="dev"]
 baseline_scores={v["view_id"]:f.ell(v["view_id"]) for v in f.views}
 root=ROOT/model;out=root/"readouts";out.mkdir(exist_ok=True)
 records=[];models={};support_cache={};matrix_cache={}
 def save(name,obj,devscore):
  obj["name"]=name
  with (out/(name+".pkl")).open("wb") as file:pickle.dump(obj,file)
  metrics=pair_metrics(dev,devpairs,devscore)
  records.append(dict(name=name,parameters=obj["fit"]["parameters"],dev=metrics,component=obj.get("component"),
   rank=obj.get("rank"),regularization=obj.get("regularization"),weights=obj.get("weights")))
  models[name]=obj
  lines(root/"readout_scores"/(name+"_dev.jsonl"),[dict(view_id=k,ell=float(v)) for k,v in devscore.items()])
  write(root/"readout_candidates.json",records)
 def candidate(name,headset,rank,component,reg,weights,support_type="paired"):
  allowed=h2.get("by_rank",{}).get(str(rank),h2).get("nuisance_enabled",False)
  if name.startswith("cr_") and not allowed:
   component="signal" if component=="both" else "bias" if component=="nuisance" else component
  key=(tuple(headset),rank)
  if key not in support_cache:
   saved=root/("supports_r"+str(rank)+".npz")
   if list(headset)==sel["heads"] and saved.exists():
    with np.load(saved) as arrays:support_cache[key]={k:arrays[k] for k in arrays.files}
   else:support_cache[key]=supports(f,headset,rank)
  support=dict(support_cache[key])
  if tuple(headset) not in matrix_cache:matrix_cache[tuple(headset)]=(f.matrix(train,headset),f.matrix(dev,headset))
  fitraw,devraw=matrix_cache[tuple(headset)]
  total=support["S"].shape[1]+support["N"].shape[1]
  if support_type=="random":
   rng=np.random.default_rng(SEED)
   q,_=np.linalg.qr(rng.normal(size=(len(support["mu"]),max(total,1))))
   ns=support["S"].shape[1];support["S"]=q[:,:ns].astype(np.float32);support["N"]=q[:,ns:total].astype(np.float32)
  if component=="ordinary":
   target_dim=(support["S"].shape[1] if active in ("signal","both") else 0)+(support["N"].shape[1] if active in ("nuisance","both") and allowed else 0)
   support["O"]=basis((fitraw-support["mu"])/support["scale"],target_dim,trim=False) if target_dim else np.empty((fitraw.shape[1],0),np.float32)
  x=project(fitraw,support,component)
  fitted=optimize(ell,x,yd,ps,correct,reg,weights,"linear")
  obj=dict(fit=fitted,support=support,heads=headset,component=component,rank=rank,regularization=reg,weights=weights)
  ds=predict(obj,np.array([f.ell(v["view_id"]) for v in dev]),devraw)
  save(name,obj,{v["view_id"]:float(s) for v,s in zip(dev,ds)})
 def scalar(name):
  fit=optimize(ell,np.empty((len(ell),0)),yd,[],correct,0,{},name)
  obj=dict(fit=fit,component="bias",heads=[])
  ds=predict(obj,np.array([f.ell(v["view_id"]) for v in dev]))
  save(name,obj,{v["view_id"]:float(s) for v,s in zip(dev,ds)})
 save("full",dict(fit=dict(mode="full",w=np.empty(0),b=0,parameters=0)),{v["view_id"]:f.ell(v["view_id"]) for v in dev})
 for name in ("bias","temperature","logistic"):scalar(name)
 weights=dict(invariance=1.,sensitivity=1.,preservation=1.,AP=True,CF=True)
 # Main components follow diagnostic gates; exploratory component models remain separately labelled.
 active="both" if h2["nuisance_enabled"] and h3["signal_claim_enabled"] else "nuisance" if h2["nuisance_enabled"] else "signal" if h3["signal_claim_enabled"] else "bias"
 if model=="medgemma":
  selected=read(ROOT/"qwen/development_selection.json")["aster_cr"]
  q=next(r for r in read(ROOT/"qwen/readout_candidates.json") if r["name"]==selected)
  rank=q.get("rank") or 8;reg=q.get("regularization") or .01
  ww=q.get("weights") or weights
  candidate("cr_confirm",sel["heads"],rank,active,reg,ww)
  candidate("ordinary_confirm",sel["heads"],rank,"ordinary",reg,dict(weights,invariance=0.,sensitivity=0.,preservation=0.))
  write(root/"readout_training.json",dict(seconds=time.time()-started,fit_views=len(train),dev_views=len(dev),
    active_component=active,candidate_count=len(records),confirmation_recipe_from=selected,no_grid_search=True))
  status(model+"_readout",state="complete",candidates=len(records))
  return
 if active=="bias":
  candidate("cr_diagnostic_off",sel["heads"],8,"bias",.01,weights)
  candidate("ordinary_r8",sel["heads"],8,"ordinary",.01,dict(weights,invariance=0.,sensitivity=0.,preservation=0.))
  for name,component,ww,heads,stype in [
   ("nuisance_only","nuisance",weights,sel["heads"],"paired"),
   ("signal_only","signal",weights,sel["heads"],"paired"),
   ("complete_exploratory","both",weights,sel["heads"],"paired"),
   ("ce_only","bias",dict(weights,invariance=0.,sensitivity=0.,preservation=0.),sel["heads"],"paired"),
   ("no_AP","bias",dict(weights,AP=False),sel["heads"],"paired"),
   ("no_CF","bias",dict(weights,CF=False),sel["heads"],"paired"),
   ("random_heads","bias",weights,sel["random_heads"],"paired"),
   ("random_support","bias",weights,sel["heads"],"random")]:
   candidate(name,heads,8,component,.01,ww,stype)
  write(root/"readout_training.json",dict(seconds=time.time()-started,fit_views=len(train),dev_views=len(dev),
    active_component=active,candidate_count=len(records),rank_grid_skipped="Neither diagnostic gate supports a main correction component",
    exploratory_components_retained=True))
  status(model+"_readout",state="complete",candidates=len(records))
  return
 for rank in (4,8,16,"full"):
  for reg in (.001,.01,.1):
   candidate(f"cr_r{rank}_l{reg}",sel["heads"],rank,active,reg,weights)
 # Five core ablations and single-component comparisons at the prespecified rank/regularizer.
 for name,component,ww,heads,stype in [
  ("nuisance_only","nuisance",weights,sel["heads"],"paired"),
  ("signal_only","signal",weights,sel["heads"],"paired"),
  ("complete_exploratory","both",weights,sel["heads"],"paired"),
  ("ce_only",active,dict(weights,invariance=0.,sensitivity=0.,preservation=0.),sel["heads"],"paired"),
  ("no_AP",active,dict(weights,AP=False),sel["heads"],"paired"),
  ("no_CF",active,dict(weights,CF=False),sel["heads"],"paired"),
  ("random_heads",active,weights,sel["random_heads"],"paired"),
  ("random_support",active,weights,sel["heads"],"random")]:
  candidate(name,heads,8,component,.01,ww,stype)
 for rank in (4,8,16,"full"):candidate(f"ordinary_r{rank}",sel["heads"],rank,"ordinary",.01,dict(weights,invariance=0.,sensitivity=0.,preservation=0.))
 for term in ("invariance","sensitivity","preservation"):
  for value in (.1,10.):
   candidate(f"cr_{term}_{value}",sel["heads"],8,active,.01,dict(weights,**{term:value}))
 write(root/"readout_training.json",dict(seconds=time.time()-started,fit_views=len(train),dev_views=len(dev),active_component=active,
  nuisance_enabled=h2["nuisance_enabled"],signal_claim_enabled=h3["signal_claim_enabled"],candidate_count=len(records),
  fitted_projections="fit-only, source-disjoint from development",trainable_budget="actual coefficients; support estimates reported separately"))
 status(model+"_readout",state="complete",candidates=len(records))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");train(p.parse_args().model)
