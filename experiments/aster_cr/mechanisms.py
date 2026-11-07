"""Fit-only localization, cross-source selectivity, and held-out information probes."""
import argparse,math
import numpy as np
from collections import defaultdict
from .common import *
from .features import Features,basis,orth_nuisance
def weighted_pairs(f,kind,split="fit"):
 ps=[p for p in f.pairs if p["kind"]==kind and p["split"]==split]
 result=[]
 for p in ps:
  ls,rs=f.ell(p["left"]),f.ell(p["right"])
  wrong=int((ls>0)!=bool(p["left_label"]) or (rs>0)!=bool(p["right_label"]))
  boundary=int(min(abs(ls),abs(rs))<1)
  result.append((p,1+wrong+boundary))
 return result
def localize(model):
 from .engine import Engine
 from experiments.aster_closed_form.m1 import _effects
 import torch
 f=Features(model); eng=Engine(model);oldfile=Path("/home/gcx/aster_existing_data_20261106/m1/selection.json") if model=="qwen" else Path("/home/gcx/medgemma_m1_repair_20261106/selection/selection.json")
 old=read(oldfile).get("H",[]);k=len(old) or 51
 ranks={};fit_samples={};allattr={}
 for kind in ("A","P","C","F"):
  attr=np.zeros(eng.num_heads);den=0
  wp=weighted_pairs(f,kind)
  for p,w in wp:
   left=f.get(p["left"]);right=f.get(p["right"])
   grad=left.get("grads")
   if grad is None:raise ValueError("Fit gradient missing for localization")
   d=(right["heads"]-left["heads"])*grad
   attr+=w*np.abs(d.reshape(eng.num_heads,-1).sum(-1));den+=w
  attr/=max(den,1);allattr[kind]=attr
  candidate=np.argsort(-attr)[:64].tolist()
  # Reuse the established bidirectional donor intervention implementation.
  # Bounded, source-balanced fit subset; no dev ranking or normalization grid.
  ordered=sorted(wp,key=lambda pw:(-pw[1],pw[0]["pair_id"]))
  selected=[];seen=set()
  for p,w in ordered:
   if p["source_id"] in seen:continue
   if abs(f.ell(p["right"])-f.ell(p["left"]))<1e-8:continue
   selected.append((p,w));seen.add(p["source_id"])
   if len(selected)>=12:break
  fit_samples[kind]=[p["pair_id"] for p,w in selected]
  sums=defaultdict(float);weights=0
  for p,w in selected:
   contexts=(eng.prepare(f.index[p["left"]]["prompt"]),eng.prepare(f.index[p["right"]]["prompt"]))
   left=dict(f.get(p["left"]),ell=f.ell(p["left"]));right=dict(f.get(p["right"]),ell=f.ell(p["right"]))
   for h in candidate:
    e=_effects(eng,contexts,left,right,[h],right["ell"]-left["ell"])
    sums[h]+=w*.5*(abs(e["ie_forward"])+abs(e["ie_reverse"]))
    append(ROOT/model/"mechanisms/fit_exact.jsonl",dict(pair_id=p["pair_id"],kind=kind,head=h,weight=w,**e))
   weights+=w;del contexts
  ranks[kind]=sorted(candidate,key=lambda h:(-sums[h]/max(weights,1),h))
 # Alternate interference and signal rankings to preserve both feature types.
 nsort=sorted(set(ranks["A"]+ranks["P"]),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ("A","P")))
 ssort=sorted(set(ranks["C"]+ranks["F"]),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ("C","F")))
 selected=[]
 for i in range(max(len(nsort),len(ssort))):
  for ranking in (nsort,ssort):
   if i<len(ranking) and ranking[i] not in selected:selected.append(ranking[i])
  if len(selected)>=k:break
 selected=selected[:k]
 rng=np.random.default_rng(SEED);random=rng.choice(eng.num_heads,len(selected),replace=False).tolist()
 write(ROOT/model/"selection.json",dict(heads=selected,old_heads=old,random_heads=random,ranks=ranks,
     exact_fit_pairs=fit_samples,weights="1 + original_error + min_abs_margin_below_1",ranking_split="fit"))
 status(model+"_localization",state="complete",selected_heads=len(selected))
def supports(f,heads,rank):
 fit=[v for v in f.views if v["split"]=="fit" and v["variant"]=="full"]
 x=f.matrix(fit,heads);mu=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-5)
 nd=[];sd=[]
 for p in f.pairs:
  if p["split"]!="fit" or p["kind"] not in ("A","P","C","F"):continue
  d=(f.heads(p["right"],heads)-f.heads(p["left"],heads))/scale
  (nd if p["kind"] in ("A","P") else sd).append(d)
 dim=len(mu)
 S=basis(np.asarray(sd).reshape(-1,dim),rank)
 N=orth_nuisance(S,basis(np.asarray(nd).reshape(-1,dim),rank))
 direction=np.mean(nd,axis=0) if nd else np.zeros(dim)
 if N.shape[1]:direction=(direction@N)@N.T
 return dict(heads=np.array(heads),mu=mu,scale=scale,S=S,N=N,direction=direction)
def h2(model):
 import torch
 from sklearn.linear_model import Ridge
 from sklearn.metrics import r2_score
 from .engine import Engine
 f=Features(model);sel=read(ROOT/model/"selection.json");h=sel["heads"];sup=supports(f,h,8)
 root=ROOT/model;np.savez_compressed(root/"supports_h2.npz",**sup)
 eng=Engine(model);records=[]
 for p in f.pairs:
  if p["split"]!="dev" or p["kind"] not in ("A","P","C","F"):continue
  a,b=p["left"],p["right"];za=f.heads(a);zb=f.heads(b);d=(zb[h]-za[h]).reshape(-1)/sup["scale"]
  low=((d@sup["N"])@sup["N"].T)*sup["scale"]
  signal=((d@sup["S"])@sup["S"].T)*sup["scale"]
  context=eng.prepare(f.index[a]["prompt"]);base=f.ell(a)
  operations=dict(full_head=zb[h]-za[h],low_rank=low.reshape(len(h),-1),signal=signal.reshape(len(h),-1))
  rr=dict(pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],group_id=p["group_id"],total_effect=f.ell(b)-base,
    original_correct=int((base>0)==bool(p["left_label"])),label=p["left_label"])
  for name,delta in operations.items():
   ell=eng.run(context,replacements={j:za[j]+delta[i] for i,j in enumerate(h)},capture=False)["ell"]
   rr[name]=ell-base;rr[name+"_correct"]=int((ell>0)==bool(p["left_label"]))
  rh=sel["random_heads"]
  ell=eng.run(context,replacements={j:zb[j] for j in rh},capture=False)["ell"];rr["random_heads"]=ell-base
  if sel["old_heads"]:
   ell=eng.run(context,replacements={j:zb[j] for j in sel["old_heads"]},capture=False)["ell"];rr["old_heads"]=ell-base
  if p["kind"] in ("C","F"):
   fixed=sup["direction"]*sup["scale"];fixed=fixed.reshape(len(h),-1)
   l=eng.run(context,replacements={j:za[j]+fixed[i] for i,j in enumerate(h)},capture=False)["ell"]
   other=eng.prepare(f.index[b]["prompt"])
   r=eng.run(other,replacements={j:zb[j]+fixed[i] for i,j in enumerate(h)},capture=False)["ell"]
   rr["same_nuisance_direction_difference"]=(r-l)-(f.ell(b)-base);del other
  append(root/"mechanisms/h2_dev.jsonl",rr);records.append(rr);del context
 # Predict unseen-source interference score differences, fit only.
 tr=[];ty=[];dx=[];dy=[]
 for p in f.pairs:
  if p["kind"] not in ("A","P"):continue
  diff=(f.heads(p["right"],h)-f.heads(p["left"],h))/sup["scale"]
  z=diff@sup["N"];target=f.ell(p["right"])-f.ell(p["left"])
  if p["split"]=="fit":tr.append(z);ty.append(target)
  elif p["split"]=="dev":dx.append(z);dy.append(target)
 r2=None
 if tr and dx and sup["N"].shape[1] and len(dy)>1:
  estimator=Ridge(alpha=.01).fit(np.stack(tr),ty);r2=float(r2_score(dy,estimator.predict(np.stack(dx))))
 sig=[r for r in records if r["kind"] in ("C","F")]
 leak=float(np.sqrt(np.mean([r["low_rank"]**2 for r in sig]))) if sig else None
 true=float(np.sqrt(np.mean([r["total_effect"]**2 for r in sig]))) if sig else None
 ratio=leak/true if true and leak is not None else None
 gate=bool(r2 is not None and r2>0 and ratio is not None and ratio<=.1)
 write(root/"h2.json",dict(dev_interference_r2=r2,signal_leakage_rms=leak,true_signal_rms=true,leakage_ratio=ratio,
  nuisance_enabled=gate,criterion="R2 > 0 and leakage RMS / true signal RMS <= 0.1",dev_pairs=len(records),
  corrected=sum(not r["original_correct"] and r["low_rank_correct"] for r in records if r["kind"] in ("A","P")),
  new_errors=sum(r["original_correct"] and not r["low_rank_correct"] for r in records if r["kind"] in ("A","P"))))
def h3(model):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import RidgeClassifier
 f=Features(model);sel=read(ROOT/model/"selection.json")
 fit=[v for v in f.views if v["split"]=="fit" and v["variant"]=="full"]
 dev=[v for v in f.views if v["split"]=="dev" and v["variant"]=="full"]
 y=np.array([v["label"] for v in fit]);yd=np.array([v["label"] for v in dev])
 errors=np.array([(f.ell(v["view_id"])>0)!=bool(v["label"]) for v in dev])
 rng=np.random.default_rng(SEED);result={}
 for kind,heads in [("residual",None),("selected_heads",sel["heads"]),("all_heads",None)]:
  x=f.matrix(fit,heads,kind="residual" if kind=="residual" else "heads")
  d=f.matrix(dev,heads,kind="residual" if kind=="residual" else "heads")
  mu=x.mean(0);scale=np.maximum(x.std(0),1e-5);x=(x-mu)/scale;d=(d-mu)/scale
  dim=min(64,len(fit)-1,x.shape[1]);pca=PCA(n_components=dim,svd_solver="randomized",random_state=SEED)
  x=pca.fit_transform(x);d=pca.transform(d)
  clf=RidgeClassifier(alpha=1.).fit(x,y);pred=clf.predict(d)
  shuffled=RidgeClassifier(alpha=1.).fit(x,rng.permutation(y)).predict(d)
  result[kind]=dict(dev_n=len(dev),accuracy=float(np.mean(pred==yd)),shuffle_accuracy=float(np.mean(shuffled==yd)),
    error_nodes=int(errors.sum()),error_accuracy=float(np.mean(pred[errors]==yd[errors])) if errors.any() else None,
    error_shuffle_accuracy=float(np.mean(shuffled[errors]==yd[errors])) if errors.any() else None)
  del x,d,pca
 # Positive decoding evidence is kept separate from the >=100 error-node evidence target.
 r=result["selected_heads"]
 positive=bool(r["error_nodes"]>=100 and r["error_accuracy"]>r["error_shuffle_accuracy"])
 write(ROOT/model/"h3.json",dict(probes=result,signal_claim_enabled=positive,
    error_subset_evidence="sufficient" if r["error_nodes"]>=100 else "insufficient",
    interpretation="Accuracy on held-out sources; insufficient errors do not establish absence of encoded signal."))
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--stage",choices=["localize","h2","h3"],required=True)
 a=p.parse_args();globals()[a.stage](a.model)
if __name__=="__main__":main()
