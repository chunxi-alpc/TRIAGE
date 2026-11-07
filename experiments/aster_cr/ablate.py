"""Core ablations share the selected main configuration and fit data."""
import argparse,pickle
import numpy as np
from .common import *
from .features import Features,basis
from .mechanisms import supports
from .readout import optimize,predict,project,TRAIN_KINDS
from .statistics import pair_metrics
def run(model):
 root=ROOT/model;selected=read(root/"development_selection.json")["aster_cr"]
 with (root/"readouts"/(selected+".pkl")).open("rb") as file:parent=pickle.load(file)
 if "support" not in parent:
  write(root/"ablations.json",dict(state="not_applicable",selected=selected,reason="No correction model passed development path selection"))
  return
 f=Features(model);sel=read(root/"selection.json")
 train=[v for v in f.views if v["split"]=="fit" and v["kind"] in TRAIN_KINDS]
 dev=[v for v in f.views if v["split"]=="dev" and v["kind"] in TRAIN_KINDS]
 ix={v["view_id"]:i for i,v in enumerate(train)}
 ell=np.array([f.ell(v["view_id"]) for v in train],np.float32);y=np.array([v["label"] for v in train])
 ps=[(p["kind"],ix[p["left"]],ix[p["right"]],p["right_label"]-p["left_label"]) for p in f.pairs if p["left"] in ix and p["right"] in ix]
 devpairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="dev"]
 weights=parent["weights"];rank=parent["rank"];reg=parent["regularization"];component=parent["component"]
 recipes=[
 ("ce_only",dict(weights,invariance=0.,sensitivity=0.,preservation=0.),parent["heads"],"paired"),
 ("no_AP",dict(weights,AP=False),parent["heads"],"paired"),
 ("no_CF",dict(weights,CF=False),parent["heads"],"paired"),
 ("random_heads",weights,sel["random_heads"],"paired"),
 ("random_support",weights,parent["heads"],"random"),
 ("ordinary_matched",dict(weights,invariance=0.,sensitivity=0.,preservation=0.),parent["heads"],"ordinary")]
 records=read(root/"readout_candidates.json")
 for name,ww,heads,kind in recipes:
  sup=dict(parent["support"]) if heads==parent["heads"] else supports(f,heads,rank)
  x=f.matrix(train,heads);d=f.matrix(dev,heads);comp=component
  if kind=="random":
   ns=sup["S"].shape[1];nn=sup["N"].shape[1];rng=np.random.default_rng(SEED)
   q,_=np.linalg.qr(rng.normal(size=(x.shape[1],max(ns+nn,1))))
   sup["S"]=q[:,:ns].astype(np.float32);sup["N"]=q[:,ns:ns+nn].astype(np.float32)
  if kind=="ordinary":
   comp="ordinary";dim=len(parent["fit"]["w"])
   sup["O"]=basis((x-sup["mu"])/sup["scale"],dim,trim=False) if dim else np.empty((x.shape[1],0),np.float32)
  fit=optimize(ell,project(x,sup,comp),y,ps,(ell>0)==y,reg,ww,"linear")
  obj=dict(name=name,fit=fit,support=sup,heads=heads,component=comp,rank=rank,regularization=reg,weights=ww,ablation_parent=selected)
  with (root/"readouts"/(name+".pkl")).open("wb") as file:pickle.dump(obj,file)
  values=predict(obj,np.array([f.ell(v["view_id"]) for v in dev]),d)
  scores={v["view_id"]:float(s) for v,s in zip(dev,values)}
  lines(root/"readout_scores"/(name+"_dev.jsonl"),[dict(view_id=k,ell=s) for k,s in scores.items()])
  record=dict(name=name,parameters=fit["parameters"],dev=pair_metrics(dev,devpairs,scores),component=comp,rank=rank,regularization=reg,weights=ww,ablation_parent=selected)
  records=[r for r in records if r["name"]!=name]+[record]
 write(root/"readout_candidates.json",records)
 write(root/"ablations.json",dict(state="complete",selected=selected,rank=rank,regularization=reg,component=component,recipes=[r[0] for r in recipes]))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");run(p.parse_args().model)
