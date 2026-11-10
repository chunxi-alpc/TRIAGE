"""Refit only the neural probe; preserves all completed random-control results."""
import csv,gzip,json,os,time
import numpy as np
from run import HERE,Data,project,arr_pairs,metrics,dump,train_nn,finalize_metrics
started=time.time()
d=Data()
fit,fl,fr,fy=arr_pairs(d,("A","P"),"fit")
dev,dl,dr,dy=arr_pairs(d,("A","P"),"dev")
am=np.array([p["kind"]=="A" for p in dev])
fit_src=np.array([p["source_id"] for p in fit])
raw=d.raw(d.heads)
with np.load(HERE/"selected_support_r16.npz") as a:sup={k:a[k] for k in a.files}
z=project(raw,sup);xmain=z[fr]-z[fl];xdev=z[dr]-z[dl]
old=list(csv.DictReader((HERE/"metrics.csv").open()))
allmetrics=[]
int_fields={"seed","parameters","n","requested_rank","effective_rank","head_count"}
for r in old:
 if r["method"]=="mlp16":continue
 out={}
 for k,v in r.items():
  if v=="":continue
  if k in int_fields:out[k]=int(v)
  elif k not in ("method","group","data_origin","rank_design"):out[k]=float(v)
  else:out[k]=v
 allmetrics.append(out)
temp=HERE/"pair_predictions.refit.csv.gz"
predfile=gzip.open(temp,"wt",newline="")
writer=None
with gzip.open(HERE/"pair_predictions.csv.gz","rt",newline="") as f:
 reader=csv.DictReader(f)
 writer=csv.DictWriter(predfile,fieldnames=reader.fieldnames);writer.writeheader()
 for r in reader:
  if r["method"]!="mlp16":writer.writerow(r)
def record(name,seed,pf,pd,seconds,params,extra=None):
 for split,ps,y,pred in [("fit",fit,fy,pf),("dev",dev,dy,pd)]:
  for p,yy,yp in zip(ps,y,pred):
   writer.writerow(dict(method=name,seed=seed,split=split,pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],target=float(yy),prediction=float(yp)))
 for group,mask in [("A",am),("AP",np.ones(len(dy),bool))]:
  allmetrics.append(dict(method=name,seed=seed,group=group,parameters=params,fit_seconds=seconds,
   data_origin="real_cache_new_readout_training",**metrics(dy[mask],pd[mask]),**(extra or {})))
train_nn(fit_src,xmain,fy,xdev,raw,fl,fr,d,record)
predfile.close()
os.replace(temp,HERE/"pair_predictions.csv.gz")
finalize_metrics(allmetrics)
dump("nn_refit.json",dict(status="complete",seconds=time.time()-started,
 procedure="Internal hyperparameter selection reestimates mu/scale/S/N solely on internal training sources; complete fit sources used for final neural models. Head selector fixed from fit sources."))

