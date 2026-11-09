"""Add fixed-S16 dimension controls to completed results without refitting random controls."""
import csv,gzip,json,os,time
import numpy as np
from sklearn.linear_model import Ridge
from run import HERE,Data,project,arr_pairs,metrics,dump,finalize_metrics,orth_nuisance
started=time.time();d=Data();raw=d.raw(d.heads)
fit,fl,fr,fy=arr_pairs(d,("A","P"),"fit");dev,dl,dr,dy=arr_pairs(d,("A","P"),"dev")
am=np.array([p["kind"]=="A" for p in dev])
with np.load(HERE/"selected_support_r16.npz") as a:sup={k:a[k] for k in a.files}
old=list(csv.DictReader((HERE/"metrics.csv").open()));allmetrics=[]
text_fields={"method","group","data_origin","rank_design"}
int_fields={"seed","parameters","n","requested_rank","effective_rank","head_count"}
for r in old:
 if r["method"].startswith("fixedS16_"):continue
 out={}
 for k,v in r.items():
  if v=="":continue
  if k in int_fields:out[k]=int(v)
  elif k not in text_fields:out[k]=float(v)
  else:out[k]=v
 if out["method"] in ("selected_N4","selected_N8","selected_N32"):out["rank_design"]="tied_signal_nuisance_rank"
 allmetrics.append(out)
temp=HERE/"pair_predictions.capacity.csv.gz";predfile=gzip.open(temp,"wt",newline="")
with gzip.open(HERE/"pair_predictions.csv.gz","rt",newline="") as f:
 reader=csv.DictReader(f);writer=csv.DictWriter(predfile,fieldnames=reader.fieldnames);writer.writeheader()
 for r in reader:
  if not r["method"].startswith("fixedS16_"):writer.writerow(r)
for rank in (4,8,32):
 tt=time.time()
 with np.load(HERE/("selected_support_r"+str(rank)+".npz")) as a:sp={k:a[k] for k in a.files}
 fixed=dict(sp);fixed["S"]=sup["S"];fixed["N"]=orth_nuisance(sup["S"],sp["U"])
 name="fixedS16_N"+str(rank)
 np.savez_compressed(HERE/("fixedS16_support_r"+str(rank)+".npz"),**fixed)
 z=project(raw,fixed);xf=z[fr]-z[fl];xd=z[dr]-z[dl]
 mod=Ridge(alpha=.01,fit_intercept=False).fit(xf,fy);pf=mod.predict(xf);pd=mod.predict(xd)
 np.savez_compressed(HERE/(name+"_model.npz"),coef=mod.coef_,alpha=.01,fit_intercept=False)
 for split,ps,y,pred in [("fit",fit,fy,pf),("dev",dev,dy,pd)]:
  for p,yy,yp in zip(ps,y,pred):
   writer.writerow(dict(method=name,seed="",split=split,pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],target=float(yy),prediction=float(yp)))
 for group,mask in [("A",am),("AP",np.ones(len(dy),bool))]:
  allmetrics.append(dict(method=name,seed=None,group=group,parameters=len(mod.coef_),fit_seconds=time.time()-tt,
   data_origin="real_cache_new_readout_training",**metrics(dy[mask],pd[mask]),requested_rank=rank,
   effective_rank=int(fixed["N"].shape[1]),rank_design="fixed_signal_rank16"))
predfile.close();os.replace(temp,HERE/"pair_predictions.csv.gz");finalize_metrics(allmetrics)
bootstrap=list(csv.DictReader((HERE/"bootstrap_summary.csv").open()))
for r in bootstrap:
 controls=[x for x in allmetrics if x["method"]==r["control"] and x["group"]=="A"]
 main=float(r["main_A_r2"]);n=sum(main>x["r2"] for x in controls)
 r["main_exceeds_control_seed_count"]=n;r["main_exceeds_control_seed_fraction"]=n/len(controls)
with (HERE/"bootstrap_summary.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(bootstrap[0]));w.writeheader();w.writerows(bootstrap)
config=json.loads((HERE/"config.json").read_text())
config["primary_dimension_design"]="fixedS16_N4/8/16/32; selected_N4/8/32 are secondary tied-S/N-rank comparisons"
dump("config.json",config)
dump("capacity_supplement.json",dict(status="complete",seconds=time.time()-started,
 purpose="Change only nuisance rank r4/8/16/32 while fixing the main signal space S16, raw-coordinate normalization and selected heads. No development/test tuning."))

