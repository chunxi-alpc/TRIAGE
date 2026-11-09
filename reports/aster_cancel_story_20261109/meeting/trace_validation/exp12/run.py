"""Experiments 1/2: held-out prediction controls and capacity, real Qwen3 cache."""
import argparse,csv,gzip,json,time,sys,os
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd
SCRIPT_DIR=Path(__file__).resolve().parent
HERE=Path(os.environ.get("TRACE_EXP12_OUTPUT",str(SCRIPT_DIR)))
BASE=SCRIPT_DIR.parent
sys.path.insert(0,str(BASE))
from shared import Data,orth_nuisance
SEEDS=list(range(11,61))
NN_SEEDS=[11,22,33]
ALPHA=.01
def dump(name,obj):
 (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def rows_csv(name,rr):
 if not rr:return
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with (HERE/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def metrics(y,p):
 e=y-p;sst=np.sum((y-y.mean())**2)
 return dict(n=len(y),r2=float(1-np.sum(e*e)/sst) if sst>0 else None,rmse=float(np.sqrt(np.mean(e*e))))
def arr_pairs(d,kinds,split):
 pp=[p for p in d.pairs if p["split"]==split and p["kind"] in kinds]
 l=np.array([d.ix[p["left"]] for p in pp]);r=np.array([d.ix[p["right"]] for p in pp])
 return pp,l,r,d.ell[r]-d.ell[l]
def project(raw,sup,component="N"):
 return (((raw-sup["mu"])/sup["scale"])@sup[component]).astype(np.float64)
def train_nn(fit_src,xmain,fy,xdev,raw,fl,fr,d,record):
 # All support/scaler tuning stays in the internal training sources.
 import torch
 from torch import nn
 torch.set_num_threads(8)
 rng=np.random.default_rng(20261109)
 inner_sources=np.array(sorted(set(fit_src)));rng.shuffle(inner_sources)
 inner_val_sources=set(inner_sources[:max(1,len(inner_sources)//5)])
 va=np.array([s in inner_val_sources for s in fit_src]);tr=~va
 inner_support=d.support(d.heads,16,train_sources=set(fit_src[tr]))
 inner_z=project(raw,inner_support)
 inner_x=inner_z[fr]-inner_z[fl]
 np.savez_compressed(HERE/"nn_inner_support.npz",**inner_support)
 def net():return nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
 def nn_train(seed,lr,decay,epochs,x,y,xv=None,yv=None):
  torch.manual_seed(seed)
  xm=x.mean(0);xs=np.maximum(x.std(0),1e-6);ym=y.mean();ys=max(y.std(),1e-6)
  xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32)
  if xv is not None:xvt=torch.tensor((xv-xm)/xs,dtype=torch.float32);yvt=torch.tensor((yv-ym)/ys,dtype=torch.float32)
  model=net();opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
  best=float("inf");bestepoch=epochs;state=None
  for epoch in range(1,epochs+1):
   model.train();opt.zero_grad();loss=(model(xt).squeeze(1)-yt).square().mean();loss.backward();opt.step()
   if xv is not None and epoch%10==0:
    model.eval()
    with torch.no_grad():lossv=float((model(xvt).squeeze(1)-yvt).square().mean())
    if lossv<best:best=lossv;bestepoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
  if state is not None:model.load_state_dict(state)
  return model,xm,xs,ym,ys,best,bestepoch
 tune=[];tt=time.time()
 for lr in (.001,.003):
  for decay in (.0001,.01):
   _,_,_,_,_,lossv,epoch=nn_train(11,lr,decay,400,inner_x[tr],fy[tr],inner_x[va],fy[va])
   tune.append(dict(lr=lr,weight_decay=decay,inner_normalized_mse=lossv,epochs=epoch))
 chosen=min(tune,key=lambda x:x["inner_normalized_mse"])
 tuning_seconds=time.time()-tt
 dump("nn_tuning.json",dict(architecture="16->32->1; ReLU",input="projected right-minus-left trace difference",fit_intercept_linear=False,
    inner_preprocessing="Coordinate normalization and S/N subspaces refit on inner training sources; heads frozen from all fit sources, a conditional internal tuning design.",
    inner_split_seed=20261109,inner_training_sources=sorted(set(fit_src[tr])),inner_validation_sources=sorted(inner_val_sources),
    trials=tune,chosen=chosen,tuning_seconds=tuning_seconds,
    deployment_note="This nonlinear difference predictor is an information probe. It is not a single-input score correction; experiment 4 trains a separate single-input function."))
 for seed in NN_SEEDS:
  tt=time.time()
  model,xm,xs,ym,ys,_,_=nn_train(seed,chosen["lr"],chosen["weight_decay"],chosen["epochs"],xmain,fy)
  model.eval()
  with torch.no_grad():
   pf=model(torch.tensor((xmain-xm)/xs,dtype=torch.float32)).squeeze(1).numpy()*ys+ym
   pd=model(torch.tensor((xdev-xm)/xs,dtype=torch.float32)).squeeze(1).numpy()*ys+ym
  torch.save(model.state_dict(),HERE/("mlp16_seed"+str(seed)+".pt"))
  np.savez_compressed(HERE/("mlp16_seed"+str(seed)+"_scaler.npz"),x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys)
  record("mlp16",seed,pf,pd,time.time()-tt,sum(p.numel() for p in model.parameters()),dict(tuning_seconds=tuning_seconds))

def finalize_metrics(allmetrics):
 rows_csv("metrics.csv",allmetrics)
 agg=[]
 for family in dict.fromkeys(r["method"] for r in allmetrics):
  for group in ("A","AP"):
   rr=[r for r in allmetrics if r["method"]==family and r["group"]==group]
   vals=np.array([r["r2"] for r in rr]);err=np.array([r["rmse"] for r in rr])
   agg.append(dict(method=family,group=group,repeats=len(rr),r2_mean=float(vals.mean()),r2_std=float(vals.std()),
      r2_min=float(vals.min()),r2_max=float(vals.max()),rmse_mean=float(err.mean()),parameters=rr[0]["parameters"],
      fit_seconds_mean=float(np.mean([r["fit_seconds"] for r in rr]))))
 rows_csv("summary.csv",agg)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 rr=[r for r in agg if r["group"]=="A"]
 fig,ax=plt.subplots(figsize=(12,5.5))
 ax.bar(range(len(rr)),[r["r2_mean"] for r in rr],yerr=[r["r2_std"] for r in rr],color=["#4377a8" if r["method"].startswith("selected") else "#b2c2cf" for r in rr],capsize=3)
 ax.axhline(0,color="black",lw=.7)
 ax.set_xticks(range(len(rr)),[r["method"] for r in rr],rotation=40,ha="right")
 ax.set_ylabel("Development A-pair R2");ax.set_title("Real-cache prediction controls (error bars: seed SD)")
 fig.tight_layout();fig.savefig(HERE/"prediction_comparison.png",dpi=180);fig.savefig(HERE/"prediction_comparison.pdf");plt.close(fig)

def main():
 started=time.time();HERE.mkdir(parents=True,exist_ok=True);d=Data()
 fit,fl,fr,fy=arr_pairs(d,("A","P"),"fit")
 dev,dl,dr,dy=arr_pairs(d,("A","P"),"dev")
 am=np.array([p["kind"]=="A" for p in dev])
 src=np.array([p["source_id"] for p in dev])
 fit_src=np.array([p["source_id"] for p in fit])
 raw=d.raw(d.heads)
 supports={}
 for rank in (4,8,16,32):
  supports[rank]=d.support(d.heads,rank)
  np.savez_compressed(HERE/("selected_support_r"+str(rank)+".npz"),**supports[rank])
 sup=supports[16]
 zmain=project(raw,sup); xmain=zmain[fr]-zmain[fl]; xdev=zmain[dr]-zmain[dl]
 allmetrics=[];preds={};train_times={}
 predfile=gzip.open(HERE/"pair_predictions.csv.gz","wt",newline="")
 writer=csv.DictWriter(predfile,fieldnames=["method","seed","split","pair_id","kind","source_id","target","prediction"])
 writer.writeheader()
 def record(name,seed,pf,pd,seconds,params,extra=None):
  key=name if seed is None else name+"_seed"+str(seed)
  preds[key]=pd.copy()
  for split,ps,y,pred in [("fit",fit,fy,pf),("dev",dev,dy,pd)]:
   for p,yy,yp in zip(ps,y,pred):
    writer.writerow(dict(method=name,seed=seed,split=split,pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],target=float(yy),prediction=float(yp)))
  for group,mask in [("A",am),("AP",np.ones(len(dy),bool))]:
   allmetrics.append(dict(method=name,seed=seed,group=group,parameters=params,fit_seconds=seconds,
      data_origin="real_cache_new_readout_training",**metrics(dy[mask],pd[mask]),**(extra or {})))
  train_times[key]=seconds
 def ridge(name,z,seed=None,target=None,extra=None):
  tt=time.time();xf=z[fr]-z[fl];xd=z[dr]-z[dl]
  mod=Ridge(alpha=ALPHA,fit_intercept=False).fit(xf,fy if target is None else target)
  pf=mod.predict(xf);pd=mod.predict(xd)
  seconds=time.time()-tt
  np.savez_compressed(HERE/(name+("" if seed is None else "_seed"+str(seed))+"_model.npz"),coef=mod.coef_,alpha=ALPHA,fit_intercept=False)
  record(name,seed,pf,pd,seconds,len(mod.coef_),extra)
  return pd
 ridge("selected_N16",zmain)
 # Historical reproduction uses the frozen fit-only support and coefficient procedure.
 legacy_file=Path("/home/gcx/aster_cr_repair_v2_run_20261108/qwen/supports_r16.npz")
 with np.load(legacy_file) as a:old={k:a[k] for k in a.files}
 oldz=project(raw,old)
 legacy=Ridge(alpha=.01,fit_intercept=False).fit(oldz[fr]-oldz[fl],fy)
 oldpred=legacy.predict(oldz[dr]-oldz[dl])
 dump("reference_reproduction.json",dict(
    data_origin="real_cache_recomputed",fit_AP=len(fit),dev_A=int(am.sum()),dev_AP=len(dev),
    reference_A_r2=.4063326595932645,recomputed_A=metrics(dy[am],oldpred[am]),
    reestimated_A=metrics(dy[am],preds["selected_N16"][am]),
    support_max_abs_N_difference=float(np.max(np.abs(sup["N"]-old["N"]))) if sup["N"].shape==old["N"].shape else None,
    frozen_vs_reestimated_prediction_max_abs=float(np.max(np.abs(oldpred-preds["selected_N16"])))))
 ridge("signal_S16",project(raw,sup,"S"))
 # Estimate the unorthogonalized nuisance basis from identical standardized fit A/P differences.
 nd=((raw[fr]-raw[fl])/sup["scale"]).astype(np.float32)
 _,sing,vt=randomized_svd(nd,n_components=16,random_state=20261108)
 keep=sing>max(1e-7,float(sing.max())*1e-6)
 un=vt[keep].T.astype(np.float32)
 ridge("unorthogonal_N16",(((raw-sup["mu"])/sup["scale"])@un).astype(np.float64))
 for rank in (4,8,32):
  ridge("selected_N"+str(rank),project(raw,supports[rank]),extra=dict(requested_rank=rank,effective_rank=int(supports[rank]["N"].shape[1]),rank_design="tied_signal_nuisance_rank"))
  fixed=dict(supports[rank]);fixed["S"]=sup["S"];fixed["N"]=orth_nuisance(sup["S"],fixed["U"])
  np.savez_compressed(HERE/("fixedS16_support_r"+str(rank)+".npz"),**fixed)
  ridge("fixedS16_N"+str(rank),project(raw,fixed),extra=dict(requested_rank=rank,effective_rank=int(fixed["N"].shape[1]),rank_design="fixed_signal_rank16"))
 for seed in SEEDS:
  tt=time.time();rng=np.random.default_rng(seed)
  q,_=np.linalg.qr(rng.normal(size=(raw.shape[1],16)))
  rz=(((raw-sup["mu"])/sup["scale"])@q.astype(np.float32)).astype(np.float64)
  ridge("random_subspace16",rz,seed,extra=dict(feature_seconds=time.time()-tt))
  shuffled=np.random.default_rng(seed).permutation(fy)
  ridge("shuffled_target",zmain,seed,target=shuffled)
  tt=time.time()
  # Sample uniformly among all model attention heads, with no exclusion of selected heads.
  allraw=d.raw(None)
  head_dim=raw.shape[1]//len(d.heads)
  nh=allraw.shape[1]//head_dim
  hh=np.random.default_rng(seed).choice(nh,len(d.heads),replace=False).tolist()
  rs=d.support(hh,16)
  rr=d.raw(hh)
  rz=project(rr,rs)
  feature_seconds=time.time()-tt
  ridge("random_heads51",rz,seed,extra=dict(feature_seconds=feature_seconds,head_count=len(hh),effective_rank=int(rs["N"].shape[1])))
  np.savez_compressed(HERE/("random_heads51_seed"+str(seed)+"_support.npz"),**rs)
  print("finished control seed",seed,flush=True)
 # Group bootstrap: fixed seeds retained separately; contrast against their mean metric.
 source_order=sorted(set(src[am]))
 use=np.flatnonzero(am);ya=dy[use];sa=src[use]
 keys=list(preds)
 sse=np.array([[np.sum((ya[sa==s]-preds[k][use][sa==s])**2) for s in source_order] for k in keys])
 counts=np.array([np.sum(sa==s) for s in source_order])
 sums=np.array([np.sum(ya[sa==s]) for s in source_order])
 sums2=np.array([np.sum(ya[sa==s]**2) for s in source_order])
 rng=np.random.default_rng(20261109)
 weights=rng.multinomial(len(source_order),np.ones(len(source_order))/len(source_order),size=2000)
 bn=weights@counts;ss=weights@sums;ss2=weights@sums2
 tss=ss2-ss*ss/bn
 br2=1-(weights@sse.T)/tss[:,None]
 midx=keys.index("selected_N16")
 boot=[]
 mainci=np.quantile(br2[:,midx],[.025,.975])
 for family in ("random_subspace16","random_heads51","shuffled_target","signal_S16","unorthogonal_N16"):
  ix=[i for i,k in enumerate(keys) if k==family or k.startswith(family+"_seed")]
  control=np.mean(br2[:,ix],axis=1)
  delta=br2[:,midx]-control
  base=np.mean([metrics(ya,preds[keys[i]][use])["r2"] for i in ix])
  ci=np.quantile(delta,[.025,.975])
  boot.append(dict(main="selected_N16",control=family,seeds=len(ix),main_A_r2=metrics(ya,preds["selected_N16"][use])["r2"],
    main_exceeds_control_seed_count=sum(metrics(ya,preds["selected_N16"][use])["r2"]>metrics(ya,preds[keys[i]][use])["r2"] for i in ix),
    main_exceeds_control_seed_fraction=float(np.mean([metrics(ya,preds["selected_N16"][use])["r2"]>metrics(ya,preds[keys[i]][use])["r2"] for i in ix])),
    control_mean_A_r2=float(base),difference=float(metrics(ya,preds["selected_N16"][use])["r2"]-base),
    difference_low=float(ci[0]),difference_high=float(ci[1]),main_low=float(mainci[0]),main_high=float(mainci[1]),
    bootstrap_replicates=2000,sources=len(source_order)))
 np.savez_compressed(HERE/"bootstrap_draws.npz",source_multiplicity=weights,method_names=np.array(keys),r2=br2)
 rows_csv("bootstrap_summary.csv",boot)
 train_nn(fit_src,xmain,fy,xdev,raw,fl,fr,d,record)
 predfile.close()
 finalize_metrics(allmetrics)
 dump("config.json",dict(seed_controls=SEEDS,seed_NN=NN_SEEDS,alpha=ALPHA,fit_intercept=False,fit_kinds=["A","P"],
    primary_dev_kind="A",secondary_dev_kinds=["A","P"],rank_grid=[4,8,16,32],primary_dimension_design="fixedS16_N4/8/16/32; selected_N4/8/32 are secondary tied-S/N-rank comparisons",
    effective_ranks={str(k):dict(signal=int(v["S"].shape[1]),nuisance=int(v["N"].shape[1])) for k,v in supports.items()},
    bootstrap_seed=20261109,bootstrap_replicates=2000,bootstrap_unit="source_id",bootstrap_control_statistic="mean per-seed R2, not ensemble prediction",
    fit_pairs=len(fit),dev_pairs=len(dev),dev_A_pairs=int(am.sum()),fit_sources=len(set(fit_src)),dev_sources=len(set(src)),
    elapsed_seconds=time.time()-started,GPU_used=False,data_origin="real_cache_new_readout_training",
    standardization="Raw head coordinates standardized by fit full views only; no further scaling for ridge. NN scaler fit within internal training sources during tuning, all fit during refit."))
 dump("done.json",dict(status="complete",elapsed_seconds=time.time()-started))
if __name__=="__main__":main()

