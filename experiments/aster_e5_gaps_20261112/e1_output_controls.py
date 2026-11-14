"""E1 endpoint-native output controls; all inference features are reused read-only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("OMP_NUM_THREADS","4")
import sys, time, json, csv
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev
from experiments.aster_cr_retrain_paired_v2_20261111.common import Store, rows, read, write, lines
from experiments.aster_cr_retrain_paired_v2_20261111.training import pair_weights
BASE=Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
OUT=ROOT/"reports/aster_e5_gaps_20261112/output_controls_e1"
OUT.mkdir(parents=True,exist_ok=True)
NAMES=["temperature_e1_endpoint_nll","intercept_e1_endpoint_nll","affine_e1_endpoint_nll","linear_readout_e1_last_attention_ce"]
REGS=[0.001,0.01,0.1]
START=time.time()

def progress(stage,**kw):
    write(OUT/"status.json",dict(stage=stage,seconds=time.time()-START,**kw))
    print(stage,flush=True)

def residual(store,key):
    rec=store.record(key)
    if store.runtime is not None and store.runtime.ready:
        ix=store.runtime.row(key,rec)
        if ix is not None:
            return store.runtime.heads[ix][-32:].astype(np.float32).reshape(-1)
    with np.load(rec["feature"],allow_pickle=False) as z:
        return z["heads"].reshape(-1,128)[-32:].astype(np.float32).reshape(-1)

def matrix(store,keys):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return np.asarray(list(pool.map(lambda k:residual(store,k),keys)),dtype=np.float32)

def nll(scores,labels,weights=None):
    return float(np.average(np.logaddexp(0.,scores)-labels*scores,weights=weights))

def scalar_fit(e,y,w,mode):
    def fun(t):
        if mode=="temperature":
            a=np.exp(t[0]); s=e*a; jac=np.column_stack((e*a,))
        elif mode=="intercept":
            s=e+t[0]; jac=np.ones((len(e),1))
        else:
            s=e*t[0]+t[1]; jac=np.column_stack((e,np.ones(len(e))))
        r=(expit(s)-y)*w
        return float(np.sum(w*(np.logaddexp(0.,s)-y*s))),jac.T@r
    init=np.array([0.]) if mode!="affine" else np.array([1.,0.])
    bounds=[(-5.,5.)] if mode=="temperature" else None
    opt=minimize(fun,init,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
    a,b=(float(np.exp(opt.x[0])),0.) if mode=="temperature" else (1.,float(opt.x[0])) if mode=="intercept" else tuple(map(float,opt.x))
    return dict(scale=a,bias=b,objective=float(opt.fun),converged=bool(opt.success),optimizer_message=str(opt.message),iterations=int(opt.nit),parameters=1 if mode!="affine" else 2)

def linear_fit(x,y,w,reg):
    # Ordinary pointwise binary CE on the unmodified model's final attention-head outputs.
    # No A/P or C/F pair objective, ASTER head selection, or ASTER subspace.
    def fun(theta):
        a=theta[:-1]; s=x@a+theta[-1]; r=(expit(s)-y)*w
        loss=np.sum(w*(np.logaddexp(0.,s)-y*s))+0.5*reg*np.dot(a,a)
        return float(loss),np.r_[x.T@r+reg*a,r.sum()]
    opt=minimize(fun,np.zeros(x.shape[1]+1),jac=True,method="L-BFGS-B",
                 options=dict(maxiter=400,ftol=1e-10,gtol=1e-6,maxcor=15))
    return opt

def get_scope(split):
    vv=[v for v in ev.data_rows(BASE,"historical_views.jsonl") if v.get("split")==split]
    pp=[p for p in ev.data_rows(BASE,"historical_pairs.jsonl") if p.get("split")==split]
    pv=[v for name in ["path_features.jsonl","path_interference_views.jsonl"] for v in ev.data_rows(BASE,name) if v.get("split")==split]
    cases=[c for c in ev.case_rows(ev.HIST/"data/path_cases.jsonl") if c["split"]==split]
    return vv,pp,pv,cases

def run():
    if (OUT/"completion.json").exists():
        print("Already complete; refusing duplicate fit/evaluation.",flush=True);return
    progress("loading_existing_cache")
    store=Store(BASE)
    pairs=rows(BASE/"data/E1_pairs.jsonl")
    assert len(pairs)==2540
    keys=[]; labels={}; weights=defaultdict(float)
    pw=pair_weights(pairs,"E1")
    for p,weight in zip(pairs,pw):
        for side in ["left","right"]:
            key=p[side]; label=int(p[side+"_label"])
            if key in labels:assert labels[key]==label
            else:keys.append(key);labels[key]=label
            weights[key]+=float(weight)/2
    y=np.array([labels[k] for k in keys],float)
    w=np.array([weights[k] for k in keys],float); w/=w.sum()
    ell=np.array([store.ell(k) for k in keys],float)
    lines(OUT/"training_endpoint_index.jsonl",[dict(uid=k,label=labels[k],weight=float(weights[k])) for k in keys])
    protocol=dict(dataset="E1 / Dataset A",pairs=len(pairs),endpoint_occurrences=2*len(pairs),unique_endpoints=len(keys),
      feature="fixed final attention layer: all 32 heads concatenated, 4096 dimensions",model="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B",
      cache_root=str(BASE),native_objective="weighted pointwise binary negative log likelihood",
      weighting="E1 equal pair category and within-category source; half each pair weight to each endpoint, summed for shared inputs",
      pairwise_loss=False,aster_selected_heads=False,aster_subspace=False,training_sources=80,
      regularization_candidates=REGS,selection="natural development-node NLL only; freeze before loading test residuals",
      deterministic_solver=True,initialization="zero coefficients; affine scale starts at one",
      gpu_used=False,model_forward_passes=0,bootstrap_draws=2000,bootstrap_unit="guideline source",
      target_task_evaluation="NLI4CT and MedPIC are not evaluated",
      historical_test="40 sources, previously used; not new independent confirmation",
      previous_intercept="Existing E1 independent intercept trained on 761 original natural fit nodes is separate and reused elsewhere; this output is endpoint-supervised refit",
      original_implementation_reference="experiments/aster_cr/readout.py: scalar temperature/bias/logistic objectives; ordinary readout is now explicit final-attention-layer CE without ASTER-selected feature support")
    write(OUT/"protocol.json",protocol)
    progress("fitting_scalar_controls",unique_endpoints=len(keys))
    fitted={}
    for name,mode in zip(NAMES[:3],["temperature","intercept","affine"]):
        fitted[name]=scalar_fit(ell,y,w,mode)
    dev=get_scope("dev")
    dv=[v for v in dev[0] if v.get("kind","original")=="original" and v.get("original",False) and v.get("label") is not None]
    dkeys=[ev.uid(v) for v in dv]; dy=np.array([v["label"] for v in dv],float)
    progress("reading_fit_and_development_residuals")
    x=matrix(store,keys).astype(float)
    mu=np.sum(x*w[:,None],axis=0)
    scale=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0));scale=np.maximum(scale,1e-4)
    x=(x-mu)/scale
    dx=(matrix(store,dkeys).astype(float)-mu)/scale
    candidates=[]; candidate_weights=[]
    for reg in REGS:
        progress("fitting_linear_readout",regularization=reg)
        opt=linear_fit(x,y,w,reg)
        ds=dx@opt.x[:-1]+opt.x[-1]
        candidates.append(dict(regularization=reg,dev_nll=nll(ds,dy),dev_accuracy=float(np.mean((ds>0)==dy)),
             objective=float(opt.fun),converged=bool(opt.success),optimizer_message=str(opt.message),iterations=int(opt.nit)))
        candidate_weights.append(opt.x)
    best=min(range(len(candidates)),key=lambda i:(candidates[i]["dev_nll"],REGS[i]))
    theta=candidate_weights[best]
    fitted[NAMES[3]]=dict(candidates[best],parameters=len(theta),feature="fixed final attention layer, all 32 heads")
    np.savez(OUT/"linear_parameters.npz",mean=mu,scale=scale,weight=theta[:-1],bias=theta[-1])
    write(OUT/"scalar_parameters.json",{k:v for k,v in fitted.items() if k!=NAMES[3]})
    freeze=dict(frozen=True,time=time.strftime("%Y-%m-%d %H:%M:%S"),selection_split="dev",natural_dev_nodes=len(dv),
                chosen_regularization=REGS[best],linear_candidates=candidates,methods=fitted,test_data_used_for_selection=False)
    write(OUT/"frozen.json",freeze)
    del x,dx
    progress("frozen_parameters_evaluating_historical_sets")
    all_summary=[]
    for split in ["dev","test"]:
        vv,pp,pv,cases=dev if split=="dev" else get_scope(split)
        views=list({ev.uid(v):v for v in [*vv,*pv]}.values())
        skeys=[ev.uid(v) for v in views]
        se=np.array([store.ell(k) for k in skeys])
        progress("reading_evaluation_residuals",split=split,unique_inputs=len(skeys))
        sx=(matrix(store,skeys).astype(float)-mu)/scale
        linear=sx@theta[:-1]+theta[-1]
        del sx
        scores={"raw":dict(zip(skeys,map(float,se)))}
        for name in NAMES:
            vals=linear if name==NAMES[3] else se*fitted[name]["scale"]+fitted[name]["bias"]
            scores[name]=dict(zip(skeys,map(float,vals)))
        # Reuse the exact established path traversal, pair definitions and source-bootstrap evaluator.
        old=ev.score_views
        ev.score_views=lambda _store,_views,_models:scores
        models={name:dict(directory=str(OUT),metadata={}) for name in NAMES}
        try: result=ev.scope_evaluate(split,vv,pp,store,models,out=OUT,cases=cases,path_views=pv)
        finally:ev.score_views=old
        nat=[v for v in vv if v.get("kind","original")=="original" and v.get("original",False) and v.get("label") is not None]
        nat=list({ev.uid(v):v for v in nat}.values())
        sources=sorted({v["source_id"] for v in vv}|{c["source_id"] for c in cases})
        boot=ev.SourceBootstrap(sources)
        yy=np.array([v["label"] for v in nat]); sg=[v["source_id"] for v in nat]
        basecorrect=np.array([(scores["raw"][ev.uid(v)]>0)==v["label"] for v in nat])
        extras={}; interval=[]
        for name in scores:
            ls=np.array([scores[name][ev.uid(v)] for v in nat])
            pred=ls>0; correct=pred==yy
            pm={"natural_brier":boot.mean((expit(ls)-yy)**2,sg),
                "natural_log_loss":boot.mean(np.logaddexp(0,ls)-yy*ls,sg),
                "natural_corrected_error_rate":boot.mean((~basecorrect)&correct,sg),
                "natural_introduced_error_rate":boot.mean(basecorrect&(~correct),sg)}
            extras[name]=pm
            result["models"][name].update({k:float(v[0]) for k,v in pm.items()})
            result["models"][name].update(natural_corrected_errors=int(np.sum((~basecorrect)&correct)),
                                         natural_introduced_errors=int(np.sum(basecorrect&(~correct))))
            for metric,vals in pm.items():
                interval.append(dict(scope=split,comparison=name,metric=metric,**boot.summary(vals),sources=len(sources),draws=2000))
        for name in NAMES:
            for metric in extras[name]:
                interval.append(dict(scope=split,comparison=name+" minus raw",metric=metric,
                   **boot.summary(extras[name][metric]-extras["raw"][metric]),sources=len(sources),draws=2000))
        result["bootstrap"].extend(interval)
        write(OUT/"evaluation"/split/"metrics.json",result)
        ev.write_csv(OUT/"probability_bootstrap.csv",interval,replace_scope=split)
        lines(OUT/"evaluation"/split/"scoring_predictions.jsonl",
             [dict(uid=k,source_id=v["source_id"],label=v.get("label"),case_id=v.get("case_id"),gid=v.get("gid"),
                variant=v.get("variant"),scores={name:ss[k] for name,ss in scores.items()},
                probabilities={name:float(expit(ss[k])) for name,ss in scores.items()},
                predictions={name:int(ss[k]>0) for name,ss in scores.items()}) for k,v in zip(skeys,views)])
        all_summary.extend(dict(scope=split,method=name,**r) for name,r in result["models"].items())
        progress("scope_complete",split=split,methods=list(scores))
    ev.write_csv(OUT/"main_table.csv",all_summary)
    ev.write_csv(OUT/"all_source_bootstrap.csv",
       [r for split in ["dev","test"] for r in read(OUT/"evaluation"/split/"metrics.json")["bootstrap"]])
    write(OUT/"completion.json",dict(complete=True,seconds=time.time()-START,methods=NAMES,
        gpu_used=False,new_forward_passes=0,training_pairs=len(pairs),unique_endpoints=len(keys),
        dev_only_selected=True,historical_test_source_count=40,scope=["dev","test"]))
    (OUT/"REPORT.md").write_text(
      "# E1 endpoint output controls\n\nCompleted CPU-only fits and cached historical evaluation.\n\n"
      "Training: E1 Dataset A, 2540 pairs. Shared endpoints receive summed half-pair weights; "
      "weights preserve E1 category/source balance. Native pointwise binary NLL only. "
      "No pairwise invariance/evidence loss.\n\n"
      "Temperature, endpoint intercept and affine calibration refit the existing output objectives. "
      "The independent intercept fitted on 761 original natural nodes is a separate existing result and is not rerun.\n\n"
      "The ordinary supervised linear readout uses the frozen model's fixed final attention layer (all 32 heads, 4096 dimensions), "
      "not ASTER-selected heads or ASTER subspaces. L2 candidates 0.001/0.01/0.1 are chosen by 199 natural development-node NLL "
      "and frozen before test residual loading. This is a native supervised control.\n\n"
      "All 40 test sources and 806 natural paths have prior use. Results are historical evaluation, not independent confirmation. "
      "NLI4CT and MedPIC are not re-evaluated. Positive temperature preserves decision signs; assess its Brier/log loss.\n\n"
      "main_table.csv gives common pair/path/probability metrics. all_source_bootstrap.csv gives 2000 source-cluster "
      "bootstrap 95% intervals and paired differences to raw. evaluation/{dev,test}/ contains complete scoring, "
      "pair, and traversed-path predictions. Frozen parameters and solver status are preserved.\n",encoding="utf-8")
    progress("complete")
if __name__=="__main__":
    try:run()
    except Exception as e:
        progress("failed",error=str(e));raise

