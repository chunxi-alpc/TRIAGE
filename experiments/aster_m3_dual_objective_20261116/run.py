"""Reproducible cache-only runner for issue 75. No model execution imports."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import argparse, json, time, shutil, sys, gzip
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as old
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, lines
from . import evaluation as ev
from .solver import solve, pair_weights, WEIGHT_TOTAL, center, calibrate

ROOT=Path(__file__).resolve().parents[2]
CACHE=Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
OUT=ROOT/"reports/aster_m3_dual_objective_20261116"
ETAS=[0,.01,.03,.1,.3,1,3,10,30,100]
LAMBDAS=[.0001,.001,.01,.1,1,10,100]
GRID=[dict(id="e%02d_l%02d"%(i,j),eta=e,regularization=l)
      for i,e in enumerate(ETAS) for j,l in enumerate(LAMBDAS)]
def log(s): print(time.strftime("%H:%M:%S"),s,flush=True)
def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(s) for s in f if s.strip()]
def compressed_json(path,value):
    with gzip.open(path,"wt",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False)
def compressed_lines(path,records):
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
def unique(vv): return list({old.uid(v):v for v in vv}.values())
def data(name): return old.data_rows(CACHE,name+".jsonl")
def take(rr,sources): return [r for r in rr if r["source_id"] in sources]
def natural(vv):
    return [v for v in unique(vv) if v.get("kind","original")=="original" and
        (v.get("original") or ("original" not in v and v.get("variant","original") in ("full","original")))]

def build_scopes():
    folds=read(CACHE/"folds.json")
    if len(folds["all_sources"])!=80: raise ValueError("Expected 80 fitting sources")
    training=data("E1_pairs")
    nv=data("natural_views")
    bv=[dict(v,original=False) for v in data("E2_views")]
    bp=data("E2_pairs")
    hv,hp=data("historical_views"),data("historical_pairs")
    pv=data("path_features")+data("path_interference_views")
    cases=old.case_rows(old.HIST/"data/path_cases.jsonl")
    scopes={}
    for f in folds["folds"]:
        name="fold"+str(f["fold"])
        allowed=set(f.get("validation_sources",f.get("heldout_sources",[])))
        if len(f["train_sources"])!=64 or len(allowed)!=16: raise ValueError("Invalid fold source counts")
        scopes[name]=dict(name=name,sources=sorted(allowed),pairs=take(bp,allowed),
            views=unique(take(bv,allowed)+take(nv,allowed)),natural=take(nv,allowed),
            path_views=take([v for v in pv if v["split"]=="fit"],allowed),
            cases=take([c for c in cases if c["split"]=="fit"],allowed),
            train_sources=f["train_sources"],old_scope="cv_E1_"+name+"_common_frozen_pool")
    val=[s for sc in scopes.values() for s in sc["sources"]]
    if len(val)!=80 or set(val)!=set(folds["all_sources"]): raise ValueError("OOF sources not exactly once")
    for split,n in (("dev",20),("test",40)):
        vv=[v for v in hv if v["split"]==split]
        allowed=sorted({v["source_id"] for v in vv})
        if len(allowed)!=n or set(allowed)&set(folds["all_sources"]): raise ValueError("Split source mismatch")
        scopes[split]=dict(name=split,sources=allowed,
            pairs=[p for p in hp if p["split"]==split],views=unique(vv),natural=natural(vv),
            path_views=[v for v in pv if v["split"]==split],
            cases=[c for c in cases if c["split"]==split],old_scope=split)
    if set(scopes["dev"]["sources"])&set(scopes["test"]["sources"]): raise ValueError("dev/test overlap")
    # E1 CV evaluates the same common frozen Dataset B for every model.
    for sc in scopes.values():
        ep=CACHE/"evaluation"/sc["old_scope"]
        needed=["metrics.json","pair_predictions.jsonl","predictions.jsonl","path_predictions.jsonl"]
        for n in needed:
            if not (ep/n).is_file(): raise FileNotFoundError(str(ep/n))
        rr=[r for r in rows(ep/"pair_predictions.jsonl") if r["method"]=="E1"]
        if Counter(r["pair_id"] for r in rr)!=Counter(p["pair_id"] for p in sc["pairs"]):
            raise ValueError("Existing E1 comparison uses a different pair pool: "+sc["name"])
    return scopes,training,nv,folds

def load_scores(required):
    rec={}
    for path in [CACHE/"data/reused_scores.jsonl",*sorted((CACHE/"cache").glob("scores_*.jsonl"))]:
        with path.open() as f:
            for line in f:
                r=json.loads(line);uid=r.get("uid",r.get("view_id"))
                if uid in required and r.get("status","ok")=="ok" and "feature" in r and "ell" in r:
                    rec[uid]=dict(feature=r["feature"],ell=float(r["ell"]))
    return rec

def prepare(scopes,training,nv,folds):
    supports,required={},{}
    for fold in ["full"]+["fold"+str(i) for i in range(1,6)]:
        d=CACHE/"models/E1"/fold
        for name in ("m1.json","m2.npz","m3.npz","model.json"):
            if not (d/name).is_file(): raise FileNotFoundError(str(d/name))
        m=read(d/"model.json");s=dict(np.load(d/"m2.npz",allow_pickle=False))
        if s["N"].shape!=(6528,16) or len(s["heads"])!=51: raise ValueError("Unexpected M1/M2 shape")
        if list(s["heads"])!=read(d/"m1.json")["heads"]: raise ValueError("M1/M2 heads disagree")
        train=set(folds["all_sources"] if fold=="full" else scopes[fold]["train_sources"])
        if train!=set(m.get("train_sources",m.get("training_sources",[]))):
            raise ValueError("Model training sources disagree: "+fold)
        if fold!="full" and set(m["validation_sources"])!=set(scopes[fold]["sources"]):
            raise ValueError("Model validation sources disagree: "+fold)
        train_pairs=take(training,train);cal=take(nv,train)
        ids={p[side] for p in train_pairs for side in ("left","right")}
        ids.update(old.uid(v) for v in cal)
        ids.update(str(u) for u in s["standardization_ids"])
        evals=[scopes["dev"],scopes["test"]] if fold=="full" else [scopes[fold]]
        for sc in evals: ids.update(old.uid(v) for v in sc["views"]+sc["path_views"])
        supports[fold]=s;required[fold]=sorted(ids)
    alluids=set().union(*map(set,required.values()))
    rec=load_scores(alluids)
    missing=[dict(uid=u,reason="missing finite score or feature file",
        feature=rec.get(u,{}).get("feature")) for u in sorted(alluids)
        if u not in rec or not np.isfinite(rec[u]["ell"]) or not Path(rec[u]["feature"]).is_file()]
    manifest=dict(cache_root=str(CACHE),required_unique_inputs=len(alluids),
        required_fold_input_projections=sum(map(len,required.values())),available_score_records=len(rec),
        missing=missing,M1_M2_reused=6,cache_reuse_ratio=(len(alluids)-len(missing))/len(alluids),
        GPU_tasks=0,model_forward_count=0,gradient_computation_count=0,
        data="E1 Dataset A fitting; E2 common frozen Dataset B OOF; historical dev/test",
        M4="training natural-node additive intercept; beta=1",
        validation="existence, source splits, numeric array coverage; no hashes")
    manifest.update(
        model_directories=[str(CACHE/"models/E1"/f) for f in supports],
        model_files_per_directory=["m1.json","m2.npz","m3.npz","model.json"],
        data_files=[str(CACHE/"data"/(n+".jsonl")) for n in ["E1_pairs","E2_views","E2_pairs",
            "natural_views","historical_views","historical_pairs","path_features","path_interference_views"]],
        score_tables=[str(CACHE/"data/reused_scores.jsonl")]+[str(p) for p in sorted((CACHE/"cache").glob("scores_*.jsonl"))],
        runtime_arrays=[str(CACHE/"runtime"/n) for n in ["fit_heads.npy","fit_present.npy","fit_uids.json","fit_sources.json"]],
        historical_cases=str(old.HIST/"data/path_cases.jsonl"),folds_file=str(CACHE/"folds.json"))
    write(OUT/"cache_manifest.json",manifest)
    if missing: raise RuntimeError("Required cache missing; see cache_manifest.json")
    runtime=CACHE/"runtime"
    rtids=read(runtime/"fit_uids.json",[])
    rtindex={u:i for i,u in enumerate(rtids)}
    rtsources=read(runtime/"fit_sources.json",[])
    heads=np.load(runtime/"fit_heads.npy",mmap_mode="r")
    present=np.load(runtime/"fit_present.npy",mmap_mode="r")
    projected={}
    for fold,s in supports.items():
        target=OUT/"runtime"/(fold+"_coordinates.npz")
        ids=required[fold]
        if target.exists():
            z=np.load(target,allow_pickle=False)
            if z["uids"].tolist()!=ids: raise ValueError("Projection cache UID mismatch")
            T=z["coordinates"];raw=z["raw"]
        else:
            h=s["heads"].astype(int)
            N,mu,scale=[s[k].astype(np.float64) for k in ("N","mu","scale")]
            def project(u):
                r=rec[u];idx=rtindex.get(u)
                if idx is not None and rtsources[idx]==r["feature"] and int(present[idx])&1:
                    a=heads[idx][h].reshape(-1).astype(np.float64)
                else:
                    with np.load(r["feature"],allow_pickle=False) as z:
                        a=z["heads"]
                        a=a.reshape(-1,a.shape[-1])
                        if len(a)!=1152 and "head_ids" in z:
                            mapping={int(k):i for i,k in enumerate(z["head_ids"].reshape(-1))}
                            a=a[[mapping[int(k)] for k in h]]
                        else: a=a[h]
                        a=a.reshape(-1).astype(np.float64)
                if not np.all(np.isfinite(a)): raise ValueError("Missing/nonfinite selected heads: "+u)
                t=((a-mu)/scale)@N
                if not np.all(np.isfinite(t)): raise ValueError("Nonfinite coordinates: "+u)
                return t
            T=np.empty((len(ids),16),np.float64)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for start in range(0,len(ids),512):
                    try: T[start:start+512]=list(pool.map(project,ids[start:start+512]))
                    except Exception as ex:
                        manifest["missing"].append(dict(fold=fold,reason=str(ex)))
                        write(OUT/"cache_manifest.json",manifest)
                        raise
            raw=np.array([rec[u]["ell"] for u in ids])
            np.savez_compressed(target,uids=np.array(ids),coordinates=T,raw=raw)
        projected[fold]=dict(uids=ids,index={u:i for i,u in enumerate(ids)},T=T,raw=raw)
        log("cached coordinates ready: "+fold+" "+str(len(ids)))
    manifest["projection_complete"]=True
    manifest["cache_reuse_ratio"]=1.0
    write(OUT/"cache_manifest.json",manifest)
    return supports,projected

def fit(fold,configuration,supports,projected,training,nv,scopes,folds):
    s,p=supports[fold],projected[fold]
    train=set(folds["all_sources"] if fold=="full" else scopes[fold]["train_sources"])
    pp=take(training,train);ap=[r for r in pp if r["kind"] in ("A","P")]
    cf=[r for r in pp if r["kind"] in ("C","F")]
    def mat(rr):
        li=[p["index"][r["left"]] for r in rr];ri=[p["index"][r["right"]] for r in rr]
        return p["T"][ri]-p["T"][li],p["raw"][ri]-p["raw"][li]
    X,y=mat(ap);E,_=mat(cf)
    wa,wc=pair_weights(ap,"E1",WEIGHT_TOTAL),pair_weights(cf,"E1",WEIGHT_TOTAL)
    w,diag=solve(X,y,E,wa,wc,configuration["eta"],configuration["regularization"])
    idx=[p["index"][str(u)] for u in s["standardization_ids"]]
    mu_c=center(w,p["T"][idx],s["standardization_weights"])
    cal=[v for v in take(nv,train) if v.get("variant","full")=="full"]
    idx=[p["index"][old.uid(v)] for v in cal]
    b=calibrate(p["raw"][idx],p["T"][idx],[v["label"] for v in cal],w,mu_c)
    params=dict(w=w.tolist(),mu_c=mu_c,b=b,beta=1.,eta=configuration["eta"],
        regularization=configuration["regularization"],train_sources=sorted(train),
        calibration_ids=[old.uid(v) for v in cal],AP_pair_ids=[r["pair_id"] for r in ap],
        CF_pair_ids=[r["pair_id"] for r in cf],AP_weights=wa.tolist(),CF_weights=wc.tolist(),
        support_directory=str(CACHE/"models/E1"/fold))
    diag.update(fold=fold,candidate=configuration["id"],mu_c=mu_c,b=b,calibration_nodes=len(cal),
        AP_pairs=len(ap),CF_pairs=len(cf))
    return params,diag

def predictions(sc,fold,params,projected,method):
    p=projected[fold]
    score=p["raw"]+p["T"]@np.array(params["w"])-params["mu_c"]+params["b"]
    return ev.predict(sc,dict(zip(p["uids"],score)),dict(zip(p["uids"],p["raw"])),method)

def old_predictions(sc,method):
    path=CACHE/"evaluation"/sc["old_scope"]
    prs=[{k:v for k,v in r.items() if k in (
        "pair_id","source_id","kind","group","left","right","left_label","right_label",
        "left_score","right_score","raw_difference","difference","correction_difference",
        "paired_correct","relation_correct","direction_degraded")}
        for r in rows(path/"pair_predictions.jsonl") if r["method"]==method]
    prs=[dict(r,method="old_E1" if method=="E1" else method) for r in prs]
    wanted={old.uid(v):v for v in sc["natural"]}
    ns=[]
    for r in rows(path/"predictions.jsonl"):
        u=r["view_id"]
        if u in wanted:
            v=wanted[u];score=r["scores"][method]
            ns.append(dict(method="old_E1" if method=="E1" else method,view_id=u,
                source_id=v["source_id"],label=v["label"],score=score,
                correct=bool((score>0)==int(v["label"]))))
    if len(ns)!=len(wanted): raise ValueError("Old natural predictions incomplete")
    paths=[dict(r,method="old_E1" if method=="E1" else method) for r in rows(path/"path_predictions.jsonl")
        if r["method"]==method]
    return prs,ns,paths

def combine(parts): return tuple([r for part in parts for r in part[i]] for i in range(3))

def save_scope(name,observations,sources):
    directory=OUT/"evaluation"/name;directory.mkdir(parents=True,exist_ok=True)
    boot=ev.SourceBootstrap(sources)
    np.savez_compressed(directory/"source_draws.npz",sources=np.array(boot.sources),weights=boot.weights)
    metrics,summary,bootstrap,source=[],[],[],[]
    samples={}
    for method,obs in observations.items():
        mm,cc=ev.aggregate(*obs,boot);samples[method]=mm
        summary.append(dict(scope=name,method=method,over_budget_rate="N/A",**ev.point(mm,cc)))
        source.extend(dict(scope=name,method=method,**r) for r in ev.per_source(*obs))
        for metric,values in mm.items():
            bootstrap.append(dict(scope=name,comparison=method,metric=metric,**boot.summary(values)))
        for fn,rr in zip(("pair_predictions","natural_predictions","path_predictions"),obs):
            compressed_lines(directory/(method+"_"+fn+".jsonl.gz"),rr)
    for left,right in (("dual","old_E1"),("dual","ap_only"),("dual","raw"),("ap_only","old_E1")):
        for metric in samples[left].keys() & samples[right].keys():
            bootstrap.append(dict(scope=name,comparison=left+" minus "+right,metric=metric,
                **boot.summary(samples[left][metric]-samples[right][metric])))
    old.write_csv(directory/"metrics.csv",summary)
    old.write_csv(directory/"source_metrics.csv",source)
    old.write_csv(directory/"bootstrap.csv",bootstrap)
    write(directory/"metrics.json",dict(scope=name,source_count=len(boot.sources),methods=summary,
        bootstrap_draws=2000,over_budget_rate="N/A"))
    return summary,bootstrap,source

def execute():
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"runtime").mkdir(exist_ok=True)
    config=dict(issue=75,selection_protocol="80-source five-fold OOF; 20-source dev independent validation; 40-source reused holdout",
        resolution="User confirmed OOF selection on 2026-11-16; supersedes issue evaluation section 1-3",
        eta_grid=ETAS,lambda_grid=LAMBDAS,bootstrap_draws=2000,bootstrap_seed=old.SEED,
        evaluation_weighting="E1 pooled pair/node/path moments; sources resampled with replacement",
        fitting_weighting="training.pair_weights(E1); each AP/CF group sums to 4887",
        M1_heads=51,M2_rank=16,M4="same E1 training natural-node additive intercept; beta=1",
        manuscript_difference="PR74 positive-slope Platt not used in this M3-only experiment",
        GPUs=[],GPU_memory_GiB=0,model_forwards=0,gradient_computations=0,
        grid_scope="OOF selection, never dev/test grid",execution="CPU only")
    write(OUT/"configuration.json",config)
    scopes,training,nv,folds=build_scopes()
    write(OUT/"folds.json",folds)
    supports,projected=prepare(scopes,training,nv,folds)
    log("all required caches verified; starting 350 CPU solves")
    boots=ev.SourceBootstrap(folds["all_sources"])
    np.savez_compressed(OUT/"selection_source_draws.npz",sources=np.array(boots.sources),weights=boots.weights)
    grid,grid_ci,diagnostics,models=[],[],[],{}
    for c in GRID:
        parts=[];models[c["id"]]={}
        for fold in ["fold"+str(i) for i in range(1,6)]:
            params,d=fit(fold,c,supports,projected,training,nv,scopes,folds)
            models[c["id"]][fold]=params;diagnostics.append(d)
            parts.append(predictions(scopes[fold],fold,params,projected,c["id"]))
        mm,cc=ev.aggregate(*combine(parts),boots)
        grid.append(dict(**c,**ev.point(mm,cc)))
        for metric,values in mm.items():
            grid_ci.append(dict(candidate=c["id"],metric=metric,**boots.summary(values)))
    mask=ev.pareto_mask(grid)
    for row,m in zip(grid,mask):row["pareto"]=m
    best=max(grid,key=ev.selection_key)
    ap=max([r for r in grid if r["eta"]==0],key=ev.selection_key)
    chosen=dict(dual=best,ap_only=ap,positive_joint_effect=best["J"]>0,
        selection_set="hyperparameter-selection OOF results (80 sources)",
        tie_break_order=["max J","min abs(ER-1)","min CF direction degradation",
            "max CF paired accuracy","max natural path accuracy","max lambda"],
        dev_test_used_for_selection=False)
    old.write_csv(OUT/"oof_grid_70.csv",grid)
    old.write_csv(OUT/"pareto_front.csv",[r for r in grid if r["pareto"]])
    old.write_csv(OUT/"grid_bootstrap.csv",grid_ci)
    write(OUT/"selection.json",chosen)
    compressed_json(OUT/"oof_candidate_parameters.json.gz",models)
    log("selection frozen: "+best["id"]+" eta="+str(best["eta"])+" lambda="+str(best["regularization"]))
    all_metrics,all_ci,all_source=[],[],[]
    method_models={}
    for method,c in (("dual",best),("ap_only",ap)):
        method_models[method]=models[c["id"]].copy()
        params,d=fit("full",c,supports,projected,training,nv,scopes,folds)
        method_models[method]["full"]=params;diagnostics.append(d)
        for fold,params in method_models[method].items():
            write(OUT/"models"/method/fold/"m3.json",params)
    oof={m:[] for m in ("dual","ap_only","old_E1","raw")}
    for name,sc in scopes.items():
        fold="full" if name in ("dev","test") else name
        observations={method:predictions(sc,fold,method_models[method][fold],projected,method)
                      for method in ("dual","ap_only")}
        observations.update(old_E1=old_predictions(sc,"E1"),raw=old_predictions(sc,"raw"))
        metrics,ci,sr=save_scope(name,observations,sc["sources"])
        all_metrics+=metrics;all_ci+=ci;all_source+=sr
        if fold!="full":
            for method,obs in observations.items():oof[method].append(obs)
    metrics,ci,sr=save_scope("oof_selection",{m:combine(v) for m,v in oof.items()},folds["all_sources"])
    all_metrics+=metrics;all_ci+=ci;all_source+=sr
    old.write_csv(OUT/"metrics.csv",all_metrics)
    old.write_csv(OUT/"bootstrap.csv",all_ci)
    old.write_csv(OUT/"source_metrics.csv",all_source)
    old.write_csv(OUT/"solver_diagnostics.csv",diagnostics)
    write(OUT/"completion.json",dict(complete=True,CPU_seconds_elapsed=time.monotonic()-start,
        solves=len(diagnostics),candidate_count=len(grid),M1_M2_cache_reuse_ratio=1.,
        required_input_cache_reuse_ratio=1.,GPU_tasks=0,model_forward_count=0,gradient_computation_count=0))
    log("experiment complete")

def main():
    try: execute()
    except Exception as ex:
        write(OUT/"failure.json",dict(error=type(ex).__name__,message=str(ex),
            GPU_tasks=0,model_forward_count=0))
        raise
if __name__=="__main__": main()
