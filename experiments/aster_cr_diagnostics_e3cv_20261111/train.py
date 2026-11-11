"""Only missing E3 folds; pooled exact interventions on physical GPUs 0 and 1."""
import argparse, os, shutil, subprocess, sys, time, traceback
import numpy as np
from .common import *
from experiments.aster_cr_retrain_paired_v2_20261111 import training as tr
from experiments.aster_cr_retrain_paired_v2_20261111.m1_plan import make_plan, finish_plan
from experiments.aster_cr_retrain_paired_v2_20261111.patch_queue import build_queue, _run_jobs

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(experiment_version=VERSION, historical_root=str(BASE), previous_commit="d37487a",
        frozen_group="E1",gpus=[0,1,5,6],model="Qwen3-8B",model_dtype="bfloat16",label_projection="float32",
        heads=51,rank=16,regularization=.01,delta=tr.DELTA,AP_weight_total=4887,
        bootstrap_draws=2000,seed=20261111,
        score_variants={"raw":"ell","pair_only":"ell+c","original_intercept_only":"ell+b",
        "independent_intercept":"ell+b0","full":"ell+c+b"},
        principal_comparisons=["full minus independent_intercept","E3 minus E2","E3 minus E1"],
        reuse=["E0-E3 full","E0-E2 five folds","budget sensitivity"],
        weighting="E3 natural at M1/M2/M3; every training C/F pair constrained",
        history="40-source, 60-source and target-task results have prior use",
        monitoring="Observe startup only; automatic execution records progress and failures")
    write(OUT/"configuration.json",config)
    for name in ("folds.json","frozen.json","budget_sensitivity.json"):
        shutil.copy2(BASE/name,OUT/name)
    store=Store(BASE); pairs=rows(BASE/"data/E3_pairs.jsonl"); paths=[]
    folds=read(BASE/"folds.json")["folds"]
    for f in folds:
        assert len(f["train_sources"])==64 and len(f["validation_sources"])==16
        assert not set(f["train_sources"]) & set(f["validation_sources"])
        target=OUT/"models/E3"/("fold"+str(f["fold"])); path=target/"m1_plan.json"
        if not path.exists():make_plan("E3",pairs,store,target,f["train_sources"])
        paths.append(path)
        write(OUT/"status/prepare.json",dict(state="running",completed_fold=f["fold"]))
    # Copy only requested existing exact-cache entries. New results are local to this experiment.
    exact=OUT/"cache/exact";exact.mkdir(parents=True,exist_ok=True)
    keys=set()
    for path in paths:
        for item in read(path)["kinds"].values():
            keys.update(p["left"]+"__"+p["right"] for p in item["representatives"])
    for key in keys:
        source=BASE/"cache/exact"/(key+".json");dest=exact/(key+".json")
        if source.exists() and not dest.exists():shutil.copy2(source,dest)
    manifest=build_queue(paths,out=OUT)
    write(OUT/"status/prepare.json",dict(state="complete",**{k:v for k,v in manifest.items() if k!="jobs"}))
    print({k:manifest[k] for k in ("unique_pairs","jobs_total","pending_pair_heads","cached_pair_heads")},flush=True)

def gpu_worker(shard):
    import torch
    from experiments.aster_cr_retrain_paired_v2_20261111 import worker
    torch.set_num_threads(4);worker.CACHE=OUT/"cache"
    worker.CONFIG=read(BASE/"configuration.json")
    class Patcher(worker.Patcher):
        def context(self,uid):
            old=BASE/"cache/prefix"/(uid+".pt")
            if old.exists():return torch.load(old,map_location=self.engine.device,weights_only=False)
            return super().context(uid)
    patcher=Patcher(Store(BASE))
    _run_jobs(read(OUT/"runtime/patch_jobs.json")["jobs"],patcher,shard,out=OUT,shards=4)

def fit():
    store=Store(BASE);pairs=rows(BASE/"data/E3_pairs.jsonl")
    natural=rows(BASE/"data/natural_views.jsonl")
    for f in read(BASE/"folds.json")["folds"]:
        fold="fold"+str(f["fold"]);target=OUT/"models/E3"/fold
        if read(target/"model.json").get("complete"):continue
        start=time.time();allowed=set(f["train_sources"])
        ps=[p for p in pairs if p["source_id"] in allowed and p.get("split","fit")=="fit"]
        nv=[v for v in natural if v["source_id"] in allowed]
        def exact(pair,heads):
            c=read(OUT/"cache/exact"/(pair["left"]+"__"+pair["right"]+".json"))
            return [(int(h),*c[str(int(h))]) for h in heads]
        heads,m1=finish_plan(read(target/"m1_plan.json"),store,exact,target)
        support,m2=tr._m2("E3",heads,ps,store,[],target)
        m3=tr.refit_m3("E3",ps,store,nv,support,target)
        write(target/"model.json",dict(complete=True,experiment_version=VERSION,group="E3",fold=fold,
             heads=heads,train_sources=f["train_sources"],training_sources=f["train_sources"],
             validation_sources=f["validation_sources"],intercept_fit_sources=f["train_sources"],
             M1=m1,M2=m2,M3=m3,delta=tr.DELTA,rank=16,regularization=.01,
             seconds=time.time()-start))
        write(OUT/"status/train.json",dict(state="running",completed_fold=fold))
    write(OUT/"status/train.json",dict(state="complete",folds=5))

def launch():
    jobs=[]
    manifest=read(OUT/"runtime/patch_jobs.json")
    if manifest["pending_pair_heads"]:
        for i,gpu in enumerate((0,1,5,6)):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            log=(OUT/"logs"/("gpu_"+str(gpu)+".log")).open("a")
            p=subprocess.Popen([sys.executable,"-u","-m",__package__+".train","gpu","--shard",str(i)],
                env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            jobs.append((p,log))
        write(OUT/"status/gpu_workers.json",dict(workers=[dict(pid=p.pid,gpu=(0,1,5,6)[i]) for i,(p,l) in enumerate(jobs)]))
        codes=[p.wait() for p,l in jobs]
        for p,l in jobs:l.close()
        if any(codes):raise RuntimeError("GPU patch worker failed: "+str(codes))
    fit()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","launch","gpu","fit"])
    p.add_argument("--shard",type=int,default=0);a=p.parse_args()
    (OUT/"logs").mkdir(parents=True,exist_ok=True)
    try:
        if a.mode=="prepare":prepare()
        elif a.mode=="gpu":gpu_worker(a.shard)
        elif a.mode=="fit":fit()
        else:launch()
    except Exception as e:
        write(OUT/"status"/("failed_"+a.mode+str(a.shard)+".json"),dict(error=str(e),traceback=traceback.format_exc()))
        raise
