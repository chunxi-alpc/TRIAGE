"""Durable stage controller for the authorized unfinished experiments."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from .common import *
PACKAGE="experiments.aster_cr_retrain_paired_v2_20261111"
GPUS=CONFIG["gpus"]

def state(stage,**details):
    write(OUT/"run_status.json",dict(stage=stage,time=time.strftime("%Y-%m-%d %H:%M:%S"),controller_pid=os.getpid(),**details))

def launch_workers(mode,stage,extra):
    state(stage,state="running",gpus=GPUS)
    jobs=[]
    for shard,gpu in enumerate(GPUS):
        if mode=="train" and extra==["--phase","full"] and shard>=3:continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
        cmd=[sys.executable,"-u","-m",PACKAGE+".worker","--mode",mode,"--shard",str(shard),"--shards",str(len(GPUS)),*extra]
        log=(OUT/"logs"/(stage+"_"+str(shard)+".log")).open("a")
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs.append((proc,log,shard))
    write(OUT/"active_workers.json",dict(stage=stage,workers=[dict(pid=p.pid,shard=s,gpu=GPUS[s]) for p,l,s in jobs]))
    errors=[]
    for proc,log,shard in jobs:
        ret=proc.wait();log.close()
        if ret:errors.append(dict(shard=shard,returncode=ret,log=str(OUT/"logs"/(stage+"_"+str(shard)+".log"))))
    if errors:raise RuntimeError(json.dumps(errors))
    update_manifest(stage)

def update_manifest(stage):
    old=read(OUT/"input_cache_manifest.json")
    inventory=read(OUT/"data_inventory.json")
    allrecords={}
    for p in [DATA/"reused_scores.jsonl",*sorted(CACHE.glob("scores_*.jsonl"))]:
        for r in rows(p):
            if r.get("status","ok")=="ok":allrecords[r.get("uid",r["view_id"])]=r
    fresh=[r for r in allrecords.values() if str(CACHE) in r.get("feature","")]
    manifest=dict(old,updated=time.strftime("%Y-%m-%d %H:%M:%S"),last_completed_stage=stage,unique_inputs=inventory["unique_inputs"],
                  available_inputs=len(allrecords),fresh_inputs=len(fresh),reused_inputs=len(allrecords)-len(fresh),
                  deduplication="exact prompt and label string equality; stable unique IDs; no hashes",
                  learned_estimates="M1 rankings/M2 moments and spaces/M3 weights are independently fitted within each training fold")
    write(OUT/"input_cache_manifest.json",manifest)

def folds():
    target=OUT/"folds.json"
    if target.exists():return
    if (OUT/"historical_folds.json").exists():
        write(target,read(OUT/"historical_folds.json"));return
    sources=sorted({p["source_id"] for p in rows(DATA/"E2_pairs.jsonl")})
    rng=np.random.default_rng(CONFIG["seed"]);rng.shuffle(sources)
    chunks=np.array_split(np.array(sources),5)
    rr=[]
    for i,c in enumerate(chunks):
        valid=c.tolist();train=[s for s in sources if s not in valid]
        rr.append(dict(fold=i+1,train_sources=sorted(train),validation_sources=sorted(valid)))
    write(target,dict(seed=CONFIG["seed"],all_sources=sorted(sources),folds=rr))

def model_heads(include_cv=False):
    paths=[OUT/"models"/g/"full"/"m2.npz" for g in ("E0","E1","E2","E3")]
    if include_cv:paths.extend((OUT/"models").glob("*/fold*/m2.npz"))
    values=set()
    for path in paths:
        with np.load(path) as a:values.update(int(h) for h in a["heads"])
    return sorted(values)

def sensitivity(group):
    from .training import refit_m3
    from .worker import wanted
    store=Store();original=OUT/"models"/group/"full"
    with np.load(original/"m2.npz") as a:support={k:a[k] for k in a.files}
    pairs=rows(DATA/(group+"_pairs.jsonl"));natural=rows(DATA/"natural_views.jsonl")
    ap=[p for p in pairs if p["kind"] in ("A","P")]
    # Relative budgets use the same historical reference scale for fair interpretation.
    reference_rms=2.923349027297149
    records=[]
    for relative in CONFIG["sensitivity_relative_budgets"]:
        target=OUT/"models"/group/("budget_"+str(relative))
        if read(target/"model.json").get("complete"):
            records.append(dict(relative_budget=relative,reference_RMS=reference_rms,**read(target/"m3.json")));continue
        target.mkdir(parents=True,exist_ok=True);shutil.copy2(original/"m2.npz",target/"m2.npz");shutil.copy2(original/"m1.json",target/"m1.json")
        result=refit_m3(group,pairs,store,natural,support,target,delta=relative*reference_rms)
        metadata=read(original/"model.json")
        metadata.update(complete=True,budget_relative=relative,delta=relative*reference_rms,M3=result,
                        fresh_modules=["M3"],M1_M2_reused_from="selected main data scheme, freshly estimated in this run")
        write(target/"model.json",metadata);records.append(dict(relative_budget=relative,reference_RMS=reference_rms,**result))
    write(OUT/"budget_sensitivity.json",dict(selected_group=group,reference="fixed historical fit AP RMS",reference_RMS=reference_rms,results=records))

def main():
    folds()
    launch_workers("extract","fit_cache",["--stages","fit,dev"])
    launch_workers("train","full_fits",["--phase","full"])
    launch_workers("train","source_cv",["--phase","folds"])
    all_heads=model_heads(include_cv=True)
    launch_workers("extract","development_path_cache",["--stages","path_fit,path_dev,path_interference_fit,path_interference_dev","--heads",",".join(map(str,all_heads))])
    from .evaluation import evaluate_development,evaluate_all
    state("development_comparison",state="running")
    development=evaluate_development(include_cv=True)
    write(OUT/"development_selection.json",development)
    group=development["frozen_group"]
    freeze=dict(frozen=True,selected_group=group,time=time.strftime("%Y-%m-%d %H:%M:%S"),
                configuration=CONFIG,selection=development,selection_split="development only; source-CV diagnostics also reported",
                target_test_labels_used_for_fit=False)
    write(OUT/"frozen.json",freeze)
    sensitivity(group)
    final_heads=model_heads()
    launch_workers("extract","historical_evaluation_cache",["--stages","test,path_test,path_interference_test,new60","--heads",",".join(map(str,final_heads))])
    state("historical_evaluation",state="running")
    # External forwards are also released only after source-method freezing.
    launch_workers("extract","cross_task_cache",["--stages","nli4ct,medpic","--heads",",".join(map(str,final_heads))])
    state("evaluation_and_report",state="running")
    result=evaluate_all(frozen_group=group)
    update_manifest("complete")
    write(OUT/"completion.json",dict(complete=True,time=time.strftime("%Y-%m-%d %H:%M:%S"),selected_group=group,summary=result))
    state("complete",state="complete",selected_group=group)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        state("failed",state="error",error=str(exc),traceback=traceback.format_exc())
        with (OUT/"REPORT.zh.md").open("a",encoding="utf-8") as f:f.write("\n?????"+str(exc)+"?????? run_status.json ? logs/?\n")
        raise
