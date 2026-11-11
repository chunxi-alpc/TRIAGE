"""Four-GPU pooled M1 interventions with independently fitted source-fold estimates."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from .common import *
PACKAGE="experiments.aster_cr_retrain_paired_v2_20261111"
GPUS=[0,1,5,6]

def tasks():
    ff=read(OUT/"folds.json")["folds"]
    return [(g,"full",None) for g in ("E1","E2","E3")]+[(g,"fold"+str(f["fold"]),f) for f in ff for g in ("E1","E2")]

def done(path):
    try:return read(path/"model.json").get("complete",False)
    except json.JSONDecodeError:return False

def plan_worker(shard,shards):
    from .m1_plan import make_plan
    store=Store();pair_cache={}
    for g,name,fold in tasks()[shard::shards]:
        target=OUT/"models"/g/name
        if done(target):continue
        sources=fold["train_sources"] if fold else None
        existing=read(target/"m1_plan.json")
        expected=sorted(sources or read(OUT/"folds.json")["all_sources"])
        if existing.get("group")==g and existing.get("training_sources")==expected:continue
        if g not in pair_cache:pair_cache[g]=rows(DATA/(g+"_pairs.jsonl"))
        make_plan(g,pair_cache[g],store,target,train_sources=sources)
        write(OUT/"status"/("plan_"+str(shard)+".json"),dict(state="running",last_group=g,last_fold=name,pid=os.getpid()))
    write(OUT/"status"/("plan_"+str(shard)+".json"),dict(state="complete",pid=os.getpid()))

def old_training_processes():
    found=[]
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:
            cmd=(path/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            if PACKAGE+".worker" in cmd and "--mode train" in cmd and (path/"cwd").resolve()==REPO:found.append(int(path.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found

def terminate_owned(pid):
    if not pid or pid==os.getpid():return
    p=Path("/proc")/str(pid)
    try:
        if (p/"cwd").resolve()!=REPO:return
        cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        if PACKAGE+".worker" not in cmd and PACKAGE+".controller" not in cmd:return
        os.kill(pid,signal.SIGTERM)
    except (ProcessLookupError,FileNotFoundError):pass

def handover():
    previous=dict(controller=read(OUT/"controller_process.json"),active=read(OUT/"active_workers.json"),early=read(OUT/"early_cv_workers.json"))
    write(OUT/"scheduling_before_pooled_queue.json",previous)
    terminate_owned(previous["controller"].get("pid"))
    # Stop waiting continuation processes before terminating their full-fit parents.
    gpu_pids=set()
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True)
    for line in p.stdout.splitlines():
        try:gpu_pids.add(int(line.strip()))
        except ValueError:pass
    current=old_training_processes()
    for pid in current:
        if pid not in gpu_pids:terminate_owned(pid)
    # All old M1 representatives are deterministic minima within source/category.
    pairkeys=set()
    for g in ("E1","E2"):
        representatives={}
        for p in rows(DATA/(g+"_pairs.jsonl")):
            key=(p["kind"],p["source_id"])
            if key not in representatives or p["pair_id"]<representatives[key]["pair_id"]:representatives[key]=p
        pairkeys.update((p["left"],p["right"]) for p in representatives.values())
    held=[];pending=sorted(pairkeys);started=time.time()
    try:
        while pending:
            remaining=[]
            for left,right in pending:
                path=CACHE/"exact"/(left+"__"+right+".lock");path.parent.mkdir(parents=True,exist_ok=True)
                handle=path.open("a+")
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(handle)
                except BlockingIOError:handle.close();remaining.append((left,right))
            pending=remaining
            if pending:time.sleep(.05)
            if time.time()-started>300:raise RuntimeError("Timed out waiting for safe single-pair patch checkpoints")
        for pid in old_training_processes():terminate_owned(pid)
        time.sleep(2)
    finally:
        for handle in held:
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
    write(OUT/"scheduling_handover.json",dict(complete=True,seconds=time.time()-started,safe_pair_checkpoints=len(pairkeys),old_training_pids=current,progress_preserved=True,raw_scores_recomputed=False))
    write(OUT/"early_cv_workers.json",dict(workers={},retired=True))
    write(OUT/"early_cv_worker.json",dict(retired=True))
    write(OUT/"controller_process.json",dict(pid=os.getpid(),command=[sys.executable,"-m",PACKAGE+".optimized_pipeline"],gpus=GPUS))

def read_exact(pair,heads):
    cache=read(CACHE/"exact"/(pair["left"]+"__"+pair["right"]+".json"))
    missing=[h for h in heads if str(int(h)) not in cache]
    if missing:raise RuntimeError("Missing completed exact patch: "+pair["pair_id"])
    return [(int(h),*cache[str(int(h))]) for h in heads]

def fit_worker(shard,shards):
    import torch
    from .training import _m2,refit_m3
    from .m1_plan import finish_plan
    torch.set_num_threads(4);store=Store()
    vv=rows(DATA/"historical_views.jsonl");natural=rows(DATA/"natural_views.jsonl");pair_cache={}
    all_tasks=tasks()+[("E0","fold"+str(f["fold"]),f) for f in read(OUT/"folds.json")["folds"]]
    for g,name,fold in all_tasks[shard::shards]:
        target=OUT/"models"/g/name
        if done(target):continue
        target.mkdir(parents=True,exist_ok=True)
        with (target/".fit.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if done(target):continue
            sources=sorted(fold["train_sources"] if fold else read(OUT/"folds.json")["all_sources"]);allowed=set(sources)
            if g not in pair_cache:pair_cache[g]=rows(DATA/(g+"_pairs.jsonl"))
            pairs=[p for p in pair_cache[g] if p["source_id"] in allowed and p.get("split","fit")=="fit"]
            allviews=list({v["view_id"]:v for v in rows(DATA/(g+"_views.jsonl"))+vv}.values())
            views=[v for v in allviews if v.get("split","fit")=="fit" and v["source_id"] in allowed]
            calibrators=[v for v in natural if v["source_id"] in allowed]
            prior=read(target/"model.json");start=time.time()
            if g=="E0":
                with np.load(target/"m2.npz") as a:support={k:a[k] for k in a.files}
                heads=support["heads"].tolist();m1=read(target/"m1.json");m2=prior.get("M2",{"historical_reuse":True})
            else:
                plan=read(target/"m1_plan.json");heads,m1=finish_plan(plan,store,read_exact,target)
                support=None
                if (target/"m2.npz").exists():
                    try:
                        with np.load(target/"m2.npz") as a:checkpoint={k:a[k] for k in a.files}
                        if np.array_equal(checkpoint["heads"],np.array(heads)):support=checkpoint
                    except Exception:pass
                if support is None:support,m2=_m2(g,heads,pairs,store,views,target)
                else:m2=prior.get("M2",dict(rank=16,S_rank=support["S"].shape[1],N_rank=support["N"].shape[1],resumed_completed_checkpoint=True,projected_feature_entropy_effective_rank=float(support.get("projected_feature_entropy_effective_rank",0))))
            if (target/"m3.npz").exists() and read(target/"m3.json").get("success"):
                m3=read(target/"m3.json")
            else:m3=refit_m3(g,pairs,store,calibrators,support,target,delta=CONFIG["delta"])
            meta=dict(prior,complete=True,group=g,fold=name,heads=heads,training_sources=sources,train_sources=sources,validation_sources=fold["validation_sources"] if fold else [],M1=m1,M2=m2,M3=m3,rank=16,regularization=.01,delta=CONFIG["delta"],fresh_modules=["M3"] if g=="E0" else ["M1","M2","M3"],prediction="ell + (((heads_selected-mu)/scale) @ N) @ w + b",seconds=time.time()-start,gpu_cost_policy="raw pair/head patch costs shared in runtime/patch_jobs.json and status/patch_queue_*.json; no duplicated model forwards")
            write(target/"model.json",meta)
            append(OUT/"fit_jobs.jsonl",dict(group=g,fold=name,complete=True,summary=meta))
            write(OUT/"status"/("fit_cpu_"+str(shard)+".json"),dict(state="running",last_group=g,last_fold=name))
    write(OUT/"status"/("fit_cpu_"+str(shard)+".json"),dict(state="complete"))

def spawn(module,mode,stage,gpus=None,extra=None):
    jobs=[]
    for shard in range(4):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[shard]) if gpus else "",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
        command=[sys.executable,"-u","-m",PACKAGE+"."+module,"--shard",str(shard)]
        if mode:command.extend(["--mode",mode])
        if extra:command.extend(extra)
        log=(OUT/"logs"/(stage+"_"+str(shard)+".log")).open("a")
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs.append((process,log))
    return jobs

def join(jobs):
    errors=[]
    for p,log in jobs:
        ret=p.wait();name=log.name;log.close()
        if ret:errors.append(dict(pid=p.pid,returncode=ret,log=name))
    if errors:raise RuntimeError(json.dumps(errors))

def state(stage,**kw):write(OUT/"run_status.json",dict(stage=stage,state="running",controller_pid=os.getpid(),time=time.strftime("%Y-%m-%d %H:%M:%S"),**kw))

def main():
    from .controller import launch_workers,model_heads,sensitivity,update_manifest
    state("independent_m1_plans")
    join(spawn("optimized_pipeline","plan","m1_plans"))
    state("safe_patch_queue_handover")
    handover()
    from .patch_queue import build_queue
    paths=[OUT/"models"/g/n/"m1_plan.json" for g,n,f in tasks() if not done(OUT/"models"/g/n)]
    manifest=build_queue(paths)
    state("pooled_exact_patching",jobs=manifest.get("jobs",[]).__len__() if isinstance(manifest.get("jobs"),list) else manifest.get("job_count"))
    jobs=spawn("patch_queue",None,"pooled_patch",gpus=GPUS)
    write(OUT/"active_workers.json",dict(stage="pooled_exact_patching",workers=[dict(pid=p.pid,gpu=g,shard=i) for i,((p,l),g) in enumerate(zip(jobs,GPUS))],policy="different pair jobs; original per-group and per-fold candidate sets"))
    join(jobs)
    state("module_fits_and_development_cache")
    cpu=spawn("optimized_pipeline","fit","module_fits")
    # Raw feature extraction is independent of fitted heads and labels.
    launch_workers("extract","development_path_cache",["--stages","path_fit,path_dev,path_interference_fit,path_interference_dev"])
    join(cpu)
    finish_evaluation()

def owned_worker_running(worker):
    proc=Path("/proc")/str(worker["pid"])
    try:
        command=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        return (proc/"cwd").resolve()==REPO and PACKAGE+".worker" in command
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return False

def ensure_extraction(stage,extra):
    from .controller import launch_workers,update_manifest
    active=read(OUT/"active_workers.json")
    workers=active.get("workers",[])
    if active.get("stage")==stage and len(workers)==len(GPUS):
        state(stage,gpus=GPUS,resumed_existing_workers=True)
        while any(owned_worker_running(w) for w in workers):
            time.sleep(10)
        statuses=[read(OUT/"status"/("extract_"+str(w["shard"])+".json")) for w in workers]
        if all(status.get("pid")==worker["pid"] and status.get("state")=="complete" for worker,status in zip(workers,statuses)):
            update_manifest(stage)
            return
    launch_workers("extract",stage,extra)

def finish_evaluation():
    from .controller import launch_workers,model_heads,sensitivity,update_manifest
    from .evaluation import evaluate_development,evaluate_all
    frozen=read(OUT/"frozen.json")
    if frozen.get("frozen"):
        group=frozen["selected_group"]
    else:
        state("development_comparison")
        development=evaluate_development(include_cv=True);group=development["frozen_group"]
        write(OUT/"frozen.json",dict(frozen=True,selected_group=group,time=time.strftime("%Y-%m-%d %H:%M:%S"),configuration=read(OUT/"configuration.json"),selection=development,selection_split="development only",target_test_labels_used_for_fit=False))
    sensitivity(group)
    selected=model_heads()
    ensure_extraction("historical_evaluation_cache",["--stages","test,path_test,path_interference_test,new60","--heads",",".join(map(str,selected))])
    ensure_extraction("cross_task_cache",["--stages","nli4ct,medpic","--heads",",".join(map(str,selected))])
    state("evaluation_and_report")
    result=evaluate_all(frozen_group=group)
    from .finalize_report import finalize_report
    finalize_report(out=OUT)
    update_manifest("complete")
    write(OUT/"completion.json",dict(complete=True,time=time.strftime("%Y-%m-%d %H:%M:%S"),selected_group=group,summary=result))
    write(OUT/"run_status.json",dict(stage="complete",state="complete",controller_pid=os.getpid(),selected_group=group))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["main","plan","fit","resume-evaluation"],default="main");p.add_argument("--shard",type=int,default=0)
    a=p.parse_args()
    try:
        if a.mode=="main":main()
        elif a.mode=="resume-evaluation":
            write(OUT/"controller_process.json",dict(pid=os.getpid(),command=[sys.executable,"-m",PACKAGE+".optimized_pipeline","--mode","resume-evaluation"],gpus=GPUS))
            finish_evaluation()
        elif a.mode=="plan":plan_worker(a.shard,4)
        else:fit_worker(a.shard,4)
    except Exception as exc:
        write(OUT/"optimized_error.json",dict(mode=a.mode,shard=a.shard,error=str(exc),traceback=traceback.format_exc()))
        if a.mode in ("main","resume-evaluation"):write(OUT/"run_status.json",dict(stage="failed",state="error",controller_pid=os.getpid(),error=str(exc)))
        raise
