"""Merge M1 candidate requests and schedule distinct input pairs across GPU workers."""
import argparse,fcntl,json,os,time,traceback
from pathlib import Path
from .common import OUT,CONFIG,read,write

def _job_id(job):
    return str(job["left"])+"__"+str(job["right"])

def _cache_path(job,out):
    return Path(out)/"cache"/"exact"/(_job_id(job)+".json")

def _missing(job,out):
    cached=read(_cache_path(job,out))
    return [int(h) for h in job["heads"] if str(int(h)) not in cached]

def build_queue(plan_paths,out=OUT):
    """Union candidate heads for each ordered unique-input pair; reuse all exact caches."""
    out=Path(out);merged={};paths=sorted(Path(p) for p in plan_paths)
    for path in paths:
        plan=read(path)
        if "kinds" not in plan:raise ValueError("M1 plan has no kinds: "+str(path))
        for kind,detail in plan["kinds"].items():
            candidates=[int(h) for h in detail["candidates"]]
            for pair in detail["representatives"]:
                key=(str(pair["left"]),str(pair["right"]))
                if key not in merged:
                    merged[key]=dict(left=key[0],right=key[1],pair_id=pair["pair_id"],
                                     heads=set(),pair_ids=set(),plans=set(),kinds=set())
                job=merged[key];job["heads"].update(candidates)
                job["pair_ids"].add(str(pair["pair_id"]));job["plans"].add(str(path));job["kinds"].add(kind)
    jobs=[];requested=0;cached_heads=0;cached_pairs=0
    for job in merged.values():
        candidates=sorted(job["heads"]);requested+=len(candidates)
        cached=read(_cache_path(job,out))
        needed=[h for h in candidates if str(h) not in cached]
        cached_heads+=len(candidates)-len(needed)
        if not needed:cached_pairs+=1;continue
        jobs.append(dict(left=job["left"],right=job["right"],pair_id=job["pair_id"],heads=needed,
                         job_id=_job_id(job),pair_ids=sorted(job["pair_ids"]),
                         plan_paths=sorted(job["plans"]),kinds=sorted(job["kinds"])))
    # Longer jobs start first, reducing the final imbalance among the four cards.
    jobs.sort(key=lambda j:(-len(j["heads"]),j["left"],j["right"]))
    manifest=dict(created_at=time.strftime("%Y-%m-%d %H:%M:%S"),plan_paths=[str(p) for p in paths],
        unique_pairs=len(merged),jobs_total=len(jobs),requested_unique_pair_heads=requested,
        pending_pair_heads=sum(len(j["heads"]) for j in jobs),cached_pair_heads=cached_heads,
        completely_cached_pairs=cached_pairs,jobs=jobs,
        scheduling="nonblocking per-pair task claims; each process keeps one lazy canonical model",
        cache_policy="existing per-head exact scores are authoritative; completion markers do not override cache contents")
    write(out/"runtime/patch_jobs.json",manifest)
    return manifest

def _run_jobs(jobs,patcher,shard,out=OUT,shards=4):
    """CPU-testable scheduler core. The supplied patcher owns the one reusable model."""
    out=Path(out);started=time.monotonic()
    lockdir=out/"runtime/patch_job_locks";lockdir.mkdir(parents=True,exist_ok=True)
    donedir=out/"runtime/patch_job_done";donedir.mkdir(parents=True,exist_ok=True)
    statuspath=out/"status"/("patch_queue_"+str(shard)+".json")
    preferred=jobs[shard%max(shards,1)::max(shards,1)]
    preferred+= [j for i,j in enumerate(jobs) if i%max(shards,1)!=shard%max(shards,1)]
    local_done=set();completed=0;skipped=0;computed_heads=0;wait_started=None
    stat=dict(state="starting",pid=os.getpid(),shard=shard,
              physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),jobs_total=len(jobs),
              completed_here=0,cache_skipped=0,computed_pair_heads=0)
    write(statuspath,stat)
    while len(local_done)<len(jobs):
        claimed=False
        for job in preferred:
            key=_job_id(job)
            if key in local_done:continue
            needed=_missing(job,out)
            if not needed:
                local_done.add(key);skipped+=1;continue
            handle=(lockdir/(key+".lock")).open("a+")
            acquired=False
            try:
                try:
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:
                    continue
                needed=_missing(job,out)
                if not needed:
                    local_done.add(key);skipped+=1;continue
                claimed=True;wait_started=None
                stat.update(state="running",job_id=key,pair_id=job["pair_id"],pending_heads=len(needed),
                            completed_here=completed,cache_skipped=skipped,computed_pair_heads=computed_heads,
                            seconds=time.monotonic()-started)
                write(statuspath,stat)
                before=dict(getattr(patcher,"cost",{}))
                patcher(job,needed)
                remaining=_missing(job,out)
                if remaining:raise RuntimeError("Exact cache is incomplete after patching "+key)
                completed+=1;computed_heads+=len(needed);local_done.add(key)
                marker=dict(complete=True,job_id=key,pair_id=job["pair_id"],left=job["left"],right=job["right"],
                    heads=job["heads"],new_heads_here=needed,worker_pid=os.getpid(),shard=shard,
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    patch_cost_delta={k:getattr(patcher,"cost",{}).get(k,0)-v for k,v in before.items()})
                write(donedir/(key+".json"),marker)
                stat.update(completed_here=completed,cache_skipped=skipped,computed_pair_heads=computed_heads,
                    seconds=time.monotonic()-started,cost=dict(getattr(patcher,"cost",{})))
                write(statuspath,stat)
            finally:
                if acquired:fcntl.flock(handle,fcntl.LOCK_UN)
                handle.close()
            if claimed:break
        if len(local_done)==len(jobs):break
        if not claimed:
            now=time.monotonic()
            if wait_started is None or now-wait_started>=10:
                stat.update(state="waiting_for_other_pairs",observed_complete=len(local_done),
                            completed_here=completed,cache_skipped=skipped,
                            computed_pair_heads=computed_heads,seconds=now-started)
                write(statuspath,stat);wait_started=now
            time.sleep(1)
    stat.update(state="complete",observed_complete=len(local_done),completed_here=completed,cache_skipped=skipped,
        computed_pair_heads=computed_heads,seconds=time.monotonic()-started,cost=dict(getattr(patcher,"cost",{})))
    for key in ("job_id","pair_id","pending_heads"):stat.pop(key,None)
    write(statuspath,stat);write(out/"runtime"/("patch_worker_"+str(shard)+".json"),stat)
    return stat

def run_queue(shard,out=OUT,shards=4):
    """Run one GPU worker; CUDA_VISIBLE_DEVICES selects its single authorized card."""
    import torch
    from .common import Store
    from .worker import Patcher
    torch.set_num_threads(4)
    out=Path(out);manifest=read(out/"runtime/patch_jobs.json")
    if "jobs" not in manifest:raise FileNotFoundError(out/"runtime/patch_jobs.json")
    store=Store(out);patcher=Patcher(store)
    try:
        return _run_jobs(manifest["jobs"],patcher,int(shard),out=out,shards=int(shards))
    except Exception as exc:
        write(out/"status"/("patch_queue_"+str(shard)+".json"),
              dict(state="error",error=str(exc),traceback=traceback.format_exc(),pid=os.getpid(),
                   shard=shard,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("build","run"),default="run")
    p.add_argument("--plans",nargs="+");p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=4)
    a=p.parse_args()
    if a.mode=="build":
        if not a.plans:p.error("--plans is required for build mode")
        result=build_queue(a.plans)
        print(json.dumps({k:v for k,v in result.items() if k!="jobs"}),flush=True)
    else:print(json.dumps(run_queue(a.shard,shards=a.shards)),flush=True)
