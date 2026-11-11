"""GPU extraction and complete-module training workers; all output stays in the new report."""
import argparse,copy,fcntl,os,time,traceback
from pathlib import Path
import numpy as np
from .common import *

def locked(path):
    path=Path(path)
    from contextlib import contextmanager
    @contextmanager
    def lock():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a+") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    return lock()

def wanted(record,grads=False,heads=None):
    if not record or record.get("status","ok")!="ok":return False
    if grads and not record.get("has_grads",False):return False
    available=record.get("available_heads")
    if available=="all":return True
    if available is not None:return heads is not None and set(heads).issubset(set(available))
    with np.load(record["feature"]) as a:
        if grads and "grads" not in a:return False
        h=a["heads"]
        return h.size==1152*128 or (heads is not None and "head_ids" in a and set(heads).issubset(a["head_ids"].tolist()))

def save_prefix(context,uid):
    import torch
    target=CACHE/"prefix"/(uid+".pt")
    if target.exists():return
    target.parent.mkdir(parents=True,exist_ok=True)
    host=copy.copy(context);host.input_ids=context.input_ids.cpu()
    host.prefix=copy.copy(context.prefix);host.prefix.layers=[copy.copy(x) for x in context.prefix.layers]
    for layer in host.prefix.layers:
        if layer.keys is not None:
            layer.keys=layer.keys.cpu();layer.values=layer.values.cpu()
    tmp=target.with_name(target.name+"."+str(os.getpid())+".tmp")
    torch.save(host,tmp);tmp.replace(target)

def extract(shard,shards,stages,limit=None,heads=None,smoke=False):
    import torch
    from experiments.aster_cr.engine import Engine
    torch.set_num_threads(4)
    store=Store();stage_set=set(stages.split(","))
    owned=[v for i,v in enumerate(store.inputs.values()) if i%shards==shard and stage_set.intersection(v.get("stages",[v.get("split")]))]
    todo=[v for v in owned if not wanted(store.records.get(v["view_id"]),v.get("required_grads",False) and "fit" in stage_set,heads)]
    if limit:todo=todo[:limit]
    report=OUT/"status"/("extract_"+str(shard)+".json")
    stat=dict(state="starting",pid=os.getpid(),shard=shard,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
              stages=stages,total=len(owned),pending=len(todo),completed=0,reused=len(owned)-len(todo))
    write(report,stat)
    if not todo:stat["state"]="complete";write(report,stat);return
    prefix_index=read(OUT/"runtime/prefix_required_uids.json")
    prefix_uids=set(prefix_index.get("uids",[])) if prefix_index else None
    eng=Engine("qwen");started=time.time()
    for i,v in enumerate(todo):
        uid=v["view_id"];grad=bool(v.get("required_grads",False) and "fit" in stage_set)
        with locked(CACHE/"locks"/(uid+".lock")):
            context=eng.prepare(v["prompt"])
            result=eng.run(context,capture=True,grad=grad)
            if not np.isfinite(result["ell"]):raise ValueError("Nonfinite score for "+uid)
            arrays={"heads":result.pop("heads").numpy().astype(np.float16)}
            if "grads" in result:arrays["grads"]=result.pop("grads").numpy().astype(np.float16)
            target=CACHE/"features"/(uid+".npz");target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**arrays)
            if "fit" in v.get("stages",[]) and (prefix_uids is None or uid in prefix_uids):save_prefix(context,uid)
            rec=dict(view_id=uid,uid=uid,feature=str(target),status="ok",available_heads="all",has_grads=grad,**result)
            append(CACHE/("scores_"+str(shard)+".jsonl"),rec)
            del context
        stat.update(state="running",completed=i+1,seconds=time.time()-started,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30,per_second=(i+1)/max(time.time()-started,.001))
        if i<8 or (i+1)%50==0:write(report,stat)
    stat.update(state="complete",counts=eng.counts,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    write(report,stat)
    if smoke:
        store.refresh();patcher=Patcher(store);patcher.engine=eng
        fresh=set(v["view_id"] for v in todo)
        pair=next(p for p in rows(DATA/"E2_pairs.jsonl") if p["left"] in store.records and p["right"] in store.records and (p["left"] in fresh or p["right"] in fresh))
        exact=patcher(pair,[736])
        if not np.isfinite(np.asarray(exact,dtype=float)).all():raise ValueError("Nonfinite exact patch result")
        check_uid=todo[0]["view_id"];context=patcher.context(check_uid)
        replay=eng.run(context,capture=False)["ell"];difference=abs(replay-store.ell(check_uid))
        if difference>1e-5:raise ValueError("Saved-prefix score replay mismatch: "+str(difference))
        del context
        write(OUT/"smoke.json",dict(stat,score_protocol="canonical immutable-prefix final-token BF16; FP32 label projection",finite=True,activation_shape=[1152,128],grads_checked=any(v.get("required_grads") for v in todo),exact_bidirectional_patch_checked=True,saved_prefix_replay_max_difference=difference,prefix_cache_cost=patcher.cost))

class Patcher:
    def __init__(self,store):
        self.store=store;self.engine=None;self.cost=dict(prefix_builds=0,patched_forwards=0,patch_cache_hits=0)
    def context(self,uid):
        import torch
        path=CACHE/"prefix"/(uid+".pt")
        # Published prefix files are immutable and can be read by all GPUs.
        if path.exists():return torch.load(path,map_location=self.engine.device,weights_only=False)
        with locked(CACHE/"prefix_locks"/(uid+".lock")):
            if path.exists():return torch.load(path,map_location=self.engine.device,weights_only=False)
            context=self.engine.prepare(self.store.inputs[uid]["prompt"]);save_prefix(context,uid)
            self.cost["prefix_builds"]+=1
            return context
    def __call__(self,pair,heads):
        path=CACHE/"exact"/(pair["left"]+"__"+pair["right"]+".json")
        with locked(path.with_suffix(".lock")):
            cached=read(path);needed=[int(h) for h in heads if str(h) not in cached]
            self.cost["patch_cache_hits"]+=len(heads)-len(needed)
            if needed:
                if self.engine is None:
                    from experiments.aster_cr.engine import Engine
                    self.engine=Engine("qwen")
                left=self.store.get(pair["left"]);right=self.store.get(pair["right"])
                a=self.context(pair["left"]);b=self.context(pair["right"])
                if CONFIG.get("patch_engine")=="single_row_read_only":
                    from .fast_patch import batched_single_head
                    forwards=batched_single_head(self.engine,a,right["heads"],needed,batch_size=1)
                    reverses=batched_single_head(self.engine,b,left["heads"],needed,batch_size=1)
                    for h in needed:cached[str(h)]=[float(forwards[h]),float(reverses[h])]
                    self.cost["patched_forwards"]+=2*len(needed)
                else:
                    for j,h in enumerate(needed):
                        forward=self.engine.run(a,replacements={h:right["heads"][h]},capture=False)["ell"]
                        reverse=self.engine.run(b,replacements={h:left["heads"][h]},capture=False)["ell"]
                        cached[str(h)]=[float(forward),float(reverse)]
                        self.cost["patched_forwards"]+=2
                        if (j+1)%16==0:write(path,cached)
                write(path,cached);del a,b
            return [(int(h),*cached[str(int(h))]) for h in heads]

def train(shard,shards,phase):
    import torch
    from .training import fit_group
    def alive(pid):
        p=Path("/proc")/str(pid)/"stat"
        try:return p.exists() and p.read_text().split()[2]!="Z"
        except FileNotFoundError:return False
    # Continuation workers wait for their current full fit. Later controller
    # slots join the continuation, keeping exactly one model per GPU.
    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES")
    registry=read(OUT/"early_cv_workers.json").get("workers",{})
    early=registry.get(physical_gpu,{})
    if not early and physical_gpu=="6":early=read(OUT/"early_cv_worker.json")
    if phase=="folds" and early:
        if early.get("pid")==os.getpid():
            while alive(early.get("wait_for_pid",-1)):time.sleep(2)
        elif alive(early.get("pid",-1)):
            write(OUT/"status"/("join_"+str(shard)+".json"),dict(state="joined_early_worker",pid=os.getpid(),active_gpu_worker_pid=early["pid"],physical_gpu=physical_gpu))
            while alive(early["pid"]):time.sleep(2)
    torch.set_num_threads(4)
    store=Store();patcher=Patcher(store)
    views=rows(DATA/"historical_views.jsonl");natural=rows(DATA/"natural_views.jsonl")
    folds=read(OUT/"folds.json")["folds"]
    tasks=([(g,"full",None) for g in ("E1","E2","E3")] if phase=="full" else
           [(g,"fold"+str(f["fold"]),f) for f in folds for g in ("E0","E1","E2")])
    preferred=tasks[shard::shards]+[t for i,t in enumerate(tasks) if i%shards!=shard]
    pair_cache={};view_cache={}
    started=time.time();completed_here=0
    def completed(task):
        try:return bool(read(OUT/"models"/task[0]/task[1]/"model.json").get("complete"))
        except __import__("json").JSONDecodeError:return False
    while True:
        pending=[t for t in preferred if not completed(t)]
        if not pending:break
        claimed=False
        for g,name,fold in pending:
            target=OUT/"models"/g/name;target.mkdir(parents=True,exist_ok=True)
            handle=(target/".fit.lock").open("a+")
            try:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if read(target/"model.json").get("complete"):continue
                claimed=True;sources=None if fold is None else fold["train_sources"]
                if g not in pair_cache:pair_cache[g]=rows(DATA/(g+"_pairs.jsonl"))
                pairs=pair_cache[g]
                if g not in view_cache:
                    pairviews=rows(DATA/(g+"_views.jsonl"))
                    view_cache[g]=list({v["view_id"]:v for v in pairviews+views}.values()) if pairviews else views
                allviews=view_cache[g]
                write(OUT/"status"/("train_"+str(shard)+".json"),dict(state="running",group=g,fold=name,pid=os.getpid(),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),started=time.time(),schedule="claim unfinished group/fold under exclusive task lock"))
                historical=read(target/"model.json")
                if g=="E0" and historical.get("m1_m2_historical_reuse"):
                    from .training import refit_m3
                    with np.load(target/"m2.npz") as a:support={k:a[k] for k in a.files}
                    allowed=set(sources)
                    subset=[p for p in pairs if p["source_id"] in allowed]
                    calibrators=[v for v in natural if v["source_id"] in allowed]
                    m3=refit_m3(g,subset,store,calibrators,support,target,delta=CONFIG["delta"])
                    historical.update(M3=m3,delta=CONFIG["delta"],fresh_modules=["M3"],complete=True)
                    write(target/"model.json",historical)
                    result={"historical_M1_M2_reused":True,"M3":m3}
                else:
                    result=fit_group(g,pairs,store,allviews,natural,target,train_sources=sources,patcher=patcher,delta=CONFIG["delta"])
                metadata=read(target/"model.json")
                metadata.update(complete=True,group=g,fold=name,train_sources=sources or read(OUT/"folds.json")["all_sources"],validation_sources=fold["validation_sources"] if fold else [],gpu_cost=dict(patcher.cost),total_worker_seconds=time.time()-started,worker_pid=os.getpid(),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
                write(target/"model.json",metadata)
                append(OUT/"fit_jobs.jsonl",dict(group=g,fold=name,complete=True,worker_pid=os.getpid(),summary=result))
                completed_here+=1
            finally:
                try:fcntl.flock(handle,fcntl.LOCK_UN)
                finally:handle.close()
            if claimed:break
        if not claimed:time.sleep(2)
    write(OUT/"status"/("train_"+str(shard)+".json"),dict(state="complete",phase=phase,seconds=time.time()-started,cost=patcher.cost,pid=os.getpid(),completed_jobs=completed_here,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["extract","train"],required=True)
    p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=4)
    p.add_argument("--stages",default="fit,dev");p.add_argument("--limit",type=int);p.add_argument("--heads")
    p.add_argument("--smoke",action="store_true");p.add_argument("--phase",default="full")
    p.add_argument("--fanout-child",action="store_true")
    a=p.parse_args()
    try:
        if a.mode=="extract":
            selected_heads=[int(x) for x in a.heads.split(",")] if a.heads else None
            parallel=int(CONFIG.get("extraction_workers_per_gpu",1))
            can_parallel=not a.fanout_child and a.limit is None and not a.smoke and not set(a.stages.split(",")).intersection({"fit","dev"})
            if parallel==2 and can_parallel:
                import subprocess,sys
                child_cmd=[sys.executable,"-u","-m","experiments.aster_cr_retrain_paired_v2_20261111.worker","--mode","extract","--shard",str(a.shard*2+1),"--shards",str(a.shards*2),"--stages",a.stages,"--fanout-child"]
                if a.heads:child_cmd.extend(["--heads",a.heads])
                child_log=(OUT/"logs"/("extract_actor_"+str(a.shard*2+1)+"_"+a.stages.split(",")[0]+".log")).open("a")
                child=subprocess.Popen(child_cmd,stdout=child_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(OUT/"actor_groups"/(a.stages.split(",")[0]+"_"+str(a.shard)+".json"),dict(gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),parent=os.getpid(),child=child.pid,shards=[a.shard*2,a.shard*2+1],total_shards=a.shards*2))
                extract(a.shard*2,a.shards*2,a.stages,a.limit,selected_heads,a.smoke)
                ret=child.wait();child_log.close()
                if ret:raise RuntimeError("Companion extraction worker exited with code "+str(ret))
            else:
                extract(a.shard,a.shards,a.stages,a.limit,selected_heads,a.smoke)
        else:train(a.shard,a.shards,a.phase)
    except Exception as exc:
        write(OUT/"status"/("error_"+str(a.shard)+".json"),dict(state="error",mode=a.mode,error=str(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
