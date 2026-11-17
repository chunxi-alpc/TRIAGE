"""Four queue-assigned replicas share Qwen inputs and confirmation pairs."""
import argparse
import fcntl
import gzip
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write,append,load_inputs
from .prepare import OUT,PRIVATE,SOURCE
from .revision import Config,ExactPatcher,paired_effect

WORKERS=4

def enabled(key):
    return key=="qwen" and os.environ.get("ASTER_PARALLEL_WORKERS")=="4"

def shard(items,worker):
    if worker not in range(WORKERS):raise ValueError("Invalid worker")
    return items[worker::WORKERS]

def initialize(key):
    session=str(time.time_ns())
    write(PRIVATE/key/"parallel/session.json",dict(session=session,leader_pid=os.getpid(),workers=WORKERS))
    return session

def alive(pid):
    try:return Path("/proc",str(pid),"stat").read_text().split(") ",1)[1].split()[0]!="Z"
    except FileNotFoundError:return False

def phase_paths(key,request):
    stage=PRIVATE/key/"parallel"/request["session"]/request["stage"]
    public=OUT/"models"/key/"parallel"/request["session"]/request["stage"]
    return stage,public

def run_shard(key,request,engine,inputs,worker,on_progress=None):
    import torch
    from .runtime import FeatureStore
    stage,public=phase_paths(key,request)
    public.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();computed=0;reused=0;forwards=0
    assigned=shard(request["items"],worker)
    def status(complete=False):
        value=dict(complete=complete,worker=worker,gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   stage=request["stage"],completed=computed+reused,newly_computed=computed,
                   reused=reused,total=len(assigned),seconds=time.monotonic()-started,
                   confirmation_forwards=forwards,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        write(public/("worker"+str(worker)+".json"),value)
        if on_progress:on_progress()
        return value
    status()
    def check_workers():
        failures=list(stage.glob("failure_*.json"))
        if failures:raise RuntimeError("Parallel stage failed: "+read(failures[0])["error"])
        session=read(PRIVATE/key/"parallel/session.json")
        if session.get("leader_pid") and not alive(session["leader_pid"]):
            raise RuntimeError("Parallel leader exited")
        for helper in (1,2,3):
            state=read(PRIVATE/key/"parallel"/("worker"+str(helper)+"_process.json"))
            if state.get("session")==request["session"] and not alive(state["pid"]):
                raise RuntimeError("Parallel helper exited")
    try:
        if request["kind"]=="extract":
            scope=request["scope"]
            if scope=="test" and not read(OUT/"models"/key/"frozen.json").get("frozen"):
                raise ValueError("Test extraction before frozen parameters")
            gradients=set(read(OUT/"models"/key/"gradient_plan.json")["required_ids"]) if scope=="fit" else set()
            cache=PRIVATE/key/"cache"
            scorefile=cache/("scores_"+scope+".jsonl")
            lockfile=cache/("scores_"+scope+".lock")
            cache.mkdir(parents=True,exist_ok=True)
            with lockfile.open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                existing=set(FeatureStore(PRIVATE/key,(scope,)).records)
            for uid in assigned:
                check_workers()
                if uid in existing:
                    reused+=1
                    continue
                context=engine.prepare(inputs[uid]["prompt"])
                result=engine.run(context,capture=True,grad=uid in gradients)
                arrays={field:result.pop(field).numpy().astype(np.float32)
                        for field in ("heads","grads") if field in result}
                if not np.isfinite(result["ell"]) or not all(np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("Nonfinite feature: "+uid)
                target=cache/"features"/(uid+".npz")
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():target.rename(target.with_name(target.name+".interrupted."+str(time.time_ns())))
                np.savez_compressed(target,**arrays)
                record=dict(result,uid=uid,view_id=uid,feature=str(target),status="ok",
                            has_grads=uid in gradients,available_heads="all")
                with lockfile.open("a") as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    append(scorefile,record)
                computed+=1
                del context,arrays
                if computed<=3 or computed%100==0:status()
        elif request["kind"]=="patch":
            cfg=Config(**request["config"])
            store=FeatureStore(PRIVATE/key,("fit",))
            patcher=ExactPatcher(store,inputs,engine,cfg)
            for pair in assigned:
                check_workers()
                heads=request["candidates"][pair["kind"]]["heads"]
                target=PRIVATE/key/"paired"/(pair["pair_id"]+".json")
                if target.exists():
                    batch=read(target)
                    if len(batch)!=len(heads) or {int(r["head"]) for r in batch}!=set(heads):
                        raise ValueError("Incomplete saved pair cache")
                    for row in batch:paired_effect(row,cfg)
                    reused+=1
                else:
                    def save_raw(row):
                        with gzip.open(public/("raw_worker"+str(worker)+".jsonl.gz"),"at",encoding="utf-8") as f:
                            f.write(json.dumps(row,allow_nan=False)+"\n")
                    batch=patcher.measure(pair,heads,on_record=save_raw)
                    write(target,batch)
                    computed+=1
                    forwards+=2+4*len(heads) if heads else 0
                status()
        else:raise ValueError("Unknown parallel phase")
        result=status(True)
        write(stage/("done_"+str(worker)+".json"),result)
        return result
    except Exception as exc:
        failure=dict(worker=worker,error=str(exc),traceback=traceback.format_exc())
        write(public/("failure_"+str(worker)+".json"),failure)
        write(stage/("failure_"+str(worker)+".json"),failure)
        raise

def execute(key,request,engine,inputs,progress):
    session=read(PRIVATE/key/"parallel/session.json")["session"]
    request=dict(request,session=session)
    identifiers=[x if isinstance(x,str) else x["pair_id"] for x in request["items"]]
    if len(set(identifiers))!=len(identifiers):raise ValueError("Duplicate parallel work items")
    stage,public=phase_paths(key,request)
    stage.mkdir(parents=True,exist_ok=False)
    write(stage/"request.json",request)
    write(PRIVATE/key/"parallel/active.json",dict(request=str(stage/"request.json"),session=session))
    def report_progress():
        states=[read(public/("worker"+str(i)+".json")) for i in range(WORKERS)]
        complete=sum(x.get("completed",0) for x in states)
        fields=dict(completed=complete+request.get("cached_at_start",0),
                    total=len(request["items"])+request.get("cached_at_start",0),workers=WORKERS,
                    active_gpus=[x["gpu"] for x in states if x.get("gpu") is not None],
                    confirmation_forwards=sum(x.get("confirmation_forwards",0) for x in states))
        elapsed=time.time()-request["created_at"]
        if request["kind"]=="patch" and complete:
            fields["estimated_remaining_hours"]=(len(request["items"])-complete)*elapsed/complete/3600
        progress(request["stage"],**fields)
    run_shard(key,request,engine,inputs,0,on_progress=report_progress)
    last=0
    while True:
        failures=[read(p) for p in stage.glob("failure_*.json")]
        if failures:raise RuntimeError("Parallel worker failed: "+json.dumps(failures))
        done=list(stage.glob("done_*.json"))
        states=[read(public/("worker"+str(i)+".json")) for i in range(WORKERS)]
        if time.monotonic()-last>15 or len(done)==WORKERS:
            report_progress()
            last=time.monotonic()
        if len(done)==WORKERS:break
        for worker in range(1,WORKERS):
            state=read(PRIVATE/key/"parallel"/("worker"+str(worker)+"_process.json"))
            if state.get("session")==session and not alive(state["pid"]):
                raise RuntimeError("Parallel helper exited before stage completion")
        time.sleep(1)
    summary=dict(complete=True,workers=states,wall_seconds=time.time()-request["created_at"],
                 completed=len(request["items"]),reused_before_stage=request.get("cached_at_start",0))
    write(public/"summary.json",summary)
    return summary

def extract_parallel(key,scope,inputs,engine,progress,ids,pending):
    return execute(key,dict(kind="extract",stage="extract_"+scope,scope=scope,items=pending,
                            cached_at_start=len(ids)-len(pending),created_at=time.time()),engine,inputs,progress)

def patch_parallel(key,pairs,candidates,cfg,engine,inputs,progress):
    from dataclasses import asdict
    execute(key,dict(kind="patch",stage="exact_patching",items=pairs,candidates=candidates,
                     config=asdict(cfg),created_at=time.time()),engine,inputs,progress)
    return [row for p in pairs for row in read(PRIVATE/key/"paired"/(p["pair_id"]+".json"))]

def serve(key,worker):
    from experiments.aster_e678_adaptive_20261116.run import require_queue
    require_queue()
    import torch
    torch.set_num_threads(4)
    runtime=PRIVATE/key/"parallel"
    runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/("worker"+str(worker)+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            session=read(runtime/"session.json")
            if session and alive(session["leader_pid"]):break
            if read(OUT/"models"/key/"failure.json"):raise RuntimeError("Leader failed before parallel initialization")
            time.sleep(1)
        write(runtime/("worker"+str(worker)+"_process.json"),dict(pid=os.getpid(),session=session["session"]))
        from experiments.aster_e678.engine import Engine
        from .runtime import checkpoint_identity
        root=OUT/"models"/key/"parallel"/session["session"]
        engine=Engine(key,metadata_path=root/("worker"+str(worker)+"_model_metadata.json"))
        checkpoint_identity(key,engine)
        inputs=load_inputs(SOURCE)
        completed=set()
        while True:
            if read(OUT/"models"/key/"completion.json").get("complete"):return
            if read(OUT/"models"/key/"failure.json") or not alive(session["leader_pid"]):
                raise RuntimeError("Parallel leader failed or exited")
            active=read(runtime/"active.json")
            if active.get("session")==session["session"] and active["request"] not in completed:
                request=read(active["request"])
                run_shard(key,request,engine,inputs,worker)
                completed.add(active["request"])
            else:time.sleep(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=("qwen",),required=True)
    parser.add_argument("--worker",type=int,choices=(1,2,3),required=True)
    args=parser.parse_args()
    serve(args.model,args.worker)
