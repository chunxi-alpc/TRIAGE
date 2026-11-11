"""Consume only previously unscored fit-path inputs to benchmark extraction concurrency."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from .common import OUT,CACHE,DATA,REPO,Store,read,rows,write,append
PACKAGE="experiments.aster_cr_retrain_paired_v2_20261111"
def probe(group):
    import torch
    from experiments.aster_cr.engine import Engine
    torch.set_num_threads(4);store=Store();inputs=read(OUT/"runtime/extraction_probe_inputs.json")["groups"][group]
    eng=Engine("qwen");start=time.perf_counter();start_epoch=time.time();count=0;tokens=0
    for uid in inputs:
        v=store.inputs[uid];context=eng.prepare(v["prompt"]);r=eng.run(context,capture=True)
        arrays=dict(heads=r.pop("heads").numpy().astype(np.float16))
        target=CACHE/"features"/(uid+".npz");target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,**arrays)
        append(CACHE/("scores_speed_"+str(group)+".jsonl"),dict(view_id=uid,uid=uid,feature=str(target),status="ok",available_heads="all",has_grads=False,**r))
        count+=1;tokens+=r["tokens"];del context
    elapsed=time.perf_counter()-start
    write(OUT/("runtime/extraction_probe_"+str(group)+".json"),dict(group=group,inputs=count,seconds=elapsed,per_second=count/elapsed,start_epoch=start_epoch,end_epoch=time.time(),mean_tokens=tokens/max(count,1),peak_gib=torch.cuda.max_memory_allocated()/2**30))
def coordinate():
    # GPU 6 becomes available when its existing fit-cache worker exits. Other GPUs continue training.
    old=read(OUT/"active_workers.json");workers=old.get("workers",[])
    wait=next((w["pid"] for w in workers if w["gpu"]==6),None) if old.get("stage")=="fit_cache" else None
    if wait:
        while Path("/proc/"+str(wait)).exists():
            stat=Path("/proc/"+str(wait)+"/stat")
            if stat.exists() and stat.read_text().split()[2]=="Z":break
            time.sleep(2)
    store=Store();chosen=[]
    for v in store.inputs.values():
        stages=set(v.get("stages",[]))
        if "path_fit" in stages and not stages.intersection({"fit","dev","test"}) and v["view_id"] not in store.records:
            chosen.append(v["view_id"])
            if len(chosen)==192:break
    if len(chosen)<192:raise RuntimeError("Not enough new fit-path inputs for disjoint performance probe")
    write(OUT/"runtime/extraction_probe_inputs.json",dict(groups=[chosen[i::3] for i in range(3)],inputs_are_future_required_evaluation_cache=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="6",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
    def launch(group):
        log=(OUT/"logs"/("extraction_probe_"+str(group)+".log")).open("a")
        proc=subprocess.Popen([sys.executable,"-u","-m",PACKAGE+".extraction_probe","--group",str(group)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        return proc,log
    p,log=launch(0)
    if p.wait():raise RuntimeError("Single-worker probe failed")
    log.close()
    jobs=[launch(g) for g in (1,2)]
    for p,log in jobs:
        if p.wait():raise RuntimeError("Double-worker probe failed")
        log.close()
    single=read(OUT/"runtime/extraction_probe_0.json")
    double=[read(OUT/("runtime/extraction_probe_"+str(g)+".json")) for g in (1,2)]
    seconds=max(d["end_epoch"] for d in double)-min(d["start_epoch"] for d in double)
    combined=sum(d["inputs"] for d in double)/seconds
    write(OUT/"extraction_performance.json",dict(single=single,double=double,double_combined_per_second=combined,throughput_ratio=combined/single["per_second"],distinct_fresh_inputs_cached=192,gpu=6,all_forward_shapes_remain_single_input=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--group",type=int);a=p.parse_args()
    if a.group is None:coordinate()
    else:probe(a.group)
