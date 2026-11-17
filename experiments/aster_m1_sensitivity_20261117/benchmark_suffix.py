"""Queue-only numerical equivalence and throughput comparison on training inputs."""
import json,time,os
from pathlib import Path
import numpy as np
from .prepare import OUT,RUN,PRIVATE,DATA
from .audit import read,lines
from .suffix import SuffixRunner
from experiments.aster_cr_retrain_paired_v2_20261111.common import write,load_inputs
from experiments.aster_e678_adaptive_20261116.run import require_queue

def run():
    gpu=require_queue()
    root=OUT/"speedup";root.mkdir(exist_ok=True)
    from experiments.aster_e678.engine import Engine
    import torch
    torch.set_num_threads(4)
    records={r["uid"]:r for r in lines(PRIVATE/"cache/scores_fit.jsonl")}
    # Three deterministic input lengths from the actual missing-request population.
    populations=[]
    for w in range(4):
        requests=read(RUN/("requests_"+str(w)+".json"))
        populations.extend(requests)
    targets=sorted({r["target"] for r in populations},key=lambda u:(records[u].get("tokens",0),u))
    uids=[targets[0],targets[len(targets)//2],targets[-1]]
    chosen={}
    for uid in uids:
        by_layer={}
        for row in populations:
            if row["target"]==uid:
                by_layer.setdefault(row["head"]//32,row)
        chosen[uid]=[by_layer[k] for k in sorted(by_layer)]
    del populations
    engine=Engine("qwen",metadata_path=root/"model_metadata.json")
    inputs=load_inputs(DATA)
    runner=SuffixRunner(engine)
    values=[];serial_seconds=0.;suffix_seconds=0.
    @__import__('functools').lru_cache(maxsize=64)
    def heads(uid):
        with np.load(records[uid]["feature"],allow_pickle=False) as f:
            return f["heads"].astype("float32").reshape(1152,-1)
    for uid,requests in chosen.items():
        context=engine.prepare(inputs[uid]["prompt"])
        fresh=runner.prepare(context)
        if abs(fresh-records[uid]["ell"])>1e-4+1e-5*abs(fresh):raise ValueError("Baseline cache mismatch")
        # Include self replacement, in addition to actual matched-control requests.
        checks=requests+[dict(target=uid,donor=uid,head=r["head"]) for r in requests[::max(1,len(requests)//4)]]
        for r in checks:
            donor=heads(r["donor"])[r["head"]]
            torch.cuda.synchronize();start=time.perf_counter()
            original=float(engine.run(context,replacements={r["head"]:donor},capture=False)["ell"])
            torch.cuda.synchronize();serial=time.perf_counter()-start
            start=time.perf_counter()
            fast=runner.score(r["head"],donor)
            torch.cuda.synchronize();suffix=time.perf_counter()-start
            serial_seconds+=serial;suffix_seconds+=suffix
            values.append(dict(target=uid,donor=r["donor"],head=r["head"],layer=r["head"]//32,
                tokens=records[uid].get("tokens"),serial=original,suffix=fast,
                absolute_difference=abs(original-fast),serial_seconds=serial,suffix_seconds=suffix))
            write(root/"progress.json",dict(completed=len(values),gpu=gpu))
    result=dict(complete=True,gpu=gpu,checks=len(values),training_targets=len(uids),
        bitwise_equal=all(r["serial"]==r["suffix"] for r in values),
        max_absolute_difference=max(r["absolute_difference"] for r in values),
        serial_seconds=serial_seconds,suffix_seconds=suffix_seconds,
        measured_forward_speedup=serial_seconds/suffix_seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30,
        no_scientific_parameter_change=True,baseline_and_all_downstream_operations_preserved=True)
    write(root/"equivalence_records.json",values)
    write(root/"benchmark.json",result)
    if not result["bitwise_equal"]:raise RuntimeError("Suffix replay changed scalar scores; retain serial implementation")
    write(root/"benchmark_done.json",result)
    print(json.dumps(result))

if __name__=="__main__":
    try:run()
    except Exception as exc:
        import traceback
        write(OUT/"speedup"/"benchmark_failure.json",dict(error=str(exc),traceback=traceback.format_exc()))
        raise
