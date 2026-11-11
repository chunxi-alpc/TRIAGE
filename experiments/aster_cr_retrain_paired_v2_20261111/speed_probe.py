"""Measure batching on cached training inputs; do not fit or generate data."""
import os,time,json
from pathlib import Path
import numpy as np
import torch
from .common import OUT,CACHE,DATA,Store,rows,write
from .worker import Patcher
from .fast_patch import batched_single_head,prefix_nbytes

def main():
    torch.set_num_threads(4)
    store=Store();patcher=Patcher(store)
    from experiments.aster_cr.engine import Engine
    eng=Engine("qwen");patcher.engine=eng
    saved=next((CACHE/"exact").glob("*.json"))
    left,right=saved.stem.split("__")
    pair=next(p for p in rows(DATA/"E2_pairs.jsonl") if p["left"]==left and p["right"]==right)
    a=patcher.context(left);b=patcher.context(right)
    la=store.get(left);rb=store.get(right)
    # Spread candidates over the historical selected layers.
    heads=rows(DATA/"E2_pairs.jsonl") # immediately release metadata after choosing fixed heads below
    heads=json.loads((OUT/"models/E0/full/m1.json").read_text())["heads"][:16]
    references={};started=time.perf_counter()
    for context,donor,name in ((a,rb,"forward"),(b,la,"reverse")):
        references[name]={h:eng.run(context,replacements={h:donor["heads"][h]},capture=False)["ell"] for h in heads}
    torch.cuda.synchronize();single_seconds=time.perf_counter()-started
    results=[]
    for batch in (1,4,8,16):
        before=time.perf_counter()
        measured={}
        for context,donor,name in ((a,rb,"forward"),(b,la,"reverse")):
            measured[name]=batched_single_head(eng,context,donor["heads"],heads,batch_size=batch)
        torch.cuda.synchronize();seconds=time.perf_counter()-before
        ref=np.array([references[n][h] for n in ("forward","reverse") for h in heads])
        actual=np.array([measured[n][h] for n in ("forward","reverse") for h in heads])
        errors=np.abs(actual-ref)
        results.append(dict(batch_size=batch,seconds=seconds,speedup=single_seconds/seconds,
                            max_abs_logit_difference=float(errors.max()),mean_abs_logit_difference=float(errors.mean()),
                            rms_logit_difference=float(np.sqrt(np.mean(errors**2))),peak_gib=torch.cuda.max_memory_allocated()/2**30,
                            patched_scores=measured))
    write(OUT/"patch_performance.json",dict(pair_id=pair["pair_id"],heads=heads,original_single_seconds=single_seconds,prefix_gib=prefix_nbytes(a)/2**30,results=results,algorithm="actual single-head intervention in every row; canonical BF16 final token with FP32 projection"))
if __name__=="__main__":main()
