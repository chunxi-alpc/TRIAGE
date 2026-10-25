"""Eight deterministic cache checks and end-to-end baseline cost, never full rerun."""
import json,time
import numpy as np
import torch
from run_gpu import inputs,DEFAULT_OLD,DEFAULT_OUT,MODEL,atomic,sha,HERE
from sparse_engine import SparseEngine

def main():
    rows,cache=inputs(DEFAULT_OLD);e=SparseEngine(str(MODEL));results=[];start=time.time()
    for i in np.linspace(0,799,8,dtype=int):
        c=rows[i];safe={k:c[k] for k in ('patient','rules','options')};old=cache[c['id'],c['variant']]
        timings={};scores={}
        for name,kw in [('original',{}),('zero_midlate4',dict(mode='sparse',p=old['p'],geometry='midlate4',strength=0.))]:
            torch.cuda.synchronize();t=time.time();_,values=e.answer(safe,**kw);torch.cuda.synchronize()
            timings[name]=time.time()-t;scores[name]=values
        results.append(dict(sorted_record_index=int(i),variant=c['variant'],seconds=timings,
            original_cache_max_abs=float(np.max(np.abs(np.array(scores['original'])-old['scores']['original']))),
            original_cache_same_answer=int(np.argmax(scores['original']))==old['answers']['original'],
            zero_vs_original_max_abs=float(np.max(np.abs(np.array(scores['zero_midlate4'])-scores['original'])))))
    manifests=[json.loads((DEFAULT_OUT/f'shard{gpu}/manifest.json').read_text()) for gpu in (0,1,6)]
    core=['sparse_attention.py','sparse_engine.py','run_gpu.py']
    unchanged=all(m['code_sha256'][f]==sha(HERE/f) for m in manifests for f in core)
    assert unchanged,'core code changed during full GPU run'
    result=dict(rows=8,forward_calls=16,selection='8 evenly spaced sorted record indices; no label selection',results=results,
        unchanged_gpu_core=unchanged,elapsed_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    atomic(HERE/'runtime_validation.json',result)
    print(json.dumps(dict(rows=8,unchanged_gpu_core=unchanged,original_cache_max_abs=max(r['original_cache_max_abs'] for r in results),same_answers=sum(r['original_cache_same_answer'] for r in results))))
if __name__=='__main__':main()
