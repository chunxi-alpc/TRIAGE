"""Unified-queue-only entry: fit/dev extraction -> adaptive fit -> dev -> freeze -> test."""
import argparse
import fcntl
import os
import time
import traceback
from pathlib import Path
from collections import ChainMap
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, load_inputs
from experiments.aster_cr_retrain_paired_v2_20261111 import training
from experiments.aster_e678.run import extract as legacy_extract
from experiments.aster_e678.evaluate import DynamicStore
from .adaptive_m1 import ScopedStore, pair_ids, representatives
from .prepare import OUT, SOURCE, KEYS
from .patcher import ShamDonorPatcher

SCHEDULER=Path("/home/gcx/aster-gpu-queue-20261113/scheduler.py")

def require_queue(gpu=None):
    assigned=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if assigned not in ("0","1","5","6") or (gpu is not None and assigned != str(gpu)):
        raise RuntimeError("GPU must be allocated from 0,1,5,6 by the unified queue")
    # The shared scheduler launches workers directly; standalone flags cannot bypass this.
    parent=Path("/proc")/str(os.getppid())/"cmdline"
    cmd=parent.read_bytes().split(b"\0") if parent.exists() else []
    if str(SCHEDULER).encode() not in cmd:
        raise RuntimeError("No unified scheduler parent; no model loaded")
    return int(assigned)

class SplitStore(DynamicStore):
    """Load only declared score files; fitting never even opens scores_test.jsonl."""
    def __init__(self, root, stages=("fit","dev")):
        self.out=Path(root)
        self.required_heads=np.asarray([],int)
        self.records={}
        for stage in stages:
            if stage not in ("fit","dev","test"):raise ValueError(stage)
            for r in rows(self.out/"cache"/("scores_"+stage+".jsonl")):
                if r.get("status","ok")=="ok" and "ell" in r and "feature" in r:
                    self.records[r.get("uid",r.get("view_id"))]=r
        self.scores=self.records

def extract(root,stage,inputs,engine):
    if stage=="test" and not read(Path(root)/"frozen.json").get("frozen"):
        raise ValueError("Test feature extraction before parameter freeze")
    return legacy_extract(root,stage,inputs,engine)

def freeze_identity(root, engine, inputs, ids):
    path=root/"cache/identity.json"
    metadata=engine.describe()
    tokenizer=engine.tokenizer.backend_tokenizer.to_str()
    identity=dict(model=metadata,tokenizer=tokenizer)
    if path.exists() and read(path)!=identity:
        raise ValueError("Checkpoint/tokenizer/scoring protocol changed")
    write(path,identity)
    # Exact prompt comparisons avoid reusing a cache under a changed input.
    for uid in ids:
        p=root/"cache/prompts"/(uid+".json")
        current=dict(prompt=inputs[uid]["prompt"],rendered=engine.render(inputs[uid]["prompt"]))
        if p.exists() and read(p)!=current:
            raise ValueError("Input prompt changed: "+uid)
        if not p.exists():write(p,current)
    return dict(model=metadata,identity_file=str(path))

def smoke(key):
    require_queue()
    root=OUT/"smoke"/key
    progress(root,"loading_model")
    from experiments.aster_e678.engine import Engine
    engine=Engine(key,metadata_path=root/"model_metadata.json")
    base=OUT/"models"/key
    pairs=rows(base/"data/dev_pairs.jsonl")
    reps=representatives(pairs,"dev")["A"][:2]
    extra={r["uid"]:r for r in rows(base/"data/extra_inputs.jsonl")}
    inputs=ChainMap(extra,load_inputs(SOURCE))
    fit_pair=rows(base/"data/E1_pairs.jsonl")[0]
    fit_ids=sorted(pair_ids([fit_pair]));dev_ids=sorted(pair_ids(reps))
    write(root/"stage_ids.json",dict(fit=fit_ids,dev=dev_ids))
    write(root/"gradient_ids.json",[fit_pair["left"]])
    identity=freeze_identity(root,engine,inputs,fit_ids+dev_ids)
    progress(root,"extract_fit")
    extract(root,"fit",inputs,engine)
    progress(root,"extract_dev")
    extract(root,"dev",inputs,engine)
    store=SplitStore(root)
    patcher=ShamDonorPatcher(root/"patches",store,inputs,engine,reps,identity)
    true=patcher.effects(reps[0],reps[0],[0])
    sham=patcher.effects(reps[0],reps[1],[0],True)
    if not np.isfinite(np.concatenate([true,sham])).all():raise ValueError("Invalid smoke scores")
    patcher.close()
    import torch
    write(root/"complete.json",dict(complete=True,model=key,total_heads=engine.num_heads,
        true_effect=true.tolist(),sham_effect=sham.tolist(),peak_gib=torch.cuda.max_memory_allocated()/2**30))

def progress(root,stage,**fields):
    write(root/"execution.json",dict(state="running",stage=stage,pid=os.getpid(),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),**fields))

def run_model(key,gpu=None):
    require_queue(gpu)
    root=OUT/"models"/key
    if read(root/"completion.json").get("complete"):return
    if read(OUT/"data_audit.json").get("protocol_conflict") or not read(OUT/"data_resolution.json").get("resolved"):
        raise RuntimeError("Development-data conflict requires resolution before model loading")
    cfg=read(root/"configuration.json")
    if not cfg:raise ValueError("Run CPU preparation first")
    progress(root,"loading_model")
    from experiments.aster_e678.engine import Engine
    engine=Engine(key,metadata_path=root/"model_metadata.json")
    inputs=ChainMap({r["uid"]:r for r in rows(root/"data/extra_inputs.jsonl")},load_inputs(SOURCE))
    stages=read(root/"stage_ids.json")
    progress(root,"prepare_input_identity")
    identity=freeze_identity(root,engine,inputs,stages["fit"]+stages["dev"])
    progress(root,"extract_fit")
    extract(root,"fit",inputs,engine)
    progress(root,"extract_dev")
    extract(root,"dev",inputs,engine)
    store=SplitStore(root)
    pairs=rows(root/"data/E1_pairs.jsonl");dev=rows(root/"data/dev_pairs.jsonl")
    views=rows(root/"data/E1_views.jsonl");natural=rows(root/"data/natural_views.jsonl")
    target=root/"models/E1/full"
    if not read(target/"model.json").get("complete"):
        progress(root,"adaptive_M1_M2_M3")
        patcher=ShamDonorPatcher(root/"cache/adaptive_exact",store,inputs,engine,dev,identity)
        try:
            summary=training.fit_group("E1",pairs,store,views,natural,target,patcher=patcher,m1_mode="adaptive",
                adaptive=dict(dev_pairs=dev,dev_features=store,total_heads=engine.num_heads,heads_per_layer=engine.n_heads))
        finally:patcher.close()
        if summary["status"]=="selection_incomplete":
            write(root/"completion.json",dict(complete=False,diagnostics_complete=True,status="selection_incomplete"))
            progress(root,"selection_incomplete")
            write(root/"queue_done.json",dict(complete=True,outcome="selection_incomplete"))
            from .report import report
            report()
            return
    from .evaluate import evaluate
    if not read(root/"frozen.json").get("frozen"):
        progress(root,"evaluate_dev")
        development=evaluate(root,"dev",store)
        write(root/"frozen.json",dict(frozen=True,configuration=cfg,
            development_role="feature selection; model-selection performance",
            development=development,thresholds_selected_from_test=False))
    progress(root,"extract_test")
    freeze_identity(root,engine,inputs,stages["test"])
    extract(root,"test",inputs,engine)
    progress(root,"evaluate_test")
    evaluate(root,"test",SplitStore(root,("test",)))
    progress(root,"complete")
    write(root/"completion.json",dict(complete=True,status="complete",model=key))
    write(root/"queue_done.json",dict(complete=True,outcome="complete"))
    from .report import report
    report()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=KEYS,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    root=OUT/("smoke" if a.smoke else "models")/a.model
    root.mkdir(parents=True,exist_ok=True)
    with (root/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.smoke:smoke(a.model)
            else:run_model(a.model)
        except Exception as exc:
            write(root/"execution.json",dict(state="failed",error=str(exc),traceback=traceback.format_exc()))
            raise
if __name__=="__main__":main()
