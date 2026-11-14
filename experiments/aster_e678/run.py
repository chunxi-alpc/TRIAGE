"""Resumable per-checkpoint E1 fit, development freeze, and missing evaluations."""
import argparse, fcntl, gc, json, os, time, traceback
from pathlib import Path
from collections import Counter, ChainMap
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from experiments.aster_cr_retrain_paired_v2_20261111.common import rows, read, write, append, load_inputs
from experiments.aster_cr_retrain_paired_v2_20261111 import training
from .prepare import OUT, SOURCE
from .evaluate import DynamicStore, evaluate_model

def state(root, stage, **kw):
    write(root / "run_status.json", dict(state="running", stage=stage, pid=os.getpid(),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw))
class Patcher:
    def __init__(self, root, store, inputs, engine):
        self.root, self.store, self.inputs, self.engine = root, store, inputs, engine
    def __call__(self, pair, heads):
        path = self.root / "cache/exact" / (pair["left"] + "__" + pair["right"] + ".json")
        cached = read(path)
        needed = [int(h) for h in heads if str(h) not in cached]
        if needed:
            left = self.store.get(pair["left"])["heads"]
            right = self.store.get(pair["right"])["heads"]
            # Keep only two immutable prefixes; no unbounded disk prefix cache.
            a = self.engine.prepare(self.inputs[pair["left"]]["prompt"])
            b = self.engine.prepare(self.inputs[pair["right"]]["prompt"])
            for i,h in enumerate(needed):
                forward = self.engine.run(a, replacements={h:right[h]}, capture=False)["ell"]
                reverse = self.engine.run(b, replacements={h:left[h]}, capture=False)["ell"]
                cached[str(h)] = [float(forward), float(reverse)]
                if (i+1) % 16 == 0:
                    write(path, cached)
            write(path, cached)
            del a,b
        return [(int(h), *cached[str(int(h))]) for h in heads]
def available(record, need_grad):
    return bool(record and record.get("status", "ok") == "ok" and
                Path(record.get("feature", "")).is_file() and
                (not need_grad or record.get("has_grads", False)))
def extract(root, stage, inputs, engine):
    import torch
    store = DynamicStore(root)
    ids = read(root / "stage_ids.json")[stage]
    gradients = set(read(root / "gradient_ids.json", [])) if stage == "fit" else set()
    pending = [uid for uid in ids if not available(store.records.get(uid), uid in gradients)]
    started = time.monotonic()
    for i,uid in enumerate(pending):
        v = inputs[uid]
        context = engine.prepare(v["prompt"])
        result = engine.run(context, capture=True, grad=uid in gradients)
        arrays = {"heads":result.pop("heads").numpy().astype(np.float16)}
        if "grads" in result:
            arrays["grads"] = result.pop("grads").numpy().astype(np.float16)
        target = root / "cache/features" / (uid + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
        append(root / "cache" / ("scores_" + stage + ".jsonl"), dict(result, uid=uid, view_id=uid,
               feature=str(target), status="ok", available_heads="all", has_grads=uid in gradients))
        del context
        if i < 3 or (i+1) % 50 == 0:
            state(root, "extract_" + stage, completed=i+1, total=len(pending),
                  peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-started)
    write(root / "performance" / ("extract_" + stage + ".json"), dict(complete=True,
          newly_computed=len(pending), reused=len(ids)-len(pending), total=len(ids),
          seconds=time.monotonic()-started, peak_gib=torch.cuda.max_memory_allocated()/2**30))
def calibration(root, store):
    vv=rows(root / "data/natural_views.jsonl")
    scores=np.array([store.ell(v["view_id"]) for v in vv])
    labels=np.array([v["label"] for v in vv])
    derivative=lambda b: float(np.mean(expit(scores+b)-labels))
    lo,hi=-1.,1.
    while derivative(lo)>0: lo*=2
    while derivative(hi)<0: hi*=2
    b=float(brentq(derivative,lo,hi))
    write(root / "models/E1/full/calibration.json", dict(b=b, split="source fit", nodes=len(vv),
          method="intercept-only logistic calibration", target_labels_used=False))
def run_model(key, gpu, unified_scheduler=False):
    if gpu not in (0,1,4,5,6):
        raise ValueError("GPU outside user-authorized set")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    import torch
    torch.set_num_threads(4)
    root=OUT / "models" / key
    scope_policy=read(OUT / "only_missing_policy.json")
    if key in scope_policy.get("stopped_model_pipelines", []):
        raise RuntimeError("Full pipeline stopped by user: run only previously unrun experiments")
    config=read(root / "configuration.json")
    if not config:
        raise RuntimeError("Run prepare first")
    if read(root / "completion.json").get("complete"):
        return
    if not unified_scheduler:
        if not any(read(OUT / "allocation.json").get(k) for k in ("assigned_by_codex00", "scheduling_confirmed")):
            raise RuntimeError("Coordinated GPU handoff is not available; no GPU model was loaded")
        allocation=read(OUT / "allocation.json")
        if gpu not in allocation.get("gpus", []):
            raise RuntimeError("GPU not assigned to Codex05")
    config["scheduler"]="unified_gpu_queue" if unified_scheduler else "previous_handoff_queue"
    config.update(gpus=[gpu], state="running")
    write(root / "configuration.json",config)
    extra={r["uid"]:r for r in rows(read(root/"extra_inputs.json")["path"])}
    inputs=ChainMap(extra,load_inputs(SOURCE))
    from .engine import Engine
    engine=None
    def get_engine():
        nonlocal engine
        if engine is None:
            state(root,"loading_model")
            engine=Engine(key,metadata_path=root / "model_metadata.json")
        return engine
    target=root / "models/E1/full"
    if not read(target / "model.json").get("complete"):
        extract(root,"fit",inputs,get_engine())
        store=DynamicStore(root)
        state(root,"independent_M1_M2_M3")
        summary=training.fit_group("E1",rows(root/"data/E1_pairs.jsonl"),store,
            rows(root/"data/E1_views.jsonl"),rows(root/"data/natural_views.jsonl"),target,
            patcher=Patcher(root,store,inputs,get_engine()),delta=.2923349)
        # The reused core fits arbitrary head shapes; only its display histogram assumed Qwen.
        summary["M1"]["layer_distribution"]={str(k):v for k,v in sorted(
            Counter(h//engine.n_heads for h in summary["heads"]).items())}
        summary.update(complete=True,model_key=key,head_geometry=dict(layers=engine.n_layers,
            heads_per_layer=engine.n_heads,head_dim=engine.head_dim))
        write(target/"m1.json",summary["M1"])
        write(target/"model.json",summary)
    if not (target/"calibration.json").exists():
        calibration(root,DynamicStore(root))
    if not read(root/"frozen.json").get("frozen"):
        extract(root,"dev",inputs,get_engine())
        state(root,"source_development_evaluation")
        development=evaluate_model(root,scopes=("dev",))
        write(root/"frozen.json",dict(frozen=True,model_key=key,source_protocol="E1-balanced-2540",
            configuration=config,development=development,selection_split="source development only",
            selection="fixed E1 head/rank/regularization/budget; independently fitted checkpoint parameters",
            parameter_directory=str(target),time=time.strftime("%Y-%m-%d %H:%M:%S"),
            target_test_labels_used_for_fit=False,target_labels_used_for_calibration=False))
    scopes=["test"] + ([] if key.endswith("_base") else ["nli4ct","medpic","nli4ct_official5500","medpic_official467"])
    for scope in scopes:
        extract(root,scope,inputs,get_engine())
        state(root,"evaluate_"+scope)
        evaluate_model(root,scopes=(scope,))
    from .official_mega import run as run_official, summarize
    state(root,"mega_official")
    run_official(key,root,engine=get_engine())
    summarize(root)
    write(root/"completion.json",dict(complete=True,model_key=key,
        scopes=scopes+["mega_official"],time=time.strftime("%Y-%m-%d %H:%M:%S")))
    state(root,"complete",complete=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--gpu",required=True,type=int)
    a=p.parse_args()
    root=OUT/"models"/a.model
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/".run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        run_model(a.model,a.gpu)
    except Exception as exc:
        write(root/"run_status.json",dict(state="error",error=str(exc),traceback=traceback.format_exc()))
        raise
if __name__=="__main__":
    main()
