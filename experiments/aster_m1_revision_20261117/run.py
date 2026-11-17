"""Finite queue worker. No standalone GPU launches or parameter search."""
import argparse
import fcntl
import gzip
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, lines, load_inputs
from experiments.aster_e678_adaptive_20261116.run import require_queue
from .prepare import OUT, PRIVATE, REPO, SOURCE, KEYS, datasets
from .revision import Config, ExactPatcher, screen, partition_sources, confirm, paired_effect, fit_selected
from .runtime import checkpoint_identity, extract, digest, digest_file

class ResourceBlocked(RuntimeError):
    pass

def publish(key, stage):
    """Serialize commits from concurrent model workers; publish anonymous outputs only."""
    log=PRIVATE/"publish.log";log.parent.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/"publish.lock").open("a") as lock, log.open("a") as handle:
        fcntl.flock(lock,fcntl.LOCK_EX)
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
        if branch!="fix/aster-m1-estimand-20261117":raise RuntimeError("Unexpected PR branch")
        target="reports/aster_m1_revision_20261117/models/"+key
        subprocess.run(["git","add","-f","--",target],cwd=REPO,stdout=handle,stderr=handle,check=True)
        changed=subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO).returncode
        if changed:
            subprocess.run(["git","commit","-m","results(aster): "+key+" "+stage],cwd=REPO,stdout=handle,stderr=handle,check=True)
        subprocess.run(["git","push","origin","HEAD:fix/aster-m1-estimand-20261117"],
                       cwd=REPO,stdout=handle,stderr=handle,check=True)

def freeze_parameters(root):
    targets=sorted((root/"models").glob("*/full/m[23].npz"))
    targets += sorted((root/"models").glob("*/full/model.json"))
    targets += [root/"independent_intercept.json"]
    return {str(p.relative_to(root)):digest_file(p) for p in targets}

def write_training_constraints(root, pairs, store, target):
    from experiments.aster_cr_retrain_paired_v2_20261111.training import coordinates
    support=dict(np.load(target/"m2.npz",allow_pickle=False))
    m3=dict(np.load(target/"m3.npz",allow_pickle=False))
    cache={}
    def correction(uid):
        if uid not in cache:cache[uid]=float(coordinates(uid,store,support)@m3["w"])
        return cache[uid]
    records=[]
    for p in pairs:
        raw=store.ell(p["right"])-store.ell(p["left"])
        diff=correction(p["right"])-correction(p["left"])
        excess=max(abs(diff)-float(m3["delta"]),0.) if p["kind"] in ("C","F") else None
        records.append(dict(pair_id=p["pair_id"],source_id=p["source_id"],kind=p["kind"],
                            raw_difference=raw,correction_difference=diff,adjusted_difference=raw+diff,
                            budget_excess=excess))
    lines(target/"training_pair_constraints.jsonl",records)

def run_model(key,publish_results=False,resume=False):
    require_queue()
    root=OUT/"models"/key
    runtime=PRIVATE/key
    runtime.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();engine=None
    def progress(stage,**kwargs):
        state=dict(state="running",stage=stage,pid=os.getpid(),gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                   updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),**kwargs)
        write(root/"execution.json",state)
        with (root/"events.jsonl").open("a") as f:f.write(json.dumps(state)+"\n")
    with (runtime/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(root/"completion.json").get("complete"):return
        if (root/"execution.json").exists() and not resume:
            raise RuntimeError("Prior run exists; inspect artifacts before explicit --resume")
        if key!="qwen":
            qwen=read(OUT/"models/qwen/completion.json")
            if not qwen.get("engineering_accepted"):raise RuntimeError("Qwen end-to-end acceptance required")
        try:
            import torch
            torch.set_num_threads(4)
            progress("loading_model")
            from experiments.aster_e678.engine import Engine
            engine=Engine(key,metadata_path=root/"model_metadata.json")
            progress("checkpoint_identity")
            identity=checkpoint_identity(key,engine)
            if (root/"failure.json").exists():
                history=root/"failure_history"
                history.mkdir(exist_ok=True)
                (root/"failure.json").rename(history/(str(time.time_ns())+".json"))
            data=datasets(); pairs=data["E1_pairs.jsonl"]
            inputs=load_inputs(SOURCE)
            cfg=Config()
            progress("extract_fit")
            store,input_hash=extract(key,"fit",inputs,engine,progress)
            identity["inputs_sha256"]=input_hash
            if publish_results:publish(key,"fit extraction and gradient coverage")
            screening,confirmation=partition_sources(pairs,cfg)
            ps=[p for p in pairs if p["source_id"] in set(confirmation)]
            progress("candidate_selection")
            candidates=screen(pairs,store,screening,cfg)
            count=sum(2+4*len(candidates[p["kind"]]["heads"]) for p in ps if candidates[p["kind"]]["heads"])
            manifest=dict(identity=identity,config=asdict(cfg),screening_sources=screening,
                          confirmation_sources=confirmation,candidates=candidates,
                          estimated_confirmation_forwards=count)
            manifest["protocol_sha256"]=digest(manifest)
            write(root/"selection_plan.json",manifest)
            counts={k:len(v["heads"]) for k,v in candidates.items()}
            write(root/"resource_preflight.json",dict(passed=count<=cfg.max_patch_forwards,
                  estimated_confirmation_forwards=count,limit=cfg.max_patch_forwards,
                  candidate_counts=counts,confirmation_pairs=len(ps),
                  formula="sum(2 + 4 * candidate_heads[kind]) for each nonempty confirmation pair",
                  no_candidate_truncation=True))
            if count>cfg.max_patch_forwards:
                raise ResourceBlocked("Confirmation requires "+str(count)+" forwards; ceiling "+str(cfg.max_patch_forwards))
            progress("exact_patching",estimated_forwards=count)
            target=root/"models/E1/full"
            target.mkdir(parents=True,exist_ok=True)
            patcher=ExactPatcher(store,inputs,engine,cfg)
            records=[]
            for i,p in enumerate(ps):
                path=runtime/"paired"/(p["pair_id"]+".json")
                if path.exists():
                    batch=read(path)
                    # Every reused row is checked against the current frozen plan and numerics.
                    if {int(r["head"]) for r in batch}!=set(candidates[p["kind"]]["heads"]):
                        raise ValueError("Incomplete saved pair cache")
                    for row in batch:paired_effect(row,cfg)
                else:
                    def save_raw(row):
                        with gzip.open(root/"raw_patch_attempts.jsonl.gz","at",encoding="utf-8") as f:
                            f.write(json.dumps(row,allow_nan=False)+"\n")
                    batch=patcher.measure(p,candidates[p["kind"]]["heads"],on_record=save_raw)
                    write(path,batch)
                records.extend(batch)
                if i<3 or (i+1)%10==0:progress("exact_patching",completed_pairs=i+1,total_pairs=len(ps))
            with gzip.open(target/"paired_effects.jsonl.gz","wt",encoding="utf-8") as f:
                for row in records:f.write(json.dumps(row,allow_nan=False)+"\n")
            result=confirm(ps,candidates,records,engine.num_heads,cfg)
            result["heads_per_layer"]=engine.n_heads
            result["selected_local_heads"]=[dict(head=h,layer=h//engine.n_heads,local_head=h%engine.n_heads) for h in result["heads"]]
            write(target/"m1.json",result)
            if publish_results:publish(key,"M1 paired confirmation")
            progress("M2_M3")
            if not read(target/"model.json").get("complete"):
                fit_selected(pairs,store,data["E1_views.jsonl"],data["natural_views.jsonl"],
                             target,result,manifest["protocol_sha256"])
                write_training_constraints(root,pairs,store,target)
            sensitivity={}
            for mass in (.80,.95):
                selection=confirm(ps,candidates,records,engine.num_heads,replace(cfg,retained_mass=mass))
                dest=root/"models"/("sensitivity_"+str(round(mass*100)))/"full"
                dest.mkdir(parents=True,exist_ok=True)
                write(dest/"m1.json",selection)
                selection["retained_mass"]=mass
                model=read(dest/"model.json")
                if not model.get("complete"):
                    model=fit_selected(pairs,store,data["E1_views.jsonl"],data["natural_views.jsonl"],
                                       dest,selection,digest(dict(parent=manifest["protocol_sha256"],retained_mass=mass)))
                    write_training_constraints(root,pairs,store,dest)
                sensitivity[str(mass)]=dict(heads=selection["heads"],status=model["status"])
            write(root/"sensitivity.json",dict(retained_mass=sensitivity,reused_identical_patch_cache=True,
                                              main_mass=.90,test_based_selection=False))
            from .evaluate import independent_intercept,evaluate
            if not (root/"independent_intercept.json").exists():
                write(root/"independent_intercept.json",independent_intercept(data["natural_views.jsonl"],store))
            parameters=freeze_parameters(root)
            existing_freeze=read(root/"fit_frozen.json")
            if existing_freeze and existing_freeze["parameters"]!=parameters:
                raise ValueError("Previously frozen parameters changed")
            if not existing_freeze:
                write(root/"fit_frozen.json",dict(frozen=True,parameters=parameters,selection_plan=manifest["protocol_sha256"]))
            if publish_results:publish(key,"M2 M3 and fit-only calibration")
            progress("extract_dev")
            dev_store,_=extract(key,"dev",inputs,engine,progress)
            if not read(root/"evaluation/dev/metrics.json").get("complete"):
                evaluate(root,"dev",dev_store)
            if freeze_parameters(root)!=parameters:raise ValueError("Parameters changed during dev evaluation")
            write(root/"frozen.json",dict(frozen=True,parameters=parameters,model_keys=list(KEYS),
                  main_mass=.90,sensitivity=[.80,.95],parameters_selected_from_test=False,
                  dev_role="model-selection results",test_role="historically reused 40-source holdout",
                  expansion_gate="engineering acceptance, independent of measured performance"))
            if publish_results:publish(key,"development evaluation and frozen manifest")
            progress("extract_test")
            test_store,_=extract(key,"test",inputs,engine,progress)
            if not read(root/"evaluation/test/metrics.json").get("complete"):
                evaluate(root,"test",test_store)
            if freeze_parameters(root)!=parameters:raise ValueError("Parameters changed during test evaluation")
            model=read(target/"model.json")
            write(root/"completion.json",dict(complete=True,status="intercept_only_fallback" if model["status"]=="intercept_only_fallback" else "complete",
                  engineering_accepted=True,performance_evaluated=True,seconds=time.monotonic()-started,
                  peak_gib=torch.cuda.max_memory_allocated()/2**30,engine_counts=engine.counts))
            progress("complete")
            write(root/"queue_done.json",dict(complete=True))
            if publish_results:publish(key,"historical holdout evaluation complete")
        except Exception as exc:
            state="resource_blocked" if isinstance(exc,ResourceBlocked) else "engineering_failed"
            if state!="resource_blocked" and read(root/"fit_frozen.json").get("frozen"):
                state="evaluation_incomplete"
            failure=dict(complete=False,status=state,error=str(exc),seconds=time.monotonic()-started,
                         traceback=traceback.format_exc(),engineering_accepted=False)
            if engine is not None:
                failure["engine_counts"]=engine.counts
                failure["peak_gib"]=torch.cuda.max_memory_allocated()/2**30
            write(root/"failure.json",failure)
            write(root/"execution.json",failure)
            if publish_results:publish(key,state)
            raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=KEYS,required=True)
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    run_model(args.model,args.publish,args.resume)
