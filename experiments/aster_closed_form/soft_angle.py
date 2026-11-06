"""Frozen soft-angle comparison, using saved bases, prompts and reference scores."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from .common import rows, write_json, append, status, SEED
from .operators import soft_angle_from_saved
from . import pipeline as base
SOURCE = Path("/home/gcx/aster_existing_data_20261106")
ROOT = Path("/home/gcx/aster_soft_angle_20261106")

def prepare(root):
    torch.set_num_threads(4)
    hard = torch.load(SOURCE/"operators/aster.pt", map_location="cpu", weights_only=False)
    soft, checks = soft_angle_from_saved(hard, theta=45)
    assert len(soft) == 51
    assert all(c["algebra"]["signal_preservation_error"] < 1e-5 for c in checks.values())
    assert all(c["algebra"]["angle_bound_satisfied"] for c in checks.values())
    assert all(c["high_angle_vs_stored_hard_error"] < 1e-5 for c in checks.values())
    torch.save(soft, root/"operators/soft_angle.pt")
    write_json(root/"results/operator_checks.json", checks)
    config = json.loads((SOURCE/"configuration.json").read_text())
    config.update(variant="soft_angle", source_run=str(SOURCE),
        energy_targets=[.95], rank_caps=[4], angle_degrees=[45], amplification_candidates=[1],
        fixed=dict(eta=.95, kmax=4, theta=45, heads=51, amplification=1),
        selection="none; fixed before scoring", basis_source=str(SOURCE/"operators/aster.pt"),
        reuse="Saved U_N, S, mu and type_bases; no refitting",
        method_formula="D = U V diag(s / max(s^2, sin(45deg)^2)) Q^T",
        scale=dict(main_inputs=1290, main_pairs=1053, mechanism_dev_pairs=59,
                   mechanism_test_pairs=298, historical_inputs=569, path_cases=432, nested_path_cases=100))
    write_json(root/"configuration.json", config)
    write_json(root/"resource_plan.json", dict(gpus=[0,1], model=config["model"],
        model_copies=2, precision="bfloat16", expected_gib_per_gpu=[17,30],
        expected_minutes=[20,40], scale=config["scale"],
        monitoring="First 20 inputs per GPU, then completion only"))
    write_json(root/"operators/final.json", [dict(method="soft_angle", path=str(root/"operators/soft_angle.pt"),
        factor_columns=sum(x["N"].shape[1] for x in soft.values()))])
    write_json(root/"results/reuse.json", dict(source=str(SOURCE),
        reused=["All main input definitions and pair labels", "51 selected heads and per-type ranks",
                "Saved nuisance and signal bases and reference means", "Full/hard/calibrated scores",
                "Four existing joint repair components", "Historical templates and full/hard scores",
                "Full and hard autonomous paths"],
        recomputed=["Soft main scores for all 1290 inputs", "Soft joint repairs for every dev/test A/P pair",
                    "Soft historical scores for all 569 saved inputs", "Soft autonomous traversal for all 432 cases"],
        exclusions_added=0, parameter_search=False))
    return checks

def progress(root, shard, stage, done, total, start, engine):
    status(root, f"worker_{shard}", state="running", current_stage=stage, done=done, total=total,
        stage_seconds=time.time()-start, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, counts=engine.counts)

def verify_engine(root, shard, engine, operators, views):
    v=views[shard]
    context=engine.prepare(v["prompt"])
    natural=engine.run(context,capture=False)["ell"]
    zero={h:dict(N=torch.empty(engine.head_dim,0), B=torch.empty(0,engine.head_dim),
        mu=op["mu"], S=op["S"]) for h,op in operators.items()}
    identity=engine.run(context,operators=zero,capture=False)["ell"]
    reference=base.base_scores(SOURCE)["full"][v["view_id"]]
    assert natural==identity
    assert abs(natural-reference)<1e-5
    # Engine hooks enforce sequence length 1 on the replayed final position.
    first_layer=min(operators)//engine.n_heads
    captured={}
    handle=engine.layers[first_layer].self_attn.o_proj.register_forward_hook(
        lambda mod,args,out: captured.update(value=args[0].detach().float().cpu()))
    try:
        soft=engine.run(context,operators=operators,capture=True)
    finally:
        handle.remove()
    raw=soft["heads"][first_layer]
    actual=captured["value"].reshape(engine.n_heads,engine.head_dim)
    unchanged=[]; errors=[]
    for j in range(engine.n_heads):
        h=first_layer*engine.n_heads+j
        if h in operators:
            op=operators[h]
            expected=(raw[j]-op["N"]@(op["B"]@(raw[j]-op["mu"]))).to(engine.model.dtype).float()
            errors.append(float((actual[j]-expected).abs().max()))
        else:
            unchanged.append(torch.equal(actual[j],raw[j]))
    assert captured["value"].shape[1]==1 and all(unchanged) and max(errors,default=0)==0
    write_json(root/"results"/f"zero_intervention_{shard}.json", dict(view_id=v["view_id"],
        natural=natural,zero=identity,reused_reference=reference,zero_equal=True,
        difference_from_reused=natural-reference,target_position="final token before o_proj",
        selected_heads=sorted(operators), inspected_layer=first_layer,
        non_target_heads_unchanged=all(unchanged), factor_application_max_error=max(errors,default=0)))
    del context

def main_scores(root, shard, engine, operators):
    views=list(base.load_data(root)[0].values())
    output=root/"scores"/f"soft_angle_{shard}.jsonl"
    saved={r["view_id"] for r in rows(output) if r.get("status")=="complete"}
    verify_engine(root, shard, engine, operators, views)
    todo=views[shard::2]; start=time.time(); count=0
    for i,v in enumerate(todo):
        if v["view_id"] not in saved:
            context=engine.prepare(v["prompt"])
            result=engine.run(context,operators=operators,capture=False)
            append(output,dict(view_id=v["view_id"],method="soft_angle",status="complete",**result))
            del context
            count+=1
        progress(root,shard,"main",i+1,len(todo),start,engine)
        if count==20:
            write_json(root/"timing"/f"first20_{shard}.json",dict(inputs=20,seconds=time.time()-start,
                seconds_per_input=(time.time()-start)/20,main_stage_projected_minutes=len(todo)*(time.time()-start)/20/60,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
            count+=1
    return dict(inputs=len(todo),seconds=time.time()-start)

def historical_scores(root,shard,engine,operators):
    from . import engine as module
    from .existing import historical_template
    system, make_prompt=historical_template()
    original_system=module.SYSTEM
    inputs=(rows(root/"diagnostics/fact_preference_inputs.jsonl")+
            rows(root/"diagnostics/deletion_restore_inputs.jsonl"))[shard::2]
    output=root/"diagnostics"/f"soft_angle_{shard}.jsonl"
    saved={r["score_id"] for r in rows(output) if r.get("status")=="complete"}
    start=time.time()
    try:
        module.SYSTEM=system
        for i,r in enumerate(inputs):
            if r["score_id"] not in saved:
                context=engine.prepare(make_prompt(r))
                result=engine.run(context,operators=operators,capture=False)
                append(output,dict(status="complete",method="soft_angle",**result,
                    **{k:r[k] for k in ("score_id","example_id","node_id","source_id","view","gold")}))
                del context
            progress(root,shard,"historical",i+1,len(inputs),start,engine)
    finally:
        module.SYSTEM=original_system
    return dict(inputs=len(inputs),seconds=time.time()-start)

def joint_repairs(root,shard,engine,operators):
    from .m1 import _heads
    views,pairs=base.load_data(root)
    pairs=[p for p in pairs if p["split"] in ("dev","test") and p["kind"] in ("A","P")][shard::2]
    output=root/"mechanisms"/f"soft_angle_{shard}.jsonl"
    saved={r["pair_id"] for r in rows(output) if r.get("status")=="complete"}
    start=time.time()
    for i,p in enumerate(pairs):
        if p["pair_id"] not in saved:
            a,b=[torch.load(base.feature_file(root,p[side]),map_location="cpu",weights_only=False)
                 for side in ("left","right")]
            z0,z1=_heads(a),_heads(b)
            forward,reverse={},{}
            for h,op in operators.items():
                piece=op["N"]@(op["B"]@(z1[h]-z0[h]))
                forward[h]=z0[h]+piece
                reverse[h]=z1[h]-piece
            context=engine.prepare(views[p["left"]]["prompt"])
            fe=engine.run(context,replacements=forward,capture=False)["ell"]
            del context
            context=engine.prepare(views[p["right"]]["prompt"])
            re=engine.run(context,replacements=reverse,capture=False)["ell"]
            del context
            te=float(b["ell"])-float(a["ell"])
            df,dr=fe-float(a["ell"]),re-float(b["ell"])
            append(output,dict(**{k:p[k] for k in ("pair_id","kind","subtype","group_id","source_id","split")},
                component="soft_angle",head_count=len(operators),status="complete",
                normalized_effect_eligible=abs(te)>=.1,forward_ell=fe,reverse_ell=re,
                ie_forward=df,ie_reverse=dr,te=te,
                forward_normalized=df/te if abs(te)>=.1 else None,
                reverse_normalized=dr/-te if abs(te)>=.1 else None,
                recovery=.5*(df/te+dr/-te) if abs(te)>=.1 else None))
        progress(root,shard,"mechanisms",i+1,len(pairs),start,engine)
    return dict(pairs=len(pairs),seconds=time.time()-start)

def worker(root,shard):
    from .engine import Engine
    from .paths import run_paths
    torch.set_num_threads(4);torch.manual_seed(SEED)
    status(root,f"worker_{shard}",state="loading",physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    start=time.time()
    engine=Engine()
    ops=torch.load(root/"operators/soft_angle.pt",map_location="cpu",weights_only=False)
    timing={}
    timing["main"]=main_scores(root,shard,engine,ops)
    timing["historical"]=historical_scores(root,shard,engine,ops)
    timing["mechanisms"]=joint_repairs(root,shard,engine,ops)
    before=time.time()
    progress(root,shard,"paths",0,216,before,engine)
    records=run_paths(engine,rows(root/"data/path_cases.jsonl"),ops,None,None,root/"paths",
        shard=shard,shards=2,methods=("soft_angle",),operator_method="soft_angle")
    assert len(records)==216 and all(r["status"]=="complete" for r in records)
    timing["paths"]=dict(cases=len(records),seconds=time.time()-before)
    write_json(root/"timing"/f"worker_{shard}.json",dict(stages=timing,total_seconds=time.time()-start,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,counts=engine.counts))
    status(root,f"worker_{shard}",state="complete",elapsed_seconds=time.time()-start)

def launch(root):
    start=time.time();procs=[]
    for gpu in (0,1):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
        with (root/"logs"/f"worker_{gpu}.log").open("a") as log:
            procs.append(subprocess.Popen([sys.executable,"-m","experiments.aster_closed_form.soft_angle",
                "--root",str(root),"--stage","worker","--shard",str(gpu)],
                cwd=root/"code",env=env,stdout=log,stderr=log))
    status(root,"all",state="running",worker_pids=[p.pid for p in procs],gpus=[0,1])
    codes=[p.wait() for p in procs]
    write_json(root/"runtime.json",dict(seconds=time.time()-start,gpus=[0,1],exit_codes=codes))
    if any(codes):raise RuntimeError("A worker failed; see server logs")
    status(root,"all",state="scoring_complete",elapsed_seconds=time.time()-start)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("prepare","all","worker"),default="all")
    parser.add_argument("--shard",type=int,default=0)
    args=parser.parse_args()
    try:
        if args.stage=="prepare":prepare(args.root)
        elif args.stage=="all":launch(args.root)
        else:worker(args.root,args.shard)
    except Exception as exc:
        status(args.root,f"worker_{args.shard}" if args.stage=="worker" else "all",
            state="failed",error=str(exc))
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
