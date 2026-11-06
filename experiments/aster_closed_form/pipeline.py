"""Resumable server-only execution of the registered automatic pilot."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .common import *
def load_data(root):
    vs=rows(root/"data/views.jsonl");return {v["view_id"]:v for v in vs},rows(root/"data/pairs.jsonl")
def feature_file(root,vid):
    return root/"features"/(vid.replace("/","_")+".pt")
def load_features(root):
    import torch
    return {v["view_id"]:torch.load(feature_file(root,v["view_id"]),map_location="cpu",weights_only=False)
            for v in rows(root/"data/views.jsonl") if feature_file(root,v["view_id"]).exists()}
def base_scores(root):
    scores={}
    for path in sorted((root/"scores").glob("baseline_*.jsonl")):
        for r in rows(path):scores.setdefault(r["method"],{})[r["view_id"]]=r.get("ell")
    return scores
def all_scores(root):
    scores=base_scores(root)
    for path in sorted((root/"scores").glob("final_*.jsonl")):
        for r in rows(path):scores.setdefault(r["method"],{})[r["view_id"]]=r.get("ell")
    fpath=root/"results/fusion.json"
    if fpath.exists():scores.update(json.loads(fpath.read_text())["scores"])
    return scores
def collect(root,shard=0,shards=2):
    import torch
    from .engine import Engine
    views,pairs=load_data(root);todo=list(views.values())[shard::shards]
    grad_ids={p["left"] for p in pairs if p["split"]=="fit" and p["kind"] in ("A","P")}
    output=root/"scores"/f"baseline_{shard}.jsonl"
    saved={(r["view_id"],r["method"]):r for r in rows(output)}
    engine=Engine();cache={};start=time.time()
    for ix,v in enumerate(todo):
        f=feature_file(root,v["view_id"]);f.parent.mkdir(parents=True,exist_ok=True)
        methods=["full","ignore","direct","semantic","boundary_guideline","boundary_facts"]
        if v["cohort"]=="definition":methods.append("definition_restore")
        if not f.exists():
            ctx=engine.prepare(v["prompt"])
            result=engine.run(ctx,grad=v["view_id"] in grad_ids)
            result.update({k:v[k] for k in ("view_id","split","source_id","group_id","cohort","original","label")})
            torch.save(result,f);del ctx
        else:result=torch.load(f,map_location="cpu",weights_only=False)
        cache[v["prompt"]]=result["ell"]
        for method in methods:
            key=(v["view_id"],method)
            if key in saved:continue
            unavailable=(method=="semantic" and not v.get("semantic_available",False) or
                         method=="boundary_guideline" and not v.get("boundary_annotation_available",False) or
                         method=="boundary_facts" and not v.get("decisive_annotation_available",False))
            if unavailable:
                row=dict(view_id=v["view_id"],method=method,ell=None,status="failed",error="Required automatic extraction/annotation unavailable")
                append(output,row);saved[key]=row
                continue
            text=v["prompts"][method];before=time.time()
            try:
                if text not in cache:
                    ctx=engine.prepare(text);cache[text]=engine.run(ctx,capture=False)["ell"];del ctx
                row=dict(view_id=v["view_id"],method=method,ell=cache[text],status="complete",seconds=time.time()-before)
            except Exception as exc:
                row=dict(view_id=v["view_id"],method=method,ell=None,status="failed",error=str(exc))
                if "out of memory" in str(exc).lower():torch.cuda.empty_cache()
            append(output,row);saved[key]=row
        status(root,f"collect_{shard}",state="running",done=ix+1,total=len(todo),elapsed_seconds=time.time()-start,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,counts=engine.counts)
        if ix==19:
            write_json(root/f"qwen_pilot20_{shard}.json",dict(nodes=20,seconds=time.time()-start,
                projection_hours=len(todo)*(time.time()-start)/20/3600,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    status(root,f"collect_{shard}",state="complete",done=len(todo),elapsed_seconds=time.time()-start,counts=engine.counts)
def m1_worker(root,kind):
    from .engine import Engine
    from .m1 import run_m1
    views,pairs=load_data(root);features=load_features(root)
    result=run_m1(kind,Engine(),views,pairs,features,root/"m1")
    status(root,"m1_"+kind,state="complete",selected_heads=len(result["H"]),target_met=result["target_met"])
def fit_grid(root):
    import torch
    from .operators import configuration_grid,fit_operators,total_edit_rank
    views,pairs=load_data(root);features=load_features(root)
    a=json.loads((root/"m1/m1_A.json").read_text());p=json.loads((root/"m1/m1_P.json").read_text())
    heads=sorted(set(a["H"])|set(p["H"]))
    types={h:[t for t,r in (("A",a),("P",p)) if h in r["H"]] for h in heads}
    selection=dict(H_A=a["H"],H_P=p["H"],H=heads,T_h=types,
                   epsilon=.1,rho=.8,candidates_per_type=64,fit_sources="fit",selection_sources="dev")
    write_json(root/"m1/selection.json",selection)
    manifest=[]
    for i,config in enumerate(configuration_grid()):
        ops,diag=fit_operators(features,pairs,heads,types,**config)
        method=f"grid_{i:02d}";path=root/"operators"/(method+".pt");path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(ops,path);write_json(root/"operators"/(method+".json"),diag)
        manifest.append(dict(method=method,path=str(path),config=config,heads=len(heads),rank=total_edit_rank(ops)))
    write_json(root/"operators/grid.json",manifest)
def score_operators(root,stage,shard=0,shards=2):
    import torch
    from .engine import Engine
    views,_=load_data(root)
    config_path=root/"operators"/("grid.json" if stage=="grid" else "final.json")
    configs=json.loads(config_path.read_text())
    if stage=="amp_dev":
        configs=[c for c in configs if c["method"]=="aster" or c["method"].startswith("amplify_")]
    elif stage=="final":
        amp=json.loads((root/"results/amplification_selection.json").read_text())["selected"]
        configs=[c for c in configs if not c["method"].startswith("amplify_") or c["method"]==amp]
    candidates=[v for v in views.values() if stage not in ("grid","amp_dev") or v["split"]=="dev"][shard::shards]
    ops={c["method"]:torch.load(c["path"],map_location="cpu",weights_only=False) for c in configs}
    score_prefix="final_amp_dev" if stage=="amp_dev" else stage
    output=root/"scores"/f"{score_prefix}_{shard}.jsonl";saved={(r["view_id"],r["method"]) for r in rows(output)}
    engine=Engine();start=time.time()
    for ix,v in enumerate(candidates):
        needed=[c for c in configs if (v["view_id"],c["method"]) not in saved]
        if needed:
            try:
                ctx=engine.prepare(v["prompt"])
            except Exception as exc:
                for c in needed:
                    append(output,dict(view_id=v["view_id"],method=c["method"],ell=None,status="failed",error=str(exc)))
                continue
            if stage=="grid" and ix==0:
                current=engine.run(ctx,capture=False)["ell"]
                first=next(iter(ops.values()))
                identity={h:dict(N=torch.empty(engine.head_dim,0),B=torch.empty(0,engine.head_dim),
                                mu=torch.zeros(engine.head_dim),S=torch.empty(engine.head_dim,0)) for h in first}
                zero=engine.run(ctx,operators=identity,capture=False)["ell"]
                assert current==zero, "Zero intervention changed the score"
                reference=base_scores(root)["full"].get(v["view_id"])
                write_json(root/"results"/f"zero_intervention_{shard}.json",
                    dict(view_id=v["view_id"],natural=current,zero=zero,reused_reference=reference,
                         difference_from_reused=current-reference,zero_equal=True,
                         position="final token before o_proj",heads=sorted(first)))
            for c in needed:
                try:r=engine.run(ctx,operators=ops[c["method"]],capture=False);row=dict(ell=r["ell"],status="complete")
                except Exception as exc:
                    row=dict(ell=None,status="failed",error=str(exc))
                    if "out of memory" in str(exc).lower():torch.cuda.empty_cache()
                row.update(view_id=v["view_id"],method=c["method"])
                append(output,row)
            del ctx
        status(root,f"{stage}_{shard}",state="running",done=ix+1,total=len(candidates),elapsed_seconds=time.time()-start,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,counts=engine.counts)
        if ix==min(19,len(candidates)-1):
            write_json(root/"timing"/f"{stage}_{shard}_first20.json",
                dict(inputs=ix+1,configurations=len(configs),seconds=time.time()-start,
                     remaining_stage_hours=(len(candidates)-ix-1)*(time.time()-start)/(ix+1)/3600,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,counts=engine.counts))
    status(root,f"{stage}_{shard}",state="complete",done=len(candidates),elapsed_seconds=time.time()-start,counts=engine.counts)
def select_and_ablations(root):
    import torch
    from .evaluation import evaluate,selection_key,fit_fusions
    from .operators import fit_operators,total_edit_rank,layer_matched_random
    views,pairs=load_data(root);features=load_features(root)
    scores=base_scores(root);grid_scores={}
    for path in sorted((root/"scores").glob("grid_*.jsonl")):
        for r in rows(path):grid_scores.setdefault(r["method"],{})[r["view_id"]]=r.get("ell")
    candidates=json.loads((root/"operators/grid.json").read_text())
    metrics=evaluate(views,pairs,grid_scores,split="dev",cohort=None,bootstrap=0)["methods"]
    def key(c):return (*selection_key(metrics[c["method"]]),-c["heads"],-c["rank"],c["config"]["theta"])
    selected=max(candidates,key=key)
    m1=json.loads((root/"m1/selection.json").read_text())
    choice=dict(selected,selection_rule="PA macro across all development pairs, original accuracy, fewer heads, lower edit rank, larger angle",
                selection_cohort="main + definition; source-disjoint from test",
                development_metrics=metrics,candidates=candidates,**m1)
    write_json(root/"results/selection.json",choice)
    from .engine import SYSTEM
    write_json(root/"results/deployment.json",dict(model=MODEL,labels=["\u662f","\u5426"],tie_label="\u5426",
        thinking=False,system=SYSTEM,operator_file=str(root/"operators/aster.pt"),
        H=m1["H"],T_h=m1["T_h"],configuration=selected["config"],
        position="last input token, query outputs before o_proj",online="one direct label forward per node",
        context_cache="unedited prefix only"))
    frozen=copy.deepcopy(torch.load(selected["path"],map_location="cpu",weights_only=False))
    final=[]
    def save(name,ops,diag=None):
        path=root/"operators"/(name+".pt");torch.save(ops,path)
        final.append(dict(method=name,path=str(path),rank=total_edit_rank(ops)))
        if diag is not None:write_json(root/"operators"/(name+".json"),diag)
    save("aster",frozen)
    for anchor in ("original","zero"):
        ops,diag=fit_operators(features,pairs,m1["H"],m1["T_h"],anchor=anchor,**selected["config"])
        save("anchor_"+anchor,ops,diag)
    for var in ("per_type_mean","no_C","no_F","orthogonal","leace"):
        ops,diag=fit_operators(features,pairs,m1["H"],m1["T_h"],variant=var,**selected["config"])
        save(var,ops,diag)
    for seed in (13,37,73):
        h,t=layer_matched_random(m1["H"],m1["T_h"],seed)
        ops,diag=fit_operators(features,pairs,h,t,**selected["config"]);save("random_"+str(seed),ops,diag)
    for amp in (1.25,1.5,2.0):
        amplified=copy.deepcopy(frozen)
        for op in amplified.values():op["amplification"]=amp
        save("amplify_"+str(amp),amplified)
    write_json(root/"operators/final.json",final)
    fusions=fit_fusions(views,pairs,scores,selection_cohort=None)
    write_json(root/"results/fusion.json",fusions)
def select_amplification(root):
    if (root/"results/amplification_selection.json").exists():return
    from .evaluation import evaluate,selection_key
    views,pairs=load_data(root);scores=all_scores(root)
    amp_names=["aster","amplify_1.25","amplify_1.5","amplify_2.0"]
    available={m:scores[m] for m in amp_names if m in scores}
    metrics=evaluate(views,pairs,available,split="dev",cohort=None,bootstrap=0)["methods"]
    selected=max(available,key=lambda n:(selection_key(metrics[n]),-amp_names.index(n)))
    write_json(root/"results/amplification_selection.json",dict(selected=selected,development=metrics,
                  selected_on="dev",amplification=1 if selected=="aster" else float(selected.split("_")[1])))
def mechanisms(root,split):
    import torch
    from .engine import Engine
    from .mechanisms import run_mechanisms
    views,pairs=load_data(root);features=load_features(root)
    ops=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    selection=json.loads((root/"results/selection.json").read_text())
    run_mechanisms(Engine(),views,pairs,features,ops,selection,root/"mechanisms",split=split)
    status(root,"mechanisms_"+split,state="complete")
def path_worker(root,shard=0,shards=2):
    import torch
    from .engine import Engine
    from .paths import run_paths
    f=json.loads((root/"results/fusion.json").read_text())
    spec=f["specs"][f["best_method"]]
    masks={r["key"]:r for r in rows(root/"data/path_semantic_masks.jsonl")}
    ops=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    run_paths(Engine(),rows(root/"data/path_cases.jsonl"),ops,spec,masks,root/"paths",shard=shard,shards=shards)
    status(root,"paths_"+str(shard),state="complete")
def report(root):
    import torch
    from .evaluation import evaluate,signal_probe,path_metrics
    from .report import render_report,coverage_status
    views,pairs=load_data(root);scores=all_scores(root);features=load_features(root)
    select_amplification(root)
    amp=json.loads((root/"results/amplification_selection.json").read_text())
    # Unselected signal amplification candidates are development-only in main tables.
    table_scores={m:s for m,s in scores.items() if not m.startswith("amplify_")}
    table_scores["erase_plus_selected_amplification"]=scores[amp["selected"]]
    evaluations={name:evaluate(views,pairs,{m:s for m,s in table_scores.items() if cohort!="main" or m!="definition_restore"},split=split,cohort=cohort,bootstrap=2000 if split=="test" else 0)
                 for name,split,cohort in (("main_test","test","main"),("definition_test","test","definition"),("main_dev","dev","main"))}
    write_json(root/"results/evaluations.json",evaluations)
    ops=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    h3=dict(probe_main=signal_probe(views,features,ops,scores["full"]),amplification=amp,
            erase=evaluations["main_test"]["methods"].get("aster"),
            erase_plus_signal=evaluations["main_test"]["methods"].get("erase_plus_selected_amplification"))
    write_json(root/"results/h3.json",h3)
    selection=json.loads((root/"results/selection.json").read_text())
    h2=json.loads((root/"operators"/(selection["method"]+".json")).read_text())
    mechanism_summaries={sp:json.loads((root/"mechanisms"/f"mechanisms_{sp}.json").read_text()) for sp in ("dev","test")}
    h2["interventions"]={sp:{"nodes":v["h2_node_count"],"pairs":v["h2_pair_count"],"effects":v["h2"]} for sp,v in mechanism_summaries.items()}
    write_json(root/"results/h2.json",h2)
    write_json(root/"results/local_theory.json",{sp:v["local_theory"] for sp,v in mechanism_summaries.items()})
    write_json(root/"results/operator_checks.json",{h:v["algebra"] for h,v in h2["heads"].items()})
    # Mechanism module saves split-specific products; retain direct links and complete content.
    h1={t:json.loads((root/"m1"/f"m1_{t}.json").read_text()) for t in ("A","P")}
    h1["interventions"]={sp:v["h1"] for sp,v in mechanism_summaries.items()}
    h1["mechanism_outputs"]=[str(p.relative_to(root)) for p in (root/"mechanisms").rglob("*.json")]
    write_json(root/"results/h1.json",h1)
    pathrows=[]
    for p in (root/"paths").glob("*.jsonl"):pathrows+=rows(p)
    if pathrows:write_json(root/"results/path_results.json",path_metrics(pathrows))
    manifest=json.loads((root/"data/manifest.json").read_text())
    manifest["valid_counts"]=json.loads((root/"data/valid_counts.json").read_text())
    write_json(root/"results/data_manifest.json",manifest)
    completion=coverage_status(root,evaluations,pipeline_state="complete")
    write_json(root/"results/status.json",dict(state="complete",status=completion["experiment_state"],automatic_pilot=True,clinical_verified=False,**completion))
    render_report(root,evaluations,metadata=manifest)
    status(root,"all",state="complete",experiment_state=completion["experiment_state"])
    launch_path=root/"launch.json"
    if launch_path.exists():
        launch=json.loads(launch_path.read_text())
        launch.update(state="complete",experiment_state=completion["experiment_state"])
        write_json(launch_path,launch)
def mask_paths(root,shard=0,shards=1):
    from .construct import Generator,MASK_INSTRUCTION
    from .data import canonical_guide,condition
    from .paths import enumerate_contexts
    if shards < 1 or not 0 <= shard < shards:raise ValueError("Invalid path-mask shard")
    root=Path(root)
    entries=enumerate_contexts(rows(root/"data/path_cases.jsonl"))[shard::shards]
    output=root/"data"/(f"path_semantic_masks_{shard}.jsonl" if shards>1 else "path_semantic_masks.jsonl")
    stage=f"path_semantic_extraction_{shard}" if shards>1 else "path_semantic_extraction"
    saved={r["key"] for r in rows(root/"data/path_semantic_masks.jsonl")+rows(output)}
    todo=[e for e in entries if e["key"] not in saved]
    status(root,stage,state="loading",done=0,total=len(todo),assigned=len(entries),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not todo:
        status(root,stage,state="complete",done=0,total=0,assigned=len(entries),elapsed_seconds=0)
        return
    gen=Generator();started=time.time()
    for i,e in enumerate(todo):
        # enumerate_contexts intentionally excludes patients, gold paths and labels.
        payload=dict(current_node=e["gid"],current_condition=e["condition"],
            tree=canonical_guide(e["guideline"]),
            definitions=[dict(index=j,text=d["quote"]) for j,d in enumerate(e["definitions"])])
        try:
            v,raw,ni,no=gen.generate(MASK_INSTRUCTION,payload,max_new=1200)
            record=dict(key=e["key"],value=v,status="complete",input_tokens=ni,output_tokens=no)
        except Exception as exc:
            tree=payload.get("tree",{})
            record=dict(key=e["key"],status="failed",error=str(exc),
                        value=dict(keep_action_ids=list(tree.get("leaves",{})),
                                   keep_definition_indices=list(range(len(payload.get("definitions",[]))))))
        append(output,record)
        status(root,stage,state="running",done=i+1,total=len(todo),assigned=len(entries),elapsed_seconds=time.time()-started,
               physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),peak_reserved_gib=gen.torch.cuda.max_memory_reserved()/2**30)
    status(root,stage,state="complete",done=len(todo),total=len(todo),assigned=len(entries),elapsed_seconds=time.time()-started)
def merge_path_masks(root,shards=2):
    root=Path(root);merged={}
    for path in [root/"data/path_semantic_masks.jsonl"]+[root/"data"/f"path_semantic_masks_{s}.jsonl" for s in range(shards)]:
        for r in rows(path):
            previous=merged.get(r["key"])
            if previous is None or previous.get("status")!="complete":merged[r["key"]]=r
    write_rows(root/"data/path_semantic_masks.jsonl",merged.values())
    timings=[json.loads(p.read_text()).get("elapsed_seconds",0) for p in
             [root/"status"/f"path_semantic_extraction_{s}.json" for s in range(shards)] if p.exists()]
    status(root,"path_semantic_extraction",state="complete",done=len(merged),shards=shards,
           elapsed_seconds=max(timings,default=0),worker_seconds=sum(timings))

def spawn(root,stage,gpu,**kwargs):
    cmd=[sys.executable,"-m","experiments.aster_closed_form","--stage",stage,"--root",str(root)]
    for k,v in kwargs.items():cmd+=["--"+k.replace("_","-"),str(v)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
    (root/"logs").mkdir(exist_ok=True,parents=True)
    log=(root/"logs"/(stage+"_"+str(gpu)+".log")).open("a")
    return subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=log)
def join(processes):
    failures=[]
    for p in processes:
        code=p.wait()
        if code:failures.append(dict(pid=p.pid,returncode=code))
    if failures:raise RuntimeError("Stage failed: "+str(failures))
def all_stages(root,construction_pid=None):
    start=time.time();status(root,"all",state="waiting_for_construction")
    if construction_pid:
        while True:
            try:os.kill(construction_pid,0)
            except ProcessLookupError:break
            st=root/"status/construction.json"
            if st.exists() and json.loads(st.read_text()).get("state")=="complete":break
            time.sleep(20)
        st=json.loads((root/"status/construction.json").read_text())
        if st.get("state")!="complete":raise RuntimeError("Construction process stopped before completion")
    else:
        from .construct import merge_construction_shards
        workers=[spawn(root,"construct",gpu,shard=gpu,shards=2) for gpu in (0,1)]
        write_json(root/"construction_workers.json",dict(pids=[p.pid for p in workers],gpus=[0,1],shards=2))
        status(root,"all",state="construction",worker_pids=[p.pid for p in workers],gpus=[0,1])
        status(root,"construction",state="running",worker_pids=[p.pid for p in workers],shards=2)
        join(workers)
        merge_construction_shards(root,shards=2)
        status(root,"construction",state="complete",done=len(rows(root/"data/constructions.jsonl")),shards=2)
    from .data import build
    build(root)
    # Both GPUs perform the remaining label-independent extraction, then both score.
    status(root,"all",state="path_semantic_extraction",gpus=[0,1])
    join([spawn(root,"path-masks",gpu,shard=gpu,shards=2) for gpu in (0,1)])
    merge_path_masks(root,shards=2)
    status(root,"all",state="collect",gpus=[0,1])
    join([spawn(root,"collect",gpu,shard=gpu,shards=2) for gpu in (0,1)])
    status(root,"all",state="m1")
    join([spawn(root,"m1",0,kind="A"),spawn(root,"m1",1,kind="P")])
    fit_grid(root)
    join([spawn(root,"grid",0,shard=0,shards=2),spawn(root,"grid",1,shard=1,shards=2)])
    select_and_ablations(root)
    join([spawn(root,"amp-dev",0,shard=0,shards=2),spawn(root,"amp-dev",1,shard=1,shards=2)])
    select_amplification(root)
    join([spawn(root,"final",0,shard=0,shards=2),spawn(root,"final",1,shard=1,shards=2)])
    select_amplification(root)
    join([spawn(root,"mechanisms",0,split="dev"),spawn(root,"mechanisms",1,split="test")])
    join([spawn(root,"paths",0,shard=0,shards=2),spawn(root,"paths",1,shard=1,shards=2)])
    write_json(root/"runtime.json",dict(total_seconds=time.time()-start,automatic_pilot=True,gpus=[0,1]))
    report(root)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--stage",choices=["prepare","construct","build","path-masks","collect","m1","fit-grid","grid","select","amp-dev","final","mechanisms","paths","report","all"],default="all")
    p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=2)
    p.add_argument("--kind",choices=["A","P"]);p.add_argument("--split",default="test")
    p.add_argument("--construction-pid",type=int)
    a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    try:
        if a.stage=="prepare":
            from .data import prepare
            prepare(root)
        elif a.stage=="construct":
            from .construct import construct
            construct(root,shard=a.shard,shards=a.shards)
        elif a.stage=="build":
            from .data import build
            build(root)
        elif a.stage=="path-masks":mask_paths(root,a.shard,a.shards)
        elif a.stage=="collect":collect(root,a.shard,a.shards)
        elif a.stage=="m1":m1_worker(root,a.kind)
        elif a.stage=="fit-grid":fit_grid(root)
        elif a.stage=="grid":score_operators(root,"grid",a.shard,a.shards)
        elif a.stage=="select":select_and_ablations(root)
        elif a.stage=="amp-dev":score_operators(root,"amp_dev",a.shard,a.shards)
        elif a.stage=="final":score_operators(root,"final",a.shard,a.shards)
        elif a.stage=="mechanisms":mechanisms(root,a.split)
        elif a.stage=="paths":path_worker(root,a.shard,a.shards)
        elif a.stage=="report":report(root)
        elif a.stage=="all":all_stages(root,a.construction_pid)
    except Exception as exc:
        status(root,a.stage,state="failed",error=str(exc));traceback.print_exc();raise
if __name__=="__main__":main()
