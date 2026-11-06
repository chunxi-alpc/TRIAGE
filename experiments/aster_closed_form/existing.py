"""Existing-data-only entry point. Never invokes construction or annotation."""
import argparse, ast, copy, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from collections import Counter, defaultdict
from .common import rows, write_rows, write_json, append, status, SEED
from . import pipeline as base

ORIGINAL=Path("/home/gcx/aster_closed_form_20261105")
HISTORICAL=Path("/home/gcx/semantic_masking_followup_20261105")
ROOT=Path("/home/gcx/aster_existing_data_20261106")
CODE=Path(__file__).resolve().parents[2]

def historical_template():
    source=(HISTORICAL/"score.py").read_text()
    tree=ast.parse(source)
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SYSTEM" for t in node.targets):
            selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=="make_prompt":
            selected.append(node)
    namespace={}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(HISTORICAL/"score.py"),"exec"),namespace)
    return namespace["SYSTEM"],namespace["make_prompt"]

def prepare(root):
    if (root/"data/prepared.json").exists():
        return json.loads((root/"data/prepared.json").read_text())
    for folder in ("data", "scores", "operators", "mechanisms", "paths", "results", "status", "logs", "diagnostics", "timing", "m1"):
        (root/folder).mkdir(parents=True,exist_ok=True)
    problems=json.loads((ORIGINAL/"analysis_failure/condition_consistency.json").read_text())["records"]
    removed=[r for r in problems if r["assessment"] in ("unsupported_flip","needs_clarification")]
    assert Counter(r["assessment"] for r in removed)=={"unsupported_flip":7,"needs_clarification":2}
    gids={r["group_id"] for r in removed}
    original_pairs=rows(ORIGINAL/"data/pairs.jsonl")
    rejected=[p for p in original_pairs if p["kind"]=="C" and p["group_id"] in gids]
    removed_ids={p["right"] for p in rejected}
    assert len(rejected)==len(removed_ids)==9
    pairs=[p for p in original_pairs if p not in rejected]
    views=[v for v in rows(ORIGINAL/"data/views.jsonl") if v["view_id"] not in removed_ids]
    by_id={v["view_id"]:v for v in views}
    sources=defaultdict(set)
    for v in views:
        sources[v["source_id"]].add(v["split"])
    assert all(len(s)==1 for s in sources.values())
    for p in pairs:
        for side in ("left","right"):
            v=by_id[p[side]]
            assert v["label"]==p[side+"_label"]
            assert (v["split"],v["source_id"],v["group_id"])==(p["split"],p["source_id"],p["group_id"])
        assert (p["left_label"]!=p["right_label"]) == (p["kind"] in ("C","F"))
    assert Counter(v["split"] for v in views)=={"fit":561,"dev":119,"test":610}
    expected={"fit":{"A":156,"P:clinical":57,"P:preference":57,"C":13,"F":158},
              "dev":{"A":40,"P:clinical":10,"P:preference":9,"C":1,"F":40},
              "test":{"A":200,"P:clinical":49,"P:preference":49,"C":14,"F":200}}
    actual={s:dict(Counter(p["kind"]+(":"+p["subtype"] if p["kind"]=="P" else "") for p in pairs if p["split"]==s)) for s in expected}
    assert actual==expected and len(views)==1290 and len(pairs)==1053
    write_rows(root/"data/views.jsonl",views)
    write_rows(root/"data/pairs.jsonl",pairs)
    write_rows(root/"data/excluded_C.jsonl",removed)
    write_json(root/"data/valid_counts.json",dict(views=len(views),pairs=len(pairs),by_split=actual))
    for name in ("groups_main.jsonl","semantic_masks.jsonl","path_semantic_masks.jsonl"):
        shutil.copy2(ORIGINAL/"data"/name,root/"data"/name)
    shutil.copytree(ORIGINAL/"m1",root/"m1",dirs_exist_ok=True)
    (root/"features").symlink_to(ORIGINAL/"features",target_is_directory=True)
    for path in (ORIGINAL/"scores").glob("baseline_*.jsonl"):
        write_rows(root/"scores"/path.name,[r for r in rows(path) if r["view_id"] in by_id])
    for split in ("dev","test"):
        for name in (f"h1_pairs_{split}.jsonl",f"h1_random_heads_{split}.json",f"mechanism_subset_{split}.json"):
            shutil.copy2(ORIGINAL/"mechanisms"/name,root/"mechanisms"/name)
        write_rows(root/"mechanisms"/f"h2_pairs_{split}.jsonl",
                   [r for r in rows(ORIGINAL/"mechanisms"/f"h2_pairs_{split}.jsonl") if r["operation"]=="full"])

    # All existing vignette cases from the original twenty test guidelines.
    from experiments.aster_representation.tree import tree_guideline
    mega=Path("/home/gcx/TRIAGE/data/MEGA-CDP/train_val")
    groups=rows(root/"data/groups_main.jsonl")
    tests={g["guideline_id"]:g for g in groups if g["split"]=="test"}
    raw={r["filename"]:r for r in rows(mega/"decision_trees.jsonl")}
    path_records={r["guideline_id"]:r for r in rows(mega/"case_paths.jsonl") if r["guideline_id"] in tests}
    old_cases={c["case_id"]:c for c in rows(ORIGINAL/"data/path_cases.jsonl")}
    maps={sid:tree_guideline(json.loads(raw[g["source_filename"]]["text"])["decision_tree"])[1] for sid,g in tests.items()}
    cases=[]
    for c in rows(mega/"vignette_cases.jsonl"):
        sid=c["guideline_id"]
        if sid not in tests:
            continue
        g=tests[sid]; mapping=maps[sid]
        rp=path_records[sid]["paths"][int(c["path_id"])-1]
        expected_path=[dict(gid="G"+str(mapping[str(n["index"])]["node"]),label=int(n["decision_outcome"]=="yes")) for n in rp if "decision_outcome" in n]
        item=dict(case_id=f"path-{sid}-{c['path_id']}",source_id=g["source_id"],patient=c["text"],
                  guideline=g["guideline"],definitions=g["definitions"],gold_path=expected_path,
                  gold_leaf="L"+str(mapping[str(rp[-1]["index"])]["node"]))
        if item["case_id"] in old_cases:
            assert item==old_cases[item["case_id"]]
        item["original_100_subset"]=item["case_id"] in old_cases
        cases.append(item)
    cases.sort(key=lambda c:tuple(int(s) for s in c["case_id"].split("-")[1:]))
    assert len(cases)==432 and sum(c["original_100_subset"] for c in cases)==100
    assert len({c["case_id"] for c in cases})==432 and len({c["source_id"] for c in cases})==20
    write_rows(root/"data/path_cases.jsonl",cases)

    history=rows(HISTORICAL/"data/inputs_full.jsonl")
    facts=[r for r in history if r["population"]=="constructed" and r["split"]=="test" and r["view"] in ("full","direct","condition_ancestors")]
    full=[r for r in facts if r["view"]=="full"]
    assert len(full)==168 and len({r["node_id"] for r in full})==28
    deletion_full=[r for r in history if r["population"]=="original" and r["split"]=="test" and r["view"]=="full" and r["changes"].get("definition_lost") and r["changes"].get("restoration_available")]
    deletion_ids={r["node_id"] for r in deletion_full}
    deletion=[r for r in history if r["node_id"] in deletion_ids and r["population"]=="original" and
              (r["view"]!="irrelevant_restore" or r["changes"].get("matched_irrelevant_available"))]
    assert len(deletion_full)==11 and sum(r["changes"]["matched_irrelevant_available"] for r in deletion_full)==10
    write_rows(root/"diagnostics/fact_preference_inputs.jsonl",facts)
    write_rows(root/"diagnostics/deletion_restore_inputs.jsonl",deletion)
    fpairs=[]
    for nid in sorted({r["node_id"] for r in full}):
        index={(r["metric_level"],r["background"]):r for r in full if r["node_id"]==nid}
        for bg in ("neutral","cue"):
            a,b=index[("below",bg)],index[("above",bg)]
            assert a["gold"]!=b["gold"]
            fpairs.append(dict(pair_id=nid+":F:"+bg,kind="F",node_id=nid,source_id=a["source_id"],
                               left=a["example_id"],right=b["example_id"]))
        for level in ("below","above","same_side"):
            a,b=index[(level,"neutral")],index[(level,"cue")]
            assert a["gold"]==b["gold"]
            fpairs.append(dict(pair_id=nid+":P:"+level,kind="P",node_id=nid,source_id=a["source_id"],
                               left=a["example_id"],right=b["example_id"]))
    assert Counter(p["kind"] for p in fpairs)=={"F":56,"P":84}
    write_rows(root/"diagnostics/fact_preference_pairs.jsonl",fpairs)
    hs,_=historical_template()
    write_json(root/"diagnostics/template.json",dict(system=hs,source=str(HISTORICAL/"score.py"),
        policy="Original saved user prompts, system, labels, and input formats; natural and ASTER scores recomputed together. No historical score mixing."))
    cfg=json.loads((ORIGINAL/"configuration.json").read_text())
    cfg.update(selection="main dev only",path_cases=432,definition_targets=None,data_mode="existing_only",
               original_run=str(ORIGINAL),historical_run=str(HISTORICAL),new_cases=0,new_annotations=0)
    write_json(root/"configuration.json",cfg)
    manifest=dict(seed=SEED,main_nodes=dict(Counter(g["split"] for g in groups)),
        main_sources={s:len({g["source_id"] for g in groups if g["split"]==s}) for s in expected},
        dev_sources=sorted({g["source_id"] for g in groups if g["split"]=="dev"}),
        removed_C=9,path_cases=432,original_path_subset=100,
        historical_fact_preference_nodes=28,historical_fact_pairs=56,historical_preference_pairs=84,
        deletion_nodes=11,irrelevant_restore_nodes=10,automatic_exploratory=True,
        sources=[str(ORIGINAL),str(HISTORICAL),str(mega/"vignette_cases.jsonl"),str(mega/"case_paths.jsonl"),str(mega/"decision_trees.jsonl")])
    write_json(root/"data/manifest.json",manifest)
    reuse=dict(reused=["main natural activations and gradients","full / ignore / direct / successful semantic and boundary scores",
                       "M1 attribution, exact ranking, prefixes and head types","H1","H2 full-head interventions"],
               recomputed=["M2/M3 and 18 development candidates","fusion training and selection","edited scores and ablations",
                           "H2 operator-dependent interventions","joint four-component diagnostic","local theory and H3",
                           "historical diagnostics with original prompt templates","all 432 autonomous paths"],
               missing_policy="Unavailable existing semantic/boundary inputs reported on available subsets; no generation and no fallback counted as masking",
               historical_isolation="Historical diagnoses never enter fit or selection",old_outputs_unchanged=True)
    write_json(root/"results/reuse.json",reuse)
    summary=dict(inputs=1290,pairs=1053,by_split=actual,path_cases=432,historical_inputs=len(facts)+len(deletion),
                 existing_input_validation="labels, pair indices and source splits agree")
    write_json(root/"data/prepared.json",summary)
    status(root,"prepare",state="complete",**summary)
    return summary

def selection_notes(root):
    choice=json.loads((root/"results/selection.json").read_text())
    from .evaluation import selection_key
    metrics=choice["development_metrics"]
    best=selection_key(metrics[choice["method"]])
    tied=[c for c in choice["candidates"] if selection_key(metrics[c["method"]])==best]
    choice.update(selection_cohort="main development pairs only",development_C_pairs=1,
                  tied_on_primary_metrics=[dict(method=c["method"],rank=c["rank"],config=c["config"]) for c in tied],
                  tie_interpretation="Development C has one pair. Ties do not establish parameter superiority.")
    write_json(root/"results/selection.json",choice)

def joint_components(root,engine,split):
    import torch
    from .m1 import _heads
    from .mechanisms import _summarize
    views,pairs=base.load_data(root)
    features=base.load_features(root)
    operators=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    heads=sorted(int(h) for h in operators)
    output=root/"mechanisms"/f"joint_components_{split}.jsonl"
    saved={(r["pair_id"],r["component"]):r for r in rows(output)}
    h1={(r["pair_id"],r["control"]):r for r in rows(root/"mechanisms"/f"h1_pairs_{split}.jsonl")}
    h2={(r["pair_id"],str(r["head"]),r["operation"]):r for r in rows(root/"mechanisms"/f"h2_pairs_{split}.jsonl")}
    components=("full_head","nuisance_orthogonal","screened_orthogonal","oblique")
    selected=[p for p in pairs if p["split"]==split and p["kind"] in ("A","P")]
    for index,p in enumerate(selected):
        missing=[c for c in components if (p["pair_id"],c) not in saved]
        if not missing:
            continue
        a,b=features[p["left"]],features[p["right"]]
        z0,z1=_heads(a),_heads(b)
        te=float(b["ell"])-float(a["ell"])
        contexts=None
        try:
            for component in missing:
                cached=h1.get((p["pair_id"],"selected_union")) if component=="full_head" else h2.get((p["pair_id"],"joint","nuisance")) if component=="oblique" else None
                if cached is not None:
                    effect={k:cached.get(k) for k in ("forward_ell","reverse_ell","ie_forward","ie_reverse","te","forward_normalized","reverse_normalized","recovery")}
                    origin="unchanged full-head repair" if component=="full_head" else "recomputed H2 joint repair"
                else:
                    if contexts is None:
                        contexts=(engine.prepare(views[p["left"]]["prompt"]),engine.prepare(views[p["right"]]["prompt"]))
                    forward,reverse={},{}
                    for h in heads:
                        op=operators[h]
                        delta=z1[h]-z0[h]
                        n,bmat,un=op["N"],op["B"],op["U_N"]
                        if component=="full_head":piece=delta
                        elif component=="nuisance_orthogonal":piece=un@(un.T@delta)
                        elif component=="screened_orthogonal":piece=n@(n.T@delta)
                        else:piece=n@(bmat@delta)
                        forward[h]=z0[h]+piece
                        reverse[h]=z1[h]-piece
                    fe=engine.run(contexts[0],replacements=forward,capture=False)["ell"]
                    re=engine.run(contexts[1],replacements=reverse,capture=False)["ell"]
                    df,dr=fe-float(a["ell"]),re-float(b["ell"])
                    effect=dict(forward_ell=fe,reverse_ell=re,ie_forward=df,ie_reverse=dr,te=te,
                        forward_normalized=df/te if abs(te)>=.1 else None,
                        reverse_normalized=dr/-te if abs(te)>=.1 else None,
                        recovery=.5*(df/te+dr/-te) if abs(te)>=.1 else None)
                    origin="new bidirectional joint intervention"
                item=dict(pair_id=p["pair_id"],kind=p["kind"],subtype=p.get("subtype"),group_id=p["group_id"],
                    source_id=p["source_id"],split=split,component=component,head_count=len(heads),
                    normalized_effect_eligible=abs(te)>=.1,status="complete",origin=origin,**effect)
                append(output,item);saved[(p["pair_id"],component)]=item
        finally:
            del contexts
        status(root,"joint_"+split,state="running",done=index+1,total=len(selected))
    summary=_summarize(list(saved.values()),["kind","component"])
    write_json(root/"mechanisms"/f"joint_components_{split}.json",dict(pairs=len(selected),summary=summary,
        normalization="All A/P pairs retained; normalized recovery only when absolute total effect >= 0.1",
        selection_use=False))
    status(root,"joint_"+split,state="complete",pairs=len(selected))

def historical_worker(root,shard,engine=None):
    import torch
    from . import engine as module
    engine=engine or module.Engine()
    saved_system=module.SYSTEM
    system,make_prompt=historical_template()
    module.SYSTEM=system
    operators=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    inputs=rows(root/"diagnostics/fact_preference_inputs.jsonl")+rows(root/"diagnostics/deletion_restore_inputs.jsonl")
    inputs=inputs[shard::2]
    path=root/"diagnostics"/f"scores_{shard}.jsonl"
    saved={(r["score_id"],r["method"]) for r in rows(path)}
    start=time.time()
    try:
        for i,r in enumerate(inputs):
            missing=[m for m in ("natural","aster") if (r["score_id"],m) not in saved]
            if not missing:
                continue
            context=None
            try:
                context=engine.prepare(make_prompt(r))
                for method in missing:
                    try:
                        result=engine.run(context,operators=operators if method=="aster" else None,capture=False)
                        item=dict(status="complete",ell=result["ell"],prediction=result["prediction"],tokens=result["tokens"])
                    except Exception as exc:
                        item=dict(status="failed",ell=None,error=str(exc))
                    item.update(score_id=r["score_id"],example_id=r["example_id"],node_id=r["node_id"],
                        source_id=r["source_id"],view=r["view"],method=method,gold=r["gold"])
                    append(path,item)
            except Exception as exc:
                for method in missing:
                    append(path,dict(score_id=r["score_id"],example_id=r["example_id"],node_id=r["node_id"],
                        source_id=r["source_id"],view=r["view"],method=method,gold=r["gold"],
                        status="failed",ell=None,error=str(exc)))
            finally:
                del context
            status(root,"historical_"+str(shard),state="running",done=i+1,total=len(inputs),
                elapsed_seconds=time.time()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        status(root,"historical_"+str(shard),state="complete",done=len(inputs),elapsed_seconds=time.time()-start)
    finally:
        module.SYSTEM=saved_system

def worker(root,stage,shard):
    import torch
    torch.set_num_threads(4);torch.manual_seed(SEED)
    if stage in ("grid","amp_dev","final"):
        base.score_operators(root,stage,shard,2)
    elif stage=="mechanisms":
        from .engine import Engine
        from .mechanisms import run_mechanisms
        engine=Engine()
        split=("dev","test")[shard]
        views,pairs=base.load_data(root);features=base.load_features(root)
        ops=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
        selection=json.loads((root/"results/selection.json").read_text())
        run_mechanisms(engine,views,pairs,features,ops,selection,root/"mechanisms",split=split)
        del features
        joint_components(root,engine,split)
        historical_worker(root,shard,engine)
    elif stage=="paths":
        base.path_worker(root,shard,2)
    else:
        raise ValueError(stage)

def parallel(root,stage):
    procs=[]
    for gpu in (0,1):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
        log=(root/"logs"/f"{stage}_{gpu}.log").open("a")
        command=[sys.executable,"-m","experiments.aster_closed_form.existing","--root",str(root),
                 "--stage",stage,"--shard",str(gpu)]
        procs.append(subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=log))
        log.close()
    status(root,"all",state=stage,worker_pids=[p.pid for p in procs],gpus=[0,1])
    failed=[]
    for p in procs:
        if p.wait():
            failed.append(p.pid)
    if failed:
        raise RuntimeError(f"{stage} workers failed: {failed}; see logs")

def run(root):
    import torch
    torch.set_num_threads(4)
    start=time.time()
    prepare(root)
    status(root,"all",state="fit_grid",gpus=[0,1])
    base.fit_grid(root)
    parallel(root,"grid")
    base.select_and_ablations(root)
    selection_notes(root)
    parallel(root,"amp_dev")
    base.select_amplification(root)
    parallel(root,"final")
    parallel(root,"mechanisms")
    parallel(root,"paths")
    from .existing_report import make_report
    status(root,"all",state="report")
    make_report(root)
    write_json(root/"runtime.json",dict(total_seconds=time.time()-start,gpus=[0,1]))
    status(root,"all",state="complete",report=str(root/"REPORT.zh.md"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("prepare","all","grid","amp_dev","final","mechanisms","paths","report"),default="all")
    parser.add_argument("--shard",type=int,default=0)
    args=parser.parse_args()
    try:
        if args.stage=="prepare":prepare(args.root)
        elif args.stage=="all":run(args.root)
        elif args.stage=="report":
            from .existing_report import make_report
            make_report(args.root)
        else:worker(args.root,args.stage,args.shard)
    except Exception as exc:
        status(args.root,args.stage,state="failed",error=str(exc))
        traceback.print_exc()
        raise
if __name__=="__main__":main()
