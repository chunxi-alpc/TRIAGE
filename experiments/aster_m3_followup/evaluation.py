import time,gc,re
import torch
from .common import *
from .scoring import *
def pair_reference_prompt(view, group):
    """Use structural ancestors from this view's rendered tree, without outcomes."""
    current_conditions=dict(re.findall(r"(?ms)^(G\d+):[ \t]*(.*?)(?=\n  是 ->)",view["prompt"]))
    ancestors=[]
    for ancestor in group.get("ancestors",[]):
        condition=current_conditions.get(ancestor["gid"],ancestor["condition"])
        ancestors.append(re.sub(r"（判断为[是否]）$","",condition.strip()))
    condition=re.sub(r"（判断为[是否]）$","",view["condition"].strip())
    return ("原祖先条件：\n"+"\n".join(ancestors)+"\n\n病历：\n"+view["patient"]
        +"\n\n当前节点："+view["gid"]+"\n当前条件："+condition
        +"\n请判断当前条件。只输出“是”或“否”。\n答案：")
def progress(stage,start,n,total,engine):
    if n==20 or n%100==0 or n==total:
        elapsed=time.perf_counter()-start
        status(stage,state="complete" if n==total else "running",done=n,total=total,
            elapsed_seconds=elapsed,seconds_per_item=elapsed/max(1,n),
            estimated_total_hours=elapsed/max(1,n)*total/3600,
            peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,engine_counts=engine.counts)
def p0(engine,shard):
    start=time.perf_counter();basis=frozen()
    editors={m:load_editor(engine,m) for m in ["learned_reference","learned_self"]}
    cases={c["case_id"]:c for c in rows(OLD/"data/test.jsonl")}
    tasks=rows(ROOT/"p0/missing_nodes.jsonl")[shard::2]
    path=ROOT/"p0"/f"fixed_nodes_{shard}.jsonl"
    done={(r["case_id"],r["gid"],r["method"]) for r in rows(path)}
    for i,t in enumerate(tasks):
        for m in t["methods"]:
            key=(t["case_id"],t["gid"],m)
            if key in done:continue
            e,s=editors.get(m,(None,None))
            ell,diag=node(engine,cases[t["case_id"]],t["gid"],m,e,s,basis)
            append(path,dict(case_id=t["case_id"],gid=t["gid"],method=m,label=t["label"],ell=ell,prediction=int(ell>0),**diag))
        progress(f"p0_nodes_{shard}",start,i+1,len(tasks),engine)
    for t in rows(ROOT/"p0/changed_cases.jsonl")[shard::2]:
        for m in ["full","learned_reference","learned_self","aster"]:
            e,s=editors.get(m,(None,None))
            ell,diag=node(engine,cases[t["case_id"]],t["gid"],m,e,s,basis)
            append(ROOT/"p0"/f"changed_node_reruns_{shard}.jsonl",dict(case_id=t["case_id"],gid=t["gid"],method=m,
                ell=ell,prediction=int(ell>0),original_ell=t["scores"][m]["ell"] if t["scores"][m] else None,**diag))
    status(f"p0_nodes_{shard}",state="complete",tasks=len(tasks),elapsed_seconds=time.perf_counter()-start)
def calibration(engine):
    if (ROOT/"calibration/mega.json").exists() and (ROOT/"calibration/medguide.json").exists():return
    for dataset in ["mega","medguide"]:
        path=ROOT/"calibration"/f"{dataset}_scores.jsonl";done={r["key"] for r in rows(path)}
        if dataset=="mega":
            for c in rows(OLD/"data/train.jsonl"):
                indices=sorted(set([0,len(c["gold_path"])//2,len(c["gold_path"])-1]))
                for i in indices:
                    n=c["gold_path"][i];key=c["case_id"]+":"+n["gid"]
                    if key in done:continue
                    ell,_=node(engine,c,n["gid"],"full")
                    append(path,dict(key=key,case_id=c["case_id"],gid=n["gid"],ell=ell,label=n["label"]))
        else:
            for c in rows(ROOT/"data/medguide_calibration.jsonl"):
                if c["sample_id"] in done:continue
                r=mcq(engine,c,"full");r["key"]=c["sample_id"];append(path,r)
        write(ROOT/"calibration"/f"{dataset}.json",fit_bias(rows(path),binary=dataset=="mega"))
    status("calibration",state="complete")
def evaluate(engine,name,dataset):
    path=ROOT/"predictions"/dataset/f"{name}.jsonl"
    if (ROOT/"status"/f"eval_{dataset}_{name}.json").exists() and read(ROOT/"status"/f"eval_{dataset}_{name}.json")["state"]=="complete":return
    done={r["case_id"] for r in rows(path)};start=time.perf_counter()
    if name=="bias" and (dataset.startswith("medguide") or dataset=="pairs"):
        full_status=ROOT/"status"/f"eval_{dataset}_full.json"
        while not full_status.exists() or read(full_status).get("state")!="complete":
            job=ROOT/"jobs"/f"{dataset}_full.json"
            if job.exists() and read(job).get("state")=="failed":
                raise RuntimeError("Full baseline failed; calibration requires its saved scores")
            time.sleep(2)
    editor,spec=load_editor(engine,name) if name not in ["full","aster","bias","direct_reference"] else (None,None)
    basis=frozen() if name=="aster" else None
    if dataset.startswith("medguide"):
        split=dataset.removeprefix("medguide_")
        cases=rows(ROOT/"data"/f"medguide_{split}.jsonl")
        bias=read(ROOT/"calibration/medguide.json")["bias"] if name=="bias" else 0.
        full_cache={r["case_id"]:r for r in rows(ROOT/"predictions"/dataset/"full.jsonl")} if name=="bias" else {}
        for i,c in enumerate(cases):
            if c["sample_id"] not in done:
                if c["sample_id"] in full_cache:
                    r=dict(full_cache[c["sample_id"]]);r["label_logits"]=list(r["label_logits"])
                    r["label_logits"][0]+=bias
                    r["prediction"]="ABCDEFGH"[int(np.argmax(r["label_logits"]))]
                    r.update(method=name,correct=r["prediction"]==r["label"],reused_from="full label_logits")
                elif name=="bias":raise ValueError("Completed full baseline lacks MCQ scores")
                else:r=mcq(engine,c,name,editor,spec,basis,bias)
                append(path,r)
            progress(f"eval_{dataset}_{name}",start,i+1,len(cases),engine)
    elif dataset=="mega":
        cases=rows(OLD/"data/test.jsonl");bias=read(ROOT/"calibration/mega.json")["bias"] if name=="bias" else 0.
        cached=old_predictions() if name=="bias" else {}
        for i,c in enumerate(cases):
            if c["case_id"] not in done:
                scores={n["gid"]:n for n in cached[c["case_id"],"full"]["predictions"]} if cached else None
                append(path,predict_case(engine,c,name,editor,spec,basis,bias,scores))
            progress(f"eval_{dataset}_{name}",start,i+1,len(cases),engine)
    elif dataset=="pairs":
        ps=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="test"]
        views={v["view_id"]:v for v in rows(ROOT/"data/views.jsonl")}
        groups={g["group_id"]:g for g in rows(LEGACY/"data/groups_main.jsonl")}
        node_path=ROOT/"predictions/pair_views"/f"{name}.jsonl"
        scored={r["view_id"]:r for r in rows(node_path)}
        full_cache={r["view_id"]:r for r in rows(ROOT/"predictions/pair_views/full.jsonl")} if name=="bias" else {}
        bias=read(ROOT/"calibration/mega.json")["bias"] if name=="bias" else 0.
        for i,p in enumerate(ps):
            if p["pair_id"] not in done:
                for side in ["left","right"]:
                    vid=p[side];v=views[vid]
                    if vid in scored:continue
                    if vid in full_cache:
                        row=dict(full_cache[vid]);row["ell"]=float(row["ell"])+bias
                        row.update(prediction=int(row["ell"]>0),reused_from="full ell plus fitted bias")
                        scored[vid]=row;append(node_path,row);continue
                    if name=="bias":raise ValueError("Completed full baseline lacks pair scores")
                    ref=None
                    if spec and spec["reference"]=="reference":
                        donor=pair_reference_prompt(v,groups[v["group_id"]])
                        ref=ref_heads(engine,donor)
                    context=engine.prepare(v["prompt"])
                    if name in ["full","bias"]:out=engine.run(context,capture=False)
                    elif name=="aster":out=engine.run(context,operators=basis,capture=False)
                    else:
                        if spec["reference"]=="self":ref=engine.run(context)["heads"].flatten(0,1).to(engine.device)
                        out=engine.run(context,m3=editor,reference=ref,capture=False)
                    ell=out["ell"]+bias
                    row=dict(view_id=vid,ell=ell,label=v["label"],prediction=int(ell>0),**diagnostics(out))
                    scored[vid]=row;append(node_path,row)
                a,b=[scored[p[s]] for s in ["left","right"]]
                append(path,dict(case_id=p["pair_id"],source_id=p["source_id"],method=name,kind=p["kind"],
                    left=a,right=b,correct=a["prediction"]==a["label"] and b["prediction"]==b["label"],
                    prediction_flipped=a["prediction"]!=b["prediction"],label_flipped=a["label"]!=b["label"],status="complete"))
            progress(f"eval_{dataset}_{name}",start,i+1,len(ps),engine)
    status(f"eval_{dataset}_{name}",state="complete",records=len(rows(path)),elapsed_seconds=time.perf_counter()-start,
        peak_memory_gb=torch.cuda.max_memory_allocated()/2**30)
    del editor;gc.collect();torch.cuda.empty_cache()
