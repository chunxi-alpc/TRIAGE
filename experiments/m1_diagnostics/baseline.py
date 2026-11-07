"""Exploratory original-text baselines; no label inference off the annotated path."""
import time
from .common import *
from .data import parse_tree,flatten_tree,mega_prompt
from .engine import Engine,TooLong

def medguide_case(e,case):
    r=case["native"];options=r["options"];letters=list("ABCDEFGH"[:len(options)])
    if not 2<=len(options)<=8 or r["answer"] not in letters:
        return dict(status="unscorable",reason="unsupported_original_options_or_answer")
    ids=[e.tokenizer.encode(letter,add_special_tokens=False) for letter in letters]
    if any(len(x)!=1 for x in ids):raise ValueError("Option letters must be single tokens")
    c=e.prepare(r["prompt"],task="mcq")
    logits=e.run(c,capture=False,label_ids=[x[0] for x in ids])["label_logits"]
    predicted=letters[int(np.argmax(logits))]
    return dict(status="complete",prediction=predicted,answer=r["answer"],correct=predicted==r["answer"],
      options=letters,label_logits=logits,tokens=c.length,prompt=r["prompt"])

def mega_case(e,case,dest):
    if not case["tree_text"] or not case["annotated_path"]:
        return dict(status="unscorable",reason="original_record_has_no_matching_tree_or_path")
    try:tree=parse_tree(case["tree_text"]);nodes=flatten_tree(tree)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(status="unscorable",reason="original_tree_cannot_be_parsed",detail=str(exc))
    saved=read(dest) if dest.exists() else {}
    scored=saved.get("node_scores",{})
    labels={str(s["index"]):s["decision_outcome"] for s in case["annotated_path"] if "decision_outcome" in s}
    def score(node):
        index=str(node["index"]);key=index+"::"+node["description"]
        if key not in scored:
            prompt=mega_prompt(case,node)
            c=e.prepare(prompt);out=e.run(c,capture=False)
            scored[key]=dict(index=index,condition=node["description"],ell=out["ell"],prediction="yes" if out["ell"]>0 else "no",
                label=labels.get(index),tokens=c.length)
            write(dest,dict(sample_id=case["sample_id"],source_id=case["source_id"],model=e.model_name,precision=PRECISION,
                status="running",node_scores=scored))
        return scored[key]
    annotated=[]
    for step in case["annotated_path"]:
        if "decision_outcome" not in step:continue
        item=score(dict(index=step["index"],description=step["condition"]))
        annotated.append(dict(item,label=step["decision_outcome"],correct=item["prediction"]==step["decision_outcome"]))
    predicted_path=[];current=tree
    while True:
        predicted_path.append(current["index"])
        if current.get("leaf") or not ("yes" in current or "no" in current):break
        item=score(current);branch=item["prediction"]
        if branch not in current:
            return dict(status="unscorable",reason="original_tree_missing_branch",node_scores=scored,annotated_nodes=annotated)
        current=current[branch]
    gold_path=[s["index"] for s in case["annotated_path"]]
    return dict(status="complete",node_scores=scored,annotated_nodes=annotated,
        node_correct=sum(n["correct"] for n in annotated),node_total=len(annotated),
        annotated_case_correct=all(n["correct"] for n in annotated),path_correct=predicted_path==gold_path,
        predicted_path=predicted_path,gold_path=gold_path,
        off_path_nodes=sum(str(i) not in labels for i in predicted_path[:-1]),
        off_path_labels_generated=False)

def run(model,engine=None):
    e=engine or Engine(model);start=time.time()
    for dataset,file in [("medguide","medguide_200.jsonl"),("mega","mega_200.jsonl")]:
        cases=rows(ROOT/"data"/file)
        for i,case in enumerate(cases):
            dest=ROOT/"baseline"/model/PRECISION/dataset/(case["sample_id"].replace(":","_")+".json")
            old=read(dest) if dest.exists() else {}
            if old.get("status") in ["complete","unscorable","overlength"]:continue
            try:
                result=medguide_case(e,case) if dataset=="medguide" else mega_case(e,case,dest)
            except TooLong as exc:
                result=dict(status="overlength",tokens=exc.tokens,limit=exc.limit,
                  partial_node_scores=read(dest).get("node_scores",{}) if dest.exists() else {})
            write(dest,dict(sample_id=case["sample_id"],source_id=case["source_id"],row_index=case["row_index"],
                model=model,precision=PRECISION,dataset=dataset,**result))
            status("baseline_"+model,state="running",dataset=dataset,done=i+1,total=200,elapsed=time.time()-start)
            if i==19:
                write(ROOT/"timing"/f"{model}_{dataset}_pilot20.json",dict(cases=20,elapsed_seconds=time.time()-start))
        start=time.time()
    status("baseline_"+model,state="complete")
    return e

def summary():
    out=dict(tables=[],overlap={})
    by={}
    for model in MODELS:
        for dataset in ["medguide","mega"]:
            vals=[read(p) for p in sorted((ROOT/"baseline"/model/PRECISION/dataset).glob("*.json"))]
            vals=[r for r in vals if r.get("status")!="running"];good=[r for r in vals if r["status"]=="complete"]
            by[model,dataset]={r["sample_id"]:r for r in good}
            base=dict(model=model,dataset=dataset,frozen_cases=200,finished=len(vals),scored=len(good),
                sources=len({r["source_id"] for r in vals}),scored_sources=len({r["source_id"] for r in good}),
                overlength=sum(r["status"]=="overlength" for r in vals),unscorable=sum(r["status"]=="unscorable" for r in vals))
            if dataset=="medguide":base.update(correct=sum(r["correct"] for r in good),wrong=sum(not r["correct"] for r in good))
            else:base.update(correct=sum(r["path_correct"] for r in good),wrong=sum(not r["path_correct"] for r in good),
                node_correct=sum(r["node_correct"] for r in good),node_wrong=sum(r["node_total"]-r["node_correct"] for r in good),
                node_total=sum(r["node_total"] for r in good),annotated_cases_wrong=sum(not r["annotated_case_correct"] for r in good))
            out["tables"].append(base)
    for dataset in ["medguide","mega"]:
        a,b=by["qwen",dataset],by["medgemma",dataset];common=sorted(set(a)&set(b))
        field="correct" if dataset=="medguide" else "path_correct"
        aw={i for i in common if not a[i][field]};bw={i for i in common if not b[i][field]}
        out["overlap"][dataset]=dict(common_scored=len(common),both_wrong=len(aw&bw),qwen_only_wrong=len(aw-bw),
            medgemma_only_wrong=len(bw-aw),both_correct=len(common)-len(aw|bw),
            both_wrong_sources=len({a[i]["source_id"] for i in aw&bw}),both_wrong_ids=sorted(aw&bw))
        if dataset=="mega":
            an={(i,str(n["index"])):n for i in common for n in a[i]["annotated_nodes"]}
            bn={(i,str(n["index"])):n for i in common for n in b[i]["annotated_nodes"]}
            ids=set(an)&set(bn);x={k for k in ids if not an[k]["correct"]};y={k for k in ids if not bn[k]["correct"]}
            out["overlap"]["mega_nodes"]=dict(common_nodes=len(ids),both_wrong=len(x&y),qwen_only_wrong=len(x-y),medgemma_only_wrong=len(y-x))
    return out
