"""Retrospective analysis only: old precision, no new forwards or selection."""
from .common import *
from .statistics import summarize, ranking_stability
from collections import Counter
from functools import lru_cache

def old_scores(model):
    target=ROOT/"cpu"/f"{model}_scores.json"
    if target.exists():return read(target)
    values={}
    if model=="qwen":
        for f in sorted((QWEN/"scores").glob("baseline_*.jsonl")):
            values.update({r["view_id"]:r["ell"] for r in rows(f) if r.get("method")=="full" and r.get("ell") is not None})
    for v in rows(LEGACY[model]/"data/views.jsonl"):
        if v["view_id"] not in values:values[v["view_id"]]=feature(model,v["view_id"])["ell"]
    write(target,values);return values

def exact_rows(model,kind):
    out={(r["pair_id"],r["head"]):r for r in rows(LEGACY[model]/f"m1/exact_{kind}.jsonl")}
    if model=="medgemma" and kind=="A":
        for r in rows(REPAIR/"selection/128/patches.jsonl"):
            if len(r["heads"])==1:out[(r["pair_id"],r["heads"][0])]=dict(r,head=r["heads"][0])
    return out

def combination(combos,singles,hs):
    result=[]
    for r in combos:
        pid=r["pair_id"]
        if not all((pid,h) in singles for h in hs):continue
        item={k:r[k] for k in ["pair_id","group_id","source_id","kind","split"]}
        for direction in ["forward","reverse"]:
            total=sum(singles[pid,h]["ie_"+direction] for h in hs)
            joint=r["ie_"+direction]
            item[direction]=dict(joint=joint,sum_single=total,difference=joint-total,
                ratio=joint/total if total!=0 else None,small_sum=abs(total)<EPSILON)
        result.append(item)
    summary=dict(pairs=len(result),rows=result)
    for d in ["forward","reverse"]:
        a=[r[d] for r in result]
        summary[d]=dict(mean_joint=float(np.mean([x["joint"] for x in a])) if a else None,
            mean_sum_single=float(np.mean([x["sum_single"] for x in a])) if a else None,
            differences=distribution([x["difference"] for x in a]),small_sums=sum(x["small_sum"] for x in a),
            ratios=distribution([x["ratio"] for x in a if x["ratio"] is not None]))
    return summary

def run():
    path=ROOT/"cpu/summary.json"
    if path.exists():return read(path)
    status("cpu",state="running")
    allpairs=rows(QWEN/"data/pairs.jsonl");byid={p["pair_id"]:p for p in allpairs}
    views={v["view_id"]:v for v in rows(QWEN/"data/views.jsonl")}
    scores={m:old_scores(m) for m in MODELS}
    output=dict(seed=SEED,bootstrap=BOOTSTRAPS,precision={"qwen":"legacy_bf16_logits","medgemma":"bf16_backbone_fp32_label_projection"},
        effects=[],paired_effects=[],TE=[],score_metrics=[],heads=[],stability={},combination={},reproduction=[])
    cached=[]
    for split in ["dev","test"]:
        for r in rows(QWEN/f"mechanisms/h1_pairs_{split}.jsonl"):
            if r["control"] in ["type_selected","selected_union","all_query_heads"]:
                cached.append(dict(r,model="qwen",scheme="frozen",precision="legacy_bf16_logits"))
    for f in sorted((REPAIR/"evaluation").glob("pairs_*.jsonl")):
        for r in rows(f):
            if r["scheme"] in ["prefix64","prefix128"] and r["control"] in ["type_selected","selected_union","all_query_heads"]:
                cached.append(dict(r,model="medgemma",scheme="frozen" if r["scheme"]=="prefix128" else "historical",
                                   precision=PRECISION))
    write_rows(ROOT/"cpu/legacy_effects.jsonl",cached)
    grouped=defaultdict(list)
    for r in cached:grouped[(r["model"],r["scheme"],r["split"],r["kind"],r["control"])].append(r)
    for (m,scheme,split,kind,control),rs in grouped.items():
        eligible=[p for p in allpairs if p["split"]==split and p["kind"]==kind]
        universe=len(eligible) if split in ["fit","dev","test"] else len(rs)
        result=summarize(rs)
        result.update(model=m,scheme=scheme,split=split,kind=kind,control=control,universe_pairs=universe,
          missing_patch_pairs=universe-len(rs),all_pairs_estimable=len(rs)==universe,
          all_estimator_scope="全部已有修补配对；缺失缓存不填零、不补算测试")
        output["effects"].append(result)
        if m=="qwen" and split=="dev" and control=="type_selected":
            expected=read(QWEN/f"m1/m1_{kind}.json")["selected_recovery"]
            actual=result["estimates"]["original_effective"]
            assert abs(actual-expected)<1e-10
            output["reproduction"].append(dict(model=m,kind=kind,expected=expected,actual=actual,passed=True))
        if m=="medgemma" and scheme=="frozen" and split=="dev" and kind=="A" and control=="type_selected":
            expected=read(REPAIR/"selection/selection.json")["A_dev_recovery"]
            actual=result["estimates"]["original_effective"]
            assert abs(actual-expected)<1e-10
            output["reproduction"].append(dict(model=m,kind=kind,expected=expected,actual=actual,passed=True))
    for split in ["fit","dev","test"]:
        for kind in ["A","P","F"]:
            ps=[p for p in allpairs if p["split"]==split and p["kind"]==kind]
            common=[]
            for p in ps:
                if all(p[s] in scores[m] for m in MODELS for s in ["left","right"]) and all(
                    abs(scores[m][p["right"]]-scores[m][p["left"]])>=EPSILON for m in MODELS):common.append(p["pair_id"])
            for m in MODELS:
                records=[]
                for p in ps:
                    if not all(p[s] in scores[m] for s in ["left","right"]):continue
                    a,b=[scores[m][p[s]] for s in ["left","right"]]
                    records.append(dict(p,left_ell=a,right_ell=b,te=b-a,effective=abs(b-a)>=EPSILON,
                       both_wrong=int(a>0)!=views[p["left"]]["label"] and int(b>0)!=views[p["right"]]["label"]))
                output["TE"].append(dict(model=m,split=split,kind=kind,pairs=len(records),requested_pairs=len(ps),
                    effective=sum(r["effective"] for r in records),nodes=len({r["group_id"] for r in records}),
                    sources=len({r["source_id"] for r in records}),abs_TE=distribution([abs(r["te"]) for r in records]),
                    zero_denominators=sum(r["te"]==0 for r in records),both_wrong=sum(r["both_wrong"] for r in records),
                    common_effective_pairs=len(common)))
                write_rows(ROOT/"cpu"/f"{m}_{split}_{kind}_pair_scores.jsonl",records)
                if kind in ["A","F"]:
                    output["score_metrics"].append(dict(model=m,split=split,kind=kind,pairs=len(records),
                      sources=len({r["source_id"] for r in records}),mean_abs_change=float(np.mean([abs(r["te"]) for r in records])) if records else None,
                      margin_exceeded=sum(abs(r["te"])>abs(r[s+"_ell"]) for r in records for s in ["left","right"]),
                      margin_denominator=2*len(records),prediction_flips=sum((r["left_ell"]>0)!=(r["right_ell"]>0) for r in records),
                      flip_denominator=len(records),F_reference_used=False))
            for control in ["type_selected","selected_union"]:
                for m in MODELS:
                    rs=[r for r in grouped[m,"frozen",split,kind,control] if r["pair_id"] in common]
                    if rs:output["paired_effects"].append(dict(model=m,split=split,kind=kind,control=control,
                        common_effective_universe=len(common),**summarize(rs)))
    output["LSR"]=[]
    for m in MODELS:
        for split in ["fit","dev","test"]:
            a=rows(ROOT/"cpu"/f"{m}_{split}_A_pair_scores.jsonl");f=rows(ROOT/"cpu"/f"{m}_{split}_F_pair_scores.jsonl")
            common={r["group_id"] for r in a}&{r["group_id"] for r in f}
            av=[abs(r["te"]) for r in a if r["group_id"] in common];fv=[abs(r["te"]) for r in f if r["group_id"] in common]
            denominator=float(np.mean(fv)) if fv else 0
            output["LSR"].append(dict(model=m,split=split,common_nodes=len(common),
                value=float(np.mean(av))/denominator if av and denominator else None,
                numerator_mean_abs_A=float(np.mean(av)) if av else None,denominator_mean_abs_F=denominator,
                definition="共同节点上 mean(|ell_A_right-ell_A_left|) / mean(|ell_F_right-ell_F_left|)，零分母无法估计。"))
    selections=frozen()
    for m in MODELS:
        config=read(MODELS[m]/"config.json");config=config.get("text_config",config)
        types=config.get("layer_types",["full_attention"]*config["num_hidden_layers"])
        for name in ["H_A","H_P","H"]:
            hs=selections[m][name]
            output["heads"].append(dict(model=m,set=name,heads=len(hs),total=selections[m]["total_heads"],
              fraction=len(hs)/selections[m]["total_heads"],layer_counts=dict(Counter(h//32 for h in hs)),
              local_heads=sum(types[h//32]=="sliding_attention" for h in hs),
              global_heads=sum(types[h//32]!="sliding_attention" for h in hs)))
        for kind in ["A","P"]:
            status("cpu",state="running",model=m,kind=kind,step="source_bootstrap")
            ps=[p for p in allpairs if p["kind"]==kind and p["split"]=="fit" and all(p[s] in scores[m] for s in ["left","right"])
                and abs(scores[m][p["right"]]-scores[m][p["left"]])>=EPSILON]
            matrix=[];valid=[]
            for p in ps:
                left,right=feature(m,p["left"]),feature(m,p["right"])
                if left.get("grads") is None:continue
                t=float(right["ell"])-float(left["ell"])
                matrix.append((((right["heads"]-left["heads"])*left["grads"]).sum(-1).reshape(-1).double()/t).numpy());valid.append(p)
            k=len(selections[m]["H_"+kind])
            attr=np.array(matrix)
            output["stability"][m+"_"+kind+"_gradient"]=ranking_stability(attr,[p["source_id"] for p in valid],range(selections[m]["total_heads"]),k)
            expected=read(LEGACY[m]/f"m1/attribution_{kind}.json")["ranking"]
            maxdiff=max(abs(float(attr.mean(0)[r["head"]])-r["attribution"]) for r in expected)
            output["reproduction"].append(dict(model=m,kind=kind,gradient_max_difference=maxdiff,passed=maxdiff<1e-8))
            assert maxdiff<1e-8
            singles=exact_rows(m,kind);candidate_ids=sorted({h for pid,h in singles})
            exact_ps=[p for p in ps if all((p["pair_id"],h) in singles for h in candidate_ids)]
            emat=[[singles[p["pair_id"],h]["recovery"] for h in candidate_ids] for p in exact_ps]
            output["stability"][m+"_"+kind+"_exact"]=ranking_stability(emat,[p["source_id"] for p in exact_ps],candidate_ids,k)
            combos=[dict(r,**{x:byid[r["pair_id"]][x] for x in ["kind","split","source_id","group_id"]})
                for r in rows(REPAIR/"selection/128/patches.jsonl") if m=="medgemma"
                and r["pair_id"] in byid and byid[r["pair_id"]]["kind"]==kind
                and set(r["heads"])==set(selections[m]["H_"+kind])]
            combos+=grouped[m,"frozen","dev",kind,"type_selected"]+grouped[m,"frozen","test",kind,"type_selected"]
            output["combination"][m+"_"+kind]=combination(combos,singles,selections[m]["H_"+kind])
    write(path,output);status("cpu",state="complete",tables=len(output["effects"]))
    return output
