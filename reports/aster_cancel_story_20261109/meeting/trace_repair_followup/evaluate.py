"""Source-balanced real-tree evaluation, development selection and frozen final comparison."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import argparse,csv,gzip,importlib.util,json,sys,time
from collections import defaultdict,Counter
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_CACHE=Path("/home/gcx/aster_trace_repair_20261110")
SEED=20261110
BOOTSTRAPS=2000

def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def rows(p):
    p=Path(p)
    if not p.exists():return []
    opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"rt",encoding="utf-8") as f:return [json.loads(s) for s in f if s.strip()]
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item())+"\n",encoding="utf-8")
def csvout(p,rr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rr:return
    opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
        w.writeheader()
        for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def csvrows(p):
    p=Path(p);opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"rt",encoding="utf-8") as f:return list(csv.DictReader(f))
sys.path.insert(0,str(HERE.parents[3]))
from experiments.aster_closed_form.data import canonical_guide
def canonical(guide):
    return guide if "nodes" in guide else canonical_guide(guide)
def execute(case,node_scores):
    tree=canonical(case["guideline"]);nodes={n["id"]:n for n in tree["nodes"]}
    gid=tree["nodes"][0]["id"];gold=case["gold_path"];actual=[];first=None;error=None
    while gid in nodes:
        if len(actual)>len(nodes):error="cycle";break
        if gid not in node_scores:error="unavailable_score";break
        pred=int(node_scores[gid]>0);actual.append(dict(gid=gid,label=pred))
        j=len(actual)-1
        if first is None and (j>=len(gold) or actual[-1]!=gold[j]):
            first=dict(gid=gid,order=j,gold_gid=gold[j]["gid"] if j<len(gold) else None,
                gold_label=gold[j]["label"] if j<len(gold) else None,prediction=pred,score=float(node_scores[gid]))
        gid=nodes[gid]["yes" if pred else "no"]
    correct=None if error=="unavailable_score" else bool(not error and actual==gold and gid==case["gold_leaf"])
    return dict(correct=correct,first_error=first,path=actual,leaf=gid,error=error)

def read_scores(cache,split,tasks):
    out={}
    paths=sorted((cache/"scoring"/split).glob("**/scores.jsonl"))
    for p in paths:
        for r in rows(p):
            if r.get("status")=="ok":
                key=r.get("id",r.get("task_id",r.get("view_id")))
                if key in tasks:out[key]=r
    return out

def predict(args,tasks,scores,metadata,stage_dir):
    predictions={name:{} for name in metadata};predictions.setdefault("raw",{})
    for key,r in scores.items():predictions["raw"][key]=float(r["ell"])
    supplied=Path(args.predictions) if args.predictions else args.training_dir/(args.stage+"_predictions.csv.gz")
    if supplied.exists():
        for r in csvrows(supplied):
            if r["method"] in predictions and r["id"] in tasks:predictions[r["method"]][r["id"]]=float(r["score"])
    missing={key for key in scores if any(key not in predictions[name] for name in metadata)}
    if missing:
        spec=importlib.util.spec_from_file_location("followup_train",HERE/"train.py")
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        api=module.load_models(args.training_dir)
        batch=[];shape=None
        def flush():
            if not batch:return
            ids=[r[0] for r in batch];ells=np.array([scores[k]["ell"] for k in ids],float)
            heads=np.stack([r[1] for r in batch]);head_ids=batch[0][2]
            residual=None if batch[0][3] is None else np.stack([r[3] for r in batch])
            result=api.predict_all(ells,heads,head_ids=head_ids,residual=residual)
            for name,values in result.items():
                if name not in predictions:continue
                for key,value in zip(ids,np.atleast_1d(values)):predictions[name][key]=float(value)
            batch.clear()
        for key in sorted(missing):
            with np.load(scores[key]["feature"]) as a:
                heads=a["heads"].astype(np.float32)
                if heads.ndim>2:heads=heads.reshape(-1,heads.shape[-1])
                head_ids=a["head_ids"].astype(int) if "head_ids" in a.files else None
                residual=a["residual"].astype(np.float32).reshape(-1) if "residual" in a.files else None
            signature=(heads.shape,None if head_ids is None else tuple(head_ids),None if residual is None else residual.shape)
            if batch and (signature!=shape or len(batch)>=64):flush()
            shape=signature;batch.append((key,heads,head_ids,residual))
        flush()
    # Prediction data are written only for the stage requested by the caller.
    csvout(stage_dir/"predictions.csv.gz",[dict(id=k,method=name,score=v) for name,d in predictions.items() for k,v in d.items()])
    missing_by_method={name:len(tasks)-sum(k in d for k in tasks) for name,d in predictions.items()}
    return predictions,missing_by_method

def rollouts(cases,tasks,scores,predictions):
    lookup=defaultdict(dict)
    for key,t in tasks.items():lookup[t["case_id"],t.get("variant","original")][t["gid"]]=key
    pathrows=[];noderows=[]
    for name,pred in predictions.items():
        for c in cases:
            variant=c.get("variant","original");mapping=lookup[c["case_id"],variant]
            ns={gid:pred[k] for gid,k in mapping.items() if k in pred}
            result=execute(c,ns)
            first=result["first_error"]
            if first is not None:
                tk=mapping.get(first["gid"]);first["task_id"]=tk
                first["raw_score"]=float(scores[tk]["ell"]) if tk in scores else None
            pathrows.append(dict(method=name,split=c["split"],source_id=c["source_id"],case_id=c["case_id"],
                variant=variant,correct=result["correct"],first_error_gid=first["gid"] if first else None,
                first_error_order=first["order"] if first else None,first_error=first,path=result["path"],leaf=result["leaf"],error=result["error"]))
            for order,n in enumerate(c["gold_path"]):
                k=mapping.get(n["gid"]);value=pred.get(k)
                if value is None:continue
                noderows.append(dict(method=name,source_id=c["source_id"],case_id=c["case_id"],variant=variant,
                    gid=n["gid"],gold_order=order,label=n["label"],raw_score=scores[k]["ell"],score=value,
                    correct=int(value>0)==n["label"],task_id=k))
    return pathrows,noderows

def hierarchical(rr,field="correct"):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rr:
        if r.get(field) is not None:groups[r["source_id"]][r["case_id"]].append(float(r[field]))
    source={s:float(np.mean([np.mean(values) for values in cases.values()])) for s,cases in groups.items()}
    return source

def path_summaries(pathrows,metadata):
    summary=[];bysource=[]
    for name in sorted({r["method"] for r in pathrows}):
        method=[r for r in pathrows if r["method"]==name]
        for group in ("natural","A","all"):
            rr=[r for r in method if group=="all" or (r["variant"]=="original")== (group=="natural")]
            if not rr:continue
            sd=hierarchical(rr);available=[r for r in rr if r["correct"] is not None]
            summary.append(dict(method=name,group=group,source_macro_accuracy=float(np.mean(list(sd.values()))) if sd else None,
                case_variant_accuracy=float(np.mean([r["correct"] for r in available])) if available else None,
                observed_complete_accuracy=float(np.mean([r["correct"] for r in available])) if available else None,
                source_macro_lower_bound=float(np.mean(list(hierarchical([{**r,"correct":False if r["correct"] is None else r["correct"]} for r in rr]).values()))),
                source_macro_upper_bound=float(np.mean(list(hierarchical([{**r,"correct":True if r["correct"] is None else r["correct"]} for r in rr]).values()))),
                correct=sum(r["correct"] for r in available),n=len(rr),available=len(available),missing=len(rr)-len(available),
                sources=len(sd),parameters=metadata[name].get("parameters"),budget=metadata[name].get("budget"),
                family=metadata[name].get("family"),main=metadata[name].get("main",True)))
            if len(available)!=len(rr):
                summary[-1]["source_macro_accuracy"]=None
                summary[-1]["case_variant_accuracy"]=None
            for s,value in sd.items():
                sr=[r for r in rr if r["source_id"]==s and r["correct"] is not None]
                expected=sum(r["source_id"]==s for r in rr)
                bysource.append(dict(method=name,group=group,source_id=s,accuracy=value if len(sr)==expected else None,
                    observed_complete_accuracy=value,correct=sum(r["correct"] for r in sr),n=expected,available=len(sr),missing=expected-len(sr)))
    return summary,bysource

def source_weights(rr):
    counts=Counter(r["source_id"] for r in rr);n=len(counts)
    return np.array([1/(n*counts[r["source_id"]]) for r in rr]) if n else np.array([])
def pair_group(pp,tasks,scores,pred,kind,scope,source_id=None):
    valid=[]
    for p in pp:
        if p["kind"] not in kind:continue
        if source_id is not None and p["source_id"]!=source_id:continue
        if p["left"] not in pred or p["right"] not in pred:continue
        yl=p.get("left_label",tasks[p["left"]].get("label"));yr=p.get("right_label",tasks[p["right"]].get("label"))
        if scope=="gold_labelled" and (yl is None or yr is None):continue
        valid.append(dict(**p,yl=yl,yr=yr))
    if not valid:return None,[]
    w=source_weights(valid);l=[p["left"] for p in valid];r=[p["right"] for p in valid]
    raw=np.array([scores[b]["ell"]-scores[a]["ell"] for a,b in zip(l,r)],float)
    new=np.array([pred[b]-pred[a] for a,b in zip(l,r)],float);dc=new-raw
    weighted=lambda a:float(np.sum(w*a))
    var=lambda a:weighted((a-weighted(a))**2)
    mm=dict(n=len(valid),sources=len({p["source_id"] for p in valid}),raw_mean=weighted(raw),corrected_mean=weighted(new),raw_second_moment=weighted(raw**2),corrected_second_moment=weighted(new**2),raw_rms=np.sqrt(weighted(raw**2)),
        corrected_rms=np.sqrt(weighted(new**2)),correction_rms=np.sqrt(weighted(dc**2)),
        weighted="source_equal_then_pair_equal",case_weighted_correction_rms=float(np.sqrt(np.mean(dc**2))))
    if scope=="gold_labelled":
        paired=np.array([int(pred[p["left"]]>0)==p["yl"] and int(pred[p["right"]]>0)==p["yr"] for p in valid],float)
        raw_paired=np.array([int(scores[p["left"]]["ell"]>0)==p["yl"] and int(scores[p["right"]]["ell"]>0)==p["yr"] for p in valid],float)
        mm.update(paired_accuracy=weighted(paired),raw_paired_accuracy=weighted(raw_paired),
            case_weighted_paired_accuracy=float(np.mean(paired)),labelled_pairs=len(valid))
    if set(kind)<=set(("A","P","F0")):
        mm["VR"]=1-var(new)/var(raw) if var(raw)>0 else None
        mm["case_weighted_VR"]=1-float(np.var(new))/float(np.var(raw)) if np.var(raw)>0 else None
    else:
        direction=np.array([p["yr"]-p["yl"] for p in valid],float)
        signed=raw*direction;nsigned=new*direction
        mm.update(raw_signed_response_mean=weighted(signed),corrected_signed_response_mean=weighted(nsigned),ER=weighted(nsigned)/weighted(signed) if weighted(signed)!=0 else None,
            direction_degraded=int(np.sum((signed>0)&(nsigned<=0))),
            direction_improved=int(np.sum((signed<=0)&(nsigned>0))))
    details=[dict(pair_id=p["pair_id"],source_id=p["source_id"],kind=p["kind"],left=p["left"],right=p["right"],
        left_label=p["yl"],right_label=p["yr"],raw_difference=raw[i],corrected_difference=new[i],correction_difference=dc[i],
        corrected_left=pred[p["left"]],corrected_right=pred[p["right"]]) for i,p in enumerate(valid)]
    return mm,details

def pair_summaries(pairs,tasks,scores,predictions,metadata):
    summary=[];bysource=[];details=[]
    for name,pred in predictions.items():
        for group,kinds in (("A",("A",)),("P",("P",)),("AP",("A","P")),("CF",("C","F")),("F0",("F0",))):
            for scope in ("all_scored","gold_labelled"):
                mm,dd=pair_group(pairs,tasks,scores,pred,kinds,scope)
                if mm is None:continue
                budget=metadata[name].get("evidence_budget_absolute",metadata[name].get("delta"))
                if group=="CF" and budget is not None:
                    mm["exceeds_fit_budget"]=sum(abs(d["correction_difference"])>budget+1e-7 for d in dd)
                    mm["fit_budget_absolute"]=budget
                summary.append(dict(method=name,group=group,scope=scope,budget=metadata[name].get("budget"),**mm))
                if scope=="gold_labelled":
                    details.extend(dict(method=name,**d) for d in dd)
                    for sid in sorted({p["source_id"] for p in pairs if p["kind"] in kinds}):
                        sm,_=pair_group(pairs,tasks,scores,pred,kinds,scope,sid)
                        if sm:bysource.append(dict(method=name,group=group,source_id=sid,**sm))
    return summary,bysource,details

def select(summary,metadata,missing_by_method):
    stats={(r["method"],r["group"]):r for r in summary}
    raw=stats["raw","natural"]["source_macro_accuracy"];candidates=[]
    for name,m in metadata.items():
        if not m.get("main",True) or m.get("family") not in ("baseline","trace"):continue
        natural=stats.get((name,"natural"),{});perturb=stats.get((name,"A"),{})
        available=not missing_by_method.get(name,1) and natural.get("source_macro_accuracy") is not None and perturb.get("source_macro_accuracy") is not None
        passes=available and natural["source_macro_accuracy"]>=raw-.005-1e-12
        candidates.append(dict(name=name,family=m["family"],parameters=m.get("parameters",1e12),
            budget=m.get("budget"),natural=natural.get("source_macro_accuracy"),
            A=perturb.get("source_macro_accuracy"),passes_natural_gate=passes,complete=available))
    def choose(family):
        allc=[c for c in candidates if c["family"]==family and c["complete"]]
        eligible=[c for c in allc if c["passes_natural_gate"]]
        pool=eligible or allc
        if not pool:raise RuntimeError("No complete development candidates for "+family)
        def key(c):
            b=c["budget"]
            return (-c["A"],float(c["parameters"] or 0),float(b) if b is not None else float("inf"),c["name"])
        best=sorted(pool,key=key)[0]
        return dict(**best,status="passed" if eligible else "natural_gate_failed_kept_for_negative_result")
    return dict(protocol="Development only. Source macro natural decline <=0.005 vs raw; maximize source macro A path accuracy; tie fewer parameters then smaller budget then name. Seeds not selected.",
        raw_natural=raw,baseline=choose("baseline"),trace=choose("trace"),candidates=candidates)

def bootstrap(pathrows,baseline,trace,seed=SEED,nboot=BOOTSTRAPS):
    out={}
    for group in ("natural","A","all"):
        rr=[r for r in pathrows if group=="all" or (r["variant"]=="original")==(group=="natural")]
        if any(r["correct"] is None for r in rr if r["method"] in (baseline,trace)):
            out[group]=dict(status="not_evaluated_missing_scores",expected=len([r for r in rr if r["method"]==baseline]))
            continue
        aa=hierarchical([r for r in rr if r["method"]==baseline]);bb=hierarchical([r for r in rr if r["method"]==trace])
        ss=sorted(set(aa)&set(bb))
        if not ss:continue
        a=np.array([aa[s] for s in ss]);b=np.array([bb[s] for s in ss]);rng=np.random.default_rng(seed)
        draw=rng.integers(0,len(ss),size=(nboot,len(ss)))
        vals=(b-a)[draw].mean(1);av=a[draw].mean(1);bv=b[draw].mean(1)
        amap={(r["source_id"],r["case_id"],r["variant"]):r for r in rr if r["method"]==baseline}
        bmap={(r["source_id"],r["case_id"],r["variant"]):r for r in rr if r["method"]==trace}
        keys=[k for k in amap.keys()&bmap.keys() if amap[k]["correct"] is not None and bmap[k]["correct"] is not None]
        fixed=sum(not amap[k]["correct"] and bmap[k]["correct"] for k in keys)
        broken=sum(amap[k]["correct"] and not bmap[k]["correct"] for k in keys)
        out[group]=dict(sources=len(ss),baseline_accuracy=float(a.mean()),trace_accuracy=float(b.mean()),difference=float((b-a).mean()),
            difference_ci95=np.quantile(vals,[.025,.975]).tolist(),baseline_ci95=np.quantile(av,[.025,.975]).tolist(),
            trace_ci95=np.quantile(bv,[.025,.975]).tolist(),fixed=int(fixed),new_errors=int(broken),matched_case_variants=len(keys),
            source_effects=[dict(source_id=s,baseline=aa[s],trace=bb[s],difference=bb[s]-aa[s]) for s in ss])
    return dict(baseline=baseline,trace=trace,seed=seed,replicates=nboot,unit="paired source resampling",metrics=out)

def pair_bootstrap(pairsource,baseline,trace):
    output={}
    for group in ("A","AP","CF"):
        aa={r["source_id"]:r for r in pairsource if r["method"]==baseline and r["group"]==group}
        bb={r["source_id"]:r for r in pairsource if r["method"]==trace and r["group"]==group}
        ss=sorted(aa.keys()&bb.keys())
        if not ss:continue
        rng=np.random.default_rng(SEED);draw=rng.integers(0,len(ss),size=(BOOTSTRAPS,len(ss)))
        def values(mapping):
            collect=lambda field:np.array([mapping[s][field] for s in ss])
            raw_mean=collect("raw_mean");new_mean=collect("corrected_mean")
            raw_2=collect("raw_second_moment");new_2=collect("corrected_second_moment")
            rms=collect("correction_rms")**2
            result={"correction_rms":(float(np.sqrt(np.mean(rms))),np.sqrt(rms[draw].mean(1)))}
            if group=="CF":
                a=collect("raw_signed_response_mean");b=collect("corrected_signed_response_mean")
                result["ER"]=(float(b.mean()/a.mean()),b[draw].mean(1)/a[draw].mean(1))
            else:
                rv=raw_2.mean()-raw_mean.mean()**2;nv=new_2.mean()-new_mean.mean()**2
                rb=raw_2[draw].mean(1)-raw_mean[draw].mean(1)**2
                nb=new_2[draw].mean(1)-new_mean[draw].mean(1)**2
                result["VR"]=(float(1-nv/rv),1-nb/rb)
            return result
        av=values(aa);bv=values(bb);output[group]={}
        for metric in av:
            a,ad=av[metric];b,bd=bv[metric];valid=np.isfinite(ad)&np.isfinite(bd)
            output[group][metric]=dict(sources=len(ss),baseline=a,trace=b,difference=b-a,
                difference_ci95=np.quantile((bd-ad)[valid],[.025,.975]).tolist(),
                baseline_ci95=np.quantile(ad[valid],[.025,.975]).tolist(),trace_ci95=np.quantile(bd[valid],[.025,.975]).tolist(),
                valid_replicates=int(valid.sum()))
    return dict(seed=SEED,replicates=BOOTSTRAPS,metrics=output,scope="Gold-labelled pairs, common sources, same source bootstrap draws for both methods")

def diagnose_errors(pathrows,cases,tasks,predictions):
    gold={(c["case_id"],n["gid"]):n["label"] for c in cases if c.get("variant","original")=="original" for n in c["gold_path"]}
    tk={(t["case_id"],t.get("variant","original"),t["gid"]):key for key,t in tasks.items()}
    variants=defaultdict(set)
    for c in cases:variants[c["case_id"]].add(c.get("variant","original"))
    details=[]
    for r in pathrows:
        if r["method"]!="raw" or r["variant"]!="original" or r["correct"] is not False:continue
        gid=r["first_error_gid"];label=gold.get((r["case_id"],gid));observed=[]
        for variant in sorted(variants[r["case_id"]]):
            if variant=="original":continue
            key=tk.get((r["case_id"],variant,gid))
            if key in predictions["raw"]:
                value=predictions["raw"][key]
                observed.append(dict(variant=variant,score=value,correct=int(value>0)==label))
        details.append(dict(source_id=r["source_id"],case_id=r["case_id"],gid=gid,label=label,
            raw_score=r["first_error"].get("raw_score") if r["first_error"] else None,
            observed_A_variants=len(observed),any_A_variant_correct=any(x["correct"] for x in observed),
            all_observed_A_variants_wrong=bool(observed) and not any(x["correct"] for x in observed),
            variants=observed))
    return dict(raw_natural_error_paths=len(details),A_covered_first_errors=sum(r["observed_A_variants"]>0 for r in details),
        first_error_with_correct_A_variant=sum(r["any_A_variant_correct"] for r in details),
        all_observed_A_variants_wrong=sum(r["all_observed_A_variants_wrong"] for r in details),
        interpretation="Each endpoint uses the independently supplied reference label. A correct edited endpoint is an observed controlled sensitivity, not proof that the full endpoint score is a target or a validated causal clinical diagnosis.",
        first_error_source_nodes=[dict(source_id=s,gid=g,n=n) for (s,g),n in Counter((r["source_id"],r["gid"]) for r in details).most_common()],
        details=details)

def transition_rows(pathrows,baseline,trace):
    aa={(r["case_id"],r["variant"]):r for r in pathrows if r["method"]==baseline}
    bb={(r["case_id"],r["variant"]):r for r in pathrows if r["method"]==trace}
    return [dict(case_id=k[0],variant=k[1],source_id=aa[k]["source_id"],baseline_correct=aa[k]["correct"],
        trace_correct=bb[k]["correct"],fixed=aa[k]["correct"] is False and bb[k]["correct"] is True,
        new_error=aa[k]["correct"] is True and bb[k]["correct"] is False,
        baseline_first_error=aa[k]["first_error"],trace_first_error=bb[k]["first_error"]) for k in sorted(aa.keys()&bb.keys())]

def evaluate(args):
    started=time.time();args.training_dir=Path(args.training_dir);args.cache=Path(args.cache);args.out=Path(args.out)
    stage_dir=args.out/args.stage;stage_dir.mkdir(parents=True,exist_ok=True)
    freeze=args.out/"frozen.json"
    frozen=load_json(freeze)
    if args.stage=="final" and not frozen.get("frozen"):raise RuntimeError("Freeze development selection before any final evaluation")
    tasklist=rows(args.cache/"data"/("tasks_"+args.stage+".jsonl"))+rows(args.cache/"data"/("tasks_diagnostic_"+args.stage+".jsonl"));tasks={t["id"]:t for t in tasklist}
    cases=rows(args.cache/"data"/("cases_"+args.stage+".jsonl"))
    pairlist=rows(args.cache/"data"/("pairs_"+args.stage+".jsonl"))+rows(args.cache/"data"/("pairs_diagnostic_"+args.stage+".jsonl"))
    pairs=list({p["pair_id"]:p for p in pairlist}.values())
    modeldata=load_json(args.training_dir/"models.json");mlist=modeldata if isinstance(modeldata,list) else modeldata.get("models",[])
    metadata={m["name"]:m for m in mlist}
    metadata.setdefault("raw",dict(name="raw",family="baseline",architecture="original model",parameters=0,main=True,budget=None))
    if args.stage=="final":
        # Parameters and candidate names were fixed on development, never selected here.
        old={m["name"]:m for m in frozen["models"]}
        if set(old)!=set(metadata):raise RuntimeError("Model manifest names changed after final freeze")
        metadata=old
    scores=read_scores(args.cache,args.stage,tasks)
    predictions,missing=predict(args,tasks,scores,metadata,stage_dir)
    pathrows,noderows=rollouts(cases,tasks,scores,predictions)
    summary,bysource=path_summaries(pathrows,metadata)
    pairsum,pairsource,pairdetails=pair_summaries(pairs,tasks,scores,predictions,metadata)
    csvout(stage_dir/"path_results.csv.gz",pathrows);csvout(stage_dir/"gold_node_results.csv.gz",noderows)
    csvout(stage_dir/"path_summary.csv",summary);csvout(stage_dir/"source_path_summary.csv",bysource)
    csvout(stage_dir/"pair_summary.csv",pairsum);csvout(stage_dir/"source_pair_summary.csv",pairsource)
    csvout(stage_dir/"pair_predictions.csv.gz",pairdetails)
    coverage=dict(stage=args.stage,tasks=len(tasks),scores=len(scores),cases=len(cases),sources=len({c["source_id"] for c in cases}),
        case_variants=dict(Counter(c.get("variant","original") for c in cases)),pairs=dict(Counter(p["kind"] for p in pairs)),
        missing_predictions=missing,model_count=len(metadata),provenance="New real Qwen3-8B cache and frozen-node-function evaluation; no synthetic outcomes")
    if args.stage=="dev":
        selection=select(summary,metadata,missing)
        dump(args.out/"dev_selection.json",selection)
        frozen=dict(frozen=True,created_at=time.strftime("%Y-%m-%d %H:%M:%S"),selection=selection,
            models=list(metadata.values()),training_dir=str(args.training_dir),final_scores_used=False,
            source_disjoint_protocol="Data manifest assigns whole guideline sources to fit/dev/final. Final scoring starts only after this file exists.",
            metrics="Full-tree autonomous rollout for each case variant. Within source average variants then cases; final metric averages sources.",
            final_evaluation_seed=SEED,bootstrap_replicates=BOOTSTRAPS)
        dump(freeze,frozen)
    else:selection=frozen["selection"]
    contrast=bootstrap(pathrows,selection["baseline"]["name"],selection["trace"]["name"])
    dump(stage_dir/"contrast.json",contrast)
    dump(stage_dir/"pair_contrast.json",pair_bootstrap(pairsource,contrast["baseline"],contrast["trace"]))
    dump(stage_dir/"error_diagnosis.json",diagnose_errors(pathrows,cases,tasks,predictions))
    csvout(stage_dir/"selected_transitions.csv",transition_rows(pathrows,contrast["baseline"],contrast["trace"]))
    coverage["seconds"]=time.time()-started;dump(stage_dir/"coverage.json",coverage)
    print(json.dumps(dict(stage=args.stage,coverage=coverage,selection={k:selection[k] for k in ("baseline","trace")},contrast=contrast["metrics"]),ensure_ascii=True))
    return args

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["dev","final"],required=True);p.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--training-dir",type=Path,default=HERE/"training");p.add_argument("--out",type=Path,default=HERE/"evaluation")
    p.add_argument("--predictions",type=Path);args=p.parse_args();evaluate(args)
if __name__=="__main__":main()

