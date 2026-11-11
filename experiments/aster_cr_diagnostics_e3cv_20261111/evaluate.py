"""CPU ablations and pooled out-of-fold source bootstrap; original decoders and paths."""
import argparse, itertools, time, traceback
from collections import defaultdict
import numpy as np
from .common import *
from .score_variant import fit_intercepts,parameters,seed_existing_scores,scores_for
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev
uid=ev.uid

def comparisons(names):
    c=[(n,"raw") for n in names if n!="raw"]
    for g in GROUPS:
        c.extend([(g+"_full","independent_intercept"),(g+"_full",g+"_pair_only")])
    c.extend([("E3_full","E2_full"),("E3_full","E1_full"),("E2_full","E1_full")])
    return [(a,b) for a,b in c if a in names and b in names]

def transition_rows(scope,node_records,pairs,paths,questions,names):
    datasets={}
    datasets["node_all"]={n:{r["view_id"]:(r["predictions"][n]==r["label"] if r["label"] is not None else None)
        for r in node_records if r["in_evaluation_views"]} for n in names}
    datasets["node_natural"]={n:{r["view_id"]:(r["predictions"][n]==r["label"] if r["label"] is not None else None)
        for r in node_records if r["natural"]} for n in names}
    for group in sorted({r["group"] for r in pairs}):
        for field in ("paired_correct","relation_correct"):
            datasets["pair_"+group+"_"+field]={n:{r["pair_id"]:r[field] for r in pairs
                if r["method"]==n and r["group"]==group} for n in names}
    for interference in (False,True):
        datasets["path_interference" if interference else "path_natural"]={
            n:{r["case_id"]+"|"+r["variant"]:r["correct"] for r in paths
            if r["method"]==n and r["interference"]==interference} for n in names}
    if questions:
        datasets["question_natural"]={n:{r["item_id"]:r["correct"] for r in questions if
            r["method"]==n and r["variant"] in ("full","original")} for n in names}
    output=[]
    for level,by_method in datasets.items():
        for lhs,rhs in comparisons(names):
            old,new=by_method[rhs],by_method[lhs]
            valid=[k for k in old.keys()&new.keys() if old[k] is not None and new[k] is not None]
            if not old and not new:continue
            output.append(dict(scope=scope,comparison=lhs+" minus "+rhs,level=level,
                total=len(old.keys()&new.keys()),valid_denominator=len(valid),
                wrong_to_right=sum(not old[k] and new[k] for k in valid),
                right_to_wrong=sum(old[k] and not new[k] for k in valid),
                kept_right=sum(old[k] and new[k] for k in valid),
                kept_wrong=sum(not old[k] and not new[k] for k in valid)))
    return output

def summarize(scope,nodes,pairs,paths,questions,native=None,folds=None):
    target=OUT/"evaluation"/scope;target.mkdir(parents=True,exist_ok=True)
    names=list(nodes[0]["scores"]);sources=sorted({r["source_id"] for r in nodes})
    boot=ev.SourceBootstrap(sources);metrics={};tables=[];source_rows=[]
    bypair=defaultdict(list);bypath=defaultdict(list);byq=defaultdict(list)
    for r in pairs:bypair[r["method"]].append(r)
    for r in paths:bypath[r["method"]].append(r)
    for r in questions:byq[r["method"]].append(r)
    natural=[r for r in nodes if r["natural"]];labeled=[r for r in natural if r["label"] is not None]
    for name in names:
        mm,cc,ss=ev.pair_statistics(bypair[name],boot,native=bool(native))
        if natural:
            correct=[r["predictions"][name]==r["label"] for r in labeled]
            mm["natural_node_accuracy"]=boot.mean(correct,[r["source_id"] for r in labeled])
            cc.update(natural_node_count=len(labeled),natural_node_correct=sum(correct),
                natural_node_total_count=len(natural),natural_node_unlabeled_count=len(natural)-len(labeled))
        for interfere in (False,True):
            rr=[r for r in bypath[name] if r["interference"]==interfere]
            if rr:
                key="interference_path" if interfere else "natural_path"
                mm[key+"_accuracy"]=boot.mean([r["correct"] for r in rr],[r["source_id"] for r in rr])
                cc[key+"_count"]=len(rr);cc[key+"_correct"]=sum(r["correct"] for r in rr)
        if native=="medpic":
            qq=[r for r in byq[name] if r["variant"] in ("full","original")]
            mm["natural_native_accuracy"]=boot.mean([r["correct"] for r in qq],[r["source_id"] for r in qq])
            cc.update(natural_native_count=len(qq),natural_native_correct=sum(r["correct"] for r in qq))
            mm["natural_selected_options"]=boot.mean([len(r["predicted"]) for r in qq],[r["source_id"] for r in qq])
            mm["natural_none_selected_rate"]=boot.mean([r["none_selected"] for r in qq],[r["source_id"] for r in qq])
        elif native=="nli4ct":
            mm["natural_native_accuracy"]=mm["natural_node_accuracy"]
            cc.update(natural_native_count=cc["natural_node_count"],natural_native_correct=cc["natural_node_correct"])
        metrics[name]=mm
        tables.append(dict(scope=scope,method=name,**cc,**{k:ev.finite(v[0]) for k,v in mm.items() if k not in cc}))
        source_rows.extend(dict(scope=scope,**r) for r in ss)
    bootstrap=[]
    for name,mm in metrics.items():
        for metric,vals in mm.items():
            bootstrap.append(dict(scope=scope,comparison=name,metric=metric,**boot.summary(vals),sources=len(sources),draws=2000))
    for lhs,rhs in comparisons(names):
        for metric in sorted(metrics[lhs].keys()&metrics[rhs].keys()):
            bootstrap.append(dict(scope=scope,comparison=lhs+" minus "+rhs,metric=metric,
                **boot.summary(metrics[lhs][metric]-metrics[rhs][metric]),sources=len(sources),draws=2000))
    transitions=transition_rows(scope,nodes,pairs,paths,questions,names)
    ev.write_csv(target/"metrics.csv",tables);ev.write_csv(target/"source_bootstrap.csv",bootstrap)
    ev.write_csv(target/"source_metrics.csv",source_rows);ev.write_csv(target/"transitions.csv",transitions)
    result=dict(scope=scope,experiment_version=VERSION,models={r["method"]:r for r in tables},
        source_count=len(sources),sources=sources,bootstrap_draws=2000,
        pooled_folds=folds,estimator="Recompute pooled metric for each source resample, never average fold ratios",
        native_task=native,history="Previously used data",complete=True)
    write(target/"metrics.json",result)
    return result

def scope_data(scope,fold=None):
    rd=lambda n:ev.data_rows(BASE,n)
    if fold:
        valid=set(fold["validation_sources"])
        pv=[v for v in rd("path_features.jsonl")+rd("path_interference_views.jsonl")
            if v.get("split")=="fit" and v["source_id"] in valid]
        cases=[c for c in ev.case_rows(ev.HIST/"data/path_cases.jsonl")
            if c["split"]=="fit" and c["source_id"] in valid]
        if scope=="common":
            vv=[dict(v,original=False) for v in rd("E2_views.jsonl") if v["source_id"] in valid]
            vv += [v for v in rd("natural_views.jsonl") if v["source_id"] in valid]
            pp=[p for p in rd("E2_pairs.jsonl") if p["source_id"] in valid]
        else:
            vv=[v for v in rd("historical_views.jsonl") if v["split"]=="fit" and v["source_id"] in valid]
            pp=[p for p in rd("historical_pairs.jsonl") if p["split"]=="fit" and p["source_id"] in valid]
            pv=[];cases=[]
        return vv,pp,pv,cases,None
    if scope in ("dev","test"):
        vv=[v for v in rd("historical_views.jsonl") if v["split"]==scope]
        pp=[p for p in rd("historical_pairs.jsonl") if p["split"]==scope]
        pv=[v for v in rd("path_features.jsonl")+rd("path_interference_views.jsonl") if v["split"]==scope]
        cases=[c for c in ev.case_rows(ev.HIST/"data/path_cases.jsonl") if c["split"]==scope]
        return vv,pp,pv,cases,None
    if scope=="new60":
        vv=rd("new60_views.jsonl");pp=rd("new60_pairs.jsonl")
        cases=[c for c in ev.case_rows(ev.TRACE/"data/cases_final.jsonl") if c.get("variant")=="original"]
        return vv,pp,vv,cases,None
    return ([v for v in rd(scope+"_views.jsonl") if v["split"]=="test"],
            [p for p in rd(scope+"_pairs.jsonl") if p["split"]=="test"],[],[],scope)

def verify_previous(scope,models,nodes,paths,questions,fold,pool):
    mapping={g:scope if fold=="full" else "cv_"+g+"_"+fold+"_"+("common_frozen_pool" if pool=="common" else "historical_pool")
             for g in models if fold=="full" or g!="E3"}
    node_by={r["view_id"]:r for r in nodes};path_by={(r["method"],r["case_id"],r["variant"]):r for r in paths}
    qby={(r["method"],r["item_id"],r["variant"]):r for r in questions}
    maxerr=0.;matched=0;pc=0;qc=0
    for g,old_scope in mapping.items():
        for r in rows(BASE/"evaluation"/old_scope/"predictions.jsonl"):
            now=node_by[r["view_id"]];err=abs(now["scores"][g+"_full"]-r["scores"][g]);maxerr=max(maxerr,err)
            assert err<1e-8 and now["predictions"][g+"_full"]==r["predictions"][g],(scope,g,"node replay")
            matched+=1
        for r in rows(BASE/"evaluation"/old_scope/"path_predictions.jsonl"):
            if r["method"]!=g:continue
            now=path_by[(g+"_full",r["case_id"],r["variant"])]
            assert now["path"]==r["path"] and now["leaf"]==r["leaf"] and now["correct"]==r["correct"],(scope,g,"path replay")
            pc+=1
        for r in rows(BASE/"evaluation"/old_scope/"question_predictions.jsonl"):
            if r["method"]!=g:continue
            now=qby[(g+"_full",r["item_id"],r["variant"])]
            assert now["predicted"]==r["predicted"],(scope,g,"decoder replay")
            qc+=1
    return dict(full_score_max_difference=maxerr,matched_nodes=matched,matched_paths=pc,matched_questions=qc)

def evaluate_scope(scope,store,models,intercept,data,fold="full",pool=None):
    target=OUT/"evaluation"/scope
    marker=read(target/"completion.json")
    if (marker.get("complete") and marker.get("experiment_version")==VERSION
        and marker.get("fold")==fold and marker.get("group")==list(GROUPS)
        and marker.get("intercept_fit_sources")==intercept["fit_sources"]):return
    started=time.time();vv,pp,pv,cases,native=data
    vv=list({uid(v):v for v in vv}.values());pv=list({uid(v):v for v in pv}.values())
    allviews=list({uid(v):v for v in [*pv,*vv]}.values());vkeys={uid(v) for v in vv}
    old_scopes=[scope] if fold=="full" else ["cv_"+g+"_"+fold+"_"+("common_frozen_pool" if pool=="common" else "historical_pool") for g in GROUPS if g!="E3"]
    seed_existing_scores(models,old_scopes)
    scores,descriptors=scores_for(store,allviews,models,intercept)
    decisions,questions=ev.decisions_for(vv,scores,native)
    pair_records=ev.observations(vv,pp,scores,decisions,native=native or False)
    path_records=ev.path_predictions(cases,pv,scores) if cases else []
    nodes=[]
    for v in allviews:
        natural=uid(v) in vkeys and v.get("kind","original")=="original" and (v.get("original") or
            ("original" not in v and v.get("variant","original") in ("full","original")))
        key=uid(v)
        nodes.append(dict(view_id=key,source_id=v["source_id"],label=v.get("label"),label_available=v.get("label") is not None,
            natural=bool(natural),in_evaluation_views=key in vkeys,case_id=v.get("case_id"),gid=v.get("gid"),
            variant=v.get("variant"),fold=fold,
            scores={n:ss[key] for n,ss in scores.items()},predictions={n:dd[key] for n,dd in decisions.items()}))
    if native=="medpic":
        items={r["id"]:r for r in rows(ev.TARGET/"medpic/items.jsonl")}
        for q in questions:
            item=items[q["item_id"]];none=[k for k,v in item["options"].items() if "none of" in v.lower()]
            q.update(option_count=len(item["options"]),selected_option_count=len(q["predicted"]),
                none_selected=bool(set(none)&set(q["predicted"])),none_options=none)
    verify=verify_previous(scope,models,nodes,path_records,questions,fold,pool)
    bypair={(r["method"],r["pair_id"]):r for r in pair_records}
    maxdiff=0.
    for r in pair_records:
        name=r["method"];reference=None
        if name=="independent_intercept" or name.endswith("_original_intercept_only"):reference="raw"
        elif name.endswith("_full"):reference=name[:-5]+"_pair_only"
        if reference:
            maxdiff=max(maxdiff,abs(r["difference"]-bypair[(reference,r["pair_id"])]["difference"]))
    assert maxdiff<1e-10
    verify.update(intercept_pair_difference_max_error=maxdiff,
        unavailable_path_count=sum(bool(r["error"]) for r in path_records),
        unknown_label_count=sum(r["label"] is None for r in nodes),
        autonomous_paths=True,all_available_branch_nodes_scored=True)
    assert verify["unavailable_path_count"]==0
    target.mkdir(parents=True,exist_ok=True)
    for name,records in [("predictions",nodes),("pair_predictions",pair_records),("path_predictions",path_records),("question_predictions",questions)]:
        lines(target/(name+".jsonl"),records)
    keys=list(scores["raw"])
    np.savez_compressed(target/"scores_by_variant.npz",view_ids=np.array(keys),
        **{n:np.array([ss[k] for k in keys],dtype=np.float64) for n,ss in scores.items()})
    write(target/"score_manifest.json",dict(experiment_version=VERSION,variants=descriptors,fold=fold,
        intercept_fit_sources=intercept["fit_sources"],score_fields=list(scores)))
    result=summarize(scope,nodes,pair_records,path_records,questions,native=native)
    # Metrics affected only by score differences must be invariant to intercept-only changes.
    for lhs,rhs in [("independent_intercept","raw")]+[(g+"_original_intercept_only","raw") for g in GROUPS]+[(g+"_full",g+"_pair_only") for g in GROUPS]:
        for metric in ("ap_score_change_rms","ap_vr","cf_correction_rms","cf_er","cf_direction_degradation_count","cf_over_budget_rate"):
            a,b=result["models"][lhs].get(metric),result["models"][rhs].get(metric)
            if a is not None and b is not None:assert abs(a-b)<1e-9,(scope,lhs,rhs,metric,a,b)
    write(target/"verification.json",verify)
    write(target/"completion.json",dict(complete=True,experiment_version=VERSION,scope=scope,
        group=list(GROUPS),fold=fold,intercept_fit_sources=intercept["fit_sources"],
        score_variants=list(scores),seconds=time.time()-started))

def pool_cv(pool):
    scope="cv_pooled_"+pool;fold_names=["cv_fold"+str(k)+"_"+pool for k in range(1,6)]
    records={}
    for name in ("predictions","pair_predictions","path_predictions","question_predictions"):
        records[name]=[r for f in fold_names for r in rows(OUT/"evaluation"/f/(name+".jsonl"))]
    sources={r["source_id"] for r in records["predictions"]}
    assert len(sources)==80
    seen={}
    for r in records["predictions"]:
        previous=seen.setdefault(r["source_id"],r["fold"])
        assert previous==r["fold"]
    target=OUT/"evaluation"/scope;target.mkdir(parents=True,exist_ok=True)
    summarize(scope,records["predictions"],records["pair_predictions"],records["path_predictions"],
        records["question_predictions"],folds=fold_names)
    # Per-sample OOF predictions remain in five fold files without duplicate storage.
    write(target/"prediction_references.json",dict(folds=fold_names,sources=80,one_fold_per_source=True))
    write(target/"completion.json",dict(complete=True,experiment_version=VERSION,scope=scope,group=list(GROUPS),
        fold="pooled_out_of_fold",folds=fold_names,intercept_fit_sources_by_fold={
            k:v["fit_sources"] for k,v in read(OUT/"independent_intercepts.json").items() if k!="full"}))

def main(mode):
    store=Store(BASE)
    if mode=="intercepts":
        fit_intercepts(store);return
    intercepts=read(OUT/"independent_intercepts.json")
    if mode=="full":
        models=parameters()
        for scope in ("dev","test","new60","nli4ct","medpic"):
            evaluate_scope(scope,store,models,intercepts["full"],scope_data(scope))
            write(OUT/"status/evaluate_full.json",dict(state="running",last_scope=scope))
    elif mode=="cv":
        for f in read(BASE/"folds.json")["folds"]:
            fold="fold"+str(f["fold"]);models=parameters(fold)
            for g,m in models.items():
                train=set(m["metadata"].get("training_sources",m["metadata"].get("train_sources",[])))
                assert train==set(f["train_sources"]) and not train&set(f["validation_sources"])
            for pool in ("common","historical"):
                evaluate_scope("cv_"+fold+"_"+pool,store,models,intercepts[fold],scope_data(pool,f),fold,pool)
            write(OUT/"status/evaluate_cv.json",dict(state="running",last_fold=fold))
        for pool in ("common","historical"):pool_cv(pool)
    write(OUT/"status"/("evaluate_"+mode+".json"),dict(state="complete"))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["intercepts","full","cv"]);a=p.parse_args()
    try:main(a.mode)
    except Exception as e:
        write(OUT/"status"/("failed_evaluate_"+a.mode+".json"),dict(error=str(e),traceback=traceback.format_exc()))
        raise
