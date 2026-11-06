"""Recompute the consolidated ASTER tables from published scores. CPU only."""
import argparse,csv,gzip,json,math
from collections import defaultdict,Counter
from pathlib import Path
from .evaluation import evaluate,pred,path_metrics,source_bootstrap
from .existing_report import interval,metric_row

METHODS=("full","aster","soft_angle","global_mean","same_fact","direct_reference","calibrated_full")
def read_json(path):
    return json.loads(Path(path).read_text())
def rows(path):
    p=Path(path)
    if not p.exists():p=Path(str(p)+".gz")
    if p.suffix==".gz":
        with gzip.open(p,"rt",encoding="utf-8") as f:return [json.loads(s) for s in f if s.strip()]
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def table(root,name,records):
    save(root/(name+".json"),records)
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (root/(name+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def corrections(items,method):
    full=[pred(r["full_score"])==r["gold"] for r in items]
    new=[pred(r[method+"_score"])==r["gold"] for r in items]
    return dict(inputs=len(items),correct=sum(new),baseline_correct=sum(full),
        corrected=sum(not a and b for a,b in zip(full,new)),new_errors=sum(a and not b for a,b in zip(full,new)))

def main(repo):
    old=repo/"results/aster_existing_data_20261106"
    soft=repo/"results/aster_soft_angle_20261106"
    anchor=repo/"results/aster_reference_anchor_20261106"
    out=repo/"results/aster_synthesis_20261106";out.mkdir(parents=True,exist_ok=True)
    main_inputs=rows(anchor/"results/main_inputs.jsonl")
    historical=rows(anchor/"results/historical_inputs.jsonl")
    views={v["view_id"]:v for v in rows(old/"data/views.jsonl")}
    pairs=rows(old/"data/pairs.jsonl")
    assert len(main_inputs)==len(views)==1290 and len(pairs)==1053 and len(historical)==569
    scores={m:{r["input_id"]:r[m+"_score"] for r in main_inputs} for m in METHODS}
    main_table=[];input_changes=[];evaluations={};pair_differences={}
    published=read_json(anchor/"results/main_effects.json")
    for split in ("fit","dev","test","all"):
        vs={k:dict(v,split="all") if split=="all" else v for k,v in views.items()}
        ps=[dict(p,split="all") if split=="all" else p for p in pairs]
        e=evaluate(vs,ps,scores,split=split,bootstrap=2000)
        evaluations[split]=e
        current=[r for r in main_inputs if split=="all" or r["split"]==split]
        for method,value in e["methods"].items():
            stored=published[split]["methods"][method]
            assert math.isclose(value["macro_PA"],stored["macro_PA"],abs_tol=1e-12)
            assert value["all_input_accuracy"]==stored["all_input_accuracy"]
            for kind in ("A","P","C","F"):
                assert value["PA_"+kind]["denominator"]==stored["PA_"+kind]["denominator"]
            for scope in ("all","reference_changed","reference_unchanged"):
                rs=[r for r in current if scope=="all" or r["reference_changed"]==(scope=="reference_changed")]
                counts=corrections(rs,method)
                input_changes.append(dict(split=split,scope=scope,method=method,
                    full_input_denominator=len(current),**counts))
                if scope=="all":
                    main_table.append({**metric_row(method,value),**counts,"split":split,
                        "full_input_denominator":len(current)})
        pair_differences[split]={}
        sv={k:v for k,v in vs.items() if v["split"]==split}
        sp=[p for p in ps if p["split"]==split]
        for reference in ("global_mean","direct_reference"):
            c=source_bootstrap(sv,sp,{"full":scores[reference],"same_fact":scores["same_fact"]})
            pair_differences[split][reference]=c["methods"]["same_fact"]["differences"]["full"]
    table(out,"main_comparison",main_table)
    table(out,"main_input_changes",input_changes)
    save(out/"main_effects.json",evaluations)
    save(out/"reference_comparison_intervals.json",pair_differences)
    history_table=[]
    for population in ("all","constructed","original"):
        pool=[r for r in historical if population=="all" or r["population"]==population]
        routes=["all"]+sorted({r["view"] for r in pool})
        for route in routes:
            current=[r for r in pool if route=="all" or r["view"]==route]
            for scope in ("all","reference_changed","reference_unchanged"):
                rs=[r for r in current if scope=="all" or r["reference_changed"]==(scope=="reference_changed")]
                for m in METHODS[:-1]:
                    history_table.append(dict(population=population,view=route,scope=scope,method=m,
                        full_input_denominator=len(current),**corrections(rs,m)))
    table(out,"historical_comparison",history_table)
    mechanism_records=rows(soft/"results/mechanism_all_pairs.jsonl")
    assert len(mechanism_records)==1785
    mechanisms=[]
    for split in ("dev","test"):
        for kind in ("A","P"):
            for component in ("full_head","nuisance_orthogonal","screened_orthogonal","oblique","soft_angle"):
                rs=[r for r in mechanism_records if r["split"]==split and r["kind"]==kind and r["component"]==component]
                eligible=[r for r in rs if r.get("recovery") is not None]
                te2=sum(r["te"]**2 for r in rs)
                numerator=sum(r["te"]*(r["ie_forward"]-r["ie_reverse"])/2 for r in rs)
                mechanisms.append(dict(split=split,kind=kind,component=component,total_pairs=len(rs),
                    normalized_pairs=len(eligible),mean_normalized_recovery=sum(r["recovery"] for r in eligible)/len(eligible),
                    all_pair_effect_regression=numerator/te2 if te2 else None,
                    recovery_ci95=interval([(r["source_id"],r["recovery"]) for r in eligible])))
    table(out,"mechanism_chain",mechanisms)
    h1=read_json(old/"results/h1.json")
    table(out,"m1_head_controls",[dict(split=split,**r) for split,rs in h1["interventions"].items() for r in rs])
    evidence=read_json(old/"analysis_postrun/mechanism_evidence.json")
    evidence["current_scope"]="Current closed-form version; signal/energy estimates refer to the fixed fitted basis"
    save(out/"basis_and_selection_evidence.json",{k:evidence[k] for k in ("current_scope","fit_nuisance_energy",
        "signal_basis_energy","ranks","algebra","selection","local_theory","calibration","ci_main_difference")})
    probe=read_json(old/"results/h3.json")["probe"]
    save(out/"signal_probe.json",{k:v for k,v in probe.items() if k not in ("coefficients","predictions")})
    path_records=[]
    for root in (old,soft):
        for p in sorted((root/"paths").glob("predictions_*.jsonl*")):
            path_records+=rows(p)
    keys=[(r["method"],r["case_id"]) for r in path_records]
    assert len(keys)==len(set(keys))
    cases={r["case_id"]:r for r in rows(old/"data/path_cases.jsonl")}
    baseline={r["case_id"]:r for r in path_records if r["method"]=="full"}
    path_table=[]
    for scope in ("all_432","original_100"):
        selected=[r for r in path_records if scope=="all_432" or cases[r["case_id"]]["original_100_subset"]]
        pm=path_metrics(selected,n_resamples=2000)
        for m,x in pm["methods"].items():
            rs=[r for r in selected if r["method"]==m]
            corrected=sum(not baseline[r["case_id"]]["correct"] and r["correct"] for r in rs)
            new_errors=sum(baseline[r["case_id"]]["correct"] and not r["correct"] for r in rs)
            available=[r for r in rs if r["status"]=="complete"]
            avail_full=sum(baseline[r["case_id"]]["correct"] for r in available)
            path_table.append(dict(scope=scope,method=m,**x,corrected=corrected,new_errors=new_errors,
                unavailable_inputs=sum(r["status"]=="unavailable_input" for r in rs),
                technical_failures=sum(r["status"]=="failed" for r in rs),
                available_subset_n=len(available),available_subset_full_correct=avail_full,
                available_subset_method_correct=sum(r["correct"] for r in available),
                delta_ci95=interval([(r["source_id"],int(r["correct"])-int(baseline[r["case_id"]]["correct"])) for r in rs])))
    table(out,"autonomous_paths",path_table)
    old_methods=read_json(repo/"docs/ASTER/results/single-turn.json")
    table(out,"historical_learned_path_results",[dict(method=m,n=x["registered"],correct=x["exact_path"],
        accuracy=x["exact_path"]/x["registered"],version="learned_representation_layer18_epoch3",
        sources=old_methods["source_clusters"],technical_failures=x["technical_failures"]) for m,x in old_methods["methods"].items()])
    save(out/"historical_learned_pairwise.json",old_methods["paired_pathacc"])
    agreements=[]
    for split in ("test","all"):
        rs=[r for r in main_inputs if split=="all" or r["split"]==split]
        for left,right in (("aster","calibrated_full"),("same_fact","direct_reference"),("aster","soft_angle")):
            agreements.append(dict(split=split,left=left,right=right,inputs=len(rs),
                prediction_disagreements=sum(pred(r[left+"_score"])!=pred(r[right+"_score"]) for r in rs),
                score_disagreements=sum(r[left+"_score"]!=r[right+"_score"] for r in rs)))
    table(out,"prediction_agreement",agreements)
    coverage=rows(anchor/"results/reference_coverage.jsonl")
    identity=[r for r in main_inputs+historical if not r["reference_changed"]]
    assert len(identity)==390 and all(r["full_score"]==r["same_fact_score"]==r["direct_reference_score"] for r in identity)
    summary=dict(status="complete",new_model_forwards=0,new_annotations=0,parameter_changes=0,
        main_inputs=1290,main_pairs=1053,historical_inputs=569,unchanged_reference_inputs=390,
        all_published_main_metrics_reproduced=True,main_method_count=len(METHODS),
        preserved_pair_denominators=True,reference_coverage=coverage,
        test_prediction_agreement=[r for r in agreements if r["split"]=="test"],
        mechanism_pair_records=len(mechanism_records),path_cases=len(cases),
        provenance="Read published scores, not model predictions recomputed with a new template",
        unit="source for intervals; main node variants and historical routes are not independent replications")
    save(out/"validation/recompute.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("reference_coverage","test_prediction_agreement")},ensure_ascii=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();main(args.repo_root.resolve())

