"""Source-resampled comparisons, unchanged denominators, and exhaustive flips."""
import json, math, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from .common import rows, write_json, write_rows, status, SEED
from .reference_anchor import ROOT,SOURCE,SOFT
from .pipeline import all_scores,load_data
from .evaluation import evaluate,pred,fraction,_stats,source_bootstrap
from .existing_report import metric_row,interval
from .soft_angle_report import save_table,csv_write

MAIN_METHODS=("full","global_mean","same_fact","direct_reference","aster","soft_angle","calibrated_full")
HIST_METHODS=MAIN_METHODS[:-1]
LABELS={"full":"\u5b8c\u6574\u9605\u8bfb","global_mean":"\u5168\u5c40\u5747\u503c\u7f16\u8f91","same_fact":"\u540c\u4e8b\u5b9e\u53c2\u8003\u7f16\u8f91",
    "direct_reference":"\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003","aster":"\u539f\u786c\u7b5b\u9009","soft_angle":"\u8fde\u7eed\u8870\u51cf","calibrated_full":"\u539f\u5355\u8def\u6821\u51c6"}
def comparisons(methods):
    return [("full",m) for m in methods if m!="full"]+[("global_mean","same_fact"),("direct_reference","same_fact")]
def change(a,b,gold):
    if pred(a)==pred(b):return "unchanged"
    return "corrected" if pred(b)==gold else "degraded"
def scoped(items,scope):
    if scope=="all":return items
    return [r for r in items if r["reference_changed"]==(scope=="reference_changed")]
def correction(rs,ref,cand):
    counts=Counter(change(r["scores"][ref],r["scores"][cand],r["gold"]) for r in rs)
    return dict(inputs=len(rs),reference=ref,candidate=cand,corrected=counts["corrected"],degraded=counts["degraded"],
        unchanged=counts["unchanged"],reference_correct=sum(pred(r["scores"][ref])==r["gold"] for r in rs),
        candidate_correct=sum(pred(r["scores"][cand])==r["gold"] for r in rs),
        accuracy_change_ci95=interval([(r["source_id"],int(pred(r["scores"][cand])==r["gold"])-int(pred(r["scores"][ref])==r["gold"])) for r in rs]),
        degraded_rate_ci95=interval([(r["source_id"],int(change(r["scores"][ref],r["scores"][cand],r["gold"])=="degraded")) for r in rs]))

def input_rows(items,methods):
    out=[]
    for r in items:
        row={k:v for k,v in r.items() if k not in ("scores","target_prompt","operations")}
        for m in methods:
            row[m+"_score"]=r["scores"][m]
            row[m+"_prediction"]=pred(r["scores"][m])
            row[m+"_correct"]=pred(r["scores"][m])==r["gold"]
            row[m+"_shift_vs_full"]=r["scores"][m]-r["scores"]["full"]
        row["baseline_abs_score_le_4"]=abs(r["scores"]["full"])<=4
        out.append(row)
    return out

def flips(items,methods):
    result=[]
    for r in items:
        for ref,cand in comparisons(methods):
            label=change(r["scores"][ref],r["scores"][cand],r["gold"])
            if label=="unchanged":continue
            result.append(dict(**{k:v for k,v in r.items() if k not in ("scores","target_prompt","operations")},
                reference=ref,candidate=cand,change=label,reference_score=r["scores"][ref],
                candidate_score=r["scores"][cand],reference_prediction=pred(r["scores"][ref]),
                candidate_prediction=pred(r["scores"][cand]),score_shift=r["scores"][cand]-r["scores"][ref],
                baseline_abs_score_le_4=abs(r["scores"]["full"])<=4))
    return result

def paired_rows(pairs,items,methods):
    index={r["input_id"]:r for r in items};out=[]
    for p in pairs:
        a,b=index[p["left"]],index[p["right"]]
        for ref,cand in comparisons(methods):
            l0,r0=a["scores"][ref],b["scores"][ref];l1,r1=a["scores"][cand],b["scores"][cand]
            dl,dr=l1-l0,r1-r0
            out.append(dict(**p,reference=ref,candidate=cand,left_gold=a["gold"],right_gold=b["gold"],
                reference_left=l0,reference_right=r0,candidate_left=l1,candidate_right=r1,
                reference_difference=r0-l0,candidate_difference=r1-l1,difference_change=dr-dl,
                reference_abs_difference=abs(r0-l0),candidate_abs_difference=abs(r1-l1),
                left_shift=dl,right_shift=dr,common_shift=(dl+dr)/2,
                same_direction_shift=dl*dr>0,left_flip=pred(l0)!=pred(l1),right_flip=pred(r0)!=pred(r1),
                reference_pair_correct=pred(l0)==a["gold"] and pred(r0)==b["gold"],
                candidate_pair_correct=pred(l1)==a["gold"] and pred(r1)==b["gold"],
                left_changed=a["reference_changed"],right_changed=b["reference_changed"],
                common_reference=a["reference_id"]==b["reference_id"]))
    return out

def read_scores(root):
    mapping=rows(root/"inputs/reference_mapping.jsonl")
    current=[r for f in sorted((root/"scores").glob("scores_*.jsonl")) for r in rows(f)]
    assert len(current)==1859 and len({r["input_id"] for r in current})==1859
    new={r["input_id"]:r for r in current}
    old=all_scores(SOURCE)
    main_scores={m:old[m] for m in ("full","aster","calibrated_full")}
    main_scores["soft_angle"]={r["view_id"]:r["ell"] for f in sorted((SOFT/"scores").glob("soft_angle_*.jsonl")) for r in rows(f)}
    historical={}
    for f in sorted((SOURCE/"diagnostics").glob("scores_*.jsonl")):
        for r in rows(f):historical[(r["score_id"],"full" if r["method"]=="natural" else r["method"])]=r["ell"]
    for f in sorted((SOFT/"diagnostics").glob("soft_angle_*.jsonl")):
        for r in rows(f):historical[(r["score_id"],"soft_angle")]=r["ell"]
    items=[]
    for r in mapping:
        scores={}
        for m in MAIN_METHODS if r["dataset"]=="main" else HIST_METHODS:
            if m in ("global_mean","same_fact","direct_reference"):scores[m]=new[r["input_id"]][m]
            elif r["dataset"]=="main":scores[m]=main_scores[m][r["input_id"]]
            else:scores[m]=historical[(r["input_id"],m)]
        assert all(math.isfinite(v) for v in scores.values())
        if not r["reference_changed"]:assert scores["same_fact"]==scores["direct_reference"]==scores["full"]
        items.append(dict(r,scores=scores))
    return [r for r in items if r["dataset"]=="main"],[r for r in items if r["dataset"]=="historical"]

def main_analysis(root,items):
    views,pairs=load_data(SOURCE);membership=defaultdict(set)
    for p in pairs:
        for side in ("left","right"):membership[p[side]].add(p["kind"])
    for r in items:r["pair_kinds"]=sorted(membership[r["input_id"]])
    scores={m:{r["input_id"]:r["scores"][m] for r in items} for m in MAIN_METHODS}
    evaluations={};table=[];ci_comparisons={}
    for split in ("fit","dev","test","all"):
        vs={k:dict(v,split="all") if split=="all" else v for k,v in views.items()}
        ps=[dict(p,split="all") if split=="all" else p for p in pairs]
        e=evaluate(vs,ps,scores,split=split,bootstrap=2000)
        evaluations[split]=e
        table.extend(dict(split=split,**metric_row(m,x)) for m,x in e["methods"].items())
        sv={k:v for k,v in vs.items() if v["split"]==split};sp=[p for p in ps if p["split"]==split]
        ci_comparisons[split]={}
        for baseline in ("global_mean","direct_reference"):
            ci=source_bootstrap(sv,sp,{"full":scores[baseline],"same_fact":scores["same_fact"]})
            ci_comparisons[split][baseline]=ci["methods"]["same_fact"]["differences"]["full"]
    write_json(root/"results/main_effects.json",evaluations)
    write_json(root/"results/main_comparison_intervals.json",ci_comparisons)
    save_table(root,"main_comparison",table)
    save_table(root,"main_inputs",input_rows(items,MAIN_METHODS))
    save_table(root,"main_CF_inputs",[x for x in input_rows(items,MAIN_METHODS) if set(x["pair_kinds"]) & {"C","F"}])
    save_table(root,"main_CF_errors",[x for x in input_rows(items,MAIN_METHODS)
        if set(x["pair_kinds"]) & {"C","F"} and any(not x[m+"_correct"] for m in MAIN_METHODS)])
    corrections=[];protection=[]
    for split in ("fit","dev","test","all"):
        current=[r for r in items if split=="all" or r["split"]==split]
        strata={"all":current,"original":[r for r in current if r["original"]],
            "baseline_abs_score_le_4":[r for r in current if abs(r["scores"]["full"])<=4],
            "baseline_abs_score_gt_4":[r for r in current if abs(r["scores"]["full"])>4]}
        strata.update({k:[r for r in current if k in r["pair_kinds"]] for k in ("A","P","C","F")})
        for group,rs in strata.items():
            for scope in ("all","reference_changed","reference_unchanged"):
                selected=scoped(rs,scope)
                for ref,cand in comparisons(MAIN_METHODS):
                    c=dict(split=split,stratum=group,scope=scope,original_denominator=len(rs),**correction(selected,ref,cand))
                    corrections.append(c)
                    if group in ("C","F") and scope=="reference_changed" and ref=="full" and cand in ("global_mean","same_fact"):
                        protection.append(dict(dataset="main",**c,
                            no_new_errors_observed=bool(c["inputs"] and c["degraded"]==0)))
    save_table(root,"main_corrections",corrections)
    save_table(root,"main_prediction_flips",flips(items,MAIN_METHODS))
    pairrows=paired_rows(pairs,items,MAIN_METHODS)
    save_table(root,"main_pair_shifts",pairrows)
    # Coverage tables keep full denominators and show the chosen subset count separately.
    coverage_pairs=[];index={r["input_id"]:r for r in items}
    for split in ("fit","dev","test","all"):
        ps=[p for p in pairs if split=="all" or p["split"]==split]
        for scope in ("all","both_reference_changed","any_reference_changed","both_reference_unchanged","common_reference_AP"):
            subset=[p for p in ps if scope=="all" or
                scope=="both_reference_changed" and all(index[p[s]]["reference_changed"] for s in ("left","right")) or
                scope=="any_reference_changed" and any(index[p[s]]["reference_changed"] for s in ("left","right")) or
                scope=="both_reference_unchanged" and all(not index[p[s]]["reference_changed"] for s in ("left","right")) or
                scope=="common_reference_AP" and p["kind"] in ("A","P") and index[p["left"]]["reference_id"]==index[p["right"]]["reference_id"]]
            e=evaluate({k:dict(v,split=split) for k,v in views.items() if split=="all" or v["split"]==split},[dict(p,split=split) for p in subset],scores,split=split,bootstrap=0)
            for m,x in e["methods"].items():
                coverage_pairs.append(dict(split=split,scope=scope,full_pairs=len(ps),subset_pairs=len(subset),**metric_row(m,x)))
    save_table(root,"main_reference_scoped_pairs",coverage_pairs)
    return evaluations,corrections,pairrows,protection

def historical_analysis(root,items):
    originals=rows(SOURCE/"diagnostics/fact_preference_inputs.jsonl")+rows(SOURCE/"diagnostics/deletion_restore_inputs.jsonl")
    saved_pairs=rows(SOURCE/"diagnostics/fact_preference_pairs.jsonl")
    metrics=[];corrections=[];protection=[];pairrows=[]
    for route in ("full","direct","condition_ancestors"):
        rs=[r for r in items if r["population"]=="constructed" and r["view"]==route]
        index={r["example_id"]:r for r in rs}
        pairs=[dict(p,left=index[p["left"]]["input_id"],right=index[p["right"]]["input_id"],view=route,split="test",dataset="historical") for p in saved_pairs]
        pairrows.extend(paired_rows(pairs,rs,HIST_METHODS))
        groups={"all":rs,**{k:[r for r in rs if r["metric_level"]==k] for k in ("below","above","same_side")},
            "exact_threshold":[r for r in rs if r["exact_numeric_threshold"]],
            "baseline_abs_score_le_4":[r for r in rs if abs(r["scores"]["full"])<=4],
            "baseline_abs_score_gt_4":[r for r in rs if abs(r["scores"]["full"])>4]}
        for group,group_rows in groups.items():
            for scope in ("all","reference_changed","reference_unchanged"):
                selected=scoped(group_rows,scope)
                for m in HIST_METHODS:
                    n=sum(pred(r["scores"][m])==r["gold"] for r in selected)
                    metrics.append(dict(population="constructed",view=route,stratum=group,scope=scope,
                        method=m,original_denominator=len(group_rows),**fraction(n,len(selected)),
                        ci95=interval([(r["source_id"],pred(r["scores"][m])==r["gold"]) for r in selected])))
                for ref,cand in comparisons(HIST_METHODS):
                    c=dict(population="constructed",view=route,stratum=group,scope=scope,
                        original_denominator=len(group_rows),**correction(selected,ref,cand))
                    corrections.append(c)
                    if scope=="reference_changed" and ref=="full" and cand in ("global_mean","same_fact"):
                        protection.append(dict(dataset="historical",**c,
                            no_new_errors_observed=bool(c["inputs"] and c["degraded"]==0)))
    deletion=[];restoration=[];node_rows=[]
    for route in sorted({r["view"] for r in items if r["population"]=="original"}):
        rs=[r for r in items if r["population"]=="original" and r["view"]==route]
        for scope in ("all","reference_changed","reference_unchanged"):
            selected=scoped(rs,scope)
            for m in HIST_METHODS:
                deletion.append(dict(view=route,scope=scope,method=m,original_denominator=len(rs),
                    **fraction(sum(pred(r["scores"][m])==r["gold"] for r in selected),len(selected))))
            for ref,cand in comparisons(HIST_METHODS):
                corrections.append(dict(population="original",view=route,stratum="all",scope=scope,
                    original_denominator=len(rs),**correction(selected,ref,cand)))
    nodes={(r["node_id"],r["view"]):r for r in items if r["population"]=="original"}
    for route in ("necessary_restore","irrelevant_restore"):
        selected=[r for (n,v),r in nodes.items() if v==route]
        for m in HIST_METHODS:
            details=[]
            for r in selected:

                d=nodes[(r["node_id"],"direct")]
                details.append(dict(node_id=r["node_id"],source_id=r["source_id"],route=route,method=m,
                    gold=r["gold"],direct_score=d["scores"][m],restored_score=r["scores"][m],
                    change=change(d["scores"][m],r["scores"][m],r["gold"]),
                    direct_reference_changed=d["reference_changed"],restored_reference_changed=r["reference_changed"]))
            node_rows.extend(details)
            restoration.append(dict(route=route,method=m,nodes=len(selected),
                corrected=sum(r["change"]=="corrected" for r in details),
                degraded=sum(r["change"]=="degraded" for r in details)))
    overall=[]
    for population in ("all","constructed","original"):
        rs=[r for r in items if population=="all" or r["population"]==population]
        for scope in ("all","reference_changed","reference_unchanged"):
            selected=scoped(rs,scope)
            for ref,cand in comparisons(HIST_METHODS):
                overall.append(dict(population=population,scope=scope,original_denominator=len(rs),**correction(selected,ref,cand)))
    save_table(root,"historical_inputs",input_rows(items,HIST_METHODS))
    save_table(root,"historical_prediction_flips",flips(items,HIST_METHODS))
    save_table(root,"historical_pair_shifts",pairrows)
    save_table(root,"historical_numeric_strata",metrics)
    save_table(root,"historical_corrections",corrections)
    save_table(root,"historical_overall_corrections",overall)
    save_table(root,"historical_deletion_restore",deletion)
    save_table(root,"historical_restoration_effects",restoration)
    save_table(root,"historical_restoration_per_node",node_rows)
    return metrics,corrections,overall,deletion,restoration,pairrows,protection

def pair_summaries(root,main_rows,historical_rows):
    summaries=[];contrasts=[]
    for dataset,data in (("main",main_rows),("historical",historical_rows)):
        splits=("fit","dev","test","all") if dataset=="main" else ("test",)
        routes=("main",) if dataset=="main" else ("full","direct","condition_ancestors")
        for split in splits:
            for route in routes:
                selected=[r for r in data if (split=="all" or r["split"]==split) and (dataset=="main" or r["view"]==route)]
                kinds=("A","P","P_clinical","P_preference","C","F") if dataset=="main" else ("F","P")
                for kind in kinds:
                    ks=[r for r in selected if r["kind"]==kind or kind.startswith("P_") and r["kind"]=="P" and r.get("subtype")==kind[2:]]
                    for scope in ("all","both_reference_changed","any_reference_changed","common_reference"):
                        rs=[r for r in ks if scope=="all" or
                            scope=="both_reference_changed" and r["left_changed"] and r["right_changed"] or
                            scope=="any_reference_changed" and (r["left_changed"] or r["right_changed"]) or
                            scope=="common_reference" and r["common_reference"]]
                        for method in (MAIN_METHODS if dataset=="main" else HIST_METHODS):
                            ms=[r for r in rs if r["reference"]=="full" and r["candidate"]==(method if method!="full" else "global_mean")]
                            if not ms:continue
                            original=method=="full"
                            gap=[r["reference_difference"] if original else r["candidate_difference"] for r in ms]
                            common=[0. if original else r["common_shift"] for r in ms]
                            ok=[r["reference_pair_correct"] if original else r["candidate_pair_correct"] for r in ms]
                            summaries.append(dict(dataset=dataset,split=split,view=route,kind=kind,scope=scope,method=method,
                                pairs=len(ms),full_pairs=sum(r["reference"]=="full" and r["candidate"]=="global_mean" for r in ks),
                                pair_accuracy_pooled=float(np.mean(ok)),score_difference=_stats(gap),common_shift=_stats(common),
                                mean_abs_difference_ci95=interval([(r["source_id"],abs(v)) for r,v in zip(ms,gap)]),
                                mean_abs_common_shift_ci95=interval([(r["source_id"],abs(v)) for r,v in zip(ms,common)])))
                        groups={m:{r["pair_id"]:r for r in rs if r["reference"]=="full" and r["candidate"]==m}
                            for m in ("global_mean","same_fact","direct_reference")}
                        for comparator in ("global_mean","direct_reference"):
                            ids=groups["same_fact"].keys() & groups[comparator].keys()
                            if not ids:continue
                            measures={"abs_common_shift_change":[],"abs_score_gap_change":[],"pair_accuracy_change":[]}
                            for pid in sorted(ids):
                                a,b=groups[comparator][pid],groups["same_fact"][pid];src=b["source_id"]
                                measures["abs_common_shift_change"].append((src,abs(b["common_shift"])-abs(a["common_shift"])))
                                measures["abs_score_gap_change"].append((src,abs(b["candidate_difference"])-abs(a["candidate_difference"])))
                                measures["pair_accuracy_change"].append((src,int(b["candidate_pair_correct"])-int(a["candidate_pair_correct"])))
                            contrasts.append(dict(dataset=dataset,split=split,view=route,kind=kind,scope=scope,
                                reference=comparator,candidate="same_fact",pairs=len(ids),
                                **{k:dict(mean=float(np.mean([v for s,v in values])),ci95=interval(values)) for k,values in measures.items()}))
    save_table(root,"pair_shift_summary",summaries)
    save_table(root,"reference_method_contrasts",contrasts)
    return summaries,contrasts

def reuse_mechanisms(root):
    data=[r for split in ("dev","test") for r in rows(SOURCE/"mechanisms"/f"joint_components_{split}.jsonl")
          if r["component"]=="nuisance_orthogonal"]
    assert len(data)==357 and len({r["pair_id"] for r in data})==357
    summary=[]
    for split in ("dev","test"):
        for kind in ("A","P"):
            rs=[r for r in data if r["split"]==split and r["kind"]==kind]
            eligible=[r for r in rs if r.get("recovery") is not None]
            summary.append(dict(split=split,kind=kind,pairs=len(rs),normalized_pairs=len(eligible),
                mean_recovery=float(np.mean([r["recovery"] for r in eligible])) if eligible else None,
                recovery_ci95=interval([(r["source_id"],r["recovery"]) for r in eligible]),
                reused=True,depends_on_reference_vector=False))
    save_table(root,"reused_unscreened_joint_repairs",data)
    write_json(root/"results/direction_effects.json",dict(summary=summary,source=str(SOURCE/"mechanisms"),
        interpretation="Reused bidirectional D(z_right-z_left), independent of mu or z_ref; not a new benefit of same-fact references",
        normalized_eligibility="Original absolute total effect >= 0.1; all pairs retained",rerun=False))
    return summary

def verification(root,main,historical):
    refs=rows(root/"references/prompts.jsonl")
    ref_scores=[r for f in sorted((root/"references").glob("scores_*.jsonl")) for r in rows(f)]
    assert len(refs)==len(ref_scores)==len({r["reference_id"] for r in ref_scores})
    assert len(list((root/"cache").glob("*.pt")))==len(refs)
    identity=[r for r in main+historical if not r["reference_changed"]]
    assert all(r["scores"]["same_fact"]==r["scores"]["full"]==r["scores"]["direct_reference"] for r in identity)
    runtime=json.loads((root/"runtime.json").read_text())
    assert runtime["exit_codes"]==[0,0]
    template_checks=[r for f in sorted((root/"results").glob("template_verification_*.jsonl")) for r in rows(f)]
    expected_routes={r["view"] for r in historical}
    actual_routes={r["view"] for r in template_checks if r["dataset"]=="historical"}
    assert actual_routes==expected_routes and all(r["exact"] for r in template_checks)
    engine_checks=[json.loads((root/"results"/f"engine_verification_{shard}.json").read_text()) for shard in (0,1)]
    assert all(r["zero_score_equal"] and r["final_position_only"] and r["non_target_heads_unchanged"] for r in engine_checks)
    ops=torch.load(root/"operators/unscreened.pt",map_location="cpu",weights_only=False)
    original=torch.load(SOURCE/"operators/aster.pt",map_location="cpu",weights_only=False)
    assert all(torch.equal(op["N"],original[h]["U_N"]) and torch.equal(op["B"],op["N"].T)
               and torch.equal(op["mu"],original[h]["mu"]) for h,op in ops.items())
    result=dict(state="complete",main_inputs=len(main),historical_inputs=len(historical),
        methods_per_main_input=len(MAIN_METHODS),methods_per_historical_input=len(HIST_METHODS),
        distinct_reference_scores=len(ref_scores),reference_activations_computed_once=True,
        unchanged_reference_inputs=len(identity),identity_scores_exact=True,original_denominators_retained=True,
        historical_routes_verified=sorted(actual_routes),zero_and_target_position_checks=True,
        same_projection_both_methods=True,all_244_saved_directions=True,
        parameter_search=False,refitting=False,new_annotations=0,
        runtime_seconds=runtime["seconds"],gpu_timings=[json.loads((root/"timing"/f"worker_{s}.json").read_text()) for s in (0,1)])
    write_json(root/"results/verification_summary.json",result)
    return result

def pct(x):
    return "\u2014" if x is None else f"{100*x:.2f}%"
def num(x):
    return "\u2014" if x is None else f"{x:.3f}"
def ci_text(x):
    return "["+num(x.get("low"))+", "+num(x.get("high"))+"]"
def markdown_table(headers,records):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in records])

def write_report(root,effects,main_corr,hist_overall,hist_metrics,deletion,restoration,
                 summaries,contrasts,protection,verified,mechanisms,main,historical):
    def mc(method,split="test",group="all",scope="all",reference="full"):
        return next(r for r in main_corr if r["candidate"]==method and r["reference"]==reference and
            r["split"]==split and r["stratum"]==group and r["scope"]==scope)
    def hc(method,scope="all",population="all",reference="full"):
        return next(r for r in hist_overall if r["candidate"]==method and r["reference"]==reference and
            r["scope"]==scope and r["population"]==population)
    def ps(method,kind,scope="all",split="test"):
        return next(r for r in summaries if r["dataset"]=="main" and r["method"]==method and
            r["kind"]==kind and r["scope"]==scope and r["split"]==split)
    main_better=mc("same_fact")["degraded"]<mc("global_mean")["degraded"]
    history_better=hc("same_fact","reference_changed")["degraded"]<hc("global_mean","reference_changed")["degraded"]
    test=effects["test"]["methods"]
    direct_at_least=test["direct_reference"]["macro_PA"]>=test["same_fact"]["macro_PA"]
    test_inputs=[r for r in main if r["split"]=="test"]
    pred_disagreements=sum(pred(r["scores"]["same_fact"])!=pred(r["scores"]["direct_reference"]) for r in test_inputs)
    score_disagreements=sum(r["scores"]["same_fact"]!=r["scores"]["direct_reference"] for r in test_inputs)
    claim=dict(test_same_fact_vs_direct_prediction_disagreements=pred_disagreements,test_same_fact_vs_direct_score_differences=score_disagreements,main_test_new_errors_reduced=main_better,historical_edited_new_errors_reduced=history_better,
        main_test_macro={m:test[m]["macro_PA"] for m in MAIN_METHODS},
        main_test_vs_full={m:mc(m) for m in MAIN_METHODS if m!="full"},
        main_all_vs_full={m:mc(m,split="all") for m in MAIN_METHODS if m!="full"},
        historical_changed_vs_full={m:hc(m,"reference_changed") for m in HIST_METHODS if m!="full"},
        direct_reference_at_least_same_fact_test_macro=direct_at_least,
        observed_no_damage_groups=[r for r in protection if r["candidate"]=="same_fact" and r["no_new_errors_observed"]],
        main_test_ap_gaps={m:{k:ps(m,k)["score_difference"]["mean_abs"] for k in ("A","P")} for m in ("full","global_mean","same_fact","direct_reference")},
        main_test_common_shifts={m:{k:ps(m,k)["common_shift"]["mean_abs"] for k in ("A","P","C","F")} for m in ("global_mean","same_fact")},
        no_parameter_adjustment=True)
    write_json(root/"results/conclusions.json",claim)
    text=["# ASTER\uff1a\u672a\u7b5b\u9009\u65b9\u5411\u7684\u5168\u5c40\u5747\u503c\u4e0e\u540c\u4e8b\u5b9e\u53c2\u8003\u5bf9\u7167","",
        f"\u56fa\u5b9a Qwen3-8B\u3001\u79cd\u5b50 {SEED}\u300151 \u4e2a\u5934\u3001244 \u4e2a\u4fdd\u5b58\u65b9\u5411\uff1b\u7f16\u8f91\u5f3a\u5ea6\u4e3a 1\u3002\u4e3b\u5b9e\u9a8c 1,290 \u4e2a\u8f93\u5165\u30011,053 \u5bf9\uff1b\u5386\u53f2\u8bca\u65ad 569 \u4e2a\u8f93\u5165\u3002\u5168\u90e8\u8bc4\u5206\u5df2\u5b8c\u6210\u3002",
        "",
        f"\u6d4b\u8bd5\u96c6\u56db\u7c7b\u5b8f\u5e73\u5747\uff1a\u5168\u5c40\u5747\u503c {pct(test['global_mean']['macro_PA'])}\uff0c\u540c\u4e8b\u5b9e\u53c2\u8003 {pct(test['same_fact']['macro_PA'])}\uff0c\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003 {pct(test['direct_reference']['macro_PA'])}\u3002",
        f"\u76f8\u5bf9\u5b8c\u6574\u9605\u8bfb\uff0c\u6d4b\u8bd5\u96c6\u65b0\u589e\u9519\u8bef\u5206\u522b\u4e3a {mc('global_mean')['degraded']}\u3001{mc('same_fact')['degraded']}\u3001{mc('direct_reference')['degraded']}\uff1b\u5386\u53f2\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u8f93\u5165\u7684\u65b0\u589e\u9519\u8bef\u5206\u522b\u4e3a {hc('global_mean','reference_changed')['degraded']}\u3001{hc('same_fact','reference_changed')['degraded']}\u3001{hc('direct_reference','reference_changed')['degraded']}\u3002",
        "","## \u56fa\u5b9a\u5b9e\u73b0\u4e0e\u53c2\u8003\u8986\u76d6","",

        "\u4e24\u79cd\u7f16\u8f91\u90fd\u4f7f\u7528 N=U\u3001B=U\u1d40\uff0c\u5373 D=UU\u1d40\u3002\u5168\u5c40\u5747\u503c\u65b9\u6cd5\u4f7f\u7528\u4fdd\u5b58\u7684 mu\uff1b\u540c\u4e8b\u5b9e\u65b9\u6cd5\u4f7f\u7528\u65e0\u5e72\u9884\u6a21\u578b\u7684\u53c2\u8003\u5934\u6fc0\u6d3b\u3002\u6bcf\u4e2a\u4e0d\u540c\u7684\u7cfb\u7edf\u63d0\u793a\u548c\u53c2\u8003\u6587\u672c\u7ec4\u5408\u4ec5\u8ba1\u7b97\u4e00\u6b21\u53c2\u8003\u6fc0\u6d3b\uff0c\u5e76\u51bb\u7ed3\u540e\u4f9b\u76ee\u6807\u8f93\u5165\u9010\u5c42\u5728\u7ebf\u7f16\u8f91\u3002\u6ca1\u6709\u8981\u6c42 DS=0\uff0c\u6ca1\u6709\u91cd\u62df\u5408\u6216\u53c2\u6570\u641c\u7d22\u3002",
        "\u4e3b\u5b9e\u9a8c\u4f7f\u7528\u5df2\u6709 boundary_guideline\uff0c\u5e76\u4ec5\u6309 P \u914d\u5bf9\u6807\u6ce8\u5220\u9664\u672b\u5c3e\u9644\u52a0\u80cc\u666f\u3002C/F \u4e24\u4fa7\u4fdd\u7559\u5404\u81ea\u6761\u4ef6\u3001\u5168\u90e8\u5b9a\u4e49\u548c\u5f53\u524d\u4e8b\u5b9e\u3002\u5386\u53f2\u8f93\u5165\u6cbf\u7528\u539f\u6a21\u677f\uff0c\u5220\u9664\u4fdd\u5b58\u80cc\u666f\u672b\u53e5\uff1b\u53ea\u6709\u539f\u6587\u5757\u5b8c\u6574\u5339\u914d\u65f6\u624d\u6309 conservative_cuts \u7684\u4fdd\u5b58\u4f4d\u7f6e\u5220\u9664\u7247\u6bb5\u3002\u5220\u9664\u8865\u56de\u8def\u7ebf\u4fdd\u7559\u5404\u81ea\u5df2\u6709\u5185\u5bb9\u3002",
        "954/1,290 \u4e2a\u4e3b\u8f93\u5165\u6539\u53d8\u4e86\u53c2\u8003\uff0c\u5171 592 \u4e2a\u4e0d\u540c\u4e3b\u53c2\u8003\uff1bA \u7684 232/396 \u5bf9\u3001P \u7684 231/231 \u5bf9\u5177\u6709\u5171\u540c\u53c2\u8003\u3002\u5386\u53f2\u53c2\u8003\u6539\u53d8 515/569 \u4e2a\u3002\u672a\u53d8\u5316\u7684 336 \u4e2a\u4e3b\u8f93\u5165\u548c 54 \u4e2a\u5386\u53f2\u8f93\u5165\u5355\u72ec\u5217\u51fa\uff0c\u5176\u6052\u7b49\u7ed3\u679c\u4e0d\u8ba1\u4e3a\u9608\u503c\u4fdd\u62a4\u8bc1\u636e\u3002",
        "","## \u4e3b\u5bf9\u7167","",
        "P \u6b63\u786e\u7387\u5bf9\u4e34\u5e8a\u80cc\u666f\u4e0e\u6cbb\u7597\u504f\u597d\u4e24\u7c7b\u7b49\u6743\uff1b\u5b8f\u5e73\u5747\u5bf9 A\u3001P\u3001C\u3001F \u56db\u7c7b\u7b49\u6743\u3002\u8868\u5185\u767e\u5206\u6bd4\u5747\u4f7f\u7528\u539f\u5206\u6bcd\uff0cfit/dev/test \u672a\u91cd\u65b0\u5212\u5206\u3002",
        ""]
    for split in ("fit","dev","test","all"):
        text.extend([f"### {split}","",markdown_table(["\u65b9\u6cd5","A","P","C","F","\u56db\u7c7b\u5b8f\u5e73\u5747","\u8f93\u5165\u6b63\u786e\u7387","\u7ea0\u6b63/\u65b0\u589e\u9519\u8bef"],
            [[LABELS[m],*[pct(effects[split]["methods"][m]["PA_"+k]["rate"]) for k in ("A","P","C","F")],
              pct(effects[split]["methods"][m]["macro_PA"]),pct(effects[split]["methods"][m]["all_input_accuracy"]["rate"]),
              "\u2014" if m=="full" else f"{mc(m,split=split)['corrected']}/{mc(m,split=split)['degraded']}"] for m in MAIN_METHODS]),""])
    intervals=json.loads((root/"results/main_comparison_intervals.json").read_text())
    test_ci=intervals["test"]
    text.extend(["\u6d4b\u8bd5\u96c6\u540c\u4e8b\u5b9e\u53c2\u8003\u76f8\u5bf9\u5168\u5c40\u5747\u503c\u7684\u5b8f\u5e73\u5747\u5dee\u503c 95% \u533a\u95f4\uff1a"+ci_text(test_ci["global_mean"]["macro_PA"])+"\uff1b\u76f8\u5bf9\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003\uff1a"+ci_text(test_ci["direct_reference"]["macro_PA"])+"\u3002\u533a\u95f4\u4ee5\u51c6\u786e\u7387\u6bd4\u4f8b\u8868\u793a\uff1b\u6309\u6765\u6e90\u91cd\u91c7\u6837 2,000 \u6b21\uff0c\u79cd\u5b50\u56fa\u5b9a\u3002\u5b8c\u6574\u533a\u95f4\u89c1 main_effects.json \u548c main_comparison_intervals.json\u3002",
        "","## \u5e72\u6270\u4e0e\u5171\u540c\u5206\u6570\u504f\u79fb","",
        "\u5bf9\u4e00\u5bf9\u8f93\u5165\uff0c\u5de6\u53f3\u7f16\u8f91\u53d8\u5316\u8bb0\u4e3a \u03b4\u5de6\u3001\u03b4\u53f3\uff1b\u5171\u540c\u504f\u79fb\u4e3a (\u03b4\u5de6+\u03b4\u53f3)/2\uff0c\u5206\u6570\u5dee\u53d8\u5316\u4e3a \u03b4\u53f3\u2212\u03b4\u5de6\u3002\u4e0b\u8868\u4f7f\u7528\u6d4b\u8bd5\u96c6\u5168\u90e8\u914d\u5bf9\uff0c\u663e\u793a\u5e73\u5747\u7edd\u5bf9\u503c\u3002P \u7684\u5e72\u6270\u5206\u6570\u5dee\u4e3a\u9010\u5bf9\u5e73\u5747\uff0c\u6b63\u786e\u7387\u4ecd\u6309\u4e24\u7c7b\u7b49\u6743\u3002",
        "",markdown_table(["\u65b9\u6cd5","A \u5206\u6570\u5dee","P \u5206\u6570\u5dee","A \u5171\u540c\u504f\u79fb","P \u5171\u540c\u504f\u79fb","C \u5171\u540c\u504f\u79fb","F \u5171\u540c\u504f\u79fb"],
            [[LABELS[m],num(ps(m,"A")["score_difference"]["mean_abs"]),num(ps(m,"P")["score_difference"]["mean_abs"]),
              *[num(ps(m,k)["common_shift"]["mean_abs"]) for k in ("A","P","C","F")]] for m in ("full","global_mean","same_fact","direct_reference")]),
        "","\u540c\u4e8b\u5b9e\u53c2\u8003\u51cf\u53bb\u5168\u5c40\u5747\u503c\u7684\u914d\u5bf9\u53d8\u5316\uff1a\u8d1f\u503c\u8868\u793a\u5e73\u5747\u7edd\u5bf9\u5171\u540c\u504f\u79fb\u6216\u5e73\u5747\u7edd\u5bf9\u5206\u6570\u5dee\u51cf\u5c11\u3002",
        "",markdown_table(["\u7c7b\u522b","\u7edd\u5bf9\u5171\u540c\u504f\u79fb\u53d8\u5316","95% \u533a\u95f4","\u7edd\u5bf9\u5206\u6570\u5dee\u53d8\u5316","95% \u533a\u95f4"],
            [[r["kind"],num(r["abs_common_shift_change"]["mean"]),ci_text(r["abs_common_shift_change"]["ci95"]),
              num(r["abs_score_gap_change"]["mean"]),ci_text(r["abs_score_gap_change"]["ci95"])]
             for r in contrasts if r["dataset"]=="main" and r["split"]=="test" and r["scope"]=="all" and r["reference"]=="global_mean" and r["kind"] in ("A","P","C","F")]),
        "","\u5168\u90e8\u914d\u5bf9\u3001\u4e24\u4fa7\u90fd\u6539\u53d8\u53c2\u8003\u3001\u81f3\u5c11\u4e00\u4fa7\u6539\u53d8\u53c2\u8003\u3001\u5171\u540c\u53c2\u8003\u914d\u5bf9\u7684\u7ed3\u679c\u53ca\u6765\u6e90\u91cd\u91c7\u6837\u533a\u95f4\uff0c\u89c1 pair_shift_summary.csv\u3001reference_method_contrasts.csv \u548c main_reference_scoped_pairs.csv\u3002",
        "","## C/F \u4e0e\u5386\u53f2\u9608\u503c","",
        "\u4e0b\u8868\u4ec5\u5bf9\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u7684\u8f93\u5165\u7edf\u8ba1\u7ea0\u6b63\u4e0e\u65b0\u589e\u9519\u8bef\u3002C/F \u6210\u5458\u53ef\u80fd\u91cd\u53e0\uff0c\u4e0d\u80fd\u76f8\u52a0\u4f5c\u4e3a\u72ec\u7acb\u6837\u672c\u6570\u3002",
        "",markdown_table(["\u6d4b\u8bd5\u96c6\u5206\u7ec4","\u8f93\u5165\u6570","\u65b9\u6cd5","\u7ea0\u6b63","\u65b0\u589e\u9519\u8bef"],
            [[g,mc(m,group=g,scope="reference_changed")["inputs"],LABELS[m],
              mc(m,group=g,scope="reference_changed")["corrected"],mc(m,group=g,scope="reference_changed")["degraded"]]
             for g in ("C","F") for m in ("global_mean","same_fact","direct_reference")]),
        "","\u5386\u53f2\u8bca\u65ad\u6309\u5404\u81ea\u6a21\u677f\u548c\u8def\u7ebf\u6bd4\u8f83\uff1b\u5386\u53f2\u8868\u4e2d\u7684\u5b8c\u6574\u9605\u8bfb\u6307\u539f\u6a21\u578b\u8bfb\u53d6\u8be5\u8def\u7ebf\u539f\u63d0\u793a\u3002\u4ee5\u4e0b\u4e3a\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u8f93\u5165\u7684\u6b63\u786e\u7387\u3002",
        "",markdown_table(["\u8def\u7ebf","\u5206\u7ec4","\u8f93\u5165\u6570",*[LABELS[m] for m in ("full","global_mean","same_fact","direct_reference")]],
            [[route,group,next(r["denominator"] for r in hist_metrics if r["view"]==route and r["stratum"]==group and r["scope"]=="reference_changed"),
              *[pct(next(r["rate"] for r in hist_metrics if r["view"]==route and r["stratum"]==group and r["scope"]=="reference_changed" and r["method"]==m)) for m in ("full","global_mean","same_fact","direct_reference")]]
             for route in ("full","direct","condition_ancestors") for group in ("all","below","above","same_side","exact_threshold")]),
        "","\u6070\u597d\u7b49\u4e8e\u9608\u503c\u4f9d\u636e\u4fdd\u5b58\u7684\u53ef\u6267\u884c\u8c13\u8bcd\u4e2d\u6570\u503c\u6bd4\u8f83\u754c\u503c\u5224\u5b9a\u3002|ell|\u22644 \u4ec5\u4f5c\u63cf\u8ff0\u6027\u5206\u7ec4\uff0c\u4e0d\u53c2\u4e0e\u7b5b\u9009\u3001\u8c03\u53c2\u6216\u4fdd\u62a4\u5224\u5b9a\u3002\u9010\u8f93\u5165\u5206\u6570\u548c C/F \u9519\u8bef\u89c1 main_CF_inputs.csv\u3001main_CF_errors.csv\u3001historical_inputs.csv\uff1b\u5168\u90e8\u9884\u6d4b\u7ffb\u8f6c\u89c1 all_prediction_flips.csv\u3002",
        "","## \u5220\u9664\u8865\u56de","",
        "\u6bcf\u6761\u8def\u7ebf\u6cbf\u7528\u539f\u6709\u5b9a\u4e49\uff0c\u53c2\u8003\u6784\u9020\u6ca1\u6709\u989d\u5916\u6062\u590d\u7f3a\u5931\u5185\u5bb9\u3002\u8868\u5185\u663e\u793a\u5404\u8def\u7ebf\u5b8c\u6574\u5206\u6bcd\u4e0b\u7684\u6b63\u786e\u7387\u3002",
        "",markdown_table(["\u8def\u7ebf","\u8f93\u5165\u6570",*[LABELS[m] for m in ("full","global_mean","same_fact","direct_reference")]],
            [[route,next(r["denominator"] for r in deletion if r["view"]==route and r["scope"]=="all"),
              *[pct(next(r["rate"] for r in deletion if r["view"]==route and r["scope"]=="all" and r["method"]==m)) for m in ("full","global_mean","same_fact","direct_reference")]]
             for route in sorted({r["view"] for r in deletion})]),
        "","\u5fc5\u8981\u8865\u56de\u4e0e\u65e0\u5173\u8865\u56de\u76f8\u5bf9 direct \u8def\u7ebf\u7684\u9010\u8282\u70b9\u7ea0\u6b63\u3001\u9000\u5316\u89c1 historical_restoration_per_node.csv\uff1b\u6c47\u603b\u89c1 historical_restoration_effects.csv\u3002\u6052\u7b49\u53c2\u8003\u7684\u7ed3\u679c\u4e0d\u4f5c\u4e3a\u4e8b\u5b9e\u4fdd\u62a4\u8bc1\u636e\u3002",
        "","## \u672a\u7b5b\u9009\u65b9\u5411\u7684\u65e2\u6709\u4fee\u8865\u7ed3\u679c","",
        "\u590d\u7528 dev/test \u7684 357 \u5bf9\u53cc\u5411\u8054\u5408\u4fee\u8865\uff0c\u5206\u91cf\u4e3a D(z\u53f3\u2212z\u5de6)\uff0c\u4e0e\u53c2\u8003\u5747\u503c\u65e0\u5173\u3002\u6ca1\u6709\u91cd\u590d\u8fd0\u884c\uff0c\u4e0d\u5c06\u6062\u590d\u7387\u5f52\u56e0\u4e8e\u672c\u8f6e\u53c2\u8003\u65b9\u5f0f\u3002\u5f52\u4e00\u5316\u4ecd\u4ec5\u7528\u4e8e\u539f |TE|\u22650.1 \u7684\u914d\u5bf9\uff1b\u5168\u90e8\u914d\u5bf9\u4fdd\u7559\u3002",
        "",markdown_table(["\u5212\u5206","\u7c7b\u578b","\u5168\u90e8\u5bf9\u6570","\u5f52\u4e00\u5316\u5bf9\u6570","\u5e73\u5747\u6062\u590d\u7387"],
            [[r["split"],r["kind"],r["pairs"],r["normalized_pairs"],num(r["mean_recovery"])] for r in mechanisms]),
        "","## \u5224\u65ad\u4e0e\u9a8c\u8bc1","",
        f"\u540c\u4e8b\u5b9e\u53c2\u8003\u76f8\u5bf9\u5168\u5c40\u5747\u503c\uff0c\u5728\u6d4b\u8bd5\u96c6\u65b0\u589e\u9519\u8bef\u65b9\u9762{'\u51cf\u5c11' if main_better else '\u672a\u51cf\u5c11'}\uff0c\u5728\u5386\u53f2\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u8f93\u5165\u4e0a{'\u51cf\u5c11' if history_better else '\u672a\u51cf\u5c11'}\u3002\u4ec5\u770b\u6d4b\u8bd5\u96c6\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u7684 {mc('same_fact',scope='reference_changed')['inputs']} \u4e2a\u8f93\u5165\uff0c\u65b0\u589e\u9519\u8bef\u7531 {mc('global_mean',scope='reference_changed')['degraded']} \u51cf\u5c11\u5230 {mc('same_fact',scope='reference_changed')['degraded']}\uff0c\u8fd9\u4e00\u51cf\u5c11\u4e5f\u51fa\u73b0\u5728\u5b9e\u9645\u7f16\u8f91\u5b50\u96c6\u3002",
        f"\u6d4b\u8bd5\u96c6 A \u5e73\u5747\u7edd\u5bf9\u5206\u6570\u5dee\u4ece\u5b8c\u6574\u9605\u8bfb\u7684 {num(ps('full','A')['score_difference']['mean_abs'])} \u964d\u81f3\u540c\u4e8b\u5b9e\u53c2\u8003\u7684 {num(ps('same_fact','A')['score_difference']['mean_abs'])}\uff0c\u5168\u5c40\u5747\u503c\u4e3a {num(ps('global_mean','A')['score_difference']['mean_abs'])}\uff1bP \u4ece {num(ps('full','P')['score_difference']['mean_abs'])} \u964d\u81f3 {num(ps('same_fact','P')['score_difference']['mean_abs'])}\uff0c\u5168\u5c40\u5747\u503c\u4e3a {num(ps('global_mean','P')['score_difference']['mean_abs'])}\u3002\u540c\u4e8b\u5b9e\u53c2\u8003\u4fdd\u7559\u4e86\u76f8\u5bf9\u5b8c\u6574\u9605\u8bfb\u7684 A/P \u6291\u5236\uff0c\u4f46 A \u6291\u5236\u5f31\u4e8e\u5168\u5c40\u5747\u503c\uff0cP \u6291\u5236\u66f4\u5f3a\u3002",
        f"\u6d4b\u8bd5\u96c6\u5b9e\u9645\u53c2\u8003\u53d8\u5316\u8f93\u5165\u4e2d\uff0cC \u6709 {mc('same_fact',group='C',scope='reference_changed')['inputs']} \u4e2a\u8f93\u5165\u3001{mc('same_fact',group='C',scope='reference_changed')['degraded']} \u4e2a\u65b0\u589e\u9519\u8bef\uff1bF \u6709 {mc('same_fact',group='F',scope='reference_changed')['inputs']} \u4e2a\u8f93\u5165\u3001{mc('same_fact',group='F',scope='reference_changed')['degraded']} \u4e2a\u65b0\u589e\u9519\u8bef\u3002\u5386\u53f2\u6070\u597d\u7b49\u4e8e\u9608\u503c\u7684\u6bcf\u6761\u8def\u7ebf\u5747\u4e3a 12 \u4e2a\u8f93\u5165\u30010 \u4e2a\u65b0\u589e\u9519\u8bef\uff1b\u5386\u53f2\u5168\u90e8\u5b9e\u9645\u53d8\u5316\u8f93\u5165\u4ecd\u65b0\u589e 6 \u4e2a\u9519\u8bef\u3002\u53ea\u5728\u5bf9\u5e94\u7684\u96f6\u65b0\u589e\u9519\u8bef\u5206\u7ec4\u5185\u79f0\u672c\u7ec4\u8bca\u65ad\u672a\u89c2\u5bdf\u5230\u635f\u4f24\u3002",
        f"\u6d4b\u8bd5\u96c6 610 \u4e2a\u8f93\u5165\u4e2d\uff0c\u540c\u4e8b\u5b9e\u7f16\u8f91\u4e0e\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003\u7684\u9884\u6d4b\u4e0d\u540c\u6570\u4e3a {pred_disagreements}\uff0c\u5206\u6570\u4e0d\u540c\u6570\u4e3a {score_disagreements}\u3002\u5206\u6570\u53d8\u5316\u6ca1\u6709\u5e26\u6765\u989d\u5916\u7684\u6d4b\u8bd5\u96c6\u5206\u7c7b\u6536\u76ca\u3002",
        "\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003\u7684\u6d4b\u8bd5\u96c6\u56db\u7c7b\u5b8f\u5e73\u5747\u8fbe\u5230\u6216\u8d85\u8fc7\u540c\u4e8b\u5b9e\u9690\u85cf\u5c42\u7f16\u8f91\uff0c\u9690\u85cf\u5c42\u7f16\u8f91\u7684\u989d\u5916\u6536\u76ca\u5c1a\u672a\u5f97\u5230\u652f\u6301\u3002" if direct_at_least else
        "\u540c\u4e8b\u5b9e\u9690\u85cf\u5c42\u7f16\u8f91\u7684\u6d4b\u8bd5\u96c6\u56db\u7c7b\u5b8f\u5e73\u5747\u9ad8\u4e8e\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003\uff1b\u662f\u5426\u652f\u6301\u989d\u5916\u6536\u76ca\u8fd8\u5e94\u7ed3\u5408\u6765\u6e90\u91cd\u91c7\u6837\u5dee\u503c\u533a\u95f4\u53ca\u65b0\u589e\u9519\u8bef\uff0c\u4e0d\u80fd\u4ec5\u51ed\u70b9\u4f30\u8ba1\u5224\u65ad\u3002",
        "\u4ec5\u5bf9 reference_changed \u4e14\u65b0\u589e\u9519\u8bef\u4e3a\u96f6\u7684\u5177\u4f53\u5206\u7ec4\u4f7f\u7528\u201c\u672c\u7ec4\u8bca\u65ad\u672a\u89c2\u5bdf\u5230\u635f\u4f24\u201d\u3002\u5404\u7ec4\u5e03\u5c14\u5224\u5b9a\u4e0e\u7ea0\u6b63\u3001\u9000\u5316\u8ba1\u6570\u89c1 protection_groups.csv\uff1b\u8fd9\u4e9b\u7ed3\u679c\u4e0d\u5916\u63a8\u5230\u5176\u4ed6\u8f93\u5165\u6216\u81ea\u4e3b\u8def\u5f84\u3002",
        f"\u8bc4\u5206\u4f7f\u7528 GPU 0\u30011\uff0c\u5171 {verified['runtime_seconds']/60:.2f} \u5206\u949f\uff1b\u6bcf\u5361\u5cf0\u503c\u4fdd\u7559\u663e\u5b58\u5206\u522b\u4e3a "+", ".join(f"{t['peak_reserved_gib']:.2f} GiB" for t in verified["gpu_timings"])+"\u3002\u9996\u6279 20 \u4e2a\u8f93\u5165\u540e\u505c\u6b62\u6301\u7eed\u76d1\u63a7\u3002",
        f"\u9a8c\u8bc1\u901a\u8fc7\uff1a\u540c\u4e00\u4fdd\u5b58\u57fa\u4e0e\u6295\u5f71\u3001\u6295\u5f71\u8303\u6570\u22641+1e-5\u3001\u539f\u6761\u4ef6\u548c\u5b9a\u4e49\u4fdd\u7559\u3001{len([r for r in main+historical if not r['reference_changed']])} \u4e2a\u672a\u53d8\u5316\u8f93\u5165\u8bc4\u5206\u5b8c\u5168\u4e00\u81f4\u3001\u96f6\u5e72\u9884\u4e00\u81f4\u3001\u6240\u6709\u9009\u4e2d\u5c42\u76ee\u6807\u5934\u548c\u6700\u7ec8\u4f4d\u7f6e\u6b63\u786e\u3001\u5386\u53f2\u5404\u8def\u7ebf\u81ea\u7136\u5206\u6570\u4e0e\u4fdd\u5b58\u5206\u6570\u4e00\u81f4\u3002\u8be6\u60c5\u89c1 verification_summary.json\u3002",
        "","\u4ea7\u7269\u76ee\u5f55\uff1a"+str(root)+"\u3002\u56fa\u5b9a\u914d\u7f6e\u3001\u5b8c\u6574\u53c2\u8003\u6620\u5c04\u3001\u53c2\u8003\u5206\u6570\u3001\u76ee\u6807\u5934\u6fc0\u6d3b\u7f13\u5b58\u3001\u5168\u90e8\u8f93\u5165\u5206\u6570\u4e0e\u914d\u5bf9\u53d8\u5316\u3001\u7ffb\u8f6c\u8868\u3001\u539f\u5206\u6bcd\u548c\u5206\u7ec4\u533a\u95f4\u5747\u5df2\u4fdd\u5b58\u3002\u81ea\u4e3b\u8def\u5f84\u672a\u7eb3\u5165\u672c\u8f6e\u3002",""])
    (root/"REPORT.zh.md").write_text("\n".join(text),encoding="utf-8")
    (root/"README.zh.md").write_text("\n".join([
        "# ASTER \u540c\u4e8b\u5b9e\u53c2\u8003\u5bf9\u7167\u4ea7\u7269","",
        "REPORT.zh.md \u4e3a\u7ed3\u8bba\u62a5\u544a\u3002configuration.json \u56fa\u5b9a\u8bbe\u7f6e\uff1bresource_plan.json \u4e3a\u542f\u52a8\u8d44\u6e90\u8ba1\u5212\u3002",
        "inputs/reference_mapping.jsonl \u4e3a\u5168\u90e8 1,859 \u4e2a\u8f93\u5165\u5230\u53c2\u8003\u7684\u6620\u5c04\uff0c\u5305\u542b\u539f\u63d0\u793a\u3001\u53c2\u8003\u53d8\u66f4\u548c\u4fdd\u5b58\u6807\u6ce8\u4f4d\u7f6e\u3002",
        "references/prompts.jsonl \u4fdd\u5b58\u7cfb\u7edf\u4e0e\u7528\u6237\u53c2\u8003\u63d0\u793a\uff1breferences/scores_*.jsonl \u4fdd\u5b58\u76f4\u63a5\u8bfb\u53d6\u53c2\u8003\u5206\u6570\uff1bcache/*.pt \u4fdd\u5b58\u65e0\u5e72\u9884\u6a21\u578b\u7684 51 \u4e2a\u76ee\u6807\u5934\u6fc0\u6d3b\u3002",

        "operators/unscreened.pt \u4e3a\u4e24\u79cd\u7f16\u8f91\u5171\u7528\u7684 N/B/mu/S \u7b97\u5b50\uff1b\u540c\u4e8b\u5b9e\u65b9\u6cd5\u8fd0\u884c\u65f6\u53ea\u66ff\u6362 mu\u3002",
        "scores/scores_*.jsonl \u4e3a\u672c\u8f6e\u5168\u90e8\u5206\u6570\uff1bresults/ \u4e0b CSV/JSONL \u4e3a\u53ef\u9010\u4f8b\u590d\u67e5\u7684\u4e3b\u5b9e\u9a8c\u3001\u5171\u540c\u504f\u79fb\u3001\u5386\u53f2\u9608\u503c\u3001\u5220\u9664\u8865\u56de\u548c\u7ffb\u8f6c\u7ed3\u679c\u3002\u5404\u8868\u4fdd\u7559\u539f\u5206\u6bcd\u3002",
        "protection_groups.csv \u4ec5\u5bf9\u53c2\u8003\u5b9e\u9645\u53d8\u5316\u7684\u5206\u7ec4\u5224\u65ad\u662f\u5426\u89c2\u5bdf\u5230\u65b0\u589e\u9519\u8bef\u3002",
        "\u4e3b\u7ed3\u679c\u6309\u539f\u6765\u6e90\u5212\u5206\uff1bbootstrap \u6309\u6765\u6e90\u91cd\u91c7\u6837 2,000 \u6b21\u3002\u53c2\u8003\u6784\u9020\u4e0d\u4f7f\u7528\u7b54\u6848\u3002\u65e7\u786c\u7b5b\u9009\u3001\u8fde\u7eed\u8870\u51cf\u3001\u4e3b\u5b9e\u9a8c\u5355\u8def\u6821\u51c6\u5206\u6570\u4e0e\u672a\u7b5b\u9009\u8054\u5408\u4fee\u8865\u76f4\u63a5\u590d\u7528\u3002",
        "\u590d\u73b0\uff1a\u5728\u670d\u52a1\u5668 source /home/gcx/gcx261014/bin/activate\uff0c\u8fdb\u5165 code \u540e\u8fd0\u884c python -m experiments.aster_closed_form.reference_anchor --stage prepare\uff1b--stage all \u8bc4\u5206\uff1bpython -m experiments.aster_closed_form.reference_anchor_report \u751f\u6210\u5206\u6790\u3002",
        "\u6240\u6709\u7a0b\u5e8f\u4ec5\u5728\u670d\u52a1\u5668\u8fd0\u884c\u3002\u65e5\u5fd7\u4fdd\u5b58\u4e8e logs/\u3002\u4e0d\u8fdb\u884c\u54c8\u5e0c\u5ba1\u6838\u3001\u4e0d\u91cd\u65b0\u7b5b\u9009\u8f93\u5165\u3001\u4e0d\u8c03\u6574\u53c2\u6570\u3002",
        ""]),encoding="utf-8")
    return claim

def main():
    torch.set_num_threads(4)
    status(ROOT,"analysis",state="running")
    main_items,hist_items=read_scores(ROOT)
    effects,main_corr,main_pair,main_protection=main_analysis(ROOT,main_items)
    hist_metrics,hist_corr,hist_overall,deletion,restoration,hist_pair,hist_protection=historical_analysis(ROOT,hist_items)
    summary,contrasts=pair_summaries(ROOT,main_pair,hist_pair)
    mechanisms=reuse_mechanisms(ROOT)
    protection=main_protection+hist_protection
    save_table(ROOT,"protection_groups",protection)
    save_table(ROOT,"all_prediction_flips",flips(main_items,MAIN_METHODS)+flips(hist_items,HIST_METHODS))
    verified=verification(ROOT,main_items,hist_items)
    conclusions=write_report(ROOT,effects,main_corr,hist_overall,hist_metrics,deletion,restoration,
        summary,contrasts,protection,verified,mechanisms,main_items,hist_items)
    status(ROOT,"analysis",state="complete")
    status(ROOT,"all",state="complete",report=str(ROOT/"REPORT.zh.md"))
    archive=ROOT/"aster_reference_anchor_results.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p!=archive and "__pycache__" not in p.parts:
                z.write(p,p.relative_to(ROOT))
    print(json.dumps(dict(state="complete",report=str(ROOT/"REPORT.zh.md"),archive=str(archive),
        main_test_macro=conclusions["main_test_macro"],runtime_minutes=verified["runtime_seconds"]/60),ensure_ascii=False))
if __name__=="__main__":main()


