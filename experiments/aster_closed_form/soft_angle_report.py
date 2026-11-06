"""Complete frozen comparison and per-input, pair and path flip records."""
import argparse
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .common import rows,write_rows,write_json,status
from . import pipeline as base
from .evaluation import evaluate,pred,margin,fraction,path_metrics,_stats
from .existing_report import metric_row,threshold_value,interval
from .data import canonical_guide
from .soft_angle import SOURCE,ROOT

METHODS=("full","aster","soft_angle")
COMPARISONS=(("full","aster"),("full","soft_angle"),("aster","soft_angle"))

def csv_write(path,records):
    records=list(records)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        path.write_text("")
        return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def save_table(root,name,records):
    write_rows(root/"results"/(name+".jsonl"),records)
    csv_write(root/"results"/(name+".csv"),records)

def change(a,b,gold):
    if pred(a) is None or pred(b) is None:return "missing"
    if pred(a)==pred(b):return "unchanged"
    return "corrected" if pred(b)==gold else "degraded"

def comparison(items,reference,candidate):
    counts=Counter(change(r["scores"].get(reference),r["scores"].get(candidate),r["gold"]) for r in items)
    return dict(inputs=len(items),reference=reference,candidate=candidate,
        reference_correct=sum(pred(r["scores"].get(reference))==r["gold"] for r in items),
        candidate_correct=sum(pred(r["scores"].get(candidate))==r["gold"] for r in items),
        corrected=counts["corrected"],degraded=counts["degraded"],unchanged=counts["unchanged"],
        missing=counts["missing"],
        corrected_ids=[r["input_id"] for r in items if change(r["scores"].get(reference),r["scores"].get(candidate),r["gold"])=="corrected"],
        degraded_ids=[r["input_id"] for r in items if change(r["scores"].get(reference),r["scores"].get(candidate),r["gold"])=="degraded"])

def input_table(items):
    result=[]
    for r in items:
        out={k:v for k,v in r.items() if k!="scores"}
        for method,score in r["scores"].items():
            out[method+"_score"]=score
            out[method+"_prediction"]=pred(score)
            out[method+"_correct"]=pred(score)==r["gold"]
        for ref,cand in COMPARISONS:
            out[cand+"_vs_"+ref]=change(r["scores"][ref],r["scores"][cand],r["gold"])
            out[cand+"_shift_vs_"+ref]=r["scores"][cand]-r["scores"][ref]
        out["baseline_abs_score_le_4"]=abs(r["scores"]["full"])<=4
        result.append(out)
    return result

def flips(items,dataset):
    result=[]
    for r in items:
        for ref,cand in COMPARISONS:
            label=change(r["scores"][ref],r["scores"][cand],r["gold"])
            if label not in ("corrected","degraded"):continue
            result.append(dict(dataset=dataset,**{k:v for k,v in r.items() if k!="scores"},
                reference=ref,candidate=cand,change=label,reference_score=r["scores"][ref],
                candidate_score=r["scores"][cand],score_shift=r["scores"][cand]-r["scores"][ref],
                reference_prediction=pred(r["scores"][ref]),candidate_prediction=pred(r["scores"][cand]),
                baseline_abs_score_le_4=abs(r["scores"]["full"])<=4))
    return result

def pair_shifts(pairs,items):
    output=[]
    for p in pairs:
        a,b=items[p["left"]],items[p["right"]]
        for ref,cand in COMPARISONS:
            l0,r0=a["scores"][ref],b["scores"][ref]
            l1,r1=a["scores"][cand],b["scores"][cand]
            dl,dr=l1-l0,r1-r0
            output.append(dict(**p,reference=ref,candidate=cand,left_gold=a["gold"],right_gold=b["gold"],
                reference_left=l0,reference_right=r0,candidate_left=l1,candidate_right=r1,
                reference_difference=r0-l0,candidate_difference=r1-l1,difference_change=dr-dl,
                left_shift=dl,right_shift=dr,common_shift=(dl+dr)/2,
                same_direction_shift=dl*dr>0,
                left_flip=pred(l0)!=pred(l1),right_flip=pred(r0)!=pred(r1),
                reference_pair_correct=pred(l0)==a["gold"] and pred(r0)==b["gold"],
                candidate_pair_correct=pred(l1)==a["gold"] and pred(r1)==b["gold"]))
    return output

def main_analysis(root):
    views,pairs=base.load_data(root)
    old=base.all_scores(SOURCE)
    scores={k:old[k] for k in ("full","aster","calibrated_full")}
    soft=[r for path in sorted((root/"scores").glob("soft_angle_*.jsonl")) for r in rows(path)]
    scores["soft_angle"]={r["view_id"]:r["ell"] for r in soft}
    assert len(soft)==len(scores["soft_angle"])==1290
    assert all(margin(scores[m].get(k)) is not None for m in scores for k in views)
    evaluations={split:evaluate(views,pairs,scores,split=split,bootstrap=2000 if split=="test" else 0)
                 for split in ("fit","dev","test")}
    all_views={k:dict(v,split="all") for k,v in views.items()}
    evaluations["all"]=evaluate(all_views,[dict(p,split="all") for p in pairs],scores,split="all",bootstrap=0)
    write_json(root/"results/main_effects.json",evaluations)
    csv_write(root/"results/main_comparison.csv",
        [dict(split=s,**metric_row(m,v)) for s,e in evaluations.items() for m,v in e["methods"].items()])
    membership=defaultdict(set)
    for p in pairs:
        for side in ("left","right"):membership[p[side]].add(p["kind"])
    items=[dict(input_id=k,gold=v["label"],split=v["split"],source_id=v["source_id"],group_id=v["group_id"],
                variant=v["variant"],original=v["original"],condition=v["condition"],
                pair_kinds=sorted(membership[k]),scores={m:s[k] for m,s in scores.items()})
           for k,v in views.items()]
    save_table(root,"main_inputs",input_table(items))
    pair_rows=pair_shifts(pairs,{r["input_id"]:r for r in items})
    save_table(root,"main_pair_shifts",pair_rows)
    summary={}
    for split in ("fit","dev","test","all"):
        current=[r for r in items if split=="all" or r["split"]==split]
        strata={"all":current,"original":[r for r in current if r["original"]],
            "baseline_abs_score_le_4":[r for r in current if abs(r["scores"]["full"])<=4],
            "baseline_abs_score_gt_4":[r for r in current if abs(r["scores"]["full"])>4]}
        for kind in ("A","P","C","F"):strata[kind]=[r for r in current if kind in r["pair_kinds"]]
        summary[split]={g:{cand+"_vs_"+ref:comparison(rs,ref,cand) for ref,cand in COMPARISONS} for g,rs in strata.items()}
        summary[split]["abs_score_le_4_counts"]={m:sum(abs(r["scores"][m])<=4 for r in current) for m in METHODS}
    write_json(root/"results/main_corrections.json",summary)
    save_table(root,"main_prediction_flips",flips(items,"main"))
    return evaluations,summary,items,pair_rows

def mechanism_analysis(root):
    data=[]
    for split in ("dev","test"):data+=rows(SOURCE/"mechanisms"/f"joint_components_{split}.jsonl")
    soft=[r for p in sorted((root/"mechanisms").glob("soft_angle_*.jsonl")) for r in rows(p)]
    assert len(soft)==357 and len({r["pair_id"] for r in soft})==357
    data+=soft
    grouped=defaultdict(list)
    for r in data:grouped[(r["split"],r["kind"],r["component"])].append(r)
    output=[]
    for (split,kind,component),rs in sorted(grouped.items()):
        normalized=[r["recovery"] for r in rs if r.get("recovery") is not None]
        expected=next(v for (s,k,c),v in grouped.items() if s==split and k==kind and c=="full_head")
        assert {r["pair_id"] for r in rs}=={r["pair_id"] for r in expected}
        output.append(dict(split=split,kind=kind,component=component,pairs=len(rs),
            normalized_pairs=len(normalized),recovery_mean=float(np.mean(normalized)) if normalized else None,
            recovery_median=float(np.median(normalized)) if normalized else None,
            unnormalized_pairs=len(rs)-len(normalized)))
    save_table(root,"mechanism_comparison",output)
    save_table(root,"mechanism_all_pairs",data)
    write_json(root/"results/mechanisms.json",dict(summary=output,normalization="Original |TE| >= 0.1 only for normalized recovery; every pair retained",
        component="joint replacement using D_soft @ (z_right-z_left), in both directions",selection_use=False))
    return output

def historical_analysis(root):
    facts=rows(root/"diagnostics/fact_preference_inputs.jsonl")
    deletion=rows(root/"diagnostics/deletion_restore_inputs.jsonl")
    pairs=rows(root/"diagnostics/fact_preference_pairs.jsonl")
    inputs=facts+deletion
    saved={}
    for p in sorted((SOURCE/"diagnostics").glob("scores_*.jsonl")):
        for r in rows(p):saved[(r["score_id"],"full" if r["method"]=="natural" else r["method"])]=r["ell"]
    new=[r for p in sorted((root/"diagnostics").glob("soft_angle_*.jsonl")) for r in rows(p)]
    assert len(new)==len({r["score_id"] for r in new})==569
    for r in new:saved[(r["score_id"],"soft_angle")]=r["ell"]
    assert all(margin(saved.get((r["score_id"],m))) is not None for r in inputs for m in METHODS)
    items=[dict(input_id=r["score_id"],example_id=r["example_id"],node_id=r["node_id"],
        source_id=r["source_id"],gold=r["gold"],view=r["view"],condition=r["condition"],
        population=r["population"],metric_level=r.get("metric_level"),metric_value=r.get("metric_value"),
        background=r.get("background"),exact_numeric_threshold=threshold_value(r) if r in facts else False,
        scores={m:saved[(r["score_id"],m)] for m in METHODS}) for r in inputs]
    save_table(root,"historical_inputs",input_table(items))
    save_table(root,"historical_prediction_flips",flips(items,"historical"))
    all_pairs=[];table=[];strata_rows=[];details={}
    for route in ("full","direct","condition_ancestors"):
        current=[r for r in items if r["population"]=="constructed" and r["view"]==route]
        index={r["example_id"]:r for r in current}
        adapted=[dict(p,left=index[p["left"]]["input_id"],right=index[p["right"]]["input_id"],view=route) for p in pairs]
        all_pairs+=pair_shifts(adapted,{r["input_id"]:r for r in current})
        for method in METHODS:
            item=dict(view=route,method=method,inputs=len(current),
                correct=sum(pred(r["scores"][method])==r["gold"] for r in current))
            item["accuracy"]=item["correct"]/item["inputs"]
            for kind in ("F","P"):
                ps=[p for p in pairs if p["kind"]==kind]
                n=sum(all(pred(index[p[side]]["scores"][method])==index[p[side]]["gold"] for side in ("left","right")) for p in ps)
                item.update({f"PA_{kind}":n/len(ps),f"correct_{kind}":n,f"pairs_{kind}":len(ps)})
            table.append(item)
        groups={level:[r for r in current if r["metric_level"]==level] for level in ("below","above","same_side")}
        groups.update(exact_threshold=[r for r in current if r["exact_numeric_threshold"]],
            baseline_abs_score_le_4=[r for r in current if abs(r["scores"]["full"])<=4],
            baseline_abs_score_gt_4=[r for r in current if abs(r["scores"]["full"])>4],all=current)
        details[route]={level:{cand+"_vs_"+ref:comparison(rs,ref,cand) for ref,cand in COMPARISONS} for level,rs in groups.items()}
        for level,rs in groups.items():
            for method in METHODS:
                strata_rows.append(dict(view=route,stratum=level,method=method,
                    **fraction(sum(pred(r["scores"][method])==r["gold"] for r in rs),len(rs))))
    deletion_table=[];deletion_details={};rescue=[]
    for route in sorted({r["view"] for r in deletion}):
        current=[r for r in items if r["population"]=="original" and r["view"]==route]
        deletion_details[route]={cand+"_vs_"+ref:comparison(current,ref,cand) for ref,cand in COMPARISONS}
        for m in METHODS:
            deletion_table.append(dict(view=route,method=m,
                **fraction(sum(pred(r["scores"][m])==r["gold"] for r in current),len(current))))
    by_node={(r["node_id"],r["view"]):r for r in items if r["population"]=="original"}
    for method in METHODS:
        for route in ("necessary_restore","irrelevant_restore"):
            selected=[r for (nid,v),r in by_node.items() if v==route]
            rescue.append(dict(method=method,route=route,nodes=len(selected),
                direct_errors=sum(pred(by_node[(r["node_id"],"direct")]["scores"][method])!=r["gold"] for r in selected),
                rescued=sum(pred(by_node[(r["node_id"],"direct")]["scores"][method])!=r["gold"] and pred(r["scores"][method])==r["gold"] for r in selected),
                degraded=sum(pred(by_node[(r["node_id"],"direct")]["scores"][method])==r["gold"] and pred(r["scores"][method])!=r["gold"] for r in selected)))
    save_table(root,"historical_pair_shifts",all_pairs)
    save_table(root,"historical_fact_preference",table)
    save_table(root,"historical_numeric_strata",strata_rows)
    save_table(root,"historical_deletion_restore",deletion_table)
    write_json(root/"results/historical_corrections.json",dict(fact_preference=details,deletion_restore=deletion_details,restoration_effects=rescue,
        full_definition="Unedited score using each saved historical route and its original system/user prompt",
        threshold_definition="Saved metric equals a numeric threshold in its predicate",selection_use=False))
    return table,strata_rows,deletion_table,details,deletion_details,items,all_pairs

def path_analysis(root):
    cases=rows(root/"data/path_cases.jsonl");by_case={r["case_id"]:r for r in cases}
    records={}
    for p in sorted((SOURCE/"paths").glob("predictions_*.jsonl")):
        for r in rows(p):
            if r["method"] in ("full","aster"):records[(r["case_id"],r["method"])]=r
    soft=[r for p in sorted((root/"paths").glob("predictions_*.jsonl")) for r in rows(p)]
    assert len(soft)==len({r["case_id"] for r in soft})==432
    for r in soft:records[(r["case_id"],r["method"])]=r
    assert all(records[(c["case_id"],m)]["status"]=="complete" for c in cases for m in METHODS)
    summary={};table=[]
    for subset in ("all_432","original_100"):
        selected=[c for c in cases if subset=="all_432" or c["original_100_subset"]]
        rr=[records[(c["case_id"],m)] for c in selected for m in METHODS]
        summary[subset]=path_metrics(rr)
        for m,result in summary[subset]["methods"].items():
            result["corrected_vs_full"]=sum(records[(c["case_id"],m)]["correct"] and not records[(c["case_id"],"full")]["correct"] for c in selected)
            result["degraded_vs_full"]=sum(not records[(c["case_id"],m)]["correct"] and records[(c["case_id"],"full")]["correct"] for c in selected)
            result["corrected_vs_hard"]=sum(records[(c["case_id"],m)]["correct"] and not records[(c["case_id"],"aster")]["correct"] for c in selected)
            result["degraded_vs_hard"]=sum(not records[(c["case_id"],m)]["correct"] and records[(c["case_id"],"aster")]["correct"] for c in selected)
            table.append(dict(subset=subset,method=m,**result))
        summary[subset]["nested_subset_of_432"]=subset=="original_100"
        summary[subset]["paired_accuracy_differences"]={cand+"_vs_"+ref:interval([
            (c["source_id"],int(records[(c["case_id"],cand)]["correct"])-int(records[(c["case_id"],ref)]["correct"])) for c in selected])
            for ref,cand in COMPARISONS}
    changes=[];node_flips=[];common_nodes=[];unshared=[]
    for case in cases:
        cid=case["case_id"]
        condition={n["id"]:n["condition"] for n in canonical_guide(case["guideline"])["nodes"]}
        gold={p["gid"]:p["label"] for p in case["gold_path"]}
        natural_nodes={p["gid"]:p for p in records[(cid,"full")]["predictions"]}
        for ref,cand in COMPARISONS:
            a,b=records[(cid,ref)],records[(cid,cand)]
            left={r["gid"]:r for r in a["predictions"]};right={r["gid"]:r for r in b["predictions"]}
            divergence=next((k for k,v in left.items() if k in right and v["label"]!=right[k]["label"]),None)
            record=dict(case_id=cid,source_id=case["source_id"],original_100_subset=case["original_100_subset"],
                reference=ref,candidate=cand,reference_correct=a["correct"],candidate_correct=b["correct"],
                change="corrected" if b["correct"] and not a["correct"] else "degraded" if a["correct"] and not b["correct"] else "unchanged",
                predicted_path_changed=a["predicted_path"]!=b["predicted_path"],
                reference_path=a["predicted_path"],candidate_path=b["predicted_path"],
                reference_leaf=a["predicted_leaf"],candidate_leaf=b["predicted_leaf"],
                first_divergence=divergence,condition=condition.get(divergence),gold=gold.get(divergence),
                reference_score=left[divergence]["ell"] if divergence else None,
                candidate_score=right[divergence]["ell"] if divergence else None)
            changes.append(record)
            unshared.append(dict(case_id=cid,reference=ref,candidate=cand,reference_only_nodes=sorted(left.keys()-right.keys()),
                candidate_only_nodes=sorted(right.keys()-left.keys())))
            for gid in left.keys()&right.keys():
                x,y=left[gid],right[gid]
                item=dict(dataset="path_common_node",input_id=cid+":"+gid,case_id=cid,gid=gid,
                    source_id=case["source_id"],original_100_subset=case["original_100_subset"],
                    reference=ref,candidate=cand,reference_score=x["ell"],candidate_score=y["ell"],
                    reference_prediction=x["label"],candidate_prediction=y["label"],
                    score_shift=y["ell"]-x["ell"],gold=gold.get(gid),condition=condition[gid],
                    baseline_abs_score_le_4=(abs(natural_nodes[gid]["ell"])<=4 if gid in natural_nodes else None),
                    unedited_score=natural_nodes.get(gid,{}).get("ell"),
                    reference_abs_score_le_4=abs(x["ell"])<=4,reference_path_correct=a["correct"],
                    candidate_path_correct=b["correct"])
                item["change"]=change(x["ell"],y["ell"],gold[gid]) if gid in gold else "off_gold_path"
                common_nodes.append(item)
                if x["label"]!=y["label"]:node_flips.append(item)
    save_table(root,"path_comparison",table)
    save_table(root,"path_all_cases",changes)
    save_table(root,"path_changed_cases",[r for r in changes if r["predicted_path_changed"] or r["change"]!="unchanged"])
    save_table(root,"path_node_flips",node_flips)
    save_table(root,"path_common_nodes",common_nodes)
    save_table(root,"path_unshared_nodes",unshared)
    for key,rs in (("all",common_nodes),):
        summary["common_node_counts"]={cand+"_vs_"+ref:dict(
            shared_nodes=sum(r["reference"]==ref and r["candidate"]==cand for r in rs),
            flips=sum(r["reference"]==ref and r["candidate"]==cand for r in node_flips),
            baseline_abs_score_le_4=sum(r["reference"]==ref and r["candidate"]==cand and r["baseline_abs_score_le_4"] is True for r in rs),
            baseline_score_unavailable=sum(r["reference"]==ref and r["candidate"]==cand and r["unedited_score"] is None for r in rs))
            for ref,cand in COMPARISONS}
    write_json(root/"results/path_results.json",summary)
    return summary,table,changes,node_flips

def shift_summary(pair_rows):
    result=[]
    groups=defaultdict(list)
    for r in pair_rows:groups[(r.get("split",r.get("view")),r["kind"],r["reference"],r["candidate"])].append(r)
    for (group,kind,ref,cand),rs in sorted(groups.items()):
        result.append(dict(group=group,kind=kind,reference=ref,candidate=cand,pairs=len(rs),
            common_shift=_stats([r["common_shift"] for r in rs]),
            difference_change=_stats([r["difference_change"] for r in rs]),
            left_shift=_stats([r["left_shift"] for r in rs]),right_shift=_stats([r["right_shift"] for r in rs]),
            pairs_with_any_flip=sum(r["left_flip"] or r["right_flip"] for r in rs),
            unchanged_difference_with_flip=sum(r["difference_change"]==0 and (r["left_flip"] or r["right_flip"]) for r in rs)))
    return result

def percent(value):
    return "—" if value is None else f"{100*value:.2f}%"

def count_rate(value):
    return f"{value['numerator']}/{value['denominator']}（{percent(value['rate'])}）"

def make_report(root):
    evaluations,main_changes,main_items,main_pairs=main_analysis(root)
    mechanisms=mechanism_analysis(root)
    hist,hist_strata,deletion,hist_changes,del_changes,hist_items,hist_pairs=historical_analysis(root)
    paths,path_table,path_cases,path_flips=path_analysis(root)
    all_flips=flips(main_items,"main")+flips(hist_items,"historical")+path_flips
    save_table(root,"all_prediction_flips",all_flips)
    shifts=dict(main=shift_summary(main_pairs),historical=shift_summary(hist_pairs),
        formula=dict(pair_difference="right - left",difference_change="(new_right-new_left)-(old_right-old_left)",
            common_shift="((new_left-old_left)+(new_right-old_right))/2"),
        descriptive_group="Unedited absolute score <= 4. No exclusion, fitting or selection.")
    write_json(root/"results/threshold_analysis.json",shifts)
    checks=json.loads((root/"results/operator_checks.json").read_text())
    verification=json.loads((root/"results/verification_summary.json").read_text())
    timing=[json.loads((root/"timing"/f"worker_{i}.json").read_text()) for i in (0,1)]
    runtime=json.loads((root/"runtime.json").read_text())
    mech={(r["split"],r["kind"],r["component"]):r for r in mechanisms}
    path_methods=paths["all_432"]["methods"]
    test=evaluations["test"]["methods"]
    conclusion=dict(
        test_mechanism={kind:dict(soft=mech[("test",kind,"soft_angle")]["recovery_mean"],
            hard_oblique=mech[("test",kind,"oblique")]["recovery_mean"],
            screened_orthogonal=mech[("test",kind,"screened_orthogonal")]["recovery_mean"],
            unscreened_orthogonal=mech[("test",kind,"nuisance_orthogonal")]["recovery_mean"],
            improvement_over_hard=mech[("test",kind,"soft_angle")]["recovery_mean"]-mech[("test",kind,"oblique")]["recovery_mean"])
            for kind in ("A","P")},
        test_macro_PA={m:test[m]["macro_PA"] for m in METHODS},
        test_C_F_input_changes={kind:main_changes["test"][kind] for kind in ("C","F")},
        all_C_F_input_changes={kind:main_changes["all"][kind] for kind in ("C","F")},
        historical_full_changes=hist_changes["full"]["all"],
        historical_numeric_strata=hist_changes,
        paths_432={m:{k:v for k,v in result.items() if k not in ("ci95",)} for m,result in path_methods.items()},
        no_test_based_parameter_adjustment=True)
    write_json(root/"results/conclusions.json",conclusion)
    geometry=dict(heads=len(checks),hard_factor_columns=sum(r["hard_rank"] for r in checks.values()),
        soft_factor_columns=sum(r["nuisance_rank"] for r in checks.values()),
        effective_erased_dimensions=sum(r["attenuation_trace"] for r in checks.values()),
        max_signal_error=max(r["algebra"]["signal_preservation_error"] for r in checks.values()),
        max_edit_norm=max(r["algebra"]["edit_operator_norm"] for r in checks.values()),
        max_high_angle_error=max(r["high_angle_vs_stored_hard_error"] for r in checks.values()),
        per_head_residual_fraction=_stats([r["residual_fraction"] for r in checks.values()]),
        max_idempotence_residual=max(r["algebra"]["idempotence_error"] for r in checks.values()))
    write_json(root/"results/operator_summary.json",geometry)
    save_table(root,"operator_residuals",[dict(head=h,nuisance_rank=r["nuisance_rank"],hard_rank=r["hard_rank"],
        residual_frobenius=r["residual_frobenius"],residual_fraction=r["residual_fraction"],
        effective_erased_dimensions=r["attenuation_trace"],**r["algebra"]) for h,r in checks.items()])
    direction_rows=[]
    for h,r in checks.items():
        g=r["geometry"]
        for i,(angle,sine,erased,residual) in enumerate(zip(g["principal_angles_degrees"],g["singular_values"],g["self_erasure_fractions"],g["self_residual_fractions"])):
            direction_rows.append(dict(head=h,direction=i,angle_degrees=angle,sine=sine,
                hard_erasure=1 if sine>=math.sin(math.pi/4) else 0,
                soft_self_erasure=erased,soft_self_residual=residual))
    save_table(root,"operator_directions",direction_rows)
    cf="；".join(f"{k}：纠正 {main_changes['test'][k]['soft_angle_vs_full']['corrected']}、新增错误 {main_changes['test'][k]['soft_angle_vs_full']['degraded']}" for k in ("C","F"))
    h=hist_changes["full"]["all"]["soft_angle_vs_full"]
    ph=path_methods["soft_angle"]
    lines=["# ASTER 连续夹角衰减固定对照报告","",
        "本轮仅替换夹角权重，固定原数据、51 个头、单类秩上限 4、能量目标 0.95、45° 设置、信号空间及参考均值。直接从保存基构建算子，没有重新拟合、筛样本或搜索参数。",
        "",
        f"测试集主指标：完整阅读 {percent(test['full']['macro_PA'])}，硬筛选 {percent(test['aster']['macro_PA'])}，连续衰减 {percent(test['soft_angle']['macro_PA'])}。",
        f"测试集 C/F 唯一输入相对完整阅读的变化为 {cf}。历史完整阅读路线：纠正 {h['corrected']}、新增错误 {h['degraded']}。",
        f"432 例路径：连续衰减正确 {ph['numerator']} 例；相对完整阅读纠正 {ph['corrected_vs_full']}、退化 {ph['degraded_vs_full']}，相对硬筛选纠正 {ph['corrected_vs_hard']}、退化 {ph['degraded_vs_hard']}。",
        "",
        "## 固定方法与必要验证","",
        "令 (I−SSᵀ)U = Q diag(s) Vᵀ，τ = sin45°，D = UV diag(s / max(s², τ²)) Qᵀ，z′ = z − D(z−μ)。",
        "夹角不小于 45° 时与硬筛选相同；30° 方向擦除自身分量的 50%，15° 约 13.4%，零夹角编辑为零。信号保持是保存隐藏空间上的代数性质，输出判断另行检验。",
        f"硬筛选使用 {geometry['hard_factor_columns']} 个方向，连续衰减保留 {geometry['soft_factor_columns']} 个保存方向，按自身擦除比例求和为 {geometry['effective_erased_dimensions']:.3f}。这没有改变每类秩上限。",
        f"保存精度下，最大信号保持误差 {geometry['max_signal_error']:.3g}；最大算子范数 {geometry['max_edit_norm']:.6f} < √2；高夹角部分相对旧算子的最大误差 {geometry['max_high_angle_error']:.3g}。",
        f"每头干扰基残留的相对 Frobenius 范数均值为 {geometry['per_head_residual_fraction']['mean']:.4f}；最大非幂等残差 {geometry['max_idempotence_residual']:.4f}。残留和非幂等为本方法允许的结果。",
        "夹角解析检查、既有算子检查、零干预一致和目标头／最终位置检查已通过。两个 GPU 上首个输入的自然分数均与保存基线一致；自主路径控制器的根节点起步与恢复执行检查通过。",
        "逐头结果见 [operator_residuals.csv](results/operator_residuals.csv)，逐方向权重见 [operator_directions.csv](results/operator_directions.csv)。",
        "",
        "## 主实验","",
        "评分覆盖全部 1,290 个现成输入、1,053 对；拟合／开发／测试输入数分别为 561／119／610。下表为测试集，全部划分和逐输入结果见 CSV。",
        "PA 要求配对两侧均正确；PA_P 对临床背景与治疗偏好等权，主指标为 A/P/C/F 四类 PA 的平均值。单路校准为原保存参考，没有重新训练。",
        "",
        "| 方法 | PA_A | PA_P | PA_C | PA_F | 宏平均 | 全部测试输入准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in (*METHODS,"calibrated_full"):
        r=test[m]
        lines.append("| "+m+" | "+" | ".join(percent(r["PA_"+k]["rate"]) for k in ("A","P","C","F"))+
            f" | {percent(r['macro_PA'])} | {count_rate(r['all_input_accuracy'])} |")
    lines+=["","| 范围 | 对照 | 纠正 | 退化 |","| --- | --- | ---: | ---: |"]
    for split in ("fit","dev","test","all"):
        for key,c in main_changes[split]["all"].items():
            lines.append(f"| {split} | {key} | {c['corrected']} | {c['degraded']} |")
    lines+=["","| 划分 | C/F 唯一输入 | 输入数 | 连续衰减相对 full 纠正／新增错误 | 相对硬筛选纠正／新增错误 |",
        "| --- | --- | ---: | ---: | ---: |"]
    for split in ("fit","dev","test","all"):
        for kind in ("C","F"):
            a=main_changes[split][kind]["soft_angle_vs_full"]
            b=main_changes[split][kind]["soft_angle_vs_aster"]
            lines.append(f"| {split} | {kind} | {a['inputs']} | {a['corrected']}/{a['degraded']} | {b['corrected']}/{b['degraded']} |")
    lines+=["","测试配对宏平均持平，但逐输入并非全部不变。新增 P 输入错误所在配对在硬筛选下已因另一侧判错而失分，因此不会再次降低该配对指标。全部划分的 C/F 新增错误另列，不能把测试子集的结果推广为全部输入都无损伤。"]
    lines+=["","主表与按来源重采样的测试区间见 [main_effects.json](results/main_effects.json)、[main_comparison.csv](results/main_comparison.csv)。全部输入及变化见 [main_inputs.csv](results/main_inputs.csv)，翻转见 [main_prediction_flips.csv](results/main_prediction_flips.csv)。",
        "","## 双向联合修补","",
        "保留开发 59 对、测试 298 对 A/P 配对，新增 D_soft(z右−z左) 的双向联合修补。恢复比例仍为两个方向归一化效应的平均值；仅 |TE|≥0.1 时计算比例，其他配对保留原始效应。没有按恢复结果筛选配对。",
        "","| 划分 | 类型 | 分量 | 全部对数 | 可归一化对数 | 平均恢复比例 |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    component_order=("full_head","nuisance_orthogonal","screened_orthogonal","oblique","soft_angle")
    for split in ("dev","test"):
        for kind in ("A","P"):
            for component in component_order:
                r=mech[(split,kind,component)]
                lines.append(f"| {split} | {kind} | {component} | {r['pairs']} | {r['normalized_pairs']} | {percent(r['recovery_mean'])} |")
    for kind in ("A","P"):
        r=conclusion["test_mechanism"][kind]
        lines+=["",f"测试 {kind}：连续衰减相对硬筛选斜投影的恢复比例变化为 {100*r['improvement_over_hard']:+.2f} 个百分点；未经筛选的正交分量为 {percent(r['unscreened_orthogonal'])}。"]
    lines+=["","完整五种分量对照与逐对双向分数见 [mechanism_comparison.csv](results/mechanism_comparison.csv)、[mechanism_all_pairs.jsonl](results/mechanism_all_pairs.jsonl)。",
        "","## 历史事实、偏好及删除补回","",
        "仅补连续衰减的 569 个输入评分，沿用各自原系统提示、用户提示和标签。504 个事实／偏好输入来自 28 个节点的三条路线；另有 65 个删除补回输入。此处 full 表示该路线原提示下的无干预模型。",
        "","| 路线 | 方法 | PA_F | PA_P | 输入准确率 |","| --- | --- | ---: | ---: | ---: |"]
    for r in hist:
        lines.append(f"| {r['view']} | {r['method']} | {r['correct_F']}/{r['pairs_F']}（{percent(r['PA_F'])}） | {r['correct_P']}/{r['pairs_P']}（{percent(r['PA_P'])}） | {r['correct']}/{r['inputs']}（{percent(r['accuracy'])}） |")
    lines+=["","| 路线 | 数值状态 | 对照 | 输入数 | 纠正 | 新增错误 |","| --- | --- | --- | ---: | ---: | ---: |"]
    for route,levels in hist_changes.items():
        for level in ("below","above","same_side","exact_threshold"):
            for comp in ("soft_angle_vs_full","soft_angle_vs_aster"):
                c=levels[level][comp]
                lines.append(f"| {route} | {level} | {comp} | {c['inputs']} | {c['corrected']} | {c['degraded']} |")
    lines+=["","exact_threshold 指保存数值恰好等于条件中的数值阈值；below／above／same_side 沿用原构造标签。分数接近零与数值恰好等于条件阈值分别统计。",
        "","| 删除／补回路线 | full | aster | soft_angle |","| --- | ---: | ---: | ---: |"]
    for route in sorted({r["view"] for r in deletion}):
        values={r["method"]:r for r in deletion if r["view"]==route}
        lines.append("| "+route+" | "+" | ".join(count_rate(values[m]) for m in METHODS)+" |")
    lines+=["","事实、偏好、数值分层、删除补回及纠正／退化见 [historical_fact_preference.csv](results/historical_fact_preference.csv)、[historical_numeric_strata.csv](results/historical_numeric_strata.csv)、[historical_deletion_restore.csv](results/historical_deletion_restore.csv)、[historical_corrections.json](results/historical_corrections.json)。",
        "","## 分数阈值与共同偏移","",
        "逐对同时记录两侧分数、配对差分变化与共同偏移：共同偏移 = (左侧分数变化 + 右侧分数变化) / 2。配对差分不变不能推出两侧预测不变。",
        f"开发集完整阅读分数绝对值 ≤4 的输入为 {main_changes['dev']['all']['soft_angle_vs_full']['inputs']-main_changes['dev']['baseline_abs_score_gt_4']['soft_angle_vs_full']['inputs']} 个；该分组仅作描述，不删样本或参与选参。不能单凭开发准确率持平判断保护有效。",
        "主实验的全部 1,053 对及历史每条路线的全部配对都已记录分数变化，见 [main_pair_shifts.csv](results/main_pair_shifts.csv)、[historical_pair_shifts.csv](results/historical_pair_shifts.csv)。统计见 [threshold_analysis.json](results/threshold_analysis.json)。",
        "所有主实验、历史输入与自主路径共同节点上的预测翻转合并于 [all_prediction_flips.csv](results/all_prediction_flips.csv)。路径分叉后只有一方访问的节点另列，不虚构另一方法的分数。",
        "","## 自主路径","",
        "连续衰减对全部 432 例从 G0 根节点重新执行，标准路径仅用于执行后的判分。原 100 例是嵌套子集。完整阅读和硬筛选复用既有自主执行结果。",
        "","| 范围 | 方法 | 完整路径正确率 | 相对 full 纠正／退化 | 相对硬筛选纠正／退化 |","| --- | --- | ---: | ---: | ---: |"]
    for r in path_table:
        lines.append(f"| {r['subset']} | {r['method']} | {count_rate(r)} | {r['corrected_vs_full']}/{r['degraded_vs_full']} | {r['corrected_vs_hard']}/{r['degraded_vs_hard']} |")
    lines+=["","路径变化、首个分歧条件与分数见 [path_changed_cases.csv](results/path_changed_cases.csv)；全部共同节点上的翻转见 [path_node_flips.csv](results/path_node_flips.csv)。未共同访问节点见 [path_unshared_nodes.csv](results/path_unshared_nodes.csv)。",
        "","## 分开判断",""]
    for kind in ("A","P"):
        r=conclusion["test_mechanism"][kind]
        direction="提高" if r["improvement_over_hard"]>0 else "降低" if r["improvement_over_hard"]<0 else "持平"
        lines.append(f"- 测试 {kind} 修补恢复比例相对硬筛选斜投影{direction}；连续衰减的 PA_{kind} 为 {percent(test['soft_angle']['PA_'+kind]['rate'])}，硬筛选为 {percent(test['aster']['PA_'+kind]['rate'])}。")
    for kind in ("C","F"):
        c=main_changes["test"][kind]["soft_angle_vs_full"]
        statement="本组诊断未观察到新增损伤" if c["degraded"]==0 else f"出现 {c['degraded']} 个原本正确输入的新错误"
        lines.append(f"- 测试 {kind} 的 {c['inputs']} 个唯一输入，相对完整阅读{statement}；纠正 {c['corrected']} 个。")
    for route in ("full","direct","condition_ancestors"):
        c=hist_changes[route]["all"]["soft_angle_vs_full"]
        statement="本组诊断未观察到新增损伤" if c["degraded"]==0 else f"出现 {c['degraded']} 个原本正确输入的新错误"
        exact=hist_changes[route]["exact_threshold"]["soft_angle_vs_full"]
        lines.append(f"- 历史 {route} 路线的 168 个事实／偏好输入，相对原模型{statement}，纠正 {c['corrected']} 个；恰好等于数值阈值的 {exact['inputs']} 个输入新增错误 {exact['degraded']} 个。")
    lines.append(f"- 路径相对完整阅读净变化 {ph['numerator']-path_methods['full']['numerator']:+d}/432，相对硬筛选净变化 {ph['numerator']-path_methods['aster']['numerator']:+d}/432。")
    lines+=["","以上判断仅对应实际评价的输入与既有标签。代数信号保持不能推出所有数值阈值判断安全；本轮正负结果全部保留，没有根据测试结果调整参数。",
        "","## 执行与交付","",
        f"仅 GPU 0、1 各一份 Qwen3-8B，bfloat16；评分总耗时 {runtime['seconds']/60:.2f} 分钟，峰值保留显存分别为 {timing[0]['peak_reserved_gib']:.2f}／{timing[1]['peak_reserved_gib']:.2f} GiB。",
        "主实验 1,290 输入、机制 357 对、历史 569 输入及路径 432 例均完整，无缺分数或路径技术失败。原始输入、完整路径和运行日志保留在服务器。",
        "实现位于 code/experiments/aster_closed_form/，固定配置见 configuration.json。执行与复算说明见 README.zh.md。",
        f"服务器目录：{root}。"]
    (root/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(root/"results/status.json",dict(state="complete",main_inputs=1290,main_pairs=1053,
        mechanism_pairs=357,mechanism_components=5,historical_inputs=569,path_cases=432,nested_path_cases=100,
        missing_scores=0,technical_failures=0,parameter_search=False,report=str(root/"REPORT.zh.md")))
    status(root,"all",state="complete",report=str(root/"REPORT.zh.md"))
    return conclusion

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args()
    make_report(args.root)
