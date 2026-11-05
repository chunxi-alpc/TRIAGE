"""Q_auto, paired masking, fact response, and development-selected binary fusion."""
import argparse
import collections
import json
import math
from common import CONFIG, DATA, RESULTS, fraction, read_rows, sigmoid, write_json, write_rows

def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None

def correctness(rows, view):
    selected=[r for r in rows if view in r["scores"]]
    return fraction(sum(r["scores"][view]["prediction"]==r["gold"] for r in selected),len(selected))

def examples(stage):
    inputs={r["score_id"]:r for r in read_rows(DATA/f"inputs_{stage}.jsonl")}
    output={}
    failures=[]
    for shard in (0,1):
        for score in read_rows(RESULTS/f"scores_{stage}_{shard}.jsonl"):
            if score["score_id"] not in inputs:
                continue
            if score["status"]!="complete":
                failures.append(score)
                continue
            source=inputs[score["score_id"]]
            if source["example_id"] not in output:
                output[source["example_id"]]=dict(source,scores={})
            output[source["example_id"]]["scores"][source["view"]]=score
    return list(output.values()),failures

def q_auto(stage):
    nodes=[r for r in read_rows(DATA/"nodes.jsonl")
           if r["role"]=="statistical" and (stage=="full" or r["pilot"])]
    annotations={r["node_id"]:r for r in read_rows(DATA/"annotations.jsonl")}
    rows=[]
    for node in nodes:
        a=annotations.get(node["node_id"],{})
        value=a.get("annotation",{})
        rows.append(dict(node_id=node["node_id"],publisher=node["publisher"],
            source_id=node["source_id"],traceable=a.get("validation",{}).get("source_traceable",False),
            dependency=value.get("source_semantic_dependency"),
            dependency_evidence_valid=a.get("validation",{}).get("dependency_evidence_valid",False),
            mixed=value.get("source_condition_action_mixed"),
            dataset_completed=value.get("dataset_definition_completed"),
            annotation_status=a.get("status","missing"),
            original_case_available=node["original_case"] is not None,
            unresolved_references=a.get("unresolved_references",[])))
    summaries={}
    for label,subset in [("all",rows)]+[(p,[r for r in rows if r["publisher"]==p]) for p in ("NCCN","CMA")]:
        trace=[r for r in subset if r["traceable"]]
        known=[r for r in trace if type(r["dependency"]) is bool and r["dependency_evidence_valid"] and not r["unresolved_references"]]
        dependent=[r for r in known if r["dependency"]]
        mixed_known=[r for r in trace if type(r["mixed"]) is bool]
        completion_known=[r for r in dependent if type(r["dataset_completed"]) is bool]
        summaries[label]=dict(sampled_nodes=len(subset),traceable_nodes=len(trace),
            Q_auto=fraction(len(dependent),len(known)),
            mixed_condition_action=fraction(sum(r["mixed"] for r in mixed_known),len(mixed_known)),
            dataset_completed_definition=fraction(sum(r["dataset_completed"] for r in completion_known),len(completion_known)),
            source_evidence_missing=fraction(len(subset)-len(trace),len(subset)),
            dependency_unknown=fraction(len(trace)-len(known),len(trace)),
            original_case_missing=fraction(sum(not r["original_case_available"] for r in subset),len(subset)))
    write_rows(RESULTS/f"q_nodes_{stage}.jsonl",rows)
    return summaries

def paired_accuracy(rows, first, second):
    selected=[r for r in rows if first in r["scores"] and second in r["scores"]]
    ca=[r["scores"][first]["prediction"]==r["gold"] for r in selected]
    cb=[r["scores"][second]["prediction"]==r["gold"] for r in selected]
    return dict(n=len(selected),first=first,second=second,
        first_accuracy=mean(ca),second_accuracy=mean(cb),
        second_minus_first=mean(int(b)-int(a) for a,b in zip(ca,cb)),
        correct_to_wrong=fraction(sum(a and not b for a,b in zip(ca,cb)),len(selected)),
        wrong_to_correct=fraction(sum(not a and b for a,b in zip(ca,cb)),len(selected)),
        prediction_changed=fraction(sum(r["scores"][first]["prediction"]!=r["scores"][second]["prediction"] for r in selected),len(selected)),
        mean_absolute_probability_change=mean(abs(r["scores"][first]["p_yes"]-r["scores"][second]["p_yes"]) for r in selected))

def masking_metrics(rows):
    original=[r for r in rows if r["population"]=="original" and r["role"]=="statistical" and r["source_traceable"]]
    groups={"all":original}
    for dep in (True,False):
        groups["dependent" if dep else "independent"]=[r for r in original if r["source_semantic_dependency"] is dep]
    groups["definition_already_completed"]=[r for r in original if r["dataset_definition_completed"] is True]
    groups["direct_information_insufficient"]=[r for r in original if r["direct_information_sufficient"] is False]
    groups["direct_information_sufficient"]=[r for r in original if r["direct_information_sufficient"] is True]
    groups["full_information_sufficient"]=[r for r in original if r["full_information_sufficient"] is True]
    result={}
    for name,subset in groups.items():
        direct=paired_accuracy(subset,"full","direct")
        direct["A_accuracy_drop"]=None if direct["second_minus_first"] is None else -direct["second_minus_first"]
        restore=[r for r in subset if r.get("changes",{}).get("restoration_available")]
        matched=[r for r in restore if r["changes"]["matched_irrelevant_available"]]
        result[name]=dict(n=len(subset),full_vs_direct=direct,
            full_vs_conservative=paired_accuracy(subset,"full","conservative"),
            necessary_recovery=paired_accuracy(restore,"direct","necessary_restore"),
            irrelevant_recovery=paired_accuracy(matched,"direct","irrelevant_restore"),
            necessary_vs_irrelevant=paired_accuracy(matched,"irrelevant_restore","necessary_restore"),
            full_vs_condition_ancestors=paired_accuracy(subset,"full","condition_ancestors"))
    disagreements=[r for r in original if r.get("teacher_answer") in ("yes","no")
                   and int(r["teacher_answer"]=="yes")!=r["gold"]]
    result["label_disagreement"]=dict(n=len(disagreements),original_n=len(original),
                                     example_ids=[r["example_id"] for r in disagreements],
                                     handling="Kept original dataset labels; not outcome-filtered")
    return result

def effect_pairs(rows, factor, left, right, view):
    others=[k for k in ("metric_level","background","wording") if k!=factor]
    groups=collections.defaultdict(dict)
    for row in rows:
        if view in row["scores"]:
            groups[(row["node_id"],*(row[k] for k in others))][row[factor]]=row
    pairs=[(g[left],g[right]) for g in groups.values() if left in g and right in g]
    delta=[b["scores"][view]["p_yes"]-a["scores"][view]["p_yes"] for a,b in pairs]
    return dict(n=len(pairs),prediction_changed=fraction(sum(a["scores"][view]["prediction"]!=b["scores"][view]["prediction"] for a,b in pairs),len(pairs)),
        accuracy_difference=mean(int(b["scores"][view]["prediction"]==b["gold"])-int(a["scores"][view]["prediction"]==a["gold"]) for a,b in pairs),
        mean_probability_difference=mean(delta),mean_absolute_probability_difference=mean(abs(x) for x in delta))

def fact_metrics(rows):
    rows=[r for r in rows if r["population"]=="constructed"]
    result={}
    for view in ("direct","condition_ancestors","full"):
        valid=[r for r in rows if view in r["scores"]]
        if view=="condition_ancestors":
            valid=[r for r in valid if r["ancestors_information_sufficient"] is True]
        groups=collections.defaultdict(dict)
        for row in valid:
            groups[(row["node_id"],row["background"],row["wording"])][row["metric_level"]]=row
        cross=[(g["below"],g["above"]) for g in groups.values() if "below" in g and "above" in g]
        same=[(g["above"],g["same_side"]) for g in groups.values() if "above" in g and "same_side" in g]
        wrong=lambda r:r["scores"][view]["prediction"]!=r["gold"]
        errors=sum(wrong(a) or wrong(b) for a,b in cross)
        directed=sum((b["scores"][view]["p_yes"]-a["scores"][view]["p_yes"])*(b["gold"]-a["gold"])>0 for a,b in cross)
        both=sum(not wrong(a) and not wrong(b) for a,b in cross)
        result[view]=dict(nodes=len({r["node_id"] for r in valid}),
            P_b=fraction(errors,len(cross)),
            single_case_error=fraction(sum(wrong(r) for r in valid),len(valid)),
            crossing_both_correct=fraction(both,len(cross)),
            probability_changes_in_correct_direction=fraction(directed,len(cross)),
            crossing_prediction_changed=fraction(sum(a["scores"][view]["prediction"]!=b["scores"][view]["prediction"] for a,b in cross),len(cross)),
            noncrossing_prediction_stable=fraction(sum(a["scores"][view]["prediction"]==b["scores"][view]["prediction"] for a,b in same),len(same)),
            noncrossing_both_correct=fraction(sum(not wrong(a) and not wrong(b) for a,b in same),len(same)),
            background_effect=effect_pairs(valid,"background","neutral","cue",view),
            wording_effect=effect_pairs([r for r in valid if r["wording_changed"]],"wording","original","neutral",view),
            wording_noop_nodes=len({r["node_id"] for r in valid if not r["wording_changed"]}),
            P_b_strata={})
        for background in ("neutral","cue"):
            for wording in ("original","neutral"):
                pairs=[(a,b) for a,b in cross if a["background"]==background and a["wording"]==wording]
                result[view]["P_b_strata"][background+"/"+wording]=fraction(sum(wrong(a) or wrong(b) for a,b in pairs),len(pairs))
    result["downstream_action_exchange"]=paired_accuracy(rows,"full","action_swap")
    return result

def fusion_probability(method, parameter, full, blind):
    if method=="probability_average":
        return parameter*full["p_yes"]+(1-parameter)*blind["p_yes"]
    if method=="cad_full":
        return sigmoid((1+parameter)*full["margin"]-parameter*blind["margin"])
    if method=="cad_blind":
        return sigmoid((1+parameter)*blind["margin"]-parameter*full["margin"])
    raise ValueError(method)

def fusion_summary(rows, probabilities, blind):
    correct=[int(p>=0.5)==r["gold"] for r,p in zip(rows,probabilities)]
    both=[r["scores"]["full"]["prediction"]!=r["gold"] and r["scores"][blind]["prediction"]!=r["gold"] for r in rows]
    repaired=sum(b and c for b,c in zip(both,correct))
    damaged=sum(not b and not c for b,c in zip(both,correct))
    return dict(n=len(rows),accuracy=fraction(sum(correct),len(rows)),
                B=fraction(sum(both),len(rows)),
                double_wrong_correction=fraction(repaired,sum(both)),
                new_error_among_at_least_one_correct=fraction(damaged,len(rows)-sum(both)),
                new_error_among_all=fraction(damaged,len(rows)))

def fit_logistic(development,blind):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    x=np.array([[r["scores"]["full"]["margin"],r["scores"][blind]["margin"]] for r in development])
    y=np.array([r["gold"] for r in development])
    groups=np.array([r["source_id"] for r in development])
    if len(set(y))<2:
        return None,dict(status="unavailable",reason="development labels contain only one class")
    n_groups=len(set(groups))
    if n_groups<2:
        return None,dict(status="unavailable",reason="fewer than two development sources")
    cv=GroupKFold(n_splits=min(5,n_groups)); candidates=[]
    for c in CONFIG["logistic_c"]:
        hits=0; total=0; folds=[]
        for train,validation in cv.split(x,y,groups):
            if len(set(y[train]))<2:
                continue
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=CONFIG["seed"]))
            model.fit(x[train],y[train])
            pred=model.predict(x[validation])
            hits+=int((pred==y[validation]).sum());total+=len(validation)
            folds.append(dict(training_sources=sorted(set(groups[train])),validation_sources=sorted(set(groups[validation]))))
        candidates.append(dict(C=c,correct=hits,n=total,accuracy=hits/total if total else -1))
    best=max(candidates,key=lambda r:r["accuracy"])
    if best["accuracy"]<0:
        return None,dict(status="unavailable",reason="No valid source-grouped development fold")
    model=make_pipeline(StandardScaler(),LogisticRegression(C=best["C"],max_iter=3000,random_state=CONFIG["seed"]))
    model.fit(x,y)
    scale=model.named_steps["standardscaler"];reg=model.named_steps["logisticregression"]
    info=dict(status="fit",C=best["C"],cv_candidates=candidates,
              selection="GroupKFold within development sources only",folds=folds,
              feature_order=["full_log_odds",blind+"_log_odds"],
              feature_mean=scale.mean_.tolist(),feature_scale=scale.scale_.tolist(),
              coefficients=reg.coef_.tolist(),intercept=reg.intercept_.tolist(),
              development_sources=sorted(set(groups)),development_n=len(y))
    return model,info

def fusion_metrics(rows,blind,stage):
    paired=[r for r in rows if "full" in r["scores"] and blind in r["scores"]]
    dev=[r for r in paired if r["split"]=="development"]
    test=[r for r in paired if r["split"]=="test"]
    if not dev:
        return dict(status="unavailable",reason="No paired development scores")
    chosen={};dev_results={}
    for method,grid in (("probability_average",CONFIG["weights"]),("cad_full",CONFIG["alphas"]),("cad_blind",CONFIG["alphas"])):
        candidates=[]
        for parameter in grid:
            probabilities=[fusion_probability(method,parameter,r["scores"]["full"],r["scores"][blind]) for r in dev]
            summary=fusion_summary(dev,probabilities,blind)
            candidates.append(dict(parameter=parameter,accuracy=summary["accuracy"]["rate"]))
        best=max(candidates,key=lambda x:x["accuracy"])
        chosen[method]=best["parameter"];dev_results[method]=candidates
    model,logistic=fit_logistic(dev,blind)
    selection=dict(blind_view=blind,parameters=chosen,development_candidates=dev_results,logistic=logistic,
        selection_scope="All development examples pooled; held-out test labels are never used",
        tie_break="first parameter in ascending configured grid",
        development_sources=sorted({r["source_id"] for r in dev}),
        test_sources=sorted({r["source_id"] for r in test}))
    write_json(RESULTS/f"fusion_selection_{stage}_{blind}.json",selection)
    evaluated=[]
    # Pilot numbers are explicitly development-only; never label them held-out performance.
    target=test if stage=="full" else dev
    import numpy as np
    for row in target:
        scores=dict(full=row["scores"]["full"]["p_yes"],blind=row["scores"][blind]["p_yes"])
        for method,parameter in chosen.items():
            scores[method]=fusion_probability(method,parameter,row["scores"]["full"],row["scores"][blind])
        if model is not None:
            x=np.array([[row["scores"]["full"]["margin"],row["scores"][blind]["margin"]]])
            scores["logistic"]=float(model.predict_proba(x)[0,1])
        evaluated.append(dict(example_id=row["example_id"],node_id=row["node_id"],source_id=row["source_id"],
            population=row["population"],split=row["split"],gold=row["gold"],probabilities=scores,
            direct_information_sufficient=row["direct_information_sufficient"]))
    write_rows(RESULTS/f"fusion_predictions_{stage}_{blind}.jsonl",evaluated)
    populations={}
    for population in ("original","constructed"):
        idx=[i for i,r in enumerate(target) if r["population"]==population]
        subset=[target[i] for i in idx]
        methods=set().union(*(evaluated[i]["probabilities"].keys() for i in idx)) if idx else []
        populations[population]={method:fusion_summary(subset,[evaluated[i]["probabilities"][method] for i in idx],blind)
                                 for method in sorted(methods)}
    return dict(status="complete",evaluation_split="test" if stage=="full" else "development_pilot",
                selection=selection,populations=populations,
                input_pair="full/"+blind,paired_dev=len(dev),paired_test=len(test))

def percent(value):
    return "NA" if value is None else f"{100*value:.2f}%"

def report(stage,metrics):
    q=metrics["Q_auto"]["all"]; lines=[
        "# 自动预实验：信息丢失、关键事实响应和概率融合",
        "",
        "本报告由程序自动生成，原文关联、语义依赖及病例改写未经人工复核。",
        "原始病例指 MEGA-CDP 已生成的病例；构造病例均标为“基于真实指南构造”，不代表自然发生频率。",
        f"阶段：{stage}；完整打分 {metrics['complete_score_count']} 条；失败记录 {metrics['failed_score_count']} 条。",
        "",
        f"Q_auto = {q['Q_auto']['numerator']}/{q['Q_auto']['denominator']} = {percent(q['Q_auto']['rate'])}。",
        f"抽样统计节点 {q['sampled_nodes']}；原文依据不可回溯比例 {percent(q['source_evidence_missing']['rate'])}。",
        "混写比例与语义依赖比例分别统计；数据集已补全定义的节点另列。",
        "",
        "| 节点组 | 配对数 | 完整准确率 | 直接遮挡准确率 | A（准确率下降） |",
        "|---|---:|---:|---:|---:|"]
    for group,data in metrics["A"].items():
        if "full_vs_direct" not in data:continue
        r=data["full_vs_direct"]
        lines.append(f"| {group} | {r['n']} | {percent(r['first_accuracy'])} | {percent(r['second_accuracy'])} | {percent(r['A_accuracy_drop'])} |")
    lines+=["","| 输入 | 跨阈值病例对 | P_b | 单例错误率 | 概率按正确方向变化 |",
            "|---|---:|---:|---:|---:|"]
    for view,data in metrics["P_b"].items():
        if "P_b" not in data:continue
        lines.append(f"| {view} | {data['P_b']['denominator']} | {percent(data['P_b']['rate'])} | {percent(data['single_case_error']['rate'])} | {percent(data['probability_changes_in_correct_direction']['rate'])} |")
    lines+=["","背景、措辞、不跨阈值变化、处置交换及必要／无关说明补回的完整配对结果见 metrics_"+stage+".json。"]
    for blind,fusion in metrics["fusion"].items():
        lines+=["",f"## 融合：full / {blind}"]
        if fusion["status"]!="complete":
            lines.append(fusion.get("reason","Unavailable"));continue
        lines.append("评估部分："+fusion["evaluation_split"]+"。参数只使用开发来源选择。")
        lines+=["","| 病例 | 方法 | N | B | 准确率 | 双错纠正率 | 原至少一路正确而融合错误 |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for population,methods in fusion["populations"].items():
            for method,r in methods.items():
                lines.append(f"| {population} | {method} | {r['n']} | {percent(r['B']['rate'])} | {percent(r['accuracy']['rate'])} | {percent(r['double_wrong_correction']['rate'])} | {percent(r['new_error_among_at_least_one_correct']['rate'])} |")
    lines+=["","## 解释",
        "外部遮挡可能损失必要语义；即使信息充分，模型仍可能未能正确响应关键事实。两路输出的共同错误能否被外部融合纠正，需要通过具体方法验证。",
        "信息不足的错误不能用于证明固有偏见。当前实验不能定位错误来自哪个训练阶段。",
        "B 不是融合无效的证明；总体准确率、双错纠正率和新增错误必须一起报告。",
        "既有内部方法参考交换：一致参考与相反事实参考均为 386/400，尚未显示可靠使用参考事实方向；未重新训练内部模块。",
        "",
        "逐样本依据：data/nodes.jsonl、data/annotations.jsonl；输入及变动：data/inputs_"+stage+".jsonl、data/edits_"+stage+".jsonl。",
        "两路二分类分数：results/scores_"+stage+"_*.jsonl；融合选择与预测：results/fusion_*_"+stage+"_*.json*。",
        "CAD 采用对数概率形式，参见 https://aclanthology.org/2024.naacl-short.69/ 。"]
    (RESULTS/f"REPORT_{stage}.zh.md").write_text("\n".join(lines)+"\n")

def analyze(stage):
    rows,failures=examples(stage)
    score_count=sum(len(r["scores"]) for r in rows)
    metrics=dict(stage=stage,automatic_pilot=True,Q_auto=q_auto(stage),A=masking_metrics(rows),
        P_b=fact_metrics(rows),
        A_by_split={part:masking_metrics([r for r in rows if r["split"]==part]) for part in ("development","test")},
        P_b_by_split={part:fact_metrics([r for r in rows if r["split"]==part]) for part in ("development","test")},
        fusion={blind:fusion_metrics(rows,blind,stage) for blind in ("direct","condition_ancestors")},
        complete_score_count=score_count,failed_score_count=len(failures),
        scope="Node-level binary scores; original MEGA-CDP vignettes and constructed cases reported separately")
    write_json(RESULTS/f"metrics_{stage}.json",metrics)
    report(stage,metrics)
    print(json.dumps(dict(stage=stage,examples=len(rows),scores=score_count,failures=len(failures))))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["pilot","full"],default="pilot")
    analyze(parser.parse_args().stage)
