"""Corrected denominators and development-only selection of fusion and calibration."""
import argparse
import collections
import json
import numpy as np
from common import CONFIG, DATA, RESULTS, fraction, read_rows, write_json, write_rows
import analyze as base
from dependency_rules import row_dependency
from score import make_prompt, settings

def examples(stage):
    inputs={r["score_id"]:r for r in read_rows(DATA/f"inputs_{stage}.jsonl")}
    latest={}
    for shard in (0,1):
        for row in read_rows(RESULTS/f"scores_{stage}_{shard}.jsonl"):
            latest[row["score_id"]]=row
    output,failures={},[]
    for key,row in inputs.items():
        score=latest.get(key)
        if (not score or score.get("status")!="complete" or
            score.get("user_prompt")!=make_prompt(row) or
            score.get("model_settings")!=settings(score.get("label_token_ids"))):
            failures.append(dict(score_id=key,reason="missing, failed or incompatible score"))
            continue
        if row["example_id"] not in output:
            output[row["example_id"]]=dict(row,scores={})
        # Labels always come from the repaired inputs, including reused scores.
        output[row["example_id"]]["scores"][row["view"]]=dict(score,gold=row["gold"],
            correct=score["prediction"]==row["gold"])
    return list(output.values()),failures

def masking_metrics(rows):
    original=[r for r in rows if r["population"]=="original" and r["role"]=="statistical"]
    groups=dict(all=original,
        dependent=[r for r in original if row_dependency(r) is True],
        independent=[r for r in original if row_dependency(r) is False],
        unknown_dependency=[r for r in original if row_dependency(r) is None],
        definition_already_completed=[r for r in original if row_dependency(r) is True and r["dataset_definition_completed"] is True])
    result={}
    for name,pool in groups.items():
        changed=[r for r in pool if r["changes"]["direct_changed"]]
        direct=base.paired_accuracy(changed,"full","direct")
        direct["A_accuracy_drop"]=None if direct["second_minus_first"] is None else -direct["second_minus_first"]
        conservative=[r for r in pool if r["changes"]["conservative_changed"]]
        restore=[r for r in changed if r["changes"]["restoration_available"]]
        matched=[r for r in restore if r["changes"]["matched_irrelevant_available"]]
        result[name]=dict(n=len(pool),designated_deletion_count=len(changed),
            unchanged_input_count=len(pool)-len(changed),
            conservative_unchanged_count=len(pool)-len(conservative),
            full_vs_direct=direct,
            full_vs_conservative=base.paired_accuracy(conservative,"full","conservative"),
            necessary_recovery=base.paired_accuracy(restore,"direct","necessary_restore"),
            irrelevant_recovery=base.paired_accuracy(matched,"direct","irrelevant_restore"),
            necessary_vs_irrelevant=base.paired_accuracy(matched,"irrelevant_restore","necessary_restore"),
            full_vs_condition_ancestors=base.paired_accuracy(pool,"full","condition_ancestors"))
    return result

def fact_metrics(rows):
    constructed=[r for r in rows if r["population"]=="constructed"]
    result={}
    for view in ("full","direct","condition_ancestors"):
        valid=[r for r in constructed if view in r["scores"]]
        primary=[r for r in valid if r["background"]=="neutral" and r["wording"]=="original"]
        groups=collections.defaultdict(dict)
        for row in primary:
            groups[row["node_id"]][row["metric_level"]]=row
        cross=[(g["below"],g["above"]) for g in groups.values() if "below" in g and "above" in g]
        same=[(g["above"],g["same_side"]) for g in groups.values() if "above" in g and "same_side" in g]
        wrong=lambda r:r["scores"][view]["prediction"]!=r["gold"]
        result[view]=dict(nodes=len(groups),primary_population="neutral background + original complete condition; one pair per node",
            P_b=fraction(sum(wrong(a) or wrong(b) for a,b in cross),len(cross)),
            single_case_error=fraction(sum(wrong(r) for r in primary),len(primary)),
            all_background_single_case_error=fraction(sum(wrong(r) for r in valid),len(valid)),
            crossing_both_correct=fraction(sum(not wrong(a) and not wrong(b) for a,b in cross),len(cross)),
            probability_changes_in_correct_direction=fraction(sum(
                (b["scores"][view]["p_yes"]-a["scores"][view]["p_yes"])*(b["gold"]-a["gold"])>0 for a,b in cross),len(cross)),
            crossing_prediction_changed=fraction(sum(a["scores"][view]["prediction"]!=b["scores"][view]["prediction"] for a,b in cross),len(cross)),
            noncrossing_prediction_stable=fraction(sum(a["scores"][view]["prediction"]==b["scores"][view]["prediction"] for a,b in same),len(same)),
            noncrossing_both_correct=fraction(sum(not wrong(a) and not wrong(b) for a,b in same),len(same)),
            background_effect=base.effect_pairs(valid,"background","neutral","cue",view),
            wording_effect=base.effect_pairs([r for r in valid if r["wording_changed"]],"wording","original","neutral",view),
            wording_noop_nodes=len({r["node_id"] for r in valid if not r["wording_changed"]}),
            P_b_strata={})
        for background in ("neutral","cue"):
            subgroup=collections.defaultdict(dict)
            for row in valid:
                if row["background"]==background and row["wording"]=="original":
                    subgroup[row["node_id"]][row["metric_level"]]=row
            pairs=[(g["below"],g["above"]) for g in subgroup.values() if "below" in g and "above" in g]
            result[view]["P_b_strata"][background+"/original"]=fraction(sum(wrong(a) or wrong(b) for a,b in pairs),len(pairs))
        result[view]["wide_crossing_nodes"]=[r["node_id"] for r in primary if r["metric_level"]=="below" and
            "wide crossing" in r.get("construction",{}).get("note","").lower()]
    result["downstream_action_exchange"]=base.paired_accuracy(constructed,"full","action_swap")
    return result

def feature_matrix(rows,views):
    return np.array([[r["scores"][view]["margin"] for view in views] for r in rows])

def fit_logistic(development,views):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    x=feature_matrix(development,views)
    y=np.array([r["gold"] for r in development])
    groups=np.array([r["source_id"] for r in development])
    if len(set(y))<2 or len(set(groups))<2:
        return None,dict(status="unavailable",reason="Need two labels and two development sources")
    cv=GroupKFold(n_splits=min(5,len(set(groups))))
    candidates=[]
    for c in CONFIG["logistic_c"]:
        hits=total=0;folds=[]
        for train,validation in cv.split(x,y,groups):
            if len(set(y[train]))<2:
                continue
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=CONFIG["seed"]))
            model.fit(x[train],y[train]);pred=model.predict(x[validation])
            hits+=int((pred==y[validation]).sum());total+=len(validation)
            folds.append(dict(training_sources=sorted(set(groups[train])),validation_sources=sorted(set(groups[validation]))))
        candidates.append(dict(C=c,correct=hits,n=total,accuracy=hits/total if total else -1,folds=folds))
    best=max(candidates,key=lambda c:c["accuracy"])
    if best["accuracy"]<0:
        return None,dict(status="unavailable",reason="No valid development fold")
    model=make_pipeline(StandardScaler(),LogisticRegression(C=best["C"],max_iter=3000,random_state=CONFIG["seed"]))
    model.fit(x,y)
    scale=model.named_steps["standardscaler"];reg=model.named_steps["logisticregression"]
    return model,dict(status="fit",C=best["C"],cv_candidates=candidates,
        feature_order=views,feature_mean=scale.mean_.tolist(),feature_scale=scale.scale_.tolist(),
        coefficients=reg.coef_.tolist(),intercept=reg.intercept_.tolist(),
        development_n=len(y),development_sources=sorted(set(groups)),
        selection="GroupKFold within original development sources; first C breaks ties")

def fusion_metrics(rows,blind,stage):
    paired=[r for r in rows if "full" in r["scores"] and blind in r["scores"]]
    dev=[r for r in paired if r["split"]=="development"]
    test=[r for r in paired if r["split"]=="test"]
    if not dev:
        return dict(status="unavailable",reason="No paired development scores")
    chosen,grid_results={},{}
    for method,grid in (("probability_average",CONFIG["weights"]),("cad_full",CONFIG["alphas"]),("cad_blind",CONFIG["alphas"])):
        candidates=[]
        for parameter in grid:
            probabilities=[base.fusion_probability(method,parameter,r["scores"]["full"],r["scores"][blind]) for r in dev]
            candidates.append(dict(parameter=parameter,accuracy=base.fusion_summary(dev,probabilities,blind)["accuracy"]["rate"]))
        chosen[method]=max(candidates,key=lambda c:c["accuracy"])["parameter"]
        grid_results[method]=candidates
    models,logistics={},{}
    for name,views in (("logistic",["full",blind]),("calibration_full",["full"]),("calibration_blind",[blind])):
        models[name],logistics[name]=fit_logistic(dev,views)
    selection=dict(blind_view=blind,parameters=chosen,development_candidates=grid_results,
        logistic=logistics["logistic"],single_path_calibration={k:v for k,v in logistics.items() if k!="logistic"},
        cad_baseline_fallback={name:("full" if name=="cad_full" else blind) for name in ("cad_full","cad_blind") if chosen[name]==0},
        selection_scope="All repaired development examples pooled; original source split unchanged; test labels unused",
        development_sources=sorted({r["source_id"] for r in dev}),test_sources=sorted({r["source_id"] for r in test}))
    write_json(RESULTS/f"fusion_selection_{stage}_{blind}.json",selection)
    target=test if stage=="full" else dev
    evaluated=[]
    for row in target:
        scores=dict(full=row["scores"]["full"]["p_yes"],blind=row["scores"][blind]["p_yes"])
        for method,parameter in chosen.items():
            scores[method]=base.fusion_probability(method,parameter,row["scores"]["full"],row["scores"][blind])
        for name,model in models.items():
            if model is not None:
                scores[name]=float(model.predict_proba(feature_matrix([row],logistics[name]["feature_order"]))[0,1])
        evaluated.append(dict(example_id=row["example_id"],node_id=row["node_id"],source_id=row["source_id"],
            population=row["population"],split=row["split"],gold=row["gold"],probabilities=scores,
            direct_changed=row["changes"]["direct_changed"]))
    write_rows(RESULTS/f"fusion_predictions_{stage}_{blind}.jsonl",evaluated)
    populations={}
    for population in ("original","constructed"):
        idx=[i for i,r in enumerate(target) if r["population"]==population]
        methods=sorted(set().union(*(evaluated[i]["probabilities"] for i in idx))) if idx else []
        populations[population]={method:base.fusion_summary([target[i] for i in idx],
            [evaluated[i]["probabilities"][method] for i in idx],blind) for method in methods}
    return dict(status="complete",evaluation_split="test" if stage=="full" else "development_pilot",
        selection=selection,populations=populations,input_pair="full/"+blind,
        paired_dev=len(dev),paired_test=len(test),
        direct_unchanged_test=sum(not r["changes"]["direct_changed"] for r in test))

def write_report(stage,metrics):
    p=base.percent;q=metrics["Q_auto"]["all"]
    prep=json.loads((RESULTS/"revision_preparation.json").read_text())
    lines=["# 修订自动预实验","","状态：自动预实验。保留原40份指南、379个统计节点和来源划分；未训练内部干预模块。",
        "原始病例指MEGA-CDP生成的病例。构造病例与语义依赖仍为自动研究标注。",
        f"阶段：{stage}；有效分数 {metrics['complete_score_count']}；缺失或失败 {metrics['failed_score_count']}。",
        "",
        "旧结果被替代的原因：原遮挡大多未执行指定删除；旧构造标签丢失区间或复合条件，背景还改变了病例事实；这些标签参与开发调参，因此原始病例与构造病例的全部融合选择均重新计算。",
        "",
        f"固定构造池50个条件；机械修复后可构造 {prep['repaired_constructible']} 个，其余原因逐项保存。最终入组数见下表及 input_exclusions_{stage}.jsonl。",
        f"Q_auto：{q['Q_auto']['numerator']}/{q['Q_auto']['denominator']} = {p(q['Q_auto']['rate'])}；依赖未知比例 {p(q['dependency_unknown']['rate'])}。",
        "混写、原文语义依赖和问题已补全定义分别记录，未知不计为否。"]
    if q["Q_auto"]["numerator"]==0:
        lines+=["本次固定抽样中未发现已被原文证据支持的语义依赖正例；不能据此推断其他指南或节点不存在依赖。"]
    lines+=["","| 节点组 | 实际删除配对 | 未改动输入 | 完整准确率 | 删除后准确率 | 准确率下降 |",
            "|---|---:|---:|---:|---:|---:|"]
    for group,data in metrics["A"].items():
        a=data["full_vs_direct"]
        lines.append(f"| {group} | {a['n']} | {data['unchanged_input_count']} | {p(a['first_accuracy'])} | {p(a['second_accuracy'])} | {p(a['A_accuracy_drop'])} |")
    lines+=["","五种输入均保存：完整、直接遮挡、保守遮挡、必要说明补回、无关说明补回。补回只使用可定位的原文；没有可补回定义时不伪造定义，相关比较分母为零。",
        "完整上下文是节点映射对应的原文块及相关上下文，包含指定处置；并非整本指南。",
        "",
        f"| 输入 | {'测试' if stage=='full' else '开发'}节点对 | P_b（中性背景＋原条件） | 单例错误率 |",
        "|---|---:|---:|---:|"]
    for view,data in metrics["P_b_by_split"]["test" if stage=="full" else "development"].items():
        if "P_b" in data:
            lines.append(f"| {view} | {data['P_b']['denominator']} | {p(data['P_b']['rate'])} | {p(data['single_case_error']['rate'])} |")
    lines+=["","每个节点只贡献一对主指标；背景、不跨阈值、真实措辞变化及处置交换另报。原条件已中性时只保留一份，不增加重复样本。"]
    for blind,fusion in metrics["fusion"].items():
        lines+=["",f"## 融合：full / {blind}"]
        if fusion["status"]!="complete":
            lines.append(fusion["reason"]);continue
        for method,view in fusion["selection"]["cad_baseline_fallback"].items():
            lines.append(f"{method} 选择系数0，退回 {view} 单路基线。")
        lines+=["","| 病例 | 方法 | N | 准确率 | 双错比例 | 双错纠正率 | 原至少一路正确却被判错 |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for population,methods in fusion["populations"].items():
            for method,r in methods.items():
                lines.append(f"| {population} | {method} | {r['n']} | {p(r['accuracy']['rate'])} | {p(r['B']['rate'])} | {p(r['double_wrong_correction']['rate'])} | {p(r['new_error_among_at_least_one_correct']['rate'])} |")
        lines+=["","calibration_full 与 calibration_blind 仅用一路分数做逻辑回归校准，使用与两路融合相同的开发来源交叉验证和C网格。与它们的比较用于判断收益是否需要两路信息。"]
    lines+=["","详细分组指标：results/metrics_"+stage+".json；逐样本原文和变化：data/edits_"+stage+".jsonl；完整输入：data/inputs_"+stage+".jsonl；",
            "完整条件：data/construction_specs.jsonl；分数：results/scores_"+stage+"_*.jsonl；融合参数与预测：results/fusion_*。",
            "仅当完整渲染输入和模型设置一致时复用旧分数，标签从修订输入重绑定。旧目录保持原样。"]
    (RESULTS/f"REPORT_{stage}.zh.md").write_text("\n".join(lines)+"\n")

def analyze(stage):
    rows,failures=examples(stage)
    metrics=dict(stage=stage,automatic_pilot=True,Q_auto=base.q_auto(stage),
        A=masking_metrics(rows),P_b=fact_metrics(rows),
        A_by_split={part:masking_metrics([r for r in rows if r["split"]==part]) for part in ("development","test")},
        P_b_by_split={part:fact_metrics([r for r in rows if r["split"]==part]) for part in ("development","test")},
        fusion={view:fusion_metrics(rows,view,stage) for view in ("direct","condition_ancestors")},
        complete_score_count=sum(len(r["scores"]) for r in rows),failed_score_count=len(failures))
    write_rows(RESULTS/f"missing_scores_{stage}.jsonl",failures)
    write_json(RESULTS/f"metrics_{stage}.json",metrics)
    write_report(stage,metrics)
    return metrics

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["pilot","full"],required=True)
    analyze(parser.parse_args().stage)
