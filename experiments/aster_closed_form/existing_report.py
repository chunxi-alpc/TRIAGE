"""Tables and necessary completion checks for the existing-data experiment."""
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .common import rows,write_rows,write_json,status
from . import pipeline as base
from .evaluation import evaluate,pred,margin,fraction,signal_probe,path_metrics

def interval(items,seed=20261105):
    """Cluster bootstrap of scalar observations, with source as the unit."""
    items=[(str(s),float(v)) for s,v in items if v is not None and math.isfinite(float(v))]
    if not items:return dict(low=None,high=None,n=0,sources=0)
    sources=sorted({s for s,_ in items})
    bins=np.zeros((len(sources),2))
    index={s:i for i,s in enumerate(sources)}
    for source,value in items:bins[index[source]]+=(value,1)
    sample=np.random.default_rng(seed).integers(0,len(sources),(2000,len(sources)))
    summed=bins[sample].sum(1);rates=summed[:,0]/summed[:,1]
    return dict(low=float(np.quantile(rates,.025)),high=float(np.quantile(rates,.975)),
                n=len(items),sources=len(sources),resamples=2000)

def table_csv(path,records):
    path=Path(path)
    if not records:return
    fields=list(dict.fromkeys(k for row in records for k in row))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def metric_row(name,m):
    result=dict(method=name)
    for kind in ("A","P","C","F","P_clinical","P_preference"):
        x=m["PA_"+kind]
        result["PA_"+kind]=x["rate"];result["n_"+kind]=x["denominator"]
    result.update(macro_PA=m["macro_PA"],original_accuracy=m["original_accuracy"]["rate"],
                  all_input_accuracy=m["all_input_accuracy"]["rate"],
                  scored=m["score_coverage"]["numerator"],inputs=m["score_coverage"]["denominator"],
                  four_input_all_correct=m["historical_fourway"].get("rate"))
    return result

def threshold_value(row):
    variable=row.get("construction",{}).get("variable")
    def walk(expr):
        if not isinstance(expr,dict):return []
        values=[]
        if expr.get("field")==variable and expr.get("op") in ("<","<=",">",">=","==","!=") and isinstance(expr.get("value"),(int,float)):
            values.append(expr["value"])
        for key in ("args",):
            for child in expr.get(key,[]):values+=walk(child)
        if "arg" in expr:values+=walk(expr["arg"])
        return values
    return row.get("metric_value") in walk(row.get("construction",{}).get("predicate"))

def history_report(root):
    inputs=rows(root/"diagnostics/fact_preference_inputs.jsonl")
    pairs=rows(root/"diagnostics/fact_preference_pairs.jsonl")
    deletion=rows(root/"diagnostics/deletion_restore_inputs.jsonl")
    scores={}
    for path in (root/"diagnostics").glob("scores_*.jsonl"):
        for r in rows(path):scores[(r["score_id"],r["method"])]=r
    def value(r,method):
        return margin(scores.get((r["score_id"],method)))
    def ok(r,method):
        return pred(value(r,method))==r["gold"]
    fact_result={}
    for route in ("full","direct","condition_ancestors"):
        current=[r for r in inputs if r["view"]==route]
        examples={r["example_id"]:r for r in current}
        for method in ("natural","aster"):
            name=route+":"+method
            results={}
            for kind in ("F","P"):
                ps=[p for p in pairs if p["kind"]==kind]
                success=[(p["source_id"],int(ok(examples[p["left"]],method) and ok(examples[p["right"]],method))) for p in ps]
                responses=[]
                for p in ps:
                    left,right=examples[p["left"]],examples[p["right"]]
                    a,b=value(left,method),value(right,method)
                    if a is not None and b is not None:
                        responses.append(dict(pair_id=p["pair_id"],source_id=p["source_id"],response=b-a,
                            truth_aligned=(b-a)*(right["gold"]-left["gold"]) if kind=="F" else None))
                results["PA_"+kind]=dict(**fraction(sum(v for _,v in success),len(success)),ci95=interval(success))
                results["responses_"+kind]=responses
            results["accuracy"]=fraction(sum(ok(r,method) for r in current),len(current))
            results["technical_failures"]=sum(value(r,method) is None for r in current)
            results["strata"]={}
            for level in ("below","above","same_side","exact_threshold"):
                selected=[r for r in current if threshold_value(r)] if level=="exact_threshold" else [r for r in current if r["metric_level"]==level]
                results["strata"][level]=dict(**fraction(sum(ok(r,method) for r in selected),len(selected)),
                    ci95=interval([(r["source_id"],int(ok(r,method))) for r in selected]),
                    node_count=len({r["node_id"] for r in selected}))
            fact_result[name]=results
    write_json(root/"results/historical_fact_preference.json",dict(nodes=28,inputs=168,F_pairs=56,P_pairs=84,
        methods=fact_result,threshold_rule="Exact saved metric value equals a numeric threshold of the saved predicate; labels unchanged",
        selection_use=False,scoring_template=str(root/"diagnostics/template.json")))
    table_csv(root/"results/historical_fact_preference.csv",
        [dict(method=name,PA_F=m["PA_F"]["rate"],PA_P=m["PA_P"]["rate"],accuracy=m["accuracy"]["rate"],
              threshold_accuracy=m["strata"]["exact_threshold"]["rate"],threshold_inputs=m["strata"]["exact_threshold"]["denominator"],
              technical_failures=m["technical_failures"]) for name,m in fact_result.items()])

    deletion_result={}
    for route in sorted({r["view"] for r in deletion}):
        selected=[r for r in deletion if r["view"]==route]
        for method in ("natural","aster"):
            deletion_result[route+":"+method]=dict(**fraction(sum(ok(r,method) for r in selected),len(selected)),
                ci95=interval([(r["source_id"],int(ok(r,method))) for r in selected]),
                technical_failures=sum(value(r,method) is None for r in selected))
    deletion_nodes=sorted({r["node_id"] for r in deletion})
    by_key={(r["node_id"],r["view"]):r for r in deletion}
    rescue={}
    for method in ("natural","aster"):
        rescue[method]={}
        for route in ("necessary_restore","irrelevant_restore"):
            ids=[nid for nid in deletion_nodes if (nid,route) in by_key]
            direct_errors=[nid for nid in ids if not ok(by_key[(nid,"direct")],method)]
            rescue[method][route]=dict(nodes=len(ids),direct_errors=len(direct_errors),
                rescued=sum(ok(by_key[(nid,route)],method) for nid in direct_errors),
                degraded=sum(ok(by_key[(nid,"direct")],method) and not ok(by_key[(nid,route)],method) for nid in ids))
    write_json(root/"results/historical_deletion_restore.json",dict(nodes=11,irrelevant_control_nodes=10,
        methods=deletion_result,restoration_effects=rescue,
        interpretation="Existing context deletion and restoration with unchanged truth labels, not C truth flips",selection_use=False))
    table_csv(root/"results/historical_deletion_restore.csv",
              [dict(method=k,**{f:v for f,v in r.items() if f!="ci95"}) for k,r in deletion_result.items()])
    return fact_result,deletion_result

def path_report(root):
    from .paths import METHODS
    cases=rows(root/"data/path_cases.jsonl")
    by_case={c["case_id"]:c for c in cases}
    records={}
    for path in (root/"paths").glob("predictions_*.jsonl"):
        for r in rows(path):records[(r["case_id"],r["method"])]=r
    planned=[]
    for c in cases:
        for m in METHODS:
            planned.append(records.get((c["case_id"],m),dict(case_id=c["case_id"],source_id=c["source_id"],method=m,
                status="failed",correct=False,error="Expected path record missing")))
    reports={}
    for subset in ("all_432","original_100"):
        selected=[r for r in planned if subset=="all_432" or by_case[r["case_id"]]["original_100_subset"]]
        report=path_metrics(selected)
        full={r["case_id"]:r for r in selected if r["method"]=="full"}
        for method,m in report["methods"].items():
            rr=[r for r in selected if r["method"]==method]
            m["unavailable_existing_inputs"]=sum(r["status"]=="unavailable_input" for r in rr)
            m["technical_failures"]=sum(r["status"] not in ("complete","unavailable_input") for r in rr)
            m["corrected_vs_full"]=sum(r["status"]=="complete" and r["correct"] and
                                      full[r["case_id"]]["status"]=="complete" and not full[r["case_id"]]["correct"] for r in rr)
            m["degraded_vs_full"]=sum(r["status"]=="complete" and not r["correct"] and
                                     full[r["case_id"]]["status"]=="complete" and full[r["case_id"]]["correct"] for r in rr)
            available=[r for r in rr if r["status"]!="unavailable_input"]
            m["available_input_accuracy"]=fraction(sum(r["status"]=="complete" and r["correct"] for r in available),len(available))
            m["same_available_subset"]={compare:fraction(sum(records.get((r["case_id"],compare),{}).get("correct",False) for r in available),len(available)) for compare in ("full","aster")}
        report["nested_subset_of_432"]=subset=="original_100"
        reports[subset]=report
    write_json(root/"results/path_results.json",reports)
    table_csv(root/"results/path_results.csv",[dict(subset=s,method=name,correct=m["numerator"],cases=m["denominator"],
        accuracy=m["rate"],ci_low=m.get("ci95",[None,None])[0],ci_high=m.get("ci95",[None,None])[1],
        corrected=m["corrected_vs_full"],degraded=m["degraded_vs_full"],
        unavailable_inputs=m["unavailable_existing_inputs"],technical_failures=m["technical_failures"])
        for s,r in reports.items() for name,m in r["methods"].items()])
    return reports

def make_report(root):
    import torch
    views,pairs=base.load_data(root)
    scores=base.all_scores(root)
    amp=json.loads((root/"results/amplification_selection.json").read_text())
    scores={m:s for m,s in scores.items() if not m.startswith("amplify_") or m==amp["selected"]}
    if amp["selected"]=="aster":scores["erase_plus_selected_amplification"]=scores["aster"]
    else:scores["erase_plus_selected_amplification"]=scores.pop(amp["selected"])
    flags={"semantic":"semantic_available","boundary_guideline":"boundary_annotation_available","boundary_facts":"decisive_annotation_available"}
    limited={m:("semantic_available" if "semantic" in m else flags[m]) for m in scores if "semantic" in m or m in flags}
    full_scores={m:s for m,s in scores.items() if m not in limited}
    main=evaluate(views,pairs,full_scores,bootstrap=2000)
    main["all_view_changes_vs_full"]={}
    from .evaluation import _correction
    test_views={k:v for k,v in views.items() if v["split"]=="test"}
    for method,s in full_scores.items():
        main["all_view_changes_vs_full"][method]=_correction(test_views,s,scores["full"])
    write_json(root/"results/main_effects.json",main)
    main_csv=[]
    for method,result in main["methods"].items():
        record=metric_row(method,result)
        for key,ci in main["confidence_intervals"]["methods"][method].items():
            if key!="differences":
                record[key+"_ci_low"]=ci["low"]
                record[key+"_ci_high"]=ci["high"]
        main_csv.append(record)
    table_csv(root/"results/main_effects.csv",main_csv)
    subsets={}
    for method,flag in limited.items():
        eligible={k:v for k,v in views.items() if v.get(flag,False)}
        ps=[p for p in pairs if p["left"] in eligible and p["right"] in eligible]
        subset=evaluate(eligible,ps,{m:scores[m] for m in dict.fromkeys(("full","aster",method))},bootstrap=2000)
        subset["coverage"]=dict(test_inputs=subset["input_count"],total_test_inputs=610,
            test_input_fraction=subset["input_count"]/610,
            pair_counts=subset["pair_counts"],total_pair_counts=main["pair_counts"],
            criterion=flag,selection_independent_of_model_outcomes=True)
        subsets[method]=subset
    write_json(root/"results/available_subsets.json",subsets)
    table_csv(root/"results/available_subsets.csv",[dict(subset=group,**metric_row(m,v)) for group,r in subsets.items() for m,v in r["methods"].items()])
    choice=json.loads((root/"results/selection.json").read_text())
    ops=torch.load(root/"operators/aster.pt",map_location="cpu",weights_only=False)
    diag=json.loads((root/"operators"/(choice["method"]+".json")).read_text())
    checks={h:v["algebra"] for h,v in diag["heads"].items()}
    assert all(c["signal_preservation_error"]<1e-5 and c["erasure_error"]<1e-5 and
               c["idempotence_error"]<1e-5 and c["angle_bound_satisfied"] for c in checks.values())
    write_json(root/"results/operator_checks.json",checks)
    features=base.load_features(root)
    h3=signal_probe(views,features,ops,scores["full"])
    write_json(root/"results/h3.json",dict(probe=h3,amplification=amp))
    del features
    mechanism={}
    for split in ("dev","test"):
        summary=json.loads((root/"mechanisms"/f"mechanisms_{split}.json").read_text())
        joint=rows(root/"mechanisms"/f"joint_components_{split}.jsonl")
        clustered=[]
        for kind in ("A","P"):
            for component in ("full_head","nuisance_orthogonal","screened_orthogonal","oblique"):
                rs=[r for r in joint if r["kind"]==kind and r["component"]==component]
                finite=[r for r in rs if r.get("recovery") is not None]
                clustered.append(dict(kind=kind,component=component,total_pairs=len(rs),normalized_pairs=len(finite),
                    recovery=float(np.mean([r["recovery"] for r in finite])) if finite else None,
                    ci95=interval([(r["source_id"],r["recovery"]) for r in finite])))
        mechanism[split]=dict(h1=summary["h1"],h2=summary["h2"],local_theory=summary["local_theory"],joint_components=clustered)
    write_json(root/"results/mechanisms.json",mechanism)
    write_json(root/"results/h1.json",dict(A=json.loads((root/"m1/m1_A.json").read_text()),
        P=json.loads((root/"m1/m1_P.json").read_text()),interventions={s:m["h1"] for s,m in mechanism.items()}))
    write_json(root/"results/h2.json",dict(operator_geometry=diag,interventions={s:m["h2"] for s,m in mechanism.items()}))
    write_json(root/"results/local_theory.json",{s:m["local_theory"] for s,m in mechanism.items()})
    table_csv(root/"results/joint_components.csv",[dict(split=s,**{k:v for k,v in r.items() if k!="ci95"},
        ci_low=r["ci95"]["low"],ci_high=r["ci95"]["high"]) for s,m in mechanism.items() for r in m["joint_components"]])
    fact,deleted=history_report(root)
    paths=path_report(root)
    manifest=json.loads((root/"data/manifest.json").read_text())
    technical=sum(m["score_coverage"]["missing"] for m in main["methods"].values())
    limited_failures=sum(subset["methods"][name]["score_coverage"]["missing"] for name,subset in subsets.items())
    historical_failures=sum(m["technical_failures"] for m in fact.values())+sum(m["technical_failures"] for m in deleted.values())
    path_failures=sum(m["technical_failures"] for m in paths["all_432"]["methods"].values())
    zero=[json.loads((root/"results"/f"zero_intervention_{s}.json").read_text()) for s in (0,1)]
    assert all(v["zero_equal"] for v in zero)
    completion=dict(state="complete",technical_failed_scores=technical,limited_subset_failed_scores=limited_failures,historical_failed_scores=historical_failures,
        path_technical_failures=path_failures,existing_input_gaps={m:610-r["input_count"] for m,r in subsets.items()},
        checks=dict(data_counts=1290,pairs=1053,source_splits="disjoint",zero_intervention=True,operator_properties=True,
                    target_position_and_heads="server engine tests",path_cases=432,nested_path_subset=100,
                    no_new_annotations=True,no_test_selection=True))
    completion["result_status"]="complete_with_technical_failures" if technical+limited_failures+historical_failures+path_failures else "complete"
    write_json(root/"results/status.json",completion)
    def pct(value):return "?" if value is None else f"{100*value:.2f}%"
    def fmt(value):
        if not isinstance(value,dict):return pct(value)
        return f"{value['numerator']}/{value['denominator']} ({pct(value['rate'])})" if value.get("numerator") is not None else pct(value["rate"])
    lines=["# ASTER 现成数据实验报告",
        "",
        "本实验仅复用已有数据。主实验保留原 200 个节点、来源划分及种子 20261105，仅移除已登记的 7 条明确错误和 2 条有歧义的 C 变体。",
        "共使用 1,290 个输入、1,053 个配对。拟合、开发、测试输入分别为 561、119、610 个；未新增病例、条件或标签。",
        "",
        f"固定配置：{choice['method']}，能量目标 {choice['config']['eta']}，单类秩上限 {choice['config']['kmax']}，夹角 {choice['config']['theta']} 度，{choice['heads']} 个头，总编辑秩 {choice['rank']}。",
        f"开发 C 只有 1 对；有 {len(choice['tied_on_primary_metrics'])} 个配置在主指标和原输入准确率上并列。最终选择遵循预设的头数、总编辑秩和较大夹角规则，不能据此认定所选参数充分优于其他参数。",
        f"开发集选择的信号放大倍数为 {amp['amplification']}；输出层对照中选中的方法为 {json.loads((root/'results/fusion.json').read_text())['best_method']}。",
        "",
        "## 主效果与消融","",
        "配对正确率 PA 要求一对输入的两侧均判断正确。PA_P 由临床背景与治疗偏好两类正确率各占一半；主指标为 A、P、C、F 四类 PA 的平均值。按指南来源重采样得到的 95% 区间见 JSON 和 CSV。",
        "",
        "| 方法 | PA_A | PA_P | PA_C | PA_F | 宏平均 | 原输入准确率 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name,m in main["methods"].items():
        lines.append("| "+" | ".join([name,*[fmt(m["PA_"+k]) for k in ("A","P","C","F")],pct(m["macro_PA"]),fmt(m["original_accuracy"])])+" |")
    a=main["methods"]["aster"];f=main["methods"]["full"]
    delta=a["macro_PA"]-f["macro_PA"]
    ci=main["confidence_intervals"]["methods"]["aster"]["differences"]["full"]["macro_PA"]
    lines+=["",f"ASTER 相对完整阅读的宏平均变化为 {delta*100:+.2f} 个百分点，按指南来源重采样得到的差值 95% 区间为 [{ci['low']*100:+.2f}, {ci['high']*100:+.2f}] 个百分点。",
        "原输入与全部输入的纠正和退化、同节点双错纠正、分数响应及历史四输入全对率见 main_effects.json。本实验为自动探索性验证，正负结果均保留。",
        "","## 语义遮挡与边界对照可用子集","",
        "| 对照 | 测试输入覆盖数／610 | 完整阅读宏平均 | ASTER 宏平均 | 对照宏平均 |",
        "| --- | --- | --- | --- | --- |"]
    for method,r in subsets.items():
        lines.append(f"| {method} | {r['input_count']}/610 | {pct(r['methods']['full']['macro_PA'])} | {pct(r['methods']['aster']['macro_PA'])} | {pct(r['methods'][method]['macro_PA'])} |")
    lines+=["","各行在相同的现成输入或标注子集上比较。现成输入缺失与实际执行失败分开统计；执行失败保留在对应评价分母中，不将保留全文的回退输入计作语义遮挡。",
        "","## 双向联合修补与机制诊断","",
        "| 配对类型 | 修补分量 | 全部配对数 | 可归一化配对数 | 平均恢复比例 |",
        "| --- | --- | --- | --- | --- |"]
    for r in mechanism["test"]["joint_components"]:
        val="?" if r["recovery"] is None else f"{r['recovery']:.4f}"
        lines.append(f"| {r['kind']} | {r['component']} | {r['total_pairs']} | {r['normalized_pairs']} | {val} |")
    lines+=["", "四种修补使用同一批 A/P 配对，分别对应完整头差分、未经夹角筛选的干扰空间正交分量、筛选后的正交分量和最终斜投影分量。该诊断不参与选参；归一化恢复比例仅在总效应绝对值不小于 0.1 的配对上计算，其他配对仍保留原始效应。",
        f"H3 在原模型判错的全部 {h3['full_error_count']} 个有效测试输入上评价，其中原输入错误子集为 {h3['original_full_error_count']} 个。信号探针准确率为 {fmt(h3.get('accuracy',fraction(0,0)))}，原输入错误子集为 {fmt(h3.get('original_error_accuracy',fraction(0,0)))}。",
        "H1、逐头与联合 H2、固定开发和测试各 20 节点的局部理论结果，以及完整阅读与遮挡的背景响应差分，见 mechanisms.json、h1.json、h2.json、local_theory.json 和 main_effects.json。",
        "","## 历史事实与偏好诊断","",
        "复用 28 个历史测试节点，形成 56 对事实配对和 84 对治疗偏好配对。保留原输入、条件、标签和评分模板，使用固定的主实验算子评价；不参与主实验拟合或选参。",
        "",
        "| 输入路线／方法 | PA_F | PA_P | 输入准确率 | 阈值边界准确率 |",
        "| --- | --- | --- | --- | --- |"]
    for name,r in fact.items():
        lines.append(f"| {name} | {fmt(r['PA_F'])} | {fmt(r['PA_P'])} | {fmt(r['accuracy'])} | {fmt(r['strata']['exact_threshold'])} |")
    lines+=["","各数值状态 below、above、same_side 的结果见 historical_fact_preference.json。阈值边界按既有数值和条件判断，不改写条件或标签。",
        "","## 历史删除与补回诊断","",
        "复用 11 个测试节点，其中 10 个有无关说明补回对照。这些病例用于删除与补回诊断，不计作条件真值翻转 C。",
        "","| 输入路线／方法 | 准确率 | 技术失败数 |","| --- | --- | --- |"]
    for name,r in deleted.items():lines.append(f"| {name} | {fmt(r)} | {r['technical_failures']} |")
    lines+=["","## 自主路径","",
        "全部 432 例从根节点自主执行。原 100 例作为嵌套子集另列，不视为独立重复实验。完整路径正确要求预测路径和终点均符合既有标准答案。",
        "","| 范围 | 方法 | 完整路径正确率 | 纠正数／退化数 | 现成输入缺失数 | 技术失败数 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for scope,r in paths.items():
        for name,m in r["methods"].items():
            lines.append(f"| {scope} | {name} | {fmt(m)} | {m['corrected_vs_full']}/{m['degraded_vs_full']} | {m['unavailable_existing_inputs']} | {m['technical_failures']} |")
    lines+=["","现成输入缺失与技术失败分开报告，完整范围的分母保持为计划病例数。可用输入子集上的准确率，以及同子集完整阅读与 ASTER 的结果，见 path_results.json。",
        "","## 复用、验证与产物","",
        "原模型激活、梯度、可复用基线分数、M1、H1 和不受算子变化影响的完整头修补结果直接复用。M2/M3、开发网格、融合选择、编辑评分、相关消融、探针和路径按依赖重新计算。",
        "必要验证包括已有标签和配对索引、来源划分、零干预一致、目标头与位置，以及算子的信号保持、擦除、幂等和幅度约束。结果见 results/operator_checks.json、results/zero_intervention_*.json 和 logs/verification.log。",
        "","全部结果表位于 results/；固定配置见 results/selection.json、results/fusion.json 和 operators/aster.pt；复用说明见 results/reuse.json。",
        f"完成状态：{completion['result_status']}。主表缺分数 {technical}，可用子集对照缺分数 {limited_failures}，历史诊断缺分数 {historical_failures}，路径技术失败数 {path_failures}。",
        ""]
    (root/"REPORT.zh.md").write_text("\n".join(lines),encoding="utf-8")

    launch_path=root/"launch.json"
    if launch_path.exists():
        launch=json.loads(launch_path.read_text())
        from datetime import datetime,timezone
        launch.update(state="complete",completed=datetime.now(timezone.utc).isoformat(),
                      result_status=completion["result_status"],report=str(root/"REPORT.zh.md"))
        write_json(launch_path,launch)
    return completion
