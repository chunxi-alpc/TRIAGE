"""Post-run error analysis from saved scores only; no model calls or relabeling."""
import argparse, json, statistics, gzip
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

def read_rows(path):
    if not path.exists():path=Path(str(path)+".gz")
    text=gzip.open(path,"rt",encoding="utf-8").read() if path.suffix==".gz" else path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def fraction(correct, total):
    return {"correct": int(correct), "total": int(total), "rate": correct / total if total else None}

def main(root):
    out = root / "analysis_postrun"
    out.mkdir(exist_ok=True)
    cases = {r["case_id"]: r for r in read_rows(root / "data/path_cases.jsonl")}
    main_nodes = {(r["source_id"], r["gid"]) for r in read_rows(root / "data/groups_main.jsonl")}
    paths = defaultdict(dict)
    for path in sorted((root / "paths").glob("predictions_*.jsonl*")):
        for row in read_rows(path):
            paths[row["case_id"]][row["method"]] = row
    common, changes = [], []
    for cid, methods in paths.items():
        case = cases[cid]
        full, aster = methods["full"], methods["aster"]
        fm = {p["gid"]: p for p in full["predictions"]}
        am = {p["gid"]: p for p in aster["predictions"]}
        gold = {p["gid"]: p["label"] for p in case["gold_path"]}
        for gid in sorted(fm.keys() & am.keys()):
            common.append(dict(case_id=cid, source_id=case["source_id"], gid=gid, gold=gold.get(gid),
                full_ell=fm[gid]["ell"], aster_ell=am[gid]["ell"], delta=am[gid]["ell"]-fm[gid]["ell"],
                full_label=fm[gid]["label"], aster_label=am[gid]["label"],
                changed=fm[gid]["label"] != am[gid]["label"],
                main_node_covered=(case["source_id"], gid) in main_nodes,
                full_path_correct=full["correct"], aster_path_correct=aster["correct"]))
        if full["correct"] == aster["correct"]:
            continue
        changed_gold = [p for p in case["gold_path"] if p["gid"] in fm and p["gid"] in am
                        and fm[p["gid"]]["label"] != am[p["gid"]]["label"]]
        first = changed_gold[0]
        gid = first["gid"]
        group = next(g for g in case["guideline"]["groups"] if g["group_id"] == gid)
        changes.append(dict(case_id=cid, source_id=case["source_id"],
            outcome="corrected" if aster["correct"] else "degraded",
            original_100_subset=case["original_100_subset"], gid=gid, gold=first["label"],
            full_ell=fm[gid]["ell"], aster_ell=am[gid]["ell"],
            delta=am[gid]["ell"]-fm[gid]["ell"],
            main_node_covered=(case["source_id"], gid) in main_nodes,
            condition=group["rules"][0]["condition"], patient=case["patient"],
            first_gold_error_in_full=next((p["gid"] for p in case["gold_path"] if p["gid"] in fm and fm[p["gid"]]["label"]!=p["label"]), None),
            first_gold_error_in_aster=next((p["gid"] for p in case["gold_path"] if p["gid"] in am and am[p["gid"]]["label"]!=p["label"]), None)))
    source_bins = defaultdict(lambda: [0, 0])
    for cid, methods in paths.items():
        source_bins[cases[cid]["source_id"]][0] += int(methods["aster"]["correct"])-int(methods["full"]["correct"])
        source_bins[cases[cid]["source_id"]][1] += 1
    b = np.array([source_bins[k] for k in sorted(source_bins)], dtype=float)
    samples = np.random.default_rng(20261105).integers(0, len(b), (2000, len(b)))
    totals = b[samples].sum(1)
    ci = np.quantile(totals[:,0]/totals[:,1], [.025,.975]).tolist()
    changed_common = [r for r in common if r["changed"]]
    path_result = dict(cases=len(cases), full_correct=sum(d["full"]["correct"] for d in paths.values()),
        aster_correct=sum(d["aster"]["correct"] for d in paths.values()),
        corrected=sum(c["outcome"]=="corrected" for c in changes),
        degraded=sum(c["outcome"]=="degraded" for c in changes),
        paired_difference_ci95_by_source=ci, source_count=len(source_bins), resamples=2000,
        common_node_count=len(common), common_node_flips=changed_common,
        common_abs_full_margin_median=statistics.median(abs(r["full_ell"]) for r in common),
        common_abs_full_margin_le_2_5=sum(abs(r["full_ell"])<=2.5 for r in common),
        outcome_changes=changes,
        policy="Observed common nodes only; no counterfactual scores or labels inferred for unvisited branches.")

    scores = {}
    for path in sorted((root/"diagnostics").glob("scores_*.jsonl*")):
        for row in read_rows(path):
            scores[(row["score_id"],row["method"])] = row
    inputs = read_rows(root/"diagnostics/fact_preference_inputs.jsonl")
    pairs = read_rows(root/"diagnostics/fact_preference_pairs.jsonl")
    historical = {}
    def thresholds(expr, variable):
        if not isinstance(expr, dict): return []
        found = [expr["value"]] if expr.get("field")==variable and expr.get("op") in ("<","<=",">",">=","==","!=") and isinstance(expr.get("value"),(int,float)) else []
        for child in expr.get("args", []): found += thresholds(child,variable)
        if "arg" in expr: found += thresholds(expr["arg"],variable)
        return found
    for route in ("full","direct","condition_ancestors"):
        current = [r for r in inputs if r["view"]==route]
        by_example = {r["example_id"]: r for r in current}
        def sc(row,method): return scores[(row["score_id"],method)]
        def ok(row,method): return sc(row,method)["prediction"]==row["gold"]
        flips=[]
        strata={}
        for level in ("below","above","same_side"):
            for bg in ("neutral","cue"):
                subset=[r for r in current if r["metric_level"]==level and r["background"]==bg]
                strata[level+":"+bg]={method:fraction(sum(ok(r,method) for r in subset),len(subset)) for method in ("natural","aster")}
        for row in current:
            natural,aster=sc(row,"natural"),sc(row,"aster")
            if natural["prediction"]!=aster["prediction"]:
                flips.append(dict(**{k:row[k] for k in ("score_id","node_id","source_id","metric_level","metric_value","background","gold","condition","patient")},
                    predicate=row["construction"]["predicate"],full_ell=natural["ell"],aster_ell=aster["ell"],
                    delta=aster["ell"]-natural["ell"],outcome="corrected" if ok(row,"aster") else "degraded"))
        pair_summary={}
        for kind in ("F","P"):
            selected=[p for p in pairs if p["kind"]==kind]
            details=[]
            counts=Counter()
            for p in selected:
                good={m:ok(by_example[p["left"]],m) and ok(by_example[p["right"]],m) for m in ("natural","aster")}
                for m,g in good.items():counts[m]+=int(g)
                if good["natural"]!=good["aster"]:
                    details.append(dict(pair_id=p["pair_id"],node_id=p["node_id"],source_id=p["source_id"],
                        outcome="corrected" if good["aster"] else "degraded",left=p["left"],right=p["right"]))
            pair_summary[kind]=dict(natural=fraction(counts["natural"],len(selected)),
                aster=fraction(counts["aster"],len(selected)),changed_pairs=details)
        exact=[r for r in current if r.get("metric_value") in thresholds(r["construction"].get("predicate"),r["construction"].get("variable"))]
        historical[route]=dict(inputs=len(current),nodes=len({r["node_id"] for r in current}),
            exact_threshold={m:fraction(sum(ok(r,m) for r in exact),len(exact)) for m in ("natural","aster")},
            accuracy={m:fraction(sum(ok(r,m) for r in current),len(current)) for m in ("natural","aster")},
            corrected=sum(x["outcome"]=="corrected" for x in flips),
            degraded=sum(x["outcome"]=="degraded" for x in flips),input_flips=flips,strata=strata,pairs=pair_summary)

    cautions=[
        dict(case_id="path-11-25",observation="病历先否认可疑神经症状，后又记载查体可见行走不稳、可诱发构音不清；G33询问病史或当前表现。",
             implication="同一现成病例中的正反叙述使gold=0的事实对应关系不清；本分析保留原标签，不据此重新判分。"),
        dict(case_id="path-2432-11",observation="病历既说经皮穿刺在解剖条件允许时已完成，后又说经皮不可行、内镜不需要，因为经皮路径曾经可达。",
             implication="条件同时询问经皮可行或因不可行而需内镜，且病例混有时态；保留既有gold=0。"),
        dict(case_id="path-800-2",observation="先记利尿剂使尿量初增，随后记尿量未维持、液体超负荷持续、呋塞米挑战后反应差。",
             implication="G4问当前是否有利尿反应，早期反应与当前状态的对应关系不够直接；保留既有gold=1。")]
    evidence=dict(protocol="Post-run descriptive analysis of saved predictions. No models, new inputs, exclusions, relabeling, or parameter selection.",
        paths=path_result,historical=historical,existing_text_cautions=cautions,
        interpretation_limits=["Main-node coverage is a (source_id,gid) check; original_100_subset is a case subset, not node coverage.",
            "Historical natural and ASTER scores use the same saved inputs and template. Absolute accuracy is not compared across the main and historical templates.",
            "Small score margin is an observed location of flips, not a proposed gate or a proven root cause of representation failure.",
            "Text cautions are not independent clinical adjudication and do not change any reported denominator."])
    (out/"error_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[
        "# ASTER 路径与历史诊断错误分析","",
        "本报告只分析现成输入与保存分数；未新增输入、重跑模型、修改标签或重选参数。分数为“是”减“否”，大于零判是，同分判否。",
        "",
        "432 例路径中，完整阅读正确 404 例，ASTER 正确 401 例：纠正 2 例、退化 5 例。",
        f"按指南来源配对重采样，ASTER 减完整阅读的路径正确率差值为 -0.69 个百分点，95% 区间为 [{ci[0]*100:.2f}, {ci[1]*100:.2f}] 个百分点。",
        "",
        "## 路径变化发生在哪里","",
        "| 病例 | 变化 | 首个分歧节点 | 原标签 | 完整阅读分数 | ASTER 分数 | 原100病例子集 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for c in changes:
        lines.append(f"| {c['case_id']} | {'纠正' if c['outcome']=='corrected' else '退化'} | {c['gid']} | {c['gold']} | {c['full_ell']:.2f} | {c['aster_ell']:.2f} | {'是' if c['original_100_subset'] else '否'} |")
    lines += ["",
        f"两种方法共同访问 {len(common)} 个病例—节点，其中仅 {len(changed_common)} 次预测翻转。上述 7 次改变路径成败；另 1 次发生在已偏离标准路径的节点，两种方法整条路径仍然错误。",
        "这 8 次翻转对应的节点均未包含在主实验 200 节点中，但均来自原 20 份测试指南，不是新指南。原 100 病例子集包含其中 1 个退化病例，但病例子集不等于节点覆盖。",
        f"7 次成败变化的完整阅读分数绝对值均不超过 2.5；所有共同节点的绝对分数中位数为 {path_result['common_abs_full_margin_median']:.3f}。固定编辑主要改变了未覆盖节点上的少数接近判定阈值的结果。",
        "低分数说明这些决策容易被编辑推过零点；5 个退化和 2 个纠正不能说明所有近阈值输入都更差，也不能单凭这一点确定是信号空间遗漏、干扰方向估计偏差，还是模型对病例冲突的处理发生变化。",
        "",
        "## 历史事实与偏好诊断","",
        "完整阅读路线下，同一批 168 个输入、同一保存提示，natural 为 119/168 正确，ASTER 为 114/168。全部 5 个变化都是正确变错误，集中于 2 个节点；没有纠正。",
        "",
        "| 条件和数值 | 背景 | 原标签 | natural 分数 | ASTER 分数 |",
        "| --- | --- | --- | ---: | ---: |"]
    for c in historical["full"]["input_flips"]:
        lines.append(f"| {c['condition']}；数值={c['metric_value']} | {c['background']} | {c['gold']} | {c['full_ell']:.2f} | {c['aster_ell']:.2f} |")
    lines += ["",
        "肾结石条件要求 10–20 mm，21 mm 的 neutral/cue 两个输入都由否翻成是；咳嗽峰流量条件要求 <270 L/min，271 的 cue 和 272 的 neutral/cue 也都由否翻成是。分数上移 2.5–3.25，损伤了原本正确的阈值判断。",
        "事实配对从 25/56 降为 22/56，偏好配对从 59/84 降为 57/84。事实配对损失是 21 mm 两个背景及 271 cue；偏好配对损失是 21 mm 和 272 的背景对。",
        "below 的 56 个输入全部不变；above 退化 3 个，same_side 退化 2 个。neutral 退化 2 个，cue 退化 3 个。完整的各路线、各层统计和具体配对编号见 error_evidence.json。",
        "恰好等于条件阈值的 12 个输入在两种方法下均为 12/12 正确；上述损失发生在阈值附近而非恰好等于阈值，二者不矛盾。",
        "这支持一个有限结论：在拟合数据上保护的信号方向没有充分覆盖这些现成阈值判断。不能把不同模板间的绝对准确率差异解释成 ASTER 效果；此处结论来自历史模板内部的同输入配对。",
        "",
        "## 现成病例的解释问题","",
        "5 个路径退化病例中，有 3 个包含正反叙述或时间状态不清的情况，原文与既有标签均保留："]
    for c in cautions:
        lines += [f"- {c['case_id']}：{c['observation']} {c['implication']}"]
    lines += ["",
        "这些问题可能使路径准确率混合了模型变化与病例表达问题；本轮未新增筛选、排除或标注，不能用它们撤销已报告的 5 个退化。",
        "",
        "## 可直接推进的检查","",
        "优先查看上述 7 个共同分歧节点和历史 2 个阈值节点的已有信号/干扰投影诊断，解释固定编辑为何产生这些分数变化；把主实验覆盖节点与未覆盖节点的结果分开描述。若以后做新实验，应预先固定检查方案，不据这 7 例反向选择配置。",
        "",
        "复算：在服务器的实验运行目录执行 python analysis_postrun/analyze_errors.py --root .。脚本仅读取保存分数，写本报告和证据 JSON。",""]
    (out/"error_analysis.zh.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(paths_corrected=path_result["corrected"],paths_degraded=path_result["degraded"],
        common_nodes=len(common),flips=len(changed_common),paired_ci95=ci,
        historical_full_corrected=historical["full"]["corrected"],historical_full_degraded=historical["full"]["degraded"]),ensure_ascii=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("/home/gcx/aster_existing_data_20261106"))
    main(p.parse_args().root)
