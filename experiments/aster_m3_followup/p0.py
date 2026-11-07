"""Recompute frozen path comparisons; no model calls and no parameter selection."""
from .common import *
def run():
    cases=rows(OLD/"data/test.jsonl");pred=old_predictions()
    sources=[c["source_id"] for c in cases]
    comparisons={}
    for a,b in [("learned_reference","full"),("learned_self","full"),
                ("learned_reference","learned_self"),("learned_reference","aster")]:
        comparisons[a+" minus "+b]=paired(
            [pred[c["case_id"],a]["correct"] for c in cases],
            [pred[c["case_id"],b]["correct"] for c in cases],sources)
    splits={s:rows(OLD/"data"/f"{s}.jsonl") for s in ["train","dev","test"]}
    counts={s:dict(cases=len(rs),sources=len({r["filename"] for r in rs}),
        gold_nodes=sum(len(r["gold_path"]) for r in rs)) for s,rs in splits.items()}
    overlaps={a+" vs "+b:len({r["filename"] for r in splits[a]}&{r["filename"] for r in splits[b]})
        for a,b in [("train","dev"),("train","test"),("dev","test")]}
    tasks=[];changes=[]
    for c in cases:
        cid=c["case_id"]
        for n in c["gold_path"]:
            missing=[m for m in METHODS if n["gid"] not in {q["gid"] for q in pred[cid,m]["predictions"]}]
            if missing:tasks.append(dict(case_id=cid,gid=n["gid"],label=n["label"],methods=missing))
        if pred[cid,"full"]["correct"]==pred[cid,"learned_reference"]["correct"]:continue
        scores={m:{q["gid"]:q for q in pred[cid,m]["predictions"]} for m in METHODS}
        first=next(n for n in c["gold_path"] if n["gid"] in scores["full"]
            and n["gid"] in scores["learned_reference"]
            and scores["full"][n["gid"]]["label"]!=scores["learned_reference"][n["gid"]]["label"])
        gid=first["gid"]
        condition=next(g["rules"][0]["condition"] for g in c["guideline"]["groups"] if g["group_id"]==gid)
        changes.append(dict(case_id=cid,source_id=c["source_id"],gid=gid,label=first["label"],condition=condition,
            outcome="corrected" if pred[cid,"learned_reference"]["correct"] else "degraded",
            first_error_full=next((n["gid"] for n in c["gold_path"] if n["gid"] in scores["full"]
                and scores["full"][n["gid"]]["label"]!=n["label"]),None),
            scores={m:scores[m].get(gid) for m in METHODS},
            path_magnitude={m:pred[cid,m]["magnitude"] for m in METHODS}))
    write_rows(ROOT/"p0/missing_nodes.jsonl",tasks)
    write_rows(ROOT/"p0/changed_cases.jsonl",changes)
    write(ROOT/"p0/paired.json",dict(comparisons=comparisons,split_counts=counts,
        source_filename_overlaps=overlaps,missing_node_method_scores=sum(len(x["methods"]) for x in tasks),
        independence="Official MEGA split, disjoint filenames in this training/development/test selection; repeatedly used benchmark, not a new blinded test."))
    lines=["# M3 P0：冻结结果直接比较","",
        "MEGA-CDP 测试分母为 7,458 例、440 个指南来源；金标准路径共 45,627 个节点。训练/开发为 200/100 例，分别来自 20/10 个来源，文件名与测试来源无交集。该官方测试集已用于历史研究，不称为新的盲测。","",
        "|比较|净变化|差值（百分点）|来源 bootstrap 95% 区间|来源配对置换 p|McNemar p|",
        "|---|---:|---:|---|---:|---:|"]
    for k,v in comparisons.items():
        lo,hi=v["source_bootstrap_95ci"]
        lines.append(f"|{k}|{v['net']:+d}|{100*v['difference']:+.4f}|[{100*lo:+.4f}, {100*hi:+.4f}]|{v['source_permutation_p']:.4f}|{v['mcnemar_exact_p']:.4f}|")
    lines+=["","按病例配对，来源 bootstrap 保留同一指南内的全部病例，共 20,000 次。来源置换对整组交换方法；McNemar 为病例级补充，不能代替来源相关性处理。","",
        "6 个纠正的原始分数差均为 +0.25 或 +0.5；2 个退化的原始分数差为 0 或 +0.5。分数差为是/否 logit 差，零分按否处理。所有变化集中在阈值附近，支持局部阈值变化的解释，但需要校准对照才能归因。","",
        "原结果只保存路径平均编辑量，没有保存逐节点 gate。8 例 gate 和编辑量将用冻结检查点补跑，并保留原分数与补跑分数。固定金标准路径复用已保存节点，只补缺失节点；金标准仅指定评估节点，不进入提示。","",
        "计算成本：完整阅读为一次前缀计算和一次末位置评分；自身参考复用同一前缀，需两次末位置评分；辅助参考为两套提示，各计算前缀和末位置。两种旧学习式版本均不是一次末位置前向。"]
    (ROOT/"p0/REPORT.zh.md").write_text("\n".join(lines)+"\n")
    status("p0_statistics",state="complete",comparisons=len(comparisons),changed_cases=len(changes))
    return comparisons
if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False))
