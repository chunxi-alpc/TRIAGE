#!/usr/bin/env python3
"""Dev96-only comparison against existing A1; no inference or expansion."""
import json,hashlib,sys,csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(REPO))
from experiments.care_module1_v8_1.analyze import dataset,evaluate
OLD=REPO/"experiments/care_module1_semantic_v8"
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def jsonl(p,x):p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in x))
def main():
    cases=dataset("dev");out=HERE/"analysis";out.mkdir(exist_ok=True)
    paths={"A1":OLD/"results/dev_A1/predictions.jsonl","A1_U":HERE/"results/dev_A1_U/predictions.jsonl"}
    manifest=json.loads((paths["A1_U"].parent/"manifest.json").read_text())
    assert manifest["status"]=="completed"
    metrics={};details={};runs={}
    for arm,path in paths.items():
        rows=read(path);assert len(rows)==96 and {r["case_id"] for r in rows}=={c["case_id"] for c in cases}
        m,rs=evaluate(cases,rows);m["arm"]=arm;m["prediction_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        if arm=="A1":
            for k in ("added_model_calls","added_inference_seconds","added_input_tokens","added_output_tokens","added_total_tokens"):m[k]=0
        metrics[arm]=m;details[arm]=rs;runs[arm]=rows
    baseline={(r["case_id"],r["rule_id"]):r for r in details["A1"]}
    changes=[];counts={k:0 for k in ["corrected","broken","stayed_correct","stayed_wrong","state_changed","U_recovered","U_lost","F_to_T_added","F_to_T_removed"]}
    for r in details["A1_U"]:
        a=baseline[r["case_id"],r["rule_id"]]
        group="stayed_correct" if a["correct"] and r["correct"] else "broken" if a["correct"] else "corrected" if r["correct"] else "stayed_wrong"
        counts[group]+=1
        counts["state_changed"]+=a["predicted"]!=r["predicted"]
        counts["U_recovered"]+=r["gold"]=="U" and a["predicted"]!="U" and r["predicted"]=="U"
        counts["U_lost"]+=r["gold"]=="U" and a["predicted"]=="U" and r["predicted"]!="U"
        counts["F_to_T_added"]+=r["gold"]=="F" and a["predicted"]!="T" and r["predicted"]=="T"
        counts["F_to_T_removed"]+=r["gold"]=="F" and a["predicted"]=="T" and r["predicted"]!="T"
        if a["predicted"]!=r["predicted"]:
            changes.append(dict(r,A1_predicted=a["predicted"],A1_reason=a["reason_short"],change_type=group))
    protocol=json.loads((HERE/"protocol.json").read_text())
    frozen={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in protocol["freeze_sha256"].items()}
    assert all(frozen.values())
    added=[a for r in runs["A1_U"] for a in r["attempts"]]
    completion={"completed_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"scope":"dev96 only",
      "new_model_requests":len(added),"format_retries":sum(a.get("attempt",1)>1 for a in added),
      "failed_cases":sum(r["status"]!="ok" for r in runs["A1_U"]),"truncated_attempts":sum(a.get("truncated",False) for a in added),
      "peak_allocated_gib":manifest["peak_allocated_gib"],"peak_reserved_gib":manifest["peak_reserved_gib"],
      "elapsed_seconds":manifest["elapsed_seconds"],"frozen_hash_matches":frozen,"expanded":False}
    ratios={k:metrics["A1_U"][k]/metrics["A1"][k] for k in ("model_calls","input_tokens","output_tokens","total_tokens","summed_case_inference_seconds")}
    write(out/"metrics.json",metrics);write(out/"changes.json",dict(counts=counts,changes=changes))
    write(out/"cost_ratios.json",ratios);write(HERE/"completion_manifest.json",completion)
    jsonl(out/"rule_predictions.jsonl",[dict(arm=a,**r) for a,rs in details.items() for r in rs])
    jsonl(out/"errors.jsonl",[dict(arm=a,**r) for a,rs in details.items() for r in rs if not r["correct"]])
    fields=["arm","correct","rules","accuracy","U_recall","F_to_T","F_to_T_rate","model_calls","added_model_calls","input_tokens","output_tokens","total_tokens","summed_case_inference_seconds"]
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(metrics.values())
    a=metrics["A1"];b=metrics["A1_U"]
    lines=["# A1 增加针对 U 的对照示例：原96例开发集", "",
      "报告时间："+completion["completed_at"],"",
      "本轮只运行一个新增版本及原96例开发集。A1结果直接复用缓存，未运行历史192例或任何72例数据集，也未启动后续实验。","",
      "## 唯一改动","",
      "沿用原A1系统指令、目标输入、输出字段及顺序、Phi-4＋TrialMatchAI LoRA、NF4 4bit、batch8、greedy、384输出token预算。只替换训练演示：从三个演示中的四段病例，改成两个对照组中的六段新编病例。OR组覆盖T/F/U和NOT U；例外组覆盖明确不存在、明确存在和未知，并示范无关未知不影响已成立的T或F。示例在推理前冻结，不使用开发集原病例或金标作演示。","",
      "## 结果","",
      "|指标|原A1缓存|A1＋U对照示例|","|---|---:|---:|",
      f"|规则准确率|{a['correct']}/{a['rules']}（{a['accuracy']:.2%}）|{b['correct']}/{b['rules']}（{b['accuracy']:.2%}）|",
      f"|U正确识别|{a['confusion']['U']['U']}/{a['per_state']['U']['support']}（{a['U_recall']:.2%}）|{b['confusion']['U']['U']}/{b['per_state']['U']['support']}（{b['U_recall']:.2%}）|",
      f"|U精确率|{a['per_state']['U']['precision']:.2%}|{b['per_state']['U']['precision']:.2%}|",
      f"|F→T|{a['F_to_T']}/{a['F_to_T_denominator']}|{b['F_to_T']}/{b['F_to_T_denominator']}|",
      f"|宏平均F1|{a['macro_f1']:.4f}|{b['macro_f1']:.4f}|",
      f"|目标反事实配对全对|{a['target_pair_correct']}/{a['target_pair_total']}|{b['target_pair_correct']}/{b['target_pair_total']}|","",
      f"相对A1纠正 {counts['corrected']} 条、改错 {counts['broken']} 条；恢复原本漏识别的U {counts['U_recovered']} 条，同时丢失原本正确U {counts['U_lost']} 条；新增F→T {counts['F_to_T_added']} 条、消除F→T {counts['F_to_T_removed']} 条。","",
      "## 成本","",
      "|指标|原A1完整成本|本轮候选完整成本|","|---|---:|---:|",
      f"|模型请求|{a['model_calls']}|{b['model_calls']}|",
      f"|本轮实际新增请求|0|{b['added_model_calls']}|",
      f"|输入token|{a['input_tokens']}|{b['input_tokens']}|",
      f"|输出token|{a['output_tokens']}|{b['output_tokens']}|",
      f"|总token|{a['total_tokens']}|{b['total_tokens']}|",
      f"|批次生成秒|{a['summed_case_inference_seconds']:.1f}|{b['summed_case_inference_seconds']:.1f}|","",
      f"输入token比A1增加 {(ratios['input_tokens']-1):.2%}，总token增加 {(ratios['total_tokens']-1):.2%}。历史与本轮耗时来自不同运行，不能视为严格同条件在线延迟比较。GPU0峰值已分配显存 {manifest['peak_allocated_gib']:.2f} GiB，含加载运行时间 {manifest['elapsed_seconds']:.1f} 秒。格式重试 {completion['format_retries']} 次，失败 {completion['failed_cases']} 例。","",
      "## 结果解释与扩展建议","", "本轮总正确数增加10条，由T类增加8条、F类增加6条、U类减少4条组成。它改善了总体准确率和F→T，但没有实现U识别改善，宏平均F1也由0.6976下降至0.6877。建议暂不以U改进方案扩展；保留为总体准确率候选，是否扩展由用户另行决定。","", "丢失的6条原本正确U全部集中在3个仅提供亲属年龄的病例，涉及患者本人信息缺失与嵌套逻辑；恢复的2条U来自同一年龄矛盾病例。这说明简单OR/例外未知演示未在本开发集稳定覆盖更复杂的信息归属和组合条件，不能据此断言模型内部原因。","", "## 改变实例",""]
    for group in ("corrected","broken"):
        chosen=sorted([r for r in changes if r["change_type"]==group],key=lambda r:(r["gold"]!="U",r["case_id"],r["rule_id"]))
        for r in chosen[:3]:
            lines += [f"### {group}：{r['case_id']} / {r['rule_id']}","",
              f"金标 {r['gold']}；A1 {r['A1_predicted']} → 新版 {r['predicted']}。","",
              "**病例：** "+r["patient"],"","**规则：** "+r["rule_text"],"",
              "**A1理由：** "+str(r["A1_reason"]),"","**新版理由：** "+str(r["reason_short"]),""]
    lines+=["## 复现与范围","",
      "运行入口见 README.md；逐例原始输出保存在 results/dev_A1_U/predictions.jsonl，全部变化保存在 analysis/changes.json。A1原始缓存保持不变。",
      "这是在已反复使用的开发集上的探索结果，不能代替新数据验证。是否扩展需在查看上述收益和代价后另作决定；本轮已停在96例。",""]
    (HERE/"REPORT.zh.md").write_text("\n".join(lines))
    print(json.dumps({"baseline_correct":a["correct"],"candidate_correct":b["correct"],
      "U_correct":[a["confusion"]["U"]["U"],b["confusion"]["U"]["U"]],
      "F_to_T":[a["F_to_T"],b["F_to_T"]],"total_token_ratio":ratios["total_tokens"],"changes":counts,"expanded":False}))
if __name__=="__main__":main()
