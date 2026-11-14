#!/usr/bin/env python3
"""Summarize the authored 124-case post-hoc review; never write gold or predictions."""
import collections, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LABELS = {"A": "明确的 A1 判断错误", "B": "标签定义或上下文差异", "C": "无法确定"}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    rows = [json.loads(x) for x in (HERE/"u_error_review.jsonl").read_text().splitlines()]
    original = [json.loads(x) for x in (EXP/"results/predictions.jsonl").read_text().splitlines()]
    expected = {x["case_id"]: x for x in original if x["mapped_gold"] == "U" and x["predicted_a1_state"] in ("T","F")}
    assert len(rows) == len(expected) == 124
    assert len({r["case_id"] for r in rows}) == 124 and {r["case_id"] for r in rows} == set(expected)
    for row in rows:
        src = expected[row["case_id"]]
        assert row["original_expert_eligibility"] == src["expert_eligibility"] == "not enough information"
        assert row["original_predicted_state"] == src["predicted_a1_state"]
        assert row["criterion_type"] == src["criterion_type"]
        assert row["category"] in LABELS and row["rationale_zh"]
    guards = json.loads((HERE/"review_provenance.json").read_text())
    for relative, wanted in guards["protected_file_sha256"].items():
        assert digest(EXP/relative) == wanted, "Original artifact changed: " + relative
    counts = collections.Counter(r["category"] for r in rows)
    summary = {
        "review_population": 124, "official_population": 1015, "original_gold_U": 293,
        "original_correct_U": 169, "original_U_recall": 169/293,
        "categories": {k: {"name":v, "n":counts[k], "percent_of_124":100*counts[k]/124} for k,v in LABELS.items()},
        "A_reason_groups": dict(collections.Counter(r["reason_group"] for r in rows if r["category"]=="A")),
        "by_prediction": {state:dict(collections.Counter(r["category"] for r in rows if r["original_predicted_state"]==state)) for state in ("T","F")},
        "by_criterion_type": {typ:dict(collections.Counter(r["category"] for r in rows if r["criterion_type"]==typ)) for typ in ("inclusion","exclusion")},
        "gold_modified": False, "predictions_modified":False, "original_metrics_modified":False,
        "reviewer":"Codex, single non-blinded post-hoc text review; not independent clinician adjudication",
        "counts_are_review_classifications_not_proven_causal_shares":True,
        "no_corrected_gold_or_adjusted_accuracy":True,
        "review_rows_sha256":digest(HERE/"u_error_review.jsonl"),
        "protected_artifacts_verified":len(guards["protected_file_sha256"])
    }
    (HERE/"u_error_review_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    lines=["# 124 条专家 U → A1 T/F 的逐条分析","",
      "本次复核覆盖全部124条分歧：84条预测F、40条预测T。逐条查看冻结输入中的完整患者说明、完整单条标准、A1最终状态及理由；这些病例的原专家标签均为not enough information。没有重新运行GPU、修改金标、修改原预测或调整原评分。","",
      "这是Codex进行的一次非盲、事后文本复核，不是独立临床专家裁定；分类是解释性分析，不是新增T/F/U金标，也不是已识别的真实因果比例。","",
      "## 分类规则","",
      "- **A：明确的 A1 判断错误。** 完整输入中可定位到明确的文本或逻辑缺陷，如把缺失当否定、漏必要条件、混淆对象或时间。仅理由不佳但完整输入另有合理支持时，不归A。",
      "- **B：标签定义或上下文差异。** 存在具体可辩护的另一种规则解释、证据完整性要求、评估时点、群体指代或标准片段缺失。表示分歧可能来自这些因素，不表示专家标签已被证明错误或A1已被证明正确。",
      "- **C：无法确定。** 主要依赖医学诊断、分级、预后或其他尚不足以裁定的推断，既无明确文本错误，也无足够具体的口径解释。",
      "- 对既有上下文缺失又有直接逻辑缺陷的样例，若缺陷不依赖补齐上下文即可确定，归A；不能如此区分时归B或C。每条只选一个主分类，理由保留其他因素。","",
      "## 数量与占比","",
      "| 分类 | 数量 | 占124条 |","|---|---:|---:|"]
    for k,v in LABELS.items():
        lines.append(f"| {v} | {counts[k]} | {100*counts[k]/124:.2f}% |")
    lines += ["","明确A1错误占本次复核的多数，不能主要用六类到三类映射来解释U召回不足；但B/C合计34条不能确定归于A1。此结论不等于重新裁定专家标签，也不据此重算准确率。","",
      "原U召回率仍为169/293=57.68%，原全量投影一致率仍为722/960=75.21%，宏F1仍为0.7334。","",
      "## 明确 A1 错误的主要类型","",
      "| 原因 | 数量 | 占A类90条 |","|---|---:|---:|"]
    for reason,n in summary["A_reason_groups"].items():
        lines.append(f"| {reason} | {n} | {100*n/counts['A']:.2f}% |")
    lines += ["","原因标签是每条复核后的描述性归组，不是关键词自动分类；一个样例可能含多种缺陷，表中仅计主原因。","",
      "## 按输出和标准类型拆分","",
      "| 分组 | A | B | C | 合计 |","|---|---:|---:|---:|---:|"]
    for field, values in (("original_predicted_state",("T","F")),("criterion_type",("inclusion","exclusion"))):
        for value in values:
            c=collections.Counter(r["category"] for r in rows if r[field]==value)
            lines.append(f"| {value} | {c['A']} | {c['B']} | {c['C']} | {sum(c.values())} |")
    lines += ["","## 关键实例与复核修正","",
      "- 949：A1承认没有肝酶信息却给F；831、783：只核实部分必要条件；250：把子宫尺寸当肌瘤尺寸。这些不需要改写TrialGPT标签即可定位A1问题。",
      "- 526：饮食控制与药物治疗要求可能构成明确反例；810：实际有经食管超声未见异常的记录，不能只看A1简短理由就认定其没有反证。",
      "- 937：A1引用配合研究的理由不充分，但病例确有咳痰和痰培养，原标准只要求很可能能提供样本；输出本身有可辩护解释，归B。",
      "- 889：原先举例指出会说话不等于会英语，这个理由缺陷仍成立；但完整病例还含英语单词倒拼任务，可能是语言线索，因此本次不把输出本身认定为明确错误，归C。",
      "- 491：群体病例中部分成员怀孕，未指定评估哪一人；232、480：术中决定或未来退出流程与当前状态不同；均属具体上下文或评估对象差异。","",
      "## 全部逐条结论","",
      "编号为官方annotation_id。表中仅使用派生概述；患者原文、标准原文及长模型输出仍保存在服务器private/。每条JSONL包含冻结输入及原预测记录的SHA256，可定位并复核。","",
      "| annotation_id | A1 | 分类 | 原因 | 逐条依据 |","|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['annotation_id']} | {r['original_predicted_state']} | {r['category']} | {r['reason_group']} | {r['rationale_zh'].replace('|','/')} |")
    lines += ["","## 可复核文件与限制","",
      "- u_error_review.jsonl：124条独立的事后复核记录，每条含原标签引用、原预测引用、分类、具体理由和源记录哈希。没有建议金标字段。",
      "- u_error_review_summary.json：由逐条记录生成的计数、占比及分层。",
      "- review_provenance.json：受保护原始金标、输入、原始预测、原结果文件的哈希及原实验版本。",
      "- build_u_review.py：只读取原实验产物并生成本分析的汇总和报告；不写回prepare/evaluate或原始结果。服务器执行 python experiments/trialgpt_a1_full_20261021/analysis/build_u_review.py 可复核计数与哈希。",
      "- B类解释是可定位的候选解释，不是专家原标注意图；数据中的GPT-4解释不能当作专家理由，本次未用其作裁决依据。",
      "- C类保留不确定，不用外部医学推断补写患者事实。分类未经过独立临床专家复核，不能用于训练标签或宣称改进后的模型成绩。",
      "- 单条标准缺少原试验上下文是已知输入限制；本次未补充这些内容重跑模型，原任务与统计分母保持一致。",""]
    (HERE/"U_ERROR_REVIEW.zh.md").write_text("\n".join(lines))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":
    main()
