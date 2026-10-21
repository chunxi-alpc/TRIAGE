#!/usr/bin/env python3
"""Score all official TrialGPT expert labels, without changing A1 or its output."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from prepare import COUNT, HERE, LABEL_MAP, RULE_ID, sha256

CLASSES = ("T", "F", "U")
PRED_CLASSES = ("T", "F", "U", "ERROR")


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hash_id(prefix, value):
    return hashlib.sha256((prefix + ":" + value).encode()).hexdigest()[:16]


def decode_a1(record):
    if record is None:
        return "ERROR"
    rules = record.get("rules")
    if record.get("status") != "ok" or not isinstance(rules, list) or len(rules) != 1:
        return "ERROR"
    rule = rules[0]
    if not isinstance(rule, dict) or rule.get("rule_id") != RULE_ID:
        return "ERROR"
    v = rule.get("state")
    return v if v in CLASSES else "ERROR"


def class_metrics(rows):
    """Rows are strictly gold-mapped T/F/U; ERROR counts as an incorrect prediction."""
    den = len(rows)
    correct = sum(x["prediction"] == x["mapped_gold"] for x in rows)
    confusion = {g: {p: 0 for p in PRED_CLASSES} for g in CLASSES}
    for x in rows:
        confusion[x["mapped_gold"]][x["prediction"]] += 1
    per_class = {}
    f1 = []
    for label in CLASSES:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in CLASSES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(score)
        per_class[label] = {"gold": actual, "predicted": predicted,
                            "true_positive": tp, "precision": precision,
                            "recall": recall, "f1": score}
    return {"correct": correct, "total": den,
            "accuracy": correct / den if den else None,
            "macro_f1": sum(f1) / len(f1),
            "per_class": per_class, "confusion": confusion}


def grouped_stats(rows, field):
    out = {}
    for value in sorted({str(row[field]) for row in rows}):
        values = [x for x in rows if str(x[field]) == value and x["mapped_gold"] is not None]
        all_in_group = sum(str(x[field]) == value for x in rows)
        core = class_metrics(values)
        out[value] = {"all_rows": all_in_group, "mapped_rows": len(values),
                      "correct": core["correct"], "accuracy": core["accuracy"]}
    return out


def cluster_bootstrap(rows, field, replicates=10000, seed=20261021):
    """Resample complete patient/trial clusters; report descriptive 95% interval."""
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["mapped_gold"] in CLASSES:
            grouped[row[field]].append(int(row["prediction"] == row["mapped_gold"]))
    keys = sorted(grouped)
    if not keys:
        return {"clusters": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    totals = {k: (sum(v), len(v)) for k, v in grouped.items()}
    samples = []
    for _ in range(replicates):
        picked = [totals[rng.choice(keys)] for _ in keys]
        denominator = sum(n for c, n in picked)
        if denominator:
            samples.append(sum(c for c, n in picked) / denominator)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"clusters": len(keys), "bootstrap": len(samples), "seed": seed,
            "ci95": [lo, hi]}


def evaluate(labels, pred_records, source_manifest):
    by_id = {}
    for p in pred_records:
        cid = p.get("case_id")
        if not cid or cid in by_id:
            raise ValueError("missing or duplicate case_id in A1 predictions")
        by_id[cid] = p
    expected = {x["case_id"] for x in labels}
    if len(labels) != COUNT or len(expected) != COUNT:
        raise ValueError("incomplete expert population")
    extras = set(by_id) - expected
    if extras:
        raise ValueError("predictions include cases outside frozen official source")
    derived = []
    for lab in labels:
        cid = lab["case_id"]
        prediction = decode_a1(by_id.get(cid))
        g = lab["mapped_gold"]
        derived.append({
            "case_id": cid, "patient_id": lab["patient_id"], "trial_id": lab["trial_id"],
            "criterion_type": lab["criterion_type"],
            "training_flag": lab["training_flag"],
            "evidence_present": lab["evidence_present"],
            "expert_eligibility": lab["expert_eligibility"],
            "mapped_gold": g, "prediction": prediction,
            "correct": prediction == g if g is not None else None,
            "source_input_valid": lab["valid_input"],
        })
    eligible = [x for x in derived if x["mapped_gold"] in CLASSES]
    core = class_metrics(eligible)
    status = ("complete" if len(by_id) == sum(x["valid_input"] for x in labels)
              and all(x["prediction"] != "ERROR" or x["case_id"] in by_id or not x["source_input_valid"]
                      for x in derived) else "incomplete")
    # Valid A1 calls may return a parse failure: this is a scored failure,
    # not an incomplete job. Missing predicted case IDs are incomplete.
    counts = collections.Counter(x["prediction"] for x in derived)
    summary = {
        "status": status,
        "dataset": source_manifest["source_dataset"],
        "dataset_revision": source_manifest["revision"],
        "dataset_parquet_sha256": source_manifest["parquet_sha256"],
        "total_official_rows": COUNT,
        "a1_prediction_rows": len(by_id),
        "official_rows_without_prediction": sum(x["case_id"] not in by_id for x in derived),
        "expert_native_labels": dict(collections.Counter(x["expert_eligibility"] for x in derived)),
        "unmapped_not_applicable": sum(x["mapped_gold"] is None for x in derived),
        "mapped_population": len(eligible),
        "mapping_coverage_of_all_rows": len(eligible) / COUNT,
        "valid_model_state": sum(x["prediction"] in CLASSES for x in derived),
        "parse_or_missing_error_count": counts["ERROR"],
        "mapped_scoring": core,
        "strata": {
            "criterion_type": grouped_stats(derived, "criterion_type"),
            "training_flag": grouped_stats(derived, "training_flag"),
            "expert_evidence_present": grouped_stats(derived, "evidence_present"),
        },
        "patient_cluster": cluster_bootstrap(derived, "patient_id"),
        "trial_cluster": cluster_bootstrap(derived, "trial_id"),
        "limitations": [
            "Experts label trial inclusion/exclusion eligibility; mapped states are not literal rule truth labels.",
            "Not applicable has no frozen A1 TFU equivalent and remains in the full denominator.",
            "Frozen PR12 A1 prompt describes invented routing simulation; this is domain transfer.",
            "One criterion per input differs from historical three-rule A1 prompts.",
            "The 1015 rows reuse only 53 patient notes and 103 clinical trials.",
            "No score threshold or hyperparameter selected on expert labels.",
        ],
    }
    return summary, derived


def write_results(out, summary, rows):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # Anonymous hashes only; original detailed records stay under private/.
    public_rows = []
    for row in rows:
        public_rows.append({
            "case_id": row["case_id"],
            "patient_hash": hash_id("patient", row["patient_id"]),
            "trial_hash": hash_id("trial", row["trial_id"]),
            "criterion_type": row["criterion_type"],
            "training_flag": row["training_flag"],
            "expert_evidence_present": row["evidence_present"],
            "expert_eligibility": row["expert_eligibility"],
            "mapped_gold": row["mapped_gold"],
            "predicted_a1_state": row["prediction"],
            "correct": row["correct"],
        })
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for x in public_rows:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {**summary, "predictions_sha256": sha256(out / "predictions.jsonl")}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    core = summary["mapped_scoring"]
    report = [
        "# 冻结 A1 × TrialGPT 专家标准全量评估",
        "",
        "状态：" + summary["status"].upper() + "。仅统计已实际出现的冻结 A1 原始预测。",
        "",
        "官方专家标注母集：" + str(COUNT) + " 条。",
        "专家 T/F/U 投影可评分条数：" + str(summary["mapped_population"]) + "。",
        "原生 not applicable（不能映射）：" + str(summary["unmapped_not_applicable"]) + "。",
        "A1 实际预测条数：" + str(summary["a1_prediction_rows"]) + "。",
        "A1 解析/缺失失败数：" + str(summary["parse_or_missing_error_count"]) + "。",
        "专家标签投影一致率：" + str(core["correct"]) + "/" +
            str(core["total"]) + "（" +
            (("%.4f" % core["accuracy"]) if core["accuracy"] is not None else "N/A") + "）。",
        "宏 F1：" + ("%.4f" % core["macro_f1"]),
        "",
        "完整细节：summary.json / predictions.jsonl。患者级和试验级 bootstrap 均只作为",
        "此数据集中群相关样本的描述性区间。",
        "",
        "**解释限制：** 这是官方专家资格标签的三类投影一致率，不是原生 TrialGPT 六类任务",
        "的准确率；A1 原 system prompt、few-shot、模型和推理上限未改变。不能与",
        "TrialGPT 论文 87.3% 的原生任务结果直接比较。全部原始输入和逐例长输出仅保留 private/。",
    ]
    if summary["status"] != "complete":
        report.insert(4, "**INCOMPLETE：尚有未返回的 A1 请求，不得引用以上结果为全量准确率。**")
    (out / "REPORT.zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private", type=Path, default=HERE / "private")
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--predictions", type=Path, nargs="*")
    args = p.parse_args()
    manifest = json.loads((args.private / "dataset_manifest.json").read_text())
    if sha256(args.private / "expert_labels_all.jsonl") != manifest["file_sha256"]["expert_labels_all.jsonl"]:
        raise ValueError("Expert gold file changed since freeze")
    labels = read_jsonl(args.private / "expert_labels_all.jsonl")
    files = args.predictions or [args.private / ("a1_%d/predictions.jsonl" % i) for i in range(4)]
    preds = [row for path in files for row in read_jsonl(path)]
    summary, rows = evaluate(labels, preds, manifest)
    write_results(args.out, summary, rows)
    print(json.dumps({"status": summary["status"],
                      "a1_predictions": summary["a1_prediction_rows"],
                      "mapped_accuracy": summary["mapped_scoring"]["accuracy"]}))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

