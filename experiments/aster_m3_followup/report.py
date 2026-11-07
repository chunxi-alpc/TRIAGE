"""Summarize saved complete denominators, with explicit pending/skipped stages."""
import re
from collections import Counter
from .common import *

def family(source):
    return re.sub(r"_?\d+$", "", source)

def evaluate_table(dataset, expected, baseline=None, planned=()):
    records = {}
    pending = {m: 0 for m in planned}
    for method, values in (baseline or {}).items():
        by = {r["case_id"]: r for r in values}
        if len(by) == expected:
            records[method] = list(by.values())
            pending.pop(method, None)
        else:
            pending[method] = len(by)
    for path in sorted((ROOT / "predictions" / dataset).glob("*.jsonl")):
        by = {r["case_id"]: r for r in rows(path)}
        if len(by) == expected:
            records[path.stem] = list(by.values())
            pending.pop(path.stem, None)
        else:
            pending[path.stem] = len(by)
    medguide = dataset.startswith("medguide")
    summary = {}
    for method, values in records.items():
        source_ids = {r["source_id"] for r in values}
        summary[method] = dict(
            n=len(values), correct=sum(r["correct"] for r in values),
            accuracy=sum(r["correct"] for r in values) / len(values),
            source_pages=len(source_ids),
            statistical_sources=len({family(s) for s in source_ids}) if medguide else len(source_ids))
    comparisons = {}
    references = ["full", "bias", "learned_reference", "random_13", "random_37", "random_73",
                  "online_n", "paired_n", "paired_s"]
    by_method = {m: {r["case_id"]: r for r in values} for m, values in records.items()}
    for a, left in by_method.items():
        for b in references:
            if a == b or b not in by_method:
                continue
            right = by_method[b]
            if set(left) != set(right):
                continue
            ids = sorted(left)
            sources = [family(left[i]["source_id"]) if medguide else left[i]["source_id"] for i in ids]
            comparisons[a + " minus " + b] = paired(
                [left[i]["correct"] for i in ids], [right[i]["correct"] for i in ids], sources)
    return dict(expected=expected, state="complete" if not pending else "partial",
                methods=summary, comparisons=comparisons, pending=pending,
                source_unit="disease_family" if medguide else "guideline_source"), records

def fixed_nodes(cases, original):
    scores = {(cid, method, n["gid"]): n
              for (cid, method), record in original.items() for n in record["predictions"]}
    for path in sorted((ROOT / "p0").glob("fixed_nodes_*.jsonl")):
        for r in rows(path):
            scores[r["case_id"], r["method"], r["gid"]] = dict(label=r["prediction"], ell=r["ell"])
    expected = sum(len(c["gold_path"]) for c in cases)
    first_errors = []
    wrong_cases = 0
    for case in cases:
        if original[case["case_id"], "full"]["correct"]:
            continue
        wrong_cases += 1
        for node in case["gold_path"]:
            value = scores.get((case["case_id"], "full", node["gid"]))
            if value is not None and value["label"] != node["label"]:
                first_errors.append((case, node))
                break
    result = {}
    for method in METHODS:
        total = correct = first_scored = first_fixed = 0
        left, right, sources = [], [], []
        strata = {}
        for case in cases:
            conditions = {g["group_id"]: g["rules"][0]["condition"] for g in case["guideline"]["groups"]}
            for node in case["gold_path"]:
                value = scores.get((case["case_id"], method, node["gid"]))
                if value is None:
                    continue
                total += 1
                ok = value["label"] == node["label"]
                correct += int(ok)
                base = scores.get((case["case_id"], "full", node["gid"]))
                if base is not None:
                    left.append(ok)
                    right.append(base["label"] == node["label"])
                    sources.append(case["source_id"])
                condition = conditions[node["gid"]]
                kind = ("numeric" if re.search(r"[<>=≤≥]|\d+\s*(?:岁|年|hours|mg|ms|%)", condition, re.I)
                        else "negation" if re.search(r"not |without |无|否认|排除", condition, re.I)
                        else "compound" if re.search(r"\b(and|or)\b|且|或者|或", condition, re.I)
                        else "other")
                entry = strata.setdefault(kind, dict(n=0, correct=0))
                entry["n"] += 1
                entry["correct"] += int(ok)
        for case, node in first_errors:
            value = scores.get((case["case_id"], method, node["gid"]))
            if value is not None:
                first_scored += 1
                first_fixed += int(value["label"] == node["label"])
        comparison = paired(left, right, sources) if total == expected and len(left) == expected else None
        result[method] = dict(
            expected=expected, state="complete" if total == expected else "partial",
            scored=total, correct=correct, accuracy=correct / total if total else None,
            corrected_vs_full=sum(a and not b for a, b in zip(left, right)),
            degraded_vs_full=sum(not a and b for a, b in zip(left, right)),
            paired_node_denominator=len(left), comparison_vs_full=comparison,
            first_error_corrected=first_fixed, first_error_scored=first_scored,
            first_error_expected=len(first_errors), full_wrong_cases=wrong_cases,
            condition_strata=strata,
            condition_strata_definition="exclusive keyword categories: numeric, negation, compound, other; not manual A/P/C/F labels")
    write(ROOT / "p0/fixed_node_summary.json", result)
    return result

def changed_diagnostics():
    changes = rows(ROOT / "p0/changed_cases.jsonl")
    reruns = {}
    for path in sorted((ROOT / "p0").glob("changed_node_reruns_*.jsonl")):
        for r in rows(path):
            reruns[r["case_id"], r["gid"], r["method"]] = r
    values = []
    for case in changes:
        row = dict(case)
        row["reruns"] = {method: reruns.get((case["case_id"], case["gid"], method))
                         for method in ["full", "learned_reference", "learned_self", "aster"]}
        values.append(row)
    expected = len(changes) * 4
    matched = sum(r is not None for row in values for r in row["reruns"].values())
    differences = [r["ell"] - r["original_ell"] for row in values
                   for r in row["reruns"].values() if r is not None and r.get("original_ell") is not None]
    result = dict(state="complete" if matched == expected else "partial",
                  cases=len(changes), expected_method_nodes=expected, scored_method_nodes=matched,
                  rerun_score_changes=sum(abs(d) > 0 for d in differences),
                  max_abs_rerun_score_difference=max(map(abs, differences), default=None), values=values)
    write(ROOT / "p0/changed_node_summary.json", result)
    return result

def common_errors(pilot_rows):
    medgemma = {}
    for path in (DIAG / "baseline/medgemma/bf16_backbone_fp32_label_projection/medguide").glob("*.json"):
        row = read(path)
        if row.get("status") == "complete":
            medgemma[row["sample_id"]] = row
    result = {}
    for method, values in pilot_rows.items():
        shared = [r for r in values if r["case_id"] in medgemma]
        result[method] = dict(
            n=len(shared), source_families=len({family(r["source_id"]) for r in shared}),
            both_wrong=sum(not r["correct"] and not medgemma[r["case_id"]]["correct"] for r in shared),
            qwen_only_wrong=sum(not r["correct"] and medgemma[r["case_id"]]["correct"] for r in shared),
            medgemma_only_wrong=sum(r["correct"] and not medgemma[r["case_id"]]["correct"] for r in shared),
            both_correct=sum(r["correct"] and medgemma[r["case_id"]]["correct"] for r in shared))
    return result

def difference_text(table, a, b):
    value = table["comparisons"].get(a + " minus " + b)
    if value is None:
        return None
    lo, hi = value["source_bootstrap_95ci"]
    finding = ("区间高于零" if lo > 0 else "区间低于零" if hi < 0 else "区间包含零")
    return (f"{a} 相对 {b}：纠正 {value['corrected']}、新增错误 {value['degraded']}，"
            f"净变化 {value['net']:+d}；差值 {100 * value['difference']:+.3f} 个百分点，"
            f"来源 95% 区间 [{100 * lo:+.3f}, {100 * hi:+.3f}]，"
            f"来源配对置换 p={value['source_permutation_p']:.4f}，{finding}。")

def report_table(title, table):
    lines = ["## " + title, "", f"固定分母：{table['expected']}。"]
    if table["state"] == "skipped":
        return lines + ["", "未触发预先约定的扩展评测；该分母是待评测数据量，没有全量成绩。"]
    if table["state"] == "not_decided":
        return lines + ["", "等待试测结果决定是否启动；尚无扩展评测结论。"]
    lines += ["", "|方法|正确/总数|准确率|相对完整阅读纠正/退化|差值来源 95% 区间（百分点）|",
              "|---|---:|---:|---:|---|"]
    for method, row in table["methods"].items():
        comparison = table["comparisons"].get(method + " minus full")
        changes = f"{comparison['corrected']}/{comparison['degraded']}" if comparison else "—"
        ci = (f"[{100 * comparison['source_bootstrap_95ci'][0]:+.3f}, "
              f"{100 * comparison['source_bootstrap_95ci'][1]:+.3f}]") if comparison else "—"
        lines.append(f"|{method}|{row['correct']}/{row['n']}|{100 * row['accuracy']:.3f}%|{changes}|{ci}|")
    if table["pending"]:
        lines += ["", "尚未形成完整分母的结果："+ "，".join(f"{m} {n}/{table['expected']}" for m, n in table["pending"].items())+"。"]
    return lines

def run(final=False):
    cfg = read(ROOT / "configuration.json")
    cases = rows(OLD / "data/test.jsonl")
    original = old_predictions()
    names = [v["name"] for v in cfg["variants"]]
    all_methods = ["full", "aster", "learned_reference", "learned_self", "direct_reference", "bias"] + names
    medguide_methods = [m for m in all_methods if m != "direct_reference"]
    pair_methods = list(medguide_methods)
    prior = {m: [r for (cid, k), r in original.items() if k == m] for m in METHODS}
    mega, mega_rows = evaluate_table("mega", len(cases), prior, all_methods)
    pilot, pilot_rows = evaluate_table("medguide_pilot", cfg["medguide"]["pilot"], planned=medguide_methods)
    decision_path = ROOT / "confirmation_decision.json"
    decision = read(decision_path) if decision_path.exists() else None
    confirm, confirm_rows = evaluate_table("medguide_confirm", cfg["medguide"]["confirm"], planned=medguide_methods)
    if decision is None:
        confirm["state"] = "not_decided"
    elif not decision["run"]:
        confirm["state"] = "skipped"
        confirm["pending"] = {}
    pairs_data = rows(ROOT / "data/pairs.jsonl")
    test_pairs = [p for p in pairs_data if p["split"] == "test"]
    pairs, pair_rows = evaluate_table("pairs", len(test_pairs), planned=pair_methods)
    pair_counts = {split: dict(Counter(p["kind"] for p in pairs_data if p["split"] == split))
                   for split in ["fit", "dev", "test"]}
    pair_kinds = {
        m: {k: dict(n=len(xs), both_correct=sum(r["correct"] for r in xs),
                    flipped=sum(r["prediction_flipped"] for r in xs),
                    truth_flips=sum(r["label_flipped"] for r in xs))
            for k in ["A", "P", "C", "F", "F_same"] if (xs := [r for r in values if r["kind"] == k])}
        for m, values in pair_rows.items()}
    nodes = fixed_nodes(cases, original)
    changed = changed_diagnostics()
    common = common_errors(pilot_rows)
    training = {}
    for name in names:
        path = ROOT / "models" / name / "complete.json"
        training[name] = dict(state="complete", **read(path)) if path.exists() else dict(state="pending")
    missing = []
    for label, table in [("MEGA", mega), ("MedGUIDE pilot", pilot), ("pairs", pairs), ("MedGUIDE confirmation", confirm)]:
        if table["state"] not in ["complete", "skipped"]:
            missing.append(label)
    if any(r["state"] != "complete" for r in nodes.values()):
        missing.append("fixed gold nodes")
    if changed["state"] != "complete":
        missing.append("changed-node reruns")
    if any(r["state"] != "complete" for r in training.values()):
        missing.append("training")
    state = "complete" if final and not missing else "in_progress"
    pipeline = read(ROOT / "status/pipeline.json") if (ROOT / "status/pipeline.json").exists() else {}
    if pipeline.get("state") == "failed" and not final:
        state = "failed"
    primary = cfg.get("primary_new_method", "paired_ns")
    conclusions = {}
    for dataset, table in [("mega", mega), ("medguide_pilot", pilot), ("medguide_confirm", confirm)]:
        texts = [difference_text(table, primary, "full"), difference_text(table, primary, "bias"),
                 difference_text(table, "paired_n", "online_n")]
        conclusions[dataset] = [t for t in texts if t]
    summary = dict(state=state, final_requested=final, incomplete_stages=missing,
                   primary_new_method=primary, mega=mega, medguide_pilot=pilot,
                   medguide_confirm=confirm, confirmation_decision=decision, pairs=pairs,
                   pair_counts=pair_counts, pair_kinds=pair_kinds, medgemma_common_errors=common,
                   nodes=nodes, changed_nodes=changed, training=training, conclusions=conclusions)
    write(ROOT / "summary.json", summary)
    status_text = ("已完成全部实际启动的阶段。" if state == "complete"
                   else "运行失败；仅报告已保存的完整分母结果。" if state == "failed"
                   else "运行中；仅报告已保存的完整分母结果。")
    if state == "complete" and confirm["state"] == "skipped":
        status_text += "MedGUIDE 扩展评测未触发，未计为已评测。"
    lines = ["# ASTER M3 P0–P3", "", "状态："+status_text,
             f"预先指定的新主方法为 {primary}，其余支持空间和选头配置作为对照；不按测试成绩改选主方法。",
             "P0 的旧检查点统计见 [直接配对分析](p0/REPORT.zh.md)。主实验模型为 Qwen3-8B；MedGemma 复用既有 200 题预测计算共同错误。"]
    for label, texts in conclusions.items():
        if texts:
            lines += ["", label+"："+ " ".join(texts)]
    for title, table in [("MEGA-CDP 自主完整路径", mega), ("MedGUIDE 200 题试测", pilot), ("MedGUIDE 扩展来源留出", confirm)]:
        lines += [""] + report_table(title, table)
    lines += ["", "## 固定金标准路径", "",
              "|方法|已评分/固定节点数|节点正确|与 full 配对节点数|纠正/退化|首错纠正/已评分首错节点|",
              "|---|---:|---:|---:|---:|---:|"]
    for m, r in nodes.items():
        lines.append(f"|{m}|{r['scored']}/{r['expected']}|{r['correct']}|{r['paired_node_denominator']}|"
                     f"{r['corrected_vs_full']}/{r['degraded_vs_full']}|{r['first_error_corrected']}/{r['first_error_scored']}|")
    lines += ["", "该表沿同一金标准路径指定评估节点，金标准不进入模型提示。部分完成时的纠正/退化只针对共同已评分节点；"
              "完整节点分母形成后才计算来源聚类区间。首错纠正仅表示修正原路径的第一个错误节点，不等于整条路径正确。"
              "条件类型按关键词分为数值、否定、复合及其他，详表保存在 fixed_node_summary.json；这些类别不是 A/P/C/F 标注。"]
    lines += ["", "## 6 个纠正与 2 个退化的节点补评", "",
              f"已完成 {changed['scored_method_nodes']}/{changed['expected_method_nodes']} 个方法—节点记录。"
              " gate 和编辑量均为该节点 51 个头的平均值；原结果文件只有路径平均编辑量。",
              "", "|病例/节点|变化|原 full 分数|原参考分数→补跑|参考 gate|参考节点编辑量|自身 gate|自身节点编辑量|",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    def number(value, fmt=".5g"):
        return "待补评" if value is None else format(value, fmt)
    for case in changed["values"]:
        ref = case["reruns"]["learned_reference"] or {}
        own = case["reruns"]["learned_self"] or {}
        old_ref = case["scores"].get("learned_reference") or {}
        old_full = case["scores"].get("full") or {}
        lines.append(f"|{case['case_id']}/{case['gid']}|{case['outcome']}|{number(old_full.get('ell'))}|"
                     f"{number(old_ref.get('ell'))}→{number(ref.get('ell'))}|{number(ref.get('gate'))}|"
                     f"{number(ref.get('magnitude'))}|{number(own.get('gate'))}|{number(own.get('magnitude'))}|")
    lines += ["", f"补跑与原保存分数不同的记录数为 {changed['rerun_score_changes']}；"
              f"最大绝对差为 {number(changed['max_abs_rerun_score_difference'])}。"
              "当前 8 例的原 full 分数均在阈值附近；局部阈值变化的现象与校准一致，最终归因仍须结合预设 bias 和随机头对照。"]
    lines += ["", "## 配对诊断分母", "", "|类型|训练|开发|测试|", "|---|---:|---:|---:|"]
    for kind in ["A", "P", "C", "F", "F_same"]:
        lines.append("|"+kind+"|"+"|".join(str(pair_counts[s].get(kind, 0)) for s in ["fit", "dev", "test"])+"|")
    lines += ["", "各格为“两侧均正确/配对数；预测翻转数”。A/P/F_same 期望保持判断，C/F 期望按标签翻转。",
              "", "|方法|A|P|C|F|F_same|", "|---|---|---|---|---|---|"]
    for method, types in pair_kinds.items():
        cells = [f"{types[k]['both_correct']}/{types[k]['n']}；{types[k]['flipped']}" if k in types else "—"
                 for k in ["A", "P", "C", "F", "F_same"]]
        lines.append("|"+method+"|"+"|".join(cells)+"|")
    if pairs["pending"]:
        lines += ["", "未完成的配对诊断："+"，".join(f"{m} {n}/{pairs['expected']}" for m, n in pairs["pending"].items())+"。"]
    lines += ["", "## 两模型共同错误", "", "|Qwen 方法|共同病例数|两者都错|仅 Qwen 错|仅 MedGemma 错|两者都对|",
              "|---|---:|---:|---:|---:|---:|"]
    for method, value in common.items():
        lines.append(f"|{method}|{value['n']}|{value['both_wrong']}|{value['qwen_only_wrong']}|"
                     f"{value['medgemma_only_wrong']}|{value['both_correct']}|")
    lines += ["", "## 方法、来源与统计", "",
              "随机头采用 3 个固定种子，匹配 M1 的层分布、51 个头及 31,771 个参数，允许抽中 M1 原头；"
              "全部候选头和全部头是容量上限对照。具体头集合、重合关系及参数量见 p2/head_controls.json。"
              "M1 的必要性通过 learned_reference 与三个同训练目标随机头版本比较，不能从 paired_ns 与随机头的差异单独归因。",
              "paired_n、paired_s、paired_ns 均以当前头表示作为参考，只需一次末位置前向。online_n 是相同 N 空间、"
              "相同单次前向且不使用配对损失的对照；paired_n 对 online_n 才直接比较配对训练的作用。"
              "三种支持空间的参数量不同，空间与容量因素一并报告。",
              "A/P/F_same 使用两侧 CE 和 JS 保持项，C/F 使用两侧 CE 与按标签定向的 margin 损失，各配对类型分别平均；"
              "原病例监督、正确节点 KL 和编辑量正则保留。历史构造配对与原版病例结果分列。",
              "MedGUIDE 保留原选项提示，辅助参考只使用 profile，不读取 path、answer_text 或答案。"
              "原提示不含完整指南树，结果解释为原生选择题迁移。单一 bias 为 A 选项的加性偏置，"
              "仅用另选的校准病例拟合；MEGA bias 仅用原训练病例节点。",
              f"MedGUIDE 试测 {cfg['medguide']['pilot']} 题覆盖 53 个页面、17 个疾病家族，属于已观察的探索样本。"
              f"扩展数据 {cfg['medguide']['confirm']} 题与校准集按疾病家族互斥，并排除原试测病例；"
              "扩展家族已有试测暴露，不能称为全流程未见来源。若未触发扩展，数据规模不作为已完成评测分母。",
              "MedGUIDE 区间按各评测集实际出现的疾病家族聚类：试测为 17 个家族，扩展集为预留家族，"
              "并非每个表均有 17 个聚类。MEGA 按指南来源聚类。配对 bootstrap 为 20,000 次，"
              "来源置换整组交换方法；病例级 McNemar 作为补充。完整比较和统计来源数见 summary.json。"
              "多个消融的区间为逐项区间，不能把测试择优解释为确认性证据。",
              "Qwen 主评分保留旧 M3 的 BF16 标签投影；既有 MedGemma 试测使用 FP32 标签投影。"
              "共同错误按病例 ID 配对，并分别记录设置。训练用时、峰值显存及选轮记录见各 models/<name>/complete.json。"]
    (ROOT / "REPORT.zh.md").write_text("\n".join(lines)+"\n")
    (ROOT / "README.md").write_text(
        "# M3 后续实验\n\n"+status_text+"完整结果见 [REPORT.zh.md](REPORT.zh.md)，"
        "机器可读统计见 [summary.json](summary.json)，旧模型直接比较见 [P0](p0/REPORT.zh.md)。\n")
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    run(args.final)
