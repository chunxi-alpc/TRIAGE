"""Analyze ACTUAL saved Llama inference, retaining the complete case denominator.

Results are descriptive and exploratory; B/C have constructed action labels.
All outputs stay local. This script never synthesizes model predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import random
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def mean(items):
    return sum(items) / len(items) if items else None


def metric(records):
    n = len(records)
    completed = [x for x in records if x["result"].get("status") == "completed"]
    correct = sum(bool(x["result"].get("correct")) for x in completed)
    return {
        "n": n, "completed": len(completed), "failed_or_missing": n - len(completed),
        "correct": correct, "accuracy_full_denominator": correct / n if n else None,
    }


def ci_by_family(rows, left, right, seed=261022, draws=2000):
    families = defaultdict(list)
    for r in rows:
        families[r["family"]].append(r)
    groups = list(families)
    if not groups:
        return None
    rnd = random.Random(seed)
    changes = []
    for _ in range(draws):
        sample = [r for _ in groups for r in families[rnd.choice(groups)]]
        if sample:
            changes.append(sum(int(r[left].get("correct", False)) - int(r[right].get("correct", False))
                               for r in sample) / len(sample))
    changes.sort()
    return [changes[int(.025 * (len(changes) - 1))], changes[int(.975 * (len(changes) - 1))]]


def summarize_attention(items, criterion):
    selected = []
    for x in items:
        res = x["result"]
        if res.get("status") != "completed" or res.get("correct") != criterion:
            continue
        layers = res.get("attention", [])
        if not layers:
            continue
        app = mean([layer["app_rule_mass_before"] for layer in layers])
        sib = mean([layer["sibling_rule_mass_before"] for layer in layers])
        # Per-token normalization is only a descriptive concentration proxy,
        # not a causal estimate. Report raw mass too.
        app_norm = mean([
            layer["app_rule_mass_before"] / layer["app_rule_tokens"]
            for layer in layers if layer["app_rule_tokens"] > 0
        ])
        sib_norm = mean([
            layer["sibling_rule_mass_before"] / layer["sibling_rule_tokens"]
            for layer in layers if layer["sibling_rule_tokens"] > 0
        ])
        selected.append((app, sib, app_norm, sib_norm))
    return {
        "n": len(selected),
        "app_mass": mean([v[0] for v in selected]),
        "sibling_mass": mean([v[1] for v in selected]),
        "app_mass_per_token": mean([v[2] for v in selected if v[2] is not None]),
        "sibling_mass_per_token": mean([v[3] for v in selected if v[3] is not None]),
    }


def error_category(case, variant, result):
    if result.get("status") != "completed":
        return "failed_or_missing"
    if result.get("correct"):
        return "correct"
    pred = result.get("prediction")
    closedbook = case.get("closedbook", {}).get("prediction")
    edited_sibling = case["replacement_answer"] if variant == "C" else case["sibling_answer"]
    matches_memory = pred is not None and pred == closedbook
    matches_sibling = pred == edited_sibling
    if matches_memory and matches_sibling:
        return "overlap_closedbook_and_sibling"
    if matches_sibling:
        return "sibling_action"
    if matches_memory:
        return "closedbook_choice"
    return "other"


def svg_strength(path, points):
    """Dependency-free SVG for the fixed strength scan."""
    if not points:
        return
    width, height = 770, 440
    x0, x1, y0, y1 = 70, 730, 42, 374
    values = sorted(set(x for x, _, _ in points))
    xs = {v: x0 + (x1 - x0) * i / max(len(values) - 1, 1) for i, v in enumerate(values)}
    variants = ("A", "B", "C")
    colors = {"A": "#475569", "B": "#0284c7", "C": "#b45309"}
    def y(v):
        return y1 - v * (y1 - y0)
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<text x="70" y="23" font-size="16">Signed attention: MCQA accuracy by strength</text>']
    for tick in (0, .25, .5, .75, 1):
        yy = y(tick)
        content.extend([
            f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e2e8f0"/>',
            f'<text x="25" y="{yy+5:.1f}" font-size="12">{tick:.2f}</text>',
        ])
    for key in variants:
        values_for_key = sorted((s, score) for s, variant, score in points if variant == key and score is not None)
        if values_for_key:
            coords = " ".join(f"{xs[s]:.1f},{y(score):.1f}" for s, score in values_for_key)
            content.append(f'<polyline points="{coords}" stroke="{colors[key]}" fill="none" stroke-width="2.5"/>')
            for s, score in values_for_key:
                content.append(f'<circle cx="{xs[s]:.1f}" cy="{y(score):.1f}" r="4" fill="{colors[key]}"/>')
    for s in values:
        content.append(f'<text x="{xs[s]:.1f}" y="398" text-anchor="middle" font-size="12">{s:g}</text>')
    content.append('<text x="360" y="428" font-size="12">Attention score bias strength</text>')
    for i, key in enumerate(variants):
        content.append(f'<text x="{530 + 65*i}" y="25" fill="{colors[key]}" font-size="13">Variant {key}</text>')
    content.append("</svg>")
    path.write_text("\n".join(content), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as sink:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="Private prepared cases.jsonl")
    ap.add_argument("--predictions", type=Path, required=True, help="Private runner output")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {x["case_id"]: x for x in read_jsonl(args.cases)}
    actual = {x["case_id"]: x for x in read_jsonl(args.predictions)}
    if not expected:
        raise ValueError("No prepared cases to evaluate")
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate prediction IDs")
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise ValueError(f"Predictions contain unknown case IDs: {sorted(unexpected)[:5]}")
    metrics, errors, attention = [], [], []
    by_case_variant = defaultdict(list)
    strength_points = []
    for variant in "ABC":
        arm_names = sorted({
            arm for case in actual.values()
            for arm in case.get("variants", {}).get(variant, {})
        })
        for arm in arm_names:
            paired = []
            for case_id, original in expected.items():
                inference = actual.get(case_id, {})
                arms = inference.get("variants", {}).get(variant, {})
                result = arms.get(arm, {"status": "not_run"})
                base = arms.get("baseline", {"status": "not_run"})
                sample = {
                    "case_id": case_id, "family": original["family"],
                    "result": result, "baseline": base,
                }
                paired.append(sample)
                if arm == "baseline":
                    cb = inference.get("closedbook", {})
                    category = error_category({
                        **original, "closedbook": cb
                    }, variant, result)
                    errors.append({
                        "case_id": case_id, "variant": variant, "category": category,
                        "gold": original["versions"][variant]["gold"],
                        "prediction": result.get("prediction"),
                        "closedbook": cb.get("prediction"),
                    })
            info = metric(paired)
            info.update({"variant": variant, "arm": arm})
            if arm != "baseline":
                info["corrected_vs_baseline"] = sum(
                    bool(x["result"].get("correct")) and not bool(x["baseline"].get("correct"))
                    for x in paired)
                info["damaged_vs_baseline"] = sum(
                    not bool(x["result"].get("correct")) and bool(x["baseline"].get("correct"))
                    for x in paired)
                info["delta_accuracy_vs_baseline"] = (
                    info["corrected_vs_baseline"] - info["damaged_vs_baseline"]) / len(paired)
                boot_rows = [
                    {"family": x["family"], "new": x["result"], "base": x["baseline"]}
                    for x in paired
                ]
                info["family_bootstrap_delta_ci"] = ci_by_family(boot_rows, "new", "base")
                if arm != "manual_zero":
                    zero = [actual.get(case_id, {}).get("variants", {}).get(variant, {}).get("manual_zero", {"status": "not_run"}) for case_id in expected]
                    info["delta_accuracy_vs_manual_zero"] = (
                        sum(bool(x["result"].get("correct")) for x in paired) - sum(bool(x.get("correct")) for x in zero)
                    ) / len(paired)
            metrics.append(info)
            for correctness in (True, False):
                attention.append({
                    "variant": variant, "arm": arm, "correct": correctness,
                    **summarize_attention(paired, correctness),
                })
            if arm.startswith("signed@"):
                strength_points.append((float(arm.split("@", 1)[1]), variant, info["accuracy_full_denominator"]))
            by_case_variant[variant].append(info)
    m = {variant: {r["arm"]: r for r in by_case_variant[variant]} for variant in "ABC"}
    primary = {}
    for variant in "ABC":
        arms = m[variant]
        for name in ("baseline", "whole", "changed", "positive", "unsigned", "explicit"):
            primary[f"{variant}_{name}_accuracy"] = arms.get(name, {}).get("accuracy_full_denominator")
    if "baseline" in m["A"] and "baseline" in m["C"]:
        primary["baseline_A_minus_C"] = (
            m["A"]["baseline"]["accuracy_full_denominator"] - m["C"]["baseline"]["accuracy_full_denominator"])
    categories = Counter(r["category"] for r in errors)
    summary = {
        "status": "completed" if metrics and len(actual) == len(expected) and all(
            r["failed_or_missing"] == 0 for r in metrics) else "incomplete",
        "prepared_cases": len(expected), "recorded_cases": len(actual),
        "missing_case_ids": sorted(set(expected) - set(actual)),
        "experimental_labels": "B/C are synthetic changes to published-path actions, not NCCN gold",
        "original_sibling_path_identity": "oracle from source gold path for experimental routing; not in prompt as an explicit label",
        "methods": "whole/changed/positive/unsigned/signed/reversed are direct score biases, NOT official SEKA",
        "primary": primary,
        "wrong_answer_category_counts_baseline": dict(categories),
        "metric_records": metrics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = []
    for r in metrics:
        flat.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    write_csv(args.output / "metrics.csv", flat)
    write_csv(args.output / "errors.csv", errors)
    write_csv(args.output / "attention.csv", attention)
    svg_strength(args.output / "strength_curve.svg", strength_points)
    brief = (
        f"# ASTER MedGUIDE 预实验结果\n\n"
        f"状态：{summary['status']}。准备病例 {len(expected)}，已有预测 {len(actual)}。\n\n"
        "本任务的 B/C 正确动作是实验者构造的反事实标签，不能被表述为原始医疗指南的建议。"
        "这里的闭卷同选项预测只是操作性闭卷偏好，不直接证明参数记忆。\n\n"
        "已写出 metrics.csv、errors.csv、attention.csv、strength_curve.svg 和 summary.json。"
        "注意力统计是描述性指标；是否改善决策应以同病例正确率和纠正／损害比较为准。\n"
    )
    (args.output / "REPORT.zh.md").write_text(brief, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "prepared": len(expected),
                      "recorded": len(actual), "metric_rows": len(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
