#!/usr/bin/env python3
"""Fixed 24-case Qwen interface smoke and decision-impact comparison.

This imports the unchanged PR11 engine. It never fits intervention parameters.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import json
import os
import sys
import tarfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PR11 = REPO / "experiments/causal_attention_20261019"
DATA = REPO / "experiments/care_reliability_20261019/data/mechanism_cases.jsonl"
sys.path.insert(0, str(PR11))
from interface import normalize_structure, build_interface, signed_action_targets

FIXED_CONFIGS = [
    dict(name="baseline", layers=[], strength=1.0, variant="C", query_window=1,
         method="direct", with_structure=False),
    dict(name="structure_only", layers=[], strength=1.0, variant="C", query_window=1,
         method="direct", with_structure=True),
    dict(name="attention_only", layers=list(range(18, 24)), strength=1.0, variant="C",
         query_window=1, method="direct", with_structure=False),
    dict(name="combined", layers=list(range(18, 24)), strength=1.0, variant="C",
         query_window=1, method="direct", with_structure=True),
]


def load_cases():
    return [json.loads(s) for s in DATA.read_text().splitlines()
            if json.loads(s)["split"] == "confirmation"]


def freeze():
    dest = HERE / "qwen_selection.json"
    if dest.exists():
        return json.loads(dest.read_text())
    cases = load_cases()
    selection = []
    modes = ("missing", "historical_only", "relative_only", "contradictory_current")
    for j, family in enumerate(sorted({c["family_id"] for c in cases})):
        rows = [c for c in cases if c["family_id"] == family]
        pair_id = sorted({c["pair_id"] for c in rows
                          if c.get("pair_eligible") and c["variant"] == "canonical"})[0]
        chosen = sorted([c for c in rows if c["pair_id"] == pair_id],
                        key=lambda c: c["reference_state"], reverse=True)
        chosen += [next(c for c in rows if c.get("uncertainty_mode") == modes[j % 4])]
        assert len(chosen) == 3 and {c["reference_state"] for c in chosen} == {"T", "F", "U"}
        selection += [{"case_id": c["case_id"], "family_id": family,
                       "reference_state": c["reference_state"], "pair_id": c["pair_id"],
                       "uncertainty_mode": c.get("uncertainty_mode")} for c in chosen]
    result = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cases": selection, "n": len(selection),
        "selection_rule": "Before V8 results: each of 8 confirmation families contributes the first canonical T/F pair and one U, cycling missing/historical_only/relative_only/contradictory_current.",
        "purpose": "Historical regression, interface integration and decision impact only; not independent confirmation or intervention parameter search.",
        "configs": FIXED_CONFIGS, "source_commit": "f0c3c28a30096bdda37c021f6db9de85754cb8e7",
        "source_selection": "experiments/causal_attention_20261019/selection_group4.json",
        "qwen_input": "Unchanged PR11 full natural patient and original pre-parsed guidance; no gold labels or structured patient facts. V8 contributes only applicability states.",
    }
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def chosen_cases():
    manifest = freeze()
    lookup = {c["case_id"]: c for c in load_cases()}
    return [lookup[r["case_id"]] for r in manifest["cases"]]


def check_interface():
    from engine import full_context, MODEL
    from transformers import AutoTokenizer
    case = chosen_cases()[0]
    raw = {"rules": [
        {"rule_id": rule["rule_id"], "state": state,
         "evidence_quote": case["patient"].split(". ")[0] if i == 0 else "not in patient"}
        for i, (rule, state) in enumerate(zip(case["rules"], "TFU"))]}
    structure = normalize_structure(case, raw)
    prompt, spans = full_context(case, structure)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    offsets = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    result = build_interface(case, prompt, spans, structure, offsets)
    assert len(result) == len(case["rules"])
    assert [r["state"] for r in result] == list("TFU")
    assert all(r["action_token_indices"] for r in result)
    assert result[1]["evidence_location_status"] == "not_exact_patient_substring"
    assert "optional_evidence_char_span" in result[0]
    failed = normalize_structure(case, {"status": "parse_failure", "rules": []})
    assert all(r["state"] is None for r in failed["rules"])
    assert signed_action_targets(build_interface(case, prompt, spans, failed, offsets)) == []
    out = {"status": "passed", "case_id": case["case_id"],
           "checks": ["all original rule/action IDs retained",
                      "full-prompt character-to-token offsets",
                      "unlocated evidence does not gate T/F/U",
                      "missing parsed states remain null, never U"],
           "full_prompt_tokens": len(offsets), "S": result}
    (HERE / "qwen_interface_check.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def summarize(cases, rows, baseline):
    by_id = {r["case_id"]: r for r in rows}
    base = {r["case_id"]: r for r in baseline}
    good = {c["case_id"]: by_id[c["case_id"]].get("prediction") == c["gold_answer"]
            for c in cases}
    pairs = defaultdict(list)
    for c in cases:
        if c.get("pair_eligible"):
            pairs[c["pair_id"]].append(c["case_id"])
    return {
        "cases": len(cases), "correct": sum(good.values()),
        "accuracy": sum(good.values()) / len(cases),
        "technical_failures": sum(r["status"] != "completed" for r in rows),
        "module1_parse_failures": sum(r.get("module1_parse_failure", False) for r in rows),
        "attention_changed_cases": sum(any(
            d["attention_absolute_change_mean"] > 1e-9 for d in r.get("attention_diagnostics", []))
            for r in rows),
        "pair_count": sum(len(p) == 2 for p in pairs.values()),
        "pair_correct": sum(len(p) == 2 and all(good[x] for x in p) for p in pairs.values()),
        "by_target_state": {s: {"n": sum(c["reference_state"] == s for c in cases),
                                  "correct": sum(good[c["case_id"]] for c in cases if c["reference_state"] == s)}
                            for s in "TFU"},
        "by_family": {f: {"n": sum(c["family_id"] == f for c in cases),
                          "correct": sum(good[c["case_id"]] for c in cases if c["family_id"] == f)}
                      for f in sorted({c["family_id"] for c in cases})},
        "prediction_changes_vs_baseline": sum(by_id[c["case_id"]].get("prediction") != base[c["case_id"]].get("prediction") for c in cases),
        "corrected_vs_baseline": sum(good[c["case_id"]] and base[c["case_id"]].get("prediction") != c["gold_answer"] for c in cases),
        "damaged_vs_baseline": sum(not good[c["case_id"]] and base[c["case_id"]].get("prediction") == c["gold_answer"] for c in cases),
        "mean_inference_seconds": sum(r.get("elapsed_seconds", 0) for r in rows) / len(rows),
    }


def historical(cases, out):
    wanted = {c["case_id"] for c in cases}
    paths = {
        "baseline": "results/group1_confirmation_main/baseline.jsonl",
        "structure_only": "results/group4_fixed/structure_only.jsonl",
        "attention_only": "results/group4_fixed/attention_only.jsonl",
        "combined": "results/group4_fixed/combined.jsonl",
    }
    selected = {}
    with tarfile.open(PR11 / "artifacts/raw_run_records.tar.gz", "r:gz") as archive:
        for arm, path in paths.items():
            f = archive.extractfile(path)
            selected[arm] = [r for r in map(json.loads, f) if r["case_id"] in wanted]
            assert len(selected[arm]) == len(cases)
            (out / ("historical_" + arm + ".jsonl")).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected[arm]))
    return selected, {arm: summarize(cases, rows, selected["baseline"]) for arm, rows in selected.items()}


def run(predictions, variant):
    import torch
    from engine import QwenEngine, full_context

    class SemanticQwenEngine(QwenEngine):
        def encode_case(self, case, config, structure=None):
            if structure is None:
                raise ValueError("V8 structure is required; reference structure must not be used")
            prompt, spans = full_context(case, structure if config.get("with_structure") else None)
            encoded = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                                     add_special_tokens=False)
            offsets = encoded.pop("offset_mapping")[0].tolist()
            self.last_interface = build_interface(case, prompt, spans, structure, offsets)
            targets = signed_action_targets(self.last_interface)
            signs = torch.zeros(len(offsets), device=self.device, dtype=torch.float32)
            for target in targets:
                signs[target["token_indices"]] = target["sign"]
            return prompt, {k: v.to(self.device) for k, v in encoded.items()}, targets, signs

    cases = chosen_cases()
    records = {r["case_id"]: r for r in map(json.loads, Path(predictions).read_text().splitlines())}
    missing = [c["case_id"] for c in cases if c["case_id"] not in records]
    if missing:
        raise ValueError("Selected predictions not ready: " + str(missing))
    out = HERE / "results" / ("qwen_" + variant)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = SemanticQwenEngine()
    history, history_summary = historical(cases, out)
    (out / "provenance.json").write_text(json.dumps(
        model.provenance | {"argv": sys.argv, "variant": variant,
                            "predictions": str(predictions), "configs": FIXED_CONFIGS,
                            "selection": "qwen_selection.json",
                            "scope": "interface and decision impact; no layer/head/strength search",
                            "clinical": False}, indent=2) + "\n")
    structures = {c["case_id"]: normalize_structure(c, records[c["case_id"]]) for c in cases}
    all_rows = {}
    for cfg in FIXED_CONFIGS:
        rows = []
        with (out / (cfg["name"] + ".jsonl")).open("x") as stream:
            for case in cases:
                structure = structures[case["case_id"]]
                try:
                    row = model.score(case, cfg, structure)
                    row["S"] = model.last_interface
                except Exception as exc:
                    row = {"status": "technical_failure", "is_model_prediction": False,
                           "error": repr(exc), "traceback": traceback.format_exc()}
                row.update(case_id=case["case_id"], family_id=case["family_id"],
                           reference_state=case["reference_state"], gold_answer=case["gold_answer"],
                           arm=cfg["name"], variant=variant, config=cfg,
                           module1_parse_failure=structure["parse_failure"])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
        all_rows[cfg["name"]] = rows
        print(json.dumps({"arm": cfg["name"], "cases": len(rows),
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    # One no-op check: an all-U interface creates no targets and must exactly
    # recover the same baseline input and logits, without touching medical labels.
    c = cases[0]
    all_u = {"status": "ok", "rules": [
        {"rule_id": r["rule_id"], "state": "U", "evidence_quote": ""}
        for r in c["rules"]]}
    noop = model.score(c, FIXED_CONFIGS[2], all_u)
    base = all_rows["baseline"][0]
    difference = max(abs(a - b) for a, b in zip(base["candidate_logits"], noop["candidate_logits"]))
    no_op_check = {"case_id": c["case_id"], "fixture": "synthetic all-U interface",
                   "same_prompt": base["prompt"] == noop["prompt"],
                   "same_prediction": base["prediction"] == noop["prediction"],
                   "max_candidate_logit_difference": difference,
                   "target_token_count": noop["target_token_count"], "raw": noop}
    no_op_check["passed"] = no_op_check["same_prompt"] and difference == 0 and noop["target_token_count"] == 0
    (out / "noop_check.json").write_text(json.dumps(no_op_check, indent=2) + "\n")
    summary = {
        "v8_variant": variant, "cases": len(cases),
        "v8": {arm: summarize(cases, rows, all_rows["baseline"]) for arm, rows in all_rows.items()},
        "pr11_fixed_historical_same_cases": history_summary,
        "same_cases_comparison": {arm: {
            "prediction_changes": sum(x.get("prediction") != y.get("prediction") for x, y in
                                       (({r["case_id"]: r for r in all_rows[arm]}[c["case_id"]],
                                         {r["case_id"]: r for r in history[arm]}[c["case_id"]]) for c in cases)),
            "correct_delta": sum(r.get("prediction") == r["gold_answer"] for r in all_rows[arm]) - history_summary[arm]["correct"]}
            for arm in all_rows},
        "noop_check_passed": no_op_check["passed"],
        "limitations": [
            "These are 24 selected historical regression cases, not independent confirmation.",
            "Qwen sees the same original pre-parsed guidance in every arm; the module-one A0/A1 input remains natural-language rules.",
            "Structure-only and combined effects do not establish an attention mechanism.",
            "Every action and rule is retained; unmatched evidence never blocks T/F/U.",
        ],
    }
    for arm in all_rows:
        current = {r["case_id"]: r for r in all_rows[arm]}
        prior = {r["case_id"]: r for r in history[arm]}
        comparison = summary["same_cases_comparison"][arm]
        comparison["corrected_vs_pr11_fixed"] = sum(
            current[c["case_id"]].get("prediction") == c["gold_answer"] and
            prior[c["case_id"]].get("prediction") != c["gold_answer"] for c in cases)
        comparison["damaged_vs_pr11_fixed"] = sum(
            current[c["case_id"]].get("prediction") != c["gold_answer"] and
            prior[c["case_id"]].get("prediction") == c["gold_answer"] for c in cases)
        comparison["identical_full_prompts"] = sum(
            current[c["case_id"]].get("prompt") == prior[c["case_id"]].get("prompt") for c in cases)
    structure_by_id = {r["case_id"]: r for r in all_rows["structure_only"]}
    summary["combined_vs_structure_only"] = {
        "prediction_changes": sum(r.get("prediction") != structure_by_id[r["case_id"]].get("prediction")
                                  for r in all_rows["combined"]),
        "correct_delta": summary["v8"]["combined"]["correct"] - summary["v8"]["structure_only"]["correct"],
    }
    summary["baseline_reproduction"] = {
        "same_predictions": summary["same_cases_comparison"]["baseline"]["prediction_changes"] == 0,
        "identical_full_prompts": summary["same_cases_comparison"]["baseline"]["identical_full_prompts"],
        "max_candidate_logit_difference": max(
            abs(x - y) for c in cases
            for x, y in zip(
                next(r for r in all_rows["baseline"] if r["case_id"] == c["case_id"])["candidate_logits"],
                next(r for r in history["baseline"] if r["case_id"] == c["case_id"])["candidate_logits"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "completion.json").write_text(json.dumps({
        "status": "completed", "forward_calls": model.forward_calls,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    report = [
        "# V8 Qwen 小规模接入结果", "",
        f"采用 {variant}，固定历史 confirmation 中 24 例：8 个家族各一对 T/F 和一个 U；病例清单在查看 V8 结果前冻结。",
        "沿用 PR11 固定 C 配置：18–23 层，强度 1，最后一个 query。没有重新选择层、头或强度。",
        "Qwen 完整提示保持 PR11 的预解析规则内容；各臂只改变预测状态提示或原定动作 token 干预。",
        "", "| 方案 | V8 正确/24 | PR11 修复版同例正确/24 |", "|---|---:|---:|",
    ]
    for arm, label in [("baseline", "无结构无干预"), ("structure_only", "仅结构提示"),
                       ("attention_only", "仅干预"), ("combined", "结构提示+干预")]:
        report.append(f"| {label} | {summary['v8'][arm]['correct']} | {history_summary[arm]['correct']} |")
    evidence_rows = [s for r in all_rows["baseline"] for s in r.get("S", [])]
    evidence_found = sum("optional_evidence_char_span" in s for s in evidence_rows)
    reproduction = summary["baseline_reproduction"]
    report += ["",
               f"V8 共 {len(evidence_rows)} 条规则保留原始动作，证据原文可定位 {evidence_found} 条；证据定位只作诊断。",
               f"零操作检查通过：{no_op_check['passed']}；全 U 测试未创建动作目标，候选 logit 最大差值 {difference}。",
               f"与旧 baseline 相同的完整提示为 {reproduction['identical_full_prompts']}/24，预测全部相同：{reproduction['same_predictions']}，候选 logit 最大差值 {reproduction['max_candidate_logit_difference']}。",
               f"结构提示+干预相对于仅结构提示，正确数变化为 {summary['v8']['combined']['correct']-summary['v8']['structure_only']['correct']}；保留所有逐例变化与干预诊断。",
               f"仅干预与组合各有 {summary['v8']['attention_only']['attention_changed_cases']} 和 {summary['v8']['combined']['attention_changed_cases']} 例实际改变注意力分布；仅干预未改变最终选项，组合相对仅结构提示改变选项 {summary['combined_vs_structure_only']['prediction_changes']} 例。",
               "本实验只检查接口和决策影响；历史数据已经用于开发，合成数据不是临床验证，结构提示带来的变化不能归因于注意力干预。", ""]
    (out / "QWEN_REPORT.zh.md").write_text("\n".join(report))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--predictions")
    p.add_argument("--variant", choices=["A0", "A1"])
    args = p.parse_args()
    if args.prepare:
        print(json.dumps(freeze(), ensure_ascii=False))
    if args.check:
        print(json.dumps(check_interface(), ensure_ascii=False))
    if args.predictions:
        if not args.variant:
            p.error("--variant is required with --predictions")
        run(args.predictions, args.variant)

if __name__ == "__main__":
    main()
