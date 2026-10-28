#!/usr/bin/env python3
"""Analyze the frozen rule-reading / answer-readout follow-up.

No fitting, case selection, or parameter search occurs here. The input must
contain all 24 previously fixed prompts under all eight diagnostic conditions.
Scientific outputs are written only after completeness and numeric validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS = (
    "baseline", "projected4", "projected32", "projected_minus32",
    "score_plus8", "score_minus8", "note_gold", "note_sibling",
)
LABELS = {
    "baseline": "Baseline", "projected4": "Key +4", "projected32": "Key +32",
    "projected_minus32": "Key −32", "score_plus8": "Score +8",
    "score_minus8": "Score −8", "note_gold": "Gold note",
    "note_sibling": "Sibling note",
}
POLARITY = {
    "baseline": 0, "projected4": 1, "projected32": 1,
    "projected_minus32": -1, "score_plus8": 1, "score_minus8": -1,
    "note_gold": 1, "note_sibling": -1,
}
COLORS = {
    "baseline": "#64748B", "projected4": "#4F83B6", "projected32": "#008A91",
    "projected_minus32": "#008A91", "score_plus8": "#DE8B33",
    "score_minus8": "#DE8B33", "note_gold": "#AA6893", "note_sibling": "#AA6893",
}
CONTRASTS = (
    ("key_sign", "projected32", "projected_minus32"),
    ("score_sign", "score_plus8", "score_minus8"),
    ("note_content", "note_gold", "note_sibling"),
    ("key_dose", "projected32", "projected4"),
)
FLOOR = 1e-300


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def load_rows(path):
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def prompt_id(row):
    return row["case_id"], row["variant"]


def raw_margin(row):
    return float(row["logits_A"] - row["logits_B"])


def numeric(value, label, probability=False):
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid finite number: {label}={value!r}")
    if probability and not (-1e-7 <= value <= 1 + 1e-6):
        raise ValueError(f"Probability/mass outside [0,1]: {label}={value}")


def validate(rows, status):
    if status.get("status") != "complete":
        raise ValueError("Refuse to analyze an incomplete run")
    if len(rows) != 192:
        raise ValueError(f"Expected 24 × 8 = 192 records, found {len(rows)}")
    index = {}
    head_set = None
    for row in rows:
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"Unregistered condition {condition}")
        key = prompt_id(row) + (condition,)
        if key in index:
            raise ValueError(f"Duplicate input condition: {key}")
        index[key] = row
        if row["gold_rule"] not in ("A", "B") or row["prediction"] not in ("A", "B"):
            raise ValueError("Constrained labels must be A or B")
        if sorted(row["rule_order"]) != ["A", "B"] or row["last_rule"] != row["rule_order"][-1]:
            raise ValueError("Rule-order/last-rule metadata mismatch")
        if bool(row["correct"]) != (row["prediction"] == row["gold_rule"]):
            raise ValueError("Incorrect correctness flag")
        for name in ("logits_A", "logits_B", "gold_margin"):
            numeric(row[name], name)
        for name in ("mass_A", "mass_B", "candidate_probability_mass",
                     "candidate_probability_A", "candidate_probability_B",
                     "conditional_probability_A", "conditional_probability_B"):
            numeric(row[name], name, probability=True)
        if row["mass_A"] + row["mass_B"] > 1 + 1e-6:
            raise ValueError("Disjoint target masses exceed one")
        margin = raw_margin(row) * (1 if row["gold_rule"] == "A" else -1)
        if abs(margin - row["gold_margin"]) > 1e-5:
            raise ValueError("Gold margin does not match full-vocabulary candidate logits")
        expected = "A" if row["logits_A"] >= row["logits_B"] else "B"
        if row["prediction"] != expected:
            raise ValueError("Prediction does not match the constrained candidate scores")
        coverage = row["candidate_probability_A"] + row["candidate_probability_B"]
        if abs(coverage - row["candidate_probability_mass"]) > 2e-6:
            raise ValueError("Candidate probability coverage mismatch")
        if abs(row["conditional_probability_A"] + row["conditional_probability_B"] - 1) > 2e-6:
            raise ValueError("Conditional candidate probabilities must sum to one")
        actual_heads = {(int(h["layer"]), int(h["kv_head"])) for h in row["per_head_mass"]}
        if len(actual_heads) != 4 or len(row["per_head_mass"]) != 4:
            raise ValueError("Expected four unique frozen KV heads")
        if head_set is None:
            head_set = actual_heads
        elif actual_heads != head_set:
            raise ValueError("The selected head set changed between conditions")
        for h in row["per_head_mass"]:
            for label in ("mass_A", "mass_B"):
                numeric(h[label], "per_head_" + label, probability=True)
        for label in ("mass_A", "mass_B"):
            if abs(mean(row["per_head_mass"], label) - row[label]) > 2e-6:
                raise ValueError("Aggregate target mass is not the four-head mean")
        if not row["top_tokens"]:
            raise ValueError("Missing full-vocabulary top-token diagnostic")
    baselines = [r for r in rows if r["condition"] == "baseline"]
    prompts = {prompt_id(r) for r in baselines}
    if len(prompts) != 24 or len({r["case_id"] for r in baselines}) != 12:
        raise ValueError("Expected 24 unique prompts and 12 paired configurations")
    groups = sorted({r["group_id"] for r in baselines})
    if len(groups) != 3:
        raise ValueError("Expected the three frozen confirmation guidelines")
    stable = ("triplet_id", "group_id", "gold_rule", "patient_state", "action_permutation",
              "rule_order_index", "rule_order", "last_rule")
    for p in prompts:
        base = index[p + ("baseline",)]
        for c in CONDITIONS:
            if p + (c,) not in index:
                raise ValueError(f"Missing registered condition {p}, {c}")
            row = index[p + (c,)]
            if any(row[k] != base[k] for k in stable):
                raise ValueError(f"Patient/configuration identity changed for {p}, {c}")
            if c not in ("note_gold", "note_sibling") and row["prompt_sha256"] != base["prompt_sha256"]:
                raise ValueError("An internal-edit condition changed the prompt text")
    for case in {r["case_id"] for r in baselines}:
        pair = [r for r in baselines if r["case_id"] == case]
        if len(pair) != 2 or {r["gold_rule"] for r in pair} != {"A", "B"}:
            raise ValueError("Each paired configuration must contain both opposite gold rules")
    return index, baselines, groups, sorted(head_set)


def baseline_analysis(baselines, groups):
    pairs = defaultdict(dict)
    for row in baselines:
        pairs[row["case_id"]][row["gold_rule"]] = row
    paired = []
    for case, pair in sorted(pairs.items()):
        a, b = pair["A"], pair["B"]
        ma, mb = raw_margin(a), raw_margin(b)
        paired.append({"case_id": case, "group_id": a["group_id"],
                       "action_permutation": a["action_permutation"],
                       "rule_order_index": a["rule_order_index"], "rule_order": a["rule_order"],
                       "patient_state_gold_A": a["patient_state"], "patient_state_gold_B": b["patient_state"],
                       "prediction_gold_A": a["prediction"], "prediction_gold_B": b["prediction"],
                       "margin_A_minus_B_under_gold_A": ma, "margin_A_minus_B_under_gold_B": mb,
                       "patient_relative_signal": (ma - mb) / 2, "common_label_bias": (ma + mb) / 2,
                       "absolute_patient_margin_change": abs(ma - mb),
                       "same_prediction": a["prediction"] == b["prediction"],
                       "both_correct": bool(a["correct"] and b["correct"]),
                       "both_wrong": bool(not a["correct"] and not b["correct"])})
    effects = []
    for factor in ("patient", "action_permutation", "rule_order_index"):
        matches = defaultdict(dict)
        for row in baselines:
            if factor == "patient":
                key, value = (row["group_id"], row["action_permutation"], row["rule_order_index"]), row["gold_rule"]
            else:
                other = "rule_order_index" if factor == "action_permutation" else "action_permutation"
                key, value = (row["group_id"], row[other], row["gold_rule"]), row[factor]
            matches[key][value] = row
        for match, values in sorted(matches.items()):
            keys = ("A", "B") if factor == "patient" else (0, 1)
            if set(values) != set(keys):
                raise ValueError(f"Incomplete baseline factor comparison: {factor}, {match}")
            a, b = (values[k] for k in keys)
            dm = raw_margin(b) - raw_margin(a)
            effects.append({"factor": factor, "group_id": a["group_id"], "match_key": list(match),
                            "from_case_id": a["case_id"], "to_case_id": b["case_id"],
                            "from_gold_rule": a["gold_rule"], "to_gold_rule": b["gold_rule"],
                            "from_patient_state": a["patient_state"], "to_patient_state": b["patient_state"],
                            "from_rule_order": a["rule_order"], "to_rule_order": b["rule_order"],
                            "from_raw_margin": raw_margin(a), "to_raw_margin": raw_margin(b),
                            "raw_margin_change": dm, "absolute_raw_margin_change": abs(dm),
                            "prediction_flip": a["prediction"] != b["prediction"]})
    factor_summary, guide_summary = [], []
    for group in ["all"] + groups:
        subset = baselines if group == "all" else [r for r in baselines if r["group_id"] == group]
        pp = paired if group == "all" else [r for r in paired if r["group_id"] == group]
        guide_summary.append({"group_id": group, "n_prompts": len(subset), "n_pairs": len(pp),
                              "correct": sum(r["correct"] for r in subset),
                              "same_prediction_pairs": sum(r["same_prediction"] for r in pp),
                              "both_correct_pairs": sum(r["both_correct"] for r in pp),
                              "both_wrong_pairs": sum(r["both_wrong"] for r in pp),
                              "predict_last_rule": sum(r["prediction"] == r["last_rule"] for r in subset),
                              "attention_favors_last_rule": sum(r["mass_" + r["last_rule"]] > r["mass_" + ("B" if r["last_rule"] == "A" else "A")] for r in subset),
                              "mean_gold_margin": mean(subset, "gold_margin"),
                              "mean_abs_patient_relative_signal": float(np.mean([abs(r["patient_relative_signal"]) for r in pp])),
                              "mean_abs_common_label_bias": float(np.mean([abs(r["common_label_bias"]) for r in pp]))})
        for factor in ("patient", "action_permutation", "rule_order_index"):
            ee = [r for r in effects if r["factor"] == factor and (group == "all" or r["group_id"] == group)]
            factor_summary.append({"group_id": group, "factor": factor, "n_comparisons": len(ee),
                                   "prediction_flips": sum(r["prediction_flip"] for r in ee),
                                   "mean_absolute_raw_margin_change": mean(ee, "absolute_raw_margin_change"),
                                   "mean_raw_margin_change": mean(ee, "raw_margin_change")})
    return paired, effects, factor_summary, guide_summary


def derive_rows(rows, index):
    derived, head_rows = [], []
    for row in rows:
        base = index[prompt_id(row) + ("baseline",)]
        gold, other = row["gold_rule"], "B" if row["gold_rule"] == "A" else "A"
        mg, mo = row["mass_" + gold], row["mass_" + other]
        bg, bo = base["mass_" + gold], base["mass_" + other]
        favored = gold if POLARITY[row["condition"]] >= 0 else other
        top = row["top_tokens"][0]
        out = {k: row[k] for k in ("case_id", "triplet_id", "group_id", "variant", "gold_rule",
                                  "patient_state", "action_permutation", "rule_order_index", "last_rule",
                                  "condition", "key_gain", "score_bias", "note_rule", "prediction", "correct",
                                  "logits_A", "logits_B", "gold_margin", "candidate_probability_mass",
                                  "candidate_probability_A", "candidate_probability_B",
                                  "conditional_probability_A", "conditional_probability_B",
                                  "prompt_sha256", "token_count", "prefix_length")}
        out.update({"mass_gold": mg, "mass_other": mo, "mass_non_rule": max(0., 1 - mg - mo), "favored_rule": favored,
                    "mass_favored": row["mass_" + favored], "mass_gold_change": mg - bg,
                    "mass_other_change": mo - bo,
                    "signed_mass_contrast_change": (mg - mo) - (bg - bo),
                    "log_gold_other_ratio": math.log(max(mg, FLOOR)) - math.log(max(mo, FLOOR)),
                    "delta_log_gold_other_ratio": math.log(max(mg, FLOOR)) - math.log(max(mo, FLOOR)) - math.log(max(bg, FLOOR)) + math.log(max(bo, FLOOR)),
                    "log_ratio_floor_used": min(mg, mo, bg, bo) < FLOOR,
                    "gold_margin_change": row["gold_margin"] - base["gold_margin"],
                    "baseline_gold_margin": base["gold_margin"],
                    "baseline_prediction": base["prediction"], "baseline_correct": bool(base["correct"]),
                    "prediction_flip": row["prediction"] != base["prediction"],
                    "correction": bool(row["correct"] and not base["correct"]),
                    "harm": bool(not row["correct"] and base["correct"]),
                    "predict_last_rule": row["prediction"] == row["last_rule"],
                    "gold_mass_ge_0_8": mg >= .8, "gold_mass_ge_0_9": mg >= .9,
                    "favored_mass_ge_0_8": row["mass_" + favored] >= .8,
                    "favored_mass_ge_0_9": row["mass_" + favored] >= .9,
                    "top_token_id": top["token_id"], "top_token_text": top["text"],
                    "top_token_probability": top["probability"],
                    "global_top_is_candidate": top["token_id"] in (362, 425),
                    "global_top_is_gold": top["token_id"] == (362 if gold == "A" else 425)})
        derived.append(out)
        for head in row["per_head_mass"]:
            hrow = {"case_id": row["case_id"], "variant": row["variant"],
                    "group_id": row["group_id"], "condition": row["condition"],
                    "layer": head["layer"], "kv_head": head["kv_head"],
                    "mass_gold": head["mass_" + gold], "mass_other": head["mass_" + other],
                    "mass_non_rule": max(0., 1 - head["mass_A"] - head["mass_B"]),
                    "mass_favored": head["mass_" + favored]}
            for role_name, label in (("gold", gold), ("other", other)):
                for role in ("id", "action"):
                    if f"mass_{label}_{role}" in head:
                        hrow[f"mass_{role_name}_{role}"] = head[f"mass_{label}_{role}"]
                if f"response_{label}" in head:
                    for name, value in head[f"response_{label}"].items():
                        hrow[f"actual_query_response_{role_name}_{name}"] = value
            head_rows.append(hrow)
    return derived, head_rows


def summarize(derived, groups):
    summaries, pairs = [], []
    for condition in CONDITIONS:
        rr = [r for r in derived if r["condition"] == condition]
        by_case = defaultdict(list)
        for row in rr:
            by_case[row["case_id"]].append(row)
        for case, pp in sorted(by_case.items()):
            pairs.append({"condition": condition, "case_id": case, "group_id": pp[0]["group_id"],
                          "both_correct": all(r["correct"] for r in pp),
                          "both_wrong": all(not r["correct"] for r in pp),
                          "same_prediction": pp[0]["prediction"] == pp[1]["prediction"],
                          "mean_gold_margin": mean(pp, "gold_margin"),
                          "mean_gold_margin_change": mean(pp, "gold_margin_change"),
                          "both_gold_mass_ge_0_8": all(r["gold_mass_ge_0_8"] for r in pp),
                          "both_gold_mass_ge_0_9": all(r["gold_mass_ge_0_9"] for r in pp),
                          "both_reading_improved": all(r["delta_log_gold_other_ratio"] > 0 for r in pp)})
        for group in ["all"] + groups:
            ss = rr if group == "all" else [r for r in rr if r["group_id"] == group]
            pp = [r for r in pairs if r["condition"] == condition and (group == "all" or r["group_id"] == group)]
            out = {"condition": condition, "group_id": group, "n_prompts": len(ss), "n_pairs": len(pp),
                   "correct": sum(r["correct"] for r in ss), "accuracy": mean(ss, "correct"),
                   "pair_correct": sum(r["both_correct"] for r in pp), "pair_accuracy": mean(pp, "both_correct"),
                   "prediction_A": sum(r["prediction"] == "A" for r in ss),
                   "prediction_B": sum(r["prediction"] == "B" for r in ss),
                   "prediction_flips": sum(r["prediction_flip"] for r in ss),
                   "flips_A_to_B": sum(r["baseline_prediction"] == "A" and r["prediction"] == "B" for r in ss),
                   "flips_B_to_A": sum(r["baseline_prediction"] == "B" and r["prediction"] == "A" for r in ss),
                   "corrections": sum(r["correction"] for r in ss), "harms": sum(r["harm"] for r in ss),
                   "predict_last_rule": sum(r["predict_last_rule"] for r in ss),
                   "last_rule_rate": mean(ss, "predict_last_rule"),
                   "gold_margin_improved": sum(r["gold_margin_change"] > 0 for r in ss),
                   "global_top_is_candidate": sum(r["global_top_is_candidate"] for r in ss),
                   "global_top_is_gold": sum(r["global_top_is_gold"] for r in ss)}
            for gold in ("A", "B"):
                gs = [r for r in ss if r["gold_rule"] == gold]
                out[f"n_gold_{gold}"] = len(gs)
                out[f"correct_gold_{gold}"] = sum(r["correct"] for r in gs)
                out[f"corrections_gold_{gold}"] = sum(r["correction"] for r in gs)
                out[f"harms_gold_{gold}"] = sum(r["harm"] for r in gs)
                out[f"prediction_flips_gold_{gold}"] = sum(r["prediction_flip"] for r in gs)
            for key in ("gold_margin", "gold_margin_change", "mass_gold", "mass_other", "mass_non_rule", "mass_favored",
                        "delta_log_gold_other_ratio", "candidate_probability_mass",
                        "conditional_probability_A", "conditional_probability_B"):
                out["mean_" + key] = mean(ss, key)
            for key in ("gold_mass_ge_0_8", "gold_mass_ge_0_9", "favored_mass_ge_0_8", "favored_mass_ge_0_9"):
                out["n_" + key] = sum(r[key] for r in ss)
                out["rate_" + key] = mean(ss, key)
            summaries.append(out)
    return summaries, pairs


def summarize_heads(head_rows, groups, heads):
    result = []
    for c in CONDITIONS:
        for group in ["all"] + groups:
            for layer, head in heads:
                ss = [r for r in head_rows if r["condition"] == c and r["layer"] == layer and r["kv_head"] == head
                      and (group == "all" or r["group_id"] == group)]
                if not ss:
                    raise ValueError("Missing registered head measurements")
                out = {"condition": c, "group_id": group, "layer": layer, "kv_head": head, "n_prompts": len(ss)}
                for key in ("mass_gold", "mass_other", "mass_non_rule", "mass_favored"):
                    out["mean_" + key] = mean(ss, key)
                for key in ss[0]:
                    if key.startswith("actual_query_response_") or key in (
                            "mass_gold_id", "mass_gold_action", "mass_other_id", "mass_other_action"):
                        out["mean_" + key] = mean(ss, key)
                for target in ("gold", "other", "favored"):
                    for threshold, label in ((.8, "0_8"), (.9, "0_9")):
                        n = sum(r["mass_" + target] >= threshold for r in ss)
                        out[f"n_{target}_mass_ge_{label}"] = n
                        out[f"rate_{target}_mass_ge_{label}"] = n / len(ss)
                result.append(out)
    return result


def contrasts(derived, groups):
    index = {prompt_id(r) + (r["condition"],): r for r in derived}
    prompts = sorted({prompt_id(r) for r in derived})
    records = []
    for name, plus, minus in CONTRASTS:
        for p in prompts:
            a, b = index[p + (plus,)], index[p + (minus,)]
            records.append({"contrast": name, "positive_condition": plus, "reference_condition": minus,
                            "case_id": p[0], "variant": p[1], "group_id": a["group_id"],
                            "gold_rule": a["gold_rule"], "same_prompt": a["prompt_sha256"] == b["prompt_sha256"],
                            "prediction_differs": a["prediction"] != b["prediction"],
                            "correct_positive": a["correct"], "correct_reference": b["correct"],
                            "correctness_difference": int(a["correct"]) - int(b["correct"]),
                            "gold_margin_difference": a["gold_margin"] - b["gold_margin"],
                            "gold_mass_difference": a["mass_gold"] - b["mass_gold"],
                            "log_ratio_difference": a["log_gold_other_ratio"] - b["log_gold_other_ratio"],
                            "candidate_probability_mass_difference": a["candidate_probability_mass"] - b["candidate_probability_mass"]})
    summary = []
    for name, plus, minus in CONTRASTS:
        for group in ["all"] + groups:
            ss = [r for r in records if r["contrast"] == name and (group == "all" or r["group_id"] == group)]
            out = {"contrast": name, "positive_condition": plus, "reference_condition": minus,
                   "group_id": group, "n_prompts": len(ss),
                   "predictions_differ": sum(r["prediction_differs"] for r in ss),
                   "positive_condition_more_correct": sum(r["correctness_difference"] > 0 for r in ss),
                   "reference_condition_more_correct": sum(r["correctness_difference"] < 0 for r in ss)}
            for key in ("correctness_difference", "gold_margin_difference", "gold_mass_difference",
                        "log_ratio_difference", "candidate_probability_mass_difference"):
                out["mean_" + key] = mean(ss, key)
            margin_differences = [r["gold_margin_difference"] for r in ss]
            out["n_gold_margin_difference_positive"] = sum(v > 0 for v in margin_differences)
            out["n_gold_margin_difference_negative"] = sum(v < 0 for v in margin_differences)
            out["median_gold_margin_difference"] = float(np.median(margin_differences))
            out["min_gold_margin_difference"] = min(margin_differences)
            out["max_gold_margin_difference"] = max(margin_differences)
            summary.append(out)
    return records, summary


def summarize_top_tokens(derived, groups):
    result = []
    for c in CONDITIONS:
        for group in ["all"] + groups:
            rr = [r for r in derived if r["condition"] == c and (group == "all" or r["group_id"] == group)]
            for token in sorted({r["top_token_id"] for r in rr}):
                ss = [r for r in rr if r["top_token_id"] == token]
                result.append({"condition": c, "group_id": group, "top_token_id": token,
                               "top_token_text": ss[0]["top_token_text"], "n_prompts": len(rr),
                               "top_token_count": len(ss), "top_token_rate": len(ss) / len(rr),
                               "mean_probability_when_top": mean(ss, "top_token_probability")})
    return result


def make_figure(factor_summary, guide_summary, derived, summaries, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "svg.fonttype": "none", "svg.hashsalt": "triage-readout-20261028",
                         "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.4))
    fig.subplots_adjust(left=.075, right=.985, top=.89, bottom=.17, wspace=.31, hspace=.57)
    fig.suptitle("Rule reading and answer readout · Fixed follow-up diagnostics", fontsize=18,
                 fontweight="bold", y=.98)
    fig.text(.5, .945, "All 24 fixed prompts · 12 paired configurations · 3 confirmation guidelines",
             ha="center", color="#64748B", fontsize=11)

    ax = axs[0, 0]
    factors = ("patient", "action_permutation", "rule_order_index")
    names = ("Patient fact", "Action mapping", "Rule order")
    vals = [next(r for r in factor_summary if r["group_id"] == "all" and r["factor"] == f) for f in factors]
    ax.bar(range(3), [r["mean_absolute_raw_margin_change"] for r in vals], color=["#008A91", "#DE8B33", "#AA6893"], width=.52, alpha=.8)
    gids = sorted({r["group_id"] for r in factor_summary} - {"all"})
    for g, offset in zip(gids, (-.11, 0, .11)):
        y = [next(r for r in factor_summary if r["group_id"] == g and r["factor"] == f)["mean_absolute_raw_margin_change"] for f in factors]
        ax.scatter(np.arange(3) + offset, y, s=32, color="#334155", marker="o", zorder=3)
    ymax = max(r["mean_absolute_raw_margin_change"] for r in factor_summary)
    ax.set_ylim(0, max(ymax * 1.28, .02))
    for i, r in enumerate(vals):
        guide_max = max(z["mean_absolute_raw_margin_change"] for z in factor_summary if z["factor"] == factors[i])
        ax.text(i, guide_max + ymax * .075, f"{r['prediction_flips']}/{r['n_comparisons']} answer flips", ha="center", fontsize=9.3)
    all_base = next(r for r in guide_summary if r["group_id"] == "all")
    ax.set_title(f"A  Baseline sensitivity · last rule chosen {all_base['predict_last_rule']}/24", loc="left", pad=13)
    ax.set_xticks(range(3), names)
    ax.set_ylabel("Mean absolute change in logit(A) − logit(B)")
    ax.text(.02, .97, "Bars: all matched comparisons; dots: guideline means", transform=ax.transAxes,
            va="top", fontsize=8.7, color="#64748B")

    ax = axs[0, 1]
    for c in CONDITIONS[1:]:
        ss = [r for r in derived if r["condition"] == c]
        marker = "x" if POLARITY[c] < 0 else ("s" if c == "projected4" else "o")
        ax.scatter([r["signed_mass_contrast_change"] for r in ss], [r["gold_margin_change"] for r in ss],
                   label=LABELS[c], s=24 if c == "projected4" else 31,
                   color=COLORS[c], marker=marker, alpha=.65, linewidths=.8)
    ax.axhline(0, color="#94A3B8", lw=.8)
    ax.axvline(0, color="#94A3B8", lw=.8)
    ax.set_title("B  Local reading change versus answer change", loc="left", pad=13)
    ax.set_xlabel("Change in (correct − sibling) attention mass")
    ax.set_ylabel("Change in correct-label logit margin")
    ax.legend(fontsize=7.8, ncols=3, frameon=True, facecolor="white", edgecolor="none",
              framealpha=.96, loc="upper center", bbox_to_anchor=(.5, 1.02))
    ax.margins(y=.17)

    ax = axs[1, 0]
    all_s = {r["condition"]: r for r in summaries if r["group_id"] == "all"}
    xx = np.arange(len(CONDITIONS))
    bars1 = ax.bar(xx - .18, [all_s[c]["correct"] for c in CONDITIONS], width=.34, color="#008A91", label="Correct prompts / 24")
    bars2 = ax.bar(xx + .18, [all_s[c]["pair_correct"] for c in CONDITIONS], width=.34, color="#DE8B33", label="Both-patient correct pairs / 12")
    ax.bar_label(bars1, fontsize=8, padding=2)
    ax.bar_label(bars2, fontsize=8, padding=2)
    ax.set_title("C  Answer counts · every registered condition", loc="left", pad=13)
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel("Correct counts (denominators shown in legend)")
    ax.set_xticks(xx, [LABELS[c] for c in CONDITIONS], rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    ax = axs[1, 1]
    coverage = [all_s[c]["mean_candidate_probability_mass"] for c in CONDITIONS]
    ax.bar(xx, coverage, width=.57, color=[COLORS[c] for c in CONDITIONS], alpha=.55)
    for g, offset in zip(gids, (-.13, 0, .13)):
        y = [next(r for r in summaries if r["condition"] == c and r["group_id"] == g)["mean_candidate_probability_mass"] for c in CONDITIONS]
        ax.scatter(xx + offset, y, s=24, color="#334155", zorder=3)
    ax.set_title("D  Full-vocabulary support for the scored labels", loc="left", pad=13)
    ax.set_ylabel("P( A) + P( B) before restricting to two labels")
    ax.set_ylim(0, 1.06)
    ax.set_xticks(xx, [LABELS[c] for c in CONDITIONS], rotation=35, ha="right")
    ax.text(.02, .97, "Bars: all prompts; dots: guideline means", transform=ax.transAxes,
            va="top", fontsize=8.7, color="#64748B")
    internal = [all_s[c] for c in ("projected32", "projected_minus32", "score_plus8", "score_minus8")]
    n_a_b = sum(r["flips_A_to_B"] for r in internal)
    n_b_a = sum(r["flips_B_to_A"] for r in internal)
    fig.text(.014, .025,
             "Markers in B show every prompt, not selected examples. Crosses reverse the key/score sign or give the sibling note.\n"
             "Fixed frozen directions and four heads · oracle routing signs · all gains and notes registered before this follow-up.\n"
             f"Strong internal-edit flips across four conditions: A→B {n_a_b}; B→A {n_b_a}. This does not establish two-way answer routing.\n"
             "Three source guidelines; controlled text variants are not independent clinical patients. No population confidence claim.",
             fontsize=9.3, color="#64748B", va="bottom")
    for ext in ("png", "svg", "pdf"):
        metadata = {"Date": None} if ext == "svg" else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        fig.savefig(figures / f"rule_readout_diagnostic.{ext}", dpi=190, metadata=metadata,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_narrative(output, summaries, guide_summary, factor_summary, contrast_summary):
    all_s = {r["condition"]: r for r in summaries if r["group_id"] == "all"}
    base = next(r for r in guide_summary if r["group_id"] == "all")
    ff = {r["factor"]: r for r in factor_summary if r["group_id"] == "all"}
    lines = ["# 规则读取与答案读出的固定诊断", "",
             "本页由完整真实记录自动生成，不进行拟合、重新选头、增益搜索或案例筛选。所有八配置均包含原固定的 24 个提示、12 个配对配置；独立来源为 3 个指南、6 个基础患者状态。", "",
             "## 基线的规则顺序与患者敏感性", "",
             f"基线在 {base['predict_last_rule']}/24 个提示中选择最后出现的规则；选中头聚合注意力在 {base['attention_favors_last_rule']}/24 个提示中偏向最后规则。相反患者中 {base['same_prediction_pairs']}/12 对给出相同答案。", "",
             "| 唯一改变的因素 | 答案翻转／匹配比较 | 平均绝对 A−B 间隔变化 |", "|---|---:|---:|"]
    for factor, label in (("patient", "患者决定事实"), ("action_permutation", "处置映射"), ("rule_order_index", "规则顺序")):
        x = ff[factor]
        lines.append(f"| {label} | {x['prediction_flips']}/{x['n_comparisons']} | {x['mean_absolute_raw_margin_change']:.6f} |")
    lines += ["", "全部匹配比较、患者相对信号与共同标签偏好、三个指南的分别结果均在配套 CSV 中。比较使用完整配对，不按答案效果选例。", "",
              "## 固定配置的全部结果", "",
              "| 配置 | 答对／24 | 配对全对／12 | 翻转 | 纠正 | 伤害 | 平均正确标签间隔变化 | 平均正确规则质量 | 平均候选概率覆盖 | 选最后规则／24 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        x = all_s[c]
        lines.append(f"| {c} | {x['correct']} | {x['pair_correct']} | {x['prediction_flips']} | {x['corrections']} | {x['harms']} | {x['mean_gold_margin_change']:+.6f} | {x['mean_mass_gold']:.6f} | {x['mean_candidate_probability_mass']:.6f} | {x['predict_last_rule']} |")
    lines += ["", "纠正与伤害均相对于同一提示的 baseline。相反符号与 sibling note 是诊断性反向控制，不是拟采用的决策策略。", "",
              "## 离散翻转方向与标签分层", "",
              "下表保留全部患者，A/B 的金标分母各为 12。斜线分别列出两个标签或两个翻转方向的计数，不表示比例。", "",
              "| 配置 | 预测 A／B | 翻转 A→B／B→A | 金标 A／B 答对 | 金标 A／B 被纠正 | 金标 A／B 被伤害 |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        x = all_s[c]
        lines.append(f"| {c} | {x['prediction_A']}／{x['prediction_B']} | {x['flips_A_to_B']}／{x['flips_B_to_A']} | {x['correct_gold_A']}／{x['correct_gold_B']} | {x['corrections_gold_A']}／{x['corrections_gold_B']} | {x['harms_gold_A']}／{x['harms_gold_B']} |")
    strong = [all_s[c] for c in ("projected32", "projected_minus32", "score_plus8", "score_minus8")]
    a_b, b_a = sum(x["flips_A_to_B"] for x in strong), sum(x["flips_B_to_A"] for x in strong)
    lines += ["", f"四种强内部编辑条件合计出现 {a_b} 次 A→B、{b_a} 次 B→A 翻转（这是同批患者在四条件中的事件计数，不增加独立样本）。正负编辑的正确数不同，并不能因此解释为两个答案方向都已稳定受控；必须同时检查预测标签偏移、纠正和伤害的金标分层。各指南同样的完整计数位于 `condition_summary.csv`。", "",
              "## 固定读取阈值的达成率", "",
              "目标质量门槛只用于描述干预是否达到较强读取；不是调整参数或筛样的依据。表中“被增强目标”在反向条件下是兄弟规则。baseline 的被增强目标列按正确规则计算，仅作为参照。", "",
              "| 配置 | 正确规则质量≥0.8／24 | 正确规则质量≥0.9／24 | 被增强目标质量≥0.8／24 | 被增强目标质量≥0.9／24 |",
              "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        x = all_s[c]
        lines.append(f"| {c} | {x['n_gold_mass_ge_0_8']} | {x['n_gold_mass_ge_0_9']} | {x['n_favored_mass_ge_0_8']} | {x['n_favored_mass_ge_0_9']} |")
    lines += ["", "逐头的全部质量与阈值达成率见 `head_summary.csv`，包含每个条件、每个指南及全部提示；没有把四头平均超过阈值当成所有头都超过阈值。", "",
              "## 正反控制与剂量差的匹配比较", "",
              "| 比较（前者减后者） | 答案不同／24 | 正确标签间隔差 | 正确规则质量差 |",
              "|---|---:|---:|---:|"]
    for x in contrast_summary:
        if x["group_id"] == "all":
            lines.append(f"| {x['positive_condition']} − {x['reference_condition']} | {x['predictions_differ']} | {x['mean_gold_margin_difference']:+.6f} | {x['mean_gold_mass_difference']:+.6f} |")
    lines += ["", "## 读出范围", "",
              f"基线完整词表的最高概率词元在 {all_s['baseline']['global_top_is_candidate']}/24 个提示中属于规范 A/B 候选，平均候选概率覆盖为 {all_s['baseline']['mean_candidate_probability_mass']:.4f}。因此原候选评分并非只读取两个几乎没有模型支持的词元。", "",
              f"显式正确注释得到 {all_s['note_gold']['correct']}/24 正确、{all_s['note_gold']['pair_correct']}/12 配对全对；相反注释得到 {all_s['note_sibling']['correct']}/24 正确。这是已提供规则 ID 后的输出通路控制，不检验模型能否自动判断患者条件。", "",
              "强内部编辑下，完整词表的最高概率词元可能移出规范 A/B。逐条件、逐指南的最高词元计数见 `top_token_summary.csv`，逐提示文本保存在 `patient_effects.csv`；不能把受限二选一正确率当成自由生成成绩，也不能仅凭首词元改变判定完整回答错误。", "",
              "![固定诊断的完整结果](../figures/rule_readout_diagnostic.png)", "",
              "候选概率覆盖是完整词表上的 P( A)+P( B)，没有先限制答案集合；二选一正确率不代表自由生成正确率。注意力质量取固定四个 KV 头及其共享查询头的目标编号与处置均值，不代表全模型注意力。", "",
              "所有统计是这批已使用开发指南上的后续诊断。没有新增独立指南、没有总体显著性推断，也不能将极强增益或 oracle 文本提示解释为正式端到端方法收益。"]
    (output / "ANALYSIS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="experiments/rule_readout_diagnostic_20261028/results/records.jsonl")
    parser.add_argument("--status", default="experiments/rule_readout_diagnostic_20261028/results/status.json")
    parser.add_argument("--output", default="experiments/rule_readout_diagnostic_20261028/analysis")
    parser.add_argument("--figures", default="experiments/rule_readout_diagnostic_20261028/figures")
    args = parser.parse_args()
    rows = load_rows(args.records)
    status = json.loads(Path(args.status).read_text())
    index, baselines, groups, heads = validate(rows, status)
    paired, factors, factor_summary, guide_summary = baseline_analysis(baselines, groups)
    derived, head_rows = derive_rows(rows, index)
    summaries, all_pairs = summarize(derived, groups)
    head_summary = summarize_heads(head_rows, groups, heads)
    contrast_rows, contrast_summary = contrasts(derived, groups)
    top_summary = summarize_top_tokens(derived, groups)
    output, figures = Path(args.output), Path(args.figures)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, data in (("baseline_patient_pairs", paired), ("baseline_factor_comparisons", factors),
                       ("baseline_factor_summary", factor_summary), ("baseline_guideline_summary", guide_summary),
                       ("patient_effects", derived), ("per_head_masses", head_rows), ("head_summary", head_summary),
                       ("condition_summary", summaries), ("paired_effects", all_pairs),
                       ("direction_contrasts", contrast_rows), ("contrast_summary", contrast_summary),
                       ("top_token_summary", top_summary)):
        write_csv(output / f"{name}.csv", data)
    summary = {"schema_version": "rule_readout_diagnostic.analysis.v1", "status": "complete",
               "n_records": len(rows), "n_unique_prompts": 24, "n_paired_configurations": 12,
               "n_base_patient_states": len({(r["group_id"], r["patient_state"]) for r in baselines}),
               "n_guidelines": len(groups), "guidelines": groups, "selected_heads": heads,
               "conditions": list(CONDITIONS), "condition_summary": [r for r in summaries if r["group_id"] == "all"],
               "baseline_factor_summary": [r for r in factor_summary if r["group_id"] == "all"],
               "baseline_guideline_summary": guide_summary,
               "contrast_summary": [r for r in contrast_summary if r["group_id"] == "all"],
               "head_summary": [r for r in head_summary if r["group_id"] == "all"],
               "top_token_summary": [r for r in top_summary if r["group_id"] == "all"],
               "mass_thresholds": [.8, .9], "thresholds_are_descriptive_only": True,
               "log_ratio_floor": FLOOR, "log_ratio_floor_uses": sum(r["log_ratio_floor_used"] for r in derived),
               "source_sha256": digest(__file__),
               "inputs": {str(args.records): digest(args.records), str(args.status): digest(args.status)},
               "scope": "All fixed reused development prompts; no case selection, fitting, new tuning, or population confidence claim."}
    write_json(output / "summary.json", summary)
    make_figure(factor_summary, guide_summary, derived, summaries, figures)
    write_narrative(output, summaries, guide_summary, factor_summary, contrast_summary)
    print(json.dumps({"status": "complete", "records": len(rows), "conditions": len(CONDITIONS),
                      "prompts": 24, "pairs": 12, "guidelines": 3, "output": str(output), "figures": str(figures)}))


if __name__ == "__main__":
    main()
