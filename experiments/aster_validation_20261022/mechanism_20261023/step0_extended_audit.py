#!/usr/bin/env python3
"""Extended, read-only step-0 audit of Qwen200, with optional private-case checks.

Public mode uses only tracked scores and selected IDs. Private mode inspects
prepared cases on the experiment server but emits aggregate counts and IDs only.
No model, tokenizer, GPU, or external service is required.
"""
import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

from step0_reanalysis import ROOT, cluster_ci, read_jsonl

METHODS = {"visible_prune": "branch_visible",
           "hidden_prune": "branch_hidden",
           "hidden_top2": "branch_hidden"}
VARIANTS = "ABCD"


def normalized_action(text):
    """Conservative string-level equivalence; never a clinical synonym test."""
    x = unicodedata.normalize("NFKC", str(text)).strip().casefold()
    return " ".join(x.rstrip(" .;:。；：").split())


def family_proxy(disease):
    """Heuristic prefix only; NOT a validated guideline-source identifier."""
    match = re.match(r"[a-zA-Z]+", disease)
    return match.group(0).lower() if match else disease


def indexed(scores):
    out = {}
    for row in scores:
        key = (row["id"], row["variant"], row["method"])
        if key in out:
            raise ValueError("duplicate score row: " + repr(key))
        out[key] = row
    return out


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def group_summary(ids, lookup, disease, group_key):
    groups = collections.defaultdict(list)
    for id_ in ids:
        a = lookup[id_, "A", "branch_visible"]
        d = lookup[id_, "D", "branch_visible"]
        groups[group_key(disease[id_])].append(int(a["correct"]) - int(d["correct"]))
    return {k: {"n": len(v), "net_A_minus_D": sum(v),
                "effect_A_minus_D": sum(v) / len(v)}
            for k, v in sorted(groups.items())}


def public_audit(scores, selected_ids):
    ids = sorted(set(selected_ids))
    if len(ids) != len(selected_ids):
        raise ValueError("selection has duplicate IDs")
    lookup = indexed(scores)
    disease, stratum = {}, {}
    transition = collections.Counter()
    by_stratum = {}
    for id_ in ids:
        a, d = lookup[id_, "A", "branch_visible"], lookup[id_, "D", "branch_visible"]
        if a["gold"] != d["gold"]:
            raise ValueError("A/D gold branch changed: " + id_)
        if a["disease"] != d["disease"] or a["stratum"] != d["stratum"]:
            raise ValueError("A/D case fields changed: " + id_)
        disease[id_], stratum[id_] = a["disease"], a["stratum"]
        key = ("correct" if a["correct"] else "wrong",
               "correct" if d["correct"] else "wrong",
               "changed" if a["prediction"] != d["prediction"] else "same")
        transition["/".join(key)] += 1
    if any(st not in ("K", "U") for st in stratum.values()):
        raise ValueError("unexpected K/U label")
    diffs = [int(lookup[i, "A", "branch_visible"]["correct"]) -
             int(lookup[i, "D", "branch_visible"]["correct"]) for i in ids]
    by_disease = group_summary(ids, lookup, disease, lambda x: x)
    by_family = group_summary(ids, lookup, disease, family_proxy)
    loo = {}
    for key, value in by_family.items():
        rest = len(ids) - value["n"]
        loo[key] = {"n_excluded": value["n"],
                    "effect_without_group": fraction(sum(diffs) - value["net_A_minus_D"], rest)}
    for group in ("all", "K", "U"):
        chosen = [i for i in ids if group == "all" or stratum[i] == group]
        delta = [int(lookup[i, "A", "branch_visible"]["correct"]) -
                 int(lookup[i, "D", "branch_visible"]["correct"]) for i in chosen]
        by_stratum[group] = {"n": len(chosen), "net_correct_A_minus_D": sum(delta),
                             "effect_A_minus_D": fraction(sum(delta), len(chosen)),
                             "cluster_ci_disease_descriptive":
                                 cluster_ci(delta, [disease[i] for i in chosen]),
                             "cluster_ci_family_proxy_descriptive":
                                 cluster_ci(delta, [family_proxy(disease[i]) for i in chosen])}
    alerts = {}
    for v in VARIANTS:
        tp = fp = fn = tn = 0
        visible_errors = hidden_errors_disagreement = 0
        for id_ in ids:
            vb = lookup[id_, v, "branch_visible"]
            hb = lookup[id_, v, "branch_hidden"]
            va = lookup[id_, v, "visible_prune"]
            ha = lookup[id_, v, "hidden_prune"]
            disagree = vb["prediction"] != hb["prediction"]
            wrong = not va["correct"]
            if disagree and wrong: tp += 1
            elif disagree: fp += 1
            elif wrong: fn += 1
            else: tn += 1
            visible_errors += wrong
            if disagree: hidden_errors_disagreement += not ha["correct"]
            if not disagree and va["prediction"] != ha["prediction"]:
                raise ValueError("agreeing branch yielded different action: " + id_)
        alerts[v] = {
            "n": len(ids), "target": "visible_prune final-action error against constructed gold",
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision_error_when_alert": fraction(tp, tp + fp),
            "error_detection_sensitivity": fraction(tp, tp + fn),
            "false_alert_rate_among_correct": fraction(fp, fp + tn),
            "coverage_when_silent": fraction(fn + tn, len(ids)),
            "accuracy_when_silent": fraction(tn, fn + tn),
            "hidden_errors_among_alerted": hidden_errors_disagreement,
            "total_visible_errors": visible_errors,
        }
    return {
        "provenance": {
            "n": len(ids), "new_model_calls": 0, "private_case_data_in_output": False,
            "disease_labels": len(by_disease),
            "family_prefix_groups": len(by_family),
            "family_warning": "Prefixes are sensitivity proxies, NOT verified independent guideline sources.",
        },
        "A_to_D_visible_transition": dict(sorted(transition.items())),
        "A_to_D_group_effects": by_stratum,
        "by_disease": by_disease, "by_family_proxy": by_family,
        "leave_one_family_proxy_out": loo,
        "alert_for_visible_answer_error": alerts,
        "statistical_scope": [
            "Bootstrap percentile intervals are descriptive; not Holm-adjusted or preregistered.",
            "Case-level McNemar does not account for shared guideline origin.",
            "Cluster ID and actual independent source tree have not been matched.",
            "Observed errors refer to constructed answer labels, not independently adjudicated clinical errors.",
        ],
    }


def branch_action_labels(case, variant, branch_ids):
    options = case["options"]
    norm_options = {label: normalized_action(value) for label, value in options.items()}
    rules = case["variant_rules"][variant]
    matched = set()
    for branch_id in branch_ids:
        if not isinstance(branch_id, int) or not 0 <= branch_id < len(case["branches"]):
            raise ValueError("invalid branch index in prepared case")
        for ri in case["branches"][branch_id]["rule_indices"]:
            action = normalized_action(rules[ri]["action"])
            matched.update(label for label, text in norm_options.items() if text == action)
    return matched


def private_case_audit(cases, scores, selected_ids):
    ids = set(selected_ids)
    lookup = indexed(scores)
    by_id = {}
    for case in cases:
        if case["id"] in by_id:
            raise ValueError("duplicate private case ID")
        by_id[case["id"]] = case
    if set(by_id) != ids:
        raise ValueError("private prepared cases differ from frozen selected IDs")
    integrity = collections.Counter()
    integrity_examples = collections.defaultdict(list)
    action_cells = collections.defaultdict(lambda: collections.Counter())
    action_examples = collections.defaultdict(lambda: collections.defaultdict(list))
    oracle = collections.defaultdict(collections.Counter)
    top2 = collections.defaultdict(collections.Counter)
    a24 = {"ids": [], "A_gold_action_in_wrong_branch": [],
           "A_gold_action_absent_from_wrong_branch": [],
           "D_gold_action_in_predicted_branch": [],
           "D_gold_action_absent_from_predicted_branch": []}
    for id_ in sorted(ids):
        c = by_id[id_]
        rules_a, rules_d = c["variant_rules"]["A"], c["variant_rules"]["D"]
        ai, si = c["app_rule"], c["sib_rule"]
        if len(rules_a) != len(rules_d) or not 0 <= ai < len(rules_a) or not 0 <= si < len(rules_a) or ai == si:
            integrity["invalid_rule_indices_or_length"] += 1
            integrity_examples["invalid_rule_indices_or_length"].append(id_)
            continue
        structural_ok = all(
            ra["condition"] == rd["condition"] and
            (rd["action"] == rules_a[si]["action"] if j == ai else
             rd["action"] == rules_a[ai]["action"] if j == si else
             rd["action"] == ra["action"])
            for j, (ra, rd) in enumerate(zip(rules_a, rules_d))
        )
        if not structural_ok:
            integrity["D_not_exact_two_rule_action_swap"] += 1
            integrity_examples["D_not_exact_two_rule_action_swap"].append(id_)
        sibling = c["sib_branch"]
        if len(c["branches"][sibling]["rule_indices"]) > 1:
            integrity["D_swap_into_multi_rule_sibling"] += 1
            integrity_examples["D_swap_into_multi_rule_sibling"].append(id_)
        if len({normalized_action(v) for v in c["options"].values()}) < len(c["options"]):
            integrity["duplicate_normalized_option_text"] += 1
            integrity_examples["duplicate_normalized_option_text"].append(id_)
        for v in VARIANTS:
            gold_branch = c["gold_branch"][v]
            gold = c["gold_answer"][v]
            allowed_gold = branch_action_labels(c, v, [gold_branch])
            if gold not in allowed_gold:
                integrity["constructed_gold_missing_from_gold_branch_" + v] += 1
                integrity_examples["constructed_gold_missing_from_gold_branch_" + v].append(id_)
            orow = lookup[id_, v, "gold_branch_prune"]
            if not orow["correct"]:
                ora = branch_action_labels(c, v, [gold_branch])
                oracle[v]["wrong"] += 1
                oracle[v]["wrong_option_permitted_by_gold_rules"] += orow["prediction"] in ora
                oracle[v]["wrong_option_absent_from_gold_rules"] += orow["prediction"] not in ora
                if orow["original_action_selected"]:
                    oracle[v]["wrong_choose_original"] += 1
            for method, bmethod in METHODS.items():
                b = lookup[id_, v, bmethod]
                a = lookup[id_, v, method]
                if a.get("upstream_branch_parse_failure"):
                    action_cells[(v, method)]["parse_fail"] += 1
                    continue
                first = a.get("predicted_branch")
                if first is None or first != ord(b["prediction"]) - ord("A"):
                    raise ValueError("branch-action linkage mismatch: " + id_ + "/" + v + "/" + method)
                indices = [first]
                if method == "hidden_top2" and a.get("runner_up") is not None:
                    indices.append(a["runner_up"])
                allowed = branch_action_labels(c, v, indices)
                cell = "branch_" + ("correct" if b["correct"] else "wrong") + \
                       "_action_" + ("correct" if a["correct"] else "wrong")
                action_cells[(v, method)][cell] += 1
                action_cells[(v, method)]["predicted_option_in_retained_rules"] += a["prediction"] in allowed
                action_cells[(v, method)]["predicted_option_absent_from_retained_rules"] += a["prediction"] not in allowed
                if not b["correct"] and a["correct"]:
                    kind = "wrong_branch_right_answer_gold_" + ("present" if a["gold"] in allowed else "absent")
                    action_cells[(v, method)][kind] += 1
                    action_examples[(v, method)][kind].append(id_)
                if b["correct"] and not a["correct"]:
                    kind = "correct_branch_wrong_answer_option_" + ("present" if a["prediction"] in allowed else "absent")
                    action_cells[(v, method)][kind] += 1
                    action_examples[(v, method)][kind].append(id_)
                if method == "hidden_top2":
                    first_allowed = branch_action_labels(c, v, [first])
                    if len(indices) == 2:
                        second_allowed = branch_action_labels(c, v, [indices[1]])
                        status = ("first" if a["prediction"] in first_allowed else "") + \
                                 ("second" if a["prediction"] in second_allowed else "")
                        top2[v]["selected_option_origin_" + (status or "neither")] += 1
            # The 24 A events are a prespecified retrospective inspection set.
        ba = lookup[id_, "A", "branch_hidden"]
        aa = lookup[id_, "A", "hidden_prune"]
        if not ba["correct"] and aa["correct"]:
            da = lookup[id_, "D", "hidden_prune"]
            a24["ids"].append(id_)
            a_allowed = branch_action_labels(c, "A", [aa["predicted_branch"]])
            d_allowed = branch_action_labels(c, "D", [da["predicted_branch"]])
            a24["A_gold_action_in_wrong_branch" if aa["gold"] in a_allowed
                else "A_gold_action_absent_from_wrong_branch"].append(id_)
            a24["D_gold_action_in_predicted_branch" if da["gold"] in d_allowed
                else "D_gold_action_absent_from_predicted_branch"].append(id_)
    return {
        "scope": "PRIVATE prepared-case structure; output omits patient and rule text",
        "integrity_counts": dict(sorted(integrity.items())),
        "integrity_case_ids": dict(sorted(integrity_examples.items())),
        "method_variant_cells": {v + "/" + m: dict(counter)
                                 for (v, m), counter in sorted(action_cells.items())},
        "method_variant_example_ids": {v + "/" + m: dict(examples)
                                       for (v, m), examples in sorted(action_examples.items())},
        "oracle_error_rule_membership": {v: dict(oracle[v]) for v in VARIANTS},
        "top2_option_source": {v: dict(top2[v]) for v in VARIANTS},
        "A_wrong_branch_right_answer_cohort": a24,
        "interpretation_limits": [
            "Action inclusion uses conservative normalized string identity, not clinical equivalence.",
            "An answer absent from retained rule actions shows a surface mismatch, not memory causation.",
            "An answer present in the retained rules does not prove the model followed those rules.",
            "A/D semantic coherence and clinical reasonableness require independent clinician review.",
            "Multiple actions in a branch may be valid alternatives, not logical contradictions.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT)
    parser.add_argument("--cases", type=Path, help="Private prepared 200-case JSONL; never commit it")
    parser.add_argument("--output", required=True, type=Path, help="Write a new audit JSON")
    args = parser.parse_args()
    ids = json.loads((args.results / "priority200/selection.json").read_text())["ids"]
    if len(ids) != 200:
        raise ValueError("expected fixed200 selection")
    scores = read_jsonl(args.results / "priority200/qwen/full_parser_v2/scores.jsonl")
    out = public_audit(scores, ids)
    if args.cases:
        out["private_case_checks"] = private_case_audit(read_jsonl(args.cases), scores, ids)
    else:
        out["private_case_checks"] = {"status": "not_performed_requires_private_prepared_cases"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.output, "for", len(ids), "cases; private checks:", bool(args.cases))


if __name__ == "__main__":
    main()
