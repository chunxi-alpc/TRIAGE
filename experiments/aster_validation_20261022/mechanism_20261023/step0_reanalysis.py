#!/usr/bin/env python3
"""Read-only, CPU-only reanalysis of fixed Qwen200 ASTER results.

Uses only tracked JSONL and selection.json unless --cases is explicitly given.
No model import, no network, no inference, no test-set tuning.
"""
import argparse
import collections
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"
METHOD_BRANCH = {
    "visible_prune": "branch_visible",
    "hidden_prune": "branch_hidden",
    "hidden_top2": "branch_hidden",
    "hidden_signed": "branch_hidden",
    "hidden_unsigned": "branch_hidden",
}
VARIANTS = ("A", "B", "C", "D")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_mcnemar_p(losses, gains):
    """Unadjusted exact two-sided sign test on discordant paired outcomes."""
    n = losses + gains
    if n == 0:
        return 1.0
    k = min(losses, gains)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def cluster_ci(diffs, clusters, repetitions=2000):
    """Descriptive percentile CI; resample disease labels, not independent rows."""
    grouped = collections.defaultdict(list)
    for value, group in zip(diffs, clusters):
        grouped[group].append(value)
    keys = sorted(grouped)
    if len(keys) < 2:
        return None
    sums = [sum(grouped[k]) for k in keys]
    counts = [len(grouped[k]) for k in keys]
    rng = random.Random(20261023)
    draws = []
    for _ in range(repetitions):
        indices = [rng.randrange(len(keys)) for _ in keys]
        draws.append(sum(sums[i] for i in indices) / sum(counts[i] for i in indices))
    draws.sort()

    def percentile(q):
        at = q * (len(draws) - 1)
        lo = int(at)
        hi = min(len(draws) - 1, lo + 1)
        return draws[lo] * (1 - (at - lo)) + draws[hi] * (at - lo)

    return [percentile(0.025), percentile(0.975)]


def index_method(rows):
    result = {}
    for row in rows:
        key = (row["id"], row["variant"], row["method"])
        if key in result:
            raise ValueError("Duplicate mechanism record: " + str(key))
        result[key] = row
    return result


def analyse(scores, cad_scores, d_scores, selected_ids, private_cases=None):
    ids = set(selected_ids)
    assert len(ids) == len(selected_ids) == 200
    assert len(scores) == 10426 and {r["id"] for r in scores} == ids
    assert len(cad_scores) == 9000 and {r["id"] for r in cad_scores} == ids
    assert len(d_scores) == 1200 and {r["id"] for r in d_scores} == ids
    s = index_method(scores)
    assert len(s) == len(scores)
    by_id = {id_: {} for id_ in ids}
    for row in scores:
        by_id[row["id"]][(row["variant"], row["method"])] = row
    assert all(len(v) == (13 * 4 + (13 if any(k[0] == "CS" for k in v) else 0)) for v in by_id.values())
    strata = {i: by_id[i][("A", "branch_hidden")]["stratum"] for i in ids}
    assert collections.Counter(strata.values()) == {"K": 76, "U": 124}
    diseases = {i: by_id[i][("A", "branch_hidden")]["disease"] for i in ids}
    assert len(set(diseases.values())) == 23
    out = {
        "provenance": {
            "cases": 200, "K": 76, "U": 124,
            "disease_labels": 23, "mechanism_rows": len(scores),
            "cad_rows": len(cad_scores), "d_option_order_rows": len(d_scores),
            "analysis_scope": "fixed200 only; retrospective descriptive audit",
            "new_inference": False,
        },
        "branch_A_visible_vs_D_visible": {},
        "decomposition": {}, "visibility_disagreement": {},
        "top2_transitions": {}, "oracle_errors": {},
        "cad_old128_vs_filtered_original": {},
        "A_wrong_branch_right_then_D": {},
        "semantic_wrong_branch_check": {"status": "requires_private_prepared_cases"},
    }
    for group in ("all", "K", "U"):
        chosen = sorted(i for i in ids if group == "all" or strata[i] == group)
        a = [s[i, "A", "branch_visible"] for i in chosen]
        d = [s[i, "D", "branch_visible"] for i in chosen]
        both = sum(x["correct"] and y["correct"] for x, y in zip(a, d))
        loses = sum(x["correct"] and not y["correct"] for x, y in zip(a, d))
        gains = sum(not x["correct"] and y["correct"] for x, y in zip(a, d))
        neither = len(chosen) - both - loses - gains
        diffs = [int(x["correct"]) - int(y["correct"]) for x, y in zip(a, d)]
        out["branch_A_visible_vs_D_visible"][group] = {
            "n": len(chosen), "A_correct": both + loses,
            "D_correct": both + gains, "both_correct": both,
            "A_correct_D_wrong": loses, "A_wrong_D_correct": gains,
            "both_wrong": neither,
            "branch_prediction_changed": sum(x["prediction"] != y["prediction"] for x, y in zip(a, d)),
            "delta_A_minus_D": sum(diffs) / len(chosen),
            "mcnemar_p_unadjusted_exploratory": exact_mcnemar_p(loses, gains),
            "disease_cluster_ci_descriptive": cluster_ci(diffs, [diseases[i] for i in chosen]),
        }
    for method, bmethod in METHOD_BRANCH.items():
        out["decomposition"][method] = {}
        for v in VARIANTS:
            pairs = [(s[i, v, bmethod], s[i, v, method]) for i in sorted(ids)]
            bc_ac = sum(b["correct"] and a["correct"] for b, a in pairs)
            bc_aw = sum(b["correct"] and not a["correct"] for b, a in pairs)
            bw_ac = sum(not b["correct"] and a["correct"] for b, a in pairs)
            bw_aw = sum(not b["correct"] and not a["correct"] for b, a in pairs)
            bc = bc_ac + bc_aw
            bw = bw_ac + bw_aw
            assert bc + bw == 200
            out["decomposition"][method][v] = {
                "n": 200, "branch_correct_action_correct": bc_ac,
                "branch_correct_action_wrong": bc_aw,
                "branch_wrong_action_correct": bw_ac,
                "branch_wrong_action_wrong": bw_aw,
                "branch_accuracy": bc / 200,
                "action_accuracy": (bc_ac + bw_ac) / 200,
                "action_correct_given_branch_correct": bc_ac / bc if bc else None,
                "action_correct_given_branch_wrong": bw_ac / bw if bw else None,
                "correct_branch_contribution": bc_ac / 200,
                "wrong_branch_contribution": bw_ac / 200,
            }
    a_hits = []
    for i in sorted(ids):
        a = by_id[i]
        branch = a["A", "branch_hidden"]
        action = a["A", "hidden_prune"]
        if not branch["correct"] and action["correct"]:
            d = a["D", "hidden_prune"]
            a_hits.append({
                "id": i, "stratum": strata[i],
                "original_action_in_A": bool(action["original_action_selected"]),
                "A_D_same_option": action["prediction"] == d["prediction"],
                "D_action_correct": bool(d["correct"]),
                "D_original_action_selected": bool(d["original_action_selected"]),
            })
    out["A_wrong_branch_right_then_D"] = {
        "n": len(a_hits),
        "all_A_choose_original": all(x["original_action_in_A"] for x in a_hits),
        "same_answer_in_D": sum(x["A_D_same_option"] for x in a_hits),
        "D_correct": sum(x["D_action_correct"] for x in a_hits),
        "D_choose_original": sum(x["D_original_action_selected"] for x in a_hits),
        "K": sum(x["stratum"] == "K" for x in a_hits),
        "U": sum(x["stratum"] == "U" for x in a_hits),
        "case_ids": [x["id"] for x in a_hits],
    }
    for v in VARIANTS:
        agree, disagree = [], []
        for i in sorted(ids):
            rr = by_id[i]
            vb, hb = rr[v, "branch_visible"], rr[v, "branch_hidden"]
            visible, hidden = rr[v, "visible_prune"], rr[v, "hidden_prune"]
            (disagree if vb["prediction"] != hb["prediction"] else agree).append((i, visible, hidden))
        assert len(agree) + len(disagree) == 200
        equal_action_on_agreement = sum(x["prediction"] == y["prediction"] for _, x, y in agree)
        out["visibility_disagreement"][v] = {
            "n": 200, "disagree": len(disagree), "agree": len(agree),
            "visible_wrong_if_disagree": sum(not x["correct"] for _, x, _ in disagree),
            "hidden_wrong_if_disagree": sum(not y["correct"] for _, _, y in disagree),
            "visible_wrong_if_agree": sum(not x["correct"] for _, x, _ in agree),
            "hidden_wrong_if_agree": sum(not y["correct"] for _, _, y in agree),
            "same_final_action_if_agree": equal_action_on_agreement,
            "agreement_subset_accuracy_visible": sum(x["correct"] for _, x, _ in agree) / len(agree),
            "alert_rate": len(disagree) / 200,
            "disagree_hidden_else_visible_accuracy": (
                sum(y["correct"] for _, _, y in disagree)
                + sum(x["correct"] for _, x, _ in agree)
            ) / 200,
        }
        top_old = [s[i, v, "hidden_prune"] for i in sorted(ids)]
        top_new = [s[i, v, "hidden_top2"] for i in sorted(ids)]
        top_discard = sum(x["correct"] and not y["correct"] for x, y in zip(top_old, top_new))
        top_rescue = sum(not x["correct"] and y["correct"] for x, y in zip(top_old, top_new))
        wrong = [i for i in sorted(ids) if not s[i, v, "branch_hidden"]["correct"]]
        out["top2_transitions"][v] = {
            "n": 200, "top1_correct": sum(x["correct"] for x in top_old),
            "top2_correct": sum(x["correct"] for x in top_new),
            "top1_right_top2_wrong": top_discard,
            "top1_wrong_top2_right": top_rescue,
            "mcnemar_p_unadjusted_exploratory": exact_mcnemar_p(top_discard, top_rescue),
            "wrong_first_branch": len(wrong),
            "runner_up_is_gold_among_wrong": sum(
                s[i, v, "hidden_top2"]["runner_up"] ==
                ord(s[i, v, "branch_hidden"]["gold"]) - ord("A")
                for i in wrong
            ),
        }
        gold = [s[i, v, "gold_branch_prune"] for i in sorted(ids)]
        err = [r for r in gold if not r["correct"]]
        out["oracle_errors"][v] = {
            "n": 200, "wrong": len(err),
            "parse_failures": sum(bool(x["parse_failure"]) for x in err),
            "wrong_choose_original": sum(bool(x["original_action_selected"]) for x in err),
            "wrong_choose_sibling": sum(bool(x["sibling_action_selected"]) for x in err),
            "case_ids": [x["id"] for x in err],
        }

    cad = {}
    for r in cad_scores:
        key = (r["id"], r["variant"][0], r["method"], bool(r["assigned_wrong"]), r["revision"])
        if key in cad:
            raise ValueError("Duplicate CAD record: " + str(key))
        cad[key] = r
    assert len(cad) == 9000
    for v in ("A", "B", "C"):
        old_only, new_only = [], []
        old_correct = new_correct = 0
        old_original = new_original = 0
        by_stratum = {st: {"n": 0, "old_only": 0, "new_only": 0} for st in ("K", "U")}
        for i in sorted(ids):
            a = cad[i, v, "cad", False, "CAD128"]
            b = cad[i, v, "cad", False, "CAD_filtered_original"]
            assert a["stratum"] == b["stratum"] == strata[i]
            old_correct += bool(a["correct"])
            new_correct += bool(b["correct"])
            old_original += a["prediction"] == a["original_answer"]
            new_original += b["prediction"] == b["original_answer"]
            st = by_stratum[strata[i]]
            st["n"] += 1
            if a["correct"] and not b["correct"]:
                old_only.append(i)
                st["old_only"] += 1
            if b["correct"] and not a["correct"]:
                new_only.append(i)
                st["new_only"] += 1
        out["cad_old128_vs_filtered_original"][v] = {
            "n": 200, "old_correct": old_correct,
            "filtered_correct": new_correct,
            "old_only_ids": old_only, "filtered_only_ids": new_only,
            "old_only": len(old_only), "filtered_only": len(new_only),
            "mcnemar_p_unadjusted_exploratory": exact_mcnemar_p(len(old_only), len(new_only)),
            "old_choose_original": old_original, "filtered_choose_original": new_original,
            "by_stratum": by_stratum,
            "cannot_identify_masked_gold_token_from_public_scores": True,
        }

    # These files contain final predictions, not actual retained rule text.
    # Full action-set membership requires private, prepared test.jsonl (optional).
    if private_cases is not None:
        cases = {r["id"]: r for r in private_cases}
        assert set(cases) == ids and len(private_cases) == 200
        member = {}
        for method in ("visible_prune", "hidden_prune", "hidden_top2"):
            bmethod = METHOD_BRANCH[method]
            member[method] = {}
            for v in VARIANTS:
                totals = collections.Counter()
                for i in sorted(ids):
                    c = cases[i]
                    b = s[i, v, bmethod]
                    a = s[i, v, method]
                    if b["correct"] or not a["correct"]:
                        continue
                    selected = [a["predicted_branch"]]
                    if method == "hidden_top2" and a["runner_up"] is not None:
                        selected.append(a["runner_up"])
                    rules = c["variant_rules"][v]
                    allowed = {
                        rules[j]["action"].strip().casefold()
                        for branch_id in selected
                        for j in c["branches"][branch_id]["rule_indices"]
                    }
                    gold_text = c["options"][c["gold_answer"][v]].strip().casefold()
                    totals["wrong_branch_right"] += 1
                    totals["gold_action_in_kept_rules"] += gold_text in allowed
                    totals["gold_action_absent_from_kept_rules"] += gold_text not in allowed
                member[method][v] = dict(totals)
        out["semantic_wrong_branch_check"] = {
            "status": "performed_with_private_prepared_cases",
            "membership_by_exact_normalized_action_text": member,
            "warning": "Absence of gold action in retained rules is not proof of parameter-memory causality.",
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT)
    parser.add_argument("--cases", type=Path, help="Optional private prepared priority200/test.jsonl; never publish raw case text.")
    parser.add_argument("--output", type=Path, required=True, help="Write audit JSON outside tracked baseline files.")
    args = parser.parse_args()
    folder = args.results
    result = analyse(
        read_jsonl(folder / "priority200/qwen/full_parser_v2/scores.jsonl"),
        read_jsonl(folder / "cad200_final/cad_scores.jsonl"),
        read_jsonl(folder / "d_option_order/paired_scores.jsonl"),
        json.loads((folder / "priority200/selection.json").read_text(encoding="utf-8"))["ids"],
        read_jsonl(args.cases) if args.cases else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {result['provenance']['cases']} cases; wrote {args.output}")


if __name__ == "__main__":
    main()
