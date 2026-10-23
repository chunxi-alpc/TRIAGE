#!/usr/bin/env python3
"""Exploratory, CPU-only Qwen200 cluster-sign-flip sensitivity analysis.

All case outcomes are frozen. The exact sign-flip enumeration is a test of
CLUSTER-LEVEL SYMMETRY; it is NOT a permutation test of randomized treatments,
NOT a clinical-guideline-source test, and NOT part of the original preregistered
Holm family. The disease-prefix grouping is only a cancer-type proxy.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from step0_reanalysis import ROOT, read_jsonl

CONTRASTS = {
    # Difference is accuracy(first) - accuracy(second).
    "visible_A_minus_D_branch": (("A", "branch_visible"), ("D", "branch_visible")),
    "hidden_minus_visible_A": (("A", "hidden_prune"), ("A", "visible_prune")),
    "hidden_minus_visible_B": (("B", "hidden_prune"), ("B", "visible_prune")),
    "hidden_minus_visible_C": (("C", "hidden_prune"), ("C", "visible_prune")),
    "hidden_minus_visible_D": (("D", "hidden_prune"), ("D", "visible_prune")),
}


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cancer_prefix_proxy(label):
    """Heuristic cancer-type prefix, NOT a verified NCCN version/source ID."""
    match = re.match(r"^[A-Za-z]+", label)
    return match.group(0).lower() if match else label


def exact_cluster_signflip(cluster_sums):
    """Two-sided exact tail of sum(+/- fixed cluster sums), via dynamic programming.

    The null assumption is sign symmetry of cluster-level paired effects.
    Nothing here implies clusters were independently randomized.
    """
    sums = list(cluster_sums)
    if not sums or any(not isinstance(v, int) for v in sums):
        raise ValueError("Need integer paired outcome differences for at least one cluster")
    distribution = {0: 1}
    for value in sums:
        next_distribution = collections.defaultdict(int)
        for total, count in distribution.items():
            next_distribution[total + value] += count
            next_distribution[total - value] += count
        distribution = dict(next_distribution)
    observed = sum(sums)
    total = 2 ** len(sums)
    tail = sum(count for value, count in distribution.items()
               if abs(value) >= abs(observed))
    assert sum(distribution.values()) == total
    return {
        "observed_net_correct": observed,
        "clusters": len(sums),
        "nonzero_net_clusters": sum(value != 0 for value in sums),
        "tail_assignments": tail,
        "total_sign_assignments": total,
        "p_two_sided": tail / total,
    }


def holm_adjust(p_values):
    """Exploratory Holm family, separate from the original experimental family."""
    sorted_items = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    n = len(sorted_items)
    result = {}
    previous = 0.0
    for rank, (key, pvalue) in enumerate(sorted_items):
        adjusted = max(previous, min(1.0, (n - rank) * pvalue))
        result[key] = adjusted
        previous = adjusted
    return result


def index_scores(rows):
    result = {}
    for row in rows:
        key = (row["id"], row["variant"], row["method"])
        if key in result:
            raise ValueError("Duplicate outcome: " + str(key))
        result[key] = row
    return result


def summarize_one(values, grouping):
    groups = collections.defaultdict(list)
    for row in values:
        groups[grouping(row)].append(row["delta"])
    sums = {key: sum(v) for key, v in sorted(groups.items())}
    n = len(values)
    overall = sum(row["delta"] for row in values)
    leave_one_out = {}
    for group in groups:
        remainder = n - len(groups[group])
        if remainder:
            leave_one_out[group] = {
                "n_excluded": len(groups[group]),
                "effect_without_group": (overall - sums[group]) / remainder,
            }
    return {
        **exact_cluster_signflip(sums.values()),
        "cluster_net_differences": sums,
        "largest_cluster_n": max(map(len, groups.values())),
        "largest_cluster_fraction": max(map(len, groups.values())) / n,
        "leave_one_group_out_min_effect":
            min(x["effect_without_group"] for x in leave_one_out.values())
            if leave_one_out else None,
        "leave_one_group_out_max_effect":
            max(x["effect_without_group"] for x in leave_one_out.values())
            if leave_one_out else None,
    }


def analyse(scores, selected_ids):
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate frozen selected IDs")
    ids = sorted(selected_ids)
    selected = set(ids)
    lookup = index_scores(scores)
    if {row["id"] for row in scores} != selected:
        raise ValueError("Outcomes and frozen case IDs differ")
    disease = {}
    strata = {}
    for id_ in ids:
        a = lookup[id_, "A", "branch_visible"]
        d = lookup[id_, "D", "branch_visible"]
        if a["gold"] != d["gold"]:
            raise ValueError("A/D gold branch differs for " + id_)
        if a["disease"] != d["disease"] or a["stratum"] != d["stratum"]:
            raise ValueError("A/D pairing metadata mismatch: " + id_)
        disease[id_] = a["disease"]
        strata[id_] = a["stratum"]
    out = {
        "provenance": {
            "cases": len(ids),
            "disease_tree_keys": len(set(disease.values())),
            "cancer_prefix_proxy_groups":
                len({cancer_prefix_proxy(d) for d in disease.values()}),
            "K": sum(st == "K" for st in strata.values()),
            "U": sum(st == "U" for st in strata.values()),
            "new_inference": False,
        },
        "method": {
            "statistic": "sum of signed paired correctness differences within groups",
            "test": "exact two-sided cluster sign-flip using dynamic programming",
            "assumption": "cluster-level sign symmetry under the null, not randomized assignment",
            "independent_guideline_sources_verified": False,
            "cancer_prefix_is_source_id": False,
            "comparison_family": "five exploratory contrasts defined here; not original Holm family",
            "bootstrap_ci_source": "existing extended_public.json; not recalculated here",
        },
        "contrasts": {},
        "exploratory_holm": {},
    }
    for name, (first, second) in CONTRASTS.items():
        records = []
        for id_ in ids:
            lhs = lookup[id_, first[0], first[1]]
            rhs = lookup[id_, second[0], second[1]]
            if lhs["disease"] != rhs["disease"] or lhs["stratum"] != rhs["stratum"]:
                raise ValueError("Method pairing mismatch: " + id_ + "/" + name)
            if type(lhs["correct"]) is not bool or type(rhs["correct"]) is not bool:
                raise ValueError("Correctness must be Boolean")
            records.append({
                "id": id_, "disease": disease[id_],
                "delta": int(lhs["correct"]) - int(rhs["correct"]),
            })
        net = sum(r["delta"] for r in records)
        out["contrasts"][name] = {
            "n": len(records),
            "net_correct_first_minus_second": net,
            "paired_accuracy_delta": net / len(records),
            "disease_tree_proxy": summarize_one(records, lambda r: r["disease"]),
            "cancer_type_prefix_proxy": summarize_one(
                records, lambda r: cancer_prefix_proxy(r["disease"])
            ),
        }
    for unit in ("disease_tree_proxy", "cancer_type_prefix_proxy"):
        out["exploratory_holm"][unit] = holm_adjust({
            name: result[unit]["p_two_sided"]
            for name, result in out["contrasts"].items()
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    scores_path = args.results / "priority200/qwen/full_parser_v2/scores.jsonl"
    ids_path = args.results / "priority200/selection.json"
    selected = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
    if len(selected) != 200:
        raise ValueError("Frozen 200-case cohort required")
    result = analyse(read_jsonl(scores_path), selected)
    result["provenance"]["scores_sha256"] = source_digest(scores_path)
    result["provenance"]["selection_sha256"] = source_digest(ids_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print("STEP0_CLUSTER_SENSITIVITY_JSON=" +
          json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
