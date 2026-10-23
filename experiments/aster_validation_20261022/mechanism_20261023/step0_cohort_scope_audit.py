#!/usr/bin/env python3
"""Read-only scope/overlap audit: frozen Qwen200 vs PR25 frozen 800-case manifest.

This script never reads model logits or predicts outcomes for the 800 cases.
It assesses cohort composition and protocol compatibility only. The original
200 data tree IDs are treated as observed tree IDs, not proven independent
NCCN guideline sources. Strict new-tree analysis is post-hoc descriptive.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from step0_reanalysis import ROOT, read_jsonl
from step0_cluster_sensitivity import cancer_prefix_proxy

PR25_FROZEN_COMMIT = "2407c15824f6c2d957dc166f3ad70f57361cabab"
PR25_MANIFEST_PATH = "experiments/qwen800_methods_20261023/results/case_manifest.jsonl"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope_cell(rows):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ID within a cohort subset")
    return {
        "n": len(rows),
        "K": sum(r["stratum"] == "K" for r in rows),
        "U": sum(r["stratum"] == "U" for r in rows),
        "trees": len({r["disease"] for r in rows}),
        "D": sum("D" in r["variants"] for r in rows),
        "CS": sum("CS" in r["variants"] for r in rows),
        "strict_protocol": sum(r["pair_protocol"] == "PR23_strict" for r in rows),
        "single_action_gold_branch": sum(r["single_action_gold_branch"] for r in rows),
    }


def analyse(old_scores, original_ids, expanded_rows):
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Duplicate original200 ID")
    if len(expanded_rows) != len({r["id"] for r in expanded_rows}):
        raise ValueError("Duplicate expanded-cohort ID")
    old_map = {}
    for r in old_scores:
        if r["variant"] == "A" and r["method"] == "branch_hidden":
            if r["id"] in old_map:
                raise ValueError("Duplicate original branch record")
            old_map[r["id"]] = r["disease"]
    if set(old_map) != set(original_ids):
        raise ValueError("Original200 cases lack matching A hidden branch records")
    old_ids = set(original_ids)
    old_trees = set(old_map.values())
    old_cancer_prefixes = {cancer_prefix_proxy(tree) for tree in old_trees}
    allowed_protocols = {"PR23_strict", "extended_offpath"}
    for r in expanded_rows:
        if r["pair_protocol"] not in allowed_protocols:
            raise ValueError("Unexpected protocol: " + str(r["pair_protocol"]))
        if r["stratum"] not in ("K", "U"):
            raise ValueError("Unexpected K/U group")
        if r["id"] in old_ids and r["disease"] != old_map[r["id"]]:
            raise ValueError("Disease tree for reused case changed: " + r["id"])
        if not isinstance(r["single_action_gold_branch"], bool):
            raise ValueError("Invalid unique action branch field")
        if not isinstance(r["variants"], list):
            raise ValueError("Expected variant list")
        if "A" not in r["variants"] or "B" not in r["variants"] or "C" not in r["variants"]:
            raise ValueError("Missing required A/B/C case variants")
    subsets = {
        "all": expanded_rows,
        "no_case_overlap_with_200": [r for r in expanded_rows if r["id"] not in old_ids],
        "no_tree_overlap_with_200": [r for r in expanded_rows if r["disease"] not in old_trees],
        "no_cancer_prefix_overlap_with_200": [
            r for r in expanded_rows
            if cancer_prefix_proxy(r["disease"]) not in old_cancer_prefixes
        ],
        "no_case_overlap_strict": [
            r for r in expanded_rows
            if r["id"] not in old_ids and r["pair_protocol"] == "PR23_strict"
        ],
        "no_case_overlap_strict_single": [
            r for r in expanded_rows
            if r["id"] not in old_ids
            and r["pair_protocol"] == "PR23_strict"
            and r["single_action_gold_branch"]
        ],
        "no_tree_overlap_strict": [
            r for r in expanded_rows
            if r["disease"] not in old_trees and r["pair_protocol"] == "PR23_strict"
        ],
        "no_cancer_prefix_overlap_strict": [
            r for r in expanded_rows
            if cancer_prefix_proxy(r["disease"]) not in old_cancer_prefixes
            and r["pair_protocol"] == "PR23_strict"
        ],
        "no_cancer_prefix_overlap_strict_single": [
            r for r in expanded_rows
            if cancer_prefix_proxy(r["disease"]) not in old_cancer_prefixes
            and r["pair_protocol"] == "PR23_strict"
            and r["single_action_gold_branch"]
        ],
        "no_tree_overlap_strict_single": [
            r for r in expanded_rows
            if r["disease"] not in old_trees
            and r["pair_protocol"] == "PR23_strict"
            and r["single_action_gold_branch"]
        ],
    }
    per_tree = collections.defaultdict(list)
    for r in expanded_rows:
        per_tree[r["disease"]].append(r)
    return {
        "provenance": {
            "original200_n": len(original_ids),
            "pr25_cohort_n": len(expanded_rows),
            "pr25_frozen_commit": PR25_FROZEN_COMMIT,
            "pr25_manifest_path": PR25_MANIFEST_PATH,
            "new_model_calls": 0,
            "tested_800_predictions": False,
        },
        "counts": {key: scope_cell(rows) for key, rows in subsets.items()},
        "original200_tree_ids": sorted(old_trees),
        "new_tree_ids": sorted(set(per_tree) - old_trees),
        "original200_cancer_prefix_proxies": sorted(old_cancer_prefixes),
        "new_cancer_prefix_proxies": sorted(
            {cancer_prefix_proxy(tree) for tree in per_tree} - old_cancer_prefixes
        ),
        "original200_overlap_ids": sorted(
            set(r["id"] for r in expanded_rows) & old_ids
        ),
        "by_tree": {
            key: {
                **scope_cell(rows),
                "present_in_original200": key in old_trees,
            }
            for key, rows in sorted(per_tree.items())
        },
        "interpretation_limits": [
            "The original200 cohort is not an independent validation set for PR25.",
            "Nonoverlapping case IDs do not guarantee independence of source trees.",
            "Post-hoc new-tree/strict/single-action intersections are descriptive.",
            "The PR23 strict and PR25 extended_offpath construction protocols differ.",
            "Cancer-type prefixes are a heuristic, not a validated oncology guideline source/version.",
            "New trees are data-tree IDs, not verified independent NCCN guideline versions.",
            "No PR25 action outcome has been evaluated by this script.",
            "The 84 CS cases are one-tree synthetic changes, not cross-tree validation.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT)
    p.add_argument("--pr25-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    old_scores_path = args.results / "priority200/qwen/full_parser_v2/scores.jsonl"
    old_ids_path = args.results / "priority200/selection.json"
    old_ids = json.loads(old_ids_path.read_text(encoding="utf-8"))["ids"]
    if len(old_ids) != 200:
        raise ValueError("Expected frozen old200")
    old_scores = read_jsonl(old_scores_path)
    expanded = read_jsonl(args.pr25_manifest)
    if len(expanded) != 800:
        raise ValueError("Expected frozen PR25 cohort of 800")
    out = analyse(old_scores, old_ids, expanded)
    out["provenance"].update({
        "old_scores_sha256": sha256(old_scores_path),
        "old_selection_sha256": sha256(old_ids_path),
        "pr25_case_manifest_sha256": sha256(args.pr25_manifest),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("STEP0_COHORT_SCOPE_JSON=" +
          json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
