"""Paired full-path differences on the registered original-view test cases."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import read_rows
from .metrics import strict_score

FIELDS = ("both_correct", "corrected", "regressed", "both_wrong")


def count_by_source(cases, predictions, controls, primary="no_js"):
    reference = {}
    for row in read_rows(cases):
        reference[row["case_id"]] = {k: row[k] for k in ("source_id", "gold_rules", "gold_leaf")}
    arms = {k: {} for k in [primary, *controls]}
    for input_path in predictions:
        files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
        for path in files:
            for row in read_rows(path):
                kind, cid = row["kind"], row["case_id"]
                if kind not in arms or cid not in reference or row.get("view", "original") != "original":
                    continue
                arms[kind][cid] = (row["status"] == "complete" and
                                   strict_score(reference[cid], row["result"]["text"])["exact_path"])
    for kind, values in arms.items():
        if len(values) != len(reference):
            raise ValueError(f"{kind}: recorded {len(values)} of {len(reference)} registered cases")
    counts = {}
    for cid, case in reference.items():
        for control in controls:
            key = (case["source_id"], control)
            table = counts.setdefault(key, dict.fromkeys(FIELDS, 0))
            main, other = arms[primary][cid], arms[control][cid]
            field = ("both_correct" if main and other else "corrected" if main
                     else "regressed" if other else "both_wrong")
            table[field] += 1
    return [{"source_id": s, "control": c, **table} for (s, c), table in counts.items()]


def summarize(rows, control):
    group = [r for r in rows if r["control"] == control]
    if not group or len({r["source_id"] for r in group}) != len(group):
        raise ValueError("Expected one contingency table per source cluster for " + control)
    counts = np.array([[int(r[k]) for k in FIELDS] for r in group])
    sizes = counts.sum(axis=1)
    if (counts < 0).any() or (sizes == 0).any():
        raise ValueError("Invalid contingency table")
    delta = counts[:, 1] - counts[:, 2]
    both, corrected, regressed, neither = map(int, counts.sum(axis=0))
    n = int(sizes.sum())
    draws = np.random.default_rng(20261102).integers(len(group), size=(2000, len(group)))
    sample = 100 * delta[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    equal = 100 * (delta / sizes)[draws].mean(axis=1)
    return {
        "control": control, "n": n, "source_clusters": len(group),
        "primary_correct": both + corrected, "control_correct": both + regressed,
        "both_correct": both, "corrected": corrected, "regressed": regressed, "both_wrong": neither,
        "net_cases": corrected - regressed, "paired_net_pp": 100 * (corrected - regressed) / n,
        "cluster_ci95_pp": np.quantile(sample, [.025, .975]).tolist(),
        "source_equal_net_pp": 100 * float((delta / sizes).mean()),
        "source_equal_ci95_pp": np.quantile(equal, [.025, .975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-counts", type=Path)
    inputs.add_argument("--cases", type=Path)
    parser.add_argument("--predictions", nargs="+", type=Path)
    parser.add_argument("--control", default="ordinary")
    args = parser.parse_args()
    if args.source_counts:
        if args.predictions:
            parser.error("use --predictions with --cases")
        with args.source_counts.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        if not args.predictions:
            parser.error("--cases requires --predictions")
        rows = count_by_source(args.cases, args.predictions, [args.control])
    print(json.dumps(summarize(rows, args.control), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
