#!/usr/bin/env python3
"""Paired, guideline-clustered statistics for pre-scored binary outcomes.

Input JSONL must contain one row per case with a stable guideline cluster ID,
baseline_correct and method_correct booleans, and optional status fields.
Failures must be encoded as False for correctness and retained in the input.
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path, group_key="guideline_id"):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no} must be an object")
            if group_key not in row:
                raise KeyError(f"line {line_no} missing cluster key {group_key!r}")
            for key in ("baseline_correct", "method_correct"):
                if not isinstance(row.get(key), bool):
                    raise ValueError(f"line {line_no}: {key} must be bool; score failures as false")
            rows.append(row)
    if not rows:
        raise ValueError("no rows")
    return rows


def summarize(rows, bootstrap_draws=10000, seed=20261029, group_key="guideline_id"):
    grouped = defaultdict(list)
    corrected = regressed = 0
    for row in rows:
        b, m = row["baseline_correct"], row["method_correct"]
        grouped[str(row[group_key])].append(int(m) - int(b))
        corrected += int((not b) and m)
        regressed += int(b and (not m))
    cluster_means = {g: sum(ds) / len(ds) for g, ds in grouped.items()}
    guide_macro_delta = sum(cluster_means.values()) / len(cluster_means)
    micro_delta = sum(sum(ds) for ds in grouped.values()) / len(rows)
    rng = random.Random(seed)
    groups = sorted(cluster_means)
    boot = []
    for _ in range(bootstrap_draws):
        sampled = [rng.choice(groups) for _ in groups]
        boot.append(sum(cluster_means[g] for g in sampled) / len(sampled))
    boot.sort()
    lo = boot[int(0.025 * (len(boot) - 1))]
    hi = boot[int(0.975 * (len(boot) - 1))]
    return {
        "n_cases": len(rows),
        "n_guideline_clusters": len(grouped),
        "baseline_accuracy_micro": sum(bool(r["baseline_correct"]) for r in rows) / len(rows),
        "method_accuracy_micro": sum(bool(r["method_correct"]) for r in rows) / len(rows),
        "corrected": corrected,
        "regressed": regressed,
        "net_corrected_minus_regressed": corrected - regressed,
        "micro_delta": micro_delta,
        "guideline_macro_delta": guide_macro_delta,
        "guideline_cluster_bootstrap_95ci_macro_delta": [lo, hi],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "failure_rows": sum(bool(r.get("technical_failure", False)) for r in rows),
        "invalid_output_rows": sum(bool(r.get("invalid_output", False)) for r in rows),
        "coverage_rows": sum(bool(r.get("intervention_applied", False)) for r in rows),
        "note": "All input rows are retained. For the primary analysis, encode failed/invalid predictions as incorrect rather than dropping them."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="paired scored JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-key", default="guideline_id")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    rows = load_rows(args.input, args.group_key)
    result = summarize(rows, args.bootstrap_draws, args.seed, args.group_key)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
