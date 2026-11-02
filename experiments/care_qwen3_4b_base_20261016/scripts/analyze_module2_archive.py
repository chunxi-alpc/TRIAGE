#!/usr/bin/env python3
"""New paired analysis of archived CLINICAL O1 predictions; no model inference."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 261017
REPLICATES = 10000

def cluster_ci(values):
    """Paired family bootstrap, reproducible in JS and Python via LCG32."""
    seed = SEED
    draws = []
    for _ in range(REPLICATES):
        n = d = 0
        for _ in values:
            seed = (1664525 * seed + 1013904223) & 0xffffffff
            a, b = values[math.floor(seed / 4294967296 * len(values))]
            n += a
            d += b
        if d:
            draws.append(n / d)
    draws.sort()
    def quantile(p):
        k = (len(draws) - 1) * p
        i = math.floor(k)
        return draws[i] + (draws[min(i + 1, len(draws) - 1)] - draws[i]) * (k - i)
    return [quantile(.025), quantile(.975)]

def truth(value):
    return value == "True"

def mean(values):
    return sum(values) / len(values)

def gold_margin(row):
    p = json.loads(row["candidate_probabilities"])
    g = "ABCD".index(row["gold_answer"])
    return p[g] - max(v for i, v in enumerate(p) if i != g)

def analyze(root):
    source = root / "results/obs1_clinical/predictions.csv"
    with source.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "test"]
    lookup = {(r["method"], r["case_id"]): r for r in rows}
    assert len(rows) == len(lookup) == 1152
    for fold in range(3):
        info = json.loads((root / f"results/obs1_clinical/fold{fold}/manifest.json").read_text())
        a, b, c = (set(info[k]) for k in ("train_families", "dev_families", "test_families"))
        assert not (a & b or a & c or b & c)
        for r in rows:
            if int(r["outer_fold"]) != fold:
                continue
            assert r["group_id"] in c and r["case_id"] in info["test_case_ids"]
            if r["method"] != "baseline":
                assert float(r["strength"]) == info["selection"][r["method"]]
    metrics, transitions, per_family = [], [], []
    for method in ("baseline", "seka", "pasta"):
        data = [r for r in rows if r["method"] == method]
        pairs = defaultdict(list)
        for r in data:
            pairs[r["pair_id"]].append(r)
        assert len(data) == 384 and len(pairs) == 192
        assert all(len(v) == 2 and {r["subset"] for r in v} == {"app", "non"} for v in pairs.values())
        subsets = [
            ("all", lambda r: True),
            ("eligible", lambda r: truth(r["prior_eligible"])),
            ("stable_agreement", lambda r: truth(r["prior_stable"]) and not truth(r["prior_eligible"])),
        ]
        definitions = [
            ("accuracy", lambda r: True, lambda r: int(r["correct"])),
            ("app_accuracy", lambda r: r["subset"] == "app", lambda r: int(r["correct"])),
            ("non_accuracy", lambda r: r["subset"] == "non", lambda r: int(r["correct"])),
            ("misuse", lambda r: r["subset"] == "non", lambda r: int(r["misapplied"])),
            ("gold_margin", lambda r: True, gold_margin),
        ]
        for subset, include in subsets:
            for metric, select, value in definitions:
                d = [r for r in data if include(r) and select(r)]
                families = list(dict.fromkeys(r["group_id"] for r in d))
                grouped = [[r for r in d if r["group_id"] == g] for g in families]
                delta = lambda r: value(r) - value(lookup["baseline", r["case_id"]])
                metrics.append(dict(
                    method=method, subset=subset, metric=metric, n=len(d), families=len(families),
                    estimate=mean([value(r) for r in d]),
                    ci95=cluster_ci([(sum(value(r) for r in v), len(v)) for v in grouped]),
                    delta=mean([delta(r) for r in d]),
                    delta_ci95=cluster_ci([(sum(delta(r) for r in v), len(v)) for v in grouped])))
        for subset, include in [
            ("all", lambda v: True),
            ("complete_eligible", lambda v: all(truth(r["prior_eligible"]) for r in v)),
        ]:
            ps = [v for v in pairs.values() if include(v)]
            families = list(dict.fromkeys(v[0]["group_id"] for v in ps))
            grouped = [[v for v in ps if v[0]["group_id"] == g] for g in families]
            value = lambda v: int(all(int(r["correct"]) for r in v))
            delta = lambda v: value(v) - int(all(int(lookup["baseline", r["case_id"]]["correct"]) for r in v))
            metrics.append(dict(
                method=method, subset=subset, metric="pair_accuracy", n=len(ps), families=len(families),
                estimate=mean([value(v) for v in ps]),
                ci95=cluster_ci([(sum(value(v) for v in vs), len(vs)) for vs in grouped]),
                delta=mean([delta(v) for v in ps]),
                delta_ci95=cluster_ci([(sum(delta(v) for v in vs), len(vs)) for vs in grouped])))
        for g in dict.fromkeys(r["group_id"] for r in data):
            d = [r for r in data if r["group_id"] == g]
            ps = [v for v in pairs.values() if v[0]["group_id"] == g]
            per_family.append(dict(
                group_id=g, method=method,
                accuracy=mean([int(r["correct"]) for r in d]),
                pair_accuracy=mean([int(all(int(r["correct"]) for r in v)) for v in ps]),
                corrected=sum(int(r["correct"]) and not int(lookup["baseline", r["case_id"]]["correct"]) for r in d),
                harmed=sum(not int(r["correct"]) and int(lookup["baseline", r["case_id"]]["correct"]) for r in d)))
        if method != "baseline":
            for r in data:
                b = lookup["baseline", r["case_id"]]
                if r["prediction"] != b["prediction"]:
                    transitions.append(dict(
                        method=method, case_id=r["case_id"], pair_id=r["pair_id"],
                        group_id=r["group_id"], subset=r["subset"], gold=r["gold_answer"],
                        baseline=b["prediction"], intervention=r["prediction"],
                        corrected=not int(b["correct"]) and bool(int(r["correct"])),
                        harmed=bool(int(b["correct"])) and not int(r["correct"]),
                        misuse_before=int(b["misapplied"]), misuse_after=int(r["misapplied"]),
                        prior_eligible=truth(r["prior_eligible"])))
    return dict(metrics=metrics, transitions=transitions, per_family=per_family)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--verify", action="store_true", help="compare against saved connector-runtime analysis")
    args = p.parse_args()
    result = analyze(args.root)
    out = args.root / "results/module2_followup"
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        saved = json.loads((out / "archived_clinical_pairs.json").read_text())
        def close(a, b):
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
            if isinstance(a, float):
                return abs(a - b) < 1e-10
            return a == b
        assert all(close(result[k], saved[k]) for k in result), "Reproduction mismatch"
    else:
        (out / "python_reproduction.json").write_text(json.dumps(result, indent=2))
    # Optional publication exports from saved results only.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    selected = [r for r in result["metrics"] if r["subset"] == "all" and r["metric"] == "pair_accuracy"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for i, r in enumerate(selected):
        ax.errorbar(i, r["estimate"] * 100, yerr=[[100 * (r["estimate"] - r["ci95"][0])],
            [100 * (r["ci95"][1] - r["estimate"])]], fmt="o", capsize=4, color="#0072B2")
    ax.set(xticks=range(3), xticklabels=["Baseline", "Official SEKA", "Official PASTA"],
           ylabel="Both members correct (%)", ylim=(0, 65),
           title="Archived clinical O1: 192 pairs / 12 held-out families")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"archived_pair_accuracy.{ext}", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
