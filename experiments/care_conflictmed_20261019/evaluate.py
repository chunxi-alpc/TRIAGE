#!/usr/bin/env python3
"""Strict paired evaluation: parse failures count wrong; cluster by original pair."""
import argparse
import collections
import json
import math
import random
from pathlib import Path

from prepare import read_jsonl


def mean(values):
    return sum(values) / len(values) if values else None


def bootstrap_by_group(rows, reps=2000, seed=20261019):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(float(r["correct"]))
    if not groups:
        return None
    samples = list(groups.values())
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        drawn = [samples[rng.randrange(len(samples))] for __ in samples]
        values.append(sum(sum(x) for x in drawn) / sum(map(len, drawn)))
    values.sort()
    return [values[int(.025 * reps)], values[min(reps - 1, math.ceil(.975 * reps) - 1)]]


def evaluate(labels, inputs, predictions, require_complete=True):
    gold = {r["id"]: r for r in labels}
    cases = {r["id"]: r for r in inputs}
    if len(gold) != len(labels) or len(cases) != len(inputs):
        raise ValueError("Duplicate gold/case ID")
    if set(gold) != set(cases):
        raise ValueError("Inputs/labels coverage mismatch")
    by_method = collections.defaultdict(dict)
    for p in predictions:
        item_id, method = p.get("id"), p.get("method")
        if item_id not in gold or not isinstance(method, str) or not method:
            raise ValueError("Unknown item or missing method")
        if item_id in by_method[method]:
            raise ValueError(f"Duplicate {method}:{item_id}")
        if p.get("prediction") not in {"A", "B", None, "INVALID"}:
            raise ValueError(f"Invalid prediction encoding for {item_id}")
        by_method[method][item_id] = p
    out = {"status": "EVALUATED_MODEL_PREDICTIONS", "methods": {}}
    for method, predicted in sorted(by_method.items()):
        expected = set(gold)
        if require_complete and set(predicted) != expected:
            raise ValueError(f"{method}: incomplete {len(predicted)}/{len(expected)}; pass --allow-incomplete explicitly")
        joined = []
        for item_id, p in predicted.items():
            g = gold[item_id]
            inp = cases[item_id]
            joined.append({"id": item_id, "group": g["group"], "track": g["track"], "split": g["split"],
                           "source_kind": inp["source_kind"], "correct": p.get("prediction") == g["correct_choice"],
                           "valid": p.get("prediction") in {"A", "B"},
                           "retrieved": inp.get("retrieved_option"), "choice": p.get("prediction"),
                           "gold_choice": g["correct_choice"], "variant": inp["variant"]})
        strata = collections.defaultdict(list)
        for r in joined:
            strata[(r["track"], r["split"], r["source_kind"])].append(r)
        metrics = {}
        for (track, split, source), rows in sorted(strata.items()):
            key = f"{track}/{split}/{source}"
            metrics[key] = {"n": len(rows), "accuracy": mean([r["correct"] for r in rows]),
                            "invalid": sum(not r["valid"] for r in rows),
                            "cluster_95ci": bootstrap_by_group(rows)}
            if source.startswith("excerpt_"):
                excerpt_follow = [r["choice"] == r["retrieved"] for r in rows]
                wrong_excerpts = [r for r in rows if r["retrieved"] != r["gold_choice"]]
                metrics[key]["excerpt_follow_rate"] = mean(excerpt_follow)
                metrics[key]["wrong_excerpt_n"] = len(wrong_excerpts)
                metrics[key]["wrong_excerpt_follow_rate"] = mean([r["choice"] == r["retrieved"] for r in wrong_excerpts])
        # Baseline known subset is an observed behavioral proxy, not mechanistic proof.
        by_group = collections.defaultdict(list)
        for r in joined:
            if r["track"] == "advice_choice" and r["source_kind"] == "none":
                by_group[(r["split"], r["group"])].append(r)
        known = {k for k, rs in by_group.items() if len(rs) == 2 and all(x["correct"] for x in rs)}
        known_by_split = collections.Counter(s for s, _ in known)
        other_counts = collections.Counter(s for s, _ in by_group)
        out["methods"][method] = {"predictions": len(predicted), "coverage": len(predicted) / len(expected),
                                  "metrics": metrics, "closed_book_consistent_groups": dict(known_by_split),
                                  "advice_groups_observed": dict(other_counts)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    a = p.parse_args()
    metrics = evaluate(list(read_jsonl(a.labels)), list(read_jsonl(a.inputs)),
                       list(read_jsonl(a.predictions)), not a.allow_incomplete)
    if a.out.exists():
        raise FileExistsError("Refuse to overwrite analysis")
    a.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"status": metrics["status"], "methods": list(metrics["methods"])}, indent=2))


if __name__ == "__main__":
    main()
