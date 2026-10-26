#!/usr/bin/env python3
"""Frozen MedPIC exact-option-set scoring. Missing/invalid predictions are incorrect.

Prediction contract (JSONL): sample_id, method, seed, prediction (list or strict option string).
The gold file is read only by this evaluator after config_lock.frozen=true.
No question/answer text or gold set is copied to output.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from paired_statistics import compare_registered_family


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option_set(value, allowed):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        elif re.fullmatch(r"[A-Z](?:\s*[,;\s]\s*[A-Z])*", value):
            value = re.findall(r"[A-Z]", value)
        else:
            return None
    if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
        return None
    result = set(value)
    if not result <= set(allowed):
        return None
    return result


def score_questions(gold_rows, predictions, methods, seeds):
    gold = {str(row["instance_id"]): row for row in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError("duplicate official instance_id")
    methods, seeds = list(methods), list(map(int, seeds))
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate method or seed declarations")
    lookup = {}
    for row in predictions:
        key = (str(row["sample_id"]), row["method"], int(row["seed"]))
        if key[0] not in gold or key[1] not in methods or key[2] not in seeds:
            raise ValueError("unexpected sample, method or seed; no silent exclusions")
        if key in lookup:
            raise ValueError("duplicate prediction key")
        lookup[key] = row
    scores, aggregate = [], []
    for method in methods:
        for seed in seeds:
            grouped = defaultdict(list)
            invalid, missing = 0, 0
            for sample_id, row in gold.items():
                expected = option_set(row["answer"], row["options"])
                if expected is None:
                    raise ValueError("invalid official option-set label; preserve data and report")
                predrow = lookup.get((sample_id, method, seed))
                pred = option_set(predrow.get("prediction"), row["options"]) if predrow else None
                missing += int(predrow is None)
                invalid += int(pred is None)
                correct = int(pred is not None and pred == expected)
                family = row.get("benchmark_task_family", "missing")
                scores.append({"sample_id": sample_id, "method": method, "seed": seed, "accuracy": correct,
                               "parse_valid": pred is not None, "prediction_missing": predrow is None,
                               "task_family": family})
                grouped["all"].append(correct)
                grouped[family].append(correct)
            aggregate.append({"method": method, "seed": seed, "denominator": len(gold),
                              "accuracy": sum(grouped["all"]) / len(gold),
                              "groups": {k: {"n":len(v), "correct":sum(v), "accuracy":sum(v)/len(v)} for k,v in grouped.items()},
                              "parse_failures_including_missing": invalid, "missing_predictions": missing,
                              "run_coverage_complete": missing == 0})
    return scores, aggregate


def validated_pairs(pair_manifest, ids):
    if not pair_manifest or pair_manifest.get("status") != "complete_verified":
        return None
    pairs = pair_manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 89:
        raise ValueError("official pair mapping must contain all 89 links")
    canonical = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or any(x not in ids for x in pair):
            raise ValueError("invalid official pair link")
        canonical.add(tuple(sorted(pair)))
    if len(canonical) != len(pairs):
        raise ValueError("duplicate pair link")
    return sorted(canonical)


def merge_linked_clusters(clusters, pairs):
    """Never split linked patients across resampling units; source clusters stay intact."""
    if any(v in (None, "") for v in clusters.values()):
        raise ValueError("empty cluster identifiers")
    parent = {str(c):str(c) for c in clusters.values()}
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for a, b in pairs or []:
        if a not in clusters or b not in clusters:
            continue
        x, y = find(str(clusters[a])), find(str(clusters[b]))
        parent[max(x,y)] = min(x,y)
    return {sid:find(str(cid)) for sid,cid in clusters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--clusters", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.config_lock.read_text())
    if lock.get("frozen") is not True or not lock.get("frozen_at"):
        raise ValueError("Formal external evaluation requires frozen method/configuration and timestamp")
    methods, seeds = lock["methods"], lock["seeds"]
    if sorted(seeds) != [0,1,2]:
        raise ValueError("Formal confirmation reports all prespecified seeds 0/1/2")
    data = json.loads(args.gold.read_text())
    if len(data) != 467:
        raise ValueError("Full external MedPIC evaluation requires all 467 official questions")
    preds = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    scored, metrics = score_questions(data, preds, methods, seeds)
    pairs = validated_pairs(json.loads(args.pairs.read_text()) if args.pairs else None, {str(r["instance_id"]) for r in data})
    clusters = json.loads(args.clusters.read_text()) if args.clusters else {}
    clusters = merge_linked_clusters(clusters, pairs)
    comparisons = lock["primary_comparisons"]
    comparisons_result = compare_registered_family(scored, comparisons, clusters, seeds)
    if not all(m["run_coverage_complete"] for m in metrics):
        for comparison in comparisons_result:
            comparison["status"] = "blocked"
            comparison["reason"] = "Incomplete prediction coverage; missing outputs counted wrong, inferential claim withheld"
    pair_metrics = []
    by_key = {(r["sample_id"],r["method"],r["seed"]):r["accuracy"] for r in scored}
    if pairs:
        for method in methods:
            for seed in seeds:
                correct = sum(by_key[(a,method,seed)] and by_key[(b,method,seed)] for a,b in pairs)
                pair_metrics.append({"method":method,"seed":seed,"n_pairs":len(pairs),"correct_pairs":correct,"pair_accuracy":correct/len(pairs)})
    mean_metrics = [{"method":method,"accuracy_mean_over_seeds":sum(m["accuracy"] for m in metrics if m["method"]==method)/len(seeds)} for method in methods]
    output = {"schema_version":"aster.medpic_evaluation.v1", "created_at":datetime.now(timezone.utc).isoformat(),
              "status":"complete_verified" if all(m["run_coverage_complete"] for m in metrics) else "blocked",
              "source_hashes":{"gold":sha256(args.gold),"predictions":sha256(args.predictions),"config_lock":sha256(args.config_lock),
                               "clusters":sha256(args.clusters) if args.clusters else None,"pairs":sha256(args.pairs) if args.pairs else None,
                               "evaluator":sha256(Path(__file__)),"statistics":sha256(Path(__file__).with_name("paired_statistics.py"))},
              "metrics":metrics,"seed_means":mean_metrics,"primary_comparisons":comparisons_result,
              "pair_status":"complete_verified" if pairs else "blocked",
              "pair_reason":None if pairs else "Official 89-pair mapping unavailable; no guessed links",
              "pair_metrics":pair_metrics,"scored_rows":scored,
              "inference_performed_by_evaluator":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"status":output["status"],"pair_status":output["pair_status"]}))


if __name__ == "__main__":
    main()
