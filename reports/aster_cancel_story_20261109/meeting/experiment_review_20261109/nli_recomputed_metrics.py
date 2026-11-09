#!/usr/bin/env python3
"""Recompute NLI4CT-2024 metrics from archived binary predictions and official metadata.

This script performs no model inference, fitting, or selection. The archived
predictions are from TRIAGE; the minimal manifest is derived from the official
gold_test.json. Definitions are explicit and do not invoke the official scorer:
- macro F1 on the 500 original statements;
- consistency on all label-preserving relations;
- faithfulness on label-altering relations conditional on the method answering
  the original statement correctly;
- unconditional pair accuracy on label-altering relations.

Example (repository-root paths supplied by the caller):
  python nli_recomputed_metrics.py --predictions-dir PATH --manifest PATH --output PATH
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

LABELS = {"Contradiction": 0, "Entailment": 1}
CACHE_REPOSITORY = "chunxi-alpc/TRIAGE"
CACHE_REVISION = "742e6b32c59fade4ddd2639314785209e85f1006"
CACHE_DIRECTORY = "reports/aster_cancel_story_20261109/meeting/primary_full/nli4ct_all"
CACHE_BLOB_SHA = {
    "path_results_shard0.jsonl.gz": "b91a9d15d17fd55a963b51d259b6052dad6712b2",
    "path_results_shard1.jsonl.gz": "9830a24b81d5e7d221988c5d426e7063b0baa2e7",
    "path_results_shard2.jsonl.gz": "d5b515ba91243728ebdb889e280ff53ea96d6da1",
    "path_results_shard3.jsonl.gz": "3117bec24f1c0c306b5f85eb30fa611d621e2cc2",
}
EXPECTED_FULL_CORRECT = {
    "raw": 4073, "fit_bias_only": 4033, "old_linear": 4031,
    "optimized_balanced_fit_calibration": 4048,
    "optimized_strict_fit_calibration": 4042,
    "trace_only_balanced_fit_calibration": 4040,
}
EXPECTED_METHODS = set(EXPECTED_FULL_CORRECT)

def fraction(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator if denominator else None,
    }

def macro_f1(gold, predicted):
    per_class = {}
    for label in (0, 1):
        tp = sum(y == label and p == label for y, p in zip(gold, predicted))
        fp = sum(y != label and p == label for y, p in zip(gold, predicted))
        fn = sum(y == label and p != label for y, p in zip(gold, predicted))
        support = sum(y == label for y in gold)
        denom = 2 * tp + fp + fn
        per_class[str(label)] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "support": support, "f1": 2 * tp / denom if denom else None,
        }
    assert all(v["support"] > 0 for v in per_class.values()), "Missing gold class"
    assert all(v["f1"] is not None for v in per_class.values())
    return {"value": sum(v["f1"] for v in per_class.values()) / 2,
            "per_class": per_class}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    original_ids = sorted(k for k, v in records.items() if v["causal_type"] is None)
    preserving = [(v["causal_type"][1], k) for k, v in records.items()
                  if v["causal_type"] and v["causal_type"][0] == "Preserving"]
    altering = [(v["causal_type"][1], k) for k, v in records.items()
                if v["causal_type"] and v["causal_type"][0] == "Altering"]
    assert (len(records), len(original_ids), len(preserving), len(altering)) == (5500, 500, 4136, 864)
    original_set = set(original_ids)
    gold = {k: LABELS[v["label"]] for k, v in records.items()}
    assert all(a in original_set for a, b in preserving + altering)
    assert all(gold[a] == gold[b] for a, b in preserving)
    assert all(gold[a] != gold[b] for a, b in altering)
    predictions = defaultdict(dict)
    cached_status = Counter()
    input_files = []
    full_record_count = 0
    for name, expected_sha in CACHE_BLOB_SHA.items():
        path = args.predictions_dir / name
        payload = path.read_bytes()
        blob_sha = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\x00" + payload).hexdigest()
        assert blob_sha == expected_sha, f"Unexpected archive bytes: {name}"
        input_files.append({"name": name, "blob_sha": blob_sha, "bytes": len(payload)})
        for line in gzip.decompress(payload).decode().splitlines():
            row = json.loads(line)
            full_record_count += 1
            if row["dataset_split"] != "test":
                continue
            cached_status[row["status"]] += 1
            assert row["status"] == "ok" and row["error"] is None
            assert row["dataset"] == "NLI4CT-2024"
            case_id = row["case_id"].removeprefix("NLI4CT-2024:test:")
            assert case_id in records
            assert len(row["predicted_path"]) == len(row["gold_path"]) == 1
            predicted = row["predicted_path"][0]["label"]
            cached_gold = row["gold_path"][0]["label"]
            assert predicted in (0, 1) and cached_gold == gold[case_id]
            assert bool(row["correct"]) == (predicted == cached_gold)
            method = row["method"]
            assert case_id not in predictions[method], f"Duplicate {method}/{case_id}"
            predictions[method][case_id] = predicted
    assert set(predictions) == EXPECTED_METHODS
    assert full_record_count == 6 * 9542
    results = {}
    for method, pred in sorted(predictions.items()):
        assert set(pred) == set(records)
        all_correct = sum(pred[k] == gold[k] for k in records)
        assert all_correct == EXPECTED_FULL_CORRECT[method], f"Mismatch with archived full metric: {method}"
        y = [gold[k] for k in original_ids]
        p = [pred[k] for k in original_ids]
        eligible = [(a, b) for a, b in altering if pred[a] == gold[a]]
        faithful = sum(pred[b] != pred[a] for a, b in eligible)
        unconditional_flip = sum(pred[b] != pred[a] for a, b in altering)
        pair_correct = sum(pred[a] == gold[a] and pred[b] == gold[b] for a, b in altering)
        assert faithful == pair_correct, "Binary label-altering identity failed"
        results[method] = {
            "full_test_accuracy": fraction(all_correct, len(records)),
            "original_accuracy": fraction(sum(a == b for a, b in zip(y, p)), len(y)),
            "original_macro_f1": macro_f1(y, p),
            "preserving_prediction_consistency": fraction(sum(pred[a] == pred[b] for a, b in preserving), len(preserving)),
            "altering_faithfulness_given_original_correct": fraction(faithful, len(eligible)),
            "altering_unconditional_prediction_change": fraction(unconditional_flip, len(altering)),
            "altering_pair_accuracy": fraction(pair_correct, len(altering)),
            "eligible_originals_for_conditional_faithfulness": len({a for a, b in eligible}),
        }
    output = {
        "status": "recomputed_from_archived_predictions_no_inference_no_fitting",
        "official_metadata": {k: v for k, v in manifest.items() if k != "records"},
        "archive": {"repository": CACHE_REPOSITORY, "revision": CACHE_REVISION,
                    "directory": CACHE_DIRECTORY, "files": input_files},
        "coverage": {"all_cached_records": full_record_count, "test_records_per_method": len(records),
                     "methods": len(predictions), "original_statements": len(original_ids),
                     "preserving_relations": len(preserving), "altering_relations": len(altering),
                     "test_status_counts": dict(cached_status)},
        "definitions": {
            "original_macro_f1": "Unweighted mean of F1 for Contradiction and Entailment on the 500 original statements.",
            "consistency": "Fraction of 4136 label-preserving relations with equal predictions on original and variant; not conditioned on correctness.",
            "faithfulness": "Among label-altering relations whose original is correctly answered by this method, fraction with a changed prediction; denominator is method-specific.",
            "pair_accuracy": "Fraction of all 864 label-altering relations with both endpoints correct.",
            "qualification": "These explicitly defined metrics are recomputed from archived predictions; not a claim that the current official evaluate.py was executed.",
            "units": "All stored metric values are fractions; multiply by 100 for percentages.",
            "uncertainty": "Point estimates and integer counts only; no newly estimated confidence intervals or claims of significance.",
        },
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for method, m in results.items():
        print(method, "macro_F1", round(100*m["original_macro_f1"]["value"], 4),
              "consistency", round(100*m["preserving_prediction_consistency"]["value"],4),
              "conditional_faithfulness", round(100*m["altering_faithfulness_given_original_correct"]["value"],4),
              "eligible", m["altering_faithfulness_given_original_correct"]["denominator"],
              "pair_correct",m["altering_pair_accuracy"]["numerator"])

if __name__ == "__main__":
    main()

