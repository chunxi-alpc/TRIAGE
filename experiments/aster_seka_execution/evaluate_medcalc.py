#!/usr/bin/env python3
"""CPU native correctness evaluation, separate from model input generation."""
import argparse
import json
from pathlib import Path

from medcalc_native import OfficialMedCalc, digest, grouped_point_metrics
from medcalc_data import read_rows, sha_file


def evaluate(rows, predictions, official):
    expected = {str(row["Row Number"]): row for row in rows}
    lookup = {}
    for prediction in predictions:
        key = str(prediction["sample_id"])
        if key in lookup or key not in expected:
            raise ValueError("Duplicate or unexpected prediction ID")
        lookup[key] = prediction
    scored = []
    for key, row in expected.items():
        pred = lookup.get(key)
        if pred is None or pred.get("generation", {}).get("status") != "ok":
            result = {"correct": 0, "status": "missing_or_failed_prediction"}
        else:
            result = official.score(row, pred["generation"]["text"])
        scored.append({"sample_id": key, "category": row["Category"],
                       "source_cluster_sha256": digest([row.get("Note Type"), row.get("Note ID")]),
                       "note_text_sha256": digest(row["Patient Note"]), **result})
    return {"scored_rows": scored, "metrics": grouped_point_metrics(scored),
            "denominator": len(rows), "missing_predictions": len(expected) - len(lookup),
            "official_correctness_functions": official.source_hashes,
            "official_stale_1047_standard_error_used": False,
            "cluster_intervals": "not_computed_here; use paired source/note groups across frozen methods/seeds"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--predictions-private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.config_lock.read_text())
    rows, data_sha = read_rows(args.official_root, args.split, lock, evaluation=True)
    predictions = [json.loads(line) for line in args.predictions_private.read_text().splitlines() if line.strip()]
    result = evaluate(rows, predictions, OfficialMedCalc(args.official_root))
    result.update(split=args.split, data_sha256=data_sha, predictions_sha256=sha_file(args.predictions_private),
                  config_lock_sha256=sha_file(args.config_lock))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": result["denominator"], "missing": result["missing_predictions"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
