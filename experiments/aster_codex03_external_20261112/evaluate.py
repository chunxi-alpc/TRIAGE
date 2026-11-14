"""Evaluate completed external-method scores with the existing MEGA-CDP protocol.

This module runs on CPU. Raw references are read from historical score caches;
no model is loaded or rerun. Development and frozen evaluation outputs are kept
in separate directories. --validate-raw writes a fixture only under validation/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from experiments.aster_cr_retrain_paired_v2_20261111.common import load_records
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import (
    SourceBootstrap, observations, pair_statistics, path_predictions, uid,
)
from .data import RAW_SOURCE_OUT, load_bundle

COST_FIELDS = (
    "parameter_count", "trainable_parameter_count", "stored_parameter_count",
    "training_wall_seconds", "training_gpu_hours", "training_peak_memory_gib",
    "inference_wall_seconds", "inference_gpu_hours", "inference_forward_count",
    "inference_peak_memory_gib", "inference_seconds_per_input",
)
TABLE_METRICS = {
    "ap_rms": "ap_score_change_rms",
    "ap_vr": "ap_vr",
    "cf_er": "cf_er",
    "cf_correction_rms": "cf_correction_rms",
    "cf_both_correct_accuracy": "cf_paired_accuracy",
    "natural_path_accuracy": "natural_path_accuracy",
    "interference_path_accuracy": "interference_path_accuracy",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, records):
    records = list(records)
    keys = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in records)


def read_scores(path, needed, allowed=None, validate_raw=False):
    allowed = needed if allowed is None else allowed
    scores = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not validate_raw and row.get("execution_kind") == "validation_fixture":
                raise ValueError("Validation scores cannot be exported as actual method results")
            key = uid(row)
            if key not in allowed:
                continue
            if row.get("status", "ok") != "ok":
                raise ValueError(f"Non-success prediction at line {line_number}: {key}")
            value = float(row["ell"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite prediction: {key}")
            if key in scores and scores[key] != value:
                raise ValueError(f"Conflicting duplicate prediction: {key}")
            scores[key] = value
    missing = needed - scores.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} required inputs; first: {sorted(missing)[0]}")
    return scores


def raw_scores(needed):
    records = load_records(RAW_SOURCE_OUT)
    missing = needed - records.keys()
    if missing:
        raise ValueError(f"Historical raw cache lacks {len(missing)} inputs; first: {sorted(missing)[0]}")
    scores = {key: float(records[key]["ell"]) for key in needed}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Historical raw cache contains a non-finite score")
    return scores


def evaluate(method, scores_path, split, output, metadata=None, validate_raw=False):
    if split not in ("dev", "test"):
        raise ValueError("Evaluation split must be dev or test")
    if not method or Path(method).name != method or method in (".", "..", "raw"):
        raise ValueError("Use a method name without path components; raw is reserved")
    supplied = dict(metadata or {})
    metadata = {}
    if scores_path is not None:
        sidecar = Path(scores_path).with_suffix(".metadata.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not validate_raw and (metadata.get("execution_kind") == "validation_fixture"
                             or supplied.get("execution_kind") == "validation_fixture"
                             or "fixture" in method.lower()):
        raise ValueError("Validation fixtures require --validate-raw and cannot become method results")
    metadata.update(supplied)
    bundle = load_bundle(split)
    views, pairs = bundle["views"], bundle["pairs"]
    cases, path_views = bundle["cases"], bundle["path_views"]
    all_views = {uid(v): v for v in [*views, *path_views]}
    available = set(all_views)
    needed = {uid(v) for v in views}
    if not needed or not cases or not pairs:
        raise ValueError("Expected nonempty complete MEGA-CDP views, pairs and paths")
    for pair in pairs:
        if pair["left"] not in needed or pair["right"] not in needed:
            raise ValueError("Pair endpoint is missing from the frozen bundle")
    raw = raw_scores(available)
    adjusted = raw.copy() if validate_raw and scores_path is None else read_scores(scores_path, needed, available, validate_raw)
    scores = {"raw": raw, method: adjusted}
    decisions = {name: {key: int(value > 0) for key, value in values.items()}
                 for name, values in scores.items()}
    pair_records = [row for row in observations(views, pairs, scores, decisions)
                    if row["method"] == method]
    all_paths = path_predictions(cases, path_views, scores)
    raw_paths = {(r["case_id"], r["variant"]): r for r in all_paths if r["method"] == "raw"}
    paths = [row for row in all_paths if row["method"] == method]
    # Every frozen case must have one natural and three interference paths.
    counts_by_case = Counter((r["case_id"], r["interference"]) for r in paths)
    for case in cases:
        cid = case["case_id"]
        if counts_by_case[cid, False] != 1 or counts_by_case[cid, True] != 3:
            raise ValueError(f"Incomplete natural/interference path variants for {cid}")
    for row in paths:
        if row["error"]:
            raise ValueError(f"Path execution failed: {row['case_id']} / {row['variant']}: {row['error']}")
        reference = raw_paths[row["case_id"], row["variant"]]
        if reference["error"]:
            raise ValueError("Historical raw path execution failed")
        row.update(raw_correct=reference["correct"], raw_path=reference["path"],
                   raw_leaf=reference["leaf"], raw_first_error=reference["first_error"],
                   corrected=bool(row["correct"] and not reference["correct"]),
                   new_error=bool(reference["correct"] and not row["correct"]))
    sources = [r["source_id"] for r in pair_records] + [r["source_id"] for r in paths]
    boot = SourceBootstrap(sources)
    metrics, counts, per_source = pair_statistics(pair_records, boot)
    # The historical helper also computes an ASTER-specific correction budget.
    # External methods are not subject to that budget; do not report it here.
    metrics = {key: value for key, value in metrics.items() if "over_budget" not in key}
    counts = {key: value for key, value in counts.items() if "over_budget" not in key}
    for row in pair_records:
        row.pop("evidence_budget", None)
        row.pop("over_budget", None)
    for row in per_source:
        row.pop("over_budget_count", None)
    path_summary = []
    for interference in (False, True):
        records = [r for r in paths if r["interference"] == interference]
        prefix = "interference_path" if interference else "natural_path"
        metrics[prefix + "_accuracy"] = boot.mean([r["correct"] for r in records],
                                                   [r["source_id"] for r in records])
        counts.update({prefix + "_count": len(records),
                       prefix + "_correct": sum(r["correct"] for r in records),
                       prefix + "_raw_correct": sum(r["raw_correct"] for r in records),
                       prefix + "_corrected": sum(r["corrected"] for r in records),
                       prefix + "_new_errors": sum(r["new_error"] for r in records)})
    for variant in sorted({r["variant"] for r in paths}):
        records = [r for r in paths if r["variant"] == variant]
        path_summary.append(dict(method=method, split=split, variant=variant, count=len(records),
                                 correct=sum(r["correct"] for r in records),
                                 accuracy=sum(r["correct"] for r in records) / len(records),
                                 corrected=sum(r["corrected"] for r in records),
                                 new_errors=sum(r["new_error"] for r in records)))
    predictions = []
    for key, view in all_views.items():
        if key not in adjusted:
            continue
        label = view.get("label")
        gold = None if label is None else int(label)
        correct = None if gold is None else decisions[method][key] == gold
        raw_correct = None if gold is None else decisions["raw"][key] == gold
        predictions.append(dict(uid=key, method=method, split=split, source_id=view["source_id"],
                                case_id=view.get("case_id"), gid=view.get("gid"),
                                kind=view.get("kind"), variant=view.get("variant"), label=gold,
                                ell=adjusted[key], raw_ell=raw[key], prediction=decisions[method][key],
                                raw_prediction=decisions["raw"][key], correct=correct,
                                raw_correct=raw_correct,
                                corrected=None if gold is None else bool(correct and not raw_correct),
                                new_error=None if gold is None else bool(raw_correct and not correct)))
    intervals = {key: boot.summary(values) for key, values in metrics.items()}
    estimates = {key: value["estimate"] for key, value in intervals.items()}
    costs = {key: metadata.get(key, metadata.get("costs", {}).get(key)) for key in COST_FIELDS}
    for key, value in costs.items():
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Cost field must be a nonnegative finite number or null: {key}")
    if costs["inference_seconds_per_input"] is None and costs["inference_wall_seconds"] is not None:
        costs["inference_seconds_per_input"] = costs["inference_wall_seconds"] / len(adjusted)
    table_counts = {key: value for key, value in counts.items()
                    if key in ("ap_pairs", "ap_sources", "cf_pairs", "cf_sources",
                               "cf_paired_labeled_count", "cf_paired_unlabeled_count")
                    or key.startswith(("natural_path_", "interference_path_"))}
    table_counts["cf_both_correct_count"] = sum(row["paired_correct"] is True
                                               for row in pair_records if row["group"] == "CF")
    table = dict(method=method, split=split, result_kind="validation_fixture" if validate_raw else "actual_experiment",
                 complete=not validate_raw, model="Qwen3-8B", train_data="E1 / Dataset A", train_pairs=2540,
                 **{key: estimates.get(original) for key, original in TABLE_METRICS.items()},
                 **table_counts, **costs)
    result = dict(method=method, split=split, complete=not validate_raw,
                  result_kind=table["result_kind"], evaluation_status="computed",
                  model="Qwen3-8B", train_data="E1 / Dataset A", train_pairs=2540,
                  unique_inputs=len(adjusted), candidate_inputs=len(available), required_pair_inputs=len(needed),
                  unscored_path_inputs=len(available - adjusted.keys()), pair_count=len(pairs), case_count=len(cases),
                  path_count=len(paths), source_count=len(boot.sources),
                  metrics=estimates, counts=counts, costs=costs, intervals=intervals,
                  metadata=metadata, scores_path=None if scores_path is None else str(Path(scores_path).resolve()),
                  raw_cache=str(RAW_SOURCE_OUT), bootstrap_draws=boot.draws,
                  decision_rule="ell > 0", metric_units="accuracy, VR and ER are fractions; RMS is in logit units",
                  test_history="Frozen historical evaluation scope; previously evaluated, not newly unseen",
                  missing_cost_fields=[key for key, value in costs.items() if value is None])
    output = Path(output)
    destination = output / ("validation" if validate_raw else "evaluation") / split / method
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "predictions.jsonl", predictions)
    write_jsonl(destination / "pair_predictions.jsonl", pair_records)
    write_jsonl(destination / "path_predictions.jsonl", paths)
    write_csv(destination / "pair_predictions.csv", pair_records)
    write_csv(destination / "path_predictions.csv", paths)
    write_csv(destination / "source_metrics.csv", per_source)
    write_csv(destination / "path_metrics.csv", path_summary)
    write_csv(destination / "statistics.csv", [dict(method=method, split=split, metric=key, **summary)
                                               for key, summary in intervals.items()])
    write_json(destination / "training_and_implementation.json", metadata)
    write_json(destination / "statistics.json", result)
    filename = "section4_5.fixture.csv" if validate_raw else "section4_5.csv"
    write_csv(destination / filename, [table])
    if not validate_raw:
        # Only complete actual-method files contribute to the split-specific main table.
        rows = []
        for path in sorted((output / "evaluation" / split).glob("*/section4_5.csv")):
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows.extend(r for r in csv.DictReader(f)
                            if r.get("complete") == "True" and r.get("result_kind") == "actual_experiment")
        write_csv(output / "evaluation" / split / "section4_5.csv", rows)
    write_json(destination / "status.json", dict(complete=not validate_raw, result_kind=table["result_kind"],
                                                evaluation_status="computed", split=split, method=method))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--validate-raw", action="store_true")
    args = parser.parse_args()
    if not args.validate_raw and args.scores is None:
        parser.error("--scores is required for actual experiment evaluation")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    result = evaluate(args.method, args.scores, args.split, args.output, metadata, args.validate_raw)
    print(json.dumps({key: result[key] for key in ("method", "split", "result_kind", "unique_inputs", "pair_count", "path_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
