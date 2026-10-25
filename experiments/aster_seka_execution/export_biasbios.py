#!/usr/bin/env python3
"""Server CPU only: verify native BiasBios top1/topk and anonymize every row.

No inference, no deduplication by names, no prompt or generation text exported.
Published source: SEKA aab9cfed80dc012fb9e83f8bdefb8675c33f40e7.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashed(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def text_hash(value):
    require(isinstance(value, str), "Expected text field for hash-only export")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_id(source_sha, start, end, absolute_row, original_id):
    return hashed(["SEKA-BiasBios-v1", source_sha, f"{start}:{end}", absolute_row, original_id])


def read_slice(path, start, end):
    rows, index = [], 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if index >= end:
                break
            if index >= start:
                rows.append(json.loads(line))
            index += 1
    require(len(rows) == end - start, "Requested processed dataset slice is incomplete")
    return rows


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def export_rows(raw, source, source_sha, start=5000, end=10000,
                marker_start="**", marker_end="**"):
    samples, saved_metrics = raw["samples"], raw["metrics"]
    require(len(samples) == len(source) == end - start, "Result/source row count mismatch")
    require(isinstance(saved_metrics["k"], int) and saved_metrics["k"] > 0, "Invalid native top-k field")
    k = saved_metrics["k"]
    labels = sorted({row["target_mediated"].lower() for row in source})
    label_set = set(labels)
    require(k <= len(labels), "Native k exceeds candidate occupation count")
    outputs, original_ids = [], []
    for offset, (sample, original) in enumerate(zip(samples, source)):
        require(sample["id"] == original["id"], "Result IDs do not match standard slice order")
        original_ids.append(str(original["id"]))
        target, predictions = sample["target"], sample["predictions"]
        require(target == original["target_mediated"].lower(), "Result target differs from source")
        require(isinstance(predictions, list) and len(predictions) == k
                and len(set(predictions)) == k and set(predictions).issubset(label_set),
                "Invalid native top-k occupation predictions")
        logps = [float(value) for value in sample["logp_predictions"]]
        logp_target = float(sample["logp_target"])
        require(len(logps) == k and all(math.isfinite(x) and x <= 1e-6 for x in logps)
                and math.isfinite(logp_target) and logp_target <= 1e-6, "Invalid saved log probabilities")
        require(all(a >= b - 1e-6 for a, b in zip(logps, logps[1:])), "Top-k log probabilities are not sorted")
        if target in predictions:
            require(math.isclose(logp_target, logps[predictions.index(target)], abs_tol=1e-6, rel_tol=1e-6),
                    "Target probability differs from its top-k entry")
        # Exact upstream prompt_in_context_from_sample with context_suffix='\n\n'.
        plain = original["context"].rstrip(". ") + ".\n\n" + original["prompt"]
        marked = plain.replace(original["attribute"], marker_start + original["attribute"] + marker_end)
        require(sample["prompt"] in (plain, marked),
                "Saved prompt does not match full-context official preprocessing")
        output = {"sample_id": row_id(source_sha, start, end, start + offset, original["id"]),
                  "absolute_processed_row_index": start + offset,
                  "source_name_id_sha256": hashed(["BiasBios-source-name", original["id"]]),
                  "target": target, "predictions": predictions, "k": k,
                  "top1_correct": predictions[0] == target, "topk_correct": target in predictions,
                  "logp_predictions": logps, "logp_target": logp_target,
                  "prompt_sha256": text_hash(sample["prompt"]),
                  "generation_sha256": text_hash(sample["generation"]),
                  "generation_is_empty": sample["generation"] == "",
                  "generation_token_count": (len(sample["generation_token_ids"])
                                             if isinstance(sample.get("generation_token_ids"), list) else None),
                  "fluency_saved": finite_or_none(sample["fluency"]),
                  "consistency_saved": finite_or_none(sample["consistency"])}
        outputs.append(output)
    counts = Counter(original_ids)
    recomputed = {"top1_accuracy": sum(row["top1_correct"] for row in outputs) / len(outputs),
                  "topk_accuracy": sum(row["topk_correct"] for row in outputs) / len(outputs), "k": k}
    metadata = {"source_rows": len(source), "unique_original_name_ids": len(counts),
                "repeated_name_id_groups": sum(count > 1 for count in counts.values()),
                "duplicate_name_id_extra_rows": sum(count - 1 for count in counts.values()),
                "unique_anonymous_sample_ids": len({row["sample_id"] for row in outputs}),
                "candidate_occupation_count": len(labels), "recomputed_metrics": recomputed,
                "nonfinite_saved_fluency_rows": sum(row["fluency_saved"] is None for row in outputs),
                "nonfinite_saved_consistency_rows": sum(row["consistency_saved"] is None for row in outputs)}
    require(metadata["unique_anonymous_sample_ids"] == len(outputs), "Anonymous sample IDs are not unique")
    return outputs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Official result.json")
    parser.add_argument("--processed-data", type=Path, required=True, help="Author-reformatted biasbios JSONL")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=5000)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--marker-start", default="**")
    parser.add_argument("--marker-end", default="**")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "benchmark": "BiasBios", "new_gpu_forwards": 0,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "raw_text_published": False,
              "scope": "standard implementation validation; not clinical independent confirmation",
              "independent_recalculation": "Native top1/topk from saved occupational predictions and targets, aligned to original ordered data rows",
              "limits": "Does not reconstruct full-vocabulary logits, regenerate outputs, or recompute textual fluency/TF-IDF consistency; those saved values are explicitly labeled",
              "generation_token_count_policy": "null without original token IDs; decoded text is not retokenized",
              "deduplicated": False, "source_name_id_is_patient_identity": False,
              "native_definition_source": "https://github.com/waylonli/SEKA/blob/aab9cfed80dc012fb9e83f8bdefb8675c33f40e7/benchmarks/biasbios/evaluate.py#L365"}
    code = 0
    try:
        require(0 <= args.start < args.end, "Invalid source slice")
        source_sha = digest(args.processed_data)
        raw = json.loads(args.raw.read_text(encoding="utf-8"))
        run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        require(run.get("process_state") == "exited" and run.get("exit_code") == 0,
                "Official inference run did not exit successfully")
        require(source_sha in run.get("input_files_sha256", {}).values(),
                "Processed data hash is absent from the inference run input manifest")
        source = read_slice(args.processed_data, args.start, args.end)
        rows, metadata = export_rows(raw, source, source_sha, args.start, args.end,
                                     args.marker_start, args.marker_end)
        path = args.output_dir / "predictions.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        report.update(metadata, raw_file_sha256=digest(args.raw), processed_data_sha256=source_sha,
                      run_manifest_sha256=digest(args.run_manifest), predictions_sha256=digest(path),
                      split=f"train[{args.start}:{args.end}]", saved_metrics={
                          "top1_accuracy": finite_or_none(raw["metrics"]["top1_accuracy"]),
                          "topk_accuracy": finite_or_none(raw["metrics"]["topk_accuracy"]),
                          "k": raw["metrics"]["k"]},
                      source_commit=run.get("source_commit"),
                      model_files_sha256=run.get("model_files_sha256", {}),
                      costs={key: run.get(key) for key in ("forward_attempts", "forward_completed", "elapsed_seconds",
                             "gpu_lock_wall_seconds", "peak_allocated_gib", "peak_reserved_gib")})
        for key in ("top1_accuracy", "topk_accuracy"):
            require(math.isclose(metadata["recomputed_metrics"][key], float(raw["metrics"][key]),
                                 rel_tol=1e-10, abs_tol=1e-12), "Saved native metric differs: " + key)
        report["status"] = "complete_verified"
    except Exception as exc:
        code = 1
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=digest(Path(__file__)))
        (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "new_gpu_forwards": 0, "raw_text_published": False}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
