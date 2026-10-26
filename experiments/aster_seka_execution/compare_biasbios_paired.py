#!/usr/bin/env python3
"""Server CPU: compare two verified BiasBios exports, preserving all 5000 rows.

Only exact processed-biography duplicates define optional bootstrap clusters.
Names are not patient IDs. Input matching compares prefill, never decode suffixes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import time

from counterfact_entity_ci import bootstrap_effects, digest, hashed, iter_json_objects, jsonl, require
from export_biasbios import row_id


def load_verified(folder):
    folder = Path(folder)
    audit_path, prediction_path = folder / "audit.json", folder / "predictions.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("status") == "complete_verified" and audit.get("benchmark") == "BiasBios",
            "Requires a verified BiasBios export")
    require(audit.get("split") == "train[5000:10000]" and audit.get("source_rows") == 5000,
            "Requires the complete official 5000-case slice")
    require(digest(prediction_path) == audit["predictions_sha256"], "Export fingerprint changed")
    rows = jsonl(prediction_path)
    by_id = {row["sample_id"]: row for row in rows}
    require(len(rows) == len(by_id) == 5000, "Missing or duplicate anonymous row IDs")
    require(sorted(row["absolute_processed_row_index"] for row in rows) == list(range(5000, 10000)),
            "Original split row order/coverage differs")
    require(audit["recomputed_metrics"]["k"] == 3 and all(row["k"] == 3 for row in rows),
            "Only actual top-3 exports can be labeled top3")
    for row in rows:
        require(row["top1_correct"] == (row["predictions"][0] == row["target"])
                and row["topk_correct"] == (row["target"] in row["predictions"]), "Saved correctness flags differ")
    for field, metric in (("top1_correct", "top1_accuracy"), ("topk_correct", "topk_accuracy")):
        require(math.isclose(sum(bool(row[field]) for row in rows) / 5000,
                             audit["recomputed_metrics"][metric], abs_tol=1e-12), "Native aggregate differs")
    return by_id, audit, {"audit_sha256": digest(audit_path), "predictions_sha256": digest(prediction_path),
                         "processed_data_sha256": audit["processed_data_sha256"],
                         "raw_file_sha256": audit["raw_file_sha256"], "costs": audit.get("costs", {})}


def source_clusters(path, expected_sha, cases):
    require(digest(path) == expected_sha, "Processed biography file differs from verified exports")
    groups = {}
    for index, row in enumerate(itertools.islice(iter_json_objects(path), 10000)):
        if index < 5000:
            continue
        sid = row_id(expected_sha, 5000, 10000, index, row["id"])
        require(sid in cases, "Original row identity is absent from the exports")
        context = row["context"]
        require(isinstance(context, str) and bool(context), "Processed biography context missing")
        groups[sid] = hashed(["BiasBios-exact-processed-context-v1", context])
    require(set(groups) == set(cases), "Processed biography slice is incomplete")
    return groups


def score_pairs(candidate, reference):
    require(set(candidate) == set(reference) and bool(candidate), "Paired export coverage differs")
    ids = sorted(candidate, key=lambda sid: candidate[sid]["absolute_processed_row_index"])
    rows, metrics = [], {}
    for sid in ids:
        a, b = candidate[sid], reference[sid]
        require(a["absolute_processed_row_index"] == b["absolute_processed_row_index"]
                and a["target"] == b["target"], "Paired source row/target differs")
        row = {"sample_id": sid, "absolute_processed_row_index": a["absolute_processed_row_index"],
               "prompt_sha256_identical": a["prompt_sha256"] == b["prompt_sha256"]}
        for name, field in (("top1", "top1_correct"), ("top3", "topk_correct")):
            av, bv = int(bool(a[field])), int(bool(b[field]))
            row[name] = {"seka": av, "matched_zero": bv, "difference": av - bv,
                         "corrected": av == 1 and bv == 0, "regressed": av == 0 and bv == 1}
        rows.append(row)
    for name in ("top1", "top3"):
        values = [row[name] for row in rows]
        metrics[name] = {"seka_accuracy": sum(x["seka"] for x in values) / len(values),
                         "matched_zero_accuracy": sum(x["matched_zero"] for x in values) / len(values),
                         "difference": sum(x["difference"] for x in values) / len(values),
                         "difference_percentage_points": 100 * sum(x["difference"] for x in values) / len(values),
                         "corrected": sum(x["corrected"] for x in values),
                         "regressed": sum(x["regressed"] for x in values),
                         "correction_rate": sum(x["corrected"] for x in values) / len(values),
                         "regression_rate": sum(x["regressed"] for x in values) / len(values),
                         "unchanged": sum(x["difference"] == 0 for x in values),
                         "denominator_all_cases": len(values)}
    return rows, metrics


def tensor_key(value, allow_none=False):
    if value is None and allow_none:
        return ("explicit_none",)
    if not isinstance(value, dict) or not all(key in value for key in ("sha256", "shape", "dtype")):
        return None
    return value["sha256"], tuple(value["shape"]), value["dtype"]


def compare_prefill_rows(left, right, expected_cases=5000, batch_size=16):
    def select(trace):
        prefills = []
        for row in trace:
            value = row.get("input_ids")
            shape = value.get("shape") if isinstance(value, dict) else None
            if not isinstance(shape, list) or len(shape) != 2:
                return None
            # In this standard cached generation task all input prompts exceed one token.
            if shape[1] > 1:
                prefills.append(row)
        expected_batches = math.ceil(expected_cases / batch_size)
        expected_sizes = [min(batch_size, expected_cases - i * batch_size) for i in range(expected_batches)]
        if len(prefills) != expected_batches or [row["input_ids"]["shape"][0] for row in prefills] != expected_sizes:
            return None
        return prefills
    a, b = select(left), select(right)
    if a is None or b is None:
        return {"status": "blocked", "reason": "Trace cannot be resolved into exactly one cached prefill per declared batch",
                "decode_suffix_equality_required": False}, []
    comparisons = []
    for index, (x, y) in enumerate(zip(a, b)):
        ids_x, ids_y = tensor_key(x.get("input_ids")), tensor_key(y.get("input_ids"))
        mask_x = tensor_key(x["attention_mask"], True) if "attention_mask" in x else None
        mask_y = tensor_key(y["attention_mask"], True) if "attention_mask" in y else None
        known = all(item is not None for item in (ids_x, ids_y, mask_x, mask_y))
        comparisons.append({"batch_index": index, "seka_call_index": x.get("completed_call_index"),
                            "matched_zero_call_index": y.get("completed_call_index"),
                            "batch_rows": x["input_ids"]["shape"][0],
                            "input_ids_identical": ids_x == ids_y if known else None,
                            "attention_mask_identical": mask_x == mask_y if known else None,
                            "status": "blocked" if not known else "complete_verified" if ids_x == ids_y and mask_x == mask_y else "failed"})
    status = ("blocked" if any(row["status"] == "blocked" for row in comparisons) else
              "failed" if any(row["status"] == "failed" for row in comparisons) else "complete_verified")
    return {"status": status, "prefill_batches": len(a), "cases_covered": expected_cases,
            "matched_prefill_batches": sum(row["status"] == "complete_verified" for row in comparisons),
            "seka_total_forward_calls": len(left), "matched_zero_total_forward_calls": len(right),
            "decode_suffix_equality_required": False,
            "note": "Prefills are paired by batch order, not global call number; decode tokens and early stopping may differ"}, comparisons


def read_trace(manifest_path, export_audit):
    if manifest_path is None:
        return None, {"status": "blocked", "reason": "Run manifest not supplied; prefill matching remains unknown"}
    require(digest(manifest_path) == export_audit["run_manifest_sha256"], "Run manifest differs from the verified export")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    path = Path(manifest["forward_input_trace"])
    if not path.is_absolute():
        path = Path(manifest_path).parent / path
    require(digest(path) == manifest["forward_input_trace_sha256"], "Forward trace fingerprint changed")
    rows = jsonl(path)
    require(len(rows) == manifest["forward_input_trace_rows"] == manifest["forward_completed"], "Incomplete forward trace")
    require([row["completed_call_index"] for row in rows] == list(range(1, len(rows) + 1)), "Trace call sequence is incomplete")
    return rows, {"status": "complete_verified", "manifest_sha256": digest(manifest_path),
                  "trace_sha256": digest(path), "rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seka-export", type=Path, required=True)
    parser.add_argument("--zero-export", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, help="Optional for exact duplicate-biography grouped intervals")
    parser.add_argument("--seka-run-manifest", type=Path)
    parser.add_argument("--zero-run-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "BiasBios standard implementation validation; not independent clinical confirmation",
              "new_gpu_forwards": 0, "split": "train[5000:10000]", "names_used_as_patient_ids": False,
              "all_negative_results_retained": True, "deduplicated": False}
    code = 0
    try:
        a, aa, ae = load_verified(args.seka_export)
        b, ba, be = load_verified(args.zero_export)
        require(aa["processed_data_sha256"] == ba["processed_data_sha256"], "Compared runs use different processed data")
        require(aa.get("model_files_sha256") and aa["model_files_sha256"] == ba.get("model_files_sha256"),
                "Compared runs use different or unrecorded checkpoints")
        rows, metrics = score_pairs(a, b)
        report.update(exports={"seka": ae, "matched_zero": be}, metrics=metrics, n_cases=5000,
                      prompt_hashes_identical=all(row["prompt_sha256_identical"] for row in rows),
                      prompt_hash_mismatches=sum(not row["prompt_sha256_identical"] for row in rows))
        if args.processed_data:
            groups = source_clusters(args.processed_data, aa["processed_data_sha256"], a)
            counts = Counter(groups.values())
            for row in rows:
                row["duplicate_biography_source_cluster"] = groups[row["sample_id"]]
            report["bootstrap"] = {"status": "blocked", "unit": "exact processed context text SHA256; not verified person identity",
                "n_clusters": len(counts), "repeated_text_clusters": sum(size > 1 for size in counts.values()),
                "largest_text_cluster": max(counts.values()), "replicates": args.bootstrap_samples, "seed": 0,
                "weighting": "native case-weighted statistic: sampled cluster sum / sampled case count",
                "interpretation": "Marginal 95% intervals conditional on these runs; assumes distinct biography texts are independent. Different texts may still share a person/source, so these are not patient-cluster CIs."}
            if len(counts) >= 2:
                data = [[row[name][field] for name in ("top1", "top3")
                         for field in ("seka", "matched_zero", "difference", "corrected", "regressed")] for row in rows]
                intervals = bootstrap_effects(data, [groups[row["sample_id"]] for row in rows], args.bootstrap_samples, 0)
                for index, name in enumerate(("top1", "top3")):
                    for offset, field in enumerate(("seka_accuracy", "matched_zero_accuracy", "difference", "correction_rate", "regression_rate")):
                        metrics[name]["ci95_" + field] = intervals[index * 5 + offset]
                    metrics[name]["ci95_difference_percentage_points"] = [100 * value for value in intervals[index * 5 + 2]]
                report["bootstrap"]["status"] = "complete_verified"
        else:
            report["bootstrap"] = {"status": "blocked", "reason": "No processed biography source supplied; point estimates retained, names are not substituted for patient clusters"}
        name_counts = Counter(row["source_name_id_sha256"] for row in a.values())
        report["name_id_metadata"] = {"repeated_name_groups": sum(size > 1 for size in name_counts.values()),
                                       "extra_rows_with_repeated_names": sum(size - 1 for size in name_counts.values()),
                                       "used_for_clustering": False}
        traces = []
        for manifest_path, audit in ((args.seka_run_manifest, aa), (args.zero_run_manifest, ba)):
            try:
                traces.append(read_trace(manifest_path, audit))
            except Exception as exc:
                traces.append((None, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        (left, left_meta), (right, right_meta) = traces
        prefill_rows = []
        if left is None or right is None:
            prefill = {"status": "failed" if "failed" in (left_meta["status"], right_meta["status"]) else "blocked",
                       "reason": "Both verified original traces are needed for prefill matching; descriptive comparison retained"}
        else:
            prefill, prefill_rows = compare_prefill_rows(left, right, batch_size=args.batch_size)
        report["prefill_matching"] = {**prefill, "seka_trace": left_meta, "matched_zero_trace": right_meta}
        report["matched_input_status"] = ("failed" if not report["prompt_hashes_identical"] else prefill["status"])
        for filename, values in (("paired_cases.jsonl", rows), ("prefill_comparison.jsonl", prefill_rows)):
            path = args.output_dir / filename
            with path.open("x", encoding="utf-8") as handle:
                for row in values:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            report[filename + "_sha256"] = digest(path)
        report.update(status="complete_verified", status_scope="paired descriptive/statistical calculation; input-matching and cluster evidence have separate statuses")
    except Exception as exc:
        code = 1
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=digest(Path(__file__)),
                      supporting_source_sha256={name: digest(Path(__file__).with_name(name)) for name in ("counterfact_entity_ci.py", "export_biasbios.py")})
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "new_gpu_forwards": 0,
                      "matched_input_status": report.get("matched_input_status", "not_checked")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
