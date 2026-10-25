#!/usr/bin/env python3
"""Independent server-only MEGA train_val evaluation; never opens test.

Official parsing, judge requests, Acc normalization and DTW are called unchanged.
Raw requests/responses stay in private/. Public results contain IDs and numbers.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from mega_trainval_reader import DATASET_REVISION, OFFICIAL_COMMIT, snapshot_digest
from counterfact_entity_ci import bootstrap_effects


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity(row):
    return row.get("case_path_id"), row.get("path_index"), row.get("filename")


def align_predictions(manifest, rows):
    """Retain the complete frozen denominator; fail rather than discard extras."""
    expected = {identity(entry["identity"]): entry for entry in manifest["selected"]}
    require(len(expected) == len(manifest["selected"]) > 0, "Invalid frozen identity set")
    supplied = {}
    for row in rows:
        key = identity(row)
        require(key in expected, "Prediction contains an extra or unidentifiable case")
        require(key not in supplied, "Prediction contains a duplicate frozen case")
        supplied[key] = row
    aligned = []
    for entry in manifest["selected"]:
        original = supplied.get(identity(entry["identity"]))
        if original is None:
            original = {**entry["identity"], "language": "en", "llm_output": "", "status": "error",
                        "evaluation_input_failure": "missing_prediction"}
        else:
            original = dict(original)
            require(original.get("id") == entry["identity"]["id"], "Prediction source ID differs from frozen identity")
            if original.get("seed", 0) != 0:
                raise ValueError("This development evaluator accepts the frozen method seed 0 only")
        aligned.append((entry, original))
    return aligned


def parse_with_official(parser_module, record):
    try:
        if record.get("status") == "error" or record.get("evaluation_input_failure"):
            raise ValueError("failed_or_missing_generation")
        return parser_module.parse_record(record), None
    except Exception as exc:
        failure = {key: record.get(key) for key in ("id", "case_path_id", "path_index", "filename", "language")}
        failure.update(llm_output=record.get("llm_output", ""), parsed_success=False)
        return failure, {"class": type(exc).__name__, "reason_sha256": value_hash(str(exc))}


def native_metrics(pathway, accuracy, summary_module, pathway_module):
    """Use native functions with fixed failure scores; never invent binary Acc."""
    result = {"pcc_dtw_cost": 1., "acc_normalized": 0., "pathway_valid": False, "accuracy_valid": False,
              "pathway_judge_results": len(pathway.get("results", [])),
              "accuracy_judge_results": len(accuracy.get("results", [])), "metric_failures": {}}
    try:
        require(pathway.get("skipped") is not True, "Official pathway judge marked skipped")
        matrix = summary_module.average_and_normalize_matrices(pathway.get("results"), 5, True, "qwen")
        require(matrix and matrix[0] and all(math.isfinite(x) for row in matrix for x in row), "Invalid normalized pathway matrix")
        # Exact native PCC path: similarity normalization -> 1-similarity -> official DTW.
        _, ratio, _ = pathway_module.dynamic_time_warping([[1. - x for x in row] for row in matrix])
        require(math.isfinite(ratio) and 0 <= ratio <= 1, "Invalid native DTW ratio")
        result.update(pcc_dtw_cost=float(ratio), pathway_valid=True)
    except Exception as exc:
        result["metric_failures"]["pathway"] = {"class": type(exc).__name__, "reason_sha256": value_hash(str(exc))}
    try:
        require(accuracy.get("skipped") is not True, "Official accuracy judge marked skipped")
        normalized = summary_module.normalize_accuracy_scores(accuracy.get("results"), 5, True)
        value = math.fsum(normalized) / len(normalized)
        require(math.isfinite(value) and 0 <= value <= 1, "Invalid native normalized accuracy")
        result.update(acc_normalized=value, accuracy_valid=True)
    except Exception as exc:
        result["metric_failures"]["accuracy"] = {"class": type(exc).__name__, "reason_sha256": value_hash(str(exc))}
    return result


class RequestRecorder:
    """One record per actual create() attempt, including official retries."""
    def __init__(self, client, raw_path, metadata_path):
        self.client, self.context, self.rows = client, {}, []
        self.raw = Path(raw_path).open("x", encoding="utf-8")
        self.metadata = Path(metadata_path).open("x", encoding="utf-8")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        started = time.monotonic()
        forwards_before = getattr(self.client, "forward_attempts", None)
        completed_before = getattr(self.client, "forward_completed", None)
        row = {**self.context, "attempt_index": len(self.rows) + 1,
               "request_sha256": value_hash(request), "status": "running"}
        raw = {**self.context, "attempt_index": row["attempt_index"], "request": request}
        try:
            response = self.client.chat.completions.create(**request)
            payload = response.model_dump()
            raw["response"] = payload
            usage = payload.get("usage") or {}
            row.update(status="ok", response_sha256=value_hash(payload),
                       usage={key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                       finish_reasons=[choice.get("finish_reason") for choice in payload.get("choices", [])])
            if "local_raw" in payload:
                row["local_generation_costs"] = {key: payload["local_raw"].get(key) for key in (
                    "generation_elapsed_seconds", "process_peak_allocated_gib_so_far", "process_peak_reserved_gib_so_far")}
            return response
        except Exception as exc:
            raw["error"] = {"type": type(exc).__name__, "message": str(exc)}
            row.update(status="error", error_type=type(exc).__name__, error_sha256=value_hash(str(exc)), usage=None)
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            if forwards_before is not None:
                row["local_forward_attempts"] = self.client.forward_attempts - forwards_before
                row["local_forward_completed"] = self.client.forward_completed - completed_before
            self.rows.append(row)
            self.raw.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")
            self.metadata.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.raw.flush()
            self.metadata.flush()

    def close(self):
        self.raw.close()
        self.metadata.close()

    def cost_summary(self):
        groups = {}
        for row in self.rows:
            key = row.get("method", "unknown")
            item = groups.setdefault(key, {"attempts": 0, "successful_responses": 0, "failed_attempts": 0,
                                          "elapsed_request_seconds": 0., "prompt_tokens_known": 0,
                                          "completion_tokens_known": 0, "requests_without_usage": 0,
                                          "local_forward_attempts_known": 0, "local_forward_completed_known": 0})
            item["attempts"] += 1
            item["successful_responses"] += row["status"] == "ok"
            item["failed_attempts"] += row["status"] == "error"
            item["elapsed_request_seconds"] += row["elapsed_seconds"]
            item["local_forward_attempts_known"] += row.get("local_forward_attempts", 0)
            item["local_forward_completed_known"] += row.get("local_forward_completed", 0)
            usage = row.get("usage") or {}
            if any(usage.get(name) is None for name in ("prompt_tokens", "completion_tokens")):
                item["requests_without_usage"] += 1
            else:
                item["prompt_tokens_known"] += usage["prompt_tokens"]
                item["completion_tokens_known"] += usage["completion_tokens"]
        return {"by_method": groups, "all_attempts": len(self.rows),
                "scope": "all create attempts including official retry attempts; failed requests may have unknown token use",
                "raw_request_response_artifact": "private/raw_judge_requests.jsonl; server-only"}


def load_official(parser_path):
    parser_path = parser_path.resolve()
    root = parser_path.parents[3]
    directory = root / "mega_cdp/scripts/evaluate"
    require(parser_path == directory / "parse_prediction_output_singleturn.py", "Unexpected official parser location")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(commit == OFFICIAL_COMMIT, "Official MEGA checkout is not the fixed commit")
    names = ["parse_prediction_output_singleturn.py", "pathway_consistency_eval_batch_singleturn.py",
             "accuracy_eval_batch_singleturn.py", "gt_path_references.py", "pairwise_score_parser.py",
             "resume_utils.py", "summarize_eval_metrics.py", "pathway_metrics.py", "optimal_transport.py"]
    paths = [directory / name for name in names]
    paths += [root / "mega_cdp/template" / language / filename for language, filename in (
        ("template_en", "PATHWAY_CONSISTENCY_EVAL_TEMPLATE_PATHWAY.txt"),
        ("template_zh", "ZH_PATHWAY_CONSISTENCY_EVAL_TEMPLATE_PATHWAY.txt"),
        ("template_en", "ACCURACY_EVAL_TEMPLATE.txt"),
        ("template_zh", "ZH_ACCURACY_EVAL_TEMPLATE.txt"))]
    fingerprints = {}
    for path in paths:
        relative = str(path.relative_to(root))
        content = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        expected = subprocess.check_output(["git", "-C", str(root), "rev-parse", f"{OFFICIAL_COMMIT}:{relative}"], text=True).strip()
        require(blob == expected, "Official evaluation source/template differs from the pinned commit")
        fingerprints[relative] = hashlib.sha256(content).hexdigest()
    sys.path.insert(0, str(directory))
    def module(filename):
        name = "aster_official_" + filename.removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded
    loaded = {"parser": module(names[0]), "pathway": module(names[1]), "accuracy": module(names[2]),
              "summary": module("summarize_eval_metrics.py"), "metrics": importlib.import_module("pathway_metrics")}
    require(Path(loaded["metrics"].__file__).resolve() == directory / "pathway_metrics.py", "Unexpected cached metrics module")
    return loaded, fingerprints


def compare_methods(rows, reference, bootstrap_samples=10000):
    import numpy as np
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["sample_id"]] = row
    require(reference in by_method, "Reference method is missing")
    result = []
    for name, current in by_method.items():
        if name == reference:
            continue
        prior = by_method[reference]
        require(set(current) == set(prior), "Paired methods have incomplete frozen coverage")
        ids = sorted(current)
        groups = [current[sid]["tree_cluster"] for sid in ids]
        for metric, higher in (("acc_normalized", True), ("pcc_dtw_cost", False)):
            deltas = np.asarray([current[sid][metric] - prior[sid][metric] for sid in ids])
            benefit = deltas if higher else -deltas
            item = {"method": name, "reference": reference, "metric": metric,
                    "difference_method_minus_reference": float(deltas.mean()),
                    "higher_is_better": higher, "n_cases": len(ids), "n_tree_clusters": len(set(groups)),
                    "score_improved": int((benefit > 1e-12).sum()), "score_degraded": int((benefit < -1e-12).sum()),
                    "score_unchanged": int((abs(benefit) <= 1e-12).sum()),
                    "binary_corrected": None, "binary_regressed": None,
                    "correction_note": "Native scores are continuous; only score improvement/degradation is defined",
                    "ci95": None, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": 0}
            if len(set(groups)) >= 2:
                item["ci95"] = bootstrap_effects(deltas[:, None], groups, bootstrap_samples, 0)[0]
                item["status"] = "complete_verified"
            else:
                item.update(status="blocked", reason="Fewer than two source-tree groups; point estimate retained")
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-val-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--official-parser", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="METHOD=prediction.jsonl")
    parser.add_argument("--reference", help="Existing method name for paired comparisons")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--model", type=Path, help="Existing local Qwen3-8B judge checkpoint")
    parser.add_argument("--served-model-name", default="Qwen3_8B")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-context-length", type=int, default=32768)
    parser.add_argument("--judge-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    private = args.output_dir / "private"
    private.mkdir(mode=0o700)
    started = time.monotonic()
    report = {"status": "running", "scope": "train_val development only; not independent clinical confirmation",
              "test_files_read": False, "independent_model_claim": False,
              "judge_track": "official local Qwen alternative; differs from paper's GPT-5.4 Mini judge",
              "judge_weight_relationship": "same Qwen checkpoint may serve generation and evaluation in separate roles/stages",
              "raw_text_published": False, "all_failures_in_denominator": True,
              "binary_accuracy_claim": False, "method_parameters_changed": False}
    code, recorder, client = 0, None, None
    try:
        train_val = args.train_val_root.resolve()
        require(train_val.name == "train_val", "Explicit train_val directory is required")
        snapshot = train_val.parent
        frozen = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
        require(frozen.get("split") == "train_val" and frozen.get("expected_dataset_revision") == DATASET_REVISION,
                "Only the fixed train_val manifest is accepted")
        require(snapshot_digest(snapshot, train_val / "vignette_cases.jsonl") == frozen["source_file_sha256"],
                "Frozen development input file changed")
        reference_path = train_val / "gt_paths.jsonl"
        reference_sha = snapshot_digest(snapshot, reference_path)
        predictions = {}
        for item in args.prediction:
            name, sep, path = item.partition("=")
            require(sep and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in predictions, "Invalid/duplicate method mapping")
            predictions[name] = Path(path)
        if len(predictions) > 1:
            require(args.reference in predictions, "An explicit available reference method is required for comparisons")
        official, source_hashes = load_official(args.official_parser)
        config = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "method_order": list(predictions),
                  "method_development_seed": 0, "prediction_sha256": {name: digest(path) for name, path in predictions.items()},
                  "freeze_manifest_sha256": digest(args.freeze_manifest), "train_val_reference_sha256": reference_sha,
                  "official_code_commit": OFFICIAL_COMMIT, "official_source_files_sha256": source_hashes,
                  "judge": {"backend": args.backend, "served_name": args.served_model_name,
                            "model_path": str(args.model) if args.model else None,
                            "model_config_sha256": digest(args.model / "config.json") if args.model else None,
                            "model_weights_fingerprint_source": ("coordinator run_official_job run_manifest.json records all weight shards"
                                                                 if args.backend == "local" else "separate API service manifest required"),
                            "api_base_url": args.api_base_url if args.backend == "api" else None,
                            "expected_results_per_metric": 5, "temperature": .7, "top_p": .8, "top_k": 20,
                            "thinking": False, "max_new_tokens": args.judge_max_tokens,
                            "context_length": args.judge_context_length if args.backend == "local" else None,
                            "local_rng_seed": args.judge_seed if args.backend == "local" else None,
                            "upstream_seed_note": "Upstream API did not specify a seed; local adaptation sets torch seed once",
                            "request_workers": 1, "official_transport_attempts_max": 3,
                            "sdk_internal_retries": 0},
                  "statistics": {"bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": 0,
                                 "group": "frozen source-guideline document, all its paths together",
                                 "method_reference": args.reference, "multiplicity": "marginal 95% CIs, no significance claim"}}
        write_json(args.output_dir / "evaluation_config.json", config)
        references = official["pathway"].load_gt_path_references(reference_path)
        aligned = {name: align_predictions(frozen, read_jsonl(path)) for name, path in predictions.items()}
        if args.backend == "local":
            require(args.model is not None, "Local judge checkpoint path is required")
            from mega_local_qwen_client import LocalQwenClient
            client = LocalQwenClient(args.model, args.served_model_name, args.judge_context_length, args.judge_seed)
        else:
            from openai import OpenAI
            client = OpenAI(api_key="EMPTY", base_url=args.api_base_url, max_retries=0)
        recorder = RequestRecorder(client, private / "raw_judge_requests.jsonl", args.output_dir / "judge_attempts.jsonl")
        judge_args = SimpleNamespace(qwen_model_name=args.served_model_name, qwen_max_tokens=args.judge_max_tokens, expected_results=5)
        scored = []
        for name, cases in aligned.items():
            with (private / (name + "_official.jsonl")).open("x", encoding="utf-8") as log:
                for entry, raw in cases:
                    parsed, parse_failure = parse_with_official(official["parser"], raw)
                    native = {}
                    for metric in ("pathway", "accuracy"):
                        recorder.context = {"method": name, "sample_id": entry["sample_id"], "metric": metric}
                        native[metric] = official[metric].evaluate_record(parsed, references, recorder, judge_args)
                    log.write(json.dumps({"sample_id": entry["sample_id"], "parsed": parsed, **native}, ensure_ascii=False) + "\n")
                    log.flush()
                    values = native_metrics(native["pathway"], native["accuracy"], official["summary"], official["metrics"])
                    scored.append({"sample_id": entry["sample_id"], "tree_cluster": entry["source_cluster"],
                                   "method": name, "method_seed": 0, "parsed_success": parsed["parsed_success"],
                                   "parse_failure": parse_failure, **values})
        with (args.output_dir / "case_metrics.jsonl").open("x", encoding="utf-8") as handle:
            for row in scored:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        summaries = {}
        for name in predictions:
            rows = [row for row in scored if row["method"] == name]
            require(len(rows) == len(frozen["selected"]), "Internal evaluation denominator changed")
            summaries[name] = {"n_cases": len(rows), "n_tree_clusters": len({row["tree_cluster"] for row in rows}),
                               "PCC_DTW_cost_lower_better": math.fsum(row["pcc_dtw_cost"] for row in rows) / len(rows),
                               "Acc_normalized_1_to_5_higher_better": math.fsum(row["acc_normalized"] for row in rows) / len(rows),
                               "parse_failures": sum(not row["parsed_success"] for row in rows),
                               "pathway_failures_scored_one": sum(not row["pathway_valid"] for row in rows),
                               "accuracy_failures_scored_zero": sum(not row["accuracy_valid"] for row in rows)}
        comparisons = compare_methods(scored, args.reference, args.bootstrap_samples) if args.reference else []
        report.update(status="complete_verified", status_scope="evaluation completed, including failures assigned native worst scores",
                      methods=summaries, comparisons=comparisons, denominator_per_method=len(frozen["selected"]),
                      confidence_scope="Development sample and fixed method seed only; source-group independence assumed; five judge samples are not five patients",
                      official_metric_scope="Native Acc normalization and DTW PCC; ancillary OT metrics are not calculated")
    except Exception as exc:
        code = 1
        report.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
    finally:
        if recorder is not None:
            recorder.close()
            report["judge_request_costs"] = recorder.cost_summary()
        if client is not None and hasattr(client, "cost_summary"):
            try:
                report["judge_process_costs"] = client.cost_summary()
            except Exception as exc:
                report["judge_process_costs"] = {"status": "failed", "error_type": type(exc).__name__,
                                                 "note": "Consult outer process run_manifest for failed-run GPU accounting"}
        elif client is not None:
            report["judge_process_costs"] = {"backend": "local_api", "gpu_peak_memory": None,
                                             "note": "API cannot measure service GPU memory; use separate service run manifest"}
        report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=digest(Path(__file__)),
                      local_client_sha256=digest(Path(__file__).with_name("mega_local_qwen_client.py")),
                      supporting_source_sha256={name: digest(Path(__file__).with_name(name)) for name in (
                          "mega_trainval_reader.py", "counterfact_entity_ci.py")})
        write_json(args.output_dir / "evaluation_report.json", report)
    print(json.dumps({"status": report["status"], "scope": "train_val development", "independent_model_claim": False}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
