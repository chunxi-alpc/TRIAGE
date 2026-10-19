#!/usr/bin/env python3
"""Run full-context reference/V6/V7 by prompt/bias/learned-subspace factorial arms.

--help and --preflight-only work without Torch. Missing checkpoints or learned
operators are reported as blocked; no CPU oracle is substituted for a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path[:0] = [str(REPOSITORY), str(EXPERIMENT)]
from joint.context import full_context, prompt_sha256, prediction_input
from joint.evaluate import read_jsonl, evaluate
from joint.structures import StructureProvider, signed_targets


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def preflight(args, cases, protocol):
    failures = []
    manifest_path = args.cases.parent / "manifest.json"
    if not manifest_path.exists():
        failures.append("frozen data manifest missing")
    else:
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("files", {}).get(args.cases.name, {}).get("sha256")
        if expected != sha256(args.cases):
            failures.append("data digest differs from frozen manifest")
    for library in ("torch", "transformers"):
        if importlib.util.find_spec(library) is None:
            failures.append(library + " is not installed in this execution environment")
    if not args.model or not args.model.is_dir():
        failures.append("local fixed Qwen3-4B-Base checkpoint is unavailable")
    for case in cases:
        if prediction_input(case).keys() != {"guideline", "patient", "question"}:
            raise AssertionError("Prediction input allowlist violated")
        prompt, spans = full_context(case)
        if prompt != case["prompt"]:
            raise ValueError("Saved complete prompt differs from current builder")
        for rule in case["rules"]:
            start, end = spans[rule["rule_id"]]
            if prompt[start:end] != rule["action"]:
                raise ValueError("Complete-context action span mismatch")
    learned_status = "not_supplied"
    if args.learned_assets:
        try:
            from joint.model_backend import LearnedAssets
            LearnedAssets(args.learned_assets, {c["family_id"] for c in cases if c["split"] == "confirmation"})
            learned_status = "manifest_and_file_digests_verified"
        except Exception as error:
            learned_status = "blocked: " + type(error).__name__ + ": " + str(error)
    return {
        "status": "ready_for_real_model_initialization" if not failures else "blocked_model_runtime",
        "failures": failures, "learned_subspace_status": learned_status,
        "data_sha256": sha256(args.cases), "protocol_sha256": sha256(args.protocol),
        "source_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "cases_in_requested_split": len(cases), "rule_families": len({c["family_id"] for c in cases}),
        "sources": args.sources, "modes": args.modes,
        "reference_diagnostic_authorized": args.include_reference_diagnostic,
        "model_inference_completed": False,
        "numeric_results": None,
        "limitations": ["new data are invented mechanism cases, not independent clinical validation",
                        "schema and wording were used for development regression fixes; not blind throughout method development",
                        "joint learned arm is blocked if verified learned assets are absent",
                        "CPU validation does not validate real attention hooks"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=EXPERIMENT / "data/mechanism_cases.jsonl")
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "module2/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--module1-device", default="cuda:0")
    parser.add_argument("--module1-cache")
    parser.add_argument("--module1-base-model")
    parser.add_argument("--module1-adapter-model")
    parser.add_argument("--module1-evidence-model")
    parser.add_argument("--sources", nargs="+", choices=["reference", "reference_positive", "v6", "v7"], default=["v6", "v7"])
    parser.add_argument("--modes", nargs="+", choices=["structured_prompt", "signed_bias", "learned_subspace"],
                        default=["structured_prompt", "signed_bias", "learned_subspace"])
    parser.add_argument("--include-reference-diagnostic", action="store_true")
    parser.add_argument("--learned-assets", type=Path, default=EXPERIMENT / "module2/frozen_learned_assets.json")
    parser.add_argument("--split", choices=["train", "dev", "confirmation"], default="confirmation")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if any(s.startswith("reference") for s in args.sources) and not args.include_reference_diagnostic:
        parser.error("--sources reference requires --include-reference-diagnostic")
    if len(args.sources) != len(set(args.sources)) or len(args.modes) != len(set(args.modes)):
        parser.error("Repeated sources/modes are prohibited")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new isolated directory. No files were overwritten.")
    cases = [c for c in read_jsonl(args.cases) if c["split"] == args.split]
    if not cases:
        parser.error("Requested split is empty")
    protocol = json.loads(args.protocol.read_text())
    status = preflight(args, cases, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "preflight.json", status)
    if args.preflight_only or status["failures"]:
        save(args.output / "execution_status.json", status)
        print(json.dumps(status, indent=2))
        return 0 if args.preflight_only else 2
    journal = args.output / "predictions.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise ValueError("Prediction journal already exists; choose an isolated output directory")
    from joint.model_backend import LearnedAssets, QwenJointBackend
    assets = None
    if status["learned_subspace_status"] == "manifest_and_file_digests_verified":
        assets = LearnedAssets(args.learned_assets, {c["family_id"] for c in cases if c["split"] == "confirmation"})
    options = {"device": args.module1_device}
    for arg, name in ((args.module1_cache, "cache_dir"), (args.module1_base_model, "base_model"),
                      (args.module1_adapter_model, "adapter_model"), (args.module1_evidence_model, "evidence_model")):
        if arg:
            options[name] = arg
    started = time.time()
    save(args.output / "execution_status.json", status | {"status": "initializing_real_models"})
    try:
        backend = QwenJointBackend(args.model, args.device, protocol, assets)
    except Exception as error:
        if getattr(error, "compatibility_audit", None) is not None:
            save(args.output / "model_compatibility.json", error.compatibility_audit)
        status.update(status="blocked_model_initialization", failures=[type(error).__name__ + ": " + str(error)])
        save(args.output / "execution_status.json", status)
        print(json.dumps(status, indent=2))
        return 2
    save(args.output / "model_provenance.json", backend.provenance)
    save(args.output / "model_compatibility.json", backend.compatibility_audit)
    providers = {s: StructureProvider(s, include_reference_diagnostic=args.include_reference_diagnostic,
                                       model_options=options) for s in args.sources}
    records = []
    zero_records = []
    zero_journal = args.output / "zero_checks.jsonl"
    source_errors = {}
    shared_adapters = None
    save(args.output / "execution_status.json", status | {"status": "running_real_model_inference"})
    with journal.open("x", buffering=1) as stream, zero_journal.open("x", buffering=1) as zero_stream:
        def append(row):
            row.update(data_sha256=status["data_sha256"], protocol_sha256=status["protocol_sha256"])
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            records.append(row)
        def append_zero(row):
            row.update(data_sha256=status["data_sha256"], protocol_sha256=status["protocol_sha256"])
            zero_stream.write(json.dumps(row, allow_nan=False) + "\n")
            zero_records.append(row)
        for case in cases:
            prompt, _ = full_context(case)
            try:
                output = backend.score(prompt, [], "baseline", case["outer_fold"])
                append({"case_id": case["case_id"], "source": "none", "mode": "baseline",
                        "status": "completed", "is_model_prediction": True, "prompt_sha256": prompt_sha256(prompt), **output})
            except Exception as error:
                append({"case_id": case["case_id"], "source": "none", "mode": "baseline",
                        "status": "model_error", "is_model_prediction": False, "error": type(error).__name__ + ": " + str(error)})
        for source, provider in providers.items():
            try:
                provider.load(shared_adapters)
                if provider.instance is not None and shared_adapters is None:
                    # Both experimental versions use the exact same frozen
                    # TrialMatchAI and MedNLI model instances. This reduces
                    # memory without transferring predictions or gold labels.
                    shared_adapters = {key: getattr(provider.instance, key)
                                       for key in ("trialmatch", "mednli", "nli_batch_size", "candidate_limit")}
            except Exception as error:
                source_errors[source] = type(error).__name__ + ": " + str(error)
            for case in cases:
                if source in source_errors:
                    append_zero({"case_id": case["case_id"], "source": source,
                                 "status": "blocked_structure_model", "zero_forward_attempted": False,
                                 "error": source_errors[source]})
                    for mode in args.modes:
                        append({"case_id": case["case_id"], "source": source, "mode": mode,
                                "status": "blocked_structure_model", "is_model_prediction": False, "error": source_errors[source]})
                    continue
                try:
                    structure, raw = provider.predict(case)
                except Exception as error:
                    structure, raw = {"source": source, "status": "parse_failure", "parse_failure": True,
                                      "rules": [], "error": type(error).__name__ + ": " + str(error)}, None
                save(args.output / "structures" / source / (case["case_id"] + ".json"),
                     {"case_id": case["case_id"], "input_sha256": prompt_sha256(json.dumps(prediction_input(case), sort_keys=True)),
                      "normalized": structure, "raw": raw})
                prompt, action_spans = full_context(case, structure)
                targets = signed_targets(structure, action_spans)
                cached_structured = None
                try:
                    cached_structured, zero_audit = backend.signed_zero_control(prompt, case["outer_fold"])
                except Exception as error:
                    zero_audit = {"status": "model_error", "zero_forward_attempted": False,
                                  "error": type(error).__name__ + ": " + str(error)}
                append_zero({"case_id": case["case_id"], "source": source,
                             "prompt_sha256": prompt_sha256(prompt), "outer_fold": case["outer_fold"],
                             "prompt_tokens": cached_structured.get("prompt_tokens") if cached_structured else None,
                             **zero_audit})
                for mode in args.modes:
                    base = {"case_id": case["case_id"], "source": source, "mode": mode,
                            "structure": structure, "prompt_sha256": prompt_sha256(prompt),
                            "reference_diagnostic": source.startswith("reference"),
                            "signed_zero_control_status": zero_audit["status"]}
                    if mode == "structured_prompt":
                        if cached_structured is not None:
                            append(base | {"status": "completed", "is_model_prediction": True,
                                           "reused_zero_control_baseline": True, **cached_structured})
                        else:
                            append(base | {"status": "model_error", "is_model_prediction": False,
                                           "error": zero_audit.get("error", "Structured zero-control baseline failed")})
                        continue
                    if mode == "signed_bias" and zero_audit["status"] != "passed":
                        append(base | {"status": "zero_control_failed", "is_model_prediction": False,
                                       "reason": "Empty-target direct-mask path failed same-prompt task-output equality",
                                       "zero_control": zero_audit})
                        continue
                    if mode == "learned_subspace" and assets is None:
                        append(base | {"status": "blocked_learned_assets", "is_model_prediction": False,
                                       "reason": status["learned_subspace_status"]})
                        continue
                    try:
                        output = backend.score(prompt, targets, mode, case["outer_fold"])
                        append(base | {"status": "completed", "is_model_prediction": True, **output})
                    except Exception as error:
                        append(base | {"status": "model_error", "is_model_prediction": False,
                                       "error": type(error).__name__ + ": " + str(error)})
    zero_coverage = {}
    for source in args.sources:
        rows = [row for row in zero_records if row["source"] == source]
        checked = sum(row["status"] in {"passed", "failed_output_mismatch"} for row in rows)
        zero_coverage[source] = {
            "expected_case_source_checks": len(cases), "recorded_case_source_checks": len(rows),
            "zero_forward_attempted": sum(bool(row["zero_forward_attempted"]) for row in rows),
            "comparisons_completed": checked, "comparison_coverage": checked / len(cases),
            "passed": sum(row["status"] == "passed" for row in rows),
            "output_mismatches": sum(row["status"] == "failed_output_mismatch" for row in rows),
            "model_errors": sum(row["status"] == "model_error" for row in rows),
            "blocked_structure_model": sum(row["status"] == "blocked_structure_model" for row in rows),
        }
    zero_passed = len(zero_records) == len(cases) * len(args.sources) and all(r["status"] == "passed" for r in zero_records)
    save(args.output / "zero_checks.json", {
        "status": "passed_all_actual_case_source_checks" if zero_passed else "failed_or_incomplete",
        "expected_case_source_checks": len(cases) * len(args.sources), "by_source": zero_coverage,
        "zero_check_log_sha256": sha256(zero_journal),
        "comparison": "exact candidate logits and A/B/C/D probabilities; structured outputs reused",
        "nonzero_intervention_effects_validated": False,
    })
    result = evaluate(read_jsonl(args.cases), records, args.split)
    save(args.output / "metrics.json", result)
    complete = all(r["status"] == "completed" for r in records) and zero_passed
    status.update(status="completed_real_model_inference" if complete else "partial_real_model_inference",
                  model_inference_completed=True, journal_records=len(records), actual_forward_calls=backend.forward_calls,
                  elapsed_seconds=time.time()-started, prediction_log_sha256=sha256(journal),
                  source_load_failures=source_errors, shared_frozen_module1_adapters=True,
                  signed_zero_checks_passed=zero_passed, signed_zero_coverage_by_source=zero_coverage,
                  model_provenance_sha256=sha256(args.output / "model_provenance.json"))
    save(args.output / "execution_status.json", status)
    print(json.dumps(status, indent=2))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
