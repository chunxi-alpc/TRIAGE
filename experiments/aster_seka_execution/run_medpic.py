#!/usr/bin/env python3
"""Server-only native MedPIC runner. Formal questions are opened only after freeze.

--synthetic uses two built-in invented cases and never accepts a dataset file.
--engine fake is a CPU orchestration test, never a medical model evaluation.
The program cannot create or modify a configuration freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REVISION = "9ef6db4f13865b14fc2e6be3f94dcfaf3a0cf983"
PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def runner_source_hashes():
    import medpic_execution, native_generation_baselines
    return {Path(p).name: sha_file(p) for p in
            (__file__, medpic_execution.__file__, native_generation_baselines.__file__)}


def synthetic_rows():
    # These are invented symbolic examples, not clinical advice or benchmark data.
    return [
        {"instance_id": "synthetic_medpic_alpha", "patient_vignette":
            "Alpha is present. Beta is absent. The other observation is unchanged.",
         "question": "For this invented symbolic exercise, which options are supported?",
         "options": {"A": "Select the alpha option", "B": "Select the beta option", "C": "Select the delta option"}},
        {"instance_id": "synthetic_medpic_unknown", "patient_vignette":
            "No measurement of omega is available. The marker has not been tested.",
         "question": "For this invented symbolic exercise, which options are supported?",
         "options": {"A": "Select the omega option", "B": "Select the zeta option"}},
    ]


def input_rows(lock, *, synthetic, questions=None, engine="qwen"):
    """Check authorization before even hashing/opening a formal question file."""
    if synthetic:
        if questions is not None:
            raise ValueError("Synthetic mode uses only built-in invented inputs; no dataset path accepted")
        return synthetic_rows(), {"synthetic": True, "rows": 2, "dataset_file_opened": False}
    if engine != "qwen":
        raise ValueError("Fake inference is restricted to synthetic inputs")
    if lock.get("frozen") is not True or not lock.get("frozen_at"):
        raise ValueError("Formal 467-question inference is prohibited until the coordinator freezes the configuration")
    data = lock.get("dataset", {})
    if data.get("revision") != REVISION or data.get("rows") != 467 or not data.get("questions_sha256"):
        raise ValueError("Freeze the actual MedPIC revision, full 467-row count and file hash")
    if questions is None or sha_file(questions) != data["questions_sha256"]:
        raise ValueError("Formal question file differs from its frozen fingerprint")
    rows = json.loads(Path(questions).read_text())
    if not isinstance(rows, list) or len(rows) != 467:
        raise ValueError("Formal run requires all 467 questions")
    ids = [str(row["instance_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate official instance_id")
    # The method never receives the raw row, answer, taxonomy, GF/CF or pair fields.
    projected = [{"instance_id": row["instance_id"], "patient_vignette": row["patient_vignette"],
                  "question": row["question"], "options": row["options"]} for row in rows]
    return projected, {"synthetic": False, "rows": len(projected), "dataset_file_opened": True,
                       "revision": REVISION, "questions_sha256": data["questions_sha256"],
                       "instance_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


def validate_lock(lock, *, synthetic, seed):
    from medpic_option_transfer import TransferConfig, require_lock
    from medpic_execution import method_names
    from native_generation_baselines import FusionConfig
    config = TransferConfig(**lock["config"])
    config.validate()
    sources = lock.get("fusion", {}).get("condition_sources", ["A0", "A1"])
    params = lock.get("fusion", {}).get("parameters", asdict(FusionConfig()))
    fusion = FusionConfig(**params)
    fusion.validate()
    methods = method_names(sources)
    if not synthetic:
        require_lock(lock, config, seed)
        frozen_fusion = lock.get("fusion", {})
        if set(frozen_fusion.get("parameters", {})) != set(asdict(FusionConfig())) or not frozen_fusion.get("condition_sources"):
            raise ValueError("Explicitly freeze all vocabulary-fusion parameters and A sources")
        if lock.get("runner_source_hashes") != runner_source_hashes() or lock.get("methods") != methods:
            raise ValueError("Runner source or full method list differs from the frozen record")
        provenance = lock.get("fusion", {}).get("provenance", {})
        if not provenance.get("kind"):
            raise ValueError("Freeze fusion parameters and their calibration/transfer provenance")
        if not lock.get("model_fingerprint") or not lock.get("tokenizer_chat_template_sha256"):
            raise ValueError("Freeze the actual model and tokenizer fingerprints")
        if not isinstance(lock.get("primary_comparisons"), list):
            raise ValueError("Declare the primary comparison family before formal inference")
        operator = lock.get("operator", {})
        if (operator.get("kind") != "SEKA" or operator.get("source_commit") != PIN or
                not isinstance(operator.get("layers"), str) or
                not isinstance(operator.get("amplify_pos"), (int, float)) or
                not isinstance(operator.get("amplify_neg"), (int, float))):
            raise ValueError("Freeze the actual SEKA layer selection and numeric gains")
    return config, fusion, sources, methods


def fake_callbacks(fusion_config=None):
    """CPU-only test backend with invented scores and honest no-GPU metadata."""
    import torch
    def infer(prompt, policy, control, budget, seed):
        from aster_seka_adapter import Control
        if control in {Control.WRONG_SPAN, Control.MISALIGNED_RULE}:
            return {"status": "control_unavailable", "text": "", "control_executed": False,
                    "control_unavailable_reason": "Fake backend does not construct real tokenizer masks",
                    "actual_forward_calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.,
                    "peak_allocated_gib": 0., "peak_reserved_gib": 0.}
        result = {"status": "ok", "text": "T" if prompt.stage == "condition" else 'FINAL_ANSWER: ["A"]',
                  "label_logits": {"T": 2., "F": 0., "U": 1.}, "actual_forward_calls": 0,
                  "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.,
                  "peak_allocated_gib": 0., "peak_reserved_gib": 0., "backend": "synthetic_fake_no_model",
                  "control_executed": False, "_first_token_scores_tensor": torch.tensor([2., 0., 1., 3.])}
        return result
    def baseline(primary, auxiliary, *, kind, control, max_new_tokens, seed):
        from native_generation_baselines import FusionConfig, mix_vocabulary_logits
        base = torch.tensor([[2., 0., 1., 3.]])
        combined, details = mix_vocabulary_logits(base, base + .5, kind, fusion_config or FusionConfig(), control)
        return {"status": "ok", "text": 'FINAL_ANSWER: ["A"]', "actual_forward_calls": 0,
                "input_tokens_per_path": [0, 0], "output_tokens": 0, "wall_seconds": 0.,
                "peak_allocated_gib": 0., "peak_reserved_gib": 0., "backend": "synthetic_fake_no_model",
                "first_log_probs": combined[0].log_softmax(-1), "first_raw_base": base[0],
                "step_trace": [details]}
    return infer, baseline


def qwen_runtime(args, lock, config, fusion):
    import torch
    from src.model import SEKALLM
    from aster_seka_adapter import StageSEKAAdapter
    from medpic_option_transfer import chat_prompt, fingerprint, seka_infer
    from native_generation_baselines import generate_native
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one coordinator-assigned GPU")
    actual = subprocess.check_output(["git", "-C", str(args.seka_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError("Official SEKA revision differs from the registered source")
    if any(value is None for value in (args.model, args.model_fingerprint, args.pos, args.neg)):
        raise ValueError("Qwen runtime requires the existing model/fingerprint and both SEKA projections")
    supplied = lock.get("operator", {})
    layers, positive, negative = supplied.get("layers", "last10"), supplied.get("amplify_pos", 2.4), supplied.get("amplify_neg", 0.)
    if not args.synthetic:
        if lock["model_fingerprint"] != args.model_fingerprint or supplied.get("source_commit") != actual:
            raise ValueError("Checkpoint/source differs from the frozen record")
        actual_projection = {"positive": sha_file(args.pos), "negative": sha_file(args.neg)}
        if lock.get("projection_sha256") != actual_projection:
            raise ValueError("Projection files differ from the frozen record")
    rope = lock.get("runtime_rope_scaling")
    kwargs = {"rope_scaling": rope} if rope is not None else {}
    operator = SEKALLM(args.model, device="cuda", pos_pt=str(args.pos), neg_pt=str(args.neg), layers=layers,
                       amplify_pos=float(positive), amplify_neg=float(negative), torch_dtype="auto",
                       attn_implementation="sdpa", **kwargs)
    operator.model.eval()
    adapter = StageSEKAAdapter(operator, kind="seka", source_commit=actual,
                               projection_paths=[str(args.pos), str(args.neg)])
    chat_hash = fingerprint(operator.tok.chat_template)
    if not args.synthetic and lock["tokenizer_chat_template_sha256"] != chat_hash:
        raise ValueError("Actual chat template differs from the frozen record")
    base_infer = seka_infer(adapter, config)

    def vocabulary(primary, auxiliary, *, kind, control, max_new_tokens, seed):
        operator.remove_projection()
        started, handle, count = time.perf_counter(), None, [0]
        torch.cuda.reset_peak_memory_stats(operator.device)
        result = {"status": "error", "text": ""}
        try:
            p, a = chat_prompt(primary, operator.tok), chat_prompt(auxiliary, operator.tok)
            limit = min(config.context_limit, int(operator.model.config.max_position_embeddings))
            def counter(*unused):
                count[0] += 1
            handle = operator.model.register_forward_hook(counter)
            result = generate_native(operator.model, operator.tok, p.text, a.text, kind=kind, config=fusion,
                                     control=control, max_new_tokens=max_new_tokens, max_context_tokens=limit)
            if result["actual_forward_calls"] != count[0]:
                raise ValueError("Outer forward count differs from vocabulary fusion record")
            result.update(status="ok", requested_case_seed=seed,
                          greedy_seed_note="Reused official native generation helper fixes RNG 0; deterministic greedy; not independent seeds")
        except Exception as error:
            result.update(status="error", text="", error_type=type(error).__name__, error=str(error),
                          **getattr(error, "partial_generation", {}))
        finally:
            if handle is not None:
                handle.remove()
            operator.remove_projection()
        torch.cuda.synchronize(operator.device)
        result.update(actual_forward_calls=count[0], wall_seconds=time.perf_counter() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(operator.device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(operator.device) / 2**30)
        return result

    runtime = {"source_commit": actual, "model_fingerprint": args.model_fingerprint,
               "projection_manifest": adapter.projections, "tokenizer_chat_template_sha256": chat_hash,
               "layers": layers, "amplify_pos": positive, "amplify_neg": negative,
               "model_config": operator.model.config.to_dict(),
               "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")}}
    return adapter, base_infer, vocabulary, runtime


def cli():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--engine", choices=["fake", "qwen"], default="qwen")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parent = Path(__file__).resolve().parent.parent
    parser.add_argument("--audit-dir", type=Path, default=parent / "data_audit")
    parser.add_argument("--clinical-dir", type=Path, default=parent / "clinical_dev")
    parser.add_argument("--adapter-dir", type=Path, default=parent / "method_agent")
    parser.add_argument("--seka-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--pos", type=Path)
    parser.add_argument("--neg", type=Path)
    return parser.parse_args()


def main(args):
    for path in (args.audit_dir, args.clinical_dir, args.adapter_dir, args.seka_root):
        if path is not None:
            sys.path.insert(0, str(path.resolve()))
    import torch
    from medpic_native_adapter import native_input
    from medpic_option_transfer import attach_controls, option_evidence, seka_infer, source_manifest
    from medpic_execution import run_case
    lock = json.loads(args.config_lock.read_text())
    config, fusion, fusion_sources, methods = validate_lock(lock, synthetic=args.synthetic, seed=args.seed)
    # This check runs before model creation, question hashing and question reading.
    rows, input_manifest = input_rows(lock, synthetic=args.synthetic, questions=args.questions, engine=args.engine)
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._created_run = True
    score_dir = args.output_dir / "score_tensors"
    score_dir.mkdir()
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "scope": "synthetic" if args.synthetic else "formal_MedPIC_frozen_test", "engine": args.engine,
                "model_inference": args.engine == "qwen", "config_lock_sha256": sha_file(args.config_lock),
                "source_hashes": source_manifest(), "runner_source_hashes": runner_source_hashes(),
                "input_manifest": input_manifest, "seed": args.seed, "declared_seeds": [0, 1, 2],
                "methods": methods, "expected_prediction_rows": len(rows) * len(methods),
                "fusion_parameters": asdict(fusion), "fusion_condition_sources": fusion_sources,
                "fusion_provenance": lock.get("fusion", {}).get("provenance"),
                "frozen_input_copied_without_modification": True,
                "same_output_budget_not_equal_forward_count": "fusion uses two full-vocabulary paths; all actual forward costs reported"}
    write_json(args.output_dir / "run.json", manifest)
    write_json(args.output_dir / "config_lock.copy.json", lock)
    adapter = None
    if args.engine == "fake":
        infer, baseline = fake_callbacks(fusion)
        manifest["runtime"] = {"backend": "synthetic_fake_no_model", "GPU_allocated": False}
    else:
        if args.seka_root is None:
            raise ValueError("Qwen execution requires the pinned existing SEKA checkout")
        adapter, infer, baseline, manifest["runtime"] = qwen_runtime(args, lock, config, fusion)
    write_json(args.output_dir / "run.json", manifest)
    files = {key: (args.output_dir / name).open("x") for key, name in {
        "predictions": "predictions.jsonl", "traces": "traces.private.jsonl",
        "costs": "costs.jsonl", "conditions": "conditions.private.jsonl",
        "events": "call_events.private.jsonl"}.items()}
    counts, trace_statuses, written = Counter(), Counter(), 0
    started = time.perf_counter()
    try:
        for row in rows:
            sample_id = str(row["instance_id"])
            tensor_records = {}
            def persist_call(entry, tensors):
                tensor_record = None
                if tensors:
                    stem = hashlib.sha256(json.dumps([sample_id, args.seed, entry["trace_id"]]).encode()).hexdigest()
                    path = score_dir / (stem + ".pt")
                    torch.save(tensors, path)
                    tensor_record = {"path": str(path.relative_to(args.output_dir)), "sha256": sha_file(path)}
                    tensor_records[entry["trace_id"]] = tensor_record
                files["events"].write(json.dumps({"sample_id": sample_id, "seed": args.seed,
                    "score_tensor_file": tensor_record, **entry}, ensure_ascii=False) + "\n")
                files["events"].flush()
            try:
                task = native_input(row)
                if adapter is not None:
                    evidence = option_evidence(task)
                    order = [key for key, _ in task.options]
                    def build_control(prepared):
                        target = prepared.evidence.spans[0].rule_id if prepared.stage == "condition" and prepared.evidence.spans else None
                        return attach_controls(prepared, adapter, evidence, order, target=target)
                    infer = seka_infer(adapter, config, control_builder=build_control)
                bundle = run_case(task, sample_id=sample_id, seed=args.seed, config=config,
                                  lock_sha256=manifest["config_lock_sha256"], infer=infer,
                                  generate_baseline=baseline, fusion_sources=fusion_sources,
                                  fusion_base_temperature=fusion.temperature_base, on_trace=persist_call)
                bundle.pop("score_tensors")  # Already persisted per call, no duplicated tensor files.
                for trace in bundle["private_traces"]:
                    trace_statuses[trace["generation"]["status"]] += 1
                    files["traces"].write(json.dumps({"sample_id": sample_id, "seed": args.seed,
                                                      "score_tensor_file": tensor_records.get(trace["trace_id"]),
                                                      **trace}, ensure_ascii=False) + "\n")
                files["conditions"].write(json.dumps({"sample_id": sample_id, "seed": args.seed,
                                                       "assessments": bundle["assessments"]}, ensure_ascii=False) + "\n")
                files["costs"].write(json.dumps({"sample_id": sample_id, "seed": args.seed,
                                                 **bundle["cost_summary"]}, ensure_ascii=False) + "\n")
                predictions = bundle["predictions"]
            except Exception as error:
                # Every case/method stays in the full denominator on outer failures.
                predictions = [{"sample_id": sample_id, "seed": args.seed, "method": method,
                                "prediction": None, "parse_valid": False, "status": "error",
                                "error_type": type(error).__name__, "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                                "config_lock_sha256": manifest["config_lock_sha256"]} for method in methods]
                files["traces"].write(json.dumps({"sample_id": sample_id, "seed": args.seed,
                    "outer_error": {"type": type(error).__name__, "message": str(error)},
                    "note": "Case retained; partial cost may be unavailable; never replaced with zero"}) + "\n")
                trace_statuses["outer_error"] += 1
            for prediction in predictions:
                counts[prediction["status"]] += 1
                files["predictions"].write(json.dumps(prediction, ensure_ascii=False) + "\n")
                written += 1
            for handle in files.values():
                handle.flush()
            manifest.update(completed_cases=written // len(methods), written_prediction_rows=written,
                            prediction_status_counts=dict(counts), trace_status_counts=dict(trace_statuses))
            write_json(args.output_dir / "run.json", manifest)
        complete = written == manifest["expected_prediction_rows"]
        manifest.update(status="complete_verified" if complete and not counts["error"] and not trace_statuses["error"] and not trace_statuses["outer_error"] else "failed",
                        execution_finished=True, coverage_complete=complete,
                        finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
                        task_accuracy="not_evaluated_by_this_runner", clinical_claim="none")
        write_json(args.output_dir / "run.json", manifest)
    finally:
        for handle in files.values():
            handle.close()
    return manifest


if __name__ == "__main__":
    args = cli()
    try:
        outcome = main(args)
        print(json.dumps({"status": outcome["status"], "scope": outcome["scope"],
                          "rows": outcome["written_prediction_rows"], "output_dir": str(args.output_dir)}))
        sys.exit(0 if outcome["status"] == "complete_verified" else 1)
    except Exception as error:
        if getattr(args, "_created_run", False):
            write_json(args.output_dir / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
            run_path = args.output_dir / "run.json"
            if run_path.exists():
                failed = json.loads(run_path.read_text())
                failed.update(status="failed", execution_finished=False, failure_file="failure.json")
                write_json(run_path, failed)
        raise
