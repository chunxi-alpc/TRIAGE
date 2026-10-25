"""Additional native MEGA development baselines reusing frozen rule/TFU outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_mega_development import file_sha, write_json
from native_generation_baselines import (FusionConfig, cropped_guideline,
                                          direct_action_mapping, generate_native)


METHODS = ["text_two_stage", "text_emphasis", "top1_crop", "top2_crop", "applicable_crop",
           "fusion_static", "fusion_dynamic", "fusion_static_zero", "fusion_dynamic_zero",
           "cad", "cad_zero", "cad_reverse", "spotlight_no_hook", "spotlight", "spotlight_zero",
           "spotlight_wrong", "spotlight_reverse", "direct_action_mapping"]


class MeasuredFailure(RuntimeError):
    def __init__(self, error, measurement):
        super().__init__(f"{type(error).__name__}: {error}")
        self.measurement = measurement


def cli():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", required=True, nargs="+", choices=METHODS)
    parser.add_argument("--fusion-config", type=Path,
                        help="Optional previously frozen development-calibration artifact; never fit here")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="Required with fusion-config; verifies disjoint train_val guideline groups")
    parent = Path(__file__).resolve().parent.parent
    parser.add_argument("--adapter-dir", type=Path, default=parent / "method_agent")
    parser.add_argument("--reader-dir", type=Path, default=parent / "data_audit")
    parser.add_argument("--baseline-dir", type=Path, default=parent / "baseline_adapters")
    return parser.parse_args()


def main(args):
    for path in (args.adapter_dir, args.reader_dir, args.baseline_dir):
        sys.path.insert(0, str(path.resolve()))
    from aster_seka_adapter import PredictedSpans
    from chunked_spotlight import generate_chunked_spotlight
    from clinical_core import (SELECTOR_VERSION, chat_prompt, official_prompt, parse_quoted_rules_individually,
                               sha, support_spans)
    from mega_trainval_reader import OFFICIAL_COMMIT, TEMPLATES, iter_method_tasks, render_official_prompt

    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Do not duplicate baseline methods")
    primary = json.loads((args.primary_run / "protocol.json").read_text())
    manifest = json.loads(args.input_manifest.read_text())
    if primary["seed"] != 0 or primary["input_manifest_sha256"] != file_sha(args.input_manifest):
        raise ValueError("Primary run must be the identical frozen seed-0 development input")
    if primary["model_fingerprint"] != args.model_fingerprint:
        raise ValueError("Primary and additional baseline checkpoint fingerprints differ")
    if primary.get("reference_files_read") or primary.get("test_files_read"):
        raise ValueError("Primary artifacts must originate from the label-free development method")
    if subprocess.check_output(["git", "-C", str(args.official_root), "rev-parse", "HEAD"], text=True).strip() != OFFICIAL_COMMIT:
        raise ValueError("Official MEGA source revision changed")
    tasks = list(iter_method_tasks(args.dataset_root, manifest))
    max_new, context_limit = primary["max_new_tokens"], primary["max_context_tokens"]
    config = FusionConfig()
    origin = {"kind": "historical_parameter_transfer_not_native_calibration",
              "source": "TRIAGE PR33 a8c1e49c0fbda941282daf25908004407eb0b1bf",
              "native_scope": "full-vocabulary two-generation-path extension; not old option-support reproduction"}
    if args.fusion_config:
        supplied = json.loads(args.fusion_config.read_text())
        config = FusionConfig(**supplied["parameters"])
        origin = supplied["provenance"]
        if not origin.get("calibration_manifest_sha256") or origin.get("scope") != "train_val_calibration":
            raise ValueError("A native calibration artifact must name its separate train_val calibration manifest")
        if origin["calibration_manifest_sha256"] == file_sha(args.input_manifest):
            raise ValueError("Cannot treat this development evaluation batch as separate calibration")
        if not args.calibration_manifest or file_sha(args.calibration_manifest) != origin["calibration_manifest_sha256"]:
            raise ValueError("Supply the actual frozen calibration manifest named by the configuration")
        calibration = json.loads(args.calibration_manifest.read_text())
        if calibration.get("split") != "train_val" or calibration["config"]["partition"] != "calibration":
            raise ValueError("Fusion calibration must use the separately frozen train_val calibration partition")
        if {r["source_cluster"] for r in calibration["selected"]} & {r["source_cluster"] for r in manifest["selected"]}:
            raise ValueError("Fusion calibration and this evaluation batch share guideline groups")
    config.validate()
    stage_path = args.primary_run / "stage_predictions.private.jsonl"
    case_artifacts = {}
    with stage_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            case = case_artifacts.setdefault(row["sample_id"], {"A0": {}})
            if row["stage_configuration"] == "guideline_extraction":
                case["cache_key"] = row["cache_key"]
            if row["stage_configuration"] == "A0" and row.get("prediction"):
                prediction = row["prediction"]
                if prediction["rule_id"] in case["A0"]:
                    raise ValueError("Duplicate primary condition prediction")
                case["A0"][prediction["rule_id"]] = prediction
    original_path = args.primary_run / "A0B0.predictions.private.jsonl"
    cached_two_stage = {row["sample_id"]: row for row in map(json.loads, original_path.read_text().splitlines())}
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._created_run = True
    protocol = {"task_state": "running", "execution_status": "started", "seed": 0,
                "scope": "train_val native-path development only", "methods": args.methods,
                "model_fingerprint": args.model_fingerprint, "input_manifest_sha256": file_sha(args.input_manifest),
                "primary_protocol_sha256": file_sha(args.primary_run / "protocol.json"),
                "primary_stages_sha256": file_sha(stage_path), "primary_A0B0_sha256": file_sha(original_path),
                "max_new_tokens": max_new, "max_context_tokens": context_limit,
                "fusion_parameters": asdict(config), "fusion_provenance": origin,
                "CAD_alpha": 1., "Spotlight_target": .1, "Spotlight_backend": "query_chunked_eager",
                "Spotlight_query_chunk": 128,
                "other_generation_backend": "sdpa", "thinking": False, "chat_role": "system",
                "runtime_rope_scaling": primary.get("runtime_rope_scaling"),
                "expected_cases": len(tasks), "expected_rows": len(tasks) * len(args.methods),
                "source_hashes": {p.name: file_sha(p) for p in [Path(__file__), Path(__file__).with_name("native_generation_baselines.py"),
                                   Path(__file__).with_name("chunked_spotlight.py"),
                                   Path(__file__).with_name("clinical_core.py"), args.baseline_dir / "baseline_operators.py"]},
                "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers"]},
                "reference_files_read": False, "test_files_read": False,
                "direct_mapping_scope": "one exception-free positive quoted rule; no reconstructed tree edges"}
    protocol["additional_eager_reference_generation"] = any(m.startswith("spotlight") for m in args.methods) and "spotlight_no_hook" not in args.methods
    protocol["additional_plain_first_token_probe"] = any(m.endswith("_zero") and not m.startswith("spotlight") for m in args.methods)
    write_json(args.output_dir / "protocol.json", protocol)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one coordinator-assigned GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    runtime_config = {"rope_scaling": primary["runtime_rope_scaling"]} if primary.get("runtime_rope_scaling") is not None else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", attn_implementation="sdpa", **runtime_config).to("cuda").eval()
    write_json(args.output_dir / "runtime_model_config.json", model.config.to_dict())
    protocol["runtime_model_config_sha256"] = file_sha(args.output_dir / "runtime_model_config.json")
    write_json(args.output_dir / "protocol.json", protocol)
    if sha(tokenizer.chat_template) != primary["chat_template_sha256"]:
        raise ValueError("Chat template differs from the primary clinical experiment")
    output_files = {method: (args.output_dir / f"{method}.predictions.private.jsonl").open("x") for method in args.methods}
    numeric_file = (args.output_dir / "numeric_probes.private.jsonl").open("x")
    counts = Counter()

    def measured(call):
        forwards = [0]
        def count_forward(*unused):
            forwards[0] += 1
        hook = model.register_forward_hook(count_forward)
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        try:
            result = call()
        except Exception as error:
            measurement = {"actual_forward_calls": forwards[0], "elapsed_seconds": time.perf_counter() - start,
                           "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 2**30,
                           "peak_reserved_gib": torch.cuda.max_memory_reserved(model.device) / 2**30,
                           **getattr(error, "partial_generation", {})}
            raise MeasuredFailure(error, measurement) from error
        finally:
            hook.remove()
        torch.cuda.synchronize(model.device)
        if result.get("actual_forward_calls", forwards[0]) != forwards[0]:
            raise ValueError("Instrumented forward count mismatch")
        result.update(actual_forward_calls=forwards[0], elapsed_seconds=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(model.device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(model.device) / 2**30)
        return result

    try:
        for identity, task in tasks:
            completed = set()
            try:
                artifact = case_artifacts[identity.sample_id]
                cache_path = args.primary_run / "private_guideline_extractions" / (artifact["cache_key"] + ".json")
                extraction = json.loads(cache_path.read_text())
                if extraction["guideline_sha256"] != sha(task.guideline):
                    raise ValueError("Cached extraction belongs to a different guideline")
                rules = parse_quoted_rules_individually(extraction["generation"]["text"], task.guideline)[0] if extraction["validation_status"] in {"quotes_validated", "quotes_partially_validated"} else ()
                tfu = [artifact["A0"][rule.rule_id] for rule in rules]
                template = (args.official_root / TEMPLATES["zh" if task.language == "zh" else "en"]).read_text()
                if official_prompt(template, task.guideline, task.vignette).text != render_official_prompt(task, args.official_root, manifest):
                    raise ValueError("Official native question/template mismatch")
                selected = support_spans(rules, tfu)
                evidence = PredictedSpans(sha(SELECTOR_VERSION), selected)
                base_plain = official_prompt(template, task.guideline, task.vignette, evidence, tfu)
                base = chat_prompt(base_plain, tokenizer)
                if cached_two_stage[identity.sample_id].get("generation", {}).get("prompt_sha256") != sha(base.text):
                    raise ValueError("Reconstructed full-information prompt differs from primary A0B0")
                passages = [task.guideline[span.start:span.end] for span in selected]
                support_plain = replace(base_plain, text=base_plain.text + "\nAttend to these predicted relevant quotations (unverified):\n" + "\n\n".join(passages)) if passages else base_plain
                support = chat_prompt(support_plain, tokenizer)
                prior = chat_prompt(official_prompt(template, "", task.vignette, tfu=tfu), tokenizer)
                reference_log_probs, reference_raw = None, None
                if any(method.endswith("_zero") and not method.startswith("spotlight") for method in args.methods):
                    try:
                        probe = measured(lambda: generate_native(model, tokenizer, base.text, None, kind="plain",
                                                                 max_new_tokens=1, max_context_tokens=context_limit))
                        reference_log_probs = probe.pop("first_log_probs")
                        reference_raw = probe.pop("first_raw_base")
                    except Exception as error:
                        probe = {"status": "error", "error": str(error), **getattr(error, "measurement", {})}
                    numeric_file.write(json.dumps({"sample_id": identity.sample_id, "purpose": "one-token no-fusion numerical reference",
                                                   "prompt_sha256": sha(base.text), **probe}, ensure_ascii=False) + "\n")
                    numeric_file.flush()
                eager_reference = None
                # The identical query-chunked no-steering reference comes first.
                requested = list(args.methods)
                if any(m.startswith("spotlight") for m in requested) and "spotlight_no_hook" not in requested:
                    requested.insert(0, "spotlight_no_hook")
                elif "spotlight_no_hook" in requested:
                    requested.remove("spotlight_no_hook")
                    requested.insert(0, "spotlight_no_hook")
                for method in requested:
                    try:
                        crop_meta = None
                        if method == "text_two_stage":
                            source = cached_two_stage[identity.sample_id]
                            result = {"text": source["llm_output"], "status": source["status"],
                                      "actual_forward_calls": 0, "elapsed_seconds": 0.,
                                      "reused_from": str(original_path), "reused_source_sha256": file_sha(original_path)}
                        elif method == "direct_action_mapping":
                            text, reason = direct_action_mapping(rules, tfu, task.language)
                            if text is not None and len(tokenizer.encode(text, add_special_tokens=False)) > max_new:
                                text, reason = None, "direct action quotation exceeds the same output-token budget"
                            result = {"text": text or "", "status": "ok" if text is not None else "mapping_unresolved",
                                      "reason": reason, "actual_forward_calls": 0, "elapsed_seconds": 0.,
                                      "symbolic_tree_execution": False, "unverified_extracted_mapping": True}
                        elif method.startswith("spotlight"):
                            control = {"spotlight": "normal", "spotlight_no_hook": "no_hook", "spotlight_zero": "matched_zero",
                                       "spotlight_wrong": "wrong_span", "spotlight_reverse": "reverse"}[method]
                            length = len(tokenizer.encode(base.text, add_special_tokens=True))
                            if length + max_new > context_limit:
                                raise ValueError("Full Spotlight prompt exceeds frozen context budget")
                            if not selected:
                                control = "no_hook"
                            def spotlight_call():
                                return generate_chunked_spotlight(model, tokenizer, base.text, base.absolute_spans(selected),
                                                                 target=.1, control=control, query_chunk=128,
                                                                 max_new_tokens=max_new, max_context_tokens=context_limit)
                            result = measured(spotlight_call)
                            score = result.pop("first_log_probs")
                            if method == "spotlight_no_hook":
                                eager_reference = score
                            if method == "spotlight_zero" and eager_reference is not None:
                                result["numerical_control"] = {"reference": "same query-chunked kernel, bias disabled, identical prompt",
                                    "max_abs_first_token_logprob_difference": float((score - eager_reference).abs().max()),
                                    "argmax_changed": bool(score.argmax() != eager_reference.argmax())}
                            result["no_support_fallback"] = not bool(selected)
                            if method not in args.methods:
                                numeric_file.write(json.dumps({"sample_id": identity.sample_id, "purpose": "matching eager reference",
                                                               **result}, ensure_ascii=False) + "\n")
                                numeric_file.flush()
                                continue
                        else:
                            prompt, auxiliary, kind, control = base.text, None, "plain", "normal"
                            if method == "text_emphasis":
                                prompt = support.text
                            elif method.endswith("_crop"):
                                cropped, crop_meta = cropped_guideline(task.guideline, rules, tfu, method[:-5])
                                prompt = chat_prompt(official_prompt(template, cropped, task.vignette, tfu=tfu), tokenizer).text
                            elif method.startswith("fusion_"):
                                kind = "dynamic" if "dynamic" in method else "static"
                                auxiliary = support.text
                                control = "matched_zero" if method.endswith("_zero") else "normal"
                            elif method.startswith("cad"):
                                kind, auxiliary = "cad", prior.text
                                control = {"cad": "normal", "cad_zero": "matched_zero", "cad_reverse": "reverse"}[method]
                            result = measured(lambda: generate_native(model, tokenizer, prompt, auxiliary, kind=kind, config=config,
                                                                       control=control, max_new_tokens=max_new, max_context_tokens=context_limit))
                            score, raw = result.pop("first_log_probs"), result.pop("first_raw_base")
                            if method.endswith("_zero") and reference_log_probs is not None:
                                reference = (reference_raw / config.temperature_base).log_softmax(-1) if method.startswith("fusion") else reference_log_probs
                                result["numerical_control"] = {"reference": "same primary prompt, declared base temperature",
                                    "max_abs_first_token_logprob_difference": float((score - reference).abs().max()),
                                    "argmax_changed": bool(score.argmax() != reference.argmax())}
                            result.update(primary_prompt_sha256=sha(prompt), auxiliary_prompt_sha256=sha(auxiliary) if auxiliary else None,
                                          crop_metadata=crop_meta)
                        result.setdefault("status", "ok")
                    except Exception as error:
                        result = {"text": "", "status": "error", "error_type": type(error).__name__, "error": str(error),
                                  **getattr(error, "measurement", {})}
                    if method not in args.methods:
                        numeric_file.write(json.dumps({"sample_id": identity.sample_id, "purpose": "matching eager reference failed", **result}) + "\n")
                        numeric_file.flush()
                        continue
                    row = {**asdict(identity), "language": task.language, "sample_id": identity.sample_id,
                           "method": method, "model": args.model, "llm_output": result.get("text", ""),
                           "status": result["status"], "generation": result, "seed": 0,
                           "input_manifest_sha256": protocol["input_manifest_sha256"],
                           "extraction_cache_sha256": file_sha(cache_path), "extraction_status": extraction["validation_status"],
                           "predicted_support_span_count": len(selected), "no_predicted_support": not bool(selected),
                           "cost_references": {"primary_stage_file_sha256": protocol["primary_stages_sha256"],
                                               "extraction_cache_key": artifact["cache_key"], "condition_configuration": "A0",
                                               "include_shared_extraction_and_condition_cost": True},
                           "native_output_parse": "not_run", "judge_evaluation": "not_run"}
                    output_files[method].write(json.dumps(row, ensure_ascii=False) + "\n")
                    output_files[method].flush()
                    counts[f"{method}:{row['status']}"] += 1
                    completed.add(method)
            except Exception as error:
                for method in args.methods:
                    if method in completed:
                        continue
                    row = {**asdict(identity), "language": task.language, "sample_id": identity.sample_id,
                           "method": method, "model": args.model, "llm_output": "", "status": "error",
                           "error_type": type(error).__name__, "error": str(error), "preparation_failed": True}
                    output_files[method].write(json.dumps(row, ensure_ascii=False) + "\n")
                    output_files[method].flush()
                    counts[f"{method}:error"] += 1
            protocol["outcome_counts"] = dict(counts)
            write_json(args.output_dir / "protocol.json", protocol)
        protocol.update(execution_status="finished", verification_pending=True, written_rows=sum(counts.values()))
    finally:
        for file in output_files.values():
            file.close()
        numeric_file.close()
        write_json(args.output_dir / "protocol.json", protocol)


if __name__ == "__main__":
    arguments = cli()
    try:
        main(arguments)
    except Exception as error:
        if getattr(arguments, "_created_run", False):
            path = arguments.output_dir / "protocol.json"
            value = json.loads(path.read_text()) if path.exists() else {}
            value.update(task_state="blocked", execution_status="failed", error_type=type(error).__name__, error=str(error))
            write_json(path, value)
        raise
