"""Native MEGA attention baselines using frozen primary A0 TFU and evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from author_attention import generate_pasta_native, make_adaseka, routing_last_token_logits


PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"
METHODS = ["seka_execution_reuse", "pasta_no_hook", "pasta_zero", "pasta",
           "adaseka_no_hook", "adaseka_zero", "adaseka_05", "adaseka_3"]


def cli():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("dataset-root", "input-manifest", "official-root", "primary-run", "seka-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--pasta-heads", type=Path)
    parser.add_argument("--adaseka-experts", type=Path)
    parser.add_argument("--artifact-provenance", type=Path,
                        help="Coordinator's completed YaRN model/head/expert construction manifest")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--clinical-dir", type=Path, default=root / "clinical_dev")
    parser.add_argument("--adapter-dir", type=Path, default=root / "method_agent")
    parser.add_argument("--reader-dir", type=Path, default=root / "data_audit")
    return parser.parse_args()


def verify_artifact_binding(args, primary, experts, need_pasta, need_adaseka, file_sha, text_sha):
    """Bind the existing coordinator/audit schemas to actual model and artifacts."""
    if not (need_pasta or need_adaseka):
        return {"status": "not_needed_for_existing_SEKA_output_reuse"}
    if args.artifact_provenance is None:
        raise ValueError("Provide the completed clinical attention construction manifest")
    bundle = json.loads(args.artifact_provenance.read_text())
    if bundle["status"] != "complete_verified" or Path(bundle["model_path"]).resolve() != Path(args.model).resolve():
        raise ValueError("Clinical attention artifacts were not verified for this model directory")
    hashes = bundle["model_files_sha256"]
    mapping_digest = text_sha(json.dumps(hashes, sort_keys=True))
    if mapping_digest != primary["model_fingerprint"] or "config.json" not in hashes:
        raise ValueError("Artifact checkpoint mapping differs from primary clinical model fingerprint")
    for relative, expected in hashes.items():
        if file_sha(Path(args.model) / relative) != expected:
            raise ValueError("Actual clinical model file differs from construction manifest: " + relative)
    config = json.loads((Path(args.model) / "config.json").read_text())
    if config.get("rope_scaling") != primary.get("runtime_rope_scaling"):
        raise ValueError("Artifact model config and primary runtime YaRN configuration differ")
    checked = {"status": "complete_verified", "model_mapping_sha256": mapping_digest,
               "actual_model_config_sha256": hashes["config.json"], "rope_scaling": config.get("rope_scaling")}
    if need_pasta:
        item = bundle["pasta"]
        if Path(item["path"]).resolve() != args.pasta_heads.resolve() or file_sha(args.pasta_heads) != item["sha256"]:
            raise ValueError("Actual PASTA heads differ from construction manifest")
        if file_sha(item["run_manifest_path"]) != item["run_manifest_sha256"]:
            raise ValueError("PASTA construction run manifest changed")
        build = json.loads(Path(item["run_manifest_path"]).read_text())
        if (build.get("process_state") != "exited" or build.get("exit_code") != 0
                or build.get("model_files_sha256") != hashes or build.get("source_commit") != PIN):
            raise ValueError("PASTA completed construction does not match the actual clinical checkpoint/source")
        checked["pasta"] = item
    if need_adaseka:
        item = bundle["adaseka"]
        if (Path(item["expert_config_path"]).resolve() != args.adaseka_experts.resolve()
                or file_sha(args.adaseka_experts) != item["expert_config_sha256"]
                or file_sha(item["audit_path"]) != item["audit_sha256"]):
            raise ValueError("AdaSEKA expert configuration or completed audit changed")
        audit = json.loads(Path(item["audit_path"]).read_text())
        if (audit.get("status") != "complete_verified" or audit.get("config_ready") is not True
                or audit.get("expert_config_sha256") != item["expert_config_sha256"]
                or set(audit["experts"]) != set(experts)):
            raise ValueError("AdaSEKA four-expert audit is not complete for this configuration")
        for name, path in experts.items():
            detail = audit["experts"][name]
            if (Path(detail["path"]).resolve() != Path(path).resolve() or file_sha(path) != detail["sha256"]
                    or detail["checkpoint_files_sha256"] != hashes
                    or file_sha(detail["manifest_path"]) != detail["manifest_sha256"]):
                raise ValueError("AdaSEKA expert provenance differs from actual clinical model: " + name)
        checked["adaseka"] = item
    return checked


def main(args):
    for path in (args.clinical_dir, args.adapter_dir, args.reader_dir, args.seka_root):
        sys.path.insert(0, str(path.resolve()))
    from run_mega_development import file_sha, write_json
    from clinical_core import (SELECTOR_VERSION, chat_prompt, official_prompt,
                               parse_quoted_rules_individually, sha, support_spans)
    from aster_seka_adapter import Control, PhasePolicy, PredictedSpans
    from mega_trainval_reader import OFFICIAL_COMMIT, TEMPLATES, iter_method_tasks, render_official_prompt
    from pastalib.pasta import read_head_config

    primary = json.loads((args.primary_run / "protocol.json").read_text())
    manifest = json.loads(args.input_manifest.read_text())
    if primary["seed"] != 0 or primary["input_manifest_sha256"] != file_sha(args.input_manifest):
        raise ValueError("Use the same frozen primary seed-0 development batch")
    if primary["model_fingerprint"] != args.model_fingerprint or primary["model_dtype"] != "torch.bfloat16":
        raise ValueError("Clinical attention baselines must match the primary BF16 model fingerprint")
    for directory, expected in [(args.seka_root, PIN), (args.official_root, OFFICIAL_COMMIT)]:
        if subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip() != expected:
            raise ValueError("Pinned author source revision changed")
    need_pasta = any(name.startswith("pasta") for name in args.methods)
    need_adaseka = any(name.startswith("adaseka") for name in args.methods)
    if need_pasta and not args.pasta_heads:
        raise ValueError("Provide the actual YaRN-compatible author PASTA head configuration")
    if need_adaseka and not args.adaseka_experts:
        raise ValueError("Provide the actual four-expert YaRN AdaSEKA configuration")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate method names")
    heads = read_head_config(str(args.pasta_heads)) if need_pasta else None
    experts = {}
    if need_adaseka:
        raw = json.loads(args.adaseka_experts.read_text())
        if not isinstance(raw, dict) or len(raw) != 4:
            raise ValueError("This comparison requires the real four-expert bank, not a single-expert substitute")
        experts = {name: str((args.adaseka_experts.parent / path).resolve()) if not Path(path).is_absolute() else path
                   for name, path in raw.items()}
    artifact_binding = verify_artifact_binding(args, primary, experts, need_pasta, need_adaseka, file_sha, sha)
    tasks = list(iter_method_tasks(args.dataset_root, manifest))
    stage_path = args.primary_run / "stage_predictions.private.jsonl"
    cases = {}
    for line in stage_path.read_text().splitlines():
        record = json.loads(line)
        case = cases.setdefault(record["sample_id"], {"A0": {}})
        if record["stage_configuration"] == "guideline_extraction":
            case["cache_key"] = record["cache_key"]
        elif record["stage_configuration"] == "A0" and record.get("prediction"):
            prediction = record["prediction"]
            case["A0"][prediction["rule_id"]] = prediction
    source_path = args.primary_run / "A0B1.predictions.private.jsonl"
    source_rows = {row["sample_id"]: row for row in map(json.loads, source_path.read_text().splitlines())}
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._created_run = True
    requested = list(args.methods)
    # Both references are genuinely run if needed, with their extra costs saved.
    execution_order = list(requested)
    for prefix in ("pasta", "adaseka"):
        if any(name.startswith(prefix) for name in execution_order):
            ref = prefix + "_no_hook"
            if ref in execution_order:
                execution_order.remove(ref)
            execution_order.insert(0, ref)
    protocol = {"task_state": "running", "execution_status": "started", "scope": "MEGA train_val development",
                "seed": 0, "methods": requested, "execution_order": execution_order,
                "model_fingerprint": args.model_fingerprint, "dtype": "torch.bfloat16",
                "precision_scope": "Matched clinical BF16 comparison; differs from G1 AdaSEKA default FP32",
                "source_revision": PIN, "runtime_rope_scaling": primary.get("runtime_rope_scaling"),
                "max_new_tokens": primary["max_new_tokens"], "max_context_tokens": primary["max_context_tokens"],
                "input_manifest_sha256": file_sha(args.input_manifest), "primary_protocol_sha256": file_sha(args.primary_run / "protocol.json"),
                "primary_stage_sha256": file_sha(stage_path), "primary_A0B1_sha256": file_sha(source_path),
                "pasta_head_config_sha256": file_sha(args.pasta_heads) if need_pasta else None,
                "pasta_alpha": .01, "pasta_zero_alpha": 1., "pasta_scale_position": "exclude", "pasta_query_chunk": 128,
                "adaseka_strengths": [.5, 3.], "adaseka_zero_strength": 0.,
                "adaseka_top_k_singular": 5, "adaseka_combination_method": "weighted_top_k",
                "adaseka_temperature": 1., "adaseka_routing_parameters_origin": "Frozen G1 preset, retained before any clinical run",
                "adaseka_routing_logit_storage": "Only routing pre-forward sets logits_to_keep=1; all hidden states unchanged",
                "adaseka_expert_config_sha256": file_sha(args.adaseka_experts) if need_adaseka else None,
                "adaseka_experts": {name: {"path": path, "sha256": file_sha(path)} for name, path in experts.items()},
                "artifact_provenance_sha256": file_sha(args.artifact_provenance) if args.artifact_provenance else None,
                "artifact_binding": artifact_binding,
                "source_hashes": {p.name: file_sha(p) for p in [Path(__file__), Path(__file__).with_name("author_attention.py"),
                                   args.adapter_dir / "aster_seka_adapter.py", args.clinical_dir / "clinical_core.py",
                                   args.seka_root / "pastalib/pasta.py", args.seka_root / "src/model/adaptive_seka_llm.py"]},
                "expected_cases": len(tasks), "expected_requested_rows": len(tasks) * len(requested),
                "test_files_read": False, "reference_files_read": False}
    write_json(args.output_dir / "protocol.json", protocol)
    operator = adapter = model = None
    rope = primary.get("runtime_rope_scaling")
    if need_pasta or need_adaseka:
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one coordinator-assigned GPU")
        if need_adaseka:
            operator, adapter = make_adaseka(args.model, experts, device="cuda", source_commit=PIN, rope_scaling=rope)
            model, tokenizer = operator.model, operator.tok
            protocol["loaded_expert_manifest"] = adapter.projections
        else:
            settings = {"rope_scaling": rope} if rope is not None else {}
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                        attn_implementation="sdpa", **settings).to("cuda").eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        write_json(args.output_dir / "runtime_model_config.json", model.config.to_dict())
        protocol["runtime_model_config_sha256"] = file_sha(args.output_dir / "runtime_model_config.json")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    write_json(args.output_dir / "protocol.json", protocol)
    files = {name: (args.output_dir / f"{name}.predictions.private.jsonl").open("x") for name in requested}
    extra = (args.output_dir / "additional_references.private.jsonl").open("x")
    counts = Counter()

    def measured(function):
        calls = {"forward_attempts": 0, "forward_completed": 0,
                 "routing_forward_attempts": 0, "routing_forward_completed": 0}
        def routing_call(keyword):
            return keyword.get("output_hidden_states") is True and keyword.get("use_cache") is False
        def count_attempt(module, positional, keyword):
            calls["forward_attempts"] += 1
            calls["routing_forward_attempts"] += int(routing_call(keyword))
        def count_completed(module, positional, keyword, output):
            calls["forward_completed"] += 1
            calls["routing_forward_completed"] += int(routing_call(keyword))
        torch.cuda.reset_peak_memory_stats(model.device)
        before = model.register_forward_pre_hook(count_attempt, with_kwargs=True)
        after = model.register_forward_hook(count_completed, with_kwargs=True)
        torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        try:
            result = function()
            torch.cuda.synchronize(model.device)
        except Exception as error:
            result = {"text": "", "status": "error", "error_type": type(error).__name__, "error": str(error)}
        finally:
            before.remove()
            after.remove()
            if operator is not None:
                operator.remove_projection()
        result.update(**calls, actual_forward_calls=calls["forward_completed"], elapsed_seconds=time.perf_counter() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(model.device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(model.device) / 2**30)
        return result

    try:
        for identity, task in tasks:
            completed = set()
            try:
                source = source_rows[identity.sample_id]
                info = cases[identity.sample_id]
                cache_path = args.primary_run / "private_guideline_extractions" / (info["cache_key"] + ".json")
                extraction = json.loads(cache_path.read_text())
                if extraction["guideline_sha256"] != sha(task.guideline):
                    raise ValueError("Frozen guideline extraction fingerprint mismatch")
                rules = parse_quoted_rules_individually(extraction["generation"]["text"], task.guideline)[0] if extraction["validation_status"] in {"quotes_validated", "quotes_partially_validated"} else ()
                tfu = [info["A0"][rule.rule_id] for rule in rules]
                template = (args.official_root / TEMPLATES["zh" if task.language == "zh" else "en"]).read_text()
                if official_prompt(template, task.guideline, task.vignette).text != render_official_prompt(task, args.official_root, manifest):
                    raise ValueError("Official original template changed")
                evidence = PredictedSpans(sha(SELECTOR_VERSION), support_spans(rules, tfu))
                prompt = chat_prompt(official_prompt(template, task.guideline, task.vignette, evidence, tfu), tokenizer)
                if source.get("generation", {}).get("prompt_sha256") != sha(prompt.text):
                    raise ValueError("A0 full-information prompt differs from the cached SEKA execution baseline")
                references = {}
                for method in execution_order:
                    if method == "seka_execution_reuse":
                        result = {"text": source["llm_output"], "status": source["status"], "actual_forward_calls": 0,
                                  "elapsed_seconds": 0., "reused_from": "primary A0B1", "source_sha256": protocol["primary_A0B1_sha256"],
                                  "original_generation": source.get("generation"), "new_model_run": False}
                    elif method.startswith("pasta"):
                        control = {"pasta": "normal", "pasta_zero": "matched_zero", "pasta_no_hook": "no_hook"}[method]
                        result = measured(lambda: generate_pasta_native(model, tokenizer, prompt, heads, control=control,
                                                                        max_new_tokens=primary["max_new_tokens"],
                                                                        max_context_tokens=primary["max_context_tokens"]))
                    else:
                        strength = 3. if method == "adaseka_3" else .5
                        control = Control.NO_HOOK if method == "adaseka_no_hook" else (Control.MATCHED_ZERO if method == "adaseka_zero" else Control.NORMAL)
                        def ada_call():
                            operator.set_amplify_factor(strength)
                            ids, _, _ = adapter._encode(prompt, prompt.evidence.spans)
                            if ids.numel() + primary["max_new_tokens"] > primary["max_context_tokens"]:
                                raise ValueError("Complete AdaSEKA input exceeds the frozen context budget")
                            with routing_last_token_logits(model) as routing:
                                value = adapter.generate(prompt, PhasePolicy(False, True), control=control,
                                                         max_new_tokens=primary["max_new_tokens"], generation_seed=0,
                                                         capture_first_token_scores=True)
                            value["routing_storage_adaptation"] = dict(routing)
                            value["requested_strength"] = strength
                            value["status"] = ("control_unavailable" if control == Control.MATCHED_ZERO else "no_support") if value["no_support_fallback"] and control != Control.NO_HOOK else "ok"
                            return value
                        result = measured(ada_call)
                    scores = result.pop("first_token_scores", None)
                    family = "pasta" if method.startswith("pasta") else "adaseka"
                    if method.endswith("_no_hook") and scores is not None:
                        references[family] = {"scores": scores, "input_ids_sha256": result.get("input_ids_sha256")}
                    if method.endswith("_zero") and scores is not None and family in references:
                        reference = references[family]
                        actual_input_hash = result.get("input_ids_sha256")
                        result["numerical_control"] = {"reference": family + "_no_hook",
                            "same_full_input": bool(actual_input_hash) and actual_input_hash == reference["input_ids_sha256"],
                            "actual_input_ids_sha256": actual_input_hash, "reference_input_ids_sha256": reference["input_ids_sha256"],
                            "max_abs_first_token_score_difference": float((scores - reference["scores"]).abs().max()),
                            "argmax_changed": bool(scores.argmax() != reference["scores"].argmax())}
                    result.setdefault("status", "ok")
                    row = {**asdict(identity), "language": task.language, "sample_id": identity.sample_id,
                           "method": method, "model": args.model, "llm_output": result.get("text", ""),
                           "status": result["status"], "generation": result, "generation_seed": 0,
                           "prompt_sha256": sha(prompt.text), "extraction_cache_sha256": file_sha(cache_path),
                           "cost_references": {"primary_stage_sha256": protocol["primary_stage_sha256"],
                                               "condition_configuration": "A0", "extraction_cache_key": info["cache_key"],
                                               "include_shared_primary_cost": True},
                           "native_output_parse": "not_run", "judge_evaluation": "not_run"}
                    target = files[method] if method in files else extra
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    target.flush()
                    if method in files:
                        counts[f"{method}:{row['status']}"] += 1
                        completed.add(method)
            except Exception as error:
                for method in requested:
                    if method in completed:
                        continue
                    row = {**asdict(identity), "language": task.language, "sample_id": identity.sample_id,
                           "method": method, "model": args.model, "llm_output": "", "status": "error",
                           "error_type": type(error).__name__, "error": str(error), "preparation_failed": True}
                    files[method].write(json.dumps(row, ensure_ascii=False) + "\n")
                    files[method].flush()
                    counts[f"{method}:error"] += 1
            protocol["outcome_counts"] = dict(counts)
            write_json(args.output_dir / "protocol.json", protocol)
        protocol.update(execution_status="finished", verification_pending=True, written_rows=sum(counts.values()))
    finally:
        for handle in files.values():
            handle.close()
        extra.close()
        write_json(args.output_dir / "protocol.json", protocol)


if __name__ == "__main__":
    arguments = cli()
    try:
        main(arguments)
    except Exception as error:
        if getattr(arguments, "_created_run", False):
            sys.path.insert(0, str(arguments.clinical_dir))
            from run_mega_development import write_json
            path = arguments.output_dir / "protocol.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            state.update(task_state="blocked", execution_status="failed", error_type=type(error).__name__, error=str(error))
            write_json(path, state)
        raise
