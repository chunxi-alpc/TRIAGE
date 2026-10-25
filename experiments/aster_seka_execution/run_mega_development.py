"""Frozen MEGA train_val development only. Run on the coordinator's server GPU.

Native judge evaluation is a separate process; this program never opens its
reference trees, reference paths or answer files. All private text stays remote.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def cli():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--seka-root", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path(__file__).resolve().parent.parent / "method_agent")
    parser.add_argument("--reader-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data_audit")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--pos", required=True)
    parser.add_argument("--neg", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--extraction-max-new-tokens", type=int, required=True)
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--rope-scaling-json", type=Path,
                        help="Optional explicit runtime RoPE configuration; default preserves checkpoint configuration")
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--include-text-controls", action="store_true")
    return parser.parse_args()


def main(args):
    for path in (args.reader_dir, args.adapter_dir, args.seka_root):
        sys.path.insert(0, str(path.resolve()))
    from aster_seka_adapter import Control, PhasePolicy, PreparedPrompt, PredictedSpans, StageSEKAAdapter
    from mega_trainval_reader import (OFFICIAL_COMMIT, TEMPLATES, iter_method_tasks,
                                      render_official_prompt, value_digest)
    from clinical_core import (EXTRACTION_INSTRUCTION, EXTRACTION_PARSER_VERSION, SELECTOR_VERSION, attach_controls, chat_prompt,
                               condition_prompt, empty_evidence, official_prompt, parse_quoted_rules_individually,
                               sha, support_spans, verify_tfu_tokens)
    from src.model import SEKALLM

    if min(args.max_new_tokens, args.extraction_max_new_tokens, args.max_context_tokens) <= 0:
        raise ValueError("All budgets must be positive")
    rope_scaling = json.loads(args.rope_scaling_json.read_text()) if args.rope_scaling_json else None
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise ValueError("Runtime rope_scaling must be a JSON object")
    actual_seka = subprocess.check_output(["git", "-C", str(args.seka_root), "rev-parse", "HEAD"], text=True).strip()
    actual_mega = subprocess.check_output(["git", "-C", str(args.official_root), "rev-parse", "HEAD"], text=True).strip()
    if actual_seka != PIN or actual_mega != OFFICIAL_COMMIT:
        raise ValueError("Official source checkout differs from the registered source revision")
    manifest = json.loads(args.input_manifest.read_text())
    if manifest.get("split") != "train_val" or manifest["config"]["method_development_seed"] != 0:
        raise ValueError("Only the frozen seed-0 train_val development manifest is accepted")
    # Model code receives only task inputs, while identity is held by this logger.
    tasks = list(iter_method_tasks(args.dataset_root, manifest))
    if not tasks:
        raise ValueError("No frozen development cases")
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args._created_run = True
    cache_dir = args.output_dir / "private_guideline_extractions"
    cache_dir.mkdir()
    arms = ["full_guideline", "A0B0", "A0B1", "A1B0", "A1B1",
            "A1B1_zero", "A1B1_wrong", "A1B1_misaligned", "A1B1_reverse"]
    if args.include_text_controls:
        arms += ["text_emphasis", "predicted_crop"]
    protocol = {
        "task_state": "running", "execution_status": "started", "scope": "MEGA train_val development only",
        "model": args.model, "model_fingerprint": args.model_fingerprint,
        "source_revisions": {"SEKA": actual_seka, "MEGA": actual_mega},
        "input_manifest_sha256": file_sha(args.input_manifest),
        "source_hashes": {str(p.name): file_sha(p) for p in [
            Path(__file__), Path(__file__).with_name("clinical_core.py"),
            args.adapter_dir / "aster_seka_adapter.py", args.reader_dir / "mega_trainval_reader.py"]},
        "seed": 0, "planned_final_seeds": [0, 1, 2], "greedy": True, "thinking": False,
        "chat_role": "system", "attention_backend": "sdpa", "batch_size": 1,
        "max_new_tokens": args.max_new_tokens, "extraction_max_new_tokens": args.extraction_max_new_tokens,
        "max_context_tokens": args.max_context_tokens, "truncation": False,
        "runtime_rope_scaling": rope_scaling,
        "runtime_rope_scaling_file_sha256": file_sha(args.rope_scaling_json) if args.rope_scaling_json else None,
        "layers": "last10", "amplify_pos": 2.4, "amplify_neg": 0.,
        "projection_note": "Existing SEKA CounterFact-derived fixed setting; not clinically optimized or verified",
        "selector_version": SELECTOR_VERSION, "extraction_prompt_sha256": sha(EXTRACTION_INSTRUCTION),
        "extraction_parser_version": EXTRACTION_PARSER_VERSION,
        "quotation_validation": "Complete array items only; invalid JSON escapes become literal backslashes; unchanged verbatim-quote checks, original item IDs, repair/rejection/truncation audit",
        "arms": arms, "expected_cases": len(tasks), "expected_native_predictions": len(tasks) * len(arms),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers"]},
        "reference_files_read": False, "test_files_read": False,
        "rules_semantically_verified": False, "symbolic_executor_implemented": False,
        "native_judge_evaluation": "not_run_by_this_program",
    }
    write_json(args.output_dir / "protocol.json", protocol)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly the coordinator-assigned single GPU; upstream SEKA otherwise shards automatically")
    runtime_config = {"rope_scaling": rope_scaling} if rope_scaling is not None else {}
    operator = SEKALLM(args.model, device="cuda", pos_pt=args.pos, neg_pt=args.neg, layers="last10",
                       amplify_pos=2.4, amplify_neg=0., torch_dtype="auto", attn_implementation="sdpa", **runtime_config)
    adapter = StageSEKAAdapter(operator, kind="seka", source_commit=actual_seka,
                               projection_paths=[args.pos, args.neg])
    label_ids = verify_tfu_tokens(operator.tok)
    protocol.update(tfu_label_token_ids=label_ids, model_dtype=str(next(operator.model.parameters()).dtype),
                    projection_manifest=adapter.projections, generation_config=operator.model.generation_config.to_dict(),
                    chat_template_sha256=sha(operator.tok.chat_template))
    write_json(args.output_dir / "runtime_model_config.json", operator.model.config.to_dict())
    protocol["runtime_model_config_sha256"] = file_sha(args.output_dir / "runtime_model_config.json")
    write_json(args.output_dir / "protocol.json", protocol)
    stage_file = (args.output_dir / "stage_predictions.private.jsonl").open("x")
    native_files = {arm: (args.output_dir / f"{arm}.predictions.private.jsonl").open("x") for arm in arms}
    extraction_cache = {}
    guide_case_counts = Counter(sha(task.guideline) for _, task in tasks)
    outcome_counts = Counter()
    rows, run_start = 0, time.perf_counter()

    def infer(prompt, policy, control, budget, capture=True):
        calls, handle, result, scores = [0], None, {}, None
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(operator.device)
        try:
            input_ids, _, _ = adapter._encode(prompt, prompt.evidence.spans)
            if input_ids.numel() + budget > args.max_context_tokens:
                raise ValueError(f"Input plus generation budget exceeds frozen context limit: {input_ids.numel()} + {budget}")
            def count_forward(*unused):
                calls[0] += 1

            handle = operator.model.register_forward_hook(count_forward)
            result = adapter.generate(prompt, policy, max_new_tokens=budget, control=control,
                                      generation_seed=0, capture_first_token_scores=capture)
            scores = result.pop("first_token_scores", None)
            result["status"] = "ok"
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__, "error": str(error),
                      "text": "", "prompt_sha256": sha(prompt.text), "stage": prompt.stage,
                      "requested_control": control.value, "generation_seed": 0}
        finally:
            if handle is not None:
                handle.remove()
            operator.remove_projection()
        result.update(actual_forward_calls=calls[0], wall_seconds=time.perf_counter() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(operator.device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(operator.device) / 2**30)
        return result, scores

    def log_stage(identity, label, record):
        payload = {"sample_id": identity.sample_id, "stage_configuration": label, **record}
        stage_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stage_file.flush()

    def extraction(task):
        key = value_digest([sha(task.guideline), args.model_fingerprint, sha(EXTRACTION_INSTRUCTION),
                            args.extraction_max_new_tokens, 0, False, protocol["runtime_model_config_sha256"],
                            EXTRACTION_PARSER_VERSION])
        if key in extraction_cache:
            return (*extraction_cache[key], True)
        prompt = PreparedPrompt(EXTRACTION_INSTRUCTION + task.guideline, {}, empty_evidence(), "execution", "rules")
        result, _ = infer(chat_prompt(prompt, operator.tok), PhasePolicy(False, False),
                          Control.NO_HOOK, args.extraction_max_new_tokens, capture=False)
        rules, item_audit, validation_error = (), [], None
        if result["status"] == "ok":
            try:
                rules, item_audit = parse_quoted_rules_individually(result["text"], task.guideline)
            except Exception as error:
                validation_error = f"{type(error).__name__}: {error}"
        else:
            validation_error = result.get("error", "Generation failed")
        rejected = sum(item["status"] == "rejected" for item in item_audit)
        array_incomplete = any(not item["array_closed"] for item in item_audit)
        validation_status = ("quotes_partially_validated" if rejected or array_incomplete else "quotes_validated") if rules else "failed"
        record = {"cache_key": key, "guideline_sha256": sha(task.guideline), "generation": result,
                  "extraction_parser_version": EXTRACTION_PARSER_VERSION,
                  "validation_status": validation_status, "item_audit": item_audit,
                  "rejected_rule_count": rejected,
                  "truncated_item_count": sum(item["truncated"] for item in item_audit),
                  "invalid_json_escape_repair_count": sum(len(item["invalid_escape_repairs"]) for item in item_audit),
                  "array_closed": not array_incomplete if item_audit else None,
                  "validation_error": validation_error, "rules": [asdict(r) for r in rules],
                  "semantic_correctness": "not_verified", "coverage": "not_verified"}
        write_json(cache_dir / f"{key}.json", record)
        extraction_cache[key] = (rules, record)
        return rules, record, False

    def condition_assessments(identity, task, rules):
        assessments = {"A0": [], "A1": []}
        for rule in rules:
            prompt = attach_controls(chat_prompt(condition_prompt(task.vignette, rules, rule.rule_id), operator.tok),
                                     adapter, rules, target_rule=rule.rule_id)
            reference = None
            variants = [("A0", Control.NO_HOOK), ("A1", Control.NORMAL), ("A1_zero", Control.MATCHED_ZERO),
                        ("A1_wrong", Control.WRONG_SPAN), ("A1_misaligned", Control.MISALIGNED_RULE),
                        ("A1_reverse", Control.REVERSE)]
            for name, control in variants:
                missing = ((control == Control.WRONG_SPAN and not prompt.evidence.wrong_spans) or
                           (control == Control.MISALIGNED_RULE and not prompt.evidence.misaligned_rule_spans))
                if missing:
                    log_stage(identity, name, {"rule_id": rule.rule_id, "status": "control_unavailable",
                                              "reason": "No changed equal-token-count rule/span mapping exists under this control",
                                              "actual_forward_calls": 0, "prediction": None})
                    continue
                result, scores = infer(prompt, PhasePolicy(name != "A0", False), control, 1)
                prediction = {"rule_id": rule.rule_id, "label": "U", "status": result["status"],
                              "label_probabilities": None, "candidate_normalized_probabilities": None}
                if scores is not None and torch.isfinite(scores).all():
                    indices = torch.tensor(list(label_ids.values()), dtype=torch.long)
                    label_scores = scores[indices]
                    conditional = label_scores.softmax(-1)
                    full_probs = scores.log_softmax(-1)[indices].exp()
                    prediction.update(label=list(label_ids)[int(conditional.argmax())],
                                      label_probabilities=dict(zip(label_ids, full_probs.tolist())),
                                      candidate_normalized_probabilities=dict(zip(label_ids, conditional.tolist())),
                                      label_logits=dict(zip(label_ids, label_scores.tolist())),
                                      score_definition="Within-rule T/F/U first-token generation scores; not generated confidence")
                elif result["status"] == "ok":
                    result.update(status="error", error="Missing or non-finite first-token scores")
                    prediction["status"] = "error"
                    scores = None
                if result["status"] != "ok":
                    prediction["fallback_reason"] = "Assessment failed; U is an explicit missing-information fallback, not probability 1"
                if name in assessments:
                    assessments[name].append(prediction)
                if name == "A0":
                    reference = scores
                if name == "A1_zero" and reference is not None and scores is not None:
                    result["numerical_control"] = {
                        "reference": "A0_no_hook", "max_abs_first_token_score_difference": float((scores - reference).abs().max()),
                        "argmax_changed": bool(scores.argmax() != reference.argmax())}
                log_stage(identity, name, {"rule_id": rule.rule_id, "prediction": prediction, "generation": result})
        return assessments

    try:
        for identity, task in tasks:
            case_written = set()
            try:
                rules, extract_record, cache_hit = extraction(task)
                log_stage(identity, "guideline_extraction", {"cache_key": extract_record["cache_key"], "cache_hit": cache_hit,
                          "rule_count": len(rules), "validation_status": extract_record["validation_status"],
                          "error": extract_record["validation_error"],
                          "actual_forward_calls": 0 if cache_hit else extract_record["generation"]["actual_forward_calls"]})
                assessed = condition_assessments(identity, task, rules)
                language = "zh" if task.language == "zh" else "en"
                template = (args.official_root / TEMPLATES[language]).read_text()
                original = official_prompt(template, task.guideline, task.vignette)
                if original.text != render_official_prompt(task, args.official_root, manifest):
                    raise ValueError("Native full-guideline prompt differs from the frozen official template")
                prepared = {}
                for a in ("A0", "A1"):
                    selected = support_spans(rules, assessed[a])
                    evidence = PredictedSpans(sha(SELECTOR_VERSION), selected)
                    plain = official_prompt(template, task.guideline, task.vignette, evidence, assessed[a])
                    prepared[a] = attach_controls(chat_prompt(plain, operator.tok), adapter, rules)
                cached_b0, cached_b0_scores = None, None
                for arm in arms:
                    missing = False
                    if arm == "full_guideline":
                        prompt, control, policy = chat_prompt(original, operator.tok), Control.NO_HOOK, PhasePolicy(False, False)
                    elif arm in {"text_emphasis", "predicted_crop"}:
                        evidence = support_spans(rules, assessed["A0"])
                        if arm == "text_emphasis":
                            plain = official_prompt(template, task.guideline, task.vignette, tfu=assessed["A0"])
                            quotations = [task.guideline[s.start:s.end] for s in evidence]
                            plain = replace(plain, text=plain.text + "\nAttend to these predicted relevant quotations (unverified):\n" + "\n\n".join(quotations))
                        else:
                            labels = {p["rule_id"]: p["label"] for p in assessed["A0"]}
                            passages = list(dict.fromkeys(r.rule_quote for r in rules if labels.get(r.rule_id, "U") != "F"))
                            cropped = "\n\n".join(passages) if passages else task.guideline
                            plain = official_prompt(template, cropped, task.vignette, tfu=assessed["A0"])
                        prompt, control, policy = chat_prompt(plain, operator.tok), Control.NO_HOOK, PhasePolicy(False, False)
                    else:
                        a = arm[:2]
                        prompt = prepared[a]
                        policy = PhasePolicy(a == "A1", arm[3] == "1")
                        suffix = arm.partition("_")[2]
                        control = {"": Control.NORMAL if policy.execution_on else Control.NO_HOOK,
                                   "zero": Control.MATCHED_ZERO, "wrong": Control.WRONG_SPAN,
                                   "misaligned": Control.MISALIGNED_RULE, "reverse": Control.REVERSE}[suffix]
                        missing = ((control == Control.WRONG_SPAN and not prompt.evidence.wrong_spans) or
                                   (control == Control.MISALIGNED_RULE and not prompt.evidence.misaligned_rule_spans))
                    if missing and cached_b0 is not None:
                        result, scores = dict(cached_b0), cached_b0_scores
                        result.update(status="control_unavailable", requested_control=control.value,
                                      reason="Changed equal-token-count span/rule mapping unavailable; retained original-context prediction",
                                      reused_from="A1B0", actual_forward_calls=0, wall_seconds=0.)
                    else:
                        result, scores = infer(prompt, policy, control, args.max_new_tokens)
                    if arm == "A1B0":
                        cached_b0, cached_b0_scores = dict(result), scores
                    if arm == "A1B1_zero" and scores is not None and cached_b0_scores is not None:
                        result["numerical_control"] = {
                            "reference": "A1B0_no_hook_same_TFU", "max_abs_first_token_score_difference": float((scores - cached_b0_scores).abs().max()),
                            "argmax_changed": bool(scores.argmax() != cached_b0_scores.argmax())}
                    row = {**asdict(identity), "sample_id": identity.sample_id, "method": arm,
                           "language": task.language,
                           "llm_output": result.get("text", ""), "model": args.model,
                           "status": result["status"],
                           "error": result.get("error"), "generation": result,
                           "guideline_extraction_status": extract_record["validation_status"],
                           "condition_rule_count": len(rules), "method_fallback": not bool(rules),
                           "cost_references": {
                               "guideline_extraction_cache_key": None if arm == "full_guideline" else extract_record["cache_key"],
                               "guideline_cache_cases": guide_case_counts[sha(task.guideline)],
                               "condition_configuration": None if arm == "full_guideline" else
                                   ("A0" if arm.startswith("A0") or arm in {"text_emphasis", "predicted_crop"} else "A1"),
                               "note": "Shared extraction/condition calls are logged once; include their amortized cost when comparing methods"},
                           "guideline_sha256": sha(task.guideline), "original_template_sha256": sha(template),
                           "max_new_tokens": args.max_new_tokens, "generation_seed": 0,
                           "judge_evaluation": "not_run", "native_output_parse": "not_run"}
                    native_files[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
                    native_files[arm].flush()
                    rows += 1
                    case_written.add(arm)
                    outcome_counts[f"{arm}:{row['status']}"] += 1
            except Exception as error:
                # Retain every frozen case/arm, including preparation and parsing
                # failures; never reduce the native evaluation denominator.
                for arm in arms:
                    if arm in case_written:
                        continue
                    failure = {**asdict(identity), "sample_id": identity.sample_id, "method": arm,
                               "language": task.language,
                               "llm_output": "", "model": args.model, "status": "error",
                               "error_type": type(error).__name__, "error": str(error),
                               "preparation_failed": True, "generation_seed": 0,
                               "judge_evaluation": "not_run", "native_output_parse": "not_run"}
                    native_files[arm].write(json.dumps(failure, ensure_ascii=False) + "\n")
                    native_files[arm].flush()
                    rows += 1
                    outcome_counts[f"{arm}:error"] += 1
                log_stage(identity, "case_preparation_failure",
                          {"status": "error", "error_type": type(error).__name__, "error": str(error)})
            protocol["completed_native_prediction_rows"] = rows
            protocol["native_outcome_counts"] = dict(outcome_counts)
            write_json(args.output_dir / "protocol.json", protocol)
        protocol.update(execution_status="finished", verification_pending=True,
                        completed_native_prediction_rows=rows, elapsed_seconds=time.perf_counter() - run_start,
                        cached_guideline_count=len(extraction_cache))
    except Exception as error:
        protocol.update(task_state="blocked", execution_status="failed", error_type=type(error).__name__,
                        error=str(error), completed_native_prediction_rows=rows, partial_results_retained=True)
        raise
    finally:
        stage_file.close()
        for handle in native_files.values():
            handle.close()
        write_json(args.output_dir / "protocol.json", protocol)


if __name__ == "__main__":
    arguments = cli()
    try:
        main(arguments)
    except Exception as failure:
        if getattr(arguments, "_created_run", False):
            path = arguments.output_dir / "protocol.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            state.update(task_state="blocked", execution_status="failed", error_type=type(failure).__name__,
                         error=str(failure), partial_results_retained=True)
            write_json(path, state)
        raise
