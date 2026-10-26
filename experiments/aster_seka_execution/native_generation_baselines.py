"""Full-vocabulary native generation baselines, not answer-option classifiers.

Fusion is an explicitly labelled generation extension of TRIAGE PR33. The prior
option-support distribution is not reused as a fictitious vocabulary distribution.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from transformers import LogitsProcessor


@dataclass(frozen=True)
class FusionConfig:
    beta: float = .75
    temperature_base: float = 1.
    temperature_support: float = .25
    dynamic_amplitude: float = .25

    def validate(self):
        values = [self.beta, self.temperature_base, self.temperature_support, self.dynamic_amplitude]
        if not all(math.isfinite(v) for v in values) or not 0 <= self.beta <= 1:
            raise ValueError("Invalid frozen fusion configuration")
        if min(self.temperature_base, self.temperature_support) <= 0 or self.dynamic_amplitude < 0:
            raise ValueError("Temperatures must be positive and dynamic amplitude nonnegative")


def mix_vocabulary_logits(base: torch.Tensor, auxiliary: torch.Tensor, kind: str,
                         config: FusionConfig, control: str = "normal"):
    """Both distributions span the identical complete tokenizer vocabulary."""
    config.validate()
    if base.shape != auxiliary.shape or base.ndim != 2 or base.shape[0] != 1:
        raise ValueError("Expected paired batch-one full-vocabulary logits")
    if control not in {"normal", "matched_zero", "reverse"}:
        raise ValueError("Unknown dual-path control")
    base, auxiliary = base.float(), auxiliary.float()
    if not torch.isfinite(base).all() or not torch.isfinite(auxiliary).all():
        raise ValueError("Nonfinite raw vocabulary logits")
    support_raw = auxiliary.softmax(-1)
    top = support_raw.topk(2, dim=-1).values
    margin = float((top[0, 0] - top[0, 1]).detach())
    if kind == "cad":
        alpha = 0. if control == "matched_zero" else (-1. if control == "reverse" else 1.)
        scores = (1 + alpha) * base - alpha * auxiliary
        return scores, {"alpha": alpha, "raw_support_margin": margin}
    if kind not in {"static", "dynamic"}:
        raise ValueError("Unknown full-vocabulary fusion")
    beta = config.beta
    if kind == "dynamic":
        beta = min(1., max(0., beta + config.dynamic_amplitude * (2 * margin - 1)))
    if control == "matched_zero":
        beta = 0.
    elif control == "reverse":
        raise ValueError("A negative mixture weight is not a probability distribution; use CAD reverse instead")
    log_base = (base / config.temperature_base).log_softmax(-1)
    log_support = (auxiliary / config.temperature_support).log_softmax(-1)
    # Handle endpoints exactly; no candidate filtering and no tiny-probability floor.
    if beta == 0:
        mixed = log_base
    elif beta == 1:
        mixed = log_support
    else:
        mixed = torch.logaddexp(log_base + math.log1p(-beta), log_support + math.log(beta))
    return mixed, {"beta": beta, "raw_support_margin": margin}


class DualPathProcessor(LogitsProcessor):
    """HF manages the primary cache; a separate auxiliary cache follows the same tokens."""
    def __init__(self, model, auxiliary_inputs, kind, config, control):
        self.model, self.inputs, self.kind = model, auxiliary_inputs, kind
        self.config, self.control = config, control
        self.raw = None
        self.cache = None
        self.trace = []
        self.first_log_probs = None
        self.first_raw_base = None
        self.calls = 0

    def capture_raw(self, module, positional_args, output):
        # Copy before potentially in-place generation processors change scores.
        self.raw = output.logits[:, -1, :].detach().float().clone()
        self.calls += 1

    def __call__(self, input_ids, scores):
        raw_base = self.raw
        if raw_base is None or not torch.equal(raw_base, scores.float()):
            raise ValueError("Nonidentity HF logits processor needs a separately verified composition")
        if self.first_raw_base is None:
            self.first_raw_base = raw_base[0].cpu()
        details, auxiliary = {}, None
        if self.control == "no_hook" or self.kind == "plain":
            combined = raw_base
        else:
            if not self.trace:
                next_ids = self.inputs["input_ids"]
            else:
                # The actual selected token, not an independently decoded branch.
                next_ids = input_ids[:, -1:]
                self.inputs["attention_mask"] = torch.cat(
                    [self.inputs["attention_mask"], torch.ones_like(next_ids)], dim=-1)
            output = self.model(input_ids=next_ids, attention_mask=self.inputs["attention_mask"],
                                past_key_values=self.cache, use_cache=True, return_dict=True,
                                logits_to_keep=1)
            self.cache = output.past_key_values
            auxiliary = output.logits[:, -1, :].float()
            combined, details = mix_vocabulary_logits(raw_base, auxiliary, self.kind, self.config, self.control)
        log_probs = combined.log_softmax(-1)
        token = int(combined.argmax(-1)[0])
        if self.first_log_probs is None:
            self.first_log_probs = log_probs[0].detach().cpu()
        self.trace.append({"step": len(self.trace), "token_id": token,
                           "base_argmax": int(raw_base.argmax(-1)[0]),
                           "auxiliary_argmax": None if auxiliary is None else int(auxiliary.argmax(-1)[0]),
                           "selected_log_probability": float(log_probs[0, token]), **details})
        return combined


@contextmanager
def attention_backend(model, name: str):
    configs = {id(model.config): model.config}
    for layer in model.model.layers:
        configs[id(layer.self_attn.config)] = layer.self_attn.config
    previous = [(config, config._attn_implementation) for config in configs.values()]
    try:
        for config, _ in previous:
            config._attn_implementation = name
        yield
    finally:
        for config, value in previous:
            config._attn_implementation = value


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_native(model, tokenizer, primary: str, auxiliary: str | None, *, kind: str,
                    config: FusionConfig = FusionConfig(), control: str = "normal",
                    max_new_tokens: int, max_context_tokens: int) -> dict[str, Any]:
    """Greedy native text generation; two actual model forwards per generated token."""
    import transformers
    if transformers.__version__ != "4.51.3" or model.config.model_type != "qwen3" or model.training:
        raise ValueError("This implementation targets evaluation-mode Qwen3 / transformers 4.51.3")
    if kind not in {"plain", "static", "dynamic", "cad"}:
        raise ValueError("Invalid native generation method")
    if max_new_tokens < 1 or max_context_tokens < 1:
        raise ValueError("Invalid native generation budget")
    if kind != "plain" and auxiliary is None:
        raise ValueError("A complete auxiliary prompt is required for dual-path generation")
    torch.manual_seed(0)
    batches = []
    for prompt in [primary] + ([auxiliary] if auxiliary is not None else []):
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
        if batch["input_ids"].numel() + max_new_tokens > max_context_tokens:
            raise ValueError("Complete prompt plus generation budget exceeds the frozen context limit")
        batches.append({k: v.to(model.device) for k, v in batch.items()})
    processor = DualPathProcessor(model, batches[1] if len(batches) == 2 else None, kind, config, control)
    handle = model.register_forward_hook(processor.capture_raw)
    _sync(model.device)
    start = time.perf_counter()
    try:
        output = model.generate(**batches[0], do_sample=False, temperature=None, top_p=None, top_k=None,
                                max_new_tokens=max_new_tokens, logits_to_keep=1,
                                logits_processor=[processor], pad_token_id=tokenizer.eos_token_id,
                                return_dict_in_generate=False)
        _sync(model.device)
    except Exception as error:
        error.partial_generation = {
            "partial_output_token_ids": [step["token_id"] for step in processor.trace],
            "partial_output_text": tokenizer.decode([step["token_id"] for step in processor.trace], skip_special_tokens=True),
            "partial_step_trace": processor.trace, "actual_forward_calls_before_failure": processor.calls,
        }
        raise
    finally:
        handle.remove()
    tokens = output[0, batches[0]["input_ids"].shape[-1]:].tolist()
    if tokens != [step["token_id"] for step in processor.trace]:
        raise ValueError("Actual native generation suffix differs from the shared dual-cache decisions")
    expected_calls = len(tokens) * (1 if kind == "plain" or control == "no_hook" else 2)
    if processor.calls != expected_calls:
        raise ValueError("Actual forward count differs from the recorded path count")
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "output_token_ids": tokens,
            "input_tokens_per_path": [batch["input_ids"].numel() for batch in batches],
            "output_tokens": len(tokens), "actual_forward_calls": processor.calls,
            "first_log_probs": processor.first_log_probs, "first_raw_base": processor.first_raw_base,
            "step_trace": processor.trace, "kind": kind, "control": control,
            "max_new_tokens": max_new_tokens, "generation_seed": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "generation_processor_check": "raw pre-processor logits equal HF scores at every primary step",
            "suffix_alignment": "identical chosen token fed to both independent caches"}


def ranked_rules(rules, tfu):
    predictions = {item["rule_id"]: item for item in tfu}
    valid = []
    for rule in rules:
        prediction = predictions.get(rule.rule_id, {})
        probs = prediction.get("candidate_normalized_probabilities")
        if prediction.get("status") == "ok" and isinstance(probs, dict) and "T" in probs:
            valid.append((float(probs["T"]), rule))
    return [rule for score, rule in sorted(valid, key=lambda item: (-item[0], item[1].rule_id))]


def cropped_guideline(guideline, rules, tfu, kind):
    labels = {item["rule_id"]: item.get("label", "U") for item in tfu}
    if kind in {"top1", "top2"}:
        selected = ranked_rules(rules, tfu)[:1 if kind == "top1" else 2]
    elif kind == "applicable":
        if any(labels.get(rule.rule_id, "U") == "U" for rule in rules):
            return guideline, {"fallback": "unknown_conditions_keep_original_context", "selected_rules": []}
        selected = [rule for rule in rules if labels.get(rule.rule_id) == "T"]
    else:
        raise ValueError("Invalid predicted cropping method")
    passages = list(dict.fromkeys(rule.rule_quote for rule in selected))
    if not passages:
        return guideline, {"fallback": "no_predicted_support_keep_original_context", "selected_rules": []}
    return "\n\n".join(passages), {"fallback": None, "selected_rules": [rule.rule_id for rule in selected]}


def direct_action_mapping(rules, tfu, language):
    """A mechanical one-rule action baseline, not a recovered decision-tree executor."""
    labels = {item["rule_id"]: item for item in tfu}
    if not rules or any(labels.get(rule.rule_id, {}).get("label", "U") == "U" for rule in rules):
        return None, "incomplete_condition_assessment"
    selected = [rule for rule in rules if labels.get(rule.rule_id, {}).get("label") == "T"]
    if len(selected) != 1 or selected[0].exception_quotes:
        return None, "not_one_unambiguous_exception_free_rule; no invented path ordering"
    rule = selected[0]
    if language == "zh":
        output = (f"### 临床决策过程：\n1. 对患者条件“{rule.condition_quote}”的预测判断为真，"
                  f"对应指南原文动作“{rule.action_quote}”。\n### 临床决策结论：{rule.action_quote}")
    else:
        output = (f"### Clinical Decision Process:\n1. The predicted patient assessment for '{rule.condition_quote}' "
                  f"is true, mapped to the quoted guideline action '{rule.action_quote}'.\n"
                  f"### Clinical Decision Outcome: {rule.action_quote}")
    return output, None
