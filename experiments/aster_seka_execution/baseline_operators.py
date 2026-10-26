"""CAD formula adaptation and Spotlight paper reimplementation for Qwen3.

CAD source: xhan77/context-aware-decoding@1281d7fcd0e7e49d786c78d3170a3048dd9f6dc2.
Spotlight: EACL 2026 long paper 174, Eq. 3-4. This is not author-released code.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F


def cad_scores(context_logits: torch.Tensor, prior_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """No candidate filtering; arithmetic is in float32 for all CAD controls."""
    if context_logits.shape != prior_logits.shape or not math.isfinite(alpha):
        raise ValueError("Invalid paired CAD logits or alpha")
    return (1.0 + alpha) * context_logits.float() - alpha * prior_logits.float()


def spotlight_bias_logits(logits: torch.Tensor, highlighted: torch.Tensor, *, target: float,
                          gain: float, legal: torch.Tensor | None = None):
    """Per query/head log-ratio bias; gain=0/-1 are numerical/reverse controls.

    highlighted and legal broadcast to [batch, query_heads, query, key]. The
    logsumexp mass computation avoids losing tiny nonzero masses to underflow.
    """
    if not 0 < target <= 1 or gain not in {-1.0, 0.0, 1.0}:
        raise ValueError("Use a fixed valid target and normal/zero/reverse gain")
    raw = logits.float()
    selected = highlighted.bool().expand_as(raw)
    if legal is not None:
        selected = selected & legal.bool().expand_as(raw)
    has_selected = selected.any(dim=-1)
    chosen = raw.masked_fill(~selected, float("-inf"))
    log_mass = torch.logsumexp(chosen, dim=-1) - torch.logsumexp(raw, dim=-1)
    active = has_selected & (log_mass < math.log(target))
    # where removes the +inf from rows having no causally visible highlighted key.
    bias = torch.where(active, math.log(target) - log_mass, torch.zeros_like(log_mass))
    correction = torch.where(selected, (gain * bias).unsqueeze(-1), torch.zeros_like(raw))
    return raw + correction, active


def encode_spans(tokenizer: Any, text: str, spans: Sequence[tuple[int, int]]):
    """Encode the original text once without inserting/removing marker strings."""
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                        padding=False, truncation=False, add_special_tokens=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    for a, b in spans:
        if not 0 <= a < b <= len(text):
            raise ValueError("Invalid character span")
    mask = torch.tensor([[any(y > a and x < b and y > x for a, b in spans) for x, y in offsets]], dtype=torch.bool)
    visible = encoded["attention_mask"].bool() & torch.tensor([[y > x for x, y in offsets]])
    return encoded, mask & visible, visible


def wrong_span_mask(original: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Deterministic first eligible non-highlighted tokens, preserving token count."""
    if original.ndim != 2 or original.shape[0] != 1 or original.shape != visible.shape:
        raise ValueError("Only a single unpadded example is supported")
    count = int(original.sum())
    candidates = torch.nonzero(visible[0] & ~original[0], as_tuple=False).flatten()
    if count == 0 or candidates.numel() < count:
        raise ValueError("Cannot construct a disjoint equal-token-count wrong-span control")
    wrong = torch.zeros_like(original)
    wrong[0, candidates[:count]] = True
    return wrong


class SpotlightQwen:
    """Temporary custom attention function; upstream QKV/RoPE/cache stay intact."""
    def __init__(self, model: Any, mask: torch.Tensor, *, target: float = .1, gain: float = 1.0):
        import transformers
        from transformers.models.qwen3 import modeling_qwen3
        if transformers.__version__ != "4.51.3" or model.config.model_type != "qwen3":
            raise ValueError("This adapter is pinned to transformers 4.51.3 Qwen3")
        if model.training or mask.ndim != 2 or mask.shape[0] != 1:
            raise ValueError("Use evaluation mode and one example per call")
        self.model, self.mask, self.target, self.gain = model, mask, target, gain
        self.hf = modeling_qwen3
        self.name = f"aster_spotlight_{id(self)}"
        self.previous = []
        self.calls = 0
        self.layers = set()
        self.active_rows = None

    def __enter__(self):
        configs = {id(self.model.config): self.model.config}
        for layer in self.model.model.layers:
            configs[id(layer.self_attn.config)] = layer.self_attn.config
        self.hf.ALL_ATTENTION_FUNCTIONS[self.name] = self._attention
        for config in configs.values():
            self.previous.append((config, config._attn_implementation))
            config._attn_implementation = self.name
        return self

    def __exit__(self, *exc):
        for config, previous in self.previous:
            config._attn_implementation = previous
        del self.hf.ALL_ATTENTION_FUNCTIONS[self.name]

    def _attention(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key = self.hf.repeat_kv(key, module.num_key_value_groups)
        value = self.hf.repeat_kv(value, module.num_key_value_groups)
        logits = torch.matmul(query, key.transpose(2, 3)) * scaling
        legal = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            logits = logits + causal_mask
            # Qwen3's upstream causal/padding mask is additive: zero or -inf/min.
            legal = causal_mask == 0
        key_length = key.shape[-2]
        if key_length < self.mask.shape[-1]:
            raise ValueError("Prompt key cache shorter than the highlighted input")
        highlighted = F.pad(self.mask.to(query.device), (0, key_length - self.mask.shape[-1]), value=False)
        edited, active = spotlight_bias_logits(logits, highlighted[:, None, None, :],
                                              target=self.target, gain=self.gain, legal=legal)
        weights = F.softmax(edited, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        self.calls += 1
        self.layers.add(module.layer_idx)
        current = active.sum()
        self.active_rows = current if self.active_rows is None else self.active_rows + current
        return output, weights


@dataclass
class DecodeResult:
    text: str
    first_token_scores: torch.Tensor
    output_token_ids: list[int]
    metadata: dict[str, Any]


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def cad_first_token(model, tokenizer, context: str, prior: str, *, alpha: float = 1.0,
                    control: str = "normal", seed: int = 0) -> DecodeResult:
    """Two exact one-token generate calls; official CounterFact efficacy budget.

    The upstream CAD formula acts on raw logits. We reject nonidentity generation
    processors rather than silently mixing processed scores under that name.
    no_hook runs one contextual path; matched_zero still evaluates both paths.
    """
    if control not in {"normal", "matched_zero", "reverse", "no_hook"}:
        raise ValueError("Invalid CAD control")
    if model.training:
        raise ValueError("Expected model.eval()")
    torch.manual_seed(seed)
    effective_alpha = 0.0 if control in {"matched_zero", "no_hook"} else (-alpha if control == "reverse" else alpha)
    contexts = [context] if control == "no_hook" else [context, prior]
    inputs = []
    for text in contexts:
        batch = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False, padding=False)
        inputs.append({k: v.to(model.device) for k, v in batch.items()})
    prompt_lengths = [batch["input_ids"].numel() for batch in inputs]
    _sync(model.device)
    start, logits = time.perf_counter(), []
    for batch in inputs:
        captured_raw = []

        def capture_before_processors(module, args, forward_output):
            captured_raw.append(forward_output.logits[:, -1, :].detach().float().clone())

        handle = model.register_forward_hook(capture_before_processors)
        try:
            output = model.generate(**batch, max_new_tokens=1, do_sample=False,
                                    temperature=None, top_k=None, top_p=None, return_dict_in_generate=True,
                                    output_scores=True, output_logits=True, pad_token_id=tokenizer.eos_token_id)
        finally:
            handle.remove()
        if len(output.scores) != 1 or len(output.logits) != 1:
            raise ValueError("Expected exactly one generated token per CAD path")
        if len(captured_raw) != 1:
            raise ValueError("Expected exactly one model forward per CAD path")
        # output.logits may share a tensor mutated in-place by a logits processor.
        # Capture before generation processors so identity checks cannot be fooled.
        raw, processed = captured_raw[0], output.scores[0].float()
        if not torch.equal(raw, processed):
            raise ValueError("Nonidentity generation processor: raw CAD formula needs an explicit processor adaptation")
        logits.append(raw)
    score = logits[0] if len(logits) == 1 else cad_scores(logits[0], logits[1], effective_alpha)
    if not torch.isfinite(score).all():
        raise ValueError("Non-finite CAD scores")
    generated = [int(score.argmax(dim=-1)[0])]
    _sync(model.device)
    return DecodeResult(tokenizer.decode(generated, skip_special_tokens=True), score[0].detach().cpu(), generated, {
        "method": "CAD_formula_Qwen_adaptation", "control": control, "alpha": effective_alpha,
        "candidate_filtering": False, "forward_calls": len(inputs), "prompt_tokens_per_path": prompt_lengths,
        "output_tokens": 1, "max_new_tokens": 1, "generation_seed": seed,
        "greedy": True, "elapsed_seconds": time.perf_counter() - start,
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "prior_sha256": hashlib.sha256(prior.encode()).hexdigest(),
        "score_semantics": "raw float32 CAD logits; exact identity of generation processors verified on both paths",
    })


@torch.inference_mode()
def spotlight_generate(model, tokenizer, text: str, spans: Sequence[tuple[int, int]], *, target: float = .1,
                       control: str = "normal", max_new_tokens: int = 1, seed: int = 0) -> DecodeResult:
    if control not in {"normal", "matched_zero", "wrong_span", "reverse", "no_hook"} or max_new_tokens <= 0:
        raise ValueError("Invalid Spotlight control/budget")
    if model.config._attn_implementation != "eager":
        raise ValueError("Load the model with attn_implementation='eager' for all matched comparisons")
    torch.manual_seed(seed)
    inputs, mask, visible = encode_spans(tokenizer, text, spans)
    no_support = not bool(mask.any())
    unavailable_reason = "No evidence tokens in the original input" if no_support and control != "no_hook" else None
    if control == "wrong_span" and not no_support:
        mask = wrong_span_mask(mask, visible)
    gain = 0.0 if control == "matched_zero" else (-1.0 if control == "reverse" else 1.0)
    operator = None if control == "no_hook" or unavailable_reason else SpotlightQwen(model, mask, target=target, gain=gain)
    input_count = inputs["input_ids"].numel()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    _sync(model.device)
    start = time.perf_counter()
    with operator if operator is not None else nullcontext():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                temperature=None, top_k=None, top_p=None, return_dict_in_generate=True,
                                output_scores=True, pad_token_id=tokenizer.eos_token_id)
    _sync(model.device)
    tokens = output.sequences[0, inputs["input_ids"].shape[-1]:].tolist()
    first_scores = output.scores[0][0].detach().float().cpu()
    if not torch.isfinite(first_scores).all():
        raise ValueError("Non-finite Spotlight generation scores")
    return DecodeResult(tokenizer.decode(tokens, skip_special_tokens=True), first_scores, tokens, {
        "method": "Spotlight_paper_reimplementation_Qwen", "control": control, "target_mass": target,
        "status": ("no_support" if control == "normal" and no_support else "control_unavailable") if unavailable_reason else "ok",
        "no_support": no_support, "control_unavailable_reason": unavailable_reason,
        "intervention_applied": operator is not None, "effective_control": "no_hook" if operator is None else control,
        "gain": gain, "forward_calls": len(output.scores), "input_tokens": input_count,
        "output_tokens": len(tokens), "max_new_tokens": max_new_tokens, "generation_seed": seed,
        "greedy": True, "elapsed_seconds": time.perf_counter() - start,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "span_mask_sha256": hashlib.sha256(json.dumps(mask.tolist()).encode()).hexdigest(),
        "highlighted_tokens": int(mask.sum()),
        "attention_calls": operator.calls if operator is not None else 0,
        "layers_touched": sorted(operator.layers) if operator is not None else [],
        "rows_below_target": int(operator.active_rows) if operator is not None and operator.active_rows is not None else 0,
        "score_semantics": "Hugging Face generation scores with the frozen generation configuration",
    })
