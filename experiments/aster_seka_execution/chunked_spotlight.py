"""Memory-bounded query-chunked Spotlight; all clinical controls use this kernel.

Paper reimplementation, not author code. No full [heads, query, key] matrix is
allocated. Upstream Qwen may still allocate its [1, 1, query, key] causal mask.
"""
from __future__ import annotations

import hashlib
import json
import time

import torch
from torch.nn import functional as F
from transformers import LogitsProcessor

from baseline_operators import SpotlightQwen, encode_spans, spotlight_bias_logits, wrong_span_mask


class ChunkedSpotlightQwen(SpotlightQwen):
    def __init__(self, model, mask, *, target=.1, gain=1., query_chunk=128, apply_bias=True):
        super().__init__(model, mask, target=target, gain=gain)
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.query_chunk = query_chunk
        self.apply_bias = apply_bias
        self.chunk_calls = 0
        self.max_score_elements = 0

    def _attention(self, module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if kwargs.get("sliding_window") is not None:
            raise ValueError("Sliding-window attention requires a separately verified mask adaptation")
        key = self.hf.repeat_kv(key, module.num_key_value_groups)
        value = self.hf.repeat_kv(value, module.num_key_value_groups)
        batch, heads, q_length, dim = query.shape
        k_length = key.shape[-2]
        if k_length < self.mask.shape[-1]:
            raise ValueError("Key cache shorter than the complete highlighted prompt")
        highlight = F.pad(self.mask.to(query.device), (0, k_length - self.mask.shape[-1]), value=False)
        output = torch.empty((batch, q_length, heads, dim), device=query.device, dtype=query.dtype)
        for start in range(0, q_length, self.query_chunk):
            end = min(start + self.query_chunk, q_length)
            logits = torch.matmul(query[:, :, start:end, :], key.transpose(2, 3)) * scaling
            legal = None
            if attention_mask is not None:
                if attention_mask.shape[-2] == 1:
                    causal = attention_mask[:, :, :, :k_length]
                else:
                    causal = attention_mask[:, :, start:end, :k_length]
                logits = logits + causal
                legal = causal == 0
            if self.apply_bias:
                logits, active = spotlight_bias_logits(logits, highlight[:, None, None, :],
                                                       target=self.target, gain=self.gain, legal=legal)
                current = active.sum()
                self.active_rows = current if self.active_rows is None else self.active_rows + current
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=module.training)
            output[:, start:end, :, :] = torch.matmul(weights, value).transpose(1, 2)
            self.chunk_calls += 1
            self.max_score_elements = max(self.max_score_elements, logits.numel())
        self.calls += 1
        self.layers.add(module.layer_idx)
        # The clinical runner explicitly disables output_attentions. Returning a
        # full attention matrix here would defeat the memory-bounded algorithm.
        return output, None


class FirstScoreRecorder(LogitsProcessor):
    def __init__(self):
        self.first = None
        self.steps = 0

    def __call__(self, input_ids, scores):
        if self.first is None:
            self.first = scores[0].detach().float().cpu()
        self.steps += 1
        return scores


@torch.inference_mode()
def generate_chunked_spotlight(model, tokenizer, text, spans, *, control="normal", target=.1,
                               query_chunk=128, max_new_tokens, max_context_tokens):
    if control not in {"normal", "no_hook", "matched_zero", "wrong_span", "reverse"}:
        raise ValueError("Unknown Spotlight control")
    inputs, mask, visible = encode_spans(tokenizer, text, spans)
    original_count = int(mask.sum())
    if inputs["input_ids"].numel() + max_new_tokens > max_context_tokens:
        raise ValueError("Complete Spotlight prompt plus output budget exceeds frozen context limit")
    if control == "wrong_span" and original_count:
        mask = wrong_span_mask(mask, visible)
    gain = 0. if control == "matched_zero" else (-1. if control == "reverse" else 1.)
    operator = ChunkedSpotlightQwen(model, mask, target=target, gain=gain, query_chunk=query_chunk,
                                    apply_bias=control != "no_hook" and bool(original_count))
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    recorder = FirstScoreRecorder()
    calls = [0]
    def count_forward(*unused):
        calls[0] += 1
    handle = model.register_forward_hook(count_forward)
    torch.manual_seed(0)
    if str(model.device).startswith("cuda"):
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    try:
        with operator:
            output = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                    max_new_tokens=max_new_tokens, output_attentions=False, logits_to_keep=1,
                                    logits_processor=[recorder], pad_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=False, output_scores=False)
        if str(model.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)
    finally:
        handle.remove()
    tokens = output[0, inputs["input_ids"].shape[-1]:].tolist()
    if calls[0] != len(tokens) or recorder.steps != len(tokens):
        raise ValueError("Unexpected forward or score count in chunked native generation")
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "output_token_ids": tokens,
            "first_log_probs": recorder.first.log_softmax(-1), "control": control,
            "method_status": "Spotlight_paper_reimplementation_query_chunked",
            "no_hook_definition": "same query-chunked attention function with attention bias disabled",
            "query_chunk": query_chunk, "target_mass": target, "gain": gain,
            "input_tokens": inputs["input_ids"].numel(), "output_tokens": len(tokens),
            "actual_forward_calls": calls[0], "elapsed_seconds": time.perf_counter() - start,
            "attention_calls": operator.calls, "query_chunk_calls": operator.chunk_calls,
            "largest_score_matrix_elements": operator.max_score_elements,
            "layers_touched": sorted(operator.layers),
            "rows_below_target": int(operator.active_rows) if operator.active_rows is not None else None,
            "span_mask_sha256": hashlib.sha256(json.dumps(mask.tolist()).encode()).hexdigest(),
            "highlighted_tokens": int(mask.sum()), "no_support_fallback": original_count == 0,
            "generation_seed": 0, "max_new_tokens": max_new_tokens,
            "attention_matrix_export": False,
            "remaining_memory_cost": "Upstream Qwen full shared causal mask may still use quadratic memory"}
