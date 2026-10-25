"""Native-path wrappers around the SEKA authors' Qwen PASTA and AdaSEKA APIs.

PASTA's unchanged edit_multisection_attention is called on query chunks to avoid
allocating a full per-head 55k-by-55k mask. This is a disclosed execution adaptation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json

import torch
from torch.nn import functional as F
from transformers import LogitsProcessor


def encode_evidence(prompt, tokenizer):
    inputs = tokenizer(prompt.text, return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=True, truncation=False, padding=False)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    ranges = []
    mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    # Keep individual author multisection ranges. Merging adjacent ranges changes
    # their relative scaling against generated keys in upstream exclude mode.
    for start, end in prompt.absolute_spans(prompt.evidence.spans):
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
        if indices:
            ranges.append(torch.tensor([[indices[0], indices[-1] + 1]], dtype=torch.long))
            mask[0, indices] = True
    return inputs, ranges, mask


class AuthorPastaChunks:
    """Use the unmodified author mask operator; only query scheduling is adapted."""
    def __init__(self, model, pasta, token_ranges, input_length, *, apply_edit, query_chunk=128):
        from transformers.models.qwen3 import modeling_qwen3
        self.model, self.pasta, self.hf = model, pasta, modeling_qwen3
        self.ranges, self.input_length = token_ranges, input_length
        self.apply_edit, self.query_chunk = apply_edit, query_chunk
        self.name = f"aster_author_pasta_chunks_{id(self)}"
        self.previous = []
        self.attention_calls = self.author_operator_calls = self.max_score_elements = 0

    def __enter__(self):
        configs = {id(self.model.config): self.model.config}
        for layer in self.model.model.layers:
            configs[id(layer.self_attn.config)] = layer.self_attn.config
        self.hf.ALL_ATTENTION_FUNCTIONS[self.name] = self.attention
        for config in configs.values():
            self.previous.append((config, config._attn_implementation))
            config._attn_implementation = self.name
        return self

    def __exit__(self, *unused):
        for config, original in self.previous:
            config._attn_implementation = original
        del self.hf.ALL_ATTENTION_FUNCTIONS[self.name]

    def attention(self, module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if kwargs.get("sliding_window") is not None:
            raise ValueError("This Qwen adaptation requires the checkpoint's non-sliding attention")
        key = self.hf.repeat_kv(key, module.num_key_value_groups)
        value = self.hf.repeat_kv(value, module.num_key_value_groups)
        batch, heads, query_length, dim = query.shape
        key_length = key.shape[-2]
        output = torch.empty((batch, query_length, heads, dim), dtype=query.dtype, device=query.device)
        for start in range(0, query_length, self.query_chunk):
            end = min(start + self.query_chunk, query_length)
            if attention_mask is None:
                raise ValueError("Qwen custom attention backend did not provide its causal mask")
            causal = (attention_mask[:, :, :, :key_length] if attention_mask.shape[-2] == 1
                      else attention_mask[:, :, start:end, :key_length])
            if self.apply_edit and module.layer_idx in self.pasta.head_config:
                _, modified = self.pasta.edit_multisection_attention(
                    module=module, input_args=(), input_kwargs={"attention_mask": causal},
                    head_idx=self.pasta.head_config[module.layer_idx], token_ranges=self.ranges,
                    input_len=self.input_length)
                causal = modified["attention_mask"]
                self.author_operator_calls += 1
            logits = torch.matmul(query[:, :, start:end, :], key.transpose(2, 3)) * scaling + causal
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=module.training)
            output[:, start:end] = torch.matmul(weights, value).transpose(1, 2)
            self.max_score_elements = max(self.max_score_elements, logits.numel())
        self.attention_calls += 1
        return output, None


class FirstScore(LogitsProcessor):
    def __init__(self):
        self.first = None

    def __call__(self, input_ids, scores):
        if self.first is None:
            self.first = scores[0].detach().float().cpu()
        return scores


@torch.inference_mode()
def generate_pasta_native(model, tokenizer, prompt, head_config, *, control,
                          max_new_tokens, max_context_tokens, query_chunk=128):
    from pastalib.pasta import PASTA
    if control not in {"normal", "no_hook", "matched_zero"}:
        raise ValueError("Unknown frozen PASTA control")
    inputs, ranges, mask = encode_evidence(prompt, tokenizer)
    length = inputs["input_ids"].shape[-1]
    if length + max_new_tokens > max_context_tokens:
        raise ValueError("Full PASTA input exceeds the frozen context budget")
    alpha = 1. if control == "matched_zero" else .01
    # New object per run prevents an old cached log(alpha) surviving alpha changes.
    pasta = PASTA(model=model, tokenizer=tokenizer, head_config=head_config,
                  alpha=alpha, scale_position="exclude")
    operator = AuthorPastaChunks(model, pasta, ranges, length,
                                apply_edit=control != "no_hook" and bool(ranges), query_chunk=query_chunk)
    recorder = FirstScore()
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    torch.manual_seed(0)
    with operator:
        sequences = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None, logits_to_keep=1,
                                   output_attentions=False, return_dict_in_generate=False, output_scores=False,
                                   logits_processor=[recorder], pad_token_id=tokenizer.eos_token_id)
    token_ids = sequences[0, length:].tolist()
    return {"text": tokenizer.decode(token_ids, skip_special_tokens=True), "output_token_ids": token_ids,
            "first_token_scores": recorder.first, "input_tokens": length, "output_tokens": len(token_ids),
            "control": control, "alpha": alpha, "scale_position": "exclude", "query_chunk": query_chunk,
            "status": "no_support" if not ranges and control == "normal" else
                      ("control_unavailable" if not ranges and control == "matched_zero" else "ok"),
            "intervention_applied": operator.apply_edit, "no_support_fallback": not bool(ranges),
            "author_operator": "pastalib.pasta.PASTA.edit_multisection_attention (unchanged)",
            "method_status": "SEKA-author Qwen PASTA port with disclosed query-chunk execution adaptation",
            "no_hook_definition": "same query-chunked attention kernel without author mask edits",
            "section_token_ranges": [value.tolist()[0] for value in ranges],
            "author_operator_calls": operator.author_operator_calls,
            "attention_calls": operator.attention_calls, "largest_score_matrix_elements": operator.max_score_elements,
            "input_ids_sha256": hashlib.sha256(json.dumps(inputs["input_ids"].tolist()).encode()).hexdigest(),
            "span_mask_sha256": hashlib.sha256(json.dumps(mask.tolist()).encode()).hexdigest(),
            "generation_seed": 0, "max_new_tokens": max_new_tokens}


def make_adaseka(model_path, expert_paths, *, device, source_commit, rope_scaling=None):
    """Actual author class and existing StageSEKAAdapter, explicitly BF16."""
    from src.model.adaptive_seka_llm import AdaptiveSEKALLM
    from aster_seka_adapter import StageSEKAAdapter
    settings = {"rope_scaling": rope_scaling} if rope_scaling is not None else {}
    operator = AdaptiveSEKALLM(model_path, device=device, expert_paths=expert_paths, layers="last10",
                               amplify_factor=.5, top_k_singular=5, combination_method="weighted_top_k",
                               temperature=1., torch_dtype=torch.bfloat16, attn_implementation="sdpa", **settings)
    adapter = StageSEKAAdapter(operator, kind="adaseka", source_commit=source_commit,
                               projection_paths=list(expert_paths.values()))
    return operator, adapter


@contextmanager
def routing_last_token_logits(model):
    """Storage adaptation only: author routing reads all hidden states, not logits."""
    record = {"routing_forward_calls": 0, "logits_to_keep": 1,
              "hidden_states": "all original positions/layers preserved"}
    def trim_unused_logits(module, positional, keyword):
        if keyword.get("output_hidden_states") is True and keyword.get("use_cache") is False:
            keyword = {**keyword, "logits_to_keep": 1}
            record["routing_forward_calls"] += 1
        return positional, keyword
    handle = model.register_forward_pre_hook(trim_unused_logits, with_kwargs=True)
    try:
        yield record
    finally:
        handle.remove()
