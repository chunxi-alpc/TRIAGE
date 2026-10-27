"""Patient-conditioned key measurements for the offline applicability pilot.

The supported adapter is deliberately narrow: full-attention Qwen3, batch one,
no padding, default RoPE, and an evaluation-mode model. Keys are measured after
``k_norm`` and before RoPE. This module never reads module-one predictions.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


VARIANTS = ("positive", "negative", "neutral")


def validate_qwen3(model):
    """Validate assumptions which determine the meaning of an edited KV head."""
    cfg = model.config
    if getattr(cfg, "model_type", None) != "qwen3":
        raise ValueError("The applicability adapter currently supports Qwen3 only")
    if model.training:
        raise ValueError("Use model.eval(): dropout would invalidate paired measurements")
    layers = int(cfg.num_hidden_layers)
    q_heads, kv_heads = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    dim = int(getattr(cfg, "head_dim", cfg.hidden_size // q_heads))
    if q_heads % kv_heads or dim % 2:
        raise ValueError("Expected equal contiguous GQA groups and an even head dimension")
    types = getattr(cfg, "layer_types", ["full_attention"] * layers)
    if len(types) != layers or any(t != "full_attention" for t in types):
        raise ValueError("Sliding/hybrid attention needs a separately validated adapter")
    scaling = getattr(cfg, "rope_scaling", None)
    if scaling and scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError("Non-default/dynamic RoPE needs a separately validated adapter")
    implementation = getattr(cfg, "_attn_implementation", None)
    if implementation not in ("eager", "sdpa"):
        raise ValueError("Use eager or sdpa attention for the audited Qwen3 adapter")
    if len(model.model.layers) != layers:
        raise ValueError("Model layers do not match its configuration")
    for layer in model.model.layers:
        if not all(hasattr(layer.self_attn, name) for name in ("q_norm", "k_norm")):
            raise ValueError("Expected Qwen3 q_norm/k_norm hooks")
    return {"layers": layers, "query_heads": q_heads, "kv_heads": kv_heads,
            "head_dim": dim, "queries_per_kv_head": q_heads // kv_heads}


def rotate_half(value):
    """The split-half rotation used by Qwen3, not interleaved-pair rotation."""
    import torch
    a, b = value.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def span_token_positions(offsets, spans, prompt):
    """Map explicit half-open character spans without silently truncating them."""
    positions = set()
    for start, end in spans:
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(prompt)):
            raise ValueError("Invalid half-open character span")
        matched = [i for i, (a, b) in enumerate(offsets)
                   if b > a and a < end and b > start]
        if not matched:
            raise ValueError("A requested span has no tokens")
        # Leading separator whitespace may be merged with an action token.
        # A token crossing into actual neighbouring text is not an exact target.
        for i in matched:
            a, b = offsets[i]
            outside = prompt[a:start] if a < start else ""
            outside += prompt[end:b] if b > end else ""
            if outside.strip():
                raise ValueError("Token boundary crosses non-whitespace outside the span")
        if offsets[matched[0]][0] > start or offsets[matched[-1]][1] < end:
            # Whitespace-only span ends can be omitted by some fast tokenizers.
            first, last = offsets[matched[0]][0], offsets[matched[-1]][1]
            if prompt[start:first].strip() or prompt[last:end].strip():
                raise ValueError("The tokenizer did not cover the complete requested span")
        positions.update(matched)
    return sorted(positions)


def encode_prompt(tokenizer, prompt, spans, roles=("action",), max_length=2048):
    """Return unpadded tokens, offsets, and all rule targets for a prompt."""
    if not prompt or not getattr(tokenizer, "is_fast", False):
        raise ValueError("A nonempty prompt and a fast tokenizer are required")
    encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False, truncation=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    if encoded["input_ids"].shape[0] != 1:
        raise ValueError("The audited adapter requires batch size one")
    n = int(encoded["input_ids"].shape[1])
    if n < 2 or n > max_length:
        raise ValueError(f"Prompt has {n} tokens; require 2..{max_length}, without truncation")
    if "attention_mask" in encoded and not bool(encoded["attention_mask"].all()):
        raise ValueError("Padding is not supported by this adapter")
    def positions(role_spans):
        segments = [span for role in roles for span in role_spans.get(role, [])]
        if not segments:
            raise ValueError("No requested role spans in a rule")
        result = span_token_positions(offsets, segments, prompt)
        if not result or max(result) >= n - 1:
            raise ValueError("Rule targets must precede the final decision-prompt token")
        return result
    target = positions(spans)
    all_rules = {str(rule): positions(values)
                 for rule, values in spans.get("rules", {}).items()}
    return {"encoded": encoded, "offsets": offsets, "positions": target,
            "rule_positions": all_rules, "length": n}


def _capture_prompt(model, tokenized, geometry, candidate_ids):
    """One forward captures every rule in a prompt, then retains only summaries."""
    import torch
    union = sorted(set(tokenized["positions"]).union(
        *(set(p) for p in tokenized["rule_positions"].values())))
    index = {position: i for i, position in enumerate(union)}
    targets = {tuple(tokenized["positions"])} | {
        tuple(p) for p in tokenized["rule_positions"].values()}
    captured, handles = {}, []
    for layer_idx, layer in enumerate(model.model.layers):
        state = captured[layer_idx] = {}
        def rotary_hook(module, args, kwargs, state=state):
            if "position_embeddings" not in kwargs:
                raise ValueError("Qwen3 did not supply its actual rotary coefficients")
            cos, sin = kwargs["position_embeddings"]
            state["cos"], state["sin"] = cos[0].detach(), sin[0].detach()
        def key_hook(module, args, output, state=state):
            if output.ndim != 4 or output.shape[2:] != (geometry["kv_heads"], geometry["head_dim"]):
                raise ValueError("Unexpected Qwen3 k_norm layout")
            state["keys"] = output[0, union].detach()
        def query_hook(module, args, output, state=state):
            if output.ndim != 4 or output.shape[2:] != (geometry["query_heads"], geometry["head_dim"]):
                raise ValueError("Unexpected Qwen3 q_norm layout")
            state["query"] = output[0, -1].detach()
        handles.extend([
            layer.self_attn.register_forward_pre_hook(rotary_hook, with_kwargs=True),
            layer.self_attn.k_norm.register_forward_hook(key_hook),
            layer.self_attn.q_norm.register_forward_hook(query_hook)])
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            output = model(**{k: v.to(device) for k, v in tokenized["encoded"].items()},
                           use_cache=False, logits_to_keep=1)
            logits = output.logits[0, -1, candidate_ids].float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    results = {}
    for target in targets:
        selected = [index[p] for p in target]
        key_rows, query_rows, means, stds = [], [], [], []
        for layer_idx in range(geometry["layers"]):
            state = captured[layer_idx]
            if set(state) != {"keys", "query", "cos", "sin"}:
                raise RuntimeError("A Qwen3 activation hook was not executed")
            key = state["keys"][selected]  # token, KV head, dimension
            cos, sin = state["cos"][list(target)], state["sin"][list(target)]
            q = state["query"]
            q_post = q * state["cos"][-1] + rotate_half(q) * state["sin"][-1]
            q_grouped = q_post.float().reshape(geometry["kv_heads"],
                                              geometry["queries_per_kv_head"], -1)
            # R_j^T q_post: this is a transpose action, including any scale.
            q_mean = q_grouped.mean(dim=1)
            effective = (q_mean[None] * cos.float()[:, None]
                         - rotate_half(q_mean)[None] * sin.float()[:, None])
            k_post = key * cos[:, None] + rotate_half(key) * sin[:, None]
            scores = torch.einsum("hgd,thd->hgt", q_grouped, k_post.float())
            scores *= geometry["head_dim"] ** -0.5
            key_rows.append(key.float().mean(dim=0).cpu().numpy())
            query_rows.append(effective.mean(dim=0).cpu().numpy())
            means.append(scores.mean(dim=(1, 2)).cpu().numpy())
            stds.append(scores.std(dim=(1, 2), correction=0).cpu().numpy())
        results[target] = {
            "keys": np.stack(key_rows), "effective_queries": np.stack(query_rows),
            "baseline_score_mean": np.stack(means), "baseline_score_std": np.stack(stds)}
    return {"targets": results, "candidate_logits": logits,
            "positions": tokenized["positions"], "length": tokenized["length"],
            "ids": tokenized["encoded"]["input_ids"][0].tolist()}


def extract_triplets(model, tokenizer, triplets: Iterable[dict], max_length=2048,
                     roles=("action",), candidate_labels=("A", "B"),
                     require_neutral_alignment=True, progress=None):
    """Measure paired keys and effective queries; deduplicate complete prompts.

    Returns arrays ``[triplet, layer, KV head, dimension]`` for keys/queries,
    ``[triplet, layer, KV head]`` for score moments, and per-prompt candidate
    logits. Positive/negative target token IDs and absolute positions must match.
    Neutral alignment is also required by default to make the missing-information
    control interpretable. No truncation, target-label teacher forcing, or test
    data is accepted. These are representation measurements, not causal evidence.
    """
    geometry = validate_qwen3(model)
    rows = list(triplets)
    if not rows:
        raise ValueError("No triplets supplied")
    candidate_ids = []
    for label in candidate_labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate {label!r} is not one token; use sequence scoring")
        candidate_ids.append(ids[0])
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate labels must have distinct tokens")
    names = ("keys", "effective_queries", "baseline_score_mean", "baseline_score_std")
    collected = {name: {variant: [] for variant in VARIANTS} for name in names}
    logits = {variant: [] for variant in VARIANTS}
    metadata, audit, cache = [], [], {}
    forward_count = 0
    for row_idx, row in enumerate(rows):
        if row.get("source_split") not in {"train", "train_val", "development", "validation", "dev"}:
            raise ValueError("Extraction requires explicit development-only source provenance")
        if row.get("split") not in {"fit", "select", "confirm"}:
            raise ValueError("Expected fit/select/confirm partitions within development data")
        encoded = {v: encode_prompt(tokenizer, row["prompts"][v], row["spans"][v], roles, max_length)
                   for v in VARIANTS}
        target_ids = {v: encoded[v]["encoded"]["input_ids"][0, encoded[v]["positions"]].tolist()
                      for v in VARIANTS}
        def aligned(a, b):
            return (target_ids[a] == target_ids[b]
                    and encoded[a]["positions"] == encoded[b]["positions"]
                    and encoded[a]["length"] == encoded[b]["length"])
        if not aligned("positive", "negative"):
            raise ValueError(f"{row['id']}: positive/negative target tokens or positions are not aligned")
        neutral_aligned = aligned("positive", "neutral")
        if require_neutral_alignment and not neutral_aligned:
            raise ValueError(f"{row['id']}: neutral control changes target tokens or absolute positions")
        for variant in VARIANTS:
            prompt = row["prompts"][variant]
            if prompt not in cache:
                cache[prompt] = _capture_prompt(model, encoded[variant], geometry, candidate_ids)
                forward_count += 1
            result = cache[prompt]
            target = tuple(encoded[variant]["positions"])
            if target not in result["targets"]:
                raise ValueError("An identical prompt was assigned incompatible rule spans")
            for name in names:
                collected[name][variant].append(result["targets"][target][name])
            logits[variant].append(result["candidate_logits"])
        metadata.append({k: row.get(k) for k in (
            "id", "tree_id", "group_id", "pair_cluster_id", "source_split", "split", "target_rule_id")})
        audit.append({"id": row["id"], "positive_negative_aligned": True,
                      "neutral_aligned": neutral_aligned,
                      "token_counts": {v: encoded[v]["length"] for v in VARIANTS},
                      "target_positions": {v: encoded[v]["positions"] for v in VARIANTS},
                      "target_token_ids": target_ids["positive"],
                      "prompt_sha256": {v: hashlib.sha256(row["prompts"][v].encode()).hexdigest()
                                        for v in VARIANTS}})
        if progress is not None:
            progress(row_idx + 1, len(rows), forward_count)
    result = {name: {v: np.stack(values) for v, values in variants.items()}
              for name, variants in collected.items()}
    result.update({"candidate_logits": {v: np.stack(values) for v, values in logits.items()},
                   "candidate_labels": list(candidate_labels), "candidate_token_ids": candidate_ids,
                   "metadata": metadata, "audit": audit, "geometry": geometry,
                   "forward_count": forward_count,
                   "adapter": {"model_type": "qwen3", "attention": "full",
                               "rope": "default", "key_space": "post_k_norm_pre_rope",
                               "query_position": "last_prompt_token_before_any_answer",
                               "query_aggregation": "all contiguous GQA queries and target tokens",
                               "roles": list(roles), "module1_outputs_used": False}})
    if not all(np.isfinite(a).all() for name in names for a in result[name].values()):
        raise RuntimeError("Nonfinite key, query, or baseline-score measurement")
    return result
