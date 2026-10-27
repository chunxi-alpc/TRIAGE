"""Apply a frozen offline applicability displacement to Qwen3 rule caches.

Prefill stops before the last decision-prompt token. Only selected rule keys in
that prefix are edited. The held-out prompt token is then evaluated, so the first
answer-token logits already reflect the intervention. Values and other prefix
representations are unchanged. This is distinct from editing during prefill.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from src.module2.extraction import encode_prompt, rotate_half, validate_qwen3


@dataclass
class PrefixState:
    input_ids: object
    attention_mask: object
    cache: object
    cos: object
    sin: object
    target_positions: list
    rule_positions: dict
    geometry: dict
    model_identity: int

    @property
    def prefix_length(self):
        return int(self.input_ids.shape[1]) - 1


def prepare_prefix(model, tokenizer, prompt, spans, max_length=2048, roles=("action",)):
    """Prepare an immutable reusable prefix for baseline and signed-dose tests."""
    import torch
    geometry = validate_qwen3(model)
    tokenized = encode_prompt(tokenizer, prompt, spans, roles, max_length)
    device = model.get_input_embeddings().weight.device
    ids = tokenized["encoded"]["input_ids"].to(device)
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1], device=device)[None]
    with torch.inference_mode():
        # Default RoPE is sequence-length invariant. Non-default RoPE is rejected.
        dummy = torch.empty((1, ids.shape[1], 1), device=device,
                            dtype=model.get_input_embeddings().weight.dtype)
        cos, sin = model.model.rotary_emb(dummy, positions)
        output = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1],
                       position_ids=positions[:, :-1], use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    if type(cache).__name__ != "DynamicCache" or not hasattr(cache, "to_legacy_cache"):
        raise ValueError("Expected an unquantized DynamicCache")
    if len(cache.layers) != geometry["layers"]:
        raise ValueError("Incomplete prefix cache")
    for key, value in cache.to_legacy_cache():
        expected = (1, geometry["kv_heads"], ids.shape[1] - 1, geometry["head_dim"])
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError("Cache is truncated, padded, or has an unexpected head layout")
    return PrefixState(ids, mask, cache, cos[0], sin[0], tokenized["positions"],
                       tokenized["rule_positions"], geometry, id(model))


def _validate_entry(entry, state, kind):
    layer, head = int(entry["layer"]), int(entry["kv_head"])
    if not (0 <= layer < state.geometry["layers"] and 0 <= head < state.geometry["kv_heads"]):
        raise ValueError("Intervention addresses an invalid layer/KV head")
    positions = entry.get("positions", state.target_positions)
    if not positions or any(not isinstance(p, int) or not 0 <= p < state.prefix_length for p in positions):
        raise ValueError("Every edited token must belong to the prepared prefix")
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate token positions within an intervention")
    allowed = set(state.target_positions).union(*(set(v) for v in state.rule_positions.values()))
    if not set(positions).issubset(allowed):
        raise ValueError("Intervention positions lie outside the explicitly mapped rule targets")
    coefficient = float(entry.get("coefficient", 1.0) if kind == "shift" else entry["bias"])
    if not math.isfinite(coefficient):
        raise ValueError("Nonfinite intervention strength")
    return layer, head, positions, coefficient


def edited_cache(state, shifts=()):
    """Clone a DynamicCache and add g R_j Delta once, before any decision query."""
    import torch
    from transformers import DynamicCache
    legacy = tuple((k.clone(), v.clone()) for k, v in state.cache.to_legacy_cache())
    cache = DynamicCache.from_legacy_cache(legacy)
    occupied = set()
    for shift in shifts:
        layer, head, positions, coefficient = _validate_entry(shift, state, "shift")
        entries = {(layer, head, p) for p in positions}
        if occupied.intersection(entries):
            raise ValueError("Overlapping shifts would edit the same key twice")
        occupied.update(entries)
        key = cache.layers[layer].keys
        delta = torch.as_tensor(shift["delta"], device=key.device, dtype=torch.float32)
        if tuple(delta.shape) != (state.geometry["head_dim"],) or not bool(torch.isfinite(delta).all()):
            raise ValueError("Delta must be one finite vector of head_dim")
        if coefficient == 0:
            continue
        cos, sin = state.cos[positions].to(key.device).float(), state.sin[positions].to(key.device).float()
        rotated = coefficient * (delta[None] * cos + rotate_half(delta)[None] * sin)
        key[0, head, positions] += rotated.to(key.dtype)
    return cache


def decision_logits(model, state, shifts=(), score_bias=(), *, return_diagnostics=False,
                    diagnostic_targets=()):
    """Compute first-answer logits with a fresh cache copy for every condition.

    ``shifts`` entries contain layer, kv_head, delta, coefficient, and optional
    explicit positions. ``score_bias`` entries contain layer, kv_head, bias, and
    positions. A KV-head bias covers all its associated query heads. The latter
    is a direct-attention-score control, not another learned key direction.

    With ``return_diagnostics=True``, eager attention is required and the result
    is a dictionary with logits and actual target attention masses. Optional
    ``diagnostic_targets`` use layer/kv_head/positions and an optional rule_id.
    Otherwise intervention targets are measured; with no intervention, all
    layer/KV heads are measured at the state's target positions. No attention
    backend is changed implicitly.
    """
    import torch
    geometry = validate_qwen3(model)
    if id(model) != state.model_identity or geometry != state.geometry:
        raise ValueError("A prefix state cannot be used with another model")
    shifts, score_bias = list(shifts), list(score_bias)
    if return_diagnostics and model.config._attn_implementation != "eager":
        raise ValueError("Actual attention-mass diagnostics require eager attention")
    cache = edited_cache(state, shifts)
    by_layer, handles = {}, []
    occupied = set()
    for entry in score_bias:
        layer, head, positions, bias = _validate_entry(entry, state, "bias")
        entries = {(layer, head, p) for p in positions}
        if occupied.intersection(entries):
            raise ValueError("Overlapping score biases are ambiguous")
        occupied.update(entries)
        by_layer.setdefault(layer, []).append((head, positions, bias))
    # Validate every requested diagnostic before installing any temporary hook.
    measured, per_layer = set(), {}
    if return_diagnostics:
        targets = list(diagnostic_targets) or shifts + score_bias
        if not targets:
            targets = [{"layer": l, "kv_head": h} for l in range(geometry["layers"])
                       for h in range(geometry["kv_heads"])]
        for target in targets:
            layer, head, positions, _ = _validate_entry(target, state, "shift")
            identity = (layer, head, tuple(positions), target.get("rule_id"))
            if identity not in measured:
                per_layer.setdefault(layer, []).append((head, positions, target.get("rule_id")))
                measured.add(identity)
    for layer, entries in by_layer.items():
        def bias_hook(module, args, kwargs, entries=entries):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or hidden.shape[:2] != (1, 1):
                raise ValueError("Score-bias control is restricted to the single decision query")
            mask = kwargs.get("attention_mask")
            length = state.prefix_length + 1
            expected = (1, geometry["query_heads"], 1, length)
            if mask is None:
                mask = hidden.new_zeros(expected)
            else:
                if mask.ndim != 4 or mask.shape[-2:] != (1, length):
                    raise ValueError("Unexpected decision attention mask")
                mask = mask.expand(expected).clone()
            group = geometry["queries_per_kv_head"]
            for head, positions, bias in entries:
                mask[0, head * group:(head + 1) * group, 0, positions] += bias
            kwargs = dict(kwargs, attention_mask=mask)
            return args, kwargs
        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(
            bias_hook, with_kwargs=True))
    attention_rows = []
    if return_diagnostics:
        for layer, targets_for_layer in per_layer.items():
            def attention_hook(module, args, output, layer=layer, targets_for_layer=targets_for_layer):
                weights = output[1]
                expected = (1, geometry["query_heads"], 1, state.prefix_length + 1)
                if weights is None or tuple(weights.shape) != expected:
                    raise RuntimeError("Eager attention did not return the actual decision weights")
                group = geometry["queries_per_kv_head"]
                for head, positions, rule_id in targets_for_layer:
                    masses = weights[0, head * group:(head + 1) * group, 0][:, positions].float().sum(-1)
                    if not bool(torch.isfinite(masses).all()):
                        raise RuntimeError("Nonfinite actual attention mass")
                    attention_rows.append({"layer": layer, "kv_head": head, "rule_id": rule_id,
                        "positions": list(positions), "key_count": len(positions), "query_count": group,
                        "target_attention_mass": float(masses.mean()),
                        "per_query_mass": masses.detach().cpu().tolist(),
                        "measurement": "actual eager softmax weights at the decision prompt token"})
            handles.append(model.model.layers[layer].self_attn.register_forward_hook(attention_hook))
    try:
        with torch.inference_mode():
            output = model(input_ids=state.input_ids[:, -1:], attention_mask=state.attention_mask,
                           position_ids=state.input_ids.new_tensor([[state.prefix_length]]),
                           cache_position=state.input_ids.new_tensor([state.prefix_length]),
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = output.logits[0, -1].detach().float().cpu()
            if return_diagnostics:
                return {"logits": logits, "attention_mass": attention_rows}
            return logits
    finally:
        for handle in handles:
            handle.remove()


def measure_fixed_query_shifts(model, state, shifts):
    """Measure several candidate shifts against one unedited decision query.

    Returned RMS and sign fractions cover *every* query head sharing the KV head
    and every explicitly edited token. Actual differences include cache-dtype
    rounding of k + g R_j Delta. Queries are held at their baseline values; these
    are local score effects and do not equal the end-to-end intervention effect.
    Each shift is evaluated independently, even if their targets overlap.
    """
    import torch
    entries = list(shifts)
    validated = [_validate_entry(entry, state, "shift") for entry in entries]
    requested_layers = sorted({entry[0] for entry in validated})
    queries, handles = {}, []
    for layer in requested_layers:
        def capture(module, args, output, layer=layer):
            if output.shape[:2] != (1, 1):
                raise ValueError("Expected exactly one baseline decision query")
            queries[layer] = output[0, 0].detach()
        handles.append(model.model.layers[layer].self_attn.q_norm.register_forward_hook(capture))
    try:
        baseline_logits = decision_logits(model, state)
    finally:
        for handle in handles:
            handle.remove()
    stats = []
    dim, group = state.geometry["head_dim"], state.geometry["queries_per_kv_head"]
    for entry, (layer, head, positions, coefficient) in zip(entries, validated):
        key = state.cache.layers[layer].keys
        query = queries[layer]
        cq, sq = state.cos[-1].to(query.device), state.sin[-1].to(query.device)
        q_post = query * cq + rotate_half(query) * sq
        q_group = q_post[head * group:(head + 1) * group].float()
        delta = torch.as_tensor(entry["delta"], dtype=torch.float32, device=key.device)
        if tuple(delta.shape) != (dim,) or not bool(torch.isfinite(delta).all()):
            raise ValueError("Delta must be one finite vector of head_dim")
        co, si = state.cos[positions].to(key.device).float(), state.sin[positions].to(key.device).float()
        proposed = coefficient * (delta[None] * co + rotate_half(delta)[None] * si)
        original = key[0, head, positions]
        # This precisely follows edited_cache's addition and dtype conversion.
        actual = (original + proposed.to(original.dtype)).float() - original.float()
        scores = (q_group.to(key.device) @ actual.T) * dim ** -0.5
        analytic = (q_group.to(key.device) @ proposed.T) * dim ** -0.5
        stats.append({"layer": layer, "kv_head": head, "positions": list(positions),
                      "coefficient": coefficient, "score_count": int(scores.numel()),
                      "mean": float(scores.mean()), "rms": float(scores.square().mean().sqrt()),
                      "positive_fraction": float((scores > 0).float().mean()),
                      "zero_fraction": float((scores == 0).float().mean()),
                      "analytical_rms": float(analytic.square().mean().sqrt()),
                      "analytical_mean": float(analytic.mean()),
                      "query_scope": "all queries in the selected GQA group"})
    return {"baseline_logits": baseline_logits, "shifts": stats}
