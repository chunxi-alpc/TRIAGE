"""Qwen3 guideline-first cache editing for the 2026-10-28 mechanism pilot.

The guideline is prefetched without an intervention. Each patient receives an
independent cache view, edits only mapped guideline keys, and then evaluates the
*whole* patient/question/answer-prefix suffix. This is not a last-token patch.
Only Transformers 4.57.6, eager Qwen3 attention, default RoPE and batch one are
audited here. No parameters are fitted or selected by this execution adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from src.module2.extraction import rotate_half, validate_qwen3


@dataclass
class GuidelinePrefix:
    prefix_ids: object
    cache: object
    cos: object
    sin: object
    rule_positions: dict
    geometry: dict
    model_identity: int

    @property
    def prefix_length(self):
        return int(self.prefix_ids.shape[1])


@dataclass
class GuidelineSuffixTail:
    prefix: GuidelinePrefix
    input_ids: object
    layer: int
    hidden: object
    layer_kwargs: dict
    baseline: dict


def _runtime(model):
    import transformers
    geometry = validate_qwen3(model)
    if transformers.__version__ != "4.57.6":
        raise ValueError("Guideline cache sharing is audited for transformers 4.57.6 only")
    if model.config._attn_implementation != "eager":
        raise ValueError("Guideline diagnostics require eager Qwen3 attention")
    return geometry


def _input_ids(model, input_ids):
    import torch
    ids = torch.as_tensor(input_ids, device=model.get_input_embeddings().weight.device)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError("Expected complete unpadded input_ids with shape [1, length >= 2]")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must have an integer dtype")
    if bool((ids < 0).any()) or bool((ids >= model.config.vocab_size).any()):
        raise ValueError("Token ID outside vocabulary")
    return ids


def prepare_guideline_prefix(model, input_ids, prefix_length, rule_positions):
    """Prefill the first ``prefix_length`` tokens of a complete token sequence.

    The caller determines the guideline/suffix boundary in the actual full
    tokenization. ``rule_positions`` maps rule IDs to disjoint ID/action token
    positions, all strictly inside that boundary. Reuse the returned object
    with any full input whose prefix tokens are exactly equal.
    """
    import torch
    geometry = _runtime(model)
    ids = _input_ids(model, input_ids)
    if (not isinstance(prefix_length, int) or isinstance(prefix_length, bool)
            or not 0 < prefix_length < ids.shape[1]):
        raise ValueError("Guideline prefix must leave a nonempty patient/question suffix")
    if not isinstance(rule_positions, dict) or not rule_positions:
        raise ValueError("Explicit rule positions are required")
    mapped, occupied = {}, set()
    for rule_id, positions in rule_positions.items():
        values = list(positions)
        if (not values or any(not isinstance(p, int) or isinstance(p, bool)
                             or not 0 <= p < prefix_length for p in values)
                or len(values) != len(set(values))):
            raise ValueError("Rule positions must be unique guideline token indices")
        if occupied.intersection(values):
            raise ValueError("A target token cannot belong to two rules")
        occupied.update(values)
        mapped[str(rule_id)] = sorted(values)
    prefix_ids = ids[:, :prefix_length].detach().clone()
    positions = torch.arange(prefix_length, device=ids.device)[None]
    with torch.inference_mode():
        dummy = torch.empty((1, prefix_length, 1), device=ids.device,
                            dtype=model.get_input_embeddings().weight.dtype)
        cos, sin = model.model.rotary_emb(dummy, positions)
        output = model(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids),
                       position_ids=positions, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    if type(cache).__name__ != "DynamicCache" or len(cache.layers) != geometry["layers"]:
        raise ValueError("Expected a complete unquantized DynamicCache")
    expected = (1, geometry["kv_heads"], prefix_length, geometry["head_dim"])
    for layer in cache.layers:
        if (type(layer).__name__ != "DynamicLayer" or tuple(layer.keys.shape) != expected
                or tuple(layer.values.shape) != expected):
            raise ValueError("Expected full, unpadded, non-sliding guideline cache layers")
    return GuidelinePrefix(prefix_ids, cache, cos[0], sin[0], mapped, geometry, id(model))


def _entry(entry, prefix, kind):
    layer, head = int(entry["layer"]), int(entry["kv_head"])
    if not 0 <= layer < prefix.geometry["layers"] or not 0 <= head < prefix.geometry["kv_heads"]:
        raise ValueError("Intervention addresses an invalid layer/KV head")
    positions = entry.get("positions", prefix.rule_positions.get(str(entry.get("rule_id"))))
    if positions is None:
        raise ValueError("Supply explicit target positions or a mapped rule_id")
    positions = list(positions)
    allowed = set().union(*map(set, prefix.rule_positions.values()))
    if (not positions or any(not isinstance(p, int) or isinstance(p, bool) for p in positions)
            or len(set(positions)) != len(positions) or not set(positions).issubset(allowed)):
        raise ValueError("Every target must be a uniquely mapped guideline token")
    strength = float(entry.get("coefficient", 1.0) if kind == "shift" else entry["bias"])
    if not math.isfinite(strength):
        raise ValueError("Nonfinite intervention strength")
    return layer, head, positions, strength


def _delta(entry, prefix, device):
    import torch
    delta = torch.as_tensor(entry["delta"], device=device, dtype=torch.float32)
    if tuple(delta.shape) != (prefix.geometry["head_dim"],) or not bool(torch.isfinite(delta).all()):
        raise ValueError("Delta must be one finite head-dimensional vector")
    return delta


def edited_guideline_cache(model, prefix, shifts=()):
    """Return independent cache objects; clone keys only in edited layers.

    In the pinned DynamicLayer implementation, suffix updates concatenate and
    rebind tensors. Unedited prefix tensors and values can therefore be shared
    without mutating the original cache. This is guarded by version/type checks.
    """
    from transformers import DynamicCache
    if id(model) != prefix.model_identity or _runtime(model) != prefix.geometry:
        raise ValueError("A guideline prefix belongs to one specific model")
    legacy = list(prefix.cache.to_legacy_cache())
    occupied, copied = set(), set()
    for shift in shifts:
        layer, head, positions, coefficient = _entry(shift, prefix, "shift")
        support = {(layer, head, p) for p in positions}
        if occupied.intersection(support):
            raise ValueError("Overlapping shifts would edit a guideline key twice")
        occupied.update(support)
        key, value = legacy[layer]
        delta = _delta(shift, prefix, key.device)
        if coefficient == 0:
            continue
        if layer not in copied:
            key = key.clone()
            legacy[layer] = (key, value)
            copied.add(layer)
        cos = prefix.cos[positions].to(key.device).float()
        sin = prefix.sin[positions].to(key.device).float()
        rotated = coefficient * (delta[None] * cos + rotate_half(delta)[None] * sin)
        key[0, head, positions] += rotated.to(key.dtype)
    cache = DynamicCache(config=model.config)
    if len(cache.layers) != len(legacy):
        raise ValueError("Cache configuration does not cover the whole guideline")
    for layer, (key, value) in zip(cache.layers, legacy):
        if type(layer).__name__ != "DynamicLayer":
            raise ValueError("Only unquantized DynamicLayer cache objects are supported")
        layer.lazy_initialization(key)
        layer.keys, layer.values = key, value
    return cache


def run_guideline_suffix(model, prefix, input_ids, shifts=(), score_bias=(), *,
                         diagnostic_targets=(), response_targets=(), _tail=None):
    """Edit a fresh guideline-cache view and run the full remaining suffix.

    Shifts use ``layer, kv_head, positions|rule_id, delta, coefficient``. Direct
    score controls use ``layer, kv_head, positions|rule_id, bias`` and apply to
    every suffix query in the associated GQA group. Both leave guideline prefill
    unchanged. Diagnostic targets report actual last-query softmax masses.

    Response targets additionally supply a ``delta`` and return the theoretical
    unit-gain score response q^T R_j delta / sqrt(d), for every target token and
    every GQA query head. Under nonzero edits these are the *changed* queries;
    use an unedited call for baseline Read+/Read- and fixed-query calibration.
    The result contains full-vocabulary final logits; callers may score A/B.
    """
    import torch
    geometry = _runtime(model)
    if id(model) != prefix.model_identity or geometry != prefix.geometry:
        raise ValueError("A guideline prefix cannot be reused with another model")
    ids = _input_ids(model, input_ids)
    n, p = int(ids.shape[1]), prefix.prefix_length
    if n <= p or not torch.equal(ids[:, :p], prefix.prefix_ids):
        raise ValueError("Full input must start with the exact saved guideline tokens")
    if prefix.cache.get_seq_length() != p:
        raise RuntimeError("Reusable guideline prefix cache was changed")
    shifts, score_bias = list(shifts), list(score_bias)
    diagnostic_targets, response_targets = list(diagnostic_targets), list(response_targets)
    cache = edited_guideline_cache(model, prefix, shifts)
    bias_by_layer, diagnostic_by_layer, response_by_layer = {}, {}, {}
    occupied = set()
    for item in score_bias:
        layer, head, positions, strength = _entry(item, prefix, "bias")
        support = {(layer, head, pos) for pos in positions}
        if occupied.intersection(support):
            raise ValueError("Overlapping direct score biases are ambiguous")
        occupied.update(support)
        bias_by_layer.setdefault(layer, []).append((head, positions, strength))
    for index, item in enumerate(diagnostic_targets):
        layer, head, positions, _ = _entry(item, prefix, "shift")
        diagnostic_by_layer.setdefault(layer, []).append((index, head, positions, item.get("rule_id")))
    for index, item in enumerate(response_targets):
        layer, head, positions, _ = _entry(item, prefix, "shift")
        delta = _delta(item, prefix, ids.device)
        response_by_layer.setdefault(layer, []).append((index, head, positions, item.get("rule_id"), delta))
    if _tail is not None:
        if (_tail.prefix is not prefix or not torch.equal(_tail.input_ids, ids)
                or not 0 <= _tail.layer < geometry["layers"]):
            raise ValueError("Suffix tail belongs to a different guideline/patient input")
        requested_layers = set(bias_by_layer) | set(diagnostic_by_layer) | set(response_by_layer)
        requested_layers.update(_entry(item, prefix, "shift")[0] for item in shifts)
        if any(layer < _tail.layer for layer in requested_layers):
            raise ValueError("A suffix tail cannot edit or remeasure a skipped layer")
    handles, queries, rotary = [], {}, {}
    masses, responses = [None] * len(diagnostic_targets), [None] * len(response_targets)
    q_per_kv, q_count, suffix_length = (geometry["queries_per_kv_head"],
                                        geometry["query_heads"], n - p)
    for layer_index, entries in bias_by_layer.items():
        def apply_bias(module, args, kwargs, entries=entries):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or tuple(hidden.shape[:2]) != (1, suffix_length):
                raise ValueError("Unexpected patient/question suffix layout")
            mask = kwargs.get("attention_mask")
            shape = (1, q_count, suffix_length, n)
            if mask is None:
                key_pos = torch.arange(n, device=hidden.device)
                query_pos = torch.arange(p, n, device=hidden.device)
                allowed = key_pos[None, :] <= query_pos[:, None]
                mask = hidden.new_full((1, 1, suffix_length, n), torch.finfo(hidden.dtype).min)
                mask.masked_fill_(allowed[None, None], 0.)
            if mask.ndim != 4 or tuple(mask.shape[-2:]) != (suffix_length, n):
                raise ValueError("Unexpected causal suffix attention mask")
            mask = mask.expand(shape).clone()
            for head, positions, bias in entries:
                mask[0, head * q_per_kv:(head + 1) * q_per_kv, :, positions] += bias
            return args, dict(kwargs, attention_mask=mask)
        handles.append(model.model.layers[layer_index].self_attn.register_forward_pre_hook(
            apply_bias, with_kwargs=True))
    for layer_index, entries in diagnostic_by_layer.items():
        def capture_mass(module, args, output, layer_index=layer_index, entries=entries):
            weights = output[1]
            if weights is None or tuple(weights.shape) != (1, q_count, suffix_length, n):
                raise RuntimeError("Actual eager suffix attention weights were not returned")
            for index, head, positions, rule_id in entries:
                values = weights[0, head * q_per_kv:(head + 1) * q_per_kv, -1][:, positions].float()
                per_query = values.sum(-1)
                if not bool(torch.isfinite(per_query).all()):
                    raise RuntimeError("Nonfinite guideline attention mass")
                masses[index] = {"layer": layer_index, "kv_head": head, "rule_id": rule_id,
                    "positions": positions, "key_count": len(positions), "query_count": q_per_kv,
                    "target_attention_mass": float(per_query.mean()),
                    "per_query_mass": per_query.detach().cpu().tolist(),
                    "per_token_mass": values.mean(0).detach().cpu().tolist(),
                    "per_query_token_mass": values.detach().cpu().tolist(),
                    "measurement": "actual last suffix-query eager softmax weights"}
        handles.append(model.model.layers[layer_index].self_attn.register_forward_hook(capture_mass))
    for layer_index in response_by_layer:
        def capture_query(module, args, output, layer_index=layer_index):
            if output.ndim != 4 or tuple(output.shape[:3]) != (1, suffix_length, q_count):
                raise RuntimeError("Unexpected Qwen3 query-normalization layout")
            queries[layer_index] = output[0, -1].detach().clone()
        def capture_rotary(module, args, kwargs, layer_index=layer_index):
            if "position_embeddings" not in kwargs:
                raise RuntimeError("Actual Qwen3 rotary embeddings are unavailable")
            cos, sin = kwargs["position_embeddings"]
            rotary[layer_index] = (cos[0, -1].detach(), sin[0, -1].detach())
        attention = model.model.layers[layer_index].self_attn
        handles.append(attention.q_norm.register_forward_hook(capture_query))
        handles.append(attention.register_forward_pre_hook(capture_rotary, with_kwargs=True))
    try:
        with torch.inference_mode():
            suffix_positions = torch.arange(p, n, device=ids.device)
            if _tail is None:
                output = model(input_ids=ids[:, p:], attention_mask=torch.ones_like(ids),
                               position_ids=suffix_positions[None], cache_position=suffix_positions,
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = output.logits[0, -1].detach().float().cpu()
            else:
                # Every suffix token is recomputed from the first edited layer.
                # No query or hidden state at/above that layer is reused.
                hidden = _tail.hidden
                kwargs = dict(_tail.layer_kwargs, past_key_values=cache)
                for layer in model.model.layers[_tail.layer:]:
                    hidden = layer(hidden, **kwargs)
                hidden = model.model.norm(hidden)
                logits = model.lm_head(hidden[:, -1:])[0, -1].detach().float().cpu()
        for layer_index, entries in response_by_layer.items():
            query = queries[layer_index]
            co_q, si_q = rotary[layer_index]
            q_post = (query * co_q + rotate_half(query) * si_q).float()
            for index, head, positions, rule_id, delta in entries:
                co = prefix.cos[positions].to(delta.device).float()
                si = prefix.sin[positions].to(delta.device).float()
                rotated_delta = delta[None] * co + rotate_half(delta)[None] * si
                scores = q_post[head * q_per_kv:(head + 1) * q_per_kv] @ rotated_delta.T
                scores *= geometry["head_dim"] ** -0.5
                responses[index] = {"layer": layer_index, "kv_head": head, "rule_id": rule_id,
                    "positions": positions, "key_count": len(positions), "query_count": q_per_kv,
                    "mean": float(scores.mean()), "rms": float(scores.square().mean().sqrt()),
                    "positive_fraction": float((scores > 0).float().mean()),
                    "per_query_mean": scores.mean(1).detach().cpu().tolist(),
                    "per_token_mean": scores.mean(0).detach().cpu().tolist(),
                    "scores": scores.detach().cpu().tolist(),
                    "measurement": "unit-gain theoretical response at the actual last suffix query"}
        if prefix.cache.get_seq_length() != p:
            raise RuntimeError("Suffix execution mutated the reusable guideline cache")
        if any(value is None for value in masses + responses):
            raise RuntimeError("Incomplete suffix diagnostic coverage")
        return {"logits": logits, "attention_mass": masses, "query_response": responses,
                "prefix_length": p, "suffix_length": suffix_length}
    finally:
        for handle in handles:
            handle.remove()


def prepare_guideline_tail(model, prefix, input_ids, layer, *, diagnostic_targets=(), response_targets=()):
    """Capture all suffix tokens entering the earliest allowed edited layer.

    This invokes the full reference path once, saving its baseline diagnostics.
    Subsequent tail runs may edit this layer or any later layer. They recompute
    the whole remaining patient/question suffix at every affected layer. They
    do not expose a partially filled cache for free-running generation.
    """
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < prefix.geometry["layers"]:
        raise ValueError("Invalid suffix-tail start layer")
    captured = {}
    ids = _input_ids(model, input_ids)
    suffix_length = int(ids.shape[1]) - prefix.prefix_length
    def capture(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or tuple(hidden.shape[:2]) != (1, suffix_length):
            raise RuntimeError("Expected the entire unchanged patient/question suffix")
        captured["hidden"] = hidden.detach().clone()
        captured["kwargs"] = {key: value for key, value in kwargs.items()
                              if key not in {"hidden_states", "past_key_values", "past_key_value"}}
    handle = model.model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        baseline = run_guideline_suffix(model, prefix, ids, diagnostic_targets=diagnostic_targets,
                                        response_targets=response_targets)
    finally:
        handle.remove()
    if set(captured) != {"hidden", "kwargs"} or "position_embeddings" not in captured["kwargs"]:
        raise RuntimeError("Actual complete suffix inputs/rotary coefficients were not captured")
    return GuidelineSuffixTail(prefix, ids.detach().clone(), layer, captured["hidden"],
                               captured["kwargs"], baseline)


def run_guideline_tail(model, tail, shifts=(), score_bias=(), *, diagnostic_targets=(), response_targets=()):
    """Run the audited suffix optimization with the reference result schema."""
    return run_guideline_suffix(model, tail.prefix, tail.input_ids, shifts, score_bias,
                                diagnostic_targets=diagnostic_targets,
                                response_targets=response_targets, _tail=tail)
