"""Equivalent single-query suffix execution for a fixed Qwen3 cache edit.

This execution adapter does not fit, select, or alter an intervention. If the
only edited layer is L, the decision token's inputs to layers 0..L are unchanged.
Capture the input to L once and reevaluate L..last for each condition. Prefix
cache tensors are immutable and shared; DynamicLayer.update creates new tensors
with torch.cat. A temporary mixed-length cache is never exposed for generation.

Batch one is intended to be bitwise identical to the original implementation.
Batches 2..4 are available for separately validated nonzero conditions only;
matrix-kernel rounding may differ. Zero-intervention validation remains batch one.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.module2.extraction import rotate_half, validate_qwen3
from src.module3.applicability_shift import _validate_entry, decision_logits as reference_decision_logits


@dataclass
class TailState:
    prefix: object
    layer: int
    kv_head: int
    hidden: object
    layer_kwargs: dict
    baseline: dict


def _validate_runtime(model, state):
    import transformers
    geometry = validate_qwen3(model)
    if transformers.__version__ != "4.57.6":
        raise ValueError("Suffix cache aliasing is audited only for transformers 4.57.6")
    if id(model) != state.model_identity or geometry != state.geometry:
        raise ValueError("Prefix model identity or geometry changed")
    if model.config._attn_implementation != "eager":
        raise ValueError("The suffix diagnostic requires eager attention")
    if any(type(layer).__name__ != "DynamicLayer" for layer in state.cache.layers):
        raise ValueError("Only unquantized, non-sliding DynamicLayer caches are supported")
    return geometry


def prepare_tail(model, state, layer, kv_head, diagnostic_targets):
    """Capture unchanged suffix input from one original full decision forward."""
    _validate_runtime(model, state)
    if not 0 <= layer < state.geometry["layers"] or not 0 <= kv_head < state.geometry["kv_heads"]:
        raise ValueError("Invalid fixed suffix layer or KV head")
    captured = {}
    def capture(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or tuple(hidden.shape[:2]) != (1, 1):
            raise ValueError("Expected one unchanged decision-token hidden state")
        captured["hidden"] = hidden.detach().clone()
        captured["kwargs"] = {key: value for key, value in kwargs.items()
                              if key not in {"hidden_states", "past_key_values", "past_key_value"}}
    handle = model.model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        baseline = reference_decision_logits(model, state, return_diagnostics=True,
                                             diagnostic_targets=diagnostic_targets)
    finally:
        handle.remove()
    if set(captured) != {"hidden", "kwargs"}:
        raise RuntimeError("The suffix input capture did not execute")
    if "position_embeddings" not in captured["kwargs"]:
        raise RuntimeError("Actual Qwen3 position embeddings were not captured")
    return TailState(state, int(layer), int(kv_head), captured["hidden"], captured["kwargs"], baseline)


def _condition_entries(condition, tail):
    shifts, biases = list(condition.get("shifts", ())), list(condition.get("score_bias", ()))
    for kind, entries in (("shift", shifts), ("bias", biases)):
        occupied = set()
        for entry in entries:
            layer, head, positions, _ = _validate_entry(entry, tail.prefix, kind)
            if (layer, head) != (tail.layer, tail.kv_head):
                raise ValueError("The suffix optimization is restricted to its one fixed layer/KV head")
            keys = {(layer, head, p) for p in positions}
            if occupied.intersection(keys):
                raise ValueError("Overlapping intervention entries")
            occupied.update(keys)
    return shifts, biases


def _alias_prefix_cache(model, legacy):
    """Create distinct cache-layer objects without from_legacy_cache's copy.

    In 4.57.6, from_legacy_cache calls update(empty, prefix), whose torch.cat
    copies the entire prefix. Explicitly initialize metadata and bind immutable
    tensors instead. Future update calls rebind these new layer objects only.
    """
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    if len(cache.layers) != len(legacy):
        raise ValueError("Cache configuration does not cover the whole prefix")
    for layer, (key, value) in zip(cache.layers, legacy):
        if type(layer).__name__ != "DynamicLayer":
            raise ValueError("Only the audited DynamicLayer cache can share a prefix")
        layer.lazy_initialization(key)
        layer.keys, layer.values = key, value
    return cache


def tail_decision_batch(model, tail, conditions, *, return_diagnostics=False, diagnostic_targets=()):
    """Run one to four independently specified conditions on one fixed prefix.

    For batch sizes above one, callers must first validate output tolerances on
    their actual model. Empty/zero conditions are prohibited in a multirow batch
    so no approximate batched result can serve as a strict zero-identity check.
    """
    import torch
    state = tail.prefix
    geometry = _validate_runtime(model, state)
    conditions = list(conditions)
    batch = len(conditions)
    if not 1 <= batch <= 4:
        raise ValueError("Suffix diagnostic batch size must be 1..4")
    entries = [_condition_entries(condition, tail) for condition in conditions]
    if batch > 1:
        for shifts, biases in entries:
            nonzero = any(float(s.get("coefficient", 1.)) != 0 and
                          bool(torch.as_tensor(s["delta"]).ne(0).any()) for s in shifts)
            nonzero |= any(float(b["bias"]) != 0 for b in biases)
            if not nonzero:
                raise ValueError("Strict zero-identity conditions must run with batch size one")
    targets = list(diagnostic_targets)
    if return_diagnostics and not targets:
        targets = [{"layer": tail.layer, "kv_head": tail.kv_head,
                    "positions": state.target_positions}]
    diagnostic_by_layer = {}
    for target in targets:
        layer, head, positions, _ = _validate_entry(target, state, "shift")
        if layer < tail.layer:
            raise ValueError("A suffix run cannot remeasure a skipped layer")
        diagnostic_by_layer.setdefault(layer, []).append((head, positions, target.get("rule_id")))

    # No clone of unrelated prefix layers. update() in this audited cache type
    # concatenates and rebinds its attributes; it never changes these tensors.
    legacy = [(key.expand(batch, -1, -1, -1), value.expand(batch, -1, -1, -1))
              for key, value in state.cache.to_legacy_cache()]
    modified_key = None
    for batch_idx, (shifts, _) in enumerate(entries):
        for shift in shifts:
            _, head, positions, coefficient = _validate_entry(shift, state, "shift")
            original_key = legacy[tail.layer][0]
            delta = torch.as_tensor(shift["delta"], dtype=torch.float32, device=original_key.device)
            if tuple(delta.shape) != (geometry["head_dim"],) or not bool(torch.isfinite(delta).all()):
                raise ValueError("Delta must be one finite head-dimensional vector")
            if coefficient == 0:
                continue
            if modified_key is None:
                modified_key = original_key.clone()
            co = state.cos[positions].to(original_key.device).float()
            si = state.sin[positions].to(original_key.device).float()
            rotated = coefficient * (delta[None] * co + rotate_half(delta)[None] * si)
            modified_key[batch_idx, head, positions] += rotated.to(original_key.dtype)
    if modified_key is not None:
        legacy[tail.layer] = (modified_key, legacy[tail.layer][1])
    cache = _alias_prefix_cache(model, legacy)
    handles, masses = [], [[] for _ in conditions]
    if any(biases for _, biases in entries):
        def bias_hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or tuple(hidden.shape[:2]) != (batch, 1):
                raise ValueError("Unexpected suffix decision-query batch")
            shape = (batch, geometry["query_heads"], 1, state.prefix_length + 1)
            mask = kwargs.get("attention_mask")
            mask = hidden.new_zeros(shape) if mask is None else mask.expand(shape).clone()
            group = geometry["queries_per_kv_head"]
            for batch_idx, (_, biases) in enumerate(entries):
                for bias in biases:
                    _, head, positions, value = _validate_entry(bias, state, "bias")
                    mask[batch_idx, head * group:(head + 1) * group, 0, positions] += value
            return args, dict(kwargs, attention_mask=mask)
        handles.append(model.model.layers[tail.layer].self_attn.register_forward_pre_hook(
            bias_hook, with_kwargs=True))
    if return_diagnostics:
        for layer, layer_targets in diagnostic_by_layer.items():
            def capture_mass(module, args, output, layer=layer, layer_targets=layer_targets):
                weights = output[1]
                expected = (batch, geometry["query_heads"], 1, state.prefix_length + 1)
                if weights is None or tuple(weights.shape) != expected:
                    raise RuntimeError("Actual eager suffix attention weights are missing")
                group = geometry["queries_per_kv_head"]
                for head, positions, rule_id in layer_targets:
                    for batch_idx in range(batch):
                        values = weights[batch_idx, head * group:(head + 1) * group, 0][:, positions].float().sum(-1)
                        if not bool(torch.isfinite(values).all()):
                            raise RuntimeError("Nonfinite actual target attention mass")
                        masses[batch_idx].append({"layer": layer, "kv_head": head, "rule_id": rule_id,
                            "positions": list(positions), "key_count": len(positions), "query_count": group,
                            "target_attention_mass": float(values.mean()),
                            "per_query_mass": values.detach().cpu().tolist(),
                            "measurement": "actual eager softmax weights at the decision prompt token"})
            handles.append(model.model.layers[layer].self_attn.register_forward_hook(capture_mass))
    try:
        with torch.inference_mode():
            hidden = tail.hidden.expand(batch, -1, -1).contiguous()
            kwargs = dict(tail.layer_kwargs, past_key_values=cache)
            if kwargs.get("attention_mask") is not None:
                kwargs["attention_mask"] = kwargs["attention_mask"].expand(batch, -1, -1, -1)
            if kwargs.get("position_ids") is not None:
                kwargs["position_ids"] = kwargs["position_ids"].expand(batch, -1)
            for layer in model.model.layers[tail.layer:]:
                hidden = layer(hidden, **kwargs)
            logits = model.lm_head(model.model.norm(hidden))[:, -1].detach().float().cpu()
        if state.cache.get_seq_length() != state.prefix_length:
            raise RuntimeError("The suffix run mutated the reusable prefix cache")
        if return_diagnostics:
            return [{"logits": logits[i], "attention_mass": masses[i]} for i in range(batch)]
        return [logits[i] for i in range(batch)]
    finally:
        for handle in handles:
            handle.remove()


def tail_decision_logits(model, tail, shifts=(), score_bias=(), *, return_diagnostics=False,
                         diagnostic_targets=()):
    """Batch-one suffix path with the original decision_logits result schema."""
    return tail_decision_batch(model, tail, [{"shifts": shifts, "score_bias": score_bias}],
        return_diagnostics=return_diagnostics, diagnostic_targets=diagnostic_targets)[0]


def compare_results(reference, actual, *, logit_atol=0., attention_atol=0.):
    """Fail closed on complete-vocabulary or per-query-mass disagreement."""
    import torch
    ref_logits = reference["logits"] if isinstance(reference, dict) else reference
    new_logits = actual["logits"] if isinstance(actual, dict) else actual
    if ref_logits.shape != new_logits.shape or not bool(torch.isfinite(new_logits).all()):
        raise RuntimeError("Invalid optimized logits")
    logit_error = float((ref_logits - new_logits).abs().max())
    attention_error = 0.
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or len(reference["attention_mass"]) != len(actual["attention_mass"]):
            raise RuntimeError("Optimized attention diagnostics differ in coverage")
        for original, changed in zip(reference["attention_mass"], actual["attention_mass"]):
            for key in ("layer", "kv_head", "rule_id", "positions", "query_count", "key_count"):
                if original[key] != changed[key]:
                    raise RuntimeError("Optimized attention diagnostics changed support/order")
            error = max(abs(a-b) for a,b in zip(original["per_query_mass"], changed["per_query_mass"]))
            attention_error = max(attention_error, error)
    if logit_error > logit_atol or attention_error > attention_atol:
        raise RuntimeError(f"Suffix equivalence failed: logits={logit_error:g}, attention={attention_error:g}")
    if int(ref_logits.argmax()) != int(new_logits.argmax()):
        raise RuntimeError("Optimized execution changed the most probable vocabulary token")
    return {"max_full_vocabulary_logit_error": logit_error,
            "max_per_query_attention_mass_error": attention_error,
            "logit_atol": logit_atol, "attention_atol": attention_atol,
            "top_vocabulary_token_identical": True}
