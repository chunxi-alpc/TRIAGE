"""Batched exact single-head interventions for one immutable Qwen3 context.

This module never loads a model. Each batch row applies exactly one intervention,
using the same final-token BF16 replay and per-row FP32 label projection as Engine.
The original prefix tensors are read-only. A disposable cache presents expanded
prefix views; concatenated final-token K/V tensors live only within their layer.
"""
from __future__ import annotations
import copy
import time
from types import MethodType
import torch


def _read_only_update(layer, key_states, value_states, cache_kwargs=None):
    # Native DynamicLayer.update uses these same concatenations, then stores
    # them. Final-token patch replay never consumes the extended cache again,
    # so storing it only duplicates B copies of every layer's long prefix.
    if layer.keys is None:
        return key_states, value_states
    return (torch.cat((layer.keys, key_states), dim=-2),
            torch.cat((layer.values, value_states), dim=-2))


def _expanded_prefix(prefix, batch_size):
    replay = copy.copy(prefix)
    replay.layers = [copy.copy(layer) for layer in prefix.layers]
    for layer in replay.layers:
        if layer.keys is not None:
            layer.keys = layer.keys.expand(batch_size, *layer.keys.shape[1:])
            layer.values = layer.values.expand(batch_size, *layer.values.shape[1:])
        layer.update = MethodType(_read_only_update, layer)
    return replay


def prefix_nbytes(context):
    """Existing prefix storage, useful when choosing a patch batch size."""
    return sum(x.numel() * x.element_size()
               for layer in context.prefix.layers
               for x in (layer.keys, layer.values) if x is not None)


def batched_single_head(eng, context, donor_heads, heads, batch_size=8):
    """Return {global_head: patched_score} for separate single-head interventions.

    donor_heads is the donor input's (all_heads, head_dim) activation array.
    No prefix cache or input context is modified. Batch size controls throughput;
    comparison with batch_size=1 can assess the BF16 GEMM batch-shape effect.
    Calls on the same Engine must remain sequential, just like Engine.run.
    """
    heads = [int(h) for h in heads]
    if not heads:
        return {}
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if eng.model_name != 'qwen':
        raise ValueError('This final-token batching implementation is for Qwen3')
    if eng._active:
        raise RuntimeError('Overlapping model forwards')
    if any(h < 0 or h >= eng.num_heads for h in heads):
        raise ValueError('Head outside model')
    started = time.perf_counter()
    results = {}
    eng._active = True
    total_batches = 0
    try:
        # Keep embedding lookup shape identical to the original one-input run.
        final_embedding = eng.model.get_input_embeddings()(context.input_ids[:, -1:]).detach()
        donor = torch.as_tensor(donor_heads).reshape(eng.num_heads, eng.head_dim)
        cache_position = torch.tensor([context.length - 1], device=eng.device)
        position_ids = torch.tensor([[context.length - 1]], device=eng.device)
        for start in range(0, len(heads), int(batch_size)):
            selected = heads[start:start + int(batch_size)]
            n = len(selected)
            grouped = {}
            for row, head in enumerate(selected):
                grouped.setdefault(head // eng.n_heads, []).append((row, head % eng.n_heads, head))
            handles = []
            def make_hook(entries):
                rows = torch.tensor([r for r, _, _ in entries], device=eng.device)
                query_heads = torch.tensor([q for _, q, _ in entries], device=eng.device)
                values = donor[[h for _, _, h in entries]].to(device=eng.device, dtype=final_embedding.dtype)
                def hook(module, args):
                    raw = args[0]
                    expected = (n, 1, eng.n_heads * eng.head_dim)
                    if tuple(raw.shape) != expected:
                        raise RuntimeError('Wrong batched final-token head position')
                    work = raw.reshape(n, eng.n_heads, eng.head_dim).clone()
                    work[rows, query_heads] = values
                    return (work.reshape_as(raw),) + args[1:]
                return hook
            replay = _expanded_prefix(context.prefix, n)
            try:
                for layer_idx, entries in grouped.items():
                    handles.append(eng.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(make_hook(entries)))
                with torch.no_grad():
                    output = eng.model.model(
                        inputs_embeds=final_embedding.expand(n, -1, -1),
                        attention_mask=torch.ones((n, context.length), dtype=context.input_ids.dtype, device=eng.device),
                        past_key_values=replay, use_cache=True,
                        cache_position=cache_position, position_ids=position_ids)
                    # Preserve Engine's 1D FP32 projection operation per example.
                    for row, head in enumerate(selected):
                        score = eng._score_hidden(output.last_hidden_state[row, -1])
                        results[head] = float(score.detach())
                    del output
            finally:
                for handle in handles:
                    handle.remove()
                del replay
            total_batches += 1
        eng.counts['final_forwards'] += len(heads)
        eng.counts['batched_patch_calls'] = eng.counts.get('batched_patch_calls', 0) + total_batches
        eng.counts['final_seconds'] += time.perf_counter() - started
        return results
    finally:
        eng._active = False
