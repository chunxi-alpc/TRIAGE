"""Same eager attention arithmetic, with CPU position bookkeeping and deferred reads.

The caller supplies the absolute start of the first suffix forward. Every layer
advances its own position counter by the forwarded token count. No GPU position
scalar is copied to the CPU during decoding. Recorded masses retain the original
indexing, float conversion and sum; only their host transfer is deferred.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


def _materialize_masses(stats):
    """Pack already computed masses without arithmetic, then perform one transfer."""
    items = [(entry, rid, mass) for entry in stats for rid, mass in entry["mass"].items()]
    if not items:
        return
    values = torch.cat([mass.reshape(-1) for _, _, mass in items]).detach().cpu().tolist()
    offset = 0
    for entry, rid, mass in items:
        size = mass.numel()
        entry["mass"][rid] = values[offset:offset + size]
        offset += size


@contextmanager
def attention_probe(backend, heads, positions, gains, bias_unit, *, initial_start,
                    capture_positions=None):
    """Reference score math and bias order; only CPU synchronization is changed."""
    if isinstance(initial_start, bool) or not isinstance(initial_start, int) or initial_start < 0:
        raise ValueError("initial_start must be a nonnegative integer")
    original = {}
    stats = []
    next_position = {layer: initial_start for layer in {h[0] for h in heads}}
    for layer in sorted({h[0] for h in heads}):
        module = backend.blocks[layer].self_attn
        original[layer] = module.forward
        localheads = [h[1] for h in heads if h[0] == layer]

        def forward(self, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None,
                    _layer=layer, _heads=localheads, **kwargs):
            input_shape = hidden_states.shape[:-1]
            shape = (*input_shape, -1, self.head_dim)
            query_absolute_position = next_position[_layer] + hidden_states.shape[-2] - 1
            next_position[_layer] = query_absolute_position + 1
            q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position})
            kk = repeat_kv(k, self.num_key_value_groups)
            vv = repeat_kv(v, self.num_key_value_groups)
            scores = torch.matmul(q, kk.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :kk.shape[-2]]
            for kv in _heads:
                name = f"{_layer}:{kv}"
                if name in bias_unit:
                    for rid, indices in positions.items():
                        if gains[rid] == 0:
                            continue
                        scores[:, kv * backend.group_size:(kv + 1) * backend.group_size,
                               :, list(indices)] += gains[rid] * bias_unit[name]
            weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            for kv in (_heads if capture_positions is None or
                       query_absolute_position in capture_positions else []):
                masses = {rid: weights[0, kv * backend.group_size:(kv + 1) * backend.group_size,
                                       -1, list(indices)].float().sum(-1).detach()
                          for rid, indices in positions.items()}
                stats.append({"layer": _layer, "kv_head": kv,
                              "query_absolute_position": query_absolute_position, "mass": masses})
            result = torch.matmul(weights, vv).transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(result), weights

        module.forward = MethodType(forward, module)
    succeeded = False
    try:
        yield stats
        succeeded = True
    finally:
        for layer, forward in original.items():
            backend.blocks[layer].self_attn.forward = forward
        # Preserve the original exception if a forward failed (e.g. OOM).
        if succeeded:
            _materialize_masses(stats)
