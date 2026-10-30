"""Qwen3 eager adapter recording actual and theoretical per-token score changes."""
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from src.aster.manuscript_core import rotate

@contextmanager
def probe(backend, heads, positions, gains, bias, base_layers, edited_layers, direction_map,
          cos_prefix, sin_prefix, initial_start, last_prompt_position, capture_offsets=(0,1,2,4,8,16)):
    originals, traces = {}, []
    next_positions = {layer: initial_start for layer, _ in heads}
    for layer in sorted({layer for layer, _ in heads}):
        module = backend.blocks[layer].self_attn
        originals[layer] = module.forward
        localheads = [kv for ll, kv in heads if ll == layer]
        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                    cache_position=None, _layer=layer, _heads=localheads, **kwargs):
            shape = (*hidden_states.shape[:-1], -1, self.head_dim)
            absolute = next_positions[_layer] + hidden_states.shape[-2] - 1
            next_positions[_layer] = absolute + 1
            q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx, {"sin":sin, "cos":cos, "cache_position":cache_position})
            kk, vv = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
            scores = torch.matmul(q, kk.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:,:,:,:kk.shape[-2]]
            for kv in _heads:
                name = f"{_layer}:{kv}"
                if name in bias:
                    for rid, ids in positions.items():
                        scores[:,kv*backend.group_size:(kv+1)*backend.group_size,:,list(ids)] += gains[rid] * bias[name]
            weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if absolute-last_prompt_position in capture_offsets:
                for kv in _heads:
                    begin, end = kv*backend.group_size, (kv+1)*backend.group_size
                    query = q[0,begin:end,-1].float()
                    local_scores = scores[0,begin:end,-1].float()
                    undo_scores = local_scores.clone()
                    actuals, theories, details = {}, {}, {}
                    for rid, ids in positions.items():
                        ids = list(ids)
                        delta = edited_layers[_layer][0][0,kv,ids].float() - base_layers[_layer][0][0,kv,ids].float()
                        actual = query @ delta.T * self.scaling
                        if f"{_layer}:{kv}" in bias:
                            actual = actual + gains[rid]*bias[f"{_layer}:{kv}"]
                        actuals[rid] = actual
                        undo_scores[:,ids] -= actual
                        vec = direction_map.get((_layer,kv))
                        if vec is None:
                            theory = torch.zeros_like(actual)
                        else:
                            rotated = rotate(vec[None], cos_prefix[ids], sin_prefix[ids]).to(query.device).float()
                            theory = query @ rotated.T * self.scaling * gains[rid]
                        if f"{_layer}:{kv}" in bias:
                            theory = theory + gains[rid]*bias[f"{_layer}:{kv}"]
                        theories[rid] = theory
                    undo_weights = torch.softmax(undo_scores, -1)
                    for rid, ids in positions.items():
                        ids=list(ids)
                        details[rid] = {"target_positions":ids,
                            "score_delta_actual":actuals[rid].detach().cpu(),
                            "score_delta_theory":theories[rid].detach().cpu(),
                            "attention_per_token":weights[0,begin:end,-1,ids].float().detach().cpu(),
                            "mass":weights[0,begin:end,-1,ids].float().sum(-1).detach().cpu(),
                            "mass_same_query_undo":undo_weights[:,ids].sum(-1).detach().cpu()}
                    traces.append({"layer":_layer,"kv_head":kv,"query_heads":list(range(begin,end)),
                        "query_absolute_position":absolute,"query_offset":absolute-last_prompt_position,"rules":details})
            result = torch.matmul(weights, vv).transpose(1,2).contiguous().reshape(*hidden_states.shape[:-1],-1)
            return self.o_proj(result), weights
        module.forward = MethodType(forward,module)
    try:
        yield traces
    finally:
        for layer, forward in originals.items():
            backend.blocks[layer].self_attn.forward = forward
