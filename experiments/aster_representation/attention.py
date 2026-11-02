from contextlib import contextmanager
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

@contextmanager
def attention_control(backend):
    originals = {}
    for layer, block in enumerate(backend.blocks):
        module = block.self_attn
        originals[layer] = module.forward

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
            shape = hidden_states.shape[:-1]
            hs = (*shape, -1, self.head_dim)
            q = self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx, {'sin': sin, 'cos': cos, 'cache_position': cache_position})
            kk, vv = (repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups))
            n = q.shape[-2]
            parts = []
            start_abs = int(cache_position[0]) if cache_position is not None else kk.shape[-2] - n
            for st in range(0, n, 256):
                en = min(st + 256, n)
                scores = torch.matmul(q[:, :, st:en], kk.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask[:, :, st:en, :kk.shape[-2]]
                weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                parts.append(torch.matmul(weights, vv).transpose(1, 2).contiguous())
            out = torch.cat(parts, dim=1).reshape(*shape, -1)
            return (self.o_proj(out), None)
        module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        for layer, old in originals.items():
            backend.blocks[layer].self_attn.forward = old
