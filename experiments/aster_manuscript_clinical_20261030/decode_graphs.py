"""Optional single-token Qwen3 MLP-tail CUDA graphs; never enabled on import.

This experimental adapter replays existing eager operations from Transformers
4.57.6: the Qwen3DecoderLayer post-attention norm/MLP/residual tail, and optionally
standard Qwen3Attention Q/K/V projections, Q/K norms, and RoPE. Dynamic cache
updates, attention score/softmax/value products, and output projection stay in
the original eager order. No compilation, fusion, dtype change, cache change,
or sampling change is introduced here.
The output is cloned after every replay so consumers never receive a persistent
CUDA-graph output buffer that a later invocation could overwrite.

Only an explicit ``with decode_mlp_graphs(model)`` enables the adapter. The
caller must first compare logits, caches and complete seeded generations against
the eager reference on real weights, and measure speed and memory. This module
alone does not establish numerical equivalence or a speed improvement.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, apply_rotary_pos_emb, eager_attention_forward,
)


class _TailGraph:
    def __init__(self, layer, warmup_stream, warmup_steps):
        self.layer = layer
        weight = layer.mlp.gate_proj.weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.shape = (1, 1, layer.mlp.gate_proj.in_features)
        self.replays = 0
        self.fallbacks = 0
        self.static_input = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                warmup_output = self.eager(self.static_input)
            del warmup_output
        current.wait_stream(warmup_stream)
        # The explicit capture stream preserves the warmup dependencies. Each
        # layer has its own graph pool; no live intermediates are shared between
        # graphs. All replayed arithmetic uses the original bound submodules.
        with torch.cuda.graph(self.graph, stream=warmup_stream):
            self.static_output = self.eager(self.static_input)
        current.wait_stream(warmup_stream)

    def eager(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def eligible(self, hidden_states):
        return (not self.layer.training and torch.is_inference_mode_enabled()
                and hidden_states.device == self.device
                and hidden_states.dtype == self.dtype
                and tuple(hidden_states.shape) == self.shape
                and not torch.is_autocast_enabled(self.device.type))

    def replay(self, hidden_states):
        self.static_input.copy_(hidden_states)
        self.graph.replay()
        self.replays += 1
        return self.static_output.clone()


class _QKVGraph:
    def __init__(self, attention, warmup_stream, warmup_steps):
        self.attention = attention
        weight = attention.q_proj.weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.shape = (1, 1, attention.q_proj.in_features)
        self.rotation_shape = (1, 1, attention.head_dim)
        self.replays = 0
        self.fallbacks = 0
        self.static_input = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        self.static_cos = torch.ones(self.rotation_shape, device=self.device, dtype=self.dtype)
        self.static_sin = torch.zeros(self.rotation_shape, device=self.device, dtype=self.dtype)
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device)
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                warmup_output = self.eager(self.static_input, self.static_cos, self.static_sin)
            del warmup_output
        current.wait_stream(warmup_stream)
        with torch.cuda.graph(self.graph, stream=warmup_stream):
            self.static_outputs = self.eager(self.static_input, self.static_cos, self.static_sin)
        current.wait_stream(warmup_stream)

    def eager(self, hidden_states, cos, sin):
        attention = self.attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query_states = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states

    def eligible(self, hidden_states, position_embeddings, past_key_values):
        if (self.attention.training or not torch.is_inference_mode_enabled()
                or self.attention.config._attn_implementation != "eager"
                or not isinstance(past_key_values, DynamicCache)
                or past_key_values.get_seq_length(self.attention.layer_idx) <= 0
                or hidden_states.device != self.device or hidden_states.dtype != self.dtype
                or tuple(hidden_states.shape) != self.shape
                or torch.is_autocast_enabled(self.device.type)
                or not isinstance(position_embeddings, (tuple, list))
                or len(position_embeddings) != 2):
            return False
        return all(isinstance(value, torch.Tensor) and value.device == self.device
                   and value.dtype == self.dtype and tuple(value.shape) == self.rotation_shape
                   for value in position_embeddings)

    def replay(self, hidden_states, position_embeddings):
        self.static_input.copy_(hidden_states)
        self.static_cos.copy_(position_embeddings[0])
        self.static_sin.copy_(position_embeddings[1])
        self.graph.replay()
        self.replays += 1
        return tuple(value.clone() for value in self.static_outputs)


class DecodeMLPGraphs:
    """Explicit lifecycle for one single-threaded, inference-only Qwen3 model.

    Attention remains a dynamic call to ``layer.self_attn``. Temporary probe or
    native-memory contexts entered inside this adapter override the QKV wrapper,
    execute their eager attention code, and restore the wrapper on exit. Existing
    hooks in captured norm/MLP/projection submodules are rejected at preparation;
    graph replay cannot invoke Python hooks. Do not add such hooks while active.
    """
    def __init__(self, model, *, warmup_steps=3, qkv=True):
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
            raise ValueError("warmup_steps must be a positive integer")
        self.model = model
        self.warmup_steps = warmup_steps
        self.qkv_enabled = bool(qkv)
        self.tails = []
        self.qkvs = []
        self.attention_originals = []
        self.originals = []
        self.prepared = False
        self.installed = False

    def prepare(self):
        if self.prepared or self.installed:
            raise RuntimeError("decode graphs already prepared or installed")
        if self.model.training or self.model.config.model_type != "qwen3":
            raise ValueError("decode graphs require an eval-mode Qwen3 model")
        layers = list(self.model.model.layers)
        if not layers:
            raise ValueError("model has no decoder layers")
        device = layers[0].mlp.gate_proj.weight.device
        if device.type != "cuda":
            raise ValueError("decode graphs require a CUDA model")
        if torch.is_autocast_enabled(device.type):
            raise ValueError("autocast is not part of this fixed-dtype adapter")
        for layer in layers:
            if layer.training or layer.__class__.__name__ != "Qwen3DecoderLayer":
                raise ValueError("unsupported decoder layer")
            roots = [layer.post_attention_layernorm, layer.mlp]
            if self.qkv_enabled:
                attention = layer.self_attn
                if (attention.__class__ is not Qwen3Attention
                        or attention.forward.__func__ is not Qwen3Attention.forward
                        or attention.config._attn_implementation != "eager"):
                    raise ValueError("install decode graphs outside temporary attention adapters; eager Qwen3Attention required")
                roots += [attention.q_proj, attention.k_proj, attention.v_proj,
                          attention.q_norm, attention.k_norm]
            for root in roots:
                for module in root.modules():
                    if module.training or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
                        raise ValueError("tail module training or hooks are incompatible with graph replay")
                for parameter in root.parameters():
                    if parameter.device != device or parameter.dtype != layer.mlp.gate_proj.weight.dtype:
                        raise ValueError("tail parameters must use one CUDA device and dtype")
        with torch.inference_mode(), torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            for layer in layers:
                self.tails.append(_TailGraph(layer, stream, self.warmup_steps))
                if self.qkv_enabled:
                    self.qkvs.append(_QKVGraph(layer.self_attn, stream, self.warmup_steps))
            # Ensure captured initial outputs and graph state are ready before
            # installation. This one setup wait is outside timed decode loops.
            stream.synchronize()
        self.prepared = True
        return self

    def install(self):
        if not self.prepared or self.installed:
            raise RuntimeError("prepare once before installing decode graphs")
        for layer, tail in zip(self.model.model.layers, self.tails):
            original = layer.forward
            self.originals.append(original)

            def forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_values=None, use_cache=False, cache_position=None,
                        position_embeddings=None, _original=original, _tail=tail, **kwargs):
                cached_decode = (use_cache and past_key_values is not None
                                 and past_key_values.get_seq_length(self.self_attn.layer_idx) > 0)
                if not cached_decode or not _tail.eligible(hidden_states) or "past_key_value" in kwargs:
                    _tail.fallbacks += 1
                    return _original(hidden_states=hidden_states,
                        attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, use_cache=use_cache,
                        cache_position=cache_position, position_embeddings=position_embeddings,
                        **kwargs)
                # Exact attention and attention-residual sequence from the
                # installed Transformers 4.57.6 Qwen3DecoderLayer.forward.
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + hidden_states
                return _tail.replay(hidden_states)

            layer.forward = MethodType(forward, layer)
        for layer, qkv_graph in zip(self.model.model.layers, self.qkvs):
            attention = layer.self_attn
            original = attention.forward
            self.attention_originals.append(original)

            def attention_forward(self, hidden_states, position_embeddings,
                                  attention_mask, past_key_values=None,
                                  cache_position=None, _original=original,
                                  _qkv=qkv_graph, **kwargs):
                if (not _qkv.eligible(hidden_states, position_embeddings, past_key_values)
                        or "past_key_value" in kwargs):
                    _qkv.fallbacks += 1
                    return _original(hidden_states=hidden_states,
                        position_embeddings=position_embeddings, attention_mask=attention_mask,
                        past_key_values=past_key_values, cache_position=cache_position,
                        **kwargs)
                input_shape = hidden_states.shape[:-1]
                query_states, key_states, value_states = _qkv.replay(hidden_states, position_embeddings)
                cos, sin = position_embeddings
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs)
                attn_output, attn_weights = eager_attention_forward(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling, sliding_window=self.sliding_window,
                    **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            attention.forward = MethodType(attention_forward, attention)
        self.installed = True
        return self

    def restore(self):
        for layer, original in zip(self.model.model.layers, self.originals):
            layer.forward = original
        for layer, original in zip(self.model.model.layers, self.attention_originals):
            layer.self_attn.forward = original
        self.originals = []
        self.attention_originals = []
        self.installed = False

    def summary(self):
        return {"prepared": self.prepared, "installed": self.installed,
                "layers": len(self.tails),
                "replays_by_layer": [tail.replays for tail in self.tails],
                "fallbacks_by_layer": [tail.fallbacks for tail in self.tails],
                "graph_output_cloned": True,
                "qkv_graphs_enabled": self.qkv_enabled,
                "qkv_replays_by_layer": [graph.replays for graph in self.qkvs],
                "qkv_fallbacks_by_layer": [graph.fallbacks for graph in self.qkvs],
                "scope": "single-token post-attention norm/MLP/residual; optional QKV projections, Q/K norms, RoPE",
                "real_weight_validation_report_required": True}


@contextmanager
def decode_mlp_graphs(model, *, warmup_steps=3, qkv=True):
    adapter = DecodeMLPGraphs(model, warmup_steps=warmup_steps, qkv=qkv)
    try:
        adapter.prepare().install()
        yield adapter
    finally:
        adapter.restore()
