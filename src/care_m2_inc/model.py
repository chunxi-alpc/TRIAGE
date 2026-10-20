"""Qwen3-compatible scoring and residual intervention helpers.

The runner scores short option continuations instead of free-form generations;
this makes every method comparable and keeps answer spans out of the edit
selector.  It works with standard Hugging Face causal language models and
records all target positions used by the hook.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class Intervention:
    layer: int | None = None
    vector: torch.Tensor | None = None
    scale: float = 0.0
    positions: tuple[int, ...] | None = None


def _layers(model: Any):
    for path in ("model.layers", "transformer.h", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("Cannot locate transformer layers for residual hook")


def _output_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


@contextmanager
def residual_intervention(model: Any, intervention: Intervention, token_position: int):
    """Add a residual vector at exactly the final prompt token."""
    if intervention.vector is None or intervention.layer is None or intervention.scale == 0:
        yield {"changed_l1": 0.0, "target_position": token_position, "layer": intervention.layer}
        return
    layers = _layers(model)
    if not 0 <= intervention.layer < len(layers):
        raise ValueError(f"Invalid intervention layer: {intervention.layer}")
    vector = intervention.vector.detach().to(next(model.parameters()).device)
    if vector.ndim != 1:
        raise ValueError("Intervention vector must be one-dimensional")
    positions = tuple(intervention.positions or (token_position,))
    stats = {"changed_l1": 0.0, "target_position": token_position, "layer": intervention.layer, "positions": positions}

    def hook(_module, _inputs, output):
        hidden = _output_hidden(output)
        if hidden.shape[-1] != vector.numel():
            raise ValueError(f"Vector dimension {vector.numel()} != hidden dimension {hidden.shape[-1]}")
        updated = hidden.clone()
        for position in positions:
            if not 0 <= position < updated.shape[1]:
                continue
            delta = vector.to(dtype=updated.dtype) * float(intervention.scale)
            updated[:, position, :] = updated[:, position, :] + delta
            stats["changed_l1"] += float(delta.float().abs().sum().item())
        if isinstance(output, tuple):
            return (updated, *output[1:])
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = updated
            return output
        return updated

    handle = layers[intervention.layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


def capture_hidden(model: Any, tokenizer: Any, prompt: str, layer: int, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = _layers(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = _output_hidden(output).detach()[:, -1, :].float().cpu()

    handle = layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Layer hook did not capture a hidden state")
    return captured["hidden"][0], {"prompt_tokens": int(encoded["input_ids"].shape[1]), "layer": layer}


def _forward_logits(model: Any, tokenizer: Any, prompt: str, option: str, device: torch.device,
                    intervention: Intervention | None = None) -> tuple[float, dict[str, Any]]:
    # A leading space is intentionally part of the candidate continuation. It
    # is applied equally to every option and recorded in the run manifest.
    continuation = option if option.startswith(" ") else " " + option
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=True)["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    full_ids = full_ids.to(device)
    attention = torch.ones_like(full_ids)
    kwargs = {"input_ids": full_ids, "attention_mask": attention, "use_cache": False}
    if intervention is None:
        context = nullcontext()
    else:
        context = residual_intervention(model, intervention, prompt_len - 1)
    with context as hook_stats:
        with torch.inference_mode():
            output = model(**kwargs)
            logits = output.logits.float()
    ids = full_ids[0]
    # Teacher-force only newly appended tokens.  Candidate strings are short;
    # this also supports multi-token route names in N5.
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    token_logprob = 0.0
    for index in range(prompt_len, int(ids.shape[0])):
        token_logprob += float(lp[index - 1, ids[index]].item())
    stats = {"prompt_tokens": prompt_len, "full_tokens": int(ids.shape[0]), "token_count": int(ids.shape[0] - prompt_len),
             "hook": hook_stats if intervention is not None else None}
    return token_logprob, stats


class OptionScorer:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.forward_count = 0

    def score(self, prompt: str, options: dict[str, str], intervention: Intervention | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        scores: dict[str, float] = {}
        details: dict[str, Any] = {"options": {}, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        for key, text in options.items():
            score, stats = _forward_logits(self.model, self.tokenizer, prompt, text, self.device, intervention)
            self.forward_count += 1
            scores[key] = score
            details["options"][key] = stats
        return scores, details


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def load_qwen(model_path: str, device: str = "cuda", dtype: str = "bfloat16") -> tuple[Any, Any, torch.device, dict[str, Any]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for GPU stages") from exc
    target = torch.device(device)
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True,
                                                 attn_implementation="sdpa")
    model.to(target)
    model.eval()
    config = model.config
    fingerprint = {"model_path": str(model_path), "model_name_or_path": getattr(config, "_name_or_path", ""),
                   "model_type": getattr(config, "model_type", ""), "revision": getattr(config, "_commit_hash", None),
                   "dtype": dtype, "device": str(target), "attention_implementation": "sdpa",
                   "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
                   "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
                   "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
                   "hidden_size": int(getattr(config, "hidden_size", -1)),
                   "tokenizer_class": tokenizer.__class__.__name__}
    return model, tokenizer, target, fingerprint


def hidden_dim(model: Any) -> int:
    return int(getattr(model.config, "hidden_size", model.get_input_embeddings().weight.shape[-1]))


def default_layer(model: Any) -> int:
    return max(0, int(getattr(model.config, "num_hidden_layers", 1)) * 2 // 3)
