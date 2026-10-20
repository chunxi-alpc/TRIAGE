from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from .io import read_jsonl, write_json
from .model import Intervention, OptionScorer, default_layer, hidden_dim, load_qwen, _layers


def run(model_path: str, data_dir: str | Path, out: str | Path, device: str = "cuda") -> dict:
    model, tokenizer, target, fingerprint = load_qwen(model_path, device)
    rows = read_jsonl(Path(data_dir) / "N1/inputs.jsonl")
    prompt = rows[0]["prompt_text"]
    options = {"A": "A", "B": "B", "C": "C"}
    layer = default_layer(model)
    dim = hidden_dim(model)
    vector = torch.ones(dim, dtype=torch.float32, device=target)
    vector = vector / vector.norm()
    scorer = OptionScorer(model, tokenizer, target)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.time()
    baseline, base_info = scorer.score(prompt, options)
    zero, zero_info = scorer.score(prompt, options, Intervention(layer=layer, vector=vector, scale=0.0))
    target_edit, target_info = scorer.score(prompt, options, Intervention(layer=layer, vector=vector, scale=0.5))
    non_target_position = max(0, int(base_info["options"]["A"]["prompt_tokens"]) - 2)
    non_target, non_target_info = scorer.score(prompt, options, Intervention(layer=layer, vector=vector, scale=0.5, positions=(non_target_position,)))
    zero_delta = max(abs(baseline[key] - zero[key]) for key in baseline)
    target_delta = max(abs(baseline[key] - target_edit[key]) for key in baseline)
    layers = _layers(model)
    config = model.config
    result = {"schema_version": "care.m2.gpu_smoke.v1", "status": "gpu_smoke_passed" if zero_delta < 1e-5 and target_delta > 1e-6 else "blocked",
              "model": fingerprint, "device": str(target), "layer": layer, "hidden_dim": dim,
              "gqa": {"num_attention_heads": int(getattr(config, "num_attention_heads", -1)), "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
                      "divisible": int(getattr(config, "num_attention_heads", 0)) % max(1, int(getattr(config, "num_key_value_heads", 1))) == 0},
              "rope": {"rope_theta": getattr(config, "rope_theta", None), "rope_scaling": getattr(config, "rope_scaling", None)},
              "dtype": str(next(model.parameters()).dtype), "baseline": baseline, "zero_hook": zero,
              "target_edit": target_edit, "non_target_edit": non_target,
              "zero_max_abs_delta": zero_delta, "target_max_abs_delta": target_delta,
              "target_hook": target_info["options"]["A"]["hook"], "non_target_hook": non_target_info["options"]["A"]["hook"],
              "prompt_tokens": base_info["options"]["A"]["prompt_tokens"], "forward_count": scorer.forward_count,
              "peak_memory_bytes": int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0,
              "elapsed_seconds": round(time.time() - started, 3),
              "checks": {"zero_identity": zero_delta < 1e-5, "nonzero_target_effect": target_delta > 1e-6,
                          "target_hook_changed": float(target_info["options"]["A"]["hook"]["changed_l1"]) > 0,
                          "gqa_contract": bool(result_gqa := (int(getattr(config, "num_attention_heads", 0)) % max(1, int(getattr(config, "num_key_value_heads", 1))) == 0)),
                          "bf16": next(model.parameters()).dtype == torch.bfloat16},
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out, result)
    return result
