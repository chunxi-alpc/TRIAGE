"""Minimal Qwen3 intervention, using only selected query heads.

SEKA source: third_party/SEKA/src/model/seka_llm.py and
projection_builder_base.py. We use its k -> k + gain * k P spectral edit,
with calibration k_norm activations. Because Qwen3 uses GQA, keys are
expanded before selected-query-head edits; original KV projections are not
modified. This small adapter follows the installed Qwen3 forward tensor steps.
The prior basis is an attention-output PCA proxy, not an identified causal
prior-only subspace. It is used only with explicit valid conflict evidence.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from types import MethodType
import json
import hashlib
import torch
import torch.nn.functional as F


def _heads(components):
    if isinstance(components, (str, Path)):
        components = json.loads(Path(components).read_text())
    rows = components.get("selected_heads", [])
    return [(int(r["layer"]), int(r["head"])) for r in rows], components



def select_intervention_heads(components, projections, heads):
    """Return isolated component/projection views without changing cached artifacts.

    Use the existing bases for one or more already selected heads. This performs
    no localization, recalibration, or training and never mutates either input.
    """
    available, components = _heads(components)
    if isinstance(projections, (str, Path)):
        projections = torch.load(projections, map_location="cpu", weights_only=True)
    requested = [(int(layer), int(head)) for layer, head in heads]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Provide nonempty, unique selected heads")
    if not set(requested).issubset(available):
        raise ValueError("Subset contains a head not selected in existing components")
    subset_components = dict(components)
    subset_components["selected_heads"] = [dict(row) for row in components["selected_heads"]
        if (int(row["layer"]), int(row["head"])) in set(requested)]
    subset_projections = dict(projections)
    subset_projections["components"] = {
        f"{layer}:{head}": projections["components"][f"{layer}:{head}"]
        for layer, head in requested}
    parent_fingerprint = projections.get("fingerprint")
    subset_projections["parent_fingerprint"] = parent_fingerprint
    subset_projections["fingerprint"] = hashlib.sha256(json.dumps(
        {"parent_fingerprint": parent_fingerprint, "selected_heads": sorted(requested)},
        sort_keys=True).encode()).hexdigest()
    return subset_components, subset_projections


def _basis(samples, dimension, rank):
    if not samples:
        raise ValueError("No activations collected for projection")
    x = torch.cat(samples).float()
    # Compact uncentered SVD gives a spectral projector, not parameter training.
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    kept = min(rank, dimension, int((singular > 1e-6).sum()))
    return vh[:kept].T.contiguous(), singular[:kept].cpu()


@torch.no_grad()
def build_projections(model, tokenizer, calibration_records, components,
                      output_path, rank=4, max_length=2048, max_records=4):
    """Build low-rank key/output bases from current model and non-test prompts.

    Records accept text/prompt, source and split fields. The caller must supply
    training/calibration data; an explicit test split is rejected.
    """
    heads, artifact = _heads(components)
    if not heads:
        raise ValueError("Module2 selected_heads is empty: cannot build selected-head projections")
    cfg = model.config
    dim = cfg.head_dim
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    keys, outputs = defaultdict(list), defaultdict(list)
    handles, sources = [], []
    capture_indices = None
    by_layer = defaultdict(list)
    for layer, head in heads:
        if not 0 <= layer < cfg.num_hidden_layers or not 0 <= head < cfg.num_attention_heads:
            raise ValueError("Selected component is outside the model configuration")
        by_layer[layer].append(head)
    for layer, layer_heads in by_layer.items():
        attn = model.model.layers[layer].self_attn
        def capture_key(module, args, value, layer=layer, layer_heads=layer_heads):
            for head in layer_heads:
                # Qwen3 k_norm: [batch, seq, kv_heads, head_dim].
                key = value[..., head // group, :].reshape(-1, dim)
                indices = capture_indices if capture_indices is not None else torch.linspace(0, key.shape[0]-1, min(128,key.shape[0])).long().tolist()
                keys[f"{layer}:{head}"].append(key[indices].detach().float().cpu())
        def capture_output(module, args, layer=layer, layer_heads=layer_heads):
            value = args[0].reshape(*args[0].shape[:-1], cfg.num_attention_heads, dim)
            for head in layer_heads:
                output = value[..., head, :].reshape(-1, dim)
                indices = capture_indices if capture_indices is not None else torch.linspace(0, output.shape[0]-1, min(128,output.shape[0])).long().tolist()
                outputs[f"{layer}:{head}"].append(output[indices].detach().float().cpu())
        handles += [attn.k_norm.register_forward_hook(capture_key), attn.o_proj.register_forward_pre_hook(capture_output)]
    try:
        for index, record in enumerate(calibration_records):
            if index >= max_records:
                break
            if isinstance(record, str):
                record = {"text": record, "split": "caller_supplied_calibration"}
            split = str(record.get("split", "caller_supplied_calibration"))
            if "test" in split.lower():
                raise ValueError("Projection construction rejects test data")
            text = record.get("text", record.get("prompt"))
            if not text:
                raise ValueError("Calibration record requires text or prompt")
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            requested = sorted({int(i) for i in record.get("token_indices", []) if 0 <= int(i) < encoded["input_ids"].shape[-1]})
            capture_indices = [requested[i] for i in torch.linspace(0,len(requested)-1,min(128,len(requested))).long().tolist()] if requested else None
            model(**encoded, use_cache=False)
            sources.append({"source": record.get("source", "existing_calibration"), "split": split,
                            "sample_id": record.get("sample_id", index), "tokens": encoded["input_ids"].shape[-1],
                            "truncated":len(tokenizer.encode(text)) > max_length, "max_length":max_length,
                            "activation_tokens":len(capture_indices) if capture_indices is not None else min(128,encoded["input_ids"].shape[-1])})
    finally:
        for handle in handles:
            handle.remove()
    result = {"schema_version": "module3.projections.v1", "model_config": {
        "model_type": cfg.model_type, "layers": cfg.num_hidden_layers,
        "query_heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
        "head_dim": dim, "name_or_path": str(cfg._name_or_path)}, "components": {},
        "calibration_sources": sources, "rank": rank,
        "method": "uncentered SVD of k_norm and selected query-head attention output; no training",
        "prior_basis_semantics": "attention-output PCA proxy; not identified prior-only direction"}
    for layer, head in heads:
        key = f"{layer}:{head}"
        kb, ks = _basis(keys[key], dim, rank)
        ob, os = _basis(outputs[key], dim, rank)
        result["components"][key] = {"layer": layer, "head": head, "kv_head": head // group,
                                     "key_basis": kb, "prior_basis": ob,
                                     "key_singular_values": ks, "output_singular_values": os}
    fingerprint_input = {"model_config": result["model_config"], "heads": heads, "sources": sources, "rank": rank}
    result["fingerprint"] = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    metadata = {k: v for k, v in result.items() if k != "components"}
    metadata["components"] = [{"layer": l, "head": h, "kv_head": h // group,
                               "key_rank": result["components"][f"{l}:{h}"]["key_basis"].shape[1],
                               "prior_rank": result["components"][f"{l}:{h}"]["prior_basis"].shape[1]} for l, h in heads]
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return result


class GatedAttentionIntervention:
    """Context manager; calls within it perform gated model computation.

    Context: selected-head key edits at mapped condition/evidence/action tokens.
    Prior competition: selected-head output spectral subtraction only for
    applicable rules with explicit prior=oppose, guided=support disagreement.
    Inapplicable/unknown rules and zero action gates do not trigger either path.
    """
    def __init__(self, model, components, projection_path, module1_output,
                 token_mapping, context_strength=.1, prior_strength=.1):
        self.model = model
        self.heads, self.components = _heads(components)
        self.projections = (torch.load(projection_path, map_location="cpu", weights_only=True)
                            if isinstance(projection_path, (str, Path)) else projection_path)
        self.module1 = module1_output
        self.mapping = token_mapping
        self.context_strength, self.prior_strength = float(context_strength), float(prior_strength)
        if not 0 <= self.context_strength <= 2 or not 0 <= self.prior_strength <= 1:
            raise ValueError("Context gain must be [0,2], prior suppression [0,1]")
        self.handles, self.original_forwards = [], {}
        self.stats = {"schema_version": "module3.stats.v1", "context_changed_elements": 0,
                      "context_delta_l2": 0., "prior_changed_elements": 0., "prior_delta_l2": 0.,
                      "context_forward_calls": 0, "prior_forward_calls": 0,
                      "selected_heads": [{"layer": l, "head": h} for l, h in self.heads],
                      "gqa_edit_scope": "selected query heads only after repeat_kv; other heads untouched",
                      "rule_gates": {}, "prior_basis_is_proxy": True,
                      "context_strength":self.context_strength, "prior_strength":self.prior_strength,
                      "projection_fingerprint":self.projections.get("fingerprint"),
                      "context_operator":"token salience spectral key edit; no query-specific relation bias yet",
                      "prior_minimum_evidence_confidence":.8}
        self.token_weights, self.prior_gate = self._gates()
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("Duplicate selected heads")
        cfg = model.config
        metadata = self.projections.get("model_config", {})
        for name, current in [("query_heads", cfg.num_attention_heads), ("kv_heads", cfg.num_key_value_heads),
                              ("head_dim", cfg.head_dim), ("layers", cfg.num_hidden_layers)]:
            if metadata.get(name, current) != current:
                raise ValueError(f"Projection {name} mismatches loaded model")
        if metadata.get("name_or_path") and metadata["name_or_path"] != str(cfg._name_or_path):
            raise ValueError("Projection model path mismatches loaded model")
        projected_heads = {(int(r["layer"]), int(r["head"])) for r in self.projections["components"].values()}
        if projected_heads != set(self.heads):
            raise ValueError("Projection selected heads mismatch component artifact")
        for l, h in self.heads:
            if f"{l}:{h}" not in self.projections["components"]:
                raise ValueError(f"Missing projection for layer {l} head {h}")

    def _gates(self):
        apps = {r["rule_id"]: r for r in self.module1.get("applicability", [])}
        applicability = {rid: r.get("status") for rid, r in apps.items()}
        conflicts = {r["rule_id"]: r for r in self.module1.get("conflict_scores", [])}
        maps = {(r["rule_id"], r["condition_id"]): r for r in self.mapping.get("relations", [])}
        weights, prior = defaultdict(float), 0.
        actions = self.module1.get("action_gate", {})
        for rid, value in actions.items():
            gate = max(0., min(1., float(value))) if applicability.get(rid) == "applicable" else 0.
            conflict = conflicts.get(rid, {})
            decisive = set(apps.get(rid, {}).get("decisive_condition_ids", []))
            evidence_rows = [r for r in self.module1.get("condition_evidence_relations", [])
                             if r.get("rule_id") == rid and (not decisive or r.get("condition_id") in decisive)]
            confident = bool(evidence_rows) and all(r.get("status") in {"satisfied", "not_satisfied"}
                       and bool(r.get("evidence")) and float(r.get("confidence", 0)) >= .8 for r in evidence_rows)
            if decisive and {r.get("condition_id") for r in evidence_rows} != decisive:
                confident = False
            valid = (confident and conflict.get("status") == "disagreement" and conflict.get("prior_support") == "oppose"
                     and conflict.get("guided_support") == "support")
            suppression = gate * max(0., min(1., float(conflict.get("score", 0)))) if valid else 0.
            self.stats["rule_gates"][rid] = {"applicability": applicability.get(rid, "unknown"),
                                           "action_gate": gate, "valid_conflict": valid, "evidence_confident": confident,
                                           "prior_gate": suppression}
            prior = max(prior, suppression)
        for row in self.module1.get("relation_weights", []):
            rid, cid = row["rule_id"], row["condition_id"]
            gate = self.stats["rule_gates"].get(rid, {}).get("action_gate", 0.)
            if gate <= 0:
                continue
            relation = maps.get((rid, cid), {})
            weight = max(0., min(1., float(row.get("weight", row.get("binding_gate", 0)))))
            for token in relation.get("condition_tokens", []) + relation.get("evidence_tokens", []):
                weights[int(token)] = max(weights[int(token)], gate * weight)
            action_tokens = relation.get("action_tokens", [])
            if not action_tokens:
                action_tokens = self.mapping.get("rules", {}).get(rid, {}).get("action_tokens", [])
            for token in action_tokens:
                weights[int(token)] = max(weights[int(token)], gate)
        return {token: value for token, value in weights.items() if value > 0}, prior

    def _edit_keys(self, keys, layer, layer_heads, cache_position):
        # Input already expanded to query heads. Never mutate a shared KV view.
        if not self.token_weights or self.context_strength == 0:
            return keys
        seq = keys.shape[-2]
        if cache_position is None:
            positions = range(seq)
        else:
            positions = cache_position.flatten().tolist()
        gains = torch.tensor([self.token_weights.get(int(p), 0.) for p in positions],
                             device=keys.device, dtype=torch.float32)
        if gains.numel() != seq or not bool(gains.any()):
            return keys
        changed = keys.clone()
        for head in layer_heads:
            basis = self.projections["components"][f"{layer}:{head}"]["key_basis"].to(keys.device)
            source = keys[:, head].float()
            projected = (source @ basis) @ basis.T
            delta = self.context_strength * gains[None, :, None] * projected
            changed[:, head] = (source + delta).to(keys.dtype)
            actual = changed[:, head].float() - source
            self.stats["context_changed_elements"] += int(torch.count_nonzero(actual))
            self.stats["context_delta_l2"] += float(torch.linalg.vector_norm(actual))
        self.stats["context_forward_calls"] += 1
        return changed

    def _prior_hook(self, layer, layer_heads):
        def hook(module, args):
            if self.prior_gate <= 0 or self.prior_strength == 0:
                return None
            original = args[0]
            view = original.reshape(*original.shape[:-1], self.model.config.num_attention_heads,
                                    self.model.config.head_dim)
            changed = view.clone()
            for head in layer_heads:
                basis = self.projections["components"][f"{layer}:{head}"]["prior_basis"].to(original.device)
                source = view[..., head, :].float()
                delta = self.prior_strength * self.prior_gate * ((source @ basis) @ basis.T)
                changed[..., head, :] = (source - delta).to(original.dtype)
                actual = changed[..., head, :].float() - source
                self.stats["prior_changed_elements"] += int(torch.count_nonzero(actual))
                self.stats["prior_delta_l2"] += float(torch.linalg.vector_norm(actual))
            self.stats["prior_forward_calls"] += 1
            return (changed.reshape_as(original),) + tuple(args[1:])
        return hook

    def __enter__(self):
        try:
            return self._attach()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _attach(self):
        if self.handles or self.original_forwards:
            raise RuntimeError("Intervention context is already active")
        context_enabled = bool(self.token_weights) and self.context_strength > 0
        prior_enabled = self.prior_gate > 0 and self.prior_strength > 0
        # A disabled operator must preserve the original attention implementation.
        if not self.heads or not (context_enabled or prior_enabled):
            return self
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        by_layer = defaultdict(list)
        for layer, head in self.heads:
            by_layer[layer].append(head)
        for layer, layer_heads in by_layer.items():
            attn = self.model.model.layers[layer].self_attn
            if context_enabled:
                self.original_forwards[layer] = attn.forward
                owner = self
                def forward(attn_self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=None, cache_position=None, layer=layer,
                            layer_heads=layer_heads, **kwargs):
                    if kwargs.get("past_key_value") is not None:
                        past_key_values = kwargs.pop("past_key_value")
                    shape = (*hidden_states.shape[:-1], -1, attn_self.head_dim)
                    q = attn_self.q_norm(attn_self.q_proj(hidden_states).view(shape)).transpose(1, 2)
                    k = attn_self.k_norm(attn_self.k_proj(hidden_states).view(shape)).transpose(1, 2)
                    v = attn_self.v_proj(hidden_states).view(shape).transpose(1, 2)
                    k = repeat_kv(k, attn_self.num_key_value_groups)
                    v = repeat_kv(v, attn_self.num_key_value_groups)
                    k = owner._edit_keys(k, layer, layer_heads, cache_position)
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, attn_self.layer_idx,
                                                     {"sin": sin, "cos": cos, "cache_position": cache_position})
                    mask = attention_mask[..., :k.shape[-2]] if attention_mask is not None else None
                    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                 dropout_p=0., is_causal=mask is None and q.shape[-2] > 1,
                                 scale=attn_self.scaling)
                    result = result.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
                    return attn_self.o_proj(result), None
                attn.forward = MethodType(forward, attn)
            if prior_enabled:
                self.handles.append(attn.o_proj.register_forward_pre_hook(self._prior_hook(layer, layer_heads)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for layer, original in self.original_forwards.items():
            self.model.model.layers[layer].self_attn.forward = original
        self.original_forwards.clear()
        return False
