"""Token-level measurements for the 2026-10-28 applicability mechanism study.

The paired covariance is computed *before* averaging tokens. Offline and online
orders are different, explicit protocols; this module does not apply edits.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np

from .extraction import encode_prompt, rotate_half, validate_qwen3

VARIANTS = ("positive", "negative", "neutral")
ROLES = ("id", "action")


def online_prompt(row, variant):
    """Move the exact guideline block before patient and question, remapping spans.

    The full prompt is tokenized once. A prefix boundary is subsequently selected
    from those token offsets, never by concatenating independently tokenized text.
    """
    old = row["prompts"][variant]
    old_spans = row["spans"][variant]
    gs = old.index("Guideline (all conditions in a rule are required):\n")
    # The guide annotation includes its final newline; the next newline belongs
    # to the answer cue. Keep that annotated guide byte-for-byte intact.
    answer = old.rindex("\nAnswer:")
    guide = old[gs:answer]
    pre = old[:gs]
    prompt = guide + "\n\n" + pre + old[answer:]
    suffix_shift = len(guide) + 2
    def move(span):
        a, b = span
        if gs <= a < b <= answer:
            return [a-gs, b-gs]
        if 0 <= a < b <= gs:
            return [a+suffix_shift, b+suffix_shift]
        raise ValueError("A recorded span crosses the order boundary")
    spans = {}
    for key, value in old_spans.items():
        if key == "rules":
            spans[key] = {r: {role: [move(s) for s in ss] for role, ss in roles.items()}
                          for r, roles in value.items()}
        else:
            spans[key] = [move(s) for s in value]
    for r, roles in old_spans["rules"].items():
        for role, ss in roles.items():
            for before, after in zip(ss, spans["rules"][r][role]):
                assert old[slice(*before)] == prompt[slice(*after)]
    return prompt, spans, len(guide)


def tokenize_online(tokenizer, row, variant, max_length=2048):
    prompt, spans, guide_end = online_prompt(row, variant)
    result = encode_prompt(tokenizer, prompt, spans, ROLES, max_length)
    # The separator is patient independent and belongs to the cached prefix.
    # Qwen can merge final action punctuation with these trailing newlines; use
    # the start of 'Patient', so this complete editable token stays in the cache.
    prefix_char_end = guide_end + 2
    prefix_length = 0
    for a, b in result["offsets"]:
        if b <= prefix_char_end:
            prefix_length += 1
        else:
            break
    if prefix_length < 1 or any(max(p) >= prefix_length
                                for p in result["rule_positions"].values()):
        raise ValueError("The complete rule targets are not inside the guide prefix")
    result.update(prompt=prompt, spans=spans, prefix_length=prefix_length,
                  guide_end=guide_end)
    return result


def capture_tokens(model, tokenized, candidate_ids, *, directions=None):
    """One real full forward captures both rules, all keys and last queries.

    Retain only rule tokens and last-query measurements, not full attention
    matrices. ``directions`` are optional [layer, KV head, dim] arrays used to
    compute actual RoPE/GQA score RMS and per-query responses for fixed directions.
    """
    import torch
    geometry = validate_qwen3(model)
    union = sorted(set().union(*map(set, tokenized["rule_positions"].values())))
    index = {p: i for i, p in enumerate(union)}
    captured, handles = {}, []
    for li, layer in enumerate(model.model.layers):
        state = captured[li] = {}
        def rotary(module, args, kwargs, s=state):
            c, sn = kwargs["position_embeddings"]
            s["cos"], s["sin"] = c[0].detach(), sn[0].detach()
        def keys(module, args, output, s=state):
            s["keys"] = output[0, union].detach()
        def queries(module, args, output, s=state):
            s["query"] = output[0, -1].detach()
        def attention(module, args, output, s=state):
            weights = output[1]
            if weights is None:
                raise ValueError("Use eager attention for actual attention measurements")
            s["attention"] = weights[0, :, -1, union].detach()
        handles.extend([
            layer.self_attn.register_forward_pre_hook(rotary, with_kwargs=True),
            layer.self_attn.k_norm.register_forward_hook(keys),
            layer.self_attn.q_norm.register_forward_hook(queries),
            layer.self_attn.register_forward_hook(attention)])
    device = model.get_input_embeddings().weight.device
    try:
        with torch.inference_mode():
            out = model(**{k: v.to(device) for k, v in tokenized["encoded"].items()},
                        use_cache=False, logits_to_keep=1)
            logits = out.logits[0, -1, candidate_ids].float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    results = {}
    for rule, positions in tokenized["rule_positions"].items():
        ids = [index[p] for p in positions]
        keys, effective, mass = [], [], []
        rms = {name: [] for name in (directions or {})}
        per_query = {name: [] for name in (directions or {})}
        token_positive = {name: [] for name in (directions or {})}
        for li in range(geometry["layers"]):
            s = captured[li]
            key = s["keys"][ids]
            cos, sin = s["cos"][positions].float(), s["sin"][positions].float()
            q = s["query"]
            qp = (q * s["cos"][-1] + rotate_half(q) * s["sin"][-1]).float()
            qg = qp.reshape(geometry["kv_heads"], geometry["queries_per_kv_head"], -1)
            qm = qg.mean(1)
            eq = qm[None] * cos[:, None] - rotate_half(qm)[None] * sin[:, None]
            keys.append(key.float().cpu().numpy())
            effective.append(eq.mean(0).cpu().numpy())
            am = s["attention"][:, ids].float().sum(-1).reshape(
                geometry["kv_heads"], geometry["queries_per_kv_head"])
            mass.append(am.mean(-1).cpu().numpy())
            for name, direction in (directions or {}).items():
                delta = torch.as_tensor(direction[li], dtype=torch.float32, device=device)
                rotated = delta[None]*cos[:, None] + rotate_half(delta)[None]*sin[:, None]
                score = torch.einsum("hqd,thd->hqt", qg, rotated) / geometry["head_dim"]**.5
                rms[name].append(score.square().mean((1, 2)).sqrt().cpu().numpy())
                per_query[name].append(score.mean(-1).cpu().numpy())
                token_positive[name].append((score > 0).float().mean((1, 2)).cpu().numpy())
        results[rule] = {"token_keys": np.stack(keys),
                         "effective_query": np.stack(effective),
                         "attention_mass": np.stack(mass),
                         "unit_rms": {n: np.stack(v) for n, v in rms.items()},
                         "per_query": {n: np.stack(v) for n, v in per_query.items()},
                         "token_positive": {n: np.stack(v) for n, v in token_positive.items()}}
    return {"rules": results, "candidate_logits": logits, "geometry": geometry}


def token_pair_second_moment(positive, negative):
    """Return mean_j d_j d_j^T; never outer(mean_j d_j)."""
    a, b = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 4 or a.shape[1] == 0:
        raise ValueError("Expected aligned [layer, token, KV head, dimension] keys")
    d = (a-b).transpose(0, 2, 3, 1)
    return np.matmul(d, d.swapaxes(-1, -2)) / a.shape[1]


def row_metadata(row):
    return {key: deepcopy(row[key]) for key in
            ("id", "tree_id", "group_id", "pair_cluster_id", "split", "source_split",
             "target_rule_id", "controls", "decisive_node_index", "oracle")}


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()
