#!/usr/bin/env python3
"""Train-only subspaces and matched single-location interventions for CARE M2.

No GPU execution is performed on import. B2 calls the pinned official builder.
All tensors use QK-normalized pre-RoPE coordinates.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

SEKA_COMMIT = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
LOCATIONS = [(2, 0), (20, 5), (9, 0), (35, 0)]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

def paired_rows(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    out = []
    for pid, values in sorted(pairs.items()):
        assert len(values) == 2, (pid, len(values))
        assert {r["applicable"] for r in values} == {True, False}, pid
        a = next(r for r in values if r["applicable"])
        b = next(r for r in values if not r["applicable"])
        assert a["group_id"] == b["group_id"]
        assert a["prompt"][slice(*a["spans"]["action"])] == b["prompt"][slice(*b["spans"]["action"])]
        assert a["option_map"] == b["option_map"]
        out.append((a, b))
    return out

def fit_basis(x, rank, center=False):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all()
    if center:
        x = x - x.mean(0)
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    available = int(np.sum(s > max(s[0] * 1e-8, 1e-10))) if len(s) else 0
    if rank > available:
        raise ValueError(f"rank {rank} exceeds empirical rank {available}")
    return vh[:rank].T

def contrastive_basis(signal, background, rank, ridge=.1):
    """Generalized signal/background second moment; QR in original key coordinates."""
    a = np.asarray(signal, dtype=np.float64)
    b = np.asarray(background, dtype=np.float64)
    assert len(a) and len(b)
    ca = a.T @ a / len(a)
    cb = b.T @ b / len(b)
    reg = max(np.trace(cb) / cb.shape[0] * ridge, 1e-10)
    w, v = np.linalg.eigh(cb + reg * np.eye(cb.shape[0]))
    whitening = (v * (1 / np.sqrt(w))) @ v.T
    e, u = np.linalg.eigh(whitening @ ca @ whitening)
    available = int(np.sum(e > max(e[-1] * 1e-8, 1e-10)))
    if rank > available:
        raise ValueError(f"B4 rank {rank} exceeds signal rank {available}")
    directions = whitening @ u[:, np.argsort(e)[::-1][:rank]]
    q, _ = np.linalg.qr(directions)
    return q[:, :rank]

def training_matrices(rows, keys, location, tolerance=.15, control_rows=None):
    """Only same-rule, same-label numeric controls matched to flip magnitude."""
    L, h = location
    k = lambda r: np.asarray(keys[r["case_id"]][L, h], dtype=np.float64)
    raw = np.stack([k(r).mean(0) for r in rows])
    flips, controls, audit = [], [], []
    pairs = paired_rows(rows)
    pool = rows if control_rows is None else control_rows
    for a, b in pairs:
        assert k(a).shape == k(b).shape
        flips.append(k(a).mean(0) - k(b).mean(0))
        distance = abs(float(a["value"]) - float(b["value"]))
        candidates = []
        for i, x in enumerate(pool):
            if (x["group_id"], x["rule_id"]) != (a["group_id"], a["rule_id"]):
                continue
            for y in pool[i + 1:]:
                if (x["group_id"], x["rule_id"], x["gold_label"]) != (y["group_id"], y["rule_id"], y["gold_label"]):
                    continue
                if x["applicable"] != y["applicable"]:
                    continue
                delta = abs(float(x["value"]) - float(y["value"]))
                if delta and distance and abs(delta / distance - 1) <= tolerance:
                    candidates.append((abs(delta / distance - 1), x["case_id"], y["case_id"], x, y))
        if candidates:
            _, _, _, x, y = min(candidates, key=lambda z: z[:3])
            # Both vectors compare the exact same action tokens.
            assert k(x).shape == k(y).shape
            controls.append(k(x).mean(0) - k(y).mean(0))
            audit.append(dict(pair_id=a["pair_id"], control=[x["case_id"], y["case_id"]],
                              flip_distance=distance, control_distance=abs(float(x["value"]) - float(y["value"]))))
    return raw, np.stack(flips), np.stack(controls) if controls else None, audit

def normalized_delta(k, operator, epsilon):
    """Equal Frobenius key disturbance at the selected head/span per example."""
    u = k.float() @ operator.to(k.device, torch.float32)
    un = torch.linalg.vector_norm(u)
    if float(un) <= 1e-10 or epsilon == 0:
        return torch.zeros_like(k, dtype=torch.float32)
    return epsilon * torch.linalg.vector_norm(k.float()) / un * u

@contextlib.contextmanager
def key_edit(model, idx, location, operator, epsilon, mode="amplify", donor=None):
    L, h = location
    report = {}
    def hook(module, inputs, out):
        k = out[0, idx, h].float()
        if mode == "patch":
            assert donor is not None and tuple(donor.shape) == tuple(k.shape)
            delta = (donor.to(k.device).float() - k) @ operator.to(k.device).float()
        else:
            delta = normalized_delta(k, operator, epsilon)
            if mode == "ablate":
                delta = -(k @ operator.to(k.device).float())
        changed = out.clone()
        changed[0, idx, h] = (k + delta).to(out.dtype)
        realized = changed[0, idx, h].float() - k
        report.update(requested_norm=float(delta.norm()), realized_norm=float(realized.norm()),
                      relative_norm=float(realized.norm() / k.norm().clamp_min(1e-10)))
        return changed
    handle = model.model.layers[L].self_attn.k_norm.register_forward_hook(hook)
    try:
        yield report
    finally:
        handle.remove()

def logit_delta(model, capture, idx, location, operator, epsilon):
    """Fixed-layer QK algebra after exactly one RoPE; output [Q,T,T]."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    L, h = location
    q, k = capture["q"].to(model.device), capture["k"].to(model.device)
    n, nq, d = q.shape
    delta = normalized_delta(k[idx, h], operator, epsilon)
    positions = torch.arange(n, device=model.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(
        torch.zeros((1, n, model.config.hidden_size), device=model.device, dtype=q.dtype), positions)
    cos, sin = cos[0], sin[0]
    # Mirror the real BF16 cast and rotation, including key-addition rounding.
    before = k[idx, h].to(q.dtype)
    after = (k[idx, h].float() + delta).to(q.dtype)
    rotate = lambda x, c, s: x * c + rotate_half(x) * s
    dq = rotate(after, cos[idx], sin[idx]).float() - rotate(before, cos[idx], sin[idx]).float()
    qr = rotate(q, cos[:, None], sin[:, None]).float()
    repeat = nq // model.config.num_key_value_heads
    assert repeat == 4
    heads = list(range(h * repeat, (h + 1) * repeat))
    scores = torch.einsum("tqd,sd->qts", qr[:, heads], dq) / math.sqrt(d)
    delta_mask = torch.zeros((nq, n, n), device=model.device, dtype=torch.float32)
    for j, t in enumerate(idx):
        delta_mask[heads, t:, t] = scores[:, t:, j]
    return delta_mask

@contextlib.contextmanager
def attention_bias(model, location, delta_mask, uniform=False):
    """Same causal entries and query heads; positive uniform bias matched by RMS."""
    L, h = location
    m = delta_mask
    if uniform:
        repeat = model.config.num_attention_heads // model.config.num_key_value_heads
        # Support must include valid zero-valued QK entries, so derive target columns explicitly.
        support = getattr(attention_bias, "_support", None)
        assert support is not None and support.shape == m.shape
        rms = m[support].square().mean().sqrt()
        m = support.to(m.dtype) * rms
    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        assert mask is not None and mask.ndim == 4
        mask = mask.expand(1, model.config.num_attention_heads, -1, -1).float().clone()
        kwargs["attention_mask"] = (mask + m.unsqueeze(0)).to(model.dtype)
        return args, kwargs
    handle = model.model.layers[L].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()

def bias_support(n, idx, location, device):
    support = torch.zeros((32, n, n), device=device, dtype=torch.bool)
    _, h = location
    for t in idx:
        support[4*h:4*h+4, t:, t] = True
    return support

def collect(model, tok, row, locations, capture_q=False):
    from run_experiments import tokenize_span, causal_inputs
    enc, idx, _ = tokenize_span(tok, row["prompt"], row["spans"]["action"])
    assert enc["input_ids"].shape[1] <= 1024
    keys, full, handles = {}, {}, []
    for L, h in locations:
        def kh(module, args, out, L=L, h=h):
            keys[L, h] = out[0, idx, h].float().cpu().numpy().copy()
            if capture_q:
                full.setdefault(L, {})["k"] = out[0].detach().clone()
        handles.append(model.model.layers[L].self_attn.k_norm.register_forward_hook(kh))
        if capture_q:
            def qh(module, args, out, L=L):
                full.setdefault(L, {})["q"] = out[0].detach().clone()
            handles.append(model.model.layers[L].self_attn.q_norm.register_forward_hook(qh))
    try:
        logits = model(**causal_inputs(enc, model.device)).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    ids = []
    for letter in "ABCD":
        value = tok.encode(" " + letter, add_special_tokens=False)
        assert len(value) == 1
        ids.append(value[0])
    probabilities = logits[ids].softmax(-1).cpu().tolist()
    return keys, full, idx, probabilities

def official_projectors(model, tok, rows, layers, output, clinical):
    """Only prompt assembly/extraction adapts. SVD and pos/neg definition are official."""
    from src.model.projection_builder_base import ProjectionBuilderBase
    from run_experiments import tokenize_span, causal_inputs
    class Builder(ProjectionBuilderBase):
        def iter_examples(self):
            return iter(rows)
        def get_triplets(self, row):
            ctx = row["patient"] + ("\nClinical guidance: " + row["guidance"] if clinical else "\nGuideline: " + row["modified_rule"])
            action = row["target_action"] if clinical else "select Route Cedar"
            return [(ctx, row["query"], action, "What type of patient information is being presented?")]
    builder = Builder.__new__(Builder)
    builder.layers = layers
    builder.device = "cpu"
    builder.model_path = "Qwen3-4B-Base"
    builder.top_pct = .9
    builder.min_diff = 0.
    builder.feature = None
    buffers = [[[[ ] for _ in range(8)] for _ in layers] for _ in range(3)]
    audit = []
    for row in rows:
        ctx, rel, action, irr = builder.get_triplets(row)[0]
        token_ids = []
        for c, text in enumerate([f"Context: {ctx} ", f"Question: {rel}\nContext: {ctx}", f"Question: {irr}\nContext: {ctx}"]):
            start = text.index(action)
            enc, idx, _ = tokenize_span(tok, text, [start, start + len(action)])
            token_ids.append(enc["input_ids"][0, idx].tolist())
            captured, handles = {}, []
            for L in layers:
                def kh(module, args, out, L=L):
                    captured[L] = out[0, idx].float().cpu().clone()
                handles.append(model.model.layers[L].self_attn.k_norm.register_forward_hook(kh))
            try:
                model(**causal_inputs(enc, model.device))
            finally:
                for handle in handles:
                    handle.remove()
            for j, L in enumerate(layers):
                for h in range(8):
                    buffers[c][j][h].append(captured[L][:, h])
            audit.append(dict(case_id=row["case_id"], condition=c, token_ids=token_ids[-1]))
        assert token_ids[0] == token_ids[1] == token_ids[2]
    output.mkdir(parents=True, exist_ok=True)
    builder._compute_and_save_traditional(*buffers, len(layers), 8, str(output))
    torch.save(dict(H=buffers[0], H_relevant=buffers[1], H_irrelevant=buffers[2]), output / "builder_keys.pt")
    save_json(output / "training_span_audit.json", audit)
    pos = torch.load(output / "Qwen3-4B-Base_pos_proj.pt", weights_only=True)["proj"]
    neg = torch.load(output / "Qwen3-4B-Base_neg_proj.pt", weights_only=True)["proj"]
    return {(L, h): (pos[j, h] + .25 * neg[j, h]) / 2 for j, L in enumerate(layers) for h in range(8)}

def o2_cache(artifacts, rows, locations):
    """Reuse token tensors only after exact archive and data-hash verification."""
    manifest = json.loads((artifacts / "results/manifest.json").read_text())
    path = artifacts / "results/obs2/key_embeddings.npz"
    assert sha256(path) == manifest["artifact_hashes"]["results/obs2/key_embeddings.npz"]
    data_path = artifacts / "data/derived/pilot.jsonl"
    assert sha256(data_path) == manifest["artifact_hashes"]["data/derived/pilot.jsonl"]
    data = np.load(path, allow_pickle=False)
    assert data["token_keys"].ndim == 6
    assert list(data["states"]) == ["RA", "RN", "IA", "IN"]
    gids = [str(g) for g in data["group_ids"]]
    states = ["RA", "RN", "IA", "IN", "RA2", "RN2"]
    lookup = {r["case_id"]: r for r in rows}
    cache = {}
    for i, g in enumerate(gids):
        for j, state in enumerate(states):
            case_id = f"{g}-{state}"
            if case_id in lookup:
                assert str(data["split"][i]) == lookup[case_id]["split"]
                cache[case_id] = {(L, h): data["token_keys"][i, j, L, :, h, :].copy() for L, h in locations}
    assert set(lookup) <= cache.keys()
    return cache
