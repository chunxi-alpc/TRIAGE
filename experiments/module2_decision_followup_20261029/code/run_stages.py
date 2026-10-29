#!/usr/bin/env python3
"""Server-only staged follow-up. Prepared code; not yet executed or validated.

Run from the requested commit in an isolated worktree. Every stage writes into
a fresh directory and records failures. Never run this file on confirmation
until train/selection artifacts are frozen. See protocol.json and RUN.zh.txt.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def journal(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def digest_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prepare(a, p, out):
    """Download exact official revisions only; retain hashes and prior files."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from src.module2.minimal_pairs import build_minimal_pairs, load_development_trees
    old = a.repo / 'experiments/module2_applicability_20261027/results'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.repo,
                               text=True).strip() != p['base_commit']:
        raise ValueError('Use an isolated checkout of the specified base commit')
    source = Path(a.trees) if a.trees else Path(hf_hub_download(
        'nchennnn/MEGA-CDP', 'train_val/decision_trees.jsonl', repo_type='dataset',
        revision=p['source_revision']))
    if sha(source) != p['source_sha256']:
        raise ValueError('Source SHA differs from the original frozen source')
    prior = read(old / 'data/audit.json')
    exclude = set(prior['group_split_assignments'])
    if exclude != set(p['original_guidelines']):
        raise ValueError('Prior guideline manifest differs from protocol')
    source_rows = [r for r in load_development_trees(source)
                   if f"guideline:{r.get('guideline_id')}" not in exclude]
    data, audit = build_minimal_pairs(source_rows, seed=p['seed'],
        max_groups=p['new_guidelines'], max_depth=4, max_pairs_per_tree=1,
        max_trees_per_group=1, source_revision=p['source_revision'])
    actual = {s: len({r['group_id'] for r in data if r['split'] == s})
              for s in ('fit', 'select', 'confirm')}
    if actual != p['split']:
        raise ValueError(f'Cannot construct the planned split: {actual}')
    for split in actual:
        f = out / f'{split}.jsonl'
        f.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                             for r in data if r['split'] == split))
    model = Path(a.model) if a.model else Path(snapshot_download(
        p['model_id'], revision=p['model_revision'],
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json']))
    for name, expected in [('model.safetensors', p['weight_sha256']),
                           ('tokenizer.json', p['tokenizer_sha256'])]:
        if sha(model / name) != expected:
            raise ValueError(f'Model file hash mismatch: {name}')
    immutable = [old / 'data/triplets.jsonl', old / 'data/audit.json',
                 old / 'qwen3_0_6b_cpu/operators.npz',
                 old / 'qwen3_0_6b_cpu/selection.json',
                 old / 'qwen3_0_6b_cpu/causal/predictions.jsonl']
    dump(out / 'manifest.json', {'model_path': str(model.resolve()),
        'source_path': str(source.resolve()), 'source_sha256': sha(source),
        'model_files': {f.name: sha(f) for f in model.iterdir()
                        if f.is_file() and f.suffix in ('.json', '.safetensors')},
        'old_files': {str(f): sha(f) for f in immutable},
        'new_files': {s: sha(out / f'{s}.jsonl') for s in actual},
        'excluded_guidelines': sorted(exclude), 'audit': audit,
        'untouched_across_other_projects': 'not established',
        'confirmation_outcomes_read': False})


class Engine:
    def __init__(self, a, p):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.np, self.p = torch, np, p
        self.device = f'cuda:{a.gpu}'
        if a.gpu not in (0, 1) or not torch.cuda.is_available():
            raise RuntimeError('GPU 0 or 1 is required; no local/CPU fallback')
        torch.set_num_threads(4)
        torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prep = read(a.run / 'prepare/manifest.json')
        for name, expected in [('model.safetensors', p['weight_sha256']),
                               ('tokenizer.json', p['tokenizer_sha256'])]:
            if sha(Path(prep['model_path']) / name) != expected:
                raise ValueError(f'Changed model artifact: {name}')
        self.tokenizer = AutoTokenizer.from_pretrained(prep['model_path'], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(prep['model_path'],
            local_files_only=True, torch_dtype=torch.float32,
            attn_implementation='eager').to(self.device).eval()
        self.model.requires_grad_(False)
        labels = [self.tokenizer.encode(x, add_special_tokens=False) for x in ('A', 'B')]
        if any(len(x) != 1 for x in labels):
            raise ValueError('Original primary A/B scoring needs single tokens')
        self.labels = [x[0] for x in labels]
        self.environment = {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'gpu': a.gpu, 'dtype': 'float32', 'tf32': False}

    def state(self, sample):
        from src.module3.applicability_shift import prepare_prefix
        from src.module2.extraction import encode_prompt
        t = self.torch
        state = prepare_prefix(self.model, self.tokenizer, sample['prompt'],
                               sample['spans'], max_length=4096,
                               roles=('condition', 'id', 'action'))
        # Clone outside inference_mode so autograd may save these constants.
        state.legacy = tuple((k.detach().clone(), v.detach().clone())
                             for k, v in state.cache.to_legacy_cache())
        state.cos, state.sin = state.cos.clone(), state.sin.clone()
        state.role_positions = {}
        for role in ('condition', 'id', 'action'):
            enc = encode_prompt(self.tokenizer, sample['prompt'], sample['spans'],
                                (role,), max_length=4096)
            state.role_positions[role] = enc['rule_positions']
        state.queries, handles = {}, []
        for l, layer in enumerate(self.model.model.layers):
            def cap(module, args, output, l=l):
                state.queries[l] = output[0, -1].detach().clone()
            handles.append(layer.self_attn.q_norm.register_forward_hook(cap))
        try:
            with t.no_grad():
                state.baseline = self.forward(state).detach()
        finally:
            for h in handles:
                h.remove()
        state.gold = sample['gold']
        return state

    def forward(self, state, key_edits=(), biases=(), replacement=None):
        from transformers import DynamicCache
        t = self.torch
        legacy = [(k.clone(), v.clone()) for k, v in state.legacy]
        for l, h, positions, delta in key_edits:
            legacy[l][0][0, h, positions] = legacy[l][0][0, h, positions] + delta
        if replacement:
            l, h, positions, donor, component = replacement
            ci = 0 if component == 'key' else 1
            legacy[l][ci][0, h, positions] = donor.legacy[l][ci][0, h, positions]
        cache = DynamicCache.from_legacy_cache(tuple(legacy))
        grouped = collections.defaultdict(list)
        for l, h, positions, bias in biases:
            grouped[l].append((h, positions, bias))
        handles = []
        for l, entries in grouped.items():
            def hook(module, args, kwargs, entries=entries):
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                shape = (1, state.geometry['query_heads'], 1, state.prefix_length + 1)
                mask = kwargs.get('attention_mask')
                mask = hidden.new_zeros(shape) if mask is None else mask.expand(shape).clone()
                group = state.geometry['queries_per_kv_head']
                for h, pos, bias in entries:
                    support = t.zeros_like(mask)
                    support[0, h * group:(h + 1) * group, 0, pos] = 1
                    mask = mask + support * bias
                return args, dict(kwargs, attention_mask=mask)
            handles.append(self.model.model.layers[l].self_attn.register_forward_pre_hook(
                hook, with_kwargs=True))
        try:
            return self.model(input_ids=state.input_ids[:, -1:],
                attention_mask=state.attention_mask,
                position_ids=state.input_ids.new_tensor([[state.prefix_length]]),
                cache_position=state.input_ids.new_tensor([state.prefix_length]),
                past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1]
        finally:
            for h in handles:
                h.remove()

    def edits(self, state, heads, vectors, tau, mode, bias_only=False):
        """Calibrate on unchanged queries, separately for the two rule factors."""
        from src.module2.extraction import rotate_half
        t, g = self.torch, state.geometry['queries_per_kv_head']
        edits, biases, diagnostics = [], [], []
        for rule in ('A', 'B'):
            positive = rule == state.gold
            if (mode == 'enhance' and not positive) or (mode == 'suppress' and positive):
                continue
            sign = 1. if positive else -1.
            pos = state.role_positions['action'][rule]
            proposed, scores = [], []
            for i, (l, h) in enumerate(heads):
                if bias_only:
                    b = vectors[i]
                    scores.append(b.expand(g * len(pos)))
                    proposed.append(b)
                else:
                    d = vectors[i]
                    dpost = d[None] * state.cos[pos] + rotate_half(d)[None] * state.sin[pos]
                    q = state.queries[l][h*g:(h+1)*g]
                    qp = q * state.cos[-1] + rotate_half(q) * state.sin[-1]
                    scores.append((qp @ dpost.T / math.sqrt(d.shape[-1])).reshape(-1))
                    proposed.append(dpost)
            rms = t.cat(scores).square().mean().sqrt()
            if not bool(t.isfinite(rms)) or float(rms.detach()) <= 1e-12:
                raise ValueError('Direction has no readable energy; record failed cell')
            scale = sign * float(tau) / rms
            for (l, h), v in zip(heads, proposed):
                if bias_only:
                    biases.append((l, h, pos, v * scale))
                else:
                    edits.append((l, h, pos, v * scale))
            actual = []
            for (l, h), v in zip(heads, proposed):
                if bias_only:
                    actual.append((v * scale).expand(g * len(pos)))
                else:
                    original = state.legacy[l][0][0, h, pos]
                    delta = (original + v * scale) - original
                    q = state.queries[l][h*g:(h+1)*g]
                    qp = q * state.cos[-1] + rotate_half(q) * state.sin[-1]
                    actual.append((qp @ delta.T / math.sqrt(v.shape[-1])).reshape(-1))
            measured = float(t.cat(actual).square().mean().sqrt().detach())
            diagnostics.append({'rule': rule, 'target_rms': tau, 'actual_rms': measured})
            if tau and abs(measured / tau - 1) > .05:
                raise ValueError('Actual score RMS differs from target by over 5%')
        return edits, biases, diagnostics

    def outcome(self, logits, sample):
        z = logits[self.labels].detach().float().cpu().tolist()
        pred = 'AB'[int(z[1] > z[0])]
        gi = 'AB'.index(sample['gold'])
        return {'logits': z, 'margin': z[gi] - z[1-gi], 'pred': pred,
                'correct': int(pred == sample['gold'])}

    def sequence_score(self, prompt, continuation):
        t = self.torch
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        full = self.tokenizer.encode(prompt + continuation, add_special_tokens=False)
        if full[:len(prefix)] != prefix:
            return {'score': None, 'reason': 'continuation retokenizes prompt boundary'}
        ids = t.tensor([full], device=self.device)
        with t.no_grad():
            logits = self.model(input_ids=ids, use_cache=False).logits
            lp = logits[0, len(prefix)-1:-1].log_softmax(-1)
            target = ids[0, len(prefix):]
            score = lp.gather(1, target[:, None]).sum()
        return {'score': float(score), 'tokens': target.tolist()}


def samples(a, split):
    # Target A and target B rows share texts; keep one reciprocal pair only.
    data = [r for r in rows(a.run / f'prepare/{split}.jsonl') if r['target_rule_id'] == 'A']
    answer = []
    for r in data:
        for variant in ('positive', 'negative'):
            answer.append({'id': r['id'] + ':' + variant, 'pair': r['id'],
                'group': r['group_id'], 'variant': variant, 'split': split,
                'mapping': r['controls']['action_permutation'],
                'prompt': r['prompts'][variant], 'spans': r['spans'][variant],
                'gold': r['oracle'][variant]['selected_rule_id']})
    return answer


def public(sample):
    return {k: v for k, v in sample.items() if k not in ('prompt', 'spans')}


def audit(a, p, out, e):
    from src.module2.extraction import encode_prompt
    t = e.torch
    dump(out / 'tokens.json', {s: e.tokenizer.encode(s, add_special_tokens=False)
        for s in ('A', 'B', ' A', ' B', '\nA', '\nB', ' YES', ' NO', ' UNKNOWN')})
    for split in ('fit', 'select'):
        data = samples(a, split)
        for j, s in enumerate(data):
            state = e.state(s)
            with t.no_grad():
                direct = e.model(input_ids=state.input_ids, use_cache=False,
                                 logits_to_keep=1).logits[0, -1]
                zero = e.forward(state)
            err = float((direct[e.labels] - state.baseline[e.labels]).abs().max())
            if err > 1e-3 or not t.equal(zero, state.baseline):
                raise RuntimeError(f'Cache/full or zero mismatch: {err}')
            donor = data[j+1 if j % 2 == 0 else j-1]
            alignment = {}
            for role in ('condition', 'id', 'action'):
                aa = encode_prompt(e.tokenizer, s['prompt'], s['spans'], (role,), 4096)
                bb = encode_prompt(e.tokenizer, donor['prompt'], donor['spans'], (role,), 4096)
                ok = aa['length'] == bb['length'] and aa['rule_positions'] == bb['rule_positions']
                for rid, pos in aa['rule_positions'].items():
                    ok = ok and t.equal(aa['encoded']['input_ids'][0, pos],
                                       bb['encoded']['input_ids'][0, pos])
                alignment[role] = bool(ok)
            if not all(alignment.values()):
                raise RuntimeError('Counterfactual patch spans are not aligned')
            secondary = {c: e.sequence_score(s['prompt'], c) for c in (' A', ' B')}
            r = {**public(s), **e.outcome(state.baseline, s), 'alignment': alignment,
                 'cached_vs_full_error': err, 'secondary_scores': secondary,
                 'length': state.input_ids.shape[1]}
            if j < 8:
                with t.no_grad():
                    generated = e.model.generate(state.input_ids, max_new_tokens=8,
                        do_sample=False, pad_token_id=e.tokenizer.eos_token_id)
                r['greedy_completion'] = e.tokenizer.decode(generated[0, state.input_ids.shape[1]:])
            journal(out / 'predictions.jsonl', r)
    summarize_rows(rows(out / 'predictions.jsonl'), out / 'summary.json')


def localize(a, p, out, e):
    t, data = e.torch, samples(a, 'fit')
    groups = sorted({s['group'] for s in data}, key=lambda g: digest_text(str(p['seed']) + g))[:6]
    data = [s for s in data if s['group'] in groups]
    for i in range(0, len(data), 2):
        pair = data[i:i+2]
        states = [e.state(s) for s in pair]
        for di in (0, 1):
            donor, recipient = states[di], states[1-di]
            s, ds = pair[1-di], pair[di]
            donor_label = 'AB'.index(ds['gold'])
            def margin(z):
                return float(z[e.labels[donor_label]] - z[e.labels[1-donor_label]])
            baseline = margin(recipient.baseline)
            gap = margin(donor.baseline) - baseline
            for role in p['localization']['roles']:
                pos = sorted(set(sum(recipient.role_positions[role].values(), [])))
                donor_pos = sorted(set(sum(donor.role_positions[role].values(), [])))
                if pos != donor_pos or not t.equal(recipient.input_ids[0, pos], donor.input_ids[0, pos]):
                    raise ValueError('Unaligned counterfactual tokens')
                for l in p['localization']['layers']:
                    for component in ('key', 'value'):
                        for h in p['localization']['kv_heads']:
                            with t.no_grad():
                                z = e.forward(recipient, replacement=(l, h, pos, donor, component))
                            delta = margin(z) - baseline
                            journal(out / 'patches.jsonl', {**public(s), 'role': role, 'layer': l,
                                'head': h, 'component': component, 'donor_gold': ds['gold'],
                                'donor_baseline_gap': gap, 'margin_change': delta,
                                'normalized_recovery': delta / gap if abs(gap) > 1e-4 else None})
                    # Residual patch: full forward, distinct from cached K/V patch.
                    captured = []
                    def capture(module, args, output):
                        x = output[0] if isinstance(output, tuple) else output
                        captured.append(x[:, pos].detach().clone())
                    handle = e.model.model.layers[l].register_forward_hook(capture)
                    try:
                        with t.no_grad():
                            e.model(input_ids=donor.input_ids, use_cache=False, logits_to_keep=1)
                    finally:
                        handle.remove()
                    def replace(module, args, output):
                        x = (output[0] if isinstance(output, tuple) else output).clone()
                        x[:, pos] = captured[0]
                        return (x, *output[1:]) if isinstance(output, tuple) else x
                    handle = e.model.model.layers[l].register_forward_hook(replace)
                    try:
                        with t.no_grad():
                            z = e.model(input_ids=recipient.input_ids, use_cache=False,
                                        logits_to_keep=1).logits[0, -1]
                    finally:
                        handle.remove()
                    delta = margin(z) - baseline
                    journal(out / 'patches.jsonl', {**public(s), 'role': role, 'layer': l,
                        'head': None, 'component': 'residual', 'donor_gold': ds['gold'],
                        'donor_baseline_gap': gap, 'margin_change': delta,
                        'normalized_recovery': delta / gap if abs(gap) > 1e-4 else None})
    records = rows(out / 'patches.jsonl')
    aggregate = collections.defaultdict(list)
    for r in records:
        if r['component'] == 'key' and r['role'] == 'action':
            aggregate[(r['layer'], r['head'])].append(r['margin_change'])
    ranking = sorted(aggregate, key=lambda x: (-sum(aggregate[x]) / len(aggregate[x]), x))
    frozen = e.np.load(a.repo / 'experiments/module2_applicability_20261027/results/qwen3_0_6b_cpu/operators.npz', allow_pickle=False)
    eligible = [x for x in ranking if float(e.np.linalg.norm(frozen['delta'][x])) > 1e-12]
    if len(eligible) < 4:
        raise ValueError('Fewer than four nonzero frozen action directions are available')
    old = read(a.repo / 'experiments/module2_applicability_20261027/results/qwen3_0_6b_cpu/selection.json')
    heads = [[r['layer'], r['kv_head']] for r in old['selected_heads']]
    dump(out / 'heads.json', {'old_single': heads[:1], 'old_multi': heads,
        'causal_single': [list(eligible[0])], 'causal_multi': [list(x) for x in eligible[:4]],
        'ranking': [{'layer': l, 'head': h, 'mean': sum(aggregate[l,h])/len(aggregate[l,h]),
                     'nonzero_frozen_direction': (l,h) in eligible}
                    for l, h in ranking], 'selection_scope': 'fit only'})


def operators(a, e, heads):
    f = a.repo / 'experiments/module2_applicability_20261027/results/qwen3_0_6b_cpu/operators.npz'
    archive = e.np.load(f, allow_pickle=False)
    basis = e.torch.tensor(e.np.stack([archive['basis'][l, h] for l, h in heads]),
                           device=e.device, dtype=e.torch.float32)
    delta = e.torch.tensor(e.np.stack([archive['delta'][l, h] for l, h in heads]),
                           device=e.device, dtype=e.torch.float32)
    return basis, delta


def train(a, p, out, e):
    t, data = e.torch, samples(a, 'fit')
    hs = read(a.run / 'localize/heads.json')
    cfg = p['training']
    for name in ('old_single', 'old_multi', 'causal_single', 'causal_multi'):
        heads = hs[name]
        basis, delta = operators(a, e, heads)
        for method in ('optimized_subspace', 'optimized_score_bias',
                       'permuted_seed0', 'permuted_seed1', 'permuted_seed2'):
            isbias = method == 'optimized_score_bias'
            coefficient = t.ones(len(heads), device=e.device) if isbias else t.einsum('hdr,hd->hr', basis, delta)
            coefficient = coefficient / coefficient.norm().clamp_min(1e-8)
            coefficient = t.nn.Parameter(coefficient)
            optimizer = t.optim.Adam([coefficient], lr=cfg['learning_rate'])
            rng = random.Random(p['seed'])
            signs = {g: 1 for g in sorted({s['group'] for s in data})}
            if method.startswith('permuted'):
                prng = random.Random(p['seed'] + int(method[-1]) + 100)
                signs = {g: prng.choice([-1, 1]) for g in signs}
            for step in range(cfg['steps']):
                optimizer.zero_grad()
                batch = rng.sample(data, cfg['batch_size'])
                losses = []
                for s in batch:
                    state = e.state(s)
                    # Group-level label permutation affects fitting objective and routing.
                    fit_s = dict(s)
                    if signs[s['group']] < 0:
                        fit_s['gold'] = 'B' if s['gold'] == 'A' else 'A'
                        state.gold = fit_s['gold']
                    vectors = coefficient if isbias else t.einsum('hdr,hr->hd', basis, coefficient)
                    keys, biases, _ = e.edits(state, heads, vectors, cfg['train_tau'], 'joint', isbias)
                    z = e.forward(state, keys, biases)
                    gi = 'AB'.index(fit_s['gold'])
                    loss = -(z[e.labels[gi]] - z[e.labels[1-gi]]) / len(batch)
                    loss.backward()
                    losses.append(float(loss.detach()) * len(batch))
                optimizer.step()
                with t.no_grad():
                    coefficient.div_(coefficient.norm().clamp_min(1e-8))
                journal(out / 'optimization.jsonl', {'head_set': name, 'method': method,
                    'step': step, 'negative_margin': sum(losses)/len(losses)})
            dump(out / f'{name}__{method}.json', {'coefficient': coefficient.detach().cpu().tolist(),
                 'heads': heads, 'group_signs': signs, 'fit_only': True})


def method_vectors(a, p, e, head_set, method, heads):
    t = e.torch
    basis, delta = operators(a, e, heads)
    if method == 'projected_mean':
        return delta, False
    if method in ('score_bias', 'reversed_score_bias'):
        sign = -1 if method == 'reversed_score_bias' else 1
        return sign * t.ones(len(heads), device=e.device), True
    if method.startswith('random_seed'):
        gen = t.Generator(device=e.device).manual_seed(p['seed'] + int(method[-1]))
        return t.randn(delta.shape, generator=gen, device=e.device), False
    saved = read(a.run / f'train/{head_set}__{method}.json')
    coefficient = t.tensor(saved['coefficient'], device=e.device)
    if method == 'optimized_score_bias':
        return coefficient, True
    return t.einsum('hdr,hr->hd', basis, coefficient), False


def summarize_rows(records, path):
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[(r.get('split'), r.get('method', 'baseline'), r.get('head_set'),
                 r.get('mode'), r.get('tau'))].append(r)
    output = []
    for key, rec in grouped.items():
        if not rec:
            continue
        gs, pairs = collections.defaultdict(list), collections.defaultdict(list)
        for r in rec:
            gs[r['group']].append(r)
            pairs[r['pair']].append(r)
        per_group = []
        for g, gr in sorted(gs.items()):
            gpairs = [rr for rr in pairs.values() if rr[0]['group'] == g]
            if any(len(pp) != 2 for pp in gpairs):
                raise ValueError('Incomplete reciprocal pair')
            per_group.append({'group': g, 'accuracy': sum(r['correct'] for r in gr)/len(gr),
                'margin': sum(r['margin'] for r in gr)/len(gr),
                'paired_accuracy': sum(all(x['correct'] for x in pp) for pp in gpairs)/len(gpairs),
                'corrected': sum(r.get('corrected', 0) for r in gr),
                'regressed': sum(r.get('regressed', 0) for r in gr)})
        output.append(dict(zip(('split','method','head_set','mode','tau'), key),
            n=len(rec), guidelines=len(gs), per_guideline=per_group,
            accuracy=sum(g['accuracy'] for g in per_group)/len(per_group),
            margin=sum(g['margin'] for g in per_group)/len(per_group),
            paired_accuracy=sum(g['paired_accuracy'] for g in per_group)/len(per_group)))
    dump(path, output)
    return output


def evaluate(a, p, out, e, confirm=False):
    t = e.torch
    headsets = read(a.run / 'localize/heads.json')
    if confirm:
        frozen = read(a.run / 'select/frozen.json')
        for file, expected in frozen['hashes'].items():
            if sha(file) != expected:
                raise ValueError(f'Frozen artifact changed: {file}')
        configs = frozen['choices'] + frozen['ablations']
    else:
        configs = [dict(method=m, head_set=h, mode=mode, tau=tau)
                   for m in p['methods'] for h in ('old_single','old_multi','causal_single','causal_multi')
                   for mode in p['modes'] for tau in p['taus']]
    prepared = []
    for c in configs:
        heads = headsets[c['head_set']]
        vectors, bias = method_vectors(a, p, e, c['head_set'], c['method'], heads)
        prepared.append((c, heads, vectors, bias))
    split = 'confirm' if confirm else 'select'
    for s in samples(a, split):
        state = e.state(s)
        baseline = e.outcome(state.baseline, s)
        journal(out / 'predictions.jsonl', {**public(s), 'method': 'baseline', **baseline})
        for c, heads, vectors, bias in prepared:
            with t.no_grad():
                keys, bs, diagnostic = e.edits(state, heads, vectors, c['tau'], c['mode'], bias)
                z = e.forward(state, keys, bs)
            r = e.outcome(z, s)
            if c['tau'] == 0 and not t.equal(z, state.baseline):
                raise ValueError('Zero intervention differs from baseline')
            filename = 'ablations.jsonl' if c.get('analysis') == 'ablation' else 'predictions.jsonl'
            journal(out / filename, {**public(s), **c, **r,
                'rms': diagnostic,
                'corrected': int(r['correct'] and not baseline['correct']),
                'regressed': int(baseline['correct'] and not r['correct'])})
    records = rows(out / 'predictions.jsonl')
    summary = summarize_rows(records, out / 'summary.json')
    for mapping in (0, 1):
        summarize_rows([r for r in records if r['mapping'] == mapping], out / f'mapping{mapping}.json')
    if confirm:
        summarize_rows(rows(out / 'ablations.jsonl'), out / 'ablation_summary.json')
    if not confirm:
        choices = []
        for method in p['methods']:
            candidates = [x for x in summary if x['method'] == method]
            winner = min(candidates, key=lambda x: (-x['accuracy'], -x['margin'],
                         x['tau'], x['head_set'], x['mode']))
            choices.append({k: winner[k] for k in ('method','head_set','mode','tau')})
        artifacts = [Path(__file__).resolve(), a.protocol.resolve(),
                     a.run / 'prepare/manifest.json', a.run / 'localize/heads.json']
        artifacts += sorted((a.run / 'prepare').glob('*.jsonl'))
        artifacts += sorted((a.run / 'train').glob('*.json'))
        old = read(a.run / 'prepare/manifest.json')['old_files']
        artifacts += [Path(f) for f in old]
        primary_bias = min([r for r in summary if r['method'] in
                          ('score_bias', 'reversed_score_bias', 'optimized_score_bias')],
                         key=lambda r: (-r['accuracy'], -r['margin'], r['method'],
                                        r['tau'], r['head_set'], r['mode']))['method']
        ablations = []
        for method in ('optimized_subspace', 'score_bias', 'optimized_score_bias'):
            chosen = next(c for c in choices if c['method'] == method)
            for head_set in ('old_single','old_multi','causal_single','causal_multi'):
                for mode in p['modes']:
                    ablations.append(dict(method=method, head_set=head_set, mode=mode,
                                          tau=chosen['tau'], analysis='ablation'))
        dump(out / 'frozen.json', {'choices': choices, 'ablations': ablations,
            'created_unix': time.time(),
            'primary_score_baseline': primary_bias,
            'hashes': {str(f.resolve()): sha(f) for f in artifacts},
            'confirmation_outcomes_read': False,
            'exposure_status': 'unseen in this follow-up; other projects not audited'})


def report(a, p, out):
    import numpy as np
    records = rows(a.run / 'confirm/predictions.jsonl')
    summary = read(a.run / 'confirm/summary.json')
    by_method = {r['method']: r for r in summary}
    candidate = by_method['optimized_subspace']
    comparisons = []
    rng = np.random.default_rng(p['bootstrap']['seed'])
    for control in ['baseline'] + [m for m in p['methods'] if m != 'optimized_subspace']:
        cgroups = {r['group']: r for r in by_method[control]['per_guideline']}
        delta = np.array([g['accuracy'] - cgroups[g['group']]['accuracy']
                          for g in candidate['per_guideline']])
        boot = rng.choice(delta, size=(p['bootstrap']['draws'], len(delta)), replace=True).mean(1)
        q = np.quantile(boot, [.0125, .9875])
        comparisons.append({'control': control, 'accuracy_difference': float(delta.mean()),
            'paired_guideline_97_5_interval': q.tolist()})
    primary_bias = read(a.run / 'select/frozen.json')['primary_score_baseline']
    required = [c for c in comparisons if c['control'] in ('baseline', primary_bias)]
    passed = all(c['accuracy_difference'] >= .05 and
                 c['paired_guideline_97_5_interval'][0] > 0 for c in required)
    dump(out / 'comparison.json', {'comparisons': comparisons, 'numeric_criterion_met': passed,
        'primary_score_baseline': primary_bias,
        'hypothesis_proven': False, 'reason': 'Cross-project untouched status remains unestablished',
        'rows': len(records), 'summary': summary})
    ablations = rows(a.run / 'confirm/ablations.jsonl')
    baseline = {r['id']: r for r in records if r['method'] == 'baseline'}
    cells = collections.defaultdict(dict)
    for r in ablations:
        cells[(r['method'],r['head_set'],r['tau'],r['id'])][r['mode']] = r
    interactions = []
    for (method, hs, tau, sid), cell in cells.items():
        if set(cell) != {'joint','enhance','suppress'}:
            raise ValueError('Incomplete factorial ablation')
        interactions.append({'method':method, 'head_set':hs, 'tau':tau, 'id':sid,
            'group':baseline[sid]['group'],
            **{f'{metric}_interaction': cell['joint'][metric] - cell['enhance'][metric]
              - cell['suppress'][metric] + baseline[sid][metric]
              for metric in ('correct','margin')}})
    dump(out / 'factorial_interactions.json', interactions)
    text = ['TRIAGE 规则选择干预：新一轮结果',
            '所有数值来自 confirm/predictions.jsonl；以来源指南为统计单位。',
            '使用适用性真值确定增强/抑制符号，因此是算子上界实验。',
            '新指南未用于本轮拟合或选择，但其他项目的历史接触情况尚未确认。', '']
    for method, r in by_method.items():
        text.append(f"{method}: 准确率 {r['accuracy']:.3%}，配对全对率 {r['paired_accuracy']:.3%}，平均间隔 {r['margin']:.6f}")
    text += ['', f'预定数值判据是否满足：{passed}。',
             '不据本报告声称假设已被证明；完整负结果、分层和改对/改错数见 JSON。']
    (out / 'REPORT.zh.txt').write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','audit','localize','train','select','confirm','report'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.json'))
    parser.add_argument('--gpu', type=int, choices=[0,1], default=0)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--trees', type=Path)
    a = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    a.repo, a.run = a.repo.resolve(), a.run.resolve()
    sys.path.insert(0, str(a.repo))
    p = read(a.protocol)
    out = a.run / a.stage
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dump(out / 'status.json', {'status': 'running', 'stage': a.stage})
    try:
        if a.stage == 'prepare':
            prepare(a, p, out)
        elif a.stage == 'report':
            if read(a.run / 'confirm/status.json')['status'] != 'completed':
                raise RuntimeError('Confirmation incomplete; no completed-result report')
            report(a, p, out)
        else:
            dependencies = {'audit':['prepare'], 'localize':['audit'],
                            'train':['localize'], 'select':['train'], 'confirm':['select']}
            for stage in dependencies[a.stage]:
                if read(a.run / stage / 'status.json')['status'] != 'completed':
                    raise RuntimeError(f'Prerequisite incomplete: {stage}')
            manifest = read(a.run / 'prepare/manifest.json')
            for file, expected in manifest['old_files'].items():
                if sha(file) != expected:
                    raise ValueError(f'Original artifact changed: {file}')
            for split, expected in manifest['new_files'].items():
                if sha(a.run / f'prepare/{split}.jsonl') != expected:
                    raise ValueError(f'Prepared split changed: {split}')
            e = Engine(a, p)
            dump(out / 'environment.json', e.environment)
            if a.stage == 'audit': audit(a, p, out, e)
            elif a.stage == 'localize': localize(a, p, out, e)
            elif a.stage == 'train': train(a, p, out, e)
            else: evaluate(a, p, out, e, confirm=a.stage == 'confirm')
            dump(out / 'memory.json', {'peak_allocated_gib': e.torch.cuda.max_memory_allocated(a.gpu)/2**30})
        dump(out / 'status.json', {'status': 'completed', 'stage': a.stage,
                                 'elapsed_seconds': time.time()-started})
    except BaseException as exc:
        (out / 'error.log').write_text(traceback.format_exc())
        dump(out / 'status.json', {'status':'failed', 'stage':a.stage,
            'error_type':type(exc).__name__, 'error':str(exc),
            'elapsed_seconds':time.time()-started})
        raise


if __name__ == '__main__':
    main()
