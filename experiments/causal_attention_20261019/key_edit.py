"""Query-aligned, paired-mean and random key edits for the new causal experiment.

The callback receives post-RoPE Q/K. The edited keys serve every causally
visible query; selected query rows only define the solve and local diagnostics.
No token is removed because its realized change has an undesired sign.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def _layer(module):
    return int(module.layer_idx)


class KeyEditor:
    """Post-RoPE key editing with one scalar calibration per GQA query group.

    query_aligned: ridge least squares over all selected causal query/head rows.
    learned_mean/random: normalize the direction, then gamma=z/(z*z+epsilon),
    where z=<mean selected Q,u>/sqrt(d) and
    epsilon=0.01*||mean selected Q||^2/d. This smooth scalar aligns the average
    group response without a four-head sign gate or per-example RMS matching.

    Learned vectors are true within-pair T-F post-RoPE action-key differences,
    averaged over training pairs separately for each selected layer and KV head.
    This intentionally tests a simple direction in the actual edited key space.
    """
    def __init__(self, method, strength, query_window=1, directions=None,
                 ridge=0.01, smooth=0.01, seed=20261019):
        self.method = method
        self.strength = float(strength)
        self.query_window = int(query_window)
        self.directions = directions or {}
        self.ridge = float(ridge)
        self.smooth = float(smooth)
        self.seed = int(seed)
        self.case = None
        self.capture = {}
        self.diagnostics = []
        self.random_directions = {}

    def set_case(self, case):
        self.case = case
        self.capture = {}
        self.diagnostics = []

    def _targets(self, engine):
        if hasattr(engine, 'targets'):
            return engine.targets
        return engine.current.get('targets', [])

    def __call__(self, module, query, key, value, attention_mask, scaling, engine):
        layer = _layer(module)
        q = query[0].float()   # [query heads, queries, d]
        k = key[0].float()     # [KV heads, keys, d], post RoPE
        heads, key_count, dimension = k.shape
        group_size = q.shape[0] // heads
        if group_size != 4:
            raise ValueError(f'Expected four GQA query heads, got {group_size}')
        q_count = q.shape[-2]
        query_indices = torch.arange(max(0, q_count-self.query_window), q_count,
                                     device=q.device)
        signs = torch.as_tensor(engine.token_signs, device=k.device, dtype=torch.float32)
        selected = torch.nonzero(signs != 0, as_tuple=False).flatten()
        if self.method == 'capture':
            records = {}
            for target in self._targets(engine):
                if target.get('kind', 'action') != 'action':
                    continue
                indices = target.get('token_indices', [])
                if not indices:
                    continue
                rule = target['rule_id']
                records[rule] = k[:, indices].mean(dim=1).detach().cpu()
            self.capture[layer] = records
            return key, {'method':'capture', 'layer':layer, 'captured_rules':list(records)}

        changed = k.clone()
        group_records = []
        for head in range(heads):
            qg = q[head*group_size:(head+1)*group_size]
            if self.method == 'random':
                identity = (layer, head, dimension)
                if identity not in self.random_directions:
                    rng = torch.Generator(device='cpu')
                    rng.manual_seed(self.seed+layer*1009+head*101)
                    u = torch.randn(dimension, generator=rng)
                    self.random_directions[identity] = u/u.norm()
                u = self.random_directions[identity].to(k.device)
            elif self.method == 'learned_mean':
                tensor = self.directions.get(layer, self.directions.get(str(layer)))
                if tensor is None:
                    raise ValueError(f'Missing freshly learned direction for layer {layer}')
                u = torch.as_tensor(tensor[head], device=k.device, dtype=torch.float32)
                u = u/u.norm().clamp_min(1e-12)
            elif self.method != 'query_aligned':
                raise ValueError(f'Unknown key-edit method: {self.method}')

            # A target can lie inside the selected window; only causal query
            # rows contribute to its solve. Targets are never silently dropped.
            by_causal_rows = defaultdict(list)
            for token in selected.tolist():
                first = max(0, token-int(query_indices[0]))
                by_causal_rows[first].append(token)
            for first, tokens in by_causal_rows.items():
                valid_queries = query_indices[first:]
                if valid_queries.numel() == 0:
                    # No selected query can see this key; record the no-op.
                    group_records.append({'kv_head':head, 'tokens':tokens,
                                          'causal_rows':0, 'status':'no_causal_support'})
                    continue
                a = qg[:, valid_queries].reshape(-1, dimension)/math.sqrt(dimension)
                if self.method == 'query_aligned':
                    gram = a @ a.T
                    regularizer = self.ridge*gram.diag().mean().clamp_min(1e-12)
                    weights = torch.linalg.solve(
                        gram+regularizer*torch.eye(gram.shape[0], device=gram.device),
                        torch.ones(gram.shape[0], device=gram.device))
                    direction = a.T @ weights
                    diagnostics = {'ridge':float(regularizer)}
                else:
                    mean_a = a.mean(dim=0)
                    z = mean_a @ u
                    epsilon = self.smooth*mean_a.square().sum()
                    gamma = z/(z.square()+epsilon).clamp_min(1e-12)
                    direction = gamma*u
                    diagnostics = {'group_z':float(z), 'epsilon':float(epsilon),
                                   'gamma':float(gamma)}
                changed[head, tokens] += (
                    self.strength*signs[tokens, None]*direction[None])
                # Actual changed BF16 keys, not the ideal FP32 direction.
                realized = changed[head, tokens].to(key.dtype).float()-k[head, tokens]
                delta = a @ realized.T
                signed = delta*signs[tokens][None]
                full_delta = torch.einsum('hqd,td->hqt',qg,realized)/math.sqrt(dimension)
                causal = torch.arange(q_count,device=q.device)[:,None] >= torch.tensor(tokens,device=q.device)[None,:]
                full_signed = (full_delta*signs[tokens][None,None])[:,causal]
                full_valid = full_delta[:,causal]
                group_records.append({
                    'kv_head':head, 'tokens':tokens, 'causal_rows':int(a.shape[0]),
                    'full_causal_entries':int(full_valid.numel()),
                    'full_causal_signed_mean':float(full_signed.mean()),
                    'full_causal_wrong_sign_fraction':float((full_signed < -1e-7).float().mean()),
                    'full_causal_score_rms':float(full_valid.square().mean().sqrt()),
                    'signed_score_mean':float(signed.mean()),
                    'score_rms':float(delta.square().mean().sqrt()),
                    'wrong_sign_fraction':float((signed < -1e-7).float().mean()),
                    'nonzero_fraction':float((delta.abs() > 1e-7).float().mean()),
                    'relative_key_change':float(realized.norm()/
                                               k[head,tokens].norm().clamp_min(1e-12)),
                    'status':'edited', **diagnostics})
        result = changed[None].to(key.dtype)
        audit = {'method':self.method, 'layer':layer, 'strength':self.strength,
                 'query_window':self.query_window, 'target_token_count':int(selected.numel()),
                 'query_group_size':group_size, 'groups':group_records,
                 'sign_gate':False, 'per_case_rms_match':False}
        self.diagnostics.append(audit)
        return result, audit


def learn_paired_directions(cases, captures, layers):
    """Average real T-F matched-pair differences, never unpaired class pools.

    Each case capture is {layer:{rule_id:Tensor[KVheads,d]}}. A pair contributes
    one difference per rule that actually switches T/F. Unknown and unchanged
    rule states contribute no training contrast. All cases remain evaluable.
    """
    paired = defaultdict(list)
    for case in cases:
        if case.get('split') != 'train':
            raise ValueError('Direction extraction accepts training cases only')
        if case.get('pair_eligible'):
            paired[case['pair_id']].append(case)
    differences = defaultdict(list)
    pair_records = []
    for pair_id, members in sorted(paired.items()):
        if len(members) != 2:
            raise ValueError(f'Training pair is not complete: {pair_id}')
        a, b = members
        states_a = {r['rule_id']:r['state'] for r in a['reference_structure']['rules']}
        states_b = {r['rule_id']:r['state'] for r in b['reference_structure']['rules']}
        for rule in sorted(set(states_a) & set(states_b)):
            if {states_a[rule], states_b[rule]} != {'T','F'}:
                continue
            true, false = (a,b) if states_a[rule] == 'T' else (b,a)
            pair_records.append({'pair_id':pair_id, 'rule_id':rule,
                                 'true_case':true['case_id'], 'false_case':false['case_id']})
            for layer in layers:
                true_k = captures[true['case_id']][layer][rule]
                false_k = captures[false['case_id']][layer][rule]
                differences[layer].append(true_k-false_k)
    result = {}
    audit = {'training_cases':len(cases), 'paired_cases':sum(map(len,paired.values())),
             'pairs':len(paired), 'rule_contrasts':len(pair_records),
             'direction_space':'post-RoPE action-key mean',
             'training_pairs':pair_records, 'layers':{}}
    for layer in layers:
        if not differences[layer]:
            raise ValueError(f'No T-F paired contrast at layer {layer}')
        samples = torch.stack(differences[layer])
        mean = samples.mean(dim=0)
        result[layer] = mean
        audit['layers'][str(layer)] = {
            'contrasts':int(samples.shape[0]),
            'kv_heads':int(mean.shape[0]),
            'dimension':int(mean.shape[1]),
            'direction_norm_per_head':mean.norm(dim=-1).tolist(),
            'mean_pair_difference_norm_per_head':samples.norm(dim=-1).mean(dim=0).tolist()}
    return result, audit

