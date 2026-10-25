"""Sparse answer-query attention: bounded support/current-action-mass bias.

The SDPA path handles every unselected query/head. Only explicitly selected
answer queries are recomputed in FP32, including for zero strength. No KV cache.
"""
import torch
import torch.nn.functional as F


def group_bias(prob, allowed, groups, support, strength, bound=2., kind='dynamic'):
    """Bias is bounded by bound*abs(strength); masked/ungrouped keys get zero.

    Duplicate rule/action and option spans aggregate into the same action mass.
    Support is normalized over visible groups. Nonaction keys remain unbiased.
    """
    if not len(support):
        return torch.zeros_like(prob)
    groups = torch.as_tensor(groups, device=prob.device, dtype=torch.long)
    support = torch.as_tensor(support, device=prob.device, dtype=prob.dtype)
    if len(groups) != prob.shape[-1] or (groups >= len(support)).any() or (groups < -1).any():
        raise ValueError('invalid action-group mapping')
    if not torch.isfinite(support).all() or (support < 0).any():
        raise ValueError('support must be finite and nonnegative')
    ix = groups.clamp_min(0).expand_as(prob)
    active = (groups >= 0) & allowed
    mass = prob.new_zeros((*prob.shape[:-1], len(support)))
    count = torch.zeros_like(mass)
    mass.scatter_add_(-1, ix, prob * active)
    count.scatter_add_(-1, ix, active.to(prob.dtype))
    target = support * (count > 0)
    total = target.sum(-1, keepdim=True)
    target = target / total.clamp_min(1e-12)
    current = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
    if kind == 'dynamic':
        delta = (target.clamp_min(1e-8).log() - current.clamp_min(1e-8).log()).clamp(-bound, bound)
    elif kind == 'static':
        delta = (target.clamp_min(1e-8).log() - target.max(-1, keepdim=True).values.clamp_min(1e-8).log()).clamp(-bound, bound)
    else:
        raise ValueError(kind)
    delta = torch.where(total > 0, delta, torch.zeros_like(delta))
    return torch.where(active, strength * delta.gather(-1, ix), 0.)


def sparse_attention(q, k, v, mask, scale, groups, support, heads, query_start,
                     strength=1., bound=2., kind='dynamic', return_details=False):
    if q.shape[-2] != k.shape[-2]:
        raise ValueError('full prefill only; cached decoding is not supported')
    if q.shape[1] % k.shape[1]:
        raise ValueError('query heads must be divisible by KV heads')
    n = q.shape[-2]
    heads = list(heads)
    if len(set(heads)) != len(heads) or any(h < 0 or h >= q.shape[1] for h in heads):
        raise ValueError('invalid head indices')
    if not 0 <= query_start <= n:
        raise ValueError('invalid answer-query boundary')
    repeat = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(repeat, 1)
    vr = v.repeat_interleave(repeat, 1)
    # Explicit causal intersection even if a supplied mask is all-True.
    causal = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()[None, None]
    if mask is None:
        full_mask = causal
    elif mask.dtype == torch.bool:
        full_mask = mask[..., :n, :n] & causal
    else:
        full_mask = mask[..., :n, :n].masked_fill(~causal, float('-inf'))
    output = F.scaled_dot_product_attention(q, kr, vr, attn_mask=full_mask, dropout_p=0., scale=scale)
    if not heads or query_start == n:
        return (output, {}) if return_details else output
    scores = (q[:, heads, query_start:].float() @ kr[:, heads].float().transpose(-1, -2)) * scale
    m = full_mask.expand(q.shape[0], q.shape[1], n, n)[:, heads, query_start:]
    if m.dtype == torch.bool:
        allowed = m
    else:
        allowed = torch.isfinite(m) & (m > -1e4)
        scores = scores + m.float()
    scores = scores.masked_fill(~allowed, float('-inf'))
    prob = scores.softmax(-1).nan_to_num(0.)
    bias = group_bias(prob, allowed, groups, support, strength, bound, kind)
    after = (scores + bias).softmax(-1).nan_to_num(0.)
    output[:, heads, query_start:] = (after @ vr[:, heads].float()).to(v.dtype)
    if return_details:
        return output, {'before': prob, 'after': after, 'bias': bias, 'allowed': allowed}
    return output
