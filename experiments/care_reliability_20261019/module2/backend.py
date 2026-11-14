"""Qwen3 real-model intervention adapter; imported only in a model environment.

The CPU reference tests do not exercise this Torch adapter. A real run must
pass its separate zero-edit, archived-operator and realized-score checks.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from geometry import anchor_factors, anchor_audit, bad_anchor_tokens, scalar_search


def load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InterventionGeometry:
    """Captured baseline queries and pre-RoPE action keys at one frozen KV head.

    Queries at this layer are unchanged by this layer's key-only intervention.
    Local certificates concern the last prompt query, not future generation.
    """
    def __init__(self, model, capture, token_indices, location, protocol):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import rotate_half
        self.torch = torch
        self.rotate_half = rotate_half
        self.model = model
        self.location = tuple(location)
        self.indices = list(token_indices)
        self.protocol = protocol
        layer, head = self.location
        q = capture[layer]['q']
        self.key = capture[layer]['k'][self.indices, head]
        if q.dtype != torch.bfloat16 or self.key.dtype != torch.bfloat16:
            raise AssertionError('Frozen score certificate requires captured BF16 queries and keys')
        length, _, self.dimension = q.shape
        self.group_size = model.config.num_attention_heads // model.config.num_key_value_heads
        assert self.group_size == 4 and self.dimension == 128
        pos = torch.arange(length, device=q.device)[None]
        cosine, sine = model.model.rotary_emb(
            torch.zeros((1, length, model.config.hidden_size), device=q.device, dtype=q.dtype), pos)
        cosine, sine = cosine[0], sine[0]
        self.cosine = cosine[self.indices]
        self.sine = sine[self.indices]
        rotated_q = q * cosine[:, None] + rotate_half(q) * sine[:, None]
        self.q_rotated = rotated_q[:, self.group_size*head:self.group_size*(head+1)].float()
        self.original_rotated_key = self.rotate_actual(self.key)
        self.valid = torch.stack(
            [torch.arange(length, device=q.device) >= token for token in self.indices], dim=-1
        )[None].expand(self.group_size, -1, -1)
        self.anchor_query_group = self.q_rotated[-1].detach().cpu().numpy()

    def rotate_actual(self, key):
        # Each operation follows the BF16 model path, including intermediate rounding.
        return (key * self.cosine + self.rotate_half(key) * self.sine).float()

    def rotate_direction(self, direction):
        # Linear ideal direction, before the separate realized-BF16 certificate.
        return (direction.float() * self.cosine.float()
                + self.rotate_half(direction.float()) * self.sine.float())

    def full_score_delta(self, changed_key):
        delta = self.rotate_actual(changed_key) - self.original_rotated_key
        return (self.torch.einsum('tgd,md->gtm', self.q_rotated, delta)
                / math.sqrt(self.dimension))[self.valid]

    def anchor_delta(self, changed_key):
        delta = self.rotate_actual(changed_key) - self.original_rotated_key
        return self.q_rotated[-1] @ delta.T / math.sqrt(self.dimension)

    def unsigned_direction(self, name, operators):
        value = self.torch.as_tensor(operators[name], dtype=self.torch.float32, device=self.key.device)
        if name in ('b4', 'random_projector'):
            return self.key.float() @ value
        return value[None].expand_as(self.key).float()

    def _audit(self, changed, sign, eligible, calibration):
        scores = self.full_score_delta(changed)
        anchor = self.anchor_delta(changed).detach().cpu().numpy()
        local = anchor_audit(anchor, sign, eligible,
                             tolerance=self.protocol['actual_anchor_sign_tolerance'])
        local.update(calibration)
        local.update(
            score_support_count=int(scores.numel()),
            full_support_negative_fraction=float((scores < 0).float().mean()),
            full_support_score_mean=float(scores.mean()),
            relative_key_norm=float((changed.float()-self.key.float()).norm()/self.key.float().norm()),
            score_support_includes_skipped_zero_entries=True,
            score_certificate_basis=self.protocol['score_certificate_basis'],
            anchor_is_last_prompt_query=True,
        )
        return local

    def key_intervention(self, name, calibration_mode, operators, sign, target_rms):
        if calibration_mode not in ('raw', 'calibrated') or sign not in (-1, 1):
            raise ValueError('Unknown calibration mode or sign')
        torch = self.torch
        unsigned = self.unsigned_direction(name, operators)
        rotated = self.rotate_direction(unsigned).detach().cpu().numpy()
        factors = anchor_factors(
            self.anchor_query_group, rotated, self.protocol['angle_floor'],
            self.protocol['cosine_tolerance'])
        consensus = factors['eligible'].copy()
        if calibration_mode == 'calibrated':
            keep = consensus.copy()
            gamma = torch.as_tensor(factors['gamma'], device=self.key.device, dtype=torch.float32)
            direction = sign * unsigned * gamma[:, None]
        else:
            keep = np.ones(len(self.indices), dtype=bool)
            direction = sign * unsigned
        quarantined = np.zeros(len(self.indices), dtype=bool)
        maximum_iterations = len(self.indices) + 1
        for iteration in range(maximum_iterations):
            active = direction * torch.as_tensor(keep[:, None], dtype=torch.float32, device=self.key.device)
            if target_rms == 0 or not keep.any() or float(active.norm()) <= 1e-12:
                changed = self.key.clone()
                state = 'zero' if target_rms == 0 else 'no_eligible_direction'
                matched = dict(target_rms=float(target_rms), actual_rms=0.0, scale=0.0,
                               relative_error=(0.0 if target_rms == 0 else None),
                               calibration_status=state)
            else:
                # Same full span and full causal support as raw/direct conditions.
                # Consensus-rejected tokens remain zero entries in this RMS.
                normalized = active / active.norm() * self.key.float().norm()
                def evaluate(scale):
                    candidate = (self.key.float() + scale*normalized).to(self.key.dtype)
                    scores = self.full_score_delta(candidate)
                    return candidate, float(scores.square().mean().sqrt())
                changed, matched = scalar_search(
                    evaluate, target_rms, self.protocol['maximum_rms_relative_error'])
            actual_anchor = self.anchor_delta(changed).detach().cpu().numpy()
            bad = bad_anchor_tokens(actual_anchor, sign, self.protocol['actual_anchor_sign_tolerance'])
            if calibration_mode == 'raw' or not bad.any():
                audit = self._audit(changed, sign, keep, matched)
                audit.update(direction=name, calibration_mode=calibration_mode,
                             gqa_ideal_consensus_fraction=float(consensus.mean()),
                             bf16_quarantined_token_count=int(quarantined.sum()),
                             bf16_quarantine_iterations=int(iteration),
                             eligible_token_indices=[self.indices[i] for i in np.flatnonzero(keep)])
                if calibration_mode == 'calibrated':
                    assert audit['anchor_group_all_nonnegative']
                return changed, audit
            newly_bad = bad & keep
            if not newly_bad.any():
                raise AssertionError('A skipped token changed: invalid actual-score certificate')
            keep[newly_bad] = False
            quarantined[newly_bad] = True
        raise AssertionError('GQA/BF16 quarantine failed to terminate')

    def direct_bias_audit(self, sign, target_rms):
        actual = float(self.torch.tensor(target_rms, dtype=self.model.dtype))
        deltas = np.full((self.group_size, len(self.indices)), sign*actual)
        result = anchor_audit(deltas, sign, tolerance=self.protocol['actual_anchor_sign_tolerance'])
        result.update(
            direction='direct_bias', calibration_mode='score_bias',
            target_rms=float(target_rms), actual_rms=abs(actual),
            relative_error=abs(actual-target_rms)/target_rms if target_rms else 0.0,
            calibration_status='matched' if target_rms else 'zero',
            gqa_ideal_consensus_fraction=1.0, bf16_quarantined_token_count=0,
            bf16_quarantine_iterations=0, score_support_count=int(self.valid.sum()),
            full_support_negative_fraction=float(sign < 0 and target_rms > 0),
            full_support_score_mean=sign*actual, relative_key_norm=None,
            score_support_includes_skipped_zero_entries=True,
            score_certificate_basis='direct BF16 attention-mask bias at the frozen causal entries',
            anchor_is_last_prompt_query=True,
            eligible_token_indices=self.indices,
        )
        return result
