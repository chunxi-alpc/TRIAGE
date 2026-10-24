"""Optional Torch BF16 adapter/hook checks, distinct from real-model inference.

These tests are explicitly skipped when Torch/Transformers are unavailable.
They use artificial tensors and a minimal module, never clinical predictions.
"""
import importlib.util
import math
import types
import unittest
from pathlib import Path

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers'))


@unittest.skipUnless(AVAILABLE, 'PyTorch/Transformers unavailable; Torch adapter has NOT been validated')
class TorchAdapterProperties(unittest.TestCase):
    def test_bf16_realized_anchor_and_full_support(self):
        import torch
        from backend import InterventionGeometry
        from run import read
        protocol = read(Path(__file__).with_name('protocol.json'))
        torch.manual_seed(11)
        q = torch.randn(8, 32, 128).to(torch.bfloat16)
        q[:, :4] = q[:, :1].expand(-1, 4, -1)
        k = torch.randn(8, 8, 128).to(torch.bfloat16)
        def rope(hidden, positions):
            theta = positions[..., None].float() * torch.linspace(.001, .1, 64, device=hidden.device)
            theta = torch.cat((theta, theta), dim=-1)
            return theta.cos().to(hidden.dtype), theta.sin().to(hidden.dtype)
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096),
            model=types.SimpleNamespace(rotary_emb=rope), dtype=torch.bfloat16)
        geometry = InterventionGeometry(model, {0: {'q': q, 'k': k}}, [2, 3, 4], (0, 0), protocol)
        operators = {'b4': np.eye(128, dtype=np.float32)}
        for sign in (-1, 1):
            changed, audit = geometry.key_intervention('b4', 'calibrated', operators, sign, .05)
            self.assertTrue(audit['anchor_group_all_nonnegative'])
            self.assertEqual(audit['calibration_status'], 'matched')
            self.assertLessEqual(audit['relative_error'], .15)
            rotated = geometry.rotate_actual(changed)
            reference = []
            for head in range(4):
                for query in range(8):
                    for token_index, token_position in enumerate([2, 3, 4]):
                        if query >= token_position:
                            before = torch.dot(geometry.q_rotated[query, head], geometry.original_rotated_key[token_index])
                            after = torch.dot(geometry.q_rotated[query, head], rotated[token_index])
                            reference.append((after-before)/math.sqrt(128))
            torch.testing.assert_close(geometry.full_score_delta(changed), torch.stack(reference), atol=2e-6, rtol=2e-4)
            zero, zero_audit = geometry.key_intervention('b4', 'calibrated', operators, sign, 0)
            self.assertTrue(torch.equal(zero, geometry.key))
            self.assertEqual(zero_audit['actual_rms'], 0)

    def test_reused_key_and_bias_hook_scope_and_zero(self):
        import torch
        from backend import load_python
        from run import ROOT
        legacy = load_python(ROOT/'experiments/care_module2_signed_20261019/diagnostic.py', 'care_hook_unit_fixture')
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_norm = torch.nn.Identity()
            def forward(self, hidden, attention_mask):
                return attention_mask
        attention = Attention()
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attention)]), dtype=torch.bfloat16)
        tensor = torch.randn(1, 5, 8, 128).to(torch.bfloat16)
        expected = tensor[0, [1, 2], 0].clone()
        changed = expected + torch.tensor(.25, dtype=torch.bfloat16)
        with legacy.edit_keys(model, (0, 0), [1, 2], changed, expected):
            observed = attention.k_norm(tensor)
            self.assertTrue(torch.equal(observed[0, [1, 2], 0], changed))
            self.assertTrue(torch.equal(observed[:, :, 1:], tensor[:, :, 1:]))
        self.assertEqual(len(attention.k_norm._forward_hooks), 0)
        mask = torch.triu(torch.full((1, 1, 5, 5), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16), diagonal=1)
        with legacy.bias_scores(model, (0, 0), [1, 2], 0):
            observed = attention(tensor, attention_mask=mask)
            self.assertTrue(torch.equal(observed, mask.expand(1, 32, -1, -1)))
        self.assertEqual(len(attention._forward_pre_hooks), 0)
        with legacy.bias_scores(model, (0, 0), [1, 2], -.1):
            observed = attention(tensor, attention_mask=mask)
            self.assertEqual(float(observed[0, 0, 3, 1]), float(torch.tensor(-.1, dtype=torch.bfloat16)))
            self.assertEqual(float(observed[0, 4, 3, 1]), 0)
            self.assertEqual(float(observed[0, 0, 0, 1]), float(mask[0, 0, 0, 1]))
        self.assertEqual(len(attention._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
