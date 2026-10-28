"""Architecture checks only: random tiny weights are not research evidence."""
import importlib.util
import unittest


HAS_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_DEPS, "torch/transformers unavailable")
class GuidelinePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(261028)
        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=128, attention_dropout=0.)
        config._attn_implementation = "eager"
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25]])
        cls.rule_positions = {"A": [2, 3], "B": [6, 7]}

    def prepare(self):
        from src.module3.guideline_prefix import prepare_guideline_prefix
        return prepare_guideline_prefix(self.model, self.ids, 8, self.rule_positions)

    def test_full_vs_split_zero_identity_and_nonmutation(self):
        import torch
        from src.module3.guideline_prefix import run_guideline_suffix
        prefix = self.prepare()
        original = [(k.clone(), v.clone()) for k, v in prefix.cache.to_legacy_cache()]
        with torch.inference_mode():
            full = self.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids),
                              use_cache=False, logits_to_keep=1).logits[0, -1].float()
        baseline = run_guideline_suffix(self.model, prefix, self.ids)
        zero = run_guideline_suffix(self.model, prefix, self.ids, shifts=[
            {"layer": 1, "kv_head": 1, "rule_id": "A", "delta": torch.ones(8), "coefficient": 0.}])
        torch.testing.assert_close(full, baseline["logits"], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(zero["logits"], baseline["logits"], rtol=0, atol=0)
        self.assertEqual(baseline["suffix_length"], 5)
        for (old_k, old_v), (key, value) in zip(original, prefix.cache.to_legacy_cache()):
            torch.testing.assert_close(old_k, key, rtol=0, atol=0)
            torch.testing.assert_close(old_v, value, rtol=0, atol=0)
        self.assertEqual(prefix.cache.get_seq_length(), 8)

    def test_rotated_multirule_multihead_edits_match_requested_support(self):
        import torch
        from src.module2.extraction import rotate_half
        from src.module3.guideline_prefix import edited_guideline_cache
        prefix = self.prepare()
        delta = torch.arange(1, 9).float() / 8
        shifts = [
            {"layer": 0, "kv_head": 0, "rule_id": "A", "delta": delta, "coefficient": 2.},
            {"layer": 0, "kv_head": 0, "rule_id": "B", "delta": delta, "coefficient": -2.},
            {"layer": 2, "kv_head": 1, "rule_id": "A", "delta": delta, "coefficient": 1.},
        ]
        edited = edited_guideline_cache(self.model, prefix, shifts)
        for layer_index, ((original_k, original_v), (key, value)) in enumerate(zip(
                prefix.cache.to_legacy_cache(), edited.to_legacy_cache())):
            expected = original_k.clone()
            for shift in shifts:
                if shift["layer"] == layer_index:
                    pos = self.rule_positions[shift["rule_id"]]
                    change = shift["coefficient"] * (delta[None] * prefix.cos[pos]
                             + rotate_half(delta)[None] * prefix.sin[pos])
                    expected[0, shift["kv_head"], pos] += change
            torch.testing.assert_close(expected, key, rtol=0, atol=0)
            torch.testing.assert_close(original_v, value, rtol=0, atol=0)
        self.assertEqual(prefix.cache.get_seq_length(), 8)

    def test_direct_bias_all_suffix_queries_and_gqa_mass_law(self):
        import math
        import torch
        from src.module3.guideline_prefix import run_guideline_suffix
        prefix = self.prepare()
        targets = [{"layer": 0, "kv_head": 1, "rule_id": r} for r in ("A", "B")]
        all_query_weights = []
        handle = self.model.model.layers[0].self_attn.register_forward_hook(
            lambda module, args, output: all_query_weights.append(output[1].detach().clone()))
        try:
            baseline = run_guideline_suffix(self.model, prefix, self.ids, diagnostic_targets=targets)
            value = 1.3
            edited = run_guideline_suffix(self.model, prefix, self.ids,
                score_bias=[{**targets[0], "bias": value}], diagnostic_targets=targets)
            zero = run_guideline_suffix(self.model, prefix, self.ids,
                score_bias=[{**targets[0], "bias": 0.}], diagnostic_targets=targets)
        finally:
            handle.remove()
        torch.testing.assert_close(zero["logits"], baseline["logits"], rtol=0, atol=0)
        self.assertGreater(float((edited["logits"] - baseline["logits"]).abs().max()), 1e-6)
        for old, new in zip(baseline["attention_mass"][0]["per_query_mass"],
                            edited["attention_mass"][0]["per_query_mass"]):
            expected = math.exp(value) * old / (1 + (math.exp(value) - 1) * old)
            self.assertAlmostEqual(new, expected, places=6)
        self.assertEqual(len(edited["attention_mass"][0]["per_query_mass"]), 2)
        # Verify all five suffix queries, not just the final diagnostic query.
        old = all_query_weights[0][0, 2:4, :, 2:4].sum(-1)
        new = all_query_weights[1][0, 2:4, :, 2:4].sum(-1)
        expected = math.exp(value) * old / (1 + (math.exp(value) - 1) * old)
        torch.testing.assert_close(new, expected, rtol=2e-6, atol=2e-7)
        # An unedited sibling changes through the same softmax denominator.
        for i, (old, new) in enumerate(zip(baseline["attention_mass"][1]["per_query_mass"],
                                          edited["attention_mass"][1]["per_query_mass"])):
            target_old = baseline["attention_mass"][0]["per_query_mass"][i]
            self.assertAlmostEqual(new, old / (1 + (math.exp(value) - 1) * target_old), places=6)

    def test_qk_response_predicts_actual_two_rule_softmax_change(self):
        import numpy as np
        import torch
        from src.module3.guideline_prefix import run_guideline_suffix
        prefix = self.prepare()
        delta = torch.tensor([.3, -.2, .7, -.1, .4, .8, -.6, .1])
        targets = [{"layer": 1, "kv_head": 0, "rule_id": r, "delta": delta} for r in ("A", "B")]
        baseline = run_guideline_suffix(self.model, prefix, self.ids,
            diagnostic_targets=targets, response_targets=targets)
        gains = [.8, -.4]
        edited = run_guideline_suffix(self.model, prefix, self.ids,
            shifts=[dict(t, coefficient=g) for t, g in zip(targets, gains)],
            diagnostic_targets=targets, response_targets=targets)
        # No lower layer was edited, so last-query vectors at layer 1 are fixed.
        for before, after in zip(baseline["query_response"], edited["query_response"]):
            np.testing.assert_allclose(before["scores"], after["scores"], atol=0, rtol=0)
        for query in range(2):
            factors, old_probabilities = [], []
            for target, gain in enumerate(gains):
                scores = np.array(baseline["query_response"][target]["scores"])[query]
                old = np.array(baseline["attention_mass"][target]["per_query_token_mass"])[query]
                factors.append(np.exp(gain * scores))
                old_probabilities.append(old)
            denominator = 1 + sum(np.sum(a * (f - 1)) for a, f in zip(old_probabilities, factors))
            for target in range(2):
                expected = np.sum(old_probabilities[target] * factors[target]) / denominator
                actual = edited["attention_mass"][target]["per_query_mass"][query]
                self.assertAlmostEqual(expected, actual, places=6)

    def test_suffix_patient_changes_reuse_same_guideline_and_restore_hooks(self):
        import torch
        from src.module3.guideline_prefix import run_guideline_suffix
        prefix = self.prepare()
        changed_ids = self.ids.clone()
        changed_ids[0, 9] = 33
        target = {"layer": 0, "kv_head": 0, "rule_id": "B", "delta": torch.ones(8)}
        original = run_guideline_suffix(self.model, prefix, self.ids)
        changed = run_guideline_suffix(self.model, prefix, changed_ids,
            shifts=[dict(target, coefficient=2.)], diagnostic_targets=[target], response_targets=[target])
        repeated = run_guideline_suffix(self.model, prefix, self.ids)
        self.assertGreater(float((changed["logits"] - original["logits"]).abs().max()), 1e-6)
        torch.testing.assert_close(repeated["logits"], original["logits"], atol=0, rtol=0)
        for layer in self.model.model.layers:
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.q_norm._forward_hooks), 0)

    def test_invalid_boundaries_targets_and_overlapping_edits_fail(self):
        import torch
        from src.module3.guideline_prefix import prepare_guideline_prefix, run_guideline_suffix
        with self.assertRaises(ValueError):
            prepare_guideline_prefix(self.model, self.ids, 13, self.rule_positions)
        with self.assertRaises(ValueError):
            prepare_guideline_prefix(self.model, self.ids, 8, {"A": [2, 8]})
        with self.assertRaises(ValueError):
            prepare_guideline_prefix(self.model, self.ids, 8, {"A": [2], "B": [2]})
        prefix = self.prepare()
        wrong_ids = self.ids.clone()
        wrong_ids[0, 0] = 40
        with self.assertRaises(ValueError):
            run_guideline_suffix(self.model, prefix, wrong_ids)
        target = {"layer": 0, "kv_head": 0, "positions": [3], "delta": torch.ones(8)}
        with self.assertRaises(ValueError):
            run_guideline_suffix(self.model, prefix, self.ids, shifts=[target, target])
        with self.assertRaises(ValueError):
            run_guideline_suffix(self.model, prefix, self.ids, shifts=[dict(target, positions=[9])])
        with self.assertRaises(ValueError):
            run_guideline_suffix(self.model, prefix, self.ids, shifts=[dict(target, coefficient=float("nan"))])
        for layer in self.model.model.layers:
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)

    def test_full_and_tail_match_all_multirule_multilayer_conditions(self):
        import numpy as np
        import torch
        from src.module3.guideline_prefix import (prepare_guideline_tail,
            run_guideline_suffix, run_guideline_tail)
        prefix = self.prepare()
        delta = torch.tensor([.3, -.2, .7, -.1, .4, .8, -.6, .1])
        targets = [{"layer": layer, "kv_head": head, "rule_id": rule, "delta": delta}
                   for layer, head in ((1, 0), (1, 1), (2, 1)) for rule in ("A", "B")]
        tail = prepare_guideline_tail(self.model, prefix, self.ids, 1,
                                     diagnostic_targets=targets, response_targets=targets)
        self.assertEqual(tuple(tail.hidden.shape[:2]), (1, 5))
        conditions = [
            {},
            {"shifts": [dict(targets[0], coefficient=0.)]},
            {"score_bias": [dict(targets[0], bias=0.)]},
            {"shifts": [dict(targets[0], coefficient=4.)]},
            {"shifts": [dict(targets[0], coefficient=-4.)]},
            {"shifts": [dict(targets[0], coefficient=2.), dict(targets[1], coefficient=-1.)]},
            {"shifts": [dict(t, coefficient=(i + 1) * (-1)**i) for i, t in enumerate(targets)]},
            {"score_bias": [dict(t, bias=(i + 1) * .2 * (-1)**i) for i, t in enumerate(targets)]},
            {"shifts": [dict(targets[0], coefficient=1.)],
             "score_bias": [dict(targets[5], bias=-.3)]},
        ]
        for condition in conditions:
            full = run_guideline_suffix(self.model, prefix, self.ids, **condition,
                                        diagnostic_targets=targets, response_targets=targets)
            actual = run_guideline_tail(self.model, tail, **condition,
                                       diagnostic_targets=targets, response_targets=targets)
            torch.testing.assert_close(full["logits"], actual["logits"], rtol=0, atol=0)
            for before, after in zip(full["attention_mass"], actual["attention_mass"]):
                np.testing.assert_allclose(before["per_query_token_mass"], after["per_query_token_mass"],
                                           atol=0, rtol=0)
            for before, after in zip(full["query_response"], actual["query_response"]):
                np.testing.assert_allclose(before["scores"], after["scores"], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            run_guideline_tail(self.model, tail, shifts=[dict(targets[0], layer=0)])
        self.assertEqual(prefix.cache.get_seq_length(), 8)
        for layer in self.model.model.layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
