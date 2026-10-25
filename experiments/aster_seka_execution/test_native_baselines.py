"""Synthetic server-only tests for full-vocabulary caches and chunked Spotlight."""
from pathlib import Path
import json
import sys
import unittest

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "method_agent"))
sys.path.insert(0, str(parent / "baseline_adapters"))

import torch

from baseline_operators import spotlight_generate
from chunked_spotlight import generate_chunked_spotlight
from clinical_core import parse_quoted_rules
from native_generation_baselines import (FusionConfig, cropped_guideline, direct_action_mapping,
                                         generate_native, mix_vocabulary_logits)


class Tokenizer:
    eos_token_id = 63

    def encode(self, text, **kwargs):
        return [ord(c) % 30 + 1 for c in text]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": self.encode(text), "attention_mask": [1] * len(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return {key: torch.tensor([value]) for key, value in result.items()} if return_tensors else result


class FormulaAndMappingTests(unittest.TestCase):
    def test_static_is_complete_vocabulary_probability_mixture(self):
        base = torch.tensor([[1., 2., -1., .5]])
        auxiliary = torch.tensor([[-1., .5, 2., 0.]])
        config = FusionConfig()
        mixed, _ = mix_vocabulary_logits(base, auxiliary, "static", config)
        expected = .25 * base.softmax(-1) + .75 * (auxiliary / .25).softmax(-1)
        torch.testing.assert_close(mixed.exp(), expected)
        self.assertEqual(mixed.shape[-1], 4)
        self.assertTrue((mixed.exp() > 0).all())

    def test_dynamic_margin_is_before_temperature(self):
        base, auxiliary = torch.zeros(1, 4), torch.tensor([[3., 1., 0., -1.]])
        _, values = mix_vocabulary_logits(base, auxiliary, "dynamic", FusionConfig())
        top = auxiliary.softmax(-1).topk(2).values[0]
        expected = .75 + .25 * (2 * float(top[0] - top[1]) - 1)
        self.assertAlmostEqual(values["beta"], expected, places=6)

    def test_fusion_zero_same_temperature_base_distribution(self):
        base, auxiliary = torch.tensor([[1., 3., 0.]]), torch.tensor([[4., -1., 2.]])
        config = FusionConfig(temperature_base=2.)
        zero, _ = mix_vocabulary_logits(base, auxiliary, "dynamic", config, "matched_zero")
        torch.testing.assert_close(zero, (base / 2).log_softmax(-1), rtol=0, atol=0)

    def test_cad_formula_zero_and_reverse(self):
        base, prior = torch.tensor([[1., 3., 0.]]), torch.tensor([[4., -1., 2.]])
        normal, _ = mix_vocabulary_logits(base, prior, "cad", FusionConfig())
        zero, _ = mix_vocabulary_logits(base, prior, "cad", FusionConfig(), "matched_zero")
        reverse, _ = mix_vocabulary_logits(base, prior, "cad", FusionConfig(), "reverse")
        torch.testing.assert_close(normal, 2 * base - prior)
        torch.testing.assert_close(zero, base, rtol=0, atol=0)
        torch.testing.assert_close(reverse, prior, rtol=0, atol=0)

    def _rules(self):
        guideline = "If A, do X. If B, do Y."
        data = [{"condition_quote": f"If {c}", "action_quote": f"do {a}", "exception_quotes": [],
                 "rule_quote": f"If {c}, do {a}."} for c, a in [("A", "X"), ("B", "Y")]]
        return guideline, parse_quoted_rules(json.dumps(data), guideline)

    def test_top_k_uses_predicted_T_probability(self):
        guideline, rules = self._rules()
        tfu = [{"rule_id": "R0001", "label": "F", "status": "ok", "candidate_normalized_probabilities": {"T": .1}},
               {"rule_id": "R0002", "label": "T", "status": "ok", "candidate_normalized_probabilities": {"T": .8}}]
        text, metadata = cropped_guideline(guideline, rules, tfu, "top1")
        self.assertEqual(text, "If B, do Y.")
        self.assertEqual(metadata["selected_rules"], ["R0002"])

    def test_unknown_applicable_crop_keeps_original(self):
        guideline, rules = self._rules()
        text, metadata = cropped_guideline(guideline, rules, [{"rule_id": "R0001", "label": "U"}], "applicable")
        self.assertEqual(text, guideline)
        self.assertEqual(metadata["fallback"], "unknown_conditions_keep_original_context")

    def test_direct_mapping_does_not_invent_multi_rule_path(self):
        _, rules = self._rules()
        output, reason = direct_action_mapping(rules, [{"rule_id": r.rule_id, "label": "T"} for r in rules], "en")
        self.assertIsNone(output)
        self.assertIn("no invented path ordering", reason)

    def test_single_rule_direct_mapping_uses_exact_action(self):
        _, rules = self._rules()
        output, reason = direct_action_mapping(rules, [{"rule_id": "R0001", "label": "T"},
                                                     {"rule_id": "R0002", "label": "F"}], "en")
        self.assertIsNone(reason)
        self.assertTrue(output.endswith("Clinical Decision Outcome: do X"))


class ActualTinyQwenTests(unittest.TestCase):
    def setUp(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=128, eos_token_id=63, pad_token_id=63, attention_dropout=0.)
        config._attn_implementation = "eager"
        self.model, self.tokenizer = Qwen3ForCausalLM(config).eval(), Tokenizer()

    def test_cached_fusion_matches_recomputed_full_prefixes(self):
        primary, auxiliary, expected = "abc", "fghijkl", []
        config = FusionConfig()
        for step in range(4):
            logits = []
            for prompt in (primary, auxiliary):
                ids = torch.tensor([self.tokenizer.encode(prompt) + expected])
                with torch.inference_mode():
                    logits.append(self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :])
            combined, _ = mix_vocabulary_logits(*logits, "static", config)
            expected.append(int(combined.argmax(-1)[0]))
            if expected[-1] == self.tokenizer.eos_token_id:
                break
        actual = generate_native(self.model, self.tokenizer, primary, auxiliary, kind="static",
                                 max_new_tokens=4, max_context_tokens=128)
        self.assertEqual(actual["output_token_ids"], expected)
        self.assertEqual(actual["actual_forward_calls"], 2 * len(expected))

    def test_fusion_zero_matches_one_path_native_suffix(self):
        plain = generate_native(self.model, self.tokenizer, "abcdef", None, kind="plain",
                                max_new_tokens=3, max_context_tokens=128)
        zero = generate_native(self.model, self.tokenizer, "abcdef", "abc", kind="dynamic", control="matched_zero",
                               max_new_tokens=3, max_context_tokens=128)
        self.assertEqual(plain["output_token_ids"], zero["output_token_ids"])
        torch.testing.assert_close(plain["first_log_probs"], zero["first_log_probs"], rtol=0, atol=1e-6)
        self.assertEqual(zero["actual_forward_calls"], 2 * plain["actual_forward_calls"])

    def test_cad_reverse_tracks_prior_cache(self):
        plain = generate_native(self.model, self.tokenizer, "uvwxyz", None, kind="plain",
                                max_new_tokens=3, max_context_tokens=128)
        reverse = generate_native(self.model, self.tokenizer, "ab", "uvwxyz", kind="cad", control="reverse",
                                  max_new_tokens=3, max_context_tokens=128)
        self.assertEqual(plain["output_token_ids"], reverse["output_token_ids"])

    def test_chunked_spotlight_matches_full_query_formula(self):
        full = spotlight_generate(self.model, self.tokenizer, "abcdefg", [(1, 3)], target=.5, max_new_tokens=2)
        chunk = generate_chunked_spotlight(self.model, self.tokenizer, "abcdefg", [(1, 3)], target=.5,
                                           query_chunk=2, max_new_tokens=2, max_context_tokens=128)
        torch.testing.assert_close(full.first_token_scores.log_softmax(-1), chunk["first_log_probs"], rtol=0, atol=1e-6)
        self.assertEqual(full.output_token_ids, chunk["output_token_ids"])

    def test_chunked_zero_matches_same_kernel_no_steering(self):
        kwargs = {"query_chunk": 2, "max_new_tokens": 3, "max_context_tokens": 128}
        plain = generate_chunked_spotlight(self.model, self.tokenizer, "abcdefg", [(1, 3)], control="no_hook", **kwargs)
        zero = generate_chunked_spotlight(self.model, self.tokenizer, "abcdefg", [(1, 3)], control="matched_zero", **kwargs)
        torch.testing.assert_close(plain["first_log_probs"], zero["first_log_probs"], rtol=0, atol=1e-7)
        self.assertEqual(plain["output_token_ids"], zero["output_token_ids"])
        self.assertEqual(self.model.config._attn_implementation, "eager")
        self.assertLessEqual(zero["largest_score_matrix_elements"], 4 * 2 * (7 + 3))

    def test_chunked_no_steering_matches_stock_eager_small_input(self):
        stock = spotlight_generate(self.model, self.tokenizer, "abcdefg", [(1, 3)], control="no_hook", max_new_tokens=2)
        chunk = generate_chunked_spotlight(self.model, self.tokenizer, "abcdefg", [(1, 3)], control="no_hook",
                                           query_chunk=2, max_new_tokens=2, max_context_tokens=128)
        torch.testing.assert_close(stock.first_token_scores.log_softmax(-1), chunk["first_log_probs"], rtol=0, atol=1e-6)
        self.assertEqual(stock.output_token_ids, chunk["output_token_ids"])

    def test_chunked_no_support_is_explicit(self):
        output = generate_chunked_spotlight(self.model, self.tokenizer, "abcdefg", [],
                                            query_chunk=2, max_new_tokens=2, max_context_tokens=128)
        self.assertTrue(output["no_support_fallback"])
        self.assertIsNone(output["rows_below_target"])


if __name__ == "__main__":
    unittest.main()
