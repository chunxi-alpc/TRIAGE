"""Synthetic tests to execute on the server; no research examples or GPU required."""
from types import SimpleNamespace
import unittest

import torch

from baseline_operators import (
    SpotlightQwen, cad_first_token, cad_scores, encode_spans,
    spotlight_bias_logits, spotlight_generate, wrong_span_mask,
)
from run_counterfact_baselines import prompts_from_row


class CharTokenizer:
    eos_token_id = 63

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {
            "input_ids": torch.tensor([[ord(c) % 30 + 1 for c in text]]),
            "attention_mask": torch.ones(1, len(text), dtype=torch.long),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor([[(i, i + 1) for i in range(len(text))]])
        return result

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)


class FakeGenerationModel:
    training = False
    device = torch.device("cpu")

    def __init__(self, changed_processor=False):
        self.calls = []
        self.changed_processor = changed_processor
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs["input_ids"].shape[-1]
        raw = torch.tensor([[0., float(n), -1.]])
        for hook in self.hooks:
            hook(self, (), SimpleNamespace(logits=raw[:, None, :]))
        if self.changed_processor:
            raw.add_(1)  # In-place processors must not fool the raw-score guard.
        scores = raw.clone()
        return SimpleNamespace(logits=(raw,), scores=(scores,))


class AlgebraTests(unittest.TestCase):
    def test_cad_exact_formula_no_filter(self):
        context, prior = torch.tensor([1., -2., .4]), torch.tensor([-3., 2., .6])
        torch.testing.assert_close(cad_scores(context, prior, 1), 2 * context - prior)
        torch.testing.assert_close(cad_scores(context, prior, 0), context, rtol=0, atol=0)
        torch.testing.assert_close(cad_scores(context, prior, -1), prior, rtol=0, atol=0)

    def test_cad_matched_zero_still_two_generate_calls(self):
        model, tokenizer = FakeGenerationModel(), CharTokenizer()
        result = cad_first_token(model, tokenizer, "abcd", "a", control="matched_zero")
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["max_new_tokens"] == 1 for call in model.calls))
        self.assertEqual(result.metadata["forward_calls"], 2)
        torch.testing.assert_close(result.first_token_scores, torch.tensor([0., 4., -1.]))

    def test_cad_no_hook_one_generate_call(self):
        model = FakeGenerationModel()
        result = cad_first_token(model, CharTokenizer(), "abcd", "a", control="no_hook")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.metadata["output_tokens"], 1)

    def test_cad_nonidentity_processor_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Nonidentity generation processor"):
            cad_first_token(FakeGenerationModel(True), CharTokenizer(), "abcd", "a")

    def test_spotlight_log_ratio_not_log_odds(self):
        logits = torch.tensor([[[[.02, .98]]]]).log()
        selected = torch.tensor([[[[True, False]]]])
        edited, active = spotlight_bias_logits(logits, selected, target=.1, gain=1.)
        self.assertTrue(active.item())
        # e^bias = target/current; the normalized mass is .1/(.98+.1), not .1.
        self.assertAlmostEqual(float(edited.softmax(-1)[0, 0, 0, 0]), .1 / 1.08, places=6)

    def test_spotlight_zero_and_reverse(self):
        logits = torch.tensor([[[[-4., 0.]]]])
        selected = torch.tensor([[[[True, False]]]])
        zero, _ = spotlight_bias_logits(logits, selected, target=.1, gain=0.)
        reverse, _ = spotlight_bias_logits(logits, selected, target=.1, gain=-1.)
        torch.testing.assert_close(zero, logits, rtol=0, atol=0)
        self.assertLess(float(reverse.softmax(-1)[0, 0, 0, 0]), float(logits.softmax(-1)[0, 0, 0, 0]))

    def test_spotlight_above_target_unchanged(self):
        logits = torch.zeros(1, 1, 1, 2)
        selected = torch.tensor([[[[True, False]]]])
        output, active = spotlight_bias_logits(logits, selected, target=.1, gain=1.)
        self.assertFalse(active.item())
        torch.testing.assert_close(output, logits, rtol=0, atol=0)

    def test_spotlight_per_head_and_per_query(self):
        logits = torch.tensor([[[[-4., 0.], [0., 0.]], [[0., -4.], [-6., 0.]]]])
        output, active = spotlight_bias_logits(logits, torch.tensor([[[[True, False]]]]), target=.1, gain=1.)
        self.assertEqual(active.tolist(), [[[True, False], [False, True]]])
        self.assertEqual(float(output[0, 0, 1, 0]), 0.)

    def test_spotlight_no_causally_visible_span_stays_masked(self):
        logits = torch.tensor([[[[0., torch.finfo(torch.float32).min]]]])
        selected = torch.tensor([[[[False, True]]]])
        output, active = spotlight_bias_logits(logits, selected, target=.1, gain=1.,
                                              legal=torch.tensor([[[[True, False]]]]))
        self.assertFalse(active.item())
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, logits, rtol=0, atol=0)

    def test_equal_count_disjoint_wrong_mask(self):
        original = torch.tensor([[False, True, True, False, False]])
        wrong = wrong_span_mask(original, torch.ones_like(original))
        self.assertEqual(int(wrong.sum()), int(original.sum()))
        self.assertFalse((wrong & original).any())
        self.assertEqual(wrong.tolist(), [[True, False, False, True, False]])

    def test_encode_original_text_no_marker_insertion(self):
        tokenizer = CharTokenizer()
        inputs, mask, visible = encode_spans(tokenizer, "abcde", [(1, 3)])
        self.assertTrue(torch.equal(inputs["input_ids"], tokenizer("abcde")["input_ids"]))
        self.assertEqual(mask.tolist(), [[False, True, True, False, False]])
        self.assertTrue(visible.all())

    def test_official_prompt_and_prior_split(self):
        row = {"context": "A lives in B", "prompt": "A lives in B. Where does A live?",
               "target_mediated": "B", "target_unmediated": "C", "attribute": "lives in B"}
        full, prior, spans = prompts_from_row(row)
        self.assertEqual(full, "Previously A lives in C. Currently A lives in B. Where does A live?")
        self.assertEqual(prior, "Where does A live?")
        self.assertEqual([full[a:b] for a, b in spans], ["lives in B"])


class TinyActualQwenTests(unittest.TestCase):
    """Exercise real Qwen attention/RoPE/cache integration on synthetic CPU tensors."""
    def setUp(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128, eos_token_id=63,
                             pad_token_id=63, attention_dropout=0.)
        config._attn_implementation = "eager"
        self.model = Qwen3ForCausalLM(config).eval()
        self.tokenizer = CharTokenizer()

    def test_actual_qwen_zero_equals_eager_with_cache(self):
        plain = spotlight_generate(self.model, self.tokenizer, "abcde", [(1, 3)],
                                   control="no_hook", max_new_tokens=2)
        zero = spotlight_generate(self.model, self.tokenizer, "abcde", [(1, 3)],
                                  control="matched_zero", max_new_tokens=2)
        torch.testing.assert_close(plain.first_token_scores, zero.first_token_scores, rtol=0, atol=1e-7)
        self.assertEqual(plain.output_token_ids, zero.output_token_ids)
        self.assertEqual(zero.metadata["layers_touched"], [0, 1])
        self.assertEqual(zero.metadata["attention_calls"], 2 * zero.metadata["output_tokens"])
        self.assertEqual(self.model.config._attn_implementation, "eager")

    def test_context_restores_registry_on_exception(self):
        from transformers.models.qwen3 import modeling_qwen3
        operator = SpotlightQwen(self.model, torch.tensor([[True, False]]))
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with operator:
                raise RuntimeError("synthetic")
        self.assertEqual(self.model.config._attn_implementation, "eager")
        self.assertNotIn(operator.name, modeling_qwen3.ALL_ATTENTION_FUNCTIONS)

    def test_normal_wrong_and_reverse_use_actual_attention(self):
        for control in ["normal", "wrong_span", "reverse"]:
            result = spotlight_generate(self.model, self.tokenizer, "abcde", [(1, 2)],
                                        target=.5, control=control)
            self.assertTrue(torch.isfinite(result.first_token_scores).all())
            self.assertEqual(result.metadata["attention_calls"], 2)


if __name__ == "__main__":
    unittest.main()
