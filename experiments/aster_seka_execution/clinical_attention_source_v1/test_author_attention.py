"""Server-only synthetic contracts; no clinical/test records are loaded.

Run with the pinned SEKA checkout and method_agent on PYTHONPATH. These tiny
models verify operator wiring, not 55k-context memory or clinical performance.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method_agent"))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from aster_seka_adapter import Control, PhasePolicy, PreparedPrompt, PredictedSpans, Span
from author_attention import generate_pasta_native, make_adaseka, routing_last_token_logits


PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"


class CharacterTokenizer:
    eos_token_id = 63

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": [[ord(char) % 30 + 1 for char in text]],
                  "attention_mask": [[1] * len(text)]}
        if return_offsets_mapping:
            result["offset_mapping"] = [[(i, i + 1) for i in range(len(text))]]
        return {key: torch.tensor(value) for key, value in result.items()}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def tiny_model(layers=2):
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         eos_token_id=63, pad_token_id=63, attention_dropout=0.)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def prompt(text="abcdefghijk"):
    return PreparedPrompt(text, {"guideline": (0, text)},
                          PredictedSpans("synthetic_contract", (Span("guideline", 1, 3, "R1"),
                                                                Span("guideline", 5, 7, "R2"))),
                          "execution", "path")


class ActualPastaTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tok, self.prompt = tiny_model(), CharacterTokenizer(), prompt()
        self.heads = {0: [0, 2], 1: [1]}

    def test_author_full_mask_matches_chunks_and_cached_generation(self):
        from pastalib.pasta import PASTA
        encoded = self.tok(self.prompt.text, return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        operator = PASTA(self.model, self.tok, head_config=self.heads, alpha=.01, scale_position="exclude")
        with torch.inference_mode(), operator.apply_steering(
                self.model, strings=[self.prompt.text], substrings=[["bc"], ["fg"]],
                model_input=encoded, offsets_mapping=offsets):
            expected = self.model.generate(**encoded, max_new_tokens=3, do_sample=False,
                                           temperature=None, top_p=None, top_k=None, logits_to_keep=1,
                                           return_dict_in_generate=True, output_scores=True,
                                           pad_token_id=self.tok.eos_token_id)
        actual = generate_pasta_native(self.model, self.tok, self.prompt, self.heads, control="normal",
                                       max_new_tokens=3, max_context_tokens=128, query_chunk=2)
        torch.testing.assert_close(actual["first_token_scores"], expected.scores[0][0], rtol=1e-5, atol=1e-6)
        self.assertEqual(actual["output_token_ids"], expected.sequences[0, len(self.prompt.text):].tolist())
        self.assertGreater(actual["author_operator_calls"], 0)
        self.assertEqual(actual["section_token_ranges"], [[1, 3], [5, 7]])
        self.assertEqual(self.model.config._attn_implementation, "eager")

    def test_zero_and_no_hook_use_same_kernel_with_real_author_calls(self):
        results = {name: generate_pasta_native(self.model, self.tok, self.prompt, self.heads,
                                              control=name, max_new_tokens=3,
                                              max_context_tokens=128, query_chunk=2)
                   for name in ["no_hook", "matched_zero"]}
        torch.testing.assert_close(results["no_hook"]["first_token_scores"],
                                   results["matched_zero"]["first_token_scores"], rtol=0, atol=0)
        self.assertEqual(results["no_hook"]["output_token_ids"], results["matched_zero"]["output_token_ids"])
        self.assertEqual(results["no_hook"]["author_operator_calls"], 0)
        self.assertGreater(results["matched_zero"]["author_operator_calls"], 0)
        self.assertEqual(results["matched_zero"]["alpha"], 1.)


class ActualAdaTests(unittest.TestCase):
    def test_actual_four_expert_class_bf16_controls_and_forward_cost(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model_dir = directory / "qwen"
            tiny_model(layers=10).save_pretrained(model_dir)
            backend = Tokenizer(models.WordLevel({"[UNK]": 0, "patient": 1, "fever": 2,
                                                  "rest": 3, "[EOS]": 63}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                eos_token="[EOS]", pad_token="[EOS]")
            tokenizer.save_pretrained(model_dir)
            experts = {}
            for name in ("synthetic", "counterfact", "biasbios", "hotpot"):
                path = directory / (name + ".pt")
                torch.save({"layers": list(range(10)),
                            "U_matrices": torch.eye(8).expand(10, 2, 8, 8).clone(),
                            "singular_values": torch.ones(10, 2, 8)}, path)
                experts[name] = str(path)
            operator, adapter = make_adaseka(str(model_dir), experts, device="cpu", source_commit=PIN)
            text = "patient fever rest"
            prepared = PreparedPrompt(text, {"guideline": (0, text)},
                PredictedSpans("synthetic_contract", (Span("guideline", 8, 13, "R1"),)), "execution", "path")
            self.assertEqual(next(operator.model.parameters()).dtype, torch.bfloat16)
            self.assertEqual(operator.top_k_singular, 5)
            self.assertEqual(len(adapter.projections), 4)
            outputs = {}
            for control in (Control.NO_HOOK, Control.MATCHED_ZERO, Control.NORMAL):
                calls = [0]
                def count(*unused):
                    calls[0] += 1
                handle = operator.model.register_forward_hook(count)
                try:
                    with routing_last_token_logits(operator.model) as routing:
                        outputs[control] = adapter.generate(prepared, PhasePolicy(False, True), control=control,
                                                           max_new_tokens=1, capture_first_token_scores=True)
                finally:
                    handle.remove()
                self.assertEqual(calls[0], 1 if control == Control.NO_HOOK else 2)
                self.assertEqual(routing["routing_forward_calls"], 0 if control == Control.NO_HOOK else 1)
            torch.testing.assert_close(outputs[Control.NO_HOOK]["first_token_scores"],
                                       outputs[Control.MATCHED_ZERO]["first_token_scores"], rtol=0, atol=0)
            self.assertEqual(outputs[Control.MATCHED_ZERO]["gains"]["amplify_factor"], 0.)
            self.assertEqual(outputs[Control.NORMAL]["gains"]["amplify_factor"], .5)

    def test_routing_storage_keeps_all_hidden_states_and_last_logits(self):
        model = tiny_model().to(torch.bfloat16)
        ids = torch.tensor([[2, 7, 11, 5, 19, 3]])
        with torch.inference_mode():
            full = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            with routing_last_token_logits(model) as record:
                reduced = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        self.assertEqual(record["routing_forward_calls"], 1)
        self.assertEqual(reduced.logits.shape[1], 1)
        self.assertEqual(len(full.hidden_states), len(reduced.hidden_states))
        for expected, actual in zip(full.hidden_states, reduced.hidden_states):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(full.logits[:, -1:, :], reduced.logits, rtol=1e-2, atol=1e-3)
        print(json.dumps({"test": "tiny_qwen_bf16_routing_storage", "input_tokens": ids.shape[1],
                          "all_hidden_states_bitwise_equal": True,
                          "last_logits_max_abs_difference": float((full.logits[:, -1:, :].float() - reduced.logits.float()).abs().max()),
                          "scope": "tiny synthetic model only; not full 55k-context validation"}))


if __name__ == "__main__":
    unittest.main()
