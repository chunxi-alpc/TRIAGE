"""Architecture/numerical tests; random weights are never experimental evidence."""
import importlib.util
import unittest


HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class CharacterTokenizer:
    """Deterministic exact-offset tokenizer for tiny architecture tests only."""
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 128 for c in text]

    def __call__(self, text, **kwargs):
        import torch
        ids = torch.tensor([self.encode(text)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "offset_mapping": torch.tensor([[[i, i + 1] for i in range(len(text))]])}


@unittest.skipUnless(HAS_MODEL_DEPS, "torch/transformers unavailable")
class Qwen3AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(91)
        torch.set_num_threads(1)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=256, attention_dropout=0.0)
        cfg._attn_implementation = "eager"
        cls.model = Qwen3ForCausalLM(cfg).eval()
        cls.tokenizer = CharacterTokenizer()
        cls.prompt = "P: Y\nCondition Y. Rule A. Action X.\nAnswer:"
        p = cls.prompt.index("X")
        cls.spans = {"action": [[p, p+1]], "rules": {"A": {"action": [[p, p+1]]}}}

    def test_rotary_transpose_matches_actual_query_key_score(self):
        import torch
        from src.module2.extraction import rotate_half
        torch.manual_seed(17)
        q, d = torch.randn(4, 8), torch.randn(8)
        angle = torch.randn(3, 4).repeat(1, 2)
        co, si = angle.cos(), angle.sin()
        rd = d[None] * co + rotate_half(d)[None] * si
        direct = torch.einsum("hd,td->ht", q, rd).mean()
        effective = (q[:, None] * co[None] - rotate_half(q)[:, None] * si[None]).mean((0,1))
        torch.testing.assert_close(effective @ d, direct)

    def test_zero_edit_matches_full_forward_and_does_not_mutate_prefix(self):
        import torch
        from src.module3.applicability_shift import prepare_prefix, decision_logits
        state = prepare_prefix(self.model, self.tokenizer, self.prompt, self.spans)
        original = [(k.clone(), v.clone()) for k,v in state.cache.to_legacy_cache()]
        ids = self.tokenizer(self.prompt)["input_ids"]
        with torch.inference_mode():
            full = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
        zero = decision_logits(self.model, state, shifts=[
            {"layer": 0, "kv_head": 1, "delta": torch.ones(8), "coefficient": 0.0}])
        torch.testing.assert_close(zero, full, atol=2e-7, rtol=2e-5)
        for (k,v),(saved_k,saved_v) in zip(state.cache.to_legacy_cache(),original):
            torch.testing.assert_close(k,saved_k,rtol=0,atol=0)
            torch.testing.assert_close(v,saved_v,rtol=0,atol=0)
        self.assertEqual(state.cache.get_seq_length(), state.prefix_length)

    def test_edit_changes_first_answer_logits_and_only_targeted_prefix_keys(self):
        import torch
        from src.module3.applicability_shift import prepare_prefix, decision_logits, edited_cache
        state = prepare_prefix(self.model, self.tokenizer, self.prompt, self.spans)
        shift = {"layer": 0, "kv_head": 1, "delta": torch.arange(1, 9).float(), "coefficient": 5.0}
        base = decision_logits(self.model, state)
        changed = decision_logits(self.model, state, shifts=[shift])
        self.assertGreater(float((changed-base).abs().max()), 1e-6)
        new = edited_cache(state, [shift])
        for layer, ((old_k,old_v),(new_k,new_v)) in enumerate(zip(
                state.cache.to_legacy_cache(),new.to_legacy_cache())):
            torch.testing.assert_close(old_v,new_v,atol=0,rtol=0)
            diff = old_k != new_k
            allowed = torch.zeros_like(diff)
            if layer == 0:
                allowed[0,1,state.target_positions] = True
            self.assertFalse(bool((diff & ~allowed).any()))

    def test_score_bias_covers_gqa_queries_and_has_zero_identity(self):
        import torch
        from src.module3.applicability_shift import prepare_prefix, decision_logits
        state = prepare_prefix(self.model, self.tokenizer, self.prompt, self.spans)
        base = decision_logits(self.model,state)
        zero = decision_logits(self.model,state,score_bias=[{"layer":0,"kv_head":1,"bias":0.0}])
        torch.testing.assert_close(zero,base,rtol=0,atol=0)
        changed = decision_logits(self.model,state,score_bias=[{"layer":0,"kv_head":1,"bias":4.0}])
        self.assertGreater(float((changed-base).abs().max()),1e-6)

    def test_sdpa_zero_bias_matches_unedited_decision(self):
        import torch
        from src.module3.applicability_shift import prepare_prefix, decision_logits
        previous = self.model.config._attn_implementation
        self.model.config._attn_implementation = "sdpa"
        try:
            state = prepare_prefix(self.model,self.tokenizer,self.prompt,self.spans)
            base = decision_logits(self.model,state)
            zero = decision_logits(self.model,state,score_bias=[{"layer":0,"kv_head":1,"bias":0.0}])
            torch.testing.assert_close(zero,base,rtol=1e-5,atol=1e-7)
            changed = decision_logits(self.model,state,score_bias=[{"layer":0,"kv_head":1,"bias":4.0}])
            self.assertGreater(float((changed-base).abs().max()),1e-6)
        finally:
            self.model.config._attn_implementation = previous

    def test_actual_attention_mass_obeys_constant_bias_odds(self):
        import math
        import torch
        from src.module3.applicability_shift import prepare_prefix, decision_logits
        state = prepare_prefix(self.model,self.tokenizer,self.prompt,self.spans)
        target = {"layer":0,"kv_head":1,"rule_id":"A"}
        base = decision_logits(self.model,state,return_diagnostics=True,diagnostic_targets=[target])
        shifted = decision_logits(self.model,state,score_bias=[{**target,"bias":4.0}],
                                  return_diagnostics=True,diagnostic_targets=[target])
        torch.testing.assert_close(base["logits"],decision_logits(self.model,state),atol=0,rtol=0)
        a,b = base["attention_mass"][0],shifted["attention_mass"][0]
        self.assertEqual(a["query_count"],2)
        self.assertGreater(b["target_attention_mass"],a["target_attention_mass"])
        for before,after in zip(a["per_query_mass"],b["per_query_mass"]):
            expected = math.exp(4)*before / (1+(math.exp(4)-1)*before)
            self.assertAlmostEqual(after,expected,places=6)

    def test_extraction_checks_alignment_and_removes_hooks(self):
        from src.module2.extraction import extract_triplets
        row = {"id":"pair", "tree_id":"tree", "group_id":"group", "pair_cluster_id":"cluster",
               "source_split":"train_val", "split":"fit", "target_rule_id":"A",
               "prompts":{"positive":self.prompt,"negative":self.prompt.replace("P: Y","P: N"),
                          "neutral":self.prompt.replace("P: Y","P: ?")},
               "spans":{v:self.spans for v in ("positive","negative","neutral")}}
        result = extract_triplets(self.model,self.tokenizer,[row,row])
        self.assertEqual(result["keys"]["positive"].shape,(2,2,2,8))
        self.assertEqual(result["forward_count"],3)
        self.assertTrue(result["audit"][0]["neutral_aligned"])
        for layer in self.model.model.layers:
            self.assertEqual(len(layer.self_attn.k_norm._forward_hooks),0)
            self.assertEqual(len(layer.self_attn.q_norm._forward_hooks),0)
        row["prompts"]["negative"] = "!" + row["prompts"]["negative"]
        with self.assertRaises(ValueError):
            extract_triplets(self.model,self.tokenizer,[row])

    def test_invalid_target_and_non_qwen_fail_closed(self):
        import torch
        from types import SimpleNamespace
        from src.module2.extraction import validate_qwen3
        from src.module3.applicability_shift import prepare_prefix, edited_cache
        with self.assertRaises(ValueError):
            validate_qwen3(SimpleNamespace(config=SimpleNamespace(model_type="llama")))
        state = prepare_prefix(self.model,self.tokenizer,self.prompt,self.spans)
        with self.assertRaises(ValueError):
            edited_cache(state,[{"layer":0,"kv_head":0,"delta":torch.ones(8),"positions":[0]}])

    def test_fixed_query_score_includes_all_gqa_queries_and_sign_reversal(self):
        import torch
        from src.module3.applicability_shift import prepare_prefix, measure_fixed_query_shifts
        state = prepare_prefix(self.model,self.tokenizer,self.prompt,self.spans)
        common = {"layer":1,"kv_head":0,"delta":torch.arange(1,9).float()}
        result = measure_fixed_query_shifts(self.model,state,[
            {**common,"coefficient":1.0},{**common,"coefficient":-1.0}])
        positive,negative = result["shifts"]
        self.assertEqual(positive["score_count"],2 * len(state.target_positions))
        self.assertAlmostEqual(positive["mean"],-negative["mean"],places=6)
        self.assertAlmostEqual(positive["rms"],negative["rms"],places=6)
        self.assertAlmostEqual(positive["rms"],positive["analytical_rms"],places=6)


if __name__ == "__main__":
    unittest.main()
