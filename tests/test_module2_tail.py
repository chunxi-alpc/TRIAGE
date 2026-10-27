"""Execution equivalence tests on tiny random weights, not research evidence."""
import importlib.util
import unittest

HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_MODEL_DEPS, "torch/transformers unavailable")
class TailExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from tests.test_module2_adapter import CharacterTokenizer
        from src.module3.applicability_shift import prepare_prefix
        from src.module3.applicability_tail import prepare_tail
        torch.set_num_threads(1)
        torch.manual_seed(616)
        cfg = Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=48,
                         num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                         head_dim=8,max_position_embeddings=256,attention_dropout=0.)
        cfg._attn_implementation="eager"
        cls.model=Qwen3ForCausalLM(cfg).eval()
        prompt="Patient Y. Condition Y. Rule A. Action X. Answer:"
        p=prompt.index("X")
        spans={"action":[[p,p+1]],"rules":{"A":{"action":[[p,p+1]]}}}
        cls.state=prepare_prefix(cls.model,CharacterTokenizer(),prompt,spans)
        cls.targets=[{"layer":2,"kv_head":1,"positions":cls.state.target_positions,"rule_id":"A"}]
        cls.tail=prepare_tail(cls.model,cls.state,2,1,cls.targets)
        cls.vector=torch.arange(1,9).float()/10

    def _reference(self,condition):
        from src.module3.applicability_shift import decision_logits
        return decision_logits(self.model,self.state,**condition,return_diagnostics=True,
                               diagnostic_targets=self.targets)

    def _optimized(self,condition):
        from src.module3.applicability_tail import tail_decision_logits
        return tail_decision_logits(self.model,self.tail,**condition,return_diagnostics=True,
                                    diagnostic_targets=self.targets)

    def test_single_condition_full_vocabulary_and_attention_are_bitwise_equal(self):
        from src.module3.applicability_tail import compare_results
        cases=[{}, {"shifts":[{"layer":2,"kv_head":1,"delta":self.vector,"coefficient":0.}]},
               {"score_bias":[{"layer":2,"kv_head":1,"bias":0.}]}]
        for sign in (1.,-1.):
            cases.append({"shifts":[{"layer":2,"kv_head":1,"delta":self.vector,"coefficient":sign}]})
            cases.append({"score_bias":[{"layer":2,"kv_head":1,"bias":sign*.1}]})
        for case in cases:
            result=compare_results(self._reference(case),self._optimized(case))
            self.assertEqual(result["max_full_vocabulary_logit_error"],0.)
            self.assertEqual(result["max_per_query_attention_mass_error"],0.)

    def test_prefix_tensor_values_remain_unchanged(self):
        import torch
        original=[(k.clone(),v.clone()) for k,v in self.state.cache.to_legacy_cache()]
        self._optimized({"shifts":[{"layer":2,"kv_head":1,"delta":self.vector,"coefficient":3.}]})
        for (k,v),(saved_k,saved_v) in zip(self.state.cache.to_legacy_cache(),original):
            self.assertTrue(torch.equal(k,saved_k))
            self.assertTrue(torch.equal(v,saved_v))

    def test_prefix_aliases_are_real_and_updates_only_rebind_new_cache_objects(self):
        import torch
        from src.module3.applicability_tail import _alias_prefix_cache
        cache=_alias_prefix_cache(self.model,self.state.cache.to_legacy_cache())
        for original,alias in zip(self.state.cache.layers,cache.layers):
            self.assertIsNot(original,alias)
            self.assertEqual(original.keys.data_ptr(),alias.keys.data_ptr())
            self.assertEqual(original.values.data_ptr(),alias.values.data_ptr())
        layer=2
        before=self.state.cache.layers[layer].keys.clone()
        cache.update(torch.zeros_like(before[:,:,:1]),torch.zeros_like(before[:,:,:1]),layer)
        self.assertEqual(cache.layers[layer].get_seq_length(),self.state.prefix_length+1)
        self.assertEqual(self.state.cache.layers[layer].get_seq_length(),self.state.prefix_length)
        self.assertTrue(torch.equal(before,self.state.cache.layers[layer].keys))

    def test_batch_four_matches_independent_reference_with_declared_tolerance(self):
        from src.module3.applicability_tail import tail_decision_batch,compare_results
        cases=[{"shifts":[{"layer":2,"kv_head":1,"delta":self.vector,"coefficient":1.}]},
               {"shifts":[{"layer":2,"kv_head":1,"delta":self.vector,"coefficient":-1.}]},
               {"score_bias":[{"layer":2,"kv_head":1,"bias":.1}]},
               {"score_bias":[{"layer":2,"kv_head":1,"bias":-.1}]}]
        results=tail_decision_batch(self.model,self.tail,cases,return_diagnostics=True,
                                    diagnostic_targets=self.targets)
        for case,result in zip(cases,results):
            compare_results(self._reference(case),result,logit_atol=1e-6,attention_atol=1e-7)

    def test_zero_conditions_cannot_use_batched_approximate_path(self):
        from src.module3.applicability_tail import tail_decision_batch
        with self.assertRaises(ValueError):
            tail_decision_batch(self.model,self.tail,[{},{}])

    def test_edits_to_other_layers_or_heads_are_rejected(self):
        with self.assertRaises(ValueError):
            self._optimized({"score_bias":[{"layer":1,"kv_head":1,"bias":.1}]})
        with self.assertRaises(ValueError):
            self._optimized({"score_bias":[{"layer":2,"kv_head":0,"bias":.1}]})

    def test_runner_wrapper_preserves_original_api_and_writes_execution_manifest(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import run_module2_causal_tail_pilot as wrapper
        import run_module2_causal_pilot as canonical
        import src.module3.applicability_shift as standard
        from src.module3.applicability_tail import compare_results
        original_decision, original_prefix = standard.decision_logits, standard.prepare_prefix
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/"run"
            def fake_run():
                destination.mkdir()
                baseline=standard.decision_logits(self.model,self.state,return_diagnostics=True,
                                                  diagnostic_targets=self.targets)
                compare_results(self.tail.baseline,baseline)
                shift={"layer":2,"kv_head":1,"delta":self.vector,"coefficient":1.}
                for _ in range(2):
                    actual=standard.decision_logits(self.model,self.state,shifts=[shift],
                        return_diagnostics=True,diagnostic_targets=self.targets)
                    expected=original_decision(self.model,self.state,shifts=[shift],
                        return_diagnostics=True,diagnostic_targets=self.targets)
                    compare_results(expected,actual)
                standard.measure_fixed_query_shifts(self.model,self.state,[shift])
            with patch.object(canonical,"main",fake_run),patch.object(sys,"argv",[
                    "run_module2_causal_tail_pilot.py","--output",str(destination),"--threads","1"]):
                wrapper.main()
            report=json.loads((destination/"execution_optimization.json").read_text())
            self.assertEqual(report["status"],"completed")
            self.assertEqual(report["batch_size"],1)
            self.assertEqual(report["thread_policy"]["prefill"],4)
            self.assertTrue(all(check["max_full_vocabulary_logit_error"]==0 for check in report["checks"]))
        self.assertIs(standard.decision_logits,original_decision)
        self.assertIs(standard.prepare_prefix,original_prefix)


if __name__=="__main__":
    unittest.main()
