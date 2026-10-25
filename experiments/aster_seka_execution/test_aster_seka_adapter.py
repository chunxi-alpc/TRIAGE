"""CPU contract tests with an API double. Run on the server, not locally.

These tests do not establish GPU numerical equivalence or clinical validity.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from aster_seka_adapter import (
    ConditionRequest, ExecutionRequest, PhasePolicy, Control, StageSEKAAdapter,
    prepare_condition, prepare_execution, projection_manifest,
)


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        ids = [ord(c) + 1 for c in text]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                "offset_mapping": torch.tensor([[(i, i + 1) for i in range(len(ids))]])}

    def decode(self, ids, **kwargs):
        return "".join(chr(int(i) - 1) for i in ids if i)


class OfficialAPIDouble:
    """Tracks the actual upstream keyword interface, without executing a model."""
    def __init__(self, path):
        self.pos_pt, self.neg_pt, self.layers = str(path), None, "last2"
        self.amplify_pos, self.amplify_neg = 2.4, 0.0
        self.device = "cpu"
        self.tok = CharacterTokenizer()
        self.model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3", num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=2, num_hidden_layers=3),
            parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]))
        self.calls, self.active, self.raise_on_generate = [], False, False

    def generate(self, *, ids, steer, steer_mask, attention_mask, return_raw, **kwargs):
        self.active = steer
        self.calls.append({"ids": ids.clone(), "mask": steer_mask.clone(), "steer": steer,
                           "g_pos": self.amplify_pos, "g_neg": self.amplify_neg, "budget": kwargs["max_new_tokens"]})
        if self.raise_on_generate:
            raise RuntimeError("Synthetic generation failure")
        tail = torch.tensor([[ord(c) + 1 for c in '["A","B"]']])
        return torch.cat([ids, tail], dim=1)

    def remove_projection(self):
        self.active = False


def condition_payload():
    return {
        "patient_history": "High fever. No cough.",
        "conditions": [{"rule_id": "R1", "condition": "High fever"},
                       {"rule_id": "R2", "condition": "Dry cough!"}],
        "evidence": {
            "selector_fingerprint": "synthetic-public-selector-v1",
            "spans": [{"source": "condition:R1", "start": 0, "end": 10, "rule_id": "R1"}],
            "wrong_spans": [{"source": "condition:R2", "start": 0, "end": 10, "rule_id": "R2"}],
            "misaligned_rule_spans": [{"source": "condition:R2", "start": 0, "end": 10, "rule_id": "R2"}],
        },
    }


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "projection.pt"
        self.proj = torch.eye(2).repeat(2, 2, 1, 1)
        torch.save({"layers": [1, 2], "proj": self.proj}, self.path)
        self.op = OfficialAPIDouble(self.path)
        self.adapter = StageSEKAAdapter(self.op, kind="seka", source_commit="a" * 40,
                                        projection_paths=[str(self.path)])
        self.prompt = prepare_condition(ConditionRequest.from_mapping(condition_payload()))
        self.on = PhasePolicy(True, True)

    def test_schema_rejects_gold_and_action_fields(self):
        for field in ("gold_answer", "gold_path", "actions", "public_options"):
            payload = condition_payload()
            payload[field] = "forbidden"
            with self.assertRaises(ValueError):
                ConditionRequest.from_mapping(payload)
        payload = condition_payload()
        payload["conditions"][0]["gold_rule"] = True
        with self.assertRaises(ValueError):
            ConditionRequest.from_mapping(payload)

    def test_phase_contract_has_exactly_four_cells(self):
        observed = {(p.condition_on, p.execution_on) for p in PhasePolicy.factorial()}
        self.assertEqual(observed, {(False, False), (False, True), (True, False), (True, True)})

    def test_no_hook_and_zero_hook_have_identical_inputs_but_distinct_calls(self):
        results = [self.adapter.generate(self.prompt, self.on, max_new_tokens=16, control=c)
                   for c in (Control.NO_HOOK, Control.MATCHED_ZERO, Control.NORMAL)]
        self.assertEqual(len({r["input_ids_sha256"] for r in results}), 1)
        self.assertEqual(len({r["span_mask_sha256"] for r in results}), 1)
        self.assertFalse(self.op.calls[0]["steer"])
        self.assertTrue(self.op.calls[1]["steer"])
        self.assertEqual(self.op.calls[1]["g_pos"], 0.0)
        self.assertEqual(self.op.calls[2]["g_pos"], 2.4)
        self.assertEqual(self.op.amplify_pos, 2.4)
        self.assertFalse(self.op.active)

    def test_reverse_restores_original_gain(self):
        self.adapter.generate(self.prompt, self.on, max_new_tokens=16, control=Control.REVERSE)
        self.assertEqual(self.op.calls[-1]["g_pos"], -2.4)
        self.assertEqual(self.op.amplify_pos, 2.4)

    def test_wrong_and_misaligned_spans_keep_prompt_and_token_budget(self):
        results = [self.adapter.generate(self.prompt, self.on, max_new_tokens=16, control=c)
                   for c in (Control.NORMAL, Control.WRONG_SPAN, Control.MISALIGNED_RULE)]
        self.assertEqual(len({r["input_ids_sha256"] for r in results}), 1)
        self.assertEqual({r["highlighted_tokens"] for r in results}, {10})
        self.assertNotEqual(results[0]["span_mask_sha256"], results[1]["span_mask_sha256"])

    def test_mask_count_mismatch_is_not_silently_accepted(self):
        payload = condition_payload()
        payload["evidence"]["wrong_spans"][0]["end"] = 5
        prompt = prepare_condition(ConditionRequest.from_mapping(payload))
        with self.assertRaises(ValueError):
            self.adapter.generate(prompt, self.on, max_new_tokens=16, control=Control.WRONG_SPAN)
        self.assertEqual(self.op.calls, [])

    def test_layer_mismatch_is_rejected(self):
        torch.save({"layers": [0, 1], "proj": self.proj}, self.path)
        with self.assertRaises(ValueError):
            StageSEKAAdapter(self.op, kind="seka", source_commit="a" * 40, projection_paths=[str(self.path)])

    def test_within_same_rule_set_misalignment_changes_association_and_mask(self):
        payload = condition_payload()
        payload["evidence"]["spans"] = [
            {"source": "condition:R1", "start": 0, "end": 5, "rule_id": "R1"},
            {"source": "condition:R2", "start": 0, "end": 5, "rule_id": "R2"},
        ]
        payload["evidence"]["misaligned_rule_spans"] = [
            {"source": "condition:R2", "start": 5, "end": 10, "rule_id": "R1"},
            {"source": "condition:R1", "start": 5, "end": 10, "rule_id": "R2"},
        ]
        prompt = prepare_condition(ConditionRequest.from_mapping(payload))
        original = self.adapter.generate(prompt, self.on, max_new_tokens=16)
        shifted = self.adapter.generate(prompt, self.on, max_new_tokens=16, control=Control.MISALIGNED_RULE)
        self.assertEqual(original["input_ids_sha256"], shifted["input_ids_sha256"])
        self.assertNotEqual(original["span_mask_sha256"], shifted["span_mask_sha256"])

    def test_failure_removes_hooks_and_restores_gains(self):
        self.op.raise_on_generate = True
        with self.assertRaises(RuntimeError):
            self.adapter.generate(self.prompt, self.on, max_new_tokens=16, control=Control.MATCHED_ZERO)
        self.assertFalse(self.op.active)
        self.assertEqual(self.op.amplify_pos, 2.4)

    def _adaptive_file(self, malformed=False):
        u = self.proj.clone()
        singular = torch.ones(2, 2, 2)
        if malformed:
            u[0, 1, 0, 0] = float("nan")
        else:
            u[0, 1] = float("nan")
            singular[0, 1] = float("nan")
        torch.save({"layers": [1, 2], "U_matrices": u, "singular_values": singular}, self.path)
        self.op.sel_layers = [1, 2]
        self.op.task_projector = SimpleNamespace(
            expert_names=["synthetic"], expert_U_matrices={"synthetic": u},
            expert_singular_values={"synthetic": singular})

    def test_official_adaptive_whole_head_nan_is_counted_as_filtered(self):
        self._adaptive_file()
        result = projection_manifest(self.op, "adaseka", [str(self.path)])
        self.assertEqual(result[0]["disabled_head_count"], 1)

    def test_adaptive_partial_nan_is_rejected(self):
        self._adaptive_file(malformed=True)
        with self.assertRaises(ValueError):
            projection_manifest(self.op, "adaseka", [str(self.path)])

    def test_no_support_is_explicit_original_context_fallback(self):
        payload = condition_payload()
        payload["evidence"]["spans"] = []
        prompt = prepare_condition(ConditionRequest.from_mapping(payload))
        result = self.adapter.generate(prompt, self.on, max_new_tokens=16)
        self.assertTrue(result["no_support_fallback"])
        self.assertFalse(result["steer"])

    def test_native_multiselect_is_returned_without_argmax(self):
        payload = {
            "patient_history": "A patient has persistent fever.",
            "question": "Which actions apply to this patient?",
            "guideline": "Rule R1: if fever, choose A and B.",
            "predicted_tfu": [{"rule_id": "R1", "true_score": .8, "false_score": .1, "unknown_score": .1}],
            "output_kind": "multi_select",
            "public_options": [{"label": "A", "text": "First action"}, {"label": "B", "text": "Second action"}],
            "evidence": {"selector_fingerprint": "synthetic-public-selector-v1", "spans": []},
        }
        prompt = prepare_execution(ExecutionRequest.from_mapping(payload))
        self.assertIn(payload["guideline"], prompt.text)
        self.assertIn(payload["patient_history"], prompt.text)
        self.assertIn(payload["question"], prompt.text)
        result = self.adapter.generate(prompt, self.on, max_new_tokens=16)
        self.assertEqual(result["text"], '["A","B"]')
        self.assertEqual(result["output_kind"], "multi_select")

    def test_path_contract_does_not_require_choices(self):
        payload = {"patient_history": "A patient has fever.", "question": "Give the decision path.",
                   "guideline": "Start -> condition -> outcome.", "predicted_tfu": [], "output_kind": "path",
                   "evidence": {"selector_fingerprint": "synthetic-public-selector-v1", "spans": []}}
        prompt = prepare_execution(ExecutionRequest.from_mapping(payload))
        self.assertEqual(prompt.output_kind, "path")
        self.assertNotIn("Options:", prompt.text)

    def test_stage_off_does_not_run_hidden_intervention(self):
        result = self.adapter.generate(self.prompt, PhasePolicy(False, True), max_new_tokens=16)
        self.assertFalse(result["steer"])
        with self.assertRaises(ValueError):
            self.adapter.generate(self.prompt, PhasePolicy(False, True), max_new_tokens=16, control=Control.REVERSE)


if __name__ == "__main__":
    unittest.main()
