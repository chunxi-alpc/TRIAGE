"""Synthetic CPU contracts. Run on the research server, never open test data."""
from dataclasses import asdict
import json
import unittest

import torch

from medpic_native_adapter import native_input, render_native
from medpic_option_transfer import (ARMS, CONDITION_INSTRUCTION, PROTOCOL, SELECTOR, TransferConfig,
    assessment, attach_controls, chat_prompt, condition_prompt, direct_applicability_set,
    execution_prompt, fingerprint, option_evidence, run_factorial, source_manifest, support_spans)


class SecretTrap(dict):
    def __getitem__(self, key):
        if key not in {"patient_vignette", "question", "options"}:
            raise AssertionError("A non-input field was accessed")
        return super().__getitem__(key)


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize, **kwargs):
        assert messages[0]["role"] == "user"
        assert kwargs["enable_thinking"] is False
        rendered = "<user>" + messages[0]["content"] + "<assistant>"
        return list(map(ord, rendered)) if tokenize else rendered

    def __call__(self, text, **kwargs):
        result = {"input_ids": list(map(ord, text))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class MaskAdapter:
    def __init__(self):
        self.operator = type("Operator", (), {"tok": CharTokenizer()})()

    def _encode(self, prompt, spans):
        intervals = prompt.absolute_spans(spans)
        mask = torch.tensor([[any(start <= i < end for start, end in intervals)
                              for i in range(len(prompt.text))]], dtype=torch.bool)
        return None, None, mask


class TransferContracts(unittest.TestCase):
    def task(self, shared=False):
        options = {"A": "Use alpha treatment", "B": "Use beta treatment", "C": "Use delta treatment"}
        if shared:
            options["C"] = "Reassess alpha treatment"
        return native_input(SecretTrap(patient_vignette="Alpha is present. Beta is absent. A distant observation is retained.",
            question="Which synthetic options apply?", options=options,
            answer="SECRET", guideline="SECRET", benchmark_task_family="SECRET", pair_id="SECRET"))

    def lock(self):
        return {"protocol": PROTOCOL, "frozen": True, "frozen_at": "synthetic-only",
                "seeds": [0, 1, 2], "arms": list(ARMS), "config": asdict(TransferConfig()),
                "selector": SELECTOR, "condition_instruction_sha256": fingerprint(CONDITION_INSTRUCTION),
                "source_hashes": source_manifest()}

    def assessments(self, labels=("T", "U", "F")):
        return [{"option": key, "label": value, "status": "ok", "candidate_normalized_probabilities": None}
                for key, value in zip(("A", "B", "C"), labels)]

    def fake_infer(self, prompt, policy, control, budget, seed):
        label = "T" if "Target option: A" in prompt.text else "U"
        return {"status": "ok", "text": label if prompt.stage == "condition" else 'FINAL_ANSWER: ["A"]',
                "label_logits": {key: 5. if key == label else -5. for key in ("T", "F", "U")},
                "actual_forward_calls": budget, "input_tokens": len(prompt.text), "output_tokens": budget,
                "wall_seconds": 0.25, "peak_allocated_gib": 1., "peak_reserved_gib": 2.}

    def test_method_uses_original_input_and_original_sentence_ranges(self):
        task = self.task()
        evidence = option_evidence(task)
        self.assertTrue(evidence["A"])
        self.assertEqual(evidence["C"], ())
        source = task.patient_vignette
        self.assertTrue(all("Alpha" in source[s.start:s.end] for s in evidence["A"]))
        self.assertTrue(all("absent" in source[s.start:s.end] for s in evidence["B"]))
        prompt = condition_prompt(task, "A", evidence)
        self.assertIn(task.patient_vignette, prompt.text)
        self.assertIn(task.question, prompt.text)
        self.assertTrue(all(value in prompt.text for _, value in task.options))
        self.assertNotIn("SECRET", prompt.text)
        self.assertEqual(set(prompt.fields), {"patient_history"})

    def test_unknown_preserves_full_input_and_shared_ranges_count_once(self):
        task = self.task(shared=True)
        decisions = self.assessments(("T", "U", "T"))
        selected = support_spans(task, decisions)
        self.assertEqual(len(selected), 1)
        prompt = execution_prompt(task, decisions)
        self.assertTrue(prompt.text.startswith(render_native(task).text))
        self.assertIn("unknown is not false", prompt.text)
        self.assertIn("Beta is absent", prompt.text)
        self.assertEqual(direct_applicability_set(task, decisions), ["A", "C"])
        self.assertEqual(support_spans(task, self.assessments(("U", "U", "U"))), ())

    def test_chat_role_and_offset_mask_are_native(self):
        original = condition_prompt(self.task(), "A")
        prepared = chat_prompt(original, CharTokenizer())
        self.assertTrue(prepared.text.startswith("<user>"))
        for start, end in prepared.absolute_spans(prepared.evidence.spans):
            self.assertIn("Alpha", prepared.text[start:end])

    def test_condition_failure_is_unknown_without_fabricated_probability(self):
        failed = assessment("A", {"status": "error"})
        self.assertEqual(failed["label"], "U")
        self.assertIsNone(failed["candidate_normalized_probabilities"])
        malformed = assessment("A", {"status": "ok", "label_logits": {"T": float("nan"), "F": 0., "U": 0.}})
        self.assertEqual(malformed["status"], "error")

    def test_factorial_inputs_budgets_and_shared_costs(self):
        calls = []
        def infer(*args):
            calls.append(args)
            return self.fake_infer(*args)
        bundle = run_factorial(self.task(), sample_id="synthetic", seed=0, lock=self.lock(), infer=infer)
        self.assertEqual(len(calls), 10)  # 2 x 3 one-token A calls + four B calls.
        self.assertEqual({p["method"] for p in bundle["predictions"]}, set(ARMS))
        for index in range(3):
            self.assertEqual(calls[index][0].text, calls[index + 5][0].text)
        self.assertEqual(calls[3][0].text, calls[4][0].text)
        self.assertEqual(calls[8][0].text, calls[9][0].text)
        self.assertTrue(all(p["prediction"] == ["A"] for p in bundle["predictions"]))
        costs = bundle["cost_summary"]
        self.assertEqual(costs["actual_physical_run"]["actual_forward_calls"], 6 + 4 * 512)
        self.assertTrue(all(v["actual_forward_calls"] == 3 + 512 for v in costs["standalone_per_method"].values()))
        json.dumps(bundle)  # Fake results contain no nonserializable score tensors.

    def test_all_failed_rows_remain_in_factorial_and_missing_cost_is_not_zero(self):
        def fail(*args):
            raise RuntimeError("Synthetic model failure")
        bundle = run_factorial(self.task(), sample_id="synthetic", seed=1, lock=self.lock(), infer=fail)
        self.assertEqual(len(bundle["predictions"]), 4)
        self.assertTrue(all(p["status"] == "error" and p["prediction"] is None for p in bundle["predictions"]))
        self.assertTrue(all(p["condition_failure_count"] == 3 for p in bundle["predictions"]))
        self.assertIsNone(bundle["cost_summary"]["actual_physical_run"]["actual_forward_calls"])

    def test_patient_only_negative_controls_change_equal_count_masks(self):
        task = self.task()
        evidence = option_evidence(task)
        prompt = condition_prompt(task, "B", evidence)
        adapter = MaskAdapter()
        prepared = attach_controls(prompt, adapter, evidence, [key for key, _ in task.options], target="B")
        _, _, main = adapter._encode(prepared, prepared.evidence.spans)
        _, _, wrong = adapter._encode(prepared, prepared.evidence.wrong_spans)
        self.assertTrue(prepared.evidence.wrong_spans)
        self.assertEqual(int(main.sum()), int(wrong.sum()))
        self.assertFalse(bool((main & wrong).any()))
        # B rotates to C, whose option has no matching patient sentence.
        self.assertEqual(prepared.evidence.misaligned_rule_spans, ())

    def test_unfrozen_or_changed_source_lock_cannot_start(self):
        lock = self.lock()
        lock["frozen"] = False
        with self.assertRaises(ValueError):
            run_factorial(self.task(), sample_id="s", seed=0, lock=lock, infer=self.fake_infer)
        lock = self.lock()
        lock["source_hashes"] = {}
        with self.assertRaises(ValueError):
            run_factorial(self.task(), sample_id="s", seed=0, lock=lock, infer=self.fake_infer)


if __name__ == "__main__":
    unittest.main()
