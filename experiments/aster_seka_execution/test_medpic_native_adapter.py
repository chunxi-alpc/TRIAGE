"""Server CPU synthetic MedPIC contracts; never opens the official dataset."""
from dataclasses import asdict
import unittest

from medpic_native_adapter import (AUGMENTED_PROTOCOL, INSTRUCTION, NATIVE_PROTOCOL, Generation,
    case_rng_seed, fingerprint, native_input, parse_final_answer, predict_native_row,
    render_augmented_open_book, render_native)


class MetadataTrap(dict):
    def __getitem__(self, key):
        if key in {"answer", "benchmark_task_family", "taxonomy_concept", "guideline", "pair_id"}:
            raise AssertionError("A held-out label/metadata field was accessed")
        return super().__getitem__(key)


class NativeContracts(unittest.TestCase):
    def row(self):
        return MetadataTrap(instance_id="synthetic_1", patient_vignette="Synthetic patient information.",
                            question="Which synthetic options apply?", options={"A": "Choice one", "B": "Choice two", "C": "Choice three"},
                            answer=["SECRET_LABEL"], benchmark_task_family="SECRET_FAMILY",
                            guideline="SECRET_GUIDELINE", pair_id="SECRET_PAIR")

    def lock(self):
        return {"frozen": True, "frozen_at": "synthetic", "protocol": NATIVE_PROTOCOL,
                "methods": ["baseline"], "seeds": [0, 1, 2], "prompt_template_sha256": fingerprint(INSTRUCTION)}

    def test_only_native_input_fields_reach_prompt(self):
        task = native_input(self.row())
        self.assertEqual(set(asdict(task)), {"patient_vignette", "question", "options"})
        prompt = render_native(task)
        self.assertNotIn("SECRET", prompt.text)
        left, right = prompt.patient_char_span
        self.assertEqual(prompt.text[left:right], task.patient_vignette)

    def test_multi_select_uses_only_designated_final_field(self):
        self.assertEqual(parse_final_answer('A is discussed in the explanation.\nFINAL_ANSWER: ["C", "B"]', ["A", "B", "C"]), ["B", "C"])
        self.assertIsNone(parse_final_answer('The options are A and C.', ["A", "B", "C"]))
        self.assertIsNone(parse_final_answer('FINAL_ANSWER: ["Z"]', ["A", "B", "C"]))
        self.assertIsNone(parse_final_answer('FINAL_ANSWER: ["A"]\nFINAL_ANSWER: ["C"]', ["A", "B", "C"]))

    def test_model_derived_context_is_separate_and_not_official(self):
        task = native_input(self.row())
        native = render_native(task)
        augmented = render_augmented_open_book(task, "Synthetic derived rule.", "model_derived_rules")
        self.assertEqual(native.protocol, NATIVE_PROTOCOL)
        self.assertEqual(augmented.protocol, AUGMENTED_PROTOCOL)
        self.assertIn("not an official guideline", augmented.text)
        self.assertNotIn("Synthetic derived rule", native.text)

    def test_prediction_callback_receives_no_raw_row(self):
        received = []
        def generate(prepared, seed):
            received.append((prepared, seed))
            return Generation('Brief synthetic explanation.\nFINAL_ANSWER: ["A", "C"]')
        public, private = predict_native_row(self.row(), generate, method="baseline", seed=0, lock=self.lock())
        self.assertEqual(public["prediction"], ["A", "C"])
        self.assertNotIn("prompt", public)
        self.assertNotIn("SECRET", private["prompt"])
        self.assertEqual(received[0][1], case_rng_seed(0, "synthetic_1"))

    def test_failed_generation_is_a_retained_invalid_prediction(self):
        def fail(prepared, seed):
            raise RuntimeError("Synthetic failure")
        public, private = predict_native_row(self.row(), fail, method="baseline", seed=0, lock=self.lock())
        self.assertEqual(public["sample_id"], "synthetic_1")
        self.assertEqual(public["status"], "error")
        self.assertIsNone(public["prediction"])

    def test_augmented_lock_cannot_be_used_as_native(self):
        lock = self.lock()
        lock["protocol"] = AUGMENTED_PROTOCOL
        with self.assertRaises(ValueError):
            predict_native_row(self.row(), lambda *args: Generation(""), method="baseline", seed=0, lock=lock)


if __name__ == "__main__":
    unittest.main()
