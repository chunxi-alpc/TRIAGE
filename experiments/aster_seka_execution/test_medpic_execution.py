"""Server CPU synthetic tests for complete orchestration and pre-read freeze."""
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import unittest

import torch

from medpic_native_adapter import native_input
from medpic_option_transfer import TransferConfig
from medpic_execution import method_names, run_case, numerical_drift
from run_medpic import fake_callbacks, input_rows, synthetic_rows, validate_lock


class ExecutionContracts(unittest.TestCase):
    def test_unfrozen_formal_request_never_opens_the_question_file(self):
        with patch.object(Path, "open", side_effect=AssertionError("file was accessed")):
            with self.assertRaisesRegex(ValueError, "prohibited"):
                input_rows({"frozen": False}, synthetic=False, questions=Path("do_not_open.json"))

    def test_synthetic_path_is_builtin_only_and_fake_cannot_be_formal(self):
        rows, info = input_rows({"frozen": False}, synthetic=True, engine="fake")
        self.assertEqual(len(rows), 2)
        self.assertFalse(info["dataset_file_opened"])
        self.assertTrue(all(set(row) == {"instance_id", "patient_vignette", "question", "options"} for row in rows))
        with self.assertRaises(ValueError):
            input_rows({}, synthetic=True, questions=Path("external.json"), engine="fake")
        with self.assertRaises(ValueError):
            input_rows({"frozen": True}, synthetic=False, engine="fake")

    def test_full_matrix_retains_all_controls_and_same_A_fusion_prompts(self):
        task = native_input(synthetic_rows()[0])
        infer, base = fake_callbacks()
        fusion_calls, events = [], []
        def baseline(primary, auxiliary, **kwargs):
            fusion_calls.append((primary.text, auxiliary.text, kwargs))
            return base(primary, auxiliary, **kwargs)
        bundle = run_case(task, sample_id="synthetic", seed=0, config=TransferConfig(), lock_sha256="synthetic",
                          infer=infer, generate_baseline=baseline, on_trace=lambda entry, tensors: events.append(entry["trace_id"]))
        by_method = {row["method"]: row for row in bundle["predictions"]}
        trace = {entry["trace_id"]: entry for entry in bundle["private_traces"]}
        self.assertEqual(len(by_method), 25)
        self.assertEqual(set(by_method), set(method_names()))
        self.assertEqual(len(fusion_calls), 8)
        self.assertEqual(len(events), len(trace))
        for a in ("A0", "A1"):
            for kind in ("static", "dynamic"):
                row = by_method[f"{a}_fusion_{kind}"]
                self.assertEqual(row["prompt_sha256"], by_method[f"{a}B0"]["prompt_sha256"])
                self.assertEqual(row["cost_references"][:-1], by_method[f"{a}B0"]["cost_references"][:-1])
                self.assertEqual(trace[f"{a}_fusion_{kind}"]["auxiliary_prompt"], trace[f"{a}_text_emphasis"]["prompt"])
            for control in ("wrong", "misaligned"):
                row = by_method[f"{a}B1_{control}"]
                self.assertEqual(row["status"], "control_unavailable")
                self.assertEqual(row["reused_from"], f"{a}B0")
                self.assertEqual(row["prediction"], by_method[f"{a}B0"]["prediction"])
                self.assertFalse(row["control_executed"])
            self.assertEqual(by_method[f"{a}B1_zero"]["numerical_control"]["max_abs_first_token_difference"], 0.)
        self.assertTrue(bundle["score_tensors"])

    def test_A0_only_fusion_does_not_silently_run_A1_fusion(self):
        infer, baseline = fake_callbacks()
        bundle = run_case(native_input(synthetic_rows()[0]), sample_id="s", seed=0, config=TransferConfig(),
                          lock_sha256="s", infer=infer, generate_baseline=baseline, fusion_sources=["A0"])
        names = {row["method"] for row in bundle["predictions"]}
        self.assertEqual(len(names), 21)
        self.assertNotIn("A1_fusion_static", names)
        self.assertIn("A1B1", names)

    def test_model_failures_keep_every_method_and_do_not_create_zero_cost(self):
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")
        bundle = run_case(native_input(synthetic_rows()[0]), sample_id="s", seed=2, config=TransferConfig(),
                          lock_sha256="s", infer=fail, generate_baseline=fail)
        self.assertEqual(len(bundle["predictions"]), 25)
        self.assertTrue(all(not row["parse_valid"] for row in bundle["predictions"]))
        self.assertIsNone(bundle["cost_summary"]["actual_physical_run"]["actual_forward_calls"])
        self.assertTrue(all(item["label"] == "U" and item["candidate_normalized_probabilities"] is None
                            for values in bundle["assessments"].values() for item in values))

    def test_logprob_zero_compares_the_declared_base_temperature(self):
        raw = torch.tensor([1., 2., -1.])
        expected = (raw / 2).log_softmax(-1)
        result = numerical_drift(raw, expected, reference_name="synthetic", logprob=True, temperature=2.)
        self.assertEqual(result["max_abs_first_token_difference"], 0.)

    def test_synthetic_draft_is_not_mutated_or_frozen(self):
        lock = {"frozen": False, "config": asdict(TransferConfig())}
        config, fusion, sources, methods = validate_lock(lock, synthetic=True, seed=0)
        self.assertFalse(lock["frozen"])
        self.assertEqual(sources, ["A0", "A1"])
        self.assertEqual(len(methods), 25)


if __name__ == "__main__":
    unittest.main()
