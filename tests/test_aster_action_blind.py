"""Offline contract checks; no model download and no clinical performance claims."""

import copy
import json
import math
from pathlib import Path
import types
import unittest

from src.aster import (
    ActionBlindBranchClassifier, HFLabelScorer, build_blind_prompt,
    fit_temperature, rule_gains, summarize_logits, validate_guideline,
)


def sample():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "examples/aster_action_blind_input.json").read_text(encoding="utf-8"))


class RecordingScorer:
    def __init__(self, scores=(4.0, 0.0, -1.0)):
        self.scores = scores
        self.calls = []

    def score(self, prompt, labels):
        self.calls.append((prompt, labels))
        return self.scores


class CharacterTokenizer:
    """A deterministic tokenizer double used only to test boundary validation."""

    chat_template = "test-template"
    model_max_length = 100000
    all_special_ids = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is False
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"] + "\nAssistant:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class ActionBlindTests(unittest.TestCase):
    def run_sample(self, value=None, scores=(4.0, 0.0, -1.0), **kwargs):
        value = value or sample()
        scorer = RecordingScorer(scores)
        classifier = ActionBlindBranchClassifier(scorer, temperature=1.0,
                        probability_threshold=0.7, margin_threshold=0.2)
        result = classifier.run(value["guideline"], value["patient"],
                                value.get("question"), value.get("options"), **kwargs)
        return result, scorer

    def test_action_permutation_question_and_options_leave_prompt_identical(self):
        original = sample()
        changed = copy.deepcopy(original)
        rules = changed["guideline"]["groups"][0]["rules"]
        rules[0]["action"], rules[1]["action"] = "完全不同的处置", "另一处置"
        changed["question"] = "是否使用完全不同的处置？"
        changed["options"] = ["另一处置", "完全不同的处置"]
        before, left = self.run_sample(original)
        after, right = self.run_sample(changed)
        self.assertEqual(left.calls, right.calls)
        self.assertEqual(before, after)
        prompt = left.calls[0][0]
        self.assertNotIn("方案X", prompt)
        self.assertNotIn("方案Y", prompt)
        self.assertIn("既往未使用H", prompt)
        self.assertIn("患者已复发", prompt)

    def test_unknown_metadata_and_raw_identifiers_do_not_enter_prompt(self):
        value = sample()
        group = value["guideline"]["groups"][0]
        group["group_id"] = "treatment-name-in-id"
        group["rules"][0]["rule_id"] = "another-treatment-in-id"
        group["title"] = "hidden treatment"
        group["rules"][0]["recommendation"] = "hidden recommendation"
        result, scorer = self.run_sample(value)
        self.assertNotIn("treatment", scorer.calls[0][0])
        self.assertNotIn("hidden", scorer.calls[0][0])
        self.assertEqual(result["groups"][0]["prediction"], "another-treatment-in-id")

    def test_optional_context_requires_verification(self):
        with self.assertRaisesRegex(ValueError, "verification"):
            self.run_sample(task_context="仅讨论本次复发。")
        _, scorer = self.run_sample(task_context="仅讨论本次复发。", task_context_verified=True)
        self.assertIn("仅讨论本次复发", scorer.calls[0][0])

    def test_distribution_margin_entropy_and_schema(self):
        result, _ = self.run_sample(include_logits=True)
        group = result["groups"][0]
        self.assertEqual(result["schema_version"], "aster.action_blind.v1")
        self.assertAlmostEqual(sum(group["probabilities"].values()), 1.0)
        self.assertEqual(group["prediction"], "R1")
        self.assertAlmostEqual(group["probability"], group["probabilities"]["R1"])
        self.assertAlmostEqual(group["margin"], group["probabilities"]["R1"] - group["probabilities"]["R2"])
        self.assertAlmostEqual(group["entropy"], -sum(p * math.log(p) for p in group["probabilities"].values()))
        self.assertTrue(group["intervention_allowed"])
        self.assertEqual(group["logits"], {"R1": 4.0, "R2": 0.0, "NONE": -1.0})
        json.dumps(result, allow_nan=False)

    def test_abstention_does_not_relabel_prediction_as_none(self):
        result, _ = self.run_sample(scores=(0.2, 0.1, -3.0))
        group = result["groups"][0]
        self.assertEqual(group["prediction"], "R1")
        self.assertTrue(group["abstain"])
        self.assertFalse(group["intervention_allowed"])
        self.assertEqual(rule_gains(group), {"R1": 0.0, "R2": 0.0})

    def test_high_confidence_none_is_not_uncertainty_but_disables_edits(self):
        result, _ = self.run_sample(scores=(-2.0, -1.0, 5.0))
        group = result["groups"][0]
        self.assertEqual(group["prediction"], "NONE")
        self.assertFalse(group["abstain"])
        self.assertFalse(group["intervention_allowed"])
        self.assertEqual(rule_gains(group), {"R1": 0.0, "R2": 0.0})

    def test_temperature_preserves_prediction(self):
        cold = summarize_logits({"R1": 3, "R2": 1, "NONE": -2}, temperature=0.5,
                               probability_threshold=0.7, margin_threshold=0.2)
        hot = summarize_logits({"R1": 3, "R2": 1, "NONE": -2}, temperature=4,
                              probability_threshold=0.7, margin_threshold=0.2)
        self.assertEqual(cold["prediction"], hot["prediction"])
        self.assertLess(cold["entropy"], hot["entropy"])

    def test_exact_tie_always_abstains(self):
        result = summarize_logits({"R1": 0, "NONE": 0}, temperature=1,
                                  probability_threshold=0, margin_threshold=0)
        self.assertTrue(result["abstain"])

    def test_invalid_numeric_configuration_and_scores(self):
        for temperature in (0, -1, float("nan"), float("inf"), 10**500, True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                ActionBlindBranchClassifier(RecordingScorer(), temperature=temperature,
                                            probability_threshold=0.7, margin_threshold=0.2)
        for threshold in (-0.1, 1.1, float("nan"), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ActionBlindBranchClassifier(RecordingScorer(), temperature=1,
                                            probability_threshold=threshold, margin_threshold=0.2)
        for bad in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(logit=bad), self.assertRaises(ValueError):
                self.run_sample(scores=(bad, 1, 0))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_sample(scores=(1, 2))

    def test_invalid_groups_fail_before_scoring(self):
        mutations = [
            lambda g: g.update(rules=[]),
            lambda g: g.update(mutually_exclusive=False),
            lambda g: g.pop("mutually_exclusive"),
            lambda g: g["rules"][0].update(rule_id="NONE"),
            lambda g: g["rules"][1].update(rule_id="R1"),
            lambda g: g["rules"][1].update(condition=g["rules"][0]["condition"]),
            lambda g: g.update(ancestor_conditions="not-a-list"),
            lambda g: g["rules"][0].update(condition=""),
        ]
        for mutate in mutations:
            value = sample()
            mutate(value["guideline"]["groups"][0])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.run_sample(value)
        value = sample()
        group = value["guideline"]["groups"][0]
        group["rules"] = [{"rule_id": f"R{i}", "condition": str(i)} for i in range(26)]
        with self.assertRaisesRegex(ValueError, "25"):
            validate_guideline(value["guideline"])

    def test_duplicate_ids_across_groups_are_rejected(self):
        value = sample()
        second = copy.deepcopy(value["guideline"]["groups"][0])
        second["group_id"] = "G2"
        value["guideline"]["groups"].append(second)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_guideline(value["guideline"])

    def test_fixed_signed_gains_and_reachability(self):
        result, _ = self.run_sample()
        group = result["groups"][0]
        self.assertEqual(rule_gains(group, g_plus=0.5, g_minus=2), {"R1": 0.5, "R2": -2.0})
        self.assertEqual(rule_gains(group, eligible=False), {"R1": 0.0, "R2": 0.0})
        for kwargs in ({"g_plus": -1}, {"g_minus": float("nan")}, {"eligible": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rule_gains(group, **kwargs)

    def test_malformed_gain_inputs_are_rejected(self):
        result, _ = self.run_sample()
        group = result["groups"][0]
        for updates in ({"prediction": "R2"}, {"abstain": "false"},
                        {"intervention_allowed": False},
                        {"probabilities": {"R1": 0.8, "R2": 0.8, "NONE": 0.1}}):
            malformed = {**group, **updates}
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                rule_gains(malformed)

    def test_gain_rule_keys_must_be_nonempty_strings(self):
        for key in (None, 1, "", " "):
            malformed = {"probabilities": {key: 0.9, "NONE": 0.1},
                         "prediction": key, "abstain": False}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "rule_id"):
                rule_gains(malformed)

    def test_optional_gain_summaries_must_agree_with_probabilities(self):
        result, _ = self.run_sample()
        group = result["groups"][0]
        for field in ("probability", "margin", "entropy"):
            for bad_value in (group[field] + 0.05, float("nan"), True):
                malformed = {**group, field: bad_value}
                with self.subTest(field=field, value=bad_value), self.assertRaises(ValueError):
                    rule_gains(malformed)
        minimal = {k: v for k, v in group.items() if k not in {"probability", "margin", "entropy"}}
        self.assertEqual(rule_gains(minimal), rule_gains(group))


class TokenBoundaryTests(unittest.TestCase):
    def scorer(self, tokenizer=None, **kwargs):
        model = types.SimpleNamespace(training=False, config=types.SimpleNamespace(max_position_embeddings=100000))
        return HFLabelScorer(model, tokenizer or CharacterTokenizer(), **kwargs)

    def test_aliases_checked_at_rendered_template_boundary(self):
        encoded, labels = self.scorer().prepare("条件", ("A", "B", "C"))
        self.assertEqual(labels, [ord("A"), ord("B"), ord("C")])
        self.assertEqual(encoded[-1], ord(":"))

    def test_multitoken_rule_ids_cannot_be_scored_by_first_token(self):
        with self.assertRaisesRegex(ValueError, "not one token"):
            self.scorer().prepare("条件", ("R1", "R2", "NONE"))

    def test_boundary_retokenization_is_rejected(self):
        class BoundaryTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if text.endswith("A"):
                    ids[-2] = 9999
                return ids
        with self.assertRaisesRegex(ValueError, "prompt boundary"):
            self.scorer(BoundaryTokenizer()).prepare("条件", ("A", "B"))

    def test_alias_collisions_and_special_ids_are_rejected(self):
        class CollisionTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if text.endswith(("A", "B")):
                    ids[-1] = 42
                return ids
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.scorer(CollisionTokenizer()).prepare("条件", ("A", "B"))
        tokenizer = CharacterTokenizer()
        tokenizer.all_special_ids = [ord("A")]
        with self.assertRaisesRegex(ValueError, "nonspecial"):
            self.scorer(tokenizer).prepare("条件", ("A", "B"))

    def test_no_silent_truncation_or_thinking_mode_override(self):
        with self.assertRaisesRegex(ValueError, "truncation"):
            self.scorer(max_input_tokens=4).prepare("很长的条件", ("A", "B"))
        for kwargs in ({"enable_thinking": True}, {"truncation": True}, {"tokenize": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.scorer(chat_template_kwargs=kwargs)

    def test_shared_model_switched_back_to_training_is_rejected(self):
        scorer = self.scorer()
        scorer.model.training = True
        with self.assertRaisesRegex(ValueError, r"remain in eval\(\)"):
            scorer.score("条件", ("A", "B"))


class CalibrationTests(unittest.TestCase):
    def records(self):
        return [{"logits": {"R1": 6.0, "NONE": 0.0}, "label": "R1"} for _ in range(3)] + [
            {"logits": {"R1": 6.0, "NONE": 0.0}, "label": "NONE"}]

    def test_single_global_temperature_reduces_development_nll(self):
        fit = fit_temperature(self.records(), split="development")
        self.assertGreater(fit["temperature"], 1)
        self.assertLess(fit["nll_after"], fit["nll_before"])
        self.assertAlmostEqual(fit["temperature"], 6 / math.log(3), places=5)
        self.assertEqual(fit["n_records"], 4)
        self.assertFalse(fit["at_boundary"])

    def test_different_group_sizes_are_pooled(self):
        records = self.records() + [{"logits": {"R1": 2.0, "R2": 1.0, "NONE": 0.0}, "label": "R2"}]
        fit = fit_temperature(records, split="calibration")
        self.assertEqual(fit["n_records"], 5)
        self.assertLessEqual(fit["nll_after"], fit["nll_before"])

    def test_flat_logits_keep_identity_temperature(self):
        records = [{"logits": {"R1": 0.0, "NONE": 0.0}, "label": "R1"}]
        fit = fit_temperature(records, split="development")
        self.assertEqual(fit["temperature"], 1.0)
        self.assertFalse(fit["at_boundary"])

    def test_large_finite_losses_do_not_overflow_when_averaged(self):
        records = [{"logits": {"R1": 1e306, "NONE": 0.0}, "label": "NONE"}
                   for _ in range(200)]
        fit = fit_temperature(records, split="development")
        self.assertTrue(math.isfinite(fit["nll_before"]))
        self.assertAlmostEqual(fit["nll_before"] / 1e306, 1.0)
        self.assertTrue(math.isfinite(fit["nll_after"]))
        self.assertLess(fit["nll_after"], fit["nll_before"])
        json.dumps(fit, allow_nan=False)

    def test_test_split_empty_data_and_invalid_labels_are_rejected(self):
        for split in ("test", "train", None):
            with self.subTest(split=split), self.assertRaises(ValueError):
                fit_temperature(self.records(), split=split)
        with self.assertRaises(ValueError):
            fit_temperature([], split="development")
        with self.assertRaises(ValueError):
            fit_temperature([{"logits": {"R1": 1, "NONE": 0}, "label": "absent"}], split="development")


if __name__ == "__main__":
    unittest.main()
