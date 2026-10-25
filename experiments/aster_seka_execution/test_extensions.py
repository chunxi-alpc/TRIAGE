"""Minimal server CPU tests; authored fixtures only, no official data rows read."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from medcalc_native import OfficialMedCalc, arithmetic_eval, method_input, predict
from medcalc_data import read_rows
from faitheval_native import (method_input as faith_input, messages as faith_messages,
                               normalize_answer, score_prediction, source_overlap_report)
from evaluate_medcalc import evaluate


class OnlyInputs(dict):
    def __getitem__(self, key):
        if key not in {"Patient Note", "Question", "context", "question"}:
            raise AssertionError("Evaluation-only metadata entered the method")
        return super().__getitem__(key)


class PublicContracts(unittest.TestCase):
    def test_both_method_inputs_exclude_labels_and_entities(self):
        row = OnlyInputs({"Patient Note": "Synthetic note.", "Question": "Synthetic question?",
                          "Relevant Entities": "SECRET", "Ground Truth Answer": "SECRET", "Lower Limit": "SECRET"})
        self.assertEqual(method_input(row).patient_note, "Synthetic note.")
        task = faith_input(OnlyInputs(context="Synthetic context.", question="Synthetic question?", answer="SECRET"))
        self.assertNotIn("SECRET", faith_messages(task, "unanswerable")[0]["content"])

    def test_formal_medcalc_freeze_blocks_before_file_read(self):
        with patch.object(Path, "open", side_effect=AssertionError("test file opened")):
            with self.assertRaisesRegex(ValueError, "freeze"):
                read_rows("/synthetic/path", "test", {"frozen": False})

    def test_arithmetic_only_parser_supports_original_numeric_operations(self):
        self.assertEqual(arithmetic_eval("round(2.5)"), 2)
        self.assertEqual(arithmetic_eval("(17, 4)"), (17, 4))
        self.assertEqual(arithmetic_eval("math.sqrt(9) + 2 ** 3"), 11)
        with self.assertRaises(ValueError):
            arithmetic_eval("__import__('os').getcwd()")

    def test_faitheval_preserves_published_substring_semantics(self):
        self.assertEqual(normalize_answer("The_red, fox!"), "red fox")
        self.assertEqual(score_prediction("It is unknown.", "unanswerable", strict=True)["correct"], 1)
        self.assertEqual(score_prediction("This is noteworthy.", "unanswerable", strict=False)["correct"], 1)
        self.assertEqual(score_prediction("There is conflicting evidence.", "inconsistent", strict=True)["correct"], 1)
        self.assertEqual(score_prediction("X", "counterfactual")["status"], "blocked")
        self.assertEqual(score_prediction("The red fox.", "counterfactual", reference_answers=["red fox"])["correct"], 1)

    def test_absent_source_ids_do_not_prove_independence(self):
        result = source_overlap_report([{"context_sha256": "a"}], [{"context_sha256": "b"}])
        self.assertEqual(result["source_independence"], "not_established")


class PinnedOfficialFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("MEDCALC_OFFICIAL_ROOT")
        if not root:
            raise unittest.SkipTest("Set MEDCALC_OFFICIAL_ROOT to the pinned server checkout; no test data are read")
        cls.official = OfficialMedCalc(root)

    def score(self, raw, calculator, truth, lower=None, upper=None):
        return self.official.score({"Calculator ID": str(calculator), "Ground Truth Answer": truth,
            "Lower Limit": truth if lower is None else lower, "Upper Limit": truth if upper is None else upper}, raw)

    def test_author_date_integer_and_weeks_rules(self):
        self.assertEqual(self.score('{"answer":"2/29/2024"}', 13, "02/29/2024")["correct"], 1)
        self.assertEqual(self.score('{"answer":"02/30/2024"}', 13, "02/29/2024")["correct"], 0)
        self.assertEqual(self.score('{"answer":"2.5"}', 4, "2")["correct"], 1)
        self.assertEqual(self.score('{"answer":"17 weeks, 4 days"}', 69, "(17, 4)")["correct"], 1)

    def test_author_decimal_inclusive_boundaries_and_failure(self):
        for answer in ("95", "105"):
            self.assertEqual(self.score('{"answer":"' + answer + '"}', 2, "100", "95", "105")["correct"], 1)
        self.assertEqual(self.score('{"answer":"94.999"}', 2, "100", "95", "105")["correct"], 0)
        self.assertEqual(self.score("no result", 2, "100", "95", "105")["correct"], 0)
        self.assertEqual(self.score('{"answer":"invalid"}', 69, "(1, 1)")["correct"], 0)

    def test_actual_author_prompt_and_full_missing_prediction_denominator(self):
        task = method_input(OnlyInputs({"Patient Note": "Synthetic values 2 and 3.", "Question": "Their sum?",
                                       "Relevant Entities": "SECRET", "Ground Truth Explanation": "SECRET"}))
        observed = []
        def generate(messages):
            observed.extend(messages)
            return {"text": '{"answer":"5"}', "status": "ok"}
        public, private = predict(task, generate, self.official, sample_id="1")
        self.assertNotIn("SECRET", str(observed))
        rows = [{"Row Number": str(i), "Category": "synthetic", "Patient Note": "Synthetic.", "Note ID": "same",
                 "Note Type": "Synthetic", "Calculator ID": "4", "Ground Truth Answer": "5",
                 "Lower Limit": "5", "Upper Limit": "5"} for i in (1, 2)]
        result = evaluate(rows, [private], self.official)
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["metrics"]["overall"]["accuracy"], .5)
        self.assertEqual(result["missing_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
