"""Synthetic software checks, never experimental observations."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("care_analyze", Path(__file__).resolve().parents[1] / "src/analyze.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class AnalysisChecks(unittest.TestCase):
    def test_cluster_ci_preserves_family(self):
        # Four items sharing one family cannot produce a spurious item-level CI.
        metric = a.one_metric({str(i): ("same_family", float(i < 2)) for i in range(4)}, 7, "check", 100)
        self.assertEqual(metric["estimate"], 0.5)
        self.assertEqual(metric["ci95"], [0.5, 0.5])
        self.assertEqual(metric["n_clusters"], 1)

    def test_exact_cluster_permutation_and_mcnemar(self):
        left = {str(i): (f"family_{i}", 1.0) for i in range(3)}
        right = {str(i): (f"family_{i}", 0.0) for i in range(3)}
        value = a.compare(left, right, 7, "check", 100, 1000)
        self.assertEqual(value["difference"], 1.0)
        self.assertEqual(value["cluster_test"]["p"], 0.25)
        self.assertEqual(value["secondary_mcnemar_exact_p"], 0.25)

    def test_pair_complete_and_unknown_metadata(self):
        rows = [a.canonical({"item_id": str(i), "pair_id": "p", "family_id": "f", "correct": True, "applicable": bool(i)}) for i in range(2)]
        units = a.metric_units(rows, {"p": {"0", "1"}})
        self.assertEqual(units["flip_pair_joint_accuracy"]["p"], ("f", 1.0))
        self.assertNotIn("new_rule_misuse", units)
        missing = a.metric_units(rows[:1], {"p": {"0", "1"}})
        self.assertNotIn("flip_pair_joint_accuracy", missing)

    def test_missing_methods_not_imputed(self):
        self.assertIsNone(a.compare({"x": ("f", 1.0)}, {}, 7, "check", 100, 1000))

    def test_holm(self):
        self.assertEqual(a.holm({"one": 0.03, "two": 0.01}), {"two": 0.02, "one": 0.03})

    def test_invalid_answers_remain_in_accuracy_and_misuse_bounds(self):
        row = a.canonical({"item_id": "bad", "family_id": "f", "correct": False, "applicable": False, "new_action_selected": None, "valid_answer": False})
        units = a.metric_units([row], {})
        self.assertEqual(units["accuracy"]["bad"][1], 0.0)
        self.assertEqual(units["invalid_answer_rate"]["bad"][1], 1.0)
        self.assertEqual(units["answer_parse_coverage"]["bad"][1], 0.0)
        self.assertEqual(units["new_rule_misuse_lower_bound"]["bad"][1], 0.0)
        self.assertEqual(units["new_rule_misuse_upper_bound"]["bad"][1], 1.0)
        self.assertNotIn("new_rule_misuse", units)

    def test_seed_reproducible(self):
        x = a.rng_for(7, "a").integers(0, 10000, 20)
        y = a.rng_for(7, "a").integers(0, 10000, 20)
        self.assertTrue(np.array_equal(x, y))


if __name__ == "__main__":
    unittest.main()
