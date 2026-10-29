import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_cluster_stats.py"
spec = importlib.util.spec_from_file_location("paired_cluster_stats", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PairedClusterStatsTests(unittest.TestCase):
    def test_corrected_regressed_and_denominator(self):
        rows = [
            {"guideline_id": "g1", "baseline_correct": False, "method_correct": True},
            {"guideline_id": "g1", "baseline_correct": True, "method_correct": False},
            {"guideline_id": "g2", "baseline_correct": False, "method_correct": False, "technical_failure": True},
        ]
        result = mod.summarize(rows, bootstrap_draws=200, seed=7)
        self.assertEqual(result["n_cases"], 3)
        self.assertEqual(result["n_guideline_clusters"], 2)
        self.assertEqual(result["corrected"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["failure_rows"], 1)
        self.assertEqual(result["net_corrected_minus_regressed"], 0)

    def test_custom_group_key(self):
        rows = [
            {"tree": "A", "baseline_correct": False, "method_correct": True},
            {"tree": "B", "baseline_correct": False, "method_correct": False},
        ]
        result = mod.summarize(rows, bootstrap_draws=200, seed=1, group_key="tree")
        self.assertEqual(result["n_guideline_clusters"], 2)
        self.assertAlmostEqual(result["guideline_macro_delta"], 0.5)

    def test_non_boolean_scores_rejected(self):
        with self.assertRaises(ValueError):
            mod.load_rows_from_objects([{"guideline_id": "g1", "baseline_correct": 1, "method_correct": True}])


if __name__ == "__main__":
    unittest.main()
