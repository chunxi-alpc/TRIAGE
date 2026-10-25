"""Synthetic contract tests only; run on the research server before using evaluator."""
import unittest

from evaluate_medpic import option_set, score_questions, merge_linked_clusters
from paired_statistics import paired_cluster_comparison, holm_adjust
from audit_server_metadata import extract_medpic_pairs


class EvaluationContracts(unittest.TestCase):
    def test_multiselect_is_exact_set(self):
        self.assertEqual(option_set(["C", "A"], "ABC"), {"A", "C"})
        self.assertEqual(option_set(["A", "A"], "ABC"), {"A"})
        self.assertIsNone(option_set("I think A, maybe B", "ABC"))
        self.assertIsNone(option_set(["Z"], "ABC"))
        self.assertEqual(option_set("A, C", "ABC"), {"A", "C"})

    def test_missing_and_wrong_sets_remain_in_denominator(self):
        gold = [{"instance_id":"1","answer":["A","C"],"options":dict.fromkeys("ABC"),"benchmark_task_family":"counterfactual"},
                {"instance_id":"2","answer":["A"],"options":dict.fromkeys("AB"),"benchmark_task_family":"guideline_following"}]
        scored, metrics = score_questions(gold, [{"sample_id":"1","method":"m","seed":0,"prediction":["A"]}], ["m"], [0])
        self.assertEqual(metrics[0]["denominator"], 2)
        self.assertEqual(metrics[0]["accuracy"], 0)
        self.assertEqual(metrics[0]["missing_predictions"], 1)
        self.assertFalse(metrics[0]["run_coverage_complete"])
        self.assertEqual(len(scored), 2)

    def test_duplicate_is_not_silently_removed(self):
        gold = [{"instance_id":"1","answer":["A"],"options":dict.fromkeys("AB")}]
        p = {"sample_id":"1","method":"m","seed":0,"prediction":["A"]}
        with self.assertRaises(ValueError):
            score_questions(gold, [p,p], ["m"], [0])

    def test_seeds_do_not_inflate_independent_clusters(self):
        rows = []
        for sample in ("1", "2", "3", "4"):
            for seed in (0,1,2):
                rows += [{"sample_id":sample,"method":"m","seed":seed,"accuracy":1},
                         {"sample_id":sample,"method":"b","seed":seed,"accuracy":0}]
        r = paired_cluster_comparison(rows,"m","b","accuracy",{"1":"x","2":"x","3":"y","4":"y"},[0,1,2],bootstrap_samples=100)
        self.assertEqual(r["n_source_patient_clusters"], 2)
        self.assertEqual(r["n_samples"], 4)
        self.assertEqual(r["ci95"], [1,1])
        self.assertEqual(r["p_value"], .5)
        self.assertEqual(r["corrections_and_regressions"]["0"]["corrected"], 4)
        blocked = paired_cluster_comparison(rows,"m","b","accuracy",{},[0,1,2],bootstrap_samples=100)
        self.assertEqual(blocked["status"], "blocked")

    def test_holm_keeps_missing_tests_in_family(self):
        self.assertEqual(holm_adjust([.01,.04,None]), [.03,.08,None])

    def test_pair_map_is_not_guessed(self):
        result = extract_medpic_pairs([{"instance_id":"CF-01-A"},{"instance_id":"CF-01-B"}])
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["pairs"])
        merged = merge_linked_clusters({"1":"sourceA","2":"sourceB","3":"sourceA"},[("1","2")])
        self.assertEqual(len(set(merged.values())),1)


if __name__ == "__main__":
    unittest.main()
