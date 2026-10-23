"""CPU-only regression tests for exploratory cluster sign-flip robustness."""
import json
import unittest

from step0_reanalysis import ROOT, read_jsonl
from step0_cluster_sensitivity import (
    analyse, cancer_prefix_proxy, exact_cluster_signflip, holm_adjust
)


class ClusterSensitivityTests(unittest.TestCase):
    def test_exact_sign_flip_counts(self):
        self.assertEqual(exact_cluster_signflip([1, 1])["p_two_sided"], .5)
        self.assertEqual(exact_cluster_signflip([1, 1, 1])["p_two_sided"], .25)
        self.assertEqual(exact_cluster_signflip([1, -1])["p_two_sided"], 1.0)
        self.assertEqual(exact_cluster_signflip([0, 0])["p_two_sided"], 1.0)
        self.assertEqual(exact_cluster_signflip([1, 1, 1])["tail_assignments"], 2)
        self.assertEqual(exact_cluster_signflip([1, 1, 1])["total_sign_assignments"], 8)
        with self.assertRaises(ValueError):
            exact_cluster_signflip([])
        with self.assertRaises(ValueError):
            exact_cluster_signflip([0.1])

    def test_proxy_and_holm(self):
        self.assertEqual(cancer_prefix_proxy("cns28"), "cns")
        self.assertEqual(cancer_prefix_proxy("cns_24"), "cns")
        self.assertEqual(cancer_prefix_proxy("hcc1"), "hcc")
        self.assertEqual(holm_adjust({"x": .01, "y": .02, "z": .2}),
                         {"x": .03, "y": .04, "z": .2})

    def test_reject_duplicate_or_mismatched_records(self):
        def make(id_, disease):
            rows = []
            for v in "ABCD":
                for method in ("branch_visible", "hidden_prune", "visible_prune"):
                    correct = v == "A"
                    rows.append({"id": id_, "disease": disease, "stratum": "U",
                                 "variant": v, "method": method, "gold": "A",
                                 "correct": correct})
            return rows
        rows = make("x", "aml_7") + make("y", "cns28")
        result = analyse(rows, ["x", "y"])
        self.assertEqual(result["contrasts"]["visible_A_minus_D_branch"]
                         ["net_correct_first_minus_second"], 2)
        self.assertEqual(result["provenance"]["cancer_prefix_proxy_groups"], 2)
        with self.assertRaises(ValueError):
            analyse(rows, ["x", "x"])
        with self.assertRaises(ValueError):
            analyse(rows + [rows[0]], ["x", "y"])
        with self.assertRaises(ValueError):
            analyse(rows, ["x"])

    def test_frozen_qwen200(self):
        selection = json.loads(
            (ROOT / "priority200/selection.json").read_text(encoding="utf-8")
        )["ids"]
        scores = read_jsonl(ROOT / "priority200/qwen/full_parser_v2/scores.jsonl")
        result = analyse(scores, selection)
        self.assertEqual(result["provenance"]["cases"], 200)
        self.assertEqual(result["provenance"]["disease_tree_keys"], 23)
        self.assertEqual(result["provenance"]["cancer_prefix_proxy_groups"], 13)
        self.assertEqual((result["provenance"]["K"], result["provenance"]["U"]),
                         (76, 124))
        expected = {
            "visible_A_minus_D_branch": (47, .235, 8 / 8192, 64 / 8388608),
            "hidden_minus_visible_A": (-5, -.025, 1536 / 8192, .3125),
            "hidden_minus_visible_B": (17, .085, 544 / 8192, .0364990234375),
            "hidden_minus_visible_C": (-9, -.045, 256 / 8192, .07421875),
            "hidden_minus_visible_D": (33, .165, 56 / 8192, .000335693359375),
        }
        for name, (net, delta, family_p, tree_p) in expected.items():
            row = result["contrasts"][name]
            self.assertEqual(row["net_correct_first_minus_second"], net)
            self.assertAlmostEqual(row["paired_accuracy_delta"], delta)
            self.assertAlmostEqual(
                row["cancer_type_prefix_proxy"]["p_two_sided"], family_p
            )
            self.assertAlmostEqual(row["disease_tree_proxy"]["p_two_sided"], tree_p)
        self.assertAlmostEqual(result["exploratory_holm"]["cancer_type_prefix_proxy"]
                               ["visible_A_minus_D_branch"], .0048828125)
        self.assertAlmostEqual(result["exploratory_holm"]["cancer_type_prefix_proxy"]
                               ["hidden_minus_visible_D"], .02734375)


if __name__ == "__main__":
    unittest.main()
