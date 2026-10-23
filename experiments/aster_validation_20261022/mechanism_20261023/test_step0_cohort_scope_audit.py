"""Cohort-overlap audit regression checks. No inference or model dependencies."""
import unittest

from step0_cohort_scope_audit import analyse


def one(id_, disease, protocol="PR23_strict", single=True, stratum="U",
        variants=None):
    return {
        "id": id_, "disease": disease, "pair_protocol": protocol,
        "single_action_gold_branch": single, "stratum": stratum,
        "variants": variants or ["A", "B", "C", "D"],
    }


class CohortScopeTests(unittest.TestCase):
    def setUp(self):
        self.old = ["old1", "old2"]
        self.scores = [
            {"id": "old1", "variant": "A", "method": "branch_hidden", "disease": "aml_7"},
            {"id": "old2", "variant": "A", "method": "branch_hidden", "disease": "bre_2"},
        ]
        self.expanded = [
            one("old1", "aml_7"),
            one("new_case_old_tree", "aml_7", protocol="extended_offpath", single=False),
            one("new_case_new_tree_strict", "cns_24", stratum="K"),
            one("new_case_new_tree_extended", "nscl_3", protocol="extended_offpath",
                single=False),
        ]

    def test_three_distinct_novelty_levels(self):
        result = analyse(self.scores, self.old, self.expanded)
        c = result["counts"]
        self.assertEqual(c["all"]["n"], 4)
        self.assertEqual(c["no_case_overlap_with_200"]["n"], 3)
        self.assertEqual(c["no_tree_overlap_with_200"]["n"], 2)
        self.assertEqual(c["no_tree_overlap_strict"]["n"], 1)
        self.assertEqual(c["no_tree_overlap_strict_single"]["n"], 1)
        self.assertEqual(c["no_cancer_prefix_overlap_with_200"]["n"], 2)
        self.assertEqual(c["no_cancer_prefix_overlap_strict_single"]["n"], 1)
        self.assertEqual(result["new_cancer_prefix_proxies"], ["cns", "nscl"])
        self.assertEqual(result["original200_overlap_ids"], ["old1"])
        self.assertEqual(result["new_tree_ids"], ["cns_24", "nscl_3"])
        self.assertFalse(result["provenance"]["tested_800_predictions"])

    def test_invalid_case_matching_rejected(self):
        wrong = [dict(x) for x in self.expanded]
        wrong[0]["disease"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Disease tree"):
            analyse(self.scores, self.old, wrong)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyse(self.scores, self.old, self.expanded + [self.expanded[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyse(self.scores, ["old1", "old1"], self.expanded)

    def test_missing_or_unexpected_protocol_rejected(self):
        wrong = [dict(x) for x in self.expanded]
        wrong[2]["pair_protocol"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected protocol"):
            analyse(self.scores, self.old, wrong)
        wrong[2]["pair_protocol"] = "PR23_strict"
        wrong[2]["variants"] = ["A", "C", "D"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            analyse(self.scores, self.old, wrong)


if __name__ == "__main__":
    unittest.main()
