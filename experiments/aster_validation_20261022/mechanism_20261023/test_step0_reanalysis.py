"""CPU-only regression checks for the frozen Qwen200 step-0 audit."""
import json
import unittest
from pathlib import Path

from step0_reanalysis import ROOT, analyse, cluster_ci, exact_mcnemar_p, read_jsonl


class TestStep0Reanalysis(unittest.TestCase):
    def test_exact_discordance_p(self):
        self.assertAlmostEqual(exact_mcnemar_p(55, 8), 9.761673086614714e-10, places=16)
        self.assertAlmostEqual(exact_mcnemar_p(16, 2), 0.001312255859375)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_cluster_ci_requires_multiple_clusters(self):
        self.assertIsNone(cluster_ci([1, 0], ["same", "same"]))
        bounds = cluster_ci([1, -1, 1, 0], ["x", "y", "x", "z"], 100)
        self.assertEqual(len(bounds), 2)
        self.assertLessEqual(bounds[0], bounds[1])

    def test_public_200_snapshot(self):
        selected = json.loads((ROOT / "priority200/selection.json").read_text(encoding="utf-8"))["ids"]
        result = analyse(
            read_jsonl(ROOT / "priority200/qwen/full_parser_v2/scores.jsonl"),
            read_jsonl(ROOT / "cad200_final/cad_scores.jsonl"),
            read_jsonl(ROOT / "d_option_order/paired_scores.jsonl"),
            selected,
        )
        row = result["branch_A_visible_vs_D_visible"]["all"]
        self.assertEqual((row["A_correct"], row["D_correct"]), (140, 93))
        self.assertEqual((row["A_correct_D_wrong"], row["A_wrong_D_correct"]), (55, 8))
        self.assertEqual(result["decomposition"]["hidden_prune"]["A"]["branch_wrong_action_correct"], 24)
        self.assertEqual(result["decomposition"]["hidden_prune"]["D"]["branch_wrong_action_correct"], 6)
        self.assertEqual(result["A_wrong_branch_right_then_D"]["n"], 24)
        self.assertEqual(result["A_wrong_branch_right_then_D"]["D_correct"], 0)
        self.assertEqual(result["visibility_disagreement"]["D"]["disagree"], 74)
        for v in ("A", "B", "C", "D"):
            gate = result["visibility_disagreement"][v]
            self.assertEqual(gate["same_final_action_if_agree"], gate["agree"])
            self.assertAlmostEqual(
                gate["disagree_hidden_else_visible_accuracy"],
                result["decomposition"]["hidden_prune"][v]["action_accuracy"],
            )
        cad = result["cad_old128_vs_filtered_original"]["B"]
        self.assertEqual((cad["old_only"], cad["filtered_only"]), (16, 2))


if __name__ == "__main__":
    unittest.main()
