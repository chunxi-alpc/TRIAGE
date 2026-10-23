"""Standard-library regression tests; no model, tokenizer, or private patient text."""
import json
import unittest

from step0_reanalysis import ROOT, read_jsonl
from step0_extended_audit import (
    branch_action_labels, family_proxy, normalized_action,
    private_case_audit, public_audit,
)


def fixture():
    rules_a = [
        {"condition": "path -> meets criterion", "action": "Action One"},
        {"condition": "path -> not meeting criterion", "action": "Action Two"},
        {"condition": "path -> not meeting criterion", "action": "Action Three"},
    ]
    rules_d = [dict(x) for x in rules_a]
    rules_d[0]["action"], rules_d[1]["action"] = rules_d[1]["action"], rules_d[0]["action"]
    c = {
        "id": "synthetic", "options": {"A": "Action One", "B": "Action Two", "C": "Action Three"},
        "branches": [{"rule_indices": [0]}, {"rule_indices": [1, 2]}],
        "variant_rules": {"A": rules_a, "B": rules_a, "C": rules_a, "D": rules_d},
        "gold_answer": {"A": "A", "B": "A", "C": "A", "D": "B"},
        "gold_branch": {v: 0 for v in "ABCD"},
        "app_rule": 0, "sib_rule": 1, "app_branch": 0, "sib_branch": 1,
    }
    scores = []
    for v in "ABCD":
        gold_action = c["gold_answer"][v]
        for method, kind, prediction, extra in (
            ("branch_visible", "branch", "B", {}),
            ("branch_hidden", "branch", "B", {}),
            ("visible_prune", "action", "A", {"predicted_branch": 1}),
            ("hidden_prune", "action", "A", {"predicted_branch": 1}),
            ("hidden_top2", "action", "B", {"predicted_branch": 1, "runner_up": 0}),
            ("gold_branch_prune", "action", "B", {"original_action_selected": False}),
        ):
            gold = "A" if kind == "branch" else gold_action
            scores.append({"id": c["id"], "variant": v, "method": method, "kind": kind,
                           "prediction": prediction, "gold": gold, "correct": prediction == gold,
                           "stratum": "K", "disease": "cns28", **extra})
    return c, scores


class ExtendedAuditTests(unittest.TestCase):
    def test_normalization_and_proxy_cannot_infer_clinical_equivalence(self):
        self.assertEqual(normalized_action("  Treatment.  "), "treatment")
        self.assertEqual(family_proxy("cns28"), family_proxy("cns_24"))
        self.assertNotEqual(normalized_action("antibiotic"), normalized_action("antiviral"))

    def test_private_rule_membership(self):
        case, rows = fixture()
        self.assertEqual(branch_action_labels(case, "A", [1]), {"B", "C"})
        self.assertEqual(branch_action_labels(case, "D", [1]), {"A", "C"})
        result = private_case_audit([case], rows, [case["id"]])
        self.assertEqual(result["integrity_counts"]["D_swap_into_multi_rule_sibling"], 1)
        self.assertNotIn("D_not_exact_two_rule_action_swap", result["integrity_counts"])
        self.assertEqual(result["A_wrong_branch_right_answer_cohort"]["ids"], ["synthetic"])
        self.assertEqual(result["A_wrong_branch_right_answer_cohort"]
                         ["A_gold_action_absent_from_wrong_branch"], ["synthetic"])
        self.assertEqual(result["A_wrong_branch_right_answer_cohort"]
                         ["D_gold_action_absent_from_predicted_branch"], ["synthetic"])
        self.assertEqual(result["method_variant_cells"]["A/hidden_prune"]
                         ["wrong_branch_right_answer_gold_absent"], 1)
        self.assertEqual(result["oracle_error_rule_membership"]["A"]
                         ["wrong_option_absent_from_gold_rules"], 1)

    def test_malformed_swap_reported(self):
        case, rows = fixture()
        case["variant_rules"]["D"][0]["action"] = "Unrelated"
        result = private_case_audit([case], rows, ["synthetic"])
        self.assertEqual(result["integrity_counts"]["D_not_exact_two_rule_action_swap"], 1)
        self.assertEqual(result["integrity_counts"]
                         ["constructed_gold_missing_from_gold_branch_D"], 1)

    def test_wrong_case_set_fails(self):
        case, rows = fixture()
        with self.assertRaises(ValueError):
            private_case_audit([case], rows, ["other"])

    def test_public_frozen_snapshot(self):
        ids = json.loads((ROOT / "priority200/selection.json").read_text())["ids"]
        rows = read_jsonl(ROOT / "priority200/qwen/full_parser_v2/scores.jsonl")
        result = public_audit(rows, ids)
        self.assertEqual(result["provenance"]["n"], 200)
        self.assertEqual(result["provenance"]["disease_labels"], 23)
        self.assertEqual(result["provenance"]["family_prefix_groups"], 13)
        changes = result["A_to_D_visible_transition"]
        self.assertEqual(changes["correct/wrong/changed"], 55)
        self.assertEqual(changes["wrong/correct/changed"], 8)
        self.assertEqual(changes["wrong/wrong/changed"], 8)
        self.assertEqual(changes["correct/correct/same"], 85)
        self.assertEqual(changes["wrong/wrong/same"], 44)
        self.assertEqual(result["by_family_proxy"]["aml"]["n"], 43)
        self.assertEqual(result["by_family_proxy"]["cns"]["n"], 29)
        self.assertEqual(result["A_to_D_group_effects"]["all"]["net_correct_A_minus_D"], 47)
        self.assertEqual(result["alert_for_visible_answer_error"]["D"]["true_positive"], 62)
        self.assertEqual(result["alert_for_visible_answer_error"]["D"]["false_positive"], 12)
        self.assertEqual(result["alert_for_visible_answer_error"]["D"]["false_negative"], 43)
        self.assertEqual(result["alert_for_visible_answer_error"]["D"]["true_negative"], 83)


if __name__ == "__main__":
    unittest.main()
