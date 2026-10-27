"""Construction tests use invented trees; they are not model experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.module2.minimal_pairs import (
    audit_triplets, build_minimal_pairs, execute_tree, load_development_trees,
    rule_applicability, split_guidelines,
)


def fixture_records(count=8):
    records = []
    for i in range(count):
        root = {"index": 1, "description": f"Does source condition {i} hold?",
                "yes": {"index": 2, "description": f"Does decisive condition {i} hold?",
                    "yes": {"index": 3, "leaf": True, "conclusion": f"Do fictional action X{i}."},
                    "no": {"index": 4, "leaf": True, "conclusion": f"Do fictional action Y{i}."}},
                "no": {"index": 5, "leaf": True, "conclusion": "Outside this fictional group."}}
        records.append({"filename": f"tree{i}", "guideline_id": i, "language": "en",
                        "text": json.dumps({"decision_tree": root})})
    return records


class MinimalPairTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.audit = build_minimal_pairs(fixture_records(), seed=17)

    def test_fact_flip_reaches_true_sibling_and_neutral_is_unknown(self):
        root = json.loads(fixture_records(1)[0]["text"])["decision_tree"]
        for row in [r for r in self.rows if r["tree_id"] == "tree0"]:
            pos = execute_tree(root, row["patient_facts"]["positive"])
            neg = execute_tree(root, row["patient_facts"]["negative"])
            self.assertEqual({pos, neg}, {3, 4})
            self.assertIsNone(execute_tree(root, row["patient_facts"]["neutral"]))
            self.assertEqual(pos, row["rules"][row["target_rule_id"]]["leaf_index"])

    def test_complete_balancing_with_no_guideline_leakage(self):
        self.assertEqual(len(self.rows), 64)
        self.assertEqual(self.audit["partitions"], {
            "fit": {"guidelines": 4, "triplets": 32},
            "select": {"guidelines": 2, "triplets": 16},
            "confirm": {"guidelines": 2, "triplets": 16}})
        for tree in {r["tree_id"] for r in self.rows}:
            block = [r for r in self.rows if r["tree_id"] == tree]
            for branch in (True, False):
                self.assertEqual(sum(r["controls"]["target_branch"] == branch for r in block), 4)
            for target_id in ("A", "B"):
                actions = [r["rules"][target_id]["action"] for r in block if r["target_rule_id"] == target_id]
                self.assertEqual(len(set(actions)), 2)

    def test_guideline_identity_groups_multiple_trees(self):
        records = fixture_records()
        records[1]["guideline_id"] = records[0]["guideline_id"]
        rows, _ = build_minimal_pairs(records, max_trees_per_group=2)
        splits = {r["split"] for r in rows if r["tree_id"] in {"tree0", "tree1"}}
        self.assertEqual(len(splits), 1)

    def test_order_and_whole_guide_are_fixed_within_triplet(self):
        for row in self.rows:
            guides = []
            for variant in ("positive", "negative", "neutral"):
                spans, text = row["spans"][variant], row["prompts"][variant]
                self.assertLess(spans["patient"][0][1], spans["question"][0][0])
                for rule in spans["rules"].values():
                    self.assertLess(rule["condition"][0][1], rule["id"][0][0])
                    self.assertLess(rule["id"][0][1], rule["action"][0][0])
                guides.append(text[slice(*spans["guide"][0])])
            self.assertEqual(len(set(guides)), 1)

    def test_rejects_test_data_and_missing_provenance(self):
        with self.assertRaises(ValueError):
            build_minimal_pairs(fixture_records(), source_split="test")
        records = fixture_records()
        records[0]["split"] = "test"
        with self.assertRaises(ValueError):
            build_minimal_pairs(records)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test" / "decision_trees.jsonl"
            with self.assertRaises(ValueError):
                load_development_trees(path)

    def test_audit_detects_nonminimal_change_and_partial_balancing(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["patient_facts"]["negative"]["1"] = False
        with self.assertRaises(ValueError):
            audit_triplets(rows)
        with self.assertRaises(ValueError):
            audit_triplets(self.rows[:-1])

    def test_audit_detects_cross_partition_leakage(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "confirm" if rows[0]["split"] != "confirm" else "fit"
        with self.assertRaises(ValueError):
            audit_triplets(rows)

    def test_false_condition_dominates_unknown(self):
        conditions = [{"node_index": "1", "required": True}, {"node_index": "2", "required": True}]
        self.assertIs(rule_applicability(conditions, {"1": False, "2": None}), False)
        self.assertIsNone(rule_applicability(conditions, {"1": True, "2": None}))

    def test_deterministic_selection_does_not_depend_on_file_order(self):
        records = fixture_records()
        rows, audit = build_minimal_pairs(reversed(records), seed=17)
        self.assertEqual(rows, self.rows)
        self.assertEqual(audit["group_split_assignments"], self.audit["group_split_assignments"])
        self.assertEqual(split_guidelines(["c", "b", "a"]), split_guidelines(["a", "b", "c"]))

    def test_repeated_predicate_and_identical_action_are_not_eligible(self):
        records = fixture_records()
        tree = json.loads(records[0]["text"])
        tree["decision_tree"]["yes"]["description"] = tree["decision_tree"]["description"]
        records[0]["text"] = json.dumps(tree)
        rows, audit = build_minimal_pairs(records)
        self.assertNotIn("tree0", {row["tree_id"] for row in rows})
        self.assertEqual(audit["skipped"]["no_eligible_terminal_siblings"], 1)


if __name__ == "__main__":
    unittest.main()
