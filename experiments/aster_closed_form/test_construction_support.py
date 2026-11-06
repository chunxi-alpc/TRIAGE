"""Necessary checks for C state selection and synchronized source text."""
import copy
import unittest

from .construction_support import repair_condition_state, source_edit_text


def fixture():
    atom = lambda field, op, value: {"field": field, "op": op, "value": value}
    group = {
        "ancestors": [{"gid": "G0", "condition": "Age >=18?", "label": 1}],
        "definitions": [{"quote": "The QTc criterion is QTc >500 ms.",
                         "source_file": "/source/definition500.md"}],
        "guideline": {"groups": [
            {"group_id": "G0", "rules": [
                {"rule_id": "R0Y", "condition": "Age >=18?（判断为是）", "action": "继续评估规则组 G1。"},
                {"rule_id": "R0N", "condition": "Age >=18?（判断为否）", "action": "L2：Age pathway."},
            ]},
            {"group_id": "G1", "rules": [
                {"rule_id": "R1Y", "condition": "QTc >500 ms?（判断为是）", "action": "L3：Avoid the drug when QTc >500 ms."},
                {"rule_id": "R1N", "condition": "QTc >500 ms?（判断为否）", "action": "L4：No QTc contraindication."},
            ]},
        ]},
    }
    value = {
        "facts": [{"age": 40, "qtc": 480}, {"age": 40, "qtc": 520}],
        "predicate": atom("qtc", ">", 500),
        "ancestor_predicates": [{"gid": "G0", "predicate": atom("age", ">=", 18)}],
        "condition_change": {
            "valid": True, "base_state": 0,
            "edits": [{"old": "QTc >500 ms", "new": "QTc >520 ms"}],
            "predicate": atom("qtc", ">", 520),
        },
    }
    return group, value


class ConstructionSupportTests(unittest.TestCase):
    def test_true_side_flip_repairs_only_state_and_provenance(self):
        group, value = fixture()
        before = copy.deepcopy(value)
        self.assertTrue(repair_condition_state(group, value))
        self.assertEqual(value["condition_change"]["base_state"], 1)
        self.assertTrue(value["condition_change"]["valid"])
        self.assertEqual(value["facts"], before["facts"])
        self.assertEqual(value["predicate"], before["predicate"])
        self.assertEqual(value["condition_change"]["predicate"], before["condition_change"]["predicate"])
        self.assertEqual(value["condition_change"]["edits"], before["condition_change"]["edits"])
        self.assertEqual(value["condition_change"]["repaired_base_state"]["modified_labels"], [0, 0])

    def test_neither_side_flips_or_already_valid_side_is_unchanged(self):
        group, value = fixture()
        value["condition_change"]["predicate"] = copy.deepcopy(value["predicate"])
        before = copy.deepcopy(value)
        self.assertFalse(repair_condition_state(group, value))
        self.assertEqual(value, before)
        group, value = fixture()
        value["condition_change"]["base_state"] = 1
        before = copy.deepcopy(value)
        self.assertFalse(repair_condition_state(group, value))
        self.assertEqual(value, before)

    def test_rejected_change_cannot_be_reenabled(self):
        group, value = fixture()
        value["condition_change"]["valid"] = False
        before = copy.deepcopy(value)
        self.assertFalse(repair_condition_state(group, value))
        self.assertEqual(value, before)

    def test_original_truths_and_original_ancestors_remain_required(self):
        group, value = fixture()
        value["facts"][0]["qtc"] = 510
        self.assertFalse(repair_condition_state(group, value))
        group, value = fixture()
        value["facts"][0]["age"] = 12
        self.assertFalse(repair_condition_state(group, value))
        group, value = fixture()
        value["facts"][1]["age"] = 12
        self.assertFalse(repair_condition_state(group, value))

    def test_changed_ancestor_failure_is_not_accepted(self):
        group, value = fixture()
        value["condition_change"]["ancestor_predicates"] = [
            {"gid": "G0", "predicate": {"field": "age", "op": ">=", "value": 50}}
        ]
        self.assertFalse(repair_condition_state(group, value))
        self.assertEqual(value["condition_change"]["base_state"], 0)
        group, value = fixture()
        value["condition_change"]["edits"].append({"old": "Age >=18?", "new": "Age >=20?"})
        self.assertFalse(repair_condition_state(group, value))
        value["condition_change"]["ancestor_predicates"] = [
            {"gid": "G0", "predicate": {"field": "age", "op": ">=", "value": 20}}
        ]
        self.assertTrue(repair_condition_state(group, value))

    def test_source_text_includes_actions_for_reference_synchronization(self):
        group, _ = fixture()
        text = source_edit_text(group)
        self.assertIn("QTc >500 ms?", text)
        self.assertIn("The QTc criterion is QTc >500 ms.", text)
        self.assertIn("Avoid the drug when QTc >500 ms.", text)
        self.assertNotIn("/source/definition500.md", text)
        self.assertNotIn("R1Y", text)
        self.assertNotIn("L3", text)


if __name__ == "__main__":
    unittest.main()
