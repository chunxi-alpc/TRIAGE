"""Focused numeric C tests: boundaries, unchanged facts, references, ancestors."""
import copy
import unittest

from .common import evaluate
from .construct import validate
from .data import replace_all
from .numeric_conditions import numeric_condition_candidate


def fixture():
    atom = lambda field, op, number: {"field": field, "op": op, "value": number}
    group = {
        "cohort": "main", "gid": "G30",
        "definitions": [{"quote": "The blood-pressure goal is DBP<80 mmHg.", "source_file": "guide80.md"}],
        "ancestors": [{"gid": "G0", "condition": "Age >=18?", "label": 1}],
        "guideline": {"groups": [
            {"group_id": "G0", "rules": [
                {"rule_id": "R0Y", "condition": "Age >=18?（判断为是）", "action": "继续评估规则组 G30。"},
                {"rule_id": "R0N", "condition": "Age >=18?（判断为否）", "action": "L2：Other."},
            ]},
            {"group_id": "G30", "parent_group_id": "G0", "parent_rule_id": "R0Y",
             "ancestor_conditions": ["Age >=18?（判断为是）"], "rules": [
                 {"rule_id": "R30Y", "condition": "DBP <80 mmHg?（判断为是）",
                  "action": "L31：Goal DBP<80 mmHg; an unrelated measurement <800."},
                 {"rule_id": "R30N", "condition": "DBP <80 mmHg?（判断为否）",
                  "action": "L32：Goal DBP<80 mmHg."},
             ]},
        ]},
    }
    value = {
        "facts": [{"age": 40, "dbp": 85}, {"age": 40, "dbp": 75}],
        "predicate": atom("dbp", "<", 80),
        "ancestor_predicates": [{"gid": "G0", "predicate": atom("age", ">=", 18)}],
    }
    return group, value


class NumericConditionTests(unittest.TestCase):
    def test_less_than_80_uses_75_and_keeps_fixed_patient(self):
        group, value = fixture()
        before = copy.deepcopy(value)
        candidate = numeric_condition_candidate(group, value)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["numeric_derivation"]["new_threshold"], 75)
        self.assertEqual(candidate["base_state"], 1)
        self.assertEqual([int(evaluate(candidate["predicate"], f)) for f in value["facts"]], [0, 0])
        self.assertEqual(value, before)
        value["condition_change"] = candidate
        self.assertTrue(validate(group, value)["C_valid"])

    def test_all_comparison_references_update_without_800_or_ID_changes(self):
        group, value = fixture()
        candidate = numeric_condition_candidate(group, value)
        guide = replace_all(group["guideline"], candidate["edits"])
        definitions = replace_all(group["definitions"], candidate["edits"])
        self.assertEqual(guide["groups"][1]["group_id"], "G30")
        self.assertEqual(guide["groups"][0]["rules"][0]["action"], "继续评估规则组 G30。")
        self.assertIn("DBP<75", guide["groups"][1]["rules"][0]["action"])
        self.assertIn("<800", guide["groups"][1]["rules"][0]["action"])
        self.assertIn("DBP<75", definitions[0]["quote"])
        self.assertEqual(definitions[0]["source_file"], "guide80.md")

    def test_boolean_values_or_no_explicit_current_threshold_are_ineligible(self):
        group, value = fixture()
        value["facts"][0]["dbp"] = False
        self.assertIsNone(numeric_condition_candidate(group, value))
        group, value = fixture()
        for rule in group["guideline"]["groups"][1]["rules"]:
            rule["condition"] = rule["condition"].replace("<80", "within target")
        self.assertIsNone(numeric_condition_candidate(group, value))

    def test_full_conjunction_other_leaf_is_unchanged(self):
        group, value = fixture()
        value["predicate"] = {"op": "and", "args": [
            value["predicate"], {"op": ">=", "field": "age", "value": 18}
        ]}
        candidate = numeric_condition_candidate(group, value)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["predicate"]["args"][1], value["predicate"]["args"][1])
        self.assertNotEqual(candidate["predicate"]["args"][0], value["predicate"]["args"][0])

    def test_touching_ancestor_without_same_field_interpretation_is_rejected(self):
        group, value = fixture()
        group["ancestors"][0]["condition"] = "Age <80?"
        value["ancestor_predicates"][0]["predicate"] = {"field": "age", "op": "<", "value": 80}
        for rule in group["guideline"]["groups"][0]["rules"]:
            rule["condition"] = rule["condition"].replace("Age >=18?", "Age <80?")
        self.assertIsNone(numeric_condition_candidate(group, value))

    def test_threshold_that_breaks_ancestor_truth_is_rejected(self):
        group, value = fixture()
        group["ancestors"][0]["condition"] = "DBP <80?"
        value["facts"] = [{"age": 40, "dbp": 60, "flag": False},
                          {"age": 40, "dbp": 70, "flag": True}]
        value["predicate"] = {"op": "and", "args": [
            {"field": "dbp", "op": "<", "value": 80},
            {"field": "flag", "op": "==", "value": True}
        ]}
        value["ancestor_predicates"][0]["predicate"] = {"field": "dbp", "op": "<", "value": 80}
        self.assertIsNone(numeric_condition_candidate(group, value))


if __name__ == "__main__":
    unittest.main()
