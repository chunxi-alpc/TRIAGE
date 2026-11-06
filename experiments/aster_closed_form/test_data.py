"""Truth, ancestor and reference-consistency checks; no model calls."""
import copy
import unittest
from pathlib import Path

from .common import DEFAULT_ROOT, evaluate, rows
from .construct import validate
from .data import ancestors, condition, prompt, replace_all, set_condition


def data_fixture():
    guide = {"groups": [
        {"group_id": "G0", "ancestor_conditions": [], "rules": [
            {"rule_id": "R0Y", "condition": "Age >=18 years?（判断为是）", "action": "继续评估规则组 G30。"},
            {"rule_id": "R0N", "condition": "Age >=18 years?（判断为否）", "action": "L2：Other pathway."},
        ]},
        {"group_id": "G30", "parent_group_id": "G0", "parent_rule_id": "R0Y",
         "ancestor_conditions": ["Age >=18 years?（判断为是）"],
         "rules": [
            {"rule_id": "R30Y", "condition": "Creatinine clearance >=30 mL/min or dialysis?（判断为是）",
             "action": "L31：Treatment A."},
            {"rule_id": "R30N", "condition": "Creatinine clearance >=30 mL/min or dialysis?（判断为否）",
             "action": "L32：Treatment B."},
        ]},
    ]}
    leaf = lambda field, op, value: {"field": field, "op": op, "value": value}
    expr = {"op": "or", "args": [leaf("clearance", ">=", 30), leaf("dialysis", "==", True)]}
    ancestor = leaf("age", ">=", 18)
    changed = {"op": "or", "args": [leaf("clearance", ">=", 10), leaf("dialysis", "==", True)]}
    group = {
        "group_id": "fixture", "gid": "G30", "cohort": "main", "guideline": guide,
        "definitions": [{"quote": "Renal eligibility means creatinine clearance >=30 mL/min or dialysis.",
                         "source_file": "/source/guideline30.md", "block_id": 30}],
        "ancestors": [{"gid": "G0", "condition": "Age >=18 years?", "label": 1}],
    }
    value = {
        "facts": [{"age": 40, "clearance": 20, "dialysis": False},
                  {"age": 40, "clearance": 40, "dialysis": False}],
        "predicate": expr,
        "ancestor_predicates": [{"gid": "G0", "predicate": ancestor}],
        "condition_change": {"valid": True, "base_state": 0,
                             "edits": [{"old": ">=30", "new": ">=10"}], "predicate": changed},
        "clinical_background": {"valid": True, "texts": ["No prior ankle fracture.", "An ankle fracture healed five years ago."],
                                "changed_fields": ["old_ankle_fracture"]},
        "treatment_preference": {"valid": True, "texts": ["Prefers conservative care.", "Prefers active care."],
                                 "changed_fields": ["treatment_preference"]},
    }
    return group, value


class DataTests(unittest.TestCase):
    def test_complete_condition_and_ancestor_truth(self):
        group, value = data_fixture()
        result = validate(group, value)
        self.assertEqual(result["original_truths"], [0, 1])
        self.assertEqual(result["ancestors_preserved"], 1)
        self.assertTrue(value["condition_change"]["valid"])
        # The unchanged disjunct prevents a false/true pair if it is true.
        invalid = copy.deepcopy(value)
        invalid["facts"][0]["dialysis"] = True
        with self.assertRaisesRegex(ValueError, "Complete predicate"):
            validate(group, invalid)

    def test_fact_flip_may_not_change_an_ancestor(self):
        group, value = data_fixture()
        value["facts"][1]["age"] = 12
        with self.assertRaisesRegex(ValueError, "ancestor"):
            validate(group, value)

    def test_background_field_used_by_current_or_ancestor_is_unavailable(self):
        group, value = data_fixture()
        value["clinical_background"]["changed_fields"] = ["dialysis"]
        value["treatment_preference"]["changed_fields"] = ["age"]
        validate(group, value)
        self.assertFalse(value["clinical_background"]["valid"])
        self.assertFalse(value["treatment_preference"]["valid"])

    def test_C_must_flip_the_full_condition(self):
        group, value = data_fixture()
        value["condition_change"]["predicate"] = copy.deepcopy(value["predicate"])
        validate(group, value)
        self.assertFalse(value["condition_change"]["valid"])

    def test_C_rejects_negative_state_indices(self):
        group, value = data_fixture()
        value["condition_change"]["base_state"] = -1
        validate(group, value)
        self.assertFalse(value["condition_change"]["valid"])

    def test_C_ancestor_edits_need_consistent_updated_predicates(self):
        group, value = data_fixture()
        value["condition_change"]["edits"].append({"old": ">=18", "new": ">=50"})
        validate(group, value)
        self.assertFalse(value["condition_change"]["valid"])
        group, value = data_fixture()
        value["condition_change"]["edits"].append({"old": ">=18", "new": ">=20"})
        value["condition_change"]["ancestor_predicates"] = [
            {"gid": "G0", "predicate": {"field": "age", "op": ">=", "value": 20}}
        ]
        validate(group, value)
        self.assertTrue(value["condition_change"]["valid"])

    def test_C_must_modify_condition_or_definition_not_metadata(self):
        group, value = data_fixture()
        value["condition_change"]["edits"] = [{"old": "guideline30.md", "new": "guideline10.md"}]
        validate(group, value)
        self.assertFalse(value["condition_change"]["valid"])

    def test_text_replacement_preserves_tree_identifiers_and_source_locations(self):
        group, _ = data_fixture()
        changed = replace_all(group["guideline"], [{"old": "30", "new": "10"}])
        self.assertEqual(changed["groups"][1]["group_id"], "G30")
        self.assertEqual(changed["groups"][1]["rules"][0]["rule_id"], "R30Y")
        self.assertEqual(changed["groups"][0]["rules"][0]["action"], "继续评估规则组 G30。")
        self.assertIn(">=10", changed["groups"][1]["rules"][0]["condition"])
        defs = replace_all(group["definitions"], [{"old": "30", "new": "10"}])
        self.assertIn(">=10", defs[0]["quote"])
        self.assertEqual(defs[0]["source_file"], group["definitions"][0]["source_file"])
        self.assertEqual(defs[0]["block_id"], 30)

    def test_C_references_are_updated_in_both_branches_definitions_and_prompt(self):
        group, value = data_fixture()
        edits = value["condition_change"]["edits"]
        guide = replace_all(group["guideline"], edits)
        definitions = replace_all(group["definitions"], edits)
        text = prompt(guide, definitions, "Age 40; clearance 20; no dialysis.", "G30")
        self.assertNotIn(">=30", text)
        self.assertIn(">=10", text)
        self.assertIn(">=10", guide["groups"][1]["rules"][1]["condition"])
        # set_condition likewise changes both branches and literal references.
        short = set_condition(guide, "G30", "Is renal eligibility satisfied?")
        self.assertEqual(condition({"guideline": short, "gid": "G30"}), "Is renal eligibility satisfied?")

    def test_definition_deletion_and_restoration_are_explicit_views(self):
        group, _ = data_fixture()
        guide = set_condition(group["guideline"], "G30", "Is renal eligibility satisfied?")
        original_defs = group["definitions"]
        changed_defs = replace_all(original_defs, [{"old": ">=30", "new": ">=10"}])
        patient = "Age 40; clearance 20; no dialysis."
        left_full = prompt(guide, original_defs, patient, "G30")
        right_full = prompt(guide, changed_defs, patient, "G30")
        self.assertNotEqual(left_full, right_full)
        self.assertEqual(prompt(guide, original_defs, patient, "G30", mode="direct"),
                         prompt(guide, changed_defs, patient, "G30", mode="direct"))
        self.assertIn(original_defs[0]["quote"],
                      prompt(guide, original_defs, patient, "G30", mode="definition_restore"))

    def test_available_prepared_groups_have_the_recorded_ancestor_chain(self):
        groups = rows(DEFAULT_ROOT/"data/groups_main.jsonl") + rows(DEFAULT_ROOT/"data/groups_definition.jsonl")
        if not groups:
            self.skipTest("No server-prepared dataset")
        for group in groups:
            with self.subTest(group=group["group_id"]):
                self.assertEqual(group["ancestors"], ancestors(group))

    def test_available_completed_constructions_have_the_required_truths(self):
        groups = {g["group_id"]: g for g in
                  rows(DEFAULT_ROOT/"data/groups_main.jsonl") + rows(DEFAULT_ROOT/"data/groups_definition.jsonl")}
        records = [r for r in rows(DEFAULT_ROOT/"data/constructions.jsonl") if r.get("status") == "complete"]
        if not records:
            self.skipTest("Construction has not produced complete records")
        for record in records:
            group = groups[record["group_id"]]
            value = record["value"]
            if group["cohort"] == "definition" and not value.get("definition_valid"):
                continue
            with self.subTest(group=group["group_id"]):
                self.assertEqual([int(evaluate(value["predicate"], facts)) for facts in value["facts"]], [0, 1])
                ancestor_rules = {a["gid"]: a["predicate"] for a in value["ancestor_predicates"]}
                for ancestor in group["ancestors"]:
                    self.assertEqual([int(evaluate(ancestor_rules[ancestor["gid"]], f)) for f in value["facts"]],
                                     [ancestor["label"]] * 2)


if __name__ == "__main__":
    unittest.main()
