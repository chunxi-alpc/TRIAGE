"""CPU integrity checks; fabricated unit fixtures are never exported as results."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT), str(REPOSITORY)]
from joint.logic import evaluate_tree, reference_structure, reference_answer, evaluate_atom, sufficient_proof
from joint.context import full_context, prediction_input
from joint.structures import normalize_structure, StructureProvider, signed_targets
from joint.evaluate import score_arm, gate_risk_coverage, paired_comparison
from joint.model_backend import compare_zero_outputs

spec = importlib.util.spec_from_file_location("mechanism_builder", EXPERIMENT / "data/build_mechanism_set.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class LogicSpecificationTests(unittest.TestCase):
    def test_complete_binary_kleene_tables(self):
        states = ("T", "F", "U")
        and_table = (("T", "F", "U"), ("F", "F", "F"), ("U", "F", "U"))
        or_table = (("T", "T", "T"), ("T", "F", "U"), ("T", "U", "U"))
        for i, left in enumerate(states):
            for j, right in enumerate(states):
                labels = {"x": left, "y": right}
                self.assertEqual(evaluate_tree(builder.AND("x", "y"), labels), and_table[i][j])
                self.assertEqual(evaluate_tree(builder.OR("x", "y"), labels), or_table[i][j])
        self.assertEqual(evaluate_tree(builder.NOT("x"), {"x": "U"}), "U")

    def test_unknown_names_and_invalid_not_fail(self):
        with self.assertRaises(ValueError): evaluate_tree("invented", {})
        with self.assertRaises(ValueError): evaluate_tree({"op": "NOT", "args": []}, {})

    def test_units_subject_history_and_conflicts(self):
        atom = builder.atom_definitions(0)["C1"]
        fact = {"evidence_id": "e", "field": "age", "value": 216, "unit": "month", "subject": "patient", "time": "current"}
        self.assertEqual(evaluate_atom(atom, [fact])["state"], "T")
        self.assertEqual(evaluate_atom(atom, [fact | {"subject": "relative"}])["state"], "U")
        self.assertEqual(evaluate_atom(atom, [fact | {"time": "historical"}])["state"], "U")
        self.assertEqual(evaluate_atom(atom, [fact, fact | {"value": 204, "evidence_id": "other"}])["state"], "U")


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.splits, cls.info = builder.generate()

    def test_fixed_family_split_and_variant_isolation(self):
        self.assertEqual(len(self.cases), 576)
        self.assertEqual(len(self.splits), 24)
        variants, pairs = defaultdict(set), defaultdict(list)
        for case in self.cases:
            self.assertEqual(case["split"], self.splits[case["family_id"]])
            variants[case["variant_group_id"]].add(case["split"])
            if case["pair_id"]: pairs[case["pair_id"]].append(case)
        self.assertTrue(all(len(v) == 1 for v in variants.values()))
        self.assertTrue(all(len(p) == 2 for p in pairs.values()))
        old = {"copd", "glycemia", "hypoglycemia", "pneumonia", "sore_throat", "venous_thromboembolism",
               "atrial_fibrillation", "renal", "adiposity", "anaemia", "blood_pressure", "bone_density"}
        self.assertFalse(set(self.splits) & old)

    def test_full_context_and_target_independence(self):
        for case in self.cases:
            prompt, spans = full_context(case)
            self.assertEqual(prompt, case["prompt"])
            self.assertIn("Route Cedar", prompt)
            self.assertIn("Route Maple", prompt)
            self.assertIn("Route Juniper", prompt)
            for rule in case["rules"]:
                start, end = spans[rule["rule_id"]]
                self.assertEqual(prompt[start:end], rule["action"])
            if case["pair_eligible"]:
                self.assertEqual(type(case["applicable"]), bool)
                self.assertEqual(case["applicable"], case["gold_answer"] == case["target_answer"])

    def test_each_pair_changes_one_current_patient_atom(self):
        pairs = defaultdict(list)
        for case in self.cases:
            if case["pair_id"]: pairs[case["pair_id"]].append(case)
        for pair in pairs.values():
            self.assertEqual({c["reference_state"] for c in pair}, {"T", "F"})
            self.assertEqual(pair[0]["option_map"], pair[1]["option_map"])
            observation = lambda c: {f["field"]: (f["value"], f.get("unit")) for f in c["structured_facts"]
                                     if f["subject"] == "patient" and f["time"] == "current"}
            a, b = map(observation, pair)
            self.assertEqual(a.keys(), b.keys())
            self.assertEqual(sum(a[key] != b[key] for key in a), 1)

    def test_oracle_recomputation_and_sufficient_proofs(self):
        for case in self.cases:
            structure = reference_structure(case["rules"], case["structured_facts"])
            self.assertEqual(structure, case["reference_structure"])
            self.assertEqual(reference_answer(structure), case["gold_action_id"])
            atom_states = {cid: atom["state"] for cid, atom in structure["atoms"].items()}
            for rule, ref in zip(case["rules"], structure["rules"]):
                if ref["state"] == "U": continue
                tree = builder.AND(rule["logic"], builder.NOT(rule["exception_logic"]))
                witness = sufficient_proof(tree, atom_states, ref["state"])
                partial = {cid: state if cid in witness else "U" for cid, state in atom_states.items()}
                self.assertEqual(evaluate_tree(tree, partial), ref["state"])

    def test_prediction_input_cannot_see_reference_answers(self):
        case = self.cases[0]
        before = prediction_input(case)
        contaminated = copy.deepcopy(case)
        contaminated["gold_answer"] = "EXPOSED_GOLD"
        contaminated["reference_structure"] = {"secret": "EXPOSED_REFERENCE"}
        contaminated["structured_facts"] = [{"secret": "EXPOSED_FACTS"}]
        self.assertEqual(before, prediction_input(contaminated))
        self.assertEqual(set(before), {"guideline", "patient", "question"})
        from src.module1.care_v6 import validate_v6_input
        for case in self.cases:
            validate_v6_input(**prediction_input(case))

    def test_reference_positive_policy_is_separate(self):
        case = self.cases[1]
        full, _ = StructureProvider("reference", include_reference_diagnostic=True).predict(case)
        positive, _ = StructureProvider("reference_positive", include_reference_diagnostic=True).predict(case)
        self.assertEqual([r["state"] for r in full["rules"]], [r["state"] for r in positive["rules"]])
        self.assertTrue(any(t["sign"] < 0 for t in signed_targets(full, case["action_spans"])))
        self.assertTrue(all(t["sign"] > 0 for t in signed_targets(positive, case["action_spans"])))

    def test_parser_failures_and_original_gates_are_preserved(self):
        case = self.cases[0]
        for failed in ("invalid_output", "ambiguous_target", "invalid_label", "truncated", "ERROR", "unparsed"):
            raw = {"rules": [{"rule_id": r["rule_id"], "applicability": "F",
                               "allow_attention_intervention": False, "confidence": None,
                               "conditions": [{"condition_id": "C1", "trialmatch": {"parse_status": failed}}]}
                              for r in case["rules"]]}
            result = normalize_structure(raw, case, "v6")
            self.assertTrue(result["parse_failure"])
            self.assertEqual(len(result["condition_parse_failures"]), 3)
            self.assertEqual(signed_targets(result, case["action_spans"]), [])

    def test_missing_outputs_remain_in_gate_denominator(self):
        cases = self.cases[:2]
        structure = copy.deepcopy(cases[0]["reference_structure"])
        fixture = {"case_id": cases[0]["case_id"], "source": "unit_fixture", "mode": "structured_prompt",
                   "status": "completed", "is_model_prediction": True, "structure": structure}
        metric = gate_risk_coverage(cases, [fixture])
        self.assertEqual(metric["evaluated_rule_opportunities"], 6)
        self.assertEqual(metric["missing_or_failed_model_outputs"], 1)
        self.assertEqual(metric["curve"][0]["coverage"], .5)

    def test_blocked_or_oracle_only_records_do_not_create_model_metrics(self):
        cases = self.cases[:2]
        fake = [{"case_id": c["case_id"], "status": "completed", "is_model_prediction": False,
                 "prediction": c["gold_answer"]} for c in cases]
        output = score_arm(cases, fake)
        self.assertEqual(output["status"], "blocked_or_not_run")
        self.assertIsNone(output["case_accuracy"])

    def test_zero_control_checks_logits_even_when_probabilities_match(self):
        original = {"candidate_logits": [1.0, 1.0, 1.0, 1.0],
                    "candidate_probabilities": [.25, .25, .25, .25]}
        self.assertEqual(compare_zero_outputs(original, original)["status"], "passed")
        shifted = original | {"candidate_logits": [2.0, 2.0, 2.0, 2.0]}
        audit = compare_zero_outputs(original, shifted)
        self.assertEqual(audit["status"], "failed_output_mismatch")
        self.assertEqual(audit["candidate_logit_max_difference"], 1.0)
        self.assertEqual(audit["candidate_probability_max_difference"], 0.0)

    def test_zero_control_failure_cannot_be_scored_as_an_intervention_effect(self):
        cases = self.cases[:2]
        valid = [{"case_id": c["case_id"], "status": "completed", "is_model_prediction": True,
                  "prediction": c["gold_answer"]} for c in cases]
        invalid = [valid[0], {"case_id": cases[1]["case_id"], "status": "zero_control_failed",
                              "is_model_prediction": False}]
        metric = score_arm(cases, invalid)
        self.assertEqual(metric["status"], "invalid_zero_control")
        self.assertIsNone(metric["case_accuracy"])
        comparison = paired_comparison(cases, valid, invalid)
        self.assertEqual(comparison["status"], "invalid_zero_control")
        self.assertIsNone(comparison["paired_accuracy_difference"])

    def test_harmonic_accuracy_is_not_pair_accuracy(self):
        cases = [self.cases[i] for i in (0, 1, 6, 7)]
        fixture = []
        for index, case in enumerate(cases):
            guess = case["gold_answer"] if index in (0, 3) else next(c for c in "ABCD" if c != case["gold_answer"])
            fixture.append({"case_id": case["case_id"], "status": "completed",
                            "is_model_prediction": True, "prediction": guess})
        metric = score_arm(cases, fixture)
        self.assertEqual(metric["H_Acc"], .5)
        self.assertEqual(metric["pair_accuracy"], 0.0)
        self.assertEqual(metric["case_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
