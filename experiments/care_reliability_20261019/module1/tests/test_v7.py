"""Semantic properties and concrete regressions; no model/GPU dependency."""
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_modules
v6, v7 = load_modules()


def atom(cid, state, support=False, refute=False, numeric=False):
    span = {"sentence_id": 1, "start": 0, "end": 4, "text": "Fact"}
    return {"condition_id": cid, "applicability": state,
            "support_evidence": [span] if support else [], "refute_evidence": [span] if refute else [],
            "numeric": {"status": state, "reason": "deterministic_numeric_comparison", "evidence": [span]} if numeric else None}


def rule(expr, exceptions=None, exception_logic=None):
    names = sorted(set(x for x in repr(expr).replace("'", " ").split() if x in {"a", "b", "c", "d"}))
    # Conditions supplied explicitly in each proof test; unused declarations
    # are harmless and let the AST, not a list order, determine a proof.
    value = {"conditions": [{"condition_id": x} for x in "abcd"], "logic": expr, "exceptions": exceptions or []}
    if exception_logic:
        value["exception_logic"] = exception_logic
    return value


class ProofTests(unittest.TestCase):
    def test_proofs_exact_against_exhaustive_partial_assignments(self):
        expressions = [
            {"op": "AND", "args": ["a", "b"]},
            {"op": "OR", "args": ["a", {"op": "AND", "args": ["b", "c"]}]},
            {"op": "NOT", "args": [{"op": "OR", "args": ["a", "b"]}]},
            {"op": "AND", "args": [{"op": "OR", "args": ["a", "b"]}, {"op": "NOT", "args": ["c"]}]},
            {"op": "OR", "args": [{"op": "AND", "args": ["a", "b"]}, {"op": "AND", "args": ["a", "c"]}, "d"]},
        ]
        for expr in expressions:
            for values in itertools.product("TFU", repeat=4):
                states = dict(zip("abcd", values))
                target = v6.eval_logic(expr, states)
                actual = v7.sufficient_proofs(expr, states)
                got = {frozenset((x["condition_id"], x["required_state"]) for x in proof) for proof in actual["proofs"]}
                exhaustive = set()
                known = [x for x in states if states[x] != "U"]
                if target != "U":
                    for bits in itertools.product((False, True), repeat=len(known)):
                        keep = {x for x, yes in zip(known, bits) if yes}
                        partial = {x: states[x] if x in keep else "U" for x in states}
                        if v6.eval_logic(expr, partial) == target:
                            exhaustive.add(frozenset((x, states[x]) for x in keep))
                expected = {p for p in exhaustive if not any(q < p for q in exhaustive)}
                self.assertEqual(got, expected, (expr, states))

    def test_and_requires_both_supports(self):
        expr = {"op": "AND", "args": ["a", "b"]}
        gate = v7.rule_evidence_gate(rule(expr), [atom("a", "T", support=True), atom("b", "T")])
        self.assertFalse(gate["allow_attention_intervention"])
        good = v7.rule_evidence_gate(rule(expr), [atom("a", "T", support=True), atom("b", "T", support=True)])
        self.assertTrue(good["allow_attention_intervention"])

    def test_or_can_use_an_alternative_grounded_proof(self):
        gate = v7.rule_evidence_gate(rule({"op": "OR", "args": ["a", "b"]}), [atom("a", "T"), atom("b", "T", support=True)])
        self.assertTrue(gate["allow_attention_intervention"])
        self.assertEqual(gate["selected_proof"], [{"condition_id": "b", "required_state": "T"}])

    def test_unneeded_disputed_or_branch_does_not_block_proof(self):
        gate = v7.rule_evidence_gate(rule({"op": "OR", "args": ["a", "b"]}), [atom("a", "T", support=True), atom("b", "U", support=True, refute=True)])
        self.assertTrue(gate["allow_attention_intervention"])

    def test_not_requires_refuting_evidence(self):
        expr = {"op": "NOT", "args": ["a"]}
        self.assertTrue(v7.rule_evidence_gate(rule(expr), [atom("a", "F", refute=True)])["allow_attention_intervention"])
        self.assertFalse(v7.rule_evidence_gate(rule(expr), [atom("a", "F", support=True)])["allow_attention_intervention"])

    def test_exclusion_or_requires_every_exception_false(self):
        r = rule("a", [{"condition_id": "b"}, {"condition_id": "c"}], {"op": "OR", "args": ["b", "c"]})
        rows = [atom("a", "T", support=True), atom("b", "F", refute=True), atom("c", "F")]
        self.assertFalse(v7.rule_evidence_gate(r, rows)["allow_attention_intervention"])
        rows[-1] = atom("c", "F", refute=True)
        self.assertTrue(v7.rule_evidence_gate(r, rows)["allow_attention_intervention"])

    def test_exclusion_and_needs_only_a_decisive_false(self):
        r = rule("a", [{"condition_id": "b"}, {"condition_id": "c"}], {"op": "AND", "args": ["b", "c"]})
        rows = [atom("a", "T", support=True), atom("b", "F", refute=True), atom("c", "U")]
        self.assertTrue(v7.rule_evidence_gate(r, rows)["allow_attention_intervention"])

    def test_unconditional_main_with_exception_uses_exception_evidence(self):
        r = {"conditions": [], "explicitly_unconditional": True, "exceptions": [{"condition_id": "e"}],
             "exception_logic": {"op": "OR", "args": ["e"]}}
        gate = v7.rule_evidence_gate(r, [atom("e", "F", refute=True)])
        self.assertTrue(gate["allow_attention_intervention"])
        self.assertEqual(gate["selected_proof"], [{"condition_id": "e", "required_state": "F"}])

    def test_false_unknown_and_conflict_do_not_open_actions(self):
        for r in [atom("a", "F", refute=True), atom("a", "U"), atom("a", "T", support=True, refute=True)]:
            self.assertFalse(v7.rule_evidence_gate(rule("a"), [r])["allow_attention_intervention"])

    def test_numeric_only_evidence_is_not_missing(self):
        check = v7.condition_evidence(atom("a", "T", numeric=True))
        self.assertEqual(check["evidence_status"], "supported")
        self.assertTrue(check["allowed"])

    def test_invalid_offsets_fail_closed(self):
        r = atom("a", "T", support=True)
        r["support_evidence"][0]["end"] = 99
        self.assertFalse(v7.condition_evidence(r, patient_text="Fact")["allowed"])

    def test_proof_limit_closes_gate_without_partial_success(self):
        r = rule({"op": "OR", "args": ["a", "b"]})
        gate = v7.rule_evidence_gate(r, [atom("a", "T", support=True), atom("b", "T", support=True)], max_proofs=1)
        self.assertFalse(gate["allow_attention_intervention"])
        self.assertFalse(gate["proofs_complete"])

    def test_malformed_logic_does_not_silently_become_unknown(self):
        for expr in [{"op": "NOT", "args": ["a", "b"]}, {"op": "XOR", "args": ["a", "b"]}, "missing"]:
            with self.assertRaises(ValueError):
                v7.sufficient_proofs(expr, {"a": "T", "b": "F"})


class IntegrationTests(unittest.TestCase):
    def test_gold_changes_cannot_change_normalized_model_input(self):
        from run_inference import normalized_input, digest_json
        item = {"case_id": "x", "patient": "Fact", "question": "Which route?",
                "rules": [{"rule_id": "r", "action": "Route Cedar", "conditions": [{"condition_id": "a", "text": "Fact", "gold_state": "T"}]}],
                "gold": {"r": "T"}, "structured_facts": {"a": True}, "reference_structure": {"r": "T"}}
        expected = digest_json(normalized_input(item))
        item["gold"] = {"r": "F"}
        item["structured_facts"] = {"a": False}
        item["reference_structure"] = {"r": "F"}
        item["rules"][0]["conditions"][0]["gold_state"] = "F"
        self.assertEqual(digest_json(normalized_input(item)), expected)

    def test_medical_unknown_is_not_a_parse_failure(self):
        from run_inference import status_and_compact
        graph = {"rules": [{"rule_id": "r", "applicability": "U", "support_evidence": [], "refute_evidence": [],
            "allow_attention_intervention": False, "exceptions": [], "conditions": [{"trialmatch": {"raw_label": "Unclear"}}]}]}
        self.assertEqual(status_and_compact(graph, [{"rule_id": "r"}])[0], "ok")
        graph["rules"][0]["conditions"][0]["trialmatch"]["raw_label"] = None
        self.assertEqual(status_and_compact(graph, [{"rule_id": "r"}])[0], "parse_failure")

    def test_cross_module_sibling_owner_cannot_override_patient(self):
        condition = {"condition_id": "a", "field": "age", "numeric": {"field": "age", "operator": ">=", "value": 18}}
        for text in ["The patient's sibling's current age is 24 years.",
                     "The patient's mentor's current age is 24 years.",
                     "Another patient's current age is 24 years."]:
            numeric = v7.scoped_numeric_observation(condition, v6.sentences_with_offsets(text))
            self.assertEqual(numeric["status"], "U", text)
            self.assertEqual(numeric["evidence"], [], text)

    def test_complete_run_numeric_only_overrides_and_opens_gate(self):
        class Trial:
            def judge(self, *args, **kwargs):
                return {"state": "F", "raw_label": "Not Met", "parse_status": "accepted"}
        class NLI:
            def predict(self, pairs, **kwargs):
                return [{"relation": "neutral", "scores": {"neutral": 1.0}, "class_id": 1} for _ in pairs]
        engine = v7.CAREModule1V7(Trial(), NLI())
        prepared = [{"rule_id": "r", "text": "Age at least 18 years", "action": "Route Cedar",
            "conditions": [{"condition_id": "a", "text": "Age at least 18 years", "field": "age",
                            "numeric": {"field": "age", "operator": ">=", "value": 18, "unit": "year"}}]}]
        result = engine.run(prepared, "The patient's current age is 240 months.", "Which symbolic route?")
        self.assertEqual(result["rules"][0]["applicability"], "T")
        self.assertTrue(result["rules"][0]["allow_attention_intervention"])
        self.assertEqual(result["rules"][0]["conditions"][0]["evidence_status"], "supported")
        self.assertEqual(result["action_gate"]["r"], 1.0)
        self.assertNotIn("logic", prepared[0])


if __name__ == "__main__":
    unittest.main()
