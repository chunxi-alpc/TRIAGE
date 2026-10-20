"""CPU-only A1 mainline API regression; model generation is stubbed."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import src.module1 as module1
from src.module1.semantic_a1 import CAREModule1A1, signal_targets, _EXAMPLES_PATH


def _rules():
    return [
        {"rule_id": "R1", "text": "If the patient has A, assign Route Cedar.", "action": "Route Cedar"},
        {"rule_id": "R2", "text": "If the patient has B, assign Route Maple.", "action": "Route Maple"},
        {"rule_id": "R3", "text": "If the patient has C, assign Route Juniper.", "action": "Route Juniper"},
    ]


def _raw(states):
    return json.dumps({"rules": [
        {"rule_id": f"R{i}", "state": state, "evidence_quote": "Patient has A.",
         "reason_short": "Evidence supports this outcome."}
        for i, state in enumerate(states, 1)
    ]})


class A1InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.engine = CAREModule1A1(adapter=object(), examples={"demonstrations": []})
        self.case = dict(guideline=_rules(), patient="Patient has A.", question="Which route?")

    def test_a1_is_mainline_factory_but_v6_is_compat(self):
        self.assertIs(module1.create_module1, module1.create_module1_a1)
        self.assertTrue(callable(module1.create_module1_v6))
        self.assertTrue(callable(module1.run_v6))

    def test_frozen_example_asset_matches_original_a1(self):
        payload = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
        self.assertEqual((payload["count_demonstrations"],
                          payload["count_patient_narratives"]), (3, 4))
        digest = hashlib.sha256(_EXAMPLES_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, "3fe71cdc1573c69a422c7892491a62fa99dc45ac538509e169c3adeac9fa08a8")

    def test_signs_and_action_sources(self):
        with patch.object(self.engine, "_generate", return_value=_raw(["T", "F", "U"])) as gen:
            result = self.engine.run(**self.case)
        gen.assert_called_once()
        self.assertEqual(result["schema_version"], "module1.a1")
        self.assertEqual([x["sign"] for x in result["rules"]], [1, -1, 0])
        self.assertEqual([x["action_text"] for x in signal_targets(result)],
                         ["Route Cedar", "Route Maple"])
        self.assertEqual(result["rules"][0]["optional_evidence_char_span"], [0, 14])

    def test_missing_json_is_technical_failure_not_u(self):
        with patch.object(self.engine, "_generate", return_value="broken") as gen:
            result = self.engine.run(**self.case)
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(result["status"], "parse_failure")
        self.assertFalse(result["has_module2_signal"])
        self.assertTrue(all(x["state"] is None and x["technical_failure"] for x in result["rules"]))
        self.assertEqual(signal_targets(result), [])

    def test_invalid_guideline_or_duplicate_rule_ids_stops_before_model(self):
        for invalid in ("long unstructured document", [], [_rules()[0]] * 2):
            with self.subTest(invalid=invalid):
                with patch.object(self.engine, "_generate") as gen, self.assertRaises(ValueError):
                    self.engine.run(guideline=invalid, patient="Patient has A.", question="Which?")
                gen.assert_not_called()

    def test_non_a1_result_not_accepted_as_signals(self):
        with self.assertRaises(ValueError):
            signal_targets({"schema_version": "module1.v6", "rules": []})


if __name__ == "__main__":
    unittest.main()
