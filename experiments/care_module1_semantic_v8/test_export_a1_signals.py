"""Small CPU-only smoke tests of the frozen A1 -> module-two signal bridge."""
from __future__ import annotations

import unittest
from export_a1_signals import export


class A1SignalExportTests(unittest.TestCase):
    def _case(self, name="c1"):
        return {"case_id": name, "split": "dev", "rules": [
            {"rule_id": "R1", "action": "Route Cedar"},
            {"rule_id": "R2", "action": "Route Maple"},
            {"rule_id": "R3", "action": "Route Juniper"},
        ], "gold": "not_model_input"}

    def _prediction(self, name="c1", states=("T", "F", "U")):
        return {"case_id": name, "arm": "A1", "status": "ok", "rules": [
            {"rule_id": f"R{i}", "state": state} for i, state in enumerate(states, 1)
        ]}

    def test_t_f_u_and_gold_independence(self):
        rows, meta = export([self._case()], [self._prediction()])
        self.assertEqual([r["sign"] for r in rows[0]["rules"]], [1, -1, 0])
        self.assertTrue(rows[0]["eligible_for_module2_signal_study"])
        self.assertFalse(meta["reference_labels_used"])
        self.assertEqual(meta["counts"]["eligible_cases"], 1)

    def test_no_signal_or_missing_prediction_still_kept(self):
        cases = [self._case("c1"), self._case("c2")]
        rows, meta = export(cases, [self._prediction("c1", ("U", "U", "U"))])
        self.assertEqual(len(rows), 2)
        self.assertEqual(meta["counts"]["no_signal_cases"], 2)
        self.assertEqual(rows[1]["prediction_status"], "missing_prediction")
        self.assertTrue(all(r["sign"] is None for r in rows[1]["rules"]))

    def test_reject_non_a1(self):
        p = self._prediction()
        p["arm"] = "A1_U"
        with self.assertRaises(ValueError):
            export([self._case()], [p])


if __name__ == "__main__":
    unittest.main()
