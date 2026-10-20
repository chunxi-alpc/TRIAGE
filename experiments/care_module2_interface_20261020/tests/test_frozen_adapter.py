"""Necessary contract checks for the frozen module-one routing boundary."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frozen_adapter import FrozenSignalAdapter, FrozenSignalError  # noqa: E402


def prediction(sample="C01_original", rule="V01", label="T", gate=True,
               evidence="supported"):
    return {
        "sample_id": sample,
        "rule_id": rule,
        "C": label,
        "allow_attention_intervention": gate,
        "evidence_status": evidence,
    }


class FrozenAdapterTests(unittest.TestCase):
    def test_modes_preserve_different_gate_semantics(self):
        rows = [
            prediction("a", label="T", gate=False),
            prediction("b", label="F", gate=False),
            prediction("c", label="F", gate=True),
            prediction("d", label="U", gate=True),
        ]
        adapter = FrozenSignalAdapter(rows)
        strict = adapter.adapt(rows, mode="strict_gate")
        diagnostic = adapter.adapt(rows, mode="label_only_diagnostic")
        self.assertEqual([r["route_sign"] for r in strict], [0, 0, -1, 0])
        self.assertEqual([r["route_sign"] for r in diagnostic], [1, -1, -1, 0])
        self.assertEqual(
            [r["frozen_allow_attention_intervention"] for r in diagnostic],
            [False, False, True, True],
        )
        self.assertTrue(all(r["saved_gate_applied"] for r in strict))
        self.assertTrue(all(not r["saved_gate_applied"] for r in diagnostic))

    def test_disputed_true_gate_is_preserved_without_repair(self):
        row = prediction(evidence="disputed", gate=True)
        output = FrozenSignalAdapter([row]).adapt([row], mode="strict_gate")[0]
        self.assertEqual(output["route_sign"], 1)
        self.assertEqual(output["frozen_evidence_status"], "disputed")
        self.assertTrue(output["frozen_allow_attention_intervention"])

    def test_no_source_or_decidability_is_inferred_for_any_label(self):
        rows = [prediction(label, label=label) for label in ("T", "F", "U")]
        adapter = FrozenSignalAdapter(rows)
        for mode in ("strict_gate", "label_only_diagnostic"):
            for result in adapter.adapt(rows, mode=mode):
                self.assertIsNone(result["source_target"])
                self.assertIsNone(result["decidability_target"])
                self.assertFalse(result["new_module1_inference"])

    def test_join_uses_both_ids_and_never_row_position(self):
        rows = [prediction("same", "V01", "T"),
                prediction("same", "V02", "F")]
        adapter = FrozenSignalAdapter(rows)
        output = adapter.adapt(list(reversed(rows)), mode="strict_gate")
        self.assertEqual([r["route_sign"] for r in output], [-1, 1])
        with self.assertRaisesRegex(FrozenSignalError, "missing frozen"):
            adapter.adapt([{"sample_id": "same", "rule_id": "V03"}],
                          mode="strict_gate", require_exact_keys=False)
        with self.assertRaisesRegex(FrozenSignalError, "missing frozen"):
            adapter.adapt([{"sample_id": "other", "rule_id": "V01"}],
                          mode="strict_gate", require_exact_keys=False)

    def test_duplicate_prediction_and_request_keys_fail(self):
        row = prediction()
        with self.assertRaisesRegex(FrozenSignalError, "duplicate key"):
            FrozenSignalAdapter([row, deepcopy(row)])
        adapter = FrozenSignalAdapter([row])
        with self.assertRaisesRegex(FrozenSignalError, "duplicate key"):
            adapter.adapt([row, deepcopy(row)], mode="strict_gate")

    def test_missing_and_invalid_keys_fail(self):
        for key in ("sample_id", "rule_id"):
            for value in (None, "", 12):
                with self.subTest(key=key, value=value):
                    row = prediction()
                    row[key] = value
                    with self.assertRaises(FrozenSignalError):
                        FrozenSignalAdapter([row])
            adapter = FrozenSignalAdapter([prediction()])
            request = prediction()
            del request[key]
            with self.assertRaises(FrozenSignalError):
                adapter.adapt([request], mode="strict_gate")

    def test_gate_is_mandatory_and_boolean_even_for_diagnostic(self):
        for value in (None, "false", "true", 0, 1):
            with self.subTest(value=value):
                row = prediction()
                row["allow_attention_intervention"] = value
                with self.assertRaisesRegex(FrozenSignalError, "saved boolean"):
                    FrozenSignalAdapter([row])
        row = prediction()
        del row["allow_attention_intervention"]
        with self.assertRaises(FrozenSignalError):
            FrozenSignalAdapter([row])

    def test_only_saved_C_controls_sign_and_inputs_are_unchanged(self):
        row = prediction(label="F", gate=True)
        row.update(A="T", B="T", conditions=[{"state": "T"}], label="T")
        original = deepcopy(row)
        adapter = FrozenSignalAdapter([row])
        request = {"sample_id": row["sample_id"], "rule_id": row["rule_id"],
                   "C": "T", "label": "T", "source_target": "context"}
        original_request = deepcopy(request)
        output = adapter.adapt([request], mode="strict_gate")[0]
        self.assertEqual(output["route_sign"], -1)
        self.assertIsNone(output["source_target"])
        self.assertEqual(row, original)
        self.assertEqual(request, original_request)
        row["C"] = "T"
        self.assertEqual(adapter.adapt([request], mode="strict_gate")[0]["route_sign"], -1)

    def test_no_silent_default_label(self):
        for value in (None, "unknown", True, ["T"]):
            with self.subTest(value=value):
                row = prediction()
                row["C"] = value
                with self.assertRaisesRegex(FrozenSignalError, "C must be"):
                    FrozenSignalAdapter([row])

    def test_exact_join_default_and_explicit_subset(self):
        rows = [prediction("a"), prediction("b")]
        adapter = FrozenSignalAdapter(rows)
        with self.assertRaisesRegex(FrozenSignalError, "unrequested frozen"):
            adapter.adapt(rows[:1], mode="strict_gate")
        self.assertEqual(len(adapter.adapt(rows[:1], mode="strict_gate",
                                           require_exact_keys=False)), 1)

    def test_empty_inputs_and_unknown_mode_fail(self):
        with self.assertRaises(FrozenSignalError):
            FrozenSignalAdapter([])
        adapter = FrozenSignalAdapter([prediction()])
        with self.assertRaises(FrozenSignalError):
            adapter.adapt([], mode="strict_gate")
        with self.assertRaisesRegex(FrozenSignalError, "unknown adapter"):
            adapter.adapt([prediction()], mode="automatic")


if __name__ == "__main__":
    unittest.main()
