"""CPU-only contracts for optional actual CounterFact method lists."""
import unittest
from counterfact_entity_ci import METHODS, selected_methods


class SelectionContracts(unittest.TestCase):
    def test_original_four_method_default_is_unchanged(self):
        self.assertEqual(selected_methods(None, {name: "/synthetic/" + name for name in METHODS}), METHODS)

    def test_base_three_methods_need_no_pasta_result(self):
        actual = {"seka": "/synthetic/seka", "matched_zero": "/synthetic/zero", "unsteered": "/synthetic/nohook", "pasta": None}
        self.assertEqual(selected_methods(["seka", "matched_zero", "unsteered"], actual), METHODS[:3])

    def test_unselected_existing_result_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            selected_methods(["seka", "matched_zero", "unsteered"], {name: "/synthetic/" + name for name in METHODS})


if __name__ == "__main__":
    unittest.main()
