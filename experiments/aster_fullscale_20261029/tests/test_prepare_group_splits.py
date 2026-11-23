import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_group_splits.py"
spec = importlib.util.spec_from_file_location("prepare_group_splits", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class GroupSplitTests(unittest.TestCase):
    def test_deterministic_and_complete(self):
        groups = [f"g{i}" for i in range(20)]
        fractions = {"fit": 0.7, "select": 0.15, "confirm": 0.15}
        a = mod.allocate(groups, fractions, 20261029)
        b = mod.allocate(groups, fractions, 20261029)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(groups))
        self.assertEqual(set(a.values()), set(fractions))

    def test_nested_group_key(self):
        row = {"metadata": {"guideline": {"tree_id": "tree-4"}}}
        self.assertEqual(mod.field(row, "metadata.guideline.tree_id"), "tree-4")

    def test_missing_or_empty_key_fails(self):
        with self.assertRaises(KeyError):
            mod.field({"x": 1}, "tree_id")
        with self.assertRaises(ValueError):
            mod.field({"tree_id": " "}, "tree_id")

    def test_invalid_fractions_fail(self):
        with self.assertRaises(ValueError):
            mod.allocate(["a", "b", "c"], {"fit": 0.8, "test": 0.3}, 1)
        with self.assertRaises(ValueError):
            mod.allocate(["a"], {"fit": 0.5, "test": 0.5}, 1)

    def test_each_group_has_one_split(self):
        groups = [f"tree-{i}" for i in range(55)]
        assignment = mod.allocate(groups, {"fit": 0.6, "select": 0.2, "heldout": 0.2}, 20261029)
        self.assertEqual(len(assignment), len(groups))
        self.assertTrue(all(v in {"fit", "select", "heldout"} for v in assignment.values()))


if __name__ == "__main__":
    unittest.main()
