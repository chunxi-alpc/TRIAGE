import unittest
import numpy as np
from offline_eval import prepare, predict, evaluate, norm, temperature, POLICIES, GRID

def raw(variant="NC", id="case0", tree="aml_7"):
    return {"id": id, "tree": tree, "variant": variant, "seed": 0,
            "split": "dev", "gold_option": 0, "option_S": [0.9, 0.1],
            "scores": {"original": [1., 0.], "mass_zero_control": [1., 0.],
                       "action_bias_1.0_all": [1.5, -0.5]}}

class TestOfflineQwen(unittest.TestCase):
    def test_normalization(self):
        np.testing.assert_allclose(norm([2, 3]), [.4, .6])
        np.testing.assert_allclose(temperature([.8, .2], 1.), [.8, .2])
    def test_no_delta_identity(self):
        x=raw()
        x["scores"]["action_bias_1.0_all"] = x["scores"]["mass_zero_control"][:]
        p=prepare(x)
        self.assertEqual(predict(p, (.5, 1., 1.), "full"),
                         predict(p, (.5, 1., 1.), "static"))
    def test_policies_do_not_use_gold(self):
        p=prepare(raw())
        for k in POLICIES:
            self.assertIn(predict(p, GRID[0], k), (0, 1))
        p["gold"]=1
        for k in POLICIES:
            self.assertIn(predict(p, GRID[0], k), (0, 1))
    def test_invalid_split(self):
        x=raw()
        x["split"]="test"
        with self.assertRaises(ValueError): prepare(x)
    def test_missing_zero_path(self):
        x=raw()
        x["scores"].pop("mass_zero_control")
        with self.assertRaises(ValueError): prepare(x)
    def test_four_tree_cv(self):
        rows=[prepare(raw(v, f"case{i}", t))
              for i,t in enumerate(("aml_7","bre_1","pancreatic_11","vulvar_3"))
              for v in ("NC","AC","IC","SW")]
        a=evaluate(rows,"static")
        self.assertEqual(a["correct"],16)
        self.assertEqual(len(a["chosen_by_tree"]),4)
    def test_grid_fixed(self):
        self.assertEqual(len(GRID),27)

if __name__ == "__main__":
    unittest.main()
