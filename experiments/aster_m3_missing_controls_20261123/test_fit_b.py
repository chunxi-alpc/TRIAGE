"""Numerical controls for the two newly fitted cells."""
import unittest

import numpy as np

from experiments.aster_m3_joint_study import j2_training, training as tr
from experiments.aster_m3_joint_study.test_training import binding_constraint_problem
from .fit_b import fit_cell


class FactorialFitTests(unittest.TestCase):
    def setUp(self):
        self.problem = binding_constraint_problem()
        self.raw = np.array([-2., -2.])
        self.weights = np.array([0.5, 0.5])

    def test_each_cell_uses_its_own_constraint_and_objective(self):
        signs = np.ones(2)
        cf = self.raw, signs, self.weights
        for method, weight, max_response in (("Asymmetric", 0., 2 * tr.DELTA),
                                               ("Direction", 1., tr.DELTA)):
            with self.subTest(method=method):
                result = fit_cell(self.problem, cf, method, 1., weight)
                w = np.asarray(result["w"])
                change = self.problem.cf_x @ w
                self.assertAlmostEqual(float(change.max()), max_response, places=6)
                self.assertLessEqual(result["train_constraints"]["max_violation"], tr.TOLERANCE)
                base = tr.objective(self.problem, w, 1.)[2]
                full = j2_training.objective_j2(self.problem, w, *cf, 1., weight)[2]
                for key in ("task", "ap", "weighted_ap", "regularization"):
                    self.assertEqual(result["final_losses"][key], base[key])
                self.assertAlmostEqual(result["final_losses"]["total"], full["total"], places=12)

    def test_signed_bound_changes_with_label_orientation(self):
        for sign, maximum in ((1, 2 * tr.DELTA), (-1, tr.DELTA)):
            cf = self.raw, np.full(2, sign), self.weights
            model = fit_cell(self.problem, cf, "Asymmetric", 1., 0.)
            delta = self.problem.cf_x @ model["w"]
            self.assertAlmostEqual(float(delta.max()), maximum, places=6)


if __name__ == "__main__":
    unittest.main()
