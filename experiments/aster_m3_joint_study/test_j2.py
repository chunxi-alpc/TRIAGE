"""Synthetic tests for J2; run only on the designated experiment server."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.special import expit

from . import training as tr
from . import j2_training as j2
from .test_training import make_synthetic_problem, binding_constraint_problem


class J2ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_synthetic_problem()
        n = len(self.problem.cf_ids)
        self.d = np.linspace(-3., 2., n)
        self.s = np.where(np.arange(n)%2, -1., 1.)
        self.v = np.arange(1., n+1.)
        self.v /= self.v.sum()
        self.w = np.linspace(-0.12, 0.09, 16)

    def test_direction_and_full_gradients_match_finite_differences(self):
        step = 1e-6
        for weight in (0.1, 1., 10.):
            with self.subTest(lambda_cf=weight):
                _, gradient, terms = j2.objective_j2(self.problem,self.w,self.d,self.s,self.v,1.,weight)
                numerical = np.empty(16)
                for index in range(16):
                    dx = np.zeros(16)
                    dx[index] = step
                    plus = j2.objective_j2(self.problem,self.w+dx,self.d,self.s,self.v,1.,weight)[0]
                    minus = j2.objective_j2(self.problem,self.w-dx,self.d,self.s,self.v,1.,weight)[0]
                    numerical[index] = (plus-minus)/(2*step)
                np.testing.assert_allclose(gradient,numerical,rtol=3e-6,atol=3e-7)
                self.assertAlmostEqual(terms["total"],terms["base_total"]+terms["weighted_cf_direction"],places=13)
        # The added analytic gradient itself agrees with a separate difference of objectives.
        _, full, _ = j2.objective_j2(self.problem,self.w,self.d,self.s,self.v,1.,1.)
        _, base, _ = tr.objective(self.problem,self.w,1.)
        scale = max(float(self.v@np.logaddexp(0.,-self.s*self.d)),1e-12)
        margin = self.s*(self.d+self.problem.cf_x@self.w)
        expected = -self.problem.cf_x.T@(self.v*self.s*expit(-margin))/scale
        np.testing.assert_allclose(full-base,expected,rtol=1e-12,atol=1e-12)

    def test_zero_cf_coefficient_preserves_the_original_objective(self):
        for lambda_ap in (0., 0.1, 1., 10.):
            base, base_gradient, base_terms = tr.objective(self.problem,self.w,lambda_ap)
            value, gradient, terms = j2.objective_j2(self.problem,self.w,self.d,self.s,self.v,lambda_ap,0.)
            self.assertEqual(value,base)
            np.testing.assert_array_equal(gradient,base_gradient)
            for name in ("task","ap","regularization","weighted_ap"):
                self.assertEqual(terms[name],base_terms[name])
            self.assertEqual(terms["weighted_cf_direction"],0.)
        # At w=0 the direction loss is normalized against this exact CF training pool.
        zero_terms = j2.objective_j2(self.problem,np.zeros(16),self.d,self.s,self.v,0.1,1.)[2]
        self.assertAlmostEqual(zero_terms["cf_direction"],1.,places=13)

    def test_endpoint_reversal_preserves_objective_and_constraint_orientation(self):
        reversed_problem = replace(self.problem,cf_x=-self.problem.cf_x)
        direct = j2.objective_j2(self.problem,self.w,self.d,self.s,self.v,1.,1.)
        reversed_result = j2.objective_j2(reversed_problem,self.w,-self.d,-self.s,self.v,1.,1.)
        self.assertAlmostEqual(direct[0],reversed_result[0],places=13)
        np.testing.assert_allclose(direct[1],reversed_result[1],atol=1e-13)
        np.testing.assert_array_equal(self.s[:,None]*self.problem.cf_x,
                                      (-self.s)[:,None]*reversed_problem.cf_x)

    def test_invalid_cf_alignment_or_label_direction_is_reported(self):
        with self.assertRaises(tr.ProtocolConflict):
            j2.objective_j2(self.problem,self.w,self.d[:-1],self.s,self.v,1.,1.)
        bad_labels = self.s.copy()
        bad_labels[0] = 0.
        with self.assertRaises(tr.ProtocolConflict):
            j2.objective_j2(self.problem,self.w,self.d,bad_labels,self.v,1.,1.)
        with self.assertRaises(tr.ProtocolConflict):
            j2.objective_j2(self.problem,self.w,self.d,self.s,self.v*2,1.,1.)
        with self.assertRaises(ValueError):
            j2.objective_j2(self.problem,self.w,self.d,self.s,self.v,1.,-0.1)


class J2FittingTests(unittest.TestCase):
    def test_asymmetric_bounds_allow_truth_direction_and_bound_harmful_direction(self):
        problem = binding_constraint_problem()
        d, weights = np.array([-2.,-2.]), np.array([0.5,0.5])
        # The task/AP objective pushes positive w[0]. With positive label directions
        # the strongest E row allows w[0]=DELTA; reversing labels caps it at DELTA/2.
        beneficial = j2.fit_j2(problem,d,np.ones(2),weights,1.,0.)
        harmful = j2.fit_j2(problem,d,-np.ones(2),weights,1.,0.)
        self.assertAlmostEqual(beneficial["w"][0],tr.DELTA,places=6)
        self.assertAlmostEqual(harmful["w"][0],tr.DELTA/2,places=6)
        self.assertAlmostEqual(beneficial["cf_max_signed_increase"],2*tr.DELTA,places=6)
        self.assertAlmostEqual(harmful["cf_max_signed_decrease"],tr.DELTA,places=6)
        self.assertGreater(beneficial["cf_original_budget_max_excess"],tr.DELTA*0.99)
        self.assertEqual(beneficial["cf_original_budget_excess_count"],1)
        self.assertLessEqual(harmful["cf_original_budget_max_excess"],tr.TOLERANCE)
        for model in (beneficial,harmful):
            self.assertEqual(model["parameter_count"],16)
            self.assertEqual(len(model["w"]),16)
            self.assertEqual(len(model["mu"]),16)
            np.testing.assert_allclose(model["w"][1:],0.,atol=1e-8)
            signed = np.asarray(model["cf_signed_correction"])
            self.assertGreaterEqual(float(signed.min()),-tr.DELTA-tr.TOLERANCE)
            self.assertLessEqual(float(signed.max()),2*tr.DELTA+tr.TOLERANCE)
            self.assertLessEqual(model["cf_max_violation"],tr.TOLERANCE)
            np.testing.assert_allclose(model["cf_lower_residuals"],-tr.DELTA-signed,atol=1e-13)
            np.testing.assert_allclose(model["cf_upper_residuals"],signed-2*tr.DELTA,atol=1e-13)

    def test_fitted_model_preserves_natural_calibration_and_cf_metadata(self):
        problem = make_synthetic_problem()
        n = len(problem.cf_ids)
        d = np.linspace(-1.,2.,n)
        s = np.where(np.arange(n)%2,-1.,1.)
        v = np.full(n,1./n)
        model = j2.fit_j2(problem,d,s,v,0.1,1.)
        self.assertEqual(model["calibration_ids"],problem.natural_ids)
        self.assertEqual(model["calibration_sources"],problem.natural_sources)
        self.assertEqual(model["train_sources"],problem.train_sources)
        self.assertEqual(model["b_shared"],problem.b0)
        self.assertEqual(model["cf_ids"],problem.cf_ids)
        np.testing.assert_array_equal(model["cf_weights"],v)
        np.testing.assert_array_equal(model["cf_raw_difference"],d)
        np.testing.assert_array_equal(model["cf_label_difference"],s)
        np.testing.assert_array_equal(model["mu"],problem.mu)
        logits = problem.ell+problem.t@np.asarray(model["w"])+model["b_independent"]
        self.assertAlmostEqual(float(np.mean(expit(logits)-problem.y)),0.,places=11)
        self.assertLessEqual(model["final_losses"]["total"],model["initial_losses"]["total"]+1e-8)
        self.assertEqual(model["regularization"],tr.REGULARIZATION)
        self.assertGreater(model["gradient_final"]["cf_direction_norm"],0.)


if __name__ == "__main__":
    unittest.main()
