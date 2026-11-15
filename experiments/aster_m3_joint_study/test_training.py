"""Synthetic CPU tests for the joint-supervision objective and fitting protocol.

Run on the experiment server from the repository root:
  python -m unittest experiments.aster_m3_joint_study.test_training -v

These fixtures contain no historical observations and produce no task results.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
from scipy.special import expit

from . import training as tr


def synthetic_inputs():
    """Unequal AP source frequencies, natural labels, and withheld poison rows."""
    rng = np.random.default_rng(4813)
    natural, pairs, ell, traces = [], [], {}, {}
    train_sources, valid_sources = ["train_a", "train_b"], ["valid_a"]
    for source in train_sources + valid_sources:
        labels = [0, 0, 1, 1, 1, 1]
        for index, label in enumerate(labels):
            key = f"{source}:natural:{index}"
            traces[key] = rng.normal(size=16) + np.linspace(-1, 1, 16)
            ell[key] = float(rng.normal() - 1.1)
            natural.append(dict(view_id=key, source_id=source, split="fit",
                                variant="full", label=label))
    # These ineligible rows have no score/trace entry. Accessing them is a bug.
    natural += [
        dict(view_id="unlabeled", source_id="train_a", split="fit", variant="full", label=None),
        dict(view_id="perturbed", source_id="train_a", split="fit", variant="action_swap", label=1),
        dict(view_id="development", source_id="train_a", split="dev", variant="full", label=0),
    ]

    def add_pair(kind, source, index, difference=None):
        pair_id = f"{source}:{kind}:{index}"
        left, right = pair_id + ":left", pair_id + ":right"
        traces[left] = rng.normal(size=16)
        traces[right] = traces[left] + (rng.normal(size=16) if difference is None else difference)
        ell[left], ell[right] = float(rng.normal()), float(rng.normal())
        pairs.append(dict(pair_id=pair_id, source_id=source, split="fit", kind=kind,
                          left=left, right=right, left_label=0, right_label=int(kind in ("C", "F"))))

    for kind, source, count in (("A", "train_a", 3), ("A", "train_b", 1),
                                ("P", "train_a", 1), ("P", "train_b", 2)):
        for index in range(count):
            add_pair(kind, source, index)
    for index, axis in enumerate(np.eye(16)):
        add_pair("C" if index % 2 == 0 else "F", train_sources[index % 2], index, axis)
    # Neither a held-out pair nor a non-fit pair may access its absent endpoints.
    pairs += [
        dict(pair_id="withheld_pair", source_id="valid_a", split="fit", kind="A",
             left="missing_valid_left", right="missing_valid_right"),
        dict(pair_id="development_pair", source_id="train_a", split="dev", kind="C",
             left="missing_dev_left", right="missing_dev_right"),
    ]
    return natural, pairs, ell, traces, train_sources, valid_sources


def make_synthetic_problem():
    return tr.make_problem(*synthetic_inputs())


def binding_constraint_problem():
    """Task and AP gradients both push w[0] beyond the bound DELTA / 2."""
    natural, pairs, ell, traces = [], [], {}, {}
    for source in ("train_a", "train_b"):
        for index, x in enumerate((-2., -1., 1., 2.)):
            key = f"{source}:{index}"
            t = np.zeros(16)
            t[0] = x
            traces[key], ell[key] = t, -x
            natural.append(dict(view_id=key, source_id=source, split="fit",
                                variant="full", label=int(x > 0)))
    for kind, source, displacement in (("A", "train_a", 1.), ("P", "train_b", 1.),
                                        ("C", "train_a", 1.), ("F", "train_b", 2.)):
        left, right = kind + ":left", kind + ":right"
        traces[left] = np.zeros(16)
        traces[right] = np.zeros(16)
        traces[right][0] = displacement
        ell[left], ell[right] = 0., -2.
        pairs.append(dict(pair_id=kind, kind=kind, source_id=source, split="fit",
                          left=left, right=right, left_label=0, right_label=int(kind in ("C", "F"))))
    return tr.make_problem(natural, pairs, ell, traces, ["train_a", "train_b"], ["valid_a"])


class ObjectiveTests(unittest.TestCase):
    def test_zero_ap_ignores_pair_target_and_matches_supervised_fit(self):
        problem = make_synthetic_problem()
        w = np.linspace(-0.15, 0.15, 16)
        changed_ap = replace(problem, ap_y=problem.ap_y + 57.)
        value, gradient, terms = tr.objective(problem, w, lambda_ap=0.)
        other_value, other_gradient, other_terms = tr.objective(changed_ap, w, lambda_ap=0.)
        self.assertEqual(terms["weighted_ap"], 0.)
        self.assertNotEqual(terms["ap"], other_terms["ap"])
        self.assertEqual(value, other_value)
        np.testing.assert_array_equal(gradient, other_gradient)
        # Both routes must use the same task loss, regularization and constraints.
        supervised = tr.fit(problem, lambda_ap=0., method="S")
        joint_zero = tr.fit(problem, lambda_ap=0., method="J")
        changed_fit = tr.fit(changed_ap, lambda_ap=0., method="J")
        for model in (joint_zero, changed_fit):
            np.testing.assert_allclose(model["w"], supervised["w"], atol=1e-10, rtol=1e-10)
            self.assertAlmostEqual(model["b_independent"], supervised["b_independent"], places=10)
        with self.assertRaises(ValueError):
            tr.fit(problem, lambda_ap=0.1, method="S")

    def test_objective_gradient_matches_central_finite_differences(self):
        problem = make_synthetic_problem()
        w = np.linspace(-0.12, 0.09, 16)
        epsilon = 1e-6
        for lambda_ap in (0., 0.1, 1., 10.):
            with self.subTest(lambda_ap=lambda_ap):
                _, analytic, terms = tr.objective(problem, w, lambda_ap)
                numerical = np.empty(16)
                for index in range(16):
                    step = np.zeros(16)
                    step[index] = epsilon
                    numerical[index] = (tr.objective(problem, w+step, lambda_ap)[0]
                                        - tr.objective(problem, w-step, lambda_ap)[0])/(2*epsilon)
                np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-7)
                self.assertAlmostEqual(terms["total"], terms["task"] + terms["weighted_ap"]
                                       + terms["regularization"], places=13)

    def test_normalization_and_category_source_weights(self):
        problem = make_synthetic_problem()
        self.assertAlmostEqual(float(problem.task_weights.sum()), 1., places=14)
        self.assertAlmostEqual(float(problem.ap_weights.sum()), 1., places=14)
        self.assertAlmostEqual(tr.objective(problem, np.zeros(16), 1.)[2]["task"], 1., places=13)
        weights = dict(zip(problem.ap_ids, problem.ap_weights))
        # Each category has half the mass; its two sources each have one quarter.
        self.assertAlmostEqual(weights["train_a:A:0"], 1./12, places=14)
        self.assertAlmostEqual(weights["train_b:A:0"], 1./4, places=14)
        self.assertAlmostEqual(weights["train_a:P:0"], 1./4, places=14)
        self.assertAlmostEqual(weights["train_b:P:0"], 1./8, places=14)
        self.assertGreaterEqual(problem.ap_scale, 1.)


class ReconstructionTests(unittest.TestCase):
    def test_centering_uses_training_natural_nodes_and_reconstructs_linear_scores(self):
        natural, pairs, ell, traces, train_sources, valid_sources = synthetic_inputs()
        problem = tr.make_problem(natural, pairs, ell, traces, train_sources, valid_sources)
        matrix = np.stack([traces[key] for key in problem.natural_ids])
        np.testing.assert_allclose(problem.mu, matrix.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(problem.t, matrix-matrix.mean(axis=0), atol=1e-14)
        w = np.linspace(-0.25, 0.2, 16)
        model = dict(w=w.tolist(), mu=problem.mu.tolist(), b_shared=-0.2, b_independent=0.7)
        expected_correction = np.sum((matrix-problem.mu)*w, axis=1)
        np.testing.assert_allclose(tr.correction(model, matrix), expected_correction, atol=1e-14)
        self.assertAlmostEqual(float(np.mean(expected_correction)), 0., places=13)
        for calibration, b in (("shared", -0.2), ("independent", 0.7)):
            np.testing.assert_allclose(tr.score(model, problem.ell, matrix, calibration),
                                       problem.ell+expected_correction+b, atol=1e-14)

    def test_historical_e1_intercept_translation_preserves_complete_scores(self):
        problem = make_synthetic_problem()
        traces = problem.t + problem.mu
        w = np.linspace(-0.17, 0.31, 16)
        old_b, b0 = 0.37, -0.29
        model = tr.centered_e1(w, old_b, problem.mu, b0)
        expected_historical = problem.ell + traces @ w + old_b
        np.testing.assert_allclose(tr.score(model, problem.ell, traces, "independent"),
                                   expected_historical, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(tr.score(model, problem.ell, traces, "shared"),
                                   problem.ell + problem.t @ w + b0, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(model["b_independent"], old_b+problem.mu@w, places=13)


class ConstraintAndSourceTests(unittest.TestCase):
    def test_binding_cf_constraint_is_satisfied_at_the_original_budget(self):
        problem = binding_constraint_problem()
        expected_w0 = tr.DELTA/2
        boundary = np.zeros(16)
        boundary[0] = expected_w0
        # The unconstrained descent direction at the boundary points outwards.
        self.assertLess(tr.objective(problem, boundary, 1.)[1][0], -0.1)
        model = tr.fit(problem, lambda_ap=1., method="J")
        w = np.asarray(model["w"])
        self.assertAlmostEqual(w[0], expected_w0, places=6)
        np.testing.assert_allclose(w[1:], 0., atol=1e-8)
        residuals = np.abs(problem.cf_x @ w) - tr.DELTA
        self.assertLessEqual(float(residuals.max()), tr.TOLERANCE)
        self.assertGreaterEqual(model["cf_active_count"], 1)
        np.testing.assert_allclose(model["cf_training_residuals"], residuals, atol=1e-14)
        self.assertAlmostEqual(model["cf_max_violation"], max(0., float(residuals.max())), places=14)

    def test_source_overlap_is_rejected(self):
        with self.assertRaises(tr.ProtocolConflict):
            tr.check_sources(["train_a", "train_b"], ["train_b", "valid_a"])
        with self.assertRaises(tr.ProtocolConflict):
            tr.check_sources([], ["valid_a"])

    def test_validation_labels_scores_and_traces_never_enter_fitting(self):
        args = synthetic_inputs()
        problem = tr.make_problem(*args)
        altered = deepcopy(args)
        natural, pairs, ell, traces, train_sources, valid_sources = altered
        for row in natural:
            if row["source_id"] in valid_sources:
                row["label"] = 1 - row["label"]
                ell[row["view_id"]] = 1e8
                traces[row["view_id"]] = np.full(16, -1e9)
        changed = tr.make_problem(*altered)
        for name in ("t", "ell", "y", "task_weights", "ap_x", "ap_y", "ap_weights", "cf_x", "mu"):
            np.testing.assert_array_equal(getattr(problem, name), getattr(changed, name))
        self.assertEqual(problem.b0, changed.b0)
        self.assertEqual(problem.task_scale, changed.task_scale)
        self.assertEqual(problem.ap_scale, changed.ap_scale)
        self.assertEqual(set(problem.natural_sources), set(train_sources))
        self.assertTrue(set(problem.pair_sources).issubset(train_sources))
        self.assertFalse(set(problem.natural_sources) & set(valid_sources))
        self.assertEqual(len(problem.natural_ids), 12)
        self.assertNotIn("withheld_pair", problem.ap_ids)
        self.assertNotIn("development_pair", problem.cf_ids)
        # Make held-out scores/features unavailable altogether: construction still works.
        for row in natural:
            if row["source_id"] in valid_sources:
                del ell[row["view_id"]]
                del traces[row["view_id"]]
        missing_heldout = tr.make_problem(*altered)
        np.testing.assert_array_equal(missing_heldout.mu, problem.mu)

    def test_calibration_schemes_use_the_same_training_natural_labels(self):
        problem = make_synthetic_problem()
        model = tr.fit(problem, lambda_ap=0.1, method="J")
        self.assertEqual(model["b_shared"], problem.b0)
        self.assertEqual(model["calibration_ids"], problem.natural_ids)
        self.assertEqual(model["calibration_sources"], problem.natural_sources)
        self.assertEqual(model["task_ids"], model["calibration_ids"])
        self.assertFalse(set(model["calibration_ids"]) & set(problem.ap_ids+problem.cf_ids))
        self.assertAlmostEqual(float(np.mean(expit(problem.ell+model["b_shared"])-problem.y)),
                               0., places=11)
        corrected = problem.ell + problem.t @ np.asarray(model["w"])
        self.assertAlmostEqual(float(np.mean(expit(corrected+model["b_independent"])-problem.y)),
                               0., places=11)
        # Fitting against all labels would change this intercept; held-out scores are irrelevant.
        poisoned_scores = np.r_[corrected, np.full(20, -25.)]
        poisoned_labels = np.r_[problem.y, np.ones(20)]
        self.assertGreater(abs(tr.fit_intercept(poisoned_scores, poisoned_labels)
                               - model["b_independent"]), 1.)

    def test_invalid_calibration_labels_are_reported(self):
        with self.assertRaises(tr.ProtocolConflict):
            tr.fit_intercept([0., 1.], [1., 1.])
        with self.assertRaises(ValueError):
            tr.fit_intercept([0., np.nan], [0., 1.])
        with self.assertRaises(ValueError):
            tr.fit_intercept([0., 1.], [0., 2.])


if __name__ == "__main__":
    unittest.main()
