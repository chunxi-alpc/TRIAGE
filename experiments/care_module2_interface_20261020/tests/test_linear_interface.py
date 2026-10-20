"""Algebra and scientific-control checks. No model inference or efficacy test."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linear_interface import (
    fit_paired_subspace, fit_mean_direction, covariance_mixture_basis,
    signed_key_projection, replace_coordinates, interchange_coordinates,
    independent_axis_basis, causal_gqa_logits, counterfactual_pair_split_audit,
)


class LinearInterfaceContracts(unittest.TestCase):
    def test_paired_difference_removes_shared_content(self):
        shared = np.array([[100., -30., 7.], [80., 20., 4.], [-50., 40., 2.]])
        signal = np.array([[2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])
        fit = fit_paired_subspace(shared + signal, shared, 1)
        np.testing.assert_allclose(fit.basis @ fit.basis.T, np.diag([1, 0, 0]), atol=1e-12)
        self.assertEqual(fit.captured_squared_energy, 1)

    def test_centering_can_destroy_constant_behavior_shift(self):
        base = np.zeros((8, 3)); positive = base + np.array([1., 2., 0.])
        np.testing.assert_allclose(fit_mean_direction(positive, base), np.array([1., 2., 0.]) / np.sqrt(5))
        self.assertEqual(fit_paired_subspace(positive, base, 1).rank, 1)
        with self.assertRaises(ValueError):
            fit_paired_subspace(positive, base, 1, center=True)

    def test_row_sign_shuffle_is_not_a_spectral_null(self):
        d = np.array([[2., 1., 0.], [3., 1., 1.], [-1., 0., 2.], [4., 2., -1.]])
        sign = np.array([1, -1, -1, 1])[:, None]
        a = fit_paired_subspace(d, np.zeros_like(d), 1).basis
        b = fit_paired_subspace(sign * d, np.zeros_like(d), 1).basis
        np.testing.assert_allclose(a @ a.T, b @ b.T, atol=1e-12)

    def test_covariance_mixture_is_reprojected(self):
        a, b = np.diag([2., 0., 0.]), np.diag([0., 1., 0.])
        q, mixture = covariance_mixture_basis(a, b, .25, 1)
        np.testing.assert_allclose(mixture, np.diag([.75, .25, 0]))
        self.assertFalse(np.allclose(mixture @ mixture, mixture))
        p = q @ q.T
        np.testing.assert_allclose(p @ p, p, atol=1e-12)
        with self.assertRaises(ValueError):
            covariance_mixture_basis(a, b, 1.5, 1)

    def test_independent_positive_negative_zero_token_controls(self):
        k = np.array([[2., 3.], [4., 5.], [6., 7.]])
        b = np.array([[1.], [0.]])
        original = k.copy()
        edited = signed_key_projection(k, b, [.5, -.5, 0])
        np.testing.assert_array_equal(edited, [[3, 3], [2, 5], [6, 7]])
        np.testing.assert_array_equal(k, original)

    def test_zero_edit_exact_identity(self):
        k = np.arange(20, dtype=float).reshape(5, 4)
        np.testing.assert_array_equal(signed_key_projection(k, np.eye(4)[:, :2], np.zeros(5)), k)

    def test_centroid_replacement_preserves_complement(self):
        h = np.array([3., 4., 5., 6.]); b = np.eye(4)[:, :2]
        edited = replace_coordinates(h, b, [-2, 8])
        np.testing.assert_allclose(b.T @ edited, [-2, 8])
        np.testing.assert_array_equal(edited[2:], h[2:])
        np.testing.assert_array_equal(replace_coordinates(h, b, [-2, 8], 0), h)
        np.testing.assert_allclose(replace_coordinates(h, b, [-2, 8], .5), .5 * (h + edited))

    def test_interchange_preserves_orthogonal_complement(self):
        b = np.eye(4)[:, :1]
        h = np.array([1., 10., 20., 30.]); donor = np.array([-5., 100., 200., 300.])
        np.testing.assert_array_equal(interchange_coordinates(h, donor, b), [-5., 10., 20., 30.])

    def test_nonorthogonal_axes_require_coordinate_conversion(self):
        axes = np.array([[1., 1.], [0., 1.], [0., 0.]])
        q, r = independent_axis_basis(axes)
        mu = np.array([2., 3.])
        np.testing.assert_allclose(q @ (r @ mu), axes @ mu, atol=1e-12)
        with self.assertRaises(ValueError):
            replace_coordinates(np.zeros(3), axes, mu)
        with self.assertRaises(ValueError):
            independent_axis_basis(np.array([[1., 2.], [0., 0.], [0., 0.]]))

    def test_gqa_edit_has_exact_head_and_causal_support(self):
        rng = np.random.default_rng(100)
        q = rng.normal(size=(4, 8, 4)); k = rng.normal(size=(4, 2, 4))
        angles = np.arange(4)[:, None] * np.array([[.2, .7]])
        cos = np.concatenate([np.cos(angles)] * 2, axis=1)
        sin = np.concatenate([np.sin(angles)] * 2, axis=1)
        before = causal_gqa_logits(q, k, cos, sin)
        changed = k.copy(); changed[1, 1] += np.array([1., 2., 3., 4.])
        after = causal_gqa_logits(q, changed, cos, sin)
        valid = np.isfinite(before)
        delta = np.zeros_like(before); delta[valid] = after[valid] - before[valid]
        np.testing.assert_array_equal(delta[:4], np.zeros_like(delta[:4]))
        np.testing.assert_array_equal(delta[4:, :, [0, 2, 3]], np.zeros_like(delta[4:, :, [0, 2, 3]]))
        np.testing.assert_array_equal(delta[4:, 0, 1], np.zeros(4))
        self.assertGreater(np.linalg.norm(delta[4:, 1:, 1]), 0)

    def test_positive_key_coefficient_can_lower_score(self):
        q = np.array([[[-1., 0.]]]); k = np.array([[[1., 0.]]])
        changed = k.copy(); changed[:, 0] = signed_key_projection(k[:, 0], np.eye(2)[:, :1], [1])
        before = causal_gqa_logits(q, k, np.ones((1, 2)), np.zeros((1, 2)))
        after = causal_gqa_logits(q, changed, np.ones((1, 2)), np.zeros((1, 2)))
        self.assertLess(float(after[0, 0, 0] - before[0, 0, 0]), 0)

    def test_nonzero_rotary_angle_against_hand_calculation(self):
        q = np.array([[[1., 0.]], [[1., 0.]]])
        k = np.array([[[0., 1.]], [[1., 0.]]])
        cos = np.array([[1., 1.], [0., 0.]])
        sin = np.array([[0., 0.], [1., 1.]])
        actual = causal_gqa_logits(q, k, cos, sin)[0]
        expected = np.array([[0., -np.inf], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_semantic_family_split_leakage_is_rejected(self):
        a = {"case_id": "a", "family_id": "f", "pair_id": "p", "split": "train"}
        b = {**a, "case_id": "b", "split": "test"}
        with self.assertRaises(ValueError):
            counterfactual_pair_split_audit([a, b])
        b["split"] = "train"
        self.assertEqual(counterfactual_pair_split_audit([a, b])["pairs"], 1)

    def test_sibling_family_counts_cannot_be_inflated(self):
        a = {"case_id": "a", "family_id": "f", "pair_id": "p", "split": "train"}
        with self.assertRaises(ValueError):
            counterfactual_pair_split_audit([a, {**a, "case_id": "b", "family_id": "other"}])
        a["base_case_id"] = "base"
        b = {"case_id": "b", "family_id": "g", "pair_id": "q", "split": "test", "base_case_id": "base"}
        with self.assertRaises(ValueError):
            counterfactual_pair_split_audit([a, b])
        del b["base_case_id"]
        with self.assertRaises(ValueError):
            counterfactual_pair_split_audit([a, b])

    def test_nonfinite_inputs_and_unsupported_rank_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_paired_subspace([[np.nan, 0]], [[0, 0]], 1)
        with self.assertRaises(ValueError):
            fit_paired_subspace([[1, 0]], [[0, 0]], 2)


if __name__ == "__main__":
    unittest.main()
