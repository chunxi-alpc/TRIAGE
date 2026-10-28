"""Numerical and scientific invariants of current token-level Module 2."""
import unittest

import numpy as np

from src.module2.mechanism import (
    cluster_bootstrap_interval, enumerate_group_sign_directions,
    evaluate_direction, fit_operators, rule_relative_effects, select_heads,
)


def metadata(n, prefix="fit"):
    return [{"id": f"{prefix}:{i}", "group_id": f"{prefix}:{i // 2}", "split": prefix}
            for i in range(n)]


def token_moment(tokens):
    # [N, token, L, H, D], equal token count only in these tiny fixtures.
    return np.einsum("ntlhd,ntlhe->lhde", tokens, tokens) / np.prod(tokens.shape[:2])


class TokenMechanismTests(unittest.TestCase):
    def fixture(self):
        tokens = np.array([[[1., 2., 0.], [1., -2., 0.]],
                           [[2., 1., 0.], [2., -1., 0.]],
                           [[1., 2., 0.], [1., -2., 0.]],
                           [[2., 1., 0.], [2., -1., 0.]]])[:, :, None, None, :]
        differences = tokens.mean(axis=1)
        fit = fit_operators(differences, token_moment(tokens), rank=2,
                            metadata=metadata(len(tokens)))
        return tokens, differences, fit

    def test_token_moment_retains_within_span_directions(self):
        tokens, differences, fit = self.fixture()
        self.assertEqual(int(fit["effective_rank"][0, 0]), 2)
        span_moment = np.einsum("nlhd,nlhe->lhde", differences, differences) / len(differences)
        span_fit = fit_operators(differences, span_moment, rank=2)
        self.assertEqual(int(span_fit["effective_rank"][0, 0]), 1)
        self.assertFalse(bool(span_fit["numerically_valid"][0, 0]))
        self.assertTrue(bool(fit["numerically_valid"][0, 0]))
        self.assertLessEqual(np.linalg.norm(fit["delta"][0, 0]), 1.)
        np.testing.assert_allclose(fit["projected_mean"][0, 0], [1.5, 0, 0])

    def test_swapped_labels_preserve_spectrum_reverse_direction(self):
        tokens, differences, fit = self.fixture()
        reversed_fit = fit_operators(-differences, token_moment(-tokens), rank=2)
        np.testing.assert_array_equal(fit["eigenvalues"], reversed_fit["eigenvalues"])
        np.testing.assert_allclose(fit["delta"], -reversed_fit["delta"])

    def test_equal_pair_weighting_is_not_equal_guide_weighting(self):
        tokens, differences, fit = self.fixture()
        idx = [0, 1, 2, 3, 0, 0]
        repeated = fit_operators(differences[idx], token_moment(tokens[idx]), rank=2)
        np.testing.assert_allclose(repeated["raw_mean"], differences[idx].mean(axis=0))
        self.assertNotEqual(float(fit["raw_mean"][0, 0, 0]),
                            float(repeated["raw_mean"][0, 0, 0]))

    def test_zero_shared_mean_and_inconsistent_accumulator(self):
        tokens, differences, _ = self.fixture()
        signed = differences.copy()
        signed[2:] *= -1
        fit = fit_operators(signed, token_moment(tokens), rank=2)
        np.testing.assert_array_equal(fit["delta"], 0.)
        self.assertFalse(bool(fit["numerically_valid"][0, 0]))
        with self.assertRaisesRegex(ValueError, "Token energy"):
            fit_operators(differences, np.zeros((1, 1, 3, 3)), rank=2)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            fit_operators(differences, -np.eye(3)[None, None], rank=2)

    def test_current_sep_and_both_state_selection(self):
        _, difference, fit = self.fixture()
        keys = np.zeros((4, 3, 1, 1, 3))
        keys[:, 0] = difference
        keys[:, 2] = difference / 2
        queries = np.zeros((4, 2, 1, 1, 3))
        queries[..., 0] = 3.
        evaluation = evaluate_direction(fit["delta"], keys, queries, metadata(4, "select"),
            fit_group_ids=fit["fit_group_ids"], draws=30)
        z = evaluation["per_pair_z"]
        np.testing.assert_allclose(evaluation["separation"], z.mean(0) / (z.std(0) + 1e-8))
        np.testing.assert_allclose(evaluation["read"], 3 * fit["delta"][..., 0] / np.sqrt(3))
        self.assertEqual(select_heads(fit, evaluation), [(0, 0)])
        # Average read remains positive; negative-state read is wrong-signed.
        queries[:, 0, ..., 0] = 9.
        queries[:, 1, ..., 0] = -1.
        bad = evaluate_direction(fit["delta"], keys, queries, metadata(4, "select"), draws=30)
        self.assertGreater(float(bad["read"][0, 0]), 0.)
        self.assertEqual(select_heads(fit, bad), [])
        keys[:, 2, ..., 0] += 2.
        penalized = evaluate_direction(fit["delta"], keys, queries, metadata(4, "select"), draws=30)
        self.assertLess(float(penalized["separation"][0, 0]), float(bad["separation"][0, 0]))

    def test_isolation_near_zero_and_single_group_intervals(self):
        _, difference, fit = self.fixture()
        keys = np.zeros((4, 3, 1, 1, 3))
        queries = np.ones((4, 2, 1, 1, 3))
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_direction(fit["delta"], keys, queries, metadata(4),
                               fit_group_ids=fit["fit_group_ids"])
        evaluation = evaluate_direction(fit["delta"], keys, queries, metadata(4, "test"), draws=20)
        np.testing.assert_array_equal(evaluation["near_zero_fraction"], 1.)
        self.assertTrue(np.isnan(evaluation["mean_cosine"]).all())
        self.assertEqual(select_heads(fit, evaluation), [])
        low, high = cluster_bootstrap_interval([1., 2.], ["a", "a"])
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_guide_bootstrap_preserves_pair_weighting(self):
        values = np.array([0., 0., 0., 12.])
        groups = ["a", "a", "a", "b"]
        # With two sampled groups, possible bootstrap means are 0, 3, 12, not 6.
        low, high = cluster_bootstrap_interval(values, groups, draws=200, seed=31)
        self.assertEqual(float(low), 0.)
        self.assertEqual(float(high), 12.)

    def test_rule_relative_effects_cancel_identical_fact_shift(self):
        app = np.array([2., 0., 0.])
        fact = np.array([0., 3., 0.])
        difference = np.stack([app + fact, app - fact])[:, None, None]
        records = [{"id": "tree:swap0:order0:targetA", "group_id": "g", "target_rule_id": "A"},
                   {"id": "tree:swap0:order0:targetB", "group_id": "g", "target_rule_id": "B"}]
        result = rule_relative_effects(difference, records, app[None, None] / 2)
        np.testing.assert_allclose(result["interaction"][0, 0, 0], app)
        np.testing.assert_allclose(result["common_fact"][0, 0, 0], fact)
        np.testing.assert_allclose(result["aligned_interaction"], 2.)
        np.testing.assert_allclose(result["aligned_common_fact"], 0.)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            rule_relative_effects(difference[:1], records[:1])

    def test_exhaustive_sign_diagnostic_retains_all_positive_fit(self):
        _, difference, fit = self.fixture()
        controls = enumerate_group_sign_directions(difference, ["a", "a", "b", "b"], fit["basis"])
        self.assertEqual(controls["signs"].shape, (4, 2))
        np.testing.assert_allclose(controls["directions"][-1], fit["delta"])
        np.testing.assert_allclose(controls["directions"][0], -fit["delta"])

    def test_selection_ties_and_empty_set_are_deterministic(self):
        fit = {"numerically_valid": np.ones((2, 2), dtype=bool), "fit_group_ids": ["fit"]}
        evaluation = {name: np.ones((2, 2)) for name in
                      ("separation", "read_positive", "read_negative", "score")}
        evaluation["group_ids"] = ["select"]
        self.assertEqual(select_heads(fit, evaluation, max_heads=3), [(0, 0), (0, 1), (1, 0)])
        evaluation["read_negative"][:] = 0
        self.assertEqual(select_heads(fit, evaluation), [])


if __name__ == "__main__":
    unittest.main()
