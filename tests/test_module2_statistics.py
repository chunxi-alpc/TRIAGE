"""Scientific invariants, including cases that must falsify a shared direction."""
import unittest
import numpy as np

from src.module2.applicability import fit_subspaces, evaluate_heads, select_heads


class ApplicabilityStatisticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.n = rng.normal(size=(24, 1, 1, 4))
        self.p = self.n + np.array([1., 0., 0., 0.])
        self.z = (self.p + self.n) / 2
        self.groups = np.repeat(['fit_a', 'fit_b', 'fit_c'], 8)

    def test_sign_swap_preserves_second_moment_but_reverses_operator(self):
        a = fit_subspaces(self.p, self.n, self.z, self.groups, rank=1)
        b = fit_subspaces(self.n, self.p, self.z, self.groups, rank=1)
        np.testing.assert_allclose(a['eigenvalues'], b['eigenvalues'])
        np.testing.assert_allclose(a['delta'], -b['delta'])

    def test_large_pairwise_distance_can_have_zero_shared_shift(self):
        d = np.tile([1., -1.], 12)[:, None, None, None] * np.array([8., 0., 0., 0.])
        fit = fit_subspaces(self.n + d, self.n, self.z, self.groups, rank=1)
        self.assertGreater(float(fit['paired_rms'][0, 0]), 7.)
        np.testing.assert_allclose(fit['delta'], 0, atol=1e-14)
        np.testing.assert_allclose(fit['coherence'], 0, atol=1e-14)

    def test_replicating_a_source_group_does_not_increase_its_weight(self):
        fit = fit_subspaces(self.p, self.n, self.z, self.groups, rank=1)
        idx = np.r_[np.arange(24), np.tile(np.arange(8), 3)]
        repeated = fit_subspaces(self.p[idx], self.n[idx], self.z[idx], self.groups[idx], rank=1)
        np.testing.assert_allclose(fit['delta'], repeated['delta'], atol=1e-12)
        np.testing.assert_allclose(fit['eigenvalues'], repeated['eigenvalues'], atol=1e-12)

    def test_reverse_query_sign_and_reversed_holdout_reject_heads(self):
        fit = fit_subspaces(self.p, self.n, self.z, self.groups, rank=1)
        groups = np.repeat(['select_a', 'select_b', 'select_c'], 8)
        queries = np.broadcast_to([1., 0., 0., 0.], self.p.shape)
        valid = evaluate_heads(fit, self.p, self.n, self.z, queries, queries, groups, draws=30)
        self.assertEqual(select_heads(fit, valid), [(0, 0)])
        reverse_q = evaluate_heads(fit, self.p, self.n, self.z, -queries, -queries, groups, draws=30)
        self.assertEqual(select_heads(fit, reverse_q), [])
        reverse_label = evaluate_heads(fit, self.n, self.p, self.z, queries, queries, groups, draws=30)
        self.assertEqual(select_heads(fit, reverse_label), [])

    def test_source_leakage_and_nonfinite_are_rejected(self):
        fit = fit_subspaces(self.p, self.n, self.z, self.groups, rank=1)
        with self.assertRaises(ValueError):
            evaluate_heads(fit, self.p, self.n, self.z, self.p, self.p, self.groups)
        broken = self.p.copy()
        broken[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_subspaces(broken, self.n, self.z, self.groups)

    def test_zero_signal_has_no_arbitrary_basis_or_selection(self):
        fit = fit_subspaces(self.n, self.n, self.z, self.groups, rank=2)
        np.testing.assert_array_equal(fit['basis'], 0.)
        np.testing.assert_array_equal(fit['delta'], 0.)


if __name__ == '__main__':
    unittest.main()
