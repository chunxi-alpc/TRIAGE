"""Meaningful CPU numerical properties; execute with Python unittest."""
import math
import unittest
import numpy as np
from geometry import (anchor_factors, anchor_audit, bad_anchor_tokens, rotate,
                      random_controls, bf16_round, bf16_rotate, scalar_search)


class GeometryProperties(unittest.TestCase):
    def test_b4_second_moment_cannot_be_label_shuffle_control(self):
        rng = np.random.default_rng(7)
        differences = rng.normal(size=(96, 16))
        swaps = rng.choice([-1, 1], size=96)
        np.testing.assert_array_equal(differences.T @ differences,
                                      (differences * swaps[:, None]).T @ (differences * swaps[:, None]))

    def test_controls_match_rank_and_use_balanced_fixed_label_swaps(self):
        signal = np.random.default_rng(2).normal(size=(96, 16))
        a = random_controls(16, 4, signal, 261019)
        b = random_controls(16, 4, signal, 261019)
        p = a['random_projector']
        np.testing.assert_allclose(p @ p, p, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(p), 4)
        self.assertEqual(int(np.sum(a['pair_label_swaps'])), 0)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a['shuffled_mean'], (signal * a['pair_label_swaps'][:, None]).mean(0))

    def test_group_sign_and_mean_anchor_bound_with_negative_label(self):
        # Heads agree for first two tokens; token 3 has opposing head signs.
        q = np.array([[2., .2, 0., 0.], [1., -.2, 0., 0.],
                      [3., .3, 0., 0.], [2., -.3, 0., 0.]])
        u = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]])
        factors = anchor_factors(q, u)
        np.testing.assert_array_equal(factors['eligible'], [True, True, False])
        for sign in [-1, 1]:
            strength = 0.37
            # The label sign must remain OUTSIDE gamma.
            edited = strength * sign * factors['gamma'][:, None] * u
            delta = q @ edited.T / math.sqrt(4)
            self.assertTrue(np.all(sign * delta >= 0))
            mean_delta = q.mean(0) @ edited.T / math.sqrt(4)
            self.assertTrue(np.all(np.abs(mean_delta) < strength))
            np.testing.assert_allclose(mean_delta, strength * sign * factors['anchor_kappa'])

    def test_zero_direction_and_opposed_group_are_skipped(self):
        q = np.array([[1., 0.], [-1., 0.]])
        u = np.array([[1., 0.], [0., 0.]])
        x = anchor_factors(q, u)
        np.testing.assert_array_equal(x['eligible'], [False, False])
        np.testing.assert_array_equal(x['gamma'], [0., 0.])

    def test_exact_rotated_key_change_matches_explicit_score_change(self):
        rng = np.random.default_rng(3)
        q, k, u = rng.normal(size=(3, 8))
        theta = np.tile(rng.normal(size=4), 2)
        c, s = np.cos(theta), np.sin(theta)
        q_rot = rotate(q, c, s)
        lhs = q_rot @ (rotate(k + .12*u, c, s) - rotate(k, c, s))
        rhs = .12 * q_rot @ rotate(u, c, s)
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_rounding_and_discrete_rms_search_are_measured(self):
        key = bf16_round(np.array([.75, -.5, .3, 1.2], dtype=np.float32))
        direction = np.array([.2, -.1, .3, .1], dtype=np.float32)
        q = np.array([[.4, -.2, .6, .1], [.7, .4, -.2, .1]])
        cosine = bf16_round(np.tile(np.cos([.25, -.75]), 2))
        sine = bf16_round(np.tile(np.sin([.25, -.75]), 2))
        original = bf16_rotate(key, cosine, sine)
        def evaluate(a):
            changed = bf16_round(key + a*direction)
            scores = q @ (bf16_rotate(changed, cosine, sine) - original) / 2
            return changed, float(np.sqrt(np.mean(scores**2)))
        changed, audit = scalar_search(evaluate, .05)
        self.assertLessEqual(audit['relative_error'], .15)
        self.assertEqual(evaluate(audit['scale'])[1], audit['actual_rms'])
        zero, audit_zero = scalar_search(evaluate, 0)
        np.testing.assert_array_equal(zero, key)
        self.assertEqual(audit_zero['actual_rms'], 0)

    def test_nonnegative_certificate_does_not_hide_zero_coverage(self):
        audit = anchor_audit(np.zeros((4, 3)), -1, np.zeros(3, dtype=bool))
        self.assertEqual(audit['anchor_nonnegative_fraction'], 1)
        self.assertEqual(audit['anchor_strict_correct_fraction'], 0)
        self.assertEqual(audit['retained_token_fraction'], 0)
        self.assertIsNone(audit['anchor_active_correct_fraction'])

    def test_rounding_violation_requires_every_group_head_to_be_checked(self):
        actual = np.array([[-.1, -.2], [-.1, +.001], [-.1, -.1], [-.2, -.1]])
        np.testing.assert_array_equal(bad_anchor_tokens(actual, -1), [False, True])
        audit = anchor_audit(actual, -1)
        self.assertFalse(audit['anchor_group_all_nonnegative'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
