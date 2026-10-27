"""Synthetic statistical checks; no model is loaded or evaluated here."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_module2_causal_pilot import (
    calibration_status, enforce_zero_identity, summarize,
)


class CausalSummaryTests(unittest.TestCase):
    methods = ('projected_mean', 'direct_score_bias')
    taus = (0., .1)

    def records(self, deltas=None):
        """Three groups with unequal numbers of pairs expose naive pooling."""
        deltas = deltas or {'projected_mean': [2., 2., 2.],
                            'direct_score_bias': [1., 1., 1.]}
        rows = []
        for method in self.methods:
            for tau in self.taus:
                for group in range(3):
                    for pair in range(group + 1):
                        for variant in ('positive', 'negative'):
                            baseline_margin = (1. if group == 2 else -1.) + pair * .01
                            delta = 0. if tau == 0 else deltas[method][group]
                            margin = baseline_margin + delta
                            baseline_mass = .2 + group * .01
                            mass = baseline_mass + delta * .01
                            rows.append({'method': method, 'tau': tau,
                                'id': f'g{group}_p{pair}', 'group_id': f'g{group}',
                                'variant': variant, 'status': 'completed',
                                'correct': margin > 0., 'baseline_correct': baseline_margin > 0.,
                                'gold_logit_margin': margin, 'gold_attention_mass': mass,
                                'target_attention_mass': mass if variant == 'positive' else .1 - delta * .01})
        return rows

    def summarize(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'causal_metrics.csv'
            result = summarize(rows, output, seed=7,
                               expected_methods=self.methods, expected_taus=self.taus)
            for name in ('causal_metrics.csv', 'causal_coverage.csv',
                         'paired_comparisons.csv', 'paired_coverage.csv'):
                self.assertTrue((Path(directory) / name).exists())
            return result

    def comparison(self, result, metric, reference='baseline'):
        return next(r for r in result['paired_comparisons'] if
                    r['method'] == 'projected_mean' and r['tau'] == .1 and
                    r['metric'] == metric and r['reference'] == reference)

    def test_paired_effects_weight_source_groups_equally(self):
        result = self.summarize(self.records({
            'projected_mean': [2., 0., -2.], 'direct_score_bias': [0., 0., 0.]}))
        for metric in ('case_accuracy', 'pair_accuracy', 'gold_logit_margin'):
            row = self.comparison(result, metric)
            self.assertAlmostEqual(row['delta_method_minus_reference'], 0., places=12)
            self.assertEqual(row['source_guidelines'], 3)
        self.assertEqual(self.comparison(result, 'case_accuracy')['observations'], 12)
        self.assertEqual(self.comparison(result, 'pair_accuracy')['observations'], 6)

    def test_pairing_uses_identity_not_journal_order(self):
        rows = self.records()
        order = np.random.default_rng(9).permutation(len(rows))
        result = self.summarize([rows[i] for i in order])
        for reference, expected in [('baseline', 2.), ('direct_score_bias', 1.)]:
            row = self.comparison(result, 'gold_logit_margin', reference)
            self.assertAlmostEqual(row['delta_method_minus_reference'], expected, places=12)
            self.assertAlmostEqual(row['ci_low'], expected, places=12)
            self.assertAlmostEqual(row['ci_high'], expected, places=12)

    def test_blocked_condition_preserves_coverage_and_blocks_whole_estimand(self):
        rows = self.records()
        row = next(r for r in rows if r['method'] == 'projected_mean' and r['tau'] == .1)
        row['status'] = 'blocked_zero_fixed_query_response'
        for key in ('correct', 'gold_logit_margin', 'gold_attention_mass', 'target_attention_mass'):
            del row[key]
        result = self.summarize(rows)
        self.assertEqual(result['status'], 'partial')
        coverage = next(r for r in result['coverage'] if r['method'] == 'projected_mean' and r['tau'] == .1)
        self.assertEqual(coverage['blocked'], 1)
        self.assertEqual(coverage['completed'], 11)
        self.assertIn('blocked_zero_fixed_query_response', coverage['reasons'])
        self.assertFalse(any(r['method'] == 'projected_mean' and r['tau'] == .1 for r in result['metrics']))
        self.assertFalse(any(r['method'] == 'projected_mean' and r['tau'] == .1 for r in result['paired_comparisons']))
        self.assertTrue(any(r['method'] == 'projected_mean' and r['tau'] == .1
                            and r['status'].startswith('blocked') for r in result['paired_coverage']))

    def test_entire_missing_dose_is_visible_without_subset_estimates(self):
        rows = [r for r in self.records() if not (r['method'] == 'projected_mean' and r['tau'] == .1)]
        result = self.summarize(rows)
        coverage = next(r for r in result['coverage'] if r['method'] == 'projected_mean' and r['tau'] == .1)
        self.assertEqual(coverage['missing'], 12)
        self.assertEqual(coverage['observed'], 0)
        self.assertEqual(result['status'], 'partial')

    def test_zero_effect_has_exact_zero_paired_difference(self):
        result = self.summarize(self.records({
            'projected_mean': [0., 0., 0.], 'direct_score_bias': [0., 0., 0.]}))
        self.assertEqual(result['status'], 'completed')
        for row in result['paired_comparisons']:
            self.assertEqual(row['delta_method_minus_reference'], 0.)
            self.assertEqual(row['ci_low'], 0.)
            self.assertEqual(row['ci_high'], 0.)

    def test_duplicate_conditions_and_changed_groups_are_rejected(self):
        rows = self.records()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.summarize(rows + [copy.deepcopy(rows[0])])
        rows = self.records()
        rows[-1]['group_id'] = 'wrong_group'
        with self.assertRaisesRegex(ValueError, 'source group'):
            self.summarize(rows)

    def test_zero_doses_must_share_the_same_baseline(self):
        rows = self.records()
        rows[0]['gold_logit_margin'] += .01
        with self.assertRaisesRegex(ValueError, 'Zero-dose'):
            self.summarize(rows)

    def test_score_mismatch_is_blocked_at_the_prespecified_threshold(self):
        self.assertEqual(calibration_status(.1049, .1)[0], 'completed')
        self.assertEqual(calibration_status(.105, .1)[0], 'completed')
        self.assertEqual(calibration_status(.106, .1)[0], 'blocked_score_rms_mismatch')
        self.assertEqual(calibration_status(0., .1)[0], 'blocked_score_rms_mismatch')
        with self.assertRaises(ValueError):
            calibration_status(float('nan'), .1)

    def test_zero_intervention_errors_fail_closed(self):
        enforce_zero_identity(1e-4, 0.)
        with self.assertRaisesRegex(RuntimeError, 'Full/prefix'):
            enforce_zero_identity(1.001e-4, 0.)
        with self.assertRaisesRegex(RuntimeError, 'exact identities'):
            enforce_zero_identity(0., 1e-12)


if __name__ == '__main__':
    unittest.main()
