"""Synthetic in-memory checks; no fabricated experiment output is persisted."""
import json
import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from run import predict_record, audit_rows, setup_model, RunBlocked, METHODS, HERE, read
from analyze import nd, point, resampled, interval, validate_journals, validate_zero_checks


def fixture_record(family, variant, applicable, correct=True):
    gold = 'A' if applicable else 'B'
    prediction = gold if correct else ('B' if applicable else 'A')
    logits = [-1.0]*4
    logits['ABCD'.index(prediction)] = 1.0
    probs = np.exp(logits)/np.exp(logits).sum()
    row = dict(case_id=f'{family}-{variant}-{applicable}', pair_id=f'{family}-{variant}',
               group_id=family, applicable=applicable, gold_answer=gold, target_answer='A',
               dataset_id='synthetic_unit_test_only', variant=variant, variant_group_id=f'{family}-base')
    return predict_record(row, 'baseline', probs.tolist(), logits, {}, 'sealed_mechanism', 0, 0)


class StatisticsProperties(unittest.TestCase):
    def zero_fixture(self):
        protocol = read(HERE / 'protocol.json')
        records = [fixture_record('unit-a', 'canonical', True), fixture_record('unit-b', 'canonical', True)]
        for record, n in zip(records, (17, 29)):
            record['prompt_tokens'] = n
        checks = []
        for index, base in enumerate(records):
            for method in METHODS if index == 0 else ('direct_bias',):
                checks.append(dict(case_id=base['case_id'], method=method, tau=0.0,
                                   prompt_tokens=base['prompt_tokens'],
                                   candidate_logits=base['candidate_logits'][:],
                                   candidate_probabilities=base['candidate_probabilities'][:],
                                   identical_candidate_logits=True, identical_candidate_probabilities=True,
                                   mask_head_expansion_checked=method == 'direct_bias'))
        zero = dict(status='passed_real_model_checks', policy=protocol['zero_checks'],
                    expected_direct_bias_cases=len(records), checks=checks)
        return zero, records, protocol

    def test_zero_journal_requires_each_case_not_just_each_method(self):
        zero, records, protocol = self.zero_fixture()
        self.assertEqual(validate_zero_checks(zero, records, protocol), 12)
        old_first_case_only = copy.deepcopy(zero)
        old_first_case_only['checks'].pop()
        with self.assertRaisesRegex(AssertionError, 'per-case direct-bias zero coverage'):
            validate_zero_checks(old_first_case_only, records, protocol)
        duplicated = copy.deepcopy(zero)
        duplicated['checks'][-1] = copy.deepcopy(duplicated['checks'][-2])
        with self.assertRaisesRegex(AssertionError, 'per-case direct-bias zero coverage'):
            validate_zero_checks(duplicated, records, protocol)

    def test_zero_journal_rejects_hidden_score_or_prompt_length_differences(self):
        zero, records, protocol = self.zero_fixture()
        altered_scores = copy.deepcopy(zero)
        altered_scores['checks'][-1]['candidate_logits'][0] += 1e-8
        with self.assertRaisesRegex(AssertionError, 'exact candidate equality'):
            validate_zero_checks(altered_scores, records, protocol)
        wrong_length = copy.deepcopy(zero)
        wrong_length['checks'][-1]['prompt_tokens'] = records[0]['prompt_tokens']
        with self.assertRaisesRegex(AssertionError, 'same input sequence length'):
            validate_zero_checks(wrong_length, records, protocol)

    def test_schema_retains_variant_identity_and_strict_pair_denominator(self):
        data = [fixture_record('unit_family', variant, applicable,
                               correct=not (variant == 'paraphrase' and not applicable))
                for variant in ('canonical', 'paraphrase', 'permuted') for applicable in (True, False)]
        self.assertTrue(all(r['dataset_id'] == 'synthetic_unit_test_only' for r in data))
        baseline = {r['case_id']: r for r in data}
        self.assertEqual(nd(data, 'paired_accuracy', baseline), (2, 3))
        self.assertEqual(nd(data, 'all_variant_pair_accuracy', baseline), (0, 1))
        self.assertAlmostEqual(point(data, 'accuracy', baseline), 5/6)
        canonical = [r for r in data if r['variant'] == 'canonical']
        self.assertEqual(nd(canonical, 'paired_accuracy', baseline), (1, 1))
        with self.assertRaises(ValueError):
            nd(data[:-1], 'all_variant_pair_accuracy', baseline)

    def test_family_draws_preserve_pairs_and_variant_clusters(self):
        a = [fixture_record('a', v, app) for v in ('canonical', 'paraphrase', 'permuted') for app in (True, False)]
        b = [fixture_record('b', v, app, correct=app) for v in ('canonical', 'paraphrase', 'permuted') for app in (True, False)]
        baseline = {r['case_id']: r for r in a+b}
        weights = np.array([[2, 0], [1, 1], [0, 2]])
        values = resampled(a+b, 'paired_accuracy', ['a', 'b'], weights, baseline)
        np.testing.assert_array_equal(values, [1, .5, 0])
        np.testing.assert_array_equal(resampled(a+b, 'all_variant_pair_accuracy', ['a', 'b'], weights, baseline), values)
        self.assertEqual(interval(values-values), [0, 0])
        h = resampled(a+b, 'h_accuracy', ['a', 'b'], weights, baseline)
        np.testing.assert_allclose(h, [1, 2/3, 0])

    def test_blocked_run_cannot_be_promoted_to_performance_report(self):
        with tempfile.TemporaryDirectory(prefix='care-analysis-rejection-test-') as name:
            folder = Path(name)/'fold0'
            folder.mkdir()
            (folder/'run_manifest.json').write_text(json.dumps(dict(status='blocked_real_inference', real_model_inference_completed=False)))
            with self.assertRaisesRegex(RuntimeError, 'has not completed real inference'):
                validate_journals(Path(name))

    def test_oracle_schema_does_not_accept_unknown_as_false(self):
        prompt = 'Question: routing\nPatient: unknown\nRule: Route Cedar\nOptions:\nA. Cedar\nB. Maple\nC. Juniper\nD. Unknown\nAnswer:'
        start = prompt.index('Route Cedar')
        row = dict(case_id='unit', pair_id='unit-pair', group_id='unit-family',
                   prompt=prompt, spans={'action': [start, start+len('Route Cedar')]},
                   target_action='Route Cedar', applicable=None, gold_answer='D', target_answer='A',
                   option_map=dict(A='Cedar', B='Maple', C='Juniper', D='Unknown'))
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            audit_rows([row], 'legacy384_regression')

    def test_missing_torch_is_blocked_before_any_model_or_gpu_access(self):
        # Deliberately no args/model object: the dependency gate must happen first.
        with mock.patch('run.importlib.util.find_spec', return_value=None):
            with self.assertRaisesRegex(RunBlocked, 'has not run'):
                setup_model(None, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
