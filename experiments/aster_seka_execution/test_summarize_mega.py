"""Server-only synthetic tests. No model, dataset, labels, GPU or subprocess."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data_audit'))
from summarize_mega import measurement, numerical, read, sha, summarize


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dump_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def measured(calls):
    return {'actual_forward_calls': calls, 'wall_seconds': calls / 10,
            'peak_allocated_gib': 2., 'peak_reserved_gib': 3.}


def fixture(root):
    ids = ['s0', 's1', 's2']
    primary, extra, evaluation = root / 'primary', root / 'extra', root / 'evaluation'
    frozen = root / 'frozen.json'
    dump(frozen, {'split': 'train_val', 'selected': [
        {'sample_id': sid, 'source_cluster': 'g0' if index < 2 else 'g1'} for index, sid in enumerate(ids)]})
    primary_protocol = {'seed': 0, 'input_manifest_sha256': sha(frozen), 'model_fingerprint': 'synthetic_model',
                        'arms': ['A0B0', 'A1B0', 'A1B1_zero']}
    dump(primary / 'protocol.json', primary_protocol)
    stages = []
    for index, sid in enumerate(ids):
        stages.append({'sample_id': sid, 'stage_configuration': 'guideline_extraction',
                       'cache_key': 'c0' if index < 2 else 'c1'})
        for variant, calls in [('A0', 1), ('A1', 2)]:
            stages.append({'sample_id': sid, 'stage_configuration': variant, 'rule_id': 'R1',
                           'generation': measured(calls)})
    dump_rows(primary / 'stage_predictions.private.jsonl', stages)
    for key, calls in [('c0', 10), ('c1', 20)]:
        dump(primary / 'private_guideline_extractions' / (key + '.json'),
             {'cache_key': key, 'rules': [{'rule_id': 'R1'}], 'generation': measured(calls)})
    for method, calls in [('A0B0', 3), ('A1B0', 4), ('A1B1_zero', 4)]:
        data = []
        for sid in ids:
            generation = measured(calls)
            if method.endswith('_zero'):
                generation['numerical_control'] = {'max_abs_first_token_score_difference': 0., 'argmax_changed': False}
            data.append({'sample_id': sid, 'method': method, 'status': 'ok', 'generation': generation})
        dump_rows(primary / (method + '.predictions.private.jsonl'), data)
    dump(extra / 'protocol.json', {'seed': 0, 'condition_variant': 'A1', 'model_fingerprint': 'synthetic_model',
        'input_manifest_sha256': sha(frozen), 'primary_protocol_sha256': sha(primary / 'protocol.json'),
        'primary_stages_sha256': sha(primary / 'stage_predictions.private.jsonl'),
        'methods': ['text_two_stage', 'fusion_static']})
    dump_rows(extra / 'text_two_stage.predictions.private.jsonl', [
        {'sample_id': sid, 'method': 'text_two_stage', 'status': 'ok', 'generation': {
            **measured(0), 'reused_from': str(primary / 'A1B0.predictions.private.jsonl'),
            'reused_source_sha256': sha(primary / 'A1B0.predictions.private.jsonl')}} for sid in ids])
    dump_rows(extra / 'fusion_static.predictions.private.jsonl', [
        {'sample_id': sid, 'method': 'fusion_static', 'status': 'error' if sid == 's2' else 'ok',
         'generation': measured(5)} for sid in ids])
    dump_rows(extra / 'numeric_probes.private.jsonl', [{'sample_id': 's0', **measured(1)}])
    methods = [
        {'name': 'A0B0', 'prediction': str(primary / 'A0B0.predictions.private.jsonl'), 'condition': 'A0', 'calibration_jobs': []},
        {'name': 'A1B1_zero', 'prediction': str(primary / 'A1B1_zero.predictions.private.jsonl'), 'condition': 'A1', 'calibration_jobs': ['seka_build']},
        {'name': 'A1_text', 'generation_method': 'text_two_stage', 'prediction': str(extra / 'text_two_stage.predictions.private.jsonl'), 'condition': 'A1', 'calibration_jobs': ['seka_build']},
        {'name': 'A1_fusion', 'generation_method': 'fusion_static', 'prediction': str(extra / 'fusion_static.predictions.private.jsonl'), 'condition': 'A1', 'calibration_jobs': ['seka_build']}]
    for item in methods:
        item['evaluation_dir'] = str(evaluation)
    scores = {'A0B0': [(.4, .6), (.4, .6), (.4, .6)], 'A1B1_zero': [(.4, .6)] * 3,
              'A1_text': [(.5, .5), (.5, .5), (.5, .5)],
              'A1_fusion': [(.8, .2), (.4, .6), (0., 1.)]}
    metrics = []
    for name, values in scores.items():
        for index, (acc, pcc) in enumerate(values):
            failed = name == 'A1_fusion' and index == 2
            metrics.append({'sample_id': ids[index], 'method': name, 'method_seed': 0,
                'tree_cluster': 'g0' if index < 2 else 'g1', 'parsed_success': not failed,
                'accuracy_valid': not failed, 'pathway_valid': not failed, 'acc_normalized': acc, 'pcc_dtw_cost': pcc})
    dump_rows(evaluation / 'case_metrics.jsonl', metrics)
    dump(evaluation / 'evaluation_config.json', {'freeze_manifest_sha256': sha(frozen),
        'prediction_sha256': {item['name']: sha(item['prediction']) for item in methods},
        'judge': {'backend': 'local', 'served_name': 'synthetic_Qwen'},
        'official_code_commit': 'synthetic_official', 'official_source_files_sha256': {}})
    dump(evaluation / 'evaluation_report.json', {'status': 'complete_verified'})
    dump(root / 'build.json', {'process_state': 'exited', 'exit_code': 0, 'forward_completed': 100,
                             'forward_attempts': 100, 'elapsed_seconds': 10.})
    return {'freeze_manifest': str(frozen), 'primary_run': str(primary), 'methods': methods,
            'calibration_jobs': {'seka_build': {'family': 'seka', 'run_manifest': str(root / 'build.json')}},
            'comparisons': [{'method': 'A1_fusion', 'reference': 'A1_text'}, {'method': 'A1_text', 'reference': 'A0B0'}]}


class SummaryTests(unittest.TestCase):
    def test_continuous_directions_full_denominator_and_cost_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = summarize(fixture(root), root, 100)
            self.assertEqual(result['n_fixed_cases'], 3)
            self.assertEqual(result['n_guideline_groups'], 2)
            fused = result['methods']['A1_fusion']
            self.assertEqual(fused['accuracy_failures_scored_zero'], 1)
            self.assertAlmostEqual(fused['metrics']['acc_normalized']['mean'], .4)
            self.assertEqual(fused['cost']['shared_extraction']['forward_completed_total'], 30)
            self.assertEqual(fused['cost']['shared_condition']['forward_completed_total'], 6)
            self.assertEqual(fused['cost']['method_inference_total']['forward_completed_total'], 51)
            reused = result['methods']['A1_text']
            self.assertEqual(reused['new_execution_in_this_file']['forward_completed_total'], 0)
            self.assertEqual(reused['cost']['execution_including_reused_original']['forward_completed_total'], 12)
            self.assertEqual(reused['cost']['method_inference_total']['forward_completed_total'], 48)
            self.assertEqual(result['actual_observed_generation_ledger']['forward_completed_total'], 88)
            self.assertEqual(len(result['calibration_jobs_counted_once']), 1)
            self.assertEqual(fused['cost']['calibration_families_without_recorded_cost'], ['fusion'])
            for item in result['comparisons'][:2]:
                self.assertEqual(item['score_improved'], 1)
                self.assertEqual(item['score_degraded'], 2)
                self.assertTrue(item['same_condition_outputs'])
                self.assertIsNone(item['binary_corrected'])
                self.assertNotIn('p_value', item)
                self.assertIsNotNone(item['ci95'])
            self.assertFalse(result['comparisons'][2]['same_condition_outputs'])
            self.assertFalse(result['independent_model_claim'])

    def test_missing_scored_case_or_changed_prediction_does_not_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fixture(root)
            metric_path = root / 'evaluation/case_metrics.jsonl'
            original = metric_path.read_text()
            metric_path.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'denominator'):
                summarize(spec, root, 100)
            metric_path.write_text(original)
            prediction = Path(spec['methods'][0]['prediction'])
            prediction.write_text(prediction.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'predictions changed'):
                summarize(spec, root, 100)

    def test_numerical_diagnostics_and_missing_attempts_are_not_fabricated(self):
        cost = measurement({'actual_forward_calls': 2, 'wall_seconds': 1., 'elapsed_seconds': .5})
        self.assertEqual(cost['seconds'], 1.)
        self.assertIsNone(cost['forward_attempts'])
        result = numerical([{'generation': {'numerical_control': {
            'max_abs_first_token_score_difference': .2, 'argmax_changed': True, 'same_full_input': False}}},
            {'generation': {}}])
        self.assertEqual(result['same_full_input_false'], 1)
        self.assertEqual(result['argmax_changed'], 1)
        self.assertEqual(result['diagnostics_missing'], 1)
        self.assertIn('not retained', result['scope'])


if __name__ == '__main__':
    unittest.main()
