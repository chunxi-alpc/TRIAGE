"""Server-only 43-arm/16-case interface regression; all files are synthetic."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import summarize_all_mega as entrypoint
from summarize_mega import sha

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data_audit'))
import paired_statistics


METHODS = {
    'primary': ['full_guideline', 'A0B0', 'A0B1', 'A1B0', 'A1B1', 'A1B1_zero', 'A1B1_wrong',
                'A1B1_misaligned', 'A1B1_reverse', 'text_emphasis', 'predicted_crop'],
    'extra_A0': ['text_two_stage', 'text_emphasis', 'top1_crop', 'top2_crop', 'applicable_crop',
                 'fusion_static', 'fusion_dynamic', 'fusion_static_zero', 'fusion_dynamic_zero', 'cad',
                 'cad_zero', 'cad_reverse', 'spotlight_no_hook', 'spotlight', 'spotlight_zero',
                 'spotlight_wrong', 'spotlight_reverse', 'direct_action_mapping'],
    'extra_A1': ['text_two_stage', 'text_emphasis', 'fusion_static', 'fusion_dynamic',
                 'fusion_static_zero', 'fusion_dynamic_zero'],
    'attention_A0': ['seka_execution_reuse', 'pasta_no_hook', 'pasta_zero', 'pasta',
                     'adaseka_no_hook', 'adaseka_zero', 'adaseka_05', 'adaseka_3']}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def fixture(root):
    ids = ['s' + str(i) for i in range(16)]
    frozen = root / 'frozen.json'
    dump(frozen, {'split': 'train_val', 'selected': [{'sample_id': sid, 'source_cluster': 'group' + str(i // 8)}
                                                  for i, sid in enumerate(ids)]})
    primary = root / 'primary'
    stages = [{'sample_id': sid, 'stage_configuration': 'guideline_extraction', 'cache_key': 'cache' + str(i // 8)}
              for i, sid in enumerate(ids)]
    lines(primary / 'stage_predictions.private.jsonl', stages)
    for key in ('cache0', 'cache1'):
        dump(primary / 'private_guideline_extractions' / (key + '.json'),
             {'cache_key': key, 'rules': [], 'generation': {'actual_forward_calls': 1, 'wall_seconds': .1}})
    mapping, metric_rows = {}, []
    for prefix, methods in METHODS.items():
        directory = root / prefix
        protocol = {'seed': 0, 'expected_cases': 16, 'execution_status': 'finished',
                    'input_manifest_sha256': sha(frozen), 'model_fingerprint': 'synthetic_checkpoint',
                    ('arms' if prefix == 'primary' else 'methods'): methods}
        if prefix != 'primary':
            protocol['primary_protocol_sha256'] = sha(primary / 'protocol.json')
            protocol['primary_stage_sha256' if prefix == 'attention_A0' else 'primary_stages_sha256'] = sha(primary / 'stage_predictions.private.jsonl')
        if prefix.startswith('extra_'):
            protocol['condition_variant'] = prefix.removeprefix('extra_')
        dump(directory / 'protocol.json', protocol)
        for method in methods:
            data = []
            for i, sid in enumerate(ids):
                generation = {'actual_forward_calls': 1, 'elapsed_seconds': .1}
                if method == 'text_two_stage':
                    source = primary / (protocol['condition_variant'] + 'B0.predictions.private.jsonl')
                    generation.update(actual_forward_calls=0, elapsed_seconds=0., reused_from=str(source),
                                      reused_source_sha256=sha(source))
                elif method == 'seka_execution_reuse':
                    generation.update(actual_forward_calls=0, elapsed_seconds=0., reused_from='primary A0B1',
                                      source_sha256=sha(primary / 'A0B1.predictions.private.jsonl'))
                failed = prefix == 'extra_A0' and method == 'cad' and i == 0
                unavailable = method in {'A1B1_wrong', 'pasta_zero'} and i == 0
                status = 'error' if failed else 'control_unavailable' if unavailable else 'ok'
                if method.endswith('_zero'):
                    generation['numerical_control'] = {'max_abs_first_token_score_difference': 0.,
                        'argmax_changed': False, 'same_full_input': True}
                data.append({'sample_id': sid, 'method': method, 'synthetic_source_group': prefix,
                             'status': status, 'generation': generation})
                metric_rows.append({'sample_id': sid, 'method': prefix + '__' + method, 'method_seed': 0,
                    'tree_cluster': 'group' + str(i // 8), 'parsed_success': not failed,
                    'accuracy_valid': not failed, 'pathway_valid': not failed,
                    'acc_normalized': 0. if failed else .5, 'pcc_dtw_cost': 1. if failed else .5})
            path = directory / (method + '.predictions.private.jsonl')
            lines(path, data)
            mapping[prefix + '__' + method] = sha(path)
    evaluation = root / 'evaluation'
    dump(evaluation / 'evaluation_config.json', {'freeze_manifest_sha256': sha(frozen),
        'prediction_sha256': mapping, 'judge': {'backend': 'local', 'served_name': 'synthetic_Qwen'},
        'official_code_commit': 'synthetic_commit', 'official_source_files_sha256': {}})
    dump(evaluation / 'evaluation_report.json', {'status': 'complete_verified'})
    lines(evaluation / 'case_metrics.jsonl', metric_rows)
    argv = ['summarize_all_mega.py', '--freeze-manifest', str(frozen), '--evaluation-dir', str(evaluation),
            '--bootstrap-samples', '100', '--statistics-dir', str(Path(paired_statistics.__file__).parent)]
    for prefix, argument, _ in entrypoint.GROUPS:
        argv += ['--' + argument.replace('_', '-'), str(root / prefix)]
    return argv


class AllMethodsTests(unittest.TestCase):
    def test_real_entrypoint_all_43_methods_16_cases_failures_and_reused_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = fixture(root)
            output = root / 'summary'
            with mock.patch.object(sys, 'argv', argv + ['--output-dir', str(output)]):
                self.assertEqual(entrypoint.main(), 0)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['fixed_method_case_pairs'], 688)
            self.assertEqual(len(report['methods']), 43)
            self.assertTrue(all(row['n_fixed_cases'] == 16 for row in report['methods'].values()))
            failed = report['methods']['extra_A0__cad']
            self.assertEqual(failed['accuracy_failures_scored_zero'], 1)
            self.assertEqual(failed['generation_status_counts']['error'], 1)
            self.assertEqual(report['methods']['attention_A0__pasta_zero']['generation_status_counts']['control_unavailable'], 1)
            for name in ('extra_A0__text_two_stage', 'extra_A1__text_two_stage', 'attention_A0__seka_execution_reuse'):
                value = report['methods'][name]
                self.assertEqual(value['new_execution_in_this_file']['forward_completed_total'], 0)
                self.assertEqual(value['cost']['execution_including_reused_original']['forward_completed_total'], 16)
                self.assertEqual(value['cost']['shared_extraction']['forward_completed_total'], 2)
            self.assertEqual(report['actual_observed_generation_ledger']['forward_completed_total'], 642)
            self.assertIn('extra_A0__fusion_static', report['methods'])
            self.assertIn('extra_A1__fusion_static', report['methods'])

    def test_missing_judge_arm_is_not_silently_dropped_and_planning_never_judges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = fixture(root)
            path = root / 'evaluation/evaluation_config.json'
            value = json.loads(path.read_text())
            del value['prediction_sha256']['attention_A0__adaseka_3']
            dump(path, value)
            failed = root / 'failed'
            with mock.patch.object(sys, 'argv', argv + ['--output-dir', str(failed)]):
                self.assertEqual(entrypoint.main(), 1)
            report = json.loads((failed / 'summary.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('attention_A0__adaseka_3', report['error'])
            planned = root / 'planned'
            with mock.patch.object(sys, 'argv', argv + ['--output-dir', str(planned), '--prepare-evaluation-map-only']):
                self.assertEqual(entrypoint.main(), 0)
            report = json.loads((planned / 'summary.json').read_text())
            self.assertEqual(report['status'], 'planned')
            self.assertTrue(report['no_judge_started'])
            mapping = json.loads((planned / 'evaluation_predictions.json').read_text())
            self.assertEqual(len(mapping['prediction_mapping']), 43)


if __name__ == '__main__':
    unittest.main()
