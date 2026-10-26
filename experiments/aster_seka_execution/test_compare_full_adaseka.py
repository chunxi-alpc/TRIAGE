"""Server-only synthetic CPU tests; no actual source/model/test data is read."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import compare_full_adaseka as entrypoint
from compare_full_adaseka import AMPLITUDES, NAMES, PIN, analyze, compare_prompt_calls, helpers, load_run

helpers()
from compare_standard import sha


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    data_path = root / 'counterfact.jsonl'
    data_path.write_text('synthetic entity source only')
    expert_paths = {}
    for name in ('synthetic', 'counterfact', 'biasbios', 'hotpotqa'):
        path = root / (name + '.pt')
        path.write_bytes(b'synthetic projection ' + name.encode())
        expert_paths[name] = str(path)
    expert_config = root / 'experts.json'
    dump(expert_config, expert_paths)
    inputs = {str(path): sha(path) for path in [data_path, expert_config, *(Path(x) for x in expert_paths.values())]}
    values = {
        'adaseka_zero': {'efficacy': [[1], [0], [1], [0]]},
        'adaseka_default1': {'efficacy': [[1], [1], [0], [0]], 'paraphrase': [[1, 1], [1, 0], [0, 0], [1, 0]]},
        'adaseka_paper3': {'efficacy': [[0], [1], [0], [0]], 'paraphrase': [[1, 0], [0, 0], [1, 0], [0, 0]]}}
    loaded = {}
    for name in NAMES:
        folder = root / name
        manifest_path = folder / 'run_manifest.json'
        traces, exports = [], {}
        for benchmark, cases in values[name].items():
            predictions, case_metrics = [], []
            for index, scores in enumerate(cases):
                native_indices = list(range(len(scores)))
                if name == 'adaseka_default1' and benchmark == 'paraphrase':
                    native_indices.reverse()  # Genuine per-case input matching survives ordering changes.
                margins = []
                for prompt_index in native_indices:
                    win = scores[prompt_index]
                    target, comparator = (-1., -2.) if win else (-2., -1.)
                    margin = math.exp(target) - math.exp(comparator)
                    prompt_hash = hashed(benchmark + ':' + str(index) + ':' + str(prompt_index))
                    predictions.append({'id': str(index), 'prompt_index': prompt_index, 'prompt_sha256': prompt_hash,
                        'target_score': target, 'comparator_score': comparator, 'probability_margin': margin, 'target_wins': bool(win)})
                    margins.append(margin)
                    ids = {'sha256': hashed('ids:' + prompt_hash), 'dtype': 'torch.int64', 'shape': [1, 5]}
                    mask = {'sha256': hashed('ones'), 'dtype': 'torch.int64', 'shape': [1, 5]}
                    traces.append({'completed_call_index': len(traces) + 1, 'input_ids': ids, 'attention_mask': None})
                    traces.append({'completed_call_index': len(traces) + 1, 'input_ids': ids, 'attention_mask': mask})
                case_metrics.append({'id': str(index), 'prompt_count': len(scores),
                    'score': sum(scores) / len(scores), 'magnitude': sum(margins) / len(margins)})
            export = folder / benchmark
            lines(export / 'predictions.jsonl', predictions)
            lines(export / 'case_metrics.jsonl', case_metrics)
            exports[benchmark] = (export, case_metrics)
        trace_path = folder / 'forward_inputs.jsonl'
        lines(trace_path, traces)
        manifest = {'process_state': 'exited', 'exit_code': 0, 'first_forward_parameter_dtype': 'torch.float32',
            'module_args': ['--adaptive-seka', '--adaptive_amplify_factor', str(AMPLITUDES[name]),
                '--batch_size', '1', '--adaptive-expert-path', str(expert_config), '--benchmarks', *values[name]],
            'command': ['runner.py', '--source', str(root)], 'input_files_sha256': inputs,
            'source_commit': PIN, 'source_files_sha256': {'official.py': 'synthetic-source-hash'},
            'model_files_sha256': {'weights.safetensors': 'synthetic-model-hash'},
            'forward_attempts': len(traces), 'forward_completed': len(traces),
            'forward_input_trace': str(trace_path), 'forward_input_trace_rows': len(traces),
            'forward_input_trace_sha256': sha(trace_path), 'elapsed_seconds': 1.}
        dump(manifest_path, manifest)
        for benchmark, (export, case_metrics) in exports.items():
            dump(export / 'audit.json', {'status': 'complete_verified', 'benchmark': benchmark,
                'anonymous_predictions_sha256': sha(export / 'predictions.jsonl'),
                'case_metrics_sha256': sha(export / 'case_metrics.jsonl'), 'run_manifest_sha256': sha(manifest_path),
                'raw_file_sha256': 'synthetic-raw-hash', 'recomputed_metrics': {
                    'score': {'mean': sum(r['score'] for r in case_metrics) / 4},
                    'magnitude': {'mean': sum(r['magnitude'] for r in case_metrics) / 4}}})
        loaded[name] = load_run(name, folder, manifest_path, data_path, sha(data_path), 4)
    mapping = {str(index): {'sample_id': hashed(str(index)), 'entity_cluster': 'e0' if index < 2 else 'e1',
                           'absolute_processed_row_index': 5000 + index} for index in range(4)}
    return loaded, mapping, data_path


class AdaStatisticsTests(unittest.TestCase):
    def test_cli_writes_success_and_preserves_original_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded, mapping, data_path = fixture(root)
            argv = ['compare_full_adaseka.py', '--data', str(data_path), '--bootstrap-samples', '100']
            for suffix, name in zip(('paper3', 'default1', 'zero'), NAMES):
                argv += ['--' + suffix + '-export-root', str(root / name),
                         '--' + suffix + '-manifest', loaded[name]['manifest_path']]
            # This is the real CLI parsing/final report path. Only the fixed 5000
            # source loader is replaced with four synthetic observations; no real
            # data or model runs are substituted into an actual research report.
            success = root / 'cli_success'
            with mock.patch.object(sys, 'argv', argv + ['--output-dir', str(success)]), \
                 mock.patch('counterfact_entity_ci.source_clusters', return_value=(mapping, sha(data_path))), \
                 mock.patch.object(entrypoint, 'load_run', side_effect=lambda name, *args, **kwargs: loaded[name]):
                self.assertEqual(entrypoint.main(), 0)
            report = json.loads((success / 'comparison.json').read_text())
            self.assertEqual(report['status'], 'complete_verified')
            self.assertEqual(report['checker_sha256'], sha(Path(entrypoint.__file__)))
            self.assertEqual(set(report['helper_sources_sha256']),
                             {'compare_standard', 'counterfact_entity_ci', 'compare_adaptive_counterfact'})
            failed = root / 'cli_failure'
            with mock.patch.object(sys, 'argv', argv + ['--output-dir', str(failed)]), \
                 mock.patch('counterfact_entity_ci.source_clusters', side_effect=ValueError('synthetic original failure')):
                self.assertEqual(entrypoint.main(), 1)
            report = json.loads((failed / 'comparison.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error_type'], 'ValueError')
            self.assertEqual(report['error'], 'synthetic original failure')

    def test_all_pairs_negative_results_entity_ci_and_native_paraphrase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded, mapping, _ = fixture(root)
            output = root / 'comparison'
            output.mkdir()
            report = analyze(loaded, mapping, output, 100)
            self.assertEqual(report['status'], 'complete_verified')
            self.assertEqual(set(report['methods']), set(NAMES))
            self.assertEqual(len(report['comparisons']), 4)
            negative = report['comparisons']['efficacy_adaseka_paper3_vs_adaseka_zero']
            self.assertEqual(negative['score_effect_percentage_points'], -25.)
            self.assertEqual((negative['corrected'], negative['regressed']), (1, 2))
            self.assertIsNotNone(negative['ci95_score_difference'])
            para = report['comparisons']['paraphrase_adaseka_paper3_vs_adaseka_default1']
            self.assertIsNone(para['corrected'])
            self.assertIsNone(para['regressed'])
            self.assertEqual((para['cases_improved'], para['cases_degraded']), (1, 3))
            self.assertEqual(para['input_matching']['status'], 'complete_verified')
            self.assertFalse(para['input_matching']['ordered_full_segment_inputs_identical'])
            self.assertEqual(report['methods']['adaseka_paper3']['call_patterns']['efficacy']['routing_forwards'], 4)
            self.assertEqual(report['methods']['adaseka_paper3']['call_patterns']['paraphrase']['routing_forwards'], 8)
            self.assertIn('only on 16', report['nohook_scope'])

    def test_explicit_none_matches_but_missing_field_or_tensor_difference_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            loaded, _, _ = fixture(Path(temporary))
            paper, zero = loaded['adaseka_paper3'], loaded['adaseka_zero']
            result, _ = compare_prompt_calls(paper, zero, 'efficacy')
            self.assertEqual(result['status'], 'complete_verified')
            call = next(iter(paper['calls']['efficacy'].values()))['routing']
            del call['attention_mask']
            result, _ = compare_prompt_calls(paper, zero, 'efficacy')
            self.assertEqual(result['status'], 'blocked')
            call['attention_mask'] = {'sha256': 'different', 'dtype': 'torch.int64', 'shape': [1, 5]}
            result, _ = compare_prompt_calls(paper, zero, 'efficacy')
            self.assertEqual(result['status'], 'failed')

    def test_actual_forward_count_and_fp32_label_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded, _, data_path = fixture(root)
            name = 'adaseka_zero'
            path = Path(loaded[name]['manifest_path'])
            manifest = json.loads(path.read_text())
            manifest['first_forward_parameter_dtype'] = 'torch.bfloat16'
            dump(path, manifest)
            with self.assertRaisesRegex(ValueError, 'FP32'):
                load_run(name, path.parent, path, data_path, sha(data_path), 4)
            manifest['first_forward_parameter_dtype'] = 'torch.float32'
            manifest['forward_attempts'] += 1
            dump(path, manifest)
            audit_path = path.parent / 'efficacy/audit.json'
            audit = json.loads(audit_path.read_text())
            audit['run_manifest_sha256'] = sha(path)
            dump(audit_path, audit)
            with self.assertRaisesRegex(ValueError, 'forward total'):
                load_run(name, path.parent, path, data_path, sha(data_path), 4)


if __name__ == '__main__':
    unittest.main()
