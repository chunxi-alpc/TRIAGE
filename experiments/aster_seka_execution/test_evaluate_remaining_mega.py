"""Server CPU synthetic contracts for queue preparation and exact-file reuse."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evaluate_remaining_mega import cost_budget, evaluator_arguments, prepare, select_pending


class RemainingEvaluationTests(unittest.TestCase):
    def test_exact_file_reuse_retains_all_methods_and_native_parameters(self):
        methods = [{'name': 'primary__p' + str(i) if i < 11 else 'extra__m' + str(i),
                    'generation_method': 'p' + str(i), 'source_group': 'primary' if i < 11 else 'extra_A0',
                    'prediction': '/synthetic/' + str(i), 'prediction_sha256': 'synthetic_hash_' + str(i)} for i in range(43)]
        config = {'prediction_sha256': {'p' + str(i): methods[i]['prediction_sha256'] for i in range(11)},
                  'method_order': ['p' + str(i) for i in range(11)],
                  'judge': {'backend': 'local', 'expected_results_per_metric': 5, 'temperature': .7, 'top_p': .8,
                            'top_k': 20, 'thinking': False, 'served_name': 'Qwen3_8B', 'max_new_tokens': 512,
                            'context_length': 32768, 'local_rng_seed': 0}, 'statistics': {'bootstrap_samples': 10000}}
        methods[11]['prediction_sha256'] = methods[0]['prediction_sha256']
        methods[13]['prediction_sha256'] = methods[12]['prediction_sha256']
        pending, mapping, reuse = select_pending(methods, config, Path('/old'), Path('/new'))
        self.assertEqual((len(mapping), len(pending)), (43, 30))
        self.assertEqual(mapping[methods[11]['name']]['evaluation_dir'], '/old')
        self.assertEqual(mapping[methods[13]['name']], mapping[methods[12]['name']])
        self.assertEqual(sum(row['new_judge_requests'] == 0 for row in reuse), 13)
        self.assertTrue(all(row['fixed_denominator'] == 16 for row in reuse))
        args = SimpleNamespace(reader_dir=Path('/p'), train_val_root=Path('/data/train_val'),
                               freeze_manifest=Path('/frozen'), official_parser=Path('/official/parser'), model=Path('/model'))
        command = evaluator_arguments(args, config, pending, Path('/new'))
        self.assertEqual(command.count('--prediction'), 30)
        self.assertEqual(command[command.index('--judge-max-tokens') + 1], '512')
        self.assertEqual(command[command.index('--judge-seed') + 1], '0')
        self.assertNotIn('--expected-results', command)  # Existing native evaluator fixes 5; no new scoring parameter.
        changed = [dict(row) for row in methods]
        changed[0]['prediction_sha256'] = 'changed_after_primary_judging'
        with self.assertRaisesRegex(ValueError, 'primary prediction changed'):
            select_pending(changed, config, Path('/old'), Path('/new'))

    def test_prepare_writes_only_gpu0_dependency_queue_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(experiment_root=root / 'p', runtime_root=root / 'runtime', output_dir=root / 'remaining_job',
                gpu=0, freeze_manifest=root / 'frozen', train_val_root=root / 'train_val', official_parser=root / 'parser',
                model=root / 'model', primary_evaluation_manifest=root / 'primary_job/run_manifest.json',
                python=Path('/synthetic/python'), source=root / 'SEKA', plan=root / 'queue.json', wait_for_pids=[428868, 428866])
            self.assertEqual(prepare(args), 0)
            queue = json.loads(args.plan.read_text())
            self.assertEqual(queue['gpu'], 0)
            self.assertEqual(queue['wait_for_pids'], [428868, 428866])
            self.assertTrue(queue['no_process_started'])
            command = queue['jobs'][0]['command']
            self.assertEqual(command[command.index('--gpu') + 1], '0')
            self.assertIn('--reader-dir', command)
            self.assertNotIn('--runtime-root', command)  # Avoid recursively fingerprinting unrelated runtime outputs.
            self.assertEqual(queue['resources']['successful_request_budget_before_parse_failures'], 5120)
            self.assertEqual(queue['resources']['transport_attempt_upper_bound'], 15360)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(cost_budget(32, {'elapsed_seconds': 110.})['rough_wall_seconds_from_primary'], 320.)


if __name__ == '__main__':
    unittest.main()
