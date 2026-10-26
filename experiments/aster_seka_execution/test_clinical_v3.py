import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from clinical_core import QuotedRule
from grounded_clinical_prompt import grounded_prompts
from request_keyed_judge import RequestKeyedRecorder, restore_response


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.torch = SimpleNamespace(manual_seed=self.set_seed,
                                     cuda=SimpleNamespace(manual_seed_all=lambda seed: None))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def set_seed(self, seed):
        self.seed = seed
    def create(self, **request):
        self.calls += 1
        return restore_response({'id': str(self.calls), 'model': 'fake',
                                 'choices': [{'message': {'role': 'assistant', 'content': str(self.seed)},
                                              'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})


class JudgeTests(unittest.TestCase):
    def recorder(self, root, label, client, namespace='v3'):
        return RequestKeyedRecorder(client, root / (label + '.raw'), root / (label + '.meta'),
                                    namespace=namespace, cache_dir=root / 'cache',
                                    events_path=root / (label + '.events'))
    def draw(self, recorder, method, answer='answer', sample='case'):
        recorder.context = {'method': method, 'sample_id': sample, 'metric': 'accuracy'}
        return [recorder.create(model='fake', messages=[{'role': 'user', 'content': answer}])
                .choices[0].message.content for _ in range(5)]
    def test_repeated_draws_not_collapsed_and_shared_across_methods(self):
        with TemporaryDirectory() as d:
            client = FakeClient(); r = self.recorder(Path(d), 'one', client)
            a = self.draw(r, 'A'); b = self.draw(r, 'B')
            self.assertEqual(a, b); self.assertEqual(len(set(a)), 5)
            self.assertEqual(client.calls, 5); self.assertEqual(r.cache_hits, 5)
            self.assertEqual(len(r.rows), 5)
            r.close()
    def test_order_independence_and_restart(self):
        with TemporaryDirectory() as d:
            root = Path(d); a = self.recorder(root, 'one', FakeClient())
            expected = self.draw(a, 'first'); a.close()
            client = FakeClient(); b = self.recorder(root, 'two', client)
            self.draw(b, 'unrelated', answer='different')
            self.assertEqual(self.draw(b, 'renamed'), expected)
            self.assertEqual(client.calls, 5); b.close()
    def test_namespace_case_and_reference_are_part_of_key(self):
        with TemporaryDirectory() as d:
            root = Path(d); c = FakeClient(); a = self.recorder(root, 'one', c)
            first = self.draw(a, 'A')
            self.assertNotEqual(first, self.draw(a, 'B', sample='other'))
            self.assertNotEqual(first, self.draw(a, 'B', answer='reference changed'))
            a.close(); b = self.recorder(root, 'two', c, namespace='other checkpoint')
            self.assertNotEqual(first, self.draw(b, 'A')); self.assertEqual(c.calls, 20); b.close()
    def test_corrupt_cache_rejected(self):
        with TemporaryDirectory() as d:
            root = Path(d); r = self.recorder(root, 'one', FakeClient())
            self.draw(r, 'A')
            for f in (root / 'cache').glob('*.json'):
                value = json.loads(f.read_text()); value['response_sha256'] = 'bad'; f.write_text(json.dumps(value))
            with self.assertRaises(ValueError): self.draw(r, 'B')
            r.close()


class MappingTests(unittest.TestCase):
    def fixture(self):
        guide = 'If X, do Y unless Z.'
        rule = QuotedRule('R1', 'If X', 'do Y', ('unless Z',), guide, 0, len(guide), 1)
        tfu = [{'rule_id': 'R1', 'label': 'T', 'status': 'ok'}]
        return guide, rule, tfu
    def test_mapping_and_span_kept_exact(self):
        guide, rule, tfu = self.fixture()
        base, support, mapping = grounded_prompts('{Guideline}\n{Case}', guide, 'patient X', (rule,), tfu)
        self.assertEqual(mapping[0]['condition_quote'], 'If X')
        self.assertEqual(mapping[0]['exception_quotes'], ['unless Z'])
        a, b = base.absolute_spans(base.evidence.spans)[0]
        self.assertEqual(base.text[a:b], guide)
        self.assertIn('R1', base.text); self.assertIn('do Y', base.text)
        self.assertTrue(support.text.startswith(base.text))
    def test_no_evidence_does_not_add_empty_emphasis(self):
        base, support, mapping = grounded_prompts('{Guideline}\n{Case}', 'guideline', 'case', (), [])
        self.assertEqual(base.text, support.text); self.assertFalse(mapping)
        guide, rule, tfu = self.fixture(); tfu[0]['label'] = 'U'
        base, support, _ = grounded_prompts('{Guideline}\n{Case}', guide, 'case', (rule,), tfu)
        self.assertEqual(base.text, support.text)
    def test_wrong_rule_id_cannot_be_attached(self):
        guide, rule, tfu = self.fixture(); tfu[0]['rule_id'] = 'wrong'
        with self.assertRaises(ValueError):
            grounded_prompts('{Guideline}\n{Case}', guide, 'case', (rule,), tfu)


if __name__ == '__main__':
    unittest.main()
