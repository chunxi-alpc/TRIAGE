"""CPU regressions for the two faults discovered in the saved dev outputs."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from care_m2_v2.candidates import (SCORING_CONTRACT, candidate_contract,
    score_candidates, score_key, recover_legacy_top)
from care_m2_v2.data import digest
from care_m2_v2.engine import Engine, fit_axis
from care_m2_v2.n5_pairs import matched_pairs
from care_m2_v2.study import train, evaluate

class Tokenizer:
    # Bare and space-prefixed labels have different IDs; ID order differs from label order.
    mapping = {'A': 2, 'B': 3, ' A': 7, ' B': 5}
    def encode(self, text, add_special_tokens=False):
        if text in self.mapping: return [self.mapping[text]]
        if text.endswith('Answer:'): return [0, 1]
        for suffix in (' A', ' B'):
            if text.endswith('Answer:' + suffix): return [0, 1, self.mapping[suffix]]
        raise ValueError(text)
    def decode(self, ids):
        return next((k for k, v in self.mapping.items() if [v] == ids), '?')

def logits():
    x = torch.zeros(10); x[2] = 9; x[7] = 1; x[5] = 4
    return x

def test_scores_continuation_not_bare_letter():
    tok = Tokenizer(); cs = candidate_contract(tok, 'Question\nAnswer:', {'A': 'one', 'B': 'two'})
    result = score_candidates(logits(), cs, tok)
    assert result['prediction'] == 'B'  # old scorer chooses A
    assert result['candidate_logits'] == {'A': 1., 'B': 4.}
    assert result['candidate_margin'] == 3
    assert result['candidate_tokens']['A']['continuation'] == ' A'
    assert not result['unconstrained_is_candidate']  # global winner is bare A

@pytest.mark.parametrize('bad', ['boundary_merge', 'multiple_tokens', 'wrong_decode'])
def test_rejects_misaligned_candidate_boundary(bad):
    class Broken(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            if text.endswith('Answer: A'):
                if bad == 'boundary_merge': return [0, 9, 7]
                if bad == 'multiple_tokens': return [0, 1, 8, 7]
            return ids
        def decode(self, ids): return 'A' if bad == 'wrong_decode' else super().decode(ids)
    with pytest.raises(ValueError): candidate_contract(Broken(), 'Answer:', ['A'])

def test_engine_uses_contract_and_does_not_reuse_bare_score_cache(tmp_path):
    engine = object.__new__(Engine)
    engine.tokenizer = Tokenizer(); engine.modelhash = 'model'; engine.calls = 0
    engine.tokens = lambda prompt: ({'input_ids': torch.tensor([[0, 1]])}, [])
    engine.edit = lambda *args: nullcontext({'residual': [], 'attention': []})
    engine.model = lambda **kwargs: SimpleNamespace(logits=logits()[None, None, :])
    engine.journal = tmp_path/'journal.jsonl'
    options = {'A': 'one', 'B': 'two'}; spec = {'edits': [], 'biases': []}
    oldkey = digest(['model', 'Answer:', options, spec])
    engine.cache = {oldkey: {'prediction': 'A', 'candidate_logits': {'A': 9, 'B': 0}}}
    first = engine.score('Answer:', options)
    assert first['prediction'] == 'B' and engine.calls == 1
    assert first['scoring_contract'] == SCORING_CONTRACT
    assert engine.score('Answer:', options)['cache_hit'] and engine.calls == 1
    assert engine.cache[oldkey]['prediction'] == 'A'
    journal = json.loads(engine.journal.read_text())
    assert journal['key'] != oldkey and journal['checksum'] == digest(journal['payload'])

def test_legacy_global_top_recovery_is_exact_even_with_ties():
    tok = Tokenizer(); cs = candidate_contract(tok, 'Answer:', ['A', 'B'])
    x = torch.zeros(10); x[7] = x[5] = 4
    legacy = {'top_unconstrained_token': tok.decode([int(x.argmax())]),
              'candidate_logits': {'A': 0, 'B': 0}, 'output_sha256': 'old', 'prediction': 'A'}
    recovered = recover_legacy_top(legacy, cs, {' A': 7, ' B': 5})
    assert recovered['prediction'] == score_candidates(x, cs, tok)['prediction'] == 'B'
    assert recovered['candidate_logits'] is None and recovered['forward_count'] == 0
    assert recover_legacy_top({**legacy, 'top_unconstrained_token': 'B'}, cs, {'B': 3}) is None
    assert recover_legacy_top(legacy, cs, {}) is None  # ambiguous decode is not recoverable

@pytest.mark.parametrize('positive,negative', [([], []), ([torch.ones(3)], [torch.ones(3), torch.zeros(3)])])
def test_axis_rejects_silent_truncation(positive, negative):
    with pytest.raises(ValueError, match='matched pairs'): fit_axis(positive, negative)

def rows():
    result = []
    for i in range(3):
        for state in ['T', 'F']:
            result.append({'task': 'N5', 'split': 'train', 'case_id': f'{i}_{state}', 'family_id': 'family',
                'pair_id': f'pair{i}', 'source_commit': 'frozen', 'a1_method': 'A1', 'prompt': f'{i}_{state}',
                'signals': [{'rule_id': 'R', 'rule_sha256': 'rulehash', 'action': 'act', 'span': [0, 1],
                             'state': state, 'signal': 1 if state == 'T' else -1}], 'spans': [[0, 1]]})
    return result

class Capture:
    def positions(self, row): return [[0] for span in row['spans']]
    def capture(self, prompt, positions=None):
        i, state = prompt.split('_'); i = int(i)
        return torch.tensor([i, i*i + (state == 'T'), (i+2)*(state == 'T'), 1.])

def test_n5_fit_matches_same_pair_rule_and_ignores_input_order_and_gold(tmp_path):
    a = rows(); b = copy.deepcopy(list(reversed(a)))
    for r in b: r['gold'] = 'bad'; r['reference_states'] = 'bad'
    left = tmp_path/'a'; right = tmp_path/'b'; left.mkdir(); right.mkdir()
    fit1 = train(Capture(), a, left)['A1']; fit2 = train(Capture(), b, right)['A1']
    assert matched_pairs(a) == matched_pairs(b)
    assert torch.allclose(fit1['differences'].mean(0), fit1['mean'], atol=1e-6)
    for key in ['positive', 'negative', 'differences', 'mean']:
        assert torch.allclose(fit1[key], fit2[key])
    assert torch.allclose(fit1['basis']@fit1['basis'].T, fit2['basis']@fit2['basis'].T)
    audit = json.loads((left/'n5_pair_audit.json').read_text())
    assert len(audit['pairs']) == 3 and audit['mother_signals'] == 6

@pytest.mark.parametrize('field,value', [('rule_sha256', 'other'), ('action', 'other'),
                                        ('source_commit', 'other'), ('a1_method', 'other')])
def test_n5_does_not_cross_match_rule_or_version(field, value):
    r = rows()[:2]
    if field in r[1]: r[1][field] = value
    else: r[1]['signals'][0][field] = value
    pairs, audit = matched_pairs(r)
    assert not pairs and audit['status_counts']['no_unique_TF_counterpart'] == 2

def test_n5_preserves_unpaired_unknown_null_and_wrong_predictions():
    r = rows()
    for state, signal in [('U', 0), (None, None), ('T', 1)]:
        extra = copy.deepcopy(r[0]); extra['case_id'] = str(state); extra['pair_id'] = extra['family_id']
        extra['signals'][0].update(state=state, signal=signal); r.append(extra)
    # Predicted states disagree with fake reference: pairs still use predictions.
    for x in r: x['reference_states'] = [{'state': 'U'}]
    pairs, audit = matched_pairs(r)
    assert len(pairs) == 3 and audit['mother_cases'] == 9 and audit['mother_signals'] == 9
    assert audit['status_counts'] == {'paired': 6, 'no_TF_signal': 2, 'no_original_pair': 1}
    assert not audit['evaluation_population_changed']
    r[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='train-only'): matched_pairs(r)

def test_old_selection_cannot_enter_corrected_evaluator(tmp_path):
    with pytest.raises(ValueError, match='incompatible candidate scoring'):
        evaluate(None, [], {}, {'scoring_contract': 'old_bare'}, tmp_path, [])

def test_zero_iia_never_captures_a_donor():
    from care_m2_v2.data import n2
    from care_m2_v2.study import runone
    class ZeroEngine:
        tokenizer = SimpleNamespace(encode=lambda s: [0, 1])
        def capture(self, *args): raise AssertionError('zero intervention must not request donor activation')
        def score(self, prompt, options, edits, biases):
            return {'prediction': 'A', 'forward_count': 0, 'cache_hit': True}
    data = n2(); receiver = next(r for r in data if r['split'] == 'dev')
    axis = {'positive': torch.ones(3), 'negative': torch.zeros(3), 'mean': torch.ones(3), 'basis': torch.eye(3)[:, :1]}
    result = runone(ZeroEngine(), receiver, {'source': 'C'}, 'iia_train', 0, {'source': axis}, [r for r in data if r['split'] != 'test'])
    assert result['forward_count'] == 0 and result['donor']['split'] == 'train'
