import json
from pathlib import Path

import numpy as np

from src.module2.mechanism_extraction import online_prompt, token_pair_second_moment


def test_online_reorder_keeps_rule_and_patient_content():
    path = Path(__file__).resolve().parents[1] / 'experiments/module2_mechanism_20261028/inputs/triplets.jsonl'
    row = json.loads(path.read_text().splitlines()[0])
    for variant in ('positive', 'negative', 'neutral'):
        prompt, spans, end = online_prompt(row, variant)
        old, old_spans = row['prompts'][variant], row['spans'][variant]
        assert prompt.startswith('Guideline')
        assert prompt.index('Patient findings') > end
        assert prompt.index('Question:') > prompt.index('Patient findings')
        for role in ('patient', 'question', 'condition', 'id', 'action', 'guide'):
            assert [prompt[slice(*s)] for s in spans[role]] == [old[slice(*s)] for s in old_spans[role]]


def test_token_second_moment_keeps_opposing_token_variation():
    # Span-average-first would incorrectly erase this real token variation.
    pos = np.array([[[[1., 2.]], [[-1., -2.]]]])
    neg = np.zeros_like(pos)
    actual = token_pair_second_moment(pos, neg)
    assert np.allclose(actual[0, 0], [[1., 2.], [2., 4.]])
    assert np.allclose(pos.mean(axis=1), 0)
