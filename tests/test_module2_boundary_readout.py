"""Boundary correction and exact saved-forward comparison, without a model run."""
from copy import deepcopy

import pytest

from scripts.replay_module2_boundary_readout import (compare_saved_forward,
    record_key, unique_records, verify_boundary)


class BoundaryTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"Answer:": [10, 11], "Answer: A": [10, 11, 362],
                "Answer: B": [10, 11, 425]}[text]


def test_actual_boundary_requires_unchanged_prefix_and_exactly_one_token():
    audit = verify_boundary(BoundaryTokenizer(), "Answer:", [10, 11])
    assert [r["token_id"] for r in audit] == [362, 425]
    assert all(r["prefix_tokens_unchanged"] and r["appended_tokens"] == 1 for r in audit)
    with pytest.raises(ValueError):
        verify_boundary(BoundaryTokenizer(), "Answer:", [11, 10])


@pytest.mark.parametrize("wrong", [[10, 99, 362], [10, 11, 362, 1], [10, 11, 32]])
def test_boundary_rejects_retokenization_multi_token_or_wrong_label(wrong):
    class WrongTokenizer(BoundaryTokenizer):
        def encode(self, text, add_special_tokens=False):
            return wrong if text == "Answer: A" else super().encode(text, add_special_tokens)
    with pytest.raises(ValueError):
        verify_boundary(WrongTokenizer(), "Answer:", [10, 11])


def original_record():
    return {"case_id": "case", "variant": "positive", "method": "baseline", "mode": "none", "gain": 0.,
            "logits_A": 2., "logits_B": 1.,
            "attention_mass": [{"layer": 17, "kv_head": 5, "rule_id": "A", "positions": [1, 2],
                                "per_query_token_mass": [[.1, .2], [.3, .4]]}]}


def test_saved_forward_comparison_is_exact_and_rejects_changes():
    original = original_record()
    actual = deepcopy(original)
    result = compare_saved_forward(original, actual, [2., 1.])
    assert result["bitwise_numeric_identity"]
    actual["attention_mass"][0]["per_query_token_mass"][0][0] += 1e-3
    with pytest.raises(RuntimeError):
        compare_saved_forward(original, actual, [2., 1.])
    with pytest.raises(RuntimeError):
        compare_saved_forward(original, original, [2.1, 1.])
    with pytest.raises(ValueError):
        compare_saved_forward(original, original, [float("nan"), 1.])


def test_saved_forward_rejects_missing_or_misidentified_attention():
    original = original_record()
    actual = deepcopy(original)
    actual["attention_mass"] = []
    with pytest.raises(ValueError):
        compare_saved_forward(original, actual, [2., 1.])
    actual = deepcopy(original)
    actual["attention_mass"][0]["rule_id"] = "B"
    with pytest.raises(ValueError):
        compare_saved_forward(original, actual, [2., 1.])


def test_replay_coverage_rejects_duplicates_missing_and_extra_cells():
    row = original_record()
    key = record_key(row)
    assert unique_records([row], {key}) == {key: row}
    with pytest.raises(ValueError):
        unique_records([row, row], {key})
    with pytest.raises(ValueError):
        unique_records([], {key})
    with pytest.raises(ValueError):
        unique_records([row], set())
