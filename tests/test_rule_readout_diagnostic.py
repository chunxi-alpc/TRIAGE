"""Contracts for the frozen follow-up diagnostic; no pretrained model required."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.run_rule_readout_diagnostic import (
    CANDIDATES, insert_note, intervention, mass_summary, readout, verify_anchor,
)


class TokenNames:
    def decode(self, ids):
        return ":".join(map(str, ids))


def test_key_stress_reverses_once_and_preserves_shared_direction():
    heads = [(1, 0), (2, 1)]
    delta = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
    original = delta.copy()
    positive, no_bias = intervention(heads, delta, "B", key_gain=32)
    negative, no_bias_negative = intervention(heads, delta, "B", key_gain=-32)
    assert not no_bias and not no_bias_negative
    assert len(positive) == len(negative) == 4
    for plus, minus in zip(positive, negative):
        expected = 32 if plus["rule_id"] == "B" else -32
        assert plus["coefficient"] == expected
        assert minus["coefficient"] == -expected
        np.testing.assert_array_equal(plus["delta"], delta[plus["layer"], plus["kv_head"]])
        np.testing.assert_array_equal(minus["delta"], plus["delta"])
    np.testing.assert_array_equal(delta, original)


def test_score_control_has_same_targets_and_excludes_mixed_interventions():
    heads, delta = [(0, 0)], np.ones((1, 1, 4))
    shifts, plus = intervention(heads, delta, "A", score_bias=8)
    reverse_shifts, minus = intervention(heads, delta, "A", score_bias=-8)
    assert shifts == reverse_shifts == []
    assert [(r["rule_id"], r["bias"]) for r in plus] == [("A", 8), ("B", -8)]
    assert [(r["rule_id"], r["bias"]) for r in minus] == [("A", -8), ("B", 8)]
    assert intervention(heads, delta, "A") == ([], [])
    with pytest.raises(ValueError):
        intervention(heads, delta, "A", key_gain=32, score_bias=8)
    with pytest.raises(ValueError):
        intervention(heads, delta, "NONE", key_gain=32)


def test_note_preserves_complete_context_and_only_swaps_route_ids():
    prompt = "Guideline: Rule A. Rule B.\n\nPatient: β = YES.\nQuestion: A or B?\n\nAnswer:"
    original_context = prompt[:-len("\nAnswer:")]
    a, b = insert_note(prompt, "A"), insert_note(prompt, "B")
    for actual in (a, b):
        assert actual.startswith(original_context)
        assert actual.endswith("\nAnswer:")
        assert actual.count("Routing result (provided by the rule parser):") == 1
        assert actual.count("Question: A or B?") == 1
    a_note = a[len(original_context):-len("\nAnswer:")]
    b_note = b[len(original_context):-len("\nAnswer:")]
    assert "Applicable rule: A." in a_note
    assert a_note.count("Applicable rule:") == 1
    assert "not applicable" not in a_note
    assert b_note == a_note.replace("Applicable rule: A.", "Applicable rule: B.")


def test_note_rejects_unexpected_answer_boundary_and_invalid_rule():
    for prompt, rule in [("Question\nAnswer: ", "A"), ("Question\nAnswer:", "C"),
                         ("Question Answer:", "B")]:
        with pytest.raises(ValueError):
            insert_note(prompt, rule)


def test_readout_separates_vocabulary_mass_from_constrained_preference():
    logits = torch.full((500,), -10.)
    logits[CANDIDATES[0]], logits[CANDIDATES[1]], logits[17] = 2., 1., 10.
    result = readout(logits, TokenNames(), "B")
    assert result["prediction"] == "A" and not result["correct"]
    assert result["gold_margin"] == pytest.approx(-1.)
    assert result["candidate_probability_mass"] < 0.001
    assert result["conditional_probability_A"] > 0.7
    assert result["conditional_probability_A"] + result["conditional_probability_B"] == pytest.approx(1.)
    assert result["candidate_probability_mass"] == pytest.approx(
        result["candidate_probability_A"] + result["candidate_probability_B"])
    assert result["top_tokens"][0]["token_id"] == 17
    assert result["top_tokens"][0]["text"] == "17"
    assert len(result["top_tokens"]) == 5


def test_readout_rejects_nonfinite_or_nonvector_logits():
    for invalid in (torch.zeros(1, 500), torch.full((500,), float("nan")),
                    torch.full((500,), float("inf"))):
        with pytest.raises(ValueError):
            readout(invalid, TokenNames(), "A")
    with pytest.raises(ValueError):
        readout(torch.zeros(500), TokenNames(), "NONE")


def _anchor_fixture():
    logits = torch.zeros(500)
    logits[CANDIDATES[0]], logits[CANDIDATES[1]] = 0.3, -0.2
    target = dict(layer=13, kv_head=4, rule_id="A", positions=[2, 5],
                  per_query_token_mass=[[0.1, 0.2], [0.05, 0.1]])
    old = dict(logits_A=float(logits[CANDIDATES[0]]), logits_B=float(logits[CANDIDATES[1]]),
               attention_mass=[deepcopy(target)], prefix_length=8, suffix_length=4)
    actual = dict(logits=logits, attention_mass=[deepcopy(target)], prefix_length=8, suffix_length=4)
    return old, actual


def test_anchor_checks_values_identity_and_token_lengths():
    old, actual = _anchor_fixture()
    exact = verify_anchor(old, actual)
    assert exact["candidate_logit_max_error"] == exact["attention_max_error"] == 0
    changed = deepcopy(actual)
    changed["logits"][CANDIDATES[0]] += 0.01
    with pytest.raises(RuntimeError):
        verify_anchor(old, changed)
    changed = deepcopy(actual)
    changed["attention_mass"][0]["positions"] = [2, 6]
    with pytest.raises(ValueError):
        verify_anchor(old, changed)
    changed = deepcopy(actual)
    changed["attention_mass"][0]["per_query_token_mass"][0][0] += 0.01
    with pytest.raises(RuntimeError):
        verify_anchor(old, changed)
    changed = deepcopy(actual)
    changed["suffix_length"] += 1
    with pytest.raises(ValueError):
        verify_anchor(old, changed)
    for old_side in (False, True):
        changed_old, changed_actual = deepcopy(old), deepcopy(actual)
        if old_side:
            changed_old["logits_A"] = float("nan")
        else:
            changed_actual["logits"][CANDIDATES[0]] = float("nan")
        with pytest.raises(ValueError):
            verify_anchor(changed_old, changed_actual)
        changed_old, changed_actual = deepcopy(old), deepcopy(actual)
        target = changed_old if old_side else changed_actual
        target["attention_mass"][0]["per_query_token_mass"][0][0] = float("nan")
        with pytest.raises(ValueError):
            verify_anchor(changed_old, changed_actual)


def test_mass_roles_use_token_identity_and_preserve_other_mass():
    actual = dict(attention_mass=[], query_response=[])
    positions = {"A": [2, 7, 10], "B": [3, 8, 11]}
    values = {"A": [0.05, 0.10, 0.15], "B": [0.08, 0.04, 0.08]}
    roles = {r: {"id": [positions[r][1]], "action": [positions[r][0], positions[r][2]]}
             for r in ("A", "B")}
    for rule in ("A", "B"):
        actual["attention_mass"].append(dict(layer=6, kv_head=2, rule_id=rule,
            positions=positions[rule], per_token_mass=values[rule], target_attention_mass=sum(values[rule])))
        actual["query_response"].append(dict(layer=6, kv_head=2, rule_id=rule,
            scores=[[-1., 2., 3.], [-2., 1., 3.]], mean=1., rms=2., positive_fraction=2/3))
    summary = mass_summary(actual, [(6, 2)], roles)
    assert summary["mass_A"] == pytest.approx(0.3)
    assert summary["mass_B"] == pytest.approx(0.2)
    head = summary["per_head_mass"][0]
    assert head["mass_A_id"] == pytest.approx(0.1)
    assert head["mass_A_action"] == pytest.approx(0.2)
    assert head["mass_B_id"] == pytest.approx(0.04)
    assert head["mass_B_action"] == pytest.approx(0.16)
    assert head["mass_other"] == pytest.approx(0.5)
    assert head["response_A"]["minimum"] == -2.
    assert head["response_A"]["maximum"] == 3.
