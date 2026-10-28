"""Pure planning/coverage checks for the frozen guide-prefix causal pilot."""
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from scripts.run_module2_mechanism_causal import CONDITIONS, compare, plan
from scripts.run_module2_mechanism import context_key


def fixture():
    heads = [(1, 0), (1, 2), (3, 1), (4, 0)]
    base = np.arange(5 * 3 * 8, dtype=float).reshape(5, 3, 8) / 100
    directions = {"projected": base, "raw_mean": base + 1,
                  "random": base - 1, "reversed": -base}
    beta = np.arange(1, 16, dtype=float).reshape(5, 3) / 50
    return heads, directions, beta


def test_frozen_grid_includes_factorial_modes_and_all_controls():
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 9
    assert ("baseline", "none", 0.) in CONDITIONS
    assert {(method, mode, gain) for method, mode, gain in CONDITIONS if method == "projected"} == {
        ("projected", "both", 1.), ("projected", "both", 4.),
        ("projected", "enhance", 4.), ("projected", "suppress", 4.)}
    assert {method for method, _, _ in CONDITIONS} == {
        "baseline", "projected", "raw_mean", "random", "reversed", "direct_bias"}


def test_all_four_heads_receive_both_oracle_signed_rules():
    heads, directions, beta = fixture()
    for gold in ("A", "B"):
        shifts, biases = plan("projected", "both", 4., heads, directions, beta, gold)
        assert not biases
        assert len(shifts) == 8
        assert {(s["layer"], s["kv_head"], s["rule_id"]) for s in shifts} == {
            (layer, head, rule) for layer, head in heads for rule in ("A", "B")}
        for shift in shifts:
            assert shift["coefficient"] == (4. if shift["rule_id"] == gold else -4.)
            np.testing.assert_array_equal(shift["delta"], directions["projected"][shift["layer"], shift["kv_head"]])


def test_enhance_and_suppress_partition_the_joint_plan():
    heads, directions, beta = fixture()
    for gold in ("A", "B"):
        enhance, _ = plan("projected", "enhance", 4., heads, directions, beta, gold)
        suppress, _ = plan("projected", "suppress", 4., heads, directions, beta, gold)
        both, _ = plan("projected", "both", 4., heads, directions, beta, gold)
        assert len(enhance) == len(suppress) == 4
        assert all(s["rule_id"] == gold and s["coefficient"] == 4. for s in enhance)
        assert all(s["rule_id"] != gold and s["coefficient"] == -4. for s in suppress)
        identity = lambda s: (s["layer"], s["kv_head"], s["rule_id"], s["coefficient"])
        assert set(map(identity, enhance + suppress)) == set(map(identity, both))


def test_direction_controls_change_vector_without_changing_target_or_gain():
    heads, directions, beta = fixture()
    for method in ("projected", "raw_mean", "random", "reversed"):
        shifts, biases = plan(method, "both", 4., heads, directions, beta, "B")
        assert not biases
        for shift in shifts:
            assert shift["coefficient"] == (4. if shift["rule_id"] == "B" else -4.)
            np.testing.assert_array_equal(shift["delta"], directions[method][shift["layer"], shift["kv_head"]])
    forward, _ = plan("projected", "both", 4., heads, directions, beta, "B")
    reverse, _ = plan("reversed", "both", 4., heads, directions, beta, "B")
    for a, b in zip(forward, reverse):
        np.testing.assert_array_equal(a["coefficient"] * a["delta"], -b["coefficient"] * b["delta"])


def test_direct_bias_reuses_frozen_head_scale_for_both_patient_states():
    heads, directions, beta = fixture()
    for gold in ("A", "B"):
        shifts, biases = plan("direct_bias", "both", 4., heads, directions, beta, gold)
        assert not shifts
        assert len(biases) == 8
        for bias in biases:
            expected = 4. * beta[bias["layer"], bias["kv_head"]]
            assert bias["bias"] == (expected if bias["rule_id"] == gold else -expected)
            assert "delta" not in bias


def test_baseline_and_empty_expert_never_intervene():
    heads, directions, beta = fixture()
    assert plan("baseline", "none", 0., heads, directions, beta, "A") == ([], [])
    for method, mode, gain in CONDITIONS:
        assert plan(method, mode, gain, [], directions, beta, "B") == ([], [])


def test_canonical_A_covers_each_physical_patient_once_and_B_is_reversed():
    source = Path(__file__).resolve().parents[1] / "experiments/module2_mechanism_20261028/inputs/triplets.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    confirmation = [r for r in rows if r["split"] == "confirm"]
    canonical = [r for r in confirmation if r["target_rule_id"] == "A"]
    assert len(canonical) == len({context_key(r) for r in canonical}) == 12
    by_identity = {(context_key(r), r["target_rule_id"]): r for r in confirmation}
    observations = set()
    for a in canonical:
        b = by_identity[context_key(a), "B"]
        assert a["oracle"]["positive"]["selected_rule_id"] == "A"
        assert a["oracle"]["negative"]["selected_rule_id"] == "B"
        for av, bv in (("positive", "negative"), ("negative", "positive")):
            assert a["prompts"][av] == b["prompts"][bv]
            assert a["patient_facts"][av] == b["patient_facts"][bv]
            assert a["oracle"][av]["selected_rule_id"] == b["oracle"][bv]["selected_rule_id"]
            observations.add((context_key(a), av))
    assert len(observations) == 24


def comparison_fixture():
    import torch
    identity = {"layer": 17, "kv_head": 5, "rule_id": "A", "positions": [2, 3]}
    return {"logits": torch.tensor([.1, .2, .3]),
            "attention_mass": [dict(identity, per_query_token_mass=[[.1, .2], [.3, .4]])],
            "query_response": [dict(identity, scores=[[.2, .4], [.3, .5]])]}


def test_compare_exact_zero_and_tolerances_are_distinct():
    reference = comparison_fixture()
    actual = deepcopy(reference)
    result = compare(reference, actual, logit_atol=0., attention_atol=0.)
    assert result["logit_max_abs_error"] == result["attention_max_abs_error"] == result["query_max_abs_error"] == 0
    actual["logits"][0] += 1e-6
    actual["attention_mass"][0]["per_query_token_mass"][0][0] += 1e-7
    actual["query_response"][0]["scores"][0][0] += 1e-7
    with pytest.raises(RuntimeError):
        compare(reference, actual, logit_atol=0., attention_atol=0.)
    compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)
    actual["query_response"][0]["scores"][0][0] += 1e-4
    with pytest.raises(RuntimeError):
        compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)


def test_compare_rejects_nonfinite_reference_or_actual_values():
    for side in ("reference", "actual"):
        for location in ("logits", "attention_mass", "query_response"):
            reference = comparison_fixture()
            actual = deepcopy(reference)
            target = reference if side == "reference" else actual
            if location == "logits":
                target[location][0] = float("nan")
            else:
                field = "per_query_token_mass" if location == "attention_mass" else "scores"
                target[location][0][field][0][0] = float("inf")
            with pytest.raises(ValueError):
                compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)


def test_compare_rejects_missing_diagnostics_and_different_support():
    for location in ("attention_mass", "query_response"):
        reference = comparison_fixture()
        actual = deepcopy(reference)
        actual[location] = []
        with pytest.raises(ValueError):
            compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)
        for field, value in (("layer", 16), ("kv_head", 6), ("rule_id", "B"), ("positions", [2, 4])):
            actual = deepcopy(reference)
            actual[location][0][field] = value
            with pytest.raises(ValueError):
                compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)


def test_compare_rejects_broadcastable_shape_mismatches():
    reference = comparison_fixture()
    actual = deepcopy(reference)
    actual["logits"] = actual["logits"][:1]
    with pytest.raises(ValueError):
        compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)
    for location, field in (("attention_mass", "per_query_token_mass"), ("query_response", "scores")):
        actual = deepcopy(reference)
        actual[location][0][field] = [actual[location][0][field][0]]
        with pytest.raises(ValueError):
            compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)
