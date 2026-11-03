import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import construct, prepare, validate_advice, validate_scenarios
from evaluate import evaluate


def fixture():
    advice = [
        {"id": 1, "correct": "If condition X holds, choose option P.",
         "wrong": "If condition X holds, choose option Q.", "disease": "fictional"},
        {"id": 2, "correct": "Route to zone A.", "wrong": "Route to zone B.", "disease": "fictional"},
    ]
    scenarios = [{"id": 0, "text": "Toy statement is supported.", "label": "correct", "disease": "fictional"},
                 {"id": 2145, "text": "Toy statement is contradicted.", "label": "wrong", "disease": "fictional"}]
    return advice, scenarios


def test_pairing_and_no_leakage():
    advice, scenarios = fixture()
    inputs, labels = construct(advice, scenarios)
    assert len(inputs) == 14 and len(labels) == 14
    assert len({r["id"] for r in inputs}) == 14
    assert not any("correct_choice" in r for r in inputs)
    assert not any("label" in r for r in inputs)
    splits = {}
    for i in inputs:
        assert splits.setdefault(i["group"], i["split"]) == i["split"]
    gold = {r["id"]: r["correct_choice"] for r in labels}
    for i in inputs:
        if i["track"] == "advice_choice":
            assert gold[i["id"]] == ("A" if i["variant"] == "AB" else "B")
            if i["source_kind"] == "none":
                assert "Retrieved excerpt" not in i["prompt"]
            else:
                assert i["options"][i["retrieved_option"]] in i["prompt"]
        else:
            assert i["source_kind"] == "none"


def test_source_mismatch_is_not_gold():
    advice, scenarios = fixture()
    inputs, labels = construct(advice, scenarios)
    for i in inputs:
        if i["track"] == "advice_choice" and i["source_kind"] != "none":
            label = next(x["correct_choice"] for x in labels if x["id"] == i["id"])
            assert any(i["retrieved_option"] != label for i in inputs if i["track"] == "advice_choice" and i["source_kind"] != "none")
            assert label in ("A", "B")


def test_duplicate_and_label_validation():
    advice, scenarios = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_advice(advice + [advice[0]], strict=False)
    with pytest.raises(ValueError, match="Unexpected label"):
        validate_scenarios([{**scenarios[0], "label": "maybe"}], strict=False)
    with pytest.raises(ValueError, match="Expected 194"):
        validate_advice(advice, strict=True)


def test_evaluation_no_gold_in_predictions_and_invalid():
    a, s = fixture()
    inputs, labels = construct(a, s)
    prediction = [{"id": g["id"], "method": "fixture_mock_not_model", "prediction": g["correct_choice"]}
                  for g in labels]
    prediction[0]["prediction"] = None
    report = evaluate(labels, inputs, prediction)
    m = report["methods"]["fixture_mock_not_model"]
    assert m["predictions"] == len(labels)
    assert any(row["invalid"] == 1 for row in m["metrics"].values())
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(labels, inputs, prediction + [prediction[0]])
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(labels, inputs, prediction[:-1])


def test_immutable_prepare(tmp_path):
    advice, scenarios = fixture()
    af, sf = tmp_path / "advice.jsonl", tmp_path / "scenarios.jsonl"
    for path, rows in ((af, advice), (sf, scenarios)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    manifest = prepare(af, sf, out, allow_partial=True)
    assert manifest["case_counts"] == {"advice_choice": 12, "scenario_judgment": 2}
    assert manifest["status"] == "DATA_PREPARED_NOT_MODEL_EVALUATED"
    with pytest.raises(FileExistsError):
        prepare(af, sf, out, allow_partial=True)


def test_conditional_flip_requires_verification_and_remains_action_blind():
    from conditions import generate
    example = {"family_id": "fictional-threshold-01", "question": "Does fictional rule R apply?",
               "rule_text": "If level is at least 30 points, apply R.",
               "patient_template": "Fictional level = {value} points.",
               "condition": {"field": "fictional_level", "operator": ">=", "threshold": 30, "unit": "points"},
               "values": {"T": 31, "F": 29},
               "provenance": {"verified": True, "reviewer": "test-fixture-review", "source_url": "https://example.org/fictional"}}
    inp, gold = generate([example])
    assert [x["state"] for x in gold] == ["T", "F", "U"]
    assert all(x["clinical_action_gold"] is None for x in gold)
    assert inp[0]["prompt"].replace("31", "<X>") == inp[1]["prompt"].replace("29", "<X>")
    assert all(x["prompt"][slice(*x["evidence_span"])] in x["prompt"] for x in inp)
    with pytest.raises(ValueError, match="reviewed"):
        generate([{**example, "provenance": {"verified": False}}])
    with pytest.raises(ValueError, match="flip applicability"):
        generate([{**example, "values": {"T": 31, "F": 40}}])
