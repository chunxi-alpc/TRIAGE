"""Strict TrialMatchAI final-answer parsing; no real model or GPU needed."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import TrialMatchAIAdapter, map_trialmatch_label

CRITERION = "Distant metastasis is present."


def answer(label="Met", kind="inclusion", criterion=CRITERION, **changes):
    data = {
        "Inclusion_Criteria_Evaluation": [],
        "Exclusion_Criteria_Evaluation": [],
        "Recap": "Single condition assessed.",
        "Final Decision": "Eligible",
    }
    key = "Exclusion_Criteria_Evaluation" if kind == "exclusion" else "Inclusion_Criteria_Evaluation"
    data[key] = [{"Criterion": criterion, "Classification": label,
                  "Justification": "The provided patient sentence supports the classification."}]
    data.update(changes)
    return json.dumps(data)


def test_exact_labels_only():
    assert map_trialmatch_label("Met") == "T"
    assert map_trialmatch_label("Not Met") == "F"
    assert map_trialmatch_label("Unclear") == "U"
    assert map_trialmatch_label("Initially Met, finally Not Met") == "U"
    assert map_trialmatch_label("Violated", "exclusion") == "T"
    assert map_trialmatch_label("Not Violated", "exclusion") == "U"


def test_final_json_is_read_after_reasoning_not_first_keyword():
    for closer in ("</think>", "<think>"):
        raw = "<think>Met may be tempting; Unclear is another possibility." + closer + answer("Not Met")
        result = TrialMatchAIAdapter._parse_response(raw, CRITERION)
        assert result["state"] == "F"
        assert result["raw_label"] == "Not Met"
        assert result["parse_error"] is None


def test_reasoning_and_prose_without_final_json_are_unknown():
    for raw in ("Met. Evidence is present.", "<think>Met"):
        result = TrialMatchAIAdapter._parse_response(raw, CRITERION)
        assert result["state"] == "U"
        assert result["parse_error"] == "invalid_json_response"


def test_fenced_json_and_exclusion_unknown_are_supported():
    result = TrialMatchAIAdapter._parse_response("```json\n" + answer("Not Violated", "exclusion") + "\n```",
                                                CRITERION, "exclusion")
    assert result["state"] == "U"
    assert result["raw_label"] == "Not Violated"
    assert result["parse_error"] is None
    result = TrialMatchAIAdapter._parse_response(answer("Unclear", "exclusion"), CRITERION, "exclusion")
    assert result["parse_error"] is None


def test_truncated_generation_is_unknown_even_if_json_looks_complete():
    result = TrialMatchAIAdapter._parse_response(answer("Met"), CRITERION, truncated=True)
    assert result["state"] == "U"
    assert result["parse_error"] == "generation_truncated"
    assert result["raw_label"] is None


def test_other_criteria_or_wrong_criterion_cannot_supply_label():
    wrong = TrialMatchAIAdapter._parse_response(answer("Met", criterion="A different condition."), CRITERION)
    assert wrong["parse_error"] == "criterion_text_mismatch"
    data = json.loads(answer())
    data["Inclusion_Criteria_Evaluation"].append(dict(data["Inclusion_Criteria_Evaluation"][0]))
    multiple = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert multiple["parse_error"] == "duplicate_target_criterion"
    data = json.loads(answer())
    data["Exclusion_Criteria_Evaluation"] = [{"Criterion": "Not used", "Classification": "Violated", "Justification": "No"}]
    wrong_side = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert wrong_side["parse_error"] is None
    assert wrong_side["state"] == "T"
    assert wrong_side["non_target_entry_count"] == 1


def test_schema_missing_or_invalid_labels_are_unknown():
    samples = [("{}", "missing_schema_fields"),
               (answer("Met because this looks likely"), "invalid_classification")]
    for raw, reason in samples:
        result = TrialMatchAIAdapter._parse_response(raw, CRITERION)
        assert result["state"] == "U"
        assert result["parse_error"] == reason
    duplicate = answer().replace('"Classification": "Met"', '"Classification": "Unclear", "Classification": "Met"')
    assert TrialMatchAIAdapter._parse_response(duplicate, CRITERION)["parse_error"] == "invalid_json_response"


def test_official_prompt_uses_chat_messages_and_explicit_single_criterion():
    messages = TrialMatchAIAdapter._messages("Patient has distant metastasis.", CRITERION, "inclusion")
    assert [row["role"] for row in messages] == ["system", "user"]
    assert "Eligibility Criteria:\nInclusion Criteria:\n" + CRITERION in messages[1]["content"]
    assert '"Classification"' in messages[1]["content"]
    assert "Met | Not Met | Unclear | Irrelevant" in messages[1]["content"]
    assert TrialMatchAIAdapter.MAX_NEW_TOKENS == 2048


def _fake_batched_adapter():
    import types
    import torch
    second = "Patient age is at least 18 years."

    class Tokenizer:
        eos_token_id, pad_token_id, eos_token, pad_token = 2, 0, "EOS", "PAD"
        padding_side = "right"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "short" if CRITERION in messages[1]["content"] else "long"
        def __call__(self, text, **kwargs):
            def ids(prompt):
                return [9, 9] if prompt == "short" else [9, 9, 9, 9]
            if isinstance(text, str):
                return {"input_ids": ids(text)}
            assert self.padding_side == "left"
            assert kwargs["padding"] is True and kwargs["truncation"] is False
            rows = [ids(prompt) for prompt in text]
            width = max(map(len, rows))
            return {"input_ids": torch.tensor([[0] * (width-len(row)) + row for row in rows]),
                    "attention_mask": torch.tensor([[0] * (width-len(row)) + [1] * len(row) for row in rows])}
        def decode(self, ids, **kwargs):
            return answer("Met") if ids[0] == 100 else answer("Not Met", criterion=second)

    class Model:
        device = "cpu"
        config = types.SimpleNamespace(max_position_embeddings=16384)
        generation_config = types.SimpleNamespace(eos_token_id=2)
        calls = []
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 2048
            assert kwargs["do_sample"] is False
            rows = []
            for prompt, mask in zip(kwargs["input_ids"], kwargs["attention_mask"]):
                completion = [100, 2, 0, 0] if int(mask.sum()) == 2 else [101, 102, 103, 2]
                rows.append(prompt.tolist() + completion)
            self.calls.append(kwargs)
            return torch.tensor(rows)

    return TrialMatchAIAdapter(Model(), Tokenizer(), "cpu"), second


def test_batched_generation_uses_left_padding_and_counts_completion_before_padding():
    adapter, second = _fake_batched_adapter()
    requests = [{"patient": "A patient.", "condition": {"text": CRITERION}, "criterion_kind": "inclusion"},
                {"patient": "Another patient.", "condition": {"text": second}, "criterion_kind": "inclusion"}]
    results = adapter.judge_many(requests, batch_size=8)
    assert [row["state"] for row in results] == ["T", "F"]
    assert [row["input_tokens"] for row in results] == [2, 4]
    assert [row["output_tokens"] for row in results] == [2, 4]
    assert all(row["generation_batch_size"] == 2 for row in results)
    assert all(row["timing_method"] == "shared_batch_wall_time_divided_by_batch_size" for row in results)
    assert results[0]["inference_seconds"] == results[1]["inference_seconds"]
    assert results[0]["batch_elapsed_seconds"] == 2 * results[0]["inference_seconds"]
    assert adapter.model.calls[-1]["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    single = adapter.judge("A patient.", {"text": CRITERION})
    assert single["raw_output"] == results[0]["raw_output"]
    assert single["generation_batch_size"] == 1


def test_batch_validation_and_context_limit_are_explicit():
    import pytest
    adapter, _ = _fake_batched_adapter()
    with pytest.raises(ValueError, match="batch_size"):
        adapter.judge_many([], batch_size=0)
    assert adapter.judge_many([]) == []
    adapter.model.config.max_position_embeddings = 2049
    result = adapter.judge("A patient.", {"text": CRITERION})
    assert result["state"] == "U"
    assert result["parse_error"] == "input_context_limit"
    assert result["output_tokens"] == 0
    assert result["generation_batch_size"] == 0


def test_precomputed_cache_preserves_duplicate_request_outputs_in_order():
    import importlib.util
    import pytest
    script = ROOT / "experiments/care_module1_v5/scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("v5_run_experiment_for_parser_tests", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = {"patient": "A patient.", "condition": {"text": CRITERION}, "criterion_kind": "inclusion"}
    cache = module.PrecomputedTrialMatch([request, request], [{"state": "T"}, {"state": "F"}])
    assert cache.judge(request["patient"], request["condition"])["state"] == "T"
    assert cache.judge(request["patient"], request["condition"])["state"] == "F"
    assert cache.remaining == 0
    with pytest.raises(RuntimeError, match="precomputed"):
        cache.judge(request["patient"], request["condition"])


def test_only_explicit_irrelevant_empty_side_placeholders_are_ignored():
    data = json.loads(answer())
    placeholder = {"Criterion": "No exclusion criteria provided.", "Classification": "Irrelevant", "Justification": "None supplied."}
    data["Exclusion_Criteria_Evaluation"] = [placeholder]
    parsed = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert parsed["state"] == "T" and parsed["parse_error"] is None
    assert parsed["ignored_non_target_placeholders"] == [placeholder]
    placeholder["Classification"] = "Not Violated"
    ignored_other = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert ignored_other["state"] == "T" and ignored_other["raw_label"] == "Met"
    assert ignored_other["non_target_entry_count"] == 1
    assert "non_target_entries_ignored" in ignored_other["parse_warnings"]


def test_fences_prefixes_and_suffixes_never_supply_labels():
    for prefix, suffix in (("Reasoning: Met. ", " Final explanation: Met."),
                           ("The final condition evaluation follows:\n```json\n", "\n```\nThe patient is eligible."),
                           ("```JSON\n", "\n```")):
        result = TrialMatchAIAdapter._parse_response(prefix + answer("Not Met") + suffix, CRITERION)
        assert result["raw_label"] == "Not Met" and result["state"] == "F"
        assert result["parse_status"] == "accepted"


def test_irrelevant_non_target_placeholders_are_explicitly_ignored():
    for kind, label in (("inclusion", "Met"), ("exclusion", "Violated")):
        other = "inclusion" if kind == "exclusion" else "exclusion"
        key = "Inclusion_Criteria_Evaluation" if other == "inclusion" else "Exclusion_Criteria_Evaluation"
        for phrase in (f"No specific {other} criteria provided", f"No {other} criteria were mentioned.",
                       f"No specific {other} criteria listed", f"No {other} criteria"):
            raw = answer(label, kind, **{key: [{"Criterion": phrase, "Classification": "Irrelevant"}]})
            result = TrialMatchAIAdapter._parse_response(raw, CRITERION, kind)
            assert result["parse_status"] == "accepted"
            assert len(result["ignored_non_target_placeholders"]) == 1


def test_placeholder_cannot_hide_a_real_non_target_condition_or_label():
    for criterion, label in (("No specific exclusion criteria provided", "Violated"),
                             ("No exclusion criteria involving active infection", "Irrelevant")):
        result = TrialMatchAIAdapter._parse_response(answer(**{"Exclusion_Criteria_Evaluation":
            [{"Criterion": criterion, "Classification": label}]}), CRITERION)
        assert result["parse_error"] is None
        assert result["state"] == "T"
        assert result["non_target_entry_count"] == 1


def test_unique_exact_target_can_be_located_among_additional_entries():
    data = json.loads(answer("Not Met"))
    data["Inclusion_Criteria_Evaluation"].insert(0, {
        "Criterion": "An unrelated condition.", "Classification": "Met", "Justification": "Unrelated."})
    parsed = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert parsed["raw_label"] == "Not Met"
    assert parsed["matched_criterion_index"] == 1
    assert parsed["ignored_target_entry_count"] == 1


def test_duplicate_or_conflicting_final_objects_are_ambiguous():
    for second in (answer("Met"), answer("Not Met")):
        result = TrialMatchAIAdapter._parse_response(answer("Met") + "\n" + second, CRITERION)
        assert result["parse_error"] == "multiple_schema_objects"
        assert result["parse_status"] == "ambiguous_target"
    data = json.loads(answer())
    data["Inclusion_Criteria_Evaluation"].append({
        "Criterion": CRITERION, "Classification": "Not Met", "Justification": "Conflict."})
    result = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert result["parse_error"] == "duplicate_target_criterion"


def test_exact_target_and_label_are_not_fuzzy_matched():
    for changed in (CRITERION.lower(), CRITERION.rstrip("."), "Distant metastasis is absent."):
        result = TrialMatchAIAdapter._parse_response(answer(criterion=changed), CRITERION)
        assert result["parse_error"] == "criterion_text_mismatch"
    for label in ("met", " Met ", "Met (confirmed)", "Not violated"):
        result = TrialMatchAIAdapter._parse_response(answer(label), CRITERION)
        assert result["parse_status"] == "invalid_label"


def test_unknown_and_format_failure_categories_are_separate():
    unclear = TrialMatchAIAdapter._parse_response(answer("Unclear"), CRITERION)
    irrelevant = TrialMatchAIAdapter._parse_response(answer("Irrelevant"), CRITERION)
    truncated = TrialMatchAIAdapter._parse_response(answer("Met"), CRITERION, truncated=True)
    mismatch = TrialMatchAIAdapter._parse_response(answer(criterion="Another condition"), CRITERION)
    assert unclear["parse_status"] == irrelevant["parse_status"] == "accepted"
    assert unclear["unknown_reason"] == "medical_unknown"
    assert irrelevant["unknown_reason"] == "irrelevant_condition"
    assert truncated["parse_status"] == truncated["unknown_reason"] == "truncated"
    assert mismatch["parse_status"] == "ambiguous_target"
    assert all(row["state"] == "U" for row in (unclear, irrelevant, truncated, mismatch))


def test_trial_wide_decision_is_a_warning_and_not_a_condition_label():
    for decision in ("Possibly", None, {"nested": "Ineligible"}):
        parsed = TrialMatchAIAdapter._parse_response(answer("Met", **{"Final Decision": decision}), CRITERION)
        assert parsed["raw_label"] == "Met" and parsed["parse_error"] is None
        assert "invalid_or_missing_final_decision_not_used" in parsed["parse_warnings"]


def test_nested_json_in_justification_is_not_a_second_answer():
    data = json.loads(answer("Not Met"))
    data["Inclusion_Criteria_Evaluation"][0]["Justification"] = answer("Met")
    parsed = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert parsed["raw_label"] == "Not Met"
    wrapped = TrialMatchAIAdapter._parse_response(json.dumps({"not_the_schema": data}), CRITERION)
    assert wrapped["parse_error"] == "missing_schema_fields"


def test_broken_json_and_duplicate_keys_are_never_repaired_by_guessing():
    for raw in (answer()[:-1], answer().replace('"Classification": "Met"',
            '"Classification": "Unclear", "Classification": "Met"'),
            answer().replace('"Final Decision": "Eligible"', '"Final Decision": NaN')):
        parsed = TrialMatchAIAdapter._parse_response(raw, CRITERION)
        assert parsed["parse_error"] == "invalid_json_response"
        assert parsed["raw_label"] is None


def test_target_whitespace_reflow_is_exact_but_wording_and_punctuation_are_not():
    reflowed = CRITERION.replace("Distant", "Distant\n  ")
    result = TrialMatchAIAdapter._parse_response(answer("Met", criterion=reflowed), CRITERION)
    assert result["parse_error"] is None and result["raw_label"] == "Met"
    assert result["target_match_kind"] == "whitespace_normalized_exact"
    result = TrialMatchAIAdapter._parse_response(answer("Met", criterion=CRITERION.rstrip(".")), CRITERION)
    assert result["parse_error"] == "criterion_text_mismatch"
