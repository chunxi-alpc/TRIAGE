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
    for raw in ("Met. Evidence is present.", "<think>Met", "Reasoning: Met. " + answer("Met")):
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
    assert multiple["parse_error"] == "expected_exactly_one_target_criterion"
    data = json.loads(answer())
    data["Exclusion_Criteria_Evaluation"] = [{"Criterion": "Not used", "Classification": "Violated", "Justification": "No"}]
    wrong_side = TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)
    assert wrong_side["parse_error"] == "expected_exactly_one_target_criterion"


def test_schema_missing_or_invalid_labels_are_unknown():
    samples = [("{}", "missing_schema_fields"),
               (answer("Met because this looks likely"), "invalid_classification"),
               (answer(**{"Final Decision": "Possibly"}), "invalid_final_decision")]
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
    assert TrialMatchAIAdapter._parse_response(json.dumps(data), CRITERION)["state"] == "U"
