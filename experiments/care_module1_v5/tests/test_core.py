from __future__ import annotations
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import (  # noqa: E402
    BioLinkBERTMedNLI, CAREModule1V5, FALSE, TRUE, UNKNOWN, _active_spans,
    _numeric_observation, eval_logic, map_trialmatch_label, sentences_with_offsets,
)


def test_module_one_default_import_does_not_load_legacy_qwen_backend():
    import src.module1
    assert "src.module1.modeler" not in sys.modules
    assert callable(src.module1.run)


def test_trialmatch_label_mapping_is_conservative():
    assert map_trialmatch_label("Met") == TRUE
    assert map_trialmatch_label("Not Met") == FALSE
    assert map_trialmatch_label("Unclear") == UNKNOWN
    assert map_trialmatch_label("Irrelevant") == UNKNOWN
    assert map_trialmatch_label("Violated", "exclusion") == TRUE
    assert map_trialmatch_label("Not Violated", "exclusion") == UNKNOWN


def test_mednli_ids_are_loaded_from_config_mapping():
    model = object()
    tokenizer = object()
    classifier = BioLinkBERTMedNLI(model, tokenizer, {0: "neutral", 1: "contradiction", 2: "entailment"})
    assert classifier.required == {"entailment": 2, "neutral": 0, "contradiction": 1}


def test_three_valued_logic_preserves_unknown():
    states = {"a": TRUE, "b": UNKNOWN, "c": FALSE}
    assert eval_logic({"op": "AND", "args": ["a", "b"]}, states) == UNKNOWN
    assert eval_logic({"op": "AND", "args": ["a", "c"]}, states) == FALSE
    assert eval_logic({"op": "OR", "args": ["b", "c"]}, states) == UNKNOWN
    assert eval_logic({"op": "NOT", "args": ["b"]}, states) == UNKNOWN


def test_offsets_survive_unicode_decimal_and_newline_splitting():
    text = "患者年龄为4岁。 Lab value is 2.8 mmol/L.\nNo distant metastasis."
    spans = sentences_with_offsets(text)
    assert [s["text"] for s in spans] == ["患者年龄为4岁。", "Lab value is 2.8 mmol/L.", "No distant metastasis."]
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)


def test_explicit_correction_supersedes_old_fact_but_keeps_live_conflicts():
    text = "Current distant metastasis is positive.\nRecord correction: all earlier statements about distant metastasis were copied in error and are withdrawn.\nCorrected record: current distant metastasis is negative.\nReport A: distant metastasis positive. Report B: distant metastasis negative. Both reports remain signed; neither is withdrawn or superseded."
    spans = sentences_with_offsets(text)
    condition = {"field": "distant", "text": "current distant metastasis"}
    active = _active_spans(condition, spans)
    assert spans[0] not in active
    assert spans[-1] in active


def test_numeric_comparison_ignores_unrelated_numbers_and_unknown_on_conflict():
    text = "The verified invasive breast tumor diameter is 3.5 cm; accession number 150 is unrelated."
    spans = sentences_with_offsets(text)
    condition = {"field": "tumor_cm", "text": "tumor diameter is at most 5 cm",
                 "numeric": {"field": "tumor_cm", "operator": "<=", "value": 5}}
    result = _numeric_observation(condition, spans)
    assert result["status"] == TRUE and result["value"] == 3.5
    text = "Tumor diameter is 3 cm. Tumor diameter is 6 cm."
    result = _numeric_observation(condition, sentences_with_offsets(text))
    assert result["status"] == UNKNOWN


def test_numeric_ranges_and_unit_conversion_are_deterministic():
    stage = {"field": "pt", "text": "original pathological T stage at most 2",
             "numeric": {"field": "pt", "operator": "<=", "value": 2}}
    result = _numeric_observation(stage, sentences_with_offsets("Original pathology: pT1/pT2."))
    assert result["status"] == TRUE
    tumor = {"field": "tumor_cm", "text": "tumor diameter at most 5 cm",
             "numeric": {"field": "tumor_cm", "operator": "<=", "value": 5, "unit": "cm"}}
    assert _numeric_observation(tumor, sentences_with_offsets("Invasive tumor diameter was 35 mm."))["status"] == TRUE
    margin = {"field": "margin_mm", "text": "margin at least 1 mm",
              "numeric": {"field": "margin_mm", "operator": ">=", "value": 1, "unit": "mm"}}
    assert _numeric_observation(margin, sentences_with_offsets("Closest margin measured 0.1 cm."))["status"] == TRUE
    age = {"field": "age", "text": "patient age below 5 years",
           "numeric": {"field": "age", "operator": "<", "value": 5, "unit": "years"}}
    assert _numeric_observation(age, sentences_with_offsets("The patient is 60 months old."))["status"] == FALSE


def test_missing_numeric_value_stays_unknown():
    condition = {"field": "tumor_cm", "text": "tumor diameter at most 5 cm",
                 "numeric": {"field": "tumor_cm", "operator": "<=", "value": 5}}
    assert _numeric_observation(condition, sentences_with_offsets("Tumor size is not reported.")) is None


class FakeTrialMatch:
    def __init__(self, output): self.output = output
    def judge(self, patient, condition, criterion_kind="inclusion"):
        return {"raw_output": self.output, "raw_label": self.output, "state": map_trialmatch_label(self.output),
                "criterion_kind": criterion_kind}


class FakeNLI:
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    def __init__(self, relation): self.relation = relation
    def predict(self, pairs, batch_size=32):
        return [{"relation": self.relation, "class_id": 0 if self.relation == "contradiction" else 2,
                 "scores": {"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95}}
                for _ in pairs]


def test_evidence_conflict_becomes_unknown_without_silent_override():
    engine = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("contradiction"))
    guideline = [{"rule_id": "R1", "text": "Has disease", "source_span": {"start": 0, "end": 10},
        "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present",
                        "source_span": {"start": 0, "end": 10}}],
        "logic": {"op": "AND", "args": ["c1"]}, "exceptions": []}]
    output = engine.run(guideline, "Current distant metastasis is present.", "Assess the rule")
    assert output["rules"][0]["applicability"] == UNKNOWN
    assert output["rules"][0]["uncertain"] is True
    assert output["metadata"]["qwen_called"] is False
    assert output["condition_evidence_relations"][0]["refute_evidence"]
    assert {"candidate_rules", "condition_evidence_relations", "applicability", "critical_spans",
            "relation_weights", "binding_gate", "action_gate", "conflict_scores"}.issubset(output)


def test_output_converts_to_existing_pipeline_token_contract():
    import importlib.util
    module_path = ROOT / "src" / "pipeline" / "tokens.py"
    spec = importlib.util.spec_from_file_location("v5_pipeline_tokens", module_path)
    tokens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokens)

    class Box:
        def __init__(self, value): self.value = value
        def tolist(self): return self.value

    class FakeTokenizer:
        is_fast = True
        def apply_chat_template(self, messages, **kwargs):
            return "SYSTEM\n" + messages[0]["content"] + "\nUSER\n" + messages[1]["content"]
        def __call__(self, text, **kwargs):
            offsets = [(index, index + 1) for index in range(len(text))]
            return {"input_ids": Box(list(range(len(text)))), "offset_mapping": [Box(offsets)]}

    engine = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("entailment"))
    patient = "Current distant metastasis is present."
    guideline = [{"rule_id": "R1", "text": "Has disease", "source_span": {"start": 0, "end": 11, "unit": "character"},
        "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present",
                        "source_span": {"start": 0, "end": 10, "unit": "character"}}],
        "logic": {"op": "AND", "args": ["c1"]}, "exceptions": []}]
    graph = engine.run(guideline, patient, "Assess the rule")
    _, _, mapping = tokens.build_prompt_and_mapping(FakeTokenizer(), patient, "Assess the rule", graph)
    assert mapping["rules"]["R1"]["source_span"] == {"start": 0, "end": 11, "unit": "character"}
    assert mapping["relations"][0]["evidence_tokens"]


def test_default_module1_class_never_uses_final_qwen_backend():
    import src.module1
    from src.module1.compat_v5 import Module1
    assert src.module1.Module1 is Module1

    class FinalBackend:
        device = "cuda:6"
        def json(self, *args, **kwargs):
            raise AssertionError("Module one must never call the final Qwen backend")
        def _load_qwen(self):
            raise AssertionError("Module one must never load the final Qwen backend")

    engine = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("entailment"))
    wrapper = Module1(backend=FinalBackend(), engine=engine)
    output = wrapper.run([{"rule_id": "R1", "text": "Distant metastasis rule",
        "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present"}],
        "action": "Consider the documented action"}],
        "Current distant metastasis is present.", "Assess the rule")
    assert output["schema_version"] == "module1.v1"
    assert output["implementation_version"] == "module1.v5"
    assert output["metadata"]["qwen_backend_used"] is False
    assert "src.module1.modeler" not in sys.modules


def test_free_text_uses_exact_offline_registry_or_rejects_before_model_load(tmp_path):
    import json
    import pytest
    from src.module1.compat_v5 import Module1
    rule = {"rule_id": "R1", "source_text": "Exact original clause.", "text": "Prepared clause.",
            "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present"}]}
    registry = tmp_path / "rules.jsonl"
    registry.write_text(json.dumps(rule) + "\n")
    wrapper = Module1(rules_path=registry,
        engine=CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("entailment")))
    output = wrapper.run("Exact original clause.", "Current distant metastasis is present.", "Assess")
    assert output["rules"][0]["rule_id"] == "R1"
    with pytest.raises(ValueError, match="unique exact match"):
        Module1(rules_path=registry).run("Unregistered clause", "Patient", "Assess")


def test_compatibility_confidence_uses_real_evidence_scores_and_correct_proof_ids():
    from src.module1.compat_v5 import _logic_proof
    assert _logic_proof({"op": "OR", "args": ["a", "b"]},
                        {"a": TRUE, "b": UNKNOWN}) == (TRUE, ["a"])
    assert _logic_proof({"op": "AND", "args": ["a", "b"]},
                        {"a": TRUE, "b": FALSE}) == (FALSE, ["b"])
    assert _logic_proof({"op": "NOT", "args": ["a"]}, {"a": FALSE}) == (TRUE, ["a"])
    rule = {"rule_id": "R1", "text": "Has disease",
            "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present"}]}
    output = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("entailment")).run(
        [rule], "Current distant metastasis is present.", "Assess")
    app, relation = output["applicability"][0], output["condition_evidence_relations"][0]
    assert app["decisive_condition_ids"] == ["c1"]
    assert app["supporters"] == ["c1"] and app["blockers"] == []
    # FakeNLI deliberately provides entailment probability .02; never manufacture
    # a high value merely because its argmax label was passed as entailment.
    assert relation["confidence"] == 0.02
    assert relation["confidence_kind"] == "uncalibrated_mednli_probability"
    assert output["binding_gate"]["R1"]["c1"] == 0.02
    assert output["decisive_relations"][0]["condition_id"] == "c1"
    assert output["conflict_scores"][0]["prior_support"] == "unknown"
    assert output["relation_weights"][0]["prior_competition_weight"] == 0.0


def test_compatibility_unknown_evidence_is_preserved_without_high_confidence():
    rule = {"rule_id": "R1", "text": "Has disease",
            "conditions": [{"condition_id": "c1", "field": "distant", "text": "distant metastasis is present"}]}
    output = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("neutral")).run(
        [rule], "Current distant metastasis is present.", "Assess")
    relation = output["condition_evidence_relations"][0]
    assert relation["status"] == "satisfied"
    assert relation["confidence"] == 0.0
    assert relation["evidence"] == []
    assert relation["unknown_evidence"]
    assert output["binding_gate"]["R1"]["c1"] == 0.0


def test_clear_numeric_comparison_keeps_value_when_nli_disagrees():
    engine = CAREModule1V5(FakeTrialMatch("Met"), FakeNLI("contradiction"))
    guideline = [{"rule_id": "R1", "text": "Tumor diameter at most 5 cm",
        "conditions": [{"condition_id": "c1", "field": "tumor_cm", "text": "tumor diameter at most 5 cm",
            "numeric": {"field": "tumor_cm", "operator": "<=", "value": 5, "unit": "cm"}}],
        "logic": {"op": "AND", "args": ["c1"]}, "exceptions": []}]
    rule = engine.run(guideline, "Invasive tumor diameter was 35 mm.", "Assess this rule")["rules"][0]
    assert rule["applicability"] == TRUE
    assert rule["conditions"][0]["conflict"] is True
    assert rule["conditions"][0]["resolution"] == "deterministic_numeric"
