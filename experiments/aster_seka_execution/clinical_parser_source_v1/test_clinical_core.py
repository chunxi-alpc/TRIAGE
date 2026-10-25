"""Synthetic contracts only; execute on the server, not on local research data."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method_agent"))

import torch

from aster_seka_adapter import PredictedSpans, PreparedPrompt, Span
from clinical_core import (attach_controls, chat_prompt, condition_prompt, empty_evidence,
                           lexical_patient_spans, official_prompt, parse_quoted_rules,
                           parse_quoted_rules_individually,
                           support_spans, verify_tfu_tokens)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        output = {"input_ids": self.encode(text), "attention_mask": [1] * len(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return output

    def apply_chat_template(self, messages, tokenize, **kwargs):
        assert kwargs["enable_thinking"] is False
        text = "<system>" + messages[0]["content"] + "</system><assistant>"
        return self.encode(text) if tokenize else text


class MaskAdapter:
    def __init__(self):
        self.operator = SimpleNamespace(tok=CharacterTokenizer())

    def _encode(self, prompt, spans):
        ranges = prompt.absolute_spans(spans)
        mask = torch.tensor([[any(a <= i < b for a, b in ranges) for i in range(len(prompt.text))]])
        return torch.tensor([self.operator.tok.encode(prompt.text)]), torch.ones_like(mask), mask


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.guideline = "If fever, give rest. If cough, arrange review."
        self.raw = json.dumps([
            {"condition_quote": "If fever", "action_quote": "give rest", "exception_quotes": [],
             "rule_quote": "If fever, give rest."},
            {"condition_quote": "If cough", "action_quote": "arrange review", "exception_quotes": [],
             "rule_quote": "If cough, arrange review."},
        ])
        self.rules = parse_quoted_rules(self.raw, self.guideline)

    def test_every_quote_exact_and_source_offsets(self):
        for rule in self.rules:
            self.assertEqual(self.guideline[rule.start:rule.end], rule.rule_quote)
        corrupted = self.raw.replace("If fever", "If very high fever", 1)
        with self.assertRaisesRegex(ValueError, "verbatim"):
            parse_quoted_rules(corrupted, self.guideline)

    def test_extraction_does_not_accept_gold_fields(self):
        data = json.loads(self.raw)
        data[0]["gold_rule"] = True
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_quoted_rules(json.dumps(data), self.guideline)

    def test_partial_quotes_keep_valid_items_and_original_ids(self):
        items = json.loads(self.raw)
        items.insert(1, {**items[0], "condition_quote": "invented condition"})
        rules, audit = parse_quoted_rules_individually(json.dumps(items), self.guideline)
        self.assertEqual([rule.rule_id for rule in rules], ["R0001", "R0003"])
        self.assertEqual(audit[1]["status"], "rejected")
        self.assertEqual(audit[1]["raw_item"]["condition_quote"], "invented condition")
        self.assertEqual(len(audit[1]["item_sha256"]), 64)

    def test_partial_validator_keeps_all_rejected_items(self):
        items = [{"not_a_rule": 1}, {"not_a_rule": 2}]
        rules, audit = parse_quoted_rules_individually(json.dumps(items), self.guideline)
        self.assertEqual(rules, ())
        self.assertEqual([entry["status"] for entry in audit], ["rejected", "rejected"])

    def test_partial_validator_does_not_guess_truncated_json(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_quoted_rules_individually(self.raw[:-3], self.guideline)

    def test_condition_action_overlap_rejected(self):
        data = json.loads(self.raw)
        data[0]["condition_quote"] = data[0]["rule_quote"]
        with self.assertRaisesRegex(ValueError, "proposed action"):
            parse_quoted_rules(json.dumps(data), self.guideline)

    def test_blind_condition_input_retains_patient_no_candidate_actions(self):
        patient = "Fever today. Previously took aspirin."
        prompt = condition_prompt(patient, self.rules, "R0001")
        self.assertIn(patient, prompt.text)
        self.assertIn("If fever", prompt.text)
        self.assertNotIn("give rest", prompt.text)
        self.assertNotIn("arrange review", prompt.text)
        self.assertIn("Previously took aspirin", prompt.text)

    def test_lexical_evidence_is_not_a_truth_label(self):
        patient = "No fever. Cough present."
        selected = lexical_patient_spans(patient, "If fever", "R0001")
        self.assertEqual([patient[s.start:s.end] for s in selected], ["No fever."])

    def test_official_question_patient_and_full_guide_preserved(self):
        template = "Use {Guideline}\nCase: {Case}\nGive a numbered clinical decision process and final outcome."
        patient = "The original patient history."
        p = official_prompt(template, self.guideline, patient, tfu=[{"rule_id": "R0001", "label": "U"}])
        self.assertIn(self.guideline, p.text)
        self.assertIn(patient, p.text)
        self.assertIn("Give a numbered clinical decision process and final outcome.", p.text)
        self.assertIn("unknown is not false", p.text)

    def test_chat_template_offsets_and_input_ids_match(self):
        template = "Guide={Guideline}; Patient={Case}; Original question."
        prompt = official_prompt(template, self.guideline, "Patient narrative.")
        prepared = chat_prompt(prompt, CharacterTokenizer())
        for name, (offset, text) in prepared.fields.items():
            self.assertEqual(prepared.text[offset:offset + len(text)], text)
        self.assertEqual(prepared.text.count(prompt.text), 1)

    def test_unknown_keeps_full_context_but_does_not_fabricate_support(self):
        self.assertFalse(support_spans(self.rules, [{"rule_id": "R0001", "label": "U"}]))
        supports = support_spans(self.rules, [{"rule_id": "R0001", "label": "T"}])
        self.assertEqual(self.guideline[supports[0].start:supports[0].end], "If fever, give rest.")

    def test_wrong_span_control_same_token_count_and_disjoint(self):
        prompt = PreparedPrompt("abcdefghijklmnop", {"patient_history": (0, "abcdefghijklmnop")},
                                PredictedSpans("synthetic", (Span("patient_history", 2, 4, "R0001"),)),
                                "condition", "tfu")
        adapter = MaskAdapter()
        prepared = attach_controls(prompt, adapter, (), target_rule=None)
        _, _, normal = adapter._encode(prepared, prepared.evidence.spans)
        _, _, wrong = adapter._encode(prepared, prepared.evidence.wrong_spans)
        self.assertEqual(int(normal.sum()), int(wrong.sum()))
        self.assertFalse((normal & wrong).any())

    def test_tfu_labels_require_three_distinct_exact_tokens(self):
        ids = verify_tfu_tokens(CharacterTokenizer())
        self.assertEqual(ids, {"T": ord("T"), "F": ord("F"), "U": ord("U")})

    def test_all_supported_rule_rotation_is_not_fake_intervention(self):
        template = "{Guideline}\n{Case}\nQuestion"
        evidence = PredictedSpans("synthetic", support_spans(self.rules, [
            {"rule_id": "R0001", "label": "T"}, {"rule_id": "R0002", "label": "T"}]))
        prompt = official_prompt(template, self.guideline, "Patient", evidence=evidence)
        controlled = attach_controls(prompt, MaskAdapter(), self.rules)
        self.assertEqual(controlled.evidence.misaligned_rule_spans, ())


if __name__ == "__main__":
    unittest.main()
