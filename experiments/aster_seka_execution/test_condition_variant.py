"""One server-only regression for actual A0/A1 input and shared-cost selection."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method_agent"))

from clinical_core import chat_prompt, official_prompt, parse_quoted_rules, sha
from run_mega_additional_baselines import load_selected_primary, prepare_condition_inputs
from test_clinical_core import CharacterTokenizer


class ConditionVariantTest(unittest.TestCase):
    def test_selected_tfu_evidence_reference_and_measured_cost_stay_together(self):
        guideline = "If fever, give rest. If cough, arrange review."
        raw = [{"condition_quote": "If fever", "action_quote": "give rest", "exception_quotes": [],
                "rule_quote": "If fever, give rest."},
               {"condition_quote": "If cough", "action_quote": "arrange review", "exception_quotes": [],
                "rule_quote": "If cough, arrange review."}]
        rules = parse_quoted_rules(json.dumps(raw), guideline)
        task = SimpleNamespace(guideline=guideline, vignette="Original patient history: cough.")
        template = "Guideline: {Guideline}\nPatient: {Case}\nOriginal question: Give the native path."
        tok = CharacterTokenizer()
        predictions = {variant: [{"rule_id": "R0001", "label": first, "status": "ok"},
                                 {"rule_id": "R0002", "label": second, "status": "ok"}]
                       for variant, first, second in [("A0", "T", "F"), ("A1", "F", "T")]}
        expected_prompts = {variant: chat_prompt(official_prompt(template, guideline, task.vignette, tfu=values), tok).text
                            for variant, values in predictions.items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stages = [{"sample_id": "synthetic", "stage_configuration": "guideline_extraction", "cache_key": "same_guide"}]
            for variant, calls, seconds in [("A0", 1, .1), ("A1", 7, .7)]:
                for prediction in predictions[variant]:
                    stages.append({"sample_id": "synthetic", "stage_configuration": variant,
                                   "prediction": prediction,
                                   "generation": {"actual_forward_calls": calls, "wall_seconds": seconds}})
                reference = {"sample_id": "synthetic", "llm_output": variant + " actual cached path", "status": "ok",
                             "generation": {"prompt_sha256": sha(expected_prompts[variant])}}
                (root / f"{variant}B0.predictions.private.jsonl").write_text(json.dumps(reference) + "\n")
            (root / "stage_predictions.private.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stages))
            default_cases, default_path, default_references = load_selected_primary(root)
            self.assertEqual(default_path.name, "A0B0.predictions.private.jsonl")
            a0 = prepare_condition_inputs(task, rules, default_cases["synthetic"], template, tok,
                                          default_references["synthetic"])
            cases, path, references = load_selected_primary(root, "A1")
            self.assertEqual(path.name, "A1B0.predictions.private.jsonl")
            self.assertEqual(references["synthetic"]["llm_output"], "A1 actual cached path")
            a1 = prepare_condition_inputs(task, rules, cases["synthetic"], template, tok,
                                          references["synthetic"], "A1")
            for variant, result, rule, calls in [("A0", a0, "R0001", 1), ("A1", a1, "R0002", 7)]:
                tfu, spans, base, support, prior, cost = result
                self.assertEqual(tfu, predictions[variant])
                self.assertEqual([span.rule_id for span in spans], [rule])
                self.assertEqual(base.text, expected_prompts[variant])
                self.assertIn(task.vignette, base.text)
                self.assertIn("Original question: Give the native path.", base.text)
                self.assertIn(json.dumps(predictions[variant], ensure_ascii=False, sort_keys=True), prior.text)
                expected_quote = rules[0 if variant == "A0" else 1].rule_quote
                self.assertTrue(support.text.endswith(expected_quote + "</system><assistant>"))
                self.assertEqual(cost["condition_configuration"], variant)
                self.assertEqual([item["actual_forward_calls"] for item in cost["condition_stage_costs"].values()], [calls, calls])
                self.assertEqual([item["wall_seconds"] for item in cost["condition_stage_costs"].values()],
                                 ([.1, .1] if variant == "A0" else [.7, .7]))
            self.assertNotEqual(a0[2].text, a1[2].text)
            with self.assertRaisesRegex(ValueError, "A1B0"):
                prepare_condition_inputs(task, rules, cases["synthetic"], template, tok,
                                         default_references["synthetic"], "A1")


if __name__ == "__main__":
    unittest.main()
