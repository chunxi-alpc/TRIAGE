"""Minimal synthetic CPU tests; never loads the benchmark or model."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_faitheval_overlap import (calibration_signatures, compare_signatures, map_trace,
                           normalized_question, prefix_rows, question_hashes, text_sha)
from faitheval_native import method_input, messages, prompt_provenance


class PublicOnly(dict):
    def __getitem__(self, key):
        if key not in {"context", "question", "choices"}:
            raise AssertionError("Evaluation metadata reached the method")
        return super().__getitem__(key)


class OverlapContracts(unittest.TestCase):
    def rows(self):
        return [(i, {"_id": identifier, "question": question, "context": [["Synthetic", ["Evidence."]]]})
                for i, (identifier, question) in enumerate((("first", "Does alpha occur?"), ("second", "Does beta occur?")))]

    def test_prefix_preserves_original_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.json"
            path.write_text(json.dumps([{"_id": "z", "question": "x ] \\\" y"}, {"_id": "a"}, {"_id": "unused"}]))
            rows = list(prefix_rows(path, 2))
            self.assertEqual([row["_id"] for _, row in rows], ["z", "a"])
            self.assertEqual([index for index, _ in rows], [0, 1])

    def test_normalization_never_removes_negation_or_digits(self):
        self.assertEqual(normalized_question("  ALPHA\u00a0 is  ２? "), "alpha is 2?")
        self.assertNotEqual(question_hashes("Alpha is 2?")["question_normalized_sha256"],
                            question_hashes("Alpha is not 2?")["question_normalized_sha256"])

    def test_only_explicit_source_id_match_is_confirmed(self):
        source = calibration_signatures(self.rows())
        test = [
            {"qid": "first", "subset": "HotpotQA", "question": "Changed wording."},
            {"qid": "second", "subset": "Other", "question": "Changed wording."},
            {"qid": "unknown", "subset": "HotpotQA", "question": " DOES ALPHA OCCUR? "},
            {"qid": "unrelated", "subset": "HotpotQA", "question": "Different."},
        ]
        records, counts = compare_signatures("synthetic", test, source, {"expert": {0: "included", 1: "rejected"}})
        self.assertEqual(records[0]["source_link_status"], "confirmed_explicit_source_id")
        self.assertEqual(records[1]["confirmed_hotpot_source_id_matches"], [])
        self.assertEqual(records[1]["unnamespaced_same_id_candidates"], [1])
        self.assertEqual(records[2]["source_link_status"], "question_overlap_source_identity_unknown")
        self.assertEqual(records[3]["source_link_status"], "unknown_no_observed_match")
        self.assertEqual(counts["all_rows"], 4)
        self.assertEqual(counts["confirmed_source_id_overlap_rows"], 1)
        self.assertNotIn("question", records[0])

    def test_selection_trace_joins_by_both_prompt_hashes_not_row_position(self):
        rows = self.rows()
        def assemble(self, context, question):
            return f"Context: {context} ", f"Question: {question}\nContext: {context}"
        context = "Synthetic Evidence."
        h, hp = assemble(None, context, rows[1][1]["question"])
        trace = [{"text_h_sha256": text_sha(h), "text_hp_sha256": text_sha(hp), "included": True}]
        membership, detail = map_trace(rows, trace, assemble)
        self.assertEqual(membership[0], "not_observed_in_trace")
        self.assertEqual(membership[1], "included")
        self.assertEqual(detail["uniquely_mapped_trace_rows"], 1)

    def test_counterfactual_preserves_public_choices_and_excludes_gold(self):
        row = PublicOnly(context="Synthetic context.", question="Synthetic question?",
                         choices={"label": ["2", "1"], "text": ["Second option", "First option"]},
                         answer="SECRET", answerKey="SECRET", justification="SECRET")
        task = method_input(row, kind="counterfactual")
        prompt = messages(task, "counterfactual")[0]["content"]
        self.assertLess(prompt.index("2. Second option"), prompt.index("1. First option"))
        self.assertNotIn("SECRET", prompt)
        self.assertFalse(prompt_provenance("counterfactual")["full_official_script_reproduced"])
        with self.assertRaises(ValueError):
            messages(method_input(row), "counterfactual")


if __name__ == "__main__":
    unittest.main()
