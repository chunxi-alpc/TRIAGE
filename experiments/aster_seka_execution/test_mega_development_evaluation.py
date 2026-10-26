"""Server CPU synthetic tests; point ASTER_MEGA_OFFICIAL_PARSER at pinned code."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mega_development_evaluation import (RequestRecorder, align_predictions, compare_methods,
                                         load_official, native_metrics, parse_with_official)


class EvaluationContracts(unittest.TestCase):
    def manifest(self):
        return {"selected": [{"sample_id": str(i), "source_cluster": f"source_{i}",
                              "identity": {"id": i, "case_path_id": i, "path_index": 0, "filename": f"doc_{i}"}}
                             for i in range(2)]}

    def test_missing_generation_retains_frozen_denominator(self):
        frozen = self.manifest()
        row = {**frozen["selected"][0]["identity"], "language": "en", "llm_output": "synthetic", "status": "ok"}
        result = align_predictions(frozen, [row])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1][1]["evaluation_input_failure"], "missing_prediction")

    def test_duplicate_case_is_not_silently_discarded(self):
        frozen = self.manifest()
        row = {**frozen["selected"][0]["identity"], "language": "en", "llm_output": "synthetic"}
        with self.assertRaises(ValueError):
            align_predictions(frozen, [row, row])

    def test_continuous_changes_are_not_binary_corrections(self):
        rows = [{"sample_id": str(i), "tree_cluster": f"tree_{i}", "method": method,
                 "acc_normalized": acc, "pcc_dtw_cost": pcc}
                for i in range(2) for method, acc, pcc in (("baseline", .5, .4), ("candidate", .75, .6))]
        result = compare_methods(rows, "baseline", bootstrap_samples=100)
        accuracy = next(row for row in result if row["metric"] == "acc_normalized")
        pathway = next(row for row in result if row["metric"] == "pcc_dtw_cost")
        self.assertEqual(accuracy["score_improved"], 2)
        self.assertEqual(pathway["score_degraded"], 2)
        self.assertIsNone(accuracy["binary_corrected"])
        self.assertIsNone(pathway["binary_regressed"])


@unittest.skipUnless(os.environ.get("ASTER_MEGA_OFFICIAL_PARSER"), "Pinned official parser path required for native CPU tests")
class OfficialNativeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official, _ = load_official(Path(os.environ["ASTER_MEGA_OFFICIAL_PARSER"]))

    def test_official_parser_valid_synthetic_output(self):
        row = {"id": 1, "case_path_id": 2, "path_index": 0, "filename": "synthetic", "language": "en", "status": "ok",
               "llm_output": "### Clinical Decision Process:\n1. Synthetic decision.\n### Clinical Decision Outcome: Synthetic outcome."}
        parsed, failure = parse_with_official(self.official["parser"], row)
        self.assertTrue(parsed["parsed_success"])
        self.assertIsNone(failure)

    def test_native_dtw_and_continuous_accuracy(self):
        result = native_metrics({"results": [[[5, 1], [1, 5]]] * 5}, {"results": [1, 2, 3, 4, 5]},
                                self.official["summary"], self.official["metrics"])
        self.assertEqual(result["pcc_dtw_cost"], 0.)
        self.assertEqual(result["acc_normalized"], .5)
        self.assertTrue(result["pathway_valid"] and result["accuracy_valid"])

    def test_incomplete_judge_repeats_receive_native_worst_scores(self):
        result = native_metrics({"results": [[[5]]] * 4}, {"results": [5] * 4},
                                self.official["summary"], self.official["metrics"])
        self.assertEqual((result["pcc_dtw_cost"], result["acc_normalized"]), (1., 0.))
        self.assertFalse(result["pathway_valid"] or result["accuracy_valid"])

    def test_all_actual_judge_retry_attempts_are_recorded(self):
        calls = []
        def create(**request):
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("Synthetic transport failure")
            payload = {"choices": [{"message": {"content": "5"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
            return SimpleNamespace(model_dump=lambda: payload, choices=[SimpleNamespace(message=SimpleNamespace(content="5"))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        args = SimpleNamespace(qwen_model_name="synthetic", qwen_max_tokens=4096)
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestRecorder(client, Path(directory) / "raw.jsonl", Path(directory) / "costs.jsonl")
            recorder.context = {"method": "synthetic", "sample_id": "1", "metric": "accuracy"}
            with patch.object(self.official["accuracy"].time, "sleep", return_value=None):
                response = self.official["accuracy"].qwen_chat(recorder, args, "Synthetic prompt")
            recorder.close()
            self.assertEqual(response, "5")
            self.assertEqual(len(recorder.rows), 2)
            self.assertEqual([row["status"] for row in recorder.rows], ["error", "ok"])
            costs = recorder.cost_summary()["by_method"]["synthetic"]
            self.assertEqual(costs["failed_attempts"], 1)
            self.assertEqual(costs["completion_tokens_known"], 1)


if __name__ == "__main__":
    unittest.main()
