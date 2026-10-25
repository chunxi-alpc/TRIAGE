"""Synthetic server CPU tests for paired BiasBios scores and prefill matching."""
import unittest
from compare_biasbios_paired import compare_prefill_rows, score_pairs


def trace(steps):
    return [{"completed_call_index": index + 1,
             "input_ids": {"shape": [batch, length], "dtype": "torch.int64", "sha256": code},
             "attention_mask": None}
            for index, (batch, length, code) in enumerate(steps)]


class PairedBiosContracts(unittest.TestCase):
    def test_different_decode_suffixes_and_call_counts_are_allowed(self):
        left = trace([(2, 10, "p0"), (2, 1, "decode_a"), (2, 1, "decode_b"), (1, 7, "p1"), (1, 1, "decode_c")])
        right = trace([(2, 10, "p0"), (2, 1, "other"), (1, 7, "p1"), (1, 1, "other2"), (1, 1, "other3"), (1, 1, "other4")])
        summary, rows = compare_prefill_rows(left, right, expected_cases=3, batch_size=2)
        self.assertEqual(summary["status"], "complete_verified")
        self.assertEqual(summary["prefill_batches"], 2)
        self.assertNotEqual(rows[1]["seka_call_index"], rows[1]["matched_zero_call_index"])

    def test_prefill_token_mismatch_is_detected(self):
        left = trace([(2, 10, "p0"), (1, 7, "p1")])
        right = trace([(2, 10, "different"), (1, 7, "p1")])
        summary, _ = compare_prefill_rows(left, right, expected_cases=3, batch_size=2)
        self.assertEqual(summary["status"], "failed")

    def test_missing_mask_is_not_treated_as_explicit_none(self):
        left = trace([(2, 10, "p0"), (1, 7, "p1")])
        right = trace([(2, 10, "p0"), (1, 7, "p1")])
        del right[0]["attention_mask"]
        summary, _ = compare_prefill_rows(left, right, expected_cases=3, batch_size=2)
        self.assertEqual(summary["status"], "blocked")

    def test_unique_rows_with_same_name_metadata_are_not_merged(self):
        def row(index, top1, top3):
            return {"absolute_processed_row_index": 5000 + index, "target": "synthetic_occupation",
                    "prompt_sha256": f"prompt{index}", "source_name_id_sha256": "same_name_hash",
                    "top1_correct": top1, "topk_correct": top3}
        candidate = {"unique0": row(0, True, True), "unique1": row(1, False, False)}
        reference = {"unique0": row(0, False, True), "unique1": row(1, True, True)}
        rows, metrics = score_pairs(candidate, reference)
        self.assertEqual(len(rows), 2)
        self.assertEqual(metrics["top1"]["corrected"], 1)
        self.assertEqual(metrics["top1"]["regressed"], 1)
        self.assertEqual(metrics["top3"]["difference"], -.5)


if __name__ == "__main__":
    unittest.main()
