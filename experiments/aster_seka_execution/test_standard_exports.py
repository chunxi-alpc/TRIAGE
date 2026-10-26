"""Synthetic contracts; run on the experiment server only."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from counterfact_entity_ci import bootstrap_effects, iter_json_objects
from export_biasbios import export_rows


class FixedDraw:
    def integers(self, low, high, size):
        assert high == 2
        return np.tile([0, 1], (size[0], 1))


class ExportContracts(unittest.TestCase):
    def test_concatenated_objects_and_jsonl_keep_identical_order(self):
        rows = [{"id": index, "entity": "synthetic", "text": "中文 }{ escaped \\\""} for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            for separator in ("", "\n", " \n\t"):
                path = Path(directory) / "objects.jsonl"
                path.write_text(separator.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
                self.assertEqual(list(iter_json_objects(path, chunk_size=7)), rows)

    def test_truncated_json_object_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.jsonl"
            path.write_text('{"id":0}{"id":', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(iter_json_objects(path, chunk_size=4))

    def test_cluster_bootstrap_retains_case_weighting(self):
        # Every draw takes both clusters: 3 successes + 1 failure = .75, not .5.
        with patch("counterfact_entity_ci.np.random.default_rng", return_value=FixedDraw()):
            intervals = bootstrap_effects([[1], [1], [1], [0]], ["a", "a", "a", "b"], 100)
        self.assertEqual(intervals, [[.75, .75]])

    def test_negative_effect_is_retained(self):
        with patch("counterfact_entity_ci.np.random.default_rng", return_value=FixedDraw()):
            intervals = bootstrap_effects([[-1], [-1], [-1], [0]], ["a", "a", "a", "b"], 100)
        self.assertEqual(intervals, [[-.75, -.75]])

    def fixture(self):
        source, samples = [], []
        for position, target in enumerate(("doctor", "artist")):
            original = {"id": "synthetic_same_name", "target_mediated": target,
                        "context": "Synthetic person works as " + target,
                        "prompt": "Synthetic person has the occupation of a/an ",
                        "attribute": "works as " + target}
            source.append(original)
            samples.append({"id": original["id"], "target": target,
                "predictions": ["doctor", "artist"], "logp_predictions": [-.1, -.2],
                "logp_target": -.1 if position == 0 else -.2,
                "prompt": original["context"] + ".\n\n" + original["prompt"],
                "generation": "Synthetic answer", "fluency": 1., "consistency": .2})
        return {"samples": samples, "metrics": {"top1_accuracy": .5, "topk_accuracy": 1., "k": 2}}, source

    def test_duplicate_names_kept_and_native_metrics_recomputed(self):
        raw, source = self.fixture()
        rows, info = export_rows(raw, source, "synthetic_sha", 5000, 5002)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["sample_id"], rows[1]["sample_id"])
        self.assertEqual(rows[0]["source_name_id_sha256"], rows[1]["source_name_id_sha256"])
        self.assertEqual(info["duplicate_name_id_extra_rows"], 1)
        self.assertEqual(info["recomputed_metrics"], {"top1_accuracy": .5, "topk_accuracy": 1., "k": 2})
        self.assertFalse(any("prompt" in row or "generation" in row or "id" in row for row in rows))

    def test_wrong_source_order_detected_despite_duplicate_name(self):
        raw, source = self.fixture()
        with self.assertRaises(ValueError):
            export_rows(raw, list(reversed(source)), "synthetic_sha", 5000, 5002)


if __name__ == "__main__":
    unittest.main()
