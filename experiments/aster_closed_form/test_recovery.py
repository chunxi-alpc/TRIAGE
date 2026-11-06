"""CPU checks for isolated input copying and the definition-pilot transition."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .common import rows, write_json, write_rows
from .recovery import definition_pilot, prepare_recovery, run_recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.root = self.base / "source", self.base / "repaired"
        self.source.mkdir()
        definitions = [
            dict(group_id=split + "-" + ident, split=split, cohort="definition",
                 source_id=split + ":" + ident, ancestors=[{}] * count)
            for split in ("fit", "dev", "test")
            for ident, count in (("x", 2), ("b", 0), ("a", 0))
        ]
        main = [dict(group_id="main-0", split="fit", cohort="main"),
                dict(group_id="main-1", split="test", cohort="main")]
        constructions = [
            dict(group_id=g["group_id"], cohort=g["cohort"], split=g["split"],
                 construction_revision=2, status="complete",
                 value={"definition_valid": False}, validation={"available": False})
            for g in main + definitions
        ]
        constructions[1]["status"] = "failed"
        files = {
            "groups_main": main, "groups_definition": definitions,
            "path_cases": [{"source_id": "test:1"}], "constructions": constructions,
            "semantic_masks": [
                dict(key="main-0:base", group_id="main-0", status="complete"),
                dict(key="main-1:base", group_id="main-1", status="failed"),
            ],
            "path_semantic_masks": [
                dict(key="test:1:G0", status="complete"),
                dict(key="test:1:G1", status="failed"),
            ],
        }
        for name, values in files.items():
            write_rows(self.source / "data" / (name + ".jsonl"), values)
        write_json(self.source / "configuration.json", {"epsilon": .1, "gpus": [0, 1]})
        write_json(self.source / "data/manifest.json", {"seed": 20261105})
        write_json(self.source / "features/old.json", {"stale": True})
        write_json(self.source / "scores/old.json", {"stale": True})
        write_json(self.source / "operators/old.json", {"stale": True})

    def test_prepare_preserves_source_and_omits_failed_masks_and_caches(self):
        source_before = {p.relative_to(self.source): p.read_bytes()
                         for p in self.source.rglob("*") if p.is_file()}
        result = prepare_recovery(self.source, self.root)
        self.assertEqual(len(rows(self.root / "data/constructions.jsonl")), 11)
        self.assertEqual(rows(self.root / "data/constructions.jsonl")[1]["status"], "failed")
        self.assertEqual([x["key"] for x in rows(self.root / "data/semantic_masks.jsonl")],
                         ["main-0:base"])
        self.assertEqual([x["key"] for x in rows(self.root / "data/path_semantic_masks.jsonl")],
                         ["test:1:G0"])
        for relative in ("features", "scores", "operators", "data/views.jsonl", "data/pairs.jsonl"):
            self.assertFalse((self.root / relative).exists())
        self.assertEqual(source_before, {p.relative_to(self.source): p.read_bytes()
                                        for p in self.source.rglob("*") if p.is_file()})
        pilot = Path(result["pilot_root"])
        self.assertEqual(len(rows(pilot / "data/groups_definition.jsonl")), 6)
        for name in ("groups_main", "path_cases", "constructions", "semantic_masks"):
            self.assertEqual(rows(pilot / "data" / (name + ".jsonl")), [])

    def test_pilot_uses_original_splits_and_ancestor_count_then_id(self):
        original = rows(self.source / "data/groups_definition.jsonl")
        selected = definition_pilot(original)
        self.assertEqual([g["group_id"] for g in selected],
                         ["fit-a", "fit-b", "dev-a", "dev-b", "test-a", "test-b"])
        self.assertEqual(len(original), 9)

    def test_prepare_is_resumable_without_overwriting_new_work(self):
        prepare_recovery(self.source, self.root)
        write_rows(self.root / "data/constructions.jsonl",
                   [dict(group_id="new", construction_revision=3)])
        prepare_recovery(self.source, self.root)
        self.assertEqual(rows(self.root / "data/constructions.jsonl"),
                         [dict(group_id="new", construction_revision=3)])

    def _worker(self, valid_first):
        def spawn(pilot, stage, gpu, shard, shards):
            self.assertEqual(stage, "construct")
            self.assertEqual((gpu, shard, shards), (gpu, gpu, 2))
            groups = rows(pilot / "data/groups_definition.jsonl")[shard::shards]
            self.assertEqual(len(groups), 3)
            constructions, masks = [], []
            for i, group in enumerate(groups):
                valid = valid_first and shard == 0 and i == 0
                constructions.append(dict(
                    group_id=group["group_id"], split=group["split"], cohort="definition",
                    construction_revision=3, status="complete",
                    value={"definition_valid": valid}, validation={"available": valid},
                ))
                masks.append(dict(
                    key=group["group_id"] + ":base", group_id=group["group_id"],
                    construction_revision=3, status="complete" if valid else "failed",
                ))
            write_rows(pilot / "data" / ("constructions_" + str(shard) + ".jsonl"), constructions)
            write_rows(pilot / "data" / ("semantic_masks_" + str(shard) + ".jsonl"), masks)
            return SimpleNamespace(pid=100 + gpu)
        return spawn

    def test_zero_valid_definitions_blocks_without_starting_full_pipeline(self):
        with patch("experiments.aster_closed_form.pipeline.spawn",
                   side_effect=self._worker(False)), \
             patch("experiments.aster_closed_form.pipeline.join"), \
             patch("experiments.aster_closed_form.pipeline.all_stages") as full:
            result = run_recovery(self.source, self.root)
        self.assertEqual(result["state"], "blocked")
        self.assertFalse(result["full_pipeline_started"])
        self.assertEqual(result["pilot"]["valid_definitions"], 0)
        full.assert_not_called()
        recorded = json.loads((self.root / "status/recovery.json").read_text())
        self.assertEqual(recorded["state"], "blocked")

    def test_one_valid_definition_merges_revision_then_starts_full_pipeline(self):
        def full_check(root):
            records = {r["group_id"]: r for r in rows(root / "data/constructions.jsonl")}
            self.assertEqual(len(records), 11)
            self.assertEqual(records["fit-a"]["construction_revision"], 3)
            self.assertEqual(records["fit-b"]["construction_revision"], 3)
            self.assertEqual(records["fit-x"]["construction_revision"], 2)
            masks = rows(root / "data/semantic_masks.jsonl")
            self.assertEqual({m["key"] for m in masks}, {"main-0:base", "fit-a:base"})
            self.assertTrue(all(m["status"] == "complete" for m in masks))
        with patch("experiments.aster_closed_form.pipeline.spawn",
                   side_effect=self._worker(True)), \
             patch("experiments.aster_closed_form.pipeline.join"), \
             patch("experiments.aster_closed_form.pipeline.all_stages",
                   side_effect=full_check) as full:
            result = run_recovery(self.source, self.root)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["pilot"]["valid_definitions"], 1)
        full.assert_called_once_with(self.root)
        run = json.loads((self.root / "recovery_run.json").read_text())
        self.assertTrue(run["started_at"].endswith("+08:00"))


if __name__ == "__main__":
    unittest.main()
