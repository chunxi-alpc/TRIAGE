"""Synthetic contract tests. Execute on the experiment server only."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mega_trainval_reader as reader


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "snapshot"
        split = self.root / "train_val"
        split.mkdir(parents=True)
        rows = []
        for group in range(6):
            name = f"document_{group}"
            path = split / "guideline_md" / name / f"{name}.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"Synthetic public guideline {group}", encoding="utf-8")
            for patient in range(2):
                rows.append({"guideline_id": group, "path_id": patient,
                             "filename": name, "text": f"Synthetic vignette {patient}",
                             "gold_path": "SECRET_GOLD", "gold_answer": "SECRET_GOLD",
                             "gt_rule": "SECRET_GOLD", "language": "en"})
        (split / "vignette_cases.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        # These files must never be opened by the method data reader.
        (split / "gt_paths.jsonl").write_text("SECRET_GOLD", encoding="utf-8")
        (split / "decision_trees.jsonl").write_text("SECRET_GOLD", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def make_hf_cache(self):
        cache = self.root.parent / "hf_cache" / "datasets--nchennnn--MEGA-CDP"
        snapshot = cache / "snapshots" / reader.DATASET_REVISION
        snapshot.parent.mkdir(parents=True)
        self.root.rename(snapshot)
        self.root = snapshot
        blobs = cache / "blobs"
        blobs.mkdir()
        files = [snapshot / "train_val" / "vignette_cases.jsonl"]
        files += sorted((snapshot / "train_val" / "guideline_md").rglob("*.md"))
        for index, path in enumerate(files):
            content = path.read_bytes()
            # HF ordinary files use git blob SHA1; LFS files use raw SHA256.
            object_id = (hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                         if index % 2 == 0 else hashlib.sha256(content).hexdigest())
            blob = blobs / object_id
            blob.write_bytes(content)
            path.unlink()
            path.symlink_to(os.path.relpath(blob, path.parent))

    def test_real_hf_cache_links_round_trip(self):
        self.make_hf_cache()
        manifest = reader.build_manifest(self.root, batch_size=2)
        tasks = list(reader.iter_method_tasks(self.root, manifest))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(manifest["snapshot_guideline_files"]), 6)
        self.assertEqual(manifest["source_file_sha256"], reader.digest(self.root / "train_val" / "vignette_cases.jsonl"))

    def test_link_to_test_file_is_rejected(self):
        self.make_hf_cache()
        forbidden = self.root / "test" / "document_0.md"
        forbidden.parent.mkdir()
        forbidden.write_text("Synthetic held-out content", encoding="utf-8")
        link = self.root / "train_val" / "guideline_md" / "document_0" / "document_0.md"
        link.unlink()
        link.symlink_to(os.path.relpath(forbidden, link.parent))
        with self.assertRaises(ValueError):
            reader.build_manifest(self.root, batch_size=2)

    def test_changed_hf_blob_content_is_rejected(self):
        self.make_hf_cache()
        manifest = reader.build_manifest(self.root, batch_size=2)
        name = manifest["selected"][0]["identity"]["filename"]
        link = self.root / "train_val" / "guideline_md" / name / f"{name}.md"
        link.resolve().write_text("Synthetic changed cache bytes", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(reader.iter_method_tasks(self.root, manifest))

    def test_unlisted_guideline_and_escaping_name_are_rejected(self):
        self.make_hf_cache()
        manifest = reader.build_manifest(self.root, batch_size=2)
        manifest["snapshot_guideline_files"] = {}
        with self.assertRaises(ValueError):
            list(reader.iter_method_tasks(self.root, manifest))
        directory = self.root / "train_val" / "guideline_md"
        for invalid in ("../escape", "/absolute", "sub\\name"):
            with self.assertRaises(ValueError):
                reader.safe_guideline_path(directory, invalid)

    def test_different_hf_snapshot_revision_is_rejected(self):
        self.make_hf_cache()
        other = self.root.with_name("0" * 40)
        self.root.rename(other)
        self.root = other
        with self.assertRaises(ValueError):
            reader.build_manifest(self.root, batch_size=2)

    def test_reference_fields_and_files_do_not_reach_method(self):
        opened = []
        real_open = Path.open

        def observe(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", observe):
            manifest = reader.build_manifest(self.root, batch_size=2)
            tasks = list(reader.iter_method_tasks(self.root, manifest))
        self.assertTrue(tasks)
        for identity, task in tasks:
            self.assertEqual(set(asdict(task)), {"guideline", "vignette", "language"})
            self.assertNotIn("SECRET_GOLD", json.dumps(asdict(task)))
        self.assertFalse(any("gt_paths" in path or "decision_trees" in path for path in opened))

    def test_test_manifest_and_split_root_rejected(self):
        with self.assertRaises(ValueError):
            reader.train_paths(self.root / "test")
        with self.assertRaises(ValueError):
            list(reader.iter_method_tasks(self.root, {"split": "test"}))

    def test_split_is_document_grouped_and_deterministic(self):
        first = reader.build_manifest(self.root, batch_size=2)
        second = reader.build_manifest(self.root, batch_size=2)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["groups"], second["groups"])
        self.assertEqual(sum(group["rows"] for group in first["groups"]), 12)
        self.assertTrue(all(group["rows"] == 2 for group in first["groups"]))
        development = {group["source_cluster"] for group in first["groups"]
                       if group["partition"] == "development"}
        self.assertTrue(all(row["source_cluster"] in development for row in first["selected"]))

    def test_input_change_after_freeze_rejected(self):
        manifest = reader.build_manifest(self.root, batch_size=2)
        selected = manifest["selected"][0]
        name = selected["identity"]["filename"]
        path = self.root / "train_val" / "guideline_md" / name / f"{name}.md"
        path.write_text("Changed guideline", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(reader.iter_method_tasks(self.root, manifest))


if __name__ == "__main__":
    unittest.main()
