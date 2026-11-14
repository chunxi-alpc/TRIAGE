"""Fail-closed tests for the split-runtime structure transfer."""
import copy
import contextlib
import io
import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from joint.run_structures import request_key, collect_requests, StrictReplay, save, file_digest, implementation_hashes, digest, main as structures_main
from joint.saved_structures import SavedStructureProvider
from joint.context import prediction_input, prompt_sha256
from joint.structures import normalize_structure

ROOT = Path(__file__).resolve().parents[1]


class StructureTransferTests(unittest.TestCase):
    def case(self):
        return json.loads((ROOT / "data/mechanism_cases.jsonl").read_text().splitlines()[0])

    def test_request_identity_uses_exact_prompt_fields(self):
        atom = {"text": "Current age is at least 18", "numeric": {"value": 18}}
        base = request_key("The patient is currently 20.", atom)
        self.assertNotEqual(base, request_key("The patient is currently 21.", atom))
        self.assertNotEqual(base, request_key("The patient is currently 20.", {"text": atom["text"] + "."}))
        self.assertNotEqual(base, request_key("The patient is currently 20.", atom, "exclusion"))
        self.assertEqual(base, request_key("The patient is currently 20.", dict(atom, condition_id="other")))
        self.assertEqual(base, request_key("The patient is currently 20.", dict(atom, numeric={"value": 99})))

    def test_collector_excludes_reference_and_reuses_versions(self):
        case = self.case()
        expected, calls = collect_requests([case])
        changed = copy.deepcopy(case)
        changed["reference_structure"] = {"rules": "do not read"}
        changed["gold_answer"] = "not a model input"
        changed["reference_state"] = "not a model input"
        changed["target_answer"] = "not a model input"
        changed["structured_facts"] = {"gold": "do not read"}
        actual, _ = collect_requests([changed])
        self.assertEqual(expected, actual)
        self.assertTrue(all(r["uses"] for r in expected))
        self.assertEqual(calls, 2 * sum(len(r["uses"]) for r in expected))

    def test_strict_cache_miss_and_copy(self):
        key = request_key("patient", {"text": "criterion"})
        cache = StrictReplay({key: {"output": {"state": "T", "raw_output": "model bytes"}}})
        output = cache.judge("patient", {"text": "criterion"})
        output["state"] = "F"
        self.assertEqual(cache.judge("patient", {"text": "criterion"})["state"], "T")
        with self.assertRaises(KeyError):
            cache.judge("different patient", {"text": "criterion"})

    def fixture(self, directory):
        case = self.case()
        raw = {"rules": [{"rule_id": r["rule_id"], "applicability": "U",
                         "allow_attention_intervention": False, "conditions": [], "exceptions": []}
                        for r in case["rules"]]}
        row = {"case_id": case["case_id"],
               "input_sha256": prompt_sha256(json.dumps(prediction_input(case), sort_keys=True)),
               "normalized": normalize_structure(raw, case, "v6"), "raw": raw}
        rel = Path("structures/v6") / (case["case_id"] + ".json")
        save(directory / rel, row)
        (directory / "trialmatch_generations.jsonl").write_text("{}\n")
        manifest = {"status": "completed", "contract": {"data_sha256": "data", "split": "train",
                    "implementation_sha256": implementation_hashes(), "case_ids": [case["case_id"]],
                    "request_manifest_sha256": digest([])},
                    "generation_journal_sha256": file_digest(directory / "trialmatch_generations.jsonl"),
                    "active_adapters": ["default"], "structure_sha256": {str(rel): file_digest(directory / rel)},
                    "python_executable": "original", "versions": {"transformers": "4.57.1"}, "models": {}}
        save(directory / "request_manifest.json", [])
        save(directory / "immutable_provenance.json", {k: manifest[k] for k in ("contract", "versions", "models")})
        manifest["immutable_provenance_sha256"] = file_digest(directory / "immutable_provenance.json")
        save(directory / "run_manifest.json", manifest)
        save(directory / "COMPLETE.json", {"manifest_sha256": file_digest(directory / "run_manifest.json")})
        return case, rel

    def test_transfer_rejects_input_and_artifact_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            case, rel = self.fixture(directory)
            provider = SavedStructureProvider("v6", [directory], "data", "train")
            self.assertEqual(provider.predict(case)[0]["source"], "v6")
            changed = copy.deepcopy(case)
            changed["patient"] += " changed"
            with self.assertRaisesRegex(ValueError, "input mismatch"):
                provider.predict(changed)
            with (directory / rel).open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "file changed"):
                provider.predict(case)

    def test_transfer_rejects_source_and_data_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "data or split"):
                SavedStructureProvider("v6", [directory], "different", "train")
            with patch("joint.saved_structures.implementation_hashes", return_value={"changed": "source"}):
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    SavedStructureProvider("v6", [directory], "data", "train")

    def test_repeated_resume_refusal_cannot_replace_original_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prepared"
            argv = ["run_structures.py", "--split", "train", "--case-limit", "1",
                    "--output", str(output), "--base-model", tmp, "--adapter-model", tmp,
                    "--evidence-model", tmp]
            with patch.object(sys, "argv", argv + ["--dry-run"]), contextlib.redirect_stdout(io.StringIO()):
                structures_main()
            manifest = json.loads((output / "run_manifest.json").read_text())
            original = {"contract": manifest["contract"], "versions": {"transformers": "original"},
                        "models": {k: {"path": tmp} for k in ("base_model", "adapter_model", "evidence_model")}}
            save(output / "immutable_provenance.json", original)
            before = (output / "run_manifest.json").read_bytes()
            for _ in range(2):
                with patch.object(sys, "argv", argv + ["--resume"]), \
                     patch("joint.run_structures.package_versions", return_value={"transformers": "changed"}), \
                     patch("joint.run_structures.model_manifest", return_value={"path": tmp}):
                    with self.assertRaisesRegex(ValueError, "Resume rejected"):
                        structures_main()
                self.assertEqual((output / "run_manifest.json").read_bytes(), before)
                self.assertEqual(json.loads((output / "immutable_provenance.json").read_text()), original)

    def test_transfer_rejects_duplicate_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "more than one"):
                SavedStructureProvider("v6", [directory, directory], "data", "train")


if __name__ == "__main__":
    unittest.main()

