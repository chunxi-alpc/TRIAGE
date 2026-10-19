"""CPU-only routing checks; no model loading, downloads, or experiment outputs."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import src.module1 as module1
from src.module1 import care_v6
from src.module1.__main__ import main


class Module1InterfaceTests(unittest.TestCase):
    def tearDown(self):
        module1.set_default_v6(None)

    def test_factory_reads_every_model_environment_setting(self):
        settings = {"DEVICE": "cuda:7", "CACHE": "/test/cache", "BASE_MODEL": "/test/phi-4",
                    "ADAPTER_MODEL": "/test/lora", "EVIDENCE_MODEL": "/test/mednli"}
        with patch.dict(os.environ, {"CARE_MODULE1_V6_" + k: v for k, v in settings.items()}), \
                patch.object(care_v6.CAREModule1V6, "from_pretrained") as load:
            self.assertIs(module1.create_module1_v6(), load.return_value)
        load.assert_called_once_with(device="cuda:7", cache_dir="/test/cache", base_model="/test/phi-4",
            adapter_model="/test/lora", evidence_model="/test/mednli", nli_batch_size=32, candidate_limit=8)

    def test_explicit_factory_arguments_override_environment(self):
        settings = {"CARE_MODULE1_V6_" + key: "environment-value" for key in
                    ("DEVICE", "CACHE", "BASE_MODEL", "ADAPTER_MODEL", "EVIDENCE_MODEL")}
        explicit = dict(device="cuda:2", cache_dir="/explicit/cache", base_model="/explicit/base",
                        adapter_model="/explicit/adapter", evidence_model="/explicit/evidence",
                        nli_batch_size=4, candidate_limit=3)
        with patch.dict(os.environ, settings), patch.object(care_v6.CAREModule1V6, "from_pretrained") as load:
            module1.create_module1_v6(**explicit)
        load.assert_called_once_with(**explicit)

    def test_package_run_creates_and_reuses_one_default_engine(self):
        module1.set_default_v6(None)
        engine = Mock()
        engine.run.return_value = {"schema_version": "module1.v6"}
        with patch.object(care_v6, "create_module1_v6", return_value=engine) as factory:
            for _ in range(2):
                self.assertEqual(module1.run([], "note", "Assess"), engine.run.return_value)
        factory.assert_called_once_with()
        self.assertEqual(engine.run.call_count, 2)

    def _cli_paths(self, folder):
        sample = {"guideline": [{"rule_id": "R1", "text": "Signal A", "conditions": [
                  {"condition_id": "c1", "text": "Signal A"}]}], "patient": "Signal A.", "question": "Assess"}
        input_path, output_path = Path(folder) / "input.json", Path(folder) / "output.json"
        input_path.write_text(json.dumps(sample), encoding="utf-8")
        return sample, output_path, ["--input", str(input_path), "--output", str(output_path)]

    def test_cli_defaults_to_v6_and_writes_the_graph(self):
        engine = Mock()
        engine.run.return_value = {"schema_version": "module1.v6"}
        with tempfile.TemporaryDirectory() as folder:
            sample, output, args = self._cli_paths(folder)
            with patch("src.module1.__main__.create_module1_v6", return_value=engine) as factory:
                main(args + ["--device", "cuda:6", "--base-model", "/explicit/phi-4"])
            factory.assert_called_once_with(device="cuda:6", cache_dir=None, base_model="/explicit/phi-4",
                                            adapter_model=None, evidence_model=None)
            engine.run.assert_called_once_with(**sample)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), engine.run.return_value)

    def test_cli_legacy_is_explicit_and_keeps_its_retrieval_option(self):
        engine = Mock()
        engine.run.return_value = {"schema_version": "module1.v1"}
        with tempfile.TemporaryDirectory() as folder:
            sample, output, args = self._cli_paths(folder)
            with patch("src.module1.modeler.Module1", return_value=engine) as legacy, \
                    patch("src.module1.__main__.create_module1_v6") as v6:
                main(args + ["--legacy", "--top-k", "5", "--device", "cpu"])
            legacy.assert_called_once_with(top_k=5, device="cpu")
            v6.assert_not_called()
            engine.run.assert_called_once_with(**sample)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], "module1.v1")

    def test_cli_rejects_silently_ignored_legacy_option(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, args = self._cli_paths(folder)
            with patch("src.module1.__main__.create_module1_v6") as factory, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args + ["--top-k", "5"])
            self.assertEqual(error.exception.code, 2)
            factory.assert_not_called()

    def test_invalid_v6_inputs_stop_before_package_or_cli_factory(self):
        invalid = (
            {"guideline": "free text"},
            {"guideline": {"rules": "bad"}},
            {"guideline": [{"rule_id": "R1", "text": "Missing conditions"}]},
            {"guideline": [{"conditions": [None]}]},
            {"patient": 42},
            {"patient": {"unexpected": "note"}},
            {"question": " "},
        )
        module1.set_default_v6(None)
        with tempfile.TemporaryDirectory() as folder:
            valid, _, args = self._cli_paths(folder)
            source = Path(folder) / "input.json"
            for changes in invalid:
                with self.subTest(changes=changes):
                    item = {**valid, **changes}
                    with patch.object(care_v6, "create_module1_v6") as factory:
                        with self.assertRaises(ValueError):
                            module1.run(**item)
                        factory.assert_not_called()
                    source.write_text(json.dumps(item), encoding="utf-8")
                    with patch("src.module1.__main__.create_module1_v6") as factory, \
                            contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            main(args)
                        self.assertEqual(error.exception.code, 2)
                        factory.assert_not_called()

    def test_validator_preserves_patient_types_and_does_not_mutate_rules(self):
        rules = [{"text": "Signal A", "conditions": [{"text": "Signal A"}]}]
        before = copy.deepcopy(rules)
        patients = ("Signal A.", ["Signal A."], ("Signal A.",), {"text": "Signal A."},
                    {"sentences": ["Signal A."]}, "", [])
        for patient in patients:
            with self.subTest(patient=patient):
                self.assertIsNone(module1.validate_v6_input(rules, patient, "Assess"))
                self.assertEqual(rules, before)
        example = json.loads((Path(__file__).resolve().parents[1] /
                              "examples/module1_v6_input.json").read_text(encoding="utf-8"))
        module1.validate_v6_input(example["guideline"], example["patient"], example["question"])


if __name__ == "__main__":
    unittest.main()
