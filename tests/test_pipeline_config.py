"""Standard-library interface regressions; no model loading or frozen-result writes."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_pipeline
from src.module1.care_v6 import CAREModule1V6
from src.pipeline import TriagePipeline


class PipelineConfigTests(unittest.TestCase):
    def args(self, *extra):
        return run_pipeline.build_parser().parse_args([
            "--input", "input.json", "--output", "output.json", *extra,
        ])

    def write_config(self, folder, **values):
        path = Path(folder) / "pipeline.json"
        path.write_text(json.dumps({"schema_version": run_pipeline.CONFIG_SCHEMA, **values}), encoding="utf-8")
        return path

    def test_default_config_preserves_cli_behavior_and_uses_repo_paths(self):
        settings = run_pipeline.resolve_settings(self.args())
        self.assertEqual(settings["components_path"], run_pipeline.ROOT / "results/module2/components.json")
        self.assertEqual(settings["projection_path"], run_pipeline.ROOT / "results/module3/projections.pt")
        self.assertEqual(settings["max_new_tokens"], 1284)
        self.assertEqual(settings["device"], "cuda:0")
        self.assertNotIn("top_k", settings)

    def test_cli_overrides_config_and_config_paths_use_repo_root(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.write_config(folder, device="cuda:6", max_new_tokens=384,
                                       components_path="calibration/heads.json", module1_cache="model_cache")
            args = self.args("--config", str(config), "--device", "cuda:1",
                             "--max-new-tokens", "512", "--projections", "other/projections.pt")
            settings = run_pipeline.resolve_settings(args)
        self.assertEqual(settings["device"], "cuda:1")
        self.assertEqual(settings["max_new_tokens"], 512)
        self.assertEqual(settings["components_path"], run_pipeline.ROOT / "calibration/heads.json")
        self.assertEqual(settings["module1_cache"], run_pipeline.ROOT / "model_cache")
        self.assertEqual(settings["projection_path"], Path("other/projections.pt"))

    def test_configuration_rejects_unused_fields_and_invalid_token_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.write_config(folder, top_k=3)
            with self.assertRaisesRegex(ValueError, "Unsupported.*top_k"):
                run_pipeline.resolve_settings(self.args("--config", str(config)))
            for invalid in (0, -1, True, "384", 1.5):
                with self.subTest(limit=invalid):
                    config = self.write_config(folder, max_new_tokens=invalid)
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        run_pipeline.resolve_settings(self.args("--config", str(config)))

    def test_invalid_input_stops_before_pipeline_creation(self):
        invalid = (
            {"guideline": "free text"},
            {"guideline": {"rules": "bad"}},
            {"guideline": [{"rule_id": "R1", "text": "Missing conditions"}]},
            {"guideline": [{"conditions": [None]}]},
            {"patient": 42},
            {"patient": ["patient"]},
            {"patient": " "},
            {"question": " "},
        )
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.json"
            valid = {"guideline": {"rules": []}, "patient": "patient", "question": "question"}
            for changes in invalid:
                with self.subTest(changes=changes):
                    source.write_text(json.dumps({**valid, **changes}), encoding="utf-8")
                    with patch("src.pipeline.TriagePipeline") as factory, contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            run_pipeline.main(["--input", str(source), "--output", str(Path(folder) / "output.json")])
                        self.assertEqual(error.exception.code, 2)
                        factory.assert_not_called()

    def test_documented_example_passes_input_validation(self):
        example = json.loads((run_pipeline.ROOT / "examples/module1_v6_input.json").read_text(encoding="utf-8"))
        run_pipeline.validate_input(example)

    def test_cli_passes_settings_and_preserves_output_before_json_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.json"
            output = Path(folder) / "new/output.json"
            source.write_text(json.dumps({"guideline": {"rules": []}, "patient": "patient", "question": "question"}))
            report = {"baseline": {"json_valid": True}, "intervened": {"json_valid": False, "text": "partial"}}
            with patch("src.pipeline.TriagePipeline") as factory:
                factory.return_value.run.return_value = report
                with self.assertRaisesRegex(AssertionError, "output text preserved"):
                    run_pipeline.main(["--input", str(source), "--output", str(output),
                                       "--module1-device", "cuda:7", "--max-new-tokens", "64"])
                self.assertEqual(factory.call_args.kwargs["module1_device"], "cuda:7")
                self.assertEqual(factory.call_args.kwargs["max_new_tokens"], 64)
                self.assertEqual(json.loads(output.read_text()), report)


class PipelineFactoryTests(unittest.TestCase):
    def test_runner_uses_shared_v6_factory_with_all_model_environment_paths(self):
        values = {
            "CARE_MODULE1_V6_DEVICE": "cuda:6",
            "CARE_MODULE1_V6_CACHE": "/tmp/triage-v6-cache",
            "CARE_MODULE1_V6_BASE_MODEL": "/models/phi4",
            "CARE_MODULE1_V6_ADAPTER_MODEL": "/models/trialmatch",
            "CARE_MODULE1_V6_EVIDENCE_MODEL": "/models/mednli",
        }
        with patch.dict(os.environ, values), patch.object(CAREModule1V6, "from_pretrained") as factory:
            pipe = TriagePipeline(backend=object(), device="cuda:0")
            self.assertIs(pipe._get_module1(), pipe._get_module1())
            factory.assert_called_once()
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["device"], "cuda:6")
            self.assertEqual(kwargs["cache_dir"], "/tmp/triage-v6-cache")
            self.assertEqual(kwargs["base_model"], "/models/phi4")
            self.assertEqual(kwargs["adapter_model"], "/models/trialmatch")
            self.assertEqual(kwargs["evidence_model"], "/models/mednli")
            explicit = TriagePipeline(backend=object(), module1_device="cuda:7")
            explicit._get_module1()
            self.assertEqual(factory.call_args.kwargs["device"], "cuda:7")

    def test_legacy_graph_is_rejected_before_backend_or_torch_import(self):
        pipe = TriagePipeline(backend=object())
        with patch.object(pipe, "load") as load:
            with self.assertRaisesRegex(ValueError, "module1.v6"):
                pipe.run([], "patient", "question", module1_output={"schema_version": "module1.v1"})
            load.assert_not_called()

    def test_live_runner_rejects_invalid_inputs_before_loading(self):
        valid = {"guideline": [], "patient": "patient", "question": "question"}
        invalid = (
            {"guideline": "free text"},
            {"guideline": {"rules": "bad"}},
            {"guideline": [{"rule_id": "R1", "text": "Missing conditions"}]},
            {"guideline": [{"conditions": [42]}]},
            {"patient": None},
            {"patient": ["patient"]},
            {"patient": " "},
            {"question": " "},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                pipe = TriagePipeline(backend=object())
                with patch.object(pipe, "load") as load, \
                        patch("src.pipeline.runner.create_module1_v6") as factory:
                    with self.assertRaises(ValueError):
                        pipe.run(**{**valid, **changes})
                    load.assert_not_called()
                    factory.assert_not_called()

    def test_empty_environment_values_use_generation_device_and_default_cache(self):
        values = {"CARE_MODULE1_V6_DEVICE": "", "CARE_MODULE1_V6_CACHE": ""}
        with patch.dict(os.environ, values), patch.object(CAREModule1V6, "from_pretrained") as factory:
            pipe = TriagePipeline(backend=object(), device="cuda:2")
            pipe._get_module1()
            self.assertEqual(factory.call_args.kwargs["device"], "cuda:2")
            self.assertIsNone(factory.call_args.kwargs["cache_dir"])
            explicit = TriagePipeline(backend=object(), device="cuda:2", module1_device="cuda:7",
                                      module1_cache="/explicit/cache")
            explicit._get_module1()
            self.assertEqual(factory.call_args.kwargs["device"], "cuda:7")
            self.assertEqual(factory.call_args.kwargs["cache_dir"], "/explicit/cache")

    def test_legacy_top_k_is_explicitly_marked_ineffective(self):
        with self.assertWarnsRegex(FutureWarning, "no effect on V6"):
            TriagePipeline(backend=object(), top_k=3)


if __name__ == "__main__":
    unittest.main()
