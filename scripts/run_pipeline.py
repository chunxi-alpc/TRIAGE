#!/usr/bin/env python3
"""Run V6 rule analysis and Qwen inference with an explicit local configuration."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pipeline_minimal.json"
CONFIG_SCHEMA = "triage_pipeline_config.v1"
CONFIG_FIELDS = {
    "components_path", "projection_path", "device", "max_new_tokens",
    "module1_device", "module1_cache",
}
sys.path.insert(0, str(ROOT))
from src.module1 import validate_v6_input


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON configuration; relative paths inside it use the repository root")
    parser.add_argument("--components", type=Path, help="Override the component artifact path")
    parser.add_argument("--projections", type=Path, help="Override the projection artifact path")
    parser.add_argument("--device", help="Override the Qwen generation device")
    parser.add_argument("--module1-device", help="Override the V6 device and CARE_MODULE1_V6_DEVICE")
    parser.add_argument("--module1-cache", type=Path, help="Override the V6 model cache directory")
    parser.add_argument("--max-new-tokens", type=positive_int, help="Override the generation token limit")
    parser.add_argument("--top-k", type=positive_int,
                        help="Deprecated compatibility option; V6 evaluates all supplied rules")
    return parser


def resolve_settings(args):
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Configuration schema_version must be {CONFIG_SCHEMA}")
    unknown = set(config) - CONFIG_FIELDS - {"schema_version"}
    if unknown:
        raise ValueError("Unsupported configuration fields: " + ", ".join(sorted(unknown)))
    settings = {
        "components_path": "results/module2/components.json",
        "projection_path": "results/module3/projections.pt",
        "device": "cuda:0",
        "max_new_tokens": 1284,
        "module1_device": None,
        "module1_cache": None,
    }
    settings.update({key: value for key, value in config.items() if key != "schema_version"})
    for key in ("components_path", "projection_path", "module1_cache"):
        value = settings[key]
        if value is None and key == "module1_cache":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration {key} must be a nonempty path string")
        path = Path(value).expanduser()
        settings[key] = path if path.is_absolute() else ROOT / path
    for key, argument in (("components_path", "components"), ("projection_path", "projections"),
                          ("device", "device"), ("max_new_tokens", "max_new_tokens"),
                          ("module1_device", "module1_device"), ("module1_cache", "module1_cache")):
        value = getattr(args, argument)
        if value is not None:
            settings[key] = value.expanduser() if isinstance(value, Path) else value
    for key in ("device", "module1_device"):
        value = settings[key]
        if value is None and key == "module1_device":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty device string")
    limit = settings["max_new_tokens"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if args.top_k is not None:
        settings["top_k"] = args.top_k
    return settings


def validate_input(item):
    if not isinstance(item, dict):
        raise ValueError("Input must be a JSON object")
    missing = {"guideline", "patient", "question"} - set(item)
    if missing:
        raise ValueError("Missing input keys: " + ", ".join(sorted(missing)))
    validate_v6_input(item["guideline"], item["patient"], item["question"])
    if not isinstance(item["patient"], str) or not item["patient"].strip():
        raise ValueError("Pipeline patient must be a nonempty string")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = resolve_settings(args)
        item = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
        validate_input(item)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    from src.pipeline import TriagePipeline
    pipe = TriagePipeline(**settings)
    result = pipe.run(item["guideline"], item["patient"], item["question"],
                      input_source=item.get("input_source"))
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["baseline"]["json_valid"] or not result["intervened"]["json_valid"]:
        raise AssertionError("Pipeline decision JSON is incomplete; output text preserved in report")
    print("Saved " + str(output), flush=True)


if __name__ == "__main__":
    main()
