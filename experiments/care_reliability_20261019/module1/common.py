"""Small CPU-safe loader and provenance helpers for the independent V7 study."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASELINE_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"


def load_modules():
    """Avoid importing the default/legacy package entry point for CPU studies."""
    package_name = "_care_reliability_module1"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "src/module1")]
        sys.modules[package_name] = package
    result = []
    for name in ("care_v6", "care_v7"):
        qualified = package_name + "." + name
        if qualified not in sys.modules:
            spec = importlib.util.spec_from_file_location(qualified, ROOT / "src/module1" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            setattr(sys.modules[package_name], name, module)
        result.append(sys.modules[qualified])
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "src/module1/care_v6.py", ROOT / "src/module1/care_v7.py"]}
