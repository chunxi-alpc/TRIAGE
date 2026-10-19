#!/usr/bin/env python3
"""Reproduce preparation checks in one new directory; never load pretrained models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="A new directory; existing checks and model logs are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    py = sys.executable
    experiment = HERE.relative_to(ROOT)
    m1, m2, joint = [experiment / part for part in ("module1", "module2", "joint")]
    data = experiment / "data"
    tasks = [
        ("v6_default_interfaces", [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_module1_interfaces.py", "-v"]),
        ("module1_numeric_and_logic", [py, str(m1 / "run_cpu_diagnostics.py"), "--output", str(output / "module1_cpu")]),
        ("module1_cached_gate_replay", [py, str(m1 / "replay_cached_gates.py"), "--source-dir", "experiments/care_module1_v6/data", "--output", str(output / "module1_cached")]),
        ("module1_input_preparation", [py, str(m1 / "run_inference.py"), "--inputs", str(data / "mechanism_cases.jsonl"), "--version", "v7", "--dry-run", "--output", str(output / "module1_prepared")]),
        ("module2_reference_and_optional_torch_tests", [py, "-m", "unittest", "discover", "-s", str(m2), "-p", "test_*.py", "-v"]),
        ("joint_data_and_evaluator_tests", [py, "-m", "unittest", "discover", "-s", str(joint), "-p", "test_*.py", "-v"]),
        ("joint_full_data_audit", [py, str(joint / "audit_cpu.py"), "--output", str(output / "joint_data_audit.json")]),
    ]
    for mode in ("legacy384_regression", "sealed_mechanism"):
        for fold in range(3):
            command = [py, str(m2 / "run.py"), "--preflight", "--mode", mode, "--fold", str(fold),
                       "--output", str(output / f"module2_{mode}_fold{fold}")]
            if mode == "sealed_mechanism":
                command += ["--data", str(data / "module2_sealed.jsonl"), "--data-manifest", str(data / "manifest.json")]
            tasks.append((f"module2_{mode}_fold{fold}_preflight", command))
    tasks.append(("joint_model_preflight", [
        py, str(joint / "run_joint.py"), "--preflight-only", "--sources", "reference", "reference_positive", "v6", "v7",
        "--include-reference-diagnostic", "--learned-assets", str(m2 / "frozen_learned_assets.json"),
        "--output", str(output / "joint_preflight")]))
    sources = list(HERE.rglob("*.py")) + [ROOT / "src/module1/care_v7.py", ROOT / "src/module1/care_v6.py"]
    sources += [HERE / p for p in ("module1/protocol.json", "module2/protocol.json", "module2/frozen_learned_assets.json", "data/manifest.json")]
    packages = {}
    for name in ("numpy", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    summary = {
        "status": "running_preparation_checks", "started_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE_COMMIT, "python": platform.python_version(), "packages": packages,
        "source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(sources)},
        "pretrained_model_inference_performed": False,
        "interpretation": "Unit tests, synthetic mechanism checks, frozen-cache replay and asset preflight only. Passing these checks does not complete the model experiments.",
        "checks": [],
    }
    def write_record(path, value):
        # Immutable records also survive shared-workspace synchronization without
        # an earlier progress snapshot replacing the final completion record.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_record(output / "run_started.json", summary)
    records = output / "check_records"
    records.mkdir()
    for name, command in tasks:
        started = time.monotonic()
        record = {"name": name, "argv": command, "cwd": str(ROOT), "log": name + ".log"}
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=300, check=False)
            record.update(returncode=completed.returncode, passed=completed.returncode == 0)
            log = completed.stdout
        except subprocess.TimeoutExpired as error:
            raw = error.stdout or ""
            log = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
            log += "\nPreparation check timed out after 300 seconds.\n"
            record.update(returncode=None, passed=False, error="timeout")
        except OSError as error:
            log = f"{type(error).__name__}: {error}\n"
            record.update(returncode=None, passed=False, error=type(error).__name__)
        (output / record["log"]).write_text(log)
        match = re.search(r"Ran (\d+) tests? in", log)
        skipped = re.search(r"skipped=(\d+)", log)
        if match:
            record["unit_tests_run"] = int(match.group(1))
            record["unit_tests_skipped"] = int(skipped.group(1)) if skipped else 0
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["checks"].append(record)
        write_record(records / (name + ".json"), record)
        print(json.dumps({k: record[k] for k in ("name", "passed", "returncode", "elapsed_seconds")}), flush=True)
    summary["status"] = "preparation_checks_passed" if all(r["passed"] for r in summary["checks"]) else "preparation_checks_failed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    joint_status = output / "joint_preflight/execution_status.json"
    if joint_status.is_file():
        summary["real_model_runtime_preflight"] = json.loads(joint_status.read_text())
    write_record(output / "validation_summary.json", summary)
    return 0 if summary["status"] == "preparation_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
