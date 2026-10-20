#!/usr/bin/env python3
"""Run and record only numerical/data-contract tests; never a model experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error"})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "skipped", "reason": reason})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "results" / "verification.json")
    args = p.parse_args()
    out = args.out.resolve()
    log = out.with_suffix(".log")
    if out.exists() or log.exists():
        p.error("refusing to overwrite verification evidence; select a new --out")
    stream = io.StringIO()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(suite)
    report = {
        "schema": "care.module2.local_verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_mathematical_contract_checks" if result.wasSuccessful() else "failed",
        "new_model_forwards": 0,
        "actual_model_P0_passed": False,
        "neural_efficacy_test": False,
        "model_or_module1_modified": False,
        "python": platform.python_version(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_seconds": time.perf_counter() - started,
        "tests": result.records,
        "source_sha256": {
            str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(HERE.rglob("*.py"))
        },
        "limits": [
            "No Transformer hooks, real activation tensors, BF16 kernels or KV caches are executed.",
            "A coordinate swap can still copy answer content encoded in the exchanged subspace.",
            "Declared split IDs are checked; semantic-family truth requires an independent content audit.",
            "Frozen-cache ID joins do not replace raw-input hash and module-version validation.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    with log.open("x", encoding="utf-8") as f:
        f.write(stream.getvalue())
    print(stream.getvalue(), end="")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "failures", "errors", "new_model_forwards")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
