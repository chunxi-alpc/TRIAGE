from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import write_jsonl


def write_case_reports(run_dir: str | Path, task: str, predictions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    run_dir = Path(run_dir)
    row_by_id = {row["case_id"]: row for row in rows}
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for item in predictions:
        row = row_by_id.get(item["case_id"], {})
        expected = item.get("gold_answer")
        correct = item.get("prediction") == expected
        grouped[(item["method"], row.get("family_id", item["case_id"]))].append(correct)
        if not correct:
            failures.append({"case_id": item["case_id"], "family_id": row.get("family_id"), "method": item["method"],
                             "prediction": item.get("prediction"), "gold_answer": expected, "control": item.get("control"),
                             "scores": item.get("scores"), "a1_signal": item.get("a1_signal")})
    with (run_dir / "per_family.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "family_id", "correct", "total", "accuracy"))
        for (method, family), values in sorted(grouped.items()):
            writer.writerow((method, family, sum(values), len(values), sum(values) / len(values)))
    write_jsonl(run_dir / "failure_cases.jsonl", failures)
