"""Exact, read-only join to the frozen PR #12 A1 new72 artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json, write_jsonl


SIGNAL_MAP = {"T": 1, "F": -1, "U": 0, "technical_failure": None, "null": None}


def _prediction_state(prediction: dict[str, Any], rule_id: str) -> tuple[str | None, str]:
    if prediction.get("status") != "ok":
        return None, "technical_failure"
    for row in prediction.get("rules", []):
        if row.get("rule_id") == rule_id:
            state = row.get("state")
            if state in {"T", "F", "U"}:
                return state, state
    return None, "technical_failure"


def build(cases_path: str | Path, gold_path: str | Path, predictions_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    gold = {row["case_id"]: row for row in read_jsonl(gold_path)}
    predictions = {row["case_id"]: row for row in read_jsonl(predictions_path)}
    rows = []
    for case_id in sorted(cases):
        case, label, prediction = cases[case_id], gold.get(case_id), predictions.get(case_id)
        if label is None or prediction is None:
            raise ValueError(f"Missing exact A1 join for {case_id}")
        rule_id = case["target_rule_id"]
        state, status = _prediction_state(prediction, rule_id)
        action = label.get("gold_action")
        options = case.get("option_map", {})
        if not action or not options:
            raise ValueError(f"No downstream action/options for {case_id}")
        rows.append({"task": "N5", "case_id": case_id, "family_id": case["family_id"],
                     "underlying_mechanism_id": "frozen_A1_new72", "root_case_id": label.get("pair_id", case_id),
                     "pair_id": label.get("pair_id", case_id), "fact_group_id": case["family_id"], "split": "test",
                     "prompt_text": case["prompt"], "gold_answer": label["gold_answer"], "gold_action": action,
                     "a1_state": state, "a1_status": status, "a1_signal": SIGNAL_MAP[status],
                     "rule_id": rule_id, "option_map": options, "a1_method_version": "semantic_A1_frozen_PR12",
                     "case_sha256": hashlib.sha256(case["patient"].encode()).hexdigest(),
                     "rule_sha256": hashlib.sha256(case["guidance"].encode()).hexdigest(),
                     "source_commit": "5a4e47a4f71aeb845c5ebf52be14f9198a12f2b1",
                     "eligibility": True, "origin_dataset": "PR12_new72_frozen_A1"})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "inputs.jsonl", [{k: v for k, v in row.items() if not k.startswith("gold_")} for row in rows])
    write_jsonl(out_dir / "labels.jsonl", [{k: v for k, v in row.items() if k.startswith("gold_")} for row in rows])
    manifest = {"schema_version": "care.m2.n5.join.v1", "rows": len(rows),
                "source_commit": rows[0]["source_commit"] if rows else None,
                "source_files": {"cases": sha256_file(cases_path), "gold": sha256_file(gold_path), "predictions": sha256_file(predictions_path)},
                "status_counts": {status: sum(row["a1_status"] == status for row in rows) for status in ("T", "F", "U", "technical_failure")},
                "eligible_rows": sum(bool(row["eligibility"]) for row in rows),
                "selection_rule": "all PR12 new72 rows with exact case/rule/text hashes; no gold-based filtering"}
    write_json(out_dir / "manifest.json", manifest)
    return manifest
