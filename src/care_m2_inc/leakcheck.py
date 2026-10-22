from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, write_json
from .oracle import action_set, dq_label, evaluate

FORBIDDEN_PROMPT_TERMS = ("gold_", "control_target", "family_id", "root_case_id", "fact_group_id", "oracle", "donor_gold")


def _group_overlap(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    by_split: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row.get(field))
    overlaps = []
    for left, right in itertools.combinations(sorted(by_split), 2):
        common = sorted(by_split[left] & by_split[right])
        if common:
            overlaps.append({"field": field, "left": left, "right": right, "values": common})
    return overlaps


def audit(data_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    report: dict[str, Any] = {"schema_version": "care.m2.leakcheck.v1", "manifest": manifest, "tasks": {}, "violations": []}
    for task in manifest["tasks"]:
        inputs = read_jsonl(data_dir / task / "inputs.jsonl")
        labels = read_jsonl(data_dir / task / "labels.jsonl")
        splits = read_jsonl(data_dir / task / "splits.jsonl")
        rows = [{**i, **l, **s} for i, l, s in zip(inputs, labels, splits)]
        task_report = {"rows": len(rows), "families": len({r["family_id"] for r in rows}), "overlaps": [], "checks": {}}
        for field in ("family_id", "underlying_mechanism_id", "root_case_id", "pair_id", "fact_group_id"):
            task_report["overlaps"].extend(_group_overlap(rows, field))
        for row in rows:
            prompt = row.get("prompt_text", "")
            for term in FORBIDDEN_PROMPT_TERMS:
                if term in prompt:
                    report["violations"].append({"task": task, "case_id": row["case_id"], "type": "prompt_forbidden_term", "term": term})
        if task == "N1":
            groups = defaultdict(list)
            for row in rows:
                groups[row["pair_id"]].append(row)
            task_report["checks"]["four_cell_blocks"] = all({(r["gold_R"], r["gold_A"]) for r in group} == set(itertools.product((False, True), repeat=2)) for group in groups.values())
            task_report["checks"]["r_a_cell_balance"] = all(sum(bool(r["gold_R"]) for r in group) == 2 and sum(bool(r["gold_A"]) for r in group) == 2 for group in groups.values())
        elif task == "N2":
            by_receiver = defaultdict(list)
            for row in rows:
                by_receiver[row["receiver_id"]].append(row)
            task_report["checks"]["neutral_receiver_prompt"] = all(len({r["prompt_text"] for r in group}) == 1 for group in by_receiver.values())
            task_report["checks"]["receiver_donor_separation"] = all(r["fact_group_id"] != next((x["fact_group_id"] for x in rows if x["case_id"] == r["donor_id"]), None) for r in rows)
            task_report["checks"]["synthetic_p_track_labeled"] = all(r.get("track") == "synthetic_prior" and not r.get("p_verified") for r in rows)
        elif task == "N3":
            ok = True
            for row in rows:
                label, possible, attrs = dq_label(row["rule_ast"], row["values"])
                ok = ok and label == row["gold_DQ"] and possible == set(row["possible_actions"]) and attrs == row["valid_question_attributes"]
            task_report["checks"]["exhaustive_dq"] = ok
            task_report["checks"]["both_dq"] = {row["gold_DQ"] for row in rows} == {"D", "Q"}
        elif task == "N4":
            groups = defaultdict(list)
            for row in rows:
                groups[row["pair_id"]].append(row)
            task_report["checks"]["sixteen_cell_blocks"] = all(len(group) == 16 and len({(r["control_target"]["source"], r["control_target"]["DQ"], r["control_target"]["A"], r["control_target"]["order"]) for r in group}) == 16 for group in groups.values())
            task_report["checks"]["order_invariant"] = all(len({r["gold_answer"] for r in group if r["rule_order"] == "original"}) == len({r["gold_answer"] for r in group if r["rule_order"] == "reversed"}) for group in groups.values())
        task_report["valid"] = not task_report["overlaps"] and all(task_report["checks"].values())
        report["tasks"][task] = task_report
        report["violations"].extend(task_report["overlaps"])
    report["valid"] = not report["violations"] and all(item["valid"] for item in report["tasks"].values())
    if out:
        write_json(out, report)
    return report
