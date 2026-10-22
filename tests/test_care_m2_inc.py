from __future__ import annotations

import json
from pathlib import Path

from care_m2_inc.generate import prepare
from care_m2_inc.leakcheck import audit
from care_m2_inc.oracle import action_set, decisive_attributes, dq_label, evaluate
from care_m2_inc.smoke import cpu_contract_smoke


def test_cpu_dataset_and_leakcheck(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manifest = prepare(data)
    assert manifest["tasks"]["N1"]["families"] == 24
    report = audit(data)
    assert report["valid"], report
    assert cpu_contract_smoke(data)["valid"]


def test_three_valued_exhaustion() -> None:
    ast = {"op": "AND", "args": ["a", {"op": "OR", "args": ["b", "c"]}]}
    values = {"a": True, "b": None, "c": False}
    assert evaluate(ast, values) is None
    assert action_set(ast, values) == {"A", "B"}
    assert decisive_attributes(ast, values) == ["b"]
    assert dq_label(ast, values)[0] == "Q"


def test_prompt_does_not_include_scorer_fields(tmp_path: Path) -> None:
    data = tmp_path / "data"
    prepare(data)
    forbidden = ("gold_", "control_target", "family_id", "root_case_id", "fact_group_id")
    for path in data.glob("*/inputs.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert not any(term in row.get("prompt_text", "") for term in forbidden)


def test_n1_four_cell_independence(tmp_path: Path) -> None:
    data = tmp_path / "data"
    prepare(data)
    labels = [json.loads(line) for line in (data / "N1/labels.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = [json.loads(line) for line in (data / "N1/splits.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [{**label, **split} for label, split in zip(labels, splits)]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    for group in grouped.values():
        assert {(row["gold_R"], row["gold_A"]) for row in group} == {(False, False), (False, True), (True, False), (True, True)}
        assert {row["gold_action"] for row in group if row["gold_R"]} == {"TARGET", "ALTERNATIVE"}
