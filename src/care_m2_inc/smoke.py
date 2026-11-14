from __future__ import annotations

from pathlib import Path

from .io import read_json, read_jsonl
from .leakcheck import audit
from .oracle import action_set, dq_label


def cpu_contract_smoke(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    leak = audit(data_dir)
    details = {"manifest": read_json(data_dir / "manifest.json"), "leakcheck": leak, "checks": {}}
    details["checks"]["manifest_present"] = (data_dir / "manifest.json").is_file()
    details["checks"]["all_jsonl_nonempty"] = all(path.stat().st_size > 0 for path in data_dir.glob("*/inputs.jsonl"))
    details["checks"]["n3_oracle_recompute"] = all(
        dq_label(row["rule_ast"], row["values"])[0] == row["gold_DQ"]
        for row in ({**i, **l} for i, l in zip(read_jsonl(data_dir / "N3/inputs.jsonl"), read_jsonl(data_dir / "N3/labels.jsonl")))
    )
    details["checks"]["n4_order_rows"] = len(read_jsonl(data_dir / "N4/inputs.jsonl")) % 16 == 0
    details["valid"] = bool(leak["valid"] and all(details["checks"].values()))
    return details
