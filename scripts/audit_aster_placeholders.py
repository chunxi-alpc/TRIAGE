#!/usr/bin/env python3
"""Audit arithmetic in the supplied ASTER placeholder manuscript.

Uses only the Python standard library. Reads the source without changing it.
This checks displayed placeholder values, never experiment validity.

Examples:
    python scripts/audit_aster_placeholders.py
    python scripts/audit_aster_placeholders.py manuscript.txt --output audit.json

Exit status is 0 when the report is produced, including when it lists issues;
2 indicates an input, parsing, or output error. No model or benchmark is run.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import re
import sys


ONE_DECIMAL = Decimal("0.1")
HUNDRED = Decimal("100")
DEFAULT_SOURCE = "docs/ASTER/sources/manuscript-20261026.txt"


def clean_cell(value: str) -> str:
    return value.strip().replace("**", "").replace("−", "-")


def read_table(text: str, number: int) -> tuple[list[str], list[list[str]]]:
    """Read the first contiguous Markdown table after an exact table heading."""
    heading = re.search(rf"(?m)^\*\*表\s*{number}(?:\s|[：:])", text)
    if heading is None:
        raise ValueError(f"找不到表 {number} 的标题")
    remaining = text[heading.end():].splitlines()
    rows: list[list[str]] = []
    for line in remaining:
        if line.strip().startswith("|"):
            cells = [clean_cell(cell) for cell in line.strip().strip("|").split("|")]
            rows.append(cells)
        elif rows:
            break
    if len(rows) < 3:
        raise ValueError(f"表 {number} 缺少表头、分隔行或数据行")
    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        raise ValueError(f"表 {number} 的 Markdown 分隔行无法解析")
    header = rows[0]
    for index, row in enumerate(rows[2:], start=1):
        if len(row) != len(header):
            raise ValueError(f"表 {number} 第 {index} 行列数与表头不一致")
    return header, rows[2:]


def number(value: str) -> Decimal:
    result = Decimal(clean_cell(value).removesuffix("%"))
    if not result.is_finite():
        raise ValueError(f"数值必须有限：{value!r}")
    return result


def round_one(value: Decimal) -> str:
    return str(value.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def harmonic_mean(a: Decimal, b: Decimal) -> Decimal:
    return Decimal(0) if a + b == 0 else 2 * a * b / (a + b)


def require_columns(header: list[str], columns: list[str], table_number: int) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"表 {table_number} 缺少列：{', '.join(missing)}")


def metric_check(displayed: Decimal, recomputed: Decimal) -> dict:
    rounded = round_one(recomputed)
    return {
        "displayed": str(displayed),
        "recomputed_from_displayed_inputs": str(recomputed),
        "recomputed_rounded": rounded,
        "consistent_at_display_precision": displayed == Decimal(rounded),
    }


def audit(text: str, raw: bytes, source_name: str, denominator: int) -> dict:
    h9, rows9 = read_table(text, 9)
    h13, rows13 = read_table(text, 13)
    h12, rows12 = read_table(text, 12)
    h3, rows3 = read_table(text, 3)
    require_columns(h9, ["方法", "NC", "IC", "AC", "SW", "CS", "NCA", "CA", "H"], 9)
    require_columns(h13, ["编号", "NCA", "CA", "H"], 13)
    require_columns(h12, ["模型", "ASTER-full", "Clerk-Prune"], 12)
    require_columns(h3, ["NC", "IC", "AC", "SW", "CS"], 3)

    table9 = []
    for row in rows9:
        values = dict(zip(h9, row))
        nca = (number(values["NC"]) + number(values["IC"])) / 2
        ca = sum(number(values[column]) for column in ("AC", "SW", "CS")) / 3
        metrics = {
            name: metric_check(number(values[name]), expected)
            for name, expected in (("NCA", nca), ("CA", ca), ("H", harmonic_mean(nca, ca)))
        }
        table9.append({
            "method": values["方法"],
            "metrics": metrics,
            "all_consistent": all(item["consistent_at_display_precision"] for item in metrics.values()),
        })

    table13 = []
    for row in rows13:
        values = dict(zip(h13, row))
        check = metric_check(
            number(values["H"]),
            harmonic_mean(number(values["NCA"]), number(values["CA"])),
        )
        table13.append({"ablation": values["编号"], "H": check})

    # Table 12 has repeated method headings for different task groups.
    # The final ASTER-full column is the generation-task BFR column.
    full_indices = [index for index, name in enumerate(h12) if name == "ASTER-full"]
    if len(full_indices) != 2:
        raise ValueError("表 12 应包含选择题和路径生成两组 ASTER-full 列")
    full_index = full_indices[-1]
    clerk_index = h12.index("Clerk-Prune")
    if clerk_index < full_indices[0]:
        raise ValueError("表 12 的 Clerk-Prune 列不在路径生成组")
    differences = [
        {"model": row[h12.index("模型")], "difference_percentage_points": str(number(row[full_index]) - number(row[clerk_index]))}
        for row in rows12
    ]
    if not differences:
        raise ValueError("表 12 没有可比较的模型")
    deltas = [Decimal(row["difference_percentage_points"]) for row in differences]
    computed_range = [str(min(deltas)), str(max(deltas))]
    # Audit only the specific cross-model prose, not unrelated numeric ranges.
    range_match = re.search(
        r"在路径生成上，ASTER\s*比最强的同信息基线高\s*([0-9.]+)\s*[–—-]\s*([0-9.]+)\s*分",
        text,
    )
    stated_range = list(range_match.groups()) if range_match else None
    range_consistent = (
        [number(value) for value in stated_range] == [min(deltas), max(deltas)]
        if stated_range else None
    )

    blind_rows = [row for row in rows3 if row[0].startswith("盲判")]
    map_rows = [row for row in rows9 if row[h9.index("方法")] == "TS-Map"]
    if len(blind_rows) != 1 or len(map_rows) != 1:
        raise ValueError("表 3 必须有一行盲判，表 9 必须有一行 TS-Map")
    blind_map = []
    for condition in ("NC", "IC", "AC", "SW", "CS"):
        blind = number(blind_rows[0][h3.index(condition)])
        mapped = number(map_rows[0][h9.index(condition)])
        blind_map.append({
            "condition": condition,
            "blind_branch_accuracy": str(blind),
            "TS_Map_answer_accuracy": str(mapped),
            "TS_Map_minus_blind_percentage_points": str(mapped - blind),
            "equal_displayed_values": mapped == blind,
        })

    possible: dict[str, list[int]] = {}
    for correct in range(denominator + 1):
        displayed = round_one(HUNDRED * Decimal(correct) / Decimal(denominator))
        possible.setdefault(displayed, []).append(correct)
    feasibility = []
    for row in rows9:
        for condition in ("NC", "IC", "AC", "SW", "CS"):
            rate = number(row[h9.index(condition)])
            counts = possible.get(round_one(rate), [])
            feasibility.append({
                "method": row[h9.index("方法")],
                "condition": condition,
                "displayed_accuracy_percent": str(rate),
                "feasible_correct_counts": counts,
                "feasible": bool(counts),
            })

    passed9 = sum(row["all_consistent"] for row in table9)
    passed13 = sum(row["H"]["consistent_at_display_precision"] for row in table13)
    return {
        "schema_version": 1,
        "source": source_name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_status": "hypothetical_placeholder_manuscript",
        "experiments_run": False,
        "notes": [
            "仅审计原稿中显示的占位数值；不验证实测结果、统计显著性或数据真实性。",
            "Decimal 从字符串读取，最终显示按 ROUND_HALF_UP 保留一位小数。",
            "表 9 用已显示的分卷准确率重算 NCA、CA、H，不先舍入中间指标；表 13 用已显示的 NCA、CA 重算 H。",
            "输入已舍入，检查一致不等于未舍入原始计数一致；最终结果仍须从锁定测试集原始计数计算。",
            "分母可实现性检查假定单次二元逐例准确率及指定固定分母；若采用宏平均、多次平均或不同分母，需另行说明。",
            "盲判与 TS-Map 相等仅在同一病例、同一盲判及确定性无损分支到选项映射的条件下应成立。",
            "表 12 的比较固定为 ASTER-full 减 Clerk-Prune；正文若采用其他比较集合，应另行核对。",
        ],
        "summary": {
            "table9_row_count": len(rows9),
            "table9_expected_row_count": 22,
            "table9_row_count_matches": len(rows9) == 22,
            "table9_rows_arithmetically_consistent": passed9,
            "table13_row_count": len(rows13),
            "table13_expected_row_count": 14,
            "table13_row_count_matches": len(rows13) == 14,
            "table13_rows_arithmetically_consistent": passed13,
            "table12_stated_range_matches_full_minus_clerk_prune": range_consistent,
            "blind_vs_map_unequal_conditions": sum(not row["equal_displayed_values"] for row in blind_map),
            "denominator": denominator,
            "denominator_accuracy_cells_checked": len(feasibility),
            "denominator_infeasible_cells": sum(not row["feasible"] for row in feasibility),
        },
        "table9_derived_metrics": table9,
        "table13_harmonic_mean": table13,
        "table12_generation_full_minus_clerk_prune": {
            "per_model": differences,
            "computed_range_percentage_points": computed_range,
            "stated_range_percentage_points": stated_range,
            "stated_range_consistent": range_consistent,
        },
        "table3_blind_vs_table9_TS_Map": blind_map,
        "table9_denominator_feasibility": {
            "denominator": denominator,
            "rounding": "ROUND_HALF_UP",
            "unit": "single_run_micro_accuracy_percent",
            "cells": feasibility,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path(DEFAULT_SOURCE), help="原稿路径")
    parser.add_argument("--output", type=Path, help="报告 JSON 路径；省略时写入标准输出")
    parser.add_argument("--denominator", type=int, default=400, help="每卷二元准确率分母（默认 400）")
    args = parser.parse_args()
    try:
        if args.denominator <= 0:
            raise ValueError("分母必须大于 0")
        if args.output and args.output.resolve() == args.source.resolve():
            raise ValueError("输出路径不得覆盖原稿")
        raw = args.source.read_bytes()
        text = raw.decode("utf-8-sig")
        with localcontext() as context:
            context.prec = 40
            report = audit(text, raw, args.source.as_posix(), args.denominator)
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, UnicodeError, ValueError, InvalidOperation) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
