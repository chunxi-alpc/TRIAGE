#!/usr/bin/env python3
"""Reanalyse the archived CARE O2 per-head table; NO new model inference.

The ranking uses TRAIN cross-validation AUC only. Held-out AUC is inspected
after selection. All findings are exploratory, especially because this
secondary analysis was motivated by seeing the archived test statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx, dy = [v - mx for v in xs], [v - my for v in ys]
    sx = sum(v * v for v in dx)
    sy = sum(v * v for v in dy)
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sx * sy) if sx > 0 and sy > 0 else float("nan")


def describe(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def row_compact(row: dict, key: str) -> dict:
    return {
        "layer": int(row["layer"]), "kv_head": int(row["kv_head"]),
        "selection_score": row[key],
        "applicability_cv_auc": row["applicability_cv_auc"],
        "applicability_test_auc": row["applicability_test_auc"],
        "relevance_test_auc": row["relevance_test_auc"],
        "applicability_shift": row["applicability_shift"],
        "same_side_shift": row["same_side_shift"],
        "cross_per_unit_minus_same_side_per_unit": row["cross_per_unit_minus_same_side_per_unit"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=root / "results/obs2/layer_head_stats.csv")
    p.add_argument("--output", type=Path, default=root / "results/module2_o2_reanalysis/head_selection.json")
    args = p.parse_args()

    with args.table.open(newline="", encoding="utf-8") as f:
        data = [
            {k: int(v) if k in ("layer", "kv_head") else float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]
    assert len(data) == 288, f"Expected 36 layers x 8 KV heads, got {len(data)}"

    col = lambda k: [row[k] for row in data]
    train_rank = sorted(data, key=lambda row: (-row["applicability_cv_auc"], row["layer"], row["kv_head"]))
    selected = next(row for row in data if row["layer"] == 2 and row["kv_head"] == 0)
    top10 = train_rank[:10]

    diverse = []
    seen_layers = set()
    for row in train_rank:
        if row["layer"] not in seen_layers:
            diverse.append(row_compact(row, "applicability_cv_auc"))
            seen_layers.add(row["layer"])
        if len(diverse) == 8:
            break

    layer_bins = []
    for start, end in [(0, 8), (9, 17), (18, 26), (27, 35)]:
        group = [row for row in data if start <= row["layer"] <= end]
        layer_bins.append({
            "layers": f"{start}-{end}", "heads": len(group),
            "train_cv_auc": describe([r["applicability_cv_auc"] for r in group]),
            "test_auc": describe([r["applicability_test_auc"] for r in group]),
            "positive_boundary_per_unit": sum(r["cross_per_unit_minus_same_side_per_unit"] > 0 for r in group),
        })

    result = {
        "status": "completed_archived_results_reanalysis",
        "date": "2026-10-17",
        "source_sha256": hashlib.sha256(args.table.read_bytes()).hexdigest(),
        "source_path": str(args.table),
        "n_layer_head_positions": len(data),
        "n_test_auc_ge_0_8": sum(r["applicability_test_auc"] >= 0.8 for r in data),
        "n_test_auc_ge_0_9": sum(r["applicability_test_auc"] >= 0.9 for r in data),
        "n_app_test_fdr_lt_0_05": sum(r["applicability_test_p_fdr"] < 0.05 for r in data),
        "n_positive_boundary_per_unit": sum(r["cross_per_unit_minus_same_side_per_unit"] > 0 for r in data),
        "correlation_app_and_same_side_shift": pearson(col("applicability_shift"), col("same_side_shift")),
        "correlation_rel_and_app_cv_auc": pearson(col("relevance_cv_auc"), col("applicability_cv_auc")),
        "applicability_test_auc_summary": describe(col("applicability_test_auc")),
        "original_relevance_selected": row_compact(selected, "applicability_cv_auc"),
        "original_head_train_app_rank": next(i + 1 for i, row in enumerate(train_rank)
                                             if row["layer"] == 2 and row["kv_head"] == 0),
        "best_train_app_cv_head": row_compact(train_rank[0], "applicability_cv_auc"),
        "top10_train_app_cv_mean_test_auc": statistics.fmean([r["applicability_test_auc"] for r in top10]),
        "layer_bins": layer_bins,
        "eight_train_selected_distinct_layers": diverse,
        "interpretation": [
            "AUC measures linear decodability of fictional applicability labels, NOT causal steering.",
            "CV-based head rank was computed from archived training CV scores; held-out figures were already reported publicly in the repository.",
            "No new model inference, hidden-state causal patching or B1-B4 projector evaluation was performed.",
            "All head-level statistics share input groups and are not independent statistical units.",
            "Boundary-per-unit statistic is exploratory and may depend on control construction/normalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_heads": result["n_layer_head_positions"],
        "original_head_train_rank": result["original_head_train_app_rank"],
        "best_head": [train_rank[0]["layer"], train_rank[0]["kv_head"]],
        "best_head_test_auc": train_rank[0]["applicability_test_auc"],
        "app_same_shift_correlation": result["correlation_app_and_same_side_shift"],
        "output": str(args.output)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
