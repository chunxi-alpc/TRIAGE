#!/usr/bin/env python3
"""Plot saved offline reanalysis; this script never runs or simulates a model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "care_m2_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ["B4_positive", "B4_signed", "d_app_signed", "score_bias_signed"]
LABELS = ["B4 positive projection", "B4 signed projection", "Signed mean direction", "Direct score bias"]


def load_metrics(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["metric"] == "paired_accuracy" and r["method"] != "baseline"]
    if len(rows) != 4 or any(r["new_model_forwards"] != "0" for r in rows):
        raise ValueError("expected four archived-inference reanalysis rows")
    return {r["method"]: r for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, default=HERE / "results/replay")
    p.add_argument("--output", type=Path, default=HERE / "results/figures")
    args = p.parse_args()
    files = [args.replay / "metrics_ci.csv", args.replay / "method_specific_metrics_ci.csv"]
    tables = [load_metrics(f) for f in files]
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)
    for ax, table, title, color in zip(axes, tables,
            ["Shared development strength", "Method-specific development strength"],
            ["#2378a5", "#25816a"]):
        value = np.array([float(table[m]["delta_vs_baseline"]) for m in METHODS]) * 100
        lo = np.array([float(table[m]["delta_ci_low"]) for m in METHODS]) * 100
        hi = np.array([float(table[m]["delta_ci_high"]) for m in METHODS]) * 100
        y = np.arange(len(METHODS))
        ax.axvline(0, color="#747b81", linestyle="--", linewidth=1)
        ax.errorbar(value, y, xerr=np.vstack([value-lo, hi-value]), fmt="o", color=color,
                    capsize=4, elinewidth=2.2, markersize=7)
        for v, yy in zip(value, y):
            ax.text(v, yy-.18, f"{v:+.2f} pp", ha="center", va="bottom", color=color, fontsize=10)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlim(-2.3, 5.1)
        ax.set_xticks([-2, 0, 2, 4])
        ax.set_xlabel("Paired accuracy change vs baseline (percentage points)", fontsize=10)
        ax.set_yticks(y, LABELS)
        ax.grid(axis="x", alpha=.18)
        ax.set_ylim(3.6, -.65)
    fig.suptitle("Archived oracle routing: sensitivity to development selection", fontsize=15, y=.97)
    fig.text(.5, .88, "192 pairs  |  12 rule families  |  95% paired family-bootstrap intervals", ha="center", fontsize=11)
    fig.text(.02, .025,
             "Offline reanalysis; new model forwards = 0. Intervals condition on fitted folds. No refitting or multiplicity correction.",
             fontsize=9, color="#4b5157")
    fig.subplots_adjust(left=.22, right=.98, bottom=.17, top=.77, wspace=.16)
    png, svg = args.output / "historical_routing.png", args.output / "historical_routing.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    metadata = {
        "status": "completed_plot_of_saved_offline_reanalysis",
        "new_model_forwards": 0,
        "inputs_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "outputs_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (png, svg)},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": metadata["status"], "new_model_forwards": 0, "output": str(args.output)}))


if __name__ == "__main__":
    main()
