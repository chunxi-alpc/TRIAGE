#!/usr/bin/env python3
"""Render the CPU-only E1 attention-head quadrant diagnostic (issue #72).

Figure contract: compare activation movement, local gradient sensitivity and
candidate-screened patch effects without assuming separated head populations.
The three panels share quadrant meanings, but retain their own percentile
thresholds and denominators.  Python/matplotlib is the exclusive renderer.
Inputs are analysis CSV/JSON files; no model or activation cache is loaded.
Exports: a 183 mm-wide three-panel figure and five-fold small multiples,
with editable PDF/SVG text and a 400 dpi PNG preview.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

VIEW_NAMES = {
    "activation": "Activation movement",
    "gradient": "Gradient attribution",
    "patch": "Exact activation patching",
}
COLORS = {
    "AP_selective": "#D88A42",
    "CF_selective": "#3F83B3",
    "mixed": "#8D73AA",
    "low": "#A4ADB6",
}
LABELS = {
    "AP_selective": "AP selective",
    "CF_selective": "CF selective",
    "mixed": "Mixed",
    "low": "Low change",
}
QUADRANT_CELLS = {
    "CF_selective": (0, 0.5),
    "mixed": (0.5, 0.5),
    "low": (0, 0),
    "AP_selective": (0.5, 0),
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row: dict, key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return float(value.lower() == "true")
    return float(value)


def quadrant(x: np.ndarray, y: np.ndarray, tx: float, ty: float) -> np.ndarray:
    return np.where(x >= tx, np.where(y >= ty, "mixed", "AP_selective"),
                    np.where(y >= ty, "CF_selective", "low"))


def quadrant_counts(labels: np.ndarray) -> dict:
    n = len(labels)
    return {q: {"count": int(np.sum(labels == q)),
                "fraction": float(np.mean(labels == q)) if n else 0.0}
            for q in COLORS}


def panel(ax, rows: list[dict], view: str, thresholds: dict | None,
          title: str, subtitle: str, consensus: bool) -> dict:
    x = np.array([number(r, f"{view}_AP_pct") for r in rows])
    y = np.array([number(r, f"{view}_CF_pct") for r in rows])
    valid = np.isfinite(x) & np.isfinite(y)
    if view == "patch" and consensus:
        valid &= np.array([number(r, "patch_covered_folds", 0) >= 3 for r in rows])
    x, y = x[valid], y[valid]
    marker_key = "m1_selected_count" if consensus else "m1_selected"
    selected = np.array([number(r, marker_key, 0) for r in rows])[valid]
    tx = float(thresholds["x"]) if thresholds and thresholds.get("x") is not None else (
        float(np.quantile(x, .75)) if len(x) else np.nan)
    ty = float(thresholds["y"]) if thresholds and thresholds.get("y") is not None else (
        float(np.quantile(y, .75)) if len(y) else np.nan)
    labels = quadrant(x, y, tx, ty)
    if len(x):
        for q, (x0, x1, y0, y1) in {
            "low": (0, tx, 0, ty),
            "AP_selective": (tx, 100, 0, ty),
            "CF_selective": (0, tx, ty, 100),
            "mixed": (tx, 100, ty, 100),
        }.items():
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                   facecolor=COLORS[q], alpha=.045, edgecolor="none", zorder=0))
        ax.plot([0, 100], [0, 100], color="#C2C6CA", linewidth=.6, linestyle=":", zorder=1)
        for q in COLORS:
            take = labels == q
            ax.scatter(x[take], y[take], s=4.5 if view != "patch" else 11,
                       color=COLORS[q], alpha=.7 if view != "patch" else .9,
                       edgecolors="none", zorder=2)
        rings = selected >= 3 if consensus else selected > 0
        sizes = 9 + 4 * selected[rings] if consensus else np.full(int(rings.sum()), 24)
        ax.scatter(x[rings], y[rings], s=sizes, facecolors="none",
                   edgecolors="#171717", linewidths=.55, zorder=4)
        ax.axvline(tx, color="#3F4448", linestyle=(0, (3, 2)), linewidth=.7, zorder=3)
        ax.axhline(ty, color="#3F4448", linestyle=(0, (3, 2)), linewidth=.7, zorder=3)
    else:
        ax.text(.5, .5, "No heads meet the\nrecord-coverage requirement", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(length=2.5, pad=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("A/P response percentile")
    ax.set_ylabel("C/F response percentile")
    ax.text(0, 1.15, title, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left")
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=6.2, ha="left", color="#3F4448")
    return {"n": len(x), "quadrants": quadrant_counts(labels), "selected_n": int((selected >= 3).sum()) if consensus else int((selected > 0).sum()),
            "thresholds": {"x": tx, "y": ty}}


def count_table(ax, result: dict, fontsize: float = 5.7) -> None:
    ax.set_axis_off()
    for q, (left, bottom) in QUADRANT_CELLS.items():
        ax.add_patch(Rectangle((left+.005, bottom+.025), .49, .45,
                               transform=ax.transAxes, facecolor=COLORS[q], alpha=.13,
                               edgecolor="none"))
        data = result["quadrants"][q]
        ax.text(left+.25, bottom+.335, LABELS[q], transform=ax.transAxes,
                ha="center", va="center", fontsize=fontsize, color="#26313C")
        ax.text(left+.25, bottom+.14, f"{data['count']:,} heads ({100*data['fraction']:.1f}%)",
                transform=ax.transAxes, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="#26313C")


def selection_legend(fig, x=.065, y=.145) -> None:
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor="#171717", markeredgewidth=.55,
                      markersize=np.sqrt(9+4*n), label=f"{n}/5 folds") for n in (3, 4, 5)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x, y),
               ncol=3, fontsize=6.3, handletextpad=.4, columnspacing=1.1, borderaxespad=0)
    fig.text(.47, y, "Black rings: M1 selection in at least 3/5 folds", va="center", fontsize=6.3)


def save(fig, directory: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=400)
    plt.close(fig)


def render_main(rows: list[dict], summary: dict, directory: Path) -> dict:
    fig = plt.figure(figsize=(183/25.4, 112/25.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, .36],
                           left=.073, right=.99, top=.83, bottom=.235,
                           wspace=.32, hspace=.30)
    views = ("activation", "gradient", "patch")
    results = {}
    for i, view in enumerate(views):
        ax = fig.add_subplot(grid[0, i])
        reference = summary["views"][view]
        n = reference["n"]
        subtitle = f"All {n:,} heads; mean of 5 folds" if view != "patch" else f"Screened candidates; {n:,} heads"
        result = panel(ax, rows, view, reference["thresholds"],
                       f"{'abc'[i]}  {VIEW_NAMES[view]}", subtitle, True)
        if result["n"] != n:
            raise ValueError(f"{view} denominator mismatch: figure {result['n']} vs summary {n}")
        for q in COLORS:
            if result["quadrants"][q]["count"] != reference["quadrants"][q]["count"]:
                raise ValueError(f"{view}/{q} count disagrees with analysis summary")
        count_table(fig.add_subplot(grid[1, i]), result)
        results[view] = result
    fig.text(.073, .965, "Attention-head responses to A/P and C/F changes", fontsize=10,
             fontweight="bold", va="top")
    fig.text(.073, .913, "Five-fold consensus  |  E1 training sources  |  Qwen3-8B", fontsize=7, color="#4D555B")
    selection_legend(fig)
    footnotes = [
        "Dashed lines: 75th percentile of each consensus axis. Counts use each panel's displayed-head denominator.",
        "AP/CF selective denotes a response-selective candidate. Gradient attribution is a local first-order score proxy.",
        "Panel c: gradient-screened candidate set; complete A/P/C/F records in at least 3/5 folds. Candidate-selection bias applies.",
    ]
    for j, line in enumerate(footnotes):
        fig.text(.073, .10-.027*j, line, fontsize=5.8, color="#4D555B")
    save(fig, directory, "FIGURE_head_quadrants")
    return results


def render_folds(rows: list[dict], directory: Path) -> dict:
    fold_ids = sorted({int(number(r, "fold")) for r in rows})
    if len(fold_ids) != 5:
        raise ValueError(f"Expected 5 fold panels, got {fold_ids}")
    fig = plt.figure(figsize=(183/25.4, 167/25.4))
    grid = fig.add_gridspec(4, 3, height_ratios=[1, .33, 1, .33],
                           left=.073, right=.99, top=.88, bottom=.105,
                           wspace=.32, hspace=.48)
    results = {}
    for i, fold in enumerate(fold_ids):
        line = 2*(i//3)
        column = i % 3
        subset = [r for r in rows if int(number(r, "fold")) == fold]
        result = panel(fig.add_subplot(grid[line, column]), subset, "activation", None,
                       f"{'abcde'[i]}  Fold {fold}", "1,152 heads; 64 training sources", False)
        count_table(fig.add_subplot(grid[line+1, column]), result, fontsize=5.6)
        results[str(fold)] = result
    ax = fig.add_subplot(grid[2:, 2])
    ax.set_axis_off()
    ax.text(0, .92, "Reading the panels", fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(0, .82, "Black rings: the 51 M1-selected\nheads in each individual fold.\n\n"
            "Axes: within-fold percentiles\nacross 1,152 attention heads.\n\n"
            "Dashed lines: 75th percentile;\nhigh includes ties.\n\n"
            "The 2 x 2 count tables follow\nthe same quadrant positions.\n\n"
            "Denominator: 1,152 heads/fold.",
            fontsize=6.4, linespacing=1.25, va="top", transform=ax.transAxes)
    fig.text(.073, .97, "Activation-movement quadrants across five folds", fontsize=10, fontweight="bold", va="top")
    fig.text(.073, .93, "Training-only standardization and source-balanced RMS movement  |  Qwen3-8B", fontsize=7, color="#4D555B")
    fig.text(.073, .055, "A/P high and C/F low: interference-sensitive candidates. A/P low and C/F high: evidence-sensitive candidates.", fontsize=6)
    fig.text(.073, .032, "Both high: mixed sensitivity. Both low: low change. Shared axes permit comparison of head locations across folds.", fontsize=6)
    save(fig, directory, "FIGURE_head_quadrants_folds")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/aster_head_quadrant_20261116"))
    args = parser.parse_args()
    directory = args.report_dir
    consensus = read_csv(directory / "head_quadrant_consensus.csv")
    folds = read_csv(directory / "head_quadrant_metrics.csv")
    summary = json.loads((directory / "quadrant_summary.json").read_text(encoding="utf-8"))
    style()
    render_main(consensus, summary, directory)
    render_folds(folds, directory)
    print("Rendered main and five-fold quadrant figures (PNG/PDF/SVG), CPU only.")


if __name__ == "__main__":
    main()
