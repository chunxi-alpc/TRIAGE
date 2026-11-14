"""Plot completed ASTER external-method results on CPU.

Run from the repository root:
python -m experiments.aster_codex03_external_20261112.plot_results

The method summary supplies plotted means and seed SD. The completed-run table
supplies denominators and per-seed source data. No model or test selection runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/aster_codex03_external_20261112"
METHODS = [
    ("RePS-E1-task-adapted", "RePS", "#788AB3"),
    ("PSR-MSE-E1-task-adapted", "PSR-MSE", "#788AB3"),
    ("CAA-E1-task-adapted", "CAA", "#788AB3"),
    ("INLP-task-adapted", "INLP", "#938A9D"),
    ("DoRA-E1-SFT", "DoRA", "#BC9476"),
]
METRICS = [
    ("ap_rms", "a", "A/P RMS", 1., "score units", "lower", "ap_pairs"),
    ("cf_both_correct_accuracy", "b", "C/F both endpoints correct", 100.,
     "%", "higher", "cf_paired_labeled_count"),
    ("natural_path_accuracy", "c", "Natural path accuracy", 100.,
     "%", "higher", "natural_path_count"),
    ("interference_path_accuracy", "c", "Interference path accuracy", 100.,
     "%", "higher", "interference_path_count"),
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def prepare(summary_path, runs_path):
    summaries = {r["method"]: r for r in read_csv(summary_path)}
    runs = {r["method"]: r for r in read_csv(runs_path)}
    if set(summaries) != {m[0] for m in METHODS}:
        raise ValueError("The summary must contain exactly the five completed methods.")
    data = {}
    exported = []
    for full, label, color in METHODS:
        row = summaries[full]
        names = row["runs"].split(";")
        n = int(row["completed_runs"])
        if n != len(names):
            raise ValueError("Run count disagrees with the summary: " + full)
        group = [runs[name] for name in names]
        if any(r["split"] != "test" or r["complete"].lower() != "true"
               or r["result_kind"] != "actual_experiment" for r in group):
            raise ValueError("Only completed actual test results may be plotted.")
        seeds = ";".join(name.rsplit("-seed", 1)[1] for name in names
                         if "-seed" in name)
        data[full] = {"n": n, "seeds": seeds, "metrics": {}}
        for key, panel, title, scale, unit, direction, countkey in METRICS:
            mean = float(row[key + "_mean"])
            sd_text = row[key + "_seed_sd"]
            sd = float(sd_text) if sd_text else None
            values = [float(r[key]) for r in group]
            if not math.isclose(mean, statistics.mean(values), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Summary mean differs from completed runs: " + full + "/" + key)
            if n > 1:
                expected_sd = statistics.stdev(values)
                if sd is None or not math.isclose(sd, expected_sd, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Summary seed SD differs from completed runs: " + full + "/" + key)
            elif sd is not None:
                raise ValueError("Single deterministic runs must have an empty SD.")
            counts = {int(r[countkey]) for r in group}
            if len(counts) != 1:
                raise ValueError("Metric denominator differs across seeds: " + full + "/" + key)
            denominator = counts.pop()
            base = {
                "panel": panel, "method": full, "display_name": label,
                "metric": key, "metric_label": title, "direction": direction,
                "plot_unit": unit, "plot_multiplier": scale,
                "denominator": denominator, "denominator_field": countkey,
                "completed_runs": n, "seeds": seeds,
            }
            metric = dict(base, mean=mean, sd=sd,
                          plotted_mean=mean * scale,
                          plotted_sd=None if sd is None else sd * scale)
            data[full]["metrics"][key] = metric
            exported.append(dict(base, record_type="summary", run_name="",
                                 raw_value=mean, raw_seed_sd="" if sd is None else sd,
                                 plot_value=mean * scale,
                                 plot_seed_sd="" if sd is None else sd * scale,
                                 source_file=relative(summary_path)))
            for run, value in zip(group, values):
                exported.append(dict(base, record_type="run", run_name=run["method"],
                                     raw_value=value, raw_seed_sd="",
                                     plot_value=value * scale, plot_seed_sd="",
                                     source_file=relative(runs_path)))
    return data, exported


def make_plot(data, outdir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.linewidth": 0.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "legend.frameon": False, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 105 / 25.4),
                             sharey=True)
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.22,
                        top=0.72, wspace=0.31)
    titles = ["A/P invariance", "C/F joint correctness", "Path correctness"]
    subtitles = ["2,293 pairs", "40 labeled pairs", "806 natural / 2,418 interference"]
    labels = [
        "A/P RMS (lower is better)",
        "Both endpoints correct (%)",
        "Correct paths (%)",
    ]
    for j, ax in enumerate(axes):
        ax.set_title(titles[j], loc="left", pad=19, fontweight="bold")
        ax.text(0, 1.04, subtitles[j], transform=ax.transAxes,
                ha="left", va="bottom", color="#555555", fontsize=6.5)
        ax.text(-0.06, 1.25, "abc"[j], transform=ax.transAxes,
                fontweight="bold", fontsize=10, ha="right")
        ax.set_xlabel(labels[j], labelpad=7)
        ax.set_ylim(4.6, -0.6)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.55)
        ax.tick_params(axis="y", length=0, pad=7)
        for y in (0.5, 1.5, 2.5, 3.5):
            ax.axhline(y, color="#EEEEEE", linewidth=0.55, zorder=0)
    ylabels = [label + (" (n=3)" if data[full]["n"] == 3 else " (1 run)")
               for full, label, _ in METHODS]
    axes[0].set_yticks(range(len(METHODS)), ylabels)
    max_rms = max(data[full]["metrics"]["ap_rms"]["plotted_mean"] +
                  (data[full]["metrics"]["ap_rms"]["plotted_sd"] or 0.)
                  for full, _, _ in METHODS)
    axes[0].set_xlim(0, math.ceil(max_rms * 1.08))
    axes[0].xaxis.set_major_locator(MultipleLocator(2))
    for ax in axes[1:]:
        ax.set_xlim(0, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
    for y, (full, _, color) in enumerate(METHODS):
        for key, panel, _, _, _, _, _ in METRICS:
            record = data[full]["metrics"][key]
            ax = axes["abc".index(panel)]
            is_interference = key == "interference_path_accuracy"
            is_path = panel == "c"
            offset = (0.14 if is_interference else -0.14) if is_path else 0.
            marker = "D" if is_interference else "o"
            ax.errorbar(record["plotted_mean"], y + offset,
                        xerr=record["plotted_sd"], fmt=marker,
                        markersize=4.1 if is_interference else 4.7,
                        color=color, markeredgecolor=color,
                        markerfacecolor="white" if is_interference else color,
                        markeredgewidth=1., ecolor=color, elinewidth=1.05,
                        capsize=2.2, capthick=0.9, zorder=4)
    handles = [
        Line2D([], [], color="#505050", marker="o", linestyle="none",
               markersize=4.5, label="Natural"),
        Line2D([], [], color="#505050", marker="D", linestyle="none",
               markerfacecolor="white", markersize=4, label="Interference"),
    ]
    axes[2].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                   ncol=2, handletextpad=0.35, columnspacing=0.9, fontsize=6.5)
    fig.text(0.17, 0.94, "Qwen3-8B | E1 / Dataset A",
             fontsize=10.5, fontweight="bold", ha="left")
    fig.text(0.17, 0.885, "External-method task adaptations | controlled test: 40 sources",
             fontsize=7.1, ha="left", color="#444444")
    fig.text(0.17, 0.075,
             "Points: mean across completed runs. Error bars: seed SD (n=3, seeds 11/22/33).",
             fontsize=6.5, ha="left")
    fig.text(0.17, 0.037,
             "CAA and INLP: one deterministic run, without error bars. All five methods are shown.",
             fontsize=6.5, ha="left")
    stem = outdir / "section4_5_external_methods"
    for extension in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=400)
    # Compact preview is produced by the same Python plotting backend.
    fig.savefig(outdir / "preview.png", dpi=130)
    plt.close(fig)


def write_readme(outdir, data):
    method_rows = []
    for full, label, _ in METHODS:
        record = data[full]
        seeds = record["seeds"] or "deterministic"
        method_rows.append("| " + label + " | " + full + " | " +
                           str(record["n"]) + " | " + seeds + " |")
    text = """# External-method results: figure for Section 4.5

![Five-method results](section4_5_external_methods.png)

This quantitative three-panel figure compares the five completed task adaptations
of external methods on the same controlled test. Panel a reports A/P score
stability, panel b reports joint correctness on C/F pairs, and panel c reports
complete-path correctness. The display order follows the method categories
(RePS, PSR, CAA, concept erasure, parameter-efficient tuning), without ranking or
filtering methods using test results.

## Data and reproducibility

- Model: Qwen3-8B.
- Fitting data: E1 / Dataset A, 2,540 pairs from 80 fitting sources.
- Development set: 20 sources; training and development choices precede test scoring.
- Evaluation scope: historical 40-source controlled test. This is the historical
  evaluation set described in the experiment report, distinct from the official
  7,458-case MEGA test.
- Plotted means and standard deviations are read directly from
  [section4_5_method_summary.csv](../section4_5_method_summary.csv).
- Per-run values and metric denominators are read from
  [section4_5_new_methods.csv](../section4_5_new_methods.csv).
- [source_data.csv](source_data.csv) retains both summary and individual-run rows,
  raw proportions, plot values, seed counts, denominators, units, and source paths.
  Select record_type=summary to reproduce the 20 plotted means.
- Reproduce from the repository root:
  `python -m experiments.aster_codex03_external_20261112.plot_results`.
  The script runs on CPU and only reads completed results.

## Methods and repeated runs

| Plot label | Full method name | Completed runs | Seeds |
|---|---|---:|---|
""" + "\n".join(method_rows) + """

All five methods are task adaptations as identified by their full names above.
PSR-MSE denotes the MSE version of PSR. Labels are shortened only in the figure.

Points are the arithmetic mean across completed runs. For the three stochastic
methods, error bars are plus/minus one sample standard deviation across seeds
11, 22, and 33 (ddof=1, n=3). CAA and INLP each have one deterministic result;
their SD fields remain empty and no error bars are added. A measured SD of zero
remains zero. Seed SD describes training-seed variation and is not a confidence
interval over evaluation examples. No significance tests or comparisons selected
from test results are added.

## Metrics and denominators

| Panel / metric | Definition | Per-run denominator | Direction / plot unit |
|---|---|---:|---|
| a: A/P RMS | Square root of the mean squared difference between the method's right and left scores across A/P pairs: sqrt(mean((ell_right - ell_left)^2)) | 2,293 A/P pairs, spanning 40 sources | Lower; score units |
| b: C/F joint correctness | Fraction of labeled C/F pairs for which both endpoint predictions equal their labels | 40 labeled C/F pairs, spanning 12 sources | Higher; percent |
| c: Natural path accuracy | Fraction of original paths whose complete node/decision sequence and terminal leaf match gold | 806 original paths | Higher; percent |
| c: Interference path accuracy | Fraction of paths under action_presence, action_neutral, and action_swap whose complete sequence and leaf match gold | 2,418 paths (3 variants per case) | Higher; percent |

The pair RMS uses the method's score difference, not its correction relative to
the raw model. Prediction uses the existing evaluation rule (ell > 0).
All proportion means and their SD are multiplied by 100 for plotting.
Thus accuracy SD is expressed in percentage points; RMS is not rescaled.
All methods and seeds have the same denominators. Seed-level accuracies are
averaged without pooling the examples across repeated runs. F0 pairs are
excluded from these panels, following the shared A/P and C/F protocol.

## Figure files

- [PNG, 400 dpi](section4_5_external_methods.png)
- [SVG with editable text](section4_5_external_methods.svg)
- [PDF with embedded TrueType text](section4_5_external_methods.pdf)
- [Compact preview](preview.png)
- [Source data](source_data.csv)

The canvas is 183 by 105 mm. Ratio axes show the full 0-100% range; the RMS axis
starts at zero and includes every mean plus its SD. No methods, seeds, or error
bars are omitted. Shape distinguishes natural and interference paths, with
consistent method-family colors across panels. Matplotlib generates every
format from the same figure; no manual image edits are applied.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=REPORT / "section4_5_method_summary.csv")
    parser.add_argument("--runs", type=Path, default=REPORT / "section4_5_new_methods.csv")
    parser.add_argument("--outdir", type=Path, default=REPORT / "figures")
    args = parser.parse_args()
    data, exported = prepare(args.summary, args.runs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    keys = list(exported[0])
    with (args.outdir / "source_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(exported)
    make_plot(data, args.outdir)
    write_readme(args.outdir, data)
    artifacts = {p.name: p.stat().st_size for p in sorted(args.outdir.iterdir()) if p.is_file()}
    print(json.dumps({"methods": len(data), "source_rows": len(exported),
                      "outdir": str(args.outdir), "artifacts_bytes": artifacts}, indent=2))


if __name__ == "__main__":
    main()
