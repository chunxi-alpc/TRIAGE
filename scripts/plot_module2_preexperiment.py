#!/usr/bin/env python3
"""Render module-two figures from saved measurements, without fitting or inference.

Every displayed point comes from a result CSV. No synthetic illustration values,
confirmation-fitted PCA, smoothing, or selected-head-only diagnostic heatmaps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "triage-module2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


POSITIVE = "#167D8D"
NEGATIVE = "#CC6543"
NEUTRAL = "#9BA2AA"
MEAN = "#173E65"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty measurement file: {path}")
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Nonfinite measurements cannot be plotted as finite evidence")
    return result


def figure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "medium",
        "axes.linewidth": .65, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, output, stem):
    files = []
    for suffix in ("pdf", "svg", "png"):
        path = output / f"{stem}.{suffix}"
        metadata = {"Creator": "TRIAGE module-two saved-result plotting"}
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif suffix == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=450, metadata=metadata)
        files.append(path.name)
    plt.close(fig)
    return files


def grid(rows, field, split):
    chosen = [row for row in rows if row["split"] == split]
    layers = sorted({int(row["layer"]) for row in chosen})
    heads = sorted({int(row["kv_head"]) for row in chosen})
    by_head = {(int(row["layer"]), int(row["kv_head"])): number(row[field]) for row in chosen}
    if not layers or len(chosen) != len(by_head) or len(by_head) != len(layers) * len(heads):
        raise ValueError(f"Expected a complete, unique layer/KV-head grid for {split}")
    values = np.array([[by_head[layer, head] for layer in layers] for head in heads])
    return values, layers, heads


def heatmaps(rows, selection, audit, model_name, output):
    specs = [
        ("A", "fit_mean_paired_l2", "select", "Fitting: mean paired distance", r"$\mathbb{E}\,\|\bar{k}^{+}-\bar{k}^{-}\|_2$", "YlGnBu"),
        ("B", "paired_sign_consistency", "confirm", "Confirmation: sign consistency", "Aligned fraction", "viridis"),
        ("C", "read", "confirm", "Confirmation: signed Read", "Standardized Read", "RdBu_r"),
        ("D", "fit_coherence", "select", "Fitting: mean-shift coherence", "Coherence", "YlGnBu"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.85))
    fig.subplots_adjust(left=.075, right=.96, bottom=.205, top=.81, hspace=.65, wspace=.40)
    fig.suptitle(f"{model_name} | Applicability key diagnostics", x=.075, ha="left", y=.975, fontsize=12.5, weight="medium")
    counts = audit["partitions"]
    subtitle = (f"{audit['source_guidelines']} development guidelines: "
                f"{counts['fit']['guidelines']} fitting / {counts['select']['guidelines']} selection / "
                f"{counts['confirm']['guidelines']} confirmation")
    fig.text(.075, .91, subtitle, fontsize=9, color="#485261")
    selected = {(int(head["layer"]), int(head["kv_head"])) for head in selection["selected_heads"]}
    metrics = {}
    for ax, (letter, field, split, title, color_label, cmap) in zip(axes.flat, specs):
        values, layers, heads = grid(rows, field, split)
        plotted_values = values
        masked_cells = []
        if field == "paired_sign_consistency":
            shift_norm, norm_layers, norm_heads = grid(rows, "delta_norm", split)
            if norm_layers != layers or norm_heads != heads:
                raise ValueError("Shift-norm grid differs from sign-consistency grid")
            no_direction = shift_norm == 0.
            plotted_values = np.ma.masked_where(no_direction, values)
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_bad("#D7DCE2")
            masked_cells = [{"layer": layers[column], "kv_head": heads[row_index]}
                            for row_index, column in np.argwhere(no_direction)]
        if field == "read":
            magnitude = max(float(np.max(np.abs(values))), np.finfo(float).eps)
            kwargs = {"norm": TwoSlopeNorm(vmin=-magnitude, vcenter=0., vmax=magnitude)}
        elif field == "paired_sign_consistency":
            kwargs = {"vmin": 0., "vmax": 1.}
        else:
            kwargs = {"vmin": 0., "vmax": max(float(values.max()), np.finfo(float).eps)}
        graphic = ax.pcolormesh(np.arange(len(layers) + 1) - .5, np.arange(len(heads) + 1) - .5,
                               plotted_values, shading="flat", cmap=cmap, rasterized=False,
                               edgecolors="face", linewidth=0., **kwargs)
        ax.set_xlim(-.5, len(layers) - .5)
        ax.set_ylim(len(heads) - .5, -.5)
        xloc = sorted(set([*range(0, len(layers), max(1, len(layers) // 7)), len(layers) - 1]))
        ax.set_xticks(xloc, [str(layers[i]) for i in xloc])
        ax.set_yticks(range(len(heads)), heads)
        ax.set_xlabel("Layer (zero-based)", labelpad=3)
        ax.set_ylabel("KV head (zero-based)", labelpad=4)
        ax.set_title(f"{letter}   {title}", loc="left", pad=9)
        for layer, head in selected:
            if layer in layers and head in heads:
                ax.add_patch(Rectangle((layers.index(layer) - .5, heads.index(head) - .5), 1, 1,
                                       fill=False, edgecolor="#101820", linewidth=.8))
        bar = fig.colorbar(graphic, ax=ax, fraction=.045, pad=.025)
        bar.ax.tick_params(labelsize=7)
        bar.set_label(color_label, size=7.1, labelpad=5)
        if field != "paired_sign_consistency":
            bar.formatter.set_powerlimits((-2, 2))
            bar.update_ticks()
        metrics[field] = {"split": "fit" if field.startswith("fit_") else split,
                          "minimum": float(values.min()), "maximum": float(values.max()),
                          "layers": layers, "kv_heads": heads,
                          "zero_shift_cells_masked": masked_cells}
    accepted = (f"Black outlines: {len(selected)} heads accepted by selection screens."
                if selected else "No head passed the prespecified selection screens.")
    fig.text(.075, .126, "Keys are means over each action span after key normalization and before RoPE; A is not a per-token distance.", fontsize=7.5)
    fig.text(.075, .094, "Gray B cells: zero fitted shift, so the strict-sign fraction has no directional interpretation. Read still shows zero response.", fontsize=7.5)
    fig.text(.075, .062, f"All layer / KV-head cells are shown. {accepted}", fontsize=7.5)
    fig.text(.075, .029, "Exploratory representation measurements; signed Read is a local score response, not decision accuracy.", fontsize=7.5, color="#485261")
    return save_figure(fig, output, "head_diagnostics"), metrics


def group_mean(values, groups):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    return np.stack([values[groups == group].mean(axis=0) for group in sorted(set(groups))]).mean(axis=0)


def paired_shifts(rows, selection, audit, model_name, output, eigenvalues):
    grouped = defaultdict(dict)
    head_order = []
    for row in rows:
        head = (int(row["layer"]), int(row["kv_head"]))
        if head not in head_order:
            head_order.append(head)
        identity = row["id"]
        entry = grouped[head].setdefault(identity, {"group_id": row["group_id"]})
        if row["variant"] in entry:
            raise ValueError("Duplicate projected point")
        entry[row["variant"]] = np.array([number(row["x"]), number(row["y"])])
    if not 1 <= len(head_order) <= 4:
        raise ValueError("Expected one to four frozen selection-ranked candidates")
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))
    fig.subplots_adjust(left=.095, right=.97, bottom=.19, top=.82, hspace=.53, wspace=.30)
    fig.suptitle(f"{model_name} | Paired applicability shifts", x=.095, ha="left", y=.975, fontsize=12.5, weight="medium")
    fig.text(.095, .93, "Confirmation pairs in frozen training-difference subspaces", fontsize=9.3, color="#485261")
    fig.text(.095, .893, f"{len(head_order)} candidates ranked on the selection partition; inclusion here does not imply acceptance.", fontsize=8.1, color="#485261")
    summaries = []
    accepted = {(int(row["layer"]), int(row["kv_head"])) for row in selection["selected_heads"]}
    for panel, (ax, head) in enumerate(zip(axes.flat, head_order)):
        points = list(grouped[head].values())
        for point in points:
            if not all(variant in point for variant in ("positive", "negative", "neutral")):
                raise ValueError("Projected triplet is incomplete")
        positives = np.stack([point["positive"] for point in points])
        negatives = np.stack([point["negative"] for point in points])
        neutrals = np.stack([point["neutral"] for point in points])
        groups = [point["group_id"] for point in points]
        for negative, positive in zip(negatives, positives):
            ax.annotate("", xy=positive, xytext=negative,
                        arrowprops={"arrowstyle": "->", "lw": .6, "color": "#A3ABB2", "alpha": .8,
                                    "shrinkA": 1., "shrinkB": 1., "mutation_scale": 5.5}, zorder=1)
        ax.scatter(neutrals[:, 0], neutrals[:, 1], c=NEUTRAL, s=12, marker="x", linewidths=.65, alpha=.65, zorder=2)
        ax.scatter(negatives[:, 0], negatives[:, 1], c=NEGATIVE, s=15, edgecolors="white", linewidths=.25, alpha=.85, zorder=3)
        ax.scatter(positives[:, 0], positives[:, 1], c=POSITIVE, s=15, edgecolors="white", linewidths=.25, alpha=.85, zorder=4)
        mean_negative, mean_positive = group_mean(negatives, groups), group_mean(positives, groups)
        ax.annotate("", xy=mean_positive, xytext=mean_negative,
                    arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": MEAN,
                                "shrinkA": 0., "shrinkB": 0., "mutation_scale": 12}, zorder=6)
        ax.scatter(*mean_negative, c=MEAN, marker="s", s=12, zorder=5)
        status = "accepted" if head in accepted else "candidate"
        ax.set_title(f"{'ABCD'[panel]}   Layer {head[0]} / KV head {head[1]} ({status})", loc="left", fontsize=9.4, pad=8)
        spectrum = np.asarray(eigenvalues[head], dtype=float)
        if not np.isfinite(spectrum).all() or (spectrum < 0).any():
            raise ValueError("Invalid saved fitting-set spectrum")
        total_energy = spectrum.sum()
        available = min(2, int(selection["rank"]))
        ratios = (spectrum[:available] / total_energy).tolist() if total_energy > 0 else [None] * available
        ratios += [None] * (2 - available)
        component_labels = [f"Difference component {i + 1}" +
                            (f" ({100 * ratio:.1f}%)" if ratio is not None else " (zero energy)")
                            for i, ratio in enumerate(ratios)]
        if available < 2:
            component_labels[1] = "Unused axis (rank one)"
        ax.set_xlabel(component_labels[0])
        ax.set_ylabel(component_labels[1])
        ax.tick_params(labelsize=7.5)
        ax.margins(.12)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#E5E9ED", lw=.5, zorder=0)
        summaries.append({"layer": head[0], "kv_head": head[1], "accepted": head in accepted,
                          "triplets": len(points), "guidelines": len(set(groups)),
                          "mean_negative": mean_negative.tolist(), "mean_positive": mean_positive.tolist(),
                          "mean_shift": (mean_positive - mean_negative).tolist(),
                          "fitting_difference_energy_ratios": ratios})
    for ax in list(axes.flat)[len(head_order):]:
        ax.set_visible(False)
    legend = [Line2D([], [], linestyle="", marker="o", color=POSITIVE, markersize=4, label="Applicable"),
              Line2D([], [], linestyle="", marker="o", color=NEGATIVE, markersize=4, label="Inapplicable"),
              Line2D([], [], linestyle="", marker="x", color=NEUTRAL, markersize=4, label="Unknown"),
              Line2D([], [], color="#A3ABB2", lw=.8, label="Paired shift"),
              Line2D([], [], color=MEAN, lw=2.2, label="Mean shift")]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.53, .105), fontsize=8.2,
               handlelength=1.5, columnspacing=1.6)
    counts = audit["partitions"]
    fig.text(.095, .076,
             f"Axes: {counts['fit']['guidelines']} fitting guidelines. Head ranking: {counts['select']['guidelines']} selection guidelines. "
             f"Points: {counts['confirm']['guidelines']} separate confirmation guidelines.", fontsize=7.5)
    fig.text(.095, .045, "Gray arrows run from inapplicable to applicable. The dark arrow is the source-guideline-weighted mean.", fontsize=7.5)
    fig.text(.095, .014, "Axis percentages: fitting-set paired-difference energy. Pair variants are not independent clinical cases.", fontsize=7.5, color="#485261")
    return save_figure(fig, output, "paired_shifts"), summaries


def causal_dose_response(rows, plan, model_name, output, coverage):
    """Show all saved doses; absent/blocked doses become gaps, not invented points."""
    styles = {
        "projected_mean": ("Projected mean", POSITIVE, "o", "-"),
        "raw_mean": ("Raw mean", "#6496AD", "s", "--"),
        "random_direction": ("Random direction", "#757C85", "^", ":"),
        "reversed_direction": ("Reversed direction", NEGATIVE, "v", "--"),
        "label_shuffled": ("Shuffled signs", "#8B70A8", "D", ":"),
        "direct_score_bias": ("Direct score bias", MEAN, "P", "-"),
    }
    metrics = (("gold_attention_mass", "Gold-rule attention mass", 100., "% of attention"),
               ("gold_logit_margin", "Gold logit margin", 1., "Gold minus other-rule logit"),
               ("pair_accuracy", "Paired branch accuracy", 100., "% of complete pairs correct"))
    by_cell = {}
    for row in rows:
        key = (row["method"], number(row["tau"]), row["metric"])
        if key in by_cell:
            raise ValueError("Duplicate causal metric cell")
        if row["method"] not in styles:
            raise ValueError(f"Unrecognized saved causal method: {row['method']}")
        mean, low, high = (number(row[name]) for name in ("mean", "ci_low", "ci_high"))
        if low > high:
            raise ValueError("Reversed causal confidence interval")
        by_cell[key] = (mean, low, high)
    doses = sorted(number(tau) for tau in plan["taus"])
    methods = [method for method in styles if method in plan["methods"]]
    if not doses or doses[0] != 0:
        raise ValueError("Causal plots require an observed unedited zero-dose baseline")
    coverage_by_cell = {}
    for row in coverage:
        key = (row["method"], number(row["tau"]))
        if key in coverage_by_cell:
            raise ValueError("Duplicate method-dose coverage cell")
        counts = {name: int(row[name]) for name in ("expected_observations", "observed", "completed", "blocked", "missing")}
        if (min(counts.values()) < 0 or counts["observed"] + counts["missing"] != counts["expected_observations"] or
                counts["completed"] + counts["blocked"] != counts["observed"]):
            raise ValueError("Inconsistent causal coverage counts")
        valid = counts["blocked"] == counts["missing"] == 0
        if (row["status"] == "completed") != valid:
            raise ValueError("Causal coverage status contradicts its counts")
        coverage_by_cell[key] = {**counts, "status": row["status"], "reasons": json.loads(row["reasons"])}
    if set(coverage_by_cell) != {(method, dose) for method in methods for dose in doses}:
        raise ValueError("Causal coverage must enumerate every planned method-dose cell")
    for method, dose, metric in by_cell:
        if coverage_by_cell[method, dose]["status"] != "completed":
            raise ValueError("Incomplete method-dose cells must not have main outcome estimates")
    incomplete_counts = {method: sum(coverage_by_cell[method, dose]["status"] != "completed" for dose in doses)
                         for method in methods}
    totals = {field: sum(row[field] for row in coverage_by_cell.values())
              for field in ("expected_observations", "observed", "completed", "blocked", "missing")}
    completed_cells = sum(row["status"] == "completed" for row in coverage_by_cell.values())
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 4.9))
    fig.subplots_adjust(left=.075, right=.975, bottom=.39, top=.76, wspace=.41)
    fig.suptitle(f"{model_name} | Oracle routing dose response", x=.075, ha="left", y=.975,
                 fontsize=12.5, weight="medium")
    head = plan["head"]
    head_status = "accepted head" if plan["head_accepted_by_offline_screen"] else "diagnostic candidate; not accepted"
    n_groups = len(plan["confirmation_groups"])
    fig.text(.075, .925, f"Layer {head['layer']} / KV head {head['kv_head']} ({head_status}); "
             f"{n_groups} confirmation guidelines", fontsize=9.2, color="#485261")
    fig.text(.075, .876, "Direction magnitudes are matched by the actual fixed-query attention-score perturbation RMS.",
             fontsize=8.1, color="#485261")
    missing = []
    for panel, (ax, (metric, title, multiplier, label)) in enumerate(zip(axes, metrics)):
        zero_means = [by_cell[(method, 0., metric)][0] for method in methods if (method, 0., metric) in by_cell]
        if not zero_means or np.ptp(zero_means) > 1e-8:
            raise ValueError("Causal methods must share the same observed zero-dose baseline")
        ax.axhline(zero_means[0] * multiplier, color="#A6ACB4", linestyle=(0, (3, 2)), lw=.75, zorder=0)
        maxima = []
        for method in methods:
            title_method, color, marker, linestyle = styles[method]
            if incomplete_counts[method]:
                title_method += f" [{incomplete_counts[method]} incomplete]"
            values = []
            for dose in doses:
                entry = by_cell.get((method, dose, metric))
                if entry is None:
                    if coverage_by_cell[method, dose]["status"] == "completed":
                        raise ValueError("A completed method-dose cell lacks a required metric")
                    missing.append({"method": method, "tau": dose, "metric": metric})
                    values.append((np.nan, np.nan, np.nan))
                else:
                    values.append(tuple(value * multiplier for value in entry))
            means, lows, highs = np.asarray(values).T
            maxima.extend(highs[np.isfinite(highs)].tolist())
            ax.fill_between(doses, lows, highs, color=color, alpha=.055, linewidth=0, zorder=1)
            ax.plot(doses, means, color=color, marker=marker, markersize=4 if method == "projected_mean" else 3.5,
                    markerfacecolor="white" if method != "projected_mean" else color,
                    markeredgewidth=.85, linestyle=linestyle,
                    linewidth=1.9 if method in {"projected_mean", "direct_score_bias"} else 1.2,
                    label=title_method, zorder=4 if method == "projected_mean" else 3)
        ax.set_title(f"{'ABC'[panel]}   {title}", loc="left", pad=9, fontsize=9.2)
        ax.set_xlabel(r"Score-perturbation RMS, $\tau$", labelpad=5)
        ax.set_ylabel(label, labelpad=4)
        ax.set_xticks(doses, [f"{dose:g}" for dose in doses])
        ax.grid(axis="y", color="#E4E8EC", lw=.6)
        ax.margins(x=.07)
        if metric == "pair_accuracy":
            ax.set_ylim(0, 103)
        elif metric == "gold_attention_mass":
            ax.set_ylim(0, min(103, max(maxima, default=1.) * 1.08))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .19), ncol=3, frameon=False,
               fontsize=8.1, handlelength=2.4, columnspacing=2.4, labelspacing=.7)
    fig.text(.075, .155, f"Coverage: {completed_cells}/{len(coverage)} complete method-dose cells; "
             f"{totals['blocked']}/{totals['expected_observations']} blocked evaluations; {totals['missing']} missing. Evaluations reuse conditions.", fontsize=7.3)
    fig.text(.075, .123, "Shading: 95% source-guideline bootstrap intervals, conditional on the frozen operator; unadjusted exploratory intervals.", fontsize=7.3)
    fig.text(.075, .091, "Ordinary causal mask, without ASTER BLIND/REVEAL masks. Oracle routing; single-token A/B branch choice only.", fontsize=7.3)
    fig.text(.075, .059, "All complete method-dose estimates are shown. Incomplete cells remain gaps and are counted in the legend.", fontsize=7.3)
    fig.text(.075, .027, f"Only {n_groups} independent guidelines. This diagnostic cannot establish efficacy or population-level equivalence.", fontsize=7.3, color="#485261")
    coverage_summary = {"totals": totals, "method_dose_cells": len(coverage), "completed_method_dose_cells": completed_cells,
                        "incomplete_by_method": incomplete_counts,
                        "cells": [{"method": key[0], "tau": key[1], **value} for key, value in coverage_by_cell.items()]}
    return save_figure(fig, output, "causal_dose_response"), missing, coverage_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal-results", type=Path,
                        help="Optional completed causal result directory; otherwise discover RESULTS/causal")
    args = parser.parse_args()
    inputs = {name: args.results / name for name in (
        "head_metrics.csv", "paired_projection.csv", "selection.json", "data_audit.json", "extraction_fingerprint.json", "status.json", "operators.npz")}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required measured results are absent: " + ", ".join(missing))
    status = json.loads(inputs["status.json"].read_text())
    if status.get("status") != "completed":
        raise ValueError("Plotting requires a completed representation run")
    selection = json.loads(inputs["selection.json"].read_text())
    audit = json.loads(inputs["data_audit.json"].read_text())
    fingerprint = json.loads(inputs["extraction_fingerprint.json"].read_text())
    if not selection.get("confirmation_not_used"):
        raise ValueError("Confirmation-independent head selection must be recorded")
    model_name = fingerprint["model_id"].split("/")[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    figure_style()
    files1, metrics = heatmaps(read_csv(inputs["head_metrics.csv"]), selection, audit, model_name, args.output)
    with np.load(inputs["operators.npz"], allow_pickle=False) as operators:
        files2, pairs = paired_shifts(read_csv(inputs["paired_projection.csv"]), selection, audit, model_name,
                                     args.output, operators["eigenvalues"])
    metadata = {
        "schema_version": "module2.figures.v1", "model": fingerprint["model_id"],
        "model_revision": fingerprint["revision"], "source_inputs": {
            name: {"file": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "plot_script_sha256": sha256(__file__), "generated_files": files1 + files2,
        "rendering_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "head_metrics": metrics, "projection_heads": pairs,
        "evidence_scope": "development-only exploratory representation pilot",
        "attention_mask": "ordinary causal; no ASTER BLIND/REVEAL masks",
        "pair_axes": "fitting-set uncentered paired-difference eigenspace; no confirmation fitting",
        "distance_unit": "mean L2 distance between action-span-mean pre-RoPE keys, not per-token L2",
        "no_model_or_subspace_refitting": True,
        "causal_figure": "not generated; causal measurements are handled only when explicitly available",
    }
    causal_dir = args.causal_results or args.results / "causal"
    causal_csv = causal_dir / "causal_metrics.csv"
    if args.causal_results is not None and not causal_csv.is_file():
        raise FileNotFoundError(f"Requested causal metrics are absent: {causal_csv}")
    if causal_csv.is_file():
        causal_inputs = {name: causal_dir / name for name in ("causal_metrics.csv", "causal_coverage.csv", "plan.json", "status.json")}
        causal_status = json.loads(causal_inputs["status.json"].read_text())
        if causal_status.get("status") not in {"completed", "partial"}:
            raise ValueError("Causal plotting requires a completed or explicitly partial causal run")
        if not causal_status.get("zero_validation_passed"):
            raise ValueError("Causal plotting requires recorded zero-intervention validation")
        plan = json.loads(causal_inputs["plan.json"].read_text())
        if plan["model_id"] != fingerprint["model_id"] or plan["revision"] != fingerprint["revision"]:
            raise ValueError("Representation and causal model provenance differ")
        if plan["selection_sha256"] != sha256(inputs["selection.json"]):
            raise ValueError("Causal head selection differs from the plotted selection record")
        files3, missing, coverage_summary = causal_dose_response(read_csv(causal_csv), plan, model_name,
            args.output, read_csv(causal_inputs["causal_coverage.csv"]))
        metadata["generated_files"].extend(files3)
        metadata["causal_figure"] = {"status": "generated", "run_status": causal_status["status"],
            "missing_metric_cells": missing, "coverage": coverage_summary,
            "attention_mask": "ordinary causal; no ASTER BLIND/REVEAL masks",
            "head_accepted_by_offline_screen": plan["head_accepted_by_offline_screen"],
            "source_inputs": {name: {"file": str(path), "sha256": sha256(path)} for name, path in causal_inputs.items()}}
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"figures": metadata["generated_files"], "model": fingerprint["model_id"]}, indent=2))


if __name__ == "__main__":
    main()
