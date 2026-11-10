"""Plot measured held-out AP suppression and C/F changes for each backbone."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .fit_controlled import OUT

MODELS = ["llama31", "gemma2", "gemma3", "medgemma"]
LABELS = ["Llama-3.1-8B", "Gemma-2-9B", "Gemma-3-27B", "MedGemma-27B"]


def main():
    records = {}
    for model in MODELS:
        with (OUT/model/"controlled_summary.csv").open(encoding="utf8") as f:
            for r in csv.DictReader(f):
                if r["split"] == "test":
                    records[(model, r["method"], r["group"])] = r
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    y = np.arange(4)
    specifications = [("AP", "variance_reduction", 100., "A/P variance reduction (%)"),
                      ("CF", "correction_rms", 1., "C/F correction RMS (logit)")]
    methods = [("aster_cr", "ASTER-CR", "#235b8e", -.12),
               ("random_trace", "Random trace", "#b37129", .12)]
    for ax, (group, metric, scale, label) in zip(axes, specifications):
        for method, display, color, offset in methods:
            rows = [records[(model, method, group)] for model in MODELS]
            center = np.array([float(r[metric])*scale for r in rows])
            low = np.array([float(r[metric+"_ci_low"])*scale for r in rows])
            high = np.array([float(r[metric+"_ci_high"])*scale for r in rows])
            # Draw percentile interval endpoints directly; the point estimate need not
            # mathematically lie inside a bootstrap percentile interval.
            ax.hlines(y+offset, low, high, color=color, linewidth=1.3)
            ax.scatter(center, y+offset, color=color, s=28, label=display, zorder=3)
        ax.axvline(0, color="0.65", lw=.8)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="0.92", linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, LABELS)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Independent model fits on the same controlled MEGA holdout", fontsize=11)
    fig.text(.5, .025, "Source-cluster paired bootstrap: 95% intervals. Actual C/F costs are not matched between methods.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.075,1,.94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT/("controlled_feature_comparison."+extension), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
