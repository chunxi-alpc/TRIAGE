"""Render paired path accuracy differences from the two pooled bootstrap tables."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator


def load_rows(path, delimiter):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def draw(report, cjk_font):
    zh = cjk_font.is_file()
    if zh:
        font_manager.fontManager.addfont(str(cjk_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk_font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": True})
    a_rows = load_rows(report / "A_comparisons.tsv", "\t")
    b_rows = load_rows(report / "b_paired_intervals.csv", ",")
    a = {(r["lhs"] + "-" + r["rhs"], r["level"]): r for r in a_rows}
    b = {(r["comparison"], r["metric"].removesuffix("_accuracy")): r for r in b_rows
         if r["calibration"] == "independent"}
    a_labels = [
        ("J_zero-R0", "零截距 J − 原模型", "Zero-intercept J − raw"),
        ("J_zero-R1", "零截距 J − R1", "Zero-intercept J − R1"),
        ("J_zero-J_shared", "零截距 J − 共享截距 J", "Zero − shared intercept"),
        ("J_zero-J_independent", "零截距 J − 独立截距 J", "Zero − independent intercept"),
        ("J_shared-R0", "共享截距 J − 原模型", "Shared-intercept J − raw"),
    ]
    b_labels = [
        ("Direction-J", "仅方向监督 − J", "Direction only − J"),
        ("Asymmetric-J", "仅非对称界 − J", "Asymmetric bounds only − J"),
        ("J2-Direction", "J2 − 仅方向监督", "J2 − direction only"),
        ("J2-Asymmetric", "J2 − 仅非对称界", "J2 − asymmetric bounds only"),
        ("interaction", "两因素交互", "Interaction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5))
    fig.subplots_adjust(left=.30, right=.97, top=.84, bottom=.14, hspace=.65, wspace=.11)
    blue, orange = "#205B82", "#AC4D20"
    for i, (table, labels, extent, tick) in enumerate(((a, a_labels, 1.56, .5),
                                                       (b, b_labels, .22, .1))):
        for j, (level, color) in enumerate((("natural_path", blue), ("interference_path", orange))):
            ax = axes[i, j]
            for y, (key, _, _) in enumerate(labels):
                row = table[key, level]
                est, lo, hi = (100 * float(row[k]) for k in ("estimate", "low", "high"))
                ax.plot([lo, hi], [y, y], color=color, lw=1.8, solid_capstyle="round")
                ax.scatter(est, y, s=31, color=color, edgecolors="white", linewidths=.45, zorder=3)
            ax.axvline(0, ls="--", lw=.9, color="#626C73", zorder=0)
            ax.set_xlim(-extent, extent)
            ax.set_ylim(len(labels)-.5, -.5)
            ax.set_yticks(range(len(labels)), labels=[r[1 if zh else 2] for r in labels] if j == 0 else [])
            ax.xaxis.set_major_locator(MultipleLocator(tick))
            ax.grid(axis="y", color="#E6E9EB", lw=.55)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#B9C2C7")
            ax.tick_params(axis="y", length=0, pad=8)
            ax.tick_params(axis="x", length=3, color="#B9C2C7")
            if i == 0:
                titles = ("自然完整路径（1,712 条）", "干扰完整路径（5,136 条）") if zh else (
                    "Natural paths (n = 1,712)", "Interference paths (n = 5,136)")
                ax.set_title(titles[j], color=color, fontsize=12, pad=12)
    xpos = .03
    fig.text(xpos, .894, "A   J 的截距对照" if zh else "A   Intercept contrasts", fontsize=13)
    fig.text(xpos, .465, "B   四格消融（独立截距）" if zh else "B   Factorial contrasts (independent intercepts)", fontsize=13)
    fig.text(.63, .064, "完整路径准确率差（百分点）" if zh else "Path accuracy difference (percentage points)",
             ha="center", fontsize=11)
    fig.text(xpos, .01, "点为实测差值，线段为 80 个来源配对重采样 2,000 次的 95% 区间。上下两行横轴刻度不同。" if zh else
             "Points show observed contrasts. Bars show 95% intervals from 2,000 paired resamples of 80 sources. X-axis scales differ by row.",
             color="#454E55", fontsize=8.5)
    fig.savefig(report / "path_contrasts.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/aster_m3_missing_controls_20261123"))
    parser.add_argument("--font", type=Path,
                        default=Path("/home/gcx/aster_prior_relevance_figure_20261123/msyh.ttc"),
                        help="Optional CJK font. English labels are used if unavailable.")
    args = parser.parse_args()
    draw(args.output, args.font)
