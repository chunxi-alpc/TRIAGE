"""Render the role table and redraw archived pilot measurements.

Figure 2 uses the bundled publication plot renderer from the
engineering-figure-banana skill. No model fitting is performed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

import plot_publication_figure as publication

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

def save(fig, name):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(OUT / (name + "." + extension), dpi=220,
                    bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

def roles():
    fig, ax = plt.subplots(figsize=(11.4, 5.5))
    ax.set_xlim(0, 1100)
    ax.set_ylim(540, 0)
    ax.set_axis_off()
    def box(x, y, w, h, fill, edge="#CBD2DA"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                               edgecolor=edge, linewidth=0.8))
    def txt(x, y, text, size=13, weight="normal", color="#243447"):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=size, fontweight=weight, color=color,
                linespacing=1.7)
    box(12, 8, 1076, 85, "#EFF4FA")
    txt(550, 34, "当前节点：eGFR ≥ 30 mL/min/1.73m²", 16, "bold")
    txt(550, 69, "患者 eGFR：34       原判断真值：是", 14)
    txt(415, 129, "保持真值 · 分数应稳定", 14, "bold")
    txt(851, 129, "翻转真值 · 分数随证据变化", 14, "bold")
    box(12, 158, 170, 135, "#F5F6F8")
    box(12, 302, 170, 135, "#F5F6F8")
    txt(97, 225, "指南信息", 15, "bold")
    txt(97, 369, "患者信息", 15, "bold")
    box(192, 158, 436, 135, "#E8F0F9")
    box(640, 158, 448, 135, "#F8E6DF")
    box(192, 302, 436, 135, "#E8F0F9")
    box(640, 302, 448, 135, "#F8E6DF")
    txt(410, 189, "A · 后续处置", 15, "bold")
    txt(410, 242, "改写分支后的处置文字\n当前条件与患者事实保持", 13)
    txt(864, 189, "C · 条件定义", 15, "bold")
    txt(864, 242, "阈值：30 → 45\n患者数值保持 34；真值变为否", 13)
    txt(410, 333, "P · 无关背景", 15, "bold")
    txt(410, 386, "改写与当前节点无关的背景\n当前条件与决定性事实保持", 13)
    txt(864, 333, "F · 决定性事实", 15, "bold")
    txt(864, 386, "患者数值：34 → 26\n条件阈值保持 30；真值变为否", 13)
    txt(550, 475, "信息角色由当前节点确定。", 13)
    txt(550, 508, "F0：证据变量在阈值同侧变化，单独作为边界测试。", 12)
    save(fig, "fig01_pair_roles")

def pilot(spec):
    fig, axes = publication.create_subplots(1, 2, tuple(spec["layout"]["figsize"]))
    rendered = []
    for ax, panel in zip(axes, spec["panels"]):
        publication.render_panel(ax, panel, fig, rendered)
        rendered.append(ax)
    fig.tight_layout(pad=spec["layout"]["tight_layout_pad"])
    save(fig, "fig02_pilot_tradeoff")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=Path, help="Optional local CJK font.")
    args = parser.parse_args()
    spec = json.loads((HERE / "fig02_spec.json").read_text(encoding="utf-8"))
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        name = font_manager.FontProperties(fname=str(args.font)).get_name()
        spec["style"]["font_family"] = [name, "DejaVu Sans"]
    publication.apply_publication_style(publication.build_style(spec))
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["pdf.fonttype"] = 42
    roles()
    pilot(spec)
    print("Rendered figures 1 and 2 in SVG, PDF, and PNG.")

if __name__ == "__main__":
    main()
