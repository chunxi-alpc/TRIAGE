#!/usr/bin/env python3
"""Plot saved ASTER-CR single-head interventions; never loads a model."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle, Patch

KINDS = ["A", "P", "C", "F"]
LABELS = {
    "A": "Action-text changes",
    "P": "Irrelevant-background changes",
    "C": "Rule-condition changes",
    "F": "Decisive-fact changes",
}
N_LAYERS, N_HEADS = 36, 32
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})
GRAY = "#dddddd"

def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def export(fig, out, name):
    for ext in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(out / (name + "." + ext), dpi=300, facecolor="white",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
    plt.close(fig)

def panel(ax, table, key, kind, selected, norm, cmap, title):
    matrix = np.full((N_LAYERS, N_HEADS), np.nan)
    measured = set()
    for row in table:
        if row["kind"] == kind and row["measured"]:
            matrix[row["layer"], row["query_head"]] = row[key]
            measured.add(row["head"])
    im = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="equal",
                   interpolation="none", cmap=cmap, norm=norm)
    for head in measured.intersection(selected):
        layer, qhead = divmod(head, N_HEADS)
        ax.add_patch(Rectangle((qhead-.45, layer-.45), .9, .9,
                               fill=False, edgecolor="#202020", linewidth=.55))
    ax.set(title=title, xlabel="Query head (0-based)", ylabel="Layer (0-based)")
    ax.set_xticks(range(0, N_HEADS, 4))
    ax.set_yticks(range(0, N_LAYERS, 5))
    ax.tick_params(length=2, labelsize=8)
    return im

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.out
    out.mkdir(parents=True, exist_ok=True)
    exact_file = run / "qwen/mechanisms/fit_exact.jsonl"
    selection_file = run / "qwen/selection.json"
    exact = read_rows(exact_file)
    selection = json.loads(selection_file.read_text())
    selected = set(selection["heads"])
    pair_index = {p["pair_id"]: p for p in read_rows(run/"data/pairs.jsonl")}
    grouped = {}
    for row in exact:
        grouped.setdefault((row["kind"], int(row["head"])), []).append(row)
    table = []
    counts = {}
    for kind in KINDS:
        kind_rows = [r for r in exact if r["kind"] == kind]
        pairs = sorted(set(r["pair_id"] for r in kind_rows))
        sources = set(pair_index[p]["source_id"] for p in pairs)
        counts[kind] = dict(pairs=len(pairs), sources=len(sources),
                           candidate_heads=len(set(r["head"] for r in kind_rows)),
                           measured_selected_heads=len(selected.intersection(
                               r["head"] for r in kind_rows)))
        for head in range(N_LAYERS*N_HEADS):
            records = grouped.get((kind, head), [])
            row = dict(kind=kind, layer=head//N_HEADS, query_head=head%N_HEADS,
                       head=head, selected=int(head in selected), measured=int(bool(records)),
                       pair_count=len(records), weight_sum="", denominator="",
                       forward="", reverse="", mean="", absolute="")
            if records:
                w = np.array([r["weight"] for r in records], dtype=float)
                te = np.array([r["te"] for r in records], dtype=float)
                fwd = np.array([r["ie_forward"] for r in records], dtype=float)
                rev = np.array([r["ie_reverse"] for r in records], dtype=float)
                denom = float(np.sum(w*np.abs(te)))
                forward = float(np.sum(w*np.sign(te)*fwd)/denom)
                reverse = float(np.sum(w*np.sign(te)*(-rev))/denom)
                row.update(weight_sum=float(w.sum()), denominator=denom,
                           forward=forward, reverse=reverse, mean=(forward+reverse)/2,
                           absolute=float(np.sum(w*(np.abs(fwd)+np.abs(rev))*.5)/w.sum()))
            table.append(row)
    with (out/"head_effects.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (out/"source").mkdir(exist_ok=True)
    shutil.copy2(exact_file, out/"source/fit_exact.jsonl")
    shutil.copy2(selection_file, out/"source/selection.json")
    pair_meta = [dict(pair_id=p, kind=pair_index[p]["kind"],
                      split=pair_index[p]["split"],
                      source_id=pair_index[p]["source_id"],
                      group_id=pair_index[p]["group_id"])
                 for p in sorted(set(r["pair_id"] for r in exact))]
    (out/"source/pairs.json").write_text(json.dumps(pair_meta, indent=2)+"\n")
    measured = [r for r in table if r["measured"]]
    signed_limit = max(abs(r[key]) for r in measured for key in ["forward","reverse","mean"])
    signed_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0, vmax=signed_limit)
    signed_cmap = plt.get_cmap("RdBu_r").copy()
    signed_cmap.set_bad(GRAY)
    absolute_limit = max(r["absolute"] for r in measured)
    absolute_norm = Normalize(vmin=0, vmax=absolute_limit)
    absolute_cmap = plt.get_cmap("viridis").copy()
    absolute_cmap.set_bad(GRAY)
    handles = [Patch(facecolor=GRAY, edgecolor="none", label="Not measured"),
               Patch(facecolor="white", edgecolor="#202020", label="Selected head (when measured)")]
    subtitle = "Qwen3-8B | repair-v2 | fit candidates | final judgment position"
    footer = ("64 candidate heads per category; gray cells are unmeasured, not zero.\n"
              "Weights: 1 + any original error + either score margin below 1.")
    for name, key, cm, norm, title, barlabel in [
        ("mean_patch_heatmap", "mean", signed_cmap, signed_norm,
         "ASTER-CR | Bidirectional single-head patching",
         "Effect-weighted patch fraction"),
        ("absolute_patch_heatmap", "absolute", absolute_cmap, absolute_norm,
         "ASTER-CR | Absolute single-head patching effects",
         "Weighted mean absolute effect (logit units)"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 10.4))
        fig.subplots_adjust(left=.075, right=.86, bottom=.17, top=.86,
                            wspace=.24, hspace=.30)
        for i, (ax, kind) in enumerate(zip(axes.flat, KINDS)):
            n = counts[kind]
            im = panel(ax, table, key, kind, selected, norm, cm,
                       f"{chr(97+i)}  {kind}: {LABELS[kind]}\n"
                       f"{n['pairs']} pairs / {n['sources']} sources")
        cax = fig.add_axes([.89, .28, .018, .43])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(barlabel, labelpad=8)
        fig.suptitle(title, y=.965, fontsize=15, weight="bold")
        fig.text(.5, .925, subtitle, ha="center", fontsize=10)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.075),
                   ncol=2, frameon=False, fontsize=9)
        fig.text(.5,.025,footer,ha="center",va="bottom",fontsize=8,color="#444444")
        export(fig,out,name)
    fig, axes = plt.subplots(4, 3, figsize=(12.2, 17.2))
    fig.subplots_adjust(left=.06, right=.89, bottom=.115, top=.92,
                        wspace=.17, hspace=.31)
    for i, kind in enumerate(KINDS):
        n=counts[kind]
        for j,(key,title) in enumerate([("forward","Forward: right to left"),
                                       ("reverse","Reverse: left to right"),
                                       ("mean","Bidirectional mean")]):
            im=panel(axes[i,j],table,key,kind,selected,signed_norm,signed_cmap,
                     f"{chr(97+i*3+j)}  {kind}: {title}")
        axes[i,0].set_ylabel(f"{LABELS[kind]}\n"
                            f"{n['pairs']} pairs | Layer")
    cax=fig.add_axes([.92,.33,.014,.35])
    fig.colorbar(im,cax=cax,label="Effect-weighted patch fraction")
    fig.suptitle("ASTER-CR | Forward and reverse single-head effects",
                 y=.975,fontsize=16,weight="bold")
    fig.text(.5,.945,subtitle,ha="center",fontsize=10)
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.05),
               ncol=2,frameon=False)
    fig.text(.5,.018,
             "Positive values move the judgment score toward the donor run; negative values move away.\n"
             "Forward and reverse are donor-oriented. Fractions are not answer accuracy or necessity.",
             ha="center",fontsize=9,color="#444444")
    export(fig,out,"directional_patch_heatmaps")
    summary=dict(model="Qwen3-8B",method="ASTER-CR",data_version="repair-v2",
                 split="fit",layers=N_LAYERS,query_heads_per_layer=N_HEADS,
                 selected_heads=len(selected),exact_records=len(exact),counts=counts,
                 measured_candidate_union=len(set(r["head"] for r in exact)),
                 shared_signed_color_limit=signed_limit,absolute_color_max=absolute_limit,
                 formulas={
                    "forward":"sum(weight * sign(TE) * IE_forward) / sum(weight * abs(TE))",
                    "reverse":"sum(weight * sign(TE) * (-IE_reverse)) / sum(weight * abs(TE))",
                    "mean":"(forward + reverse) / 2",
                    "absolute":"sum(weight * (abs(IE_forward) + abs(IE_reverse)) / 2) / sum(weight)"
                 },
                 interpretation="Mean and directional fractions are new descriptive summaries; absolute effect matches exact candidate ranking. No new model forward or training.",
                 axes="36 layers x 32 query heads; zero-based",
                 missing_cells="unmeasured, shown in gray; never imputed as zero",
                 color_scaling="global full range for each metric; no percentile clipping")
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(output=str(out),records=len(exact),counts=counts,
                          signed_limit=signed_limit,absolute_limit=absolute_limit)))

if __name__ == "__main__":
    main()

