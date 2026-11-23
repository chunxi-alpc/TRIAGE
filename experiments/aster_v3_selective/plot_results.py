"""Render the retrospective S2/S3 report figure from archived tabular results."""
from __future__ import annotations

import csv
import gzip
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2] / "reports/aster_v3_selective_20261124"


def main():
    counts = Counter()
    source_ids, input_ids = set(), set()
    with gzip.open(ROOT / "s2_events/input_events.csv.gz", "rt", encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if r["scope"] == "dataset_a_plus_natural":
                counts[r["method"], r["outcome"]] += 1
                source_ids.add(r["source_id"])
                input_ids.add(r["uid"])

    with (ROOT / "s1/s3_metrics.csv").open(encoding="utf-8") as stream:
        metrics = [r for r in csv.DictReader(stream) if r["rule"] == "ALL"]
    aggregate = {(r["split"], r["method"]): r for r in metrics if r["source"] == "ALL"}
    active_sources = {r["source"] for r in metrics if r["source"] != "ALL" and r["split"] == "fit"
                      and r["method"] == "verifier" and int(r["covered"]) > 0}
    dev = aggregate["dev", "verifier"]
    train = aggregate["fit", "verifier"]
    methods = ("E1_pair_only", "J_zero")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False,
                         "axes.spines.right":False, "axes.spines.left":False, "axes.spines.bottom":False})
    fig, (events, rules) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios":[1, 1.24]})
    fig.subplots_adjust(left=.13, right=.965, top=.71, bottom=.24, wspace=.61)
    ink, teal, rust = "#202a39", "#14776e", "#b25139"

    for ax in (events, rules):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe3e6", linewidth=.8)
        ax.tick_params(axis="both", length=0, pad=7, colors=ink)
        ax.set_ylim(1.55, -.55)
    events.set_title("Candidate decision changes", loc="left", fontweight="bold", color=ink, fontsize=16, pad=36)
    events.text(0, 1.07, f"{len(input_ids):,} training inputs from {len(source_ids)} sources",
                transform=events.transAxes, color=ink)
    largest = max(counts[m,k] for m in methods for k in ("benefit","harm"))
    for base, name in enumerate(methods):
        for delta, key, label, color in ((-.18,"benefit","Corrected",teal),(.18,"harm","Introduced error",rust)):
            value = counts[name,key]
            events.barh(base+delta, value, height=.25, color=color, label=label if base == 0 else None)
            events.text(value+.035*max(1,largest),base+delta,str(value),va="center",color=ink)
    events.set_yticks([0,1],methods)
    events.set_xlim(0,max(1,largest)*1.28)
    events.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    events.set_xlabel("Changed inputs", labelpad=8)
    events.legend(frameon=False, loc="upper left", bbox_to_anchor=(-.01,-.27),
                  ncols=2, handlelength=1.1, fontsize=10)

    rules.set_title("Rule verification on covered cases", loc="left", fontweight="bold",
                    color=ink, fontsize=16, pad=36)
    rules.text(0, 1.07, f"{int(train['covered'])} covered training nodes from {len(active_sources)} sources",
               transform=rules.transAxes, color=ink)
    for y, method in enumerate(("simple","verifier")):
        r=aggregate["fit",method]
        total, correct = int(r["covered"]), int(r["covered_correct"])
        fraction = correct/total
        rules.barh(y,1, height=.27, color="#e8ecee")
        rules.barh(y,fraction,height=.27,color="#54718c" if y == 0 else teal)
        rules.text(fraction-.014,y,f"{correct}/{total}",va="center",ha="right",color="white",weight="bold")
        rules.text(0,y+.32,f"{r['B']} corrected, {r['H']} introduced errors",color=ink,va="center",fontsize=10)
    rules.set_xlim(0,1.01)
    rules.set_yticks([0,1], ["Simple threshold","Rule verifier"])
    rules.set_xticks([0,.5,1], ["0%","50%","100%"])
    rules.set_xlabel("Accuracy on covered training nodes", labelpad=8)
    rules.text(0,-.37,f"Development coverage: {int(dev['covered']):,} / {int(dev['n']):,} nodes",
               transform=rules.transAxes, color=ink, fontweight="bold")
    dest = ROOT / "figures/results.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=165, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


if __name__ == "__main__":
    main()
