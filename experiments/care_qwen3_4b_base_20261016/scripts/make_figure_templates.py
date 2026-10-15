#!/usr/bin/env python3
"""Make *empty* academic chart layouts. Never substitute for empirical figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT=Path(__file__).resolve().parent.parent / "figures" / "templates"


def seal(fig):
    fig.text(.5,.48,"NO EMPIRICAL DATA\nRun Codex experiments before paper use",ha="center",va="center",
             fontsize=15, weight="bold",alpha=.22,rotation=0,
             bbox={"boxstyle":"round,pad=0.7","facecolor":"white","alpha":.75,"edgecolor":"none"})


def save(fig, name):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT/f"{name}.png",dpi=220,bbox_inches="tight",facecolor="white")
    fig.savefig(OUTPUT/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    ax[0].set(xlim=(0,1),ylim=(0,1),xlabel="Applicable-conflict accuracy",ylabel="Nonapplicable-conflict accuracy",title="(a) Faithfulness–misapplication trade-off")
    ax[1].set(xlim=(-.5,.5),ylim=(0,3),xlabel="Paired change relative to baseline",ylabel="Metric / method",title="(b) Paired method effects")
    for a in ax:a.grid(alpha=.12)
    fig.suptitle("O1 | Conditional-faithfulness hypothesis: plot layout only")
    seal(fig);save(fig,"O1_layout_ONLY")

    fig,ax=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
    for a,title in zip(ax,["(a) Relevance-shift PCA","(b) Applicability-shift PCA","(c) Layer × KV-head heatmap"]):
        a.set(xlabel="PC1 / KV head",ylabel="PC2 / layer",title=title)
        a.grid(alpha=.11)
    fig.suptitle("O2 | Key-space comparison: plot layout only")
    seal(fig);save(fig,"O2_layout_ONLY")

    fig,ax=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
    for a,x,y,title in zip(ax,["Alignment z","Signed weight ω","KV head"],
                           ["Count","Actual local Δa","Layer"],
                           ["(a) z distribution","(b) Weight vs logit change","(c) Mismatch heatmap"]):
        a.set(xlabel=x,ylabel=y,title=title)
        a.grid(alpha=.12)
    fig.suptitle("O3 | Key-editing sign consistency: plot layout only")
    seal(fig);save(fig,"O3_layout_ONLY")
    print("Created clearly labeled EMPTY figure templates in",OUTPUT)


if __name__=="__main__":main()
