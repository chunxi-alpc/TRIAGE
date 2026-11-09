"""Render full and low-evidence-cost views from frozen repair results."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def main():
 with (H/"exp4/main_results.csv").open() as f:rows=list(csv.DictReader(f))
 fig,axes=plt.subplots(2,2,figsize=(12,8.2))
 colors={"trace":"#2166ac","random":"#b2182b"}
 for col,split in enumerate(("dev","test")):
  rr=[r for r in rows if r["split"]==split]
  for row in (0,1):
   ax=axes[row,col]
   for family,marker in (("trace","o"),("random","s")):
    bounded=sorted([r for r in rr if r["method"].startswith(family+"_bounded")],key=lambda r:float(r["evidence_RMS"]))
    xx=[0.]+[float(r["evidence_RMS"]) for r in bounded];yy=[0.]+[100*float(r["VR"]) for r in bounded]
    ax.plot(xx,yy,color=colors[family],marker=marker,label=family+" linear",lw=1.2,markersize=5)
    if row==0:
     un=next(r for r in rr if r["method"]==family+"_unconstrained")
     ax.scatter(float(un["evidence_RMS"]),100*float(un["VR"]),c=colors[family],marker=marker,s=45)
     ax.annotate("unconstrained",(float(un["evidence_RMS"]),100*float(un["VR"])),xytext=(-4,6),textcoords="offset points",ha="right",fontsize=8)
    else:
     for r in bounded:
      ax.annotate(r["tau"],(float(r["evidence_RMS"]),100*float(r["VR"])),xytext=(5,4 if family=="trace" else -11),textcoords="offset points",fontsize=7,color=colors[family])
   nn=[r for r in rr if r["method"].startswith("nn_")]
   ax.scatter([float(r["evidence_RMS"]) for r in nn],[100*float(r["VR"]) for r in nn],c="#1b7837",marker="^",s=34,alpha=.8,label="trace neural (3 seeds)")
   ax.scatter([0],[0],c="black",s=40,label="original")
   ax.scatter([0],[0],c="gray",marker="x",s=60,label="threshold calibration")
   ax.grid(alpha=.2);ax.set_axisbelow(True)
   ax.axhline(0,color="gray",lw=.6)
   ax.set_xlabel("Evidence correction RMS (logit units)")
   ax.set_ylabel("Interference variance reduction (%)")
   ax.set_title(split+" (existing holdout)" if row==0 else split+": low evidence cost")
   if row==1:ax.set_xlim(-.025,.65);ax.set_ylim(-.6,12)
 axes[0,0].legend(fontsize=7,loc="upper right")
 fig.tight_layout()
 for suffix in ("png","pdf"):fig.savefig(H/("exp4/repair_tradeoff."+suffix),dpi=180)
 plt.close(fig)
if __name__=="__main__":main()

