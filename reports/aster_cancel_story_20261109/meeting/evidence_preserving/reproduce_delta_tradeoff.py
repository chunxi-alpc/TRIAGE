#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the measured-anchored illustrative delta interpolation and ER-VR plot.

Measured anchor values are taken from paper_story_latest.zh.md, section 8.2.
Interpolated rows are NOT model runs or statistical confidence intervals.
Requires: numpy scipy matplotlib.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
AP_FIT_RMS = 2.923349027297149
# tau, measured VR%, measured ER%, measured CF correction RMS, measured correct natural nodes
actual = [
    (0, 0.00, 100.00, 0.0000, 196),
    (.05, 2.79, 99.94, .1141, 196),
    (.10, 4.58, 99.82, .2029, 196),
    (.25, 9.50, 99.49, .5057, 196),
    (.50, 15.55, 98.74, .9056, 195),
    (1.00, 24.04, 97.49, 1.4691, 195),
    (2.00, 34.88, 95.85, 2.3533, 195),
]
unconstrained = dict(delta=float("inf"), tau=float("inf"), vr=40.71, er=94.81,
                     cf_rms=3.0869, natural="195/199", source="实测（无约束）")
illustrative_deltas = [.05, .10, .20, .35, .40, .50, .60, .75, 1.00]
tau_obs = np.asarray([r[0] for r in actual])
interp = [PchipInterpolator(tau_obs, [r[i] for r in actual]) for i in (1,2,3)]
rows = [
    dict(delta=t*AP_FIT_RMS, tau=t, vr=v, er=e, cf_rms=c,
         natural=f"{n}/199", source="实测")
    for t,v,e,c,n in actual
]
for delta in illustrative_deltas:
    tau = delta / AP_FIT_RMS
    v,e,c = [float(f(tau)) for f in interp]
    rows.append(dict(delta=delta,tau=tau,vr=v,er=e,cf_rms=c,
                     natural="未测",source="模拟插值"))
rows.sort(key=lambda r:r["delta"])
rows.append(unconstrained)
with (HERE/"delta_scan_measured_and_illustrative.csv").open("w", encoding="utf-8", newline="") as out:
    w=csv.writer(out)
    w.writerow(["delta_logit","tau","VR_percent","ER_percent","CF_correction_RMS_logit",
                "natural_node_correct","provenance"])
    for r in rows:
        w.writerow(["∞" if np.isinf(r["delta"]) else f'{r["delta"]:.4f}',
                    "∞" if np.isinf(r["tau"]) else f'{r["tau"]:.4f}',
                    f'{r["vr"]:.2f}',f'{r["er"]:.2f}',f'{r["cf_rms"]:.4f}',r["natural"],r["source"]])

fig,ax=plt.subplots(figsize=(10,6.1),dpi=160)
observed=[r for r in rows if r["source"].startswith("实测")]
simulated=[r for r in rows if r["source"]=="模拟插值"]
finite=observed[:-1]
ax.plot([r["er"] for r in finite],[r["vr"] for r in finite],
        "-o",lw=2.2,label="Measured, finite budgets")
ax.plot([finite[-1]["er"],observed[-1]["er"]],
        [finite[-1]["vr"],observed[-1]["vr"]],"--",alpha=.7)
ax.scatter([r["er"] for r in simulated],[r["vr"] for r in simulated],
           marker="D",facecolors="none",edgecolors="tab:orange",
           s=60,label="Illustrative interpolation (not rerun)")
ax.set(xlabel="Evidence-response retention ER (%)",ylabel="Nuisance variance reduction VR (%)",
       title="Delta budget: measured anchors and illustrative interpolations",
       xlim=(94,100.7),ylim=(-2,45))
ax.grid(alpha=.22)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES/"07_delta_er_vr_tradeoff_reproduced.svg")
fig.savefig(FIGURES/"07_delta_er_vr_tradeoff_reproduced.png",dpi=180)
print("Rows:",len(rows),"Measured:",len(observed),"Illustrative:",len(simulated))
