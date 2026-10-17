#!/usr/bin/env python3
"""Analyze complete REAL inference journals, with paired family uncertainty."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from module2_subspaces import save_json


def read_rows(path):
    path=Path(path)
    if not path.exists():
        path=Path(str(path)+".gz")
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def estimate(data, value, base):
    groups = sorted({r["group_id"] for r in data})
    nums = np.array([sum(value(r) for r in data if r["group_id"] == g) for g in groups])
    dens = np.array([sum(r["group_id"] == g for r in data) for g in groups])
    diffs = np.array([sum(value(r)-value(base[r["case_id"]]) for r in data if r["group_id"] == g) for g in groups])
    rng = np.random.default_rng(261017)
    samples = rng.integers(0, len(groups), size=(10000, len(groups)))
    n, d, delta = nums[samples].sum(1), dens[samples].sum(1), diffs[samples].sum(1)
    return dict(n=len(data), families=len(groups), estimate=float(nums.sum()/dens.sum()),
                ci95=np.quantile(n/d,[.025,.975]).tolist(), delta=float(diffs.sum()/dens.sum()),
                delta_ci95=np.quantile(delta/d,[.025,.975]).tolist())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--artifacts",type=Path,default=Path("/home/gcx/care_20261016/CARE_Codex_Pilot"))
    args=p.parse_args()
    statuses=[json.loads((root/"status.json").read_text()) for root in args.run]
    assert all(s["status"]=="completed_real_model_inference" for s in statuses)
    assert len({s["generalization"] for s in statuses})==1, "Do not pool O2 with clinical data"
    status=statuses[0]
    fingerprints=[json.loads((root/"run_fingerprint.json").read_text()) for root in args.run]
    for field in ("data_sha256","model_files","seka_commit","ranks","strengths","locations","source_sha256"):
        assert all(x[field]==fingerprints[0][field] for x in fingerprints), ("Incompatible parallel jobs",field)
    allrows=[]
    seen_folds=set()
    for root in args.run:
        for folder in sorted(root.glob("fold*")):
            assert (folder/"complete.json").exists(), folder
            fold=int(folder.name[4:])
            assert fold not in seen_folds, "Duplicate outer folds"
            seen_folds.add(fold)
            meta=json.loads((folder/"operators.json").read_text())
            for r in read_rows(folder/"predictions.jsonl"):
                if r["role"]=="test":
                    r["method"]="baseline" if r["config"]=="baseline" else meta[r["config"]]["method"]
                    allrows.append(r)
    keys=[(r["case_id"],r["method"],r["mode"]) for r in allrows]
    assert len(set(keys))==len(keys), "Duplicate outer-test examples"
    baseline={r["case_id"]:r for r in allrows if r["method"]=="baseline"}
    assert baseline
    screening={}
    if (args.artifacts/"results/obs1_clinical/prior_screen.csv").exists() and status["generalization"]=="held-out clinical families":
        with (args.artifacts/"results/obs1_clinical/prior_screen.csv").open(newline="") as stream:
            screening={r["case_id"]:r for r in csv.DictReader(stream)}
    metrics,errors=[],[]
    definitions=[
        ("accuracy",lambda r:True,lambda r:r["correct"]),
        ("app_accuracy",lambda r:r["applicable"],lambda r:r["correct"]),
        ("non_accuracy",lambda r:not r["applicable"],lambda r:r["correct"]),
        ("misuse",lambda r:not r["applicable"],lambda r:r["misapplied"]),
        ("gold_margin",lambda r:True,lambda r:r["gold_margin"]),
    ]
    subsets=[("all",lambda r:True)]
    if screening:
        subsets += [
            ("eligible",lambda r:screening[r["case_id"]]["eligible"]=="True"),
            ("stable_agreement",lambda r:screening[r["case_id"]]["stable"]=="True" and screening[r["case_id"]]["eligible"]!="True"),
        ]
    combinations=sorted({(r["method"],r["mode"]) for r in allrows})
    for method,mode in combinations:
        rows=[r for r in allrows if (r["method"],r["mode"])==(method,mode)]
        assert {r["case_id"] for r in rows}==baseline.keys(), "incomplete method test population"
        for subset,include in subsets:
            for metric,select,value in definitions:
                d=[r for r in rows if include(r) and select(r)]
                if d:
                    metrics.append(dict(method=method,mode=mode,subset=subset,metric=metric,**estimate(d,value,baseline)))
        pairs={}
        bpairs={}
        for r in rows:
            pairs.setdefault(r["pair_id"],[]).append(r)
            bpairs.setdefault(r["pair_id"],[]).append(baseline[r["case_id"]])
        assert all(len(v)==2 and {r["applicable"] for r in v}=={True,False} for v in pairs.values())
        paired=[dict(case_id=pid,pair_id=pid,group_id=v[0]["group_id"],
                     correct=int(all(r["correct"] for r in v)),
                     complete_eligible=bool(screening) and all(screening[r["case_id"]]["eligible"]=="True" for r in v)) for pid,v in pairs.items()]
        bpaired={pid:dict(correct=int(all(r["correct"] for r in v))) for pid,v in bpairs.items()}
        for subset,select in [("all",lambda r:True),("complete_eligible",lambda r:r["complete_eligible"])]:
            d=[r for r in paired if select(r)]
            if d:
                metrics.append(dict(method=method,mode=mode,subset=subset,metric="pair_accuracy",**estimate(d,lambda r:r["correct"],bpaired)))
        for r in rows:
            b=baseline[r["case_id"]]
            if r["prediction"]!=b["prediction"]:
                errors.append(dict(method=method,mode=mode,case_id=r["case_id"],group_id=r["group_id"],
                    gold=r["gold_answer"],before=b["prediction"],after=r["prediction"],
                    corrected=int(r["correct"] and not b["correct"]),harmed=int(not r["correct"] and b["correct"]),
                    margin_delta=r["gold_margin"]-b["gold_margin"]))
    out=args.output or args.run[0]/"analysis"
    out.mkdir(exist_ok=True)
    save_json(out/"metrics.json",dict(status="completed_saved_real_output_analysis",metrics=metrics,
        bootstrap="10000 paired clinical-family resamples conditional on fitted outer folds; O2 groups share one grammar",
        patch_interpretation="Opposite-applicability and same-label donors are oracle diagnostics, not deployable decisions",
        limitations=["B2 uses official projectors with explicit equal-key-norm calibration; it is a controlled adaptation of native SEKA.",
                     "Same-label patch numeric distances are recorded per example and may not match; stratify before causal claims."]))
    if errors:
        with (out/"answer_changes.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(errors[0]))
            writer.writeheader();writer.writerows(errors)
    for metric in ["pair_accuracy","gold_margin","misuse"]:
        data=[r for r in metrics if r["subset"]=="all" and r["metric"]==metric and r["mode"] in ("amplify","uniform_bias","baseline")]
        fig,ax=plt.subplots(figsize=(8,4))
        for i,r in enumerate(data):
            lo,hi=r["delta_ci95"]
            ax.errorbar(i,r["delta"],yerr=[[max(0,r["delta"]-lo)],[max(0,hi-r["delta"])]],fmt="o",capsize=3)
        ax.axhline(0,color=".6",ls="--")
        ax.set(xticks=range(len(data)),xticklabels=[r["method"]+"/"+r["mode"] for r in data],
               ylabel="Change from baseline",title=metric+" | paired family bootstrap 95% CI")
        ax.tick_params(axis="x",labelrotation=65)
        fig.tight_layout()
        for ext in ("png","svg","pdf"):
            fig.savefig(out/(metric+"."+ext),dpi=300)
        plt.close(fig)
if __name__=="__main__":
    main()
