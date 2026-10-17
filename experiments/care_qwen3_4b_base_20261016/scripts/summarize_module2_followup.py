#!/usr/bin/env python3
"""Final saved-data analysis: matched method comparisons, causal diagnostics and packaging.

No model inference; runs on the experiment server. All secondary contrasts are
exploratory because the saved primary results were already available.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_module2_followup import estimate
from module2_subspaces import save_json

METHODS=["baseline","B1","B2","B3","B4","random42","random43","random44"]
MODES=["amplify","uniform_bias","ablate","patch","patch_same_label"]

def read_rows(path):
    path=Path(path)
    if not path.exists():
        path=Path(str(path)+".gz")
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def family_sign_flip(group_sums):
    values=np.array(group_sums,dtype=float)
    if np.all(values==0):
        return 1.
    signs=np.array(list(itertools.product((-1.,1.),repeat=len(values))))
    null=np.abs(signs@values)
    observed=abs(values.sum())
    return float(np.mean(null>=observed-1e-12))

def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    adjusted=[0.]*len(pvalues)
    previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,pvalues[i]*(len(pvalues)-rank)))
        adjusted[i]=previous
    return adjusted

def pair_records(rows):
    pairs={}
    for r in rows:
        pairs.setdefault(r["pair_id"],[]).append(r)
    assert all(len(v)==2 and {r["applicable"] for r in v}=={True,False} for v in pairs.values())
    return [dict(case_id=pid,pair_id=pid,group_id=v[0]["group_id"],
                 correct=int(all(r["correct"] for r in v))) for pid,v in sorted(pairs.items())]

def load_dataset(run, clinical):
    roots=[run/f"clinical_fold{f}" for f in range(3)] if clinical else [run/"o2"]
    allrows=[]
    selections={}
    audit=[]
    for root in roots:
        status=json.loads((root/"status.json").read_text())
        assert status["status"]=="completed_real_model_inference"
        for folder in sorted(root.glob("fold*")):
            meta=json.loads((folder/"operators.json").read_text())
            info=json.loads((folder/"split.json").read_text())
            sets=[set(info[k]) for k in ("train","dev","test")]
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            selection=json.loads((folder/"selection.json").read_text())
            selections[root.name]=selection
            blocked=json.loads((folder/"blocked_configurations.json").read_text())
            audit.append(dict(run=root.name,status=status,journal_records=0,configurations=len(meta),
                              blocked=blocked,split=info))
            for row in read_rows(folder/"predictions.jsonl"):
                audit[-1]["journal_records"]+=1
                if row["role"]=="test":
                    row["method"]="baseline" if row["config"]=="baseline" else meta[row["config"]]["method"]
                    assert row["group_id"] in sets[2]
                    if row["method"]!="baseline":
                        chosen=selection[row["method"]]
                        assert row["config"]==chosen["config"] and row["epsilon"]==chosen["epsilon"]
                    allrows.append(row)
    ids=[(r["method"],r["mode"],r["case_id"]) for r in allrows]
    assert len(set(ids))==len(ids)
    return allrows,selections,audit

def contrast(rows, left, right, metric="pair_accuracy"):
    data=[r for r in rows if (r["method"],r["mode"])==left]
    other=[r for r in rows if (r["method"],r["mode"])==right]
    value=lambda r:r["correct"] if metric=="pair_accuracy" else r["gold_margin"]
    if metric=="pair_accuracy":
        data,other=pair_records(data),pair_records(other)
    lookup={r["case_id"]:r for r in other}
    assert data and {r["case_id"] for r in data}==lookup.keys()
    result=estimate(data,value,lookup)
    groups=sorted({r["group_id"] for r in data})
    sums=[sum(value(r)-value(lookup[r["case_id"]]) for r in data if r["group_id"]==g) for g in groups]
    return dict(left="/".join(left),right="/".join(right),metric=metric,
                p_family_sign_flip=family_sign_flip(sums),**result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--artifacts",type=Path,default=Path("/home/gcx/care_20261016/CARE_Codex_Pilot"))
    args=p.parse_args()
    run=args.run_root
    output=run/"final_analysis"
    output.mkdir(exist_ok=True)
    clinical,selections,audit=load_dataset(run,True)
    o2,o2selection,o2audit=load_dataset(run,False)
    baseline={r["case_id"]:r for r in clinical if r["method"]=="baseline"}
    clinical_cases={r["case_id"]:r for r in read_rows(args.artifacts/"data/clinical/candidates.jsonl")}
    # Verify the new contextual baseline against archived native O1 predictions.
    with (args.artifacts/"results/obs1_clinical/predictions.csv").open(newline="") as stream:
        archived={r["case_id"]:r for r in csv.DictReader(stream) if r["role"]=="test" and r["method"]=="baseline"}
    assert set(baseline)==archived.keys() and len(baseline)==384
    baseline_changes=sum(baseline[k]["prediction"]!=archived[k]["prediction"] for k in baseline)
    baseline_probability_error=max(abs(a-b) for k in baseline for a,b in zip(
        baseline[k]["candidate_probabilities"],json.loads(archived[k]["candidate_probabilities"])))
    assert baseline_changes==0
    primary=json.loads((run/"clinical_analysis/metrics.json").read_text())["metrics"]
    table=[]
    for method in METHODS:
        mode="baseline" if method=="baseline" else "amplify"
        metrics={r["metric"]:r for r in primary if r["method"]==method and r["mode"]==mode and r["subset"]=="all"}
        pair=metrics["pair_accuracy"]
        rows=[r for r in clinical if (r["method"],r["mode"])==(method,mode)]
        table.append(dict(method=method,case_accuracy=metrics["accuracy"]["estimate"],
            pair_accuracy=pair["estimate"],pair_change=pair["delta"],
            pair_change_ci_low=pair["delta_ci95"][0],pair_change_ci_high=pair["delta_ci95"][1],
            misuse=metrics["misuse"]["estimate"],app_accuracy=metrics["app_accuracy"]["estimate"],
            non_accuracy=metrics["non_accuracy"]["estimate"],
            corrected=sum(r["correct"] and not baseline[r["case_id"]]["correct"] for r in rows),
            harmed=sum(not r["correct"] and baseline[r["case_id"]]["correct"] for r in rows),
            changed_answers=sum(r["prediction"]!=baseline[r["case_id"]]["prediction"] for r in rows)))
    write_csv(output/"clinical_primary.csv",table)
    # Prespecified for this SECONDARY analysis after inspecting the primary output.
    comparisons=[
        (("B3","amplify"),("B2","amplify")),
        (("B4","amplify"),("B2","amplify")),
        (("B4","amplify"),("B1","amplify")),
        (("B4","amplify"),("B3","amplify")),
    ]+[(("B4","amplify"),(method,"amplify")) for method in ("random42","random43","random44")]
    comparisons += [((method,"amplify"),(method,"uniform_bias")) for method in ("B1","B2","B3","B4")]
    compared=[contrast(clinical,a,b) for a,b in comparisons]
    adjusted=holm([r["p_family_sign_flip"] for r in compared])
    for row,padj in zip(compared,adjusted):
        row["p_holm_secondary_11_comparisons"]=padj
    write_csv(output/"clinical_paired_comparisons.csv",compared)
    causal=[]
    for method in METHODS[1:]:
        value=contrast(clinical,(method,"patch"),(method,"patch_same_label"),"gold_margin")
        opposite={r["case_id"]:r for r in clinical if (r["method"],r["mode"])==(method,"patch")}
        sham=[r for r in clinical if (r["method"],r["mode"])==(method,"patch_same_label")]
        matched=[r for r in sham if r["disturbance"]["opposite_pair_distance"]>0 and
                 abs(r["disturbance"]["numeric_distance"]/r["disturbance"]["opposite_pair_distance"]-1)<=.15]
        donor_shifts=[]
        for r in opposite.values():
            g=clinical_cases[r["disturbance"]["donor_case_id"]]["gold_label"]
            recipient=clinical_cases[r["case_id"]]
            letters=[letter for letter,label in recipient["option_map"].items() if label==g]
            assert len(letters)==1
            donor_index="ABCD".index(letters[0])
            own_index="ABCD".index(r["gold_answer"])
            before=baseline[r["case_id"]]["candidate_probabilities"]
            after=r["candidate_probabilities"]
            donor_shifts.append(dict(case_id=r["case_id"],group_id=r["group_id"],
                shift=(after[donor_index]-after[own_index])-(before[donor_index]-before[own_index])))
        zero={r["case_id"]:dict(shift=0.) for r in donor_shifts}
        directed=estimate(donor_shifts,lambda r:r["shift"],zero)
        value.update(method=method,matched_numeric_control_cases=len(matched),control_cases=len(sham),
                     matched_control_families=len({r["group_id"] for r in matched}),
                     donor_directed_margin_change=directed["delta"],
                     donor_directed_ci_low=directed["delta_ci95"][0],
                     donor_directed_ci_high=directed["delta_ci95"][1])
        if matched:
            # Same selected operator per fold; restrict to rows with comparable numeric distances.
            data=[opposite[r["case_id"]] for r in matched]
            result=estimate(data,lambda r:r["gold_margin"],{r["case_id"]:r for r in matched})
            value.update(matched_patch_minus_sham=result["delta"],
                         matched_ci_low=result["delta_ci95"][0],matched_ci_high=result["delta_ci95"][1])
        causal.append(value)
    write_csv(output/"clinical_causal_controls.csv",causal)
    # Case-level and family-level errors; preserve semantic labels and both methods' probabilities.
    changes=[]
    family=[]
    for method in METHODS[1:]:
        rows=[r for r in clinical if (r["method"],r["mode"])==(method,"amplify")]
        for group in sorted({r["group_id"] for r in rows}):
            d=[r for r in rows if r["group_id"]==group]
            ps=pair_records(d)
            bs={r["case_id"]:r for r in pair_records([baseline[r["case_id"]] for r in d])}
            family.append(dict(method=method,group_id=group,n_cases=len(d),
                case_accuracy=float(np.mean([r["correct"] for r in d])),
                pair_accuracy=float(np.mean([r["correct"] for r in ps])),
                baseline_pair_accuracy=float(np.mean([r["correct"] for r in bs.values()]))))
        for r in rows:
            b=baseline[r["case_id"]]
            if r["prediction"]!=b["prediction"]:
                source=clinical_cases[r["case_id"]]
                changes.append(dict(method=method,case_id=r["case_id"],pair_id=r["pair_id"],
                    group_id=r["group_id"],rule_id=source["rule_id"],applicable=r["applicable"],
                    value=source["value"],gold=source["gold_label"],
                    baseline=source["option_map"][b["prediction"]],intervention=source["option_map"][r["prediction"]],
                    corrected=int(r["correct"] and not b["correct"]),harmed=int(not r["correct"] and b["correct"]),
                    gold_margin_change=r["gold_margin"]-b["gold_margin"]))
    write_csv(output/"clinical_main_answer_changes.csv",changes)
    write_csv(output/"clinical_family_metrics.csv",family)
    # O2 nonzero sensitivity keeps the original epsilon=0 main result intact.
    nonzero=[]
    nonzero_path=run/"o2_nonzero/predictions.jsonl"
    if nonzero_path.exists():
        nz=read_rows(nonzero_path)
        meta=json.loads((run/"o2/fold0/operators.json").read_text())
        for r in nz:
            r["method"]="baseline" if r["config"]=="baseline" else meta[r["config"]]["method"]
        base={r["case_id"]:r for r in nz if r["method"]=="baseline"}
        for method,mode in sorted({(r["method"],r["mode"]) for r in nz}):
            rows=[r for r in nz if (r["method"],r["mode"])==(method,mode)]
            ps=pair_records(rows)
            bp={r["case_id"]:r for r in pair_records(list(base.values()))}
            nonzero.append(dict(method=method,mode=mode,**estimate(ps,lambda r:r["correct"],bp)))
        write_csv(output/"o2_nonzero_pair_accuracy.csv",nonzero)
    # Main chart: absolute accuracy plus paired change, from the saved tables only.
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    x=np.arange(len(table))
    axes[0].bar(x,[100*r["pair_accuracy"] for r in table],
        color=["#777777","#0072B2","#56B4E9","#009E73","#D55E00","#aaaaaa","#bbbbbb","#cccccc"])
    axes[0].set(ylabel="Both members correct (%)",ylim=(0,60),title="Clinical: 192 pairs / 12 held-out families")
    for i,r in enumerate(table):
        axes[0].text(i,100*r["pair_accuracy"]+.25,f'{100*r["pair_accuracy"]:.2f}',ha="center",fontsize=8)
        if i:
            axes[1].errorbar(i,100*r["pair_change"],
                yerr=[[100*max(0,r["pair_change"]-r["pair_change_ci_low"])],
                      [100*max(0,r["pair_change_ci_high"]-r["pair_change"])]],fmt="o",capsize=3,color="#0072B2")
    axes[1].axhline(0,color=".6",ls="--")
    axes[1].set(ylabel="Change from baseline (percentage points)",title="Paired family bootstrap 95% CI")
    for ax in axes:
        ax.set_xticks(x,[r["method"] for r in table],rotation=40,ha="right")
    fig.tight_layout()
    for ext in ("png","svg","pdf"):
        fig.savefig(output/f"clinical_subspace_comparison.{ext}",dpi=300)
    plt.close(fig)
    # All methods and modes, including causal controls; intervals come from saved metrics.
    for metric in ("pair_accuracy","gold_margin"):
        fig,ax=plt.subplots(figsize=(10,4))
        for j,mode in enumerate(("amplify","uniform_bias","ablate","patch","patch_same_label")):
            d=[r for r in primary if r["subset"]=="all" and r["metric"]==metric and r["mode"]==mode]
            lookup={r["method"]:r for r in d}
            offsets=np.arange(7)+(j-2)*.14
            vals=[lookup[m]["delta"] for m in METHODS[1:]]
            low=[max(0,lookup[m]["delta"]-lookup[m]["delta_ci95"][0]) for m in METHODS[1:]]
            high=[max(0,lookup[m]["delta_ci95"][1]-lookup[m]["delta"]) for m in METHODS[1:]]
            ax.errorbar(offsets,vals,yerr=[low,high],fmt="o",markersize=4,capsize=2,label=mode)
        ax.axhline(0,color=".5",ls="--")
        ax.set(xticks=np.arange(7),xticklabels=METHODS[1:],ylabel="Change from baseline",title=metric+" | clinical family bootstrap 95% CI")
        ax.legend(frameon=False,ncol=3,fontsize=8)
        fig.tight_layout()
        for ext in ("png","svg","pdf"):
            fig.savefig(output/f"clinical_all_modes_{metric}.{ext}",dpi=300)
        plt.close(fig)
    groups=sorted({r["group_id"] for r in family})
    matrix=np.array([[100*(next(r for r in family if r["method"]==m and r["group_id"]==g)["pair_accuracy"]-
                           next(r for r in family if r["method"]==m and r["group_id"]==g)["baseline_pair_accuracy"])
                      for g in groups] for m in METHODS[1:]])
    bound=max(6.25,float(np.abs(matrix).max()))
    fig,ax=plt.subplots(figsize=(11.8,4.5))
    im=ax.imshow(matrix,cmap="RdBu_r",vmin=-bound,vmax=bound,aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]:
                ax.text(j,i,f"{matrix[i,j]:+.2f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(matrix[i,j])>.6*bound else "black")
    ax.set(yticks=np.arange(7),yticklabels=METHODS[1:],xticks=np.arange(len(groups)),
           xticklabels=[g.replace("_"," ") for g in groups],
           title="Pair accuracy change by held-out clinical family")
    ax.tick_params(axis="x",labelrotation=45)
    plt.setp(ax.get_xticklabels(),ha="right")
    fig.colorbar(im,ax=ax,label="Change from baseline (percentage points)")
    fig.tight_layout()
    for ext in ("png","svg","pdf"):
        fig.savefig(output/f"clinical_family_pair_changes.{ext}",dpi=300)
    plt.close(fig)
    data=dict(status="completed_real_results_secondary_analysis",
        baseline_matches_archived_predictions=baseline_changes==0,
        maximum_baseline_probability_difference=baseline_probability_error,
        primary=table,paired_comparisons=compared,causal_controls=causal,
        selections=selections,o2_primary_selection=o2selection,o2_nonzero=nonzero,
        execution=audit+o2audit,total_main_journal_records=sum(x["journal_records"] for x in audit+o2audit),
        limitations=["Exploratory secondary analysis after primary test outputs were available",
        "Family bootstrap and sign-flip tests condition on fitted outer folds; shared training is not resampled",
        "11 secondary pair-accuracy comparisons use Holm adjustment",
        "Random seeds have independent dev-selected heads/ranks; not an identical-head/rank intervention control",
        "Uniform bias is norm-matched to each selected key method, not independently development-tuned",
        "B2 uses official projectors with explicit disturbance normalization",
        "O2 is a single fictional grammar, not unseen-rule-family generalization",
        "Donor patching uses oracle paired patients; same-label numeric matching is imperfect",
        "BF16 numerical paths are not claimed to be exactly equivalent",
        "Primary grid includes ranks 1/4/16 and four locations, not all layers/heads/ranks",
        "Simulated patients and one fixed model; no clinical efficacy conclusion"])
    save_json(output/"final_summary.json",data)
    print(json.dumps(dict(primary=table,records=data["total_main_journal_records"],
        comparisons=compared,causal=causal,nonzero=nonzero,baseline_probability_error=baseline_probability_error)))
if __name__=="__main__":
    main()
