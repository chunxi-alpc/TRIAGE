#!/usr/bin/env python3
"""CPU-only independent arithmetic and provenance recheck of completed module-two journals."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

EXPERIMENT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(EXPERIMENT/"module2"))
from analyze import validate_journals

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def recheck(run, compare=None):
    mode, methods, baseline, protocol, manifests=validate_journals(run)
    original=read(EXPERIMENT.parents[1]/"experiments/care_qwen3_4b_base_20261016/results/manifest.json")
    assert digest(EXPERIMENT.parents[1]/"experiments/care_qwen3_4b_base_20261016/results/manifest.json")==protocol["model_manifest_sha256"]
    folds=[]
    for fold,m in enumerate(manifests):
        d=run/f"fold{fold}"
        train=read(d/"training_audit.json")
        assert m["model_weights"]=={k:v["sha256"] for k,v in original["weights"].items()}
        assert train["original_projector_max_error"]<=1e-6 and train["original_mean_max_error"]<=1e-6
        assert set(train["train_case_ids"]).isdisjoint(m["test_case_ids"])
        assert train["no_test_labels_in_direction_fit"] is True
        assert m["zero_check_forwards"]==m["cases"]+10
        folds.append({"fold":fold,"cases":m["cases"],"records":m["records"],
            "actual_forwards":m["actual_model_forwards"],"zero_forwards":m["zero_check_forwards"],
            "no_op_records":m["no_eligible_no_op_records"],
            "projector_error":train["original_projector_max_error"],"mean_direction_error":train["original_mean_max_error"],
            "seconds":m["elapsed_seconds"],"peak_reserved_gib":m["peak_reserved_GiB"],
            "physical_gpu":m["physical_gpu_visibility"]})
    primary={name:[r for r in rs if mode!="sealed_mechanism" or r["variant"]=="canonical"] for name,rs in methods.items()}
    families=sorted({r["group_id"] for r in primary["baseline"]})
    weights=np.random.default_rng(protocol["bootstrap_seed"]).multinomial(len(families),
        np.full(len(families),1/len(families)),size=protocol["bootstrap_replicates"])
    metrics={}; bootstrap={}
    for name,rs in primary.items():
        counts=[]
        for family in families:
            pairs=defaultdict(list)
            for r in rs:
                if r["group_id"]==family: pairs[r["pair_id"]].append(r)
            assert pairs and all(len(p)==2 for p in pairs.values())
            counts.append([sum(all(r["correct"] for r in p) for p in pairs.values()),len(pairs)])
        counts=np.asarray(counts)
        draws=weights@counts
        bootstrap[name]=draws[:,0]/draws[:,1]
        all_rows=methods[name]
        audits=[r["disturbance"] for r in all_rows] if name!="baseline" else []
        metrics[name]={"primary_paired_correct":int(counts[:,0].sum()),"primary_pairs":int(counts[:,1].sum()),
            "primary_paired_accuracy":float(counts[:,0].sum()/counts[:,1].sum()),
            "primary_case_correct":sum(r["correct"] for r in rs),"primary_cases":len(rs),
            "primary_case_accuracy":sum(r["correct"] for r in rs)/len(rs),
            "all_cases":len(all_rows),"no_op_all_cases":sum(a["calibration_status"]=="no_eligible_direction" for a in audits),
            "grid_fallback_all_cases":sum(a.get("search_method")=="bisection_then_fixed_grid" for a in audits),
            "maximum_relative_error":max([a["relative_error"] for a in audits if a["relative_error"] is not None],default=None),
            "mean_retained_token_fraction":float(np.mean([a["retained_token_fraction"] for a in audits])) if audits else None,
            "mean_anchor_wrong_fraction":float(np.mean([a["anchor_wrong_fraction"] for a in audits])) if audits else None}
    contrasts={}
    for name in ["b4_calibrated","mean_calibrated"]:
        delta=bootstrap[name]-bootstrap["direct_bias"]
        contrasts[name+"-direct_bias"]={"delta":metrics[name]["primary_paired_accuracy"]-metrics["direct_bias"]["primary_paired_accuracy"],
            "ci975":np.percentile(delta,[1.25,98.75]).tolist(),"ci95":np.percentile(delta,[2.5,97.5]).tolist()}
    comparison=None
    if compare:
        old={}
        for fold in range(3):
            for r in rows(compare/f"fold{fold}/predictions.jsonl"):
                old[(r["case_id"],r["method"])]=r
        new={(r["case_id"],r["method"]):r for rs in methods.values() for r in rs}
        assert old.keys()==new.keys()
        checked=("prediction","correct","candidate_logits","candidate_probabilities","gold_logit_margin","gold_probability_margin")
        changed=[{"case_id":k[0],"method":k[1],"changed_fields":[f for f in checked if old[k][f]!=new[k][f]]}
            for k in new if any(old[k][f]!=new[k][f] for f in checked)]
        comparison={"previous_run":str(compare),"records_compared":len(new),"changed_records":changed,
            "all_predictions_and_candidate_scores_identical":not changed}
    return {"status":"independently_verified_completed_real_inference","mode":mode,
        "run_root":str(run),"cases":len(baseline),"families":len(families),"methods":len(methods),
        "records":sum(m["records"] for m in manifests),"zero_checks":sum(m["zero_check_forwards"] for m in manifests),
        "folds":folds,"model_weights_match_original_manifest":True,
        "identity_environment_zero_coverage_and_budget_checks_passed":True,
        "independent_primary_metrics":metrics,"independent_primary_contrasts":contrasts,"comparison_with_previous_run":comparison,
        "source_journal_sha256":{f"fold{f}":digest(run/f"fold{f}/predictions.jsonl") for f in range(3)},
        "recheck_source_sha256":digest(Path(__file__)),
        "limits":["Applicability and action span are reference diagnostic inputs",
                  "Family intervals condition on fitted old folds, excluding refitting uncertainty",
                  "Synthetic mechanism cases do not establish clinical validity"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--compare-with",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=recheck(args.run_root,args.compare_with)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":result["status"],"mode":result["mode"],"cases":result["cases"],
        "records":result["records"],"zero_checks":result["zero_checks"],
        "primary_contrasts":result["independent_primary_contrasts"]},ensure_ascii=False))

if __name__=="__main__":main()

