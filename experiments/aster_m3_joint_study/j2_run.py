"""Separate exploratory J2, enabled by the completed fixed-intercept LP diagnosis.

All original S/J outputs remain unchanged. New hyperparameters are selected on
the original inner training-source holdout; no outer score selects a candidate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import data,training as tr
from .j2_training import fit_j2
from .evaluation import evaluate,selection_metrics
from .run import keys,method_scores,prefix_scope,git_revision
from experiments.aster_cr_retrain_paired_v2_20261111.training import pair_weights


def cf_arrays(problem, pairs, ell):
    byid={p["pair_id"]:p for p in pairs}
    rows=[byid[key] for key in problem.cf_ids]
    if any(p["source_id"] not in problem.train_sources for p in rows):
        raise tr.ProtocolConflict("J2 direction labels outside training sources")
    raw=np.array([ell[p["right"]]-ell[p["left"]] for p in rows],float)
    label=np.array([p["right_label"]-p["left_label"] for p in rows],float)
    return raw,label,pair_weights(rows,"E1",total=1.)


def run(main_output):
    main=Path(main_output).resolve();out=main/"j2";out.mkdir(parents=True,exist_ok=True)
    if data.read(main/"status.json").get("state")!="complete":
        raise ValueError("Complete original R0/R1/R2/S/J before J2")
    reason=data.read(main/"diagnostics/correction_bounds_summary.json")
    cfg=data.read(main/"configuration.json")
    frozen=dict(exploratory=True,reason=reason,representation="same outer-fold E1 16D trace; linear w",
                dimensions=16,lambda_cf_candidates=[.1,1.,10.],AP_weight="inherit selected J in the same fold",
                regularization=.01,delta=tr.DELTA,signed_CF_lower=-tr.DELTA,signed_CF_upper=2*tr.DELTA,
                constraint="-delta <= (right_label-left_label)*delta_c <= 2*delta",
                CF_direction_loss="balanced weighted softplus(-oriented corrected CF response) / baseline direction loss",
                task_and_AP_normalization="same as original S/J",natural_calibration="same source-filtered natural IDs",
                selection="same frozen inner 48/16; natural then interference then CF; smaller lambda_CF tie break",
                seed=tr.SEED,candidate_configurations=3,complete_fits=20,gpu_devices=[],gpu_memory_gib=0,
                independent_confirmation=False,outer_results_used_for_selection=False,
                history="Designed after original historical-source reevaluation; every J2 comparison exploratory",
                code_commit=git_revision())
    data.write(out/"configuration.json",frozen)
    study=data.StudyData(Path(cfg["historical_root"]),Path(cfg["cases_file"]),Path(cfg["diagnostics_root"]),cfg["path_maps"])
    pooled=dict(views=[],pairs=[],cases=[],path_views=[]);all_scores={};corrections=[];selected=[]
    started=time.monotonic()
    for f in study.folds["folds"]:
        fold=f"fold{f['fold']}";target=out/"models"/fold;sel=out/"selection"/fold
        target.mkdir(parents=True,exist_ok=True);sel.mkdir(parents=True,exist_ok=True)
        data.write(out/"status.json",dict(state="running",fold=fold,code_commit=git_revision(),exploratory=True))
        split=data.read(main/"selection"/fold/"split.json")
        data.write(sel/"split.json",split)
        with np.load(main/"models"/fold/"projected_cache.npz",allow_pickle=False) as z:
            ids=z["ids"].tolist();ell=dict(zip(ids,z["ell"].tolist()));traces=dict(zip(ids,z["traces"].copy()))
        parent=data.read(main/"models"/fold/"J.json");lam_ap=parent["lambda_ap"]
        inner=tr.make_problem(study.natural,study.train_pairs,ell,traces,split["inner_train_sources"],
                              split["inner_validation_sources"]+f["validation_sources"])
        raw,signed,weights=cf_arrays(inner,study.train_pairs,ell)
        scope=study.scope(split["inner_validation_sources"]);inner_ids=keys(scope);candidates=[]
        for lam_cf in (.1,1.,10.):
            model=fit_j2(inner,raw,signed,weights,lam_ap,lam_cf)
            name="J2_lambda_CF_"+str(lam_cf)
            model.update(fold=fold,fit_code_commit=git_revision())
            data.write(sel/(name+".json"),model)
            scores=method_scores(ell,traces,inner_ids,{"J2":model},inner.b0)
            mm=selection_metrics(**scope,scores=scores)
            candidates.append(dict(method="J2",lambda_cf=lam_cf,lambda_ap=lam_ap,
                 metrics=mm["J2_independent"],shared_metrics=mm["J2_shared"],R1=mm["R1"],
                 model_file=name+".json",final_losses=model["final_losses"]))
            data.lines(sel/(name+"_scores.jsonl.gz"),
                       [dict(view_id=k,scores={n:float(ss[k]) for n,ss in scores.items()}) for k in inner_ids])
        generic=tr.choose_candidate([dict(r,lambda_ap=r["lambda_cf"]) for r in candidates],candidates[0]["R1"])
        choice=dict(generic);choice["lambda_cf"]=choice.pop("lambda_ap")
        choice.update(inherited_lambda_ap=lam_ap,criterion=generic["criterion"].replace("lambda","lambda_CF"))
        data.write(sel/"candidates.json",dict(candidates=candidates,selection=choice,**split))
        selected.append(dict(fold=fold,**choice))
        problem=tr.make_problem(study.natural,study.train_pairs,ell,traces,f["train_sources"],f["validation_sources"])
        raw,signed,weights=cf_arrays(problem,study.train_pairs,ell)
        model=fit_j2(problem,raw,signed,weights,lam_ap,choice["lambda_cf"])
        model.update(fold=fold,validation_sources=f["validation_sources"],fit_code_commit=git_revision(),
                     M1_M2_directory=parent["M1_M2_directory"],heads=parent["heads"])
        data.write(target/"J2.json",model)
        for cal in ("shared","independent"):
            data.write(target/("J2_"+cal+"_calibration.json"),dict(b=model["b_"+cal],scheme=cal,
                       fit_ids=problem.natural_ids,fit_sources=problem.natural_sources,
                       train_sources=f["train_sources"],validation_sources=f["validation_sources"]))
        models={name:data.read(main/"models"/fold/(name+".json")) for name in ("S","J","R2")}
        models["J2"]=model
        scope=study.scope(f["validation_sources"]);outer_ids=keys(scope)
        scores=method_scores(ell,traces,outer_ids,models,problem.b0)
        source_by_id={tr.uid(v):v["source_id"] for v in scope["views"]+scope["path_views"]}
        for key in outer_ids:
            corrections.append(dict(fold=fold,view_id=fold+"::"+key,original_uid=key,source_id=source_by_id[key],
                raw_score=ell[key],correction=float(tr.correction(model,traces[key])),
                uncalibrated_score=float(ell[key]+tr.correction(model,traces[key]))))
        scoped,prefixed=prefix_scope(fold,scope,scores)
        for name,rr in scoped.items():pooled[name].extend(rr)
        for name,ss in prefixed.items():all_scores.setdefault(name,{}).update(ss)
    data.write(out/"selected_candidates.json",selected)
    data.lines(out/"corrections.jsonl.gz",corrections)
    result=evaluate(**pooled,scores=all_scores,output_dir=out/"evaluation")
    # Original study point estimates must remain identical in this separate table.
    original=data.read(main/"evaluation/metrics.json")["models"]
    for name in original:
        for metric in ("natural_path_accuracy","interference_path_accuracy","ap_vr","cf_er"):
            if abs(result["models"][name][metric]-original[name][metric])>1e-12:
                raise tr.ProtocolConflict("J2 replay changed a frozen original-study estimate")
    cf_rows=[]
    for r in data.rows(out/"evaluation/pair_predictions.jsonl"):
        if r["method"]!="J2_independent" or r["group"]!="CF":continue
        signed=(r["right_label"]-r["left_label"])*r["correction_difference"]
        cf_rows.append(dict(pair_id=r["pair_id"],source_id=r["source_id"],signed_correction=signed,
            original_budget_exceeded=abs(signed)>tr.DELTA+1e-9,
            harmful_lower_exceeded=signed< -tr.DELTA-1e-9,
            helpful_upper_exceeded=signed>2*tr.DELTA+1e-9,
            new_bound_exceeded=signed< -tr.DELTA-1e-9 or signed>2*tr.DELTA+1e-9))
    data.lines(out/"evaluation/cf_budget_checks.jsonl",cf_rows)
    data.write(out/"cf_budget_summary.json",dict(pairs=len(cf_rows),
        **{key:sum(r[key] for r in cf_rows) for key in ("original_budget_exceeded","harmful_lower_exceeded","helpful_upper_exceeded","new_bound_exceeded")}))
    data.write(out/"status.json",dict(state="complete",complete_fits=20,code_commit=git_revision(),
              seconds=time.monotonic()-started,gpu_devices=[],peak_gpu_memory_gib=0,
              exploratory=True,independent_confirmation=False,original_study_replay=True))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");args=p.parse_args()
    try:run(args.output)
    except Exception as error:
        out=Path(args.output)/"j2";out.mkdir(parents=True,exist_ok=True)
        data.write(out/"status.json",dict(state="failed",error=str(error),exploratory=True,code_commit=git_revision()))
        (out/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise
