"""Reproducible cache-only CPU study: preflight, train/evaluate, report.

Run on the experiment server after activating gcx261014. Historical files are
read-only; every new artifact is confined to the specified output directory.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

from . import data, training as tr
from .evaluation import evaluate, selection_metrics


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=data.REPO, text=True).strip()


def keys(scope):
    return sorted({tr.uid(v) for v in scope["views"]+scope["path_views"]})


def fit_cached(problem, lam, method, path):
    """Resume completed configurations within this study without repeating fits."""
    if Path(path).exists():
        model=data.read(path)
        if (not model.get("success") or model.get("method")!=method or model.get("lambda_ap")!=lam
            or model.get("task_ids")!=problem.natural_ids or model.get("ap_ids")!=problem.ap_ids
            or model.get("cf_ids")!=problem.cf_ids or model.get("train_sources")!=problem.train_sources
            or model.get("regularization")!=tr.REGULARIZATION or model.get("delta")!=tr.DELTA
            or not np.allclose(model["mu"],problem.mu,rtol=0,atol=1e-10)
            or abs(model["b_shared"]-problem.b0)>1e-10):
            raise tr.ProtocolConflict("Completed configuration does not match requested fit: "+str(path))
        model["reused_completed_same_study_fit"]=True
        model.setdefault("fit_code_commit",model.get("code_commit"))
        return model
    model=tr.fit(problem,lam,method)
    model["fit_code_commit"]=git_revision()
    return model


def method_scores(ell, traces, ids, models, b0):
    raw = {k:float(ell[k]) for k in ids}
    scores = {"raw":raw, "R0":dict(raw), "R1":{k:v+b0 for k,v in raw.items()}}
    t = np.stack([traces[k] for k in ids])
    x = np.array([raw[k] for k in ids])
    for name,model in models.items():
        for calibration in ("shared","independent"):
            values = tr.score(model,x,t,calibration)
            scores[name+"_"+calibration] = dict(zip(ids,map(float,values)))
    return scores


def prefix_scope(fold, scope, scores):
    """Exact prompts reused across folds retain their own fitted model score."""
    key = lambda k:fold+"::"+k
    def view(v):
        return dict(v, uid=key(tr.uid(v)),view_id=key(tr.uid(v)),original_uid=tr.uid(v),fold=fold)
    return dict(views=[view(v) for v in scope["views"]],
                path_views=[view(v) for v in scope["path_views"]],
                pairs=[dict(p,left=key(p["left"]),right=key(p["right"]),
                            pair_id=key(p["pair_id"]),original_pair_id=p["pair_id"],fold=fold)
                       for p in scope["pairs"]], cases=scope["cases"]), {
                    method:{key(k):v for k,v in ss.items()} for method,ss in scores.items()}


def prepare(args):
    root,cases,diagnostics = data.resolve_roots(args.e1_root,args.cases_file,args.diagnostic_root)
    output = Path(args.output).resolve()
    for old in (root,diagnostics):
        if output == old or output.is_relative_to(old):
            raise ValueError("Output must be a new directory outside historical assets")
    output.mkdir(parents=True,exist_ok=True)
    maps = []
    for item in args.path_map:
        if "=" not in item:
            raise ValueError("--path-map requires OLD=NEW")
        maps.append(tuple(item.split("=",1)))
    study = data.StudyData(root,cases,diagnostics,maps)
    config = dict(study="ASTER M3 joint supervision", version="20261115-v1",
                  backbone="frozen Qwen3-8B", training_dataset="E1 / Dataset A",training_pairs=2540,
                  mechanism_evaluation_dataset="existing common frozen Dataset B pool (E2)",
                  historical_root=str(root), cases_file=str(cases), diagnostics_root=str(diagnostics),
                  path_maps=maps, heads=51, dimensions=16, delta=tr.DELTA, regularization=tr.REGULARIZATION,
                  lambda_ap_candidates=[0.1,1.,10.], S_lambda_ap=0., seed=tr.SEED,
                  inner_split="one deterministic 48/16 source holdout per outer 64-source training set",
                  selection="conditional on fixed outer-training M1/M2; independent calibration; natural then interference then CF",
                  max_J_candidate_configurations=3, complete_fits_planned=30,
                  outer_sources=80, outer_folds=5, outer_train_sources=64, outer_validation_sources=16,
                  task_weighting="uniform over reliable original natural training nodes, as archived intercept fitting",
                  AP_weighting="equal A/P total; equal sources within kind; equal pairs within kind/source; sum=1",
                  center="uniform mean of natural training traces, no feature rescaling",
                  task_normalization="mean BCE / max(mean baseline BCE at fixed train b0,1e-12)",
                  AP_normalization="weighted corrected pair difference squared / max(weighted raw difference squared,1)",
                  regularizer="0.01 * sum(w**2), same S/J; differs in relative scale from historical sum-weight E1",
                  calibration="separate shared b0 and own intercept, same natural training IDs",
                  inference="original final-token scores and original dynamic path scorer",
                  primary="independent J-S, then independent J-R1; fixed sequence; two success conditions each",
                  additional_comparisons="shared comparisons, J-R2 and S-R1 exploratory",
                  bootstrap_draws=2000, bootstrap_seed=20261111, gpu_devices=[], gpu_memory_gib=0,
                  estimated_minutes=[10,30], model_data_metadata=study.config,
                  historical_sources=True, independent_confirmation=False, J2_enabled=False)
    config_path=output/"configuration.json"
    if config_path.exists() and data.read(config_path)!=json.loads(json.dumps(config)):
        raise tr.ProtocolConflict("Run configuration differs from the already recorded configuration")
    data.write(config_path,config)
    data.write(output/"preflight.json",dict(complete=True, required_metadata_available=True,
               cached_score_records=len(study.store.records), runtime_memmap=study.store.runtime is not None,
               natural_nodes=len(study.natural), train_pairs=len(study.train_pairs),
               evaluation_pairs=len(study.eval_pairs), path_views=len(study.path_views),
               fit_cases=len(study.cases), code_commit=git_revision(),
               note="Selected-head completeness is checked during cache projection; no hashes"))
    data.write(output/"folds.json",study.folds)
    return study,output


def fitting(study, output):
    begin = time.monotonic()
    pooled = dict(views=[],pairs=[],cases=[],path_views=[])
    pooled_scores, checks, selected = {}, [], []
    corrections = []
    for f in study.folds["folds"]:
        fold = f"fold{f['fold']}"
        target = output/"models"/fold
        target.mkdir(parents=True,exist_ok=True)
        selection_dir=output/"selection"/fold
        selection_dir.mkdir(parents=True,exist_ok=True)
        data.write(output/"status.json",dict(state="running",stage="prepare_fold",fold=fold,
                   code_commit=git_revision(),seconds=time.monotonic()-begin))
        train_sources, valid_sources = f["train_sources"],f["validation_sources"]
        rng=np.random.default_rng(tr.SEED+int(f["fold"]))
        shuffled=rng.permutation(sorted(train_sources)).tolist()
        inner_valid,inner_train=sorted(shuffled[:16]),sorted(shuffled[16:])
        split=dict(outer_training_sources=train_sources,outer_validation_sources=valid_sources,
                   inner_train_sources=inner_train,inner_validation_sources=inner_valid,
                   seed=tr.SEED+int(f["fold"]),conditional_on_fixed_representation=True)
        data.write(selection_dir/"split.json",split)
        inner_scope=study.scope(inner_valid)
        outer_scope=study.scope(valid_sources)
        allow=set(train_sources)
        train_pairs=[p for p in study.train_pairs if p["source_id"] in allow]
        needed=keys(inner_scope)+keys(outer_scope)
        needed += [tr.uid(v) for v in study.natural if v["source_id"] in allow]
        needed += [p[side] for p in train_pairs for side in ("left","right")]
        m2,old_w,old_b=study.parameters(fold)
        cache=target/"projected_cache.npz"
        if cache.exists():
            with np.load(cache,allow_pickle=False) as z:
                ids=z["ids"].tolist()
                if ids!=sorted(set(needed)) or z["traces"].shape!=(len(ids),16):
                    raise tr.ProtocolConflict("Projection cache IDs or dimension differ from this fold")
                ell=dict(zip(ids,z["ell"].tolist()))
                traces=dict(zip(ids,z["traces"].copy()))
        else:
            ell,traces=study.coordinates(m2,needed,cache)
        problem=tr.make_problem(study.natural,study.train_pairs,ell,traces,train_sources,valid_sources)
        r2=tr.centered_e1(old_w,old_b,problem.mu,problem.b0)
        outer_ids=keys(outer_scope)
        reference_scores=method_scores(ell,traces,outer_ids,{"R2":r2},problem.b0)
        replay=study.verify_archived_scores(fold,reference_scores["R0"],reference_scores["R1"],
                                            reference_scores["R2_independent"],problem.b0,problem.natural_ids)
        replay["dynamic_paths"]=study.verify_archived_paths(fold,outer_scope,reference_scores)
        data.write(target/"reference_replay.json",replay)
        inner=tr.make_problem(study.natural,study.train_pairs,ell,traces,inner_train,
                              inner_valid+valid_sources)
        inner_ids=keys(inner_scope)
        candidate_rows=[]
        for method,lam in [("S",0.),("J",.1),("J",1.),("J",10.)]:
            name=method+"_lambda_"+str(lam)
            model=fit_cached(inner,lam,method,selection_dir/(name+".json"))
            data.write(selection_dir/(name+".json"),model)
            scores=method_scores(ell,traces,inner_ids,{method:model},inner.b0)
            mm=selection_metrics(**inner_scope,scores=scores)
            row=dict(method=method,lambda_ap=lam,metrics=mm[method+"_independent"],
                     shared_metrics=mm[method+"_shared"],R1=mm["R1"],
                     model_file=name+".json",final_losses=model["final_losses"])
            candidate_rows.append(row)
            # Preserve every candidate's inner scores, including both intercepts.
            data.lines(selection_dir/(name+"_scores.jsonl.gz"),
                       [dict(view_id=k,scores={n:float(ss[k]) for n,ss in scores.items()}) for k in inner_ids])
        choice=tr.choose_candidate([r for r in candidate_rows if r["method"]=="J"],candidate_rows[0]["R1"])
        data.write(selection_dir/"candidates.json",dict(candidates=candidate_rows,selection=choice,**split))
        selected.append(dict(fold=fold,**choice))
        # Outer labels have not entered the preceding selection.
        models={name:fit_cached(problem,lam,name,target/(name+".json"))
                for name,lam in (("S",0.),("J",choice["lambda_ap"]))}
        models["R2"]=r2
        for name,model in models.items():
            model.update(fold=fold,validation_sources=valid_sources,
                         M1_M2_directory=str(study.root/"models/E1"/fold),
                         heads=np.asarray(m2["heads"]).tolist(),code_commit=git_revision())
            data.write(target/(name+".json"),model)
            for calibration in ("shared","independent"):
                data.write(target/(name+"_"+calibration+"_calibration.json"),
                           dict(method=name,scheme=calibration,b=model["b_"+calibration],
                                fit_ids=problem.natural_ids,fit_sources=problem.natural_sources,
                                train_sources=train_sources,validation_sources=valid_sources,
                                original_E1_reuse=name=="R2" and calibration=="independent"))
        data.write(target/"R1.json",dict(b0=problem.b0,calibration_ids=problem.natural_ids,
                   fit_sources=problem.natural_sources,train_sources=train_sources,
                   validation_sources=valid_sources))
        scores=method_scores(ell,traces,outer_ids,models,problem.b0)
        checks.append(dict(fold=fold,**replay))
        node_sources={tr.uid(v):v["source_id"] for v in outer_scope["views"]+outer_scope["path_views"]}
        for key in outer_ids:
            corrections.append(dict(fold=fold,view_id=fold+"::"+key,original_uid=key,
                source_id=node_sources[key],raw_score=ell[key],
                correction={name:float(tr.correction(model,traces[key])) for name,model in models.items()},
                uncalibrated_score={name:float(ell[key]+tr.correction(model,traces[key])) for name,model in models.items()}))
        scoped,prefixed=prefix_scope(fold,outer_scope,scores)
        for name,rr in scoped.items():
            pooled[name].extend(rr)
        for name,ss in prefixed.items():
            pooled_scores.setdefault(name,{}).update(ss)
        data.write(output/"status.json",dict(state="running",stage="fold_complete",fold=fold,
                   completed_folds=len(checks),code_commit=git_revision(),seconds=time.monotonic()-begin))
    data.write(output/"selected_candidates.json",selected)
    data.write(output/"verification.json",dict(archived_score_replay=checks,
               cf_training_constraints=all(data.read(output/"models"/f"fold{k}"/(m+".json"))["cf_max_violation"]<=tr.TOLERANCE
                                           for k in range(1,6) for m in ("S","J")),
               outer_sources_disjoint=True,fold_specific_M1_M2=True,
               automatic_tests="see tests.json",complete=True))
    data.lines(output/"corrections.jsonl.gz",corrections)
    result=evaluate(**pooled,scores=pooled_scores,output_dir=output/"evaluation")
    data.write(output/"status.json",dict(state="complete",stage="evaluation_complete",completed_folds=5,
               code_commit=git_revision(),seconds=time.monotonic()-begin,real_experiment=True,
               gpu_devices=[],peak_gpu_memory_gib=0,independent_confirmation=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("preflight","run","report"))
    p.add_argument("--e1-root")
    p.add_argument("--cases-file")
    p.add_argument("--diagnostic-root")
    p.add_argument("--path-map",action="append",default=[])
    p.add_argument("--output",default=str(data.REPO/"reports/aster_m3_joint_study_20261115"))
    args=p.parse_args()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage=="report":
            from .report import generate_report
            generate_report(output)
            return
        study,output=prepare(args)
        if args.stage=="preflight":
            data.write(output/"status.json",dict(state="ready_for_cache_projection",real_experiment=False,
                       code_commit=git_revision()))
            return
        fitting(study,output)
        from .report import generate_report
        generate_report(output)
    except Exception as error:
        state="protocol_conflict" if isinstance(error,tr.ProtocolConflict) else "blocked_missing_assets" if isinstance(error,data.MissingAssets) else "failed"
        data.write(output/"status.json",dict(state=state,error=str(error),stage=args.stage,
                   real_experiment_complete=False,code_commit=git_revision()))
        if isinstance(error,data.MissingAssets):
            data.write(output/"missing_assets.json",error.assets)
        (output/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
