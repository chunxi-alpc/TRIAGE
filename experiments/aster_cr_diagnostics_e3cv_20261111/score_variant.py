"""Explicit score variants with independent cache fields and calibration provenance."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from .common import *
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev

def fit_intercepts(store):
    natural=rows(BASE/"data/natural_views.jsonl")
    all_sources=read(BASE/"folds.json")["all_sources"]
    definitions=[("full",all_sources)]+[("fold"+str(f["fold"]),f["train_sources"])
                 for f in read(BASE/"folds.json")["folds"]]
    fitted={}
    for fold,sources in definitions:
        allowed=set(sources)
        vv=[v for v in natural if v["source_id"] in allowed and v.get("variant","full")=="full"]
        assert all(v.get("label") is not None for v in vv)
        x=np.array([store.ell(v["view_id"]) for v in vv]);y=np.array([v["label"] for v in vv],float)
        derivative=lambda b:float(np.mean(expit(x+b)-y))
        lo,hi=-1.,1.
        while derivative(lo)>0:lo*=2
        while derivative(hi)<0:hi*=2
        b=float(brentq(derivative,lo,hi,xtol=1e-12))
        fitted[fold]=dict(b0=b,nodes=len(vv),fit_sources=sorted(sources),
            calibration_ids=[v["view_id"] for v in vv],target_task_fit=False,
            objective="mean binary logistic loss with original score as fixed offset",
            experiment_version=VERSION,group="shared",fold=fold,derivative=derivative(b))
    assert fitted["full"]["nodes"]==761
    write(OUT/"independent_intercepts.json",fitted)
    return fitted

def parameters(fold="full"):
    return {g:ev.model_parameters(model_root(g,fold),g,fold) for g in GROUPS}

def seed_existing_scores(models,scopes):
    # Reuse stored full-method scores only in the full-method cache.
    for scope in scopes:
        path=BASE/"evaluation"/scope/"predictions.jsonl"
        for r in rows(path):
            for g,model in models.items():
                if g in r["scores"]:
                    ev._LINEAR_SCORE_CACHE.setdefault(model["directory"],{})[r["view_id"]]=r["scores"][g]

def score_variant(raw,full,original_b,b0,variant):
    if variant=="raw":return raw
    if variant=="independent_intercept":return raw+b0
    if variant=="pair_only":return full-original_b
    if variant=="original_intercept_only":return raw+original_b
    if variant=="full":return full
    raise ValueError("Unknown score_variant: "+variant)

def scores_for(store,views,models,intercept):
    full=ev.score_views(store,views,models);raw=full["raw"];b0=intercept["b0"]
    scores={"raw":raw,"independent_intercept":{k:v+b0 for k,v in raw.items()}}
    descriptors={"raw":dict(group="shared",score_variant="raw"),
                 "independent_intercept":dict(group="shared",score_variant="independent_intercept")}
    for g,m in models.items():
        for variant in ("pair_only","original_intercept_only","full"):
            name=g+"_"+variant
            scores[name]={k:float(score_variant(raw[k],full[g][k],m["b"],b0,variant)) for k in raw}
            descriptors[name]=dict(group=g,score_variant=variant,original_b=m["b"],
                model_directory=m["directory"])
    for v in descriptors.values():
        v.update(experiment_version=VERSION,fold=intercept["fold"],b0=b0,
            intercept_fit_sources=intercept["fit_sources"])
    return scores,descriptors
