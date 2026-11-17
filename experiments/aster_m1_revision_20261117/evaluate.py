"""Four frozen predictions with the existing E6 metrics and no resampling."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, lines
from .prepare import OUT, SOURCE, HIST

class PointSourceAggregation(legacy.SourceBootstrap):
    def __init__(self,sources,draws=0):
        self.sources=sorted(set(sources))
        self.index={s:i for i,s in enumerate(self.sources)}
        self.weights=np.ones((1,len(self.sources)))
        self.draws=0
    def summary(self,values):
        return dict(estimate=legacy.finite(values[0]))

def independent_intercept(natural,store):
    original=[v for v in natural if v.get("variant","full")=="full"]
    if {v["label"] for v in original}!={0,1}:raise ValueError("Both fit labels needed")
    if any(v["split"]!="fit" for v in original):raise ValueError("Non-fit intercept input")
    scores=np.array([store.ell(v["view_id"]) for v in original])
    labels=np.array([v["label"] for v in original])
    derivative=lambda b:float(np.mean(expit(scores+b)-labels))
    lo,hi=-1.,1.
    while derivative(lo)>0:lo*=2
    while derivative(hi)<0:hi*=2
    return dict(b=float(brentq(derivative,lo,hi,xtol=1e-12)),nodes=len(original),
                ids=[v["view_id"] for v in original],sources=sorted({v["source_id"] for v in original}),
                split="fit",uses_correction=False)

def prediction_models(root):
    full=legacy.model_parameters(root,"E1")
    cal=read(root/"independent_intercept.json")
    models={
      "adaptive_full":full,
      "adaptive_no_intercept":dict(full,b=0.,offset=-float(full["mu"]@full["alpha"]),
                                  directory=full["directory"]+"/no_intercept"),
      "independent_intercept":dict(full,alpha=np.zeros_like(full["alpha"]),b=cal["b"],offset=cal["b"],
                                  directory=full["directory"]+"/independent_intercept")}
    for mass in (80,95):
        target=root/"models"/("sensitivity_"+str(mass))/"full/model.json"
        if target.exists():
            models["sensitivity_"+str(mass)]=legacy.model_parameters(root,"sensitivity_"+str(mass))
    return models

def public_names(value):
    if isinstance(value,dict):
        return {("frozen_backbone" if k=="raw" else k):public_names(v) for k,v in value.items()
                if k not in ("draws","defined_draws","low","high")}
    if isinstance(value,list):return [public_names(v) for v in value]
    if isinstance(value,str):
        if value=="raw":return "frozen_backbone"
        if " minus raw" in value:return value.replace(" minus raw"," minus frozen_backbone")
    return value

def clean_result(value):
    value=public_names(value)
    if isinstance(value,dict) and "bootstrap" in value:value["point_estimates"]=value.pop("bootstrap")
    return value

def evaluate(root,scope,store):
    root=Path(root)
    if scope not in ("dev","test"):raise ValueError(scope)
    if not read(root/"fit_frozen.json").get("frozen"):raise ValueError("Fit parameters not frozen")
    if scope=="test" and not read(root/"frozen.json").get("frozen"):
        raise ValueError("Test evaluation before model manifest freeze")
    views=[v for v in rows(SOURCE/"data/historical_views.jsonl") if v["split"]==scope]
    pairs=[p for p in rows(SOURCE/"data/historical_pairs.jsonl") if p["split"]==scope]
    path_views=[v for name in ("path_features.jsonl","path_interference_views.jsonl")
                for v in rows(SOURCE/"data"/name) if v["split"]==scope]
    cases=[c for c in rows(HIST/"data/path_cases.jsonl") if c["split"]==scope]
    models=prediction_models(root)
    legacy._LINEAR_SCORE_CACHE.clear()
    old=(legacy.SourceBootstrap,legacy.DRAWS,legacy.write_csv,legacy.write,legacy.lines)
    def csv_point(path,records,**kwargs):
        path=Path(path)
        if path.name=="source_bootstrap.csv":path=path.with_name("point_estimates.csv")
        return old[2](path,public_names(list(records)),**kwargs)
    try:
        legacy.SourceBootstrap=PointSourceAggregation
        legacy.DRAWS=0
        legacy.write_csv=csv_point
        legacy.write=lambda p,v:old[3](p,clean_result(v))
        legacy.lines=lambda p,rr:old[4](p,public_names(list(rr)))
        result=clean_result(legacy.scope_evaluate(scope,views,pairs,store,models,out=root,
                                                 cases=cases,path_views=path_views))
    finally:
        legacy.SourceBootstrap,legacy.DRAWS,legacy.write_csv,legacy.write,legacy.lines=old
    evaldir=root/"evaluation"/scope
    pair_records=rows(evaldir/"pair_predictions.jsonl")
    for name,metrics in result["models"].items():
        cf=[r for r in pair_records if r["method"]==name and r["group"]=="CF"]
        if cf:
            metrics["cf_max_budget_excess"]=max(max(abs(r["correction_difference"])-r["evidence_budget"],0.) for r in cf)
            metrics["cf_raw_directed_response"]=float(np.mean([(r["right_label"]-r["left_label"])*r["raw_difference"] for r in cf]))
            metrics["cf_directed_response"]=float(np.mean([(r["right_label"]-r["left_label"])*r["difference"] for r in cf]))
            metrics["cf_directed_correction"]=float(np.mean([(r["right_label"]-r["left_label"])*r["correction_difference"] for r in cf]))
    nodes=rows(evaldir/"predictions.jsonl")
    natural=[r for r in nodes if r["kind"]=="original" and r["original"] and r["label_available"]]
    paths=rows(evaldir/"path_predictions.jsonl")
    path_index={(r["method"],r["case_id"],r["variant"]):r for r in paths}
    comparisons=[]
    source_nodes=[]
    for name in result["models"]:
        for baseline in ("frozen_backbone","independent_intercept"):
            if name==baseline:continue
            for scope_name,rr in (("natural_node",natural),
                                   ("natural_path",[r for r in paths if r["method"]==name and not r["interference"]]),
                                   ("interference_path",[r for r in paths if r["method"]==name and r["interference"]])):
                corrected=errors=0
                for r in rr:
                    if scope_name=="natural_node":
                        a=r["predictions"][name]==r["label"];b=r["predictions"][baseline]==r["label"]
                    else:
                        a=r["correct"];b=path_index[(baseline,r["case_id"],r["variant"])]["correct"]
                    corrected+=int(a and not b);errors+=int(b and not a)
                comparisons.append(dict(method=name,baseline=baseline,population=scope_name,
                                        count=len(rr),corrected=corrected,new_errors=errors))
        grouped=defaultdict(list)
        for r in natural:grouped[r["source_id"]].append(r)
        for source,rr in sorted(grouped.items()):
            source_nodes.append(dict(method=name,source_id=source,n=len(rr),
                 correct=sum(r["predictions"][name]==r["label"] for r in rr)))
    lines(evaldir/"corrections.jsonl",comparisons)
    lines(evaldir/"source_node_metrics.jsonl",source_nodes)
    result.update(confidence_intervals=False,
                  development_role="model-selection results" if scope=="dev" else "historically reused 40-source holdout",
                  parameters_frozen=True,correction_comparisons=comparisons)
    write(evaldir/"metrics.json",result)
    return result
