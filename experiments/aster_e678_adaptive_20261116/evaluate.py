"""Unchanged metric definitions; point estimates only, no fixed-head experiment."""
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write,rows
from .adaptive_m1 import ScopedStore

class PointEstimateBootstrap(legacy.SourceBootstrap):
    """Keep the legacy aggregation interface with zero resampling draws."""
    def __init__(self,sources,draws=0):
        super().__init__(sources,draws=0)
    def summary(self,values):
        return dict(estimate=legacy.finite(values[0]))

def evaluate(root,scope,store):
    root=Path(root)
    if scope not in ("dev","test"):raise ValueError(scope)
    if scope=="test" and not read(root/"frozen.json").get("frozen"):
        raise ValueError("Test evaluation before parameter freeze")
    spec=dict(views=rows(root/"data"/(scope+"_views.jsonl")),
        pairs=rows(root/"data"/(scope+"_evaluation_pairs.jsonl")),
        path_views=rows(root/"data"/(scope+"_path_views.jsonl")),
        cases=rows(root/"data"/(scope+"_cases.jsonl")))
    ids={v["view_id"] for v in spec["views"]+spec["path_views"]}
    store=ScopedStore(store,ids)
    models={"E1":legacy.model_parameters(root,"E1")}
    legacy._LINEAR_SCORE_CACHE.clear()
    old_boot,old_draws=legacy.SourceBootstrap,legacy.DRAWS
    try:
        legacy.SourceBootstrap,legacy.DRAWS=PointEstimateBootstrap,0
        result=legacy.scope_evaluate(scope,store=store,models=models,out=root,**spec)
    finally:
        legacy.SourceBootstrap,legacy.DRAWS=old_boot,old_draws
    points=result.pop("bootstrap")
    for record in points:
        record.pop("draws",None)
    result.update(method_names=dict(E1="adaptive",raw="frozen_backbone"),
                  development_role="model selection" if scope=="dev" else "held-out sources",
                  parameters_frozen=scope=="test",confidence_intervals=False,
                  fixed_51_comparison=False)
    path_rows=rows(root/"evaluation"/scope/"path_predictions.jsonl")
    lookup={(r["method"],r["case_id"],r["variant"]):r for r in path_rows}
    rr=[r for r in path_rows if r["method"]=="E1" and r["interference"]]
    previous=[lookup[("raw",r["case_id"],r["variant"])] for r in rr]
    for name,vals in (
        ("interference_path_corrected_count",[r["correct"] and not b["correct"] for r,b in zip(rr,previous)]),
        ("interference_path_new_errors",[b["correct"] and not r["correct"] for r,b in zip(rr,previous)])):
        if rr:
            points.append(dict(scope=scope,comparison="E1 minus raw",metric=name,estimate=sum(vals),
                               sources=len({r["source_id"] for r in rr})))
    result["point_estimates"]=points
    write(root/"evaluation"/scope/"metrics.json",result)
    legacy.write_csv(root/"point_estimates.csv",points,replace_scope=scope)
    # scope_evaluate writes this compatibility file; all records have zero draws.
    # Publish the correctly named point-estimate table instead.
    compatibility=root/"source_bootstrap.csv"
    if compatibility.exists():compatibility.unlink()
    return result
