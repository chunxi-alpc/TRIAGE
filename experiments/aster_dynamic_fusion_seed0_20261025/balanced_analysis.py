"""Equal 27-candidate development-only comparisons; never reads P3 labels."""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
from reproduce import prepare, predict, metric, ROOT

def configs(kind):
    if kind=="hard_anchor":
        for beta,ts in itertools.product((.1,.2,.3,.4,.5,.6,.7,.8,.9),(.25,.5,1.)):
            yield dict(kind=kind,beta=beta,t0=1.,ts=ts)
        return
    for beta,t0,ts in itertools.product((.5,.75,1.),(1.,2.,4.),(.25,.5,1.)):
        yield dict(kind=kind,beta=beta,t0=t0,ts=ts,method="action_bias_1.0_all",gamma=.25,scale=2.)

def prediction(row,c):
    k=c["kind"]
    if k=="dynamic":
        c={**c,"kind":"static","beta":float(np.clip(c["beta"]+.25*(2*row["_features"]["margin"]-1),0,1))}
    if k=="hard_anchor":
        r={**row,"_orig":np.eye(len(row["_orig"]))[int(np.argmax(row["_orig"]))]}
        return predict(r,{**c,"kind":"static"})
    if k=="reverse": c={**c,"kind":"residual","gamma":-c["gamma"]}
    if k=="no_delta": c={**c,"kind":"static"}
    return predict(row,c)

def paired(rows,a,b):
    a=np.asarray(a); b=np.asarray(b); y=np.array([r["gold_option"] for r in rows])
    trees=np.array([r["tree"] for r in rows]); names=sorted(set(trees))
    base=np.array([r["answers"]["original"] if "answers" in r else r["none"] for r in rows])
    counts=[]
    per={}
    for t in names:
        ix=trees==t; n=int(ix.sum())
        fix=int(((a==y)&(b!=y)&ix).sum()); harm=int(((a!=y)&(b==y)&ix).sum())
        dh=int((((a!=y)&(base==y)&ix).sum())-(((b!=y)&(base==y)&ix).sum()))
        counts.append([n,fix-harm,dh])
        per[t]={"n":n,"corrected_vs_comparator":fix,"regressed_vs_comparator":harm,"accuracy_delta":(fix-harm)/n,"original_answer_regression_delta":dh/n}
    x=np.array(counts); rng=np.random.default_rng(20261025)
    boot=x[rng.integers(len(x),size=(10000,len(x)))].sum(axis=1)
    ci=np.quantile(boot[:,1:]/boot[:,:1],[.025,.975],axis=0)
    return {"accuracy_delta":float(np.mean(a==y)-np.mean(b==y)),
      "corrected_vs_comparator":int(((a==y)&(b!=y)).sum()),
      "regressed_vs_comparator":int(((a!=y)&(b==y)).sum()),
      "by_tree":per,"tree_cluster_percentile_ci95":{"accuracy_delta":ci[:,0].tolist(),"original_answer_regression_delta":ci[:,1].tolist()},
      "uncertainty":"Descriptive only: four development clusters; includes all variants per tree; does not account for method-design selection."}

def fit(rows,kind):
    grid=list(configs(kind)); y=np.array([r["gold_option"] for r in rows])
    trees=np.array([r["tree"] for r in rows]); groups=sorted(set(trees))
    matrix=np.array([[prediction(r,c) for r in rows] for c in grid])
    scores=np.array([[(p[trees==t]==y[trees==t]).mean() for t in groups] for p in matrix])
    oof=np.empty(len(rows),int); folds={}; matched=np.empty(len(rows),int)
    for j,t in enumerate(groups):
        best=int(np.argmax(np.delete(scores,j,axis=1).mean(axis=1))); ix=trees==t
        oof[ix]=matrix[best,ix]
        matched[ix]=[prediction(r,{**grid[best],"kind":"no_delta"}) for r in rows if r["tree"]==t]
        folds[t]={"parameters":grid[best],"training_tree_macro_accuracy":float(np.delete(scores,j,axis=1)[best].mean()),
                  "metrics":metric([r for r in rows if r["tree"]==t],oof[ix])}
    best=int(np.argmax(scores.mean(axis=1)))
    return {"candidate_count":len(grid),"metrics":metric(rows,oof),"folds":folds,
            "selected_on_all_development":grid[best],"predictions":oof.tolist(),
            "same_parameters_without_residual":matched.tolist()}

def analyze(rows):
    names=["static","dynamic","hard_anchor"]
    if "scores" in rows[0]: names+=["residual","numeric_null","option_rotation","sign_flip","reverse"]
    result={k:fit(rows,k) for k in names}
    baselines={"original":[int(np.argmax(r["_orig"])) for r in rows],
               "direct_map":[int(np.argmax(r["_support"])) for r in rows],
               "fixed_fusion_075":[predict(r,dict(kind="static",beta=.75)) for r in rows]}
    result["baselines"]={k:metric(rows,np.array(v)) for k,v in baselines.items()}
    result["paired_vs_static"]={k:paired(rows,result[k]["predictions"],result["static"]["predictions"]) for k in names if k!="static"}
    if "residual" in result:
        result["matched_residual"]=paired(rows,result["residual"]["predictions"],result["residual"]["same_parameters_without_residual"])
        result["matched_controls"]={}
        for kind in ("numeric_null","option_rotation","sign_flip","reverse"):
            pred=[prediction(r,{**result["residual"]["folds"][r["tree"]]["parameters"],"kind":kind}) for r in rows]
            result["matched_controls"][kind]={"metrics":metric(rows,np.array(pred)),
                "paired_aligned_vs_control":paired(rows,result["residual"]["predictions"],pred)}
        result["zero_control"]={}
        scores=[r["scores"] for r in rows]
        for zero in ("mass_zero_control","zero_all","zero_last_half"):
            if zero in scores[0]:
                result["zero_control"][zero]={"changed_predictions":sum(int(np.argmax(s[zero]))!=int(np.argmax(s["original"])) for s in scores),
                    "maximum_option_logit_error":max(float(np.max(np.abs(np.array(s[zero])-s["original"]))) for s in scores)}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--fresh-root",type=Path);ap.add_argument("--output",type=Path,default=ROOT/"balanced_cached.json")
    a=ap.parse_args(); start=time.time()
    result={"scope":"development only, exploratory comparisons","grid":"27 candidates per family; identical beta/t0/ts grid for soft fusion families; hard anchor uses nine beta values and three support temperatures. Dynamic margin amplitude .25; residual full-layer strength 1, gamma .25, scale 2 fixed before fresh results.",
        "labels":"Counterfactual rule-following labels, not clinical recommendations.","models":{},"inputs":{}}
    for model in ("qwen","llama"):
        src=a.fresh_root/model/"predictions.jsonl" if a.fresh_root else ROOT/"data"/f"{model}_dev_seed0_{'logits_numeric' if model=='qwen' else 'numeric'}.jsonl"
        raw=[json.loads(x) for x in src.read_text().splitlines()]
        if len(raw)!=800 or len({(r["id"],r["variant"]) for r in raw})!=800: raise ValueError("incomplete development records")
        rows=[prepare(r) for r in raw]
        result["inputs"][model]={"sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"rows":800}
        result["models"][model]=analyze(rows)
    result["analysis_wall_seconds"]=time.time()-start
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print({m:{k:v["metrics"]["accuracy"] for k,v in x.items() if isinstance(v,dict) and "metrics" in v} for m,x in result["models"].items()})
if __name__=="__main__": main()
