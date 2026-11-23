"""Same pooled-pair moments and source bootstrap as E1; no thresholded correction metric."""
from collections import defaultdict
import numpy as np
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import (
    SourceBootstrap, ratio, finite, path_predictions, uid)

def pair_records(pairs, scores, raw, method):
    records = []
    for p in pairs:
        l, r = p["left"], p["right"]
        yl, yr = int(p["left_label"]), int(p["right_label"])
        a, b = float(scores[l]), float(scores[r])
        d, dp = float(raw[r]-raw[l]), b-a
        records.append(dict(method=method, pair_id=p["pair_id"], source_id=p["source_id"],
            kind=p["kind"], group="AP" if p["kind"] in ("A","P") else
            "CF" if p["kind"] in ("C","F") else p["kind"],
            left=l, right=r, left_label=yl, right_label=yr,
            left_score=a, right_score=b, raw_difference=d, difference=dp,
            correction_difference=dp-d, paired_correct=bool((a>0)==yl and (b>0)==yr),
            relation_correct=bool(((a>0)==(b>0))==(yl==yr)),
            direction_degraded=bool((yr-yl)*d>0 and (yr-yl)*dp<=0)))
    return records

def aggregate(records, nodes, paths, boot):
    metrics, counts = {}, {}
    for group in sorted({r["group"] for r in records}):
        rr = [r for r in records if r["group"]==group]
        raw, adjusted = [np.array([r[k] for r in rr]) for k in ("raw_difference","difference")]
        correction = adjusted-raw
        direction = np.array([r["right_label"]-r["left_label"] for r in rr])
        cols = np.column_stack((raw**2, adjusted**2, correction**2, raw, adjusted,
            [r["paired_correct"] for r in rr], [r["relation_correct"] for r in rr],
            direction*raw, direction*adjusted, [r["direction_degraded"] for r in rr]))
        total, mean, _ = boot.aggregate(cols, [r["source_id"] for r in rr])
        k = group.lower()
        metrics.update({k+"_raw_rms":np.sqrt(mean[:,0]),k+"_score_change_rms":np.sqrt(mean[:,1]),
            k+"_correction_rms":np.sqrt(mean[:,2]),k+"_paired_accuracy":mean[:,5],
            k+"_relation_accuracy":mean[:,6]})
        counts.update({k+"_pairs":len(rr),k+"_sources":len({r["source_id"] for r in rr})})
        if group=="AP":
            metrics["ap_vr"]=1-ratio(mean[:,1]-mean[:,4]**2,mean[:,0]-mean[:,3]**2)
        if group=="CF":
            metrics.update(cf_er=ratio(mean[:,8],mean[:,7]),
                cf_direction_degradation_rate=mean[:,9],
                cf_direction_degradation_count=total[:,9],
                cf_normalized_distortion=np.sqrt(ratio(mean[:,2],mean[:,0])))
    if "ap_vr" in metrics and "cf_normalized_distortion" in metrics:
        metrics["J"]=metrics["ap_vr"]-metrics["cf_normalized_distortion"]**2
    if nodes:
        metrics["natural_node_accuracy"]=boot.mean([n["correct"] for n in nodes],[n["source_id"] for n in nodes])
        counts["natural_node_count"]=len(nodes)
    for interfere in (False,True):
        pp=[p for p in paths if p["interference"]==interfere]
        k="interference_path" if interfere else "natural_path"
        if pp:
            metrics[k+"_accuracy"]=boot.mean([p["correct"] for p in pp],[p["source_id"] for p in pp])
            counts[k+"_count"]=len(pp)
            counts[k+"_unavailable_count"]=sum(p.get("error")=="unavailable_score" for p in pp)
    return metrics, counts

def predict(scope, scores, raw, method):
    prs=pair_records(scope["pairs"],scores,raw,method)
    ns=[dict(method=method,view_id=uid(v),source_id=v["source_id"],label=v["label"],
        score=float(scores[uid(v)]),correct=bool((scores[uid(v)]>0)==int(v["label"])))
        for v in scope["natural"]]
    paths=path_predictions(scope["cases"],scope["path_views"],{method:scores})
    if any(p.get("error")=="unavailable_score" for p in paths):
        raise ValueError("Missing cached path score")
    return prs,ns,paths

def point(metrics,counts=None):
    return {**(counts or {}),**{k:finite(v[0]) for k,v in metrics.items()}}

def per_source(records,nodes,paths):
    by=defaultdict(lambda:[[],[],[]])
    for i,rows in enumerate((records,nodes,paths)):
        for row in rows: by[row["source_id"]][i].append(row)
    result=[]
    for source,rr in sorted(by.items()):
        mm,cc=aggregate(*rr,SourceBootstrap([source],draws=0))
        result.append(dict(source_id=source,**point(mm,cc)))
    return result

def selection_key(row):
    return (row["J"],-abs(row["cf_er"]-1),-row["cf_direction_degradation_count"],
            row["cf_paired_accuracy"],row["natural_path_accuracy"],row["regularization"])

def pareto_mask(rows):
    return [not any(b["ap_vr"]>=a["ap_vr"] and
        b["cf_normalized_distortion"]<=a["cf_normalized_distortion"] and
        (b["ap_vr"]>a["ap_vr"] or b["cf_normalized_distortion"]<a["cf_normalized_distortion"])
        for b in rows) for a in rows]
