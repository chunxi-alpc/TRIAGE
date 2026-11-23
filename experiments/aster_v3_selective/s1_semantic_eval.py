"""Evaluate the frozen S3 rule and comparator from cached natural-node scores."""
import csv
import gzip
import time
from collections import defaultdict
from experiments.aster_closed_form.data import condition
from experiments.aster_v3_selective.s1_verifier import verify, definition
from experiments.aster_v3_selective.s1_attribution import ARCH, RAW, OUT, rows, write_csv, write_json


def aggregate(part, method):
    n = len(part)
    scope = [r for r in part if r["rule"]!="other"]
    covered = [r for r in part if r[method]!=""]
    gains = losses = right = wrong = 0
    for row in part:
        y = row["gold"]
        a = row["R0"]
        b = a if row[method]=="" else row[method]
        if a==y and b==y:
            right+=1
        elif a!=y and b!=y:
            wrong+=1
        elif b==y:
            gains+=1
        else:
            losses+=1
    correct = sum(r[method]==r["gold"] for r in covered)
    return dict(n=n, eligible=len(scope), covered=len(covered), abstained=n-len(covered),
        eligible_abstained=len(scope)-len(covered), covered_correct=correct,
        covered_accuracy=correct/len(covered) if covered else "", coverage=len(covered)/n if n else "",
        scope_coverage=len(covered)/len(scope) if scope else "", full_correct=right+gains,
        full_accuracy=(right+gains)/n if n else "", B=gains,H=losses,kept_correct=right,kept_wrong=wrong)


def main():
    start = time.perf_counter()
    data = [r for r in rows(ARCH/"data/path_features.jsonl") if r["split"] in {"fit","dev"} and r["label"] is not None]
    cases = {r["case_id"]:r for r in rows(RAW/"path_cases.jsonl") if r["split"] in {"fit","dev"}}
    uids = {r["uid"] for r in data}
    scores = {}
    for f in [ARCH/"data/reused_scores.jsonl",*sorted((ARCH/"cache").glob("scores_*.jsonl"))]:
        for r in rows(f):
            if r["uid"] in uids and r["status"]=="ok":
                if r["uid"] in scores and scores[r["uid"]]!=r["ell"]:
                    raise ValueError("Raw score identity at "+r["uid"])
                scores[r["uid"]]=r["ell"]
    if set(scores)!=uids:
        raise ValueError(f"Missing {len(uids-set(scores))} cached scores")
    result = []
    for r in data:
        c = cases[r["case_id"]]
        q = condition(c,r["gid"])
        kind,_ = definition(q)
        p,reason,span = verify(q,c["patient"])
        baseline,_,baseline_span = verify(q,c["patient"],simple=True)
        result.append(dict(split=r["split"],source_id=r["source_id"],case_id=r["case_id"],gid=r["gid"],
            uid=r["uid"],rule=kind or "other",R0=int(scores[r["uid"]]>0),gold=int(r["label"]),
            verifier=p if p is not None else "",simple=baseline if baseline is not None else "",
            reason=reason,fact_start=span[0] if span else "",fact_end=span[1] if span else "",
            simple_start=baseline_span[0] if baseline_span else "",simple_end=baseline_span[1] if baseline_span else ""))
    with gzip.open(OUT/"s3_nodes.csv.gz","wt", newline="") as f:
        w=csv.DictWriter(f, lineterminator="\n", fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    metrics=[]
    for split in ["fit","dev"]:
        stage=[r for r in result if r["split"]==split]
        for source in ["ALL"]+sorted({r["source_id"] for r in stage}):
            cohort=[r for r in stage if source=="ALL" or source==r["source_id"]]
            for rule in ["ALL","hb","sbp"]:
                part=[r for r in cohort if rule=="ALL" or rule==r["rule"]]
                for method in ["simple","verifier"]:
                    metrics.append(dict(split=split,source=source,rule=rule,method=method,**aggregate(part,method)))
    write_csv("s3_metrics.csv",metrics)
    write_json("s3_summary.json",dict(summary=[r for r in metrics if r["source"]=="ALL"],
        target_rule_sources=len({r["source_id"] for r in result if r["rule"]!="other"}),
        reused_R0_scores=len(scores),labelled_occurrences=len(result),new_verifier_decisions=sum(r["verifier"]!="" for r in result),
        run_seconds=time.perf_counter()-start,repair_iterations=0,gpu_hours=0))
if __name__=="__main__":
    main()
