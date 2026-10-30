"""Source-cluster paired estimates; no model fitting or official parameter selection."""
from __future__ import annotations
import itertools, math, random
from collections import defaultdict

def quantile(values, p):
    values=sorted(values)
    if not values: return None
    location=(len(values)-1)*p
    lo=int(location); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(location-lo)

def paired_summary(rows, *, repeats=5000, seed=20261030):
    """One planned case per row, including missing/failed predictions as zero.

    Required: case_id, source_id, baseline, treatment.
    baseline/treatment=None denotes missing and remains in the denominator.
    Technical failures can be flagged baseline_failed/treatment_failed.
    """
    if not rows:
        return {"planned_cases":0,"source_clusters":0,"status":"no_data"}
    ids=[r["case_id"] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError("duplicate case_id: counterfactual/method runs must be aggregated before case analysis")
    grouped=defaultdict(list)
    corrections=degradations=missing_baseline=missing_treatment=0
    failures_baseline=failures_treatment=0
    for r in rows:
        if not r.get("source_id"): raise ValueError("source_id is required")
        b0,t0=r["baseline"],r["treatment"]
        for v in (b0,t0):
            if v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1):
                raise ValueError("case scores must be finite in [0,1]")
        missing_baseline+=b0 is None; missing_treatment+=t0 is None
        bf,tf=bool(r.get("baseline_failed")),bool(r.get("treatment_failed"))
        failures_baseline+=bf; failures_treatment+=tf
        b=0. if b0 is None or bf else float(b0)
        t=0. if t0 is None or tf else float(t0)
        corrections+=int(b==0 and t==1); degradations+=int(b==1 and t==0)
        grouped[r["source_id"]].append(t-b)
    effects=[sum(v)/len(v) for _,v in sorted(grouped.items())]
    estimate=sum(effects)/len(effects)
    rng=random.Random(seed)
    draws=[sum(rng.choices(effects,k=len(effects)))/len(effects) for _ in range(repeats)]
    # A two-sided paired sign-flip test acts on source-cluster effects.
    active=[v for v in effects if v!=0]
    observed=abs(sum(effects))
    if len(active)<=16:
        null=[abs(sum(s*v for s,v in zip(signs,active))) for signs in itertools.product((-1,1),repeat=len(active))]
        p=sum(v>=observed-1e-12 for v in null)/len(null)
        p_method="exact_source_sign_flip"
    else:
        n=9999
        extreme=sum(abs(sum(rng.choice((-1,1))*v for v in active))>=observed-1e-12 for _ in range(n))
        p=(extreme+1)/(n+1); p_method="monte_carlo_source_sign_flip"
    return {
        "status":"descriptive_only_fewer_than_3_clusters" if len(effects)<3 else "estimated",
        "planned_cases":len(rows),"source_clusters":len(effects),
        "source_equal_paired_difference":estimate,
        "source_cluster_bootstrap_95pct":[quantile(draws,.025),quantile(draws,.975)],
        "case_weighted_paired_difference":sum(sum(v) for v in grouped.values())/len(rows),
        "corrections":corrections,"degradations":degradations,
        "missing_baseline":missing_baseline,"missing_treatment":missing_treatment,
        "technical_failures_baseline":failures_baseline,"technical_failures_treatment":failures_treatment,
        "baseline_coverage":(len(rows)-missing_baseline-failures_baseline+sum(r["baseline"] is None and bool(r.get("baseline_failed")) for r in rows))/len(rows),
        "treatment_coverage":(len(rows)-missing_treatment-failures_treatment+sum(r["treatment"] is None and bool(r.get("treatment_failed")) for r in rows))/len(rows),
        "paired_p":p,"paired_p_method":p_method,
        "bootstrap_repeats":repeats,"seed":seed,
        "pass_inference":len(effects)>=3 and quantile(draws,.025)>0,
        "cluster_effects":dict(zip(sorted(grouped),effects)),
    }

def holm(pvalues):
    """Return adjusted p values in original order, monotone in sorted raw p."""
    n=len(pvalues); result=[None]*n; previous=0.
    if any(not math.isfinite(p) or not 0<=p<=1 for p in pvalues): raise ValueError("invalid p value")
    for rank,index in enumerate(sorted(range(n),key=lambda i:pvalues[i])):
        previous=max(previous,min(1.,(n-rank)*pvalues[index]))
        result[index]=previous
    return result
