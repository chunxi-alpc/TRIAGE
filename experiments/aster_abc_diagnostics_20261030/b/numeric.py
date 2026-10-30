"""Independent mechanism diagnostics; no edits to the frozen implementation."""
import math
import torch
from src.aster.manuscript_core import parse_head, rotate

SEEDS = tuple(range(20261030, 20261035))

def directions(expert, method):
    out = {}
    seed = int(method.split(":", 1)[1]) if method.startswith("random:") else None
    rng = torch.Generator().manual_seed(seed or SEEDS[0])
    for name in expert["selected_heads"]:
        c = expert["candidates"][name]
        v = torch.tensor(c["direction"], dtype=torch.float64)
        n = v.norm()
        if method == "mean":
            v = torch.tensor(c["mean"], dtype=torch.float64)
            v = v / v.norm() * n
        elif method == "reverse":
            v = -v
        elif seed is not None:
            v = torch.randn(v.shape, generator=rng, dtype=v.dtype)
            v = v / v.norm() * n
        out[parse_head(name)] = v
    return out

def score_change(q_rotated, delta_pre_rope, cos, sin, gain):
    return q_rotated.double() @ rotate(delta_pre_rope.double()[None], cos.double(), sin.double()).T * gain / math.sqrt(q_rotated.shape[-1])

def cluster_ci(values, seed=20261030, repetitions=2000):
    x = torch.tensor(values, dtype=torch.float64)
    if not len(x):
        return {"clusters": 0, "mean": None, "ci95": None}
    if len(x) < 2:
        return {"clusters": len(x), "mean": float(x.mean()), "ci95": None}
    rng = torch.Generator().manual_seed(seed)
    samples = x[torch.randint(len(x), (repetitions, len(x)), generator=rng)].mean(1)
    return {"clusters": len(x), "mean": float(x.mean()), "ci95": [float(v) for v in torch.quantile(samples, torch.tensor([.025, .975], dtype=torch.float64))]}

def summarize(records):
    methods = sorted({r["method"] for r in records})
    out = {"unit": "source cluster; state/head/token are repeated observations",
           "planned_records": len(records), "technical_failures": sum(r["status"] != "complete" for r in records),
           "methods": {}}
    for method in methods:
        rows = [r for r in records if r["method"] == method and r["status"] == "complete" and r["state"] != "unknown"]
        clusters = {}
        for r in rows:
            clusters.setdefault(r["source_id"], []).append(r["signed_mass_change"])
        vals = [sum(x)/len(x) for x in clusters.values()]
        out["methods"][method] = dict(cluster_ci(vals), positive_clusters=sum(x>0 for x in vals), negative_clusters=sum(x<0 for x in vals))
    main = out["methods"].get("spectral", {})
    controls = [out["methods"][f"random:{s}"].get("mean") for s in SEEDS if f"random:{s}" in out["methods"]]
    mainmean = main.get("mean")
    rev = out["methods"].get("reverse", {}).get("mean")
    ci = main.get("ci95")
    out["mechanism_pass"] = bool(ci and ci[0] > 0 and rev is not None and mainmean > rev and sum(x is not None and mainmean > x for x in controls) >= 4)
    out["random_direction_means"] = controls
    out["random_null_minimum_p"] = 1/(len(controls)+1) if controls else None
    out["confirmatory_evidence"] = False
    out["interpretation"] = "Exploratory age-rule root diagnostic; no physician three-state gold and no task-benefit inference."
    return out
