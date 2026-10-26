"""Paired source/patient-cluster statistics; all seed values stay inside a cluster.

Input rows: sample_id, method, seed, numeric metric(s). No labels or texts required.
Caller supplies genuine source/patient clusters; never substitutes one row per cluster.
"""
from collections import defaultdict
import itertools
import math
import random


def quantile(values, probability):
    values = sorted(values)
    pos = (len(values) - 1) * probability
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def holm_adjust(pvalues):
    """Missing p-values remain missing and still count in registered family size."""
    result, maximum = [None] * len(pvalues), 0.0
    ordered = sorted((float(p), i) for i, p in enumerate(pvalues) if p is not None)
    family_size = len(pvalues)
    for rank, (value, index) in enumerate(ordered):
        if not 0 <= value <= 1:
            raise ValueError("p-value outside [0,1]")
        maximum = max(maximum, min(1.0, value * (family_size - rank)))
        result[index] = maximum
    return result


def paired_cluster_comparison(rows, method, reference, metric, clusters, seeds,
                              alternative="two-sided", bootstrap_samples=10000,
                              random_seed=0):
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid predeclared alternative")
    seeds = tuple(int(x) for x in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    lookup = {}
    ids = set()
    for row in rows:
        if row["method"] not in {method, reference}:
            continue
        sample_id, seed = str(row["sample_id"]), int(row["seed"])
        if seed not in seeds:
            raise ValueError("undeclared seed present")
        key = (sample_id, row["method"], seed)
        if key in lookup:
            raise ValueError("duplicate sample/method/seed")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError("nonfinite metric; handle parse failures before statistics")
        lookup[key] = value
        ids.add(sample_id)
    if not ids:
        raise ValueError("no paired observations")
    if any(sample_id not in clusters or clusters[sample_id] in (None, "") for sample_id in ids):
        return {"status": "blocked", "reason": "Genuine source/patient cluster mapping missing", "method": method, "reference": reference, "metric": metric}
    sums, counts = defaultdict(float), defaultdict(int)
    per_seed = {}
    differences = {}
    for sample_id in sorted(ids):
        ds = []
        for seed in seeds:
            try:
                d = lookup[(sample_id, method, seed)] - lookup[(sample_id, reference, seed)]
            except KeyError as exc:
                raise ValueError("incomplete paired method/seed coverage") from exc
            differences[(sample_id, seed)] = d
            ds.append(d)
        cluster = str(clusters[sample_id])
        sums[cluster] += sum(ds) / len(ds)
        counts[cluster] += 1
    for seed in seeds:
        per_seed[str(seed)] = sum(differences[(s, seed)] for s in ids) / len(ids)
    cluster_ids = sorted(sums)
    n = len(cluster_ids)
    estimate = sum(sums.values()) / sum(counts.values())
    output = {"status": "complete_verified", "method": method, "reference": reference,
              "metric": metric, "difference_method_minus_reference": estimate,
              "difference_percentage_points": estimate * 100, "per_seed_difference": per_seed,
              "n_samples": len(ids), "n_source_patient_clusters": n,
              "seeds": list(seeds), "seed_policy": "average within each sample before whole-cluster resampling",
              "alternative": alternative, "bootstrap_seed": random_seed, "bootstrap_samples": bootstrap_samples,
              "ci95": None, "p_value": None,
              "inference_caveat": "Cluster sampling assumes the supplied source/patient clusters are independent; few clusters limit resolution."}
    if metric in {"accuracy", "exact_set_accuracy"} and all(v in (0.0,1.0) for v in lookup.values()):
        output["corrections_and_regressions"] = {
            str(seed): {"corrected":sum(differences[(s,seed)] > 0 for s in ids),
                        "regressed":sum(differences[(s,seed)] < 0 for s in ids),
                        "denominator_all_samples":len(ids)} for seed in seeds}
    if n < 2:
        output.update(status="blocked", reason="Fewer than two independent clusters; point estimate only")
        return output
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = random.Random(random_seed)
    boot = []
    for _ in range(bootstrap_samples):
        chosen = [cluster_ids[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sums[c] for c in chosen) / sum(counts[c] for c in chosen))
    output["ci95"] = [quantile(boot, .025), quantile(boot, .975)]
    output["ci95_percentage_points"] = [x * 100 for x in output["ci95"]]
    observed_sum = sum(sums.values())
    def extreme(value):
        if alternative == "greater":
            return value >= observed_sum - 1e-12
        if alternative == "less":
            return value <= observed_sum + 1e-12
        return abs(value) >= abs(observed_sum) - 1e-12
    vals = [sums[c] for c in cluster_ids]
    if n <= 16:
        num = sum(extreme(sum(v * s for v, s in zip(vals, signs))) for signs in itertools.product((-1, 1), repeat=n))
        output["p_value"] = num / (2 ** n)
        output["test"] = "exact paired cluster sign-flip"
        output["permutations"] = 2 ** n
    else:
        num = sum(extreme(sum(v if rng.randrange(2) else -v for v in vals)) for _ in range(bootstrap_samples))
        output["p_value"] = (num + 1) / (bootstrap_samples + 1)
        output["test"] = "Monte Carlo paired cluster sign-flip with plus-one correction"
        output["permutations"] = bootstrap_samples
    return output


def compare_registered_family(rows, comparisons, clusters, seeds, bootstrap_samples=10000, random_seed=0):
    results = [paired_cluster_comparison(rows, clusters=clusters, seeds=seeds,
                                        bootstrap_samples=bootstrap_samples, random_seed=random_seed, **comparison)
               for comparison in comparisons]
    adjusted = holm_adjust([r.get("p_value") for r in results])
    for result, p in zip(results, adjusted):
        result["p_value_holm"] = p
        result["registered_family_size"] = len(comparisons)
    return results
