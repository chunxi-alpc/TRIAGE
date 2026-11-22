"""Numerical definitions for manuscript Table 2; no model loading or fitting on import."""
from collections import Counter, defaultdict
import numpy as np

def pair_weights(pairs, mode="hierarchical"):
    """Class/source/case/variant balance, normalized within the supplied population."""
    if not pairs:
        raise ValueError("The pair population is empty.")
    kinds = sorted({p["kind"] for p in pairs})
    sources = defaultdict(set)
    cases = defaultdict(set)
    variants = Counter()
    for p in pairs:
        q, g = p["kind"], p["source_id"]
        sources[q].add(g)
        if mode == "hierarchical":
            case = p["case_id"]
            cases[q, g].add(case)
            variants[q, g, case] += 1
        elif mode == "class_source":
            variants[q, g] += 1
        else:
            raise ValueError("Unknown weighting mode: " + mode)
    out = []
    for p in pairs:
        q, g = p["kind"], p["source_id"]
        denom = len(kinds) * len(sources[q])
        if mode == "hierarchical":
            case = p["case_id"]
            denom *= len(cases[q, g]) * variants[q, g, case]
        else:
            denom *= variants[q, g]
        out.append(1.0 / denom)
    weights = np.asarray(out, dtype=np.float64)
    return weights / weights.sum()

def normalized_subset(weights, selected):
    out = np.asarray(weights, dtype=np.float64)[selected]
    if out.size == 0 or out.sum() <= 0:
        return out
    return out / out.sum()

def fit_ridge(delta_features, delta_logits, weights, regularization):
    """Minimize sum_i weights_i (delta_logit_i + dx_i @ w)^2 + lambda ||w||^2."""
    x = np.asarray(delta_features, dtype=np.float64)
    y = np.asarray(delta_logits, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if regularization <= 0:
        raise ValueError("The manuscript requires positive ridge regularization.")
    return np.linalg.solve(
        x.T @ (weights[:, None] * x) + regularization * np.eye(x.shape[1]),
        -x.T @ (weights * y),
    )

def separable_predictability(delta_features, delta_logits, frozen_w, weights):
    x = np.asarray(delta_features, dtype=np.float64)
    y = np.asarray(delta_logits, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(y) or weights.sum() <= 0:
        return {"n": 0, "r2_sep": None, "baseline_mse": None, "residual_mse": None}
    weights = weights / weights.sum()
    baseline = float(weights @ (y * y))
    residual = y + x @ np.asarray(frozen_w, dtype=np.float64)
    error = float(weights @ (residual * residual))
    return {
        "n": len(y),
        "r2_sep": None if baseline == 0 else 1.0 - error / baseline,
        "baseline_mse": baseline,
        "residual_mse": error,
    }

def evidence_coverage(projected_evidence_differences, weights):
    """Rows contain N.T @ (I - S @ S.T) @ delta_z_standardized."""
    x = np.asarray(projected_evidence_differences, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(x) or weights.sum() <= 0:
        return {"n": 0, "r_cov": None}
    weights = weights / weights.sum()
    return {"n": len(x), "r_cov": float(np.sqrt(weights @ np.sum(x * x, axis=1)))}
