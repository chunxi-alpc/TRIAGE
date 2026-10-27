"""Offline, group-weighted applicability subspaces (NumPy only).

The second moment identifies *variation*, not its semantic sign. The signed
operator is the projection of the paired mean. No module-one predictions enter
this module. All axes are post-QK-normalization, pre-RoPE key coordinates.
"""
from __future__ import annotations

import numpy as np


def group_weights(group_ids):
    """Equal weight per source guideline, then per observation within it."""
    groups = np.asarray(group_ids, dtype=str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Need nonempty one-dimensional source group IDs")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def _validate(positive, negative, neutral, group_ids):
    arrays = [np.asarray(x, dtype=np.float64) for x in (positive, negative, neutral)]
    if arrays[0].ndim != 4 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Keys must share [sample, layer, KV head, dimension] shape")
    if arrays[0].shape[0] != len(group_ids) or any(s == 0 for s in arrays[0].shape):
        raise ValueError("Group IDs and nonempty key arrays must align")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite key representation")
    return arrays


def _mean(x, weights):
    return np.einsum("n,n...->...", weights, x)


def _project(basis, vectors):
    return np.einsum("lhdr,...lhd->...lhr", basis, vectors)


def _group_means(values, group_ids):
    groups = np.asarray(group_ids, dtype=str)
    return np.stack([values[groups == group].mean(axis=0) for group in np.unique(groups)])


def bootstrap_interval(values, group_ids, *, draws=2000, seed=261027):
    """Percentile cluster bootstrap, conditional on the already fitted operator.

    Returns NaN bounds for <2 independent groups, never fake zero-width evidence.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(group_ids) or draws < 1:
        raise ValueError("Invalid group bootstrap inputs")
    grouped = _group_means(values, group_ids)
    if len(grouped) < 2:
        empty = np.full(grouped.shape[1:], np.nan)
        return empty, empty.copy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    means = grouped[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975], axis=0))


def fit_subspaces(positive, negative, neutral, group_ids, *, rank=2,
                  baseline_score_mean=None, baseline_score_std=None, eps=1e-8):
    """Fit only on the fit partition; callers must isolate source groups first.

    Each observation is a mean over aligned action tokens, not an independent
    token. The natural anchor distance is retained; it is not a unit-norm or
    standardized shift. Score scale is estimated only on the fit partition.
    """
    positive, negative, neutral = _validate(positive, negative, neutral, group_ids)
    if rank < 1 or rank > positive.shape[-1] or eps <= 0:
        raise ValueError("Invalid rank or numerical floor")
    weights = group_weights(group_ids)
    difference = positive - negative
    mu_p, mu_n, mu_0 = [_mean(x, weights) for x in (positive, negative, neutral)]
    raw_delta = mu_p - mu_n
    omega = np.einsum("n,nlhd,nlhe->lhde", weights, difference, difference)
    eigenvalues, eigenvectors = np.linalg.eigh(omega)
    eigenvalues = np.maximum(eigenvalues[..., ::-1], 0)
    basis = eigenvectors[..., ::-1][..., :rank].copy()
    # Null eigenvectors must not become an arbitrary nonzero learned subspace.
    threshold = np.maximum(eigenvalues[..., :1] * 1e-8, eps ** 2)
    active = eigenvalues[..., :rank] > threshold
    basis *= active[..., None, :]
    delta = np.einsum("lhdr,lhr->lhd", basis, _project(basis, raw_delta))
    energy = _mean(np.sum(difference ** 2, axis=-1), weights)
    coherence = np.sum(raw_delta ** 2, axis=-1) / np.maximum(energy, eps ** 2)
    presence = (mu_p + mu_n) / 2 - mu_0
    app_norm = np.linalg.norm(raw_delta, axis=-1)
    presence_norm = np.linalg.norm(presence, axis=-1)
    if (baseline_score_mean is None) != (baseline_score_std is None):
        raise ValueError("Supply both baseline score mean and std, or neither")
    if baseline_score_mean is None:
        sigma = np.ones(positive.shape[1:3], dtype=np.float64)
        scale_source = "unscaled; no baseline scores supplied"
    else:
        means, stds = map(lambda x: np.asarray(x, dtype=np.float64),
                         (baseline_score_mean, baseline_score_std))
        expected = (2, *positive.shape[:3])
        if means.shape != expected or stds.shape != expected:
            raise ValueError("Baseline moments must be [2, sample, layer, KV head]")
        if not (np.isfinite(means).all() and np.isfinite(stds).all()) or (stds < 0).any():
            raise ValueError("Invalid baseline score moments")
        m = _mean(means.mean(axis=0), weights)
        second = _mean((stds ** 2 + means ** 2).mean(axis=0), weights)
        sigma = np.sqrt(np.maximum(second - m ** 2, eps ** 2))
        scale_source = "fit-only pooled unedited QK score standard deviation"
    return {"schema_version": "module2.applicability.v2", "basis": basis,
            "eigenvalues": eigenvalues, "delta": delta, "raw_delta": raw_delta,
            "mu_positive": mu_p, "mu_negative": mu_n, "mu_neutral": mu_0,
            "presence": presence, "sigma": sigma, "coherence": coherence,
            "paired_rms": np.sqrt(energy),
            "mean_paired_l2": _mean(np.linalg.norm(difference, axis=-1), weights),
            "presence_ratio": presence_norm / np.maximum(app_norm, eps),
            "presence_cosine": np.sum(presence * raw_delta, axis=-1) /
                np.maximum(presence_norm * app_norm, eps ** 2),
            "rank": rank, "effective_rank": active.sum(axis=-1),
            "fit_group_ids": sorted(set(map(str, group_ids))),
            "score_scale_source": scale_source, "eps": eps,
            "coordinate_system": "post_k_norm_pre_rope",
            "unit": "natural projected positive-minus-negative anchor distance"}


def evaluate_heads(fit, positive, negative, neutral, effective_queries_positive,
                   effective_queries_negative, group_ids, *, draws=2000, seed=261027,
                   require_heldout=True):
    """Measure separation and signed fixed-query response without refitting.

    Effective queries already include each target key's actual RoPE position and
    every query head sharing its KV head. Read is a local score derivative, not
    attention mass, a model decision, or a causal-use certificate.
    """
    positive, negative, neutral = _validate(positive, negative, neutral, group_ids)
    if require_heldout and set(map(str, group_ids)) & set(fit["fit_group_ids"]):
        raise ValueError("Fit and evaluation source groups overlap")
    qp, qn = map(lambda x: np.asarray(x, dtype=np.float64),
                 (effective_queries_positive, effective_queries_negative))
    if qp.shape != positive.shape or qn.shape != positive.shape:
        raise ValueError("Effective queries must align with keys")
    if not (np.isfinite(qp).all() and np.isfinite(qn).all()):
        raise ValueError("Nonfinite effective queries")
    weights = group_weights(group_ids)
    basis, delta, eps = fit["basis"], fit["delta"], fit["eps"]
    zp, zn = _project(basis, positive), _project(basis, negative)
    mp, mn = _mean(zp, weights), _mean(zn, weights)
    variance = .5 * (_mean(np.sum((zp - mp) ** 2, axis=-1), weights) +
                      _mean(np.sum((zn - mn) ** 2, axis=-1), weights))
    separation = np.linalg.norm(mp - mn, axis=-1) / np.sqrt(np.maximum(variance, eps ** 2))
    norm = np.linalg.norm(delta, axis=-1)
    aligned = np.sum((positive - negative) * delta, axis=-1) / np.maximum(norm, eps)
    denominator = np.sqrt(positive.shape[-1]) * fit["sigma"]
    read_p = np.sum(qp * delta, axis=-1) / denominator
    read_n = np.sum(qn * delta, axis=-1) / denominator
    read = (read_p + read_n) / 2
    low, high = bootstrap_interval(read, group_ids, draws=draws, seed=seed)
    low_p, high_p = bootstrap_interval(read_p, group_ids, draws=draws, seed=seed)
    low_n, high_n = bootstrap_interval(read_n, group_ids, draws=draws, seed=seed)
    mean_read = _mean(read, weights)
    return {"separation": separation, "read": mean_read,
            "read_low": low, "read_high": high,
            "read_positive": _mean(read_p, weights), "read_negative": _mean(read_n, weights),
            "read_positive_low": low_p, "read_positive_high": high_p,
            "read_negative_low": low_n, "read_negative_high": high_n,
            "read_positive_fraction": _mean((read_p > 0).astype(float), weights),
            "read_negative_fraction": _mean((read_n > 0).astype(float), weights),
            "paired_sign_consistency": _mean((aligned > 0).astype(float), weights),
            "paired_alignment": _mean(aligned, weights),
            "score": separation * np.maximum(mean_read, 0),
            "group_ids": sorted(set(map(str, group_ids))),
            "per_sample_read": read, "per_sample_alignment": aligned}


def bootstrap_stability(positive, negative, group_ids, fit, heads, *, draws=64, seed=261027):
    """Refit selected candidate subspaces with source-guideline resampling."""
    difference = np.asarray(positive, float) - np.asarray(negative, float)
    groups = np.asarray(group_ids, dtype=str)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    result = {}
    for layer, head in heads:
        ref = fit["basis"][layer, head]
        r = int(fit["effective_rank"][layer, head])
        if r == 0 or len(unique) < 2:
            result[(layer, head)] = {"mean": 0., "low": 0., "high": 0.}
            continue
        moments = np.stack([np.einsum("nd,ne->de", difference[groups == g, layer, head],
            difference[groups == g, layer, head]) / np.sum(groups == g) for g in unique])
        values = []
        for index in sampled:
            _, vectors = np.linalg.eigh(moments[index].mean(axis=0))
            u = vectors[:, -r:]
            values.append(float(np.sum((ref[:, :r].T @ u) ** 2) / r))
        result[(layer, head)] = {"mean": float(np.mean(values)),
            "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975))}
    return result


def select_heads(fit, evaluation, *, max_heads=4, min_groups=3,
                 min_sign_consistency=.6, min_coherence=.01, stability=None,
                 min_stability=.5):
    """Top K *eligible* heads; an empty set is a valid scientific result.

    Thresholds are pilot screening choices, not corrected significance tests.
    CIs condition on fit data; confirmation must use untouched source groups.
    """
    if max_heads < 1:
        raise ValueError("max_heads must be positive")
    if set(fit["fit_group_ids"]) & set(evaluation["group_ids"]):
        raise ValueError("Selection reuses fit groups")
    if len(evaluation["group_ids"]) < min_groups:
        return []
    eligible = ((evaluation["read_positive_low"] > 0) &
                (evaluation["read_negative_low"] > 0) &
                (evaluation["paired_sign_consistency"] >= min_sign_consistency) &
                (fit["coherence"] >= min_coherence) &
                (fit["effective_rank"] > 0))
    ranked = sorted([tuple(map(int, index)) for index in np.argwhere(eligible)],
                    key=lambda x: (-evaluation["score"][x], x))
    if stability is not None:
        ranked = [h for h in ranked if stability.get(h, {}).get("low", 0.) >= min_stability]
    return ranked[:max_heads]


def save_fit(path, fit):
    """Store numeric arrays without pickle; metadata are JSON in the caller."""
    np.savez_compressed(path, **{k: v for k, v in fit.items() if isinstance(v, np.ndarray)})
