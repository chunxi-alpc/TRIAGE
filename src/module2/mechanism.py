"""Token-level Module 2 statistics for the 2026-10-28 mechanism protocol.

This module implements the current manuscript, independently of the historical
span-mean estimator in ``applicability.py``. The unit of weighting is a minimal
pair: tokens are averaged *within each pair* before pairs are averaged. Group
IDs are used for partition isolation and uncertainty, not to change that target
estimand. No Module 1 outputs are accepted.

Key coordinates are post-key-normalization and pre-RoPE. Effective queries are
``mean_{target positions, shared query heads} R_j.T @ q_a``; callers must include
actual online positions and query-side RoPE, and must NOT divide by sqrt(D).
"""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np


KEY_STATES = ("positive", "negative", "neutral")
QUERY_STATES = ("positive", "negative")


def _finite_array(value, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _metadata_groups(metadata: Sequence[Mapping[str, Any]], n: int) -> np.ndarray:
    if len(metadata) != n:
        raise ValueError("Metadata length must match the number of pairs")
    if any(not str(row.get("group_id", "")) for row in metadata):
        raise ValueError("Every pair needs a nonempty source guideline group_id")
    identifiers = [row.get("id") for row in metadata]
    if all(identifier is not None for identifier in identifiers):
        if len(set(map(str, identifiers))) != len(identifiers):
            raise ValueError("Pair IDs must be unique within a partition")
    return np.asarray([str(row["group_id"]) for row in metadata])


def cluster_bootstrap_interval(values, group_ids, *, draws=2000, seed=261028):
    """Cluster-bootstrap the equal-pair mean, conditional on the fitted operator.

    Whole source guidelines are resampled. A sampled guideline contributes all
    its observations, so unequal guideline sizes retain the manuscript's
    equal-pair estimand. With fewer than two independent guidelines, bounds are
    NaN rather than a misleading zero-width interval. This is descriptive pilot
    uncertainty, not simultaneous inference over the candidate heads.
    """
    values = np.asarray(values, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=str)
    if values.ndim < 1 or not len(values) or len(values) != len(group_ids):
        raise ValueError("Bootstrap observations and source groups must align")
    if not np.isfinite(values).all() or draws < 1:
        raise ValueError("Bootstrap requires finite observations and positive draws")
    unique = np.unique(group_ids)
    if len(unique) < 2:
        empty = np.full(values.shape[1:], np.nan)
        return empty, empty.copy()
    counts = np.asarray([np.count_nonzero(group_ids == group) for group in unique])
    sums = np.stack([values[group_ids == group].sum(axis=0) for group in unique])
    indices = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    denominator = counts[indices].sum(axis=1)
    means = sums[indices].sum(axis=1) / denominator.reshape(
        (draws,) + (1,) * (values.ndim - 1))
    return tuple(np.quantile(means, [0.025, 0.975], axis=0))


def fit_operators(differences, token_second_moment, *, rank=2, eps=1e-8,
                  min_projected_norm=1e-6, rank_rtol=1e-8, metadata=None):
    """Fit token-second-moment bases and normalized shared directions.

    Parameters
    ----------
    differences : array [N, L, H, D]
        Per-pair mean aligned token differences ``mean_j(k+ - k-)``.
    token_second_moment : array [L, H, D, D]
        Already accumulated as ``mean_i mean_j outer(d_ij, d_ij)``. It must
        NOT be replaced by outer products of the span means above.
    metadata : optional sequence of mappings
        Fit-only records with ``group_id``. Stored for held-out checks.

    ``numerically_valid`` requires rank >= requested rank and a projected mean
    above the frozen norm threshold. Null directions are returned as zeros;
    they do not become arbitrary experts. Invalid heads are not repaired by
    silently lowering the requested rank.
    """
    difference = _finite_array(differences, "differences", 4)
    moment = _finite_array(token_second_moment, "token_second_moment", 4)
    n, layers, heads, dimension = difference.shape
    if moment.shape != (layers, heads, dimension, dimension):
        raise ValueError("Token second moment must have [L, H, D, D] shape")
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= dimension:
        raise ValueError("rank must be an integer between one and D")
    if eps <= 0 or min_projected_norm <= 0 or rank_rtol < 0:
        raise ValueError("Invalid numerical or near-zero thresholds")
    groups = (np.asarray([], dtype=str) if metadata is None
              else _metadata_groups(metadata, n))
    moment = (moment + np.swapaxes(moment, -1, -2)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    scale = np.maximum(np.abs(eigenvalues).max(axis=-1), eps ** 2)
    if np.any(eigenvalues[..., 0] < -np.maximum(scale * 1e-8, eps ** 2)):
        raise ValueError("Token second moment is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues[..., ::-1], 0)
    eigenvectors = eigenvectors[..., ::-1]
    threshold = np.maximum(eigenvalues[..., :1] * rank_rtol, eps ** 2)
    effective_rank = (eigenvalues > threshold).sum(axis=-1)
    active = eigenvalues[..., :rank] > threshold
    basis = eigenvectors[..., :rank].copy() * active[..., None, :]
    raw_mean = difference.mean(axis=0)
    coordinates = np.einsum("lhdr,lhd->lhr", basis, raw_mean)
    projected_mean = np.einsum("lhdr,lhr->lhd", basis, coordinates)
    projected_norm = np.linalg.norm(projected_mean, axis=-1)
    raw_norm = np.linalg.norm(raw_mean, axis=-1)
    delta = projected_mean / (projected_norm[..., None] + eps)
    raw_unit = raw_mean / (raw_norm[..., None] + eps)
    valid = (effective_rank >= rank) & (projected_norm > min_projected_norm)
    token_energy = eigenvalues.sum(axis=-1)
    mean_difference_energy = np.square(difference).sum(axis=-1).mean(axis=0)
    # Jensen's inequality is a cheap guard against an inconsistent accumulator.
    if np.any(token_energy + np.maximum(1e-10, token_energy * 1e-6)
              < mean_difference_energy):
        raise ValueError("Token energy is smaller than the supplied span-mean energy")
    return {
        "schema_version": "module2.mechanism.token_second_moment.v1",
        "basis": basis, "U": basis, "delta": delta,
        "raw_unit": raw_unit, "raw_direction": raw_unit,
        "raw_mean": raw_mean, "projected_mean": projected_mean,
        "projected_norm": projected_norm, "raw_norm": raw_norm,
        "eigenvalues": eigenvalues, "effective_rank": effective_rank,
        "numerically_valid": valid, "token_difference_energy": token_energy,
        "mean_difference_energy": mean_difference_energy,
        "mean_coherence": np.square(raw_norm) / np.maximum(mean_difference_energy, eps ** 2),
        "token_coherence": np.square(raw_norm) / np.maximum(token_energy, eps ** 2),
        "projection_retention": projected_norm / np.maximum(raw_norm, eps),
        "fit_pair_count": n, "fit_group_ids": sorted(set(groups.tolist())),
        "rank": int(rank), "eps": float(eps), "rank_rtol": float(rank_rtol),
        "min_projected_norm": float(min_projected_norm),
        "weighting": "equal minimal pair; equal token within pair",
        "coordinate_system": "post_k_norm_pre_rope",
        "normalization": "projected_mean / (projected_norm + eps)",
    }


def evaluate_direction(direction, offline_means, online_effective_queries, metadata,
                       *, fit_group_ids=(), require_heldout=True, eps=1e-8,
                       draws=2000, seed=261028):
    """Evaluate a frozen direction using current Sep and both online states.

    Offline means have shape [N, 3, L, H, D], in positive/negative/neutral order.
    Online effective queries have shape [N, 2, L, H, D], in positive/negative
    order. Reads are local QK-score derivatives, not attention mass or decisions.
    Population standard deviation (ddof=0) is used as defined in the manuscript.
    """
    direction = _finite_array(direction, "direction", 3)
    keys = _finite_array(offline_means, "offline_means", 5)
    queries = _finite_array(online_effective_queries, "online_effective_queries", 5)
    n = keys.shape[0]
    if n < 2 or keys.shape[1] != 3 or keys.shape[2:] != direction.shape:
        raise ValueError("Need >=2 pairs with offline means [N, 3, L, H, D]")
    if queries.shape != (n, 2, *direction.shape):
        raise ValueError("Online effective queries must have [N, 2, L, H, D] shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    groups = _metadata_groups(metadata, n)
    if require_heldout and set(map(str, fit_group_ids)) & set(groups.tolist()):
        raise ValueError("Fit and evaluation source guidelines overlap")
    difference = keys[:, 0] - keys[:, 1]
    midpoint_offset = (keys[:, 0] + keys[:, 1]) / 2 - keys[:, 2]
    z = np.einsum("nlhd,lhd->nlh", difference, direction)
    u = np.einsum("nlhd,lhd->nlh", midpoint_offset, direction)
    read = np.einsum("nslhd,lhd->nslh", queries, direction) / np.sqrt(direction.shape[-1])
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0, ddof=0)
    missing_penalty = np.abs(u).mean(axis=0)
    separation = z_mean / (z_std + missing_penalty + eps)
    read_mean_states = read.mean(axis=0)
    read_mean = read_mean_states.mean(axis=0)
    direction_norm = np.linalg.norm(direction, axis=-1)
    difference_norm = np.linalg.norm(difference, axis=-1)
    nonzero = (difference_norm > eps) & (direction_norm[None, ...] > eps)
    cosine = np.full_like(z, np.nan)
    np.divide(z, difference_norm * direction_norm[None, ...], out=cosine, where=nonzero)
    nonzero_count = nonzero.sum(axis=0)
    mean_cosine = np.divide(np.where(nonzero, cosine, 0).sum(axis=0), nonzero_count,
                            out=np.full_like(z_mean, np.nan), where=nonzero_count > 0)
    nonzero_positive_fraction = np.divide(((z > 0) & nonzero).sum(axis=0), nonzero_count,
        out=np.full_like(z_mean, np.nan), where=nonzero_count > 0)
    result = {
        "separation": separation, "z_mean": z_mean, "z_std": z_std,
        "missing_penalty": missing_penalty, "read": read_mean,
        "read_positive": read_mean_states[0], "read_negative": read_mean_states[1],
        "read_positive_fraction": (read[:, 0] > 0).mean(axis=0),
        "read_negative_fraction": (read[:, 1] > 0).mean(axis=0),
        "paired_sign_consistency": (z > 0).mean(axis=0),
        "nonzero_positive_fraction": nonzero_positive_fraction,
        "near_zero_fraction": (~nonzero).mean(axis=0), "mean_cosine": mean_cosine,
        "score": separation * read_mean,
        "per_pair_z": z, "per_pair_u": u, "per_pair_read": read,
        "per_pair_cosine": cosine, "per_pair_difference_norm": difference_norm,
        "per_pair_nonzero": nonzero,
        "group_ids": sorted(set(groups.tolist())), "pair_count": n,
        "pair_group_ids": groups.tolist(),
        "partition_names": sorted({str(row.get("split", "unspecified")) for row in metadata}),
        "fit_group_ids": sorted(set(map(str, fit_group_ids))),
        "weighting": "equal minimal pair", "ci_unit": "source guideline cluster",
        "ci_conditional_on": "frozen fitted direction; no multiplicity correction",
        "key_state_order": list(KEY_STATES), "query_state_order": list(QUERY_STATES),
    }
    for name, values in (("z", z), ("read", read.mean(axis=1)),
                         ("read_positive", read[:, 0]), ("read_negative", read[:, 1])):
        low, high = cluster_bootstrap_interval(values, groups, draws=draws, seed=seed)
        result[name + "_low"], result[name + "_high"] = low, high
    return result


def select_heads(fit, evaluation, *, max_heads=4):
    """Current-main TopK rule; no extra CI, coherence, or sign-rate threshold.

    Return zero-based (layer, KV head) tuples. Empty eligibility is a valid
    result. Ties are resolved by layer, then head. Selection and fit guidelines
    must be disjoint; confirmation data must never be passed here by the caller.
    """
    if not isinstance(max_heads, (int, np.integer)) or max_heads < 1:
        raise ValueError("max_heads must be a positive integer")
    if set(fit.get("fit_group_ids", ())) & set(evaluation.get("group_ids", ())):
        raise ValueError("Selection reuses fit source guidelines")
    shape = np.asarray(fit["numerically_valid"]).shape
    names = ("separation", "read_positive", "read_negative", "score")
    if any(np.asarray(evaluation[name]).shape != shape for name in names):
        raise ValueError("Evaluation and fitted head geometry differ")
    eligible = np.asarray(fit["numerically_valid"], dtype=bool).copy()
    for name in names:
        eligible &= np.isfinite(evaluation[name])
    eligible &= (evaluation["separation"] > 0)
    eligible &= (evaluation["read_positive"] > 0)
    eligible &= (evaluation["read_negative"] > 0)
    ranked = sorted((tuple(map(int, index)) for index in np.argwhere(eligible)),
                    key=lambda index: (-float(evaluation["score"][index]), index))
    return ranked[:max_heads]


def _configuration_key(row):
    if row.get("configuration_id") is not None:
        return str(row["configuration_id"])
    identifier = str(row.get("id", ""))
    if ":target" in identifier:
        return identifier.rsplit(":target", 1)[0]
    controls = row.get("controls", {})
    permutation = row.get("action_permutation", controls.get("action_permutation"))
    order = row.get("rule_order_index", controls.get("rule_order_index"))
    if row.get("pair_cluster_id") is not None and permutation is not None and order is not None:
        return f"{row['pair_cluster_id']}:swap{permutation}:order{order}"
    raise ValueError("Need configuration_id, a :target pair ID, or configuration controls")


def rule_relative_effects(differences, metadata, direction=None):
    """Decompose matched opposite-rule responses without claiming full deconfounding.

    For lexically ordered target rules A/B, supplied differences are already
    oriented from target-inapplicable to target-applicable:
    d_A = k_A(P_A)-k_A(P_B), d_B = k_B(P_B)-k_B(P_A).
    ``interaction=(d_A+d_B)/2`` and ``common_fact=(d_A-d_B)/2``.
    This cancels an identical patient-fact perturbation at both rule spans;
    different position/content responses can remain. The sign of common_fact
    depends on the explicitly recorded first target rule.
    """
    differences = _finite_array(differences, "differences", 4)
    groups = _metadata_groups(metadata, len(differences))
    configurations = {}
    for index, row in enumerate(metadata):
        configurations.setdefault((str(groups[index]), _configuration_key(row)), []).append(index)
    interaction, common, records = [], [], []
    for (group, config), indices in sorted(configurations.items()):
        if len(indices) != 2:
            raise ValueError(f"Configuration {config} needs exactly two opposite target rules")
        indices = sorted(indices, key=lambda i: str(metadata[i].get("target_rule_id", "")))
        targets = [str(metadata[i].get("target_rule_id", "")) for i in indices]
        if not all(targets) or targets[0] == targets[1]:
            raise ValueError(f"Configuration {config} lacks two distinct target rules")
        first, second = (differences[index] for index in indices)
        interaction.append((first + second) / 2)
        common.append((first - second) / 2)
        records.append({"configuration_id": config, "group_id": group,
                        "target_rule_ids": targets, "row_indices": indices,
                        "pair_ids": [str(metadata[i].get("id", "")) for i in indices]})
    interaction, common = np.stack(interaction), np.stack(common)
    result = {"interaction": interaction, "common_fact": common, "metadata": records}
    if direction is not None:
        direction = _finite_array(direction, "direction", 3)
        if direction.shape != differences.shape[1:]:
            raise ValueError("Direction and differences must share head geometry")
        result["aligned_interaction"] = np.einsum("nlhd,lhd->nlh", interaction, direction)
        result["aligned_common_fact"] = np.einsum("nlhd,lhd->nlh", common, direction)
    return result


def enumerate_group_sign_directions(differences, group_ids, basis, *, eps=1e-8,
                                    max_groups=12):
    """Enumerate clustered label-sign controls with a fixed spectral subspace.

    Every token/variant from a source guideline receives the same sign. This
    tests directional mean sensitivity; it is NOT a spectral null because
    changing a paired sign leaves every token outer product unchanged. A formal
    randomization p-value additionally needs an exchangeability justification.
    """
    differences = _finite_array(differences, "differences", 4)
    basis = _finite_array(basis, "basis", 4)
    group_ids = np.asarray(group_ids, dtype=str)
    if group_ids.shape != (len(differences),) or basis.shape[:3] != differences.shape[1:]:
        raise ValueError("Sign-control groups, differences, and basis must align")
    unique = np.unique(group_ids)
    if len(unique) > max_groups or eps <= 0:
        raise ValueError("Too many exhaustive sign configurations or invalid eps")
    group_contributions = np.stack([differences[group_ids == group].sum(axis=0)
                                    / len(differences) for group in unique])
    signs = np.asarray(list(product((-1, 1), repeat=len(unique))), dtype=np.int8)
    means = np.einsum("sg,glhd->slhd", signs, group_contributions)
    coordinates = np.einsum("lhdr,slhd->slhr", basis, means)
    projected = np.einsum("lhdr,slhr->slhd", basis, coordinates)
    norms = np.linalg.norm(projected, axis=-1)
    return {"directions": projected / (norms[..., None] + eps),
            "projected_norms": norms, "signs": signs,
            "group_ids": unique.tolist(), "raw_means": means,
            "all_positive_index": len(signs) - 1,
            "interpretation": "clustered directional diagnostic; not a spectral null"}
