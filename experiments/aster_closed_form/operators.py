"""Closed-form ASTER M2/M3 and replacements of those modules.

Fitting uses only the fit split and unedited activations. Returned operators
act on column vectors as z - N @ B @ (z - mu). Matrices are stored as CPU
float32 tensors; all decompositions use float64.

LEACE follows Theorem 4.2 of Belrose et al. (2023):
https://papers.nips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import torch

KINDS = ("A", "P", "C", "F")
EPSILON = 0.1


def configuration_grid():
    """The prespecified 2 x 3 x 3 candidates, in deterministic order."""
    return [
        {"eta": eta, "kmax": kmax, "theta": theta}
        for eta, kmax, theta in itertools.product((0.90, 0.95), (4, 8, 16), (15, 30, 45))
    ]


def _double(value):
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)


def _empty(d):
    return torch.empty((d, 0), dtype=torch.float64)


def orth(matrix):
    """Column basis, retaining only numerically independent directions."""
    matrix = _double(matrix)
    if matrix.shape[1] == 0:
        return _empty(matrix.shape[0])
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    if not singular.numel() or float(singular[0]) == 0.0:
        return _empty(matrix.shape[0])
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
    return u[:, singular > tolerance]


def energy_basis(differences, eta=0.90, kmax=4, mean_direction=False):
    """Uncentered second-moment basis or the oriented mean-difference ablation.

    differences has shape (number of pairs, head dimension). A zero-energy
    type yields an empty basis. The rank limit applies to this type only.
    """
    delta = _double(differences)
    if delta.ndim != 2:
        raise ValueError("Pair differences must be a sample-by-dimension matrix")
    if not (0 < eta <= 1) or kmax < 1:
        raise ValueError("eta must be in (0,1], and kmax must be positive")
    n, d = delta.shape
    moment = delta.T @ delta / n if n else torch.zeros((d, d), dtype=torch.float64)
    total = float(torch.trace(moment))
    if not n or total == 0.0:
        return _empty(d), moment, {
            "pairs": n, "rank": 0, "total_energy": total, "retained_energy": 0.0,
            "retained_fraction": 0.0, "target_energy": eta, "target_reached": False,
            "zero_energy": True, "eigenvalues": [0.0] * d, "rank_limited": False,
            "basis_rule": "oriented_mean_difference" if mean_direction else "uncentered_second_moment",
        }
    # SVD of the data matrix is equivalent to eigendecomposition of its
    # uncentered second moment and avoids spurious positive covariance roots.
    _, singular, vh = torch.linalg.svd(delta / math.sqrt(n), full_matrices=False)
    eigenvalues = singular.square()
    tolerance = max(delta.shape) * torch.finfo(delta.dtype).eps * singular[0]
    nonzero = int((singular > tolerance).sum())
    if mean_direction:
        mean = delta.mean(dim=0)
        norm = float(torch.linalg.vector_norm(mean))
        # Exact orientation cancellation is a zero-direction ablation outcome.
        basis = mean[:, None] / norm if norm > 0.0 else _empty(d)
        rank_needed = 1 if norm > 0.0 else 0
    else:
        rank_needed = int(torch.searchsorted(eigenvalues.cumsum(0), eta * total).item()) + 1
        rank = min(rank_needed, kmax, nonzero)
        basis = vh[:rank].T
    retained = float(torch.trace(basis.T @ moment @ basis)) if basis.shape[1] else 0.0
    return basis, moment, {
        "pairs": n, "rank": int(basis.shape[1]), "total_energy": total,
        "retained_energy": retained, "retained_fraction": retained / total,
        "target_energy": eta, "target_reached": retained / total >= eta - 1e-10,
        "zero_energy": False,
        "eigenvalues": eigenvalues.tolist() + [0.0] * max(0, d - len(eigenvalues)),
        "rank_needed_for_target": rank_needed if not mean_direction else None,
        "rank_limited": bool(not mean_direction and rank_needed > kmax),
        "basis_rule": "oriented_mean_difference" if mean_direction else "uncentered_second_moment",
    }


def oblique_factors(nuisance, signal, theta=15, variant="aster"):
    """Equations (12)-(13): principal-angle screening, then oblique erasure."""
    if variant not in ("aster", "soft_angle"):
        raise ValueError("Unknown angle variant: " + str(variant))
    nuisance, signal = _double(nuisance), _double(signal)
    if not 0 < theta <= 90:
        raise ValueError("theta must be in (0,90] degrees")
    d = nuisance.shape[0]
    if nuisance.shape[1] == 0:
        return _empty(d), torch.empty((0, d), dtype=torch.float64), {
            "principal_angles_degrees": [], "singular_values": [],
            "retained_direction_indices": [], "discarded_direction_indices": [],
            "retained_rank": 0, "discarded_rank": 0, "theta_min": theta,
        }
    residual = nuisance - signal @ (signal.T @ nuisance)
    q, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    threshold = math.sin(math.radians(theta))
    keep = singular >= threshold
    # Rotations preserve orthonormality of the retained nuisance basis.
    if variant == "soft_angle":
        nbar = nuisance @ vh.T
        weights = singular / singular.square().clamp_min(threshold**2)
        b = weights[:, None] * q.T
    else:
        nbar = nuisance @ vh[keep].T
        b = q[:, keep].T / singular[keep, None]
    angles = torch.rad2deg(torch.asin(singular.clamp(0, 1)))
    return nbar, b, {
        "variant": variant,
        "self_erasure_fractions": (singular.square() / singular.square().clamp_min(threshold**2)).tolist()
            if variant == "soft_angle" else keep.double().tolist(),
        "self_residual_fractions": (1 - singular.square() / singular.square().clamp_min(threshold**2)).tolist()
            if variant == "soft_angle" else (~keep).double().tolist(),
        "high_angle_indices": keep.nonzero(as_tuple=False).flatten().tolist(),
        "low_angle_indices": (~keep).nonzero(as_tuple=False).flatten().tolist(),
        "factor_columns": nbar.shape[1],
        "principal_angles_degrees": angles.tolist(),
        "singular_values": singular.tolist(),
        "retained_direction_indices": list(range(len(singular))) if variant == "soft_angle" else keep.nonzero(as_tuple=False).flatten().tolist(),
        "discarded_direction_indices": [] if variant == "soft_angle" else (~keep).nonzero(as_tuple=False).flatten().tolist(),
        "retained_rank": nbar.shape[1],
        "discarded_rank": 0 if variant == "soft_angle" else int((~keep).sum()), "theta_min": theta,
        "smallest_retained_sine": float(singular[keep].min()) if keep.any() else None,
    }


def _matrix_norm(matrix):
    return float(torch.linalg.matrix_norm(matrix, ord=2)) if matrix.numel() else 0.0


def algebra_diagnostics(n, b, signal, theta=None, require_signal=True, variant="aster"):
    """Measure actual factors, including stored float32 rounding."""
    n, b, signal = _double(n), _double(b), _double(signal)
    d = n.shape[0]
    update = n @ b
    e = torch.eye(d, dtype=torch.float64) - update
    basis = orth(n)
    residuals = {
        "signal_preservation_error": _matrix_norm(e @ signal - signal),
        "erasure_error": _matrix_norm(e @ n),
        "nuisance_residual_norm": _matrix_norm(e @ n),
        "exact_erasure_expected": variant != "soft_angle",
        "idempotence_expected": variant != "soft_angle",
        "idempotence_error": _matrix_norm(e @ e - e),
        "edit_range_error": _matrix_norm(update - basis @ (basis.T @ update)),
        "edit_operator_norm": _matrix_norm(update),
        "signal_preservation_expected": bool(require_signal),
        "angle_bound": 1.0 / math.sin(math.radians(theta)) if theta is not None else None,
    }
    residuals["angle_bound_satisfied"] = (
        residuals["edit_operator_norm"] <= residuals["angle_bound"] + 1e-5
        if residuals["angle_bound"] is not None else None
    )
    return residuals


def leace_factors(activations, concepts):
    """Centered empirical LEACE, with no target-truth labels or shrinkage.

    Returns low-rank N,B; N need not be orthonormal for this replacement.
    """
    x, z = _double(activations), _double(concepts)
    if x.ndim != 2 or z.ndim != 2 or len(x) != len(z):
        raise ValueError("LEACE expects aligned sample matrices")
    d = x.shape[1]
    if len(x) == 0 or z.shape[1] == 0:
        return _empty(d), torch.empty((0, d), dtype=torch.float64), {
            "samples": len(x), "concept_dimensions": z.shape[1],
            "covariance_rank": 0, "concept_rank": 0,
            "cross_covariance_before": 0.0, "cross_covariance_after": 0.0,
        }
    xc, zc = x - x.mean(0), z - z.mean(0)
    covariance = xc.T @ xc / len(x)
    cross = xc.T @ zc / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(0)
    tolerance = max(covariance.shape) * torch.finfo(values.dtype).eps * values.max()
    positive = values > tolerance
    vectors = vectors[:, positive]
    roots = values[positive].sqrt()
    whitening = (vectors / roots) @ vectors.T
    unwhitening = (vectors * roots) @ vectors.T
    whitened_cross = whitening @ cross
    u = orth(whitened_cross)
    n = unwhitening @ u
    b = u.T @ whitening
    e = torch.eye(d, dtype=torch.float64) - n @ b
    return n, b, {
        "samples": len(x), "concept_dimensions": z.shape[1],
        "covariance_rank": int(positive.sum()), "concept_rank": u.shape[1],
        "cross_covariance_before": _matrix_norm(cross),
        "cross_covariance_after": _matrix_norm(e @ cross),
        "formula": "I - W_pinv @ proj(W @ Cov(X,Z)) @ W",
        "centering": "Empirical mean for covariance; affine anchor recorded separately",
        "node_truth_used_as_concept": False,
    }


def _feature_matrix(feature):
    z = _double(feature["heads"])
    if z.ndim not in (2, 3):
        raise ValueError("heads must have shape (layers,heads,dimension) or (heads,dimension)")
    return z.reshape(-1, z.shape[-1])


def _types_for(types, head):
    allowed = types.get(head, types.get(str(head), []))
    allowed = sorted(set(allowed))
    if any(kind not in ("A", "P") for kind in allowed):
        raise ValueError("T_h may only contain A and P")
    return allowed


def _original_ids(features, fit_pairs):
    fit_ids = {p[side] for p in fit_pairs for side in ("left", "right")}
    explicit = [
        key for key, value in features.items()
        if value.get("original", value.get("is_original", False))
        and (value.get("split") == "fit" or key in fit_ids)
    ]
    if explicit:
        return sorted(set(explicit)), "explicit_fit_original_inputs"
    # Older caller data may lack view metadata. Never use dev/test records.
    fallback = sorted({p["left"] for p in fit_pairs})
    return fallback, "fallback_unique_fit_pair_left_inputs"


def _energy_in(basis, moment):
    return float(torch.trace(basis.T @ moment @ basis)) if basis.shape[1] else 0.0


def fit_operators(features: Mapping[str, Mapping[str, Any]], pairs: Iterable[Mapping[str, Any]],
                  heads, types, eta=.90, kmax=4, theta=15, variant="aster", anchor="paired"):
    """Fit M2 then M3, recomputing both for every module replacement.

    features maps IDs to raw-model ell and heads; optional original=True and
    split="fit" metadata identifies original-input anchor observations.
    Pair orientation is always the saved left -> right orientation.
    """
    variants = {
        "aster": "aster", "per_type_mean": "per_type_mean", "mean": "per_type_mean",
        "mean_difference": "per_type_mean",
        "no_C": "no_C", "no_c": "no_C", "no_F": "no_F", "no_f": "no_F",
        "orthogonal": "orthogonal", "leace": "leace", "soft_angle": "soft_angle",
    }
    if variant not in variants:
        raise ValueError("Unknown operator variant: " + str(variant))
    variant = variants[variant]
    if anchor not in ("paired", "original", "zero"):
        raise ValueError("Unknown reference anchor: " + str(anchor))
    heads = sorted(set(int(h) for h in heads))
    fit_pairs = [
        dict(p) for p in pairs if p.get("split") == "fit" and p.get("valid", True)
        and p["kind"] in KINDS
    ]
    selected = defaultdict(list)
    before = Counter(p["kind"] for p in fit_pairs)
    for pair in fit_pairs:
        left, right = features[pair["left"]], features[pair["right"]]
        if pair["kind"] in ("A", "P"):
            effect = float(right["ell"]) - float(left["ell"])
            if abs(effect) < EPSILON:
                continue
        selected[pair["kind"]].append(pair)
    original_ids, original_origin = _original_ids(features, fit_pairs)
    action_pairs = [p for p in fit_pairs if p["kind"] == "A"]
    info = {
        "variant": variant, "anchor": anchor, "eta": eta, "kmax": kmax, "theta": theta,
        "epsilon": EPSILON, "fitting_split": "fit",
        "pair_counts_before_effect_filter": {k: before[k] for k in KINDS},
        "pair_counts_after_effect_filter": {k: len(selected[k]) for k in KINDS},
        "signal_pairs_filtered_by_effect": False,
        "moment_centered": False, "pair_orientation": "saved_right_minus_left",
        "heads": {}, "total_edit_rank": 0, "number_of_heads": len(heads),
        "original_anchor_origin": original_origin,
        "paired_anchor_uses_effect_filter": False,
    }
    if not heads:
        return {}, info
    if not features:
        raise ValueError("Cannot fit selected heads without raw-model features")
    first = _feature_matrix(next(iter(features.values())))
    total_heads, d = first.shape
    if min(heads) < 0 or max(heads) >= total_heads:
        raise ValueError("Selected query-head index out of range")
    # Materialize only fitting data once per configuration.
    needed = {p[s] for p in fit_pairs for s in ("left", "right")} | set(original_ids)
    tensors = {key: _feature_matrix(features[key]) for key in needed}
    operators = {}
    for h in heads:
        allowed = _types_for(types, h)
        bases, moments, type_details = {}, {}, {}
        for kind in KINDS:
            ps = selected[kind]
            delta = torch.stack([tensors[p["right"]][h] - tensors[p["left"]][h] for p in ps]) if ps else torch.empty((0, d), dtype=torch.float64)
            basis, moment, detail = energy_basis(delta, eta, kmax, variant == "per_type_mean")
            bases[kind], moments[kind] = basis, moment
            detail["included_as_nuisance"] = kind in allowed
            detail["included_as_signal"] = kind in ("C", "F") and variant != "no_" + kind
            detail["subtypes"] = dict(Counter(p.get("subtype", "") for p in ps))
            type_details[kind] = detail
        un = orth(torch.cat([bases[k] for k in allowed], dim=1)) if allowed else _empty(d)
        protection = [k for k in ("C", "F") if variant != "no_" + k]
        us = orth(torch.cat([bases[k] for k in protection], dim=1))
        nbar, b, geometry = oblique_factors(un, us, theta, variant="soft_angle" if variant == "soft_angle" else "aster")
        energy = {}
        for kind in allowed:
            retained_energy = _energy_in(nbar, moments[kind])
            in_merged = _energy_in(un, moments[kind])
            energy[kind] = {
                "total": float(torch.trace(moments[kind])),
                "in_merged_nuisance_basis": in_merged,
                "in_retained_principal_directions": retained_energy,
                "discarded_by_angle": max(0.0, in_merged - retained_energy),
                "outside_merged_nuisance_basis": max(0.0, float(torch.trace(moments[kind])) - in_merged),
            }
        if anchor == "zero":
            mu = torch.zeros(d, dtype=torch.float64)
            anchor_source = "zero_origin"
            anchor_n = 0
        elif anchor == "paired" and action_pairs:
            mu = torch.stack([tensors[p[side]][h] for p in action_pairs for side in ("left", "right")]).mean(0)
            anchor_source = "all_unfiltered_fit_A_pair_sides"
            anchor_n = 2 * len(action_pairs)
        else:
            if not original_ids:
                raise ValueError("No fit original input for the requested reference mean")
            mu = torch.stack([tensors[key][h] for key in original_ids]).mean(0)
            anchor_source = original_origin
            anchor_n = len(original_ids)
        leace = None
        n = nbar
        if variant == "orthogonal":
            b = nbar.T
        elif variant == "leace":
            ps = [p for kind in allowed for p in selected[kind]]
            categories = sorted({p["kind"] + ":" + str(p.get("subtype", "")) for p in ps})
            category = {name: i for i, name in enumerate(categories)}
            x, concept = [], []
            for pair in ps:
                ci = category[pair["kind"] + ":" + str(pair.get("subtype", ""))]
                for side, sign in (("left", -0.5), ("right", 0.5)):
                    x.append(tensors[pair[side]][h])
                    label = torch.zeros(len(categories), dtype=torch.float64)
                    label[ci] = sign
                    concept.append(label)
            x = torch.stack(x) if x else torch.empty((0, d), dtype=torch.float64)
            concept = torch.stack(concept) if concept else torch.empty((0, 0), dtype=torch.float64)
            n, b, leace = leace_factors(x, concept)
            # The standard LEACE ablation uses its own empirical concept-data
            # mean, as required for the canonical affine minimum-distortion map.
            # Anchor-only comparisons remain separate ASTER configurations.
            aster_reference = mu.clone()
            if len(x) and anchor == "paired":
                mu = x.mean(0)
                anchor_source = "LEACE_empirical_concept_data_mean"
                anchor_n = len(x)
            leace.update(
                concept_categories=categories,
                concept_encoding="(construction side 0/1 - 0.5) times nuisance type/subtype one-hot; no node truth",
                affine_anchor=anchor_source,
                affine_anchor_is_LEACE_training_mean=bool(len(x) and torch.allclose(mu, x.mean(0), atol=1e-10, rtol=1e-10)),
                difference_from_ASTER_reference_mean=float(torch.linalg.vector_norm(mu-aster_reference)),
                preserves_signal_by_design=False,
            )
        op = {
            "N": n.float().contiguous(), "B": b.float().contiguous(),
            "mu": mu.float().contiguous(), "S": us.float().contiguous(),
            "U_N": un.float().contiguous(),
            "type_bases": {kind: basis.float().contiguous() for kind, basis in bases.items()},
            "variant": variant,
        }
        operators[h] = op
        check = algebra_diagnostics(op["N"], op["B"], op["S"],
                                    None if variant == "leace" else theta,
                                    variant not in ("orthogonal", "leace"), variant=variant)
        info["heads"][str(h)] = {
            "types": allowed, "per_type": type_details,
            "merged_nuisance_rank": un.shape[1], "merged_signal_rank": us.shape[1],
            "geometry": geometry, "nuisance_energy": energy,
            "edit_rank": n.shape[1], "anchor_source": anchor_source,
            "anchor_observations": anchor_n, "algebra": check, "leace": leace,
            "retained_nuisance_geometry_is_actual_edit_basis": variant != "leace",
        }
        info["total_edit_rank"] += n.shape[1]
    return operators, info


def layer_matched_random(heads, types, seed, heads_per_layer=32):
    """Randomize within layers, preserving A-only/P-only/both counts exactly."""
    rng = random.Random(seed)
    by_layer = defaultdict(lambda: defaultdict(list))
    for h in sorted(set(int(h) for h in heads)):
        by_layer[h // heads_per_layer][tuple(_types_for(types, h))].append(h)
    randomized, randomized_types = [], {}
    for layer in sorted(by_layer):
        slots = list(range(layer * heads_per_layer, (layer + 1) * heads_per_layer))
        rng.shuffle(slots)
        offset = 0
        for kinds, original in sorted(by_layer[layer].items()):
            assigned = slots[offset: offset + len(original)]
            offset += len(original)
            randomized.extend(assigned)
            randomized_types.update({h: list(kinds) for h in assigned})
    return sorted(randomized), randomized_types


def total_edit_rank(operators):
    return sum(int(op["N"].shape[1]) for op in operators.values())


def apply_operator(vector, operator, amplification=1.0):
    """CPU convenience function matching the online low-rank convention."""
    z = _double(vector)
    n, b, mu = (_double(operator[k]) for k in ("N", "B", "mu"))
    edited = z - n @ (b @ (z - mu))
    if amplification != 1.0:
        s = _double(operator["S"])
        edited = edited + (amplification - 1.0) * s @ (s.T @ (edited - mu))
    return edited


def soft_angle_from_saved(operators, theta=45):
    """Reuse saved bases and anchors; change only the principal-angle weights."""
    result, diagnostics = {}, {}
    for head, old in operators.items():
        n, b, geometry = oblique_factors(old["U_N"], old["S"], theta, variant="soft_angle")
        op = dict(old, N=n.float().contiguous(), B=b.float().contiguous(), variant="soft_angle")
        result[int(head)] = op
        check = algebra_diagnostics(op["N"], op["B"], op["S"], theta, variant="soft_angle")
        u, s = _double(old["U_N"]), _double(old["S"])
        q, values, vh = torch.linalg.svd(u-s@(s.T@u), full_matrices=False)
        high = values >= math.sin(math.radians(theta))
        high_update = u @ vh[high].T @ (q[:, high].T / values[high, None])
        stored_hard = _double(old["N"]) @ _double(old["B"])
        d = _double(op["N"]) @ _double(op["B"])
        error = (torch.eye(u.shape[0], dtype=u.dtype)-d) @ u
        diagnostics[str(head)] = dict(
            geometry=geometry, algebra=check,
            high_angle_vs_stored_hard_error=_matrix_norm(high_update-stored_hard),
            residual_frobenius=float(torch.linalg.vector_norm(error)),
            residual_fraction=float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(u))
                if u.numel() else 0.0,
            attenuation_trace=float(torch.trace(u.T@d@u)),
            nuisance_rank=u.shape[1], hard_rank=old["N"].shape[1],
        )
    return result, diagnostics
