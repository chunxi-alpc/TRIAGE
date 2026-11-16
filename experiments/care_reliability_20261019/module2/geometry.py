"""CPU reference mathematics. These checks are not model inference results."""
from __future__ import annotations

import math
import numpy as np


def rotate_half(x):
    x = np.asarray(x)
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimension must be even")
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate((-b, a), axis=-1)


def rotate(x, cosine, sine):
    return np.asarray(x) * cosine + rotate_half(x) * sine


def anchor_factors(q_anchor_group, rotated_directions, angle_floor=0.1,
                   cosine_tolerance=1e-8):
    """Return *unsigned* smooth factors and strict GQA consensus eligibility.

    q_anchor_group: [query heads in one KV group, head dimension].
    rotated_directions: [action tokens, head dimension], without label sign.
    Label sign MUST be multiplied after these factors; including it here and
    multiplying the signed direction would cancel the desired label sign.
    The mean-query bound applies to the mean query. Every eligible group head
    has the requested sign, but its magnitude need not obey that same bound.
    """
    q = np.asarray(q_anchor_group, dtype=np.float64)
    u = np.asarray(rotated_directions, dtype=np.float64)
    if q.ndim != 2 or u.ndim != 2 or q.shape[1] != u.shape[1]:
        raise ValueError("Expected q[G,D] and u[M,D]")
    if angle_floor <= 0:
        raise ValueError("angle_floor must be positive")
    if not np.isfinite(q).all() or not np.isfinite(u).all():
        raise ValueError("Non-finite anchor geometry")
    d = q.shape[1]
    qnorm = np.linalg.norm(q, axis=1)
    unorm = np.linalg.norm(u, axis=1)
    dot = q @ u.T
    denominator = qnorm[:, None] * unorm[None, :]
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    eligible = (np.all(cosine > cosine_tolerance, axis=0)
                | np.all(cosine < -cosine_tolerance, axis=0))
    mean_q = q.mean(axis=0)
    z = mean_q @ u.T / math.sqrt(d)
    epsilon = angle_floor ** 2 * float(mean_q @ mean_q) * unorm ** 2 / d
    gamma = np.divide(z, z * z + epsilon, out=np.zeros_like(z), where=(z * z + epsilon) > 0)
    gamma[~eligible] = 0.0
    kappa = z * gamma
    return dict(gamma=gamma, eligible=eligible, cosine=cosine,
                anchor_z=z, anchor_kappa=kappa)


def anchor_audit(actual_score_changes, sign, eligible=None, tolerance=1e-7):
    """Separate nonnegative sign certificates, strict sign and coverage.

    Inputs are realized post-RoPE score differences [G,M], including all zero
    entries at skipped action tokens. This does not certify later queries.
    """
    delta = np.asarray(actual_score_changes, dtype=np.float64)
    if delta.ndim != 2 or not np.isfinite(delta).all() or sign not in (-1, 1):
        raise ValueError("Invalid actual anchor deltas/sign")
    active = np.abs(delta) > tolerance
    signed = sign * delta
    if eligible is None:
        eligible = np.ones(delta.shape[1], dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (delta.shape[1],):
        raise ValueError("Eligibility mask length mismatch")
    return dict(
        anchor_nonnegative_fraction=float(np.mean(signed >= -tolerance)),
        anchor_strict_correct_fraction=float(np.mean(signed > tolerance)),
        anchor_nonzero_fraction=float(np.mean(active)),
        anchor_active_correct_fraction=(float(np.mean(signed[active] > tolerance))
                                        if active.any() else None),
        anchor_wrong_fraction=float(np.mean(signed < -tolerance)),
        anchor_signed_mean=float(np.mean(signed)),
        anchor_rms=float(np.sqrt(np.mean(delta ** 2))),
        anchor_absolute_max=float(np.max(np.abs(delta), initial=0)),
        target_token_count=int(delta.shape[1]),
        retained_token_count=int(eligible.sum()),
        retained_token_fraction=float(eligible.mean()),
        anchor_group_all_nonnegative=bool(np.all(signed >= -tolerance)),
        anchor_actual_deltas=delta.tolist(),
    )


def bad_anchor_tokens(actual_score_changes, sign, tolerance=1e-7):
    """Any query head with a wrong-sign realized change rejects that key."""
    return np.any(sign * np.asarray(actual_score_changes) < -tolerance, axis=0)


def random_controls(dimension, rank, signal, seed):
    """Fixed controls; no outcomes, validation scores or test cases are used.

    random_projector has exactly the B4 rank; random_vector is a rank-one
    direction for the mean contrast. shuffled_mean flips balanced training
    pair labels BEFORE their first moment. Its second moment would be
    unchanged, so this is deliberately not a shuffled B4 projector.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != dimension or len(signal) < 2:
        raise ValueError("At least two training pair vectors are required")
    if not 1 <= rank <= dimension:
        raise ValueError("Invalid projector rank")
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")[0]
    random_vector = rng.normal(size=dimension)
    random_vector /= np.linalg.norm(random_vector)
    swaps = np.ones(len(signal), dtype=np.int8)
    swaps[:len(signal) // 2] = -1
    rng.shuffle(swaps)
    shuffled = np.mean(signal * swaps[:, None], axis=0)
    if np.linalg.norm(shuffled) <= 1e-12:
        raise ValueError("Frozen shuffled mean is zero; do not redraw after results")
    return dict(random_projector=basis @ basis.T, random_vector=random_vector,
                shuffled_mean=shuffled, pair_label_swaps=swaps)


def bf16_round(x):
    """Finite CPU float32 -> BF16 round-to-nearest-even reference.

    Kept as float32 values because NumPy need not provide a BF16 dtype.
    It is only a reference check, not a claim of CUDA/SDPA equivalence.
    """
    a = np.asarray(x, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Reference rounding only supports finite tensors")
    bits = a.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_rotate(x, cosine, sine):
    return bf16_round(bf16_round(bf16_round(x) * bf16_round(cosine))
                      + bf16_round(rotate_half(bf16_round(x)) * bf16_round(sine)))


def scalar_search(evaluate, target, max_relative_error=0.15):
    """Bounded closest-value search along a discrete BF16 perturbation path.

    evaluate(a) returns (changed_keys, scalar_actual_rms). The numerical
    path is discrete, so we report and enforce an approximate RMS match.
    No answer score or outcome is supplied to this search. BF16 score RMS
    need not be monotone, so a failed bisection is followed by a fixed grid
    over its original bracket; the acceptance threshold is unchanged.
    """
    if target < 0 or not math.isfinite(target):
        raise ValueError("Nonnegative finite target required")
    if target == 0:
        keys, actual = evaluate(0.0)
        if actual != 0:
            raise AssertionError("Zero coefficient did not reproduce keys")
        return keys, dict(scale=0.0, actual_rms=0.0, target_rms=0.0,
                          relative_error=0.0, calibration_status="zero")
    lo, hi = 0.0, 0.01
    best = None
    candidate_count = 0

    def candidate(a):
        nonlocal best, candidate_count
        candidate_count += 1
        changed, rms = evaluate(a)
        if not math.isfinite(rms) or rms < 0:
            raise ValueError("Non-finite or negative RMS")
        item = (abs(rms - target), a, changed, rms)
        if best is None or item[0] < best[0]:
            best = item
        return rms

    for _ in range(30):
        if candidate(hi) >= target:
            break
        hi *= 2
    else:
        raise RuntimeError("Cannot bracket the specified actual-score RMS")
    original_bracket_hi = hi
    for _ in range(28):
        mid = (lo + hi) / 2
        if candidate(mid) < target:
            lo = mid
        else:
            hi = mid
    initial_relative_error = best[0] / target
    search_method = 'bisection'
    fallback_candidate_count = 0
    if initial_relative_error > max_relative_error:
        # The rounded, fixed-query RMS can decrease at a key-rounding boundary.
        # Bisection may discard a feasible region; this bounded, deterministic
        # grid uses no answer values and changes neither direction nor budget.
        search_method = 'bisection_then_fixed_grid'
        fallback_candidate_count = 4097
        for i in range(fallback_candidate_count):
            candidate(original_bracket_hi * i / (fallback_candidate_count - 1))
    error, scale, changed, actual = best
    relative_error = error / target
    if relative_error > max_relative_error:
        raise RuntimeError(f"RMS calibration outside frozen tolerance: {relative_error:.6g}")
    return changed, dict(scale=float(scale), actual_rms=float(actual), target_rms=float(target),
                          relative_error=float(relative_error), calibration_status="matched",
                          search_method=search_method, candidate_count=candidate_count,
                          fallback_candidate_count=fallback_candidate_count,
                          original_bracket_hi=float(original_bracket_hi),
                          bisection_relative_error=float(initial_relative_error))
