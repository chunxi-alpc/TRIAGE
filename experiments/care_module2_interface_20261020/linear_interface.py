"""Numerical reference for a typed module-two interface.

These NumPy functions validate algebra, not neural efficacy. They do not load a
model, learn source/determinacy semantics, or implement a Transformer hook.
All input activations must come from an independently audited training split.
RoPE cos/sin must come from the actual model; no position convention is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _array(value, ndim=None, name="array"):
    x = np.asarray(value, dtype=np.float64)
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return x


def _basis(value, dimension=None):
    b = _array(value, 2, "basis")
    if dimension is not None and b.shape[0] != dimension:
        raise ValueError("basis dimension mismatch")
    if b.shape[1] > b.shape[0] or not np.allclose(
        b.T @ b, np.eye(b.shape[1]), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("basis columns must be orthonormal; recalibrate coordinates after rotation")
    return b


@dataclass(frozen=True)
class SpectralFit:
    basis: np.ndarray
    singular_values: np.ndarray
    paired_mean: np.ndarray
    centered: bool
    rank: int
    captured_squared_energy: float
    sample_count: int


def fit_paired_subspace(positive, negative, rank, *, center=False):
    """Fit right singular vectors of paired differences, not raw activations.

    `positive[i]` and `negative[i]` must be the same semantic pair. The caller
    owns pairing and training-only provenance. Centering is an explicit option;
    centering removes a constant difference and can erase a mean behavior axis.
    Energy refers to squared singular values, not their cumulative sum.
    """
    p, n = _array(positive, 2, "positive"), _array(negative, 2, "negative")
    if p.shape != n.shape:
        raise ValueError("paired matrices must have identical shape")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError("rank must be a positive integer")
    d = p - n
    mean = d.mean(axis=0)
    fit = d - mean if center else d
    _, singular, vh = np.linalg.svd(fit, full_matrices=False)
    tol = max(fit.shape) * np.finfo(float).eps * singular[0]
    numerical_rank = int(np.count_nonzero(singular > tol))
    if numerical_rank < rank:
        raise ValueError(f"requested rank {rank} exceeds numerical rank {numerical_rank}")
    basis = vh[:rank].T.copy()
    # SVD signs are arbitrary. Orient against the training mean when possible;
    # this is a convention, not evidence of causal meaning.
    for j in range(rank):
        orient = float(basis[:, j] @ mean)
        if abs(orient) < 1e-12:
            orient = float(basis[np.argmax(abs(basis[:, j])), j])
        if orient < 0:
            basis[:, j] *= -1
    return SpectralFit(
        basis=basis,
        singular_values=singular,
        paired_mean=mean,
        centered=bool(center),
        rank=int(rank),
        captured_squared_energy=float(np.sum(singular[:rank] ** 2) / np.sum(singular ** 2)),
        sample_count=p.shape[0],
    )


def fit_mean_direction(positive, negative):
    p, n = _array(positive, 2, "positive"), _array(negative, 2, "negative")
    if p.shape != n.shape:
        raise ValueError("paired matrices must have identical shape")
    mean = (p - n).mean(axis=0)
    length = np.linalg.norm(mean)
    if length <= 1e-12:
        raise ValueError("paired mean has no usable direction")
    return mean / length


def covariance_mixture_basis(relevance_cov, applicability_cov, beta, rank, *, normalize_trace=True):
    """Mix symmetric PSD operators then re-estimate a rank-r eigenspace.

    Training-derived covariances are trace-normalized by default, as specified
    in the experiment protocol. Disable this only for an explicitly different
    operator definition; beta otherwise mixes quantities on different scales.
    A convex combination of projectors is generally not itself a projector.
    It must not silently be substituted for B B^T in a coordinate replacement.
    """
    a = _array(relevance_cov, 2, "relevance covariance")
    b = _array(applicability_cov, 2, "applicability covariance")
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("covariance operators must be square and have matching dimensions")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= a.shape[0]:
        raise ValueError("invalid rank")
    for value in (a, b):
        if not np.allclose(value, value.T, atol=1e-10, rtol=1e-10):
            raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(value).min() < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
    if normalize_trace:
        if np.trace(a) <= 1e-12 or np.trace(b) <= 1e-12:
            raise ValueError("trace normalization requires nonzero covariance energy")
        a, b = a / np.trace(a), b / np.trace(b)
    mixture = (1 - beta) * a + beta * b
    vals, vecs = np.linalg.eigh(mixture)
    if vals[-rank] <= max(1.0, abs(vals[-1])) * 1e-12:
        raise ValueError("requested mixture rank includes an unsupported direction")
    return vecs[:, -rank:][:, ::-1], mixture


def signed_key_projection(keys, basis, coefficients):
    """Edit one chosen KV head, with one signed coefficient per token.

    Input keys are after the model's K normalization and before RoPE. The
    coefficient expresses an intended intervention, not a guaranteed score sign.
    Unselected tokens must be given exactly zero. Do not repeat a shared KV edit
    once for each associated query head.
    """
    k = _array(keys, 2, "keys")
    b = _basis(basis, k.shape[1])
    c = _array(coefficients, 1, "coefficients")
    if c.shape != (k.shape[0],):
        raise ValueError("exactly one coefficient is required per token")
    return k + c[:, None] * ((k @ b) @ b.T)


def replace_coordinates(residual, basis, target, strength=1.0):
    """h' = h + strength * B (mu - B^T h), preserving the complement.

    Target centroids must have been measured in this exact orthonormal basis.
    Independently estimated source/determinacy axes are not assumed orthogonal.
    """
    h = _array(residual, 1, "residual")
    b = _basis(basis, len(h))
    mu = _array(target, 1, "target")
    if mu.shape != (b.shape[1],) or not math.isfinite(strength):
        raise ValueError("invalid target coordinates or strength")
    return h + float(strength) * (b @ (mu - b.T @ h))


def interchange_coordinates(base, donor, basis):
    """Swap coordinates and preserve the complement; no IIA label is invented.

    The exchanged basis can still carry answer content. This algebra does not
    establish content isolation; cross-content donor experiments are required.
    """
    h, s = _array(base, 1, "base"), _array(donor, 1, "donor")
    if h.shape != s.shape:
        raise ValueError("base and donor dimensions differ")
    b = _basis(basis, len(h))
    return replace_coordinates(h, b, b.T @ s)


def independent_axis_basis(axes):
    """Return QR basis plus change of coordinates; reject collapsed axes.

    With V=Q R, an old coefficient target a in V a becomes R a in Q. This
    function does not attach source/determinacy semantics to Q's columns.
    An old projection target c=V^T h is different: solve R^T z=c, or estimate
    centroids again in Q. Do not treat c as a synthesis coefficient a.
    """
    v = _array(axes, 2, "axes")
    if v.shape[1] > v.shape[0] or np.linalg.matrix_rank(v) < v.shape[1]:
        raise ValueError("axes are linearly dependent; a two-dimensional plane is unsupported")
    q, r = np.linalg.qr(v, mode="reduced")
    return q, r


def apply_rotary(x, cosine, sine):
    """Qwen/Llama rotate-half convention, with model-supplied cos and sin."""
    x = _array(x, 3, "tensor")  # tokens, heads, head dimension
    c, s = _array(cosine, 2, "cosine"), _array(sine, 2, "sine")
    t, _, d = x.shape
    if d % 2 or c.shape != (t, d) or s.shape != (t, d):
        raise ValueError("invalid rotary dimensions")
    half = np.concatenate((-x[..., d // 2:], x[..., :d // 2]), axis=-1)
    return x * c[:, None, :] + half * s[:, None, :]


def causal_gqa_logits(queries, keys, cosine, sine):
    """Float64 geometry reference; no cache, padding, or BF16 kernel emulation.

    Returns [query head, query token, key token]. Future keys are -inf. This
    reference only verifies indices/algebra; actual-model P0 remains required.
    """
    q, k = _array(queries, 3, "queries"), _array(keys, 3, "keys")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1]:
        raise ValueError("query/key dimensions or GQA grouping are inconsistent")
    qr, kr = apply_rotary(q, cosine, sine), apply_rotary(k, cosine, sine)
    kr = np.repeat(kr, q.shape[1] // k.shape[1], axis=1)
    score = np.einsum("ihd,jhd->hij", qr, kr) / np.sqrt(q.shape[2])
    future = np.triu(np.ones((q.shape[0], q.shape[0]), dtype=bool), k=1)
    score[:, future] = -np.inf
    return score


def counterfactual_pair_split_audit(rows):
    """Check declared IDs, not the truth of their semantic-family assignments.

    If any base_case_id is supplied, all rows must provide it. All declared
    pair/base siblings must share a family and split. Raw-text/content duplicate
    discovery remains a separate benchmark audit.
    """
    rows = list(rows)
    check_base = any("base_case_id" in row for row in rows)
    cases, families, siblings, pair_families, bases = set(), {}, {}, {}, {}
    for row in rows:
        required = {"case_id", "family_id", "pair_id", "split"}
        if not required <= row.keys() or any(not isinstance(row[k], str) or not row[k] for k in required):
            raise ValueError("missing or invalid split identifiers")
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError("unrecognized split")
        if row["case_id"] in cases:
            raise ValueError("duplicate case")
        cases.add(row["case_id"])
        for mapping, key in ((families, row["family_id"]), (siblings, row["pair_id"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("semantic family or counterfactual pair crosses splits")
            mapping[key] = row["split"]
        pair = row["pair_id"]
        if pair in pair_families and pair_families[pair] != row["family_id"]:
            raise ValueError("counterfactual pair is assigned to different families")
        pair_families[pair] = row["family_id"]
        if check_base:
            base = row.get("base_case_id")
            if not isinstance(base, str) or not base:
                raise ValueError("base_case_id must be present on every row")
            identity = (row["family_id"], row["split"])
            if base in bases and bases[base] != identity:
                raise ValueError("base-case siblings are assigned to different families or splits")
            bases[base] = identity
    if not cases:
        raise ValueError("empty split")
    return {"cases": len(cases), "families": len(families), "pairs": len(siblings),
            "base_cases": len(bases) if check_base else None}
