"""Full evidence complement and exact ridge-QP dual; no feature truncation."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from scipy.optimize import brentq
from scipy.special import expit

DELTA = 0.2923349
GRID = (0.01, 1.0, 100.0)


def orthogonal_signal(S):
    S = np.asarray(S, dtype=np.float64)
    if not S.shape[1]:
        return S.copy()
    # Reorthogonalize the archived signal span only, never the AP feature space.
    q, r = la.qr(S, mode='economic')
    if np.any(np.abs(np.diag(r)) < 1e-12):
        raise ValueError('Archived signal basis is rank deficient')
    return q


def residual(z, S):
    z = np.asarray(z, dtype=np.float64)
    q = orthogonal_signal(S)
    r = z - (z @ q) @ q.T
    return r - (r @ q) @ q.T


def disjoint(*groups):
    sets = list(map(set, groups))
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Training/development/held-out sources overlap')


def calibrate(base, correction, labels):
    center = float(np.mean(correction))
    score = base + correction - center
    fun = lambda b: float(np.mean(expit(score + b) - labels))
    lo, hi = -1., 1.
    while fun(lo) > 0: lo *= 2
    while fun(hi) < 0: hi *= 2
    return center, float(brentq(fun, lo, hi, xtol=1e-12))


def solve(X, y, weights, E, regularization, delta=DELTA):
    """Solve 1/2*(||sqrt(W)(Xw+y)||²+lambda||w||²), |Ew|<=delta.

    Q=lambda I+X'WX. Its inverse is applied in the AP row space using a
    full thin QR (no singular-value/rank truncation). The dual is
    min_v .5 v'(E Q^-1 E')v -(E w0)'v +delta||v||_1.
    Only AP- and CF-sized dense matrices are factored, never d x d.
    Positive lambda makes the primal unique; Slater holds at w=0.
    """
    start = time.monotonic()
    X, E, y, weights = [np.asarray(a, np.float64) for a in (X, E, y, weights)]
    lam = float(regularization)
    if lam <= 0: raise ValueError('lambda must be positive')
    A = np.sqrt(weights)[:, None] * X
    B, R = la.qr(A.T, mode='economic', check_finite=False)
    small = R @ R.T + lam * np.eye(R.shape[0])
    chol = la.cho_factor(small, check_finite=False)
    def inv(v):
        projected = B.T @ v
        orth = v - B @ projected
        # Remove roundoff along B before dividing by a small lambda.
        orth -= B @ (B.T @ orth)
        return B @ la.cho_solve(chol, projected, check_finite=False) + orth / lam
    q = X.T @ (weights * y)
    # Compute w0 inside the AP span, without amplifying orthogonal roundoff.
    w0 = -B @ la.cho_solve(chol, R @ (np.sqrt(weights) * y), check_finite=False)
    V = inv(E.T)
    H = E @ V
    H = (H + H.T) * 0.5
    g = E @ w0
    m = len(E)
    # Diagonal change of coordinates improves the dual's conditioning.
    scale = np.sqrt(np.maximum(np.diag(H), 1e-20))
    Hs = H / scale[:, None] / scale[None, :]
    eye = sp.eye(m, format='csc')
    problem = osqp.OSQP()
    problem.setup(P=sp.block_diag((sp.csc_matrix(Hs), sp.csc_matrix((m,m))), format='csc'),
                  q=np.r_[-g / scale, delta / scale],
                  A=sp.vstack((sp.hstack((eye,-eye)), sp.hstack((-eye,-eye))), format='csc'),
                  l=np.full(2*m, -np.inf), u=np.zeros(2*m),
                  eps_abs=1e-11, eps_rel=1e-11, max_iter=100000,
                  polishing=True, verbose=False)
    result = problem.solve(raise_error=True)
    nu = result.x[:m] / scale
    w = w0 - V @ nu
    ec = E @ w
    violation = max(0., float(np.max(np.abs(ec), initial=0.) - delta))
    stationarity = X.T @ (weights*(X@w+y)) + lam*w + E.T@nu
    primal = .5*(np.sum(weights*(X@w+y)**2)+lam*(w@w))
    # Lagrangian lower bound evaluated with the full feature-space solution.
    dual = .5*np.sum(weights*y*y) + .5*q@w0 + g@nu - .5*nu@H@nu - delta*np.abs(nu).sum()
    diag = dict(solver='OSQP signed dual with AP thin QR', status=result.info.status,
                iterations=int(result.info.iter), osqp_primal_residual=float(result.info.prim_res),
                osqp_dual_residual=float(result.info.dual_res),
                primal_objective=float(primal*2), dual_objective=float(dual*2),
                absolute_duality_gap=float(abs(primal-dual)*2),
                relative_duality_gap=float(abs(primal-dual)/max(1.,abs(primal))),
                stationarity_inf=float(np.max(np.abs(stationarity),initial=0.)),
                constraint_max_violation=violation, max_CF_correction=float(np.max(np.abs(ec),initial=0.)),
                feature_dimension=X.shape[1], AP_pairs=len(X), CF_pairs=m,
                AP_QR_columns=B.shape[1], lambda_value=lam, delta=delta,
                seconds=time.monotonic()-start)
    if violation > 1e-7 or diag['relative_duality_gap'] > 1e-6:
        raise RuntimeError('QP did not converge to requested precision: '+str(diag))
    return w, diag, ec, nu
