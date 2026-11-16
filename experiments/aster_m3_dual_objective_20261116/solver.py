"""CPU float64 unconstrained dual-objective ridge, with fixed E1 weights."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import expit
from experiments.aster_cr_retrain_paired_v2_20261111.training import pair_weights, WEIGHT_TOTAL

def solve(X_ap, y_ap, X_cf, ap_weights, cf_weights, eta, regularization):
    if eta < 0 or regularization <= 0:
        raise ValueError("eta must be nonnegative and lambda strictly positive")
    X, y, E, a, c = [np.asarray(v, dtype=np.float64)
                      for v in (X_ap, y_ap, X_cf, ap_weights, cf_weights)]
    Q = X.T @ (a[:, None] * X) + eta * E.T @ (c[:, None] * E)
    Q += regularization * np.eye(X.shape[1])
    q = X.T @ (a * y)
    w = -cho_solve(cho_factor(Q, lower=True), q)
    residual = Q @ w + q
    terms = dict(L_AP=float(a @ (y + X @ w)**2),
                 L_CF=float(c @ (E @ w)**2),
                 regularization_term=float(regularization * (w @ w)))
    eig = np.linalg.eigvalsh(Q)
    diagnostics = dict(solver="float64 Cholesky", rank=len(w), eta=float(eta),
        regularization=float(regularization), condition_number=float(np.linalg.cond(Q)),
        min_eigenvalue=float(eig[0]), hessian_min_eigenvalue=float(2*eig[0]),
        stationary_residual=float(np.linalg.norm(residual)),
        relative_stationary_residual=float(np.linalg.norm(residual)/(np.linalg.norm(q)+1e-30)),
        gradient_norm=float(2*np.linalg.norm(residual)),
        AP_weight_sum=float(a.sum()), CF_weight_sum=float(c.sum()), **terms)
    diagnostics["objective"] = terms["L_AP"]+eta*terms["L_CF"]+terms["regularization_term"]
    return w, diagnostics

def center(w, coordinates, rho):
    return float(np.asarray(rho, float) @ (np.asarray(coordinates, float) @ w))

def calibrate(raw, coordinates, labels, w, mu_c):
    """Original E1 natural-node intercept calibration, beta fixed at one."""
    base = np.asarray(raw) + coordinates @ w - mu_c
    labels = np.asarray(labels, float)
    derivative = lambda b: float(np.mean(expit(base + b) - labels))
    lo, hi = -1., 1.
    while derivative(lo) > 0: lo *= 2
    while derivative(hi) < 0: hi *= 2
    return float(brentq(derivative, lo, hi, xtol=1e-12))
