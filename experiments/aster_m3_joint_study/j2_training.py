"""Exploratory linear J2 with labeled CF direction loss and asymmetric bounds.

The original S/J implementation and artifacts remain unchanged. J2 uses the
same 16 weights, training center, fixed b0, task/AP losses and L2 penalty.
Only the CF objective and signed CF correction bounds differ.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

from . import training as tr


def _cf_context(problem, cf_raw_difference, cf_label_difference, cf_weights):
    d = np.asarray(cf_raw_difference, dtype=float)
    s = np.asarray(cf_label_difference, dtype=float)
    v = np.asarray(cf_weights, dtype=float)
    n = len(problem.cf_ids)
    if problem.t.shape[1] != 16 or problem.cf_x.shape != (n, 16):
        raise tr.ProtocolConflict("J2 requires the original 16-dimensional CF coordinates")
    if not n or any(a.shape != (n,) for a in (d, s, v)):
        raise tr.ProtocolConflict("J2 CF arrays must align with problem.cf_ids")
    if not all(np.all(np.isfinite(a)) for a in (d, s, v)):
        raise tr.ProtocolConflict("J2 CF scores, labels and weights must be finite")
    if not np.all(np.isin(s, [-1., 1.])):
        raise tr.ProtocolConflict("J2 CF label differences must be right_label-left_label in {-1,+1}")
    if np.any(v < 0) or not np.isclose(v.sum(), 1., rtol=0., atol=1e-10):
        raise tr.ProtocolConflict("J2 CF category/source weights must be nonnegative and sum to one")
    scale = max(float(v @ np.logaddexp(0., -s*d)), 1e-12)
    return d, s, v, scale


def _direction(problem, w, d, s, v, scale):
    margin = s * (d + problem.cf_x @ w)
    raw = float(v @ np.logaddexp(0., -margin))
    gradient = -(problem.cf_x.T @ (v * s * expit(-margin))) / scale
    return raw/scale, gradient, raw


def _objective(problem, w, d, s, v, scale, lambda_ap, lambda_cf):
    if not np.isfinite(lambda_cf) or lambda_cf < 0:
        raise ValueError("lambda_cf must be finite and nonnegative")
    value, gradient, base_terms = tr.objective(problem, w, lambda_ap)
    direction, direction_gradient, raw_direction = _direction(problem, w, d, s, v, scale)
    terms = dict(base_terms, base_total=value, cf_direction_raw_loss=raw_direction,
                 cf_direction=direction, weighted_cf_direction=lambda_cf*direction,
                 total=value+lambda_cf*direction)
    return terms["total"], gradient+lambda_cf*direction_gradient, terms


def objective_j2(problem, w, cf_raw_difference, cf_label_difference, cf_weights,
                 lambda_ap, lambda_cf):
    """Return total objective, its analytic gradient and all measured loss terms."""
    d, s, v, scale = _cf_context(problem, cf_raw_difference, cf_label_difference, cf_weights)
    return _objective(problem, np.asarray(w, float), d, s, v, scale, lambda_ap, lambda_cf)


def _gradients(problem, w, d, s, v, scale, lambda_ap, lambda_cf):
    result = tr.gradient_diagnostics(problem, w, lambda_ap)
    _, gd, _ = _direction(problem, w, d, s, v, scale)
    _, total, _ = _objective(problem, w, d, s, v, scale, lambda_ap, lambda_cf)
    def cosine(a, b):
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/denominator) if denominator > 1e-15 else None
    result.update(cf_direction_gradient=gd.tolist(),
                  cf_direction_norm=float(np.linalg.norm(gd)),
                  weighted_cf_direction_norm=float(np.linalg.norm(lambda_cf*gd)),
                  task_cf_direction_cosine=cosine(np.asarray(result["task_gradient"]), gd),
                  ap_cf_direction_cosine=cosine(np.asarray(result["ap_gradient"]), gd),
                  total_gradient_norm=float(np.linalg.norm(total)))
    return result


def fit_j2(problem, cf_raw_difference, cf_label_difference, cf_weights,
           lambda_ap, lambda_cf):
    """Fit J2; all CF arrays must follow the order in problem.cf_ids.

    Signed correction z_j = (y_right-y_left) * (E_j @ w) satisfies
    -DELTA <= z_j <= 2*DELTA. Negative z_j reduces the evidence response
    along the true label direction. Positive z_j may use twice the old budget.
    No extra trainable intercept, slack variable, gate or feature is introduced.
    """
    started = time.monotonic()
    if not np.isfinite(lambda_ap) or lambda_ap < 0:
        raise ValueError("lambda_ap must be finite and nonnegative")
    d, s, v, scale = _cf_context(problem, cf_raw_difference, cf_label_difference, cf_weights)
    signed_cf = s[:, None] * problem.cf_x
    bounds = LinearConstraint(signed_cf, -tr.DELTA, 2*tr.DELTA)
    initial = np.zeros(16)
    evaluate = lambda w: _objective(problem, w, d, s, v, scale, lambda_ap, lambda_cf)
    attempts = []

    def violation(w):
        signed = signed_cf @ w
        return max(0., float(np.max(-tr.DELTA-signed)),
                   float(np.max(signed-2*tr.DELTA)))

    def solve(start, ftol):
        result = minimize(lambda w: evaluate(w)[:2], start, jac=True, method="SLSQP",
                          constraints=[bounds], options=dict(ftol=ftol, maxiter=3000, disp=False))
        attempts.append(dict(solver="SLSQP", ftol=ftol, success=bool(result.success),
                             status=int(result.status), message=str(result.message),
                             iterations=int(result.nit), objective=float(result.fun),
                             cf_max_violation=violation(result.x)))
        return result

    result = solve(initial, 1e-12)
    if not result.success:
        # Retry numerical convergence at the same point without changing any bound.
        result = solve(result.x, 1e-10)
    w = np.asarray(result.x, float)
    max_violation = violation(w)
    if not result.success or max_violation > tr.TOLERANCE:
        raise RuntimeError(f"J2 optimizer failed: {result.message}; asymmetric CF violation={max_violation}")

    cf = problem.cf_x @ w
    signed = s*cf
    lower_residuals = -tr.DELTA-signed
    upper_residuals = signed-2*tr.DELTA
    pair_residuals = np.maximum(lower_residuals, upper_residuals)
    original_residuals = np.abs(cf)-tr.DELTA
    bw = tr.fit_intercept(problem.ell+problem.t@w, problem.y)
    _, _, initial_losses = evaluate(initial)
    _, _, final_losses = evaluate(w)
    lower_active = int(np.sum(np.abs(lower_residuals) < 1e-6))
    upper_active = int(np.sum(np.abs(upper_residuals) < 1e-6))
    return dict(method="J2", exploratory=True, lambda_ap=float(lambda_ap), lambda_cf=float(lambda_cf),
                regularization=tr.REGULARIZATION, delta=tr.DELTA, tolerance=tr.TOLERANCE,
                parameter_count=16, w=w.tolist(), mu=problem.mu.tolist(),
                b_shared=problem.b0, b_independent=bw,
                train_sources=problem.train_sources, natural_count=len(problem.y),
                calibration_ids=problem.natural_ids, calibration_sources=problem.natural_sources,
                task_ids=problem.natural_ids, task_weights=problem.task_weights.tolist(),
                ap_ids=problem.ap_ids, ap_weights=problem.ap_weights.tolist(),
                ap_weights_by_kind={k:float(sum(a for a,c in zip(problem.ap_weights,problem.ap_kinds) if c==k))
                                    for k in ("A", "P")},
                cf_ids=problem.cf_ids, cf_sources=problem.pair_sources[len(problem.ap_ids):],
                cf_weights=v.tolist(), cf_raw_difference=d.tolist(), cf_label_difference=s.tolist(),
                task_scale=problem.task_scale, ap_scale=problem.ap_scale, cf_direction_scale=scale,
                initial_losses=initial_losses, final_losses=final_losses,
                loss_changes={k:final_losses[k]-initial_losses[k] for k in final_losses},
                gradient_initial=_gradients(problem,initial,d,s,v,scale,lambda_ap,lambda_cf),
                gradient_final=_gradients(problem,w,d,s,v,scale,lambda_ap,lambda_cf),
                cf_correction=cf.tolist(), cf_signed_correction=signed.tolist(),
                cf_lower_residuals=lower_residuals.tolist(), cf_upper_residuals=upper_residuals.tolist(),
                cf_training_residuals=pair_residuals.tolist(), cf_max_violation=max_violation,
                max_violation=max_violation,
                cf_max_abs=float(np.max(np.abs(cf))),
                cf_max_signed_increase=max(0.,float(np.max(signed))),
                cf_max_signed_decrease=max(0.,float(np.max(-signed))),
                cf_active_count=lower_active+upper_active,
                cf_lower_active_count=lower_active, cf_upper_active_count=upper_active,
                cf_original_budget_residuals=original_residuals.tolist(),
                cf_original_budget_excess=np.maximum(original_residuals,0.).tolist(),
                cf_original_budget_max_excess=max(0.,float(np.max(original_residuals))),
                cf_original_budget_excess_count=int(np.sum(original_residuals>1e-9)),
                cf_original_budget_reporting_tolerance=1e-9,
                constraint_count=len(cf), signed_constraint_lower=-tr.DELTA,
                signed_constraint_upper=2*tr.DELTA,
                guarantee="Training signed CF correction in [-delta,2delta]; original symmetric delta guarantee is relaxed",
                solver="SLSQP", success=bool(result.success), iterations=int(result.nit),
                solver_attempts=attempts, optimizer_message=str(result.message),
                seconds=time.monotonic()-started, deterministic=True, seed=tr.SEED)
