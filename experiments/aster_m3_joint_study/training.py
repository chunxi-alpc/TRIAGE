"""Convex 16-dimensional task/AP fitting with original pairwise CF bounds.

No backbone loading, representation fitting, file writes, or GPU work on import.
S and J share this exact objective and solver; only lambda_ap changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import time
import numpy as np
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.special import expit

from experiments.aster_cr_retrain_paired_v2_20261111.training import pair_weights

DELTA = 0.2923349
REGULARIZATION = 0.01
TOLERANCE = 1e-7
SEED = 20261115


class ProtocolConflict(ValueError):
    """Requires explicit resolution; never silently select a competing protocol."""


def uid(row):
    return row.get("uid", row.get("view_id", row.get("id")))


def check_sources(train_sources, validation_sources):
    train, valid = set(train_sources), set(validation_sources)
    if not train or train & valid:
        raise ProtocolConflict("Empty training sources or train/validation source overlap")


def fit_intercept(scores, labels):
    x, y = np.asarray(scores, float), np.asarray(labels, float)
    if x.shape != y.shape or x.ndim != 1 or not len(x):
        raise ValueError("Intercept needs nonempty aligned score and label vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Invalid calibration scores or labels")
    if not 0 < y.sum() < len(y):
        raise ProtocolConflict("Calibration labels contain one class: no finite unregularized intercept")
    derivative = lambda b: float(np.mean(expit(x + b) - y))
    lo, hi = -1.0, 1.0
    while derivative(lo) > 0:
        lo *= 2
    while derivative(hi) < 0:
        hi *= 2
    return float(brentq(derivative, lo, hi, xtol=1e-12))


@dataclass
class Problem:
    t: np.ndarray
    ell: np.ndarray
    y: np.ndarray
    task_weights: np.ndarray
    ap_x: np.ndarray
    ap_y: np.ndarray
    ap_weights: np.ndarray
    cf_x: np.ndarray
    mu: np.ndarray
    b0: float
    task_scale: float
    ap_scale: float
    natural_ids: list
    ap_ids: list
    cf_ids: list
    train_sources: list
    natural_sources: list
    pair_sources: list
    ap_kinds: list


def make_problem(natural, pairs, ell, traces, train_sources, validation_sources=()):
    """Filter all fitting information to the explicit training-source allowlist."""
    check_sources(train_sources, validation_sources)
    allowed = set(train_sources)
    nv = [v for v in natural if v["source_id"] in allowed
          and v.get("split", "fit") == "fit" and v.get("variant", "full") == "full"
          and v.get("label") is not None]
    ps = [p for p in pairs if p["source_id"] in allowed and p.get("split", "fit") == "fit"]
    ids = [uid(v) for v in nv]
    if not ids or len(ids) != len(set(ids)):
        raise ProtocolConflict("Natural training nodes are empty or duplicated")
    t = np.stack([traces[k] for k in ids]).astype(float)
    if t.shape[1] != 16 or not np.all(np.isfinite(t)):
        raise ProtocolConflict("Expected finite fold-specific 16-dimensional traces")
    x = np.array([ell[k] for k in ids], float)
    y = np.array([v["label"] for v in nv], float)
    b0 = fit_intercept(x, y)
    # Same unweighted natural nodes as historical intercept fitting, no AP endpoints.
    weights = np.full(len(nv), 1.0 / len(nv))
    mu = weights @ t
    centered = t - mu
    ap = [p for p in ps if p["kind"] in ("A", "P")]
    cf = [p for p in ps if p["kind"] in ("C", "F")]
    if not ap or not cf:
        raise ProtocolConflict("Both AP and CF training pairs are required")
    pair_delta = lambda p: np.asarray(traces[p["right"]]) - np.asarray(traces[p["left"]])
    ax = np.stack([pair_delta(p) for p in ap]).astype(float)
    ay = np.array([ell[p["right"]] - ell[p["left"]] for p in ap], float)
    cx = np.stack([pair_delta(p) for p in cf]).astype(float)
    if not all(np.all(np.isfinite(a)) for a in (ax, ay, cx)):
        raise ProtocolConflict("Nonfinite pair scores or trace coordinates")
    aw = pair_weights(ap, "E1", total=1.0)
    task_scale = max(float(weights @ (np.logaddexp(0, x+b0) - y*(x+b0))), 1e-12)
    ap_scale = max(float(aw @ np.square(ay)), 1.0)
    return Problem(centered, x, y, weights, ax, ay, aw, cx, mu, b0,
                   task_scale, ap_scale, ids, [p["pair_id"] for p in ap],
                   [p["pair_id"] for p in cf], sorted(allowed),
                   [v["source_id"] for v in nv], [p["source_id"] for p in ap+cf],
                   [p["kind"] for p in ap])


def objective(problem, w, lambda_ap, regularization=REGULARIZATION):
    if lambda_ap < 0 or regularization < 0:
        raise ValueError("Loss weights must be nonnegative")
    w = np.asarray(w, float)
    z = problem.ell + problem.b0 + problem.t @ w
    residual = problem.ap_y + problem.ap_x @ w
    task_raw = float(problem.task_weights @ (np.logaddexp(0, z) - problem.y*z))
    ap_raw = float(problem.ap_weights @ np.square(residual))
    task = task_raw / problem.task_scale
    ap = ap_raw / problem.ap_scale
    regularizer = regularization * float(w @ w)
    gt = problem.t.T @ (problem.task_weights * (expit(z)-problem.y)) / problem.task_scale
    ga = 2 * problem.ap_x.T @ (problem.ap_weights * residual) / problem.ap_scale
    gr = 2 * regularization * w
    terms = dict(task=task, ap=ap, regularization=regularizer,
                 weighted_ap=lambda_ap*ap, task_raw_bce=task_raw, ap_raw_mse=ap_raw,
                 total=task+lambda_ap*ap+regularizer)
    return terms["total"], gt + lambda_ap*ga + gr, terms


def gradient_diagnostics(problem, w, lambda_ap):
    _, gt_gr, _ = objective(problem, w, 0)
    gr = 2*REGULARIZATION*w
    gt = gt_gr - gr
    residual = problem.ap_y + problem.ap_x @ w
    ga = 2*problem.ap_x.T @ (problem.ap_weights*residual) / problem.ap_scale
    denom = np.linalg.norm(gt)*np.linalg.norm(ga)
    return dict(task_gradient=gt.tolist(), ap_gradient=ga.tolist(),
                task_norm=float(np.linalg.norm(gt)), ap_norm=float(np.linalg.norm(ga)),
                weighted_ap_norm=float(np.linalg.norm(lambda_ap*ga)),
                regularization_norm=float(np.linalg.norm(gr)),
                task_ap_cosine=float(gt@ga/denom) if denom > 1e-15 else None,
                task_derivative_along_update=float(gt@w),
                weighted_ap_derivative_along_update=float(lambda_ap*ga@w))


def fit(problem, lambda_ap, method="J"):
    if method not in ("S", "J") or (method == "S" and lambda_ap != 0):
        raise ValueError("S is exactly the shared objective with lambda_ap=0")
    started = time.monotonic()
    initial = np.zeros(problem.t.shape[1])
    result = minimize(lambda w: objective(problem, w, lambda_ap)[:2], initial,
                      jac=True, method="SLSQP",
                      constraints=[LinearConstraint(problem.cf_x, -DELTA, DELTA)],
                      options=dict(ftol=1e-12, maxiter=3000, disp=False))
    attempts=[]
    def record_attempt(answer, tolerance):
        attempts.append(dict(solver="SLSQP",ftol=tolerance,success=bool(answer.success),
                             status=int(answer.status),message=str(answer.message),
                             iterations=int(answer.nit),objective=float(answer.fun),
                             cf_max_violation=max(0.,float(np.max(np.abs(problem.cf_x@answer.x)))-DELTA)))
    record_attempt(result,1e-12)
    # SLSQP can return status 8 after reaching a feasible optimum at a strict
    # objective tolerance. Restart the same convex problem at that point with
    # ordinary numerical precision. No loss weight, data, or bound is changed.
    if not result.success:
        result=minimize(lambda w: objective(problem,w,lambda_ap)[:2],result.x,
                        jac=True,method="SLSQP",
                        constraints=[LinearConstraint(problem.cf_x,-DELTA,DELTA)],
                        options=dict(ftol=1e-10,maxiter=3000,disp=False))
        record_attempt(result,1e-10)
    w = np.asarray(result.x, float)
    cf = problem.cf_x @ w
    violation = max(0., float(np.max(np.abs(cf))) - DELTA)
    if not result.success or violation > TOLERANCE:
        raise RuntimeError(f"Optimizer failed: {result.message}; CF violation={violation}")
    corrected = problem.ell + problem.t @ w
    bw = fit_intercept(corrected, problem.y)
    _, _, start_terms = objective(problem, initial, lambda_ap)
    _, _, terms = objective(problem, w, lambda_ap)
    return dict(method=method, lambda_ap=float(lambda_ap), regularization=REGULARIZATION,
                delta=DELTA, tolerance=TOLERANCE, w=w.tolist(), mu=problem.mu.tolist(),
                b_shared=problem.b0, b_independent=bw,
                train_sources=problem.train_sources,
                calibration_ids=problem.natural_ids, calibration_sources=problem.natural_sources,
                task_ids=problem.natural_ids, task_weights=problem.task_weights.tolist(),
                ap_ids=problem.ap_ids, ap_weights=problem.ap_weights.tolist(),
                ap_weights_by_kind={k:float(sum(a for a,c in zip(problem.ap_weights,problem.ap_kinds) if c==k))
                                    for k in ("A", "P")},
                cf_ids=problem.cf_ids, task_scale=problem.task_scale, ap_scale=problem.ap_scale,
                initial_losses=start_terms, final_losses=terms,
                loss_changes={k:terms[k]-start_terms[k] for k in terms},
                gradient_initial=gradient_diagnostics(problem, initial, lambda_ap),
                gradient_final=gradient_diagnostics(problem, w, lambda_ap),
                cf_max_abs=float(np.max(np.abs(cf))), cf_max_violation=violation,
                cf_active_count=int(np.sum(DELTA-np.abs(cf)<1e-6)),
                cf_training_residuals=(np.abs(cf)-DELTA).tolist(),
                constraint_count=len(cf), natural_count=len(problem.y),
                solver="SLSQP", success=bool(result.success), iterations=int(result.nit),
                solver_attempts=attempts,
                optimizer_message=str(result.message), seconds=time.monotonic()-started,
                deterministic=True, seed=SEED)


def correction(model, traces):
    return (np.asarray(traces, float) - np.asarray(model["mu"])) @ np.asarray(model["w"])


def score(model, ell, traces, calibration="independent"):
    if calibration not in ("shared", "independent"):
        raise ValueError(calibration)
    return np.asarray(ell, float) + correction(model, traces) + model["b_"+calibration]


def centered_e1(w, old_b, mu, b0):
    """Translate historical E1 into the centered convention without changing full scores."""
    return dict(method="R2", w=np.asarray(w,float).tolist(), mu=np.asarray(mu,float).tolist(),
                b_shared=float(b0), b_independent=float(old_b+np.asarray(mu)@np.asarray(w)),
                b_historical=float(old_b), reused=True)


def choose_candidate(candidates, baseline, natural_tolerance=0.005):
    """One predeclared inner split; only independently calibrated paths select J."""
    if not candidates or any(r["metrics"].get("natural_path_accuracy") is None or
                             r["metrics"].get("interference_path_accuracy") is None for r in candidates):
        raise ValueError("Inner selection requires complete natural and interference paths")
    threshold = baseline["natural_path_accuracy"] - natural_tolerance
    eligible = [r for r in candidates if r["metrics"]["natural_path_accuracy"] >= threshold]
    pool = eligible or candidates
    def key(row):
        m = row["metrics"]
        primary = [] if eligible else [-m["natural_path_accuracy"]]
        return (*primary, -m["interference_path_accuracy"],
                m["cf_direction_degradation_count"], m["cf_over_budget_rate"],
                m["cf_correction_rms"], row["lambda_ap"])
    selected = min(pool, key=key)
    return dict(lambda_ap=selected["lambda_ap"], natural_requirement_met=bool(eligible),
                threshold=threshold, criterion="natural drop <=0.005; interference; CF damage; smaller lambda",
                calibration="independent", candidates=len(candidates))
