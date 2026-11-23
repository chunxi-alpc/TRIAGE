"""Fit the two missing factorial cells using only archived 16-dimensional coordinates."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from experiments.aster_m3_joint_study import data, j2_training, training as tr
from experiments.aster_m3_joint_study.evaluation import selection_metrics
from experiments.aster_m3_joint_study.j2_run import cf_arrays
from experiments.aster_m3_joint_study.run import keys, method_scores

CELLS = ("J", "Direction", "Asymmetric", "J2")
CALIBRATIONS = ("shared", "independent")
CANDIDATES = (0.1, 1.0, 10.0)


def bounds(problem, signs, asymmetric):
    # s is exactly y_right - y_left. Its sign matters only for asymmetric bounds.
    if asymmetric:
        return signs[:, None] * problem.cf_x, -tr.DELTA, 2 * tr.DELTA
    return problem.cf_x, -tr.DELTA, tr.DELTA


def residuals(problem, signs, w, asymmetric):
    matrix, lower, upper = bounds(problem, signs, asymmetric)
    values = matrix @ w
    residual = np.maximum(lower - values, values - upper)
    return dict(lower=lower, upper=upper, train_pairs=len(values),
                min_value=float(values.min()), max_value=float(values.max()),
                max_residual=float(residual.max()),
                max_violation=max(0.0, float(residual.max())),
                violation_count_above_solver_tolerance=int(np.sum(residual > tr.TOLERANCE)),
                solver_tolerance=tr.TOLERANCE)


def fit_cell(problem, cf, method, lambda_ap, lambda_cf):
    raw, signs, weights = cf
    asymmetric = method == "Asymmetric"
    if asymmetric and lambda_cf == 0:
        def objective(w):
            return tr.objective(problem, w, lambda_ap)
    elif method == "Direction" and lambda_cf in CANDIDATES:
        context = j2_training._cf_context(problem, raw, signs, weights)
        def objective(w):
            return j2_training._objective(problem, w, *context, lambda_ap, lambda_cf)
    else:
        raise ValueError(f"Invalid missing cell: {method}, lambda_CF={lambda_cf}")
    constraint = LinearConstraint(*bounds(problem, signs, asymmetric))
    initial = np.zeros(problem.t.shape[1])
    start = initial
    attempts = []
    for tolerance in (1e-12, 1e-10):
        result = minimize(lambda w: objective(w)[:2], start, method="SLSQP", jac=True,
                          constraints=[constraint], options=dict(ftol=tolerance, maxiter=3000))
        w = result.x
        diagnostic = residuals(problem, signs, w, asymmetric)
        attempts.append(dict(ftol=tolerance, status=int(result.status), message=result.message,
                             iterations=int(result.nit), objective=float(result.fun),
                             max_violation=diagnostic["max_violation"]))
        if result.success:
            break
        start = w
    if not result.success or diagnostic["max_violation"] > tr.TOLERANCE:
        raise RuntimeError(f"{method}: optimizer or training budget failure: {attempts}")
    return dict(method=method, lambda_ap=lambda_ap, lambda_cf=lambda_cf, w=w.tolist(),
                mu=problem.mu.tolist(), b_shared=problem.b0,
                b_independent=tr.fit_intercept(problem.ell + problem.t @ w, problem.y),
                task_scale=problem.task_scale, ap_scale=problem.ap_scale,
                cf_direction_scale=None if asymmetric else context[-1],
                train_sources=problem.train_sources, task_ids=problem.natural_ids,
                calibration_ids=problem.natural_ids, ap_ids=problem.ap_ids, cf_ids=problem.cf_ids,
                regularization=tr.REGULARIZATION, delta=tr.DELTA, asymmetric_bounds=asymmetric,
                initial_losses=objective(initial)[2], final_losses=objective(w)[2],
                train_constraints=diagnostic, solver="SLSQP", solver_attempts=attempts)


def archive_scores(archive, method):
    directory = archive if method == "J" else archive / "j2"
    result = {}
    with gzip.open(directory / "evaluation/predictions.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            values = {method + "_" + cal: row["scores"][method + "_" + cal]
                      for cal in CALIBRATIONS}
            if row["view_id"] in result:
                raise tr.ProtocolConflict(f"Duplicate {method} archival prediction {row['view_id']}")
            result[row["view_id"]] = dict(R0=row["raw_score"], **values)
    return result


def validate_archives(study, archive, predictions):
    """Before fitting, compare every old score with saved weights and calibrations."""
    checks = []
    expected = set()
    for fold in study.folds["folds"]:
        fid = f"fold{fold['fold']}"
        scope = study.scope(fold["validation_sources"])
        ids = keys(scope)
        all_ids = [fid + "::" + uid for uid in ids]
        expected.update(all_ids)
        ell, traces = cache(archive, fid)
        for method in ("J", "J2"):
            model_dir = (archive if method == "J" else archive / "j2") / "models" / fid
            model = data.read(model_dir / (method + ".json"))
            if set(model["train_sources"]) != set(fold["train_sources"]):
                raise tr.ProtocolConflict(f"{fid}/{method} archive train-source mismatch")
            recalculated = method_scores(ell, traces, ids, {method: model}, model["b_shared"])
            for cal in CALIBRATIONS:
                name = method + "_" + cal
                record = data.read(model_dir / (name + "_calibration.json"))
                b = record["b"]
                if b != model["b_" + cal] or record["fit_ids"] != model["calibration_ids"]:
                    raise tr.ProtocolConflict(f"{fid}/{name} archive calibration mismatch")
                diffs = [abs(recalculated[name][uid] - predictions[method][key][name])
                         for uid, key in zip(ids, all_ids)]
                if max(diffs) > 1e-10 or any(predictions[method][key]["R0"] != ell[uid]
                                             for uid, key in zip(ids, all_ids)):
                    raise tr.ProtocolConflict(f"{fid}/{name} archive score mismatch")
                checks.append(dict(fold=fid, method=name, views=len(ids), intercept=b,
                                   max_score_difference=max(diffs)))
    for method, scores in predictions.items():
        if set(scores) != expected:
            raise tr.ProtocolConflict(f"{method} archive prediction coverage differs")
    return checks


def cache(archive, fid):
    with np.load(archive / "models" / fid / "projected_cache.npz", allow_pickle=False) as z:
        ids = z["ids"].tolist()
        return dict(zip(ids, z["ell"].tolist())), dict(zip(ids, z["traces"].copy()))


def select_direction(study, ell, traces, fold, split, lambda_ap, directory):
    train = split["inner_train_sources"]
    valid = split["inner_validation_sources"]
    if len(train) != 48 or len(valid) != 16 or set(train) | set(valid) != set(fold["train_sources"]):
        raise tr.ProtocolConflict("The archived inner 48/16 split does not cover outer training sources")
    inner = tr.make_problem(study.natural, study.train_pairs, ell, traces,
                            train, valid + fold["validation_sources"])
    cf = cf_arrays(inner, study.train_pairs, ell)
    scope = study.scope(valid)
    ids = keys(scope)
    candidates = []
    for weight in CANDIDATES:
        model = fit_cell(inner, cf, "Direction", lambda_ap, weight)
        scores = method_scores(ell, traces, ids, {"Direction": model}, inner.b0)
        metrics = selection_metrics(**scope, scores=scores)
        data.write(directory / f"inner_Direction_{weight}.json", model)
        candidates.append(dict(lambda_cf=weight, metrics=metrics["Direction_independent"],
                               R1=metrics["R1"]))
    # The archived selector takes a generic lambda in the field named lambda_ap.
    choice = tr.choose_candidate([dict(c, lambda_ap=c["lambda_cf"]) for c in candidates],
                                 candidates[0]["R1"])
    choice["lambda_cf"] = choice.pop("lambda_ap")
    data.write(directory / "selection.json", dict(**split, inherited_lambda_ap=lambda_ap,
                                                   selection=choice, candidates=candidates))
    return choice["lambda_cf"]


def run(archive, output):
    cfg = data.read(archive / "configuration.json")
    study = data.StudyData(Path(cfg["historical_root"]), Path(cfg["cases_file"]),
                           Path(cfg["diagnostics_root"]), cfg["path_maps"])
    predictions = {name: archive_scores(archive, name) for name in ("J", "J2")}
    data.write(output / "b_archive_checks.json", validate_archives(study, archive, predictions))
    constraints = []
    def out_of_fold_rows():
        for fold in study.folds["folds"]:
            fid = f"fold{fold['fold']}"
            directory = output / "b_models" / fid
            ell, traces = cache(archive, fid)
            parent = data.read(archive / "models" / fid / "J.json")
            j2 = data.read(archive / "j2/models" / fid / "J2.json")
            lam = parent["lambda_ap"]
            split = data.read(archive / "selection" / fid / "split.json")
            chosen = select_direction(study, ell, traces, fold, split, lam, directory)
            train, valid = fold["train_sources"], fold["validation_sources"]
            outer = tr.make_problem(study.natural, study.train_pairs, ell, traces, train, valid)
            cf = cf_arrays(outer, study.train_pairs, ell)
            models = {"J": parent, "J2": j2}
            for name, weight in (("Direction", chosen), ("Asymmetric", 0.0)):
                model = fit_cell(outer, cf, name, lam, weight)
                model.update(fold=fid, validation_sources=valid, heads=parent["heads"],
                             M1_M2_directory=parent["M1_M2_directory"])
                models[name] = model
                data.write(directory / (name + ".json"), model)
            for name in CELLS:
                model = models[name]
                if (model["cf_ids"] != outer.cf_ids or model["task_ids"] != outer.natural_ids
                        or model["ap_ids"] != outer.ap_ids or model["lambda_ap"] != lam
                        or model["heads"] != parent["heads"]
                        or model["regularization"] != tr.REGULARIZATION
                        or not np.allclose(model["mu"], outer.mu, atol=1e-10, rtol=0)
                        or abs(model["b_shared"] - outer.b0) > 1e-10):
                    raise tr.ProtocolConflict(f"{fid}/{name} frozen inputs mismatch")
                diag = residuals(outer, cf[1], np.asarray(model["w"]), name in ("Asymmetric", "J2"))
                if diag["max_violation"] > tr.TOLERANCE:
                    raise tr.ProtocolConflict(f"{fid}/{name} training violation: {diag}")
                constraints.append(dict(fold=fid, method=name, lambda_ap=lam,
                                        lambda_cf=model.get("lambda_cf", 0.0), **diag))
            scope = study.scope(valid)
            ids = keys(scope)
            scores = method_scores(ell, traces, ids, {n: models[n] for n in ("Direction", "Asymmetric")},
                                   parent["b_shared"])
            source = {tr.uid(v): v["source_id"] for v in scope["path_views"] + scope["views"]}
            for uid in ids:
                key = fid + "::" + uid
                old = {**predictions["J"][key], **predictions["J2"][key]}
                new = {n: s[uid] for n, s in scores.items() if n.startswith(("Direction_", "Asymmetric_"))}
                yield dict(fold=fid, view_id=key, original_uid=uid, source_id=source[uid],
                           raw_score=ell[uid], scores={**old, **new})
            print(f"{fid}: 3 inner fits, 2 outer fits, direction lambda_CF={chosen}, {len(ids)} views",
                  flush=True)
    # Publish complete pooled scores after all five folds finish.
    temporary = output / "b_scores.partial.jsonl.gz"
    data.lines(temporary, out_of_fold_rows())
    temporary.replace(output / "b_scores.jsonl.gz")
    with (output / "b_train_constraints.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=constraints[0].keys())
        writer.writeheader()
        writer.writerows(constraints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve())

