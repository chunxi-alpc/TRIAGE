"""Pairwise evaluation, fixed output baselines, and diagnostic readouts.

All predictions use ell = logit(yes) - logit(no), with a tie assigned to no.
Fit data train the calibrators/probe; development data select fixed candidates.
Missing model scores remain failures in all semantically valid denominators.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import numpy as np

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def view_map(views):
    if isinstance(views, Mapping):
        return {str(k): dict(v, view_id=str(v.get("view_id", k))) for k, v in views.items()}
    return {str(v["view_id"]): v for v in views}


def margin(value):
    """Return a finite score, or None for a failed/missing scoring unit."""
    if isinstance(value, Mapping):
        if value.get("status") in ("failed", "error", "pending"):
            return None
        value = value.get("ell", value.get("score"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pred(value):
    value = margin(value)
    return None if value is None else int(value > 0)


def fraction(numerator, denominator):
    numerator, denominator = int(numerator), int(denominator)
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def _selected(row, split, cohort):
    return row.get("split") == split and (cohort is None or row.get("cohort", "main") == cohort)


def _original(row):
    if "original" in row:
        return bool(row["original"])
    return row.get("variant") in ("base0", "original")


def _pair_subtype(pair):
    value = str(pair.get("subtype", "")).lower()
    if value in ("clinical", "clinical_background", "background"):
        return "clinical"
    if value in ("preference", "treatment_preference"):
        return "preference"
    return value or "unspecified"


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0, mean=None, mean_abs=None, std=None, median=None,
                    p05=None, p95=None, rms=None)
    return dict(n=int(a.size), mean=float(a.mean()), mean_abs=float(np.abs(a).mean()),
                std=float(a.std()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                rms=float(np.sqrt(np.mean(a * a))))


def _kind_pairs(pairs, kind, subtype=None):
    return [p for p in pairs if p.get("kind", p.get("type")) == kind and
            (subtype is None or _pair_subtype(p) == subtype)]


def _pair_metric(pairs, views, scores):
    correct = available = 0
    for pair in pairs:
        left, right = str(pair["left"]), str(pair["right"])
        lp, rp = pred(scores.get(left)), pred(scores.get(right))
        available += int(lp is not None and rp is not None)
        correct += int(lp == int(views[left]["label"]) and rp == int(views[right]["label"]))
    return dict(**fraction(correct, len(pairs)), scored_pairs=available,
                missing_score_pairs=len(pairs) - available)


def _responses(pairs, views, scores):
    raw, truth_aligned = [], []
    for pair in pairs:
        a, b = margin(scores.get(str(pair["left"]))), margin(scores.get(str(pair["right"])))
        if a is None or b is None:
            continue
        difference = b - a
        raw.append(difference)
        y0, y1 = int(views[str(pair["left"])]["label"]), int(views[str(pair["right"])]["label"])
        if y0 != y1:
            truth_aligned.append((y1 - y0) * difference)
    return dict(total_pairs=len(pairs), score_response=_stats(raw),
                truth_aligned_response=_stats(truth_aligned))


def _correction(originals, candidate, baseline):
    baseline_correct = baseline_wrong = correction = regression = unavailable = 0
    for key, v in originals.items():
        old, new, gold = pred(baseline.get(key)), pred(candidate.get(key)), int(v["label"])
        if old is None:
            unavailable += 1
        elif old == gold:
            baseline_correct += 1
            regression += int(new != gold)
        else:
            baseline_wrong += 1
            correction += int(new == gold)
    return dict(corrected=fraction(correction, baseline_wrong),
                regressed=fraction(regression, baseline_correct),
                net_corrected=correction-regression,
                baseline_missing=unavailable, original_inputs=len(originals))


def _double_wrong(rows, candidate, full, masked):
    both_wrong = corrected = unavailable = 0
    for key, v in rows.items():
        a, b, c = pred(full.get(key)), pred(masked.get(key)), pred(candidate.get(key))
        gold = int(v["label"])
        if a is None or b is None:
            unavailable += 1
        elif a != gold and b != gold:
            both_wrong += 1
            corrected += int(c == gold)
    return dict(**fraction(corrected, both_wrong), baseline_missing=unavailable,
                eligible_inputs=len(rows), definition="same fixed node and same input view")


def _historical_fourway(views, scores):
    grouped = defaultdict(dict)
    aliases = {"original0": "base0", "original1": "base1",
               "action0": "swap0", "action1": "swap1"}
    for vid, v in views.items():
        key = aliases.get(v.get("variant"), v.get("variant"))
        if key in ("base0", "base1", "swap0", "swap1"):
            grouped[str(v["group_id"])][key] = (vid, int(v["label"]))
    eligible = [g for g in grouped.values() if len(g) == 4]
    correct = sum(all(pred(scores.get(vid)) == label for vid, label in g.values()) for g in eligible)
    return dict(**fraction(correct, len(eligible)), incomplete_groups=len(grouped)-len(eligible),
                name="historical fact-state by action-view four-input correctness")


def selection_key(result):
    """Development tie break shared by fixed output baselines."""
    macro = result.get("macro_PA")
    original = result.get("original_accuracy", {}).get("rate")
    return (float(macro) if macro is not None else -math.inf,
            float(original) if original is not None else -math.inf)


def evaluate(views, pairs, scores, split="test", cohort="main", bootstrap=2000, seed=20261105):
    """Evaluate all supplied methods; no filtering by model effect or correctness."""
    all_views = view_map(views)
    selected = {k: v for k, v in all_views.items() if _selected(v, split, cohort)}
    pairs = [p for p in pairs if _selected(p, split, cohort) and p.get("valid", True)]
    dangling = [p.get("pair_id", "") for p in pairs
                if str(p["left"]) not in selected or str(p["right"]) not in selected]
    if dangling:
        raise ValueError("Paired view metadata absent or split/cohort mismatched: " + str(dangling[:3]))
    originals = {k: v for k, v in selected.items() if _original(v)}
    results = {}
    source_ids = sorted({str(v["source_id"]) for v in selected.values()} |
                        {str(p["source_id"]) for p in pairs})
    for method, current in scores.items():
        result = {}
        for kind in ("A", "C", "F"):
            result["PA_" + kind] = _pair_metric(_kind_pairs(pairs, kind), selected, current)
        clinical = _pair_metric(_kind_pairs(pairs, "P", "clinical"), selected, current)
        preference = _pair_metric(_kind_pairs(pairs, "P", "preference"), selected, current)
        pooled = _pair_metric(_kind_pairs(pairs, "P"), selected, current)
        rates = [clinical["rate"], preference["rate"]]
        result["PA_P_clinical"] = clinical
        result["PA_P_preference"] = preference
        result["PA_P"] = dict(
            rate=sum(rates)/2 if all(r is not None for r in rates) else None,
            numerator=None, denominator=clinical["denominator"] + preference["denominator"],
            clinical=clinical, preference=preference, pooled_rate=pooled["rate"],
            weighting="equal clinical and preference weights",
            unspecified_pairs=len(_kind_pairs(pairs, "P", "unspecified")),
        )
        macro_parts = [result["PA_" + k]["rate"] for k in ("A", "P", "C", "F")]
        result["macro_PA"] = float(np.mean(macro_parts)) if all(x is not None for x in macro_parts) else None
        result["missing_pair_types"] = [k for k in ("A", "P", "C", "F") if result["PA_" + k]["rate"] is None]
        result["original_accuracy"] = fraction(
            sum(pred(current.get(k)) == int(v["label"]) for k, v in originals.items()), len(originals))
        result["all_input_accuracy"] = fraction(
            sum(pred(current.get(k)) == int(v["label"]) for k, v in selected.items()), len(selected))
        present = sum(margin(current.get(k)) is not None for k in selected)
        result["score_coverage"] = dict(**fraction(present, len(selected)), missing=len(selected)-present)
        result["historical_fourway"] = _historical_fourway(selected, current)
        result["responses"] = {kind: _responses(_kind_pairs(pairs, kind), selected, current)
                               for kind in ("A", "P", "C", "F")}
        result["responses"].update({"P_" + st: _responses(_kind_pairs(pairs, "P", st), selected, current)
                                   for st in ("clinical", "preference")})
        if "full" in scores:
            result["change_vs_full"] = _correction(originals, current, scores["full"])
            result["double_wrong_correction"] = {
                masked: _double_wrong(originals, current, scores["full"], scores[masked])
                for masked in ("direct", "semantic") if masked in scores
            }
            result["all_views_double_wrong_correction"] = {
                masked: _double_wrong(selected, current, scores["full"], scores[masked])
                for masked in ("direct", "semantic") if masked in scores
            }
        results[method] = result
    output = dict(split=split, cohort=cohort, methods=results, source_count=len(source_ids),
                  node_count=len({str(v["group_id"]) for v in selected.values()}),
                  input_count=len(selected), original_input_count=len(originals),
                  pair_counts={k: len(_kind_pairs(pairs, k)) for k in ("A", "P", "C", "F")},
                  primary_metric="four pair-type macro average; P equally weights clinical and preference",
                  missing_score_policy="failure in the complete valid denominator; response statistics report coverage",
                  historical_use="exploratory reuse of previously used guideline sources",
                  clinical_verification="automatic pre-experiment; awaiting physician verification")
    output["profile_response"] = profile_response(selected, pairs, scores)
    if bootstrap:
        output["confidence_intervals"] = source_bootstrap(selected, pairs, scores, n_resamples=bootstrap, seed=seed)
    return output


def source_bootstrap(views, pairs, scores, n_resamples=2000, seed=20261105):
    """One shared guideline-source resample for all methods and differences."""
    views = view_map(views)
    sources = sorted({str(v["source_id"]) for v in views.values()} | {str(p["source_id"]) for p in pairs})
    if not sources:
        return dict(n_sources=0, n_resamples=int(n_resamples), methods={})
    lookup = {s: i for i, s in enumerate(sources)}
    n = len(sources)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(n_resamples), n))
    weights = np.zeros((int(n_resamples), n), dtype=np.int32)
    np.add.at(weights, (np.arange(int(n_resamples))[:, None], indices), 1)
    primitives = {}
    for method, current in scores.items():
        buckets = {k: np.zeros((n, 2), dtype=np.float64)
                   for k in ("A", "P_clinical", "P_preference", "C", "F", "original_accuracy")}
        for pair in pairs:
            kind = pair.get("kind", pair.get("type"))
            key = "P_" + _pair_subtype(pair) if kind == "P" else kind
            if key not in buckets:
                continue
            a, b = str(pair["left"]), str(pair["right"])
            ok = pred(current.get(a)) == int(views[a]["label"]) and pred(current.get(b)) == int(views[b]["label"])
            buckets[key][lookup[str(pair["source_id"])]] += (int(ok), 1)
        for vid, v in views.items():
            if _original(v):
                buckets["original_accuracy"][lookup[str(v["source_id"])]] += (
                    int(pred(current.get(vid)) == int(v["label"])), 1)
        samples = {}
        for key, counts in buckets.items():
            total = weights @ counts
            rate = np.divide(total[:, 0], total[:, 1], out=np.full(len(weights), np.nan),
                             where=total[:, 1] > 0)
            samples[key if key == "original_accuracy" else "PA_" + key] = rate
        samples["PA_P"] = (samples["PA_P_clinical"] + samples["PA_P_preference"]) / 2
        samples["macro_PA"] = sum(samples["PA_" + k] for k in ("A", "P", "C", "F")) / 4
        primitives[method] = samples
    def ci(a):
        finite = a[np.isfinite(a)]
        return dict(low=float(np.quantile(finite, .025)) if finite.size else None,
                    high=float(np.quantile(finite, .975)) if finite.size else None,
                    valid_resamples=int(finite.size))
    methods = {}
    for method, samples in primitives.items():
        item = {key: ci(a) for key, a in samples.items()}
        item["differences"] = {}
        for baseline in ("full", "direct", "semantic"):
            if baseline in primitives and baseline != method:
                item["differences"][baseline] = {k: ci(a-primitives[baseline][k]) for k, a in samples.items()}
        methods[method] = item
    return dict(n_sources=n, n_resamples=int(n_resamples), seed=int(seed), confidence=.95,
                unit="guideline source, preserving all within-source nodes and pairs", methods=methods)


def profile_response(views, pairs, scores):
    """Equation (23), measured directly; accuracy never substitutes for it."""
    views = view_map(views)
    pairs = _kind_pairs(pairs, "P")
    if "full" not in scores:
        return {}
    out = {}
    for masked in ("direct", "semantic"):
        if masked not in scores:
            continue
        summary = {}
        for subtype in ("all", "clinical", "preference"):
            selected = pairs if subtype == "all" else [p for p in pairs if _pair_subtype(p) == subtype]
            full_delta, mask_delta, did, rows = [], [], [], []
            method_deltas = defaultdict(list)
            for p in selected:
                a, b = str(p["left"]), str(p["right"])
                fa, fb = margin(scores["full"].get(a)), margin(scores["full"].get(b))
                ma, mb = margin(scores[masked].get(a)), margin(scores[masked].get(b))
                if any(x is None for x in (fa, fb, ma, mb)):
                    continue
                df, dm = fb-fa, mb-ma
                full_delta.append(df); mask_delta.append(dm); did.append(df-dm)
                row = dict(pair_id=p.get("pair_id"), full_response=df, masked_response=dm,
                           difference_in_differences=df-dm, fused_responses={})
                for method, current in scores.items():
                    if method in ("full", "direct", "semantic"):
                        continue
                    va, vb = margin(current.get(a)), margin(current.get(b))
                    if va is not None and vb is not None:
                        row["fused_responses"][method] = vb-va
                        method_deltas[method].append(vb-va)
                rows.append(row)
            correlation = None
            if len(full_delta) >= 2 and np.std(full_delta) > 0 and np.std(mask_delta) > 0:
                correlation = float(np.corrcoef(full_delta, mask_delta)[0, 1])
            summary[subtype] = dict(total_pairs=len(selected), paired_scored=len(did),
                full_response=_stats(full_delta), masked_response=_stats(mask_delta),
                difference_in_differences=_stats(did), response_correlation=correlation,
                other_method_responses={m: _stats(v) for m, v in method_deltas.items()}, pairs=rows)
        out[masked] = summary
    return out


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1 / (1 + np.exp(-value[positive]))
    e = np.exp(value[~positive])
    out[~positive] = e / (1 + e)
    return out


def predict_fusion(spec, full, masked=None):
    """Apply a frozen JSON-serializable output method; scalar in gives scalar out."""
    family = spec["family"]
    f = np.asarray(full, dtype=np.float64)
    m = np.asarray(masked if masked is not None else full, dtype=np.float64)
    if family == "probability":
        weight = float(spec["weight"])
        p = weight * _sigmoid(f) + (1-weight) * _sigmoid(m)
        p = np.clip(p, np.finfo(np.float64).eps, 1-np.finfo(np.float64).eps)
        value = np.log(p) - np.log1p(-p)
    elif family == "logit":
        weight = float(spec["weight"])
        value = weight*f + (1-weight)*m
    elif family == "cad":
        alpha = float(spec["alpha"])
        value = (1+alpha)*f - alpha*m
    elif family == "reverse_cad":
        alpha = float(spec["alpha"])
        value = (1+alpha)*m - alpha*f
    elif family in ("logistic", "calibration"):
        coefficients = spec["coefficients"]
        if family == "calibration":
            value = float(coefficients[0])*(f if spec["route"] == "full" else m) + float(spec["intercept"])
        else:
            value = float(coefficients[0])*f + float(coefficients[1])*m + float(spec["intercept"])
    else:
        raise ValueError("Unknown frozen output family: " + str(family))
    return float(value) if value.ndim == 0 else value


def _fusion_predictions(spec, views, scores):
    full = scores.get("full", {})
    masked = scores.get(spec.get("masked_route", "full"), {})
    out = {}
    for vid in views:
        f, m = margin(full.get(vid)), margin(masked.get(vid))
        if spec["family"] == "calibration":
            if spec["route"] == "full" and f is not None:
                out[vid] = predict_fusion(spec, f, f)
            elif spec["route"] != "full" and m is not None:
                out[vid] = predict_fusion(spec, m, m)
        elif f is not None and m is not None:
            out[vid] = predict_fusion(spec, f, m)
    return out


def fit_fusions(views, pairs, scores, selection_cohort="main"):
    """Train only on fit; select one fixed configuration per family on dev."""
    from sklearn.linear_model import LogisticRegression
    views = view_map(views)
    fit_ids = [v for v in views if views[v].get("split") == "fit"]
    fitted, failures, candidates = {}, {}, defaultdict(list)
    full = scores.get("full", {})
    training = dict(split="fit", labels="node truth for output calibration only",
                    C=1.0, solver="lbfgs", fit_intercept=True, max_iter=2000,
                    clinical_verification="automatic pre-experiment")
    def logistic(name, route, family):
        routes = ["full", route] if family == "logistic" else [route]
        ids = [vid for vid in fit_ids if all(margin(scores.get(r, {}).get(vid)) is not None for r in routes)]
        labels = np.array([int(views[vid]["label"]) for vid in ids])
        if len(set(labels.tolist())) < 2:
            failures[name] = "fit data lack two labeled classes with available scores"
            return
        x = np.array([[margin(scores[r][vid]) for r in routes] for vid in ids], dtype=np.float64)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=20261105).fit(x, labels)
        spec = dict(family=family, route=route, masked_route=route,
                    coefficients=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                    training=dict(training, n=len(ids), sources=sorted({str(views[i]["source_id"]) for i in ids})))
        fitted[name] = spec
        candidates[name].append(spec)
    for route in ("full", "direct", "semantic"):
        if route in scores:
            logistic("calibrated_" + route, route, "calibration")
    for route in ("direct", "semantic"):
        if route not in scores:
            failures["full+" + route] = "masked score route unavailable"
            continue
        for family in ("probability", "logit"):
            name = family + "_" + route
            candidates[name] = [dict(family=family, masked_route=route, weight=w) for w in WEIGHTS]
        for family in ("cad", "reverse_cad"):
            name = family + "_" + route
            candidates[name] = [dict(family=family, masked_route=route, alpha=a) for a in ALPHAS]
        logistic("logistic_" + route, route, "logistic")
    chosen, outputs, selection = {}, {}, {}
    for name, pool in candidates.items():
        scored_candidates = []
        for i, spec in enumerate(pool):
            current = _fusion_predictions(spec, views, scores)
            result = evaluate(views, pairs, {"candidate": current}, split="dev",
                              cohort=selection_cohort, bootstrap=0)["methods"]["candidate"]
            scored_candidates.append((selection_key(result), -i, spec, current, result))
        if not scored_candidates:
            continue
        best = max(scored_candidates, key=lambda c: (c[0], c[1]))
        chosen[name] = dict(best[2], selected_on="dev", selection_cohort=selection_cohort)
        outputs[name] = best[3]
        selection[name] = dict(selected=chosen[name],
            development=dict(macro_PA=best[4]["macro_PA"], original_accuracy=best[4]["original_accuracy"],
                             missing_pair_types=best[4]["missing_pair_types"]),
            candidates=[dict(spec=c[2], macro_PA=c[4]["macro_PA"],
                             original_accuracy=c[4]["original_accuracy"]) for c in scored_candidates])
    best_method = max(chosen, key=lambda m: selection_key(
        {"macro_PA": selection[m]["development"]["macro_PA"],
         "original_accuracy": selection[m]["development"]["original_accuracy"]})) if chosen else None
    return dict(specs=chosen, scores=outputs, selection=selection, best_method=best_method,
                failures=failures, protocol=dict(fit_split="fit", selection_split="dev",
                test_used_for_selection=False, weights=list(WEIGHTS), alphas=list(ALPHAS),
                logistic_C=1.0, probability_definition="convex combination of normalized binary probabilities"))


def signal_probe(views, features, operators, full_scores, split="test", cohort="main"):
    """Read out signal on all full-model test errors; original-input errors are a subset.

    Training uses fit original inputs. Neither the probe nor its predictions
    enter online ASTER; all test input variants are eligible for diagnostics.
    """
    from sklearn.linear_model import LogisticRegression
    import torch
    views = view_map(views)
    heads = sorted(int(h) for h in operators)
    by_head = {int(h): value for h, value in operators.items()}
    def coordinates(vid):
        feature = features.get(vid)
        if feature is None or "heads" not in feature:
            return None
        z = torch.as_tensor(feature["heads"]).detach().cpu().double()
        z = z.reshape(-1, z.shape[-1])
        parts = []
        for h in heads:
            op = by_head[h]
            s = torch.as_tensor(op.get("S", torch.empty(z.shape[-1], 0))).detach().cpu().double()
            mu = torch.as_tensor(op["mu"]).detach().cpu().double()
            if s.numel():
                parts.append((s.T @ (z[h] - mu)).numpy())
        return np.concatenate(parts) if parts else None
    fit_ids = [vid for vid, v in views.items() if v.get("split") == "fit" and _original(v)]
    fit_rows = [(vid, coordinates(vid)) for vid in fit_ids]
    fit_rows = [(vid, x) for vid, x in fit_rows if x is not None and np.isfinite(x).all()]
    eligible = {vid: v for vid, v in views.items() if _selected(v, split, cohort)}
    wrong = {vid: v for vid, v in eligible.items()
             if pred(full_scores.get(vid)) is not None and pred(full_scores[vid]) != int(v["label"])}
    original_wrong = {vid: v for vid, v in wrong.items() if _original(v)}
    metadata = dict(fit_split="fit", evaluation_split=split, cohort=cohort,
                    feature_definition="concatenated S.T @ (z - mu) from selected unedited heads",
                    model="linear logistic regression C=1", online_use=False,
                    evaluation_pool="all valid input views misclassified by the unedited model",
                    full_error_count=len(wrong), original_full_error_count=len(original_wrong),
                    evaluation_inputs=len(eligible),
                    full_score_missing=sum(pred(full_scores.get(vid)) is None for vid in eligible),
                    fit_inputs=len(fit_rows))
    if not fit_rows or len({int(views[vid]["label"]) for vid, x in fit_rows}) < 2:
        return dict(metadata, status="unavailable", reason="no nonempty signal coordinates or fit classes")
    train_x = np.stack([x for _, x in fit_rows])
    labels = np.array([int(views[vid]["label"]) for vid, _ in fit_rows])
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=20261105).fit(train_x, labels)
    predictions = {}
    for vid in wrong:
        x = coordinates(vid)
        if x is None or not np.isfinite(x).all():
            continue
        score = float(clf.decision_function(x[None, :])[0])
        predictions[vid] = dict(ell=score, prediction=int(score > 0), label=int(wrong[vid]["label"]),
                                correct=int(score > 0) == int(wrong[vid]["label"]))
    return dict(metadata, status="complete", coordinate_dimension=int(train_x.shape[1]),
                accuracy=fraction(sum(v["correct"] for v in predictions.values()), len(wrong)),
                scored_errors=len(predictions), missing_errors=len(wrong)-len(predictions),
                original_error_accuracy=fraction(
                    sum(predictions[vid]["correct"] for vid in original_wrong if vid in predictions),
                    len(original_wrong)),
                original_scored_errors=sum(vid in predictions for vid in original_wrong),
                original_missing_errors=sum(vid not in predictions for vid in original_wrong),
                coefficients=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                predictions=predictions)


def path_metrics(records, n_resamples=2000, seed=20261105):
    """Summarize autonomous path runs, preserving failures in case denominators.

    Input is a list of {method,case_id,source_id,correct,status?}, one per planned
    method/case. Missing methods on a registered case count as failure.
    """
    records = list(records)
    methods = sorted({r["method"] for r in records})
    cases = {}
    values = {}
    for r in records:
        cases[str(r["case_id"])] = str(r["source_id"])
        values[(r["method"], str(r["case_id"]))] = r
    sources = sorted(set(cases.values()))
    rng = np.random.default_rng(seed)
    if sources and n_resamples:
        samples = rng.integers(0, len(sources), (n_resamples, len(sources)))
        weight = np.zeros((n_resamples, len(sources)), dtype=np.int32)
        np.add.at(weight, (np.arange(n_resamples)[:, None], samples), 1)
    else:
        weight = None
    results = {}
    for method in methods:
        complete = correct = 0
        bins = np.zeros((len(sources), 2), dtype=np.float64)
        for case_id, source_id in cases.items():
            r = values.get((method, case_id), {})
            ok = bool(r.get("correct", r.get("path_correct", False))) and r.get("status", "complete") == "complete"
            complete += int(bool(r) and r.get("status", "complete") == "complete")
            correct += int(ok)
            bins[sources.index(source_id)] += (int(ok), 1)
        result = dict(**fraction(correct, len(cases)), completed=complete,
                      missing_or_failed=len(cases)-complete)
        if weight is not None:
            total = weight @ bins
            rates = total[:, 0] / total[:, 1]
            result["ci95"] = [float(np.quantile(rates, .025)), float(np.quantile(rates, .975))]
        results[method] = result
    return dict(methods=results, cases=len(cases), sources=len(sources),
                execution="autonomous root-to-leaf traversal; exact full path correctness",
                bootstrap_resamples=n_resamples, bootstrap_seed=seed)
