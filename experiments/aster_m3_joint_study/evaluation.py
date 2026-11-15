"""CPU evaluation using the archived ASTER decoders and source bootstrap.

The caller supplies out-of-fold scores. This module never fits parameters or
reads training labels. For the final analysis, concatenate the five held-out
folds before calling ``evaluate``. Historical assets are only imported/read.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math

import numpy as np

from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy


DRAWS = 2000
NONINFERIORITY = -0.005


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_safe(record), ensure_ascii=False,
                                    allow_nan=False) + "\n")


def _is_natural(view):
    return (view.get("kind", "original") == "original" and
            bool(view.get("original") or
                 ("original" not in view and
                  view.get("variant", "original") in ("full", "original"))))


def _prepare(views, pairs, cases, path_views, scores):
    views = list({legacy.uid(v): v for v in views}.values())
    # Preserve case/node aliases: different path positions can share one exact
    # prompt UID even though its score is computed once.
    path_views = list(path_views)
    cases, pairs = list(cases), list(pairs)
    all_views = {legacy.uid(v): v for v in path_views}
    all_views.update({legacy.uid(v): v for v in views})
    evaluation_ids = {legacy.uid(v) for v in views}
    scores = {name: dict(values) for name, values in scores.items()}
    if "raw" not in scores:
        if "R0" not in scores:
            raise ValueError("scores must contain raw or R0")
        scores["raw"] = dict(scores["R0"])
    for name, values in scores.items():
        missing = set(all_views) - set(values)
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} scoring inputs; first={min(missing)}")
        if any(not math.isfinite(float(values[key])) for key in all_views):
            raise ValueError(f"{name}: non-finite score")
    if "R0" in scores and any(scores["R0"][key] != scores["raw"][key]
                              for key in all_views):
        raise ValueError("R0 and raw must be identical")
    decisions, _ = legacy.decisions_for(list(all_views.values()), scores)
    pair_records = legacy.observations(views, pairs, scores, decisions)
    path_records = legacy.path_predictions(cases, path_views, scores) if cases else []
    nodes = []
    for key, view in all_views.items():
        record = dict(view_id=key, source_id=view["source_id"],
                      label=view.get("label"), label_available=view.get("label") is not None,
                      natural=key in evaluation_ids and _is_natural(view),
                      in_evaluation_views=key in evaluation_ids,
                      case_id=view.get("case_id"), gid=view.get("gid"),
                      variant=view.get("variant"), fold=view.get("fold"),
                      raw_score=float(scores["raw"][key]),
                      scores={name: float(values[key]) for name, values in scores.items()},
                      predictions={name: int(values[key]) for name, values in decisions.items()},
                      score_shift_from_raw={name: float(values[key] - scores["raw"][key])
                                            for name, values in scores.items()})
        nodes.append(record)
    return dict(nodes=nodes, pairs=pair_records, paths=path_records,
                scores=scores, cases=cases, path_views=path_views)


def _arrays(prepared, boot):
    nodes, pairs, paths = (prepared[k] for k in ("nodes", "pairs", "paths"))
    names = list(prepared["scores"])
    bypair, bypath = defaultdict(list), defaultdict(list)
    for record in pairs:
        bypair[record["method"]].append(record)
    for record in paths:
        bypath[record["method"]].append(record)
    natural = [r for r in nodes if r["natural"]]
    labeled = [r for r in natural if r["label_available"]]
    metrics, counts = {}, {}
    for name in names:
        mm, cc, _ = legacy.pair_statistics(bypair[name], boot)
        if natural:
            correct = [r["predictions"][name] == int(r["label"]) for r in labeled]
            mm["natural_node_accuracy"] = boot.mean(correct, [r["source_id"] for r in labeled])
            cc.update(natural_node_count=len(labeled), natural_node_correct=sum(correct),
                      natural_node_total_count=len(natural),
                      natural_node_unlabeled_count=len(natural) - len(labeled))
        for interference, prefix in ((False, "natural_path"), (True, "interference_path")):
            rr = [r for r in bypath[name] if r["interference"] == interference]
            if rr:
                mm[prefix + "_accuracy"] = boot.mean([r["correct"] for r in rr],
                                                       [r["source_id"] for r in rr])
                cc[prefix + "_count"] = len(rr)
                cc[prefix + "_correct"] = sum(r["correct"] for r in rr)
                cc[prefix + "_unavailable"] = sum(r["error"] == "unavailable_score" for r in rr)
        metrics[name], counts[name] = mm, cc
    return metrics, counts


def _comparison_specs(names):
    names = set(names)
    result = []
    for calibration in ("shared", "independent"):
        for left, right in (("J", "S"), ("J", "R1"), ("J", "R2"), ("S", "R1")):
            lhs = left + "_" + calibration
            rhs = right if right == "R1" else right + "_" + calibration
            if lhs in names and rhs in names:
                primary = calibration == "independent" and left == "J" and right in ("S", "R1")
                result.append(dict(lhs=lhs, rhs=rhs, comparison=lhs + " minus " + rhs,
                                   calibration=calibration,
                                   role="primary_fixed_sequence" if primary else "exploratory"))
    # Also support a single explicitly named calibration in small API consumers.
    for lhs, rhs in (("J", "S"), ("J", "R1"), ("J", "R2"), ("S", "R1")):
        if lhs in names and rhs in names:
            result.append(dict(lhs=lhs, rhs=rhs, comparison=lhs + " minus " + rhs,
                               calibration="unspecified", role="exploratory"))
    for calibration in ("shared", "independent"):
        lhs="J2_"+calibration
        for base in ("J", "R1", "S"):
            rhs=base if base=="R1" else base+"_"+calibration
            if lhs in names and rhs in names:
                result.append(dict(lhs=lhs,rhs=rhs,comparison=lhs+" minus "+rhs,
                                   calibration=calibration,role="exploratory"))
    return result


def _transition_specs(names):
    names = list(names)
    pairs = [(r["lhs"], r["rhs"]) for r in _comparison_specs(names)]
    for name in names:
        for baseline in ("R0", "R1"):
            if baseline in names and name not in (baseline, "raw"):
                pairs.append((name, baseline))
    return list(dict.fromkeys(pairs))


def _transitions(prepared):
    names = list(prepared["scores"])
    datasets = defaultdict(lambda: defaultdict(dict))
    for row in prepared["nodes"]:
        if not row["label_available"]:
            continue
        levels = (["natural_node"] if row["natural"] else [])
        if row["in_evaluation_views"]:
            levels.append("all_evaluation_nodes")
        for level in levels:
            for name in names:
                datasets[level][name][row["view_id"]] = (row["source_id"],
                    row["predictions"][name] == int(row["label"]))
    for row in prepared["paths"]:
        level = "interference_path" if row["interference"] else "natural_path"
        key = row["case_id"] + "|" + row["variant"]
        datasets[level][row["method"]][key] = (row["source_id"], bool(row["correct"]))
    for row in prepared["pairs"]:
        if row["paired_correct"] is not None:
            level = "pair_" + str(row["group"]) + "_both_correct"
            datasets[level][row["method"]][row["pair_id"]] = (row["source_id"], bool(row["paired_correct"]))
    by_source = defaultdict(lambda: dict(wrong_to_right=0, right_to_wrong=0,
                                         kept_right=0, kept_wrong=0))
    examples = []
    for level, bymethod in datasets.items():
        for lhs, rhs in _transition_specs(names):
            old, new = bymethod[rhs], bymethod[lhs]
            if set(old) != set(new):
                raise ValueError(f"Unpaired transition inputs: {level}/{lhs}/{rhs}")
            for key in sorted(old):
                source, before = old[key]
                after_source, after = new[key]
                if source != after_source:
                    raise ValueError("Paired transition source mismatch")
                category = ("kept_right" if before and after else "right_to_wrong" if before
                            else "wrong_to_right" if after else "kept_wrong")
                by_source[(lhs, rhs, level, source)][category] += 1
                examples.append(dict(comparison=lhs + " minus " + rhs, lhs=lhs, rhs=rhs,
                                     level=level, example_id=key, source_id=source,
                                     before_correct=before, after_correct=after, transition=category))
    source_rows, total = [], defaultdict(lambda: dict(wrong_to_right=0, right_to_wrong=0,
                                                      kept_right=0, kept_wrong=0))
    for (lhs, rhs, level, source), counts in sorted(by_source.items()):
        row = dict(comparison=lhs + " minus " + rhs, lhs=lhs, rhs=rhs, level=level,
                   source_id=source, total=sum(counts.values()), **counts)
        source_rows.append(row)
        for key, value in counts.items():
            total[(lhs, rhs, level)][key] += value
    totals = [dict(comparison=lhs + " minus " + rhs, lhs=lhs, rhs=rhs, level=level,
                   total=sum(counts.values()), **counts)
              for (lhs, rhs, level), counts in sorted(total.items())]
    return examples, source_rows, totals


def _first_errors(prepared):
    cases = {c["case_id"]: c for c in prepared["cases"]}
    node_ids = {}
    for view in prepared["path_views"]:
        if view.get("variant", "original") in ("original", "full"):
            node_ids[(view["case_id"], view["gid"])] = legacy.uid(view)
    paths = {(r["method"], r["case_id"]): r for r in prepared["paths"] if not r["interference"]}
    records = []
    for lhs, rhs in _transition_specs(prepared["scores"]):
        for cid, case in cases.items():
            before, after = paths.get((rhs, cid)), paths.get((lhs, cid))
            if before is None or after is None:
                continue
            gid = before.get("first_error")
            gold = {r["gid"]: r["label"] for r in case["gold_path"]}
            key, label = node_ids.get((cid, gid)), gold.get(gid)
            row = dict(comparison=lhs + " minus " + rhs, lhs=lhs, rhs=rhs,
                       source_id=case["source_id"], case_id=cid,
                       before_path_correct=before["correct"], after_path_correct=after["correct"],
                       before_first_error=gid, after_first_error=after.get("first_error"),
                       before_path_error=before.get("error"), after_path_error=after.get("error"),
                       first_error_view_id=key, gold_label=label,
                       first_error_repaired=None, before_score=None, after_score=None,
                       before_abs_score=None, after_abs_score=None,
                       before_gold_margin=None, after_gold_margin=None,
                       score_shift=None,
                       first_error_reached_after=any(r["gid"] == gid for r in after["path"]) if gid else None)
            if key is not None and label is not None:
                old = float(prepared["scores"][rhs][key])
                new = float(prepared["scores"][lhs][key])
                direction = 2 * int(label) - 1
                row.update(first_error_repaired=int(new > 0) == int(label),
                           before_score=old, after_score=new,
                           before_abs_score=abs(old), after_abs_score=abs(new),
                           before_gold_margin=direction * old, after_gold_margin=direction * new,
                           score_shift=new - old)
            records.append(row)
    return records


def _first_error_source_rows(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["comparison"], record["source_id"])].append(record)
    result = []
    for (comparison, source), rows in sorted(groups.items()):
        available = [r for r in rows if r["first_error_repaired"] is not None]
        result.append(dict(comparison=comparison, source_id=source, natural_paths=len(rows),
            wrong_paths_before=sum(not r["before_path_correct"] for r in rows),
            first_error_nodes_available=len(available),
            first_error_nodes_repaired=sum(r["first_error_repaired"] for r in available),
            wrong_paths_repaired=sum(not r["before_path_correct"] and r["after_path_correct"] for r in rows),
            before_abs_score_mean=float(np.mean([r["before_abs_score"] for r in available])) if available else None,
            after_abs_score_mean=float(np.mean([r["after_abs_score"] for r in available])) if available else None,
            before_gold_margin_mean=float(np.mean([r["before_gold_margin"] for r in available])) if available else None,
            after_gold_margin_mean=float(np.mean([r["after_gold_margin"] for r in available])) if available else None))
    return result


def selection_metrics(views, pairs, cases, path_views, scores):
    """Point estimates for inner-source selection; no resampling or file writes."""
    prepared = _prepare(views, pairs, cases, path_views, scores)
    sources = {r["source_id"] for kind in ("nodes", "pairs", "paths") for r in prepared[kind]}
    if not sources:
        raise ValueError("Empty evaluation sources")
    metrics, counts = _arrays(prepared, legacy.SourceBootstrap(sources, draws=0))
    return _safe({name: dict(counts[name], **{key: float(values[0]) for key, values in mm.items()
                                             if key not in counts[name]})
                  for name, mm in metrics.items()})


def _success(comparisons):
    stages = [("J_independent minus S_independent", "AP_increment"),
              ("J_independent minus R1", "calibrated_method_gain")]
    gate_open, results = True, []
    for comparison, label in stages:
        row = comparisons.get(comparison, {})
        natural = row.get("metrics", {}).get("natural_path_accuracy", {})
        interference = row.get("metrics", {}).get("interference_path_accuracy", {})
        defined = natural.get("low") is not None and interference.get("low") is not None
        natural_ok = bool(defined and natural["low"] >= NONINFERIORITY)
        interference_ok = bool(defined and interference["low"] > 0)
        conditions = natural_ok and interference_ok
        results.append(dict(comparison=comparison, stage=label, intervals_available=defined,
                            natural_noninferiority=natural_ok, interference_superiority=interference_ok,
                            both_conditions=conditions, formal_gate_open=gate_open,
                            passes_predeclared_sequence=bool(gate_open and conditions)))
        gate_open = gate_open and conditions
    return dict(sequence=results,
                primary_calibration="independent",
                natural_noninferiority_threshold=NONINFERIORITY,
                multiple_comparison_policy=("Fixed sequence: independent J-S, then independent J-R1. "
                    "Each stage requires both interference superiority and natural noninferiority "
                    "using source-paired 95% percentile intervals. Shared calibration, J-R2, S-R1 "
                    "and candidate-level analyses are exploratory and make no additional formal "
                    "significance claims."),
                independent_confirmation=False,
                history="Historically used sources; these are prespecified comparisons within an exploratory reevaluation")


def evaluate(views, pairs, cases, path_views, scores, output_dir):
    """Save exact-decoder outputs and pooled source-paired comparison statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(views, pairs, cases, path_views, scores)
    sources = sorted({r["source_id"] for kind in ("nodes", "pairs", "paths") for r in prepared[kind]})
    if not sources:
        raise ValueError("Empty evaluation sources")
    boot = legacy.SourceBootstrap(sources, draws=DRAWS)
    metrics, counts = _arrays(prepared, boot)
    models = {name: dict(counts[name], **{key: float(value[0]) for key, value in mm.items()
                                         if key not in counts[name]}) for name, mm in metrics.items()}
    bootstrap_rows, comparisons = [], {}
    for name, mm in metrics.items():
        for metric, values in mm.items():
            bootstrap_rows.append(dict(comparison=name, metric=metric, role="method_summary",
                                       sources=len(sources), draws=DRAWS, **boot.summary(values)))
    for spec in _comparison_specs(metrics):
        lhs, rhs = spec["lhs"], spec["rhs"]
        summary = dict(spec, metrics={})
        for metric in sorted(metrics[lhs].keys() & metrics[rhs].keys()):
            interval = boot.summary(metrics[lhs][metric] - metrics[rhs][metric])
            summary["metrics"][metric] = interval
            bootstrap_rows.append(dict(spec, metric=metric, sources=len(sources), draws=DRAWS, **interval))
        comparisons[spec["comparison"]] = summary
    for name, row in models.items():
        for reference in ("R1", "S_" + name.rsplit("_", 1)[-1] if "_" in name else "S"):
            if reference not in metrics:
                continue
            for metric in ("natural_path_accuracy", "interference_path_accuracy"):
                if metric in row and metric in models[reference]:
                    row[metric + "_difference_from_" + reference] = row[metric] - models[reference][metric]
    source_metrics = []
    grouped = {kind: defaultdict(list) for kind in ("nodes", "pairs", "paths")}
    for kind in grouped:
        for record in prepared[kind]:
            grouped[kind][record["source_id"]].append(record)
    for source in sources:
        subset = dict(prepared, **{kind: grouped[kind][source] for kind in grouped})
        sm, sc = _arrays(subset, legacy.SourceBootstrap([source], draws=0))
        for name, mm in sm.items():
            source_metrics.append(dict(source_id=source, method=name, **sc[name],
                                       **{key: float(value[0]) for key, value in mm.items() if key not in sc[name]}))
    examples, source_transitions, transitions = _transitions(prepared)
    first_errors = _first_errors(prepared)
    first_source = _first_error_source_rows(first_errors)
    files = {}
    for name, records in (("predictions", prepared["nodes"]), ("pair_predictions", prepared["pairs"]),
                          ("path_predictions", prepared["paths"]), ("transitions_by_example", examples),
                          ("natural_first_error", first_errors)):
        path = output_dir / (name + ".jsonl")
        _jsonl(path, records)
        files[name] = path.name
    for name, records in (("metrics", [dict(method=name, **row) for name, row in models.items()]),
                          ("source_bootstrap", bootstrap_rows), ("source_metrics", source_metrics),
                          ("transitions", transitions), ("source_transitions", source_transitions),
                          ("natural_first_error_by_source", first_source)):
        path = output_dir / (name + ".csv")
        legacy.write_csv(path, _safe(records))
        files[name] = path.name
    result = _safe(dict(models=models, comparisons=comparisons, success=_success(comparisons),
        counts=dict(sources=len(sources), unique_inputs=len(prepared["nodes"]),
                    pair_prediction_rows=len(prepared["pairs"]), path_prediction_rows=len(prepared["paths"])),
        sources=sources, files=files, protocol=dict(draws=DRAWS, seed=20261111,
            path_scorer="experiments.aster_cr_retrain_paired_v2_20261111.evaluation.path_predictions",
            decision="score > 0", missing_path_score="legacy unavailable_score; counted incorrect",
            score_shift_from_raw="Includes fitted intercept; uncalibrated corrections are saved by training runner",
            pooled_estimator="Source bootstrap recomputes pooled ratios; fold ratios are never averaged",
            historical_sources=True, full_models_refit=False), complete=True))
    _json(output_dir / "metrics.json", result)
    return result
