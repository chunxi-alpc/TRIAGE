#!/usr/bin/env python3
"""Clustered, paired descriptive analysis for the CARE steering pilot.

No unobserved method is imputed. Missing predictions and metadata are inventoried.
Primary inference: paired family bootstrap and family sign permutation, C - B.
Item McNemar p values are secondary because items can share a rule family.
"""
import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def first(row, *names, default=None):
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def as_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes", "applicable", "T"):
        return True
    if value in (0, "0", "false", "False", "no", "inapplicable", "F"):
        return False
    return None


def canonical(row):
    item_id = str(first(row, "item_id", "id", "example_id", default=""))
    if not item_id:
        raise ValueError("A prediction row is missing item_id/id/example_id")
    family = str(first(row, "rule_family_id", "family_id", "family", "cluster_id", "pair_id", default=item_id))
    correct = as_bool(row.get("correct"))
    if correct is None:
        prediction = first(row, "parsed_answer", "prediction", "pred")
        gold = first(row, "gold_answer", "gold", "label", "answer", "expected")
        if gold is None:
            raise ValueError(f"{item_id}: missing both correct and a ground truth answer")
        correct = prediction is not None and prediction == gold
    applicable = as_bool(first(row, "applicable", "new_applicable", "new_rule_applicable", "rule_applicable"))
    conflict = as_bool(first(row, "conflict", "has_conflict", "is_conflict"))
    old_valid = as_bool(first(row, "old_valid", "old_rule_valid"))
    new_selected = as_bool(first(row, "new_action_selected", "selected_new", "used_new_rule"))
    old_selected = as_bool(first(row, "old_action_selected", "selected_old", "used_old_rule"))
    prediction = first(row, "parsed_answer", "prediction", "pred")
    if new_selected is None and "forbidden_new_answers" in row and applicable is False:
        new_selected = prediction in row["forbidden_new_answers"]
    if old_selected is None and "forbidden_old_answers" in row and old_valid is False:
        old_selected = prediction in row["forbidden_old_answers"]
    condition = str(first(row, "condition", "case_type", "scenario", "category", default="unknown"))
    nrules = first(row, "n_new_rules", "num_new_rules", "rule_count")
    multi = as_bool(first(row, "multi_rule", "is_multi_rule", "requires_joint_actions"))
    if multi is None:
        multi = (nrules is not None and int(nrules) > 1) or "multi" in condition.lower() or "joint" in condition.lower()
    gate = as_bool(first(row, "gate_parsed", "gate_prediction", "gate_applicable", "gate_applicability_state", "gate"))
    return dict(row, item_id=item_id, family_id=family, task=str(row.get("task", "source")),
                split=str(row.get("split", "test")), condition=condition, correct=bool(correct),
                applicable=applicable, conflict=conflict, old_valid=old_valid,
                new_action_selected=new_selected, old_action_selected=old_selected,
                multi_rule=bool(multi), gate_parsed=gate)


def metric_units(rows, pair_expected):
    """metric -> unit id -> (family, binary value); None metadata is omitted."""
    units = defaultdict(dict)
    pairs = defaultdict(list)
    for row in rows:
        iid, family = row["item_id"], row["family_id"]
        put = lambda name, value: units[name].__setitem__(iid, (family, float(value)))
        put("accuracy", row["correct"])
        if row["applicable"] is True:
            put("applicable_accuracy", row["correct"])
        if row["applicable"] is False:
            put("inapplicable_accuracy", row["correct"])
            put("new_rule_misuse_lower_bound", row["new_action_selected"] is True)
            put("new_rule_misuse_upper_bound", row["new_action_selected"] is not False)
            put("new_rule_misuse_judgement_coverage", row["new_action_selected"] is not None)
            if row["new_action_selected"] is not None:
                put("new_rule_misuse", row["new_action_selected"])
        if row.get("applicability_state") == "U":
            put("unknown_applicability_accuracy", row["correct"])
        if row["old_valid"] is False:
            put("old_invalid_accuracy", row["correct"])
        if row.get("gate_applicability_state") is not None and row.get("applicability_state") is not None:
            put("gate_tristate_accuracy", row["gate_applicability_state"] == row["applicability_state"])
        if row["multi_rule"]:
            put("multi_rule_exact_accuracy", row["correct"])
        if row["applicable"] is False and row["old_valid"] is True:
            put("valid_old_fallback_accuracy", row["correct"])
        if row["old_valid"] is False and row["old_action_selected"] is not None:
            put("invalid_old_misuse", row["old_action_selected"])
        if row["conflict"] is False:
            put("no_conflict_accuracy", row["correct"])
        if row["gate_parsed"] is not None and row["applicable"] is not None:
            put("gate_accuracy", row["gate_parsed"] == row["applicable"])
            if row["applicable"] is False:
                put("gate_false_apply", row["gate_parsed"])
            else:
                put("gate_false_reject", not row["gate_parsed"])
        valid = as_bool(first(row, "parse_valid", "valid_answer"))
        if valid is not None:
            put("invalid_answer_rate", not valid)
            put("answer_parse_coverage", valid)
        if row.get("pair_id") is not None:
            pairs[str(row["pair_id"])].append(row)
    for pair_id, group in pairs.items():
        # A complete flip pair must have at least two members, both applicability states,
        # and every expected member. Other 10-case family bundles are not flip pairs.
        observed = {row["item_id"] for row in group}
        expected = pair_expected.get(pair_id, observed)
        states = {row["applicable"] for row in group}
        if len(group) >= 2 and observed == expected and True in states and False in states:
            families = {row["family_id"] for row in group}
            if len(families) != 1:
                raise ValueError(f"Pair {pair_id} spans multiple family IDs")
            units["flip_pair_joint_accuracy"][pair_id] = (next(iter(families)), float(all(row["correct"] for row in group)))
    return dict(units)


def rng_for(seed, name):
    number = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, number]))


def aggregate_units(units, family_order=None):
    families = sorted({family for family, _ in units.values()}) if family_order is None else family_order
    locations = {family: i for i, family in enumerate(families)}
    numer = np.zeros(len(families), dtype=float)
    denom = np.zeros(len(families), dtype=float)
    for family, value in units.values():
        numer[locations[family]] += value
        denom[locations[family]] += 1
    return families, numer, denom


def bootstrap(numer, denom, rng, repeats):
    count = len(numer)
    if count == 0 or not denom.sum():
        return [None, None]
    samples = rng.integers(0, count, size=(repeats, count))
    draws = numer[samples].sum(axis=1) / denom[samples].sum(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def one_metric(units, seed, name, repeats):
    families, numer, denom = aggregate_units(units)
    if not len(families):
        return None
    return {"estimate": float(numer.sum() / denom.sum()), "numerator": int(numer.sum()),
            "denominator": int(denom.sum()), "n_clusters": len(families),
            "ci95": bootstrap(numer, denom, rng_for(seed, name), repeats)}


def sign_permutation(deltas, rng, repeats):
    nonzero = deltas[np.abs(deltas) > 1e-15]
    if not len(nonzero):
        return {"p": 1.0, "type": "exact_family_sign_permutation", "replicates": 1}
    threshold = abs(float(nonzero.sum())) - 1e-12
    n = len(nonzero)
    extreme = 0
    if n <= 20:
        total = 1 << n
        for begin in range(0, total, 8192):
            bits = ((np.arange(begin, min(total, begin + 8192), dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1)
            signs = 2 * bits.astype(float) - 1
            extreme += int(np.sum(np.abs(signs @ nonzero) >= threshold))
        return {"p": extreme / total, "type": "exact_family_sign_permutation", "replicates": total}
    for begin in range(0, repeats, 8192):
        size = min(8192, repeats - begin)
        signs = rng.integers(0, 2, size=(size, n)) * 2 - 1
        extreme += int(np.sum(np.abs(signs @ nonzero) >= threshold))
    return {"p": (extreme + 1) / (repeats + 1), "type": "monte_carlo_family_sign_permutation", "replicates": repeats}


def compare(left, right, seed, name, repeats, perm_repeats):
    common = sorted(set(left) & set(right))
    if not common:
        return None
    for uid in common:
        if left[uid][0] != right[uid][0]:
            raise ValueError(f"{name}: inconsistent family ID for {uid}")
    delta_units = {uid: (left[uid][0], left[uid][1] - right[uid][1]) for uid in common}
    families, numer, denom = aggregate_units(delta_units)
    improved = sum(left[uid][1] == 1 and right[uid][1] == 0 for uid in common)
    worsened = sum(left[uid][1] == 0 and right[uid][1] == 1 for uid in common)
    discordant = improved + worsened
    return {"difference": float(numer.sum() / denom.sum()),
            "ci95": bootstrap(numer, denom, rng_for(seed, name + ":ci"), repeats),
            "n_common_units": len(common), "n_clusters": len(families),
            "excluded_left_units": len(left) - len(common), "excluded_right_units": len(right) - len(common),
            "left_one_right_zero": int(improved), "left_zero_right_one": int(worsened),
            "cluster_test": sign_permutation(numer, rng_for(seed, name + ":perm"), perm_repeats),
            "secondary_mcnemar_exact_p": float(binomtest(improved, discordant, 0.5).pvalue) if discordant else 1.0}


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    previous = 0.0
    adjusted = {}
    for index, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def cost_summary(rows):
    fields = {"total_seconds": ("total_seconds", "elapsed_seconds", "latency_seconds", "seconds"),
              "generation_seconds": ("generation_seconds", "model_seconds", "inference_seconds"),
              "scoring_plus_gate_seconds": ("scoring_plus_gate_seconds",),
              "end_to_end_seconds": ("end_to_end_seconds",),
              "batch_allocated_tokenization_and_generation_seconds": ("end_to_end_allocated_seconds",),
              "gate_seconds": ("gate_seconds",), "input_tokens": ("input_tokens", "prompt_tokens"),
              "output_tokens": ("output_tokens", "generated_tokens"),
              "peak_allocated_bytes": ("peak_allocated_bytes", "peak_memory_bytes"),
              "peak_reserved_bytes": ("peak_reserved_bytes",)}
    result = {}
    for key, aliases in fields.items():
        values = [float(value) for row in rows if (value := first(row, *aliases)) is not None]
        if values:
            data = np.array(values)
            if key.startswith("peak_"):
                result[key] = {"n": len(values), "max": float(data.max()), "measurement": "process_high_water_mark"}
            else:
                result[key] = {"n": len(values), "sum": float(data.sum()), "mean": float(data.mean()),
                               "median": float(np.median(data)), "p95": float(np.quantile(data, 0.95)), "max": float(data.max())}
    return result


def markdown(result):
    lines = ["# CARE steering pilot: observed results", "", "Only completed prediction rows are analyzed. Missing methods are not assigned zero scores.", "",
             f"Bootstrap seed: {result['bootstrap_seed']}; replicates: {result['bootstrap_replicates']}. Confidence intervals resample whole rule families.", "",
             "| Scope | Method | Metric | Numerator / denominator | Estimate | 95% CI | Families |", "|---|---|---|---:|---:|---:|---:|"]
    for scope, methods in result["metrics"].items():
        for method, metrics in methods.items():
            for metric, data in metrics.items():
                lo, hi = data["ci95"]
                lines.append(f"| {scope} | {method} | {metric} | {data['numerator']} / {data['denominator']} | {data['estimate']:.3f} | [{lo:.3f}, {hi:.3f}] | {data['n_clusters']} |")
    lines += ["", "## Paired differences on the common cohort", "", "Differences are left method minus right method. A negative misuse difference is favorable. Item-level McNemar tests are secondary and do not account for within-family dependence.", "", "| Scope | Contrast | Metric | Difference | 95% CI | Family permutation p | Primary Holm p | Secondary McNemar p |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for scope, contrasts in result["comparisons"].items():
        for contrast, metrics in contrasts.items():
            for metric, data in metrics.items():
                lo, hi = data["ci95"]
                hp = data.get("primary_holm_p")
                hp_text = f"{hp:.4g}" if hp is not None else "exploratory"
                lines.append(f"| {scope} | {contrast} | {metric} | {data['difference']:+.3f} | [{lo:.3f}, {hi:.3f}] | {data['cluster_test']['p']:.4g} | {hp_text} | {data['secondary_mcnemar_exact_p']:.4g} |")
    lines += ["", "## Completeness and limitations", ""]
    for warning in result["warnings"]:
        lines.append(f"- {warning}")
    lines += ["", "Coverage and cost summaries are preserved in the accompanying JSON. Dataset design, method approximations, and source fidelity require a separate experimental report.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="results/predictions.jsonl")
    parser.add_argument("--data", default="data/cases.jsonl", help="Canonical dataset JSONL, to merge item metadata and verify coverage")
    parser.add_argument("--gates", default="results/gates.jsonl", help="Optional stored gate outputs, joined by item ID")
    parser.add_argument("--output", default="results/analysis.json")
    parser.add_argument("--markdown", default="results/analysis.md")
    parser.add_argument("--splits", default="test,transfer")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--exploratory", action="store_true", help="Mark post-primary follow-up analyses as exploratory")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("Resampling counts must be positive")
    raw = read_jsonl(args.predictions)
    if not raw:
        raise ValueError("No observed predictions to analyze")
    allowed_splits = set(args.splits.split(","))
    data_rows = read_jsonl(args.data) if args.data else []
    metadata = {}
    for row in data_rows:
        iid = str(first(row, "item_id", "id", "example_id", default=""))
        if not iid or iid in metadata:
            raise ValueError(f"Missing or duplicate canonical dataset ID: {iid}")
        metadata[iid] = row
    stored_gates = {}
    if args.gates and Path(args.gates).exists():
        stored_gates = {str(r["item_id"]): r for r in read_jsonl(args.gates)}
    rows, seen = [], set()
    for row in raw:
        iid = str(first(row, "item_id", "id", "example_id", default=""))
        for field in ("split", "task", "condition", "pair_id", "rule_family_id", "gold", "gold_answer", "applicable", "old_valid", "conflict"):
            if field in metadata.get(iid, {}) and field in row and metadata[iid][field] != row[field]:
                raise ValueError(f"{iid}: prediction metadata disagrees with canonical dataset field {field}")
        merged = {**metadata.get(iid, {}), **row}
        if str(merged.get("method", "")).startswith(("C", "D")) and iid in stored_gates:
            merged.setdefault("gate_applicability_state", stored_gates[iid].get("new_state"))
        if merged.get("total_seconds") is not None:
            merged.setdefault("scoring_plus_gate_seconds", merged["total_seconds"] if "inference_seconds" in merged else merged["total_seconds"] + float(merged.get("gate_seconds", 0)))
        item = canonical(merged)
        if item["split"] not in allowed_splits:
            continue
        if "method" not in item:
            raise ValueError(f"{iid}: missing method")
        key = (item["method"], item["item_id"])
        if key in seen:
            raise ValueError(f"Duplicate method/item row {key}; analyze independent seeds in separate invocations")
        seen.add(key)
        rows.append(item)
    if not rows:
        raise ValueError(f"No rows in requested splits {allowed_splits}")
    warnings = ["Nonparametric bootstrap intervals can collapse to [0,0] or [1,1] on uniform strata; these intervals do not establish zero population uncertainty or equivalence.", "scoring_plus_gate_seconds adds saved gate time to model scoring time; tokenization, some hook preparation, model loading, and direction extraction are excluded. Peak memory values are process high-water marks, not per-item resets."]
    if any(row.get("scoring", "").startswith("free_generation") for row in rows):
        warnings.append("Free-generation row timings allocate batch wall time across items; they are not isolated per-item latency. Reused completions are valid method outputs but must not be counted as independent model calls or summed as actual total compute. gate_seconds=0 in this follow-up denotes unmeasured per-item gate time; full run wall time includes gate execution.")
    seeds = {str(row.get("seed", "unknown")) for row in rows}
    if len(seeds) == 1:
        warnings.append("One evaluation seed is present; intervals measure variation across rule families, not model/direction training seeds.")
    elif len(seeds) > 1:
        warnings.append("Multiple seeds are present on distinct items. These are not replicated evaluations across all items; intervals are family based only.")
    if not data_rows:
        warnings.append("No canonical dataset supplied; an item missing from every method cannot be detected.")
    methods = sorted({row["method"] for row in rows})
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    eligible_data = [row for row in data_rows if str(row.get("split", "test")) in allowed_splits]
    expected_ids = {str(first(row, "item_id", "id", "example_id")) for row in eligible_data} if data_rows else {row["item_id"] for row in rows}
    pair_expected = defaultdict(set)
    for row in eligible_data or rows:
        if row.get("pair_id") is not None:
            pair_expected[str(row["pair_id"])].add(str(first(row, "item_id", "id", "example_id")))
    coverage = {}
    for method, subset in by_method.items():
        actual = {row["item_id"] for row in subset}
        coverage[method] = {"observed": len(actual), "expected": len(expected_ids), "missing_item_ids": sorted(expected_ids - actual),
                            "unexpected_item_ids": sorted(actual - expected_ids), "statuses": dict(Counter(row.get("status", "unspecified") for row in subset))}
        if expected_ids - actual:
            warnings.append(f"{method}: {len(expected_ids - actual)} expected items are missing. Reported comparisons use the observed common cohort.")
    for field in ["applicable", "old_valid", "conflict", "new_action_selected"]:
        missing = sum(row[field] is None for row in rows)
        if missing:
            warnings.append(f"{field}: absent/unknown in {missing}/{len(rows)} rows; dependent metrics omit unknown metadata.")
    family_count = len({row["family_id"] for row in rows})
    if family_count < 20:
        warnings.append(f"Only {family_count} independent families in this analysis; bootstrap coverage and significance are exploratory.")
    scope_rows = {"all": rows}
    for task in sorted({row["task"] for row in rows}):
        scope_rows[f"task:{task}"] = [row for row in rows if row["task"] == task]
    for condition in sorted({row["condition"] for row in rows}):
        scope_rows[f"condition:{condition}"] = [row for row in rows if row["condition"] == condition]
    result = {"predictions_path": str(Path(args.predictions).resolve()), "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
              "dataset_path": str(Path(args.data).resolve()) if args.data else None,
              "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest() if args.data else None,
              "bootstrap_seed": args.seed, "bootstrap_replicates": args.bootstrap, "permutation_replicates_max": args.permutations,
              "method_order": methods, "split_filter": sorted(allowed_splits), "metrics": {}, "comparisons": {}, "coverage": coverage,
              "costs": {method: cost_summary(subset) for method, subset in by_method.items()}, "warnings": warnings}
    all_units = {}
    for scope, subset in scope_rows.items():
        units = {method: metric_units([row for row in subset if row["method"] == method], pair_expected) for method in methods}
        all_units[scope] = units
        result["metrics"][scope] = {method: {metric: one_metric(values, args.seed, f"{scope}:{method}:{metric}", args.bootstrap) for metric, values in metrics.items() if values} for method, metrics in units.items()}
        result["comparisons"][scope] = {}
        # Full inference on global/task scopes; condition-level estimates retain descriptive CIs.
        if scope.startswith("condition:"):
            continue
        infer_methods = [method for method in methods if "fixed_minus" not in method and "sensitivity" not in method]
        for right, left in itertools.combinations(infer_methods, 2):
            contrast = f"{left}-{right}"
            entries = {}
            for metric in sorted(set(units[left]) & set(units[right])):
                value = compare(units[left][metric], units[right][metric], args.seed, f"{scope}:{contrast}:{metric}", args.bootstrap, args.permutations)
                if value:
                    entries[metric] = value
            result["comparisons"][scope][contrast] = entries
    # Primary two-test family: C versus B on main/source test task (exclude transfer).
    # Pooling target transfer into the primary hypothesis would change the estimand.
    source_rows = [row for row in rows if row["split"] == "test" and "transfer" not in row["task"].lower()]
    c_method = next((m for m in methods if m == "C" or m.startswith("C_")), None)
    b_method = next((m for m in methods if m == "B" or m.startswith("B_")), None)
    result["primary_scope"] = None if args.exploratory else "source_test"
    if args.exploratory:
        warnings.append("This is an exploratory follow-up analysis; primary_scope is null and any source-test Holm values are descriptive multiplicity adjustments, not a preregistered confirmatory test.")
    for scope, subset in scope_rows.items():
        if scope.startswith("task:"):
            count = len({row["family_id"] for row in subset})
            if count < 20:
                warnings.append(f"{scope}: only {count} families; intervals and significance are exploratory.")
    result["comparisons"]["source_test"] = {}
    if b_method and c_method and source_rows:
        c_units = metric_units([row for row in source_rows if row["method"] == c_method], pair_expected)
        b_units = metric_units([row for row in source_rows if row["method"] == b_method], pair_expected)
        primary = {}
        contrast = f"{c_method}-{b_method}"
        for metric in ("accuracy", "new_rule_misuse"):
            if metric in c_units and metric in b_units:
                value = compare(c_units[metric], b_units[metric], args.seed, f"primary:{metric}", args.bootstrap, args.permutations)
                if value:
                    primary[metric] = value
        adjusted = holm({metric: value["cluster_test"]["p"] for metric, value in primary.items()})
        secondary_adjusted = holm({metric: value["secondary_mcnemar_exact_p"] for metric, value in primary.items()})
        for metric, value in primary.items():
            value["primary_holm_p"] = adjusted[metric]
            value["secondary_mcnemar_holm_p"] = secondary_adjusted[metric]
        result["comparisons"]["source_test"][contrast] = primary
    else:
        warnings.append("Primary C-minus-B source-test comparison unavailable because a method or source-test data is missing.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"analyzed_rows": len(rows), "methods": methods, "families": family_count, "output": args.output, "warnings": len(warnings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
