#!/usr/bin/env python3
"""Independent NumPy replay of archived Qwen3-4B-Base signed experiments.

This program consumes saved journals only. It imports no inference framework,
loads no model, and performs zero new model forwards. The original journals use
gold applicability, simulated cases, and the same 12 clinical rule families.
Additional permutation and leave-one-family-out analyses are exploratory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import io
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np


STATUS = "completed_offline_reanalysis_of_archived_inference"
SOURCE_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"
METHODS = ("B4_positive", "B4_signed", "d_app_signed", "score_bias_signed")
TAUS = (0.0, 0.01, 0.05, 0.1)
METRICS = (
    "paired_accuracy", "accuracy", "app_accuracy", "nonapp_accuracy",
    "gold_logit_margin", "gold_probability_margin", "misuse",
    "no_conflict_accuracy", "no_conflict_retention", "eligible_accuracy",
    "eligible_pair_accuracy",
)
EXPLORATORY_METRICS = ("paired_accuracy", "accuracy", "gold_logit_margin")
BOOTSTRAP_SEED = 261019
BOOTSTRAP_REPLICATES = 10000
PINNED_BLOBS = {
    "inputs/prior_screen.csv": "bff6d2c1d77d9a0ebe840a58d1ff158d22e2a58b",
    "results/analysis/metrics_ci.csv": "837b66b7c9a0a7f993705c3419cb38eadd9f2012",
    "results/analysis/paired_comparisons.csv": "dbc1226402910eaabae086deed93543cfe07174c",
    "results/analysis/selected_predictions.jsonl.gz": "b550f55163a9097331948000dce6a95c52ec75e0",
    "results/analysis/sensitivity.csv": "18a3720a3165f0f64da58bdad2e44140f169dfbf",
    "results/analysis/summary.json": "c1231470e862daa9792d89ebdc7f0dd64c6a1d13",
    "results/fold0/predictions.jsonl.gz": "5e6a3399f719221742d1304e49dca044b264e4cd",
    "results/fold0/provenance.json": "ebeec6c2de3303b4bdd7ae50ed15ee2cbf538773",
    "results/fold0/selection.json": "66ecf4f931317c5bf75918a23ec2b0d7d1f11e95",
    "results/fold1/predictions.jsonl.gz": "4c4d82ea67ae65d9c30e27b4fd749830ea9597a8",
    "results/fold1/selection.json": "a7b612c19de8dd1250d9c52eb7df563d90ea4854",
    "results/fold2/predictions.jsonl.gz": "c6ef72b1b22ac2fa958e739a864ae9a7ba949fca",
    "results/fold2/selection.json": "2059fa89ce37c06ddbb657f196a1975b65c2cd41",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, data: dict[str, Any]) -> None:
    data = {"status": STATUS, "new_model_forwards": 0, **data}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    require(bool(records), f"no records to write: {path}")
    fields = ["status", "new_model_forwards", *records[0]]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({"status": STATUS, "new_model_forwards": 0, **r} for r in records)
    data = stream.getvalue().encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def verify_pinned_inputs(root: Path):
    report = {}
    for relative, expected_blob in PINNED_BLOBS.items():
        data = (root / relative).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(git_blob == expected_blob, f"input differs from pinned commit: {relative}")
        report[relative] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": git_blob, "expected_git_blob_sha1": expected_blob,
            "matches_pinned_commit": True,
        }
    return report


def unique_index(records, key_fields):
    result = {}
    for row in records:
        key = tuple(row[field] for field in key_fields)
        require(key not in result, f"duplicate key {key_fields}: {key}")
        result[key] = row
    return result


def numerator_denominator(records, metric):
    if metric in ("paired_accuracy", "eligible_pair_accuracy"):
        pairs = defaultdict(list)
        for row in records:
            pairs[row["pair_id"]].append(row)
        require(all(len(pair) == 2 for pair in pairs.values()), "incomplete or duplicated pair")
        usable = [pair for pair in pairs.values()
                  if metric == "paired_accuracy" or all(r["eligible"] for r in pair)]
        return sum(all(r["correct"] for r in pair) for pair in usable), len(usable)
    if metric == "app_accuracy":
        records = [r for r in records if r["applicable"]]
    elif metric in ("nonapp_accuracy", "misuse"):
        records = [r for r in records if not r["applicable"]]
    elif metric == "no_conflict_accuracy":
        records = [r for r in records if r["no_conflict"]]
    elif metric == "no_conflict_retention":
        records = [r for r in records if r["no_conflict"] and r["baseline_correct"]]
    elif metric == "eligible_accuracy":
        records = [r for r in records if r["eligible"]]
    field = metric if metric in ("gold_logit_margin", "gold_probability_margin", "misuse") else "correct"
    return sum(r[field] for r in records), len(records)


def metric_table(records):
    result = {}
    for metric in METRICS:
        numerator, denominator = numerator_denominator(records, metric)
        result[metric] = {"value": numerator / denominator if denominator else None, "n": denominator}
    return result


def grouped_vectors(records, metric, families):
    by_family = defaultdict(list)
    for row in records:
        by_family[row["group_id"]].append(row)
    return np.asarray([numerator_denominator(by_family[family], metric) for family in families], dtype=float)


def resample_distribution(vector, weights):
    totals = weights @ vector
    return np.divide(totals[:, 0], totals[:, 1], out=np.full(len(totals), np.nan),
                     where=totals[:, 1] > 0)


def percentile_interval(values):
    return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]


def bootstrap_statistics(collection, families, weights):
    summary = {method: metric_table(rows) for method, rows in collection.items()}
    distributions = {}
    for method, rows in collection.items():
        for metric in METRICS:
            distributions[method, metric] = resample_distribution(
                grouped_vectors(rows, metric, families), weights)
    intervals = []
    for method in collection:
        for metric in METRICS:
            distribution = distributions[method, metric]
            delta_distribution = distribution - distributions["baseline", metric]
            lo, hi = percentile_interval(distribution)
            delta_lo, delta_hi = percentile_interval(delta_distribution)
            intervals.append({
                "method": method, "metric": metric, **summary[method][metric],
                "ci_low": lo, "ci_high": hi,
                "delta_vs_baseline": summary[method][metric]["value"] - summary["baseline"][metric]["value"],
                "delta_ci_low": delta_lo, "delta_ci_high": delta_hi,
            })
    comparisons = []
    for method_a, method_b in itertools.combinations(collection, 2):
        for metric in METRICS:
            lo, hi = percentile_interval(distributions[method_b, metric] - distributions[method_a, metric])
            comparisons.append({
                "method_a": method_a, "method_b": method_b, "metric": metric,
                "delta_b_minus_a": summary[method_b][metric]["value"] - summary[method_a][metric]["value"],
                "ci_low": lo, "ci_high": hi,
            })
    return summary, intervals, comparisons


def validate_journals(all_rows, screen):
    require(len(all_rows) == 9792, "unexpected journal record count")
    index = unique_index(all_rows, ("fold", "role", "case_id", "method", "tau"))
    max_errors = {"probability_sum": 0.0, "softmax_probability": 0.0,
                  "gold_logit_margin": 0.0, "gold_probability_margin": 0.0}
    case_metadata = {}
    outcomes = ("candidate_logits", "candidate_probabilities", "prediction", "correct",
                "misuse", "gold_logit_margin", "gold_probability_margin")
    zero_reuse_count = 0
    first_baselines = {}
    for row in all_rows:
        require(row["method"] in ("baseline", *METHODS), "unknown method")
        require(row["tau"] in TAUS, "unknown strength")
        require(row["role"] in ("dev", "test"), "unexpected split role")
        require(row["fold"] in (0, 1, 2), "unexpected fold")
        require(type(row["applicable"]) is bool, "applicability is not boolean")
        require(row["gold_answer"] in "ABCD" and len(row["gold_answer"]) == 1, "invalid gold option")
        probabilities = np.asarray(row["candidate_probabilities"], dtype=float)
        logits = np.asarray(row["candidate_logits"], dtype=float)
        require(probabilities.shape == logits.shape == (4,), "expected four candidate scores")
        require(bool(np.isfinite(probabilities).all() and np.isfinite(logits).all()), "non-finite candidate scores")
        require(bool((probabilities >= 0).all() and (probabilities <= 1).all()), "probability outside [0,1]")
        predicted_probability = "ABCD"[int(np.argmax(probabilities))]
        predicted_logit = "ABCD"[int(np.argmax(logits))]
        require(row["prediction"] == predicted_probability == predicted_logit, "argmax/prediction mismatch")
        require(row["correct"] == int(row["prediction"] == row["gold_answer"]), "incorrect saved correctness")
        require(row["correct"] in (0, 1) and row["misuse"] in (0, 1), "non-binary outcome")
        gold_index = "ABCD".index(row["gold_answer"])
        logit_margin = row["candidate_logits"][gold_index] - max(
            score for i, score in enumerate(row["candidate_logits"]) if i != gold_index)
        probability_margin = row["candidate_probabilities"][gold_index] - max(
            score for i, score in enumerate(row["candidate_probabilities"]) if i != gold_index)
        softmax = np.exp(logits - logits.max())
        softmax /= softmax.sum()
        observed_errors = {
            "probability_sum": abs(float(probabilities.sum()) - 1),
            "softmax_probability": float(np.max(np.abs(softmax - probabilities))),
            "gold_logit_margin": abs(logit_margin - row["gold_logit_margin"]),
            "gold_probability_margin": abs(probability_margin - row["gold_probability_margin"]),
        }
        for key, error in observed_errors.items():
            max_errors[key] = max(max_errors[key], error)
        require(observed_errors["probability_sum"] < 1e-6, "candidate probabilities are not normalized")
        require(observed_errors["softmax_probability"] < 1e-6, "candidate probabilities disagree with logits")
        require(observed_errors["gold_logit_margin"] == 0.0, "saved logit margin is not exact")
        require(observed_errors["gold_probability_margin"] == 0.0, "saved probability margin is not exact")
        metadata = {field: row[field] for field in ("pair_id", "group_id", "rule_id", "applicable", "gold_answer")}
        require(row["case_id"] not in case_metadata or case_metadata[row["case_id"]] == metadata,
                "case identity or gold metadata changes across journal records")
        case_metadata[row["case_id"]] = metadata
        if row["method"] == "baseline":
            require(row["tau"] == 0, "baseline has nonzero strength")
            if row["case_id"] in first_baselines:
                require(all(row[k] == first_baselines[row["case_id"]][k] for k in outcomes),
                        "repeated contextual baselines disagree")
            first_baselines[row["case_id"]] = row
        if row["method"] != "baseline" and row["tau"] == 0:
            baseline = index[row["fold"], row["role"], row["case_id"], "baseline", 0.0]
            require(all(row[k] == baseline[k] for k in outcomes), "zero-strength output differs from baseline")
            zero_reuse_count += 1
    require(set(case_metadata) == set(screen), "screen and journals do not cover identical cases")
    require(len(case_metadata) == 384, "unique case count is not 384")
    pair_meta = defaultdict(list)
    for case_id, metadata in case_metadata.items():
        pair_meta[metadata["pair_id"]].append((case_id, metadata))
        require(all(screen[case_id][k] == metadata[k] for k in ("pair_id", "group_id", "rule_id", "gold_answer")),
                "screen metadata mismatch")
        require(screen[case_id]["stable"] in ("True", "False") and screen[case_id]["eligible"] in ("True", "False"),
                "screen boolean not explicitly recorded")
        require((screen[case_id]["stable"] == "True") ==
                (screen[case_id]["prior_prediction"] == screen[case_id]["paraphrase_prediction"]),
                "screen stability flag disagrees with prior/paraphrase outputs")
    highlighted_options = {}
    for pair_id, members in pair_meta.items():
        require(len(members) == 2 and {m[1]["applicable"] for m in members} == {False, True},
                "pair lacks exactly one applicable and one nonapplicable case")
        highlighted_options[pair_id] = next(m[1]["gold_answer"] for m in members if m[1]["applicable"])
    # The archived protocol fixes the highlighted option across pair members.
    for row in all_rows:
        expected_misuse = int(not row["applicable"] and row["prediction"] == highlighted_options[row["pair_id"]])
        require(row["misuse"] == expected_misuse, "misuse differs from paired highlighted option")
    fold_report = {}
    all_families = {m["group_id"] for m in case_metadata.values()}
    test_family_counts = Counter()
    for fold in range(3):
        fold_rows = [r for r in all_rows if r["fold"] == fold]
        require(len(fold_rows) == 3264, "fold record count differs from 3264")
        subsets = {}
        for role, n_cases, n_families in (("dev", 64, 2), ("test", 128, 4)):
            role_rows = [r for r in fold_rows if r["role"] == role]
            case_counts = Counter(r["case_id"] for r in role_rows)
            families = {r["group_id"] for r in role_rows}
            require(len(case_counts) == n_cases and set(case_counts.values()) == {17}, "missing method/strength grid")
            require(len(families) == n_families, "unexpected number of split families")
            subsets[role] = {"cases": n_cases, "families": sorted(families)}
            if role == "test":
                test_family_counts.update(families)
        require(not set(subsets["dev"]["families"]) & set(subsets["test"]["families"]), "dev/test family overlap")
        subsets["training_families_inferred_by_complement"] = sorted(
            all_families - set(subsets["dev"]["families"]) - set(subsets["test"]["families"]))
        fold_report[str(fold)] = subsets
    require(len(test_family_counts) == 12 and set(test_family_counts.values()) == {1}, "families not tested exactly once")
    return {
        "journal_records": len(all_rows), "unique_cases": len(case_metadata), "pairs": len(pair_meta),
        "families": len(all_families), "baseline_records": sum(r["method"] == "baseline" for r in all_rows),
        "zero_strength_reuse_records": zero_reuse_count,
        "nonzero_intervention_records": sum(r["method"] != "baseline" and r["tau"] > 0 for r in all_rows),
        "all_argmax_correctness_margins_verified": True,
        "misuse_crosschecked_against_paired_applicable_gold_option": True,
        "misuse_validation_assumption": "The archived protocol fixes the highlighted action/option across each pair.",
        "maximum_numeric_errors": max_errors, "zero_strength_reuse_exact": True,
        "folds": fold_report,
    }


def replay_selections(all_rows, saved_selection):
    reports = {}
    candidate_rows = []
    for fold in range(3):
        dev = [r for r in all_rows if r["fold"] == fold and r["role"] == "dev"]
        scores = {}
        for method in METHODS:
            for tau in TAUS:
                subset = [r for r in dev if r["method"] == method and r["tau"] == tau]
                pairs_num, pairs_n = numerator_denominator(subset, "paired_accuracy")
                cases_num, cases_n = numerator_denominator(subset, "accuracy")
                scores[method, tau] = (pairs_num / pairs_n, cases_num / cases_n)
                candidate_rows.append({"fold": fold, "method": method, "tau": tau,
                                       "dev_pair_accuracy": pairs_num / pairs_n,
                                       "dev_case_accuracy": cases_num / cases_n,
                                       "dev_pairs": pairs_n, "dev_cases": cases_n})
        candidates = []
        for tau in TAUS:
            candidates.append((float(np.mean([scores[method, tau][0] for method in METHODS])),
                               float(np.mean([scores[method, tau][1] for method in METHODS])), -tau, tau))
        best = max(candidates)
        own = {method: max((*scores[method, tau], -tau, tau) for tau in TAUS)[-1] for method in METHODS}
        saved = saved_selection[fold]
        require(saved["common_tau"] == best[-1] and saved["common_dev_pair_accuracy"] == best[0],
                "common strength fails development-only replay")
        require(saved["method_specific_tau"] == own, "method-specific strengths fail development-only replay")
        require(saved["candidates"] == [list(c) for c in candidates], "development candidate table differs")
        require(saved["test_labels_used"] is False, "archive does not declare development-only selection")
        reports[str(fold)] = {"common_tau": best[-1], "common_dev_pair_accuracy": best[0],
                             "method_specific_tau": own, "selection_recomputed_from_dev_only": True,
                             "saved_selection_matches_exactly": True}
    return reports, candidate_rows


def annotate_test(all_rows, screen):
    test = [dict(r) for r in all_rows if r["role"] == "test"]
    baseline = {r["case_id"]: r for r in test if r["method"] == "baseline"}
    require(len(baseline) == 384, "test baseline not unique per case")
    for row in test:
        screening = screen[row["case_id"]]
        row["eligible"] = screening["eligible"] == "True"
        row["no_conflict"] = screening["stable"] == "True" and screening["prior_prediction"] == screening["gold_answer"]
        row["baseline_correct"] = baseline[row["case_id"]]["correct"]
    return test, baseline


def select_collection(test, baseline, selection, policy):
    result = {"baseline": list(baseline.values())}
    for method in METHODS:
        def chosen(row):
            saved = selection[row["fold"]]
            return saved["common_tau"] if policy == "common_tau" else saved["method_specific_tau"][method]
        result[method] = [r for r in test if r["method"] == method and r["tau"] == chosen(r)]
    for method, rows in result.items():
        require(len(rows) == 384 and len({r["case_id"] for r in rows}) == 384, f"invalid selected set: {method}")
    return result


def verify_selected_journal(primary, archived_rows):
    derived = [row for rows in primary.values() for row in rows]
    left = unique_index(derived, ("method", "case_id"))
    right = unique_index(archived_rows, ("method", "case_id"))
    require(set(left) == set(right) and len(left) == 1920, "selected journal identity mismatch")
    canonical = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    for key in left:
        require(canonical(left[key]) == canonical(right[key]), f"selected journal content mismatch: {key}")
    content = "\n".join(canonical(left[key]) for key in sorted(left))
    return {"records_compared": len(left), "all_fields_match_exactly": True,
            "canonical_replayed_and_archived_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "selection_source": "Raw fold journals + recorded development-only common_tau + prior screen."}


def compare_csv(archive_rows, computed_rows, key_fields):
    def keyed(rows):
        result = {}
        for row in rows:
            key = tuple(str(row[k]) for k in key_fields)
            require(key not in result, f"duplicate comparison key: {key}")
            result[key] = row
        return result
    archived, computed = keyed(archive_rows), keyed(computed_rows)
    require(set(archived) == set(computed), "replayed CSV rows differ from archive")
    errors = []
    for key, row in archived.items():
        for field, value in row.items():
            if field in key_fields:
                continue
            replayed = computed[key][field]
            if value in ("", None):
                require(replayed is None, f"missingness differs: {key} {field}")
            else:
                errors.append(abs(float(value) - float(replayed)))
    max_error = max(errors, default=0.0)
    require(max_error <= 1e-12, f"archived CSV differs, max numeric error={max_error}")
    return {"rows_compared": len(archived), "numeric_cells_compared": len(errors),
            "maximum_absolute_error": max_error, "all_numeric_cells_exact": all(e == 0 for e in errors),
            "within_absolute_tolerance_1e_12": True}


def compare_metric_summary(archive, replay):
    require(set(archive) == set(replay), "summary method keys differ")
    errors = []
    for method in archive:
        require(set(archive[method]) == set(replay[method]), "summary metric keys differ")
        for metric, expected in archive[method].items():
            require(expected["n"] == replay[method][metric]["n"], "summary denominator differs")
            actual = replay[method][metric]["value"]
            if expected["value"] is None:
                require(actual is None, "summary missingness differs")
            else:
                errors.append(abs(expected["value"] - actual))
    require(max(errors, default=0) <= 1e-12, "archived summary differs")
    return {"metric_values_compared": len(errors), "maximum_absolute_error": max(errors, default=0),
            "all_metric_values_exact": all(e == 0 for e in errors), "all_denominators_exact": True}


def transitions(collection, policy):
    baseline = {r["case_id"]: r for r in collection["baseline"]}
    cases = []
    pairs = []
    family_summaries = []
    for method in METHODS:
        by_pair = defaultdict(list)
        for row in collection[method]:
            before = baseline[row["case_id"]]
            delta = row["correct"] - before["correct"]
            transition = "corrected" if delta > 0 else "degraded" if delta < 0 else (
                "stayed_correct" if row["correct"] else "changed_wrong_answer" if row["prediction"] != before["prediction"] else "stayed_wrong")
            record = {
                "selection_policy": policy, "method": method, "case_id": row["case_id"],
                "pair_id": row["pair_id"], "family": row["group_id"], "rule_id": row["rule_id"],
                "fold": row["fold"], "tau": row["tau"], "applicable": row["applicable"],
                "gold_answer": row["gold_answer"], "baseline_prediction": before["prediction"],
                "intervention_prediction": row["prediction"], "baseline_correct": before["correct"],
                "intervention_correct": row["correct"], "correctness_delta": delta,
                "transition": transition, "answer_changed": row["prediction"] != before["prediction"],
                "baseline_misuse": before["misuse"], "intervention_misuse": row["misuse"],
                "gold_logit_margin_delta": row["gold_logit_margin"] - before["gold_logit_margin"],
                "gold_probability_margin_delta": row["gold_probability_margin"] - before["gold_probability_margin"],
                "eligible": row["eligible"], "no_conflict": row["no_conflict"],
            }
            cases.append(record)
            by_pair[row["pair_id"]].append(record)
        for pair_id, members in by_pair.items():
            require(len(members) == 2, "transition pair incomplete")
            before = int(all(r["baseline_correct"] for r in members))
            after = int(all(r["intervention_correct"] for r in members))
            transition = "corrected" if after > before else "degraded" if after < before else (
                "stayed_correct" if after else "stayed_incorrect")
            pairs.append({
                "selection_policy": policy, "method": method, "pair_id": pair_id,
                "family": members[0]["family"], "fold": members[0]["fold"], "tau": members[0]["tau"],
                "case_ids": json.dumps(sorted(r["case_id"] for r in members)),
                "baseline_pair_correct": before, "intervention_pair_correct": after,
                "pair_correctness_delta": after - before, "transition": transition,
                "member_cases_corrected": sum(r["correctness_delta"] > 0 for r in members),
                "member_cases_degraded": sum(r["correctness_delta"] < 0 for r in members),
                "member_answers_changed": sum(r["answer_changed"] for r in members),
            })
        for family in sorted({r["group_id"] for r in collection[method]}):
            c = [r for r in cases if r["method"] == method and r["family"] == family]
            p = [r for r in pairs if r["method"] == method and r["family"] == family]
            family_summaries.append(transition_summary(c, p, policy, method, family))
    totals = [transition_summary([r for r in cases if r["method"] == method],
                                 [r for r in pairs if r["method"] == method], policy, method, "ALL")
              for method in METHODS]
    return cases, pairs, family_summaries, totals


def transition_summary(cases, pairs, policy, method, family):
    return {
        "selection_policy": policy, "method": method, "family": family,
        "cases": len(cases), "pairs": len(pairs),
        "cases_corrected": sum(r["correctness_delta"] > 0 for r in cases),
        "cases_degraded": sum(r["correctness_delta"] < 0 for r in cases),
        "net_cases_correct": sum(r["correctness_delta"] for r in cases),
        "wrong_answer_changes": sum(r["transition"] == "changed_wrong_answer" for r in cases),
        "all_answer_changes": sum(r["answer_changed"] for r in cases),
        "pairs_corrected": sum(r["pair_correctness_delta"] > 0 for r in pairs),
        "pairs_degraded": sum(r["pair_correctness_delta"] < 0 for r in pairs),
        "net_pairs_correct": sum(r["pair_correctness_delta"] for r in pairs),
    }


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = min(1.0, max(previous, (len(order) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def exploratory_family_tests(primary, families):
    require(len(families) == 12, "this exact analysis expects twelve families")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(families))))
    require(signs.shape == (4096, 12), "sign enumeration is incomplete")
    contrasts = [("baseline", method, "four_methods_vs_baseline") for method in METHODS]
    contrasts += [(method, "score_bias_signed", "two_bias_vs_signed") for method in ("B4_signed", "d_app_signed")]
    tests, effects, histogram, leave_out, leave_out_summary = [], [], [], [], []
    for method_a, method_b, correction_group in contrasts:
        for metric in EXPLORATORY_METRICS:
            a = grouped_vectors(primary[method_a], metric, families)
            b = grouped_vectors(primary[method_b], metric, families)
            require(bool(np.array_equal(a[:, 1], b[:, 1])), "paired comparison denominators differ")
            difference = b[:, 0] - a[:, 0]
            total_n = float(a[:, 1].sum())
            observed_sum = float(difference.sum())
            null_sums = signs @ difference
            tolerance = 1e-12 * max(1.0, abs(observed_sum))
            extreme = int(np.count_nonzero(np.abs(null_sums) >= abs(observed_sum) - tolerance))
            row = {
                "method_a": method_a, "method_b": method_b, "metric": metric,
                "delta_b_minus_a": observed_sum / total_n,
                "families": len(families), "nonzero_family_effects": int(np.count_nonzero(difference)),
                "enumerated_sign_patterns": len(signs), "two_sided_extreme_patterns": extreme,
                "p_two_sided_exact_enumeration": extreme / len(signs),
                "holm_family": correction_group + "_within_" + metric,
                "p_holm_within_declared_family": None, "exploratory": True,
            }
            tests.append(row)
            for i, family in enumerate(families):
                effects.append({"method_a": method_a, "method_b": method_b, "metric": metric,
                                "family": family, "denominator": int(a[i, 1]),
                                "numerator_difference_b_minus_a": float(difference[i]),
                                "family_delta_b_minus_a": float(difference[i] / a[i, 1])})
            values, counts = np.unique(null_sums, return_counts=True)
            histogram.extend({"method_a": method_a, "method_b": method_b, "metric": metric,
                              "permuted_delta_b_minus_a": float(value / total_n),
                              "sign_pattern_count": int(count)} for value, count in zip(values, counts, strict=True))
            remaining_effects = []
            for i, family in enumerate(families):
                effect = float((observed_sum - difference[i]) / (total_n - a[i, 1]))
                remaining_effects.append(effect)
                leave_out.append({"method_a": method_a, "method_b": method_b, "metric": metric,
                                  "omitted_family": family, "remaining_families": 11,
                                  "remaining_denominator": int(total_n - a[i, 1]),
                                  "delta_b_minus_a": effect, "refitting_performed": False})
            leave_out_summary.append({
                "method_a": method_a, "method_b": method_b, "metric": metric,
                "full_data_delta": observed_sum / total_n,
                "leave_one_family_out_min": min(remaining_effects),
                "leave_one_family_out_max": max(remaining_effects),
                "omission_yielding_min": families[int(np.argmin(remaining_effects))],
                "omission_yielding_max": families[int(np.argmax(remaining_effects))],
                "all_omissions_strictly_positive": all(effect > 0 for effect in remaining_effects),
                "refitting_performed": False,
            })
    for group in sorted({r["holm_family"] for r in tests}):
        members = [r for r in tests if r["holm_family"] == group]
        adjusted = holm_adjust([r["p_two_sided_exact_enumeration"] for r in members])
        for row, p in zip(members, adjusted, strict=True):
            row["p_holm_within_declared_family"] = p
    return tests, effects, histogram, leave_out, leave_out_summary


def disturbance_tables(all_rows, collections):
    calibration = []
    for method in METHODS:
        for tau in TAUS[1:]:
            subset = [r for r in all_rows if r["method"] == method and r["tau"] == tau]
            rms = np.asarray([r["disturbance"]["actual_rms"] for r in subset])
            fractions = np.asarray([r["disturbance"]["delta_score_negative_fraction"] for r in subset])
            require(bool(np.isfinite(rms).all() and np.isfinite(fractions).all()), "invalid disturbance values")
            require(bool((fractions >= 0).all() and (fractions <= 1).all()), "invalid negative-score fraction")
            error = abs(rms / tau - 1)
            calibration.append({"method": method, "tau": tau, "n": len(subset),
                                "actual_rms_mean": float(rms.mean()), "max_relative_error": float(error.max()),
                                "p95_relative_error": float(np.quantile(error, 0.95)),
                                "mean_negative_score_fraction": float(fractions.mean())})
    signs = []
    for policy, collection in collections.items():
        for method in METHODS:
            for applicable in (True, False):
                rows = [r for r in collection[method] if r["applicable"] == applicable]
                nonzero = [r for r in rows if r["tau"] > 0]
                fractions = np.asarray([r["disturbance"]["delta_score_negative_fraction"] for r in nonzero])
                mean_deltas = np.asarray([r["disturbance"]["delta_score_mean"] for r in nonzero])
                signs.append({
                    "selection_policy": policy, "method": method, "applicable": applicable,
                    "requested_intervention_sign": 1 if method == "B4_positive" or applicable else -1,
                    "selected_records": len(rows), "nonzero_strength_records": len(nonzero),
                    "zero_strength_records_excluded_from_sign_summary": len(rows) - len(nonzero),
                    "mean_negative_score_fraction": float(fractions.mean()) if len(fractions) else None,
                    "min_negative_score_fraction": float(fractions.min()) if len(fractions) else None,
                    "max_negative_score_fraction": float(fractions.max()) if len(fractions) else None,
                    "inputs_with_any_negative_score_delta": int(np.count_nonzero(fractions > 0)),
                    "inputs_with_some_nonnegative_score_delta": int(np.count_nonzero(fractions < 1)),
                    "mean_per_input_score_delta": float(mean_deltas.mean()) if len(mean_deltas) else None,
                })
    return calibration, signs


def run(root: Path, output: Path):
    fingerprints = verify_pinned_inputs(root)
    all_rows = []
    selection = {}
    for fold in range(3):
        rows = read_jsonl(root / f"results/fold{fold}/predictions.jsonl.gz")
        require(all(r["fold"] == fold for r in rows), "file and record fold disagree")
        all_rows.extend(rows)
        selection[fold] = read_json(root / f"results/fold{fold}/selection.json")
    screening_rows = read_csv(root / "inputs/prior_screen.csv")
    screen_index = unique_index(screening_rows, ("case_id",))
    screen = {key[0]: row for key, row in screen_index.items()}
    journal_checks = validate_journals(all_rows, screen)
    selection_checks, dev_candidates = replay_selections(all_rows, selection)
    test, baseline = annotate_test(all_rows, screen)
    collections = {policy: select_collection(test, baseline, selection, policy)
                   for policy in ("common_tau", "method_specific_tau")}
    primary = collections["common_tau"]
    selected_check = verify_selected_journal(primary, read_jsonl(root / "results/analysis/selected_predictions.jsonl.gz"))
    families = sorted({r["group_id"] for r in test})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(families), np.full(len(families), 1 / len(families)),
                             size=BOOTSTRAP_REPLICATES)
    primary_summary, primary_ci, comparisons = bootstrap_statistics(primary, families, weights)
    secondary_summary, secondary_ci, secondary_comparisons = bootstrap_statistics(
        collections["method_specific_tau"], families, weights)
    sensitivity = []
    for tau in TAUS:
        for method in METHODS:
            table = metric_table([r for r in test if r["method"] == method and r["tau"] == tau])
            sensitivity.append({"tau": tau, "method": method, **{m: v["value"] for m, v in table.items()}})
    archived_summary = read_json(root / "results/analysis/summary.json")
    archived_comparisons = {
        "primary_metrics_ci": compare_csv(read_csv(root / "results/analysis/metrics_ci.csv"),
                                           primary_ci, ("method", "metric")),
        "all_ten_pairwise_comparisons": compare_csv(read_csv(root / "results/analysis/paired_comparisons.csv"),
                                                    comparisons, ("method_a", "method_b", "metric")),
        "all_fixed_strength_results": compare_csv(read_csv(root / "results/analysis/sensitivity.csv"),
                                                  sensitivity, ("tau", "method")),
        "primary_summary": compare_metric_summary(archived_summary["primary"], primary_summary),
        "method_specific_secondary_summary": compare_metric_summary(
            archived_summary["method_specific_secondary"], {m: secondary_summary[m] for m in METHODS}),
    }
    family_rows, all_cases, all_pairs, family_transitions, transition_totals = [], [], [], [], []
    for policy, collection in collections.items():
        for family in families:
            for method, records in collection.items():
                metrics = metric_table([r for r in records if r["group_id"] == family])
                family_rows.append({"selection_policy": policy, "family": family, "method": method,
                                    **{metric: value["value"] for metric, value in metrics.items()},
                                    **{metric + "_n": value["n"] for metric, value in metrics.items()}})
        cases, pairs, by_family, totals = transitions(collection, policy)
        all_cases.extend(cases); all_pairs.extend(pairs)
        family_transitions.extend(by_family); transition_totals.extend(totals)
    permutation, effects, histogram, leave_out, leave_out_summary = exploratory_family_tests(primary, families)
    calibration, score_signs = disturbance_tables(all_rows, collections)
    require(verify_pinned_inputs(root) == fingerprints, "an archived input changed during replay")
    model_provenance = read_json(root / "results/fold0/provenance.json")
    require(model_provenance["gold_applicability_oracle"] is True, "unexpected intervention signal source")
    verification = {
        "source_commit": SOURCE_COMMIT, "input_fingerprints": fingerprints,
        "journal_checks": journal_checks, "development_selection_replay": selection_checks,
        "selected_record_replay": selected_check, "archived_statistic_comparisons": archived_comparisons,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "families_in_sampling_order": families,
            "weight_matrix_int64_sha256": hashlib.sha256(weights.astype("<i8").tobytes()).hexdigest(),
            "procedure": "Shared multinomial family counts; percentile intervals of pooled numerators/denominators.",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "inference_framework_imported": False},
        "reanalysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    summary = {
        "source_repository": "chunxi-alpc/TRIAGE", "source_commit": SOURCE_COMMIT,
        "historical_model": "Qwen/Qwen3-4B-Base", "historical_model_revision": model_provenance["model_revision"],
        "historical_model_dtype": "BF16", "gold_applicability_oracle": True,
        "cases": 384, "pairs": 192, "families": 12, "archived_journal_records": len(all_rows),
        "patient_data": "Simulated cases; repeated analysis of the same previously used rule families.",
        "selected": selection_checks, "primary": primary_summary,
        "method_specific_secondary": {m: secondary_summary[m] for m in METHODS},
        "no_conflict_cases": sum(r["no_conflict"] for r in primary["baseline"]),
        "no_conflict_retention_denominator": primary_summary["baseline"]["no_conflict_retention"]["n"],
        "eligible_cases": sum(r["eligible"] for r in primary["baseline"]),
        "eligible_complete_pairs": primary_summary["baseline"]["eligible_pair_accuracy"]["n"],
        "primary_transitions": [r for r in transition_totals if r["selection_policy"] == "common_tau"],
        "exploratory_pair_accuracy_tests": [r for r in permutation if r["metric"] == "paired_accuracy"],
        "leave_one_family_out_pair_accuracy": [r for r in leave_out_summary if r["metric"] == "paired_accuracy"],
        "score_signs_common_tau": [r for r in score_signs if r["selection_policy"] == "common_tau"],
        "archive_verification": archived_comparisons,
        "bootstrap_interpretation": "10,000 family percentile draws conditional on saved fitted folds; twelve clusters only, no model refits.",
        "permutation_interpretation": (
            "All 4096 within-family method-label swaps, two-sided pooled mean differences, including observed patterns. "
            "A randomization-style exploratory diagnostic assumes family-level exchangeability/symmetry; this was not a randomized treatment experiment. "
            "Holm correction covers four methods versus baseline separately within each metric, and separately the two bias-versus-signed comparisons. "
            "There is no joint adjustment across the three reported metrics or selection analyses."
        ),
        "score_sign_interpretation": (
            "Fractions are equal-weight means over per-input causal attention-score supports, not a pooled fraction over all entries. "
            "A positive key coefficient can yield negative QK-score changes. Negative fractions alone do not separate zero from positive entries. "
            "Zero-strength records are counted separately and excluded from sign summaries."
        ),
        "limitations": [
            "No new inference, no new independent dataset, and no evidence of current frozen-M1 end-to-end performance.",
            "Gold applicability is an experimental oracle; M1 error is absent from these interventions.",
            "Only twelve families and the same three fitted folds are available; bootstrap intervals omit fitting variability.",
            "Sign-flip diagnostics depend on family-level exchangeability; shared fitted folds can induce dependence.",
            "The follow-up was designed after earlier results on these cases were known; all additional tests remain exploratory.",
            "Multiple choice scores and simulated cases do not establish clinical efficacy or full source/decidability control.",
            "Leave-one-family-out results remove held-out outcomes only; they do not refit directions or reselect strengths.",
            "A sparse empirical bootstrap interval excluding zero is not a confirmation of a population effect.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics_ci.csv": primary_ci, "paired_comparisons.csv": comparisons,
        "method_specific_metrics_ci.csv": secondary_ci,
        "method_specific_paired_comparisons.csv": secondary_comparisons,
        "sensitivity.csv": sensitivity, "development_selection_candidates.csv": dev_candidates,
        "per_family.csv": family_rows, "case_transitions.csv.gz": all_cases, "pair_transitions.csv.gz": all_pairs,
        "corrected_or_degraded_cases.csv": [r for r in all_cases if r["correctness_delta"] != 0],
        "corrected_or_degraded_pairs.csv": [r for r in all_pairs if r["pair_correctness_delta"] != 0],
        "transition_summary.csv": transition_totals, "family_transitions.csv": family_transitions,
        "permutation_tests.csv": permutation, "permutation_family_effects.csv": effects,
        "permutation_null_histogram.csv": histogram,
        "leave_one_family_out.csv": leave_out, "leave_one_family_out_summary.csv": leave_out_summary,
        "calibration_replay.csv": calibration, "score_sign_by_applicability.csv": score_signs,
    }
    for filename, records in tables.items():
        write_csv(output / filename, records)
    summary["output_tables"] = {filename: len(records) for filename, records in tables.items()}
    write_json(output / "verification.json", verification)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path(__file__).resolve().parents[1] / "care_module2_signed_20261019")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/replay")
    args = parser.parse_args()
    result = run(args.archive.resolve(), args.output.resolve())
    print(json.dumps({"status": STATUS, "new_model_forwards": 0,
                      "primary_paired_accuracy": {m: v["paired_accuracy"] for m, v in result["primary"].items()},
                      "exploratory_pair_accuracy_tests": result["exploratory_pair_accuracy_tests"],
                      "archive_verification": result["archive_verification"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
