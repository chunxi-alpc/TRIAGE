"""Replay the four fixed cells with the joint study's decoder and shared source draws."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.aster_m3_joint_study import data, evaluation as joint
from experiments.aster_m3_joint_study.run import keys, prefix_scope
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as legacy

CELLS = ("J", "Direction", "Asymmetric", "J2")
CALIBRATIONS = ("independent", "shared")
METHODS = tuple(f"{m}_{c}" for c in CALIBRATIONS for m in CELLS)
COMPARISONS = (("Direction", "J"), ("Asymmetric", "J"),
               ("J2", "Direction"), ("J2", "Asymmetric"))
LEVELS = {"natural_node": ("natural_node_correct", "natural_node_count"),
          "natural_path": ("natural_path_correct", "natural_path_count"),
          "interference_path": ("interference_path_correct", "interference_path_count"),
          "pair_AP_both_correct": ("ap_both_correct", "ap_paired_labeled_count"),
          "pair_CF_both_correct": ("cf_both_correct", "cf_paired_labeled_count")}
FLOWS = ("wrong_to_wrong", "wrong_to_right", "right_to_wrong", "right_to_right")
ARCHIVE = Path("/home/gcx/aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115")
OUTPUT = Path("reports/aster_m3_missing_controls_20261123")


def stream(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def prepare(archive, output):
    config = data.read(archive / "configuration.json")
    study = data.StudyData(Path(config["historical_root"]), Path(config["cases_file"]),
                           Path(config["diagnostics_root"]), config["path_maps"])
    pooled = {level: [] for level in ("cases", "views", "path_views", "pairs")}
    expected = set()
    for split in study.folds["folds"]:
        fold = f"fold{split['fold']}"
        scope = study.scope(split["validation_sources"])
        expected.update(fold + "::" + uid for uid in keys(scope))
        scoped, _ = prefix_scope(fold, scope, {})
        for name, rows in scoped.items():
            pooled[name].extend(rows)
    scores = {m: {} for m in ("raw", *METHODS)}
    for row in stream(output / "b_scores.jsonl.gz"):
        key = row["view_id"]
        if key in scores["raw"]:
            raise ValueError(f"Duplicate held-out score {key}")
        scores["raw"][key] = row["scores"]["R0"]
        for m in METHODS:
            scores[m][key] = row["scores"][m]
    if expected != scores["raw"].keys():
        missing = sorted(expected - scores["raw"].keys())
        extra = sorted(scores["raw"].keys() - expected)
        data.write(output / "b_coverage_errors.json",
                   dict(expected=len(expected), present=len(expected) - len(missing), missing=missing, extra=extra))
        raise ValueError("Held-out score coverage differs; see b_coverage_errors.json")
    prepared = joint._prepare(**pooled, scores=scores)
    if any(r["error"] == "unavailable_score" for r in prepared["paths"]):
        raise ValueError("Dynamic path requests an unavailable input")
    return prepared, sorted(study.folds["all_sources"])


def metrics_table(prepared, boot):
    arrays, counts = joint._arrays(prepared, boot)
    both = Counter((r["method"], r["group"]) for r in prepared["pairs"] if r["paired_correct"])
    table = {}
    for m in METHODS:
        row = dict(counts[m])
        row.update((k, float(values[0])) for k, values in arrays[m].items() if k not in row)
        for g in ("AP", "CF"):
            row[g.lower() + "_both_correct"] = both[m, g]
        table[m] = joint._safe(row)
    return arrays, table


def intervals(arrays, boot):
    rows = []
    for cal in CALIBRATIONS:
        columns = {cell: arrays[f"{cell}_{cal}"] for cell in CELLS}
        for metric in columns["J"]:
            values = {cell: cols[metric] for cell, cols in columns.items()}
            contrasts = {a + "-" + b: values[a] - values[b] for a, b in COMPARISONS}
            contrasts["interaction"] = ((values["J2"] - values["Direction"])
                                        - (values["Asymmetric"] - values["J"]))
            for comparison, samples in contrasts.items():
                rows.append(dict(calibration=cal, comparison=comparison, metric=metric,
                                 sources=len(boot.sources), draws=boot.draws, **boot.summary(samples)))
    return rows


def transitions(prepared, table):
    outcomes = defaultdict(lambda: defaultdict(dict))

    def add(level, method, key, source, correct):
        if method in METHODS:
            if key in outcomes[level][method]:
                raise ValueError(f"Duplicate {level} observation {key}")
            outcomes[level][method][key] = source, bool(correct)

    for r in prepared["nodes"]:
        if r["natural"] and r["label_available"]:
            for m in METHODS:
                add("natural_node", m, r["view_id"], r["source_id"], r["predictions"][m] == r["label"])
    for r in prepared["paths"]:
        level = "interference_path" if r["interference"] else "natural_path"
        add(level, r["method"], (r["case_id"], r["variant"]), r["source_id"], r["correct"])
    for r in prepared["pairs"]:
        if r["paired_correct"] is not None:
            add("pair_" + r["group"] + "_both_correct", r["method"], r["pair_id"],
                r["source_id"], r["paired_correct"])

    totals, source_rows = [], []
    for level, (correct_key, count_key) in LEVELS.items():
        for cal in CALIBRATIONS:
            for a, b in COMPARISONS:
                new_name, old_name = a + "_" + cal, b + "_" + cal
                new, old = outcomes[level][new_name], outcomes[level][old_name]
                if old.keys() != new.keys():
                    raise ValueError(f"Unpaired transitions at {level}")
                by_source = defaultdict(lambda: [0, 0, 0, 0])
                for key, (source, before) in old.items():
                    after_source, after = new[key]
                    if source != after_source:
                        raise ValueError(f"Source mismatch for {key}")
                    by_source[source][2 * int(before) + int(after)] += 1
                overall = [sum(group[i] for group in by_source.values()) for i in range(4)]
                ww, wr, rw, rr = overall
                if (sum(overall) != table[new_name][count_key]
                        or wr + rr != table[new_name][correct_key]
                        or rw + rr != table[old_name][correct_key]):
                    raise ValueError(f"Four-way transition counts inconsistent for {a}-{b}, {level}")
                meta = dict(calibration=cal, comparison=a + "-" + b, level=level)
                totals.append(dict(**meta, **dict(zip(FLOWS, overall)), total=sum(overall), net=wr - rw))
                for source, counts in sorted(by_source.items()):
                    source_rows.append(dict(**meta, source_id=source, **dict(zip(FLOWS, counts)),
                                            total=sum(counts), net=counts[1] - counts[2]))
    return totals, source_rows


def verify_archive(archive, prepared, table):
    verified = {}
    for cell, parent in (("J", archive), ("J2", archive / "j2")):
        methods = {cell + "_" + cal for cal in CALIBRATIONS}
        root = parent / "evaluation"
        saved = data.read(root / "metrics.json")["models"]
        for m in methods:
            for key in saved[m].keys() & table[m].keys():
                if abs(table[m][key] - saved[m][key]) > 1e-12:
                    raise ValueError(f"Archive point estimate differs for {m}/{key}")
        actual = {(r["method"], r["case_id"], r["variant"]): r
                  for r in prepared["paths"] if r["method"] in methods}
        n = len(actual)
        path = root / "path_predictions.jsonl.gz"
        for row in stream(path if path.exists() else path.with_suffix("")):
            if row["method"] in methods:
                key = row["method"], row["case_id"], row["variant"]
                if actual.pop(key) != row:
                    raise ValueError(f"Archived dynamic path differs for {key}")
        if actual:
            raise ValueError(f"Archived {cell} path denominator differs")
        verified[cell] = dict(archived_path_records_compared=n, all_paths_identical=True,
                              metric_absolute_tolerance=1e-12)
    return verified


def run(archive, output):
    prepared, sources = prepare(archive, output)
    boot = legacy.SourceBootstrap(sources, draws=joint.DRAWS)
    arrays, table = metrics_table(prepared, boot)
    replay = verify_archive(archive, prepared, table)
    flows, source_flows = transitions(prepared, table)
    grouped = {level: defaultdict(list) for level in ("nodes", "paths", "pairs")}
    for level, by_source in grouped.items():
        for row in prepared[level]:
            by_source[row["source_id"]].append(row)
    per_source = []
    for source in sources:
        subset = dict(prepared, **{k: rows[source] for k, rows in grouped.items()})
        _, source_metrics = metrics_table(subset, legacy.SourceBootstrap([source], draws=0))
        per_source.extend(dict(source_id=source, method=m, **row) for m, row in source_metrics.items())

    budgets = []
    for r in prepared["pairs"]:
        if r["method"] == "raw" or r["group"] != "CF":
            continue
        sign = r["right_label"] - r["left_label"]
        if sign not in (-1, 1):
            raise ValueError(f"C/F pair without oriented label: {r['pair_id']}")
        signed = sign * r["correction_difference"]
        cap = (2 if r["method"].split("_")[0] in ("J2", "Asymmetric") else 1) * legacy.DELTA
        budgets.append(dict(method=r["method"], pair_id=r["pair_id"], source_id=r["source_id"],
                            signed_correction=signed, over_original=r["over_budget"],
                            over_lower=signed < -legacy.DELTA - 1e-9, over_upper=signed > cap + 1e-9))
    for m in METHODS:
        subset = [r for r in budgets if r["method"] == m]
        if sum(r["over_original"] for r in subset) != table[m]["cf_over_budget_count"]:
            raise ValueError(f"C/F budget counts disagree for {m}")
        table[m]["cf_own_lower_exceeded"] = sum(r["over_lower"] for r in subset)
        table[m]["cf_own_upper_exceeded"] = sum(r["over_upper"] for r in subset)

    legacy.write_csv(output / "b_metrics.csv", [dict(method=m, **row) for m, row in table.items()])
    legacy.write_csv(output / "b_paired_intervals.csv", intervals(arrays, boot))
    legacy.write_csv(output / "b_transitions.csv", flows)
    legacy.write_csv(output / "b_source_transitions.csv", source_flows)
    legacy.write_csv(output / "b_source_metrics.csv", per_source)
    for level in ("paths", "pairs"):
        data.lines(output / f"b_{level}.jsonl.gz", (r for r in prepared[level] if r["method"] != "raw"))
    data.lines(output / "b_cf_budgets.jsonl.gz", budgets)
    data.write(output / "b_evaluation_checks.json", dict(archive_replay=replay,
               coverage_views=len(prepared["nodes"]), sources=len(sources),
               complete_heldout_coverage=True, four_way_counts_conserved=True,
               draws=boot.draws, seed=legacy.SEED, bootstrap="common source multinomial percentile",
               method_count=len(METHODS), dynamic_decoder="legacy.path_predictions"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.archive, args.output)
