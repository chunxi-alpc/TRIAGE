"""Issue 86, experiment A. Replay five held-out folds with the frozen J correction."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.aster_m3_joint_study import data
from experiments.aster_m3_joint_study.evaluation import DRAWS, legacy

METHODS = ("R0", "R1", "J_zero", "J_shared", "J_independent")
ARCHIVED = tuple(m for m in METHODS if m != "J_zero")
COMPARISONS = (
    ("J_zero", "R0"), ("J_zero", "R1"), ("J_zero", "J_shared"),
    ("J_zero", "J_independent"), ("J_shared", "R0"),
)
CATEGORIES = ("wrong_to_right", "right_to_wrong", "right_to_right", "wrong_to_wrong")
DENOMINATORS = {"natural_node": 761, "natural_path": 1712, "interference_path": 5136}


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def archived_scores(row, model):
    raw = row["raw_score"]
    zero = raw + row["correction"]["J"]
    if zero != row["uncalibrated_score"]["J"]:
        raise ValueError(f"Zero-intercept reconstruction failed at {row['view_id']}")
    return dict(R0=raw, R1=raw + model["b_shared"], J_zero=zero,
                J_shared=zero + model["b_shared"], J_independent=zero + model["b_independent"])


def check_coverage(required, available, kind, output):
    missing = sorted(required - available)
    if missing:
        data.write(output / "A_missing_ids.json", dict(kind=kind, required=len(required),
                   covered=len(required)-len(missing), missing=missing))
        raise ValueError(f"{kind}: {len(missing)} missing IDs. See A_missing_ids.json.")


def load_scores(archive, output, folds):
    models = {fold: data.read(archive / "models" / fold / "J.json") for fold in folds}
    for fold, model in models.items():
        raw_intercept = data.read(archive / "models" / fold / "R1.json")["b0"]
        assert model["b_shared"] == raw_intercept
        assert set(model["train_sources"]).isdisjoint(folds[fold])
    scores = {m: {} for m in METHODS}
    for row in records(archive / "corrections.jsonl.gz"):
        key, fold = row["view_id"], row["fold"]
        assert key not in scores["R0"] and row["source_id"] in folds[fold]
        for method, value in archived_scores(row, models[fold]).items():
            scores[method][key] = value
    node_rows = []
    maximum = defaultdict(float)
    expected = set()
    for row in records(archive / "evaluation/predictions.jsonl.gz"):
        key = row["view_id"]
        expected.add(key)
        if key not in scores["R0"]:
            continue
        for method in ARCHIVED:
            gap = abs(scores[method][key] - row["scores"][method])
            maximum[method] = max(maximum[method], gap)
            assert int(scores[method][key] > 0) == row["predictions"][method]
        if row["natural"]:
            assert row["label_available"]
            node_rows.append(dict(view_id=key, source_id=row["source_id"], label=int(row["label"])))
    check_coverage(expected, set(scores["R0"]), "scores", output)
    assert set(scores["R0"]) == expected and max(maximum.values()) < 1e-12
    return scores, node_rows, dict(maximum)


def load_scope(e1, cases_file, source_fold):
    cases = [r for r in records(cases_file) if r["split"] == "fit"]
    views, pairs = [], []
    for name in ("path_features.jsonl", "path_interference_views.jsonl"):
        for row in legacy.data_rows(e1, name):
            if row["split"] == "fit":
                key = source_fold[row["source_id"]] + "::" + legacy.uid(row)
                views.append(dict(row, uid=key))
    for row in legacy.data_rows(e1, "E2_pairs.jsonl"):
        prefix = source_fold[row["source_id"]] + "::"
        pairs.append(dict(row, left=prefix+row["left"], right=prefix+row["right"]))
    return cases, views, pairs


def check_pair_deltas(pairs, scores):
    counts, maximum = Counter(), defaultdict(float)
    for pair in pairs:
        group = legacy.pair_group(pair)
        counts[group] += 1
        left, right = pair["left"], pair["right"]
        for before, after in (("R0", "R1"), ("J_zero", "J_shared"), ("J_zero", "J_independent")):
            a = scores[before][right] - scores[before][left]
            b = scores[after][right] - scores[after][left]
            key = f"{group}_{before}_to_{after}"
            maximum[key] = max(maximum[key], abs(a-b))
    assert counts == {"AP": 5003, "CF": 1342}
    assert max(maximum.values()) < 1e-12
    return dict(pair_counts=dict(counts), max_abs_difference=dict(maximum))


def path_key(row):
    return row["case_id"], row["variant"]


def replay_paths(cases, views, scores, archive, output):
    check_coverage({legacy.uid(v) for v in views}, set(scores["R0"]), "path_view_scores", output)
    paths = legacy.path_predictions(cases, views, scores)
    by_method = {m: {} for m in METHODS}
    for row in paths:
        by_method[row["method"]][path_key(row)] = row
    assert len(paths) == sum(map(len, by_method.values()))
    scope = set(by_method["R0"])
    archive_keys = defaultdict(set)
    for row in records(archive / "evaluation/path_predictions.jsonl.gz"):
        method = row["method"]
        if method in ARCHIVED:
            key = path_key(row)
            archive_keys[method].add(key)
            if key in scope:
                assert by_method[method][key] == row, (key, method)
    for method in ARCHIVED:
        check_coverage(archive_keys[method], set(by_method[method]), "dynamic_paths_" + method, output)
        assert archive_keys[method] == set(by_method[method])
    assert all(set(rows) == scope for rows in by_method.values())
    gaps = [path_key(r) for r in paths if r["error"] == "unavailable_score"]
    if gaps:
        data.write(output / "A_missing_ids.json", dict(kind="dynamic_paths", missing=gaps))
        raise ValueError("Path scorer encountered unavailable input")
    return paths, by_method


def category(before, after):
    return (("wrong_to_wrong", "wrong_to_right"), ("right_to_wrong", "right_to_right"))[before][after]


def compare(level, ids, sources, correct, boot):
    """All methods and levels share the archived source draws and pooled estimator."""
    total = DENOMINATORS[level]
    assert len(ids) == len(set(ids)) == len(sources) == total
    metrics, comparisons, per_source = [], [], []
    draws = {}
    for method in METHODS:
        values = correct[method]
        draws[method] = boot.mean(values, sources)
        metrics.append(dict(level=level, method=method, correct=sum(values), total=total,
                            accuracy=draws[method][0]))
    for lhs, rhs in COMPARISONS:
        counts = defaultdict(Counter)
        for src, before, after in zip(sources, correct[rhs], correct[lhs], strict=True):
            counts[src][category(before, after)] += 1
        overall = sum(counts.values(), Counter())
        diff = sum(correct[lhs]) - sum(correct[rhs])
        assert sum(overall.values()) == total
        assert overall["wrong_to_right"] - overall["right_to_wrong"] == diff
        common = dict(level=level, lhs=lhs, rhs=rhs,
                      intercept_only=lhs == "J_zero" and rhs in ("J_shared", "J_independent"))
        comparisons.append(dict(**common, **{k:overall[k] for k in CATEGORIES}, net=diff, total=total,
                           **boot.summary(draws[lhs] - draws[rhs])))
        for src in boot.sources:
            c = counts[src]
            per_source.append(dict(**common, source_id=src, **{k:c[k] for k in CATEGORIES},
                                   net=c["wrong_to_right"]-c["right_to_wrong"], total=sum(c.values())))
    return metrics, comparisons, per_source

def first_error_boundaries(by_method, cases, views, scores):
    node_ids = {}
    for view in views:
        variant = view.get("variant", "original")
        if variant == "full":
            variant = "original"
        node_ids[view["case_id"], variant, view["gid"]] = legacy.uid(view)
    gold = {c["case_id"]: {r["gid"]: int(r["label"]) for r in c["gold_path"]} for c in cases}
    for lhs, rhs in COMPARISONS:
        for key, old in by_method[rhs].items():
            new = by_method[lhs][key]
            def boundary(gid):
                if gid is None:
                    return None
                view_id = node_ids[*key, gid]
                label = gold[key[0]].get(gid)
                before, after = scores[rhs][view_id], scores[lhs][view_id]
                sign = 2 * label - 1 if label is not None else None
                return dict(gid=gid, view_id=view_id, label=label, before_score=before,
                            after_score=after, before_gold_margin=sign*before if sign else None,
                            after_gold_margin=sign*after if sign else None,
                            reached_before=any(r["gid"]==gid for r in old["path"]),
                            reached_after=any(r["gid"]==gid for r in new["path"]))
            yield dict(lhs=lhs, rhs=rhs, case_id=key[0], variant=key[1],
                       source_id=old["source_id"], transition=category(old["correct"], new["correct"]),
                       before_first_error=boundary(old["first_error"]),
                       after_first_error=boundary(new["first_error"]))


def run(archive, output, e1=None, cases_file=None):
    output.mkdir(parents=True, exist_ok=True)
    cfg = data.read(archive / "configuration.json")
    split = data.read(archive / "folds.json")
    folds = {"fold"+str(f["fold"]): f["validation_sources"] for f in split["folds"]}
    source_fold = {s: fold for fold, sources in folds.items() for s in sources}
    assert len(source_fold) == 80 and all(len(sources) == 16 for sources in folds.values())
    assert set(source_fold) == set(split["all_sources"])
    scores, nodes, roundoff = load_scores(archive, output, folds)
    cases, views, pairs = load_scope(e1 or Path(cfg["historical_root"]),
                                     cases_file or Path(cfg["cases_file"]), source_fold)
    check_coverage({p[k] for p in pairs for k in ("left", "right")}, set(scores["R0"]),
                   "pair_endpoints", output)
    invariance = check_pair_deltas(pairs, scores)
    paths, by_method = replay_paths(cases, views, scores, archive, output)
    boot = legacy.SourceBootstrap(source_fold, draws=DRAWS)
    assert DRAWS == cfg["bootstrap_draws"] and legacy.SEED == cfg["bootstrap_seed"]
    metrics, comparisons, per_source = [], [], []
    for level in DENOMINATORS:
        if level == "natural_node":
            ids = [r["view_id"] for r in nodes]
            sources = [r["source_id"] for r in nodes]
            correct = {m:[int(scores[m][r["view_id"]]>0)==r["label"] for r in nodes] for m in METHODS}
        else:
            ids = [k for k, r in by_method["R0"].items()
                   if r["interference"] == (level == "interference_path")]
            sources = [by_method["R0"][k]["source_id"] for k in ids]
            correct = {m:[by_method[m][k]["correct"] for k in ids] for m in METHODS}
        rows, contrasts, source_rows = compare(level, ids, sources, correct, boot)
        metrics.extend(rows)
        comparisons.extend(contrasts)
        per_source.extend(source_rows)
    historical = data.read(archive/"evaluation/metrics.json")["models"]
    for row in metrics:
        if row["method"] in ARCHIVED:
            baseline = historical[row["method"]]
            assert row["correct"] == baseline[row["level"]+"_correct"]
            assert row["total"] == baseline[row["level"]+"_count"]
    tsv(output / "A_metrics.tsv", metrics)
    tsv(output / "A_comparisons.tsv", comparisons)
    tsv(output / "A_source_transitions.tsv", per_source)
    data.lines(output / "A_paths.jsonl.gz", paths)
    data.lines(output / "A_first_error_boundaries.jsonl.gz", first_error_boundaries(by_method, cases, views, scores))
    natural = {r["view_id"]: r["label"] for r in nodes}
    data.lines(output / "A_scores.jsonl.gz", (
        dict(view_id=r["view_id"], source_id=r["source_id"], fold=r["fold"],
             natural_label=natural.get(r["view_id"]), correction_J=r["correction"]["J"],
             scores={m:scores[m][r["view_id"]] for m in METHODS})
        for r in records(archive / "corrections.jsonl.gz")))
    checks = dict(state="complete", folds=len(folds), sources=len(source_fold), unique_fold_inputs=len(scores["R0"]),
                  dynamic_path_records=len(paths), compared_archived_path_records=sum(len(by_method[m]) for m in ARCHIVED),
                  bootstrap_seed=legacy.SEED, draws=DRAWS, archive_score_max_abs_gap=roundoff,
                  archive_decision_mismatches=0, archive_path_mismatches=0, missing_scores=0,
                  zero_reconstruction_max_abs_gap=0., pair_intercept_invariance=invariance)
    data.write(output / "A_verification.json", checks)
    print(json.dumps(checks, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=data.REPO/"reports/aster_m3_joint_study_20261115")
    parser.add_argument("--output", type=Path, default=data.REPO/"reports/aster_m3_missing_controls_20261123")
    parser.add_argument("--e1", type=Path, help="Override archived E1 data root")
    parser.add_argument("--cases", type=Path, help="Override archived path case file")
    args = parser.parse_args()
    run(args.archive, args.output, args.e1, args.cases)

