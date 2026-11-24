"""S2 event counts from frozen score archives, without model inference or replay."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METHODS = ("E1_pair_only", "J_zero")
TRANSITIONS = ("benefit", "harm", "kept_correct", "kept_wrong")
PRIMARY = "dataset_a_plus_natural"
NATURAL = "natural_fit"
ALL = "all_archived_inputs"
SCOPES = (NATURAL, "other_evaluation_views", "path_original_only", "path_interference_only")
ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
E1 = HOME / "aster-cr-diagnostics-e3cv-20261111/reports/aster_cr_diagnostics_e3cv_20261111"
JOINT = HOME / "aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115"
ZERO = HOME / "aster-m3-missing-controls-20261123/reports/aster_m3_missing_controls_20261123"
TRAINING = HOME / "aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111"
OUTPUT = ROOT / "reports/aster_v3_selective_20261124/s2_events"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def classification(raw_correct, candidate_correct):
    if raw_correct:
        return "kept_correct" if candidate_correct else "harm"
    return "benefit" if candidate_correct else "kept_wrong"


def source_quantile(events):
    """Inverse weighted empirical CDF, with equal total mass per represented source."""
    if not events:
        return None
    counts = Counter(row["source_id"] for row in events)
    cumulative = 0.0
    for row in sorted(events, key=lambda item: abs(item["s"])):
        cumulative += 1 / counts[row["source_id"]]
        if cumulative >= 0.75 * len(counts) - 1e-12:
            return abs(row["s"])
    raise ValueError("weighted cumulative mass did not reach q")


def changed_scores(records):
    result = []
    for row in records:
        raw, candidate = float(row["s"]), float(row["u"])
        if (raw > 0) != (candidate > 0):
            result.append(dict(row, s=raw, u=candidate))
    return result


def scope(row):
    if row["natural"]:
        return NATURAL
    if row["in_evaluation_views"]:
        return SCOPES[1]
    return SCOPES[2] if row["variant"] == "original" else SCOPES[3]


def training_metadata(root):
    metadata = {}
    endpoints = {r[side] for r in rows(root/"data/E1_pairs.jsonl") for side in ("left","right")}
    selected = set()
    for name in ("E1_views", "natural_views"):
        for row in rows(root/f"data/{name}.jsonl"):
            uid = row["uid"]
            if name == "E1_views":
                selected.add(uid)
            gid = re.search(r"[:\-](G[0-9]+)$", row["group_id"])
            if row["split"] != "fit" or gid is None:
                raise ValueError(f"unresolved training node identity: {uid}")
            key = (row["source_id"], row["case_id"], gid[1], row["label"])
            if metadata.setdefault(uid,key) != key:
                raise ValueError(f"ambiguous training UID: {uid}")
    if selected != endpoints:
        raise ValueError("Dataset A training view and pair endpoint identities differ")
    return metadata


def load_nodes(e1, zero, folds, training, out):
    joint = {}
    for row in rows(zero / "A_scores.jsonl.gz"):
        uid = row["view_id"]
        if uid in joint:
            raise ValueError(f"nonunique frozen score: {uid}")
        joint[uid] = (row["scores"]["R0"], row["scores"]["J_zero"], row["natural_label"])
    sources = defaultdict(Counter)
    natural = {}
    count = 0
    used_training = set()
    with gzip.open(out / "input_events.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        fields = ["scope", "method", "fold", "source_id", "case_id", "gid", "uid", "label", "s", "u", "outcome"]
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fields)
        writer.writeheader()
        for fold, split in folds.items():
            val_sources = set(split["validation_sources"])
            filename = e1 / f"evaluation/cv_{fold}_common/predictions.jsonl.gz"
            for row in rows(filename):
                count += 1
                uid = row["view_id"]
                source = row["source_id"]
                label = row["label"]
                raw = row["scores"]["raw"]
                raw_j, candidate_j, natural_label = joint.pop(f"{fold}::{uid}")
                if raw_j != raw or source not in val_sources:
                    raise ValueError(f"archive base or source mismatch: {fold}/{uid}")
                category = scope(row)
                if (label if category == NATURAL else None) != natural_label:
                    raise ValueError(f"natural target mismatch: {fold}/{uid}")
                if category == NATURAL:
                    if uid in natural:
                        raise ValueError(f"natural fit UID repeated: {uid}")
                    natural[uid] = dict(source_id=source, label=label, fold=fold)
                categories = [category, ALL]
                if uid in training:
                    src, case_id, gid, target = training[uid]
                    if source != src or label != target or uid in used_training:
                        raise ValueError(f"training score identity mismatch: {fold}/{uid}")
                    used_training.add(uid)
                    categories.append(PRIMARY)
                else:
                    case_id, gid = row["case_id"], row["gid"]
                for method, u in zip(METHODS, (row["scores"]["E1_pair_only"], candidate_j)):
                    changed = (raw > 0) != (u > 0)
                    outcome = ("unlabeled" if label is None else
                               classification((raw > 0) == label, (u > 0) == label))
                    for level in categories:
                        counters = sources[level, method, fold, source]
                        counters["inputs"] += 1
                        counters["changed"] += changed
                        counters[outcome] += 1
                    if uid in training or changed:
                        writer.writerow(dict(scope=PRIMARY if uid in training else category,
                                             method=method, fold=fold, source_id=source,
                                             case_id=case_id, gid=gid, uid=uid, label=label,
                                             s=raw, u=u, outcome=outcome))
    if used_training != set(training):
        raise ValueError(f"{len(set(training)-used_training)} training UIDs without frozen OOF scores")
    if joint:
        raise ValueError("unmatched J score records")
    return sources, natural, count


def path_counts(e1, zero, folds, groups):
    """Read completed paths; pair archived outcomes without executing a tree."""
    for fold, split in folds.items():
        archived = defaultdict(dict)
        filename = e1 / f"evaluation/cv_{fold}_common/path_predictions.jsonl.gz"
        for row in rows(filename):
            if row["method"] not in ("raw", "E1_pair_only"):
                continue
            case_key = (row["source_id"], row["case_id"], row["variant"], row["interference"])
            if row["method"] in archived[case_key]:
                raise ValueError(f"duplicate path record: {case_key}")
            archived[case_key][row["method"]] = row["correct"]
        for (source, _, _, interference), pair in archived.items():
            group = "interference_path" if interference else "natural_path"
            result = classification(pair["raw"], pair["E1_pair_only"])
            c = groups[group, "E1_pair_only", fold, source]
            c["inputs"] += 1
            c[result] += 1
    source_fold = {s: fold for fold, sp in folds.items() for s in sp["validation_sources"]}
    with (zero / "A_source_transitions.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row["lhs"], row["rhs"]) != ("J_zero", "R0") or row["level"] not in ("natural_path", "interference_path"):
                continue
            source = row["source_id"]
            c = groups[row["level"], "J_zero", source_fold[source], source]
            if c:
                raise ValueError("duplicate cached source transitions")
            for key, col in zip(TRANSITIONS, ("wrong_to_right", "right_to_wrong", "right_to_right", "wrong_to_wrong")):
                c[key] = int(row[col])
            c["inputs"] = int(row["total"])


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summary(c):
    b, h, k, w = (c[name] for name in TRANSITIONS)
    n = b + h + k + w
    return dict(inputs=c["inputs"], labeled=n, unlabeled=c["unlabeled"],
                **{name: c[name] for name in TRANSITIONS}, raw_correct=h+k, raw_wrong=b+w,
                benefit_rate=ratio(b, b+w), harm_rate=ratio(h, h+k), net_accuracy=ratio(b-h, n),
                oracle_correct=k+h+b, oracle_accuracy=ratio(k+h+b, n))


def summarize(groups, folds, out):
    all_sources = sorted(s for f in folds.values() for s in f["validation_sources"])
    rng = np.random.default_rng(20261111)
    draw = rng.multinomial(len(all_sources), np.full(len(all_sources), 1 / len(all_sources)), 2000)
    pooled, by_source, ci = [], [], []
    for level in (PRIMARY, ALL, *SCOPES, "natural_path", "interference_path"):
        for method in METHODS:
            source_counts = defaultdict(Counter)
            c = Counter()
            for (scope_name, meth, fold, source), counts in groups.items():
                if (scope_name, meth) != (level, method):
                    continue
                c.update(counts)
                source_counts[source].update(counts)
                by_source.append(dict(scope=level, method=method, fold=fold, source_id=source, **summary(counts)))
            if not c:
                raise ValueError(f"missing {level}/{method}")
            input_level = level in (PRIMARY, ALL, *SCOPES)
            pooled.append(dict(scope=level, method=method, sources=len(source_counts),
                               changed=c["changed"] if input_level else None,
                               benefit_sources=sum(v["benefit"] > 0 for v in source_counts.values()),
                               harm_sources=sum(v["harm"] > 0 for v in source_counts.values()), **summary(c)))
            if not input_level:
                continue
            values = np.array([[source_counts[s][k] for k in TRANSITIONS] for s in all_sources])
            b, h, k, w = (draw @ values).T
            for metric, num, den in (("benefit_rate", b, b+w), ("harm_rate", h, h+k),
                                     ("net_accuracy", b-h, b+h+k+w), ("oracle_accuracy", k+h+b, b+h+k+w)):
                values_ci = np.divide(num[den>0], den[den>0])
                lo, hi = np.quantile(values_ci,[.025,.975]) if len(values_ci) else (None,None)
                ci.append(dict(scope=level, method=method, metric=metric, low=lo, high=hi,
                               draws=2000, valid_draws=len(values_ci)))
    write_csv(out / "source_counts.csv", by_source)
    write_csv(out / "pooled_counts.csv", pooled)
    write_csv(out / "source_intervals.csv", ci)
    return pooled


def conditional_inner(joint, training, out):
    """Archive's J selection holdouts reuse the 64-source representation."""
    result = []
    for winner in read(joint / "selected_candidates.json"):
        fold = winner["fold"]
        base = joint / "selection" / fold
        split = read(base / "split.json")
        valid = set(split["inner_validation_sources"])
        assert valid.isdisjoint(split["inner_train_sources"]) and split["conditional_on_fixed_representation"]
        ids = {uid for uid, v in training.items() if v[0] in valid}
        name = f"J_lambda_{winner['lambda_ap']}"
        model = read(base / f"{name}.json")
        assert set(model["train_sources"]) == set(split["inner_train_sources"])
        seen, counter, changed_sources = set(), Counter(), defaultdict(set)
        for row in rows(base / f"{name}_scores.jsonl.gz"):
            uid = row["view_id"]
            if uid not in ids:
                continue
            if uid in seen:
                raise ValueError(f"duplicate conditional holdout UID: {fold}/{uid}")
            seen.add(uid)
            src, _, _, label = training[uid]
            s, u = row["scores"]["R0"], row["scores"]["J_shared"] - model["b_shared"]
            result_type = classification((s>0) == label, (u>0) == label)
            counter[result_type] += 1
            counter["inputs"] += 1
            changed_sources[result_type].add(src)
        if seen != ids:
            raise ValueError(f"missing conditional holdout UIDs: {fold}")
        result.append(dict(fold=fold, selected_lambda=winner["lambda_ap"], validation_sources=len(valid),
                           benefit_sources=len(changed_sources["benefit"]), harm_sources=len(changed_sources["harm"]),
                           scope="dataset_A_plus_natural_conditional_on_fixed_64_source_representation", **summary(counter)))
    write_csv(out / "j_conditional_inner.csv", result)


def parameters(out, pooled, folds):
    with gzip.open(out / "input_events.csv.gz", "rt", encoding="utf-8") as handle:
        primary = [r for r in csv.DictReader(handle) if r["scope"] == PRIMARY]
    for method in METHODS:
        selected = [r for r in primary if r["method"] == method]
        events = changed_scores(selected)
        total = next(r for r in pooled if r["scope"] == PRIMARY and r["method"] == method)
        eligible = (total["benefit"] >= 30 and total["harm"] >= 30 and
                    total["benefit_sources"] >= 10 and total["harm_sources"] >= 10)
        # Count each clinical node at most once in each direction across its variants.
        case_counts = []
        for outcome in TRANSITIONS:
            rr = [r for r in selected if r["outcome"] == outcome]
            case_counts.append(dict(outcome=outcome, inputs=len(rr),
                                    cases=len({r["case_id"] for r in rr}),
                                    case_nodes=len({(r["case_id"],r["gid"]) for r in rr}),
                                    sources=len({r["source_id"] for r in rr})))
        write_csv(out / f"{method}_case_node_counts.csv", case_counts)
        gate = None
        if eligible:
            from sklearn.linear_model import LogisticRegression
            labeled = [r for r in events if r["outcome"] in ("benefit", "harm")]
            X = [[r["s"],r["u"]-r["s"],abs(r["s"])] for r in labeled]
            labels = [int(r["outcome"] == "benefit") for r in labeled]
            cnt = Counter(r["source_id"] for r in labeled)
            weights = [len(labeled)/(len(cnt)*cnt[r["source_id"]]) for r in labeled]
            clf = LogisticRegression(C=1, penalty="l2", solver="lbfgs", max_iter=1000)
            clf.fit(X, labels, sample_weight=weights)
            gate = dict(features=["s","u_minus_s","abs_s"], coef=clf.coef_[0].tolist(),
                        intercept=float(clf.intercept_[0]), threshold=2/3, strict=True)
        config = dict(candidate=method, fit_cohort="80 archived fit sources; five source-held-out candidate folds",
                      denominator=total["labeled"], unique_sources=total["sources"], changed=len(events),
                      benefit=total["benefit"], harm=total["harm"], benefit_sources=total["benefit_sources"],
                      harm_sources=total["harm_sources"], gate_eligible=eligible, gate=gate,
                      oracle_benefit=total["benefit"], q=source_quantile(events), null_q="+infinity",
                      raw_error_upper_bound=total["raw_wrong"],
                      quantile="inverse source-equal empirical CDF on candidate-changed Dataset A/frozen natural training UIDs",
                      original_zero="negative", candidate_zero="negative",
                      apply_to="E1 full on archived development split" if method == METHODS[0]
                      else "descriptive train80 q; equivalent full80 J candidate absent")
        write_json(out / ("e1_train80_parameters.json" if method == METHODS[0] else "j_train80_parameters.json"), config)
    bounds=[]
    for fold, sp in folds.items():
        within = [r for r in primary if r["method"] == METHODS[0] and r["source_id"] in sp["train_sources"]]
        raw_errors=[r for r in within if r["outcome"] in ("benefit","kept_wrong")]
        bounds.append(dict(fold=fold, train_sources=len(sp["train_sources"]), training_uids=len(within),
                           maximum_possible_b=len(raw_errors), maximum_possible_b_sources=len({r["source_id"] for r in raw_errors}),
                           supervised_feasibility="impossible_by_raw_error_bound" if len(raw_errors)<30 else "requires_true_inner_OOF"))
    write_csv(out / "inner_training_bounds.csv",bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("e1", E1), ("training", TRAINING), ("joint", JOINT), ("zero", ZERO), ("output", OUTPUT)):
        parser.add_argument("--"+name, type=Path, default=default)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    args.output.mkdir(parents=True, exist_ok=True)
    folds = {f"fold{row['fold']}": row for row in read(args.e1/"folds.json")["folds"]}
    training = training_metadata(args.training)
    groups, natural, count = load_nodes(args.e1, args.zero, folds, training, args.output)
    path_counts(args.e1, args.zero, folds, groups)
    pooled = summarize(groups, folds, args.output)
    parameters(args.output, pooled, folds)
    conditional_inner(args.joint, training, args.output)
    result = dict(reused_candidate_scores=2*count, reused_fold_inputs=count, natural_fit_uids=len(natural), dataset_a_plus_natural_uids=len(training),
                  new_candidate_inferences=0, new_candidate_fits=0, new_gate_fits=sum(int(read(args.output/f"{k}_train80_parameters.json")["gate_eligible"]) for k in ("e1","j")), new_path_replays=0,
                  candidate_path_pairs=sum(r["inputs"] for r in pooled if r["scope"].endswith("_path")),
                  new_source_rows=sum(1 for _ in (args.output/"source_counts.csv").open())-1,
                  gpu_hours=0, cpu_process_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-started,
                  sources={"training_metadata":str(args.training),"E1":str(args.e1),"J_zero":str(args.zero),"J_selection":str(args.joint)},
                  outer_fold_nested_score_status="blocked; E1 inner weights absent; J inner M1/M2 fitted on outer 64",
                  source_bootstrap_draws=2000)
    write_json(args.output/"run.json", result)


if __name__ == "__main__":
    main()
