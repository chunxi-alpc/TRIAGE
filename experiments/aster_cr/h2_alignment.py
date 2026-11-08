"""Align H2 prediction and signal leakage for the same output correction.

Standalone CPU diagnostic; reads saved arrays and leaves the training pipeline intact.
"""
import argparse
import csv
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

KINDS = ("A", "P", "C", "F")


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None


def metrics(target, predicted):
    """predicted is the nuisance effect to subtract, including for an existing CR."""
    target, predicted = np.asarray(target, dtype=float), np.asarray(predicted, dtype=float)
    if not len(target):
        return dict(n=0, r2=None, target_rms=None, residual_rms=None,
                    residual_mean=None, normalized_mse=None, correction_rms=None,
                    signal_leakage_ratio=None)
    residual = target - predicted
    sst = float(np.sum((target - target.mean()) ** 2))
    energy = float(np.sum(target ** 2))
    trms = rms(target)
    return dict(n=len(target), r2=1-float(residual @ residual)/sst if sst > 0 else None,
                target_rms=trms, residual_rms=rms(residual),
                residual_mean=float(residual.mean()),
                normalized_mse=float(residual @ residual)/energy if energy > 0 else None,
                correction_rms=rms(predicted),
                signal_leakage_ratio=rms(predicted)/trms if trms else None)


def evaluate(pairs, predictions, scores, lookup):
    result = {}
    for group, kinds in [("AP", ("A", "P")), ("CF", ("C", "F"))] + [(k, (k,)) for k in KINDS]:
        subset = [p for p in pairs if p["kind"] in kinds]
        y = np.array([scores[p["right"]]["ell"]-scores[p["left"]]["ell"] for p in subset])
        dy = np.array([predictions[lookup[p["right"]]]-predictions[lookup[p["left"]]] for p in subset])
        result[group] = metrics(y, dy)
        result[group]["sources"] = len({p["source_id"] for p in subset})
    ap, cf = result["AP"], result["CF"]
    result["criterion_pass"] = (ap["r2"] > 0 and cf["signal_leakage_ratio"] <= .1
                               if ap["r2"] is not None and cf["signal_leakage_ratio"] is not None else None)
    return result


def self_test():
    z = np.array([[1., 2.], [4., -1.], [0., 3.]])
    w = np.array([.3, -.7])
    delta = (z[1:]-z[0]) @ w
    np.testing.assert_allclose(delta, (z @ w)[1:]-(z @ w)[0])
    correction = z @ w + 4.
    nuisance_prediction = -correction
    base = np.array([2., 7., -1.])
    np.testing.assert_allclose(np.diff(base)-np.diff(nuisance_prediction), np.diff(base+correction))
    assert metrics([1., 3.], [1., 3.])["r2"] == 1.
    assert metrics([1., 3.], [0., 0.])["signal_leakage_ratio"] == 0.
    assert metrics([], [])["r2"] is None
    print("self-test passed", flush=True)


def run(args):
    started = time.time()
    root, output = args.root, args.output
    output.mkdir(parents=True, exist_ok=True)
    model_root = root / args.model
    selection = json.loads((model_root / "development_selection.json").read_text())
    name = selection["aster_cr"]
    with (model_root / "readouts" / (name + ".pkl")).open("rb") as file:
        cr = pickle.load(file)
    # These are trusted local artifacts created by the existing experiment.
    assert cr["fit"]["mode"] == "linear"
    scores = {}
    for dirname in ("pending_test_scores", "scores"):
        for path in sorted((model_root / dirname).glob("*.jsonl")):
            for row in rows(path):
                if row.get("status") == "ok":
                    scores[row["view_id"]] = row
    views = rows(root / "data/views.jsonl")
    view_by_id = {v["view_id"]: v for v in views}
    all_pairs = [p for p in rows(root / "data/pairs.jsonl") if p["kind"] in KINDS]
    pairs = [p for p in all_pairs if p["left"] in scores and p["right"] in scores]
    available = {vid for p in pairs for vid in (p["left"], p["right"])}
    saved_dev = {r["view_id"]: r["ell"] for r in rows(model_root / "readout_scores" / (name + "_dev.jsonl"))}
    available.update(vid for vid in saved_dev if vid in scores and vid in view_by_id)
    ids = sorted(available)
    lookup = {vid: i for i, vid in enumerate(ids)}
    supports = {}
    for rank in ("4", "8", "16", "full"):
        with np.load(model_root / ("supports_r" + rank + ".npz")) as arrays:
            supports[rank] = {k: arrays[k] for k in arrays.files}
    heads = supports["16"]["heads"]
    np.testing.assert_array_equal(heads, cr["heads"])
    raw = np.empty((len(ids), len(supports["16"]["mu"])), dtype=np.float32)
    # Load each feature file once, even when multiple views reuse it.
    by_file = {}
    for vid in ids:
        by_file.setdefault(scores[vid]["feature"], []).append(lookup[vid])
    for index, (path, positions) in enumerate(by_file.items()):
        with np.load(path) as arrays:
            h = arrays["heads"]
            value = h.reshape(-1, h.shape[-1])[heads].reshape(-1).astype(np.float32)
        raw[positions] = value
        if (index+1) % 1000 == 0:
            print("loaded", index+1, "of", len(by_file), flush=True)
    result = dict(model=args.model, input_root=str(root), selected_cr=name,
                  selected_component=cr["component"], ranks=["4", "8", "16", "full"],
                  alpha=.01, fit_intercept=False, gpu_used=False,
                  target="original total logit difference: ell(right)-ell(left); not head-mediated IE",
                  criterion="AP centered R2 > 0 and CF correction RMS / original difference RMS <= 0.1",
                  actual_cr_sign="q_CR(x) = -(ell_CR(x)-ell(x)); subtract q_CR to recover CR",
                  test_selection="No fitting or model selection uses dev/test in this diagnostic",
                  coverage={}, regressions={}, actual_cr={}, checks={})
    for split in ("fit", "dev", "test"):
        expected = Counter(p["kind"] for p in all_pairs if p["split"] == split)
        observed = Counter(p["kind"] for p in pairs if p["split"] == split)
        result["coverage"][split] = {k: dict(expected=expected[k], available=observed[k],
                                                 missing=expected[k]-observed[k]) for k in KINDS}
    training_pairs = [p for p in pairs if p["split"] == "fit" and p["kind"] in ("A", "P")]
    if not training_pairs:
        raise ValueError("No fit A/P pairs available")
    left = np.array([lookup[p["left"]] for p in training_pairs])
    right = np.array([lookup[p["right"]] for p in training_pairs])
    target = np.array([scores[p["right"]]["ell"]-scores[p["left"]]["ell"] for p in training_pairs])
    coefficients = {}
    pair_records = []
    def record(method, q):
        split_results = {}
        for split in ("dev", "test"):
            subset = [p for p in pairs if p["split"] == split]
            split_results[split] = evaluate(subset, q, scores, lookup)
            for p in subset:
                y = scores[p["right"]]["ell"]-scores[p["left"]]["ell"]
                prediction = float(q[lookup[p["right"]]]-q[lookup[p["left"]]])
                pair_records.append(dict(method=method, split=split, pair_id=p["pair_id"],
                                         kind=p["kind"], source_id=p["source_id"],
                                         original_difference=y, predicted_nuisance_difference=prediction,
                                         residual_difference=y-prediction))
        return split_results
    for rank, support in supports.items():
        np.testing.assert_array_equal(heads, support["heads"])
        x = (raw-support["mu"])/support["scale"]
        projection = (x @ support["N"]).astype(np.float64)
        dx = projection[right]-projection[left]
        estimator = Ridge(alpha=.01, fit_intercept=False).fit(dx, target)
        q = projection @ estimator.coef_
        count = min(100, len(left))
        error = float(np.max(np.abs(dx[:count] @ estimator.coef_-(q[right[:count]]-q[left[:count]]))))
        np.testing.assert_allclose(dx[:count] @ estimator.coef_, q[right[:count]]-q[left[:count]], rtol=1e-9, atol=1e-8)
        result["checks"]["pair_difference_max_error_rank_"+rank] = error
        result["regressions"][rank] = dict(dimensions=projection.shape[1], fit_pairs=len(target), **record("ridge_r"+rank, q))
        coefficients["beta_r"+rank] = estimator.coef_
        print("completed rank", rank, flush=True)
        del projection, dx, x, estimator
    support = cr["support"]
    x = (raw-support["mu"])/support["scale"]
    parts = []
    if cr["component"] in ("signal", "both"):
        parts.append(x @ support["S"])
    if cr["component"] in ("nuisance", "both"):
        parts.append(x @ support["N"])
    z = np.concatenate(parts, axis=-1).astype(np.float32) if parts else np.empty((len(raw), 0), np.float32)
    # Keep float64 when differencing corrections; match existing predict's addition order.
    base = np.array([scores[vid]["ell"] for vid in ids])
    predicted_cr = base + cr["fit"]["b"] + z @ cr["fit"]["w"]
    q_cr = -(predicted_cr-base)
    result["actual_cr"] = record(name, q_cr)
    common_ids = sorted(set(ids) & set(saved_dev))
    checks = np.array([predicted_cr[lookup[vid]]-saved_dev[vid] for vid in common_ids])
    if not len(checks):
        raise ValueError("No saved development CR scores to verify")
    result["checks"]["saved_cr_scores_compared"] = len(checks)
    result["checks"]["saved_cr_scores_max_abs_error"] = float(np.max(np.abs(checks)))
    result["checks"]["saved_cr_decision_disagreements"] = int(sum(
        (predicted_cr[lookup[vid]] > 0) != (saved_dev[vid] > 0) for vid in common_ids))
    if np.max(np.abs(checks)) > 2e-4:
        raise ValueError("Reconstructed CR does not match saved scores")
    old_path = model_root / "h2.json"
    result["original_h2_separate_operations"] = json.loads(old_path.read_text())
    result["runtime_seconds"] = time.time()-started
    result["views_loaded"] = len(ids)
    result["unique_feature_files_loaded"] = len(by_file)
    write_json(output / "metrics.json", result)
    np.savez_compressed(output / "ridge_coefficients.npz", **coefficients)
    with (output / "pair_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(pair_records)
    with (output / "summary.csv").open("w", newline="") as file:
        fields = ["method", "split", "group", "n", "sources", "r2", "target_rms", "residual_rms",
                  "residual_mean", "normalized_mse", "correction_rms", "signal_leakage_ratio"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for method, res in [(f"ridge_r{k}", v) for k, v in result["regressions"].items()] + [(name, result["actual_cr"])]:
            for split in ("dev", "test"):
                for group in ("AP", "CF", *KINDS):
                    writer.writerow(dict(method=method, split=split, group=group, **res[split][group]))
    print(json.dumps(dict(status="complete", seconds=result["runtime_seconds"],
                          checks=result["checks"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/gcx/aster_cr_repair_v2_run_20261108"))
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--output", type=Path, default=Path("reports/aster_h2_alignment_20261108"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else run(args)
