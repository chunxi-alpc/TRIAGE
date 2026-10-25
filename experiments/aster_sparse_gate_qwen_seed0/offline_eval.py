#!/usr/bin/env python3
"""Qwen seed=0 exploratory analysis of reliability-gated attention residuals.

No new model inference or head-level intervention. Reuses 800 prior Qwen
GPU outputs from PR #32 (inherited by PR #33). Four previously studied
development trees only; NOT an independent test. Never reads P3 labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CORE = ("NC", "AC", "IC", "SW")
POLICIES = ("static", "full", "margin", "disagree", "direction",
            "margin_direction", "disagree_direction", "rotation",
            "flip", "reverse", "null")
GRID = [(b, t0, ts) for b in (.5, .75, 1.)
        for t0 in (1., 2., 4.) for ts in (.25, .5, 1.)]


def norm(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid action probabilities")
    return x / x.sum() if x.sum() > 0 else np.full(len(x), 1. / len(x))


def softmax(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite logits")
    e = np.exp(x - x.max())
    return e / e.sum()


def temperature(p, t):
    return softmax(np.log(np.maximum(norm(p), 1e-30)) / t)


def patient_sign(s):
    h = 0
    for c in str(s):
        h = (h * 33 + ord(c)) & 0xffffffff
    return 1 if h % 2 else -1


def prepare(r):
    if (r["split"], r["seed"]) != ("dev", 0) or r["variant"] not in CORE:
        raise ValueError("Only development, seed=0 and the four core variants")
    scores = r["scores"]
    needed = ("original", "mass_zero_control", "action_bias_1.0_all")
    if not all(x in scores for x in needed):
        raise ValueError("Matched full-layer zero-control logits are mandatory")
    p = softmax(scores["original"])
    s = norm(r["option_S"])
    delta = np.asarray(scores["action_bias_1.0_all"], float) - np.asarray(scores["mass_zero_control"], float)
    null = np.asarray(scores["mass_zero_control"], float) - np.asarray(scores["original"], float)
    if not (p.shape == s.shape == delta.shape == null.shape):
        raise ValueError("Mismatched action vectors")
    top = int(np.argmax(s))
    orig = int(np.argmax(p))
    sorted_s = np.sort(s)
    return dict(id=r["id"], tree=r["tree"], variant=r["variant"],
                gold=int(r["gold_option"]), p=p, s=s, delta=delta,
                null=null, original=orig, margin=float(sorted_s[-1]-sorted_s[-2]),
                agree=top == orig, direction=delta[top] > delta[orig])


def predict(r, c, policy):
    beta, t0, ts = c
    p = temperature(r["p"], t0)
    s = temperature(r["s"], ts)
    base = (1 - beta) * p + beta * s
    if policy == "static":
        return int(np.argmax(base))
    gate = 1.
    if "margin" in policy:
        gate *= min(1., max(0., r["margin"] / .5))
    if "disagree" in policy:
        gate *= float(not r["agree"])
    if "direction" in policy:
        gate *= float(r["direction"])
    delta = r["delta"]
    if policy == "rotation":
        delta = np.roll(delta, -1)
    elif policy == "flip":
        delta = delta * patient_sign(r["id"])
    elif policy == "reverse":
        delta = -delta
    elif policy == "null":
        delta = r["null"]
    return int(np.argmax(np.log(np.maximum(base, 1e-30))
                         + .25 * gate * np.tanh(delta / 2.)))


def evaluate(rows, policy):
    trees = sorted({r["tree"] for r in rows})
    y = np.array([r["gold"] for r in rows])
    matrix = np.array([[predict(r, c, policy) for r in rows] for c in GRID])
    tree = np.array([r["tree"] for r in rows])
    scores = np.array([
        [float(np.mean(pred[tree == t] == y[tree == t])) for t in trees]
        for pred in matrix
    ])
    out = np.empty(len(rows), int)
    matched = np.empty(len(rows), int)
    choices = {}
    for j, hold in enumerate(trees):
        pick = int(np.argmax(np.delete(scores, j, axis=1).mean(axis=1)))
        ix = np.flatnonzero(tree == hold)
        out[ix] = matrix[pick, ix]
        matched[ix] = [predict(rows[i], GRID[pick], "static") for i in ix]
        choices[hold] = list(GRID[pick])
    return {
        "correct": int(np.sum(out == y)),
        "accuracy": float(np.mean(out == y)),
        "matched_static_correct": int(np.sum(matched == y)),
        "corrected_vs_matched": int(np.sum((out == y) & (matched != y))),
        "degraded_vs_matched": int(np.sum((out != y) & (matched == y))),
        "by_variant_correct": {
            v: int(np.sum((out == y) & np.array([r["variant"] == v for r in rows])))
            for v in CORE
        },
        "chosen_by_tree": choices,
        "by_tree_correct": {
            t: int(np.sum((out == y) & (tree == t))) for t in trees
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    rows = [prepare(json.loads(line)) for line in raw.decode("utf8").splitlines() if line.strip()]
    if len(rows) != 800 or len({(r["id"], r["variant"]) for r in rows}) != 800:
        raise ValueError("Expected exactly 800 unique Qwen seed-0 development records")
    ids = {r["id"] for r in rows}
    if len(ids) != 200 or any({r["variant"] for r in rows if r["id"] == i} != set(CORE) for i in ids):
        raise ValueError("Expected 200 patients with all four counterfactual versions")
    if len({r["tree"] for r in rows}) != 4:
        raise ValueError("Expected four development trees")
    result = {
        "scope": "retrospective exploratory offline reuse; no new GPU inference",
        "seed": 0, "model": "Qwen3-8B",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "trees": sorted({r["tree"] for r in rows}),
        "records": len(rows), "patients": len(ids),
        "grid_size_per_policy": len(GRID),
        "warning": "This does not evaluate genuinely new sparse heads or dynamic query-stage attention.",
        "results": {k: evaluate(rows, k) for k in POLICIES}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    for k, v in result["results"].items():
        print(k, v["accuracy"], v["correct"], v["corrected_vs_matched"], v["degraded_vs_matched"])


if __name__ == "__main__":
    main()
