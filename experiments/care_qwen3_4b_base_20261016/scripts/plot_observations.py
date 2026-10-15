#!/usr/bin/env python3
"""Plot only actual observation outputs. Never synthesize experimental results.

Expected files:
  results/obs1/predictions.csv
  results/obs2/key_embeddings.npz, layer_head_stats.csv, selection.json
  results/obs3/logits.csv
The real model runner must produce them. See CODEX_TASK.md for schemas.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SEED = 42


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def savefig(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} / {out.with_suffix('.svg')}")


def _bool_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)
    return x.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _cluster_bootstrap(
    df: pd.DataFrame, estimator: Callable[[pd.DataFrame], float],
    n_boot: int = 1000, seed: int = SEED,
) -> tuple[float, float, float]:
    val = float(estimator(df))
    groups = df["group_id"].astype(str).unique()
    if len(groups) < 2:
        return val, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_group = {g: df.loc[df["group_id"].astype(str) == g] for g in groups}
    boots = []
    for _ in range(n_boot):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([by_group[g] for g in selected], ignore_index=True)
        x = estimator(sampled)
        if np.isfinite(x):
            boots.append(float(x))
    if not boots:
        return val, float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return val, float(lo), float(hi)


def _acc(df: pd.DataFrame, subset: str, col: str = "correct") -> float:
    d = df.loc[df["subset"] == subset]
    return float(d[col].astype(float).mean()) if len(d) else float("nan")


def plot_obs1(folder: Path, dest: Path, bootstrap_reps: int = 1000) -> None:
    path = folder / "obs1" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    require_columns(df, ["group_id", "case_id", "method", "subset", "correct", "misapplied",
                         "prior_reversion", "prior_eligible", "split", "source_type"], str(path))
    df = df[(df["split"] == "test") & _bool_series(df["prior_eligible"])].copy()
    df = df[~df["source_type"].astype(str).str.contains("synthetic_logic", case=False)]
    if df.empty:
        raise ValueError("O1 has 0 clinically validated prior-conflict test samples. "
                         "Do NOT plot synthetic conditional logic as parametric-knowledge conflict.")
    df["subset"] = df["subset"].replace({"applicable": "app", "nonapplicable": "non"})
    methods = [str(x) for x in df["method"].unique()]
    if "baseline" not in methods:
        raise ValueError("O1 requires an actual baseline row for paired comparison")
    rng = np.random.default_rng(SEED)
    points = {}
    for method in methods:
        d = df[df["method"] == method]
        if not set(["app", "non"]).issubset(d["subset"].unique()):
            warnings.warn(f"Skipping {method}; lacks both app/non subsets")
            continue
        a = _cluster_bootstrap(d, lambda x:_acc(x, "app"), bootstrap_reps, int(rng.integers(1e8)))
        n = _cluster_bootstrap(d, lambda x:_acc(x, "non"), bootstrap_reps, int(rng.integers(1e8)))
        m = _cluster_bootstrap(d, lambda x:_acc(x, "non", "misapplied"), bootstrap_reps, int(rng.integers(1e8)))
        points[method] = {"app": a, "non": n, "misapplied": m,
                          "n": len(d), "n_groups": d["group_id"].nunique()}
    if "baseline" not in points:
        raise ValueError("No paired baseline after validation")
    fig, ax = plt.subplots(figsize=(6.7, 5.1), constrained_layout=True)
    for name, p in points.items():
        app, non = p["app"], p["non"]
        xe = [[app[0]-app[1] if np.isfinite(app[1]) else 0],
              [app[2]-app[0] if np.isfinite(app[2]) else 0]]
        ye = [[non[0]-non[1] if np.isfinite(non[1]) else 0],
              [non[2]-non[0] if np.isfinite(non[2]) else 0]]
        ax.errorbar(app[0], non[0], xerr=xe, yerr=ye, fmt="o", capsize=3,
                    label=f"{name} (n={p['n']})")
    ax.set(xlabel="Applicable-conflict accuracy", ylabel="Nonapplicable-conflict accuracy",
           title="O1 | Conditional faithfulness vs. rule misapplication",
           xlim=(-0.04, 1.04), ylim=(-0.04, 1.04))
    ax.grid(alpha=.18)
    ax.legend(loc="lower left", fontsize=8)
    savefig(dest / "O1_tradeoff")

    # Paired, rule-cluster bootstrap estimates of changes relative to baseline.
    b = df[df["method"] == "baseline"].copy()
    comparison = [name for name in points if name != "baseline"]
    if not comparison:
        return
    metric_defs = [("Δ app accuracy", "app", "correct"),
                   ("Δ nonapp accuracy", "non", "correct"),
                   ("Δ nonapp misapplication", "non", "misapplied")]
    fig, ax = plt.subplots(figsize=(8.2, max(3.5, len(comparison)*1.25+1.5)), constrained_layout=True)
    for i, method in enumerate(comparison):
        method_df = df[df["method"] == method]
        for j, (label, subset, col) in enumerate(metric_defs):
            curr = method_df[method_df["subset"] == subset]
            base = b[b["subset"] == subset]
            pairs = curr.merge(base[["group_id", "case_id", col]], on=["group_id", "case_id"],
                               suffixes=("", "_baseline"), how="inner")
            if len(pairs) == 0:
                continue
            delta = _cluster_bootstrap(pairs, lambda x: float((x[col].astype(float) - x[f"{col}_baseline"].astype(float)).mean()),
                                       bootstrap_reps, seed=SEED + 53*i+j)
            y = i*4+j
            err = [[delta[0]-delta[1] if np.isfinite(delta[1]) else 0],
                   [delta[2]-delta[0] if np.isfinite(delta[2]) else 0]]
            ax.errorbar(delta[0], y, xerr=err, fmt="o", capsize=2)
            ax.text(-.97, y, f"{method} | {label}", va="center", fontsize=9)
    ax.axvline(0, ls="--", lw=1)
    ax.set(xlabel="Change from baseline (paired bootstrap 95% CI)",
           title="O1 | Effects of highlighting on correct and incorrect rule application",
           xlim=(-1.0, 1.0), yticks=[])
    ax.set_ylim(len(comparison)*4-.1, -.7)
    ax.grid(alpha=.15, axis="x")
    savefig(dest / "O1_effects")


def plot_obs2(folder: Path, dest: Path) -> None:
    sub = folder / "obs2"
    npath, spath, choice_path = sub / "key_embeddings.npz", sub / "layer_head_stats.csv", sub / "selection.json"
    for p in [npath, spath, choice_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    stats = pd.read_csv(spath)
    require_columns(stats, ["layer", "kv_head", "rel_cons", "app_cons", "split"], str(spath))
    # Selection is frozen from dev BEFORE looking at test embeddings.
    choice = json.loads(choice_path.read_text())
    if choice.get("selected_on") != "dev":
        raise ValueError("O2 selected head must come from dev, not test")
    layer, head = int(choice["layer"]), int(choice["kv_head"])
    arr = np.load(npath, allow_pickle=False)
    for k in ["keys", "group_ids", "layer_ids", "states", "split"]:
        if k not in arr.files:
            raise ValueError(f"O2 npz is missing field {k}")
    keys = arr["keys"]
    states = [str(s) for s in arr["states"]]
    assert states == ["RA", "RN", "IA", "IN"], f"Expected stable state order RA/RN/IA/IN, got {states}"
    layer_ids = [int(v) for v in arr["layer_ids"]]
    if layer not in layer_ids:
        raise ValueError("Selected layer not in embeddings")
    lix = layer_ids.index(layer)
    if keys.ndim != 5 or keys.shape[1] != 4 or head >= keys.shape[3]:
        raise ValueError("Expected keys shape [N,4,L,H,D]")
    mask = arr["split"].astype(str) == "test"
    X = keys[mask, :, lix, head, :].astype(np.float64)
    if X.shape[0] < 3:
        raise ValueError("Not enough independent test groups for O2 PCA")
    points = PCA(n_components=2, random_state=SEED).fit_transform(X.reshape(-1, X.shape[-1])).reshape(-1,4,2)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True, constrained_layout=True)
    designs = [("Relevance shift (irrelevant → relevant)", [(2,0), (3,1)]),
               ("Applicability shift (nonapplicable → applicable)", [(1,0), (3,2)])]
    rng = np.random.default_rng(SEED)
    selected = rng.choice(len(points), size=min(len(points), 50), replace=False)
    for ax, (title, pairs) in zip(axes, designs):
        all_starts, all_ends = [], []
        for j in selected:
            for s,e in pairs:
                a, b = points[j,s], points[j,e]
                ax.plot([a[0],b[0]], [a[1],b[1]], alpha=.22, lw=.8)
                ax.scatter([a[0]], [a[1]], s=11, alpha=.48, marker="o")
                ax.scatter([b[0]], [b[1]], s=11, alpha=.48, marker="x")
                all_starts.append(a); all_ends.append(b)
        aa = np.stack(all_starts).mean(axis=0)
        bb = np.stack(all_ends).mean(axis=0)
        ax.annotate("", xy=bb, xytext=aa, arrowprops={"arrowstyle":"->", "lw":2.4})
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.14)
    fig.suptitle(f"O2 | Paired Key shifts (layer {layer}, KV head {head}; {len(X)} held-out groups)")
    savefig(dest / "O2_pca_pairwise")

    test_stats = stats[stats["split"] == "test"].copy()
    if test_stats.empty:
        raise ValueError("O2 layer/head stats must include held-out test results")
    layers = sorted(test_stats["layer"].unique())
    heads = sorted(test_stats["kv_head"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, key, title in [(axes[0], "rel_cons", "Relevance-shift consistency"),
                           (axes[1], "app_cons", "Applicability-shift consistency")]:
        matrix = test_stats.pivot(index="layer", columns="kv_head", values=key).reindex(index=layers, columns=heads).values
        im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1)
        ax.set(title=title, xlabel="KV head", ylabel="Layer",
               xticks=range(len(heads)), xticklabels=heads,
               yticks=range(len(layers)), yticklabels=layers)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("O2 | Head-wise representational shifts on held-out rules")
    savefig(dest / "O2_layer_head_heatmaps")


def plot_obs3(folder: Path, dest: Path, tolerance: float = 1e-4, zero_eps: float = 1e-6) -> None:
    path = folder / "obs3" / "logits.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    require_columns(df, ["group_id", "case_id", "layer", "kv_head", "query_head",
                         "token_idx", "omega", "z", "a_before", "a_after",
                         "delta_a", "lambda", "split", "model"], str(path))
    df = df[(df["split"] == "test") & df["omega"].ne(0)].copy()
    if df.empty:
        raise ValueError("No actual nonzero-weight O3 test records")
    # Robust finite checks and numerical validity: prevent mislabeled formulas or fake results.
    fields = ["omega", "z", "a_before", "a_after", "delta_a", "lambda"]
    if not np.isfinite(df[fields].to_numpy(dtype=float)).all():
        raise ValueError("O3 contains NaN/inf in required numeric data")
    expected = df["lambda"].to_numpy() * df["omega"].to_numpy() * df["z"].to_numpy()
    derived = df["a_after"].to_numpy() - df["a_before"].to_numpy()
    observed = df["delta_a"].to_numpy()
    max_err = max(float(np.max(np.abs(observed-derived))), float(np.max(np.abs(observed-expected))))
    if max_err > tolerance:
        raise ValueError(f"O3 local QK identity failed! max error={max_err:.5g} > {tolerance:.5g}; "
                         "check RoPE/QK-norm/GQA and whether query was held fixed")
    df = df.loc[np.abs(df["z"]) > zero_eps].copy()
    if df.empty:
        raise ValueError("All O3 z values are near-zero; no direction statistics defined")
    mismatch = float((df["z"] < 0).mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.hist(df.loc[df["omega"] > 0, "z"], bins=35, alpha=.6, label="Positive target weights")
    ax.hist(df.loc[df["omega"] < 0, "z"], bins=35, alpha=.5, label="Negative target weights")
    ax.axvline(0, ls="--", lw=1.2)
    ax.set(xlabel="z = query · projected key / sqrt(d)", ylabel="Token-head observations",
           title=f"O3 | Query–projection alignment (z<0: {mismatch:.1%}, n={len(df)})")
    ax.legend(fontsize=9)
    savefig(dest / "O3_z_histogram")

    sample = df.sample(n=min(len(df), 7000), random_state=SEED)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for label, filtered in [("z≥0 (direction matched)", sample[sample["z"] >= 0]),
                            ("z<0 (direction reversed)", sample[sample["z"] < 0])]:
        ax.scatter(filtered["omega"], filtered["delta_a"], s=10, alpha=.3, label=label)
    ax.axhline(0, ls="--", lw=.8)
    ax.axvline(0, ls="--", lw=.8)
    ax.set(xlabel="Signed steering weight ω", ylabel="Actual local attention-logit change Δa",
           title="O3 | Direct signed SEKA extension: desired vs actual direction")
    ax.legend(fontsize=9)
    savefig(dest / "O3_weight_delta_scatter")

    d = df.assign(mismatch=(df["z"] < 0).astype(float))
    table = d.pivot_table(index="layer", columns="kv_head", values="mismatch", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(table.values, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set(xlabel="KV head", ylabel="Layer", title="O3 | Direction-mismatch rate by layer and KV head",
           xticks=range(len(table.columns)), xticklabels=table.columns.tolist(),
           yticks=range(len(table.index)), yticklabels=table.index.tolist())
    fig.colorbar(im, ax=ax, label="Fraction z<0")
    savefig(dest / "O3_mismatch_heatmap")
    print(f"O3 observed mismatch z<0 = {mismatch:.4f}; max algebraic error={max_err:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--obs", default="1,2,3", help="comma-separated observations, default 1,2,3")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--strict", action="store_true", help="nonzero exit if any requested observation is missing")
    args = parser.parse_args()
    failed = []
    for x in args.obs.split(","):
        try:
            if x == "1": plot_obs1(args.results, args.out, args.bootstrap_reps)
            elif x == "2": plot_obs2(args.results, args.out)
            elif x == "3": plot_obs3(args.results, args.out)
            else: raise ValueError(f"Unsupported observation: {x}")
        except (ValueError, FileNotFoundError, KeyError, AssertionError) as e:
            print(f"[NOT PLOTTED] O{x}: {type(e).__name__}: {e}")
            failed.append(x)
    if failed and args.strict:
        raise SystemExit(f"Cannot plot observations without verified data: {', '.join(failed)}")
    if failed:
        print("No results were invented. Missing or invalid observations must be run by Codex.")


if __name__ == "__main__":
    main()
