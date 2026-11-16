"""CPU-only Pareto/knee analysis of the immutable PR #76 result tables.

No fitting, prediction, bootstrap resampling, or model imports occur here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

# Set before importing any numerical or plotting library.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

X = "cf_normalized_distortion"
Y = "ap_vr"
METRICS = (Y, X, "cf_er", "cf_correction_rms",
           "natural_path_accuracy", "interference_path_accuracy")
SOURCE_REL = "reports/aster_m3_dual_objective_20261116"
SOURCE_URL = "https://github.com/chunxi-alpc/TRIAGE/tree/codex/aster-m3-dual-objective-20261116/" + SOURCE_REL
LABELS = {
    "knee_A": "Knee A", "knee_B": "Knee B",
    "auto_eta0": "自动选择", "old_E1": "旧版 E1",
    "conservative": "保守端点", "aggressive": "激进端点",
}
METRIC_LABELS = {
    Y: "A/P VR", X: "C/F 扰动", "cf_er": "C/F ER",
    "cf_correction_rms": "C/F 修正 RMS",
    "natural_path_accuracy": "自然路径准确率",
    "interference_path_accuracy": "干扰路径准确率",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def pareto_front(rows):
    """Strict dominance: minimize X, maximize Y; retain exact duplicates."""
    if not rows:
        raise ValueError("Empty grid.")
    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in (X, Y)):
            raise ValueError("Non-finite Pareto coordinate.")
    front = [dict(a) for a in rows if not any(
        b[X] <= a[X] and b[Y] >= a[Y]
        and (b[X] < a[X] or b[Y] > a[Y]) for b in rows)]
    return sorted(front, key=lambda r: (r[X], r[Y], str(r["id"])))


def tie_key(row):
    """Only resolve numerical ties: lower distortion, higher VR, then ID."""
    return (row[X], -row[Y], str(row["id"]))


def select_knees(front):
    """Both methods normalize on the same OOF Pareto front, never on E1."""
    scored = [dict(row) for row in front]
    xlo, xhi = min(r[X] for r in front), max(r[X] for r in front)
    ylo, yhi = min(r[Y] for r in front), max(r[Y] for r in front)
    for row in scored:
        row["normalized_cf_distortion"] = (row[X] - xlo) / (xhi - xlo) if xhi > xlo else 0.
        row["normalized_ap_vr"] = (row[Y] - ylo) / (yhi - ylo) if yhi > ylo else 0.
        row["balance_score_B"] = row["normalized_ap_vr"] - row["normalized_cf_distortion"]
    p, q = scored[0], scored[-1]
    px, py = p["normalized_cf_distortion"], p["normalized_ap_vr"]
    dx = q["normalized_cf_distortion"] - px
    dy = q["normalized_ap_vr"] - py
    chord_length = math.hypot(dx, dy)
    for row in scored:
        rx = row["normalized_cf_distortion"] - px
        ry = row["normalized_ap_vr"] - py
        row["distance_A"] = abs(dx * ry - dy * rx) / chord_length if chord_length else 0.
    distinct = len({(r[X], r[Y]) for r in scored})
    informative = (distinct >= 3 and xhi > xlo and yhi > ylo
                   and max(r["distance_A"] for r in scored) > 1e-12)
    result = {
        "selection_scope": "80-source five-fold OOF",
        "normalization_scope": "recomputed OOF Pareto front only",
        "dev_test_used_for_selection": False,
        "normalization_bounds": {X: [xlo, xhi], Y: [ylo, yhi]},
        "informative_knee": informative,
        "method_A": None, "method_B": None,
        "methods_agree": None,
        "all_on_or_above_chord": all(r["balance_score_B"] >= -1e-12 for r in scored),
        "tie_rule": "within rtol=1e-12/atol=1e-14: lower distortion, higher VR, then ID",
    }
    if informative:
        for name, metric in (("method_A", "distance_A"), ("method_B", "balance_score_B")):
            peak = max(r[metric] for r in scored)
            tied = [r for r in scored if math.isclose(r[metric], peak, rel_tol=1e-12, abs_tol=1e-14)]
            winner = min(tied, key=tie_key)
            result[name] = {k: winner[k] for k in
                            ("id", "eta", "regularization", X, Y, "distance_A", "balance_score_B")
                            if k in winner}
            result[name]["tied_ids"] = [r["id"] for r in tied]
        result["methods_agree"] = result["method_A"]["id"] == result["method_B"]["id"]
    return scored, result


def analyze_grid(rows):
    # Selection function accepts only the OOF grid, with no evaluation input.
    return select_knees(pareto_front(rows))


def collect_key_points(rows, front, selection, prior_selection, evaluations):
    by_id = {r["id"]: r for r in rows}
    auto_id = prior_selection["dual"]["id"]
    roles = [("knee_A", selection["method_A"]["id"]),
             ("knee_B", selection["method_B"]["id"]),
             ("auto_eta0", auto_id), ("old_E1", "old_E1"),
             ("conservative", front[0]["id"]), ("aggressive", front[-1]["id"])]
    key_rows, evaluation_rows = [], []
    for role, candidate in roles:
        oof = evaluations["oof_selection"]["old_E1"] if candidate == "old_E1" else by_id[candidate]
        meta = {
            "role": role, "candidate_id": candidate,
            "eta": oof.get("eta"), "regularization": oof.get("regularization"),
        }
        key = dict(meta, scope="oof_selection", status="evaluated",
                   **{metric: oof[metric] for metric in METRICS})
        for scope in ("oof_selection", "dev", "test"):
            if scope == "oof_selection":
                evaluated = oof
                origin = (f"evaluation/{scope}/metrics.csv:old_E1"
                          if candidate == "old_E1" else f"oof_grid_70.csv:{candidate}")
            else:
                # PR76 evaluates only selected dual/ap_only and the fixed old E1.
                method = "old_E1" if candidate == "old_E1" else "dual" if candidate == auto_id else None
                evaluated = evaluations[scope].get(method) if method else None
                origin = f"evaluation/{scope}/metrics.csv:{method}" if evaluated else None
            status = "evaluated" if evaluated else "not_evaluated"
            item = dict(meta, scope=scope, status=status,
                        reason=None if evaluated else "No existing PR76 evaluation for this configuration",
                        source_file=origin,
                        **{metric: evaluated[metric] if evaluated else None for metric in METRICS})
            evaluation_rows.append(item)
            if scope != "oof_selection":
                key[scope + "_status"] = status
        key_rows.append(key)
    return key_rows, evaluation_rows


def reused_bootstrap(source, key_rows, prior_selection):
    """Copy existing marginal intervals; do not derive paired knee intervals."""
    grid_ci = {(r["candidate"], r["metric"]): r
               for r in read_csv(source / "grid_bootstrap.csv")}
    scope_ci = {
        scope: {(r["comparison"], r["metric"]): r
                for r in read_csv(source / f"evaluation/{scope}/bootstrap.csv")}
        for scope in ("oof_selection", "dev", "test")
    }
    output = []
    for key in key_rows:
        cid = key["candidate_id"]
        for scope in scope_ci:
            for metric in METRICS:
                if scope == "oof_selection" and cid != "old_E1":
                    row = grid_ci.get((cid, metric))
                    origin = "grid_bootstrap.csv"
                else:
                    method = "old_E1" if cid == "old_E1" else (
                        "dual" if cid == prior_selection["dual"]["id"] else None)
                    row = scope_ci[scope].get((method, metric))
                    origin = f"evaluation/{scope}/bootstrap.csv"
                output.append(dict(
                    role=key["role"], candidate_id=cid, scope=scope, metric=metric,
                    status="available" if row else "not_evaluated",
                    **{k: row.get(k) if row else None for k in
                       ("estimate", "low", "high", "defined_draws")},
                    source_file=origin if row else None))
    return output


def protected_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the PR76 source directory.")
    return output


def make_figures(rows, front, selection, keys, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    by_role = {r["role"]: r for r in keys}
    knee = by_role["knee_A"]
    auto, old = by_role["auto_eta0"], by_role["old_E1"]
    fx, fy = [r[X] for r in front], [r[Y] for r in front]
    red, blue, green = "#B54745", "#456E9F", "#428778"

    def style(ax):
        ax.set_xlabel("C/F normalized distortion (lower is better)")
        ax.set_ylabel("A/P variance reduction (higher is better)")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=.16, linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(r[X] for r in rows) * 1.10)
        ax.set_ylim(0, max(r[Y] for r in rows) * 1.14)

    def save(fig, stem):
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=400, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
    ax.scatter([r[X] for r in rows], [r[Y] for r in rows], color="#B5BAC3",
               s=23, alpha=.7, label=f"All {len(rows)} configurations", zorder=2)
    ax.plot(fx, fy, "-o", color="#5B616B", linewidth=1.1, markersize=2.8,
            label=f"Pareto front ({len(front)} configurations)", zorder=3)
    ax.scatter(knee[X], knee[Y], marker="*", color=red, s=180, edgecolor="white",
               linewidth=.6, label="Knee A = B", zorder=5)
    ax.scatter(auto[X], auto[Y], marker="D", color=blue, s=47,
               label="Automatic selection / aggressive end", zorder=5)
    ax.scatter(old[X], old[Y], marker="s", color=green, s=48, label="Old hard-constraint E1", zorder=5)
    ax.annotate("Knee: eta = 1, lambda = 100\nVR 10.18%; distortion 1.41%",
                (knee[X], knee[Y]), xytext=(.0100, .045),
                arrowprops=dict(arrowstyle="-", color=red, lw=.9), color=red, fontsize=8)
    ax.annotate("eta = 0, lambda = 100", (auto[X], auto[Y]),
                xytext=(auto[X]-.0007, auto[Y]-.035), ha="right", color=blue,
                arrowprops=dict(arrowstyle="-", color=blue, lw=.7))
    ax.annotate("Old E1", (old[X], old[Y]), xytext=(12, -7),
                textcoords="offset points", color=green)
    ax.annotate("Conservative end", (fx[0], fy[0]), xytext=(12, -6),
                textcoords="offset points", color="#5B616B", fontsize=7)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("M3 Pareto trade-off and the OOF knee", loc="left", fontweight="bold")
    style(ax)
    fig.supxlabel("80-source five-fold OOF; 10 eta values x 7 lambda values. Nearby lambda settings overlap.",
                  fontsize=7)
    save(fig, "FIGURE_pareto_knee")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    etas = sorted({r["eta"] for r in rows})
    colors = plt.get_cmap("tab10")
    markers = ["o", "v", "^", "<", ">", "s", "p", "D", "h", "X"]
    for i, eta in enumerate(etas):
        group = [r for r in rows if r["eta"] == eta]
        sizes = [22 + 7 * (math.log10(r["regularization"]) + 4) for r in group]
        ax.scatter([r[X] for r in group], [r[Y] for r in group], s=sizes,
                   marker=markers[i], facecolors="none", edgecolors=colors(i),
                   linewidths=.9, label=f"eta = {eta:g}", zorder=3)
    ax.plot(fx, fy, color="#A0A4AA", linewidth=.8, zorder=1)
    ax.scatter(knee[X], knee[Y], marker="*", s=160, color=red, edgecolor="white",
               linewidth=.6, zorder=5)
    ax.annotate("Knee", (knee[X], knee[Y]), xytext=(10, -18),
                textcoords="offset points", color=red)
    ax.legend(loc="upper left", ncols=2, fontsize=7, title="C/F penalty weight", title_fontsize=8)
    ax.set_title("The eta grid traces the benefit-cost trade-off", loc="left", fontweight="bold")
    style(ax)
    fig.supxlabel("All 70 OOF configurations; marker size increases with lambda (0.0001 to 100).\n"
                  "Within each eta group, lambda settings are closely spaced and overlap.", fontsize=7)
    save(fig, "FIGURE_pareto_by_eta")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    nx = [r["normalized_cf_distortion"] for r in front]
    ny = [r["normalized_ap_vr"] for r in front]
    scored_knee = next(r for r in front if r["id"] == knee["candidate_id"])
    kx, ky = scored_knee["normalized_cf_distortion"], scored_knee["normalized_ap_vr"]
    ax = axes[0]
    ax.plot(nx, ny, "-o", color="#5B616B", markersize=2.5, linewidth=.9)
    ax.plot([0, 1], [0, 1], "--", color="#92969D", linewidth=.9, label="Endpoint chord")
    foot = (kx + ky) / 2
    ax.plot([kx, foot], [ky, foot], color=red, linewidth=1.5, label="Perpendicular distance")
    ax.scatter(kx, ky, marker="*", color=red, s=130, zorder=5)
    ax.text(.05, .9, f"Knee A = B\neta = 1, lambda = 100\nd = {scored_knee['distance_A']:.4f}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=red)
    ax.set(xlabel="Normalized C/F distortion", ylabel="Normalized A/P VR",
           title="a  Distance to the endpoint chord", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[1]
    ax.plot(nx, [r["distance_A"] for r in front], "-o", color=blue, markersize=2.5,
            linewidth=1., label="Method A: distance")
    ax.plot(nx, [r["balance_score_B"] for r in front], "-s", color=red, markersize=2,
            linewidth=1., label="Method B: balance score")
    ax.axvline(kx, color="#8A8D93", linestyle=":", linewidth=.8)
    ax.scatter([kx, kx], [scored_knee["distance_A"], scored_knee["balance_score_B"]],
               marker="*", s=90, color=[blue, red], zorder=5)
    ax.set(xlabel="Normalized C/F distortion", ylabel="Selection score",
           title="b  Both scores peak at the same point")
    ax.text(.50, .23, "A = B / sqrt(2)\non this front", ha="center",
            transform=ax.transAxes, fontsize=7.5)
    ax.legend(loc="upper right", fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=.15, linewidth=.6)
        ax.set_axisbelow(True)
    fig.supxlabel("Min-Max normalization uses only the 59 OOF Pareto points; E1 is a reference point.",
                  fontsize=7)
    save(fig, "FIGURE_knee_methods")


def fmt(value, metric):
    if value is None:
        return "未评价"
    if metric in (Y, X, "natural_path_accuracy", "interference_path_accuracy"):
        return f"{100 * value:.3f}%"
    return f"{value:.6f}" if metric == "cf_er" else f"{value:.4f}"


def make_report(source, output, rows, front, selection, keys, evaluations, cis):
    k = next(r for r in keys if r["role"] == "knee_A")
    auto = next(r for r in keys if r["role"] == "auto_eta0")
    old = next(r for r in keys if r["role"] == "old_E1")
    ranking = sorted(front, key=lambda r: r["distance_A"], reverse=True)
    winner, runner = ranking[:2]
    ci_lookup = {(r["role"], r["scope"], r["metric"]): r for r in cis}
    lines = [
        "# 新版 M3 Pareto 曲线与拐点分析",
        "",
        f"**80 来源折外结果呈现明确的收益递减趋势，拐点集中在 eta=1。两种方法均选中 "
        f"eta={k['eta']:g}、lambda={k['regularization']:g}（{k['candidate_id']}）："
        f"A/P VR 为 {fmt(k[Y], Y)}，C/F 扰动为 {fmt(k[X], X)}。**"
        f"相对当前 eta=0 配置，C/F 扰动减少 {100*(1-k[X]/auto[X]):.2f}%，"
        f"保留 {100*k[Y]/auto[Y]:.2f}% 的 A/P 消减收益。",
        "",
        "## 1. 数据、评价与选点范围",
        "",
        f"直接读取 [PR #76 结果目录]({SOURCE_URL}) 中的 oof_grid_70.csv、pareto_front.csv、"
        "selection.json、grid_bootstrap.csv，以及 evaluation/{oof_selection,dev,test}/"
        "metrics.csv 和 bootstrap.csv。来源提交为 2f1849037223fb785056c19dab364043c0d9738c。",
        "",
        "70 组配置覆盖 10 个 eta 与 7 个 lambda；每组使用 80 来源五折折外结果，"
        "汇总 5,003 个 A/P 对、1,342 个 C/F 对、1,712 条自然路径和 5,136 条干扰路径。"
        "Pareto 和拐点计算函数只接收该折外网格。选点完成后读取旧 E1、20 来源开发集与"
        "40 来源复用留出集结果，补充对照表。",
        "",
        "沿用 PR #76 的固定 51 头 M1、16 维 M2、E1 自然节点加性截距校准和 beta=1 评价协议。"
        "本次使用 1 个 CPU 线程；GPU、显存占用、训练、重新拟合、模型前向和新增 Bootstrap 抽样均为 0。"
        "所有新增代码和结果位于独立目录，PR #76 原文件和主稿保持原样。",
        "",
        "横轴 D = C/F 修正 RMS / C/F 原始响应 RMS，越小越好；"
        "纵轴 VR = 1 − Var(A/P 修正后差值) / Var(A/P 原始差值)，越大越好。"
        "C/F ER 是按标签方向对齐后的平均响应之比。路径指标沿用现有自然路径和干扰路径准确率。",
        "",
        "## 2. Pareto 前沿与拐点",
        "",
        f"按“扰动不增、VR 不降，且至少一项严格改善”的支配定义重算，"
        f"70 个配置中有 **{len(front)} 个 Pareto 配置**；与原有 pareto_front.csv 的成员集合一致。"
        "旧 E1 作为外部参考点展示，保持 70 点网格独立用于前沿和归一化。",
        "",
        "方法 A 对前沿两坐标分别 Min-Max 归一化，再按扰动升序连接两端点，"
        "计算点到端点连线的垂直距离。一般式为 "
        "d = |(x₂−x₁)(y−y₁) − (y₂−y₁)(x−x₁)| / sqrt((x₂−x₁)²+(y₂−y₁)²)。"
        "方法 B 在同一归一化前沿上最大化 S = y − x。",
        "",
        f"**方法 A：{selection['method_A']['id']}，eta=1、lambda=100，d={winner['distance_A']:.8f}。**  "
        f"\n**方法 B：{selection['method_B']['id']}，eta=1、lambda=100，S={winner['balance_score_B']:.8f}。**",
        "",
        "两种方法结果一致。本前沿归一化后的端点为 (0,0) 与 (1,1)，"
        "所有中间点均位于端点连线上方，因此 d = S / sqrt(2)，两种排序在该前沿上数学等价。"
        "如果某个前沿存在连线下方且绝对偏离更大的点，方法 A 的绝对距离与方法 B 的有向得分可能选出不同点。",
        "",
        f"前两名分别是 eta=1、lambda=100 与 eta=1、lambda=10；"
        f"距离得分相差 {winner['distance_A']-runner['distance_A']:.8f}"
        f"（最大得分的 {100*(winner['distance_A']-runner['distance_A'])/winner['distance_A']:.3f}%）。"
        "eta=1 的七个 lambda 占据距离排名前七名。因而曲线给出了清晰的 eta≈1 平衡区域，"
        "lambda=100 是已测网格中的数值最优点；相邻 lambda 的实际差异很小。",
        "",
        "|eta（lambda=100）|A/P VR|C/F 扰动|平衡得分 S|",
        "|---:|---:|---:|---:|",
    ]
    for eta in (3, 1, .3, 0):
        r = next(r for r in front if r["eta"] == eta and r["regularization"] == 100)
        lines.append(f"|{eta:g}|{fmt(r[Y],Y)}|{fmt(r[X],X)}|{r['balance_score_B']:.6f}|")
    points = {r["eta"]: r for r in rows if r["regularization"] == 100}
    left, right = points[3.], points[.3]
    before = (k[Y]-left[Y])/(k[X]-left[X])
    after = (right[Y]-k[Y])/(right[X]-k[X])
    tail = (auto[Y]-right[Y])/(auto[X]-right[X])
    lines += [
        "",
        f"沿扰动增大的方向，eta=3→1 每增加 1 个百分点 C/F 扰动，获得约 {before:.2f} 个百分点 VR；"
        f"eta=1→0.3 时该收益降至 {after:.2f}，eta=0.3→0 时降至 {tail:.2f}。"
        "这解释了 eta=1 附近的收益递减。",
        "",
        "![Pareto 前沿和拐点](FIGURE_pareto_knee.png)",
        "",
        "图 1：全部 70 个配置、重算前沿、两种方法共同拐点、当前自动选择点和旧 E1。"
        "同一 eta 下多个 lambda 的坐标接近，散点存在重叠；连线只连接已测配置。",
        "",
        "![eta 参数分布](FIGURE_pareto_by_eta.png)",
        "",
        "图 2：颜色与形状区分 eta，点大小随 lambda 增加，展示完整参数网格。",
        "",
        "![两种拐点方法](FIGURE_knee_methods.png)",
        "",
        "图 3：左图展示归一化前沿与垂直距离，右图比较两种得分。所有坐标来自 80 来源折外结果。",
        "",
        "## 3. 关键配置对照",
        "",
        "以下为同一 80 来源上的现有结果。CSV 保留原始比例值；表中 VR、扰动和准确率显示为百分数。"
        "方法 A/B 为同一配置，自动选择点也正好是最激进端点。",
        "",
        "|配置|eta|lambda|A/P VR|C/F 扰动|C/F ER|C/F 修正 RMS|自然路径准确率|干扰路径准确率|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in keys:
        eta = f"{row['eta']:g}" if row["eta"] is not None else "—"
        lam = f"{row['regularization']:g}" if row["regularization"] is not None else "—"
        lines.append("|" + "|".join([LABELS[row["role"]], eta, lam] +
                                   [fmt(row[m], m) for m in METRICS]) + "|")
    lines += [
        "",
        "### 相比当前 eta=0",
        "",
        f"Knee 的 C/F 扰动从 {fmt(auto[X],X)} 降至 {fmt(k[X],X)}，"
        f"减少 {100*(auto[X]-k[X]):.3f} 个百分点（{100*(1-k[X]/auto[X]):.2f}%）；"
        f"C/F 修正 RMS 从 {auto['cf_correction_rms']:.4f} 降至 {k['cf_correction_rms']:.4f}。"
        f"A/P VR 从 {fmt(auto[Y],Y)} 变为 {fmt(k[Y],Y)}，相差 {100*(auto[Y]-k[Y]):.3f} 个百分点，"
        f"保留 {100*k[Y]/auto[Y]:.2f}% 的收益。"
        f"自然路径准确率增加 {100*(k['natural_path_accuracy']-auto['natural_path_accuracy']):.3f} 个百分点，"
        f"干扰路径准确率下降 {100*(auto['interference_path_accuracy']-k['interference_path_accuracy']):.3f} 个百分点。",
        "",
        "### 相比旧版硬约束 E1",
        "",
        f"Knee 将 A/P VR 从 {fmt(old[Y],Y)} 提升到 {fmt(k[Y],Y)}，"
        f"增加 {100*(k[Y]-old[Y]):.3f} 个百分点，达到旧 E1 的 {k[Y]/old[Y]:.2f} 倍。"
        f"对应 C/F 扰动从 {fmt(old[X],X)} 增至 {fmt(k[X],X)}"
        f"（增加 {100*(k[X]-old[X]):.3f} 个百分点），"
        f"C/F 修正 RMS 从 {old['cf_correction_rms']:.4f} 增至 {k['cf_correction_rms']:.4f}，"
        f"ER 从 {old['cf_er']:.6f} 变为 {k['cf_er']:.6f}。"
        f"自然路径和干扰路径准确率分别增加 "
        f"{100*(k['natural_path_accuracy']-old['natural_path_accuracy']):.3f}、"
        f"{100*(k['interference_path_accuracy']-old['interference_path_accuracy']):.3f} 个百分点。"
        "旧 E1 对应更小的响应扰动，Knee 对应更强的干扰消减。",
        "",
        "## 4. 复用 Bootstrap 与现有开发集、留出集评价",
        "",
        "直接复用 PR #76 的 2,000 次来源级 Bootstrap 边际 95% 区间。"
        "下表为选中配置的既有区间；拐点位置及新配置间差值的区间尚未评价。"
        "边际区间用于描述单配置指标的不确定性，选点按折外点估计完成。",
        "",
        "|Knee 指标|点估计|既有 95% 区间|",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        ci = ci_lookup[("knee_A", "oof_selection", metric)]
        lines.append(f"|{METRIC_LABELS[metric]}|{fmt(ci['estimate'],metric)}|"
                     f"[{fmt(ci['low'],metric)}, {fmt(ci['high'],metric)}]|")
    lines += [
        "",
        "Knee A/B（eta=1、lambda=100）和保守端点缺少现成的开发集与留出集评价，"
        "相应指标全部标记“未评价”。旧 E1 和 eta=0 使用 PR #76 既有结果；"
        "这些数据在拐点选择之后加入对照表。",
        "",
        "|集合|配置|A/P VR|C/F 扰动|C/F ER|C/F 修正 RMS|自然路径准确率|干扰路径准确率|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("dev", "test"):
        for role in ("knee_A", "auto_eta0", "old_E1"):
            r = next(r for r in evaluations if r["scope"] == scope and r["role"] == role)
            name = "开发集（20 来源）" if scope == "dev" else "复用留出集（40 来源）"
            lines.append("|" + "|".join([name, LABELS[role]] + [fmt(r[m],m) for m in METRICS]) + "|")
    lines += [
        "",
        "## 5. 论文推荐配置",
        "",
        "**建议论文以 eta=1、lambda=100 作为平衡 A/P 消减与 C/F 响应保持的推荐参数。**"
        "核心结果是：在 80 来源折外评价中，约 1.41% 的 C/F 响应扰动对应 10.18% 的 A/P 方差降低；"
        "相对 eta=0，扰动减半并保留约七成消减收益。"
        "这一推荐以折外权衡为依据，开发集和 40 来源留出集的 Knee 指标标记为未评价。"
        "eta=0 保留为最大 A/P 消减对照，旧 E1 保留为低响应扰动对照。",
        "",
        "## 6. 交付与复现",
        "",
        "- 独立脚本与单元测试：experiments/aster_m3_pareto_knee_20261116/。",
        "- 三幅主图：PNG，同时提供可编辑 SVG 和 PDF。",
        "- pareto_grid_annotated.csv：70 组原始结果及重算前沿标记。",
        "- pareto_front_recomputed.csv：59 个前沿配置、归一化坐标和两种得分。",
        "- pareto_key_points.csv：六个指定关键配置在折外集的指标和开发/留出评价状态。",
        "- key_points_evaluation.csv：六个关键配置 × 三个集合；未评价指标为空。",
        "- bootstrap_key_points.csv：直接摘录的既有区间及缺失状态。",
        "- knee_selection.json、configuration.json：选点结果、分析配置和数据来源。",
        "- logs/：CPU 执行和单元测试记录。复现命令见独立脚本目录 README.md。",
        "",
    ]
    (output / "PARETO_REPORT.zh.md").write_text("\n".join(lines), encoding="utf-8")


def run(source, output):
    source = Path(source).resolve()
    output = protected_output(source, output)
    rows = read_csv(source / "oof_grid_70.csv")
    if len(rows) != 70 or any(r["ap_sources"] != 80 or r["cf_sources"] != 80 for r in rows):
        raise ValueError("Expected the specified 70 configurations evaluated on 80 OOF sources.")
    front, selection = analyze_grid(rows)
    if not selection["informative_knee"]:
        raise ValueError("This grid has no informative interior knee.")
    # Freeze the knee before loading any dev/test result or legacy reference.
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "knee_selection.json", selection)
    prior_selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    prior_front = read_csv(source / "pareto_front.csv")
    selection["prior_pareto_membership_matches"] = (
        {r["id"] for r in prior_front} == {r["id"] for r in front})
    if not selection["prior_pareto_membership_matches"]:
        raise ValueError("Recomputed front differs from PR76: inspect the input version.")
    evaluations = {
        scope: {r["method"]: r for r in read_csv(source / f"evaluation/{scope}/metrics.csv")}
        for scope in ("oof_selection", "dev", "test")
    }
    keys, evaluated = collect_key_points(rows, front, selection, prior_selection, evaluations)
    cis = reused_bootstrap(source, keys, prior_selection)
    front_by_id = {r["id"]: r for r in front}
    annotated = []
    for row in rows:
        out = dict(row)
        out["prior_pareto"] = out.pop("pareto", None)
        out.update(front_by_id.get(row["id"], {}))
        out.pop("pareto", None)
        out["pareto_recomputed"] = row["id"] in front_by_id
        annotated.append(out)
    write_csv(output / "pareto_grid_annotated.csv", annotated)
    write_csv(output / "pareto_front_recomputed.csv", front)
    write_csv(output / "pareto_key_points.csv", keys)
    write_csv(output / "key_points_evaluation.csv", evaluated)
    write_csv(output / "bootstrap_key_points.csv", cis)
    write_json(output / "knee_selection.json", selection)
    write_json(output / "configuration.json", {
        "source_pr": 76, "source_ref": "2f1849037223fb785056c19dab364043c0d9738c",
        "source_report_path": SOURCE_REL, "source_url": SOURCE_URL,
        "grid_size": len(rows), "pareto_size": len(front),
        "selection_scope": "80-source five-fold OOF", "evaluation_only_scopes": ["dev", "test"],
        "axis_x": X, "axis_y": Y, "normalization": "Pareto-front Min-Max",
        "method_A": "maximum absolute perpendicular distance to endpoint chord",
        "method_B": "maximum normalized(ap_vr) - normalized(cf_normalized_distortion)",
        "CPU_threads": 1, "GPUs": [], "GPU_memory_GiB": 0,
        "training_runs": 0, "M3_refits": 0, "model_forwards": 0, "new_bootstrap_draws": 0,
        "existing_bootstrap_draws": 2000,
        "figure_contract": {
            "claim": "eta=1 provides the OOF trade-off knee",
            "panels": ["all-grid Pareto and references", "eta parameter distribution", "method equivalence"],
            "formats": ["PNG 400 dpi", "PDF", "SVG editable text"],
            "statistics": "point estimates; existing marginal source-bootstrap intervals reported separately",
        },
    })
    make_figures(rows, front, selection, keys, output)
    make_report(source, output, rows, front, selection, keys, evaluated, cis)
    print(json.dumps({"configs": len(rows), "pareto": len(front),
                      "knee_A": selection["method_A"]["id"],
                      "knee_B": selection["method_B"]["id"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/aster_m3_pareto_knee_20261116"))
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)
