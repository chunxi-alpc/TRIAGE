"""Render the matched-cost report from saved real analysis results only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/aster_cancel_story_20261109/meeting/matched_cost_feature_analysis"
def read(name): return list(csv.DictReader((OUT/name).open()))
def render():
    summaries=read("summary.csv");front=read("pareto.csv");paths=read("path_summary.csv")
    comparisons=read("paired_comparisons.csv"); intervals=read("bootstrap_intervals.csv")
    random=read("random_distribution.csv")
    def summary(split,name): return next(r for r in summaries if r["split"]==split and r["method"]==name)
    def path(split,name): return next(r for r in paths if r["split"]==split and r["method"]==name)
    for zoom,stem in ((False,"pareto"),(True,"pareto_zoom")):
        fig,axes=plt.subplots(1,2,figsize=(11,4.4))
        for ax,split in zip(axes,("dev","test")):
            for name,color in (("N16","#2166ac"),("U16","#d95f02"),("S16","#1b7837")):
                rr=[r for r in front if r["split"]==split and r["method"]==name and r["point"] in ("grid","unconstrained")]
                rr=sorted(rr,key=lambda r:float(r["evidence_RMS"]))
                ax.plot([float(r["evidence_RMS"]) for r in rr],[100*float(r["VR"]) for r in rr],"-o",ms=3,label=name,c=color)
                s=summary(split,name)
                ax.scatter(float(s["evidence_RMS"]),100*float(s["VR"]),c=color,s=100,marker="*",zorder=5)
            for name in sorted({r["method"] for r in summaries if r["method"].startswith("random")}):
                rr=[r for r in front if r["split"]==split and r["method"]==name and r["point"] in ("grid","unconstrained")]
                rr=sorted(rr,key=lambda r:float(r["evidence_RMS"]))
                ax.plot([float(r["evidence_RMS"]) for r in rr],[100*float(r["VR"]) for r in rr],c="gray",alpha=.17,lw=.6)
            ax.axvline(.20,c="black",ls=":",lw=.8)
            if zoom: ax.set_xlim(0,.65)
            ax.set_xlabel("Actual C/F correction RMS (logit)")
            ax.set_ylabel("A/P variance reduction (%)")
            ax.set_title(split+" (retrospective)")
            ax.grid(alpha=.2)
        axes[0].legend();fig.tight_layout()
        fig.savefig(OUT/(stem+".png"),dpi=180);fig.savefig(OUT/(stem+".pdf"));plt.close(fig)
    lines=["# 同证据代价特征比较（回顾性、探索性）","",
      "在开发集相同的实际证据修正 RMS=0.20 下，正交痕迹 N 的干扰方差降低率为 4.52%，高于未正交 U 的 2.00%、证据 S 的 1.90% 和 50 个随机子空间的均值 1.39%。N 相对三个对照的来源簇配对差值区间均为正。该结果支持当前开发来源上的选择性优势。",
      "",
      "冻结后，测试集 N 的 VR 为 4.08%，但其实际证据代价升至 0.2510，U/S/随机均值为 0.2134/0.1445/0.1906。测试代价未能保持匹配，因此本次结果尚不能证明跨来源、严格同代价的稳定修复优势。N 的动态路径正确数也未增加：测试原模型 768/806，N 为 767/806，与仅截距校准相同。",
      "",
      "全部使用真实 Qwen3-8B 缓存，新增模型前向为 0，不占 GPU。复用原 fit/dev/test 的 80/20/40 来源；A/P 配对数为 4,887/1,198/2,293，C/F 为 135/60/40。当前测试集此前已被项目分析，不作为全新确认性测试。",
      "",
      "## 统一协议",
      "",
      "N、U、S 和 50 个随机子空间均为 16 维，使用同一组 51 个已选注意力头及 fit 标准化。随机种子固定为 11–60，与已有随机方向定义相同。标准化、S/N/U 和无约束岭回归均复用已完成分析的设置，未重新选头或运行五折。",
      "",
      "线性修正目标为 fit A/P 上的 Σ(原分数差+修正差)²+0.01‖w‖²，无配对回归截距；每个 fit C/F 对满足 |修正差|≤τ×fit A/P 原分数差 RMS。岭系数统一固定为已有值 0.01。仅在 dev C/F 上选择 τ，使实际 RMS 接近 0.20，不根据 dev A/P 优劣选择预算。先计算固定预算网格 [0,0.025,0.05,0.1,0.25,0.5,1,2,4]，再在第一个跨过 0.20 的相邻区间求根。全部 53 类特征均达到目标，最大绝对误差小于 0.000001 logit。",
      "",
      "每个模型的节点截距只用 fit 原始自然节点标签校准，对配对分数差和 VR/ER 没有影响。每个特征的修正参数和预算冻结后直接应用于 test；test 不缩放、不重新选预算。",
      "",
      "## 结果",
      "",
      "| 特征 | 划分 | 实际证据 RMS | A/P VR | 证据 ER | 无约束预测 R² | 方向退化 | 超 fit 预算 | 动态路径正确数 |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("dev","test"):
        for name in ("raw","bias_only","N16","U16","S16"):
            s=summary(split,name);p=path(split,name)
            probe="—" if name in ("raw","bias_only") else f"{float(s['probe_R2']):.4f}"
            excess=s["exceeds_fit_budget"] or "—"
            lines.append(f"| {name} | {split} | {float(s['evidence_RMS']):.4f} | {float(s['VR']):.2%} | {float(s['ER']):.2%} | {probe} | {s['direction_degraded']} | {excess} | {p['correct']}/{p['n']} |")
        rr=[r for r in summaries if r["split"]==split and r["method"].startswith("random")]
        pp=[r for r in paths if r["split"]==split and r["method"].startswith("random")]
        mean=lambda key:np.mean([float(r[key]) for r in rr])
        lines.append(f"| 50 随机均值 | {split} | {mean('evidence_RMS'):.4f} | {mean('VR'):.2%} | {mean('ER'):.2%} | {mean('probe_R2'):.4f} | {mean('direction_degraded'):.2f} | {mean('exceeds_fit_budget'):.2f} | {np.mean([float(r['correct']) for r in pp]):.2f}/{pp[0]['n']} |")
    lines+=["","无约束预测 R² 衡量特征对 A/P 分数变化的预测能力，区别于受约束修复残差 R²。二者分别保存为 summary.csv 的 probe_R2 与 repair_R2，预测 RMSE、修复后 A/P 分数差 RMS 均单独保留。原始模型与仅截距方法没有训练预测探针，表中以横线表示。","",
      "N、U、S 在 dev/test 的 C/F 方向退化数均为 0；个别随机配置出现 1 个退化对。逐对 fit 约束不能保证留出对都满足同一上限。N 在 dev 的 60 个 C/F 对中有 11 个超 fit 预算，在 test 的 40 个对中有 11 个超限。ER 是汇总证据响应比例；不能把接近 100% 的 ER 解释为每个证据对都没有改动。","",
      "## 来源簇配对区间","","| 比较 | 划分 | VR 差值（百分点） | 95% CI（百分点） |","|---|---|---:|---:|"]
    for r in comparisons:
        if r["metric"]=="VR":
            lines.append(f"| {r['comparison']} | {r['split']} | {100*float(r['estimate']):.2f} | [{100*float(r['ci_low']):.2f}, {100*float(r['ci_high']):.2f}] |")
    lines+=["",
      "每个来源整簇重采样 2,000 次，同一次重采样在方法间配对。随机对照使用 50 个独立修正模型的统计量均值，而非集成预测；区间描述来源不确定性，条件是这 50 个随机种子固定。随机种子的离散分布另存 random_distribution.csv。",
      "",
      "开发集 N 的 VR 高于 50/50 个随机对照，随机 VR 范围为 0.07%–2.60%。测试 N 也高于 50/50 个随机对照，但测试证据代价不同，不能把这一排序直接称为同代价复现。测试 N 相对 S 的 VR 差区间跨 0。",
      "",
      "## 完整代价曲线",
      "",
      "完整固定预算网格、无约束端点、开发集所选点以及它们直接应用于测试集后的对应点均保存在 pareto.csv。星号表示唯一 dev 所选配置。下图同时保留全部范围及低代价区域；测试曲线只作描述，不从中选择新配置。",
      "",
      "![完整代价曲线](pareto.png)","","![低代价区域](pareto_zoom.png)","","![随机子空间分布](random_distribution.png)",
      "",
      "逐例配对结果：pair_predictions.csv.gz；逐节点：node_predictions.csv.gz；动态路径：path_predictions.csv.gz；来源统计：source_metrics.csv、path_source_metrics.csv；来源区间：bootstrap_intervals.csv、paired_comparisons.csv、path_bootstrap.csv；参数：fitted_models.npz；冻结配置：selection_freeze.json；运行设置与时间：config.json。"]
    (OUT/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Report and full/zoom Pareto figures rendered from saved results.")
if __name__=="__main__":render()
