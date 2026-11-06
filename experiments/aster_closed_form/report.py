"""Render the local automatic-pilot report without inventing pending results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import path_metrics


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _find(root, names):
    for name in names:
        for folder in (root / "results", root, root / "data"):
            path = folder / name
            if path.exists():
                return path, _read(path)
    return None, None


def _rate(value, counts=False):
    if isinstance(value, dict):
        if counts and value.get("numerator") is not None:
            n, d = value.get("numerator"), value.get("denominator")
            return f"{n}/{d} ({100*value['rate']:.1f}%)" if value.get("rate") is not None else "无有效配对"
        value = value.get("rate")
    return "—" if value is None else f"{100*float(value):.1f}%"


def _number(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _ci(value):
    if not value or value.get("low") is None:
        return "—"
    return f"[{100*value['low']:.1f}, {100*value['high']:.1f}]"


def _table(headers, rows):
    if not rows:
        return "尚无已完成结果。\n"
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |"
                      for row in rows]) + "\n"


def _artifact(root, name, alternatives=()):
    path, value = _find(root, (name,)+tuple(alternatives))
    if path is None:
        return None, None
    return str(path.relative_to(root)), value


def _summary(value, prefix="", depth=0):
    """Only scalar summaries; dense matrices, per-input scores and raw logs stay in JSON."""
    if not isinstance(value, dict):
        return []
    rows = []
    for key, item in value.items():
        if key in ("predictions", "pairs", "scores", "coefficients", "intercept", "operators",
                   "heads", "grads", "N", "B", "S", "mu", "raw"):
            continue
        label = prefix + str(key)
        if isinstance(item, (int, float, bool, str)) or item is None:
            if isinstance(item, str) and len(item) > 180:
                item = item[:177]+"..."
            rows.append([label, item if item is not None else "—"])
        elif isinstance(item, dict) and depth < 2:
            rows.extend(_summary(item, label+".", depth+1))
    return rows[:70]


def _prefix_plot(root, h1):
    curves = {k: h1.get(k, {}).get("curve", []) for k in ("A", "P")}
    if not any(curves.values()):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    folder = root / "results"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
    for ax, kind in zip(axes, ("A", "P")):
        curve = [r for r in curves[kind] if r.get("recovery") is not None]
        if curve:
            ax.plot([r["k"] for r in curve], [r["recovery"] for r in curve], color="#225ea8", lw=1.8)
            selected = h1.get(kind, {})
            k = selected.get("selected_k", len(selected.get("H", [])))
            recovery = selected.get("selected_recovery")
            if recovery is not None:
                ax.scatter([k], [recovery], color="#d94801", zorder=3, s=28)
        ax.axhline(.8, color="#666666", ls="--", lw=1)
        ax.set(xlabel="Number of query heads", ylabel="Bidirectional joint recovery",
               title=kind + " interference")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "h1_prefix_curve.svg")
    fig.savefig(folder / "h1_prefix_curve.png", dpi=180)
    plt.close(fig)
    return "results/h1_prefix_curve.png"


def _mechanism_sections(root):
    lines = ["### H1：头数与联合修补效应", ""]
    rel, h1 = _artifact(root, "h1.json", ("mechanism_h1.json",))
    h1 = h1 or {}
    if h1:
        rows = []
        for kind in ("A", "P"):
            x = h1.get(kind, {})
            counts = x.get("effective_pair_counts", {})
            fit, dev = counts.get("fit", {}), counts.get("dev", {})
            rows.append([kind, x.get("selected_k", len(x.get("H", []))),
                         _number(x.get("selected_recovery")),
                         "达到" if x.get("target_met") else "未达到",
                         f"{fit.get('after', 0)}/{fit.get('before', 0)}",
                         f"{dev.get('after', 0)}/{dev.get('before', 0)}", x.get("reason", "—")])
        lines.append(_table(["干扰类型", "选中头数", "开发联合恢复率", "0.8 目标",
                             "拟合有效／全部", "开发有效／全部", "选择说明"], rows))
        lines.append(f"[选头与完整恢复曲线]({rel})。恢复率不裁剪，可能小于 0 或大于 1。\n")
        plot = _prefix_plot(root, h1)
        if plot:
            lines.append(f"![双向联合恢复率随查询头数变化]({plot})\n")
    else:
        lines.append("尚无已完成选头结果。\n")
    summaries = {}
    for split in ("dev", "test"):
        path = root / "mechanisms" / f"mechanisms_{split}.json"
        if path.exists():
            summaries[split] = _read(path)
        elif split in h1.get("interventions", {}):
            summaries[split] = {"h1": h1["interventions"][split]}
    controls = []
    for split, result in summaries.items():
        for r in result.get("h1", []):
            controls.append([split, r.get("cohort", "main"), r.get("kind"), r.get("control"),
                             r.get("count", 0), _number(r.get("recovery"))])
    if controls:
        lines.append(_table(["划分", "子集", "类型", "头集合", "配对数", "双向联合恢复率"], controls))
        lines.append("type_selected 为对应干扰的选中头；selected_union 为 A/P 并集；"
                     "random_13/37/73 在同层保持同数量；all_query_heads 为全部查询头。"
                     "这些效应分析使用可归一化干扰配对，单列于全部有效样本的行为主表。\n")
    lines.extend(["### H2：主夹角、能量与子空间修补", ""])
    rel2, h2 = _artifact(root, "h2.json", ("mechanism_h2.json",))
    h2 = h2 or {}
    for split, intervention in h2.get("interventions", {}).items():
        summaries.setdefault(split, {})["h2"] = intervention.get("effects", [])
        summaries[split]["h2_node_count"] = intervention.get("nodes")
        summaries[split]["h2_pair_count"] = intervention.get("pairs")
    if h2:
        heads = h2.get("heads", {})
        angles = [a for h in heads.values() for a in h.get("geometry", {}).get("principal_angles_degrees", [])]
        retained = sum(h.get("geometry", {}).get("retained_rank", 0) for h in heads.values())
        discarded = sum(h.get("geometry", {}).get("discarded_rank", 0) for h in heads.values())
        if angles:
            import statistics
            angle_summary = f"{min(angles):.2f} / {statistics.median(angles):.2f} / {max(angles):.2f}"
        else:
            angle_summary = "无干扰方向"
        lines.append(_table(["头数", "总编辑秩", "保留方向", "舍弃方向", "主夹角最小／中位／最大（度）"],
                            [[h2.get("number_of_heads", len(heads)), h2.get("total_edit_rank", retained),
                              retained, discarded, angle_summary]]))
        rows = []
        for kind in ("A", "P", "C", "F"):
            chosen = [h for h in heads.values() if h.get("per_type", {}).get(kind, {}).get(
                "included_as_nuisance" if kind in ("A", "P") else "included_as_signal", False)]
            detail = [h["per_type"][kind] for h in chosen]
            energy = sum(d.get("total_energy", 0) for d in detail)
            kept = sum(d.get("retained_energy", 0) for d in detail)
            angle_energy = [h.get("nuisance_energy", {}).get(kind, {}) for h in chosen]
            rows.append([kind, len(chosen), sum(d.get("rank", 0) for d in detail),
                         _number(kept/energy if energy else None),
                         sum(bool(d.get("zero_energy")) for d in detail),
                         sum(not bool(d.get("target_reached")) and not bool(d.get("zero_energy")) for d in detail),
                         _number(sum(d.get("in_retained_principal_directions", 0) for d in angle_energy))
                         if kind in ("A", "P") else "不适用",
                         _number(sum(d.get("discarded_by_angle", 0) for d in angle_energy))
                         if kind in ("A", "P") else "不适用"])
        lines.append(_table(["类型", "参与头数", "逐类型秩之和", "实际能量保留比例",
                             "零能量头", "未达能量目标头", "夹角筛选保留能量", "夹角筛选舍弃能量"], rows))
        lines.append(f"[逐头主夹角、逐类型基、合并维度与能量]({rel2})。"
                     "能量比例按该类型在参与头上的总二阶矩能量加权；合并空间维数不受单类型秩上限限制。\n")
    else:
        lines.append("尚无已完成几何结果。\n")
    patches = []
    for split, result in summaries.items():
        by_single = {}
        for r in result.get("h2", []):
            if r.get("head") == "joint":
                patches.append([split, r.get("kind"), "联合", r.get("operation"),
                                r.get("count", 0), _number(r.get("recovery"))])
            else:
                key = (r.get("kind"), r.get("operation"))
                n, total = by_single.get(key, (0, 0.0))
                if r.get("recovery") is not None:
                    by_single[key] = (n+r.get("count", 0),
                                      total+r.get("count", 0)*r["recovery"])
        for (kind, operation), (n, total) in by_single.items():
            patches.append([split, kind, "单头逐次平均", operation, n, _number(total/n if n else None)])
    if patches:
        lines.append(_table(["划分", "类型", "范围", "交换内容", "修补观测数", "双向恢复率"], patches))
        lines.append("nuisance 只交换估计干扰分量，full 交换完整头输出。"
                     "单头平均仅描述逐头干预，不能相加代替联合效果。\n")
    lines.extend(["### H3：信号探针与放大对照", ""])
    rel3, h3 = _artifact(root, "h3.json", ("mechanism_h3.json", "signal_probe.json"))
    h3 = h3 or {}
    if h3:
        probe = h3.get("probe_main", h3)
        lines.append(f"探针状态：{probe.get('status', '未完成')}；拟合原输入数："
                     f"{probe.get('fit_inputs', '—')}；信号坐标维数：{probe.get('coordinate_dimension', '—')}。\n")
        if "full_error_count" in probe:
            probe_rows = [
                ["全部测试视图", probe["full_error_count"], probe.get("scored_errors", "—"),
                 _rate(probe.get("accuracy"), True)],
                ["其中原输入子集", probe.get("original_full_error_count", "—"),
                 probe.get("original_scored_errors", "—"), _rate(probe.get("original_error_accuracy"), True)]]
        else:
            probe_rows = [["原输入（旧统计口径）", probe.get("original_full_error_count", "—"),
                           probe.get("scored_errors", "—"), _rate(probe.get("accuracy"), True)]]
        lines.append(_table(["错误样本范围", "原模型错误数", "已评分错误数", "探针正确率"], probe_rows))
        if probe.get("reason"):
            lines.append("探针说明："+str(probe["reason"])+"\n")
        amp = h3.get("amplification", {})
        lines.append(f"开发集选中的信号放大倍数：{amp.get('amplification', '尚未选定')}。"
                     "探针只作诊断，不参与在线判断。\n")
        comparisons = []
        for label, key in (("仅擦除", "erase"), ("擦除＋开发选定放大", "erase_plus_signal")):
            result = h3.get(key) or {}
            if result:
                comparisons.append([label, _rate(result.get("macro_PA")),
                                    _rate(result.get("original_accuracy"), True),
                                    _rate(result.get("change_vs_full", {}).get("corrected"), True),
                                    _rate(result.get("change_vs_full", {}).get("regressed"), True)])
        lines.append(_table(["测试方法", "四类宏平均", "原输入正确率", "原错纠正", "原对退化"], comparisons))
        if amp.get("development"):
            lines.append(_table(["开发候选", "四类宏平均", "原输入正确率"],
                         [[m, _rate(v.get("macro_PA")), _rate(v.get("original_accuracy"), True)]
                          for m, v in amp["development"].items()]))
        lines.append(f"[探针和放大完整结果]({rel3})。\n")
    else:
        lines.append("尚无已完成结果。\n")
    lines.extend(["### 局部预测、固定替换与逐层编辑", ""])
    _, local_data = _artifact(root, "local_theory.json", ("mechanism_local.json",))
    for split, result in (local_data or {}).items():
        if isinstance(result, dict) and split in ("dev", "test"):
            summaries.setdefault(split, {})["local_theory"] = result
    local = [[split, result.get("local_theory", {}).get("nodes", 0),
              _number(result.get("local_theory", {}).get("mean_abs_nonlinear_error")),
              _number(result.get("local_theory", {}).get("mean_abs_cross_layer_change")),
              _number(result.get("local_theory", {}).get("mean_abs_total_prediction_error"))]
             for split, result in summaries.items()]
    lines.append(_table(["划分", "节点数", "非线性误差绝对均值", "跨层变化绝对均值", "总预测误差绝对均值"], local))
    lines.append("非线性误差比较联合夹持下的线性预测与固定目标替换；"
                 "跨层变化比较固定目标替换与实际逐层编辑。\n")
    lines.extend(["### 算子数值性质", ""])
    relcheck, checks = _artifact(root, "operator_checks.json", ("operator_diagnostics.json",))
    if checks:
        lines.append(f"[信号保持、擦除、幂等、改动范围与幅度检查]({relcheck})。\n")
        algebra = [v for v in checks.values() if isinstance(v, dict) and "signal_preservation_error" in v]
        if algebra:
            names = ("signal_preservation_error", "erasure_error", "idempotence_error",
                     "edit_range_error", "edit_operator_norm")
            rows = [[key, _number(max(a.get(key, 0) for a in algebra), 7)] for key in names]
            rows.append(["angle_bound_violations", sum(a.get("angle_bound_satisfied") is False for a in algebra)])
            lines.append(_table(["跨头最大残差／量", "值"], rows))
        elif _summary(checks):
            lines.append(_table(["记录项", "数值／状态"], _summary(checks)))
    elif h2.get("heads"):
        algebra = [h.get("algebra", {}) for h in h2["heads"].values()]
        names = ("signal_preservation_error", "erasure_error", "idempotence_error", "edit_range_error",
                 "edit_operator_norm")
        rows = [[key, _number(max(a.get(key, 0) for a in algebra), 7)] for key in names]
        rows.append(["all_angle_bounds_satisfied", all(a.get("angle_bound_satisfied") for a in algebra)])
        lines.append(_table(["跨头最大残差／量", "值"], rows))
    else:
        lines.append("尚无已完成结果。\n")
    artifacts = []
    for folder in (root / "mechanisms", root / "m1"):
        if folder.exists():
            artifacts.extend(sorted(p for p in folder.iterdir() if p.suffix in (".json", ".jsonl")))
    if artifacts:
        lines.extend(["### 机制原始记录", ""])
        lines.extend(f"- [{p.name}]({p.relative_to(root)})" for p in artifacts)
        lines.append("")
    return lines


def _json_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coverage_status(root, evaluations, pipeline_state="complete"):
    """Separate process completion from objective coverage; no sample-count gates."""
    root = Path(root)
    evaluations = evaluations or {}
    if "methods" in evaluations:
        evaluations = {"main_test": evaluations}
    _, manifest = _find(root, ("data_manifest.json", "manifest.json"))
    manifest = manifest or {}
    main = evaluations.get("main_test") or {}
    definition = evaluations.get("definition_test") or {}
    target = int(manifest.get("definition_targets", {}).get("test", 20))
    actual = int(definition.get("node_count", 0))
    issues = []
    if not main or not main.get("input_count", 0):
        issues.append(dict(code="main_results_unavailable", scope="main_test",
                           detail="主测试结果尚未完成。"))
    if target and not actual:
        issues.append(dict(code="definition_subset_empty", scope="definition_test",
                           actual_nodes=0, target_nodes=target,
                           detail=f"定义测试子集为 0/{target} 节点，定义变化、删除与补回实验尚未完成。"))
    missing = {m: int(v.get("score_coverage", {}).get("missing", 0))
               for m, v in main.get("methods", {}).items()
               if v.get("score_coverage", {}).get("missing", 0)}
    if missing:
        primary = [m for m in ("full", "aster", "semantic", "boundary_guideline", "boundary_facts")
                   if m in missing]
        named = primary or list(missing)[:4]
        detail = "主测试存在未获得分数的输入：" + "、".join(
            f"{m} {missing[m]}/{main.get('input_count', 0)}" for m in named) + "；缺分在主表中计错。"
        issues.append(dict(code="node_scores_missing", scope="main_test", methods=missing, detail=detail))
    _, paths = _find(root, ("path_results.json", "path_metrics.json"))
    paths = paths or {}
    path_missing = {m: int(v.get("missing_or_failed", 0)) for m, v in paths.get("methods", {}).items()
                    if v.get("missing_or_failed", 0)}
    if path_missing:
        issues.append(dict(code="path_runs_failed", scope="paths", methods=path_missing,
            detail="自主路径存在技术失败：" + "、".join(f"{m} {n} 条" for m, n in path_missing.items()) + "。"))
    experiment_state = "partial" if not main or not main.get("input_count", 0) else (
        "complete_with_gaps" if issues else "complete")
    return dict(pipeline_state=pipeline_state, experiment_state=experiment_state, issues=issues,
                coverage=dict(definition_test_nodes=actual, definition_test_target=target,
                              main_test_inputs=main.get("input_count", 0),
                              main_test_missing_scores=missing, path_failed_counts=path_missing))


def _original_coverage(root, methods):
    """Read saved scores for coverage diagnostics; never rescore or alter the main pool."""
    from .evaluation import fraction, margin
    views = {str(v["view_id"]): v for v in _json_rows(root/"data/views.jsonl")
             if v.get("split") == "test" and v.get("cohort", "main") == "main" and v.get("original")}
    if not views:
        return {}
    wanted = set(methods) | {"full"}
    scores = {}
    for path in sorted((root/"scores").glob("baseline_*.jsonl")):
        for record in _json_rows(path):
            if record.get("method") in wanted:
                scores.setdefault(record["method"], {})[str(record["view_id"])] = record
    result = {}
    for method in methods:
        if method not in scores:
            continue
        ids = [vid for vid in views if margin(scores[method].get(vid)) is not None]
        correct = sum(int(margin(scores[method][vid]) > 0) == int(views[vid]["label"]) for vid in ids)
        shared = [vid for vid in ids if margin(scores.get("full", {}).get(vid)) is not None]
        full_correct = sum(int(margin(scores["full"][vid]) > 0) == int(views[vid]["label"]) for vid in shared)
        result[method] = dict(scored=len(ids), total=len(views), accuracy=fraction(correct, len(ids)),
                              full_same_subset=fraction(full_correct, len(shared)))
    return result


def _coverage_section(root, main):
    from .evaluation import fraction
    affected = [name for name, m in main.get("methods", {}).items()
                if m.get("score_coverage", {}).get("missing", 0)]
    if not affected:
        return []
    originals = _original_coverage(root, affected)
    rows = []
    for name in affected:
        m = main["methods"][name]
        cov = m["score_coverage"]
        count = int(cov.get("numerator", 0))
        correct = m.get("all_input_accuracy", {}).get("numerator")
        scored = fraction(correct, count) if correct is not None else None
        original = originals.get(name, {})
        rows.append([name, f"{cov.get('missing', 0)}/{cov.get('denominator', 0)}",
                     _rate(scored, True),
                     f"{original['scored']}/{original['total']}" if original else "未记录",
                     _rate(original.get("accuracy"), True), _rate(original.get("full_same_subset"), True)])
    return ["### 技术缺失与已评分结果", "",
            _table(["方法", "缺分／总视图", "已评分视图正确率", "原输入已评分／全部",
                    "已评分原输入正确率", "完整阅读在同一原输入子集"], rows),
            "主表保留技术失败计错的全体分母。这里的已评分准确率只描述完成子集，不替代主指标。"
            "边界标注缺失意味着原计划的上限对照尚未完成，不能把缺分造成的低分解释为纯判断退化。"
            "完成子集与缺失子集可能不同，比较时同时保留覆盖率和同一子集的完整阅读结果。", ""]


def render_report(root, evaluations=None, metadata=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if evaluations is None:
        _, evaluations = _find(root, ("evaluations.json", "evaluation.json"))
    evaluations = evaluations or {}
    if "methods" in evaluations:
        evaluations = {"main_test": evaluations}
    metadata = metadata or {}
    _, status = _find(root, ("status.json",))
    status = status or {}
    state = metadata.get("pipeline_state", status.get("pipeline_state",
                         status.get("status", status.get("state", "结果生成中"))))
    completion = coverage_status(root, evaluations, pipeline_state=state)
    process_label = "已结束" if state in ("complete", "finished") else str(state)
    experiment_label = {"complete": "已完成", "complete_with_gaps": "存在未完成内容",
                        "partial": "部分完成"}[completion["experiment_state"]]
    lines = [
        "# ASTER 闭式方法：自动预实验",
        "",
        f"计算流程：{process_label}；计划实验：{experiment_label}。"
        "本报告由已保存结果生成，程序结束不表示所有计划实验均已获得有效结果。",
        "",
        "本轮复用有历史使用记录的指南与构造病例，属于探索性验证。"
        "配对由程序及模型辅助构造，医生核验尚未完成；本报告不将自动检查写作临床核验。",
        "",
        "骨干冻结、关闭 thinking，以“是减否”的单词元分数直接判断；同分为“否”。"
        "ASTER 仅在最终判定位置、输出投影前编辑查询头，在线每个节点一次标签评分。",
        "",
        "## 数据与主效果",
        "",
        "主数据计划为拟合 16 份指南／80 节点、开发 4 份指南／20 节点、测试 20 份指南／100 节点。"
        "开发来源固定为 train_val:2308、2544、2788、688。"
        "定义子集目标为 8／2／20 节点，实际有效数量以下列结果和数据清单为准。",
        "",
    ]
    if completion["issues"]:
        lines[4:4] = ["尚未完成的内容：", ""] + [
            "- "+issue["detail"] for issue in completion["issues"]] + [""]
    main = evaluations.get("main_test")
    if main:
        lines.append(_table(["部分", "来源数", "节点数", "原输入数", "A", "P", "C", "F"], [[
            "主测试", main.get("source_count"), main.get("node_count"), main.get("original_input_count"),
            *[main.get("pair_counts", {}).get(k, 0) for k in ("A", "P", "C", "F")]]]))
        rows = []
        for name, m in main["methods"].items():
            rows.append([name, *[_rate(m.get("PA_"+k), counts=True) for k in ("A", "C", "F")],
                         _rate(m.get("PA_P_clinical"), True), _rate(m.get("PA_P_preference"), True),
                         _rate(m.get("PA_P")), _rate(m.get("macro_PA")),
                         _rate(m.get("original_accuracy"), True), _rate(m.get("score_coverage"), True)])
        lines.append(_table(["方法", "PA_A", "PA_C", "PA_F", "PA_P 临床", "PA_P 偏好",
                             "PA_P", "四类宏平均", "原输入正确率", "分数覆盖"], rows))
        lines.append("boundary_guideline 根据构造标注保留必要指南定义，病历保持不变；"
                     "boundary_facts 另外只保留决定性患者事实。两者使用测试节点的构造标注，作为信息处理上限对照。"
                     "semantic 由模型从指南中提取保留内容，额外提取时间与调用量记录在构造状态和运行文件中。\n")
        lines.extend(_coverage_section(root, main))
        lines.append("PA_P 对临床背景与治疗偏好各赋一半权重；四类宏平均再对 A/P/C/F 等权。"
                     "任一必要类型没有有效配对时，宏平均记为空，不能用其他类型补齐。"
                     "主指标包括全部语义有效测试配对，技术失败保留在分母中；效应阈值仅用于 M1/M2 拟合。\n")
        ci_rows = []
        intervals = main.get("confidence_intervals", {}).get("methods", {})
        for name, m in main["methods"].items():
            ci = intervals.get(name, {})
            change = m.get("change_vs_full", {})
            ci_rows.append([name, _ci(ci.get("macro_PA")),
                            _ci(ci.get("differences", {}).get("full", {}).get("macro_PA")),
                            _rate(change.get("corrected"), True), _rate(change.get("regressed"), True),
                            change.get("net_corrected", "—"),
                            _rate(m.get("historical_fourway"), True)])
        lines.append(_table(["方法", "宏平均 95% 区间", "相对完整阅读差值区间（百分点）",
                             "原错纠正", "原对退化", "净纠正数", "历史四输入全对"], ci_rows))
        lines.append("区间按指南来源有放回重采样，保留同来源的全部节点与配对。"
                     "历史四输入全对率是“事实两态×处置两视图”的补充指标，不是四类配对宏平均。\n")
        double = []
        for name, m in main["methods"].items():
            for baseline, result in m.get("double_wrong_correction", {}).items():
                double.append([name, "完整＋"+baseline, _rate(result, True), result.get("baseline_missing", 0)])
        lines.append("### 同一节点双错后的纠正\n")
        lines.append(_table(["方法", "两路基线", "纠正／双错", "两路分数缺失"], double))
    else:
        lines.append("主测试结果尚未完成。\n")
    lines.extend(["## 定义子集", ""])
    definition = evaluations.get("definition_test")
    if definition and definition.get("node_count", 0):
        lines.append(f"实际测试：{definition.get('source_count', 0)} 个来源、"
                     f"{definition.get('node_count', 0)} 个节点。\n")
        lines.append(_table(["方法", "PA_A", "PA_C", "PA_F", "PA_P", "原输入正确率"],
                     [[name, *[_rate(m.get("PA_"+k), True) for k in ("A", "C", "F")],
                       _rate(m.get("PA_P")), _rate(m.get("original_accuracy"), True)]
                      for name, m in definition["methods"].items()]))
    else:
        lines.append("有效定义测试节点为 0；定义变化、删除与补回结果尚未完成。"
                     "不能据此评价定义保护效果；目标数量与失败原因见构造清单。\n")
    lines.extend(["## 机制与模块消融", "",
                  "H1 的联合恢复率、H2 的子空间交换效应与 H3 的错误样本信号读出分别记录，"
                  "不由算子代数性质推断实际判断收益。", ""])
    lines.extend(_mechanism_sections(root))
    if main:
        ablation = [[name, _rate(m.get("macro_PA")), _rate(m.get("original_accuracy"), True)]
                    for name, m in main["methods"].items()
                    if any(tag in name.lower() for tag in ("random", "mean", "no_c", "no_f", "orthogonal",
                                                            "leace", "anchor", "ampl"))]
        if ablation:
            lines.append("### 模块、参考均值与放大对照\n")
            lines.append(_table(["方法", "四类宏平均", "原输入正确率"], ablation))
    lines.extend(["## 完整与遮挡的画像响应", ""])
    if main:
        rows = []
        for masked, groups in main.get("profile_response", {}).items():
            for subtype, result in groups.items():
                rows.append([masked, subtype, result.get("paired_scored"),
                             _number(result.get("full_response", {}).get("mean_abs")),
                             _number(result.get("masked_response", {}).get("mean_abs")),
                             _number(result.get("difference_in_differences", {}).get("mean_abs")),
                             _number(result.get("response_correlation"))])
        lines.append(_table(["遮挡方式", "画像类型", "配对数", "完整响应绝对均值",
                             "遮挡响应绝对均值", "差分之差绝对均值", "响应相关"], rows))
        lines.append("差分之差直接比较两路分数变化。各固定输出方法的画像响应保存在 evaluation JSON；"
                     "配对正确率提高本身不能证明画像共享项被消除。\n")
    else:
        lines.append("尚无已完成结果。\n")
    lines.extend(["## 自主完整路径", ""])
    rel, paths = _artifact(root, "path_results.json", ("paths.json", "path_metrics.json"))
    if paths is not None:
        if isinstance(paths, list):
            paths = path_metrics(paths)
        elif "records" in paths and "methods" not in paths:
            paths = path_metrics(paths["records"])
        lines.append(f"结果文件：[路径结果]({rel})。\n")
        from .evaluation import fraction
        lines.append(_table(["方法", "全体完整路径正确率", "完成数", "缺失／失败", "已完成路径正确率"],
                     [[name, _rate(m, True), m.get("completed", "—"), m.get("missing_or_failed", "—"),
                       _rate(fraction(m.get("numerator", 0), m.get("completed", 0)), True)]
                      for name, m in paths.get("methods", {}).items()]))
        if any(m.get("missing_or_failed", 0) for m in paths.get("methods", {}).values()):
            lines.append("全体路径指标将技术失败计错。已完成路径正确率单列诊断，"
                         "不同方法的完成病例集合可能不同，不能直接把其差异解释为判断能力差异。\n")
    else:
        lines.append("自主路径实验尚未完成。目标是在同 20 份测试指南中复用 100 个原病例，从根节点自主执行。\n")
    lines.extend(["## 配置与材料", ""])
    for title, name, alternatives in [
        ("数据清单", "data_manifest.json", ("manifest.json",)),
        ("最终配置", "selection.json", ("frozen_selection.json",)),
        ("输出方法选择", "fusion_selection.json", ("fusion.json", "fusions.json",)),
        ("运行资源计划", "resource_plan.json", ("resources.json",)),
        ("实际运行时长", "runtime.json", ("benchmark.json",)),
        ("状态", "status.json", ()),
    ]:
        rel, value = _artifact(root, name, alternatives)
        if rel:
            lines.append(f"- [{title}]({rel})")
    for rel in ("status/construction.json", "status/path_semantic_extraction.json"):
        if (root / rel).exists():
            lines.append(f"- [模型辅助构造和语义提取耗时]({rel})")
    for rel in ("review/physician_review.jsonl", "review/physician_review.csv", "data/physician_review.jsonl", "data/doctor_review.jsonl", "physician_review.jsonl",
                "data/physician_review.md", "doctor_review.md"):
        if (root / rel).exists():
            lines.append(f"- [医生待核验材料]({rel})")
    lines.extend(["", "算子的信号保持、擦除和幂等性质只针对估计子空间。"
                  "主夹角、修补结果和实际正确率分别决定机制是否得到支持；正负结果均保留。"
                  "本轮不包含 PASTA、ITI、LoReFT 或全面外部评测，不据此宣称超过这些方法。", ""])
    report = root / "REPORT.zh.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return str(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(render_report(args.root))


if __name__ == "__main__":
    main()
