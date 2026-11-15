"""Render a Chinese report and static figures from completed study artifacts.

No metric is inferred from an absent file. This module never trains a model,
selects a candidate, changes a historical artifact, or starts a J2 experiment.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path):
    path = Path(path)
    if not path.exists():
        compressed = Path(str(path) + ".gz")
        if not compressed.exists():
            return []
        path = compressed
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, scale=1, digits=4):
    value = _number(value)
    return "未完成" if value is None else f"{value*scale:.{digits}f}"


def _interval(row):
    if not row or row.get("estimate") is None:
        return "未完成"
    return (f"{_fmt(row.get('estimate'), 100, 3)} 个百分点，"
            f"95% CI [{_fmt(row.get('low'), 100, 3)}, {_fmt(row.get('high'), 100, 3)}]")


def _comparison(result, lhs, rhs):
    return result.get("comparisons", {}).get(lhs + " minus " + rhs, {}).get("metrics", {})


def _conditions(result, lhs, rhs):
    pair = _comparison(result, lhs, rhs)
    natural, interference = pair.get("natural_path_accuracy", {}), pair.get("interference_path_accuracy", {})
    if natural.get("low") is None or interference.get("low") is None:
        return None
    return bool(interference["low"] > 0 and natural["low"] >= -0.005)


def _comparison_sentence(result, lhs, rhs, label=None):
    pair = _comparison(result, lhs, rhs)
    label = label or lhs + " − " + rhs
    if not pair:
        return label + "：真实实验未完成。"
    outcome = _conditions(result, lhs, rhs)
    judgement = ("满足预设的干扰增益和自然非劣条件。" if outcome else
                 "预设的干扰增益和自然非劣联合条件未满足。" if outcome is False else "成功条件尚无法判定。")
    return (label + "：自然路径差 " + _interval(pair.get("natural_path_accuracy")) +
            "；干扰路径差 " + _interval(pair.get("interference_path_accuracy")) + "。" + judgement)


def _table(headers, rows):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + "|")
    return lines


def _test_summary(record, link):
    if not record:
        return "测试结果文件尚未生成。"
    count = record.get("tests_run")
    if count is None:
        return f"测试记录见 [{link}]({link})。"
    label = "合成测试" if record.get("synthetic") else "自动化测试"
    state = "通过" if record.get("successful") is True else "未全部通过" if record.get("successful") is False else "已执行"
    detail = (f"{count}项{label}{state}，失败{record.get('failures', '未记录')}，"
              f"错误{record.get('errors', '未记录')}；耗时{_fmt(record.get('seconds'), digits=3)}秒。")
    if record.get("environment"):
        detail += "执行环境：" + str(record["environment"]) + "。"
    return detail + f" 详见 [{link}]({link})。"


def _models(root, methods=("S", "J")):
    result = []
    for fold in sorted((root / "models").glob("fold*")):
        for method in methods:
            path = fold / (method + ".json")
            if path.exists():
                result.append((fold.name, method, _read(path)))
    return result


def _source_error_evidence(records, comparison):
    rows = [r for r in records if r.get("comparison") == comparison and r.get("before_first_error") is not None]
    measured = [r for r in rows if r.get("before_gold_margin") is not None and r.get("after_gold_margin") is not None]
    if not measured:
        return None
    unresolved = [r for r in measured if not r.get("first_error_repaired")]
    return dict(total=len(measured), repaired=sum(bool(r.get("first_error_repaired")) for r in measured),
                unresolved=len(unresolved),
                before_abs_mean=float(np.mean([r["before_abs_score"] for r in measured])),
                abs_shift_mean=float(np.mean([abs(r["score_shift"]) for r in measured])),
                unresolved_before_margin_mean=float(np.mean([r["before_gold_margin"] for r in unresolved])) if unresolved else None,
                unresolved_after_margin_mean=float(np.mean([r["after_gold_margin"] for r in unresolved])) if unresolved else None,
                unresolved_shift_below_distance=sum(abs(r["score_shift"]) < abs(r["before_score"]) for r in unresolved))


def _diagnosis(result, models, first_errors, bounds=None, j2_status=None, j2_result=None):
    bounds, j2_status, j2_result = bounds or {}, j2_status or {}, j2_result or {}
    if not result.get("models"):
        return ["当前没有 S/J 的真实训练与路径评价结果。下一步是恢复服务器访问并完成冻结对照；目标、约束、校准和表达能力的归因等待实测。J2 尚未启动。"]
    lines = []
    increment = _conditions(result, "J_independent", "S_independent")
    complete = _conditions(result, "J_independent", "R1")
    if increment and complete:
        lines.append("当前预设比较满足任务成功条件。下一步优先冻结方法和超参数规则，在新来源上验证；本轮不调整证据预算。")
    elif bounds.get("records"):
        lines.append(f"证据约束直接限制了当前纠错可达性。固定原模型截距 b0，在全部16维线性权重中求最有利修正，"
                     f"{bounds['impossible_at_delta_shared_b0']}/{bounds['records']} 个 R1 自然首错节点在原 C/F 预算内仍不可达；"
                     f"将预算上界按齐次性扩为2倍后，{bounds['impossible_at_twice_delta_shared_b0']} 个仍不可达，"
                     f"{bounds['newly_reachable_with_at_most_double_budget']} 个由不可达变为单节点可达。"
                     "这为标签感知证据约束提供了直接实验依据。")
        if j2_result.get("complete"):
            j2_ok = (_conditions(j2_result, "J2_independent", "J_independent") and
                     _conditions(j2_result, "J2_independent", "R1"))
            lines[0] += ("J2已完成，方向监督与非对称松弛的组合满足本轮探索性任务条件。"
                         if j2_ok else "J2已完成，当前方向监督与非对称松弛的组合尚未建立任务成功。")
            mm = _comparison(j2_result,"J2_independent","R1").get("interference_path_accuracy")
            if mm:
                lines[0] += "J2−R1干扰路径差为 " + _interval(mm) + "。"
    final_cosines = [(fold, m.get("gradient_final", {}).get("task_ap_cosine"))
                     for fold, method, m in models if method == "J"]
    available = [(fold, value) for fold, value in final_cosines if value is not None]
    negative = [(fold, value) for fold, value in available if value < 0]
    if increment is False and available:
        lines.append(f"J 的最终任务/AP 梯度在 {len(negative)}/{len(available)} 折呈负余弦，"
                     "支持两个训练目标存在局部方向冲突。它解释当前解附近的优化关系；"
                     "证据约束的可达性上界决定本轮更优先的修改方向。" if bounds.get("records") else
                     f"J 的最终任务/AP 梯度在 {len(negative)}/{len(available)} 折呈负余弦，"
                     "下一步结合各折加权梯度范数和损失变化检查 AP 项的优化贡献。")
    for method in ("S_independent", "J_independent"):
        evidence = _source_error_evidence(first_errors, method + " minus R1")
        if evidence and evidence["unresolved"]:
            lines.append(f"{method} 在 R1 的 {evidence['total']} 个可定位自然首错节点中修复 {evidence['repaired']} 个；"
                         f"{evidence['unresolved']} 个修正后仍错，其中 {evidence['unresolved_shift_below_distance']} 个"
                         "实际分数变化的绝对值小于原边界距离。"
                         f"这些未修复节点的平均有向 margin 为 {_fmt(evidence['unresolved_before_margin_mean'])} → "
                         f"{_fmt(evidence['unresolved_after_margin_mean'])} logit。")
    for method in ("S", "J"):
        shared = result.get("models", {}).get(method + "_shared", {})
        independent = result.get("models", {}).get(method + "_independent", {})
        if all(row.get("natural_path_accuracy") is not None and row.get("interference_path_accuracy") is not None
               for row in (shared, independent)):
            natural = independent["natural_path_accuracy"] - shared["natural_path_accuracy"]
            interfere = independent["interference_path_accuracy"] - shared["interference_path_accuracy"]
            lines.append(f"{method} 从共享截距换为独立截距，自然/干扰路径点估计分别变化 "
                         f"{_fmt(natural,100,3)} / {_fmt(interfere,100,3)} 个百分点；该变化单独归入校准分析。")
    raw, intercept = (result.get("models", {}).get(name, {}) for name in ("R0", "R1"))
    if all(row.get("natural_path_correct") is not None and row.get("interference_path_correct") is not None
           for row in (raw, intercept)):
        lines.append(f"R0原始分数答对自然/干扰路径 {raw['natural_path_correct']}/{raw['interference_path_correct']} 条，"
                     f"R1自然节点截距校准后为 {intercept['natural_path_correct']}/{intercept['interference_path_correct']} 条。"
                     f"校准使自然/干扰正确数分别变化 {intercept['natural_path_correct']-raw['natural_path_correct']:+d}/"
                     f"{intercept['interference_path_correct']-raw['interference_path_correct']:+d} 条。"
                     "这一描述性对照直接显示当前自然节点BCE截距校准在干扰路径上的损失。"
                     "后续应将校准目标和证据约束分别作为控制因素；首轮J−S与J−R1的主要比较保持不变。")
    if not (increment and complete):
        if bounds.get("records"):
            lines.append("共享与独立截距的结果均按各自区间判断；本轮截距变化尚未建立任务成功。"
                         "上述上界对应固定 b0 的单节点纠错，重新拟合截距及多个节点同时纠正需要用完整路径评价验证。"
                         "J2沿用相同16维痕迹和线性容量，结果作为单独探索实验保存。"
                         f"当前状态：{j2_status.get('state', '已具备启动依据，状态文件待写入')}。")
            if j2_result.get("complete") and not (_conditions(j2_result,"J2_independent","J_independent") and
                                                   _conditions(j2_result,"J2_independent","R1")):
                lines.append("若继续，优先在新训练来源中分离证据方向监督与非对称松弛的贡献，固定对照后再用新来源确认。"
                             "本轮搜索范围已经结束。当前证据同时涉及约束、局部目标冲突和边界距离，无法唯一归因为特征不足。")
        else:
            lines.append("下一轮依据梯度、错误边界距离及两种截距变化定位修改点。"
                         "J2是否开展由可达性诊断和实际运行状态决定。")
    if not lines:
        lines.append("当前诊断文件尚未提供可判定的优化与错误边界证据；下一步补齐这些记录后再确定修改点。")
    return lines


def _source_changes(source_rows, result):
    by_source = {(r.get("method"), r.get("source_id")): r for r in source_rows}
    rows = []
    for name, comparison in result.get("comparisons", {}).items():
        lhs, rhs = comparison.get("lhs"), comparison.get("rhs")
        if lhs is None or rhs is None:
            continue
        sources = sorted({source for method, source in by_source if method == lhs} &
                         {source for method, source in by_source if method == rhs})
        for metric, label in (("natural_path_accuracy", "自然路径"), ("interference_path_accuracy", "干扰路径")):
            differences = []
            for source in sources:
                left = _number(by_source[(lhs, source)].get(metric))
                right = _number(by_source[(rhs, source)].get(metric))
                if left is not None and right is not None:
                    differences.append(left-right)
            if differences:
                rows.append([name, label, sum(d > 1e-12 for d in differences),
                             sum(d < -1e-12 for d in differences), sum(abs(d) <= 1e-12 for d in differences),
                             len(differences)])
    return rows


def _bound_section(summary, records):
    text = ["## 固定截距下的纠错可达性上界", ""]
    if not summary.get("records"):
        return text + ["线性规划诊断尚未完成。", ""]
    text += [f"对 R1 的 {summary['records']} 个可定位自然路径首错节点逐一求解，有限上界 {summary.get('finite_bounds','未记录')} 个。"
             f"原预算下 {summary['impossible_at_delta_shared_b0']} 个不可达；2倍预算上界下 "
             f"{summary['impossible_at_twice_delta_shared_b0']} 个不可达，"
             f"{summary['newly_reachable_with_at_most_double_budget']} 个在不超过2倍预算时变为单节点可达。", "",
             r"记 $s_i=2y_i-1$，$E_j=t_{r_j}-t_{l_j}$，$m_i=s_i(\ell_i+b_0)$。逐节点上界为", "",
             r"\[B_i(\delta)=\max_{w\in\mathbb R^{16}}s_i(t_i-\mu_{\rm train})^\top w\quad"
             r"\text{s.t.}\quad-\delta\le E_j^\top w\le\delta,\ j\in CF_{\rm train}.\]", "",
             r"当 $m_i+B_i(\delta)<-10^{-7}$ 时，该节点在固定 $b_0$ 与这些训练约束下无法由任何16维线性权重纠正。"
             "求解不加入任务、AP或正则惩罚，因而覆盖原可行域内所有线性权重。", "",
             r"由齐次性，$B_i(2\delta)=2B_i(\delta)$。2倍预算数字由此直接计算，没有重新执行训练预算扫描。"
             "每个节点分别采用最有利权重，单节点可达数与多个节点同时修复数分别解释。"
             "这组主上界固定b0；记录中以当前bw计算的辅助边界也固定该已拟合截距。重新拟合bw后须重新检查边界。", ""]
    unique = summary.get("unique_first_error_inputs")
    if unique is None and records:
        unique = len({r["view_id"] for r in records if r.get("view_id") is not None})
    if unique is not None:
        text.insert(3, f"{summary['records']} 条首错记录对应 {unique} 个唯一输入，重复记录 "
                       f"{int(summary['records'])-int(unique)} 条。"
                       + (f"零痕迹方向 {summary['zero_trace_directions']} 个。" if "zero_trace_directions" in summary else ""))
        text.insert(4, "")
    folds = defaultdict(list)
    for record in records:
        folds[record.get("fold", "unknown")].append(record)
    rows = []
    for fold, rr in sorted(folds.items()):
        rows.append([fold, len(rr), sum(bool(r.get("impossible_at_delta_shared_b0")) for r in rr),
                     sum(bool(r.get("impossible_at_twice_delta_shared_b0")) for r in rr),
                     sum(bool(r.get("impossible_at_delta_shared_b0")) and not r.get("impossible_at_twice_delta_shared_b0") for r in rr),
                     sum(_number(r.get("max_gold_margin_with_current_fixed_bw")) is not None and
                         float(r["max_gold_margin_with_current_fixed_bw"]) < -1e-7 for r in rr)])
    text += _table(["折", "首错节点", "δ下不可达，固定b0", "2δ下不可达，固定b0", "≤2δ新增可达", "δ下不可达，固定当前bw"], rows)
    if summary.get("folds"):
        text += ["", "原 J 的训练约束与原模型证据方向：", ""]
        text += _table(["折", "训练CF对", "活跃约束", "原证据方向非正", "其中活跃"],
            [[r.get(k) for k in ("fold", "training_CF_pairs", "active_constraints", "raw_wrong_direction_CF", "active_raw_wrong_direction_CF")]
             for r in summary["folds"]])
    text += ["", "这些记录将主要问题定位到当前证据可行域与错误边界距离的关系。"
             "局部目标冲突和截距作用分别报告；特征表达能力的判断由相应对照实验承担。", "",
             "[上界汇总](diagnostics/correction_bounds_summary.json) · [逐节点LP结果](diagnostics/correction_bounds.jsonl)。", ""]
    return text


def _j2_state_text(status, result):
    if result.get("complete") and result.get("models"):
        return "探索性 J2 已完成；首轮 S/J 的主要比较和成功判定保持原记录。"
    if status:
        return "探索性 J2 当前状态：" + str(status.get("state", "状态字段缺失")) + "；结果以单独报告为准。"
    return "J2尚无运行状态文件；是否启动由已完成诊断和实际执行记录决定。"


def _write_j2_report(root, status, result, bounds):
    directory = root / "j2"
    models = _models(directory, ("J2",))
    transitions = _csv(directory / "evaluation/transitions.csv")
    budgets = _read(directory / "cf_budget_summary.json")
    tests = _read(directory / "tests.json")
    configuration = _read(directory / "configuration.json")
    measured = bool(result.get("complete") and result.get("models"))
    text = ["# ASTER-CR J2：标签感知证据方向与有限松弛（探索性）", "",
            _j2_state_text(status, result), "",
            "J2由首轮固定b0的线性规划可达性诊断触发，沿用每折16维痕迹和线性修正容量。"
            "本报告独立保存，原首轮 J−S 与 J−R1 的预设检验保留在 [主报告](REPORT.zh.md)。", ""]
    if bounds.get("records"):
        text.append(f"启动依据：原预算下 {bounds['impossible_at_delta_shared_b0']}/{bounds['records']} 个自然首错节点不可达；"
                    f"预算上界扩大到2倍后，{bounds['newly_reachable_with_at_most_double_budget']} 个由不可达变为单节点可达。")
    text += ["", "## 预定修改与证据保证", "",
             r"令 $s_j=y_{r_j}-y_{l_j}$，$\Delta c_j=w^\top(t_{r_j}-t_{l_j})$。"
             r"J2对已知且翻转标签的训练CF对使用 $-\delta\le s_j\Delta c_j\le2\delta$，$\delta=0.2923349$。"
             "损害正确证据方向的修正受δ限制，有利方向允许到2δ；"
             "训练入口要求每个CF标签差为−1或+1。", "",
             r"CF权重 $v_j$ 按C/F类型、类型内来源及来源内配对均衡，并满足 $\sum_jv_j=1$。"
             r"令 $d_j=\ell_{r_j}-\ell_{l_j}$，$Z_{CF}=\max\{\sum_jv_j\operatorname{softplus}(-s_jd_j),10^{-12}\}$，", "",
             r"\[L_{CF}=\frac{\sum_jv_j\operatorname{softplus}[-s_j(d_j+\Delta c_j)]}{Z_{CF}},\qquad"
             r"L_{J2}=L_{\rm task}+\lambda_{AP}L_{AP}+\lambda_{CF}L_{CF}+0.01\|w\|_2^2.\]", "",
             "J2加入证据方向softplus监督，CF权重候选为0.1、1、10；AP权重沿用首轮J每折已选值。"
             "内部仍用同一48/16来源划分，中心化、自然任务监督、正则和两个截距协议保持一致。"
             "所有实际归一化常数和损失项保存在各折J2模型JSON。", "",
             "原有训练逐对绝对修正≤δ的幅度保证变为上述有方向的非对称界。"
             "外折评价同时保留相对于原δ的超预算比例，另外记录J2约束的训练残差；两种口径分别展示。", "",
             "## 完整路径与证据实测", ""]
    if measured:
        rows = []
        for name in ("R1", "S_shared", "J_shared", "J2_shared", "S_independent", "J_independent", "J2_independent"):
            m = result["models"].get(name)
            if m:
                rows.append([name, f"{m.get('natural_path_correct','?')}/{m.get('natural_path_count','?')}",
                    _fmt(m.get("natural_path_accuracy"),100,3), f"{m.get('interference_path_correct','?')}/{m.get('interference_path_count','?')}",
                    _fmt(m.get("interference_path_accuracy"),100,3), _fmt(m.get("ap_vr"),100,3),
                    _fmt(m.get("cf_er"),100,3), _fmt(m.get("cf_correction_rms")), _fmt(m.get("cf_paired_accuracy"),100,3),
                    m.get("cf_direction_degradation_count"), _fmt(m.get("cf_over_budget_rate"),100,3)])
        text += _table(["方法", "自然全对/总数", "自然%", "干扰全对/总数", "干扰%", "AP方差降低%", "CF有向保留%", "CF修正RMS", "CF双端全对%", "方向退化", "超原δ预算%"], rows)
        if budgets:
            text += ["", "J2外折逐对预算检查：", ""]
            text += _table(["CF总对数", "超原δ绝对预算", "有害方向超过δ", "有利方向超过2δ", "超J2非对称界"],
                           [[budgets.get(k, "未完成") for k in ("pairs", "original_budget_exceeded", "harmful_lower_exceeded", "helpful_upper_exceeded", "new_bound_exceeded")]])
        text += ["", "## 探索性配对比较", "",
                 "J2−J、J2−R1、J2−S分别在共享和独立截距下比较。2,000次来源配对自助区间用于描述不确定性。"
                 "这些比较在首轮结果之后开展，全部属于探索性，不纳入首轮正式显著性顺序。", ""]
        rows = []
        for calibration in ("shared", "independent"):
            for reference in ("J", "R1", "S"):
                lhs = "J2_" + calibration
                rhs = reference if reference == "R1" else reference + "_" + calibration
                mm = _comparison(result, lhs, rhs)
                rows.append([lhs + " − " + rhs, _interval(mm.get("natural_path_accuracy")), _interval(mm.get("interference_path_accuracy"))])
        text += _table(["比较（探索性）", "自然路径差", "干扰路径差"], rows)
        changes = _source_changes(_csv(directory / "evaluation/source_metrics.csv"), result)
        changes = [r for r in changes if r[0].startswith("J2_")]
        if changes:
            text += ["", "### 变化的来源数", ""] + _table(["比较", "指标", "改善来源", "下降来源", "相同来源", "有效来源"], changes)
        rr = [r for r in transitions if r.get("comparison", "").startswith("J2_") and r.get("level") in ("natural_path", "interference_path", "pair_CF_both_correct")]
        if rr:
            text += ["", "### 四向错误转移", ""] + _table(["比较", "层级", "错误改对", "正确改错", "保持正确", "保持错误"],
                [[r.get(k) for k in ("comparison","level","wrong_to_right","right_to_wrong","kept_right","kept_wrong")] for r in rr])
    else:
        text.append("真实J2评价未完成，路径、证据指标与配对区间均待实际输出。")
    text += ["", "## 实际优化与全部候选", ""]
    if models:
        rows = []
        for fold, method, model in models:
            losses = model.get("final_losses", {})
            rows.append([fold, model.get("lambda_ap"), model.get("lambda_cf", model.get("lambda_direction")),
                _fmt(losses.get("task")), _fmt(losses.get("weighted_ap")),
                json.dumps({k: v for k,v in losses.items() if "direction" in k or "cf" in k}, ensure_ascii=False),
                _fmt(losses.get("regularization"),digits=7), _fmt(losses.get("total")),
                _fmt(model.get("cf_max_violation", model.get("max_constraint_violation")),digits=10)])
        text += _table(["折", "沿用AP权重", "CF方向权重", "任务项", "加权AP项", "CF方向损失项（实际字段）", "正则项", "总目标", "非对称约束最大违反量"], rows)
        text += ["", "逐折方向归一化与预算保证：", ""]
        details, gradients = [], []
        for fold, _, model in models:
            details.append([fold, _fmt(model.get("cf_direction_scale")), model.get("constraint_count"),
                _fmt(model.get("signed_constraint_lower"),digits=7), _fmt(model.get("signed_constraint_upper"),digits=7),
                _fmt(model.get("cf_max_signed_decrease"),digits=7), _fmt(model.get("cf_max_signed_increase"),digits=7),
                model.get("cf_original_budget_excess_count"), _fmt(model.get("cf_original_budget_max_excess"),digits=7)])
            grad = model.get("gradient_final", {})
            gradients.append([fold, _fmt(grad.get("task_norm")), _fmt(grad.get("weighted_ap_norm")),
                _fmt(grad.get("weighted_cf_direction_norm")), _fmt(grad.get("task_ap_cosine")),
                _fmt(grad.get("task_cf_direction_cosine")), _fmt(grad.get("ap_cf_direction_cosine"))])
        text += _table(["折", "Z_CF", "CF对", "有向下界", "有向上界", "最大有害修正", "最大有利修正", "超原δ对数", "超原δ最大量"], details)
        text += ["", "最终优化梯度：", ""]
        text += _table(["折", "任务范数", "加权AP范数", "加权CF范数", "任务/AP余弦", "任务/CF余弦", "AP/CF余弦"], gradients)
    else:
        text.append("尚无完成的J2模型。")
    candidates = []
    for path in sorted((directory / "selection").glob("fold*/candidates.json")):
        obj = _read(path)
        rows = obj if isinstance(obj, list) else obj.get("candidates", [])
        selection = {} if isinstance(obj, list) else obj.get("selection", {})
        selected = selection.get("lambda_cf", selection.get("lambda_direction")) if isinstance(selection, dict) else selection
        for candidate in rows:
            mm = candidate.get("metrics", {})
            weight = candidate.get("lambda_cf", candidate.get("lambda_direction"))
            candidates.append([path.parent.name, weight, _fmt(mm.get("natural_path_accuracy"),100,3),
                               _fmt(mm.get("interference_path_accuracy"),100,3), _fmt(mm.get("cf_correction_rms")),
                               "是" if weight is not None and weight == selected else "否"])
    if candidates:
        text += [""] + _table(["折", "CF方向权重", "内层自然%", "内层干扰%", "CF修正RMS", "选中"], candidates)
    text += ["", "## 实测结论与下一步", ""]
    if measured:
        versus_j = _conditions(result, "J2_independent", "J_independent")
        versus_r1 = _conditions(result, "J2_independent", "R1")
        for lhs, rhs, label in (("J2_independent", "J_independent", "J2−J，探索性"), ("J2_independent", "R1", "J2−R1，探索性")):
            text.append(_comparison_sentence(result, lhs, rhs, label))
        if versus_j and versus_r1:
            text.append("J2达到本轮探索性的干扰增益与自然非劣条件。下一步冻结方向监督、非对称预算和选型规则，在新来源上确认任务增益及证据代价。")
        else:
            text.append("J2的当前配对结果未同时建立相对J与R1的任务成功。原证据约束确实限制单节点可达性，"
                        "当前方向监督与非对称松弛组合的任务收益尚未通过判定。"
                        "若继续，优先在新训练来源分离方向监督、非对称松弛及其组合的贡献；方法和参数冻结后用新来源确认。"
                        "本轮搜索范围已经结束。")
    else:
        text.append("J2结论等待真实评价完成。首轮已支持优先研究证据约束；新方案的任务收益由本报告的独立探索结果判断。")
    text += ["", "## 测试与执行记录", "", _test_summary(tests, "j2/tests.json"), ""]
    execution = [("状态", status.get("state", "未记录"))]
    for key, label in (("code_commit", "执行提交"), ("seconds", "执行时长（秒）"),
                       ("complete_fits", "完成拟合数"), ("gpu_devices", "GPU编号"),
                       ("peak_gpu_memory_gib", "峰值显存GiB"), ("original_study_replay", "首轮结果重放一致"),
                       ("exploratory", "探索性"), ("independent_confirmation", "新来源独立确认")):
        if key in status:
            execution.append((label, status[key]))
    for key, label in (("seed", "随机种子"), ("lambda_cf_candidates", "CF候选权重"),
                       ("candidate_configurations", "候选配置数"), ("dimensions", "线性痕迹维数")):
        if key in configuration:
            execution.append((label, configuration[key]))
    text += _table(["字段", "实际记录"], execution)
    text += ["", "本轮五折来源已被历史实验与首轮分析使用。J2作为后续探索保留全部候选与逐例输出，新来源独立确认尚未完成。", "",
             "运行状态：[j2/status.json](j2/status.json)；实际模型与校准：[j2/models/](j2/models/)；"
             "指标和逐例预测：[j2/evaluation/metrics.json](j2/evaluation/metrics.json)。", ""]
    report = root / "J2_REPORT.zh.md"
    report.write_text("\n".join(text), encoding="utf-8")
    return dict(report=str(report), complete=measured)


def _plot(result, root):
    if not result.get("models"):
        return dict(state="not_run", reason="No measured evaluation results; no placeholder figure")
    available = any(m.get("natural_path_accuracy") is not None or m.get("interference_path_accuracy") is not None
                    for m in result["models"].values())
    if not available:
        return dict(state="not_run", reason="No measured path results; no placeholder figure")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return dict(state="blocked", reason=str(error))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for ax, calibration, title in ((axes[0, 0], "independent", "A  Independent intercepts"),
                                   (axes[0, 1], "shared", "B  Shared intercept")):
        names = [name for name in ("R0", "R1", "R2_" + calibration, "S_" + calibration, "J_" + calibration)
                 if name in result["models"]]
        for metric, label, marker, color in (("natural_path_accuracy", "Natural", "o", "#2369a1"),
                                              ("interference_path_accuracy", "Interference", "s", "#c45d22")):
            x, y = [], []
            for i, name in enumerate(names):
                value = result["models"][name].get(metric)
                if value is not None:
                    x.append(i)
                    y.append(100 * value)
            if y:
                ax.plot(x, y, marker=marker, linewidth=1.2, color=color, label=label)
        ax.set_xticks(range(len(names)), [name.split("_")[0] for name in names])
        ax.set_ylabel("Complete-path accuracy (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    ordered = []
    for calibration in ("independent", "shared"):
        for left, right in (("J", "S"), ("J", "R1"), ("J", "R2"), ("S", "R1")):
            lhs, rhs = left + "_" + calibration, right if right == "R1" else right + "_" + calibration
            values = _comparison(result, lhs, rhs)
            if values:
                ordered.append((left + "-" + right + " (" + calibration + ")", values, calibration))
    for ax, metric, title in ((axes[1, 0], "interference_path_accuracy", "C  Interference-path difference"),
                              (axes[1, 1], "natural_path_accuracy", "D  Natural-path difference")):
        for i, (_, metrics, calibration) in enumerate(ordered):
            interval = metrics.get(metric, {})
            if any(interval.get(key) is None for key in ("estimate", "low", "high")):
                continue
            center, lo, hi = (100 * interval[key] for key in ("estimate", "low", "high"))
            color = "#2369a1" if calibration == "independent" else "#777777"
            # Percentile intervals can exclude the observed estimate. Draw the
            # interval directly, avoiding a negative asymmetric-error width.
            ax.hlines(i, lo, hi, color=color, linewidth=1.5)
            ax.plot([center], [i], "o", color=color, markersize=4)
        ax.axvline(0, color="#333333", linewidth=0.8)
        if metric == "natural_path_accuracy":
            ax.axvline(-0.5, color="#b34242", linestyle="--", linewidth=0.9, label="-0.5 pp")
            ax.legend(frameon=False)
        ax.set_yticks(range(len(ordered)), [name for name, _, _ in ordered])
        ax.invert_yaxis()
        ax.set_xlabel("Difference (percentage points), source-paired 95% CI")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.15)
    directory = root / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        path = directory / ("summary." + suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(root)))
    plt.close(fig)
    return dict(state="complete", files=files)


def generate_report(output_dir):
    """Generate REPORT.zh.md; return artifact paths and actual completion status."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = _read(root / "evaluation/metrics.json")
    status, config = _read(root / "status.json"), _read(root / "configuration.json")
    verification, tests = _read(root / "verification.json"), _read(root / "tests.json")
    extended_tests = _read(root / "extended_tests.json")
    transitions = _csv(root / "evaluation/transitions.csv")
    first_errors = _jsonl(root / "evaluation/natural_first_error.jsonl")
    first_source = _csv(root / "evaluation/natural_first_error_by_source.csv")
    source_metrics = _csv(root / "evaluation/source_metrics.csv")
    bounds = _read(root / "diagnostics/correction_bounds_summary.json")
    bound_records = _jsonl(root / "diagnostics/correction_bounds.jsonl")
    j2_status = _read(root / "j2/status.json")
    j2_result = _read(root / "j2/evaluation/metrics.json")
    models = _models(root)
    measured = bool(result.get("complete") and result.get("models"))
    audit_path = root / "HISTORICAL_AUDIT.zh.md"
    audit = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    audit_conclusion = next((line for line in audit.splitlines() if line.startswith("**结论")), "历史等价性审计尚未保存。")
    text = ["# ASTER-CR 第三模块联合监督与决策校准实验", "",
            "**问题一：加入任务监督是否提高了正确率？**", "",
            _comparison_sentence(result, "S_independent", "R1", "S − R1，独立截距"), "",
            _comparison_sentence(result, "J_independent", "R2_independent", "J − R2，独立截距"), "",
            "**问题二：A/P目标是否提供额外贡献？**", "",
            _comparison_sentence(result, "J_independent", "S_independent", "首要比较 J − S，独立截距"), "",
            "**问题三：联合方法是否超过独立截距校准？**", "",
            _comparison_sentence(result, "J_shared", "R1", "共享截距 J − R1，探索性"), "",
            _comparison_sentence(result, "J_independent", "R1", "独立截距 J − R1，第二顺序比较"), "",
            "**问题四：证据保护是否仍成立？**", ""]
    j = result.get("models", {}).get("J_independent", {})
    if j:
        text.append("J 外折 C/F 有向响应保留率为 " + _fmt(j.get("cf_er"), 100, 3) +
                    "%，修正 RMS 为 " + _fmt(j.get("cf_correction_rms")) +
                    " logit，双端全对率为 " + _fmt(j.get("cf_paired_accuracy"), 100, 3) +
                    "%，方向退化 " + str(j.get("cf_direction_degradation_count", "未完成")) +
                    " 对，超预算比例 " + _fmt(j.get("cf_over_budget_rate"), 100, 3) + "%。")
        constraint_ok = len(models) == 10 and all(m.get("cf_max_violation", float("inf")) <= m.get("tolerance", 1e-7)
                                                   for _, _, m in models)
        text.append("全部十个折内 S/J 模型满足训练逐对预算容差。" if constraint_ok else "训练约束完成情况见逐折残差表。")
        for r in transitions:
            if r.get("comparison") in ("J_independent minus R1", "J_independent minus S_independent") and r.get("level") in ("natural_path", "interference_path", "pair_CF_both_correct"):
                text.append(f"{r['comparison']}，{r['level']}：错误改对 {r['wrong_to_right']}，正确改错 {r['right_to_wrong']}，"
                            f"保持正确 {r['kept_right']}，保持错误 {r['kept_wrong']}。")
    else:
        text.append("真实 S/J 逐对评价及错误转移未完成，证据保护结论待实测。历史 E1 指标单独保存在历史审计中。")
    text += ["", "**问题五：如果没有成功，下一步最值得修改什么？**", ""]
    text += [paragraph + "\n" for paragraph in _diagnosis(result, models, first_errors, bounds, j2_status, j2_result)]
    text += ["## 实验状态与结果资格", "",
             "本轮真实五折对照已完成。" if measured else "**真实五折对照未完成。下文无 S/J 实测数值的单元均记为未完成。**", "",
             "R0/R1/R2 复用或重放归档模型。S/J 是原 80 来源上的新增方法实验；各外折验证来源未进入对应折的拟合和内部选型。"
             "这些来源已有历史使用记录。本轮结果属于历史研究来源上的重新评价；方案冻结后的新来源承担独立确认。", "",
             "内部选择条件于由全部外折训练来源得到的固定 M1/M2 表示。每折仅比较 AP 权重 0.1、1、10；"
             "S 固定为 0。主要校准条件预先指定为独立截距，共享截距另列探索性结果。", "",
             "运行状态见 [status.json](status.json)，配置及版本见 [configuration.json](configuration.json)，"
             "来源与分数重建核验见 [verification.json](verification.json)。", ""]
    if status:
        text += ["```json", json.dumps(status, ensure_ascii=False, indent=2), "```", ""]
    text += ["## 路径正确率与配对区间", ""]
    if measured:
        method_rows = []
        for name in ("R0", "R1", "R2_shared", "S_shared", "J_shared", "R2_independent", "S_independent", "J_independent"):
            m = result["models"].get(name)
            if m:
                method_rows.append([name,
                    f"{m.get('natural_path_correct','?')}/{m.get('natural_path_count','?')}", _fmt(m.get("natural_path_accuracy"),100,3),
                    f"{m.get('interference_path_correct','?')}/{m.get('interference_path_count','?')}", _fmt(m.get("interference_path_accuracy"),100,3)])
        text += _table(["方法", "自然正确/总数", "自然正确率%", "干扰正确/总数", "干扰正确率%"], method_rows)
        text += ["", "每次重采样重新合并来源内的路径计数；不平均折级比率。配对区间使用完全相同的 2,000 组来源抽样，种子 20261111。", ""]
        rows = []
        for name, c in result.get("comparisons", {}).items():
            mm = c.get("metrics", {})
            rows.append([name, c.get("role"), _interval(mm.get("natural_path_accuracy")), _interval(mm.get("interference_path_accuracy"))])
        text += _table(["比较", "资格", "自然路径差", "干扰路径差"], rows)
    else:
        text += ["未执行或未完成。历史基线数值不填入本轮新增对照表。"]
    text += ["", "显著性判断固定顺序为独立截距 J−S，再到 J−R1。每步同时要求干扰差值 95% CI 下限严格大于 0、"
             "自然差值下限不低于 −0.5 个百分点；首步未通过时，第二步保留估计与区间。"
             "共享截距、J−R2、S−R1及各候选结果均为探索性，不追加正式显著性声明。", ""]
    for stage in result.get("success", {}).get("sequence", []):
        text.append(f"- {stage['comparison']}：自然非劣={stage['natural_noninferiority']}，干扰增益={stage['interference_superiority']}，"
                    f"通过预设顺序判断={stage['passes_predeclared_sequence']}。")
    text += ["", "## 配对机制与证据指标", ""]
    if measured:
        rows = []
        for name, m in result["models"].items():
            if name == "raw":
                continue
            rows.append([name, _fmt(m.get("ap_score_change_rms")), _fmt(m.get("ap_vr"),100,3),
                         _fmt(m.get("cf_er"),100,3), _fmt(m.get("cf_correction_rms")),
                         _fmt(m.get("cf_paired_accuracy"),100,3), m.get("cf_direction_degradation_count","未完成"),
                         _fmt(m.get("cf_over_budget_rate"),100,3)])
        text += _table(["方法", "AP差RMS", "AP方差降低%", "CF有向保留%", "CF修正RMS", "CF双端全对%", "方向退化数", "外折超预算%"], rows)
    else:
        text.append("未完成。")
    text += ["", "逐对结果见 [pair_predictions.jsonl](evaluation/pair_predictions.jsonl)；逐来源指标见 "
             "[source_metrics.csv](evaluation/source_metrics.csv)。三个层次分别报告，不构造综合分数。", "",
             "## 实际训练目标与归一化", "",
             r"令自然训练节点权重 $a_i=1/n$，$\mu_{\rm train}=\sum_i a_i t_i$，$c_i=w^\top(t_i-\mu_{\rm train})$。"
             "自然监督与两个截距拟合使用同一折的原始自然节点，所有训练标签来自当前训练来源。", "",
             r"$b_0=\arg\min_b\sum_i a_i\operatorname{BCE}(\ell_i+b,y_i)$，训练修正器时固定该值。", "",
             r"$Z_{\rm task}=\max\{\sum_i a_i\operatorname{BCE}(\ell_i+b_0,y_i),10^{-12}\}$，"
             r"$L_{\rm task}=\sum_i a_i\operatorname{BCE}(\ell_i+b_0+c_i,y_i)/Z_{\rm task}$。", "",
             r"AP权重为 $u_j=1/(K\,|S_k|\,n_{ks})$：类型等权、每种类型内来源等权、同类型同来源的配对等权，"
             r"$\sum_j u_j=1$，A/P均存在时各占0.5。令 $d_j=\ell_{r_j}-\ell_{l_j}$，"
             r"$Z_{AP}=\max\{\sum_j u_j d_j^2,1\}$，"
             r"$L_{AP}=\sum_j u_j[d_j+w^\top(t_{r_j}-t_{l_j})]^2/Z_{AP}$。", "",
             r"$L=L_{\rm task}+\lambda_{AP}L_{AP}+0.01\|w\|_2^2$，"
             r"对全部训练C/F配对施加 $|w^\top(t_r-t_l)|\leq0.2923349$。S取 $\lambda_{AP}=0$；J的核心新增项为AP损失。", "",
             "这里的0.01作用于分别归一化后的损失。R2保留原AP权重总和4,887的历史目标与参数；"
             "它的历史正则相对尺度见等价性审计。S/J使用完全相同的正则尺度。", "",
             "训练结束后，共享方案使用原b0；独立方案在相同自然训练节点上另拟合各自截距。"
             "原E1经中心化后的独立截距为 `b_old + mu_train @ w`，从而重建原完整分数。", ""]
    if models:
        loss_rows, gradient_rows, constraint_rows, calibration_rows = [], [], [], []
        for fold, method, model in models:
            losses = model.get("final_losses", {})
            grad = model.get("gradient_final", {})
            loss_rows.append([fold, method, model.get("lambda_ap"), _fmt(model.get("task_scale")), _fmt(model.get("ap_scale")),
                              _fmt(losses.get("task_raw_bce")), _fmt(losses.get("ap_raw_mse")),
                              _fmt(losses.get("task")), _fmt(losses.get("weighted_ap")), _fmt(losses.get("regularization"),digits=7), _fmt(losses.get("total"))])
            gradient_rows.append([fold, method, _fmt(model.get("gradient_initial",{}).get("task_ap_cosine")),
                                  _fmt(grad.get("task_ap_cosine")), _fmt(grad.get("task_norm")),
                                  _fmt(grad.get("weighted_ap_norm")), _fmt(grad.get("regularization_norm")),
                                  _fmt(model.get("loss_changes",{}).get("task")), _fmt(model.get("loss_changes",{}).get("weighted_ap"))])
            constraint_rows.append([fold, method, model.get("constraint_count"), _fmt(model.get("cf_max_abs"),digits=8),
                                    _fmt(model.get("cf_max_violation"),digits=10), model.get("cf_active_count"), model.get("iterations")])
            calibration_rows.append([fold, method, model.get("natural_count"), len(model.get("train_sources",[])),
                                     _fmt(model.get("b_shared"),digits=8), _fmt(model.get("b_independent"),digits=8),
                                     _fmt(sum(model.get("task_weights",[]))), _fmt(sum(model.get("ap_weights",[])))])
        text += _table(["折", "方法", "AP权重", "任务归一化常数", "AP归一化常数", "原BCE", "原AP MSE", "归一化任务项", "加权AP项", "正则项", "总目标"], loss_rows)
        text += ["", "### 梯度与损失变化", ""]
        text += _table(["折", "方法", "初始余弦", "最终余弦", "任务梯度范数", "加权AP梯度范数", "正则梯度范数", "任务损失变化", "加权AP损失变化"], gradient_rows)
        text += ["", "### 训练C/F约束残差", "", "逐对残差为 `abs(delta_c) − 0.2923349`，最大违反量取其正部；求解容差1e−7。完整逐对残差保存在各折模型JSON。", ""]
        text += _table(["折", "方法", "CF约束数", "最大绝对修正", "最大违反量", "活跃约束数", "迭代数"], constraint_rows)
        text += ["", "### 截距、训练样本与权重", ""]
        text += _table(["折", "方法", "自然节点", "训练来源", "共享b0", "独立bw", "任务权重和", "AP权重和"], calibration_rows)
        text.append("\n各节点和配对的ID、权重、校准来源、w与中心化均值保存在 `models/foldK/{S,J}.json`。")
    else:
        text.append("尚无已完成模型；实际优化值、梯度和约束残差未生成。")
    text += ["", "## 全部候选与选择", ""]
    candidate_rows = []
    for path in sorted((root / "selection").glob("fold*/candidates.json")):
        obj = _read(path, [])
        candidates = obj if isinstance(obj, list) else obj.get("candidates", [])
        selection = {} if isinstance(obj, list) else obj.get("selection", obj.get("selected", {}))
        selected = selection.get("lambda_ap") if isinstance(selection, dict) else selection
        for candidate in candidates:
            metrics = candidate.get("metrics", {})
            candidate_rows.append([path.parent.name, candidate.get("method", "J"), candidate.get("lambda_ap"),
                _fmt(metrics.get("natural_path_accuracy"),100,3), _fmt(metrics.get("interference_path_accuracy"),100,3),
                metrics.get("cf_direction_degradation_count","未完成"), _fmt(metrics.get("cf_over_budget_rate"),100,3),
                _fmt(metrics.get("cf_correction_rms")), "固定S" if candidate.get("method") == "S" else "是" if selected == candidate.get("lambda_ap") else "否"])
    text += (_table(["折", "方法", "AP权重", "内层自然%", "内层干扰%", "方向退化", "CF超预算%", "CF修正RMS", "选中"], candidate_rows)
             if candidate_rows else ["尚无完成的真实候选记录。"])
    text += ["", "选择先控制相对R1自然路径下降不超过0.5个百分点，再比较干扰正确率，随后比较证据损害和较小AP权重。"
             "所有候选均保留；外折16来源不参与选型。确定性求解器的随机种子和内部划分见配置，未制造多随机种子波动。", "",
             "## 错误转移与首错边界", ""]
    relevant = [r for r in transitions if r.get("level") in ("natural_path", "interference_path", "pair_CF_both_correct") and
                r.get("comparison") in result.get("comparisons", {})]
    if relevant:
        text += _table(["比较", "层级", "错误改对", "正确改错", "保持正确", "保持错误", "总数"],
                       [[r.get(k) for k in ("comparison","level","wrong_to_right","right_to_wrong","kept_right","kept_wrong","total")] for r in relevant])
    else:
        text.append("真实错误转移统计未完成。")
    for comparison in ("S_independent minus R1", "J_independent minus R1", "J_independent minus S_independent"):
        evidence = _source_error_evidence(first_errors, comparison)
        if evidence:
            text += ["", f"{comparison}：可定位首错 {evidence['total']} 个，节点修复 {evidence['repaired']} 个，"
                     f"修正前平均绝对分数 {_fmt(evidence['before_abs_mean'])}，平均绝对分数变化 {_fmt(evidence['abs_shift_mean'])} logit。"]
    text += ["", "[逐例首错及前后margin](evaluation/natural_first_error.jsonl)、"
             "[逐来源首错](evaluation/natural_first_error_by_source.csv)、[逐来源四向错误转移](evaluation/source_transitions.csv)均独立保存。"
             f"已读取逐来源首错记录 {len(first_source)} 行。", ""]
    source_changes = _source_changes(source_metrics, result)
    if source_changes:
        text += ["### 各来源正确率的变化", ""]
        text += _table(["比较", "指标", "改善来源", "下降来源", "相同来源", "有效来源"], source_changes) + [""]
    text += _bound_section(bounds, bound_records)
    text += [
             "## 历史等价性与复用", "", audit_conclusion, "",
             "完整对应关系、历史分数及路径映射见 [HISTORICAL_AUDIT.zh.md](HISTORICAL_AUDIT.zh.md)。"
             "历史监督读出、监督痕迹修正和99维非线性方法不作为当前S/J的等价对照。", "",
             "## 测试、版本与运行限制", ""]
    for title, obj in (("自动化测试", tests), ("真实数据重建与来源核验", verification)):
        text += [f"### {title}", ""]
        if title == "自动化测试":
            text.append(_test_summary(obj, "tests.json"))
            if extended_tests:
                text += ["", "扩展测试（包含首轮与J2测试模块）：" + _test_summary(extended_tests, "extended_tests.json")]
        elif obj:
            text += ["```json", json.dumps(obj, ensure_ascii=False, indent=2), "```"]
        else:
            text.append("未执行或结果文件尚未生成。")
        text.append("")
    version_fields = [(key, config[key]) for key in ("code_commit", "git_commit", "base_commit", "branch", "data_version", "model_version", "seed", "inner_seed", "bootstrap_seed", "gpus", "gpu_devices", "gpu_memory_gib") if key in config]
    if "code_commit" in status and "code_commit" not in config:
        version_fields.append(("code_commit", status["code_commit"]))
    historical_config = config.get("model_data_metadata", {})
    version_fields.extend(("historical_" + key, historical_config[key]) for key in ("data_commit", "model", "model_dtype", "frozen_model") if key in historical_config)
    if version_fields:
        text += _table(["配置字段", "值"], version_fields) + [""]
    missing = _read(root / "missing_assets.json", [])
    if missing:
        text += ["### 缺失或未确认的资产", "", "```json", json.dumps(missing, ensure_ascii=False, indent=2), "```", ""]
    figures = _plot(result if measured else {}, root)
    text += ["## 图表与结论", ""]
    if figures["state"] == "complete":
        text += ["![路径正确率与配对差值](artifacts/summary.png)", "",
                 "[PDF图](artifacts/summary.pdf)。上排区分共享和独立截距，下排显示来源配对95%区间；自然路径虚线对应−0.5个百分点。", ""]
    else:
        text += ["图表状态：" + ("没有真实评价结果，未生成占位图。" if figures["state"] == "not_run" else figures.get("reason", "未完成")), ""]
    primary = _conditions(result, "J_independent", "S_independent")
    natural_interval = _comparison(result,"J_independent","S_independent").get("natural_path_accuracy",{})
    task_judgements = []
    for lhs, rhs, label in (("S_independent", "R1", "S−R1"),
                            ("J_independent", "R2_independent", "J−R2"),
                            ("J_independent", "R1", "J−R1")):
        decision = _conditions(result, lhs, rhs)
        task_judgements.append(label + ("未完成" if decision is None else "满足双条件" if decision else "未满足双条件"))
    conclusion = [
        ["任务收益", "；".join(task_judgements) + "。正式顺序判断见上文"],
        ["AP独立贡献", "未完成" if primary is None else "J−S满足双条件" if primary else "J−S未满足双条件"],
        ["自然路径非劣（主要J−S）", "未完成" if natural_interval.get("low") is None else "满足" if natural_interval["low"] >= -0.005 else "未满足"],
        ["进入J2", _j2_state_text(j2_status, j2_result) if j2_status or j2_result else
                    "可达性诊断支持进入标签感知证据约束实验，等待执行记录" if bounds.get("records") else "由真实诊断决定，当前未完成"],
        ["结果资格", "R0/R1/R2历史复评；S/J为已使用80来源上的新增实验；新来源独立验证尚未开展"]]
    text += _table(["问题", "结论"], conclusion) + [""]
    j2_report = None
    if j2_status or j2_result or bounds.get("records"):
        j2_report = _write_j2_report(root, j2_status, j2_result, bounds)
        text += ["## J2探索性后续", "", _j2_state_text(j2_status, j2_result), "",
                 "J2保留16维线性容量，增加标签感知证据方向监督，并将训练C/F修正界改为有利方向≤2δ、"
                 "有害方向≤δ。原绝对幅度≤δ的训练保证据此改变；外折超原预算率和方向退化继续独立记录。", ""]
        if j2_result.get("complete"):
            text += [_comparison_sentence(j2_result, "J2_independent", "J_independent", "J2−J，探索性"), "",
                     _comparison_sentence(j2_result, "J2_independent", "R1", "J2−R1，探索性"), ""]
            if _conditions(j2_result,"J2_independent","J_independent") and _conditions(j2_result,"J2_independent","R1"):
                text.append("后续优先冻结J2方案，用新来源验证任务与证据指标。")
            else:
                text.append("原约束限制可达性，当前J2松弛组合尚未建立任务成功。后续优先在新训练来源分离方向监督与非对称松弛的贡献，"
                            "冻结方案后用新来源确认；本轮首要机制比较与结论保持不变，搜索范围已经结束。")
        text += ["", "全部J2候选、训练目标、证据保证变化、错误转移和结论见 [J2_REPORT.zh.md](J2_REPORT.zh.md)。", ""]
    report = root / "REPORT.zh.md"
    report.write_text("\n".join(text), encoding="utf-8")
    return dict(report=str(report), real_results_available=measured, figures=figures, j2_report=j2_report)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.output_dir), ensure_ascii=False))
