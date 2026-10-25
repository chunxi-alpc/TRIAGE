"""Render Chinese research reports from actual numeric results."""
import argparse,json
from pathlib import Path
from reproduce import ROOT
NAMES={"static":"温度校准静态融合","dynamic":"动态融合","hard_anchor":"硬锚融合","residual":"注意力残差","numeric_null":"零强度数值漂移","option_rotation":"动作循环错位","sign_flip":"病例符号翻转","reverse":"反向残差"}
def render(source,output,fresh=False):
    data=json.loads(source.read_text())
    lines=["# ASTER 实际复算与新前向报告","",
      "状态："+("两模型各 800 条新前向完成；开发集结果，不是独立确认测试。" if fresh else "历史脚本已在115服务器实际执行；新GPU前向另见 fresh_report.zh.md（存在且状态完成时有效）。"),"",
      "## 代码与统计修复","",
      "- 原 reproduce.py 与 P3 脚本实际跑通，原报告主要准确率得到复现。",
      "- 修复 P3 分种子退化/纠正未累计却输出零值的问题；新增逐树统计。",
      "- 历史后半层干预减去全层零强度 logits，不能保证数值路径相同。历史网格保留并明确标警；新增比较仅用全层残差，新前向另采全层和后半层零强度。",
      "- metric 支持列表且检查预测数量；补全逐折指标与逐例预测。",
      "- 新软融合各家固定相同27组 beta/t0/ts；动态幅度、残差通道与强度预先固定。硬锚使用9个beta×3个支持温度。相同候选数不等于完全相同有效复杂度。",
      "- 原模型单纯正温度缩放不改变argmax准确率；因此强输出层基线是温度校准的概率融合，同时保留原模型、直接映射、固定0.75融合与硬锚。",
      "- 候选网格在历史结果已知后设计；即使新前向也只是开发数据复核，不构成新独立样本。","",
      "## 按树留一结果","",
      "| 模型 | 方法 | 准确率 | 相对原答案退化率 | 纠正率 |","|---|---|---:|---:|---:|"]
    for m,x in data["models"].items():
        for k,v in x.items():
            if isinstance(v,dict) and "metrics" in v:
                z=v["metrics"];lines.append(f'| {m} | {NAMES.get(k,k)} | {100*z["accuracy"]:.3f}% | {100*z["harm_rate"]:.3f}% | {100*z["correction_rate"]:.3f}% |')
        for k,z in x["baselines"].items():
            lines.append(f'| {m} | {k} | {100*z["accuracy"]:.3f}% | {100*z["harm_rate"]:.3f}% | {100*z["correction_rate"]:.3f}% |')
    lines+=["","## 配对变化",""]
    for m,x in data["models"].items():
        for k in ("dynamic","hard_anchor","residual"):
            if k not in x["paired_vs_static"]:continue
            z=x["paired_vs_static"][k];ci=z["tree_cluster_percentile_ci95"]["accuracy_delta"]
            lines.append(f'- {m} {NAMES[k]}相对静态融合：纠正 {z["corrected_vs_comparator"]} 条、退化 {z["regressed_vs_comparator"]} 条，净准确率差 {100*z["accuracy_delta"]:+.3f} 个百分点；描述性树簇区间 [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}]。')
        if "matched_residual" in x:
            z=x["matched_residual"];ci=z["tree_cluster_percentile_ci95"]["accuracy_delta"]
            lines.append(f'- {m} 同一融合参数加残差相对去掉残差：纠正 {z["corrected_vs_comparator"]}、退化 {z["regressed_vs_comparator"]}，区间 [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] 个百分点。')
            for key,z in x["zero_control"].items():
                lines.append(f'- {m} {key} 相对原路径改变 {z["changed_predictions"]}/800 个答案，最大选项logit误差 {z["maximum_option_logit_error"]:.6g}。')
    lines+=["","各树配对纠正/退化、四变体准确率、同参数数值漂移/错位/符号/反向对照及逐例预测见对应JSON。保留所有负结果，不按结果排除病例。",
      "","## 不确定性与独立验证","",
      "按指南树整簇重采样10000次，保留每病例四变体。仅4个开发树，区间只作描述，不含事后方法设计的不确定性。开发集选参不能证明统计稳健优势。",
      "已检查既有AST​​ER、MedGUIDE及扩展病例资源：55棵来源树中53棵出现在既有记录；另外 bre_19、cns_23 在源病例表中均无病例。未找到可直接使用的独立新树标注集。完整范围见 resource_inventory.json。",
      "frozen_candidate.json 冻结两模型简单输出融合：beta=0.75、原分布温度1、支持温度0.25。仅根据开发集选择；尚未完成新树验证。P3只保留探索性复算，不作新确认。",
      "","## 标签与研究范围","",
      "NC 标签来自原数据路径对齐；AC/IC/SW 是实验反事实规则标签，不是医学处置建议。已有人工审核未完成，不宣称临床安全或完整条件适用性已验证。",
      "","## 复现","",
      "所有执行均在115服务器 gcx261014 环境。运行同目录 test_reproduction.py、reproduce.py、p3_hard_anchor_exploratory.py；balanced_analysis.py 默认使用仓库匿名数值数据。新前向用 fresh_forward.py --name qwen/llama --root 私有运行目录，分别设置 CUDA_VISIBLE_DEVICES=0/1。",
      "balanced_analysis.py --fresh-root 私有运行目录 --output balanced_fresh.json 生成新前向分析。历史来源、权重config、软件版本和开销保存在运行manifest与execution.json。"]
    if fresh:
        execution=json.loads((ROOT/"execution.json").read_text())
        lines+=["","## 实际计算成本",""]
        for m,z in execution["models"].items():
            lines.append(f'- {m}：{z["completed"]}条，墙钟 {z["elapsed_seconds"]/60:.2f} 分钟，峰值分配显存 {z["peak_GiB"]:.2f} GiB，GPU小时 {z["gpu_hours"]:.3f}。')
        lines+=["时间包括加载、真实GPU对照、条件评估和所有干预；条件输入完全相同时缓存复用。GPU小时为该模型占用卡的墙钟估计，不是计费金额。"]
    output.write_text("\n".join(lines)+"\n")
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--fresh",action="store_true");x=a.parse_args()
    render(ROOT/("balanced_fresh.json" if x.fresh else "balanced_cached.json"),ROOT/("fresh_report.zh.md" if x.fresh else "REVIEW.zh.md"),x.fresh)
