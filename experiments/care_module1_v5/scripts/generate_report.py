"""Generate a Chinese report strictly from saved run status and evaluated metrics."""
from __future__ import annotations
import argparse, collections, json, pathlib

V4 = {
    "direct": {"correct": 136, "n": 192, "accuracy": 0.7083333333333334, "macro_f1": 0.7041353837656845,
               "known_evidence_correct": 87, "known_evidence_n": 137, "joint_correct": 81, "pair_correct": 22,
               "pair_n": 32, "seconds": 79.19770555506693},
    "structured_no_guard": {"correct": 140, "n": 192, "accuracy": 0.7291666666666666, "macro_f1": 0.7471625489427929,
               "known_evidence_correct": 56, "known_evidence_n": 137, "joint_correct": 68, "pair_correct": 25,
               "pair_n": 32, "seconds": 672.9351412480464},
    "structured": {"correct": 140, "n": 192, "accuracy": 0.7291666666666666, "macro_f1": 0.7672781066731057,
               "known_evidence_correct": 68, "known_evidence_n": 137, "joint_correct": 93, "pair_correct": 23,
               "pair_n": 32, "seconds": 1331.0105511700967},
}

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def pct(x): return "未计算" if x is None else f"{x*100:.2f}%"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-status", type=pathlib.Path, required=True)
    p.add_argument("--metrics", type=pathlib.Path, required=True)
    p.add_argument("--errors", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a=p.parse_args(); status=read(a.run_status) or {}; metrics=read(a.metrics) or {}
    methods=metrics.get("methods", {})
    lines=["# CARE 模块一 V5 实验报告", "", "报告日期：2026-10-18", "",
      "本报告只记录服务器实际状态与已保存输出。V4 的 192 条回归样本已用于方法设计，因此本轮属于探索性回归，不是独立验证。",
      "", "## 实验状态", "",
      f"- 状态：`{status.get('status', 'not_started')}`。",
      f"- 服务器 GPU：{status.get('gpu_name', '未进入模型加载')}; 设备 {status.get('device', 'cuda:6')}。",
      f"- 样本/规则：192 条样本、16 条规则、6 类场景；共 32 个原始合成病例，其中 160 条为控制文本变体。",
      f"- 模型：`microsoft/phi-4` + `majdabd33/trialmatchai-phi4-reasoning-lora`；证据模型 `cnut1648/biolinkbert-mednli`。",
      f"- 实际总耗时：{status.get('elapsed_seconds', '未运行')} 秒（TrialMatchAI 加载 {status.get('trialmatch_load_seconds', '未采集')} 秒，BioLinkBERT 加载 {status.get('mednli_load_seconds', '未采集')} 秒）；峰值显存：{status.get('peak_allocated_gib', '未采集')} GiB allocated / {status.get('peak_reserved_gib', '未采集')} GiB reserved。",
      f"- 错误摘要：{status.get('error_type') or '无'}：{status.get('error') or '无'}。",
      "", "## V4 已保存对照", "",
      "| 方法 | 判定正确 | 准确率 | 宏 F1 | 已知样本证据准确率 | 判定和证据同时正确 | 反事实成对正确 | 推理秒数 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names={"direct":"Qwen3 直接判定", "structured_no_guard":"Qwen3 结构化不核验", "structured":"Qwen3 结构化加核验"}
    for key,row in V4.items():
        lines.append(f"| {names[key]} | {row['correct']}/{row['n']} | {pct(row['accuracy'])} | {row['macro_f1']:.4f} | {row['known_evidence_correct']}/{row['known_evidence_n']} ({pct(row['known_evidence_correct']/row['known_evidence_n'])}) | {row['joint_correct']}/{row['n']} ({pct(row['joint_correct']/row['n'])}) | {row['pair_correct']}/{row['pair_n']} | {row['seconds']:.1f} |")
    if methods:
        lines += ["", "## V5 实际指标", "", "| 方法 | 判定正确 | 准确率 | 宏 F1 | 已知证据定位 | 联合正确率 | 反事实成对准确率 | 平均秒/样本 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key,label in (("trialmatch_model_only","TrialMatchAI 原始条件判定 + 三值逻辑"),
                          ("trialmatch","TrialMatchAI + 确定性数值比较 + 三值逻辑"),
                          ("combined","TrialMatchAI + BioLinkBERT-MedNLI + 确定性数值比较"),
                          ("qwen_v4","V4 已存 Qwen3 直接判定")):
            row=methods.get(key,{})
            if row.get("status") == "complete":
                elapsed=row.get("mean_elapsed_seconds")
                elapsed_text=f"{elapsed:.3f}" if elapsed is not None else "未计算"
                evidence_text = ("未生成句子级证据" if not row.get("evidence_available") else
                    f"{row.get('known_evidence_correct')}/{row.get('known_evidence_n')} ({pct(row.get('known_evidence_accuracy'))})")
                joint_text = ("不适用" if row.get("joint_accuracy") is None else
                    f"{row.get('joint_correct')}/{row.get('n')} ({pct(row.get('joint_accuracy'))})")
                lines.append(f"| {label} | {row.get('correct')}/{row.get('n')} | {pct(row.get('accuracy'))} | {row.get('macro_f1',0):.4f} | {evidence_text} | {joint_text} | {row.get('counterfactual_pairs_correct')}/{row.get('counterfactual_pair_n')} ({pct(row.get('counterfactual_pair_accuracy'))}) | {elapsed_text} |")
                if key in {"trialmatch_model_only", "trialmatch", "combined"}:
                    cats=row.get("error_categories", {})
                    lines.append(f"  - {label} 错误计数：已知误判未知 {cats.get('known_to_unknown',0)}；未知误判已知 {cats.get('unknown_to_known',0)}；T→F 错误拒绝 {cats.get('false_rejection_T_to_F',0)}；F→T 错误接受 {cats.get('false_acceptance_F_to_T',0)}；已知样本证据错误 {cats.get('evidence_error_known',0)}。")
                    if row.get("by_variant"):
                        detail="；".join(f"{variant} {value['correct']}/{value['n']}，预测未知 {value['unknown_predictions']}" for variant,value in sorted(row["by_variant"].items()))
                        lines.append(f"  - 场景表现：{detail}。")
            else:
                lines.append(f"| {label} | 未运行/不完整 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 |")
        paired=metrics.get("paired_trialmatch_vs_combined")
        if paired:
            lines += ["", f"加入 BioLinkBERT 后的成对变化：纠正 {paired['combined_rescued']} 条、引入错误 {paired['combined_regressed']} 条、判定改变 {paired['prediction_changed']} 条；未知转为已知 {paired['unknown_to_known']} 条，已知转为未知 {paired['known_to_unknown']} 条。"]
    else:
        lines += ["", "## V5 实际指标", "", "模型推理未完成，不能报告 V5 准确率、宏 F1、证据准确率、联合正确率、反事实成对准确率或运行效率。"]
    if a.errors.exists() and a.errors.stat().st_size:
        errors=[json.loads(x) for x in a.errors.read_text().splitlines() if x.strip()]
        counts=collections.Counter((x.get("method"),x.get("error_category")) for x in errors)
        lines += ["", "## 错误分析", "", "预测错误按方法和类别计数："]
        for (method,category),count in sorted(counts.items()): lines.append(f"- {method} / {category}: {count}")
        lines += ["", "逐例错误 ID 见 `errors_public.jsonl`，不包含患者文本。"]
    else:
        lines += ["", "## 错误分析", "", "无已保存的模型逐例预测；模型未运行时不生成虚构错误类别。"]
    lines += ["", "## 解释和限制", "",
      "1. TrialMatchAI 的 Hugging Face 仓库提供 Phi-4 LoRA 生成适配器，不是独立三分类器。实现保存原始生成到服务器私有输出，并将官方入组标签保守映射到 T/F/U；`Irrelevant`、`Unclear` 均为 U。排除侧 `Not Violated` 也映射为 U。",
      "2. TrialMatchAI 原始输出没有可验证的病历句区间，因此纯模型组不报告证据定位与联合指标；TrialMatchAI 加确定性数值比较仅以程序实际比较的数字句评价；只有组合组统计 MedNLI 证据定位。",
      "3. BioLinkBERT 类别编号从模型配置读取。NLI 中立不作为条件为假；与 TrialMatchAI 冲突时保留冲突并将无法解决的条件设为 U。",
      "4. V4 的规则文件是预先结构化的研究用条件改写，不含完整原始指南文本和治疗动作；因此本实验能评价规则谓词适用性，但不能证明动作定位或完整临床工作流已可正式替换。原文字符偏移以该预备规则文本及输入病例字符串为基准。",
      "5. 没有纳入未使用规则和病例的独立标签集；现有 192 条数据来自合成病例与控制变体，且标签未经临床专家复核。",
      "6. 结论：在独立验证、真实指南动作片段和完整病例评估前，不应把该实现认定为 CARE 正式模块一。",
      "", "## 产物位置", "",
      "- 逐例公开预测：`predictions_public.jsonl`（无病历文本，只保留标签、字符区间和模型分数）",
      "- 原始生成与含原文证据：服务器本地 `private/predictions_private.jsonl`，由 `.gitignore` 排除",
      "- 指标：`metrics.json`；错误：`errors_public.jsonl`；运行信息：`run_status.json`",
      "- V4 输入原文继续保留在服务器 `/home/gcx/care_module1_20261018_v4/`，V5 不覆盖或重写。"]
    a.output.write_text("\n".join(lines)+"\n")
    print(str(a.output))
if __name__ == "__main__": main()
