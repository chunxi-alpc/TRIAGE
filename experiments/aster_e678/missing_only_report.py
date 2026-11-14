"""CPU report for the user's revised scope: only experiments not previously run."""
import argparse
import re
import time
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
from .prepare import OUT, SOURCE, KEYS

NAMES = {
    "qwen": "Qwen3-8B", "gemma2": "Gemma-2-9B-IT",
    "gemma2_base": "Gemma-2-9B-Base", "llama31": "Llama-3.1-8B-Instruct",
    "gemma3": "Gemma-3-27B-IT", "medgemma": "MedGemma-27B-Text-IT",
    "qwen_base": "Qwen3-8B-Base",
}
SCOPES = ("nli4ct", "nli4ct_official5500", "medpic", "medpic_official467")

def generate(finalize=False):
    policy = read(OUT / "only_missing_policy.json")
    active = policy["active_models"]
    stopped = policy["stopped_model_pipelines"]
    planned = policy.get("missing_transfer_jobs", {})
    status = dict(updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  policy=policy, models={}, E8={}, qwen_E1_reused=True,
                  all_six_E1_complete=False)
    for key in KEYS:
        root = OUT / "models" / key
        completed = bool(read(root / "completion.json").get("complete"))
        runtime = read(root / "run_status.json")
        populations = {}
        for scope in SCOPES:
            metrics = read(root / "evaluation" / scope / "metrics.json")
            if metrics.get("complete"):
                state = "complete"
            elif key in planned:
                state = "queued_missing_transfer"
            elif key in stopped:
                state = "deferred_source_not_frozen" if scope != "nli4ct" else "historical_result_exists_no_rerun"
            else:
                state = "not_required_for_Base" if key.endswith("_base") else "pending"
            populations[scope] = dict(state=state, metrics=str(root / "evaluation" / scope / "metrics.json") if metrics.get("complete") else None)
        status["models"][key] = dict(
            name=NAMES[key], state="complete" if completed else ("stopped_by_user" if key in stopped else runtime.get("state", "pending")),
            last_stage=runtime.get("stage"), root=str(root),
            full_pipeline_state="stopped_by_user" if key in stopped else runtime.get("state", "pending"),
            missing_transfer_status=read(root / "missing_transfer/status.json") if key in planned else None,
            fit_complete=bool(read(root / "models/E1/full/model.json").get("complete")),
            frozen=bool(read(root / "frozen.json").get("frozen")),
            test_complete=bool(read(root / "evaluation/test/metrics.json").get("complete")),
            populations=populations,
            official_mega_complete=(read(root / "evaluation/mega_official/metrics.json").get("state") == "complete" or bool(read(root / "evaluation/mega_official/metrics.json").get("complete"))),
        )
    from .compare_base_it import compare_pair
    for label, base, it, hpl in (("gemma2", "gemma2_base", "gemma2", 16), ("qwen", "qwen_base", "qwen", 32)):
        base_root = OUT / "models" / base
        it_root = SOURCE if it == "qwen" else OUT / "models" / it
        ready = all(read(p / "models/E1/full/model.json").get("complete") and
                    read(p / "evaluation/test/metrics.json").get("complete")
                    for p in (base_root, it_root))
        destination = OUT / "base_it" / label
        if ready and not (destination / "comparison.json").exists():
            compare_pair(base_root, it_root, destination, base_name=NAMES[base], it_name=NAMES[it], heads_per_layer=hpl)
        status["E8"][label] = dict(state="complete" if ready else "pending_Base", directory=str(destination))
    current_complete = all(status["models"][key]["state"] == "complete" for key in active)
    status["missing_transfers"] = {
        key: dict(state="complete" if all(status["models"][key]["populations"][scope]["state"] == "complete"
                                         for scope in SCOPES) else "pending",
                  planned_stages=entry["stages"], new_training=False,
                  runtime=status["models"][key]["missing_transfer_status"])
        for key, entry in planned.items()
    }
    status["current_scope_complete"] = (current_complete and all(v["state"] == "complete" for v in status["E8"].values())
                                        and all(v["state"] == "complete" for v in status["missing_transfers"].values()))
    status["phase"] = "current_scope_complete" if status["current_scope_complete"] else "running_missing_only"
    status["unrun_transfer_deferred"] = policy["deferred_evaluations"]
    write(OUT / "status_only_missing.json", status)

    lines = [
        "# E6/E7/E8 仅补未运行实验：当前状态", "",
        "更新时间：" + status["updated_at"] + "。执行依据：用户要求“没跑过的实验才跑”。", "",
        "Gemma2 IT、Gemma3、MedGemma 的三个整套流程继续停止。两个 Base 任务继续；Gemma3、MedGemma 复用已完成 E1 拟合参数，单独补完剩余开发集计算、冻结及缺失迁移评价。调度器只接续已列明的未完成阶段。", "",
        "| 模型 | 当前安排 | E1 参数冻结 | E1 受控测试 |", "|---|---|---|---|"]
    for key in KEYS:
        item = status["models"][key]
        action = ("仅补缺失迁移；复用已拟合参数" if key in planned else
                  ("已完成，复用" if item["state"] == "complete" else ("已停止重复流程" if key in stopped else "继续 Base 比较")))
        lines.append("| " + NAMES[key] + " | " + action + " | " + ("完成" if item["frozen"] else "未完成") + " | " + ("完成" if item["test_complete"] else "未完成") + " |")
    lines += ["", "## 跨任务评价", "",
              "Qwen 的 E1 四个评价总体继续直接复用。Llama 和 Gemma2 本轮已完成的跨任务结果保留。",
              "Gemma3 和 MedGemma 按既定 E1 配置补完开发集冻结，再评价 NLI 5500 与 MedPIC 188/467。NLI 500 的当前 E1 冻结迁移统计直接使用已有2864条缓存，在CPU计算；历史NLI内拟合结果单独保留。MedPIC 467含既有开发问题，作为全题描述性评价。", "",
              "| 模型 | NLI 500 | NLI 5500 | MedPIC 188 | MedPIC 467 |", "|---|---|---|---|---|"]
    for key in ("llama31", "gemma2", "gemma3", "medgemma"):
        labels = {"complete": "已完成", "queued_missing_transfer": "缺项已排队", "deferred_source_not_frozen": "待源任务冻结", "historical_result_exists_no_rerun": "复用历史记录", "pending": "待执行"}
        cells = [labels.get(status["models"][key]["populations"][scope]["state"], "未要求") for scope in SCOPES]
        lines.append("| " + NAMES[key] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Base/IT", ""]
    for label, item in status["E8"].items():
        lines.append("- " + label + "：" + ("比较完成" if item["state"] == "complete" else "等待 Base 完成") + "；" + item["directory"])
    lines += ["", "## 结果使用", "",
              "旧协议使用 A/P 4887 对、C/F 135 对；当前 E1 使用 A/P/C/F 各635对。两套协议分别标记，旧分数不填入 E1 结果列。",
              "历史806路径与官方7458路径保留既往使用记录。未运行单元保留状态，不填数值。",
              "当前范围完成包含两个 Base、两组 Base/IT 比较及两模型缺失迁移评价。Gemma3、MedGemma的E1源任务受控测试与官方MEGA，以及Gemma2尚未完成的本轮官方MEGA继续列缺项；历史结果按原协议保留，不填入对应E1列。", "",
              "- 调整记录：/home/gcx/aster-gpu-queue-20261113/only_missing_20261114/",
              "- 历史核对：/home/gcx/aster-e5-gaps-20261112/reports/aster_e5_gaps_20261112/running_history_check_20261114/",
              "- 策略：only_missing_policy.json；逐项状态：status_only_missing.json。",
              ""]
    report = OUT / ("REPORT_FUTURE_MISSING_20261114.zh.md" if planned else "REPORT_ONLY_MISSING_20261114.zh.md")
    report.write_text("\n".join(lines), encoding="utf-8")

    docs = OUT.parents[1] / "docs/ASTER/experiments.zh.md"
    content = docs.read_text(encoding="utf-8")
    transfer = sum(status["models"][k]["populations"][scope]["state"] == "complete"
                   for k in ("llama31", "gemma2", "gemma3", "medgemma") for scope in SCOPES)
    matched = sum(v["state"] == "complete" for v in status["E8"].values())
    table = {
        "E6": "| E6 | 五模型跨架构泛化 | **复用历史；重复流程已停止** | 保留本轮已完成结果；Gemma2 IT、Gemma3、MedGemma 不再自动接续整套流程 | 按用户要求只补未运行实验 |",
        "E7": "| E7 | 三个主数据集的冻结迁移 | **Qwen E1复用；其余完成" + str(transfer) + "/16** | Gemma3、MedGemma 缺失迁移已按开发冻结→迁移顺序排队；旧NLI记录按历史协议保留 | 只补缺失阶段，已完成缓存与参数复用 |",
        "E8": "| E8 | Base/IT匹配检查点 | **比较完成" + str(matched) + "/2** | Gemma2 Base、Qwen Base继续；IT使用已完成参数和结果 | 两个Base结束后自动汇总 |",
    }
    for key, row in table.items():
        content = re.sub(r"^\| " + key + r" \|.*$", lambda match, row=row: row, content, flags=re.M)
    marker = "<!-- FUTURE-MISSING-20261114 -->"
    if marker not in content:
        content += "\n" + marker + "\n### 2026-11-14 执行范围调整\n\n用户要求后续只运行未完成实验。两个 Base 继续；Gemma3、MedGemma 复用已拟合 E1 参数，仅完成缺失开发集计算、冻结和迁移评价，NLI500子集统计使用已有缓存在CPU计算。三个整套流程保持停止。详见[后续缺项队列](../../reports/aster_e678_20261112/REPORT_FUTURE_MISSING_20261114.zh.md)。\n"
    docs.write_text(content, encoding="utf-8")
    if finalize:
        if not status["current_scope_complete"]:
            raise RuntimeError("Authorized Base jobs, comparisons and missing transfer evaluations are not complete")
        write(OUT / ("future_missing_summary_complete.json" if planned else "missing_only_summary_complete.json"),
              dict(complete=True, scope="user_revised_only_missing", all_six_E1_complete=False,
                   completed_models=active, stopped_models=stopped,
                   missing_transfers=status["missing_transfers"], deferred=policy["deferred_evaluations"], report=str(report),
                   completed_at=status["updated_at"]))
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    result = generate(args.finalize)
    print(result["phase"])

if __name__ == "__main__":
    main()
