"""Write an honest E6/E7/E8 status report, including reused and pending cells."""
import time
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write
from .prepare import OUT, SOURCE, KEYS
from .engine import MODELS

QWEN_OFFICIAL=Path("/home/gcx/aster-cr-balanced-publish-20261112/reports/aster_cr_final_method_10006_20261112")
def generate():
    if (OUT / "only_missing_policy.json").exists():
        from .missing_only_report import generate as generate_missing_only
        return generate_missing_only()
    status=dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), phase="queued" if (OUT/"handoff_plan.json").exists() else "prepared",
        protocol="E1 balanced 2540, independent M1/M2/M3 for each new checkpoint",
        qwen=dict(source=str(SOURCE),formal_result_source=str(QWEN_OFFICIAL),
            historical_meGA_paths=dict(correct=768,total=806),
            nli_original=dict(correct=375,total=500),medpic_subset=dict(correct=66,total=188),
            nli_official=dict(correct=4043,total=5500),medpic_official=dict(correct=183,total=467),
            mega_official7458="not located for E1; Qwen not scheduled under this task"),
        models={}, E8={}, allocation=read(OUT/"allocation.json"),
        download=read(OUT/"status/gemma2_base_download.json"))
    for key in KEYS:
        root=OUT/"models"/key
        prepared=read(root/"prepared.json")
        completed=read(root/"completion.json").get("complete",False)
        runtime=read(root/"run_status.json")
        model=read(root/"models/E1/full/model.json")
        status["models"][key]=dict(name=MODELS[key]["name"],root=str(root),
            state="complete" if completed else runtime.get("state","queued"),
            modules_fitted=bool(model.get("complete")),parameters_frozen=read(root/"frozen.json").get("frozen",False),
            raw_cache_reused=prepared.get("raw_cache_reused",0),
            populations=read(root/"evaluation_populations.json"),
            mega_official=read(root/"evaluation/mega_official/metrics.json"))
    if all(v["state"]=="complete" for v in status["models"].values()):
        status["phase"]="complete"
    elif any((OUT/"models"/k/"process.json").exists() for k in KEYS):
        status["phase"]="running"
    if read(OUT/"handoff_completion.json").get("queue_finished") and status["phase"]!="complete":
        status["phase"]="partial"
    from .compare_base_it import compare_pair
    for label,base,it,hpl in (("gemma2","gemma2_base","gemma2",16),("qwen","qwen_base","qwen",32)):
        base_root=OUT/"models"/base
        it_root=SOURCE if it=="qwen" else OUT/"models"/it
        ready=all(read(p/"models/E1/full/model.json").get("complete") and read(p/"evaluation/test/metrics.json").get("complete") for p in (base_root,it_root))
        if ready:
            compare_pair(base_root,it_root,OUT/"base_it"/label,
                base_name=MODELS[base]["name"],it_name=MODELS[it]["name"],heads_per_layer=hpl)
        status["E8"][label]=dict(state="complete" if ready else "waiting_independent_fit",
                                base_root=str(base_root),it_root=str(it_root))
    write(OUT/"status.json",status)
    lines=[
        "# ASTER-CR E6、E7、E8 状态报告",
        "",
        "更新时间："+status["updated"]+"。本轮以已完成 Qwen E1 为准：2,540 对，A/P/C/F 各 635 对、80 个拟合来源、20 个开发来源。后续 10,006 对方案不替代本轮协议。",
        "",
        "## 当前执行状态",
        "",
        "Codex 00/01 审计已读取，当前 E1 方案无需改变。全部剩余任务交由 /home/gcx/aster-gpu-queue-20261113 统一调度，GPU 0、1、5、6 哪张空闲就分配哪张，每卡一个任务。旧独立队列已停止，旧释放文件不再控制本轮启动。当前状态："+dict(queued="排队等待资源",prepared="已准备",running="执行中",complete="已完成").get(status["phase"],status["phase"])+"。",
        "",
        "## E6：当前协议五模型",
        "",
        "| 模型 | 独立 M1/M2/M3 | 当前 E1 评价 |",
        "|---|---|---|",
        "| Qwen3-8B | 已完成，直接复用 | 受控自然路径 768/806；当前 E1 官方 7,458 路径未找到，不以历史协议结果填补，也不安排本轮重跑 |"]
    for key in ("llama31","gemma2","gemma3","medgemma"):
        item=status["models"][key]
        lines.append("| "+item["name"]+" | "+("已完成" if item["modules_fitted"] else "已准备，待调度")+" | "+("已完成" if item["state"]=="complete" else "待执行；含受控评价及官方 7,458 路径")+" |")
    lines += ["","所有新检查点独立选择 51 个头，估计证据/干扰子空间及 16 维痕迹，拟合 λ=0.01、δ=0.2923349 的受约束修正。截距沿用 E1 的 761 个源任务拟合节点校准流程；各模型完成自己的源任务开发评价后冻结。旧五模型 15 组结果保留，未导入其选头、子空间或修正参数。",
        "", "## E7：冻结跨任务评价", "",
        "| 模型 | NLI 原声明 500 | NLI 正式 5,500 | MedPIC 子集 188 | MedPIC 全部 467 |",
        "|---|---|---|---|---|",
        "| Qwen3-8B | 已有 375/500 | 已有 4043/5500 | 已有 66/188 | 已有 183/467 |"]
    for key in ("llama31","gemma2","gemma3","medgemma"):
        populations=status["models"][key]["populations"]
        cells=["已完成" if populations.get(s,{}).get("status")=="complete" else "已准备，待独立源任务参数" for s in ("nli4ct","nli4ct_official5500","medpic","medpic_official467")]
        lines.append("| "+MODELS[key]["name"]+" | "+" | ".join(cells)+" |")
    lines += ["", "Qwen 四个总体均复用已有 E1 列，不重复推理或评价。正式总体报告中的 models/E1 指向原 E1 参数；该列不使用其他方案的参数。MedPIC 467 题包含既有开发划分中的题目，按完整官方总体报告，并保留这一使用记录。",
        "", "其余四模型可复用同检查点的原始分数与全头特征；现有 NLI 测试 2,864 个输入（其中原声明 500 个）均可复用。新增 NLI 总体只补 2,636 个输入。原始缓存只在提示及标签一致时使用；旧目标任务拟合参数和修正预测均不复用。NLI 与 MedPIC 分别报告，目标标签只进入评价指标。",
        "", "## E8：Base/IT 匹配", "",
        "| 优先级 | 检查点对 | 状态 |",
        "|---|---|---|",
        "| 第一 | Gemma-2-9B-Base / Gemma-2-9B-IT | Base 权重"+("已下载" if status["download"].get("state")=="complete" else "下载中")+"；两者独立拟合已准备 |",
        "| 第二 | Qwen3-8B-Base / Qwen3-8B | Base 权重已存在；Base 待拟合，IT 复用 E1 |",
        "", "比较头集合和层分布、各自子空间谱/有效秩、同一检查点内证据与干扰子空间夹角，以及 A/P 修正、C/F 保护和决策变化。不同检查点各自选择的坐标不直接当作同一子空间比较。差异按检查点报告，不解释为某个特定后训练阶段的因果效应。",
        "", "## 资源与实验规模", "",
        "用户明确使用 GPU 0、1、5、6，由统一队列按空闲卡分配。每个检查点是独立单卡任务；不抢占现有任务。启动前资源、规模和预计时间已向用户汇报。",
        "", "每个新模型拟合输入 4,953 个、开发输入 33,341 个、受控测试输入 66,752 个（均先复用合格原始缓存）。M1 每类别每来源固定一对、64 个候选头、双向修补，最多 40,960 次末 token 前向。E6 每模型另评价官方 7,458 条动态路径；E7 只对四个 IT 模型补充，Base 无额外跨任务要求。",
        "", "预估单卡峰值：8B 模型约 18–30 GiB，Gemma2 约 22–40 GiB，27B 约 55–78 GiB。六个检查点在四张 A100 80GB 全部到位后粗估 1–3 天；这是按既有规模推算，尚无本轮 GPU 实测。正式运行后只确认启动稳定，不持续监控。",
        "", "## 产物与验证", "",
        "- 每模型目录：models/<model_key>/，包含 configuration.json、数据引用、独立参数、冻结记录、逐例预测、统计、运行状态和日志。",
        "- 程序：../../experiments/aster_e678/；unified_worker.py 执行单个检查点，finalize.py 在六模型完成后汇总。任务清单为统一队列的 codex05_jobs.json；各模型状态及失败日志独立保存。",
        "- CPU 验证：模型加载接口/分词器、动态头维数、评价与冻结流程、Base/IT 统计，以及官方动态路径遍历已通过；GPU 运行尚未验证。",
        "- 旧结果只读；不执行哈希检查；未修改论文正文。",
        "", "## 复用来源", "",
        "- Qwen E1 参数、806 路径及 500/188 结果："+str(SOURCE),
        "- Qwen E1 的 5,500/467 结果："+str(QWEN_OFFICIAL),
        "- 四模型原始特征：/home/gcx/aster_five_models_20261109/results/<model_key>/{mega,nli4ct}/",
        "- 官方 MEGA 清单：/home/gcx/aster_next_20261109/inventory/official_all_path_cases.jsonl",
        ""]
    (OUT/"REPORT.zh.md").write_text("\n".join(lines),encoding="utf-8")
    update_experiment_docs(status)
    return status

def update_experiment_docs(status):
    import re
    path=OUT.parents[1]/"docs/ASTER/experiments.zh.md"
    text=path.read_text(encoding="utf-8")
    it_keys=("llama31","gemma2","gemma3","medgemma")
    finished=sum(status["models"][k]["state"]=="complete" for k in it_keys)
    transfer=sum(status["models"][k]["populations"].get(s,{}).get("status")=="complete"
        for k in it_keys for s in ("nli4ct","nli4ct_official5500","medpic","medpic_official467"))
    matched=sum(v["state"]=="complete" for v in status["E8"].values())
    rows={
        "E6":f"| E6 | 五模型跨架构泛化 | **Qwen E1复用；新增四模型完成{finished}/4** | 各模型独立M1/M2/M3；受控与官方7,458路径分列 | 由统一队列分配GPU 0、1、5、6；详细状态见本轮报告 |",
        "E7":f"| E7 | 三个主数据集的冻结迁移 | **Qwen E1已复用；其余模型总体完成{transfer}/16** | Qwen E1：NLI375/500、4043/5500；MedPIC66/188、183/467 | 源任务冻结后分别评价；不重复Qwen |",
        "E8":f"| E8 | Base/IT匹配检查点 | **匹配比较完成{matched}/2** | 两个Base权重已就绪；Gemma2优先，Qwen IT复用E1 | 比较表示、头选择及修正行为；不作特定后训练阶段因果解释 |",
    }
    for key,row in rows.items():
        text=re.sub(r"^\| "+key+r" \|.*$",lambda match:row,text,flags=re.M)
    descriptions=dict(queued="当前为排队状态，尚未启动本轮 GPU 实验。",
        running="当前接续队列正在执行，分模型状态见本轮报告。",
        complete="本轮六个检查点均已完成，完整结果见本轮报告。",
        partial="接续队列已结束，部分任务失败或未完成；详细原因见本轮报告及日志。")
    if status["phase"] in descriptions:
        text=re.sub(r"当前为排队状态，尚未启动本轮 GPU 实验。|当前接续队列正在执行，分模型状态见本轮报告。|本轮六个检查点均已完成，完整结果见本轮报告。|接续队列已结束，部分任务失败或未完成；详细原因见本轮报告及日志。",
            lambda match:descriptions[status["phase"]],text)
    path.write_text(text,encoding="utf-8")

if __name__=="__main__":
    generate()
