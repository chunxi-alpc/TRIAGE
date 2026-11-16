"""Write per-checkpoint JSON/CSV/Markdown; missing experiments stay explicitly pending."""
from pathlib import Path
from .prepare import OUT,KEYS
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import write_csv

def _report():
    summary=[]
    states=[read(OUT/"models"/k/"completion.json") for k in KEYS]
    completed=sum(bool(x.get("complete")) for x in states)
    diagnostics=sum(x.get("status")=="selection_incomplete" for x in states)
    execution=read(OUT/"execution_plan.json")
    opening=(f"已完成实测 {completed}/7 个模型；另有 {diagnostics} 个模型以 selection_incomplete 保存诊断。"
             if completed or diagnostics else
             "代码、统计实现、开发配对补充及 CPU 测试已完成。GPU 实验等待统一队列接纳；当前没有七个模型的实测选头或性能结论。")

    if execution.get("gpu_authorized") and not completed and not diagnostics:
        opening="GPU任务已授权，统一队列执行中。实测结果将在各检查点任务结束后写入。"
    text=["# ASTER-CR 自适应注意力头筛选实验","",
          opening,"",
          "## 数据与来源划分","",
          "训练集保留 E1 的 2,540 对配对（A/P/C/F 各 635 对），来自 80 个来源；开发集沿用原 20 个来源，测试集沿用原 40 个来源。",
          "6,345 对完整训练池覆盖同一组 80 个训练来源。本次未改变训练集。",
          "经用户确认，在原开发来源补充 43 对：P 18 对、C 10 对、F 15 对。保留旧配对，四类各有 20 个独立来源；按 pair_id 固定选择每类每来源一对，共 80 对用于确认。",
          "新增配对为研究用反事实变换；来源、原文、修改位置和标签保存在 development/pair_edits.jsonl。P 添加无关行政背景，C 修改节点条件，F 修改决定性患者事实及重复表述。未使用测试来源。","",
          "## 方法实现","",
          "统一入口为 adaptive_m1/staged；training.fit_group 与 m1_plan 的 adaptive 模式调用同一实现。旧 fixed 模式保留。",
          "梯度归因覆盖四类的全部注意力头，固定目标来源梯度，打乱同类来源的激活差分，进行 2,000 次置换。模型内统一 BH 校正，q≤0.20 进入主分析。",
          "开发阶段执行真实配对和不同来源供体的双向修补，计算真实效应减去五个供体平均效应。随机种子固定，供体来源有记录。",
          "最终确认采用单侧精确符号检验（零差值作为平局剔除），统一 BY q≤0.05、2,000 次来源级 bootstrap 的均值 95% 区间下界>0，并要求至少16个有效来源。BH 同阈值另行输出。",
          "分别保存 H_A/H_P/H_C/H_F、H_N/H_E/H_shared，最终去重并集不截断。任一干扰或证据集合为空时停止 M2/M3，状态为 selection_incomplete。",
          "M2 秩上限16、M3正则化0.01、证据预算0.2923349及自然节点截距校准沿用旧方法。低维空间按实际秩拟合，不补头。",
          "N_eff 对各类候选头中正均值超额效应计算。稳定性输出为固定梯度筛选后的来源重采样 BY 支持率及正均值频率，未嵌套重算置信区间；该诊断与最终确认判定分别记录。","",
          "## 预设敏感性分析","",
          "梯度 BH q=0.10/0.20/0.30；确认 BY q=0.01/0.05/0.10；供体数3/5/10；bootstrap均为2,000次。使用嵌套的固定供体顺序。",
          "为计算q=0.30敏感性分析，对该阈值下的候选执行修补；主分析的多重检验家族仍为q≤0.20的候选。所有阈值固定，不按测试表现选优。","",
          "## 七个检查点","",
          "|检查点|状态|总头数|候选A/P/C/F|有效A/P/C/F|最终数量|比例|",
          "|---|---|---:|---|---|---:|---:|"]
    for key in KEYS:
        root=OUT/"models"/key
        m=read(root/"models/E1/full/m1.json")
        cfg=read(root/"configuration.json")
        geometry=read(OUT/"resource_plan.json").get("models",{}).get(key,{})
        stage=read(root/"execution.json")
        status=read(root/"completion.json").get("status",stage.get("stage",stage.get("state","queued" if execution.get("gpu_authorized") else "awaiting_unified_queue")))
        row=dict(model=key,status=status,total_heads=m.get("total_heads",geometry.get("total_heads")),
            selected_count=m.get("selected_count"),selection_ratio=m.get("selection_ratio"),
            candidate_by_kind=m.get("candidate_by_kind"),final_by_kind=m.get("final_by_kind"),
            nuisance_count=m.get("nuisance_count"),evidence_count=m.get("evidence_count"),
            shared_count=m.get("shared_count"),N_eff=m.get("N_eff"))
        summary.append(row)
        def show(x):return "待运行" if x is None else str(x)
        text.append("|"+"|".join(show(row.get(k)) for k in ("model","status","total_heads","candidate_by_kind",
                                                       "final_by_kind","selected_count","selection_ratio"))+"|")
        if m:
            per=["# "+key,"",f"状态：{status}","",f"最终选头：{m['selected_count']}/{m['total_heads']}","",
                 "逐头统计：main_heads.csv；四类集合与层分布：m1.json；供体：donors.json；敏感性：sensitivity.json。"]
            (root/"REPORT.zh.md").write_text("\n".join(per)+"\n",encoding="utf-8")
            plot(root,m)
    text += ["","## 测试集结果","",
             "|检查点|方法|VR|ER|C/F修正RMS|自然路径准确率|证据方向退化|超预算比例|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    test_rows=[]
    for key in KEYS:
        result=read(OUT/"models"/key/"evaluation/test/metrics.json")
        for method,label in (("E0","fixed_51"),("E1","adaptive")):
            metric=result.get("models",{}).get(method,{})
            rr=dict(model=key,method=label,**metric)
            test_rows.append(rr)
            values=[key,label]+[metric.get(k) for k in ("ap_vr","cf_er","cf_correction_rms",
                "natural_path_accuracy","cf_direction_degradation_count","cf_over_budget_rate")]
            text.append("|"+"|".join("待运行" if x is None else str(x) for x in values)+"|")
        if result:
            text += ["",f"{key} 的来源级95%区间与固定预算配对差值：models/{key}/source_bootstrap.csv。"]
    write(OUT/"test_comparison.json",test_rows)
    write_csv(OUT/"test_comparison.csv",test_rows)
    sensitivity=[]
    for key in KEYS:
        scenarios=read(OUT/"models"/key/"models/E1/full/sensitivity.json")
        for name,result in scenarios.items():
            sensitivity.append(dict(model=key,scenario=name,status=result["status"],
                heads=result["selected_count"],jaccard=result["jaccard_with_main"]))
    write(OUT/"sensitivity_summary.json",sensitivity)
    write_csv(OUT/"sensitivity_summary.csv",sensitivity)
    text += ["","## 配对评价与基线复用","",
        "沿用 evaluation.py 的 VR、ER、C/F修正RMS、自然路径准确率、证据方向退化数量及超预算比例。新增干扰路径纠正数和新增错误数。相同测试来源共用2,000次来源级重采样，输出方法差值及95%区间。",
        "基线优先检查模型、分词器、提示、训练数据、配置和原始分数。历史结果缺少完整分词器/输入身份记录时，在独立目录重跑固定51头基线，复用本次同一引擎提取的原始特征；历史学习结果保持原样。",
        "开发集用于M1特征选择，开发性能标记为模型选择结果；模型和阈值冻结后才提取测试特征并评价。","",
        "## 执行计划与状态","",
        "统一队列清单：queue_jobs.json；允许GPU为0、1、5、6，每卡一个检查点。资源与时长见resource_plan.json。入口检查调度器父进程，单独运行不会加载模型。",
        ("先进行Qwen小规模GPU冒烟，再运行七个检查点；当前没有提交清单、启动调度器或加载GPU模型。"
         if not execution.get("gpu_authorized") else "任务状态见各模型 execution.json、completion.json 与统一队列记录。"),
        "每个检查点按fit/dev原始特征→自适应M1→M2/M3→匹配基线→开发评价→冻结→测试评价执行。","",
        "## 测试与历史结果保护","",
        "CPU单元测试、集成测试和数据隔离测试见cpu_tests.log与test_results.json。所有测试在服务器环境运行，CUDA_VISIBLE_DEVICES为空。",
        "新实验及输出位于aster_e678_adaptive_20261116。历史报告目录未写入；修改的旧代码只增加显式adaptive分支、拒绝将自适应计划送入旧修补缓存，以及零秩M3的数学退化处理。","",
        "## 论文方法修改建议","",
        "方法段可将固定头预算改为来源独立的两阶段统计筛选，说明配对干预相对于随机供体的超额效应、BY确认及来源级区间。",
        "结果段应在七模型运行完成后逐模型报告选头规模、四类集合、层分布和配对测试差值。当前不改写正式实验结论，也不宣称自适应方法优于固定预算。",
        "具体实测结果、基线对照数值和论文结论建议将在模型运行完成后更新。"]
    write(OUT/"summary.json",summary)
    write_csv(OUT/"summary.csv",summary)
    (OUT/"REPORT.zh.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return summary

def plot(root,m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,3))
    layers=m["layer_distribution"]
    ax.bar([int(k) for k in layers],list(layers.values()))
    ax.set(xlabel="Layer (zero-based)",ylabel="Selected heads",title=root.name)
    fig.tight_layout()
    fig.savefig(root/"layer_distribution.png",dpi=180)
    fig.savefig(root/"layer_distribution.pdf")
    plt.close(fig)

def report():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _report()

if __name__=="__main__":report()
