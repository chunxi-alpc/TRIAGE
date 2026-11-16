"""Per-checkpoint reports for the current adaptive-only experiment scope."""
from .prepare import OUT,KEYS
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write
from experiments.aster_cr_retrain_paired_v2_20261111.evaluation import write_csv

def _report():
    states={k:read(OUT/"models"/k/"completion.json") for k in KEYS}
    completed=sum(bool(v.get("complete")) for v in states.values())
    incomplete=sum(v.get("diagnostics_complete",False) for v in states.values())
    text=["# ASTER-CR 自适应注意力头筛选实验","",
          f"实测完成 {completed}/7；selection_incomplete 诊断终点 {incomplete}/7。其他检查点状态见下表。","",
          "## 当前实验范围","",
          "按用户最新要求，本次执行七模型自适应选头、M2/M3拟合、开发与测试评价、阈值敏感性及结构诊断。取消全部置信区间计算与相应选头条件，取消固定51头重跑和对照。历史固定预算结果保持原样。",
          "来源重采样保留用于选头稳定性，最终性能报告点估计。","",
          "## 数据与方法","",
          "训练集为E1的2,540对（每类635对），来源划分80/20/40。原20个开发来源补充43对：P18、C10、F15，四类各覆盖20个来源。",
          "第一阶段按每类每训练来源一个固定代表配对，对所有注意力头进行2,000次来源置换，模型内全部类别—头统一BH校正，主阈值q≤0.20。",
          "开发阶段采用真实配对及5个不同来源随机供体的双向修补。最终确认要求：单侧精确符号检验BY q≤0.05、平均超额干预效应>0、至少16个有效开发来源；同时输出BH敏感性结果。",
          "分别构造四类头集合及干扰/证据/共享集合，不截断或补头。干扰或证据集合为空时保留selection_incomplete诊断并停止M2/M3。",
          "M2秩上限16、M3正则化0.01、证据预算0.2923349、自然节点截距校准及最终token YES-NO评分保持原配置。M2/M3只使用训练来源。",
          "仅代表配对的左侧输入计算梯度（本数据为296个唯一输入）；所有训练配对仍用于M2/M3。已提取缓存保留。","",
          "## 七模型状态与选头","",
          "|模型|状态|总头数|候选A/P/C/F|有效A/P/C/F|最终头数|比例|",
          "|---|---|---:|---|---|---:|---:|"]
    summary=[];metrics=[];sensitivity=[]
    for key in KEYS:
        root=OUT/"models"/key
        m=read(root/"models/E1/full/m1.json")
        ex=read(root/"execution.json")
        state=states[key].get("status",ex.get("stage",ex.get("state","queued")))
        geometry=read(OUT/"resource_plan.json").get("models",{}).get(key,{})
        row=dict(model=key,status=state,total_heads=m.get("total_heads",geometry.get("total_heads")),
            candidate_by_kind=m.get("candidate_by_kind"),final_by_kind=m.get("final_by_kind"),
            selected_count=m.get("selected_count"),selection_ratio=m.get("selection_ratio"),
            nuisance_count=m.get("nuisance_count"),evidence_count=m.get("evidence_count"),
            shared_count=m.get("shared_count"),N_eff=m.get("N_eff"))
        summary.append(row)
        text.append("|"+"|".join("待运行" if row.get(k) is None else str(row[k])
            for k in ("model","status","total_heads","candidate_by_kind","final_by_kind","selected_count","selection_ratio"))+"|")
        if m:
            (root/"REPORT.zh.md").write_text("\n".join(["# "+key,"",f"状态：{state}",
                f"最终选头：{m['selected_count']}/{m['total_heads']}","",
                "逐头效应和p/q：models/E1/full/per_head_statistics.csv；四类集合与层分布：m1.json；供体：donors.json。",
                "本次不计算置信区间，不重跑固定51头对照。"])+"\n",encoding="utf-8")
            plot(root,m)
        test=read(root/"evaluation/test/metrics.json")
        for method,label in (("raw","frozen_backbone"),("E1","adaptive")):
            metrics.append(dict(model=key,method=label,**test.get("models",{}).get(method,{})))
        for name,result in read(root/"models/E1/full/sensitivity.json").items():
            sensitivity.append(dict(model=key,scenario=name,status=result["status"],
                heads=result["selected_count"],jaccard=result["jaccard_with_main"]))
    text+=["","## 测试集点估计","",
           "|模型|方法|VR|ER|C/F修正RMS|自然路径准确率|证据方向退化|超预算比例|",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        text.append("|"+"|".join("待运行" if row.get(k) is None else str(row[k]) for k in
            ("model","method","ap_vr","cf_er","cf_correction_rms","natural_path_accuracy",
             "cf_direction_degradation_count","cf_over_budget_rate"))+"|")
    text+=["","各模型point_estimates.csv另列干扰路径纠正数、新增错误数及相对于原始模型的变化。指标定义沿用既有evaluation.py。","",
           "## 敏感性与稳定性","",
           "梯度BH q=0.10/0.20/0.30；确认BY q=0.01/0.05/0.10；随机供体数3/5/10。修补缓存共用，主分析检验家族按主阈值独立确定。",
           "选头稳定性使用2,000次来源级重采样，重新应用BY、正平均效应和来源数条件，报告入选频率与集合Jaccard。N_eff使用各类候选中的正均值超额效应计算。","",
           "## 执行与隔离","",
           "统一队列：/home/gcx/aster-gpu-queue-20261113/adaptive_20261116，使用GPU 0、1、5、6，每卡一个模型。先前提取缓存继续使用，已通过GPU冒烟。",
           "开发集用于M1选择，开发性能标记为模型选择结果；冻结后才提取测试特征和计算测试指标。真实/随机供体缓存独立保存，含目标及供体来源、配对、输入和头编号。",
           "测试记录见test_results.json和cpu_tests.log。全部任务结束后自动汇总并向GitHub专用分支推送。","",
           "## 论文方法建议","",
           "方法段使用来源独立的梯度置换筛选、配对相对随机供体的超额效应及BY确认描述自适应选头，并报告每模型四类集合、规模和层分布。",
           "当前范围不增加固定预算对照或置信区间。实测完成后根据逐模型结果补充具体分析；正式论文实验结论保持原样。"]
    write(OUT/"summary.json",summary);write_csv(OUT/"summary.csv",summary)
    write(OUT/"test_metrics.json",metrics);write_csv(OUT/"test_metrics.csv",metrics)
    write(OUT/"sensitivity_summary.json",sensitivity);write_csv(OUT/"sensitivity_summary.csv",sensitivity)
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
