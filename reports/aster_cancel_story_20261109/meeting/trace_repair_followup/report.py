"""Render the frozen development/final comparison and two experiment figures."""
from pathlib import Path
import csv,gzip,json,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def js(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def rr(p):
    p=Path(p)
    opener=gzip.open if p.suffix==".gz" else open
    if not p.exists():return []
    with opener(p,"rt",encoding="utf-8") as f:return list(csv.DictReader(f))
def num(r,k):
    v=r.get(k)
    return float(v) if v not in ("",None,"None") else None
def pct(x):return "未评价" if x is None else f"{100*x:.2f}%"
def fmt(x):return "未评价" if x is None else f"{x:.4f}"
def changes(pathrows,a,b,group):
    a={(r["case_id"],r["variant"]):r for r in pathrows if r["method"]==a and (group=="A")==(r["variant"]!="original")}
    b={(r["case_id"],r["variant"]):r for r in pathrows if r["method"]==b and (group=="A")==(r["variant"]!="original")}
    keys=a.keys()&b.keys()
    return sum(a[k]["correct"]=="False" and b[k]["correct"]=="True" for k in keys),sum(a[k]["correct"]=="True" and b[k]["correct"]=="False" for k in keys)
def assess_contrast(metrics):
    a=metrics.get("A",{});n=metrics.get("natural",{})
    aci=a.get("difference_ci95");nci=n.get("difference_ci95")
    interference_gain=bool(aci is not None and aci[0]>0)
    natural_noninferiority=bool(nci is not None and nci[0]>=-.005)
    return dict(interference_gain=interference_gain,natural_noninferiority=natural_noninferiority,
        demonstrated=interference_gain and natural_noninferiority,
        interference_ci_available=aci is not None,natural_ci_available=nci is not None)

def evidence_scope_note(cfcover):
    sources=int(float(cfcover.get("sources") or 0));count=int(float(cfcover.get("n") or 0))
    if not sources or not count:return "没有可评价的C/F配对，证据保护结论未评价。"
    if sources<10:return f"C/F覆盖{count}对、{sources}个来源，来源较少，只能作为有限的个例代价观察。"
    return f"C/F覆盖{count}对、{sources}个来源；证据保护结论只适用于这些实际覆盖来源。"

def response_baseline_note(cfcover):
    raw_mean=num(cfcover,"raw_signed_response_mean")
    if raw_mean is None:return "原始C/F标签方向平均分数响应未评价。"
    text=f"原始C/F标签方向平均分数响应为{raw_mean:.4f}。"
    if raw_mean<=0:
        text+="方向响应基线为非正，ER比值不表示对正确证据的保护，不据此称证据保护成功。"
    return text+"ER只衡量相对原分数响应；两侧是否正确另外报告为C/F配对正确率。"

def render():
    ev=HERE/"evaluation";frozen=js(ev/"frozen.json")
    if not frozen:raise RuntimeError("Run development evaluation and freeze before rendering")
    selection=frozen["selection"];baseline=selection["baseline"]["name"];trace=selection["trace"]["name"]
    stages=[s for s in ("dev","final") if (ev/s/"coverage.json").exists()]
    figures=ev/"figures";figures.mkdir(exist_ok=True)
    stage=stages[-1]
    paths=rr(ev/stage/"path_results.csv.gz");pathsum=rr(ev/stage/"path_summary.csv");pairs=rr(ev/stage/"pair_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,group,title in zip(axes,("natural","A"),("Natural paths","A-perturbed paths")):
        labels=["Best baseline vs raw","Selected trace vs baseline"]
        vals=[changes(paths,"raw",baseline,group),changes(paths,baseline,trace,group)]
        x=np.arange(2);ax.bar(x-.18,[v[0] for v in vals],.36,label="Fixed",color="#2166ac")
        ax.bar(x+.18,[v[1] for v in vals],.36,label="New errors",color="#b2182b")
        ax.set_xticks(x,labels,fontsize=8);ax.set_title(title);ax.set_ylabel("Case-variant count");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.suptitle(stage+" frozen comparison");fig.tight_layout()
    fig.savefig(figures/"fixed_vs_new_errors.png",dpi=180);fig.savefig(figures/"fixed_vs_new_errors.pdf");plt.close(fig)
    meta={m["name"]:m for m in frozen["models"]}
    a={r["method"]:num(r,"source_macro_accuracy") for r in pathsum if r["group"]=="A"}
    cf={r["method"]:num(r,"correction_rms") for r in pairs if r["group"]=="CF" and r["scope"]=="gold_labelled"}
    fig,ax=plt.subplots(figsize=(6.8,4.8))
    labelled=set()
    for name,m in meta.items():
        if name not in a or name not in cf or a[name] is None or cf[name] is None or a.get(baseline) is None:continue
        family=m["family"];main=m.get("main",True);color="#2166ac" if family=="trace" else "#b2182b"
        label=("Trace" if family=="trace" else "Baseline")+("" if main else " seed")
        ax.scatter(cf[name],100*(a[name]-a[baseline]),c=color,marker="o" if main else ".",s=55 if main else 25,
            alpha=.85 if main else .3,label=label if label not in labelled else None);labelled.add(label)
        if name in (trace,baseline):
            ax.annotate("selected "+family,(cf[name],100*(a[name]-a[baseline])),xytext=(4,5),textcoords="offset points",fontsize=8)
    ax.axhline(0,c="gray",lw=.7);ax.set_xlabel("Evidence correction RMS (logit units)")
    ax.set_ylabel("A-path source-macro gain vs selected baseline (pp)")
    ax.set_title(stage+"; no final selection");ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(figures/"path_gain_evidence_cost.png",dpi=180);fig.savefig(figures/"path_gain_evidence_cost.pdf");plt.close(fig)
    lines=["# 痕迹修正后续实验报告","",
      "本报告使用真实Qwen3-8B评分与保存的修正参数。所有方法、预算和种子在最终来源评分前冻结；最终评价不再选择模型。原有开发/测试和后续已查看结果只作为既有证据，不称为全新独立测试。","",
      "数据来自MEGA的指南决策树及其生成病例。旧拟合集包含80个指南来源、1712个原始生成病例；新开发集包含20个来源、256个原始生成病例；最终集包含60个来源、802个原始生成病例。新开发集与最终集每个来源最多15个病例。每个病例分别评价原始输入及处置删除、中性替换、处置互换三种A变体；变体不增加独立来源或病例数。","",
      "“新来源”指在本项目可核对的使用记录中，此前未参与选择或分析的MEGA指南来源，不是新增临床患者。来源名单、划分与使用记录保存在数据清单中。","",
      "C/F配对由同一Qwen模型进行源规则、病例事实及编辑标注，再通过程序检查与标签一致性验证，没有临床人工核验。旧拟合集的135个C/F配对用于训练证据约束；本轮新来源C/F配对仅用于冻结方法评价。P类作为节点配对诊断，不作为本轮主路径训练数据或主路径指标。","",
      "## Q1：自然判断错误是否与干扰有关？",""]
    for s in stages:
        d=js(ev/s/"error_diagnosis.json")
        lines.append(f"{s}原始自然路径有{d.get('raw_natural_error_paths',0)}条错误，其中{d.get('A_covered_first_errors',0)}条首错节点具有同病例A变体评分；{d.get('first_error_with_correct_A_variant',0)}条在至少一种A变体上判断正确，{d.get('all_observed_A_variants_wrong',0)}条在所有已观察A变体上仍错。每一侧都按独立提供的金标判断，未把full或编辑侧分数当成真值。")
    lines+=["","上述Q1诊断使用本轮新开发及最终来源的完整A覆盖：同一病例、同一节点的原始输入和三种A变体均进行真实全树评分，每一侧按各自标签判断。这区分了受控改动下可观察的改判和持续错误，不能单独证明错误的临床语义原因。旧开发集中19条PMA相关错误只属于既有诊断，不作为本轮新来源诊断或验证样本；既有文本与参考标签疑义也未用于修改本次标签。","",
      "## Q2：痕迹是否提供了校准和对照以外的信息？","",
      f"开发集选中基线为 `{baseline}`，痕迹方法为 `{trace}`。先要求自然路径来源等权正确率相对原模型下降不超过0.5个百分点，再最大化A路径正确率；同分优先更少参数、更小证据预算。仅比较预先标记的主候选，神经网络单种子不参与选择。",
      f"痕迹选择状态：{selection['trace']['status']}。若未通过自然路径条件，仍冻结并报告最优痕迹候选，保留阴性结果。",""]
    lines+=["| 划分/方法 | 自然路径来源等权正确率 | A路径来源等权正确率 | 自然修复/新增错误 | A修复/新增错误 |",
            "|---|---:|---:|---:|---:|"]
    for s in stages:
        ss=rr(ev/s/"path_summary.csv");pr=rr(ev/s/"path_results.csv.gz")
        for name in dict.fromkeys(("raw",baseline,trace)):
            ni=next((r for r in ss if r["method"]==name and r["group"]=="natural"),{})
            ai=next((r for r in ss if r["method"]==name and r["group"]=="A"),{})
            nf,nn=changes(pr,baseline,name,"natural");af,an=changes(pr,baseline,name,"A")
            lines.append(f"| {s}/{name} | {pct(num(ni,'source_macro_accuracy'))} | {pct(num(ai,'source_macro_accuracy'))} | {nf}/{nn} | {af}/{an} |")
    lines+=["","修复与新增错误以开发集选定的基线为参照，按病例变体计数。正确率先在同病例内平均有效A变体，再平均病例，最后平均指南来源；普通病例加权结果和所有预算、种子均保存在CSV。","",
      "## Q3：冻结关系能否迁移到新来源并提高正确率？",""]
    for s in stages:
        c=js(ev/s/"contrast.json");coverage=js(ev/s/"coverage.json")
        lines.append(f"{s}包含{coverage.get('sources')}个来源、{coverage.get('cases')}个病例变体；缺失预测逐方法列在coverage.json。")
        for group,label in (("natural","自然路径"),("A","A路径")):
            m=c.get("metrics",{}).get(group,{})
            if "difference" not in m:
                lines.append(f"{label}未形成完整评价，未将缺失评分当成错误或按方法删除后计算主指标。");continue
            lo,hi=m["difference_ci95"]
            lines.append(f"{label}相对冻结基线变化{100*m['difference']:+.2f}个百分点，2000次配对来源自助重采样95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines+=["","最终阶段的主要比较为开发集选中的痕迹方法对选中的最强基线。其余最终结果为预先列出的辅助比较，不用于重新选择模型。置信区间跨0时，不把观察到的均值提高解释为明确的跨来源收益。","",
      "## Q4：干扰降低的证据保护代价是什么？","",
      "| 划分/方法 | AP方差降低VR | 原分数响应比ER | CF配对正确率 | CF修正RMS | 方向退化 | 超训练证据预算 |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for s in stages:
        ps=rr(ev/s/"pair_summary.csv")
        for name in dict.fromkeys(("raw",baseline,trace)):
            ap=next((r for r in ps if r["method"]==name and r["group"]=="AP" and r["scope"]=="gold_labelled"),{})
            cfrow=next((r for r in ps if r["method"]==name and r["group"]=="CF" and r["scope"]=="gold_labelled"),{})
            lines.append(f"| {s}/{name} | {pct(num(ap,'VR'))} | {pct(num(cfrow,'ER'))} | {pct(num(cfrow,'paired_accuracy'))} | {fmt(num(cfrow,'correction_rms'))} | {cfrow.get('direction_degraded') or '未评价'} | {cfrow.get('exceeds_fit_budget') or '不适用'} |")
    for s in stages:
        ps=rr(ev/s/"pair_summary.csv")
        cfcover=next((r for r in ps if r["method"]=="raw" and r["group"]=="CF" and r["scope"]=="gold_labelled"),{})
        pcover=next((r for r in ps if r["method"]=="raw" and r["group"]=="P" and r["scope"]=="gold_labelled"),{})
        lines.append(f"{s}证据覆盖为{cfcover.get('n','0')}个C/F配对、{cfcover.get('sources','0')}个来源；P类有效配对为{pcover.get('n','0')}。")
        lines.append(evidence_scope_note(cfcover))
        lines.append(response_baseline_note(cfcover))
    lines.append("旧strict/balanced的历史开发集RMS预算档位与本轮拟合集逐对最大差约束不同，未将其超预算计数填成0。")
    lines+=["","主表使用带金标的配对及来源等权统计；全部可评分节点的A/P统计另存。证据评价仅覆盖实际存在有效C/F配对的来源，不把它们冒称覆盖全部最终来源。相同训练预算不等于留出来源具有相同实际证据RMS，应结合权衡图评价。ER是平均有方向原分数响应之比，不能单独等同正确证据。C/F配对正确率要求两侧各自按标签判断且同时正确，并按来源等权统计；ER也不能替代逐对方向退化与超预算数量。","",
      "原有条件输入和证据对齐输入仅复用已经完成的既有开发结果，本次不新增这些模板在新来源的Qwen评分，也不把它们的收益归于痕迹。新来源的主要比较均使用固定原始提示与单节点痕迹修正。条件输入删除其他节点与处置，使A输入相同，其A不变性来自输入处理。","",
      "## 汇总",""]
    last=js(ev/stage/"contrast.json").get("metrics",{})
    a=last.get("A",{});n=last.get("natural",{})
    if "difference" in a:
        lines.append(f"实验结果：在{stage}中，选定痕迹方法相对最强基线的A路径变化为{100*a['difference']:+.2f}个百分点，自然路径变化为{100*n.get('difference',0):+.2f}个百分点。")
        verdict=assess_contrast(last)
        demonstrated=verdict["demonstrated"]
        lines.append("干扰增益成立："+("是" if verdict["interference_gain"] else "否")+"（要求A路径差值95%区间下限大于0）。")
        lines.append("自然非劣成立："+("是" if verdict["natural_noninferiority"] else "否")+"（要求自然路径差值95%区间下限不低于−0.5个百分点）。")
        lines.append("总体成功判定要求上述两项同时成立，不能以自然路径点估计替代其置信区间。")
        lines.append("科学结论："+("在本次冻结来源评价中，痕迹提供了校准与对照以外的路径判断收益；证据代价仍需结合CF指标解释。" if demonstrated else "本次结果尚不足以确认痕迹在新来源上提供稳定的额外判断收益；预测干扰信息与改善最终判断应分别评价。"))
    lines+=["","图1：修复与新增错误，分别展示自然路径和A路径。","",
      "![修复与新增错误](evaluation/figures/fixed_vs_new_errors.png)","",
      "图2：A路径增益与证据改动的关系。","",
      "![路径增益与证据代价](evaluation/figures/path_gain_evidence_cost.png)","",
      "运行配置与逐对、逐路径、逐来源结果见evaluation/；模型训练、内部来源选择及完整参数见training/。"]
    rollout=js(ev/"verification_rollout.json");replay=js(HERE/"training/api_validation.json")
    lines+=["","## 交付验证","",
        f"路径软件验证使用既有真实开发缓存重放：{rollout.get('complete_path_matches','待执行')}/{rollout.get('cases','待执行')}条路径与保存的完整路径和叶子一致。金标符号oracle仅用于软件检查，不属于实验结果。",
        "training/replay_parameters.py从真实拟合集缓存读取节点，独立重载保存参数，并比较所有模型的批量和单节点预测；不重新训练，也不使用开发或最终来源进行参数选择。",
        f"参数重放状态：{replay.get('status','待流水线执行')}；覆盖{replay.get('rows','待执行')}个真实拟合节点、{replay.get('models','待执行')}个模型。详细误差保存在training/api_validation.json。"]
    (HERE/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(HERE/"REPORT.zh.md"),stage=stage,figures=4)))
if __name__=="__main__":render()

