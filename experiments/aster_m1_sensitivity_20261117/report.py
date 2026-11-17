"""Four scientific figures and Chinese research/claim reports from recorded results."""
import csv,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .prepare import OUT,REPO
from .audit import table
from .statistics import KINDS,scenarios,jaccard

def fmt(x):
    return "NA" if x is None else f"{x:.3f}"

def save(fig,path):
    fig.tight_layout()
    for ext in ["png","pdf","svg"]:fig.savefig(str(path)+"."+ext,dpi=180,bbox_inches="tight")
    plt.close(fig)

def render(cfg,plan,effect,stats,full,result,cores):
    figures=OUT/"figures";figures.mkdir(exist_ok=True)
    configs=scenarios(cfg)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    labels=dict(gradient_threshold="Gradient robust z",candidate_frequency="Five-fold frequency",
                fdr="BH threshold",direction_consistency="Bootstrap direction frequency")
    for ax,(parameter,values) in zip(axes.flat,cfg["grid"].items()):
        rows=[]
        for value in values:
            name="default" if value==cfg["defaults"][parameter] else next(x["name"] for x in configs
                if x["name"].startswith(parameter+"_") and x[parameter]==value)
            rows.append(full[name])
        for group,color in [("AP","#277DA1"),("CF","#E76F51"),("union","#30343F")]:
            ax.plot(values,[len(x["sets"][group]) for x in rows],"-o",label=group,color=color,markersize=4)
        ax.axvline(cfg["defaults"][parameter],color=".65",linestyle=":",linewidth=1)
        ax.set(xlabel=labels[parameter],ylabel="Selected heads")
        ax.set_xticks(values);ax.grid(alpha=.2)
    axes.flat[0].legend()
    save(fig,figures/"figure1_threshold_counts")
    names=list(full)
    matrix=np.array([[np.nan if (v:=jaccard(full[a]["sets"]["union"],full[b]["sets"]["union"])) is None else v
                      for b in names] for a in names])
    fig,ax=plt.subplots(figsize=(11,9))
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#dddddd")
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap)
    ax.set_xticks(range(len(names)),names,rotation=70,ha="right",fontsize=7)
    ax.set_yticks(range(len(names)),names,fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v=matrix[i,j]
            ax.text(j,i,"NA" if np.isnan(v) else f"{v:.2f}",ha="center",va="center",
                    fontsize=6,color="white" if np.isfinite(v) and v<.5 else "black")
    fig.colorbar(im,ax=ax,label="Union Jaccard")
    save(fig,figures/"figure2_jaccard")
    rng=np.random.default_rng(cfg["seed"]+7000)
    fig,axes=plt.subplots(2,2,figsize=(10,7));effect_rows=[]
    default=full["default"]
    for k,(kind,ax) in enumerate(zip(KINDS,axes.flat)):
        selected=sorted(default["sets"][kind])
        ax.set_title(kind+" (n="+str(len(selected))+" heads)")
        if not selected:
            ax.text(.5,.5,"No selected heads",ha="center",va="center",transform=ax.transAxes)
            ax.set_axis_off();continue
        values=[]
        for field in ["true","null","contrast"]:
            a=effect[field][k][:,selected]
            valid=np.isfinite(a).any(1)
            values.append(np.nanmean(a[valid],axis=1))
        n=len(values[0]);weights=rng.multinomial(n,np.full(n,1/n),size=cfg["bootstrap_draws"])
        jitter=rng.uniform(-.08,.08,n)
        for j,(field,a) in enumerate(zip(["True","Matched control","Difference"],values)):
            boot=weights@a/n;lo,hi=np.quantile(boot,[.025,.975]);mean=float(a.mean())
            ax.scatter(np.full(n,j)+jitter,a,s=12,alpha=.45,color=["#277DA1","#999999","#E76F51"][j])
            ax.errorbar(j,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt="D",color="black",capsize=5,markersize=4)
            effect_rows.append(dict(kind=kind,statistic=field,selected_heads=len(selected),sources=n,
                                    mean=mean,ci_low=float(lo),ci_high=float(hi)))
        ax.set_xticks([0,1,2],["True","Matched control","Difference"])
        ax.axhline(0,color=".65",linewidth=.8)
        ax.set_ylabel("Absolute antisymmetric effect" if k<2 else "Label-aligned signed effect")
    save(fig,figures/"figure3_source_interventions")
    table(OUT/"figure3_source_estimates.csv",effect_rows,fieldnames=["kind","statistic","selected_heads","sources","mean","ci_low","ci_high"])
    core=[r["head"] for r in cores if r["stable_core"]][:24]
    fig,axes=plt.subplots(1,2,figsize=(10,max(3,len(core)*.25+1.5)))
    if core:
        frequencies=np.array([[default["AP_frequency"][h],default["CF_frequency"][h]] for h in core])
        directions=np.array([[np.sign(stats["mean"][k,h])*stats["source_direction_consistency"][k,h]
                               for k in range(4)] for h in core])
        for ax,data,cmap,vmin,vmax,title,columns in [
            (axes[0],frequencies,"viridis",0,1,"Selection frequency in source subsets",["AP","CF"]),
            (axes[1],directions,"RdBu_r",-1,1,"Signed source direction consistency",list(KINDS))]:
            im=ax.imshow(data,vmin=vmin,vmax=vmax,cmap=cmap,aspect="auto")
            ax.set_yticks(range(len(core)),[f"L{h//32} H{h%32}" for h in core],fontsize=8)
            ax.set_xticks(range(len(columns)),columns);ax.set_title(title,fontsize=10)
            fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    else:
        for ax in axes:
            ax.text(.5,.5,"No stable core at frequency >= 0.80",ha="center",va="center",transform=ax.transAxes)
            ax.set_axis_off()
    save(fig,figures/"figure4_core_stability")
    rows=["|配置|A/P|C/F|交集|并集|稳定核心|BY并集|",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name,x in full.items():
        s=x["summary"];n=s["counts"]
        rows.append(f'|{name}|{n["AP"]}|{n["CF"]}|{n["overlap"]}|{n["union"]}|{len(s["core_heads"])}|{s["by_union_count"]}|')
    recommendation=result["recommendation"]
    meanings=dict(A="默认规则满足本轮预设稳定性标准，可以保留。",
                  B="默认规则存在稳定核心，但部分阈值扰动未满足标准；需针对相应阈值设计独立来源确认。",
                  C="现有结果不足以支持固定该默认配置，尚未建立满足预设标准的可重复头集合。")
    n=default["summary"]["counts"]
    failed=[x["configuration"] for x in result["stability"] if not x["passes_criteria"]]
    version=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    paragraphs=[
        "# 新 M1 筛选规则敏感性与稳健性验证",
        f"结论：**{recommendation}。{meanings[recommendation]}**",
        f'Qwen3-8B 的默认配置选出 {n["union"]} 个头：A/P {n["AP"]}、C/F {n["CF"]}、交集 {n["overlap"]}；来源子集入选频率达到 0.80 的默认核心共有 {len(default["summary"]["core_heads"])} 个。没有预设最终头数量。',
        "## 版本、数据与复用",
        f"实际统计执行提交：{version}。GPU 执行提交见 execution_manifest.json。PR #81 原始修补提交 f3cc86a9fa8ef53ff8bb0470496e72185144c7a7；本分支基于 b4cecd8feba6f20aface290aa355d98f11b09b21。",
        "全部分析使用 E1 的 80 个训练来源和 2,540 对配对。候选筛选与验证各 40 个来源，来源不重叠。开发/测试来源不参与新规则筛选、零对照、阈值设定或稳定性统计。",
        f'匹配对照后有效配对数：{json.dumps(plan["matched_pairs_by_kind"],ensure_ascii=False)}；有效来源数：{json.dumps(plan["eligible_sources_by_kind"],ensure_ascii=False)}。32 对缺少同来源、同类别、同标签转换的其他配对，报告为不具备对照；其他有效配对未筛掉负效应。',
        f'复用 {plan["reused_unique_scores"]:,} 个真实/自身替换分数。新增 {plan["new_unique_patch_forwards"]:,} 个匹配对照分数，另计划 {plan["new_baseline_forwards"]:,} 次基线重放；实际调用、恢复复用及显存见 workers/*/completion.json。全部所需真实修补已覆盖，新增真实修补为 0。',
        "旧零对照、旧五折功能分组与冻结集合诊断的版本和复用边界见 history_search/REPORT.zh.md。旧实验原文件保持不变。",
        "## 配置与完整结果",
        "默认：梯度稳健标准分数 2.0、五折入选频率 0.80、BH 阈值 0.05、方向一致率 0.80、至少 5 个有效来源。扫描和两个联合配置均在新对照产生前固定；详见 configuration.json 与 PROTOCOL.zh.md。",
        "\n".join(rows),
        "全部 12 个配置均对各自候选家族重新计算 BH。all_configuration_heads.csv.gz 保存全部 1,152×4 个类别—头位置在每个配置下的筛选状态与统计量；未成为候选或未测量的值标记缺失，不置零。32 个配对的对照不可用不等于整体配置不可用；有效来源数已单列。",
        "## 稳定性与默认建议",
        "本轮标准为并集 Jaccard ≥0.80、可比较功能组 Jaccard ≥0.70、并集数量变化≤25%、来源子集频率≥0.80的核心覆盖默认集合至少80%。完整结果见 stability_criteria.csv 和 source_subset_counts.csv。",
        "包含默认值并符合标准的连续扫描取值："+json.dumps(result["stable_scan_values"],ensure_ascii=False)+"。",
        "未满足全部操作标准的配置："+("、".join(failed) if failed else "无")+"。极端联合配置作描述性比较；不会据此替换默认阈值。",
        "64 次子集分析分别从候选和验证来源中不放回保留 80%，每次重算候选、来源效应统计、检验、BH 与方向一致率。子集之间有重叠，不作为独立重复证据。core_heads.csv 保存全部默认头的入选频率与功能集合归属。",
        "## 统计审计",
        "双向修补使用目标自身替换参照，反对称效应为 (uL−uR)/2。A/P 检验绝对幅度超过匹配对照的单侧假设；C/F 检验标签对齐有向响应相对匹配对照的双侧假设。不同统计量分别报告，原始效应大小不作跨组强弱排名。",
        "符号翻转以来源为单位，采用 262,143 次随机翻转；小样本枚举。随机检验 p=(b+1)/(B+1)，不产生零 p 值。每配置的 BH 家族包含全部候选类别—头假设；来源不足者 p=1。来源聚类 bootstrap 为1,999次，均值95%区间为百分位区间。",
        "符号翻转的均值检验依赖来源对比的对称性，来源符号检验另列诊断。BH 的依赖条件未经数据证明；BY 列提供更保守的依赖诊断。方向一致率附加筛选以及物理头去重并集不自动继承逐头 FDR 保证。",
        "图3与逐头区间描述当前来源上的效应，未进行选头后的区间校正。扫描用于探索稳定区间，不构成12次独立确认。新默认值若需改变，须在另行确认的独立训练来源上验证。",
        "## 四张主图",
        "![阈值与头数量](figures/figure1_threshold_counts.png)",
        "![配置间Jaccard](figures/figure2_jaccard.png)",
        "![来源级真实与匹配对照效应](figures/figure3_source_interventions.png)",
        "图3在A/P与C/F各自统计量内比较；点为来源，误差线为来源bootstrap区间。无入选头的类别显示空集合，未改用其他头补图。",
        "![稳定核心的频率与方向](figures/figure4_core_stability.png)",
        "图4按头编号展示前24个稳定核心；全量统计在CSV中。共享或分化的集合归属只描述本次干预响应，不证明机制完全独立。",
        "## 已支持与尚未支持",
        f'已测量：固定规则在当前训练来源上产生 {n["union"]} 个默认头；12个配置的数量、集合交集、Jaccard、来源效应和64次重选频率已完整报告。是否支持稳定识别以本报告 {recommendation} 类建议及逐项标准为准。',
        "尚未证明：在全新来源或其他模型上的可重复性、功能机制完全独立、最终任务准确率最优或有提升、探索后更换默认规则的无偏确认。此次未训练M2/M3、未修改M4，也没有用最终准确率选头。",
        "统计方法依据：[BH](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)、[BY及依赖条件](https://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf)、[随机置换p值](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)。",
    ]
    (OUT/"REPORT.zh.md").write_text("\n\n".join(paragraphs)+"\n",encoding="utf-8")
    paper=[
        "### M1 筛选规则的敏感性与跨来源稳定性",
        "我们在 Qwen3-8B 上评估统计式 M1 筛选规则。80个训练来源固定分为40个候选来源与40个验证来源。候选侧采用五折来源筛选，验证侧进行双向激活修补，并与同来源、类别及标签转换的匹配供体对照比较。A/P使用幅度差的单侧检验，C/F使用标签对齐有向差的双侧检验；每个配置在其候选类别—头家族内重新执行BH校正。",
        f'以2.0/0.80/0.05/0.80为默认阈值，M1选出{n["union"]}个头，其中A/P集合{n["AP"]}个，C/F集合{n["CF"]}个，交集{n["overlap"]}个。单因素与两组预定联合配置共12组，均复用真实修补和匹配对照缓存。64次训练来源子集重选中，{len(default["summary"]["core_heads"])}个默认头的入选频率达到0.80。包含默认值且满足预设稳定性标准的扫描取值为：'+json.dumps(result["stable_scan_values"],ensure_ascii=False)+"。",
        meanings[recommendation]+"本结果评价的是固定训练来源上筛选规则的稳定性和相对匹配对照的干预响应。多次扫描及重叠来源子集按探索性分析解释，效应区间为来源聚类重采样的描述性区间。A/P与C/F采用不同统计量，集合差异不用于宣称因果机制完全独立。本实验未使用下游准确率选择头集合。",
    ]
    (OUT/"PAPER_SECTION.zh.md").write_text("\n\n".join(paper)+"\n",encoding="utf-8")
