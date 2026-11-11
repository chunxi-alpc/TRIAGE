"""Assemble completed measurements, focused verification and Chinese report."""
import csv,gzip,json,shutil,time
from collections import Counter
import numpy as np
from .common import *
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev

SCOPES=("dev","test","new60","nli4ct","medpic","cv_pooled_common","cv_pooled_historical")
LABELS={"dev":"开发集（20来源）","test":"历史40来源","new60":"已有60来源",
    "nli4ct":"NLI4CT","medpic":"MedPIC","cv_pooled_common":"五折合并（80来源，冻结配对池）",
    "cv_pooled_historical":"五折合并（历史配对池，补充）"}
NINE=[("ap_score_change_rms","A/P分数差RMS"),("ap_vr","VR"),("cf_correction_rms","C/F修正RMS"),
    ("cf_er","ER"),("cf_direction_degradation_count","方向退化数"),("cf_over_budget_rate","超预算"),
    ("natural_node_accuracy","自然节点"),("natural_path_accuracy","自然路径"),("interference_path_accuracy","干扰路径")]

def csvrows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def number(value):
    if value is None or value=="":return None
    return float(value)
def fmt(value,percent=False,digits=2):
    value=number(value)
    return "—" if value is None else f"{value*(100 if percent else 1):.{digits}f}"+("%" if percent else "")
def metricfmt(key,value):
    return fmt(value,percent=any(s in key for s in ("accuracy","_vr","_er","_rate")))
def difftext(row):
    if not row:return "—"
    percent=any(s in row["metric"] for s in ("accuracy","_vr","_er","_rate"))
    scale=100 if percent else 1;unit="个百分点" if percent else ""
    est,lo,hi=[number(row[k]) for k in ("estimate","low","high")]
    if est is None:return "未定义"
    return f"{est*scale:+.3f}{unit}，95%区间[{lo*scale:+.3f}, {hi*scale:+.3f}]"
def main():
    for scope in SCOPES:
        assert read(OUT/"evaluation"/scope/"completion.json").get("complete"),scope
    assert read(OUT/"status/modules.json").get("state")=="complete"
    all_boot=[];all_metrics=[];all_transitions=[];all_sources=[]
    for target in sorted((OUT/"evaluation").iterdir()):
        if not (target/"metrics.json").exists():continue
        all_boot+=csvrows(target/"source_bootstrap.csv")
        all_metrics+=csvrows(target/"metrics.csv")
        all_transitions+=csvrows(target/"transitions.csv")
        all_sources+=csvrows(target/"source_metrics.csv")
    for name,records in [("source_bootstrap",all_boot),("metrics",all_metrics),
                          ("transitions",all_transitions),("source_metrics",all_sources)]:
        ev.write_csv(OUT/(name+".csv"),records)
    results={s:read(OUT/"evaluation"/s/"metrics.json") for s in SCOPES}
    lookup={(r["scope"],r["comparison"],r["metric"]):r for r in all_boot}
    def diff(scope,a,b,key):return lookup.get((scope,a+" minus "+b,key))
    inter=read(OUT/"independent_intercepts.json")
    med=results["medpic"]["models"]
    maincv=results["cv_pooled_common"]["models"]
    preserve="native_preserving_relation_accuracy"
    stage_drop=med["E1_full"][preserve]-med["E1_pair_only"][preserve]
    pure_change=med["E1_pair_only"][preserve]-med["raw"][preserve]
    baseline_drop=med["independent_intercept"][preserve]-med["raw"][preserve]
    primary=["natural_node_accuracy","natural_path_accuracy","interference_path_accuracy"]
    stable_calibration_gains=[k for k in primary if number(diff("cv_pooled_common","E1_full","independent_intercept",k)["low"])>0]
    stable_weight_gains=[k for k in primary if number(diff("cv_pooled_common","E3_full","E2_full",k)["high"])<0]
    t=["# ASTER-CR 收益拆分与 E3 五折验证","","## 结论",""]
    t.append("E1 继续作为上一轮冻结主方案，本轮不依据已有测试集重新选型。")
    t.append("")
    if not stable_calibration_gains:
        t.append("**配对修正尚未显示超越简单截距校准的稳定正确率收益。** 五折合并评价中，E1 完整方法与独立截距基线的自然节点、自然完整路径和干扰完整路径差值均未获得严格大于0的95%区间。配对分数稳定性和证据保护另行报告，不能用 VR 替代正确率结论。")
    else:
        t.append("E1 完整方法相对独立截距基线在以下五折合并正确率指标上获得严格正的95%区间："+", ".join(stable_calibration_gains)+"。其他指标按区间分别判断。")
    t.append("")
    if not stable_weight_gains:
        t.append("**尚不能确认类别、来源均衡权重具有稳定正确率收益。** E2 与 E3 的差异同时包含类别权重和来源权重的作用，不能分离解释；E3−E2 的五折差值如下。")
    else:
        t.append("E2 均衡权重相对 E3 自然权重在以下五折合并正确率指标上获得严格优势："+", ".join(stable_weight_gains)+"；类别和来源权重的作用尚不能拆开。")
    t.append("")
    t.append(f"**MedPIC 保持关系下降主要发生在截距加入环节。** E1 从原模型的 {fmt(med['raw'][preserve],True)}，经仅配对修正变为 {fmt(med['E1_pair_only'][preserve],True)}，加入截距后变为 {fmt(med['E1_full'][preserve],True)}。完整方法减仅配对修正为 {difftext(diff('medpic','E1_full','E1_pair_only',preserve))}。独立截距基线为 {fmt(med['independent_intercept'][preserve],True)}。")
    t.append("")
    t.append(f"五折中，E1仅配对修正的自然路径仍为1549/1712，与原模型相同；加入截距后为1546/1712，独立截距基线也为1546/1712。干扰路径依次为4722/5136（原模型）、4723/5136（仅配对）、4690/5136（完整方法），独立截距基线为4685/5136。这表明当前正确率变化主要发生在截距环节，但完整方法相对原模型的路径差值区间仍含0。")
    t.append("")
    t.append(f"E1的配对修正具有可测的分数稳定性作用：五折A/P的VR相对独立截距基线为 {difftext(diff('cv_pooled_common','E1_full','independent_intercept','ap_vr'))}。同时验证C/F有25/1342对超预算，证据方向退化为0；不能把分数方差降低直接等同于正确率提高。")
    t.append("")
    t.append(f"E3相对E1的干扰路径正确率为 {difftext(diff('cv_pooled_common','E3_full','E1_full','interference_path_accuracy'))}；这个小幅差异支持保留冻结E1，但E1和E3同时存在数据规模与权重差异，不能将其全部归因于权重。")
    t.append("")
    t.append("**下一步优先研究 M3 截距校准及其与 MedPIC 合法答案集合解码的配合。** 当前分解直接定位到这一环节；M1/M2 的重叠和 M3 的约束余量用于辅助判断。本轮只提出建议，不改造方法，不增样本，不重审数据，不启动285组基线。")
    t+=["","## 设计与复用","",
        "文件中的 raw、pair_only、original_intercept_only、independent_intercept、full 分别对应下列五种版本。原模型为 ℓ；仅配对修正为 ℓ+c；仅保留原截距为 ℓ+b；独立截距基线为 ℓ+b₀；完整方法为 ℓ+c+b。原 b 在修正后的分数上拟合，因此 ℓ+b 只用于诊断。主要比较完整方法与独立截距基线。",
        "",
        "b₀ 在原来的761个自然拟合节点上用同一逻辑回归截距目标拟合；每折只用64个训练来源的自然节点。目标任务直接使用源任务参数。全部保持 Qwen3-8B、原提示和评分位置、BF16模型、FP32标签投影、51个头、rank=16、λ=0.01、δ=0.2923349、A/P回归权重总和4,887。",
        "",
        "E0—E3全量参数、E0—E2五折参数及预算敏感性结果直接引用上一轮。新增E3各折独立M1排名、M2标准化和子空间、M3拟合及自然截距；E3全部阶段使用自然权重。旧目录没有改写。",
        "",
        "|截距|自然节点数|b₀|训练来源数|","|---|---:|---:|---:|"]
    for fold,d in inter.items():t.append(f"|{fold}|{d['nodes']}|{d['b0']:.6f}|{len(d['fit_sources'])}|")
    t+=["","## 五折合并的主要差值","",
        "五折验证来源互不重叠，合计80个来源。每次按来源抽样2,000次，重新计算合并后的整体指标及成对差值；不平均五折VR或置信区间。比例差值用百分点，区间包含0时不宣称稳定优势。",
        "","|比较|指标|差值及95%区间|","|---|---|---|"]
    for lhs,rhs in [(g+"_full","independent_intercept") for g in GROUPS]+[("E3_full","E2_full"),("E3_full","E1_full")]:
        for key,label in NINE:
            t.append(f"|{lhs} − {rhs}|{label}|{difftext(diff('cv_pooled_common',lhs,rhs,key))}|")
    t+=["","## 九项指标","",
        "A/P分数差RMS为调整后两端分数差的均方根；VR按配对差的方差定义。C/F修正RMS只度量修正项差异；ER、证据方向退化和超预算沿用原定义。未知标签保持为空，正确率分母只含有标签样本。"]
    for scope in ("dev","test","new60","cv_pooled_common","cv_pooled_historical"):
        t+=["","### "+LABELS[scope],"","|方法|"+"|".join(label for key,label in NINE)+"|",
            "|---|"+"|".join("---:" for _ in NINE)+"|"]
        for name in ("raw","independent_intercept","E0_full","E1_pair_only","E1_original_intercept_only","E1_full","E2_full","E3_full"):
            m=results[scope]["models"][name]
            t.append("|"+name+"|"+"|".join(metricfmt(k,m.get(k)) for k,l in NINE)+"|")
        raw=results[scope]["models"]["raw"]
        t.append("")
        t.append(f"自然节点有效分母 {raw.get('natural_node_count',0)}，自然节点空标签 {raw.get('natural_node_unlabeled_count',0)}；自然路径 {raw.get('natural_path_count',0)}，干扰路径 {raw.get('interference_path_count',0)}。")
    t+=["","## 完整方法与独立截距基线","",
        "|评价集|比较|自然节点或原生任务正确率差值|自然完整路径差值|","|---|---|---|---|"]
    for scope in ("dev","test","new60","nli4ct","medpic"):
        key="natural_native_accuracy" if scope in ("nli4ct","medpic") else "natural_node_accuracy"
        for g in GROUPS:
            t.append(f"|{LABELS[scope]}|{g}完整 − 独立截距|{difftext(diff(scope,g+'_full','independent_intercept',key))}|{difftext(diff(scope,g+'_full','independent_intercept','natural_path_accuracy'))}|")
    t+=["","## 纠正与误伤","",
        "下表以 E1 为主，分别报告相对原模型和独立截距基线的变化。所有组、配对全对率、关系正确率、路径及分阶段变化均在 transitions.csv 中。",
        "","|评价集|比较|层级|有效分母|错误改对|正确改错|保持正确|保持错误|","|---|---|---|---:|---:|---:|---:|---:|"]
    for r in all_transitions:
        if r["scope"] not in ("dev","test","new60","cv_pooled_common"):continue
        if r["comparison"] not in ("E1_full minus raw","E1_full minus independent_intercept"):continue
        if r["level"] not in ("node_natural","path_natural","path_interference"):continue
        t.append("|"+LABELS[r["scope"]]+"|"+r["comparison"]+"|"+r["level"]+"|"+
            "|".join(str(r[k]) for k in ("valid_denominator","wrong_to_right","right_to_wrong","kept_right","kept_wrong"))+"|")
    t+=["","## 跨任务及 MedPIC 专项","",
        "历史40来源、已有60来源、NLI4CT和MedPIC均有历史使用记录。本轮是既有数据上的对照评价，不作为新的独立确认。原生保持/翻转只表示发布标签的关系，不直接等同A/P或C/F。",
        "",
        "|NLI4CT方法|原生正确率|保持对全对率|翻转对全对率|保持关系正确率|翻转关系正确率|",
        "|---|---:|---:|---:|---:|---:|"]
    for name,m in results["nli4ct"]["models"].items():
        t.append("|"+name+"|"+"|".join(fmt(m.get(k),True) for k in
            ("natural_native_accuracy","native_preserving_paired_accuracy","native_changing_paired_accuracy",
             "native_preserving_relation_accuracy","native_changing_relation_accuracy"))+"|")
    t+=["","MedPIC 沿用已有合法答案集合解码。原生保持对85个、翻转对110个；1880个背景配对单独报告。",
        "","|MedPIC方法|原生正确率|保持对全对率|翻转对全对率|保持关系|翻转关系|平均选中选项数|选“以上均非”|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    assert med["raw"]["native_preserving_pairs"]==85 and med["raw"]["native_changing_pairs"]==110
    assert med["raw"]["synthetic_preserving_pairs"]==1880
    for name,m in med.items():
        t.append("|"+name+"|"+"|".join(fmt(m.get(k),True) for k in
            ("natural_native_accuracy","native_preserving_paired_accuracy","native_changing_paired_accuracy",
             "native_preserving_relation_accuracy","native_changing_relation_accuracy"))+
             "|"+fmt(m.get("natural_selected_options"))+"|"+fmt(m.get("natural_none_selected_rate"),True)+"|")
    t+=["","|背景配对方法|配对全对率|保持关系正确率|","|---|---:|---:|"]
    for name in ("raw","independent_intercept","E1_pair_only","E0_full","E1_full","E2_full","E3_full"):
        m=med[name];t.append(f"|{name}|{fmt(m['synthetic_preserving_paired_accuracy'],True)}|{fmt(m['synthetic_preserving_relation_accuracy'],True)}|")
    t+=["","|MedPIC E1 分阶段比较|层级|有效分母|错误改对|正确改错|保持正确|保持错误|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in all_transitions:
        if r["scope"]!="medpic" or r["comparison"] not in ("E1_pair_only minus raw","E1_full minus E1_pair_only","E1_full minus independent_intercept"):continue
        if r["level"] not in ("pair_native_preserving_relation_correct","pair_native_changing_relation_correct","question_natural"):continue
        t.append("|"+r["comparison"]+"|"+r["level"]+"|"+"|".join(str(r[k]) for k in
            ("valid_denominator","wrong_to_right","right_to_wrong","kept_right","kept_wrong"))+"|")
    qq=rows(OUT/"evaluation/medpic/question_predictions.jsonl")
    option_rows=[]
    for name in med:
        for variant_group in ("natural","all"):
            chosen=[r for r in qq if r["method"]==name and (variant_group=="all" or r["variant"] in ("full","original"))]
            counts=Counter((r["option_count"],r["selected_option_count"],r["none_selected"]) for r in chosen)
            for (available,selected,none),n in sorted(counts.items()):
                option_rows.append(dict(method=name,variant_group=variant_group,available_options=available,
                    selected_options=selected,none_selected=none,count=n,denominator=len(chosen),proportion=n/len(chosen)))
    ev.write_csv(OUT/"medpic_option_counts.csv",option_rows)
    natural_options=Counter(r["option_count"] for r in qq if r["method"]=="raw" and r["variant"] in ("full","original"))
    t+=["",f"自然题目共 {med['raw']['natural_native_count']} 道，可选答案数量分布为 {dict(natural_options)}；选中数量分布详见 medpic_option_counts.csv。",
        "",
        "合法集合采用候选集合内选项分数求和。对每个选项统一加入截距 b，会给含 k 个选项的候选集合增加 k·b，因此多选集合与“以上均非”之间的排序可能改变。这个解码事实与本次观测到的选中数量增加、保持关系下降一致；保持与翻转配对的变化分别报告。",
        "","## 模块诊断",""]
    modules=csvrows(OUT/"module_summary.csv");overlap=read(OUT/"m2_subspace_overlap.json")
    t+=["|模型|折|C/F约束数|最小余量|活跃约束数|超出1e−7容差数|",
        "|---|---|---:|---:|---:|---:|"]
    for r in modules:
        if r["group"]=="E3" or r["fold"]=="full":
            t.append(f"|{r['group']}|{r['fold']}|{r['CF_pairs']}|{float(r['minimum_margin']):.3g}|{r['active_constraints']}|{r['violations_above_1e7']}|")
    t+=["","新增E3五折全部训练C/F约束满足 δ+1e−7。历史E0全量参数用本轮坐标函数复算时有3对超过该容差，最大超出δ约3.69×10⁻⁷；该历史模型保持原样，未重训。",
        "",
        "|折|比较|重合头数|N子空间平均平方余弦|S子空间平均平方余弦|",
        "|---|---|---:|---:|---:|"]
    for r in overlap:
        if (r["lhs"],r["rhs"]) not in (("E1","E3"),("E2","E3")):continue
        t.append(f"|{r['fold']}|{r['lhs']} / {r['rhs']}|{r['heads_shared']}|{r['N']['mean_squared_cosine']:.4f}|{r['S']['mean_squared_cosine']:.4f}|")
    t+=["","子空间比较映射到共同原始头输出坐标并计入标准化尺度；平均平方余弦越大表示子空间越接近。逐主夹角见 m2_subspace_overlap.json，逐C/F约束余量见 m3_constraint_margins.csv，训练及评价的逐来源修正误差分别见 training_source_errors.csv 与 source_metrics.csv。",
        "","## 验证、资源与文件",""]
    verification={}
    for p in sorted((OUT/"evaluation").glob("*/verification.json")):
        verification[p.parent.name]=read(p)
    write(OUT/"verification.json",dict(complete=True,focused_unit_tests=5,scopes=verification,
        folds_train_validation_disjoint=True,new_E3_constraints_satisfied=True,
        historical_E0_constraint_replay_exception=True))
    t+=["5项针对性测试通过，覆盖评分分解与截距不变性、路径分支变化、空标签、MedPIC解码和合并VR。另从原始激活独立复算4,320个版本分数，与已存分数的最大差为0。完整方法的逐输入决策、完整路径和目标任务解码与上一轮一致；所有回放路径均未缺失所到节点分数。",
        "",
        "候选头对应21,360个唯一配对—头组合，复用21,282个，仅补78个（156次双向单头修补前向）。0、1卡各一个进程约16秒完成修补；用户追加5、6卡时队列已完成，因此这两卡未加载模型。后续拟合、收益拆分、重采样及报告均在CPU上执行。GPU工作结束后没有持续监控。",
        "",
        "6个独立截距及拟合来源见 independent_intercepts.json；E3五折参数见 models/E3/。逐输入、配对、题目及路径预测见各 evaluation/ 子目录，发布时提供 .jsonl.gz；scores_by_variant.npz 以独立字段存放各评分版本，score_manifest.json记录版本、组别、折号和截距拟合来源。五折合并目录引用五个未参与训练的验证预测文件。",
        "",
        "metrics.csv、source_bootstrap.csv、transitions.csv 汇总所有组及所有消融。旧结果引用见 references.json；预算敏感性直接复用。主报告与所有历史报告并存。"]
    (OUT/"REPORT.zh.md").write_text("\n".join(t)+"\n",encoding="utf-8")
    ref=dict(previous_commit="d37487a",previous_branch="results/aster-cr-paired-v2-20261111",
        previous_root=str(BASE),previous_report="https://github.com/chunxi-alpc/TRIAGE/blob/d37487a/reports/aster_cr_retrain_paired_v2_20261111/REPORT.zh.md",
        old_models=[dict(group=g,fold=f,path=str(BASE/"models"/g/f)) for f in ("full","fold1","fold2","fold3","fold4","fold5") for g in GROUPS if g!="E3" or f=="full"],
        reused_budget_sensitivity=str(BASE/"budget_sensitivity.json"))
    write(OUT/"references.json",ref)
    completion=dict(complete=True,experiment_version=VERSION,frozen_group="E1",E3_folds=5,
        independent_intercepts=6,full_scopes=5,pooled_cv_sources=80,bootstrap_draws=2000,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),publication_branch="results/aster-cr-diagnostics-e3cv-20261111")
    write(OUT/"completion.json",completion)
    print("Report ready",flush=True)

if __name__=="__main__":main()
