"""Generate the Chinese report from checkpoints, including incomplete status."""
from .common import *
from .statistics import summarize
from .baseline import summary as baseline_summary
from .cpu import combination

def fmt(v,percent=False):
    return "无法估计" if v is None else f"{100*v:.2f}%" if percent else f"{v:.4f}"
def cell(s,key="original_effective"):
    val=s["estimates"][key];ci=s.get("ci95",{}).get(key)
    text=fmt(val,True)
    if ci:text+=f" [{fmt(ci['low'],True)}, {fmt(ci['high'],True)}]"
    return text

def patch_summary():
    out=dict(effects=[],normalization=[],combination={},checks=[],residual=[],complements=[])
    group=defaultdict(list);normgroups=defaultdict(list);res=defaultdict(list);com=defaultdict(list)
    for model in MODELS:
        base=ROOT/"patch"/model/PRECISION
        for p in sorted((base/"pairs").glob("*.json")):
            d=read(p)
            if "checks" in d:out["checks"].append(dict(model=model,pair_id=d["pair_id"],**d["checks"]))
            for r in d.get("effects",{}).values():
                if not r["set"].startswith("single_"):
                    group[(model,r["split"],r["kind"],r["mode"],r["set"])].append(r)
            for name in ["type_selected","selected_union"]:
                native=d.get("effects",{}).get("native/"+name);fixed=d.get("effects",{}).get("fixed_rms/"+name)
                if native and fixed:
                    normgroups[(model,d["split"],d["kind"],name)].append(dict(native,
                        e=fixed["e"]-native["e"],ie_forward=fixed["ie_forward"]-native["ie_forward"],
                        ie_reverse=fixed["ie_reverse"]-native["ie_reverse"]))
        for kind in ["A","P"]:
            for split in ["fit","dev"]:
                for mode in ["native","fixed_rms"]:
                    singles={};combos=[]
                    for p in sorted((base/"pairs").glob("*.json")):
                        d=read(p)
                        if d["kind"]!=kind or d["split"]!=split:continue
                        for r in d.get("effects",{}).values():
                            if r["mode"]!=mode:continue
                            if r["set"].startswith("single_"):singles[r["pair_id"],int(r["set"][7:])]=r
                            elif r["set"]=="type_selected":combos.append(r)
                    if combos:out["combination"][f"{model}/{split}/{kind}/{mode}"]=combination(combos,singles,frozen()[model]["H_"+kind])
        pairs={p["pair_id"]:p for p in rows(ROOT/"data/diagnostic_pairs.jsonl")}
        for path in sorted((base/"residual").glob("*.json")):
            d=read(path)
            for r in d.get("effects",{}).values():
                key=(model,r["kind"],pairs[r["pair_id"]]["distance_group"],r["layer"],r["position"])
                res[key].append(r)
        for path in sorted((base/"complements").glob("*.json")):
            d=read(path)
            for r in d.get("effects",{}).values():com[(model,r["kind"],r["split"],r["layer"],r["part"])].append(r)
    for (model,split,kind,mode,name),rs in group.items():
        out["effects"].append(dict(model=model,split=split,kind=kind,mode=mode,set=name,**summarize(rs)))
    for (model,split,kind,name),rs in normgroups.items():
        out["normalization"].append(dict(model=model,split=split,kind=kind,set=name,**summarize(rs)))
    for (model,kind,dist,layer,pos),rs in res.items():
        out["residual"].append(dict(model=model,kind=kind,distance_group=dist,layer=layer,position=pos,**summarize(rs)))
    for (model,kind,split,layer,part),rs in com.items():
        out["complements"].append(dict(model=model,kind=kind,split=split,layer=layer,part=part,**summarize(rs)))
    return out

def _run():
    cpu=read(ROOT/"cpu/summary.json") if (ROOT/"cpu/summary.json").exists() else None
    patch=patch_summary();baseline=baseline_summary()
    write(ROOT/"results/patch_summary.json",patch);write(ROOT/"results/baseline_summary.json",baseline)
    lines=["# 冻结 M1：机制诊断与原文基础模型试测","",
      "本轮固定 Qwen3-8B 的 A/P/合并集合为 22/39/51 头，MedGemma 27B 为 125/49/141 头；保留 MedGemma 59/49/85 头历史结果。",
      "ρ=0.8、ε=0.1 保持不变。80% 仅作参照，不决定成功或失败。没有重新选头、运行 M1 v2、开展 M3 或纠错训练。",
      "旧构造配对用于回顾诊断；新增修补只使用拟合/开发数据。新增基础评估使用发布原文，超长单列，不截断改写。","",
      "## 当前状态",""]
    for stage in ["prepare","cpu","patch_qwen","patch_medgemma","baseline_qwen","baseline_medgemma","scheduler"]:
        p=ROOT/"status"/(stage+".json")
        d=read(p) if p.exists() else {}
        lines.append(f"- {stage}：{d.get('state','待运行')}"+(f"，{d['phase']} {d.get('done',0)}/{d.get('total',0)}" if "phase" in d else ""))
    lines+=["","## 资源与估时","",
      "GPU 0：Qwen3-8B，预计18–24 GiB；GPU 1：MedGemma 27B，预计55–60 GiB。约4.3万次末位置前向，另有936次残差后缀重算与每模型400例原文评估。初估3–6小时，不含排队。"]
    for model in MODELS:
        timing=ROOT/"timing"/f"{model}_pilot20.json"
        if timing.exists():
            t=read(timing)
            lines.append(f"- {model}首批20对：{t['elapsed_seconds']/60:.1f}分钟；头诊断总时长估计{t['estimated_head_stage_seconds']/3600:.2f}小时；峰值分配显存{t['peak_allocated_gib']:.2f} GiB。残差与原文部分另计。")
    complete=all((ROOT/"status"/(s+".json")).exists() and read(ROOT/"status"/(s+".json")).get("state")=="complete"
        for s in ["cpu","patch_qwen","patch_medgemma","baseline_qwen","baseline_medgemma"])
    lines+=["","完成状态："+("全部完成。" if complete else "尚未全部完成；以下只报告已经落盘的结果，待运行项不视为零效应或零错误。"),
      "","## 估计口径与不确定性","",
      "t=ell右−ell左，e=(IE正向−IE反向)/2。原口径为有效配对 mean(e/t)；效应加权为 Σsign(t)e/Σ|t|；过原点回归为 Σte/Σt²。后两项分别给出全部已缓存配对及有效配对结果。",
      "按来源整组重采样 2,000 次，固定种子 20261106；两方向、同源节点与配对一起重采样，报告百分位 95% 区间。小分母保留原始效应，零分母无法估计，恢复率不裁剪。",
      "区间只描述冻结集合在现有样本中的不确定性，不能消除既往反复选择造成的偏差；开发/测试差值不是置信区间。",
      "旧 Qwen 保留 BF16 输出读取精度；旧 MedGemma 与本轮新增前向为 BF16 骨干、FP32 标签投影。两种精度不合并。"]
    if cpu:
        lines+=["","## 冻结头数与层分布","",
          "|模型|集合|头数/总数|占比|局部/全局头|层数|","|---|---|---|---|---|---|"]
        for r in cpu["heads"]:lines.append(f"|{r['model']}|{r['set']}|{r['heads']}/{r['total']}|{fmt(r['fraction'],True)}|{r['local_heads']}/{r['global_heads']}|{len(r['layer_counts'])}|")
        lines+=["","## 旧缓存：效应分布与覆盖","",
          "|模型|划分/类型|配对/原配对|有效数|节点/来源|零分母|双错配对|绝对TE中位数/P95|","|---|---|---|---|---|---|---|"]
        for r in cpu["TE"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}|{r['pairs']}/{r['requested_pairs']}|{r['effective']}|{r['nodes']}/{r['sources']}|{r['zero_denominators']}|{r['both_wrong']}|{fmt(r['abs_TE'].get('median'))}/{fmt(r['abs_TE'].get('p95'))}|")
        lines+=["","## 旧缓存：恢复比例及95%区间","",
          "“全部”只包含确有修补缓存的配对；缺失数非零时不能代表原始全部配对。旧 Qwen 的小效应配对缺少修补结果，本轮不补算原测试集。",
          "","|模型/方案|划分/类型/集合|有效/缓存/原配对|原口径|加权全部|回归全部|加权有效|回归有效|","|---|---|---|---|---|---|---|---|"]
        for r in cpu["effects"]:
            lines.append(f"|{r['model']}/{r['scheme']}|{r['split']}/{r['kind']}/{r['control']}|{r['effective_pairs']}/{r['pairs']}/{r['universe_pairs']}|"+
                "|".join(cell(r,k) for k in ["original_effective","weighted_all","regression_all","weighted_effective","regression_effective"])+"|")
        lines+=["","## 两模型共同有效配对","",
          "|模型|划分/类型/集合|缓存/共同有效数|原口径|加权|回归|","|---|---|---|---|---|---|"]
        for r in cpu["paired_effects"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}/{r['control']}|{r['pairs']}/{r['common_effective_universe']}|{cell(r)}|{cell(r,'weighted_all')}|{cell(r,'regression_all')}|")
        lines+=["","## 排名稳定性与有效头数","",
          "入选频率指固定 k 的重采样排名前 k 名；精确排名限定在已有候选内。不是完整选头流程的稳定性检验。候选128取125时，即使随机集合也会高度重叠。",
          "有效头数定义为 (Σ|平均效应|)²/Σ平均效应²，使用梯度近似或精确单头效应，保留负效应的绝对贡献。",
          "","|模型/类型/排名|k/候选数|与原集合平均Jaccard|随机Jaccard|有效头数|","|---|---|---|---|---|"]
        for key,r in cpu["stability"].items():lines.append(f"|{key}|{r['k']}/{r['candidate_count']}|{fmt(r['jaccard']['mean'])}|{fmt(r['random_jaccard']['mean'])}|{fmt(r['effective_heads'])}|")
        lines+=["","逐头入选频率及集合重叠分布：cpu/summary.json。改动距离以各模型原生聊天模板下的词元为单位，明细：data/pair_distances.jsonl。","",
          "## A/F 分数指标","",
          "LSR 明确定义为共同节点的平均 |Δell_A| / 平均 |Δell_F|。F 仅作事实变化的分数对照，不用于同标签修补。",
          "","|模型|划分|共同节点|LSR|","|---|---|---|---|"]
        for r in cpu["LSR"]:lines.append(f"|{r['model']}|{r['split']}|{r['common_nodes']}|{fmt(r['value'])}|")
        lines+=["","|模型|划分/类型|分数变化超过原余量/两端数|实际预测翻转/配对数|","|---|---|---|---|"]
        for r in cpu["score_metrics"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}|{r['margin_exceeded']}/{r['margin_denominator']}|{r['prediction_flips']}/{r['flip_denominator']}|")
    lines+=["","## 新增前向诊断","",
      "固定 A/P 各32对：各24拟合、8开发；同一配对用于两个模型，未按恢复结果筛选。A 的跨1024词元样本来自拟合集；开发集原配对未提供该覆盖，不构造补充样本。",
      "候选为各模型梯度前128头，以及 Qwen前64/MedGemma前110头。拟合集按效应加权恢复率选出每类三个层，仅补算层内候选与候选外贡献，不更新冻结集合。",
      "","|模型|划分/类型/方式/集合|有效/全部配对|原口径|加权全部|回归全部|","|---|---|---|---|---|---|"]
    for r in patch["effects"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}/{r['mode']}/{r['set']}|{r['effective_pairs']}/{r['pairs']}|{cell(r)}|{cell(r,'weighted_all')}|{cell(r,'regression_all')}|")
    lines+=["","### 基础检查",""]
    for m in MODELS:
        rs=[r for r in patch["checks"] if r["model"]==m]
        if rs:lines.append(f"- {m}：{len(rs)}对；重复前向最大误差 {max(abs(x) for r in rs for x in r['replay_errors']):.6g}，自己修补自己最大误差 {max(abs(x) for r in rs for x in r['self_errors']):.6g}，全部头修补与donor最大分数差 {max(abs(x) for r in rs for x in r['all_head_donor_errors']):.6g}。")
    for m in MODELS:
        rs=[r for r in patch["checks"] if r["model"]==m and "full_forward_errors" in r]
        if rs:
            full_max=max(abs(x) for r in rs for x in r["full_forward_errors"])
            lines.append(f"- {m}：一次性完整前向与缓存末位置前向的最大分数绝对差为 {full_max:.6g}。缓存路径内检查一致，不代表与完整前向数值一致；解释小效应时须保留这一区别。")
    detail=ROOT/"QWEN_ANALYSIS.zh.md"
    if detail.exists():
        lines.append("- Qwen3详细分析：[QWEN_ANALYSIS.zh.md](QWEN_ANALYSIS.zh.md)。")
    lines+=["","### 组合效应与单头和","",
      "只在相同配对、相同方向且所有所需单头结果齐全时比较。比例保留零附近的原值，同时标出 |单头和|<0.1 的数量。",
      "","|模型/划分/类型/方式|配对|方向|组合均值|单头和均值|差值均值|比例中位数|小分母数|","|---|---|---|---|---|---|---|---|"]
    combos=dict(("old/"+k,v) for k,v in (cpu or {}).get("combination",{}).items());combos.update(patch["combination"])
    for key,v in combos.items():
        for d in ["forward","reverse"]:
            r=v[d];lines.append(f"|{key}|{v['pairs']}|{d}|{fmt(r['mean_joint'])}|{fmt(r['mean_sum_single'])}|{fmt(r['differences'].get('mean'))}|{fmt(r['ratios'].get('median'))}|{r['small_sums']}|")
    lines+=["","### 固定原始注意力输出RMS缩放","",
      "MedGemma固定每一层目标输入原本的注意力输出归一化分母；单头与组合修补均重跑。Qwen没有对应的注意力输出归一化，保持原分支；不修改其同名MLP前归一化。实现参照 [Gemma 3技术报告](https://arxiv.org/abs/2503.19786) 并核对本机模型代码。",
      "下表为同一配对的“固定缩放−原缩放”差值；不是两个独立区间相减。",
      "","|模型|划分/类型/集合|配对|原口径差值|加权差值|","|---|---|---|---|---|"]
    for r in patch["normalization"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}/{r['set']}|{r['pairs']}|{cell(r)}|{cell(r,'weighted_all')}|")
    lines+=["","### 三个层的候选内外贡献","",
      "|模型|划分/类型|层/部分|配对|原口径|加权全部|","|---|---|---|---|---|---|"]
    for r in patch["complements"]:lines.append(f"|{r['model']}|{r['split']}/{r['kind']}|{r['layer']}/{r['part']}|{r['pairs']}|{cell(r)}|{cell(r,'weighted_all')}|")
    lines+=["","### 残差粗扫描","",
      "固定4对窗口内A、4对跨窗口A、4对P。MedGemma层4/5/28/29/58/59；Qwen层2/3/16/17/33/34。位置为改动前共同文本末端、改动后共同文本起点和判定位置；修补层输出残差后重新计算整个后缀，不追加细扫。",
      "","|模型|类型/距离|层/位置|配对|原口径|加权全部|","|---|---|---|---|---|---|"]
    for r in patch["residual"]:lines.append(f"|{r['model']}|{r['kind']}/{r['distance_group']}|{r['layer']}/{r['position']}|{r['pairs']}|{cell(r)}|{cell(r,'weighted_all')}|")
    lines+=["","## 原文基础模型试测","",
      "MedGUIDE直接使用发布prompt、选项及答案，按实际2–8个选项计算FP32字母logit；MEGA-CDP保留原病例、树、条件和路径标签。原标注节点逐个判断，自主路径按模型判断遍历整棵树，偏离标注路径的节点没有自行生成标签。",
      "两套各200例为来源均衡探索样本，不代表全数据集成绩。官方7458例已用于其他评估；将来升级方法需另留未参与调参的确认来源。",
      "","|模型/数据|已完成/固定|可评分|正确|错误|独立来源|超长/其他无法评分|","|---|---|---|---|---|---|---|"]
    for r in baseline["tables"]:lines.append(f"|{r['model']}/{r['dataset']}|{r['finished']}/{r['frozen_cases']}|{r['scored']}|{r['correct']}|{r['wrong']}|{r['sources']}|{r['overlength']}/{r['unscorable']}|")
    lines+=["","|模型|MEGA原标注节点正确/错误/总数|原标注路径存在节点错误的病例数|","|---|---|---|"]
    for r in baseline["tables"]:
        if r["dataset"]=="mega":lines.append(f"|{r['model']}|{r['node_correct']}/{r['node_wrong']}/{r['node_total']}|{r['annotated_cases_wrong']}|")
    lines+=["","两模型错误重叠：",""]
    for k,r in baseline["overlap"].items():lines.append(f"- {k}：共同评分 {r.get('common_scored',r.get('common_nodes',0))}；双错 {r['both_wrong']}；仅Qwen错 {r['qwen_only_wrong']}；仅MedGemma错 {r['medgemma_only_wrong']}。")
    lines+=["","## 论文结论建议","",
      "1. 冻结集合的恢复率必须连同效应规模、有效配对覆盖与来源重采样区间解释。更换加权或回归口径后超过80%，不代表原指标达标。",
      "2. 当前区间仍较宽；冻结集合恢复率不能当作M3收益的普遍上界。"]
    outside=[r for r in patch["complements"] if r["part"]=="outside" and r["split"]=="dev" and
        r.get("ci95",{}).get("weighted_all",{}).get("low") is not None and r["ci95"]["weighted_all"]["low"]>0]
    norm=[r for r in patch["normalization"] if r["split"]=="dev" and
        r.get("ci95",{}).get("weighted_all",{}).get("low") is not None and
        (r["ci95"]["weighted_all"]["low"]>0 or r["ci95"]["weighted_all"]["high"]<0)]
    lines.append("3. "+("开发诊断存在候选外正向贡献，建议后续对两个模型统一研究逐层后逐头的M1 v2；这是探索证据，本轮不升级。" if outside else
        "候选外贡献暂未形成明确的开发集证据，保留冻结集合；不追加搜索。"))
    lines.append("4. "+("固定缩放在开发诊断中改变恢复效果；结合组合与单头和的差值，建议后续对两模型统一研究联合选择，不将原因归结为单一机制。" if norm else
        "归一化与组合不可加需结合双向差值共同判断；目前不据此更换方法。"))
    if not complete:lines.append("5. GPU诊断或原文试测尚未全部结束，上述机制建议为当前结果，最终完成时程序自动重写。")
    lines+=["","## 文件与续跑","",
      "- CPU明细：cpu/summary.json、cpu/legacy_effects.jsonl；所有原始小效应与零效应仍在逐对文件中。",
      "- 样本与冻结头：data/；修补按模型/精度/配对/方式/集合记录于patch/。",
      "- 原文逐例结果：baseline/；错误重叠与计数：results/baseline_summary.json。",
      "- 必要测试：logs/tests.log；首批20对估时：timing/；运行状态：status/。",
      "- 入口：python -m experiments.m1_diagnostics --stage cpu|patch|baseline|report|all；patch/baseline可指定 --model qwen|medgemma。",
      "- 启动脚本：run.sh。旧运行目录只读；不做哈希检查。"]
    final_analysis=ROOT/"FINAL_ANALYSIS.zh.md"
    if complete and final_analysis.exists():
        lines+=["", "[FINAL_ANALYSIS.zh.md](FINAL_ANALYSIS.zh.md)", ""]
    (ROOT/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status("report",state="complete" if complete else "partial",all_stages_complete=complete)
    return complete

def run():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        complete = _run()
    if complete and __import__("os").environ.get("M1_DIAGNOSTICS_NO_PUBLISH") != "1":
        from .publish import run as publish
        publish()
    return complete
