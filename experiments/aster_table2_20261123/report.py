"""Write Table 2, its node-type supplement, and the numerical report."""
from .common import REPO,OUT,read

SPLITS={"dev":"开发","test":"留出"}
def number(x,scale=1,digits=3):
    if x is None:return "不适用"
    return f"{x*scale:.{digits}f}".replace("-","−")
def interval(value,bounds,scale=1,digits=3):
    point=number(value,scale,digits)
    if value is None or bounds is None:return point
    return point+" ["+number(bounds[0],scale,digits)+", "+number(bounds[1],scale,digits)+"]"
def make_table(rows,stratified=False):
    r2="$"+chr(96)+"R^2_{\\mathrm{sep}}"+chr(96)+"$"
    cov="$"+chr(96)+"R_{\\mathrm{cov}}"+chr(96)+"$"
    header="| 评价集合 | "+("节点类型 | " if stratified else "")+"信息与配对 | 保持对 / 翻转对 | "+r2+"（%）↑ | "+cov+" ↓ |"
    align="|---|"+("---|" if stratified else "")+"---|---:|---:|---:|"
    lines=[header,align]
    for r in rows:
        parts=[SPLITS[r["split"]]]
        if stratified:parts.append(r["node_type"])
        parts.extend([r["information"]+"："+r["preserving_kind"]+" / "+r["flipping_kind"],
          f'{r["preserving_pairs"]} / {r["flipping_pairs"]}',
          interval(r["r2_sep"],r["r2_ci95"],100,2),interval(r["r_cov"],r["r_cov_ci95"],1,3)])
        lines.append("| "+" | ".join(parts)+" |")
    return "\n".join(lines)

def render(config,selection,frozen,rows,grid):
    main=[r for r in rows if r["node_type"]=="全部"]
    stratified=[r for r in rows if r["node_type"]!="全部"]
    m2=read(OUT/"m2.json")
    h=frozen["head_count"];rank=frozen["feature_rank"];lam=frozen["regularization"]
    lookup={(r["split"],r["information"]):r for r in main}
    test=lookup["test","总体"];ta=lookup["test","指南"];tp=lookup["test","患者"]
    counts=selection["selected_counts_by_kind"]
    unknown={r["split"]:r["preserving_pairs"] for r in stratified if r["node_type"]=="未标注" and r["information"]=="总体"}
    protocol=(f"Qwen3-8B使用2,540对训练配对和80个训练来源，按第3节的自适应M1与层级权重拟合；"
       f"M1的A/P/C/F选头数分别为{counts['A']}/{counts['P']}/{counts['C']}/{counts['F']}，并集为{h}个头，证据正交读出维数为{rank}。"
       f"在20个开发来源上选择正则系数λ={lam:g}，固定训练特征及岭读出后评价40个留出来源。")
    caption=("**表2：干扰可预测性与证据覆盖。** "+protocol+
       "方括号为2,000次来源自助重采样的95%区间，重采样时固定已拟合参数。"
       "开发结果参与本表无约束诊断的正则系数选择；第3.4节受约束M3的预算与正则联合选参按其独立开发协议执行。总体A/P与C/F各按类别等权，再按来源、病例、变体均衡；"
       "各行在对应子集内重新归一化既定权重。"
       "R²以百分比报告，R_cov的单位为标准化后的特征坐标。"
       "P类开发和留出分别为4对和10对，样本数随各行列出。"
       "节点类型分层结果见[附录表C15](manuscript.appendix.zh.md#appendix-c-16)，"
       "配置、逐对数值和复算代码见[表2实验报告](../../reports/aster_table2_20261123/REPORT.zh.md)。")
    if rank==0:
        analysis=(f"本配置选出{h}个头，最终读出维数为0，附加分数恒为0。"
          "因此表2中的R²与证据覆盖残差均为0；这两个零值由零维读出产生。"
          "选头及子空间记录给出了该结果的具体来源。")
    else:
        analysis=(f"A类留出配对的R²为{number(ta['r2_sep'],100,2)}%"
          f"（95%区间：{number(ta['r2_ci95'][0],100,2)}%至{number(ta['r2_ci95'][1],100,2)}%），"
          "表明证据正交特征保留了对处置改写引起的分数变化的预测信息。"
          f"A/P总体R²为{number(test['r2_sep'],100,2)}%"
          f"（{number(test['r2_ci95'][0],100,2)}%至{number(test['r2_ci95'][1],100,2)}%）；"
          f"P类为{number(tp['r2_sep'],100,2)}%"
          f"（{number(tp['r2_ci95'][0],100,2)}%至{number(tp['r2_ci95'][1],100,2)}%），"
          f"其{tp['preserving_pairs']}对留出样本的来源区间较宽。"
          f"C/F证据覆盖残差为{number(test['r_cov'])}，C类和F类分别为"
          f"{number(ta['r_cov'])}和{number(tp['r_cov'])}。"
          "这些残差刻画证据变化在读出空间中的剩余幅度，M3的逐对预算用于控制相应分数修正。")
    table=make_table(main)
    main_md=table+"\n\n"+caption+"\n\n"+analysis+"\n"
    (OUT/"table2.md").write_text(main_md,encoding="utf-8")
    supplement=('<a id="appendix-c-16"></a>\n\n### C.16 表2：按节点类型分层的干扰预测与证据覆盖\n\n'
      +make_table(stratified,True)+"\n\n"
      "**表C15：表2按节点类型分层的结果。** "
      "节点类型由既有规则中的条件运算符确定：包含and、or或not组合的为复合条件，"
      "单一数值比较的为数值条件，其余为类别或状态条件。"
      "每个分层均使用表2同一组冻结特征和读出，并在相应子集中归一化原评价权重。"
      f"开发{unknown.get('dev',0):,}对、留出{unknown.get('test',0):,}对A/P缺少结构化条件规则，归入“未标注”并保留在总体统计。无对应配对的指标记为不适用。R²以百分比报告，R_cov使用标准化特征坐标；"
      "方括号为固定参数的来源自助重采样95%区间。\n\n"
      f"正则系数从{', '.join(f'{x:g}' for x in config['ridge']['candidates'])}中选择，"
      f"选择准则为开发集A/P加权平方残差最小，得到λ={lam:g}。"
      "该读出移除M3逐对证据预算，直接对应第4.1.2节定义的无约束岭诊断；"
      "概率校准位于分数指标计算之后，未参与表2指标。"
      "逐对数值与完整开发集搜索结果见[表2报告](../../reports/aster_table2_20261123/REPORT.zh.md)。\n\n")
    (OUT/"table2_by_node_type.md").write_text(supplement,encoding="utf-8")
    manuscript=REPO/"docs/ASTER/manuscript.zh.md"
    text=manuscript.read_text(encoding="utf-8")
    section=max(text.find("#### 4.1.2"),0)
    start=text.index("\n| ",section)+1
    end=text.index("![图2：",start)
    text=text[:start]+main_md+"\n"+text[end:]
    text=text.replace("#### 4.1.2 双重分离、干扰抑制与证据保持","#### 4.1.2 干扰可预测性与证据保持",1)
    text=text.replace("该三模块配置的对应实验结果待重新拟合与评价。",
      "表2补充该版本的自适应选头、层级加权子空间与无约束岭读出诊断；完整受约束M3和路径任务评价待补。",1)
    manuscript.write_text(text,encoding="utf-8")
    appendix=REPO/"docs/ASTER/manuscript.appendix.zh.md"
    text=appendix.read_text(encoding="utf-8")
    marker='<a id="appendix-d"></a>'
    if marker not in text:marker="## 附录 D"
    existing=text.find('<a id="appendix-c-16"></a>')
    if existing>=0:
        end=text.index(marker,existing)
        text=text[:existing]+supplement+text[end:]
    else:
        text=text.replace(marker,supplement+marker,1)
    text=text.replace("该组合的拟合与评价结果待补，现有数值继续按历史配置报告。",
      "表2报告修订配置的M1/M2与无约束岭读出诊断；完整受约束M3和任务效用评价待补，E1—E8既有结果继续按历史配置报告。",1)
    appendix.write_text(text,encoding="utf-8")
    prep=read(OUT/"preparation.json")
    resource=[]
    for i in range(4):
        worker=read(OUT/"workers"/str(i)/"completion.json")
        if worker:
            resource.append(f"- GPU {worker['gpu']}：新增{worker.get('new_scores',0):,}次修补；"
              f"耗时{worker.get('seconds',0)/60:.1f}分钟；峰值显存{worker.get('peak_gib',0):.2f} GiB。")
    grid_table=["| λ | 开发R²（%） | 开发加权平方残差 | 读出系数范数 |","|---:|---:|---:|---:|"]
    for r in grid:
        grid_table.append(f"| {r['regularization']:g} | {number(r['dev_r2_sep'],100,3)} | {number(r['dev_residual_mse'],1,6)} | {number(r['coefficient_norm'],1,6)} |")
    report=("# 表2：自适应选头与层级权重的诊断结果\n\n"+analysis+"\n\n"+protocol+"\n\n"
      "## 主表\n\n"+table+"\n\n"
      "R²使用相对零修正的未中心化平方残差降低比例，负值完整保留。"
      "证据覆盖残差按Nᵀ(I−SSᵀ)Δz计算。两项指标使用冻结训练参数；"
      "开发集参与λ选择，留出集在选定λ后计算。区间按来源重采样2,000次，固定已拟合参数。\n\n"
      "## 拟合配置\n\n"
      "- 训练数据：E1 / Dataset A，A/P/C/F各635对，80个来源。\n"
      "- 开发数据：20个来源，A/P/C/F分别1,194/4/29/31对。\n"
      "- 留出数据：40个来源，A/P/C/F分别2,283/10/33/7对。\n"
      "- 配对权重：类别、来源、病例、变体逐层均衡；训练端点获得每个配对一半权重，并合并重复端点。\n"
      "- M1：来源五折，梯度稳健阈值2.0；每个来源、类别和标签转换模式最多抽取2对；匹配对照交换右端供体。\n"
      "- M1选择：类别内BH q<0.05、来源95%区间排除0、跨折候选频率和来源自助符号稳定率均≥0.80、有效来源≥5。\n"
      f"- 各类选头：A={counts['A']}、P={counts['P']}、C={counts['C']}、F={counts['F']}；并集{h}个，共享头{len(selection['shared_heads'])}个。\n"
      f"- M2：S/U/N秩分别{m2['S_rank']}/{m2['U_rank']}/{m2['N_rank']}，维数上限16；标准差下限10⁻⁵；奇异值阈值max(10⁻⁷,10⁻⁶s_max)。\n"
      f"- 无约束岭读出：A/P权重和为1，λ={lam:g}，按开发集加权平方残差选择；数值并列时选择较大λ。\n"
      "- 表2范围：M1、M2及无约束岭诊断。主稿历史表3及后续任务结果保持其原配置。\n\n"
      "## 正则系数选择\n\n"+"\n".join(grid_table)+"\n\n"
      "## 节点类型分层\n\n"+make_table(stratified,True)+"\n\n"
      f"类型由原始rules.jsonl的predicate结构确定。开发{unknown.get('dev',0):,}对、留出{unknown.get('test',0):,}对A/P缺少结构化规则，归入“未标注”并保留在总体统计。分层沿用全评价总体的既定权重并在子集内归一化。"
      "某次来源重采样未包含该子集时，该次区间统计记为无有效样本；有效次数见table2.json。\n\n"
      "## 服务器运行\n\n"
      f"共需{prep['requested_directed_scores']:,}次单向头修补分数；复用{prep['reused_directed_scores']:,}次，补算{prep['missing_directed_scores']:,}次。"
      "原始激活来自相同Qwen3-8B、输入编号、提示和计分流程的既有缓存。"
      "后缀重放复用干预层之前的计算，干预层及之后保持原运算；四卡启动时均与完整重放得到相同分数。\n\n"
      +"\n".join(resource)+"\n\n"
      "## 文件与复算\n\n"
      "- [配置](configuration.json)、[选头结果](m1/selection.json)、[逐头统计](m1/head_statistics.csv)、[来源效应](m1/source_effects.csv)。\n"
      "- [开发λ搜索](regularization_selection.csv)、[冻结参数说明](frozen.json)、[M2参数](m2.npz)、[岭系数](ridge.npz)。\n"
      "- [主表](table2.md)、[分层表](table2_by_node_type.md)、[数值表](table2_all_rows.csv)、[完整统计](table2.json)。\n"
      "- [开发逐对数值](dev_pair_metrics.csv)、[留出逐对数值](test_pair_metrics.csv)。\n"
      "- [代码](../../experiments/aster_table2_20261123/)。原始修补数据保存在服务器/home/gcx/aster-table2-runtime-20261123/。\n\n"
      "~~~bash\nsource /home/gcx/gcx261014/bin/activate\ncd /home/gcx/aster-table2-20261123\n"
      "python -m experiments.aster_table2_20261123.prepare\n"
      "python -m experiments.aster_table2_20261123.controller\n~~~\n")
    (OUT/"REPORT.zh.md").write_text(report,encoding="utf-8")
