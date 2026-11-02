# TRIAGE · ASTER

ASTER 研究大语言模型能否根据患者事实逐步执行临床指南。主方法保留完整指南，同时让同一个冻结模型单独读取患者和当前条件。在每次选择分支之前，一个可学习的四维子空间把这份处置盲参考引入当前决策。

冻结版本为 Qwen3-8B 第 18 层、第 3 轮的 `no_js`。在 MEGA-CDP 官方单轮测试上，完整路径正确率为 **89.21%（6,653／7,458）**。

- [论文正文与结果](docs/ASTER/manuscript.zh.md)
- [方法补充与复现步骤](docs/ASTER/supplementary.zh.md)
- [主方法代码与冻结干预权重](experiments/aster_representation/)

从仓库根目录查看数据准备、训练、预测和评分命令：

```bash
python -m experiments.aster_representation --help
```

旧版规则键编辑研究见[历史文档](docs/ASTER/archive/README.md)。


---

## 版本是怎样变化的

| 阶段 | 相对上一版改变了什么 | 保留下来的结论／下一步 |
|---|---|---|
| TRIAGE 初始框架 | 将指南规则、患者条件、证据和注意力干预分成模块 | 建立可执行基线；旧主线见 [pre-ASTER 快照](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026) |
| ACRE（10-15） | 从固定增强规则，改为按适用性切换增强对象 | 完整规则 Oracle 切换未优于固定增强，停止这条具体方案 |
| CARE 模块一（10-18～20） | V4 修证据 → V5 修解析 → V6 仅融合证据 → V7 可靠性门控 → V8／V8.1 语义直判 → 冻结 A1 | A1 负责提供适用性研究信号；V6 保留旧接口，二者不是同一实验配置 |
| CARE 模块二（10-16～21，与模块一并行） | 4B 全层头分析 → B1–B4 子空间 → 有符号干预／偏置 → 因果与结构对照 → N1–N5、decide-or-ask | 发现可解码信息、改动注意力与改善最终答案是三个不同问题；未建立稳定的子空间独立优势 |
| ASTER 先导（10-21～25） | 从 A1 输出转向 MedGUIDE 盲判、裁剪、融合、动态注意力、CAD、SEKA 等可比较方案 | 200／800 例、阈值／顺序和匹配对照属于各自实验；多条复杂路线未超过简单基线 |
| ASTER 离线键方向（10-27～31） | 先在 0.6B 验证差分方向和读取，再在 Qwen3-8B 扫描 KV 头、学习方向、编辑缓存键 | 找到了 8B 的四头配置，但改变局部规则注意力尚未稳定改善最终决策；见下方原始产物 |
| ASTER 表示干预（10-31～11-02，当前主线） | 从改指南 KV 键转为在每个分支的注意力输出处，融合完整指南表示与处置盲参考；学习四维子空间与门控 | 选定零起始层 18、训练第 3 轮的 no_js；完整路径 6,653／7,458。这里选的是输出干预层，不再沿用旧四个 KV 头 |

当前表示干预的完整研究快照见 [research/aster-single-turn-20261102](https://github.com/chunxi-alpc/TRIAGE/tree/d8cf67d53d6b068d9b382114de4a5fa4b22f633b)，精简后的复现入口是 [experiments/aster_representation](experiments/aster_representation/)。旧键编辑实验的 457 例中期结果与当前表示干预的 7,458 例完整测试是两套方法，分开引用。

## Qwen3-8B 扫描选头：代码和结果已经在主线

**扫描工作确实做过，入口是 PR #42。** [采集代码](experiments/aster_manuscript_20261030/collect.py)保存各层 KV 表示，[临床分析代码](experiments/aster_manuscript_clinical_20261030/analyze.py)比较秩 1／2／4／8，再由 [select_expert](src/aster/manuscript_core.py)按验证集的方向区分度和正负读取响应选头。有效候选为 **280 个 KV 头**；第 0 层的 8 个头因差分数值秩不足排除，原因记在 rank_selection.json。**rank=8 指子空间维数，最终选中的是 4 个 KV 头。**

所有层号和头号均从 0 开始；“层:头”中的头号是 KV 头号，每个 KV 头对应一组共享它的查询头。

| 冻结批次 | 选中的层:KV头 | 原始记录 |
|---|---|---|
| run03，rank 8 | 33:3、34:0、34:1、31:1 | [秩比较](experiments/aster_manuscript_20261030/public/run03/rank_selection.json) · [选中头](experiments/aster_manuscript_20261030/public/run03/selected_structure.json) · [冻结配置](experiments/aster_manuscript_20261030/public/run03/frozen.json) |
| run05，rank 8 | 21:4、21:6、24:3、29:7 | [秩比较](experiments/aster_manuscript_20261030/public/run05/rank_selection.json) · [选中头](experiments/aster_manuscript_20261030/public/run05/selected_structure.json) · [冻结配置](experiments/aster_manuscript_20261030/public/run05/frozen.json) |
| clinical01，rank 8 | **33:3、31:1、33:4、23:4** | [秩比较](experiments/aster_manuscript_clinical_20261030/public/clinical01/rank_selection.json) · [选中头与留出诊断](experiments/aster_manuscript_clinical_20261030/public/clinical01/structure.json) · [冻结配置](experiments/aster_manuscript_clinical_20261030/public/clinical01/frozen.json) |

三个批次的数据和冻结配置不同，不能混用头组。clinical01 的 43 例确认中，谱平移终点正确 20 例、原模型 19 例、文本裁剪 25 例；[457 例官方快照](experiments/aster_manuscript_clinical_20261030/public/interim_457_20261030/)仍是中期结果。[10 月 31 日选中头／对照头实验](docs/ASTER/archive/head-controls-20261031.zh.md)进一步发现：关闭选中头并未降低这 43 例的终点成绩；因此这组头有表示与读取证据，尚不能称为最有助于正确决策的头。

**选层选头的具体方法见 [说明](docs/ASTER/layer-head-selection.zh.md)。** 复现当前主方法直接用层 18；继续研究 KV 头时，从 clinical01 四头出发，与同层未选头做相同干预比较，依据开发集完整路径的纠正与退化作选择。

## 每个研究版本做了什么

下表逐一列出截至本次整理前的全部 46 个 PR；缺号 #26 不补造版本。日期为北京时间的 PR 创建日期，执行结果以相应冻结产物为准。开放或草稿只表示 GitHub 状态，不等于没有做过实验。

| 日期／PR | 工作与变化 | 合并状态 |
|---|---|---|
| 10-15 · [#1](https://github.com/chunxi-alpc/TRIAGE/pull/1) | ACRE：Qwen3-8B 的固定增强、适用性切换与同源 Oracle 对照；完整规则切换未优于固定增强。 | 开放，草稿 |
| 10-17 · [#2](https://github.com/chunxi-alpc/TRIAGE/pull/2) | CARE M2：重分析 4B 的 288 个层／KV 头，区分相关性选头与适用性选头及数值混杂。 | 本次已合并 |
| 10-17 · [#3](https://github.com/chunxi-alpc/TRIAGE/pull/3) | CARE M2：B1–B4 子空间、随机方向、偏置和因果干预；未见可靠的学习子空间优势。 | 本次已合并 |
| 10-18 · [#4](https://github.com/chunxi-alpc/TRIAGE/pull/4) | CARE M1 V4：修复证据定位，增加新规则完整病例验证。 | 开放 |
| 10-18 · [#5](https://github.com/chunxi-alpc/TRIAGE/pull/5) | CARE M1 V5：TrialMatchAI 解析修复，复评缓存输出。 | 开放 |
| 10-18 · [#6](https://github.com/chunxi-alpc/TRIAGE/pull/6) | CARE M1 V5 后续：失败条件重生成，解析器升级到 V3。 | 开放 |
| 10-18 · [#7](https://github.com/chunxi-alpc/TRIAGE/pull/7) | CARE M1 V6：仅融合证据，接入 pipeline；与后来的 A1 默认接口分别保留。 | 开放 |
| 10-19 · [#8](https://github.com/chunxi-alpc/TRIAGE/pull/8) | CARE M2：按金标适用性施加有符号 B4，比较均值平移与直接分数偏置。 | 本次已合并 |
| 10-19 · [#9](https://github.com/chunxi-alpc/TRIAGE/pull/9) | CARE V7：可靠性门控、固定对照与真实模型实验。 | 开放，草稿 |
| 10-19 · [#10](https://github.com/chunxi-alpc/TRIAGE/pull/10) | CARE 8B：ConflictMedQA／DriftMed 上下文冲突方向；独立于后来的临床规则选头。 | 开放 |
| 10-19 · [#11](https://github.com/chunxi-alpc/TRIAGE/pull/11) | CARE 因果链：4B 层段×强度与真实键编辑；注意力独立收益有限，结构修复更明显。 | 开放，草稿 |
| 10-19 · [#12](https://github.com/chunxi-alpc/TRIAGE/pull/12) | CARE M1 V8：直接做语义适用性判断，形成 A1 研究信号。 | 开放，草稿 |
| 10-20 · [#13](https://github.com/chunxi-alpc/TRIAGE/pull/13) | CARE M1 V8.1：完整条件与两轮复核；比较后保留 A1。 | 开放，草稿 |
| 10-20 · [#14](https://github.com/chunxi-alpc/TRIAGE/pull/14) | CARE A1：增加未知态 U 对照示例，准确率和 U 识别之间存在取舍。 | 开放，草稿 |
| 10-20 · [#15](https://github.com/chunxi-alpc/TRIAGE/pull/15) | CARE A1：冻结语义信号接口，供模块二研究使用；集成由 PR #16 完成。 | 开放，草稿 |
| 10-20 · [#16](https://github.com/chunxi-alpc/TRIAGE/pull/16) | 将冻结 A1 接入主线；同时保留 V6 的既有调用接口。 | 已合并 |
| 10-20 · [#17](https://github.com/chunxi-alpc/TRIAGE/pull/17) | CARE M2 增量 N1–N5：相关性／适用性、上下文／参数知识、可决断／须询问；原计划后续有实现，按分支最新产物追溯。 | 开放，草稿 |
| 10-21 · [#18](https://github.com/chunxi-alpc/TRIAGE/pull/18) | CARE decide-or-ask：决策／询问、不确定性传播与校准框架。 | 开放，草稿 |
| 10-21 · [#19](https://github.com/chunxi-alpc/TRIAGE/pull/19) | MedGUIDE：冻结 A1 与同输入 Qwen 全量对照的旧执行版本；PR 已关闭。 | 已关闭，未合并 |
| 10-21 · [#20](https://github.com/chunxi-alpc/TRIAGE/pull/20) | TrialGPT：冻结 A1 全量评估，分析 124 条 U 分歧。 | 开放，草稿 |
| 10-22 · [#21](https://github.com/chunxi-alpc/TRIAGE/pull/21) | ASTER MedGUIDE：修复先导实现，记录 Llama／Qwen 的真实试跑。 | 开放 |
| 10-22 · [#22](https://github.com/chunxi-alpc/TRIAGE/pull/22) | ASTER MedGUIDE：A/B/C 反事实预实验，进入主线。 | 已合并 |
| 10-23 · [#23](https://github.com/chunxi-alpc/TRIAGE/pull/23) | ASTER Qwen 200 例：A/B/C、CAD 与分支机制实验。 | 开放，草稿 |
| 10-23 · [#24](https://github.com/chunxi-alpc/TRIAGE/pull/24) | MedGUIDE 扩到 800 例；重算患者条件相关的阈值参考标签。 | 开放，草稿 |
| 10-23 · [#25](https://github.com/chunxi-alpc/TRIAGE/pull/25) | Qwen 800 例：比较 A/B/C/D/CS 方法，分析保留规则和 CS 错误。 | 开放，草稿 |
| 10-24 · [#27](https://github.com/chunxi-alpc/TRIAGE/pull/27) | 对照 Qwen 200／800 例，保存固定 CAD18 的完整词元轨迹。 | 开放，草稿 |
| 10-24 · [#28](https://github.com/chunxi-alpc/TRIAGE/pull/28) | 44 例阈值×规则顺序对照，补充 109 次请求。 | 开放，草稿 |
| 10-24 · [#29](https://github.com/chunxi-alpc/TRIAGE/pull/29) | 44 例先显式核对患者条件、再裁剪规则的对照。 | 开放，草稿 |
| 10-24 · [#30](https://github.com/chunxi-alpc/TRIAGE/pull/30) | ASTER E0–E8：修正金标前缀等混杂，独立控制盲判与门控，保存续跑结果。 | 开放，草稿 |
| 10-25 · [#31](https://github.com/chunxi-alpc/TRIAGE/pull/31) | ASTER P0–P3：独立测试的准确率／伤害取舍，未超过简单基线准确率。 | 开放，草稿 |
| 10-25 · [#32](https://github.com/chunxi-alpc/TRIAGE/pull/32) | ASTER：动作注意力质量守恒；Qwen 实测未超过简单融合。 | 开放，草稿 |
| 10-25 · [#33](https://github.com/chunxi-alpc/TRIAGE/pull/33) | ASTER：两模型动态融合开发实验及匹配负对照。 | 开放，草稿 |
| 10-25 · [#34](https://github.com/chunxi-alpc/TRIAGE/pull/34) | ASTER Qwen seed=0：注意力残差门控的离线对照与 GPU 协议。 | 开放，草稿 |
| 10-25 · [#35](https://github.com/chunxi-alpc/TRIAGE/pull/35) | ASTER Qwen seed=0：完成稀疏动态注意力 GPU 实验和全量对照。 | 开放 |
| 10-25 · [#36](https://github.com/chunxi-alpc/TRIAGE/pull/36) | ASTER／SEKA：官方基线复现、外部评测和后续执行计划。 | 开放，草稿 |
| 10-25 · [#37](https://github.com/chunxi-alpc/TRIAGE/pull/37) | ASTER／SEKA：执行上述研究，保存来源记录、配对结果与提取失败。 | 开放，草稿 |
| 10-27 · [#38](https://github.com/chunxi-alpc/TRIAGE/pull/38) | 离线适用性子空间：0.6B 机制预实验，局部读取改变未带来答案改善。 | 开放，草稿 |
| 10-28 · [#39](https://github.com/chunxi-alpc/TRIAGE/pull/39) | 统一 ASTER 实验章节、三类基线、评价公式和结果表。 | 已合并 |
| 10-28 · [#40](https://github.com/chunxi-alpc/TRIAGE/pull/40) | 0.6B 离线共享方向、双向读取与强编辑读出诊断；部分翻转不足以证明稳定答案控制。 | 开放，草稿 |
| 10-30 · [#41](https://github.com/chunxi-alpc/TRIAGE/pull/41) | 8B 全量旧执行：按用户要求提前停止；确认记录与失败保留，官方测试和 MedGUIDE 未开始。 | 开放，草稿 |
| 10-30 · [#42](https://github.com/chunxi-alpc/TRIAGE/pull/42) | 8B 键编辑：280 候选 KV 头、秩选择、四头冻结、43 例确认与 457 例官方中期快照。本次把原代码和公开产物收入主线。 | 开放，草稿 |
| 10-31 · [#43](https://github.com/chunxi-alpc/TRIAGE/pull/43) | 旧键编辑方案的方法修订：适用性评估、缓存注意力门控与 B0–B5；作为独立设计稿保留。 | 开放，草稿 |
| 11-03 · [#44](https://github.com/chunxi-alpc/TRIAGE/pull/44) | 为首页加入路线历史和 8B 选头索引。 | 已合并 |
| 11-03 · [#45](https://github.com/chunxi-alpc/TRIAGE/pull/45) | 更新历史分支索引。 | 已合并 |
| 11-03 · [#46](https://github.com/chunxi-alpc/TRIAGE/pull/46) | 补齐当前分支清单。 | 已合并 |
| 11-03 · [#47](https://github.com/chunxi-alpc/TRIAGE/pull/47) | 澄清 8B 扫描与 457 例官方中期结果的状态。 | 已合并 |

没有单独 PR 的支线，包括 P1–P6、适用性决策研究、模块说明和单轮表示研究，见 [完整分支索引](docs/BRANCHES.md)。

## 本次整合与后续入口

- 合并 PR #2、#3、#8：它们的实验改动已与主线一致，补齐实际合并关系。
- 将 PR #42 中主线缺少的 123 个代码、配置、公开汇总和报告文件原样纳入，保留原目录；该 PR 的整项研究状态仍独立保留。旧报告中的“当前／运行中”指报告当日，不代表现在的执行状态。
- 当前主方法继续由[论文](docs/ASTER/manuscript.zh.md)、[补充](docs/ASTER/supplementary.zh.md)和[表示干预代码](experiments/aster_representation/)说明；旧分支的主稿与计划不替换当前方法。
- 其余分支按版本索引保留各自结果和代码。PR #41 的提前停止、PR #42 的中期分母以及各版负结果均保留。

本次整理没有启动模型实验；服务器上原键编辑实现的 37 项 CPU 单元测试通过。
