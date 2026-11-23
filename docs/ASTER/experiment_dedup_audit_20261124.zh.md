# ASTER-CR 下一阶段实验去重审计（2026-11-24）

> 本审计检索了 main、Issue #70/#72/#75/#79/#86、相关已完成报告、Draft PR #29/#31/#34/#35/#71/#73/#76/#78/#82/#85/#87及各自分支。这里只核对 GitHub 证据，**没有独立重跑服务器任务或读取全部私有缓存**。将“已报告完成、有可核查工件”“不同协议、可复用”“只有计划”“尚无同协议实测”严格分开。

## 一、已完成的相同或足够接近实验：不得重新训练、推理或完整回放

| 内容 | 证据 | 后续操作 |
|---|---|---|
| E1 五折、历史有无截距、配对和完整路径、四向转移 | [E1归档](../../reports/aster_cr_diagnostics_e3cv_20261111/REPORT.zh.md) | 读取存量预测与 UID；不重新做旧五折 |
| S/J 联合监督、原 R1 的166个首次错误节点、受 C/F 约束的线性规划可达性（150/166 原预算下不可达）、首错逐例和来源分析 | [S/J报告](https://github.com/chunxi-alpc/TRIAGE/blob/experiment/aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115/REPORT.zh.md)；同目录 evaluation/natural_first_error.jsonl.gz 和 diagnostics/correction_bounds.jsonl | **不再重复166个LP、相同截距/预算的首次错误边界或 S/J 拟合**。新任务只做现有记录未覆盖的语义/来源归因 |
| J2 方向监督与非对称 C/F 约束组合 | [J2报告](https://github.com/chunxi-alpc/TRIAGE/blob/experiment/aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115/J2_REPORT.zh.md) | 仅归档复用 |
| J 零截距真实动态路径、J2四格消融、逐例首错边界、来源bootstrap、精确路径复现 | [Issue #86](https://github.com/chunxi-alpc/TRIAGE/issues/86)、[Draft PR #87](https://github.com/chunxi-alpc/TRIAGE/pull/87)、[完成报告](https://github.com/chunxi-alpc/TRIAGE/blob/cb71485e7530c82042c4cf9112eb4d4308c9a977/reports/aster_m3_missing_controls_20261123/REPORT.zh.md) | **D0只检查现存清单/哈希、分母与状态**，不重新回放27392条归档路径、不重复25次新增拟合。PR是未合并工件，读对应分支 |
| 16/32维历史特征/修正容量与预算曲线 | [正文附录C.2/C.7](manuscript.appendix.zh.md#appendix-c-2) | 不再启动通用“16→32维能否改进”的同义实验 |
| 完整证据正交补6528维 + 硬预算M3；五折、40来源与逐例记录 | [Draft PR #71](https://github.com/chunxi-alpc/TRIAGE/pull/71)、[报告](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-full-complement-70/reports/aster_full_complement_70/REPORT.zh.md) | 不重训全维/硬约束；有新增分析指标时从既有工件计算 |
| 原16维 + 软双目标 M3 的70点网格/五折与复用留出 | [Draft PR #76](https://github.com/chunxi-alpc/TRIAGE/pull/76) | 不重新做70点搜索 |
| 同70点 Pareto前沿与拐点选择 | [Draft PR #78](https://github.com/chunxi-alpc/TRIAGE/pull/78) | 不重复选择或绘图 |
| M1头四象限和稳定性试验 | [Draft PR #73](https://github.com/chunxi-alpc/TRIAGE/pull/73)、[Draft PR #82](https://github.com/chunxi-alpc/TRIAGE/pull/82) | 不重新计算梯度、选头和激活修补 |
| 自适应M1/M2的表2和规则类型分层 | [PR #85](https://github.com/chunxi-alpc/TRIAGE/pull/85)、[报告](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-table2-20261123/reports/aster_table2_20261123/REPORT.zh.md) | 直接复用结构化规则的分类；开发/留出 A/P 仍有787/1717对缺结构标签，不能称已完成全量结构难度标注 |

## 二、方法近似但不是同一任务：复用代码和教训，不重做旧队列

| 已有研究 | 证据及范围 | 本次区别 |
|---|---|---|
| 注意力残差/分数边距门控 | [PR #34](https://github.com/chunxi-alpc/TRIAGE/pull/34)、[PR #35](https://github.com/chunxi-alpc/TRIAGE/pull/35)：Qwen800开发输入和动作分数融合 | 不是 E1/J 的二元节点候选改判监督门控；复用代码和门控阴性对照，不重跑旧800例 |
| 旧 ASTER 的低伤害回退方案与独立测试 | [PR #31](https://github.com/chunxi-alpc/TRIAGE/pull/31)：原任务各模型在独立树上评价纠正/伤害/覆盖率 | 任务、修正器与数据不同；复用指标，不冒充当前 MEGA-E1/J 保护实测 |
| 显式患者事实与条件核对后路由 | [PR #29](https://github.com/chunxi-alpc/TRIAGE/pull/29)：44例 × 阈值/顺序，共176格，已记录错误与纠正 | 已实际验证相近的核对思路。后续只在 MEGA 的尚未覆盖错误类别上新增；不重跑44例 |
| CARE 完整规则/原子判断、证据定位、两轮复核 | [PR #4](https://github.com/chunxi-alpc/TRIAGE/pull/4)、[PR #5](https://github.com/chunxi-alpc/TRIAGE/pull/5)、[PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12)、[PR #13](https://github.com/chunxi-alpc/TRIAGE/pull/13) | 数据/模型/标签语义不同；借鉴解析器、拒判、回退、错误分类，不再重新做A0/A1/B1/B2 |
| 既有临床标准的错误归因 | [PR #20](https://github.com/chunxi-alpc/TRIAGE/pull/20)：TrialGPT的124条U分歧有事后文本分类 | 与 MEGA 首错不等价；可复用“缺失当否定/部分当全部/时间对象混淆”等类别，不能挪用比例 |
| 旧闭式 ASTER 的路径错误分析 | [历史诊断](../../reports/aster_existing_data_20261106/ANALYSIS.zh.md)：432条路径、低分数翻转案例 | 旧模型和旧总体；不重新跑其432路径，能用的案例仅作风险参考 |
| 曾有“新来源”确认 | [PR #52](https://github.com/chunxi-alpc/TRIAGE/pull/52)：Gemma 的20指南来源确认；[PR #31](https://github.com/chunxi-alpc/TRIAGE/pull/31)：旧 ASTER 的22测试树 | 非本次候选的 MEGA 独立确认；对应来源已被研究使用，必须纳入 D4 排重，不冒充新来源 |

## 三、还缺什么：只做真正新增单元

| V3任务 | 查重状态 | 应做的最小增量 |
|---|---|---|
| D1：MEGA R0/J_zero 首错的指南条件、患者事实、时间和否定等**语义归因** | 部分已有：首错、margin、LP、来源四向转移已做；未发现对应全量盲审和来源级结构语义归因 | 将既有首错与已有规则类型按 UID 关联；只对未知类别和新诊断原因做增量审核，未知/无法核验单列 |
| D2：E1_pair_only/J_zero 的**选择性改判门控**与**预测符号保持** | 早期门控在别的任务做过；未见与两套冻结候选完全同协议的误修正保护实测 | 先统计跨来源候选改判事件和正负例。如果太少，不训练监督门控，只做零成本/CPU确定性保护与理论上界，避免伪规律 |
| D3：仅在 D1 识别的、具备可程序核验输入的 MEGA 错误类别上做**新语义验证** | 相似机制已在 PR #29 和 CARE 试过；未见当前 MEGA 首错上的同协议记录 | 复用实现，固定最多1–2个未验证错误类别和一次工程修复，不能重复44例/旧192例 |
| D4：当前候选在**来源家族、模板均与历史评价不同**的新指南上确认 | 未见与该候选同协议的新来源验证 | 优先做来源许可证、去重、人工审核和样本量可行性；无资格就标 blocked，不用老测试替代 |

### 明确不属于本轮新实验

[Issue #79](https://github.com/chunxi-alpc/TRIAGE/issues/79) 规划了“完整正交补 + 软双目标”尚缺组合，但这是另一个项目计划，不等于 PR #71/#76/#78 的已完成工作。仓库核查未发现同组合完成报告；**本轮 V3 不自动启动 Issue #79，也不重复其已完成的组成模块**。

## 四、交给 Codex 的查重技术条件

为每个实验建立唯一身份键：基础模型与版本、分词器、输入和任务、标签定义、提示及评分位置、UID/来源划分、表示/特征维数、损失和约束、截距/校准、决策规则、数据集与评分器、逐例输出及报告状态。身份键一致且存量含可核对预测、代码、配置与真实运行报告时，标记 **reuse_completed**；禁止重拟合、主干前向、批量回放或大规模bootstrap重算。

不同模型/任务或同名但不同校准只能标 **related_not_equivalent**；只有文档计划则标 **planned_only**；已完成但未合并的PR标 **completed_unmerged**，不得假装在 main；缺少数据或无法验收的标 **partial/blocked**。新阶段仅补存量 UID/来源/语义覆盖没有的单元。

建议 Codex 的首个实际交付是 **覆盖差异清单**：逐项标历史来源、分数版本、方法、逐例 ID、结果文件与 SHA、是否可以复用、真实缺口。不要以重新运行完整历史程序代替这一清单。

> 审计限制：这里确认的是 GitHub 上公开报告与工件路径，并未验证服务器私有缓存的每一个字节。Codex 需在服务器只做轻量 SHA/样本数/标签契约检查，发现冲突才针对性修复。
