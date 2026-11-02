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

## 项目版本与实验路线

本仓库经历了 TRIAGE／CARE／ACRE 到 ASTER 的研究路线演进。**PR 和实验分支是研究版本记录，不等同于已进入主线的正式发布版本。** 下表按时间与依赖关系整理；合并状态以链接中的 GitHub 页面为准。

| 时间／版本 | 主要工作 | 结果与当前定位 |
|---|---|---|
| 2026-10-14 · TRIAGE 初始主线 | 临床指南规则检索、条件判断、证据定位与注意力干预的三模块框架；建立 Qwen3-8B、数据与运行配置 | 原始基线，完整保存在 [pre-ASTER 快照](https://github.com/chunxi-alpc/TRIAGE/tree/archive/pre-aster-main-20261026) |
| 10-15 · ACRE pilot（[PR #1](https://github.com/chunxi-alpc/TRIAGE/pull/1)） | Qwen3-8B；40 对／80 例／36 个指南；比较固定增强与适用性切换、Oracle 对照及 AttentionBias | 完整规则切换未优于固定增强；按停止条件暂停，不作为已验证的部署方法 |
| 10-16 至 10-17 · CARE M2 O2（[PR #2](https://github.com/chunxi-alpc/TRIAGE/pull/2)、[PR #3](https://github.com/chunxi-alpc/TRIAGE/pull/3)） | 重分析 288 个层／KV 头；比较原始键、SEKA 子空间、配对适用性方向、对比子空间和随机方向 | 未发现稳定的学习子空间优势；保留完整负结果与复现日志 |
| 10-18 · CARE M1 V4–V6（[PR #4](https://github.com/chunxi-alpc/TRIAGE/pull/4) 至 [PR #7](https://github.com/chunxi-alpc/TRIAGE/pull/7)） | Qwen3-8B 证据结构化实验；TrialMatchAI 解析器 V2/V3；V6 evidence-only fusion 与 pipeline 接口 | V4–V6 是连续实验版本，不应把各版指标拼成同一模型结果；V6 集成 PR 保持独立 |
| 10-19 · CARE 有符号干预与可靠性（[PR #8](https://github.com/chunxi-alpc/TRIAGE/pull/8)、[PR #9](https://github.com/chunxi-alpc/TRIAGE/pull/9)、[PR #11](https://github.com/chunxi-alpc/TRIAGE/pull/11)） | Qwen3-4B 的有符号 B4、均值方向、直接分数偏置；层段×强度扫描、反号／错配控制、真实键编辑；CARE V7 与结构修复 | 注意力编辑确实改变局部读取，但适用性特异的独立答案收益有限；结构修复收益更明确。PR #11 的扫描骨干是 4B，不是 8B |
| 10-19 · CARE 8B 上下文冲突（[PR #10](https://github.com/chunxi-alpc/TRIAGE/pull/10)） | Qwen3-8B 在 ConflictMedQA／DriftMed 上运行 3,158 次生成，学习参数知识与上下文冲突方向 | 严格已知测试事实仅 1 组；结果不足以支持 CARE 有效性结论，属于独立冲突诊断 |
| 10-19 至 10-20 · CARE M1 V8／A1 冻结（[PR #12](https://github.com/chunxi-alpc/TRIAGE/pull/12) 至 [PR #16](https://github.com/chunxi-alpc/TRIAGE/pull/16)） | 语义直判 A1、V8.1 复核、U 对照示例；冻结 A1 为模块二研究信号；PR #16 将 A1 接入 main | **PR #16 已合并**。A1 为研究信号接口，不代表替代全部历史 V6 运行契约 |
| 10-20 至 10-21 · CARE 增量机制与决策询问（[PR #17](https://github.com/chunxi-alpc/TRIAGE/pull/17)、[PR #18](https://github.com/chunxi-alpc/TRIAGE/pull/18)） | N1–N5 未测机制计划、decide-or-ask 与不确定性传播 | 主要是后续实验方案／接口研究；不得误记为已完成 GPU 结果 |
| 10-21 至 10-24 · ASTER／MedGUIDE 先导（[PR #19](https://github.com/chunxi-alpc/TRIAGE/pull/19) 至 [PR #30](https://github.com/chunxi-alpc/TRIAGE/pull/30)） | TrialGPT A1、MedGUIDE A/B/C 反事实、Qwen 200／800 例扩展、阈值×规则顺序控制及 E2–E8 | 多个 PR 是串联分支或不同数据版本；按各自冻结数据与状态引用，不能直接合并成一个成绩 |
| 10-25 · ASTER 稀疏动态注意力路线（[PR #31](https://github.com/chunxi-alpc/TRIAGE/pull/31) 至 [PR #37](https://github.com/chunxi-alpc/TRIAGE/pull/37)） | P0–P3、动作质量守恒、动态融合、Qwen seed=0、官方 SEKA 基线与外部评测 | 多项对照未超过简单基线；保留为方法筛选与消融证据 |
| 10-28 · ASTER 离线机制与协议（[PR #38](https://github.com/chunxi-alpc/TRIAGE/pull/38)、[PR #39](https://github.com/chunxi-alpc/TRIAGE/pull/39)、[PR #40](https://github.com/chunxi-alpc/TRIAGE/pull/40)） | 离线适用性子空间、可控读取与答案读出诊断；统一实验章节、指标和冻结协议 | PR #39 已合并；PR #38／#40 保留为独立机制分支，不能将局部读取变化等同端到端性能提升 |
| 10-29 至 10-30 · ASTER 全量执行与论文实现（[PR #41](https://github.com/chunxi-alpc/TRIAGE/pull/41)、[PR #42](https://github.com/chunxi-alpc/TRIAGE/pull/42)） | Qwen3-8B 固定修订；离线扫描 280 个候选 KV 头并按 rank 8 选择专家；真实临床确认、机制对照及 MEGA-CDP 官方全量执行 | **这就是 Qwen3-8B 扫描并选择注意力头的工作。** PR #42 记录的冻结头组为 33:3、31:1、33:4、23:4；43 例确认集只有探索性证据，官方 7,458 例任务在该 PR 更新时仍在运行。不得把中间结果写成最终官方成绩 |
| 11-02 · ASTER 方法修订（[PR #43](https://github.com/chunxi-alpc/TRIAGE/pull/43)） | 将适用性评估与缓存驻留注意力门控置于方法核心；键平移降为受控消融；新增 B0–B5 阶段门槛 | 文档与实验计划修订，尚未替换现有代码实现；新 B2–B5 与端到端实验待完成 |

### Qwen3-8B 注意力头扫描：明确的证据位置

- [PR #42：固定模型修订、临床确认与四卡官方测试](https://github.com/chunxi-alpc/TRIAGE/pull/42)
- [280 个候选 KV 头的 rank 选择记录（run03）](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-manuscript-implementation-20261030/experiments/aster_manuscript_20261030/public/run03/rank_selection.json)
- [run03 选中结构](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-manuscript-implementation-20261030/experiments/aster_manuscript_20261030/public/run03/selected_structure.json)
- [run05 选中结构](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-manuscript-implementation-20261030/experiments/aster_manuscript_20261030/public/run05/selected_structure.json)
- [临床确认集头选择记录](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-manuscript-implementation-20261030/experiments/aster_manuscript_clinical_20261030/public/clinical01/rank_selection.json)

注意：run03、run05 与临床确认阶段是不同冻结记录，文件中选中的头组并不完全相同。复现或引用时必须同时注明运行批次、数据划分和冻结配置，不能把不同批次的头组拼成一个结果。PR #42 的临床实验冻结头组以该 PR 正文及 clinical01 产物为准。

## 分支与合并原则

- 当前首页所述 ASTER 方法和结果是主线入口；历史 CARE／ACRE 研究通过上表和 [历史分支索引](docs/BRANCHES.md)追溯。
- 已合并的代码／文档版本：以 GitHub PR 的 merged 状态为准；当前可确认的关键整合包括 PR #16（CARE A1）、PR #22（ASTER MedGUIDE 反事实预实验）、PR #39（ASTER 实验协议）和本次 PR #44（首页版本历史与 Qwen3-8B 选头索引）。
- 仍开放的实验 PR 不因“版本整理”而直接合并。特别是 PR #41／#42 的执行实现与 PR #43 的方法修订存在研究设计差异；先完成方法决策、逐文件审查和回归测试，再决定合并或保留为独立版本。
- 旧实验、负结果、原始冻结配置和模型指纹均保留；整理首页不覆盖历史结果，也不把未完成实验标成完成。
