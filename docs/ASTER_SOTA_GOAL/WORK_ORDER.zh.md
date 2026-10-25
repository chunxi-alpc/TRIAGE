# Work 执行任务：ASTER-SEKA 复现与条件化注意力实验

**本文件是可执行研究委托，不是已经启动的任务记录。**

## 可直接交给 Work 的总指令

在 `chunxi-alpc/TRIAGE` 上执行 `docs/ASTER_SOTA_GOAL/GOAL.zh.md`。先核对最新HEAD，基于PR #35及本任务文档分支建立独立工作树；不得覆盖main或旧实验产物。目标是用官方SEKA/AdaSEKA代码构建具有真实注意力贡献的患者适用性条件化方法。先完成官方强基线复现，再比较新方法。首轮只用现有Qwen3-8B与seed=0；完成开发后冻结0/1/2种子方案。任何“优于SOTA”结论必须在同协议下得到验证，不能通过挑选最佳seed或新读过的测试标签达成。

优先检查研究服务器115、既有环境、GPU 0/1/6、本地模型、third_party和数据清单的实际可用性。不要只报告检查计划。如果资源不可用，提交已完成工作和精确阻碍，不能伪称任务已启动或推理已完成。不要自动购买云GPU或使用未授权付费API。

若环境提供多个子代理，按下列分工执行；若不提供，按相同职责串行执行，并在报告中写明没有实际多代理。代理数量不构成模型推理方法的一部分，论文不得把研发协作与多智能体临床推理混淆。

## 协调器职责

- 读取GOAL、PR #30—#35的结果和所有冻结/失败记录；明确旧数据已暴露的范围。
- 将任务注册到task_ledger.json；状态用planned/running/complete_verified/blocked/failed。
- 仅协调器分配GPU。启动前检查占用与PID；每张卡同一时刻一个所有者。使用设备锁与任务hash避免重复推理；不得终止无关用户进程。
- 基线代理和方法代理分工作树；评价与审计代理默认CPU。数据与实现阶段可并行，正式测试必须等待配置冻结。
- 保持每条预测可以追溯到input hash、checkpoint hash、source commit、projection hash、配置、token预算、seed与评价器版本。
- 汇总全部候选，不准只提交获胜项。与main整合需另行审查，本轮只开PR。

## Agent A：一手资料、数据与污染审计

输入：GOAL、configs/datasets.json、原文论文、数据卡、既有运行清单。
输出：source_registry.json、dataset_manifest.json、split_manifest.json、LICENSE_AND_EXPOSURE.zh.md。

任务：
1. 核对MedGUIDE原始任务、我们反事实构造和新临床数据任务的差别；不混淆不同上下文权限的排行榜。
2. 验证MEGA-CDP文件、train_val/test来源指南分组、与MedGUIDE重叠；验证MedPIC多选答案集合和89个链接对，记录实际pair元数据，无法精确还原时将pair指标标记blocked而非猜配。
3. MedPIC原生任务没有直接提供完整规则输入时，建立独立的统一检索/开卷赛道；只用患者和问题检索，不用gold。所有方法共享相同来源文档，保留原生评测另表。
4. 保留MedPIC全量外部测试；不基于测试错误删题或建立私有有利子集。MEGA-CDP校准只能用train_val。
5. 解析训练/开发样本格式；test正文/标签的详细检查限制在必要的数据审计，方法代理不查看gold。
6. 官方许可TBD或授权缺失则标记风险/阻碍；禁止上传指南全文、患者文本或受限数据。不能让AI代替真实医生签字。

## Agent B：官方基线复现

输入：原作者SEKA/AdaSEKA、PASTA、CAD仓库；现有third_party清单。
输出：baseline_registry.json、baseline_adapters/、projection_manifest.json、OFFICIAL_REPRODUCTION.zh.md。

任务：
1. 固定上游commit、许可证与隔离环境；读取官方构建器，不使用未匹配模型的旧投影或随机头模拟官方方法。
2. 首先在作者标准CounterFact及有权使用的BiasBios评测上复现；精确匹配checkpoint、提示、split、输出长度、头筛选、projection与指标。未取得作者checkpoint时区分“适配到新checkpoint”与“论文精确复现”。
3. 在现有Qwen3-8B构建目标checkpoint投影。官方SEKA未提供预制权重时用builder生成；无梯度校准数据不能来自最终测试。
4. Spotlight没有确认到官方完整代码时，记录“按论文复现”；不将PR #35的组级截断动态偏置标成Spotlight原实现。
5. ProbeRAG/JUICE训练或模型架构依赖未解决时，只记blocker。TrialGPT仅可用于条件模块参考，不把matching smoke当完整患者试验匹配SOTA。
6. 给出原作者任务上的复现差距后，再适配统一临床harness。

## Agent C：注意力方法研发

输入：Agent B通过验证的SEKA、Agent A冻结的训练/开发数据。
输出：aster_seka_adapter.py、phase_role_policy.json、tests/、METHOD_DIFF.zh.md。

仅实现最小增量：
- 处置盲判时，按预测span高亮患者证据和规则条件，改善T/F/U判断。
- 执行时，高亮预测适用的条件—动作片段与所需脚注；共享词元去重，未知不强制删除。
- 相同原始规则支持下比较普通SEKA与角色/阶段调度；不能同时更换验证器、提示和底座后只把增益归于注意力。
- 对照包含：仅判断阶段引导、仅执行阶段引导、两阶段引导、均关闭；与同信息的直接映射/校准融合/裁剪比较。
- 先不加入GRPO、值向量新模块或旧输出残差。二阶改动必须先记录开发假设，最多两轮并完整保留失败。
- 支持单选、原生多选集合和路径生成三种输出契约；不能把MedPIC多选或MEGA路径生成偷换成单字母任务。

必须测试：零强度同路径、RoPE及GQA/KV映射、padding/causal/cache一致性、头选择、词元span对齐、共享/重复规则、无支持回退、gold字段不可访问、掩码泄漏。键编辑不自动阻断间接信息路径，不能声称已经严格盲判而不测试。

## Agent D：冻结评价与统计

输入：统一harness、冻结候选、预定比较族；正式test在冻结后才开启。
输出：config_lock.json、全部匿名predictions、metrics、置信区间与CALIBRATION_AND_TEST.zh.md。

- 每方法最多27个预登记开发配置，比较真实计算预算、额外调用和校准样本数，不仅比较配置数量。
- 当前四树MedGUIDE只作探索回归。MEGA-CDP train_val按来源指南留出开发/校准，全部同源路径及近重复患者同组。
- MedPIC exact-set accuracy，不用argmax，也不把每个选项当独立样本；Pair指标按官方链接对计算。
- MEGA-CDP按原PCC/Acc评价。评审模型若受成本或授权限制，需要写明blocked或另设本地评审赛道，不能与作者分数混排。
- 三个预定seed均完整报告；数据seed、projection seed与生成seed各自记录。贪心重复不制造“独立重复”。
- 整指南簇/来源规则簇配对bootstrap，所有患者版本与seed同时入簇；主要比较预登记并Holm校正。少簇只作描述，不以10000次bootstrap代替独立来源。
- 模型解析失败计入完整分母；披露所有排除和数据许可问题。

## Agent E：独立复核与报告

输入：代码diff、日志、数值预测、manifest、源文件hash；不可替研究团队补造结果。
输出：AUDIT.zh.md、RUN_STATUS.json、最终PR描述。

核对：
1. 已实现 / 已测试 / 已完整运行 / 已独立确认四种状态区分。
2. 原始GPU次数、缓存命中、CPU再分析和新增前向分别计数。
3. 样本全覆盖、唯一key、方法完整、种子完整、所有配置、真实退出码、模型/投影指纹。
4. 给每张主表标记native/reproduced/adapted/derived/exploratory/confirmatory。
5. 专门验证注意力关闭后是否失去收益；改进若来自条件验证或输出融合，不得误归因。
6. 保留负对照和负结果；只有核对成功的运行才标记complete_verified。

## 资源与阶段

- 本轮使用已有研究服务器，不引入新付费服务。
- G0数据审计与基线环境可并行；G1官方复现通过后才进入G2方法优化。
- 首次模型smoke上限16例，不使用最终测试；G2先seed0。完成后冻结方法与种子，才进入G3外部测试。
- 多代理不允许并发更改同一结果文件或同一Git分支。数据、基线、方法使用各自分支，协调器合并到本计划工作分支。
- 失败时立即保留日志和部分产物；续跑以输入/配置/hash确定是否可复用。不停止后自动将运行标记成功。

## 最终交付与成功定义

成功研究目标见GOAL第7节：同底座超过最强已复现基线，外部主任务指标及非劣效性达到预设条件，注意力消融有独立贡献，多seed全部报告。

即使未达到效果门槛，执行交付也必须完整：代码+复现命令+全部结果+失败解释+资源账本+精确阻碍。不得承诺“必定第一”。没有真正Work/多代理入口时，文档交付不是已调度执行，必须明确告知用户。
