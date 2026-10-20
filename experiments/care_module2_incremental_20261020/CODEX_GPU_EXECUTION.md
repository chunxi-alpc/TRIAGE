# Codex SSH/GPU 运行手册（仅新 N1–N5）

**本文是给 Codex 未来执行的任务规范，不意味着本次 GitHub 文档提交已经用 GPU 跑过模型。**

## 0. SSH/仓库/授权

历史机器曾使用 \`gcx@10.29.128.115\`，环境 \`/home/gcx/gcx261014/bin/activate\`，Qwen 权重曾位于 \`/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base\`。路径来自旧实验报告，需要 Codex 在其**自身获授权**的连接中验证；此前其他执行环境出现过 SSH connection refused，故**不能假定连通或权重路径仍存在**。不得向 GitHub 上传 SSH 秘钥、密码、私有模型权重和患者原文。

远端示范检查（仅验证、没有启动任何 GPU 模型）：

~~~bash
ssh gcx@10.29.128.115
hostname
nvidia-smi
test -f /home/gcx/gcx261014/bin/activate
source /home/gcx/gcx261014/bin/activate
python -c "import torch, transformers; print(torch.__version__,transformers.__version__)"
~~~

连接失败时停止并报告原因，不借用未经授权的计算入口；GPU 被他人占用时不抢占、不 kill 进程。正式运行阶段由 Codex 选择空闲授权 GPU，并在 run manifest 记录 ID、型号和可用显存。历史峰值只可辅助预算，不能替代本轮实测。

GitHub 主仓库：https://github.com/chunxi-alpc/TRIAGE

工作分支：\`codex/care-module2-n1-n5-only-20261020\`

建议使用独立 worktree，保护正在进行的其他 PR。工作副本下载/检出方式随实际服务器 repo 路径选择，不重设历史工作区，不对主分支强推。代码先写入该分支，然后每阶段持续更新同一个 draft PR。

## 1. 第一个提交必须是 CPU-only 的可执行研究包

在启动 Qwen 前必须实现并验证：

- N1 R×A 同机制四格 oracle；各家族的独立变量真的可以分别翻转。
- N2 中性接收 x、C/P 两个不同接收 gold、train/test 事实组隔离；区分 synthetic prior 与真实参数知识。固定 4B 的已知事实筛查本身需要模型前向，应计入 N2 预算，不能在 CPU 阶段虚构验证。
- N3 D/Q 对所有合法补全求 \`S(x)\`；Q 属性集合不使用模型推断构建。
- N4 来源×充分性×规则适用性×顺序 16 格合法 gold；如不可满足应在数据冻结前拒绝家族并记录，不能等 test 输出后删除。
- N5 从 PR #12 不可变 commit 恢复 A1 输入/输出和规则文本，通过精确 ID、文本/规则哈希，完整保留 U/null 和不合资格母集。
- 输出 prompt allowlist 检查、fold/group overlap 报告、标签双重计算/一致性检查及冻结的 SHA256 manifest。
- 训练/dev/test 有真实底层机制组隔离，旧 600 例不得未经修订混入正式 test。

CPU 自动测试全部通过后先提交 \`status: cpu_verified\`；不把 CPU test 视为神经效果。

## 2. 开发新的 GPU CLI：命令必须真的存在

以下为**目标 CLI 规范，不是目前已存在的命令**。Codex 实现后需在 README 改为**实际可执行命令及其 help 输出**，再进行正式计算。

~~~text
python -m care_m2_inc.cli prepare  --config configs/... --out runs/.../data
python -m care_m2_inc.cli verify   --config configs/... --out runs/.../check
python -m care_m2_inc.cli smoke    --model ... --config configs/... --out runs/.../smoke
python -m care_m2_inc.cli fit      --experiment N1|N2|N3|N4|N5 --split train ...
python -m care_m2_inc.cli select   --experiment ... --split dev ...
python -m care_m2_inc.cli evaluate --experiment ... --split test ...
python -m care_m2_inc.cli analyze  --experiment ... --run ...
~~~

\`fit\` 只能见 train，\`select\` 只能见 dev，\`evaluate\` 必须读取已锁定的 selection+hash，遇到不符应拒绝。test 只能运行一次冻结选择配置；完整运行中断时通过 immutable journal + case IDs resume，不重复计算已完成且可核验的同 hash forward。所有输出唯一 \`run_id\` 目录，不能覆盖旧实验。

## 3. 执行顺序：按 S0–S6 有条件运行

| 阶段 | 前置条件 | 运行内容 | 计划 soft cap* | 阶段产物 |
| --- | --- | --- | ---: | --- |
| S0 | 有效代码和数据 | 全量 CPU 真值/泄漏、清单哈希 | 0 | split/manifest/tests |
| S1 | S0 通过 | Qwen GPU 最小 smoke：zero、hook、target、非 target、RoPE/GQA、实际编辑 | 100 | smoke.json、实际 GPU memory |
| S2/N1 | S1 通过 | R×A 独立路由；新测试的文本、直接 bias、空间/随机对照 | 1800 | R/A 成对结果与错误分层 |
| S3/N2 | S1 通过、接收/真实 P 资格明确 | 新 4B 双向 C/P，same receiver、跨内容 IIA、闭卷资格 | 1500 | 来源每向和 donor copy 指标 |
| S4/N3 | S1 通过、补全 oracle 正确 | D/Q 充分性和决定性追问 | 1200 | D/Q 双风险及问对属性 |
| S5/N4 | N2/N3 单轴达到冻结的有效性门槛 | 16 格来源/充分性/规则适用/顺序组合 | 1600 | 串扰、顺序效应及完整块指标 |
| S6/N5 | A1 冻结预测精确连接、合法动作与下游 gold | 新旧兼容的新 Qwen 数据上的 A1 同信号强基线与路由 | 800 | A1 错误传播和增量评价 |
| 合计 | 逐阶段停止门槛 | **仅规划上限** | **7000** | 非承诺必须全部运行 |

*规划数字来自研究设计，不是实际逐案网格计算结果。实现器必须在每个阶段启动前展开方法×输入×条件×候选配置得到精确 forward 需求，区分缓存基线、训练激活提取、选参与新测试。若预计超过该阶段预算，**先提交修订预算/缩小未启动的 dev 候选空间，不得删除必需控制或调整已冻结测试以凑预算**；不得静默超限。

N1 可在 N2/N3 完成前独立执行；N2/N3 互不要求彼此成功；N4 依赖 N2/N3；N5 在连接问题解决后可以做单独准备，但解释层级上仍需比较已确立的路由控制方法。若阶段停止，记录未执行的后续阶段为 \`not_run_gate_fail\`，不能空置然后给出“已完成”总报告。

## 4. 历史 GPU 结果直接拿来用，不重跑

PR #3 的 B1–B4/随机基线、PR #8 的 signed 和直接 score bias、PR #9/#11 的上游门控与结构对照、PR #10 的 Qwen3-8B 来源方向、PR #12 的冻结 A1 24 例四臂，以及 \`codex/module2-preexperiment-audit-20261020\` 的离线复算全部保存为历史 evidence。需要时读取对应原始 journal/报告/哈希，禁止启动旧完整 experiment launcher。仅对新实例运行必需同输入基线。

具体历史引用和不可合并统计分母见 [HISTORY_REUSE.md](HISTORY_REUSE.md)。

## 5. 隔离神经干预与答案泄漏

- Qwen3-4B 固定 revision \`906bfd4b4dc7f14ee4320094d8b41684abff8539\`，主轨 BF16，SDPA，batch=1，关闭 cache；记录 torch/transformers/flash backend 版本与权重及 tokenizer SHA。
- 复用 PR #8/#11 的 RoPE/GQA、QK 分数审计的正确实现；若新的残差 hook 或自定义 SDPA 路径不同，必须在 S1 重新做极小的零干预和目标-非目标效应验证，而非完整旧扫描。
- 所有干预不得直接读取 test \`gold_answer\`、\`oracle_DQ\` 或 receiver gold；控制 bit 只来自用户政策、train 派生预测或单列 oracle 上界，输出需单独标识。
- 每个 sample 的 \`action_span\` 和 \`source_span\` 必须在**最终 tokenized prompt**检查；禁止误改候选答案 token。无目标 span 时不可静默退化，应记录 no-op 原因和母集分母。
- IIA 严格用 receiver gold 评分，另计算 donor-copy 分布；probe 能区分来源、低秩具有能量、甚至编辑 attention 都不等于答对。

## 6. 每阶段提交规范与停止

每阶段需要锁定：

- \`data_manifest.json\`（来源/许可/合成逻辑/分组/哈希/无泄漏）、
- \`train_fit.json\`、\`dev_selection.json\`（拟合和选择各自输入 hash）、
- \`run_manifest.json\`（模型权重、tokenizer、host/GPU、前向数、耗时、峰值显存、git SHA）、
- \`test_predictions.jsonl\`（所有任务方法的完整逐例记录和失败）、
- \`summary.json\`、\`per_family.csv\`、\`failure_cases.jsonl\`（统计与错误解释）、
- \`STATUS.md\`（\`not_run,blocked,cpu_verified,gpu_smoke_passed,gpu_evaluated\`之一）。

历史来源可能有不同数据集许可，不允许重新公开受许可限制的原始医疗文字。大文件保存授权服务器和 SHA manifest，只有尺寸适宜且许可允许的衍生结果推送到 GitHub。

优先在每个停止门槛后提交一组小型可审查 diff；验证 PR 分支 head 后用普通 push，不 force，不篡改既有历史。**实验失败也是有效交付**：保留其日志、对应本研究假设的负结论，以及跳过下一阶段的理由。

