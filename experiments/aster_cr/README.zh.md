# ASTER-CR：新来源确认实验

全部运行位于服务器 gcx@10.29.128.115，使用 gcx261014 环境和 GPU 0、1、6。

运行目录：/home/gcx/aster_cr_20261108  
代码目录：/home/gcx/TRIAGE-worktrees/aster-cr-20261108/experiments/aster_cr

当前状态以运行目录的 status/pipeline.json 为准。REPORT.zh.md 与 summary.json 只展示已产生的结果，未完成阶段保持 pending/in_progress。

## 固定数据

- 来源 fit/dev/test：80/20/40；独立节点：761/199/372。
- 原始病例路径：1,712/420/806。
- 复用既有来源对应表，依据实际历史输入记录排除已用来源，不计算文件哈希。
- 同源不同树、病例与已有语言对应关系共用划分。明细见 data/history_usage.json 与 data/sources.jsonl。
- 原始 Full 采用发布的完整决策树和病例；完整指南文档保留为来源记录，不无差别追加进模型上下文。
- 受控变体由现有病例出发。Qwen 抽取完整规则、事实与原文证据，程序计算变体标签；抽取失败、标签不一致和不可用类型分别记录。此过程不是人工临床标注。
- C 的左右两侧都使用同一程序化条件格式，避免把条件格式变化当成规则变化。F 使用预设数值网格；同侧变化另列 F0。明确计数变量只用合法整数。
- A 分交换、去处置与等长中性替换；需要的处置定义受保护。P 的改变字段不能出现在决定性规则及祖先规则中。

## 执行流程

prepare → annotate → build → fit/dev score → P0 → localization/H2/H3 → readout → training baselines → freeze → test → MedGemma confirmation。

controller 的 prepare 阶段负责数据与 Qwen fit/dev 特征。pipeline --after-preparation 等待该阶段完成后继续所有后续阶段。阶段失败时停止后续工作；完成任务由 jobs/ 记录，重启时复用。每张卡启动前按计划等待足够空闲显存。

启动或恢复：

```bash
cd /home/gcx
source gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-cr-20261108
python -m experiments.aster_cr.controller --stage prepare
python -m experiments.aster_cr.pipeline --after-preparation
```

长期运行使用 launch.json 中记录的后台进程；不要同时再启动一个调度器。首次部署已启动，不需要用户再执行上述命令。

## 方法与验证

- 统一采用未编辑前缀缓存、最终 token 前向与 FP32 标签投影。旧完整前向分数不与新缓存前向混用。
- 同模型、同输入、同输出标签的分数和特征按文本直接复用，不做哈希。
- 选头复用已有双向干预函数，排名仅用 fit；普通配对保留，原错误和近边界配对加权。
- H2* 必须在 dev 上 R² > 0，且信号泄漏 RMS 不超过真实信号 RMS 的 10%，才启用主方法干扰项。
- H3 比较残差、选中头和全头的 fit-only 降维及 ridge，包含打乱标签对照；不足 100 个独立错误节点时保留证据不足说明。
- 主方法没有获得诊断支持时关闭相应分量；全部关闭时不浪费完整秩/正则网格，保留探索分量和核心消融。
- 普通读出与相应 CR 模型具有相同系数数量。支持空间是 fit 估计量，不能混称为梯度训练参数。
- 四个训练对照统一使用全部 fit 训练输入、3 个学习率和每阶段一轮。LoFiT 的选头及偏移训练是两个阶段，分别计成本。
- LoReFT 直接使用官方组件。LoFiT 官方代码不支持 Qwen3/Gemma3，因此按官方逐坐标缩放及偏移公式适配注意力钩子，保留两阶段。LoRA 使用官方 PEFT。最后一层对照训练残差偏移。
- ITI、SEA 采用正确/错误 fit 答案补全的二元任务适配；设置和范围写入 comparators/settings.json。SEA 使用官方投影层，对未定义的零分母坐标回退为原值，并记录次数。
- 校准、模板、CR 和训练基线均先检查 dev 路径下降不超过 0.5 个百分点，再按 PA 宏平均选择。新 test 不参与选择。
- MedGemma 重估其自身特征和权重，复用 Qwen 选定方法、秩、正则和训练基线学习率，不运行完整网格。

服务器验证已包括规则端点、事实证据、修改保持、来源 bootstrap、路径选择门槛、支持正交性、梯度、缓存与完整前向一致性、模型保存恢复。真实 Qwen 特征提取也完成首例验证。完整实验结果仍需等后台任务完成。

## 输出

- REPORT.zh.md / summary.json：汇总。
- data/variants.json：配对数、独立节点数、来源数及 C 是否达到目标。
- qwen|medgemma/diagnostics：P0、输出形式偏置、阈值响应及清理效果。
- qwen|medgemma/h2.json、h3.json：诊断与方法启用条件。
- qwen|medgemma/frozen.json：测试前固定配置。
- qwen|medgemma/readout_scores、path_results、baselines、comparators：逐例结果。
- qwen|medgemma/cost、logs：耗时、显存、前向与训练记录。
- 缓存路径的时延是已测未编辑前向时间与读出时间之和，不冒充在线墙钟基准；特征预计算全部节点的额外成本单独报告。

按来源进行 2,000 次配对 bootstrap。路径差值区间下界达到 -0.5 个百分点才报告非劣性。既有 M3、闭式和校准结果作为历史证据保留，不重新训练旧设置。
