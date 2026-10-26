# ASTER 历史结果与独立审计边界

审计日期：2026-10-25。审计者：独立审计代理。审计对象固定为 PR #35 的 cbb4d613a59618a7f98efccea60a9a469037f0ad 与 PR #36 的 f37c5f8b6339216439e5fc257419ebe78ddf5763。

本次已完成仓库文件、指标含义、源码公式、冻结记录和指纹登记的只读核查；协调器已在115服务器执行独立复算脚本v2，状态为complete_verified。本代理读取了完整返回产物 legacy_independent_audit_v2.json：PR33两模型共1600条匿名CSV统计及配对结果通过，PR35全部351组候选逐例预测、52个留树选择、匹配负对照与3个产物哈希通过。没有运行本地实验，没有新增GPU前向，没有读取MEGA-CDP或MedPIC外部测试答案。PR33仍限于CSV统计复核，未重放其全部原始logits；服务器原始模型推理未重跑。

## 可以支持的结论

已发布的 Qwen seed=0 四树开发比较中，普通静态概率融合为 500/800，动态融合 504/800，原全层动作注意力残差 506/800，新稀疏动态主方法 501/800。不能只比较主方法和原模型判断注意力贡献：PR33 残差与其同参数无残差为 506 对 505，纠正16、退化15，按树描述区间跨零；PR35 新主方法与同参数无残差均为501，纠正2、退化2，区间为[-0.375,+0.375]个百分点。

PR35 原始前向为392/800，对应零强度373/800，纠正19、退化0。这表明该开发集的原始输出受干预改善，但加入强概率融合后没有净收益。不能把原始输出的19条改善写成在强基线之上的增益。

全部负对照必须保留：PR35 反向偏置独立选参510/800，使用主方法相同参数498/800；符号翻转同参数502/800，比主方法高1条；数值漂移同参数也502/800。PR33 Llama 同参数符号翻转63.375%，高于正常残差63.250%。这些结果不支持稳定、方向特异的注意力增益。

“最优”只能按预定协议限定范围。开发表中最高的510/800来自探索性反向负对照，不应选为确认后的最优方法。当前可作为后续实用强对照的是概率融合；PR33 实际冻结的未来验证候选是静态融合 beta=0.75、t0=1、ts=0.25。它尚未在新增独立树上评测。不能宣称任何既有方法已是新主任务SOTA。

来源：[PR35完整报告](https://github.com/chunxi-alpc/TRIAGE/blob/cbb4d613a59618a7f98efccea60a9a469037f0ad/experiments/aster_sparse_dynamic_qwen_seed0/REPORT.zh.md)、[PR33完成报告](https://github.com/chunxi-alpc/TRIAGE/blob/a8c1e49c0fbda941282daf25908004407eb0b1bf/experiments/aster_dynamic_fusion_seed0_20261025/COMPLETED.zh.md)、[冻结简单融合](https://github.com/chunxi-alpc/TRIAGE/blob/a8c1e49c0fbda941282daf25908004407eb0b1bf/experiments/aster_dynamic_fusion_seed0_20261025/frozen_candidate.json)。

## 数据曝光与旧实验状态

PR31 的 P3 已完整自动评测，每模型12,288条；核心1,000病例×4变体×3数据种子=12,000条，另有CS/CS-null各48病例×3种子。22棵P3测试树与四棵开发树不同，但其结果已发布并被后续复分析。因此当前研究不能把旧P3重新称为全新独立确认。

资源清单记录55棵来源树中53棵已使用；未见于扫描记录的bre_19、cns_23都没有病例。该结论限于已扫描资源，并非已证明全世界没有其他独立数据。四个变体以及三个种子不能视为独立患者。

PR31 P3 的两份人工审核表各0/150，仍未完成。旧结论中的“伤害”是相对原模型答对变错的题目数除以全部记录，不是临床伤害。新报告宜使用“退化”。

PR30早期E1的两模型T2/T3未通过；后续实验没有抹去这些负结果。旧根状态与中间报告存在过时字段，应以实际分阶段计数和后续核对为准。PR31 legacy_audit.json 是一个中途快照，不能用其中Llama旧E5未完整计数否定另一个P3任务后来的完成记录。

官方SEKA曾在GPU6用100对合成样本构建Llama末10层投影，完成5例接口前向。它既不是Qwen检查点投影，也不是CounterFact/BiasBios完整官方复现；不得继承为G1通过。

来源：[P3核对](https://github.com/chunxi-alpc/TRIAGE/blob/cbb4d613a59618a7f98efccea60a9a469037f0ad/experiments/aster_p0_p3_20261025/results/verification_20261025/P3%E6%A0%B8%E5%AF%B9.zh.md)、[曝光资源清单](https://github.com/chunxi-alpc/TRIAGE/blob/cbb4d613a59618a7f98efccea60a9a469037f0ad/experiments/aster_dynamic_fusion_seed0_20261025/resource_inventory.json)、[初期负结果及SEKA接口记录](https://github.com/chunxi-alpc/TRIAGE/blob/cbb4d613a59618a7f98efccea60a9a469037f0ad/experiments/aster_e0_e8_20261024/results/%E6%8A%A5%E5%91%8A.zh.md)。

## 指纹和计算量

PR33 的 execution.json 保存数据、模型配置、源码和原始预测哈希。Qwen配置哈希为 f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30；两模型开发数据哈希为 6d1ebff7b71b451a213868a0b436d27309152920d7ab475c2150fc3c9e2a8bdd。

PR35 的 model_inventory.json 保存五个Qwen权重分片的SHA256，并明确 historical_weight_hash_available=false。因此当前分片可与该记录重新核对，但不能追溯证明PR33与更早实验的实际权重内容完全相同。8条缓存抽查旧原模型logit误差为0，属于抽查一致性，不是全量历史权重证明。

PR35 新前向8,800次，另有44次小试与16次缓存抽查；0.520 GPU小时只指全量新增任务，并不包含历史条件概率计算、所有小试与旧任务。PR33两模型合计约2.347 GPU小时。不要把复用条件缓存或CPU复分析计作新GPU推理。

PR35 存档哈希：results.json 为 0376c5115b995130e4ab591efe8ddb477fe97b3e352ea53898ede2f781014ee4；numeric/predictions.jsonl 为 4740fd5e81fc28df3f310dd865c5513cd7596cbe1bfdbcbf308a4996220ada69；predictions.csv 为 7f9f0416d158c6a9a282de39c013df9c1da4d4983fb1bf70820c5acf846e1f2c。服务器v2已重新读取字节并核对以上三项一致。

## 本轮可复核交付

legacy_evidence.json 保存精确来源URL、Git blob指纹、旧结果、曝光范围及尚未验证事项。verify_legacy_readonly.py 使用标准库独立实现PR35融合公式与全部351候选的预测，留树选择使用历史NumPy归约顺序，另计算精确有理数并列诊断；PR33检查匿名CSV方法计数和配对结果。它仅写指定的新输出文件，不改旧报告，不读取新外部测试。

四树区间通过枚举4^4种整树有放回抽样计算，保留全部患者变体；仅是开发集描述性区间。独立复算实现不导入旧分析器，也不重新构建历史GPU logits。Qwen PR33净差+0.125个百分点、区间[-0.5,+1.0]，Llama净差+0.375个百分点、区间[-0.5,+1.25]；PR35主配置净差0，区间[-0.375,+0.375]。独立核对没有改变研究结论。

首个v1审计在留树选择处失败，失败产物和v1脚本保留。真实原因是v1使用Python 3.12 sum，而原代码使用NumPy mean，数学并列的末位浮点差改变了两个负对照/辅助配置的选参：static_sparse在pancreatic_11折选择16而非精确并列首项15；misaligned在bre_1折选择15而非精确并列首项9。主配置不受此处差异影响。v2按原数值顺序复现后全部通过；这不构成旧预测损坏，但原文“平局取首项”应理解为浮点得分严格相等时取首项。新实验应在冻结前规定数学并列的处理方式，不能在看到外部测试后选择有利规则。

服务器v2耗时3.277秒，Python3.12.3、NumPy2.5.4，新增GPU前向0。脚本SHA256为0b836bc5bcc28b911e91b811feab2d5c958b4565bddf3bc4d7e8dd3f871f97ab；返回JSON的SHA256为8cc8fde35858b2d185abceea911ca3c3b7b4cd09a0b00667e6eb052bd8ad9dfa。

待核对事项：私有病例/原始GPU日志与实际模型权重、现有Qwen与官方release身份、全部新外部数据历史曝光、SEKA/AdaSEKA完整复现、新任务独立结果和真实人工审核。上述事项不能由此历史数值审计替代。

