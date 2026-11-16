# 交付报告与复现说明审查

审查日期：2026-10-19。对象为 REPORT.txt、REPRODUCE.txt、src/build_report.py 及其方法/依赖入口。本次只做静态审查，不修改 engine、方向、参数，不监控自由生成或 C_V6 进程。

## 总体判断

已审版本对论文与适配方法的边界说明充分：正确标出论文 2026-10-14、固定官方 commit、ClashEval 适配、五种指令包装不等于问题改写、11 个合格训练样本、重新定位至层 31、末提示 token 单次干预、原 CARE 文件缺失、C_reduced 与 C_V6 区分、D_adapt 零扰动、合成上下文旧规则不等于参数旧知识。未发现把无显著性结果描述为显著提升的表述。

目标病例准确率被限定为候选字母评分，后续自由生成单列探索性结果；领域名称变化不称临床外部验证。这些限制需要在更新后的报告继续保留。

## 已向主代理提出的修正

1. build_report.py 第 34 行在生成未完成时硬编码“552唯一完成项”，可能把计划数误写为已完成数。运行中应写计划数量，完成后读取 free_complete.json 的实际 unique_completions。

2. 第 41 行“旧规则失效为0”缺条件。care_gate.py 实际先判断 NEW=T，此时给 +1；应写“NEW=F且旧规则有效给−1；NEW=F且旧规则失效或NEW=U给0；无冲突给0”。不需要改算法。

3. REPRODUCE.txt 的 C_V6 依赖恢复不够具体。交付包不含 .deps 二进制，需要给出在新目录恢复 bitsandbytes==0.50.2 至 .deps 的明确步骤，并保留已记录的本地包来源和摘要。只指向记录文件不足以完成新的离线目录恢复。

4. build_report.py 若在复跑目录继续使用，会把第 32、42、61 行硬编码准确率/p值、D dev结果、源任务计数写成旧结果。当前数值与本次记录一致；但可复现入口应从实际文件动态计算这些说明，或明确 builder 是本次快照而非通用重跑报告器。D 的显示名也硬编码 epsilon=0。

5. Qwen3-8B 的实际权重摘要已在 model_environment_manifest.json；C_V6 如果作为交付结果，应额外固定 Phi-4、LoRA、BioLinkBERT 的实际权重/配置摘要。仅保留硬编码本地路径不能保证在其他目录复用完全相同模型。

## 依赖闭包与当前可复现范围

- engine.py 的 ROOT 根据脚本路径定位；MODEL 保留原服务器路径，文档已说明修改位置。
- data/care_gate.py 使用相对 ROOT 的 sources/care_v6_frozen.py；数值逻辑路径不依赖外部 CARE 仓库。完整 C_V6 的 .trialmatch_parser 相对导入对应 sources/trialmatch_parser.py，sources/__init__.py 已存在。
- run_rules.py 需要 data/cases.jsonl、results/direction.pt、results/selected.json；文档提供 facts→rules 顺序。
- run_free.py 使用相同冻结方向和输入，D=0 与相同 C 系数进行精确复用；它不会读取已保存 free_completions 做断点续跑，重启前需要新结果目录，文档已警示 append 文件不能重复写入。
- 复现时直接用冻结 data/facts.jsonl 不需要原始完整 parquet；重抽数据需要原服务器 parquet，文档已区分。
- analyze.py 只依赖 NumPy/SciPy 和冻结结果，可在服务器 CPU 重算统计。新目录需要 results、logs；建议同时保留 audit、sources、data、src 的完整目录结构。
- run_*.sh 硬编码原目录和环境，文档已明确在副本修改或用逐条 Python 入口。

本次未对最终压缩包逐成员核验，因为尚未提供实际包路径；这里只核对了服务器上的入口和依赖关系，不能将其表述为压缩包恢复实测。
