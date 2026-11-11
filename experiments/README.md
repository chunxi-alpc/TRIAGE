# 实验代码入口

**[方法与实验总表](../docs/ASTER/experiments.zh.md)**统一记录全部探索的规模、结论、状态和归档。

| 方向 | 代码与说明 |
|---|---|
| 闭式、连续衰减与参考对照 | [aster_closed_form](aster_closed_form/SYNTHESIS.zh.md) |
| 可学习M3五方法全量实验 | [aster_learnable_m3](aster_learnable_m3/README.zh.md) |
| 节点、迁移、选头及N/S/NS | [aster_m3_followup](aster_m3_followup/README.zh.md)；9配置训练与200题迁移完成，主测试未完成 |
| CR / repair-v2基础实现 | [aster_cr](aster_cr/README.zh.md) |
| MEGA-CDP配对数据构建与独立审核 | [aster_cr_pair_dataset_20261111](aster_cr_pair_dataset_20261111/README.md)：合格6,345对，均衡2,540对，数据留服务器 |
| CR优化、证据保护、机制验证及模拟 | [源码和结果归档](../results/aster_cr_research_20261109.zip) |
| 五模型、MedPIC和跨任务适配 | [源码和结果归档](../results/aster_cr_transfer_20261110.zip) |
| 新痕迹修复（未完成） | [代码和状态快照](aster_cr/trace_repair_20261110.zip) |
| 普通读出、M1及Gemma对照 | [普通读出](ordinary_readout.py)、[M1](m1_diagnostics/README.zh.md)、[Gemma](gemma3_head_pilot/README.md) |
| 遮挡与三数据集补充 | [遮挡](semantic_masking_followup_20261105/README.md)、[三数据集](three_dataset_analysis_20261105/README.md) |
| CARE既有实现 | [V4](care_module1_v4/README.md)、[V5](care_module1_v5/README.zh.md)、[V6](care_module1_v6/README.md)、[A1](care_module1_semantic_v8/MODULE1_A1_FREEZE.zh.md)、[有符号诊断](care_module2_signed_20261019/README.md) |
| 早期CARE、适用性及SEKA支线 | [缺失源码和原报告归档](legacy_research_20261110.zip) |
| ACRE | [方案、结果与停止结论](../docs/ACRE/README.md) |

包内保留源码结构、参数、已有结果及来源说明；大模型权重和大缓存使用原服务器路径。[运行环境](../docs/RUNNING.md)，src模块接口见各自README。
