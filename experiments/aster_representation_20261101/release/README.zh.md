# ASTER 单轮实验归档

本分支保存论文实验运行时的代码、各阶段报告和最终检查点。最终论文及复算数据见 [main 上的论文](https://github.com/chunxi-alpc/TRIAGE/blob/main/docs/ASTER/manuscript.zh.md)和[结果附件](https://github.com/chunxi-alpc/TRIAGE/tree/main/docs/ASTER/results)。历史开发或启动报告中的数据对应各自记录日期。

| 实验 | 原运行入口 |
|---|---|
| 原指南官方测试、全分母路径正确率、病例配对及来源簇区间 | [official.py](../official.py) |
| 患者事实翻转与局部兄弟交换的构造 | [matched_data.py](../matched_data.py) |
| 同输入信息、同监督的软提示训练与18节点配对分析 | [matched.py](../matched.py) |
| 六种方法全测试集的本地 BERTScore、路径对齐与汇总 | [bertscore_local.py](../bertscore_local.py) |
| 五次生成式评分、逐项复用与覆盖统计 | [api_judge.py](../api_judge.py)、[all_pathways.py](../all_pathways.py) |
| 概率上下文、盲表示分类器等方法的训练和评测 | [train.py](../train.py)、[comparators.py](../comparators.py) |

[weights/](weights/) 中保存第18层、第3轮的推理权重、层号和训练变体。普通低秩使用 ordinary，概率上下文使用 context，盲表示分类器使用 blind_readout。主方法使用 no_js。matched_soft 的检查点记录了与主方法相同的 no_js 目标，运行时由 matched.py 的 SoftContext 决定其在输入侧施加干预。零干预和固定概率融合无需学习这些参数。各文件可通过 torch.load(..., map_location="cpu", weights_only=True) 读取。

归档脚本沿用实验服务器的数据目录约定。独立环境运行冻结主方法的入口见 [main 的复现说明](https://github.com/chunxi-alpc/TRIAGE/blob/main/docs/ASTER/supplementary.zh.md)。
