# ASTER 未筛选方向：全局均值与同事实参考

[原运行报告](REPORT.zh.md) · [固定配置](configuration.json) · [总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md)

固定 Qwen3-8B、种子 20261105、51 个头、244 个未筛选方向、原数据划分和强度 1；两种新编辑使用相同的正交投影。参考激活由无干预前向计算一次并冻结。覆盖主实验 1290 输入、569 历史输入；本轮未运行自主路径。

主测试实际改参考的 406 输入中，全局均值新增错误 51 个，同事实参考 6 个；历史实际改参考的 515 输入对应 90 个和 6 个。同事实参考与直接读取参考在全部 610 个主测试输入上完全同判，隐藏层编辑的额外收益尚未得到支持。

| 产物 | 内容 |
| --- | --- |
| [主对照表](results/main_comparison.csv) | fit/dev/test/all，原分母保留 |
| [逐输入分数](results/main_inputs.csv) | 各方法、标签、参考是否改变 |
| [参考覆盖](results/reference_coverage.csv) | 主实验 954/1290、历史 515/569 改参考 |
| [配对参考覆盖](results/pair_reference_coverage.csv) | A 232/396、P 231/231 获得共同参考 |
| [配对分数与共同偏移](results/main_pair_shifts.csv) | 两侧分数、差分变化、共同偏移 |
| [分组偏移](results/pair_shift_summary.csv) | 类别、范围与分割 |
| [C/F 逐例错误](results/main_CF_errors.csv) | 所有事实类错误输入 |
| [保护分组](results/protection_groups.csv) | 实际编辑范围与未改变参考单列 |
| [历史逐输入分数](results/historical_inputs.csv) | 保留各路线原模板 |
| [历史阈值分组](results/historical_numeric_strata.csv) | below/above/same_side/恰等阈值 |
| [删除与补回](results/historical_deletion_restore.csv) | 原文完整匹配才删除保存片段 |
| [补回效果](results/historical_restoration_effects.csv) | 各定义路线分别比较 |
| [全部逐例预测翻转](results/all_prediction_flips.csv) | 不仅保留改善，也保留新增错误 |
| [检查汇总](results/verification_summary.json) | 同投影、范数、位置、零干预、恒等与模板 |
| [冻结参考提示](references/prompts.jsonl.gz) | 871 个去重参考及模板 |
| [冻结目标头激活](references/target_head_activations.pt.gz) | 各参考的分数、token 信息与选中头激活 |
| [运行资源](runtime.json) | GPU 0/1，实际约 10.42 分钟，峰值保留显存每卡约 18.23 GiB |

输入到参考的映射在 inputs/ 中；分数在 scores/ 和 references/scores_*.jsonl；算子在 operators/。大 JSONL 无损 gzip，CSV 保持可读。[publication.json](publication.json) 记录原路径到发布路径的映射；871 个缓存已合并为一个压缩文件。完整服务器运行保留在 /home/gcx/aster_reference_anchor_20261106。
