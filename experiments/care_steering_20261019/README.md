# CARE 方向干预最小实验（2026-10-19）

本目录保存 Qwen3-8B 的服务器实测、输入数据、逐例输出、环境与复现脚本。

- [完整报告（文本）](REPORT.txt)
- [完整报告（HTML）](REPORT.html)
- [复现说明](REPRODUCE.txt)
- [可下载复现包](exports/CARE_steering_reproduction_20261019.tar.gz)
- [结果图](results/test_accuracy.png)
- [原始自由生成](results/free_completions.jsonl)
- [最终答案统计](results/free_format_normalized_analysis.md)

160 条合成病例的自由生成结果（兼容常见答案格式的探索性分析）：

| 方法 | 准确率 |
| --- | ---: |
| A 无干预 | 89.375% |
| B 固定来源偏好方向 | 86.875% |
| C 简化适用性门控 | 90.000% |

C−B 为 +3.125 个百分点，家族 bootstrap 95% CI 为 [−1.875, 8.125]，探索性配对家族置换 p=0.388。设备路由迁移的 A/B/C 准确率为 88.75%/90.00%/88.75%。

现有完整 V6 模块一另测 20 例，全部因证据冲突关闭干预；自由生成与 A 相同，18/20。D 键空间适配的开发集选择零干预。原始 CARE 上传文件未找到，原版 D 尚未复现。

原始生成、首次严格解析、首次约束选项评分均保留；最终答案格式修复没有重跑模型或读取 gold 来补答案。具体方法与完成范围见报告。

论文和官方代码的核验依据见 [方法审查](sources/paper_method_audit.md)。模型权重与依赖二进制不包含在本目录，实际文件摘要和版本均已记录。

后续实际医疗知识冲突实验见 [ConflictMedQA 实测](../care_conflictmed_20261019/README.md)。该轮独立保存闭卷知识检查、错误上下文、方向校准和患者条件配对，不覆盖本目录结果。
