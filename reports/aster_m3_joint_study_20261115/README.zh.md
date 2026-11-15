# ASTER 第三模块联合监督实验交付

[中文报告](REPORT.zh.md)回答五个研究问题；[历史等价性审计](HISTORICAL_AUDIT.zh.md)说明复用范围与协议差异。新增代码位于 [experiments/aster_m3_joint_study](../../experiments/aster_m3_joint_study/README.zh.md)。

| 内容 | 位置 |
|---|---|
| 真实运行状态及代码提交 | `status.json` |
| 数据、损失、资源、随机种子和选择规则 | `configuration.json` |
| 原五折来源清单 | `folds.json` |
| 所有内部候选、模型及候选分数 | `selection/foldK/` |
| 每折S/J权重、中心、样本ID、权重、损失及约束残差 | `models/foldK/{S,J}.json` |
| 每种方法的两种截距与拟合ID | `models/foldK/*_calibration.json` |
| R0/R1/R2归档分数及完整动态路径复现 | `verification.json`、`models/foldK/reference_replay.json` |
| 原始分数、修正量和未校准分数 | `corrections.jsonl.gz` |
| 逐节点最终分数与预测 | `evaluation/predictions.jsonl` |
| 逐对A/P和C/F | `evaluation/pair_predictions.jsonl` |
| 自然和干扰完整路径 | `evaluation/path_predictions.jsonl` |
| 来源正确率、配对指标和四向转移 | `evaluation/source_metrics.csv`、`evaluation/source_transitions.csv` |
| 逐例四向转移 | `evaluation/transitions_by_example.jsonl` |
| 首错节点修复及前后分数 | `evaluation/natural_first_error.jsonl` |
| 配对差值及95%置信区间 | `evaluation/source_bootstrap.csv`、`evaluation/metrics.json` |
| 可导出结果图 | `artifacts/summary.png`、`artifacts/summary.pdf` |
| 合成测试结果 | `tests.json`、`tests.log` |

逐例JSONL可在交付时压缩为同名 `.jsonl.gz`；内容和记录数保持不变。五折投影缓存保留在服务器 `models/foldK/projected_cache.npz`，可从只读历史激活重新生成。

服务器工作目录：`/home/gcx/aster-m3-joint-20261115`。运行入口：

```bash
cd /home/gcx/aster-m3-joint-20261115
bash experiments/aster_m3_joint_study/run_server.sh
```

本轮不重跑激活修补、头选择、痕迹维度搜索、预算扫描或外部方法。历史80来源已使用过，新来源独立确认尚未开展。旧论文正文和历史实验资产保持只读。

连接便利设置：已在119服务器为 `loginuser` 生成专用SSH密钥，并把其公钥加入115服务器的 `gcx` 账号；`ssh -o BatchMode=yes gcx@10.29.128.115` 验证通过。堡垒机入口仍按其原认证方式登录；GateShell提供的命令中没有公钥登记接口。任何密码均不保存到本实验代码、配置或报告。
