# TRIAGE · ASTER

ASTER 研究大语言模型能否根据患者事实逐步执行临床指南。主方法保留完整指南，同时让同一个冻结模型单独读取患者和当前条件。在每次选择分支之前，一个可学习的四维子空间把这份处置盲参考引入当前决策。训练要求原指南和重排终点处置后的指南都选对分支。

冻结版本为 Qwen3-8B 第 18 层、第 3 轮的 `no_js`。在 MEGA-CDP 官方单轮测试上，完整路径正确率为 **89.21%（6,653／7,458）**。

- [论文正文与结果](docs/ASTER/manuscript.zh.md)
- [方法补充与复现步骤](docs/ASTER/supplementary.zh.md)
- [主方法代码与冻结干预权重](experiments/aster_representation/)

从仓库根目录查看数据准备、训练、预测和评分命令：

```bash
python -m experiments.aster_representation --help
```

旧版规则键编辑研究见[历史文档](docs/ASTER/archive/README.md)。
