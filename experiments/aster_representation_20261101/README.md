# 处置盲表示引导的分支决策

当前方法见 `docs/ASTER/faithful-reasoning-method-20261101.zh.md`；初始实验协议保留在 `docs/ASTER/representation-method-20261101.zh.md`。2026-11-02主方法改为既有`no_js_layer18/epoch3.pt`，原`full`为含JS消融，实验键不重命名。改选依据与确认集用途变化见 `docs/ASTER/main-method-amendment-20261102.zh.md`。

- `data.py`：读取既有病例，固定保存新训练／确认划分，直接复用开发集与年龄最小对；已有清单时不再准备数据。
- `intervention.py`：正交秩4子空间、处置盲映射、学习强度及行为训练目标。
- `engine.py`：`Engine.blind(case, gid)`一次保存三个候选层的表示与T/F/U诊断；`Engine.generate(public_case, module, layer)`只在当前分支查询的注意力输出编辑，临时KV不回写；统一合法路径、终点停止及处置填写。
- `train.py`：可求导的训练路径、逐轮开发选择、对照及配对评测；完整模型参数冻结。
- `orchestrate.py`：GPU0、1动态接续；先选层，再训练两项消融，冻结全部配置后执行新确认；完成条件自动跳过。

从服务器仓库工作目录运行，先激活 `/home/gcx/gcx261014/bin/activate`：

```bash
python -m experiments.aster_representation_20261101.data
python -m unittest experiments.aster_representation_20261101.test_intervention
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_representation_20261101.train --pilot
python -m experiments.aster_representation_20261101.orchestrate
```

上述是入口说明；已有数据清单、成功试跑和运行中的控制器不要重复启动。正式运行目录为 `/home/gcx/aster_representation_20261101`，启动信息在 `launch.json`，队列在 `queue_status.json`。同一轮只启动一个控制器。模型与既有数据路径沿用仓库原实验配置。

部署输入只包含病例编号、来源、患者文本、指南和问题。金标路径仅供训练和评分。旧43例、已暴露120例不进入本轮配置选择或独立确认。原指南、全部病例的终点处置交换、年龄条件翻转分别报告。

结果保存在逐例JSONL和同名summary文件；自动汇总写入 `REPORT.zh.md`，配对纠正／退化及来源等权区间写入 `paired_results.json`。年龄对按去重后的患者对计数，并保留原始最小对编号。所有历史结果仍保留，确认结果不触发新的搜索。

## 补充对照

`comparators.py`实现直接盲判查表和同参数量的盲表示分类器；`train.py --train context`在当前节点向问题加入T/F/U概率，使用相同普通低秩模块训练。它们只使用已选层，各自冻结后即可做新确认，不等待其他方法。调度器重启可接管仍在运行的原工作进程，不重跑已完成条件。补充安排见`docs/ASTER/representation-comparison-extension-20261101.zh.md`。

## 当前主方法与官方测试

主方法保留处置盲参考、当前决策位置的秩4干预，以及原指南/交换指南的两视图交叉熵；损失为CE＋0.1保持项＋0.001幅度项，无JS。第18层第3轮权重已存在，本次不重新训练。官方测试说明见 `README.official.zh.md`，优先运行主方法并复用已保存预测。现有普通低秩和上下文对照仍含JS，解释收益时须说明损失差异。
