# ASTER P1–P6 顺序预实验

基于 main `781ee863784c60f1f6710ba5eede22bc8ee6b93c`，独立分支 `exp/aster-p1-p6-20261027`。所有运行在 115 服务器、`/home/gcx/gcx261014` 环境，使用 GPU 0、1。已实现 P1–P5 的顺序执行入口。实际完成状态以 [总览](RUN_STATUS.zh.md) 和每阶段完成标记为准，不能据目录或脚本名称宣称六项完成。

阅读 [最新执行状态](RUN_STATUS.zh.md)、[P1 报告](REPORT.zh.md)、[事先固定协议](protocol.json)、[门槛状态](results/gate_status.json)。

## 数据

沿用已冻结的 MedGUIDE 开发树划分与独立 200 例开发文件，不重新划分测试集。运行脚本不打开测试病例或测试预测。四卷各 200 例；本轮可验证的 CS 为 17 例，目标规模未完成。病例和完整提示留在 `private/`，不发布来源文本。公开文件包含来源哈希、开发树标识、配对编号、标签、原始模型输出和汇总。

两条同父兄弟路径的末级条件不同，编号按固定随机顺序为 1/2。相同条件下的不同处置不作为不同可盲判分支。完整树推理与生成不在本轮 P1 的覆盖范围内。此前源路径标签并非本轮独立人工核验标签。

## 在同一服务器复现

从仓库根目录运行：

```bash
source /home/gcx/gcx261014/bin/activate
python experiments/aster_prelim_20261027/prepare.py
CUDA_VISIBLE_DEVICES='' python experiments/aster_prelim_20261027/check_data.py
bash experiments/aster_prelim_20261027/launch_p1.sh
```

本轮 `private/qwen` 或 `private/llama` 已存在时，runner 会拒绝覆盖。复跑请在新的实验目录或独立工作树操作；不要删除原始结果。源数据位置在 prepare.py 中固定，模型路径在 launch_p1.sh 中固定。模型权重逐文件 SHA256、软件版本、代码与数据哈希保存在模型 manifest。

`launch_p1.sh` 等待两个模型结束后只运行汇总，不自动启动后续阶段。`summarize.py` 仅收集本轮输出；缺失任一模型完成标记则 P1 为 INCOMPLETE，绝不当作 PASS。解析失败计错。完全相同的提示复用一次真实生成，原始逐例记录保留提示 SHA256，区分实际模型调用与逻辑记录数。

P1 门槛在推理前固定为两模型 SW 的“盲判准确率−明判准确率”百分点差的等权平均，低于 3 停止。分模型值始终单列。这个两模型汇总约定是本轮执行选择，不是用户原文给出的额外实测结果。

`check_data.py` 只检验数据配对与阈值构造，不是 P4 的注意力单元验收。CS 未达到目标数量，也没有人工核验 P6、执行 P3–P5 或调用生成路径评分器。
