# ASTER-CR：repair-v2 后续实验

> 本页保留repair-v2原阶段的运行配置。后续正式测试、优化、五模型和迁移的最新状态及归档见[实验总表](../../docs/ASTER/experiments.zh.md)，不沿用本页旧进度。

使用已完成的 repair-v2 数据：2,782 个候选节点、11,512 个输入、8,730 对变体。当前使用 GPU 0、1、5、6；MedGemma 主流程接续使用 GPU 6。服务器环境为 `/home/gcx/gcx261014`。

[结果与分析](../../reports/aster_cr_repair_v2_20261108/ANALYSIS.zh.md) · [结果快照](../../results/aster_cr_repair_v2_20261108/README.md) · [数据修复方法](repair/README.zh.md) · [独立普通读出](../../results/ordinary_readout_20261108/REPORT.zh.md)

## 数据与原始分母

来源 fit/dev/test 为 80/20/40；候选节点为 1,623/398/761，原始节点仍为 761/199/372；原始病例路径为 1,712/420/806。P/C/F 配对数、独立节点数、来源数分别报告，不因总配对增加宣称证据充分。

当前规则只读取当前条件与必要定义。事实以病例原句定位；C 在原条件对应位置修改，F 按固定网格修改所有对应事实并核对祖先，F0 保留同标签对照。A 独立构造，P 修改真实且无关的背景字段；抽取失败和原标签不一致归入抽取复核。

`data/views.jsonl` 的 `original` 标志区分原始节点与新增候选；`F`、`F0` 均接入读出、训练对照和评价。原始节点和路径准确率的分母保持不变。外部 NLI4CT、MedPIC 独立评价，不进入 MEGA 训练、选参或宏平均。

## 执行与恢复

repair-v2 数据接口和缓存导入 → Qwen fit/dev → P0、定位、H2/H3 → 读出与消融 → 四类训练对照各三个学习率 → dev 配置冻结 → test → MedGemma 关键确认 → 外部数据原生任务评价。

运行目录：`/home/gcx/aster_cr_repair_v2_run_20261108`。原始构造数据在 `/home/gcx/aster_cr_repair_20261108`，此前未修复结果在 `/home/gcx/aster_cr_20261108`。

```bash
cd /home/gcx
source gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-cr-20261108
export ASTER_CR_ROOT=/home/gcx/aster_cr_repair_v2_run_20261108
export ASTER_CR_GPUS=0,1,5,6
python -m experiments.aster_cr.resume_repair --resume-training
```

上面的恢复入口对应已完成开发读出、进入训练对照的本次运行。调度器已经在后台运行，不同时启动第二个。初次接通已有 repair-v2 时，使用独立输出目录并省略 `--resume-training`；不会再次抽取数据。旧 `prepare/annotate/build` 入口用于最初数据方案，本轮继续实验使用 `resume_repair`。

四卡扩容保留了 GPU 0、1 原训练进程，调度器接管它们，并在 5、6 启动其余对照。已完成阶段的两分片输入保留；后续尚未启动的评分使用四分片。恢复训练只跳过本轮已完成的准备和读出阶段，不导入旧实验的定位、模型或选参完成标记。

## 方法和评价

分数采用 BF16 基础模型、未编辑前缀缓存、最终 token 前向和 FP32 标签投影。仅在输入、模型及评分口径一致时复用旧 Full 分数、激活及路径缓存；新输入与依赖它们的定位、训练、选参重新计算。

选头、中心化、标准化和支持空间只用 fit。H2 各候选秩分别检查开发 R² > 0、信号扰动比 ≤ 0.1；H3 比较错误子集探针与打乱标签对照，分别报告开发与历史错误数量。不满足诊断的主方法分量关闭，探索性分量单列。

读出先检查开发路径下降不超过 0.5 个百分点，再按 A/P/C/F 配对宏平均和参数量选择。F0 单列；缺失预测按错误计，配对需要两端同时正确。消融在预选配置的秩、正则和权重下重拟合。最终配置冻结后才做本轮 test。

训练对照为 LoReFT、LoFiT、LoRA、最后一层残差偏移，学习率 0.0001/0.001/0.01，每阶段一轮；LoFiT 的选头与偏移训练分别计成本。MedGemma 只确认 Qwen 选定的配方及训练对照，不运行完整网格。

按指南来源做 2,000 次 bootstrap，区间与独立节点数同时报告。开发阈值通过不等于已经建立统计非劣性；路径差值区间下界达到 −0.5 个百分点才报告非劣性。

NLI4CT 直接生成 Entailment/Contradiction；MedPIC 输出多选字母数组，跨选项重排按文字对应。两者为基础模型原生任务评价，保持发布标签；缺失或无效输出计错。

## 结果与同步

`status/pipeline.json` 是服务器运行状态，`qwen/`、`medgemma/` 保存阶段结果，`frozen.json` 记录测试前选定配置。GitHub 的 `snapshot.json` 标明同步时点；未完成阶段不是最终成绩。

模型权重、激活缓存、完整输入、第三方代码和日志按仓库惯例保留在服务器。GitHub 同步实现、配置、结果摘要、压缩逐例明细和分析。

本次开发阶段的 CPU 汇总入口为 `python -m experiments.aster_cr.export_results --destination /path/to/repository`，不会启动模型、调参或自动推送。最终 test 结果应连同最终分析另行更新。
