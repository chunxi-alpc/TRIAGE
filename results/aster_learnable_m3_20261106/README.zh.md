# ASTER：仅改造 M3

本目录包含本轮全部配置、冻结基、训练权重、开发预测、测试预测及比较报告。完整结果见 [REPORT.zh.md](REPORT.zh.md)，机器可读结果见 [comparison.json](comparison.json)。

7,458 例原版 MEGA-CDP 测试病例，五种方法共 37,290 条完整路径预测。原模型 94.77%，原 ASTER 94.46%，新 M3＋辅助参考 94.82%，新 M3＋自身参考 94.77%，直接读取参考 86.42%。

新 M3 相对原 ASTER 提高 0.36 个百分点，按来源重采样的 95% 差值区间为 [+0.13, +0.61] 个百分点；相对原模型新增错误从 45 例降至 2 例。但相对原模型仅净增 4 例，辅助参考相对自身参考的优势尚不明确。

- `configuration.json`：固定实验设置、51 个头及各头秩。
- `frozen_m1_m2.pt`：既有头与 M2 基的冻结快照，无重新拟合。
- `reference/`、`self/`：三轮新 M3 权重、开发预测及选轮结果；均选第 1 轮。
- `data/*.jsonl.gz`：原版训练 200 例、开发 100 例、测试 7,458 例；压缩只改变存储格式，程序可直接读取。
- `test/predictions_*.jsonl`：全部测试预测，包括暂停恢复及两卡重新分配后产生的记录。报告按病例和方法汇总所有文件。
- `checks.json`：零编辑一致、仅 M3 获得梯度、51 个目标头和位置、预测不读取标签的验证结果。
- `RUN.zh.md`：运行与恢复记录；`pipeline_status.json` 和 `report_status.json` 为完成状态。

代码见 [experiments/aster_learnable_m3](../../experiments/aster_learnable_m3)。使用服务器环境 `/home/gcx/gcx261014` 和本地 Qwen3-8B；从仓库根目录运行 `python -m experiments.aster_learnable_m3.run --stage evaluate --root <结果目录> --shard 0 --shards 2`。另一张卡设置对应 `CUDA_VISIBLE_DEVICES` 并使用 `--shard 1`。已有预测会直接复用；数据读取支持原 JSONL 与仓库内的 gzip 文件，优先使用本目录冻结基快照。

旧版 89.21% 仅作为历史结果，不与本轮直接合并。运行日志、Python 缓存和锁文件留在服务器。
