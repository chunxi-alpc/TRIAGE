# ASTER-CR 10,006 对固定训练数据

后续默认训练数据入口：[configs/aster_cr_training_default.json](../../../configs/aster_cr_training_default.json)。

- [固定数据说明](FROZEN_DATASET.zh.md)：A 4,368、P 635、C 635、F 4,368；标签保持与翻转各 5,003 对；F 全来自 MedEinst。
- [冻结配对 ID 清单](data/selected_pair_ids.jsonl.gz)：仅含 pair_id、case_id、dataset_id、kind，固定具体样本。
- [数据统计](dataset_statistics.csv) · [数据清单说明](dataset_manifest.json) · [原始训练配置](configs/initial_frozen.json)
- [均衡配置测试结果](TEST_RESULTS.zh.md) · [结果、置信区间与解释](RESULTS_NOTE.zh.md) · [机器可读摘要](results_note.json)

本次决定固定后续默认训练数据；E1 仍为冻结基线，已有结果尚未证明均衡配置整体准确率优于 E1。原 E-Multi 的 10,768 对配置只作历史对照。

配对原文、完整训练索引、逐条审核、模型、大缓存与训练复现程序保留在实验服务器：`/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_multisource_20261112/balanced_10006/`。读取 JSONL 压缩清单可使用 Python 标准库 `gzip.open(path, 'rt', encoding='utf-8')`。重建训练数据仍使用服务器已有 `code/build_balanced.py`；当前清单冻结后直接复用，无需重新抽样。
