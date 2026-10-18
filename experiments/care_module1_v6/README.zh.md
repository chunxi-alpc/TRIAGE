# CARE 模块一 V6

V6 保留 TrialMatchAI 的 T/F/U 主判定；确定性数值比较只在找到明确字段、主体、时间和单位时覆盖该原子状态，缺失或跨阈值矛盾保留模型判断并记录争议。BioLinkBERT-MedNLI 只定位支持/反驳证据和字符位置，不修改适用性标签；证据缺失或争议时关闭注意力干预。

- `REPORT.zh.md`：最终中文报告。
- `data/old_192_predictions_v6.jsonl`、`data/new_100_predictions_v6.jsonl`：逐例预测。
- `data/v6_metrics.json`：A/B/C 五项核心指标和干预统计。
- `data/v6_failure_samples.jsonl`：典型失败样本。
- `data/new_set_manifest.json`：新增测试集来源和固定哈希。
- `evaluate_v6.py`：离线复评脚本。
- `scripts_run_new_v6.py`、`run_4gpu.sh`：四卡新增集运行脚本。
