# 服务器数据审计与评价工具

本目录代码由 Agent A/D 编写。提交时尚未执行；运行记录由协调器在115服务器追加。没有使用这些代码启动正式测试。

## 元数据审计

```bash
source /home/gcx/gcx261014/bin/activate
python audit_server_metadata.py --config /home/gcx/TRIAGE/configs/datasets.json --output server_data_audit.json
```

读取真实文件以核对计数、字段、唯一ID和SHA-256，结果不包含病例、指南、问题、选项、答案内容。README仅提取许可行，不复制示例。MEGA的跨划分文档哈希重合会显式报告；MedGUIDE缺原始文档来源字段时，跨数据集来源重合记unknown，不能据病种ID不同声称独立。

## 评价代码验证

```bash
python -m unittest -v test_evaluation.py
```

六项纯合成测试覆盖多选集合、缺失预测完整分母、重复key、整簇保留seed、Holm和不猜Pair。它们是评价器契约测试，不是模型实验。

## 正式 MedPIC 评价（冻结后）

预测JSONL每行：`sample_id`、`method`、`seed`、`prediction`。prediction使用选项字母列表或严格的字母列表字符串；长理由应由已冻结的模型输出解析器转换，评价器不会从理由中猜答案。错误、无效及缺失预测均保留在467题分母；缺失输出会阻止完成状态。

配置锁必须提供 `frozen: true`、`frozen_at`、`methods`、`seeds: [0,1,2]`、`primary_comparisons`。主要比较列表每项为 `method`、`reference`、`metric: "accuracy"`、`alternative`（`greater` / `less` / `two-sided`），先冻结再评价。真实簇文件是 sample_id→源指南/规则/患者簇ID字典，不允许把每题ID自动视作独立簇。可将同源组与真实患者链接取传递闭包；程序还会合并已验证Pair跨越的簇。

```bash
python evaluate_medpic.py --gold /path/to/frozen/questions.json --predictions /path/to/predictions.jsonl --config-lock /path/to/config_lock.json --clusters /path/to/source_patient_clusters.json --pairs /path/to/verified_pair_mapping.json --output /path/to/metrics.json
```

缺失官方89对映射时省略`--pairs`，Pair明确blocked，集合准确率仍可报告。缺真实簇映射时省略`--clusters`，保留点估计，区间/显著性blocked。不得自行将183题CF凑成89对。

`paired_statistics.py`也适用于MEGA逐例官方评分，输入metric如`pcc`必须为0到1比例，先固定方向（PCC改进为method-reference<0）。每个患者各seed先求平均，再以整个来源/患者簇重采样。输出各seed差异、整簇95%区间、配对簇符号置换检验及预登记比较族Holm校正；种子不充当独立样本。二元准确率另记录纠正/退化，分母为全部样本。少量来源簇的推断分辨率有限，应按实际簇数解释。

当前工具仅对完整总体准确率的预登记比较提供自动推断；GF/CF与Pair给出完整点统计。需要这些指标的确认检验时，评价代理应在第一次测试之前冻结对应比较族及真实簇定义，再调用通用配对统计函数，不得看到结果再添加有利检验。

注意：元数据审计结果的complete_verified仅表示该审计运行和计数通过，不表示许可、来源独立性或模型研究已经完成。正式模型实验的逐例原始输出、耗时、显存、模型/源码/投影指纹由推理运行清单负责；评价工具另外保存金标文件、预测、配置锁、簇映射、Pair映射与评价源码哈希。
