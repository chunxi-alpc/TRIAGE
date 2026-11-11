# ASTER-CR 配对数据构建与复现

本轮已完成历史数据恢复、单轮独立审核、新候选生成及程序验证。[构建报告与两张图](../../reports/aster_cr_pair_dataset_20261111/README.md)。版本为 `20261111-sol-agents-v2`。本目录只发布程序、JSON Schema和流程提示，**不上传数据**。

|类别|历史复用|新增合格|全部合格|
|---|---:|---:|---:|
|A|4,368|0|4,368|
|P|9|626|635|
|C|40|636|676|
|F|18|648|666|

全部合格池6,345对，每类覆盖80个拟合来源。均衡集每类635对，共2,540对，按来源、语言和树复杂度分层，优先选择不同病例。F0的82对独立保留。2,150个生成任务形成1,962个有效候选，独立审核通过1,910对，失败22对、无法确定30对；188个任务弃权或未形成有效候选。

生成由GPT-6.1 Sol／high完成；审核由独立上下文的GPT-6.1 Sol／xhigh完成，每对只审核一轮。程序验证结构、来源、标签、修改范围、重复与来源隔离；可执行的C/F规则重新计算，无法自动判定的语义由独立审核者判断。没有临床专家参与。全程使用GPU数量为0、显存为0；未训练模型、未修改方法或论文结论。

## 数据与环境

所有数据保留在服务器 `gcx@10.29.128.115`：

- 冻结实验：`/home/gcx/aster_cr_pair_dataset_20261111`。
- 历史原始输入：`/home/gcx/aster_cr_repair_v2_run_20261108/data`。
- MEGA原文：`/home/gcx/TRIAGE/data/MEGA-CDP/train_val`。
- 历史报告仓库：`/home/gcx/aster-final-two-20261111`。
- Python环境：`/home/gcx/gcx261014`；确定性程序依赖见[requirements.txt](requirements.txt)。

完整配对JSONL、病例和指南原文、逐条审核记录、任务记录、来源划分清单、人工抽查清单及v1备份均不随Git发布。仅克隆仓库不能获得完整输入，需要访问服务器已有冻结记录。

## 确定性复现

在服务器进入本实验代码目录，指定一个尚不存在、位于Git仓库外的新输出目录：

```bash
source /home/gcx/gcx261014/bin/activate
python scripts/reproduce.py \
  --frozen-root /home/gcx/aster_cr_pair_dataset_20261111 \
  --output-root /home/gcx/aster_pair_reproduction_new
```

入口读取已冻结的 `historical_pairs.jsonl`、`generated_candidates.jsonl`、`review_results.jsonl` 和 `source_splits.json`，在新目录中依次运行发布版 `validate.py --stage all --finalize` 和 `final_dataset_report.py`。原实验输入保持不变，不产生新候选或新的AI审核，不调用模型、不训练、不做哈希检查。

成功结果应为全部合格6,345对、均衡集2,540对，每类635对。程序同时核对v2均衡集每类80个来源及已记录的病例覆盖。详细日志仅写新输出目录的 `logs/`，结果见 `reports/reproduction_result.json`。已有输出目录会被拒绝，另选新目录即可。

发布版本已在服务器隔离目录 `/home/gcx/aster_pair_reproduction_publish_20261111_v2` 完整复现：全池6,345对、均衡集2,540对；均衡集A/P/C/F病例覆盖分别为635/622/635/635，每类均覆盖80个来源。复现没有重新执行AI审核。

`scripts/runtime_paths.py` 统一读取环境变量：

|环境变量|作用|默认值|
|---|---|---|
|`ASTER_PAIR_ROOT`|工作和输出目录|上述冻结实验目录|
|`ASTER_PAIR_DATA`|历史输入目录|上述历史原始输入目录|
|`ASTER_PAIR_RAW`|MEGA原文根目录|上述MEGA原文目录|
|`ASTER_PAIR_REPO`|历史报告仓库|上述历史报告仓库|

复现入口自动将 `ASTER_PAIR_ROOT` 设为新输出目录；其他变量可在启动前设置。程序和Schema始终从本代码目录加载。迁移MEGA原文时，原记录中的文档路径由 `ASTER_PAIR_RAW` 解析；记录内容和原始ID不被重写。

## 正式构建流程

原实验的执行顺序为：恢复历史配对 → 三个独立Sol审核者完成历史单轮审核 → 按历史通过数量确定缺口 → 三窗口各三作者并行生成P/C/F → 每个作者完成后交给独立Sol审核者审核一次 → 合并 → 确定性验证并构建两版数据。A历史数量充分，没有新增。

- [restore_historical.py](scripts/restore_historical.py)：从原始输入恢复历史配对和来源划分。
- [agent_b_sol_io.py](scripts/agent_b_sol_io.py)、[agent_b_sol_shard_io.py](scripts/agent_b_sol_shard_io.py)、[parallel_review.py](scripts/parallel_review.py)：历史审核材料、决定记录与分片合并；[historical_review_io.py](scripts/historical_review_io.py)仅含纯数据IO函数。
- [agent_a_sol_prepare.py](scripts/agent_a_sol_prepare.py)：历史审核完成后，按类别缺口、来源、语言和树复杂度选择有限任务批次；本轮使用 `--multiplier 1.5`。
- [parallel_generation.py](scripts/parallel_generation.py)、[normalize_sol_candidates.py](scripts/normalize_sol_candidates.py)：分片归属、智能体实际编辑记录、原文定位和候选格式转换。
- [export_blind_packets.py](scripts/export_blind_packets.py)、[parallel_generated_review.py](scripts/parallel_generated_review.py)：向独立审核者提供原文与实际修改，隐藏生成推理，记录其独立标签和七项检查。
- [merge_generated_parallel.py](scripts/merge_generated_parallel.py)、[finish_when_ready.py](scripts/finish_when_ready.py)：全部分片完成后合并并调用确定性程序。
- [validate.py](scripts/validate.py)、[report_manifest.py](scripts/report_manifest.py)、[final_dataset_report.py](scripts/final_dataset_report.py)：验证、分层选样、统计、两张图、版本记录和结果报告。

生成和审核的语义判断由指定Codex智能体完成；这些Python程序只准备材料、保存判断和执行确定性检查，不包含模型API或服务器Codex CLI。实际流程提示见[生成说明](prompts/generation_worker.md)和[审核说明](prompts/generated_review_worker.md)。未来独立构建应将 `ASTER_PAIR_ROOT` 指向新目录，并提前建立 `reports/`、`logs/`；不要对冻结实验重新生成或重新审核。

Dataset A使用均衡子集；Dataset B保留全部合格数据，未来训练采用类别均衡采样或权重。二者尚未训练，后续比较应保持模型、特征维数、目标和评价数据一致。
