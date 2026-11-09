# ASTER-CR 痕迹验证

四组实验均已在服务器完成，全部使用真实 Qwen3-8B 缓存。完整五折新增真实单头干预；其余为真实缓存复算及新增回归/小网络训练，无合成或模拟数字。未修改旧实验结果或论文正文。

## 环境与输入

- 服务器：`gcx@10.29.128.115`；环境：`source /home/gcx/gcx261014/bin/activate`。
- 工作树：`/home/gcx/TRIAGE-worktrees/aster-cr-20261108`，分支：`aster-cr-20261108`。
- 真实输入缓存：`/home/gcx/aster_cr_repair_v2_run_20261108`；部分评分记录还引用 `/home/gcx/aster_cr_20261108` 特征文件，两处输入均需保留。
- 可重建临时缓存：`/home/gcx/aster_trace_validation_cache_20261109`，不提交大规模原始激活。
- 起始仓库提交、已有未提交文件及来源名单见 `data_audit.json`。本任务只提交当前目录，未纳入已有未提交改动。
- `provenance/` 保存核对时的 features/mechanisms/readout 程序快照，不作为另一个实验入口。

## 一键运行

从仓库根目录执行：

```bash
source /home/gcx/gcx261014/bin/activate
bash reports/aster_cancel_story_20261109/meeting/trace_validation/run_all.sh
# 包含完整五折重新选头：
bash reports/aster_cancel_story_20261109/meeting/trace_validation/run_all.sh --full
```

默认运行CPU部分。完整五折使用GPU 0/1；相同pair/head的真实干预会复用，完成折会跳过。首次GPU运行已事先向用户汇报资源：实测两卡峰值均约17.97 GiB，新增3,948次最终token前向；两worker处理各自折的总时间约199/154秒。日志写入本目录，不需持续监控。

再次运行会更新本目录的复算输出，既有输入和旧实验目录保持只读。需保留多次运行时，可先复制本目录；各独立脚本的参数和输出路径见子目录README。

## 评价约定

fit/dev/test各80/20/40来源，无交集。配对两侧始终属于同一来源和划分。覆盖与检查记录在 `data_audit.json`。

主预测拟合为fit A/P共4,887对，岭系数0.01、无配对回归截距，开发A共1,194对为主指标，AP共1,198对为补充。标准化和子空间只用fit；完整五折的选头、子空间和回归均只用该折训练来源。

50个随机种子为11–60，网络种子11/22/33，来源重采样2,000次。网络超参数只从fit内部来源划分选择。实验2直接预测痕迹差，实验4学习可部署的单节点修正，二者分别标注。

证据预算预先固定0.05/0.1/0.25，约束fit C/F每对的最大绝对修正差。相同训练预算不保证相同留出证据RMS，因此完整报告同时展示实际证据代价。dev及test均有历史使用，不称全新独立测试；未按测试结果选择配置。

## 交付文件

- `REPORT.zh.md`：按Q1–Q4组织的中文报告、汇总表、两句结论。
- `summary.csv`：四组实验关键结果。各实验子目录保存全部原始结果与汇总CSV。
- `reference_audit.json`：51头、16维、1,194对及13.91%/0.4063/0.0786的逐对复算与指标定义。
- `exp12/`：随机对照、来源自助重采样、维度和网络比较、模型与压缩逐对预测。
- `exp3/`：冻结选头的条件性诊断、完整五折 `full/`、每来源指标与误差解释。
- `exp4/`：全部预设预算修复，逐节点、逐对、自主动态路径结果、配置与模型。
- `exp12/prediction_comparison.png`、`exp4/repair_tradeoff.png`：所需两张图，也提供PDF。
- `build_report.py`：直接从实测结果重新生成报告和汇总表。

大表采用gzip压缩，保留完整预测。模型权重与原始大规模缓存不随本目录重复提交。

## 已执行的复跑检查

```bash
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp12/replay.py
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --replay --folds 0 1 2 3 4
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/verify_replay.py
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp4/replay_saved.py
```

保存参数重放全部通过。实验1/2代表性6模型逐对误差为0；完整五折重放选头相同、预测误差0、新增模型前向0；修复节点/配对/动态路径节点重放最大误差约7.1×10⁻¹⁵。对应记录分别为 `exp12/replay_check.json`、`exp3/replay_verification.json`、`exp4/replay_check.json`。
