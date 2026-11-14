# Work 执行交接：本轮已完成

2026-10-19 已从已连接电脑访问原服务器，完成本分支全部既定真实模型实验。工作分支仍为 `codex/care-reliability-20261019-work`，PR #9 保持草稿。不要把下方保留的执行前状态当作当前状态。

- 模块二：旧 384 例、新 144 例均完成三折；最终目录 `/home/gcx/care_reliability_runs/module2_work_02`。
- 模块一与联合：192 例六卡结构生成及 Qwen 四来源三方式均完成；最终目录 `/home/gcx/care_reliability_runs/joint_formal_work_02`。控制器结束时间为北京时间 16:17:46，总耗时 72.4 分钟。
- 所有原始日志和失败尝试已保留并压缩提交，见 [归档说明](results/work_20261019/README.zh.md)。
- [给老师的精炼汇报](results/work_20261019/TEACHER_REPORT.zh.md)、[当前结果](README.md)、[执行状态](execution_status.json)。

当前需关注的后续问题是 Not Violated 的保守标签映射、生成截断、精确目标文本拒绝、证据冲突门控及四头全部一致限制；这些没有在本轮冻结运行中事后改动。后续优先复用原始输出，合理设置对照，不重复开展无必要检查，不按测试收益删病例。用户最新要求少做未规定的审核，集中完成明确任务。

---

## 以下是执行前的交接记录（历史）

## 目标与现状

用户已要求多个代理执行审查后提出的实验。三路代理已完成模块一修复、模块二固定对照、联合评估的代码与 CPU 验证。当前工作分支为 `codex/care-reliability-20261019-work`，基准提交为 `7557d3150214a3894f2d340df6d9964de75efce1`。

**真实 GPU 实验尚未开始。** 当前云任务没有本地电脑的执行入口；云端 SSH 连接失败不能据此判断原服务器故障。后续需从桌面端的 Work 任务使用已连接电脑访问原服务器。新任务应先读取本文件和 [当前结果](README.md)，接续现有工作。

## 先核对执行环境

仓库记录的服务器、仓库及 Python 环境为：

```text
gcx@10.29.128.115
/home/gcx/TRIAGE
/home/gcx/gcx261014
```

这些是历史记录，须在 Work 中现场核对。保留现有工作区改动，使用新 worktree。若下列新路径已经存在，检查其来源后另选新路径，不删除或覆盖。

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
git -C /home/gcx/TRIAGE status --short
git -C /home/gcx/TRIAGE fetch origin codex/care-reliability-20261019-work
git -C /home/gcx/TRIAGE worktree add \
  /home/gcx/TRIAGE-worktrees/care-reliability-20261019 \
  origin/codex/care-reliability-20261019-work
cd /home/gcx/TRIAGE-worktrees/care-reliability-20261019
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

仓库允许的物理 GPU 是 `0、1、2、3、6、7`。实际分配需依据空闲状态。启动前向用户报告卡号、显存预算、样本数与预计时长；用户已授权实验，无需再次询问是否开始这项工作。

先执行统一准备检查。Work 环境里的两个 PyTorch 检查应实际运行，不能仍以跳过视为模型适配器已通过。

```bash
python experiments/care_reliability_20261019/run_cpu_checks.py \
  --output /home/gcx/care_reliability_runs/preparation_work_01
```

模块二与联合后端按原 4B 记录核对权重、影响注意力几何的配置和软件环境。遇到不一致应先定位原环境或资产，不自动安装一套新版本，也不更换模型。

## 三路分工与资源计划

| 代理 | 任务 | 资源安排依据 |
| --- | --- | --- |
| A：模块一与联合运行 | 同批运行 V6/V7 结构及四种来源的联合矩阵；保存原始结构供模块一评价 | Qwen3-4B 与量化 Phi-4／NLI 可放两个逻辑设备；先按每进程至少 24 GiB 余量规划，再用实际峰值修正 |
| B：模块二 | 三折分别运行旧 384 回归与新机制集，固定 11 条件及基线 | Qwen3-4B，单卡 BF16，逐例；可用独立卡执行不重叠折 |
| C：结果复核 | 核验日志、数据与模型摘要、零干预检查、失败与无操作；分数据分析并整理汇报 | CPU；需要专业人员的临床证据复核不能由自动参考程序冒充 |

显存是启动规划，不是本次实测。运行时长尚未测得，先在既定训练样例上测吞吐，再据全部任务规模报告预计时长。不得用测试收益选择保留哪些病例。

模块一独立运行器支持额外输入和分片。但联合运行器本身已经保存 V6/V7 原始图，默认先利用这批同输入结果，避免在独立运行器和联合运行器中重复生成同一批输出。新确认划分每个版本涉及 192 例、576 条规则和 2352 次原子条件调用（未去重的代码预算），两个版本合计 4704 次；这不是已发生的模型调用数。

## 代理 B：模块二实际命令

原 4B 模型路径记录如下；运行器还会核对其实际文件摘要。用卡号由协调者根据现场检查设置，`cuda:0` 指该进程可见的第一张卡。

```bash
M2_MODEL=/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base
M2_RUN_ROOT=/home/gcx/care_reliability_runs/module2_work_01
```

对每折 `0、1、2` 执行下面两种模式，分别写入对应的 `fold0`、`fold1`、`fold2` 新目录。以下展示第 0 折：

```bash
python experiments/care_reliability_20261019/module2/run.py \
  --mode legacy384_regression --fold 0 --model "$M2_MODEL" \
  --output "$M2_RUN_ROOT/legacy384_regression/fold0"

python experiments/care_reliability_20261019/module2/run.py \
  --mode sealed_mechanism --fold 0 --model "$M2_MODEL" \
  --data experiments/care_reliability_20261019/data/module2_sealed.jsonl \
  --data-manifest experiments/care_reliability_20261019/data/manifest.json \
  --output "$M2_RUN_ROOT/sealed_mechanism/fold0"
```

完成三折后，两种数据分别分析：

```bash
python experiments/care_reliability_20261019/module2/analyze.py \
  --run-root "$M2_RUN_ROOT/legacy384_regression" \
  --output "$M2_RUN_ROOT/legacy_analysis"
python experiments/care_reliability_20261019/module2/analyze.py \
  --run-root "$M2_RUN_ROOT/sealed_mechanism" \
  --output "$M2_RUN_ROOT/sealed_analysis"
```

逐例零强度检查失败、算子复现失败、数值预算超限或模型身份不符时，保存日志并停止该正式比较。分析器要求完整真实日志，不接受 CPU 预检或部分折结果充当完成。

## 代理 A：模块一与联合实际命令

先核对并复用三个模块一模型的本地路径：`CARE_MODULE1_V6_BASE_MODEL`、`CARE_MODULE1_V6_ADAPTER_MODEL`、`CARE_MODULE1_V6_EVIDENCE_MODEL`。下列命令会在变量未设置时停止，避免静默使用远程模型标识。两个逻辑 GPU 的物理映射由协调者设置并记录。

```bash
python experiments/care_reliability_20261019/joint/run_joint.py \
  --model /home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base \
  --module1-base-model "${CARE_MODULE1_V6_BASE_MODEL:?Set the verified local Phi-4 directory}" \
  --module1-adapter-model "${CARE_MODULE1_V6_ADAPTER_MODEL:?Set the verified local TrialMatchAI adapter directory}" \
  --module1-evidence-model "${CARE_MODULE1_V6_EVIDENCE_MODEL:?Set the verified local BioLinkBERT directory}" \
  --learned-assets experiments/care_reliability_20261019/module2/frozen_learned_assets.json \
  --device cuda:0 --module1-device cuda:1 \
  --split confirmation --sources reference reference_positive v6 v7 \
  --include-reference-diagnostic \
  --output /home/gcx/care_reliability_runs/joint_work_01
```

如果只安排一个逻辑 GPU，先据实际显存核对两套模型能否共存，再修改设备参数。所有来源的最终答案由同一个 Qwen 实例生成；V6/V7 共享冻结的 TrialMatchAI 和 NLI 模型实例。

每个来源的三种干预方式使用相同完整提示。参考结构两组是诊断，不能称为预测模块效果；V6/V7 未开放的负向门控不会被运行器补成参考门控。参考状态、结构化事实和答案只进入显式参考诊断及评分，不进入预测器。

## 结果交付与后续判断

保留原始逐例日志、解析和技术失败、实际零干预检查、模型与输入摘要、数值预算和有效覆盖。分开报告旧数据回归、新合成机制测试、参考结构诊断和 V6/V7 预测结构效果。所有变体按家族聚类，主要分析使用 24 个标准表述配对。

本次新数据措辞曾用于解析器开发修复，不能称为对全部开发盲法的独立确证。真实临床泛化仍需新的家族隔离数据和实际专业人员证据复核。证据预算、KV 缓存、自由生成与更广模型泛化也尚未由本分支证明。

运行后把可复核的新结果提交到该实验分支，再根据实际效应与错误类型整理给老师的精炼报告。已有代码、旧预测与旧结果不覆盖；未完成项须保留明确状态。
