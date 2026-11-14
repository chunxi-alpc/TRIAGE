# Codex 03：新增外部方法实验与复现

本目录完成 Qwen3-8B 上的 RePS、PSR、CAA、INLP 和 DoRA 实验，共 11 次正式方法运行：RePS、PSR、DoRA 各使用种子 11、22、33，CAA 和 INLP 使用确定性配置。训练数据为 E1 / Dataset A 的 2,540 对，评价采用第 4.5 节受控 MEGA-CDP 划分。方法实现、训练配置、开发集选择、冻结配置和实际预测均有记录。

当前结果见 [报告目录](../../reports/aster_codex03_external_20261112/) 的 `status.json`、`section4_5_new_methods.csv`（11 行正式运行）和 `section4_5_method_summary.csv`（5 种方法汇总）。论文正文保持原状；E5 状态记录在 [experiments.zh.md](../../docs/ASTER/experiments.zh.md)。

## 交付内容与评价范围

GitHub 交付包括本目录源码、官方来源记录、审计副本、所有正式训练及冻结配置、统计 CSV/JSON，以及压缩后的逐例、逐对和路径预测。报告中 `evaluation/test/<method>/` 覆盖全部 11 次正式运行；`evaluation/dev/<method>/` 包含 CAA、INLP 和 DoRA 三个种子的 5 次完整评价。RePS、PSR 的开发集强度选择记录保存在各自的 `dev_selection_scores.jsonl` 和 `frozen.json`，其完整动态路径评价采用测试集。

逐例文件以 `.jsonl.gz` 发布，可用 `gzip.open(path, "rt", encoding="utf-8")` 逐行读取。服务器保留原始 JSONL。每个正式评价目录包含 `statistics.json`、`statistics.csv`、`source_metrics.csv`、`path_metrics.csv`、`section4_5.csv` 和 `training_and_implementation.json`。统计报告包括 A/P RMS、VR、C/F ER、C/F 修正 RMS、C/F 双端全对率、自然及干扰路径正确率、两类路径各自的纠正与新增错误、参数量以及训练和推理成本。

| 划分 | 来源数 | A/P 对 | C/F 对 | 自然路径 | 干扰路径 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fit | 80 | 1,270 | 1,270 | — | — |
| dev | 20 | 1,198 | 60 | 420 | 1,260 |
| test | 40 | 2,293 | 40 | 806 | 2,418 |

E1 的 A/P/C/F 各 635 对，5,080 条端点记录对应 4,637 个唯一输入。F0 诊断单独保存。这里的第 4.5 节受控测试集与官方 7,458 例总体评价各有独立分母。

所有方法沿用共同 chat template、关闭 thinking、FP32 是/否输出投影和 `ell > 0` 决策。拟合使用 fit，参数选择使用 dev，测试前冻结配置。RMS 单位为 logit，准确率以比例存储。VR 和 ER 按公共评价器定义保存。推理成本记录全部配对输入及实际动态路径访问的唯一输入评分；逐病例在线延迟另见效率实验。

## 方法与配置位置

以下路径均相对于报告目录 `reports/aster_codex03_external_20261112/`。

| 方法 | 保留的目标和任务适配 | 正式配置与模型目录 |
| --- | --- | --- |
| RePS-E1-task-adapted | 官方 `scaled_simpo` 双向偏好目标；E1 正确/错误答案作为偏好响应 | `reps/seed{11,22,33}/`：`training_config.json`、`training.json`、`frozen.json`、`model.pt` |
| PSR-MSE-E1-task-adapted | 官方 Focused 模块和对齐激活 MSE；使用 E1 教师提示与成功教师答案 | `psr/seed{11,22,33}/`：同上；教师配置在 `psr/teacher_fit/` |
| CAA-E1-task-adapted | 官方行为答案激活差均值及加向量函数；对同一 E1 提示比较正确/错误答案 | `caa/`：`configuration.json`、`implementation.json`、`training.json`、`frozen.json`、`vector_layer18.pt` |
| INLP-task-adapted | 官方迭代线性分类器及零空间投影；概念为“指南处置文本是否保留” | `inlp/`：`training_config.json`、`selection.json`、候选投影及概念/证据探针结果 |
| DoRA-E1-SFT | PEFT DoRA 的幅度/方向参数化；E1 单个正确答案 token 的全词表交叉熵 | `dora/full/seed{11,22,33}/`：`configuration.json`、`training_and_implementation.json`、`frozen_configuration.json`、`adapter/` |

实现差异和监督定义以各方法配置及评价目录的 `training_and_implementation.json` 为准。CAA 的原始提示末 token 表示同时提供给 INLP 拟合；方法参数独立训练。INLP 的概念分类器和证据探针结果位于 `probes_dev.json`、`probes_test.json`。

## 运行环境与仓库依赖

所有执行在服务器进行：

```bash
ssh gcx@10.29.128.115
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/aster-codex03-external-20261112
```

实际环境为 Python 3.12.3、torch 2.8.0、transformers 4.57.6、peft 0.21.2、numpy 2.5.4、scipy 1.18.1、scikit-learn 1.9.1。共同入口还使用 accelerate 1.15.0、safetensors 0.8.0、einops 0.8.2、pyarrow 26.0.0、tqdm 4.70.1 和 threadpoolctl 3.7.0。RePS 的 pyvene 0.1.8 安装在独立目录 `/home/gcx/aster_codex03_external_assets_20261112/python`，代码会将该目录加入导入路径。

本目录导入的公共源码均已包含在发布时的 `origin/main`：`aster_cr`、`aster_cr_retrain_paired_v2_20261111`、`aster_closed_form`、`m1_diagnostics`，以及其间接依赖 `aster_representation`、`gemma3_head_pilot`。从完整仓库运行即可，无需另行复制旧 worktree 的 Python 模块。其中 `common.py::InputTable` 使用 Linux 的 `os.pread`，调度和汇总程序也采用 Linux 进程/文件锁接口。

## 官方源码版本与获取入口

官方源码根目录为 `/home/gcx/aster_codex03_external_assets_20261112/official`。获取对应仓库后检出下表 commit，放入指定子目录；许可证与源码一同保留。这里列出实际调用的官方源码，完整获取记录见报告中的 `official_sources.json`。

| 方法 | 官方仓库 | 固定 commit / 实际实现版本 | 本机子目录与所需入口 |
| --- | --- | --- | --- |
| RePS | [stanfordnlp/axbench](https://github.com/stanfordnlp/axbench) | `41c8332543e5a631f9a8c0a9df38799893ace758` | `axbench/`；`axbench/models/interventions.py`、`preference_model.py::preference_loss` 及官方参考配置 |
| PSR | [Nokia-Bell-Labs/steer-like-the-llm](https://github.com/Nokia-Bell-Labs/steer-like-the-llm) | `3d916c618d146c5d657f055e432a432b0fa493c6` | `steer-like-the-llm/`；完整 `src/ase` 导入依赖 |
| CAA | [nrimsky/CAA](https://github.com/nrimsky/CAA) | `5dabbbd9a0bca5f25e174501e959de378806aa48` | `CAA-official-snapshot/`；`generate_vectors.py`、`utils/helpers.py`，以及写有左列 commit 的 `SOURCE_COMMIT` |
| INLP | [shauli-ravfogel/nullspace_projection](https://github.com/shauli-ravfogel/nullspace_projection) | `e1edcc19d808108ab71cbb3afb0389db0206a7eb` | `nullspace_projection/`；`src/debias.py`、`src/classifier.py`，保留 Git commit 信息 |
| DoRA | [NVlabs/DoRA](https://github.com/NVlabs/DoRA)；[huggingface/peft](https://github.com/huggingface/peft) | 作者参考 `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18`；实际执行 PEFT `0.21.2` | `DoRA-reference/` 保存作者说明；`peft-0.21.2-snapshot/` 保存实际安装源码，执行入口为 `LoraConfig(use_dora=True)` |

CAA 保留固定版本的核心源码快照；其历史向量、原 benchmark 数据和分析笔记本未用于本实验。RePS 直接加载官方干预类并提取原损失函数，CAA 直接提取原函数，INLP 加载作者两个源码文件，PSR 将官方 `src` 加入导入路径。具体加载过程见对应 Python 文件。实际运行采用上述环境和导入方式；上游完整 benchmark 环境中的其他模型、评测服务和训练包不属于本实验入口。

需要重建 RePS 独立依赖时，可使用：

```bash
python -m pip install --no-deps --target /home/gcx/aster_codex03_external_assets_20261112/python pyvene==0.1.8
```

## 服务器保留资产与路径配置

原始输入、预处理数据、模型权重及激活保留在服务器。GitHub 中的统计与压缩预测可直接阅读；从模型重新评分还需以下资产。

| 资产 | 服务器位置与用途 |
| --- | --- |
| Qwen3-8B | `/home/gcx/TRIAGE/models/modelscope/Qwen3-8B`；权重分片、配置、tokenizer 及 chat template。既有本机快照未记录不可变上游 revision，配置如实保留该项为空。 |
| E1 全标注及来源划分 | `/home/gcx/aster_cr_pair_dataset_20261111/balanced_train.jsonl`、`source_splits.json`；版本 `20261111-sol-agents-v2` |
| 共同输入与 raw 分数 | `/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111/` 下的 `data/inputs.jsonl`、`runtime/inputs.sqlite`、`runtime/input_index.json`、`data/reused_scores.jsonl` 和 `cache/scores_*.jsonl` |
| 数据准备元信息 | 上一目录 `data/` 下的 `E1_pairs.jsonl`、`E1_views.jsonl`、`historical_pairs.jsonl`、`historical_views.jsonl`、`path_features.jsonl`、`path_interference_views.jsonl`；路径病例来自 `/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl` |
| 已准备的本轮数据 | 本轮报告 `prepared/{fit,dev,test}/`、`prepared/E1_pairs_full.jsonl`、`prepared/source_splits.json`；输入字符串由既有 UID 索引读取 |
| 方法权重与中间表示 | 上表各方法目录中的适配器、向量和投影；`caa/raw_prompt_residuals/`、`psr/teacher_fit/`、`inlp/` 的表示与探针文件 |
| 官方代码与独立依赖 | `/home/gcx/aster_codex03_external_assets_20261112/` |

将程序迁移到其他服务器时，统一调整 `data.py` 的 `RAW_SOURCE_OUT/FROZEN_DATA/HISTORICAL_DATA`，各方法的 `ASSETS/OFFICIAL/MODEL/MODEL_PATH`，以及 `experiments/m1_diagnostics/common.py` 的 `MODELS["qwen"]`。这些路径当前为源码常量。报告输出通常通过 `--output` 指定；输入数据保持既有 UID、文本及划分。若迁移输入索引，应按原公共模块的 `prepare_runtime_index.py` 重建字节偏移索引，或使用其原有 JSONL 读取方式。

## 执行入口

数据准备和已有分数统计使用 CPU：

```bash
python -m experiments.aster_codex03_external_20261112.data

python -m experiments.aster_codex03_external_20261112.evaluate \
  --method DoRA-E1-SFT-seed11 --split test \
  --scores reports/aster_codex03_external_20261112/dora/full/seed11/test/scores.jsonl \
  --output reports/aster_codex03_external_20261112
```

评价器读取评分旁的 `*.metadata.json`，也支持 `--metadata`。从发布的压缩 JSONL 复算时先解压到工作目录，再给评价器传入原始 JSONL 路径。共同数据和 raw 分数用于完整分母、对照纠正及新增错误统计。新方法评分先覆盖全部配对输入，再按自身预测遍历每个病例的自然路径及三种干扰路径；未访问的树节点无需评分。

下表为完整模型复跑入口，运行前按服务器约定汇报 GPU、显存、规模和预计时长，并通过 `CUDA_VISIBLE_DEVICES` 使用已分配的单卡。已有队列及状态文件记录本轮执行历史。

| 方法 | 模块与阶段 |
| --- | --- |
| DoRA | `dora run-seed --seed 11 --output reports/aster_codex03_external_20261112/dora/full`；同样运行 22、33 |
| RePS | `reps all --seed 11 --output reports/aster_codex03_external_20261112/reps/seed11`；同样运行 22、33 |
| PSR | `psr teacher` 生成 fit 教师缓存，再 `psr all --seed 11 --output reports/aster_codex03_external_20261112/psr/seed11`；同样运行 22、33 |
| CAA | `caa --mode train`、`caa --mode select`、`caa --mode evaluate --split dev`、`caa --mode evaluate --split test` |
| INLP | CAA 导出原始提示表示后，`inlp fit --threads 8` 在 CPU 拟合；`inlp run` 执行冻结配置的 dev/test 评分 |

表中模块名均加前缀 `python -m experiments.aster_codex03_external_20261112.`。各入口按已有文件继续未完成阶段；复用现有正式目录会读取其已冻结配置和分数。独立新实验使用独立输出目录。完整训练参数以交付配置文件为准。

## 历史准备记录

`official_sources.json`、`source_preparation.json` 和 `validation/` 记录实验开始前的来源准备与 CPU 验证，保留当时的 `pending_Codex01`、`not_started`、`actual_experiment_results=false` 等状态。它们用于说明准备过程；当前完成状态和正式结果由报告根目录的 `status.json`、主表以及各方法训练/冻结配置给出。Codex 00/01 审计副本位于 `upstream_audit/`。

`validation/` 的 raw 缓存重放用于验证评价程序，文件名为 `section4_5.fixture.csv`，主表收录真实新方法结果。旧 GPU 5/6 控制器及后续共享队列文件作为执行记录保留，全部本轮方法现已完成。
