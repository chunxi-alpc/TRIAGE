# MedGUIDE 扩样与阈值改写

本目录从 PR #23 的 `3e4fac7` 继续，独立保存结果。报告见 [results/REPORT.zh.md](results/REPORT.zh.md)，统计见 [results/summary.json](results/summary.json)。

在 115 服务器激活 `/home/gcx/gcx261014` 环境，从仓库根目录运行 `bash experiments/medguide_expand_cs_20261023/reproduce.sh`。默认使用 GPU 0、1。每卡约 16 GiB、800 例闭卷推理；已完成这次实测，不需要持续监控。

- `prepare.py`：读取真实 Parquet 和作者 all_paths，55 树逐一分层，先取阈值候选，再轮流补足 800 例；仅给出患者范围的候选不用于 CS 执行。原表中两树无病例，实际覆盖 53 树。40 例开发集排除在外。
- `closed_book.py`：Qwen3-8B 两分片、自由贪心生成 8 token，沿用 PR23 的答案解析器；K 为闭卷答对，未用模型表现筛选样本。使用 SDPA，与旧 eager 结果不直接拼接。
- `acquire_external.py`：固定 Hugging Face 修订下载，保存实际 URL、SHA256、下载失败原因。可以使用 `--private` 和 `--output` 指定新目录。若镜像无法访问，保留下载失败记录后重试；报告中不编造外部结果。
- `external.py`：MedCalc 的 BMI/调整体重条件数值重算；DriftMed 正确/错误配对和数值变化候选；各数据来源独立统计。
- `finalize.py`：执行逐例检查，重建 84×3 条 CS 提示和金标，输出中文报告、清单和哈希。
- `test_rules.py`、`test_external.py`：边界、单位、公开数值例、分支翻转、标签保留和缺失事实等测试。

依赖现有环境：Python 3.12、torch 2.8.0+cu128、transformers 4.57.6、pyarrow、requests。模型和原始数据均已有服务器本地副本。MedGUIDE 固定修订 `1460ec478280a7295040b92f7eb985327f77dcbe`，原始数据获取脚本见上级 `aster_medguide_20261022/acquire_data.py`；来源哈希在 results/manifest.json。开发集是旧实验已固定的 40 ID：已保存为 `results/dev_exclusions.jsonl`；原始名单在 PR23 `experiments/aster_validation_20261022/results/dataset_manifest.json` 的 splits.dev.ids。

若换服务器，可通过 SOURCE、PRIVATE、DEV、MODEL、RESULTS 环境变量覆盖默认路径。原始数据、完整患者提示和详细运行日志放在 PRIVATE；GitHub 保存样本 ID、树分布、逐题预测、CS 改写、来源行号、规则证据、测试与报告。`reproduce.sh` 可以复用已有闭卷输出；重跑模型推理时设置新的 RESULTS 目录。

CS 按来源路径提供的事实判断未修改条件；不能据此声称自然语言整树推理得到临床金标。新增阈值是合成改写。MedCalc 为数值任务，DriftMed 为正确性任务，都不与 MedGUIDE 选项准确率合并。

来源：

- [MedGUIDE 原论文](https://arxiv.org/abs/2505.11613) 与 [数据集](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K)。
- [MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench)，修订 `83a1a56d6b53f8d4c0e03f0ba9699becf524a595`，CC-BY-SA-4.0；衍生的 medcalc_bmi.jsonl 同许可。
- [DriftMed](https://huggingface.co/datasets/RDBH/DriftMed)，修订 `5dcfe380f49c6f17f3c6ba5e25e0e958f9642f0e`；[原论文](https://arxiv.org/abs/2505.07968)。官方卡为 Research Use Only，未公开转发场景全文；候选清单保留来源 ID、类别、数字与文本哈希。其错误建议是人工改写，不代表历史临床指南修订。
