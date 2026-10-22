# ASTER MedGUIDE 20 例先导实验

使用真实 MedGUIDE-MCQA-8K 和原作者 `all_paths.json`，测试规则动作改变时模型是否遵循给定规则。运行状态和数值以 `results/REPORT.zh.txt`、各模型 `summary.json` 和运行 manifest 为准。随机初始化的小模型 CPU 测试不属于 MedGUIDE 模型结果。

## 本次实跑结果

两模型均已完成320次生成，全部原始输出保存在服务器。主要反事实 B 的正确数（每组20例）：

| 模型 | baseline | 直接规则提示 | 仅增强适用规则 | 无符号 | 有符号 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 5 | 17 | 7 | 6 | 7 |
| Qwen3-8B | 7 | 14 | 8 | 7 | 7 |

本轮直接规则提示改善最明显；有符号干预在 B 上没有稳定超过仅增强或无干预。完整 A/C、闭卷和配对纠正/损害见 [实跑报告](results/REPORT.zh.txt)。Llama耗时91.1秒、峰值16.95GiB；Qwen耗时126.8秒、峰值17.26GiB。11项CPU测试通过。

答案提取兼容输出开头的字母、加粗、Answer标记及选项文字；不额外调用模型作判断。保留原始生成文件，统一修正提取格式后离线重算，没有重新生成。剩余1条Llama输出无法提取明确答案，计为错误；Qwen无缺失或无法提取答案。

## 数据与 A/B/C

数据源：[MedGUIDE](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K)，固定 revision `1460ec478280a7295040b92f7eb985327f77dcbe`。实际字段包括 disease、profile、options（2–8个文本选项）、answer（A–H）、answer_text、path（临床文本字符串数组）和 prompt。prompt 已含病例和选项，本实现从 profile 和 options 重新渲染一次，避免重复或遗漏选项。

[原论文](https://arxiv.org/abs/2505.11613)链接的[作者仓库](https://anonymous.4open.science/r/Submission-MedGUIDE-187A)提供 `data/all_paths.json`：disease → 完整文本路径列表。这里按原文匹配病例 path 和末端 action；不把文本当节点 ID，不补造树节点、不从病例反推新指南。路径序列含历史治疗，确定性渲染为“临床序列 → 下一步”，不能当作专家独立标注的自然语言适用性真值。

每例输入包含该 disease 的**全部发布路径**，规则顺序由固定种子打乱，A/B/C 顺序相同。相邻分支要求完整前缀 `path[:-2]` 相同、倒数第二个条件不同，且末端动作可匹配选项。记录源路径索引，只表示原始路径位置，不声称获得原树节点 ID。

- A：适用动作 a、相邻分支动作 b，原答案 a。
- B：只把适用动作 a 改成第三个动作 c，答案变为 c；相邻分支保持 b。
- C：只把相邻分支动作 b 改成同一个 c，答案保持 a。

a、b、c 必须对应三个不同选项，选项文字及顺序始终不变。不会把两条规则改成相同动作。C 的分支误用标签随实际编辑更新为 c。原始 gold、answer_text、路径末端和选项必须一致。所有筛选理由、总数和固定抽样行号写入 manifest；不根据模型表现筛选。这里只报告可构造子集的20例先导，不代表全量 MedGUIDE。

## 对照与干预

每个模型共320次生成：每例一次闭卷，以及 A/B/C 各五组。

| 组别 | 输入/行为 |
|---|---|
| closed_book | 病例、问题和所有选项，无规则；仅 A 保存一次并报告准确率 |
| baseline | 全部规则，不干预 |
| direct_rule | 直接在提示中只提供 oracle 适用规则，不改注意力 |
| applicable_only | 全部规则，适用规则注意力分数 +1 |
| unsigned | 全部规则，适用规则与相邻分支均 +1 |
| signed | 全部规则，适用规则 +1、相邻分支 -1 |

四组 oracle 方法使用 gold source path，适用性选择没有学习，不作为端到端系统结果。direct_rule 是明确的 oracle 提示上限对照，输入较短；它与同输入的注意力组分开解释。A/B/C 全部运行五组，用 A 判断原规则是否受损。

注意力组与 baseline 输入逐字相同，使用原生 eager 接口 `(module, query, key, value, attention_mask, ...)`，把规则词元的偏置加到 softmax 前的分数遮罩中。保留因果/填充遮罩，调用原生 GQA 展开、dropout 和 `[batch, query, heads, dim]` 输出，不在输出概率上二次重算错误形状。全部层、头和查询位置干预，KV 缓存解码继续使用原始规则 key 位置。每条干预结果验证补丁实际被调用，退出或异常时恢复函数。仅支持已测 Llama/Qwen3，单进程串行使用。

使用各自 chat template，字符区间映射到模板内真实偏移，不重复添加特殊词元。Qwen3 使用 `enable_thinking=False`。两模型均 BF16、greedy、最多8个新词元，无训练、无调参。解析输出开头的选项字母，兼容加粗、Answer 标记和后接选项文字；不从任意解释正文中搜索英文 A。保存原始输出，解析修正可离线复算。

## 复算

在115服务器 gcx261014环境运行；原始文本及长输出留在 `/home/gcx/aster_medguide_20261022/private`，不提交仓库。准备脚本也可复用已有文件，取得相同内容哈希。

```bash
python experiments/aster_medguide_20261022/acquire_data.py --output-dir /path/to/private --parquet /existing/train.parquet --paths-json /existing/all_paths.json
python experiments/aster_medguide_20261022/prepare_dataset.py --parquet /path/to/private/train.parquet --paths-json /path/to/private/all_paths.json --output /path/to/private/pilot20.jsonl --limit 20
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/aster_medguide_20261022 -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 python experiments/aster_medguide_20261022/run_pilot.py --input /path/to/private/pilot20.jsonl --model /path/to/Meta-Llama-3.1-8B-Instruct --output /path/to/private/llama_predictions.jsonl
python experiments/aster_medguide_20261022/analyze_pilot.py --input /path/to/private/llama_predictions.jsonl --dataset /path/to/private/pilot20.jsonl --output /path/to/llama_summary.json
```

不带本地文件参数的 acquire_data 会从固定源下载；模型需事先下载到本地缓存/目录，运行使用 offline 模式，失败不生成虚构结果。Qwen3 复用同一输入，把 model 改为 Qwen3-8B 目录。

汇总按数据清单提供完整分母，缺失/无效答案计为错误；包括 B 原答案固执错误、以闭卷原答案为条件的固执率、当前分支动作误用、相对 baseline 的配对纠正/损害和 A/C 一致性。不能由注意力变化或20例准确率宣称临床有效性或稳定收益。数据卡许可 TBD，原始数据和指南不重新发布。

## 相关工作

通用注意力引导已有 PASTA（ICLR 2024）、SEKA/AdaSEKA（ICLR 2026）和 FAI（ICLR 2025）；这里的区别是规则适用性驱动的有符号干预及反事实控制，仍需实验验证，不能据此宣称绝对新颖。原先文献链接保留在 PR 历史，当前修改重点是实验正确性。
