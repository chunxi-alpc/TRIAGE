# ASTER：MedGUIDE 有符号规则引导先导实验

本目录补齐 ASTER 的 MedGUIDE 实验入口，不修改 TRIAGE/CARE 既有算法与历史结果。

## 相关工作与新颖性边界

已有研究明确覆盖了通用注意力引导和干扰抑制，因此“高亮规则”本身不是新颖性主张：

- PASTA，Zhang et al., ICLR 2024，推理时重加权指定输入片段的注意力。https://proceedings.iclr.cc/paper_files/paper/2024/hash/b99d6cc40b05809c3d84b57a165448cd-Abstract-Conference.html
- SEKA/AdaSEKA，Li et al., ICLR 2026，通过编辑键向量实现高效注意力引导。https://proceedings.iclr.cc/paper_files/paper/2026/hash/32ddf8ee55540539eb46083cd72361e8-Abstract-Conference.html
- FAI，Yan et al., ICLR 2025，识别并抑制干扰性示例词元。https://proceedings.iclr.cc/paper_files/paper/2025/hash/9eaeccb40e16ba5ae4e25e9fdb2fea19-Abstract-Conference.html
- MedGUIDE-MCQA-8K 数据集：https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K

本仓库已有 CARE 模块二有符号注意力实验，但其数据为模拟病例，模型为 Qwen3-4B-Base，且没有建立学习方向的稳定决策收益。本实验新增 MedGUIDE、Llama-3.1-8B-Instruct、闭卷记忆基线、规则反事实 A/B/C 和 oracle 有符号干预。当前检索未发现完全相同的组合，但这不等于证明绝对新颖；投稿前仍需系统性文献检索。

## 数据要求

MedGUIDE 的 path 描述决策路径，但公开样本未必提供完整决策树及兄弟分支。因此，构造反事实需要一份路径节点可对齐的规范化树文件，格式如下：

```json
{"nodes":[{"node_id":"n1","parent_id":"root","condition":"指标 < 30","action":"动作 A"}]}
```

数据集的每条记录必须有 profile、prompt、options、answer、answer_text、path。path 必须是节点 ID 数组，或含 node_id/id 的对象数组。脚本不会猜测节点对应关系，也不会自动编造临床动作。原数据许可目前标为 TBD；不要将原始样本或受限指南文本提交到仓库。

## 运行

```bash
pip install -r experiments/aster_medguide_20261022/requirements.txt
python experiments/aster_medguide_20261022/prepare_dataset.py --tree-json /path/to/tree.json --output /path/to/aster.jsonl --limit 20
python experiments/aster_medguide_20261022/run_pilot.py --input /path/to/aster.jsonl --model meta-llama/Llama-3.1-8B-Instruct --output /path/to/predictions.jsonl
python experiments/aster_medguide_20261022/analyze.py --input /path/to/predictions.jsonl --output /path/to/summary.json
```

先运行 20 例，成功后将 limit 调到 300。模型权重和生成的病例结果写到仓库外部目录。

## 实验臂

- A：原指南。
- B：只改适用规则动作，答案映射到新动作对应选项。
- C：只改不适用兄弟规则动作，答案保持原答案。
- 闭卷：只提供病例与问题，记录模型的记忆答案。
- Oracle 仅增强：只给适用规则加分。
- Oracle 无符号：给适用规则和兄弟规则都加分。
- Oracle 有符号：适用规则加分、兄弟规则减分。

只保留动作在选项中有唯一匹配、金标与被修改规则动作一致、规则节点与路径对齐的样本；无法满足条件时跳过，不替换无关选项。准确率之外报告固执错误率、兄弟分支误用率，以及干预相对基线的纠正/破坏数。注意力分配是诊断线索，不是因果证据。

医疗规则反事实只用于离线方法学研究，不构成临床建议。
