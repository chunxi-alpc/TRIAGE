# ASTER：MedGUIDE 三版本反事实预实验

这是可执行实现，不是仅含方案的文件。目录包含数据构造（prepare.py）、Llama 模型实际前向及注意力干预（run.py）、统计与曲线（analyze.py）、统一入口（run.sh）。已有同类干预历史结果不再重复。

## 已实现的四项实验

任务 0：读取 MedGUIDE MCQA 原始 Parquet 和原作者的完整 all_paths.json。使用病例参考路径查找同父节点但末级条件不同的兄弟路径。要求原始正确选项与参考路径的来源动作一致，并且至少存在三个可区分的来源动作；不满足条件的病例跳过并记录原因。构造 A 原版、B 只修改适用规则动作、C 只修改兄弟规则动作。三个版本始终使用同一病例、规则顺序和选项，修改动作来自已公布路径，并保留各条规则和动作的字符区间。

任务 1：对相同选项进行闭卷作答，再分别回答 A/B/C；记录原模型准确率、兄弟动作误用及闭卷同选项错误。闭卷相同选择只是操作性偏好，不能直接证明参数记忆。

任务 2：比较整段指南、仅修改规则的直接注意力分数增强与原模型。这里不使用官方 SEKA 投影，因此不将它错误地称作 SEKA 实验。原仓库已有官方 SEKA／PASTA 在模拟病例上的实验。

任务 3：对指定层、最后一个答案查询位置，分别计算指向适用规则和兄弟规则的注意力质量、归一化质量，并按答对／答错汇总。

任务 4：比较仅增强适用规则、同时增强两规则、增强适用并抑制兄弟规则、反号控制、直接在提示中说明正确规则。带符号干预固定扫描 0.5、1、2、4 四个强度。同时记录手动注意力的零偏置对照，以辨别数值重算与干预收益。

## 数据与结论边界

原始 MedGUIDE 的 path 是路径序列，不是单独验证过的互斥逻辑谓词。本实验**离线用参考路径选择两条规则**，B/C 的新标签是实验构造标签，而不是 NCCN 正式治疗建议。模型提示本身不直接包含“正确规则”的标注，除明确单列的提示上限对照外。选项中的非正确项可能被受控替换，这不是官方 MCQA 原题成绩。结构上的兄弟路径也不保证医学上的条件互斥，因此结果应解释为条件分支机制探索，而非临床指南遵循的独立证明。

如实际满足条件的病例少于 300，报告实际数量，不能补充未经验证的病例或使用未构造的标签。模型参数不训练。不同样本来自同一树时统计不把它们视作独立规则家族。来源数据卡许可为 TBD、用于学术研究，病例与原始文本全部放到被 Git 忽略的 private/。

## 运行

使用原研究服务器上的 Python、PyTorch、Transformers 4.57.x、PyArrow 和已经获授权下载的 Llama-3.1-8B-Instruct 本地权重。以下命令在仓库根目录运行：

~~~bash
source /home/gcx/gcx261014/bin/activate
export ASTER_MODEL=/absolute/path/to/Llama-3.1-8B-Instruct
export MEDGUIDE_PARQUET=/absolute/private/path/train-00000-of-00001.parquet
export MEDGUIDE_PATHS=/absolute/private/path/all_paths.json
export CUDA_VISIBLE_DEVICES=0
export ASTER_CASES=20
bash experiments/aster_medguide_20261022/run.sh
~~~

完成 20 例后，将 ASTER_CASES 改为 300。与原有记录不同的模型或参数配置必须用单独的输出位置，不要混用 runner 的断点续跑功能。真实输出仅在 private/ 内。统计会写 summary.json、metrics.csv、errors.csv、attention.csv、strength_curve.svg 与 REPORT.zh.md。

不使用模型的合成接口示例：

~~~bash
python experiments/aster_medguide_20261022/prepare.py --demo --max-cases 20 \
  --output experiments/aster_medguide_20261022/private/demo_cases.jsonl
~~~

## 与已做实验及公开论文的关系

已有 [CARE 有符号诊断](../care_module2_signed_20261019/README.md)：Qwen3-4B-Base 的 192 对模拟病例，基线成对正确 72/192，直接有符号偏置为 77/192。此前 [CARE 因果链 PR #11](https://github.com/chunxi-alpc/TRIAGE/pull/11) 显示注意力已被改变，但其独立决策收益有限。更早的 [CARE 临床来源规则观察](../care_qwen3_4b_base_20261016/results/observations_summary.md) 已包含官方 SEKA 与 PASTA。关闭且未合并的 [MedGUIDE 全量 PR #19](https://github.com/chunxi-alpc/TRIAGE/pull/19) 是冻结 A1 与 Qwen 对照，不是这里的 A/B/C 反事实任务；公开分支尚未发现完成的全量准确率。

相关工作：[PASTA，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b99d6cc40b05809c3d84b57a165448cd-Abstract-Conference.html)；[SEKA／AdaSEKA，ICLR 2026](https://github.com/waylonli/SEKA)；[MedGUIDE 数据集](https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K)。通用注意力增强及直接有符号分数偏置不是 ASTER 的全新技术。

**实际状态：代码已实现并提交；本次环境无 GPU 或 Llama 本地模型，尚无新的真实模型准确率结果。**
