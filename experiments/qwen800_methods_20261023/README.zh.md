# 固定 800 例 Qwen 方法对比

基于 PR24 `c615f83`，复用 PR23 `3e4fac7` 的 Qwen3-8B、eager attention、增益 0.5、主方法自由生成 8 token；CAD 使用 128 token，过滤 beta=0.1。CAD8 读取同一条未过滤解码的前 8 token，不重复推理。K/U 固定为 PR24 的 K426/U374（其闭卷使用 SDPA），不重新选样或调整分组。

状态见 [results/STATUS.json](results/STATUS.json)。只有 `status=complete` 才表示推理、全部键校验与统计已完成；最终报告自动保存为 `results/REPORT.zh.md` 并推送当前独立分支。完整日志与生成留在服务器 `/home/gcx/qwen800_methods_20261023/private`。

800 例中的 A/B/C 全部运行，D797，CS84。三例 D 因原选项没有不同的其他条件分支动作而不可构造，具体 ID 和原因见 case_manifest.jsonl。304 例满足 PR23 原构造；其余使用同树最近共同前缀的其他条件分支，优先第三选项；二选一题 C 改成原动作。分别报告全样本、原协议子集、原协议且单动作分支子集。原 200 例是筛选过的单动作分支子集，跨队列差值只作描述，19 个重叠 ID 单独配对核对。CS 当前为按患者值改写，旧实验固定阈值 6→12，协议不同。

13 种处置方法：完整指南基线、金标单规则裁剪、金标注意力增强、金标带符号引导、金标分支组裁剪、可见预测裁剪、隐藏预测裁剪、隐藏 Top-2、隐藏预测带符号/无符号引导、CAD8、CAD128、过滤 CAD128。另有隐藏/可见/只显示适用动作的三种分支诊断。总计 52,496 条结果，原选项顺序；本轮不包含选项轮转、噪声/错误适用性分配、增益搜索或注意力阻断。

每个版本和方法分别报告 all/K/U 准确率、解析失败、原动作选择率、分支允许动作率；方法配对比较使用 McNemar、按树重采样 2,000 次区间与 Holm 校正。CS 仅一棵树，不生成跨疾病区间。BCD、BCDCS 共同病例调和平均为描述统计。

运行环境及路径：

- 仓库 `/home/gcx/TRIAGE-worktrees/qwen800-methods`；环境 `/home/gcx/gcx261014`。
- `prepare.py` 读取 PR24 私有 cohort、公开 CS 和 K/U 清单以及已有 MedGUIDE 原始数据，生成固定病例和两分片；304 例的 ABC 提示与 PR23 构造逐字检查。
- `run.py` 两个进程各使用一张 GPU，按 `(id, variant, method)` 恢复；输入和运行代码哈希不一致会拒绝复用。
- `analyze.py` 仅在两分片完整且全部预期键存在时生成最终报告。原 200 例对比读取仓库中已提交逐题结果，不捏造旧结果。
- 在仓库根目录运行 `python -m unittest discover -s experiments/qwen800_methods_20261023 -p 'test_*.py'`。
- 正式队列由 `bash experiments/qwen800_methods_20261023/pipeline.sh` 启动。包含运行锁，任务结束后自动统计、提交、推送；失败则保存失败阶段。不要在已有完整结果上反复启动。

完整提示及原始生成不上传；逐题统计压缩保存为 scores.jsonl.gz，来源哈希、参数、分母和变体缺失原因随结果提交。
