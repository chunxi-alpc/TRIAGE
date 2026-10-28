# 规则读取与答案读出的后续诊断

本轮沿用 [PR #40 的共享方向实验](../module2_mechanism_20261028/README.md)，定位“局部读取改变而 A/B 答案不变”的原因。旧模块二离线方向学习，在主线 2e16df8 中已改为模块一；冻结代码和目录保留旧编号。

[完整结果与诊断图](RESULTS.zh.md) · [固定协议](PROTOCOL.zh.md) · [全体分析](analysis/ANALYSIS.zh.md)

## 冻结范围

仍使用相同的 Qwen3-0.6B-Base 权重、四个头、ID+action 目标及全部 24 个确认提示。新增诊断包含原基线与增益 4 重放、固定学习方向 ±32、直接分数偏置 ±8，以及正确/相反路由文本，共 192 条记录。没有重新拟合、选头、筛样或剂量搜索。

这是一轮基于已见开发结果设计的探索性诊断，只有三个确认指南。较大干预和显式 oracle 标签用于检查通路，不作为推荐部署增益或正式任务成绩。

## 复现

环境沿用 [固定依赖](../module2_mechanism_20261028/requirements.txt)。从仓库根目录执行，MODEL_DIR 指向 revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd 的本地模型目录：

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export MPLCONFIGDIR=/tmp/triage-mpl
    DIAGNOSTIC_RUN_DIR=/tmp/triage-readout-diagnostic
    python scripts/run_rule_readout_diagnostic.py --model "$MODEL_DIR" --output "$DIAGNOSTIC_RUN_DIR"
    python scripts/analyze_rule_readout_diagnostic.py --records "$DIAGNOSTIC_RUN_DIR/records.jsonl.gz" --status "$DIAGNOSTIC_RUN_DIR/status.json" --output "$DIAGNOSTIC_RUN_DIR/analysis" --figures "$DIAGNOSTIC_RUN_DIR/figures"

运行器逐条核对原规范答案读出的 baseline/projected4，保留新的完整词表概率与实际注意力。压缩记录是原 JSONL 的无损 gzip。协议、输入、权重、算子及科学源码摘要保存在运行指纹中；不同指纹的检查点不能混用。

仅重算分析时无需加载模型，将上述分析命令的 records/status 指向本目录 results 下的已完成产物即可。脚本会拒绝未完成、缺失条件或分母变化的输入。

## 已完成结果

192/192 条记录完成。学习方向 +32 的受限 A/B 正确数为 15/24（基线 12/24），含 4 次纠正和 1 次由对变错；反向 −32 为 8/24。直接分数 +8 为 14/24，反向 −8 为 10/24。正确／相反单 ID 注释分别为 24/24、0/24。

大幅增强局部读取仍未产生稳定一致的正确标签间隔响应，且强内部编辑的离散翻转都为 A→B。报告保留全部正反结果、连续分数、候选完整词表概率和三指南分层，不将这个应力点改成默认超参。

48 个旧条件锚点均在预定容差内，24 项完整/后段比较误差为零；8 项新增契约检查通过。数据、源码与图表摘要见 [ARTIFACTS.json](ARTIFACTS.json)，历史候选规范字段的范围更正见 [METADATA_CLARIFICATION.json](METADATA_CLARIFICATION.json)。
