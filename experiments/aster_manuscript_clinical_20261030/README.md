# 固定模型修订的临床实验

实际状态见 [临床报告](../../docs/ASTER/clinical-results-20261030.zh.md)。本目录公开代码和汇总，不公开临床原文、病例标识、原始提示/输出、表示或逐病例评分。

在服务器独立工作树执行，先激活 `/home/gcx/gcx261014/bin/activate` 再激活 `/home/gcx/aster_manuscript_20261030/venv-overlay/bin/activate`。所有模块命令从仓库根目录运行。Transformers 固定 4.57.6。

本次只读资源和阶段命令如下。`$R` 代表原运行 `/home/gcx/aster_manuscript_20261030/clinical01`。正式执行已启动，不得重复运行下载/构造并覆盖原目录；重建必须用新目录，且同步设置 `data.py` 的 BASE/MODEL 与来源证据路径。部分数据阶段有本次固定只读依赖，不声称只改输出参数即可任意数据集复现。

| 阶段 | 模块及参数 | 说明 |
|---|---|---|
| 模型修订 | `provenance --output NEW_MODEL_DIRECTORY` | 固定 ModelScope 提交；既有旧模型不覆盖 |
| 临床三态构造 | `data --output NEW_DATA_DIRECTORY` | 原始源树及临床病例、当前年龄独立规则、自然未知、真实分词 |
| 独立一致性复核 | `review --data DATA_DIRECTORY --output NEW_REVIEW_DIRECTORY` | 单 GPU，Llama 判断，不提供期望标签；原快照修订未知 |
| 键采集 | `collect --data DATA_DIRECTORY --output NEW_COLLECTION_DIRECTORY --shard 0` | 0/1 分片分别设置允许 GPU；使用未编辑固定 Qwen |
| 谱学习/选头 | `analyze --run $R` | 两个采集分片和独立复核完成后；新 analysis 目录 |
| 原始病例清单 | `cases --output NEW_CASE_DIRECTORY` | 完整源树与原始病例；金标仅评分 |
| 分类缓存 | `classify --run $R --shard 0` | 0/1 两分片，只用校准和开发病例 |
| 校准/阈值 | `freeze --run $R` | 独立标签及开发接受率，尚未选增益 |
| 开发增益 | `worker --run $R --phase gain --shard 0` | 0/1 两分片；随后 `gains --run $R`，按开发终点准确率选并冻结 |
| 机制诊断 | `mechanism --run $R --shard 0` | 金标根诊断与预测根诊断单列，0/1 分片 |
| 内部确认 | `worker --run $R --phase confirm --shard 0` | 原始病例13方法，旧 native OOM 保留 |
| 统计补充 | `statistics --run $R` | 只读既有产物，无重新生成、调参或按成绩重跑 |
| 官方四卡 | `launch` | 物理 0/1/5/6，各一进程；动态病例队列、共享病例分类缓存；正式执行已启动 |
| 汇总 | `report --run $R --phase confirm` 或 `--phase official_test` | 检查全部已登记单元；指南簇配对区间、失败保留 |
| 公开汇总 | `publish --run $R --out PUBLIC_DIRECTORY` | 白名单汇总，不复制病例原文/输出 |

上表模块完整前缀为 `python -m experiments.aster_manuscript_clinical_20261030.`。具体参数可读各模块 argparse。官方评分器只读源码由 `official_sources.json` 固定的 `nchennnn/MEGA-CDP@15a3a12e73a9169a870bff2ac431ec2e9968b8c7` 取得；不将第三方整套代码复制进本仓库。

`native_reference.py`、`native_balanced_reference.py` 与 `native_memory.py` 是原始完整指南对照的内存路径比较，18个原有成功病例完整生成比较按技术成功选取，不按答案正确性。初始256行分割的差异及25个原OOM均保留；正式使用2048行均匀分割，全部因果键仍参与注意力。

恢复仅补缺失病例/方法单元：在对应物理GPU可用时启动 `worker --run $R --phase official_test --shard S --dynamic --official-judge --native-query-chunks 2048 --native-balanced --resume`，其中逻辑 S=0/1/2/3 对应物理 0/1/5/6。已有成功、技术失败和评分失败均不重新生成；死进程病例锁可回收。禁止在尚活跃的同一物理GPU启动恢复副本。

`official_score.py` 采用本地固定 Qwen 作为官方算法的传输适配，不是 GPT 主表。结构编号输出按预测源树条件/结论文字渲染，真实官方格式保持原样；同病例相同表示共享评分。每请求要求五个评分结果，未冒充五次独立模型运行。

官方启动器只阻塞等待四个子进程结束后执行一次汇总，不轮询GPU；完成发布任务通过Linux进程结束通知等待，不持续监控。MedGUIDE 当前只有17家族来源归并，未运行端到端。当前年龄规则验证加辅助模型复核不等于医生金标。
