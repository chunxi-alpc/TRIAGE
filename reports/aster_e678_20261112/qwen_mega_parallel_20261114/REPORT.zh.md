# Qwen Base 官方 MEGA：历史核对与五卡续跑

## 历史核对

已查服务器结果目录、GitHub main、相关历史报告及先前覆盖91个远端引用的核对记录，未找到 Qwen3-8B-Base 在本轮 E1 冻结参数下的官方 MEGA 7,458 路径完整结果。

GitHub 中较早 Qwen Base 的完整记录是 CounterFact、BiasBios、PASTA 等实验；当前已有 Qwen 后训练模型的结果按原检查点复用。证据入口：[Base历史报告](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-execution-20261025/experiments/aster_seka_execution/REPORT.zh.md)。检索与已有核对引用见 github_evidence.json。

## 已实施的资源调整

单卡运行保存的 **2380 条完整路径**全部复用；剩余 **5078 条路径**按固定顺序分成五个互不重叠的部分，分配 GPU0、1、4、5、6。每部分 1016／1016／1016／1015／1015 条，沿用相同冻结参数、提示、评分和路径解码。部分完成病例的节点特征和预测也按病例分配并复用。

每卡预计20–28 GiB，预计剩余60–90分钟。本次新增训练为0。旧单卡进程已结束，原结果目录保留；新任务只遍历未完成路径。

各部分输出到 models/qwen_base/evaluation/mega_official/shard_resume0 至 shard_resume4。原 shard0 作为已完成结果来源保留。五部分结束后，由CPU将原有结果与新增结果按病例和方法合并，完整分母保持7,458路径、45,627金标准路径节点、418来源；最后自动完成Base/IT及全队列汇总。

## 文件

- 代码：experiments/aster_e678/parallel_qwen_mega.py
- 运行计划：models/qwen_base/evaluation/mega_official/parallel_resume_20261114/plan.json
- 队列：/home/gcx/aster-gpu-queue-20261113/status.json
- 交接记录：/home/gcx/aster-gpu-queue-20261113/qwen_mega_parallel_20261114/
- 原三个IT整套流程维持取消状态，其他已完成结果直接复用。

CPU验证通过：已完成病例排除、剩余分片互不重叠、部分病例节点缓存复用、全部分片完成后才生成全量汇总。未修改论文正文或历史报告。
