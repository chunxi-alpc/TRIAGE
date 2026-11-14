# E6、E7、E8 已有结果与缺项核对

核对日期：2026-11-14。检查了服务器工作树、历史运行目录、GitHub main 与 91 个远端分支引用中的相关结果文件，并通过 GitHub 连接器读取主实验文档、Codex 00/01 报告、旧五模型汇总和四模型官方路径报告。详细来源见 github_evidence.json；文件清单见 github_ref_inventory.json；本轮阶段记录见 current_artifacts.json。

## 复用结论

同一检查点、同一训练协议、同一冻结参数和同一评价总体的完成结果直接复用。部分完成的任务保留已拟合参数、缓存和逐例预测，只接续缺失阶段。

- 旧五模型 15 组实验已完成，保留历史记录。
- Qwen3-8B 当前 E1 的参数、受控测试、NLI 500/5500 和 MedPIC 188/467 结果直接复用。
- Llama-3.1-8B-Instruct 本轮 E1 拟合、冻结、受控测试、四项迁移及官方 MEGA 均已完成，直接复用。
- Gemma-2-9B-IT 本轮 E1 拟合、冻结、受控测试和四项迁移均已完成；官方 MEGA 没有完整完成记录，已有输出保留。
- Gemma-3-27B-IT、MedGemma-27B-Text-IT 本轮 E1 的 M1/M2/M3 均已完成。开发集冻结和后续评价尚未完成，应复用参数与缓存接续缺失阶段。
- Gemma-2-9B-Base、Qwen3-8B-Base 未找到已有相同 E1 的完整结果，继续当前任务。Gemma2 Base 已完成拟合、冻结及受控测试；Qwen Base 正在执行独立拟合。

## 旧记录与当前 E1 的对应关系

旧 MEGA 拟合使用 A/P 4,887 对、C/F 135 对。当前 E1 使用 A/P/C/F 各 635 对，合计 2,540 对。旧 NLI 实验在 NLI 数据内独立选头和拟合；本轮 NLI 使用源任务 MEGA 的冻结参数。旧协议的已修正预测按历史记录引用，当前 E1 缺失单元继续保留缺失状态。

四个 IT 模型各已复用 15,511 条原始特征记录。NLI 子集 2,864 个输入已有原始缓存；当前冻结参数对应的指标需要单独生成。Gemma2 的本轮提取记录显示 NLI 子集新增前向为 0，NLI 5,500 总体新增 2,636 条。

## 调度

共享队列已配置 GPU 0、1、4、5、6。核对时 Gemma2 Base 在 GPU 0，Qwen Base 在 GPU 4；其他空闲卡由同一调度器分配。原三个 IT 整套作业已停止，其已完成阶段保留。缺失阶段由共享调度任务拆分接续，本次核对未启动第二套调度器或重跑训练。

E6 的 Gemma2、Gemma3、MedGemma 当前 E1 官方 MEGA 完成状态仍为缺失；Gemma3、MedGemma 的源开发冻结、受控测试和 E7 四项当前协议评价也尚未完成。共享队列的后续接续范围由调度清单记录。Qwen 当前 E1 官方 MEGA 7,458 路径结果本次未定位，按原任务安排不启动 Qwen 重跑。

## 证据

- [GitHub 当前实验框架](https://github.com/chunxi-alpc/TRIAGE/blob/main/docs/ASTER/experiments.zh.md)
- [GitHub 旧五模型 15 组汇总](https://github.com/chunxi-alpc/TRIAGE/blob/main/results/aster_five_models_20261109.csv)
- [GitHub 四模型旧官方路径报告](https://github.com/chunxi-alpc/TRIAGE/blob/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/cross_model_mega/REPORT.zh.md)
- [GitHub Codex 00/01 核对](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-codex00-01-audit-20261112/docs/ASTER/baseline-audit-20261112.zh.md)
- 服务器历史参数：/home/gcx/aster_five_models_20261109/results/<model>/mega/。
- 服务器旧官方输出：/home/gcx/aster_cross_model_mega_20261111/<model>/。
- 本轮参数和逐阶段结果：/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/models/<model>/。
- 共享调度：/home/gcx/aster-gpu-queue-20261113/。
- 较早历史对照明细：historical_compatibility_reference.json，保留原报告路径。

本报告记录此次核对的结果；运行进度以共享队列和各模型完成记录为准。未修改论文正文。

## 本次核对后的共享队列安排

共享调度者确认：两个 Base 任务继续；Gemma3、MedGemma 复用已有 E1 拟合参数，分别接续剩余开发集特征、源任务冻结、NLI 5,500 与 MedPIC 188/467 的缺失评价。原整套作业保持停止。该安排不接续其受控源测试和官方 MEGA，相关 E6 单元继续记为未完成；Gemma2 IT 的官方 MEGA 同样保留未完成状态。

NLI 5,500 包含原始 500 条声明。可在获得同一冻结参数的逐例预测后做 CPU 子集汇总，不增加模型前向；旧 NLI 内拟合分数继续按历史结果引用。本核对任务未另行启动 NLI 500 实验。
