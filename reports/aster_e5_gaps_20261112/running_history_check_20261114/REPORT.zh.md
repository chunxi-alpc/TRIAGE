# 当前运行实验的历史记录核对（2026-11-14）

核对对象为共享队列中的 Codex05 五项 E6/E7/E8 作业。已检查服务器历史目录、仓库记录、GitHub main 及相关历史分支。

## 结论

三个 IT 模型已有旧协议的完整实验，两个 Base 检查点未找到相同 E1 协议的完整 ASTER-CR 结果。当前五项均有 E1 补缺内容；保留运行，本次未启动新训练、修改配置或停止进程。

| 当前模型 | 以前是否跑过 | 当前补缺 | 导入历史原始缓存 |
|---|---|---|---:|
| Gemma-2-9B-IT | 已完成旧 MEGA、NLI、三段论及 MEGA 官方路径 | E1 独立拟合及冻结评价 | 15,511 |
| Gemma-3-27B-IT | 已完成旧 MEGA、NLI、三段论及 MEGA 官方路径 | E1 独立拟合及冻结评价 | 15,511 |
| MedGemma-27B-Text-IT | 已完成旧 MEGA、NLI、三段论及 MEGA 官方路径 | E1 独立拟合及冻结评价 | 15,511 |
| Gemma-2-9B-Base | 已查记录中未见相同 E1 完成结果 | E1 Base/IT 匹配比较 | 0 |
| Qwen3-8B-Base | 早期 SEKA、PASTA 标准任务已完成；未见相同 E1 完成结果 | E1 Base/IT 匹配比较 | 0 |

## 旧协议和本轮的区别

旧五模型共15组实验已发布。三个 IT 模型均有806条受控路径、500条NLI原声明的结果；GitHub历史分支另有各7458条MEGA官方路径的完整报告。

旧MEGA训练使用4887个A/P对（A4869、P18）和135个C/F对（C66、F69）；F0另列。旧预算为各模型拟合A/P RMS的0.1倍，Gemma2、Gemma3、MedGemma分别为0.1191963、0.2255855、0.0943431。

当前正式训练为E1均衡2540对，A/P/C/F各635对，预算统一为0.2923349。旧NLI在NLI数据内单独选头和拟合，本轮使用MEGA E1参数冻结迁移。旧生成三段论不属于当前主数据集。

当前三个IT模型均重新选择51个头，与旧集合分别重合44、40、49个。旧MedGemma集合为109个头。旧官方缓存存储原选中头的向量，分别缺本轮7、11、2个头；新激活提取和修正评价有实际必要。

## 复用和计算重合

每个IT模型已导入15511条同检查点原始缓存。拟合、开发、受控测试分别可复用1785、1685、3100条，NLI子集复用2864条；这些阶段有交集，不能相加为独立输入数。

已完成的Gemma2提取记录显示：NLI子集新增前向0；5500条总体新增2636、复用2864。相应实测见各模型performance目录。

旧官方raw和bias_only结果均已存在。当前官方入口按病例节点共享一次前向，同时提取新头并输出raw、bias_only和aster_cr分数。原始分数随新头一起算出，存在该部分计算重合。当前入口只读取本轮官方缓存，未导入旧官方所选头缓存。

这批实验应表述为“保留旧结果，按E1重新拟合并补齐评价”。已查记录未发现可由相同E1完成结果整项替代的当前任务。未根据旧测试结果选择配置。历史40来源806路径及官方7458路径均保留既往使用记录。

## 证据入口

- [GitHub旧五模型15组汇总](https://github.com/chunxi-alpc/TRIAGE/blob/main/results/aster_five_models_20261109.csv)
- [GitHub旧实验索引](https://github.com/chunxi-alpc/TRIAGE/blob/main/docs/ASTER/archive/experiments_legacy_20261111.zh.md)
- [GitHub四模型官方路径完整报告](https://github.com/chunxi-alpc/TRIAGE/blob/aster-cr-20261108/reports/aster_cancel_story_20261109/meeting/cross_model_mega/REPORT.zh.md)
- [早期Qwen Base标准任务报告](https://github.com/chunxi-alpc/TRIAGE/blob/codex/aster-seka-execution-20261025/experiments/aster_seka_execution/REPORT.zh.md)

服务器旧拟合与结果：/home/gcx/aster_five_models_20261109/results/<model>/mega/。
服务器旧官方结果：/home/gcx/aster_cross_model_mega_20261111/<model>/shard*/。
当前配置、复用和实测：/home/gcx/aster-e678-gaps-20261112/reports/aster_e678_20261112/models/<model>/。

本目录matrix.json和matrix.csv保留逐项参数、头集合差异、复用数和已完成阶段提取实测。github_evidence.json保存GitHub来源链接及摘录。

本次未进行哈希检查，未覆盖历史报告，未修改论文正文。
