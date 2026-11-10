# 痕迹修正：判断正确率后续实验

本实验固定Qwen3-8B和主输入模板，检验使用判断标签训练的痕迹修正，能否提高干扰路径正确率并保护原始路径。数据为MEGA已有生成病例及有原文依据的受控编辑；模型评分与训练结果均为实际运行，不使用模拟数字。

## 环境与输入

在服务器 `gcx@10.29.128.115` 运行，激活 `source /home/gcx/gcx261014/bin/activate`。
代码工作树：`/home/gcx/aster-trace-repair-followup-20261110`。
大缓存：`/home/gcx/aster_trace_repair_20261110`，包含完整输入、原始标注生成、节点评分和头特征，不重复提交到Git。

已有输入依赖：
- repair-v2评分和原始来源划分：`/home/gcx/aster_cr_repair_v2_run_20261108`。
- 原始病例目录和冻结强基线：`/home/gcx/aster_next_20261109`。
- 冻结模型权重：`/home/gcx/aster_optimization_20261109`。
- 五折训练来源选头与子空间：仓库原 `trace_validation/exp3/full/`。

## 已固定的设计

原fit为80来源、1712病例、11722个金标节点；原始与三个有效A版本共46888任务。
新dev为20来源、256病例、19776任务；锁定final为60来源、802病例、63356任务。
同一来源的所有病例和配对始终同划分。抽样不看模型错误或分数；来源名单见 `data/source_selection.json`。

fit只收集金标路径节点，内部配置选择按金标节点全部判断正确计算路径成功；节点输入不依赖模型实际行走历史。
新dev/final收集整棵树各节点，所有方法实际从首节点走到叶子，路径和叶子都匹配才正确。
主指标先在病例内平均A版本，再平均病例和来源。普通病例加权、逐节点和逐配对指标同时保存。

新增线性与17→32→1网络使用原始/干扰交叉熵、一致性、原始正确节点保护及固定证据预算0.05/0.1/0.25。
仅分数网络、随机特征网络、扩展fit偏置/仿射校准、AP岭回归和已冻结的强修正作为对照。
新标签训练、均值尺度和内部配置选择不读取dev/final。先做旧fit来源五折，之后新dev选主方法，写入 `evaluation/frozen.json` 后才允许final评分。

新C/F与P仅在原文、事实和程序真值能支持时构造。增强提取使用同一Qwen作规则/事实/编辑标注，再做程序验证，属于构造评价对，未经临床人工核验。失败及覆盖如实保存。
原条件输入对照只复用已有结果，新来源主比较均保持固定输入。

## 三卡资源与运行

主评分使用GPU 0、1、6各一份Qwen3-8B，预计18–24 GiB/卡；初始201节点实测峰值约15.9 GiB。
GPU6同时运行数据提取模型，预报额外24–32 GiB，合计约42–50 GiB；稳定后不持续监控。
全部原始日志保存在大缓存及服务器任务日志中。

从仓库根目录重新执行：

```bash
bash reports/aster_cancel_story_20261109/meeting/trace_repair_followup/run_all.sh
```

请勿与尚未结束的同一流水线重复启动。脚本复用已冻结来源和已保存评分；已有逐候选CV结果可恢复，不重选最终来源。
模型稳定运行后不需要持续监控，父进程仅等待阶段完成；任一异常写入 `pipeline_status.json`，不会解锁未冻结最终评价。

分阶段入口：
```bash
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/train.py --stage cv
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/train.py --stage fit
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/evaluate.py --stage dev
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/evaluate.py --stage final
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/report.py
```

最终评价不重新选型。主要比较由dev冻结，2000次来源配对自助重采样评估不确定性。干扰增益区间下限>0与原始路径差值区间下限≥−0.5个百分点分别报告；不满足时保留阴性结论。

## 交付与复跑

`configuration.json`保存固定配置；`data/`保存来源、覆盖、构造代码和检查；
`training/`保存内部选择、全部导出参数及真实缓存参数重放；
`evaluation/`保存dev/final逐节点、逐对、逐路径、逐来源结果、置信区间及两张图；
`REPORT.zh.md`由实际结果生成，`replay_check.json`记录完成后的参数与覆盖复查。

运行 `python .../verify_saved.py` 可重放保存参数并检查最终覆盖，不重新运行主模型。
旧实验结果和论文正文保持不变。最终提交只包含本目录的代码及必要结果，完整提示词、特征和模型骨干权重留在原输入/大缓存位置。
