# ASTER 三模块瓶颈实验（2026-10-31）

保留方向学习、处置盲适用性判断和规则键编辑。均值差默认；谱、缓存硬屏蔽和软偏置为对照。

- 数据：历史43例／2来源；新开发100例／10来源；新验证120例／12来源，按来源、条件类型和路径结构取样，不按预测结果筛选。
- 设备：服务器GPU 0、1、5、6，环境 /home/gcx/gcx261014。
- 初轮：P0及11种算子条件；开发选β后开展新来源5条件验证。
- 后续：层窗口、实际单头决策效应、覆盖子集、匹配位移键控制、新100例选参和条件验证／真实差分诊断、模块二、同判断执行器、top-3路径概率集成、错误注入。
- 数据、原始病例输出及完整日志留在 /home/gcx/aster_bottleneck_20261031；公开目录只保存汇总、报告和图。
- 稳定后不持续监控GPU；由scheduler按进程完成事件分配下一项就绪任务，独立实验跨阶段并行。重复运行只补缺失或失败单元，保留原尝试记录。

入口：
    python -m experiments.aster_bottleneck_20261031.prepare
    python -m experiments.aster_bottleneck_20261031.run
    python -m experiments.aster_bottleneck_20261031.scheduler
    python -m experiments.aster_bottleneck_20261031.report

prepare只在新运行目录执行一次。run完成初轮；advanced需要初轮完成，也可使用事件式进程等待。评测结果不能反向重选验证配置。模块一机制对照仍只有两个年龄来源，不作通用临床机制主张。

必要验证：
    python -m unittest experiments.aster_bottleneck_20261031.test_core
    python scripts/check_aster_markdown_math.py

模块二已按用户要求简化：原始病历＋当前完整条件，由大模型直接判断T/F/U。只比较原分类器、NONE偏置校正与直接三值判断，保留置信度和祖先回退。不运行事实抽取、原子拆分、置换或额外推理对照。当前候选处置、原问题和选项不进入模块二输入。

其他实施范围：祖先中只有规则导航指令时不构造虚假的治疗历史。原behavior.py和proof.py保留为未采用的旧实现，不再自动运行。当前模块一入口为module_one.py，协议见docs/ASTER/module-one-plan-20261031.zh.md；结果保存module_one子目录。

调度器接管前写入scheduler_adopt.json，已有进程通过pidfd等待后复用输出；仅使用0、1、5、6。模块一分片完成基线即可标定；错误注入拆为每任务2例，仍使用原种子和病例。选头／选参需要的结果汇总保留实际依赖。不要与旧advanced独立启动器并行运行。
