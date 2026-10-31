# ASTER 三模块瓶颈实验（2026-10-31）

保留方向学习、处置盲适用性判断和规则键编辑。均值差默认；谱、缓存硬屏蔽和软偏置为对照。

- 数据：已暴露开发43例／2来源；新验证120例／12来源，按来源、条件类型和路径结构取样，不按预测结果筛选。
- 设备：服务器GPU 0、1、5、6，环境 /home/gcx/gcx261014。
- 初轮：P0及11种算子条件；开发选β后开展新来源5条件验证。
- 后续：层窗口、实际单头决策效应、覆盖子集、匹配位移键控制、有限行为目标方向搜索、LEACE和随机擦除、模块二、同判断执行器、top-3路径概率集成、错误注入。
- 数据、原始病例输出及完整日志留在 /home/gcx/aster_bottleneck_20261031；公开目录只保存汇总、报告和图。
- 稳定后不持续监控；阶段进程结束后顺序执行并生成报告。重复运行只补缺失或失败单元，保留原尝试记录。

入口：
    python -m experiments.aster_bottleneck_20261031.prepare
    python -m experiments.aster_bottleneck_20261031.run
    python -m experiments.aster_bottleneck_20261031.advanced
    python -m experiments.aster_bottleneck_20261031.report

prepare只在新运行目录执行一次。run完成初轮；advanced需要初轮完成，也可使用事件式进程等待。评测结果不能反向重选验证配置。模块一机制对照仍只有两个年龄来源，不作通用临床机制主张。

必要验证：
    python -m unittest experiments.aster_bottleneck_20261031.test_core
    python scripts/check_aster_markdown_math.py

模块二已按用户要求简化：原始病历＋当前完整条件，由大模型直接判断T/F/U。只比较原分类器、NONE偏置校正与直接三值判断，保留置信度和祖先回退。不运行事实抽取、原子拆分、置换或额外推理对照。当前候选处置、原问题和选项不进入模块二输入。

其他实施范围：祖先中只有规则导航指令时不构造虚假的治疗历史。行为目标使用均值、低频均值、固定随机方向的七种组合，不是全向量梯度训练。
