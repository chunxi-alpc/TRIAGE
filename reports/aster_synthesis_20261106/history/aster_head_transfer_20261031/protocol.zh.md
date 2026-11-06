> Source snapshot: /home/gcx/aster_head_transfer_20261031/protocol.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 固定对照头的相同偏置对照

本方案在本轮GPU结果产生前登记，承接上一轮在线关闭对照的结果，不根据成绩搜索新头。仅新增 control_bias_4 一组：在上一轮已固定的同层四个对照KV头（23:5、31:2、33:5、33:6）上，对相同规则ID与动作词位置施加金标准路径偏置，增益4。

将原四头冻结开发阶段的bias_unit_rms按同层既定一一对应复制到对照头，保持每层的头数、每头每个目标位置的加性分数偏置相同；不重新拟合方向，不在43例上校准。不同头原始注意力分布可以不同，因此不声称注意力概率变化量完全相等。

使用相同43例、两个来源、输入、模板、BF16、分块eager512、贪心和512输出上限。复用上一轮七组301条完整结果。先用每来源一例核对对照头零偏置接口完整输出复现，共43+2=45次新生成，另41次零编辑首步。主要比较control_bias_4对bias_4和zero的配对终点与完整路径，不将关闭头的收益等同于偏置收益。

使用GPU0、1，每卡预计35—40GB，3—6分钟。稳定后停止资源和进度监控，仅等待进程结束。官方评测保持暂停，原冻结文件与历史结果不变，不自动恢复。
