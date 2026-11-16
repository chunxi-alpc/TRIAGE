# N3 基底修改完成

31项CPU测试通过；48次开发前向完成，进程退出0。新路由beta=.5为11/16，mean和旧route为10/16，文本12/16，随机9/16。N2、N5、测试集及原冻结选择保持不变。

[完整结果](../../runs/v4_n3_mean_basis_20261020T151946Z/REPORT.zh.md)。

实际入口：`python -m care_m2_v2.n3_mean_basis prepare|run --out ...`。prepare只从已保存train差分构造两维基底；run只允许16条冻结开发提示且最多48次前向。已有模型/训练/评分与L0证据复用。`launch.py`、`process.json`和`gpu.log`保存真实执行参数、耗时、退出码及输出；`analyze.py`仅复算已保存开发结果。
