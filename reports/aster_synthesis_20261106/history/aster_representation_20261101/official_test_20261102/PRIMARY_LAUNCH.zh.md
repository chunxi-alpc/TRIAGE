> Source snapshot: /home/gcx/aster_representation_20261101/official_test_20261102/PRIMARY_LAUNCH.zh.md
> Dates and execution states below refer to that original report. Relative artifact paths refer to the source run.

# 主方法已确定并优先运行

记录时间：2026-11-02 08:38（北京时间）。

主方法为去掉JS不变性损失的处置盲表示干预，实验键保留no_js；检查点为no_js_layer18/epoch3.pt，层18、秩4、第3轮。仍在原指南和处置交换指南上训练正确分支，只去掉额外的JS项；本次直接复用已训练权重。

GPU 0、1已优先运行主方法。官方测试共7,458例，切换时已保存153例，余7305例。成功与失败的已保存记录都跳过，没有重新生成已完成预测。其他方法的已有结果全部保留，工作顺序后移；空闲卡在主方法没有可领取任务时可继续已完成主方法指南的对照，不等待另一张卡。

启动前已报告每卡预计25–45GB显存，剩余主方法约3–8小时。两卡目前均在primary阶段，仅执行no_js，已产生新的保存记录，当前未见技术失败。稳定后停止持续监控。

主方法完成后自动保存primary_summary.json和PRIMARY_REPORT.zh.md，以原指南完整路径正确率为主指标。全部对照完成后再生成完整配对比较，报告主方法相对普通低秩、上下文及盲表示分类器的纠正、退化与净提升。

主方法改选依据开发与内部确认结果，改选时没有查看正在运行的官方效能汇总。原冻结记录和含JS版本均保留，内部确认已参与选择。方法描述及论文已同步更新；普通低秩与上下文对照仍含JS，这一差异将在比较中说明，本次不扩展训练。
