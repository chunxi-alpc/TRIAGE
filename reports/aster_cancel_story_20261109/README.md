# 对消，而非擦除

明天汇报看：**[10 分钟汇报稿](meeting/meeting_brief.zh.md)**。

完整阅读看：**[论文正文](paper_story_latest.zh.md)**。

五模型、三任务的预设模拟主实验与消融已移到**[正文第 10 节](paper_story_latest.zh.md#five-model-simulations)**，全文不再使用模拟附录。第 1 节给出动机、设计与实验的对应表，第 5–6 节解释方法和成立条件，第 7 节分类介绍模型、任务、基线与消融，第 8 节报告真实开发集、留出测试、核心消融与非线性对照，第 9 节为实际合成机制实验。

当前主方法用 A/P 差分拟合干扰，用 C/F 配对约束证据修正，再单独校准阈值。真实开发集 ER=99.82%、干扰 VR=4.58%；留出测试 ER=99.60%、VR=4.10%，路径 767/806。五模型表仍是旧版无约束分数预设，表旁明确标注，不作为当前方法的真实跨模型成绩。

3,100 个留出测试输入、806 条真实路径、核心消融和三种子非线性对照已完成；测试集此前用于旧版方案评价。医生抽样核验仍待完成。其余文件仅保存配套数据、图片与复算材料；有约束拟合与实际结果见 [meeting/evidence_preserving](meeting/evidence_preserving/)。

缓存优化扩展见正文第 8.10 节：平衡配置测试 VR=14.59%、ER=99.77%、证据修正 RMS=0.2403、路径768/806；较强保护配置 VR=8.02%、证据修正 RMS=0.1201。完整搜索、模型和校准消融见 [optimization](meeting/optimization/)。

<!-- aster-official-expansion-readme:start -->
官方 test 扩展见正文第 8.11 节，覆盖 7,458 条路径。开发集错因、所有固定方法的路径及叶节点结果、来源分组和配对差值区间见 [test_expansion](meeting/test_expansion/)。原 806 条留出测试与模拟结果保留原口径。
<!-- aster-official-expansion-readme:end -->

<!-- aster-primary-full-readme:start -->
主方法全量真实任务的已完成结果与待完成阶段见正文第8.12节及 [primary_full](meeting/primary_full/)。NLI为零样本迁移，MEGA train_val保留既有fit/dev/原测试分组；缺少真实数据的三段论仅保留模拟。
<!-- aster-primary-full-readme:end -->

<!-- aster-faithfulness-readme:start -->
行为忠实性主指标、配对覆盖与方法局限见正文第8.13节及 [faithfulness](meeting/primary_full/faithfulness/)。既有MEGA配对与NLI原生变体分别报告；官方MEGA首节点A/P按实际覆盖更新，C/F和F0缺项明确保留。
<!-- aster-faithfulness-readme:end -->
