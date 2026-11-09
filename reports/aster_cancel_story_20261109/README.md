# 对消，而非擦除

完整阅读：[六章正文](paper_story_latest.zh.md)。汇报：[10分钟简版](meeting/meeting_brief.zh.md)。

正文依次为Introduction、Related Work、Method、Experiments、Analysis、Conclusion。实验仅设4.1设置、4.2实测、4.3医学模拟；分析仅设5.1消融与非线性、5.2忠实性与错误。Limitations、Ethical Considerations、References之后接附录A–D。

忠实性是主要指标，准确率为下游效果。正文保留受控留出、官方测试、固定错误池及已有MedPIC冻结迁移；详细证明放A，数据实现放B，完整实测与历史优化放C，合成机制与新旧预设模拟放D。

七个医学数据集的官方/本地/实测规模、原生划分见4.1。新增五模型、五医学任务、六条件、五种子模拟保存在原[simulated_study](simulated_study/)目录，明确标注预设模拟、待真实评测替换，不覆盖旧结果。

本次仅CPU文档与模拟工作，没有运行真实模型或重复MedPIC实测。

方向退化同时报告退化率及原正向响应分母。医学模拟占位采用共享、随证据变化的修正特征，固定原系数及种子计算，不设定退化数量；旧规则及结果归档在[medical_extension_v1](simulated_study/medical_extension_v1/)。CPU模拟及计分验证已完成，真实五模型评测待执行；用[表格同步脚本](simulated_study/render_medical_extension_tables.py)从CSV更新正文、附录和简版。

论文表格采用统一的Method名称、指标符号、单位、评价方向及完整表注；定义与分母见正文4.1及附录B.1。正式名称映射保存在[paper_table_style.json](paper_table_style.json)，由[格式同步脚本](format_paper_tables.py)维护。主结果的[ACL表格源码](tables_acl.tex)使用booktabs与tabularx，可放入ACL官方模板；本次未编译完整双栏PDF。模拟仍按逐表标识保留为待替换数值。
