# Codex 03 新增方法运行与结果

仅收录新方法真实完成的冻结评价；开发结果、smoke和CPU夹具不进入此表。

| GPU | 队列状态 | 当前任务 |
|---|---|---|
|gpu5|complete|—|
|gpu6|complete|—|

本轮5种新增方法的11次正式实验和冻结评价全部完成。

完成的测试运行：CAA-E1-task-adapted, DoRA-E1-SFT-seed11, DoRA-E1-SFT-seed22, DoRA-E1-SFT-seed33, INLP-task-adapted, PSR-MSE-E1-task-adapted-seed11, PSR-MSE-E1-task-adapted-seed22, PSR-MSE-E1-task-adapted-seed33, RePS-E1-task-adapted-seed11, RePS-E1-task-adapted-seed22, RePS-E1-task-adapted-seed33。

逐运行数据：[section4_5_new_methods.csv](section4_5_new_methods.csv)。
跨种子均值与标准差：[section4_5_method_summary.csv](section4_5_method_summary.csv)。
各方法的逐例预测、训练配置、成本和来源级区间位于 evaluation/test/<method>/。
