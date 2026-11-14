# GitHub交付说明

本目录收录5种新增方法的11次实际冻结测试结果。入口为[结果与结论](REPORT.zh.md)，[逐运行主表](section4_5_new_methods.csv)和[方法均值与种子标准差](section4_5_method_summary.csv)。

evaluation/test/保存11次完整测试；evaluation/dev/保存5次已执行的完整开发评价。RePS和PSR的开发选择分数、候选强度与冻结配置位于各自种子目录。

逐例JSONL及逐对/路径CSV以gzip无损压缩保存，文件名追加.gz；可用Python gzip.open读取。其余CSV和JSON保持原格式。官方版本和实现差异见official_sources.json与各方法配置，早期准备记录中的pending/not_started表示记录当时状态，当前完成状态见status.json。

大型权重、教师激活和完整输入缓存保留在服务器，位置见[SERVER_ARTIFACTS.json](SERVER_ARTIFACTS.json)。环境版本见[environment_versions.json](environment_versions.json)，运行入口见[代码说明](../../experiments/aster_codex03_external_20261112/README.md)。
