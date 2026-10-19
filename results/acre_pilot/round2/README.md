# 第二轮数据与失败解码记录

此目录根部的 metrics/annotation_metrics/case_results 是 source_ids_json_grammar.v2 的完整失败实验，存在状态首词元不对称问题，不能用于最终方法评价。

最终结果在 revision3/，协议 v3 将所有状态引号固定在语法中。开发样本、原始路径和 oracle 在本目录固定保存，参考动作别名是公开的事后 OAR 计数修正，详情见 data_audit/action_alias_audit。版本修正未改患者、原始决策路径和正常输入。

压缩文件可用 scripts/unpack_acre_pilot_artifacts.py --directory results/acre_pilot/round2/revision3 恢复。annotations 归档还包含冒烟使用的一个历史病例缓存，主要80例评价只使用冻结 manifest 中的病例。完整结论与复现命令见 docs/ACRE/06-pilot-results.md。
