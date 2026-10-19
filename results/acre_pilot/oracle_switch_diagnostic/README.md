# Oracle 目标切换诊断（2026-10-15）

状态：E0/E1 实际完成且独立复算。主对照不支持 OS 优于 OF，停止 E2/E3 新实验和独立验证。Oracle 均不可部署。正式中文评审见 ../../../docs/ACRE/08-oracle-switch-results.md。

β=2：O0 64/80；完整 OF 68/80、31/40 对、OAR 2/40；完整 OS 66/80、30/40 对、OAR 3/40。仅目标规则 OF-target 65/80，OS-target 68/80，是次要机制诊断，不能替代主比较。

## 复现

所有命令只在 gcx@10.29.128.115 的 /home/gcx/TRIAGE 运行，source /home/gcx/gcx261014/bin/activate，HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。新 checkout 先将原有数据和模型放到原路径。

CPU 审计：python scripts/audit_acre_switch_masks.py。它把历史归档复制到本目录 _restored 后，用已提交的 unpack_acre_pilot_artifacts.py 安全恢复，验证哈希和缓存链接，不写旧实验。

GPU 首先按 README 向用户报告资源；随后 CUDA_VISIBLE_DEVICES=0 python scripts/check_acre_oracle_operator.py，CUDA_VISIBLE_DEVICES=0 python scripts/run_acre_oracle_factorial.py --smoke。完整启动：bash scripts/launch_acre_oracle_factorial.sh，六 worker 映射 GPU 0、1、2、3、6、7。已完成 worker/case 不允许覆盖；复现须另外归档本目录或在隔离 checkout 使用新副本。

离线汇总：python scripts/summarize_acre_oracle_factorial.py --from-archive（直接使用已交付压缩记录）；完整重跑后可省略该参数以合并 worker 原始文件。独立复核：CUDA_VISIBLE_DEVICES="" python scripts/verify_acre_oracle_factorial.py。中文报告与停止记录：python scripts/report_acre_oracle_factorial.py。

只验证交付的压缩记录时，运行独立复核脚本会直接读取 case_effects.jsonl.gz，不需要加载 GPU 模型。worker/done 和详细日志不提交；若这些文件未恢复，资源退出检查标为“仅复核已提交的完成凭证”，不伪造新运行。

## 结果入口

- mask_audit.csv、mask_audit_summary.json：80 例旧 B2/B3 最终集合、角色、Jaccard 与互斥类别。
- oracle_mask_audit.csv、oracle_mask_plan.json：同一参考来源下完整与仅目标规则对照。
- annotation_error_audit.csv：1250 条规则，无失败病例排除；字符重叠精确性仅为注册决定句代理，非临床语义精确率。
- oracle_factorial_metrics.json：13 条件、全部和分层指标、固定边际、配对统计。
- case_effects.jsonl.gz：1040 条记录，全部预测、logits、目标位置、提示哈希和隔离标记。
- prediction_changes.csv、paired_comparisons.csv、error_analysis.json、stratified_margins_and_lengths.json：所有正负变化、四格、指南聚类区间、分层方向及目标长度。
- control_validity.json、factorial_metrics.json、upstream_bottleneck_analysis.md：E2/E3 未执行原因。
- resource_plan.json、smoke_checks.json、operator_gpu_checks.json、recompute_check.json：计划、真实资源与验收。
- data_lineage.json、historical_fingerprints.json、final_code_fingerprints.json、run_notes.json：版本、IDs、读取点、哈希、历史保护与工程修复。

CPU 审计可恢复缺失的未压缩冻结输入；已存在输入须字节相同，不覆盖任何已有历史结果。

CSV 含 JSON 字段和长原文。使用 Python csv 读取时先设置 csv.field_size_limit(sys.maxsize)，不能按换行手工切分。JSONL 需通过 gzip.open 逐行读取。本批开发结果不支持独立验证、等效性或部署性能主张。
