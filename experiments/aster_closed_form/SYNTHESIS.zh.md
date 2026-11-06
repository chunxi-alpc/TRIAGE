# 三轮闭式实验与全历史结果整合

> 本目录实现已完成的闭式三轮实验。最新方案保留 M1、沿用此前选定 M2 的全部未筛选方向，仅改造 M3，见[当前方法](../../docs/ASTER/manuscript.zh.md)。新 M3 尚未实现、训练和验证；下列复算命令仅重现历史结果。

本次整合将现成数据硬筛选、连续衰减、未筛选方向的同事实参考代码和结果放入同一仓库，复用保存分数生成总表，不新增模型推理。

[总分析](../../reports/aster_synthesis_20261106/ANALYSIS.zh.md) · [全部阶段](../../reports/aster_synthesis_20261106/EXPERIMENTS.zh.md) · [数据表](../../results/aster_synthesis_20261106/README.md)

## 服务器 CPU 复算

在仓库根目录激活已有实验环境：
```bash
source /home/gcx/gcx261014/bin/activate
CUDA_VISIBLE_DEVICES='' python -m experiments.aster_closed_form.recompute_synthesis --repo-root .
python reports/aster_synthesis_20261106/build_report.py
CUDA_VISIBLE_DEVICES='' python -m unittest experiments.aster_closed_form.test_operators experiments.aster_closed_form.test_soft_angle experiments.aster_closed_form.test_evaluation
```

复算仅依赖仓库的已保存输入和分数，支持 .jsonl.gz，按来源重采样 2000 次、种子 20261105。它重新计算 7 方法的 fit/dev/test/all 主表、逐输入纠正／退化、机制修补、历史输入和路径，并核对原主表和完整分母。27 项算子、连续衰减、评价测试已通过。

## 运行代码

- pipeline.py、existing_data.py、existing_report.py：原硬筛选流程与现成数据复用。
- soft_angle.py、soft_angle_report.py：冻结基上的连续衰减、机制和自主路径。
- reference_anchor.py、reference_anchor_report.py：同事实参考映射、激活缓存、全局均值／同事实投影、历史模板与逐例诊断。
- recompute_synthesis.py：从发布结果做 CPU 汇总。
- reports/aster_synthesis_20261106/build_report.py：从汇总表生成报告、索引和入口。

原实验运行脚本仍保留其服务器目录配置；重新进行模型前向时需要服务器上的 Qwen3-8B 权重、原来源产物和实验环境。它们不是在任意本地机器下载即跑的模型包。CPU 汇总不依赖这些服务器绝对路径。

服务器完整产物：
- /home/gcx/aster_existing_data_20261106
- /home/gcx/aster_soft_angle_20261106
- /home/gcx/aster_reference_anchor_20261106

本轮没有重新拟合基、选择参数、生成标注或运行新参考方法的自主路径。原历史报告保持版本归属；本文所述模块编号只指当前闭式版本。
