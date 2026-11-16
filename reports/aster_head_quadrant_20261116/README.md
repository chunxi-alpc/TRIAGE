# A/P–C/F 注意力头四象限诊断

任务：[GitHub #72](https://github.com/chunxi-alpc/TRIAGE/issues/72)。

本目录保存 Qwen3-8B 的 E1 五折缓存分析，计算使用 CPU。主图使用每折训练来源的全头激活；梯度归因与精确修补直接读取既有 M1 结果。

## 文件

- `REPORT.zh.md`：结果、计算定义、分母与解释。
- `FIGURE_head_quadrants.png/pdf/svg`：三种互补视图。
- `FIGURE_head_quadrants_folds.png/pdf/svg`：五折激活移动与 M1 选择。
- `head_quadrant_metrics.csv`：5 × 1,152 行折内指标。
- `head_quadrant_consensus.csv`：1,152 行共识指标。
- `quadrant_summary.json`：象限、稳定性、层分布、相关性与覆盖率。
- `configuration.json`：本次分析约定。
- [分析脚本](../../scripts/aster_head_quadrant/analyze.py)、[绘图脚本](../../scripts/aster_head_quadrant/plot.py)、[单元测试](../../tests/test_aster_head_quadrant.py)。

## 在服务器复现

在仓库根目录执行，使用已有实验环境：

```bash
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python scripts/aster_head_quadrant/analyze.py \
  --input-root /home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111 \
  --output-dir reports/aster_head_quadrant_20261116
python scripts/aster_head_quadrant/plot.py \
  --report-dir reports/aster_head_quadrant_20261116
python -m unittest discover -s tests -p test_aster_head_quadrant.py -v
```

分析使用服务器的只读原始缓存；已提交的 CSV 足够单独重绘图片。无需模型权重、GPU、模型前向、梯度计算或重跑 M1。运行依赖为 NumPy、SciPy 与 matplotlib；实际环境版本记录在报告中。

## 阅读约定

AP selective、CF selective、Mixed、Low change 分别表示 A/P 高且 C/F 低、A/P 低且 C/F 高、两轴都高、两轴都低。高/低依据各共识轴的第75百分位；第50百分位用于独立敏感性分析。

百分位坐标衡量该视图内头的相对位置。精确修补面板要求同一折 A/P/C/F 四类分数齐全，并至少有三折完整记录；每类分数先在原64候选头中取百分位，再计算两类平均。空值保留为缺失。

共识图圈出在至少3/5折入选 M1 的头，五折图圈出当折的51头。不同折选择集合的变化见 CSV 与摘要。
