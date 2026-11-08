# 模拟研究的独立检查

本文件检查模拟表格的数学一致性，不证明实际模型、数据集或基线性能。

## 数值检查

通过：main_seed_metrics.csv simulated status；660 rows
通过：main_seed_metrics.csv finite metrics；all numeric metrics finite
通过：main_seed_metrics.csv variance bound；VR cannot exceed 1
通过：main_seed_metrics.csv R2 bound；R2 cannot exceed 1
通过：main_seed_metrics.csv VR vs R2；same pair residual: VR-R2 is squared residual mean divided by raw variance
通过：main_seed_metrics.csv probability range；correctness rates in [0,1]
通过：main_seed_metrics.csv invariant pair probability bound；pair both correct between Frechet bounds from the same two sides
通过：main_seed_metrics.csv flip pair probability bound；pair both correct between Frechet bounds from the same two sides
通过：ablation_seed_metrics.csv simulated status；585 rows
通过：ablation_seed_metrics.csv finite metrics；all numeric metrics finite
通过：ablation_seed_metrics.csv variance bound；VR cannot exceed 1
通过：ablation_seed_metrics.csv R2 bound；R2 cannot exceed 1
通过：ablation_seed_metrics.csv VR vs R2；same pair residual: VR-R2 is squared residual mean divided by raw variance
通过：ablation_seed_metrics.csv probability range；correctness rates in [0,1]
通过：ablation_seed_metrics.csv invariant pair probability bound；pair both correct between Frechet bounds from the same two sides
通过：ablation_seed_metrics.csv flip pair probability bound；pair both correct between Frechet bounds from the same two sides
通过：main_seed_metrics.csv 5 seeds per cell；132 cells
通过：main_seed_metrics.csv nine scenarios；9
通过：ablation_seed_metrics.csv 5 seeds per cell；117 cells
通过：ablation_seed_metrics.csv nine scenarios；9
通过：main_seed_metrics.csv VR-R2 exact identity；max absolute error=4.4159554485334596e-16
通过：main_seed_metrics.csv RMS-variance-mean identity；max absolute error=8.881784197001252e-16
通过：main_seed_metrics.csv paths only in MEGA scenarios；NLI4CT and syllogism path fields must be blank
通过：ablation_seed_metrics.csv VR-R2 exact identity；max absolute error=6.76975836499949e-16
通过：ablation_seed_metrics.csv RMS-variance-mean identity；max absolute error=1.7763568394002505e-15
通过：ablation_seed_metrics.csv paths only in MEGA scenarios；NLI4CT and syllogism path fields must be blank
通过：constant b leaves main pair metrics unchanged；max absolute difference=0.0
通过：without_b leaves ablation pair metrics unchanged；max absolute difference=0.0
通过：main_summary.csv mean/SD recomputation；max mean diff=0.0; max sample SD diff=0.0
通过：ablation_summary.csv mean/SD recomputation；max mean diff=0.0; max sample SD diff=0.0
通过：ours does not win every metric in every cell；{'VR': 0, 'ER_closest_to_one': 0, 'native_accuracy': 0}

## 解释范围

1. 模型名与任务名在此仅对应参数配置。表格与图中所有结果均须标“模拟”，不得写成实际运行benchmark的结论。
2. 5个seed的标准差只描述生成过程中的随机波动，不是实际模型训练或真实任务抽样的不确定性。
3. 常数截距b不改变配对差分，但能改变以0为阈值的判断；因此去掉b只应影响阈值相关指标。
4. VR与R²使用不同的均值处理。相同残差下，VR-R²等于残差均值平方除以原配对差方差；二者不应被无条件写成完全相等。
5. ER以100%为目标。超过100%说明证据响应放大，并非更好的证据保留。
6. 模拟中的较纯痕迹与接近100%的ER来自额外设定，不能替代真实开发结果94.64%的ER。
7. 输出端线性修正也可能由合适的表示编辑实现。模拟可说明给定写回噪声与约束下的权衡，不能证明所有表示编辑都必然差于对消。
8. NLI与三段论不报告临床指南路径正确率。
