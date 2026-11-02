# Gemma 3 27B：全层扫描与选头先导实验

本目录独立实现 Gemma 3 27B 的扫描、训练、确认和汇总，复用 ASTER 的固定病例与 no_js 目标。当前为正在执行的先导实验，没有最终性能结论。Qwen 原入口保持不变。

## 数据、模型和资源

- 来源：公开 MEGA-CDP 的 train_val；沿用登记的200例训练、100例开发、100例确认。另16个年龄状态为原病例派生的8组训练对，不是独立病例或确认集。
- 服务器原始数据目录：/home/gcx/aster_representation_20261101。运行目录的 data/ 保存本轮副本；data_manifest.json 登记编号、来源与计数。
- 模型：ModelScope google/gemma-3-27b-it，master 下载快照，下载时间与文件版本字段保存在 download_manifest.json；不增加哈希检查。BF16，冻结骨干，视觉分支不输入图像。
- 仅物理 GPU 0、1；一个模型按层分布在两张 A100 80GB。实际映射见 architecture.json。使用原生 SDPA、1024局部窗口和原生全局层。
- 默认使用原生整段预填充。分块试跑与整段预填充的最大 logit 差为0.25，最高分词元相同；该尝试单独保留，不宣称数值恒等，不用于正式特征缓存。

## 固定流程

1. 从训练病例根、中间、末端节点采集完整与盲读取在 o_proj 前后的表示；用Y/N均值距离除以组内波动记录表示区分度。
2. 开发集按来源固定取两例，共20例。在根节点扫描62层×16个KV组；每组对应两个查询头。逐组施加正负0.25倍盲输出与完整输出之差，共39,680个有符号干预单元。
3. 用正确分支减错误分支的正向分数增量排序；每层选四组，再按四组平均分选择三层。并列时比较表示区分度、层号、组号。负分不删，反向响应和失败另存。
4. 三层分别训练整层与选头版本，共六模型；固定rank4、no_js、三轮、AdamW学习率0.001、权重衰减0.01、积累8病例、梯度裁剪1、种子20261101。损失沿用分支交叉熵、0.1保持项、0.001编辑幅度项。
5. 开发集选择原指南完整路径正确数最高者；并列时比较两视图均正确数、较小实际编辑幅度、较小层号和较早轮次。整层、选头分别选择。
6. 在选头方案的层，用编号最小的四个未选组训练同规格对照，开发集选轮次。冻结后，100例确认比较零干预、直接盲读、整层、选中头、未选头五组，均评测原指南和处置重排视图。

整层模块修改 o_proj 输出；选头模块仅修改 o_proj 输入中的指定查询头坐标。幅度都在投影后空间计算。两者分别有43,017和8,201个参数；同数量未选头对照也是8,201个参数。整层与选头参数量不同，结果解释中保留此差异。

## 运行

在指定服务器，先 source /home/gcx/gcx261014/bin/activate，再进入本工作树。各阶段保留完整输出，可恢复已有进度；不按成绩重跑。下载权重完成后：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemma3_head_pilot smoke
CUDA_VISIBLE_DEVICES=0,1 python -m experiments.gemma3_head_pilot all
```

可分别执行 prepare、collect、scan、train、confirm、report。训练阶段在优化器更新处保存完整恢复状态；预测与扫描按缺失单元恢复，已有失败不删除。执行 all 后依次完成剩余阶段并自动写出报告，不需要持续资源监控。

显式训练和预测可传 --model、--root、--devices、--layer、--kind、--groups、--checkpoint、--split、--output。groups 是KV组编号，不是投影后的向量片段。例如，选择层及四组由 head_selection.json 读取后，可运行：

```bash
python -m experiments.gemma3_head_pilot train --layer 31 --kind heads --groups 0,1,2,3
python -m experiments.gemma3_head_pilot predict --layer 31 --kind heads --groups 0,1,2,3 --checkpoint /path/to/epoch3.safetensors --split development --output /path/to/predictions.jsonl
```

示例编号仅展示接口，正式任务使用扫描结果，不预先指定31层或0–3组。官方7,458例不包含在本轮入口中。

## 结果和验证

服务器保留 status.json、head_selection.json、frozen_selection.json、results.json、REPORT.zh.md、costs.json、检查点和逐例记录。完整候选排名及两种方向的逐组结果均保留。源病例、患者文字和模型表示不上传公共仓库。

主指标为100例确认的完整路径正确率；同时报告相对零干预、整层和未选头对照的纠正、退化与净增加。技术失败保留在分母内。处置重排沿用历史构造，只解释所测扰动，不把所有重排假定为语义必然等价。

8项真实微型Gemma CPU测试覆盖局部/全局窗口缓存、只改选中头、投影后幅度、零干预、冻结梯度、多词元标签与选择规则。真实27B试跑验证整层/选头零干预与原生缓存路径完全一致，干预参数可反向传播，骨干无梯度。最长训练病例的运行时间与显存记录见服务器 runtime_benchmark.json。
