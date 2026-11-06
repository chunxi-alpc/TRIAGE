# ASTER：仅复用已有数据

本轮为 2026-11-06 的 Qwen3-8B 闭式 ASTER 实验。执行入口为 `experiments.aster_closed_form.existing`。它读取已登记的输入、配对和缓存，不调用病例构造、条件重写或 MedGemma。

[分析报告](../../reports/aster_existing_data_20261106/ANALYSIS.zh.md) · [完整结果包](../../results/aster_existing_data_20261106/README.md)

## 从保存分数复算表格

从仓库根目录运行以下命令即可用 CPU 重算主表、可用子集、历史诊断和路径表。脚本直接读取随仓库发布的 JSON/JSONL/GZIP，不加载模型，不依赖服务器激活缓存。输出单独写入指定目录。

```bash
source /home/gcx/gcx261014/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.aster_closed_form.recompute_existing \
  --root results/aster_existing_data_20261106 \
  --output /home/gcx/aster_existing_tables_recomputed
```

这一步复算评分表，不重新拟合算子、重做机制修补或运行 H3 探针；相应既有结果也已发布。

## 服务器模型执行

原运行保存在 `/home/gcx/aster_existing_data_20261106`。完整重跑需要注册的两个历史运行、MEGA-CDP 源文件和 Qwen3-8B 权重：

- `/home/gcx/aster_closed_form_20261105`
- `/home/gcx/semantic_masking_followup_20261105`
- `/home/gcx/TRIAGE/data/MEGA-CDP/train_val`
- `/home/gcx/TRIAGE/models/modelscope/Qwen3-8B`

从仓库根目录运行 `python -m experiments.aster_closed_form.existing --root <新的输出目录>`。程序使用 GPU 0、1，每卡一份模型；启动前应报告资源、规模与时长。默认模型阶段包括 18 配置开发评分、消融、机制、历史诊断和 432 例自主路径。每次重跑使用新目录保留本轮结果。

模块目录同时保留原闭式实验依赖和历史构造实现；本轮仅复用入口为上述 `existing`，不是旧的 `pipeline` 全流程。

## 本轮固定协议

200 节点按 80／20／100 拟合、开发、测试划分，种子 20261105。移除事先登记的 7 条错误及 2 条有歧义 C 变体，其余视图不动；共 1,290 输入、1,053 配对。M1 及不依赖新算子的缓存复用；M2/M3、融合及后续依赖重新计算。所有历史诊断独立报告，不参与拟合或选择。

固定配置为 eta=0.95、kmax=4、theta=45°，51 个头，总编辑秩 120。18 个配置在主要开发指标上并列，开发 C 仅 1 对。运行选择遵循原并列规则，不依据测试结果换参数。

发布时代码只做运行入口适配：工作进程从当前代码所在仓库启动，并自动建立必要输出子目录。这不改变已运行实验的算子、分数、配置或指标。

## 验证

服务器已有输入一致性、失败分母、432／100 路径范围、目标位置与头、零干预以及算子代数测试通过。完整实验实际评分与路径技术失败为 0。详细记录见结果包的 `validation/` 和 `results/status.json`。
