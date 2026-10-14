# TRIAGE 三模块最小实现

运行位置 `/home/gcx/TRIAGE`，环境 `/home/gcx/gcx261014`。模型复用现有 Qwen3-8B BF16；未部署新环境、下载新模型、人工标注或训练主模型。模块一代码未重写。

## 代码与入口

- 模块一真实接口：`from src import module1; module1.run(guideline=R, patient=P, question=Q)`；输出 `module1.v1`。
- 模块二：`src/module2/data.py` 从真实模块一缓存选择规则、条件与证据，构造单条件反事实；`src/module2/localizer.py` 用 `o_proj` 前向 Hook 消融选定查询头，测量适用性判断的 A/B 分数差。
- 模块三：`src/module3/intervention.py` 从当前模型选定组件的少量激活构建低秩投影；在共享键值头展开后，仅编辑选定查询头的键向量，随后正常执行 RoPE 和注意力计算。先验竞争通路在 `o_proj` 前按门控减弱输出投影分量。
- 统一入口：`src/pipeline/runner.py` 的 `TriagePipeline`，命令行 `scripts/run_pipeline.py`。实时运行模块一后复用同一 Qwen 模型进行原始及干预推理。
- 位置映射：`src/pipeline/tokens.py`，在真实聊天模板生成后使用 fast tokenizer offsets，映射原始规则全文、条件、动作及患者证据。患者引用须与原文字符区间完全一致；规则 source_span 的条款内坐标与最终提示坐标分别保存。

服务器使用方法：

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py \
  --input /absolute/path/input.json --output results/pipeline/inference.json
```

输入 JSON 必须含 `guideline`、`patient`、`question`；规则输入格式沿用模块一。默认加载 `results/module2/components.json` 与 `results/module3/projections.pt`。投影须提前从独立校准样本构建，不会从最终推理样本自动估计参数。

## 缓存定位与参数

模块二最终校准仅使用 MEGA-CDP train_val 的 guideline_id=3，缓存来源为 `results/module1_output.json` 与 `results/module1_smoke.json`；快照保存于 `results/module2/module1_calibration_cache.json`。搜索期间模块一调用次数为零。

缓存的适用标签用于离线分组，正 binding_gate 的证据位置用于选择消融词元；评分提示只含规则、患者和问题，不输入适用标签或 action_gate，避免直接透露判断结果。两组对照自动替换同一必要条件的证据句，来源、原句和反事实近似门控均有记录。

最终候选使用从零开始的编号：层35/头8、层12/头24；因果分数分别为 4.0625、1.125。分数定义为“适用样本的原始减消融 A/B 边际变化”减“不适用样本的原始减消融 A/B 边际变化”。第一组件仍有正的误用贡献，故仅为暂定候选。

模块三用上述头与 gid3 的实际规则/证据提示构建 rank=4 投影。该实现借鉴 SEKA 键向量编辑和谱投影思路；投影为少量激活的 SVD，尚非官方完整对照协方差构建方法。输出投影作为先验竞争的近似方向，不是已经证明的纯先验方向。未下载 SEKA 额外数据。

默认 context_strength=0.1、prior_strength=0.1。规则不适用或未知时 action 门控为零；先验抑制另要求明确 prior=oppose、guided=support、disagreement 和关键证据置信度至少0.8。门控仍未校准。

## 运行记录与限制

- `results/module2/components.json`、`calibration_pairs.json`、`report.json`：组件分数、反事实、真实缓存指纹与资源记录。
- `results/module3/projections.pt`、`projections.json`：实际参数及模型、组件、校准来源指纹。
- `results/module3/cpu_checks.json`、`gpu_checks.json`：实际计算改变、共享键值头作用范围、关闭门控及低置信度检查。
- `results/pipeline/smoke.json`：独立 train_val 病例 gid5/9 的实时模块一输出、原始与干预决策、门控、词元映射及干预记录。合成门控控制检查单独标记，不能视为患者真实冲突。
- `results/pipeline/coverage_smoke.json`：gid5 固定营养问题相关官方子树的追加功能检查，保留全部继承条件与三条叶规则；不修改组件或投影。完整指南 top_k=3 漏选适用规则的原始结果保留，不把缩小输入范围后的输出当作准确率提升。
- `results/triage_mvp_report.json`：最终汇总与验收记录。

缓存定位的基线适用判断为3/4正确，包含一次不适用规则误用。小样本只验证接口与计算路径，不代表临床准确率提升。关系级成对注意力偏置、校准门控、更充分组件搜索及完整基准尚未实现或开展。本次不用测试集调参。

## 最终验收

真实端到端推理共三次（两个独立病例，其中 gid5 追加一次固定问题分支检查），模块一均实时计算，同一 Qwen 实例复用。原完整指南的两次 top_k=3 推理全部 action_gate 为零；因此真实计算与生成均保持相同，原始输出与证据冲突仍保留。固定完整营养分支检查的 leaf5 action_gate=0.486737，实际修改 11,576,171 个键向量元素，下一词元最大分数变化0.5，零门控仍精确一致；两份最终决策相同。实际患者没有有效先验冲突，先验通路仅在明确标注的合成控制中验证执行。

最终校准 gid3 与最终推理 gid5/9 无指南重叠。模块二峰值15.30 GiB，模块三16.47 GiB，统一推理16.28 GiB，GPU0/1/2运行均已结束，无持续监控。原始两例共151.40秒，追加检查89.24秒；模型加载及开发时间另计。
