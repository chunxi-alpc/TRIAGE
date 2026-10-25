# MEGA 临床开发结果的独立 CPU 汇总

本目录不调用模型、不发送裁判请求、不读取数据集问题或参考答案。只对已完成裁判评估的逐例原生分数、冻结病例/指南分组、生成及构建记录进行汇总。本地没有运行测试。协调器已报告三个合成测试在 115 服务器通过，日志为 `runtime/test_summarize_mega.log`；尚无真实临床汇总。以下为服务器命令。

```bash
python experiments/aster_seka_execution/clinical_statistics/test_summarize_mega.py -v
python experiments/aster_seka_execution/clinical_statistics/summarize_mega.py --spec /absolute/path/summary_spec.json --output-dir /absolute/path/new_summary
```

需要同级 `data_audit/paired_statistics.py`；其他布局可用 `--statistics-dir` 指定。输出目录必须新建，失败也保存 `summary.json`，不会覆盖先前失败。这里没有预填的实际结果。

spec 路径可为绝对路径，或相对 spec 文件。每个方法显式列出评估名称、原生成名称和条件分支，避免 A0/A1 的同名融合方法被合并。例子是输入格式，不代表实际任务已完成：

```json
{
  "freeze_manifest": "/run/frozen_development.json",
  "primary_run": "/run/primary",
  "methods": [
    {
      "name": "A1B1",
      "generation_method": "A1B1",
      "prediction": "/run/primary/A1B1.predictions.private.jsonl",
      "condition": "A1",
      "evaluation_dir": "/run/evaluation_primary",
      "evaluation_method": "A1B1",
      "calibration_jobs": ["seka_projection"]
    },
    {
      "name": "A1_fusion_static",
      "generation_method": "fusion_static",
      "prediction": "/run/extra_A1/fusion_static.predictions.private.jsonl",
      "condition": "A1",
      "evaluation_dir": "/run/evaluation_extra",
      "evaluation_method": "A1_fusion_static",
      "calibration_jobs": ["seka_projection"]
    }
  ],
  "comparisons": [{"method": "A1B1", "reference": "A1_fusion_static"}],
  "calibration_jobs": {
    "seka_projection": {
      "family": "seka",
      "run_manifest": "/run/projection_job/run_manifest.json",
      "scope": "Actual shared SEKA positive/negative projection construction"
    }
  },
  "generation_processes": [{"run_manifest": "/run/primary_job/run_manifest.json"}]
}
```

`name` 是本次公开汇总名称；`generation_method` 必须匹配预测行；`evaluation_method` 必须匹配 `evaluation_config.json` 的实际映射。后二者省略时等于 name。`condition` 为 A0、A1 或 full_guideline 的 null，必须匹配生成协议。每个方法都必须显式提供 calibration_jobs 数组，没有则为 []。构建 family 使用 seka、pasta、adaseka、fusion；四专家通常为四个独立 adaseka job。A1 条件方法需要计入 SEKA 构建，PASTA/Ada 方法另计对应构建，失败构建可登记在全局表中，状态与成本照实保留。历史 PR33 参数转用如没有可核验的原校准成本，会明确列出 fusion 成本缺失；不能由 [] 变成免费校准。

统计使用全部冻结病例，Acc 为原生连续归一化分数且越高越好，PCC 为原生 DTW 路径成本且越低越好。解析/裁判失败由评估器记固定最差分数，汇总核对这些值，不删病例；若 case_metrics 缺行、重复、指纹变动或裁判尚未结束，则停止并记录失败，不自造裁判结果。均值与方法差的 95% 区间按完整来源指南组重采样；只有一组时保留点估计并说明区间不可计算。连续指标只给改善/退化/不变，不称二元纠正/退化。仅实际 seed=0 开发结果，不冒充种子 0/1/2 的冻结独立评估。

成本有三种独立口径，不相加：

- 方法应承担的推理：实际 B 输出（复用时追溯原输出）、所选 A 条件预测、每份指南抽取缓存一次。完整批次内这等价于将抽取成本均摊到共享该缓存的病例。记录内层 elapsed 与外层 wall 同时存在时只取 enclosing wall，不重复相加。
- 实际生成调用总账：各已提供运行目录的所有计划方法，包括负对照、额外 nohook 和数值探针、条件控制；每个真实文件行仅记一次，复用行按新增调用 0 记账。它不等于各方法承担成本之和。
- 构建、裁判、整个生成进程成本：分别按真实 manifest/评估目录唯一登记。进程时间含加载、哈希等，与逐调用时间重叠；不能再次相加。未提供进程 manifest 时不估算开销。逐例显存取保存的最大值，不能恢复首次 reset 前加载峰值。

缺失的尝试次数、耗时或条件成本保持 null/缺失标记，不由完成次数推定尝试次数。共享记录副本不会重复计入成本。数值零强度只汇总实际保存的漂移标量及输入一致性记录；完整词表分数未保存时不声称独立复算。A0/A1 方法比较明确标出条件信息是否相同；本地 Qwen 裁判与论文 GPT-5.4 Mini 不同，也可能与生成模型共享检查点，所有输出明确禁止独立模型验证的结论。

三个合成测试覆盖连续指标方向与全部失败分母、按指南共享抽取与 A1 成本、复用不免费且总账不重复、缺失裁判行/预测 hash 改变拒绝、数值诊断缺失及未知尝试次数不补造。
