# 全量 AdaSEKA 的独立 CPU 比较

本入口只读取既有原生 CounterFact 匿名导出、运行 manifest、输入 trace、四专家文件和公开 CounterFact 实体分组。不会加载模型或运行推理。本地未运行代码/测试。协调器报告首三项合成测试在服务器通过；首次真实 CLI 在最终源码指纹登记处因向只接收 Path 的辅助 sha 传入字符串失败，未写成功报告，原目录/日志保留。已修复本文件及辅助模块 __file__ 的两处转换，并增加第 4 项 CLI 成功/异常收尾测试；新测试及真实 v2 结果待服务器执行。

方法固定命名为 `adaseka_paper3`、`adaseka_default1`、`adaseka_zero`，强度分别为 3、1、0，明确都是作者 AdaSEKA 实现的 FP32 运行。强度 3 的标签仅指所用预设，不能据此声称完整复现论文。efficacy 比较三个方法的全部三对；paraphrase 仅比较两个真实有结果的正强度方法，不补造零强度结果。

服务器合成测试：

```bash
python experiments/aster_seka_execution/adaptive_statistics/test_compare_full_adaseka.py -v
```

默认读取同级 `independent_audit/{compare_standard,compare_adaptive_counterfact}.py` 和 `data_audit/counterfact_entity_ci.py`。若文件都暂放服务器 p 目录，测试可使用 `PYTHONPATH=p python p/test_compare_full_adaseka.py -v`；实际入口加 `--audit-dir p --statistics-dir p`。本代码直接复用真实 `bootstrap_effects`、原匿名导出读取函数，以及已经核对的 Ada `input_match`/参数解析辅助函数，未调用原脚本只允许 SEKA 标签的 CLI。

实际统计命令参数如下；导出根目录应含 `efficacy/audit.json`，paper3/default1 还应含 `paraphrase/audit.json`。使用新的输出目录保留既有报告和失败。

```text
python compare_full_adaseka.py
  --data <official counterfact.jsonl>
  --paper3-export-root <paper3 verified root>
  --paper3-manifest runtime/adaseka_cf_full_paper3_job/run_manifest.json
  --default1-export-root <default1 verified root>
  --default1-manifest runtime/adaseka_cf_full_adaptive_job/run_manifest.json
  --zero-export-root <zero verified root>
  --zero-manifest runtime/adaseka_cf_full_matched_zero_job/run_manifest.json
  --output-dir <new output directory>
```

实际入口固定验证原生 `train[5000:10000]` 的 5000 个 ID。核对完成状态、观察到的 FP32、batch=1、真实 Ada 标志/强度、相同检查点/作者源码/四专家、数据指纹、匿名导出与 manifest 绑定、全部 attempted/completed forward 及完整 trace。每个 efficacy 病例是路由一次加生成一次；paraphrase 是每个改写提示两次，不能把多提示病例写成总共两次。官方 wrapper 强制生成一个 token，命令中被覆盖的 max_new_tokens 只作记录，不错误拒绝有效结果。

路由 attention_mask 明确为 null 时可与 null 匹配；字段缺失仍为缺证据，null 不等于 tensor。按原导出的逐例/逐提示顺序分配真实 trace，逐条匹配 input_ids 和 mask 的 hash/dtype/shape。efficacy 按病例匹配；paraphrase 按病例及 prompt hash 匹配，并单列整个分段的调用顺序是否相同。batch=1 下提示顺序不同不会被冒称有序完全一致，也不会掩盖各提示实际输入的匹配结果。

从已保存 target/comparator log 概率再次计算每提示胜负及概率差，再核对病例均值。efficacy 给出 native delta、纠正和退化；paraphrase 保持每病例各提示胜率的连续分数，只称病例改善/下降。区间使用公开 entity 字段的精确相同值分组，整组 bootstrap，按病例加权，seed=0、默认 10000 次；均为固定既有运行条件下的边际 95% 区间，不作临床或多重检验显著性结论。

每对输出完整匿名逐例变化和逐提示路由/生成输入核对文件，全部负结果保留。过程总耗时/显存包含该运行的所有 benchmark，不能从调用比例臆测 efficacy 的单独耗时。计数可按实际分段精确拆开，路由成本单列。

FP32 nohook 目前仅在另行验证的 16 例有证据，本入口不扩展成 5000 例 nohook 对照。完整词表 logits 未保存，仅比较可得目标词/比较词概率，不声称全词表零漂移。四个合成测试覆盖负结果与实体区间、改写连续指标、实际输入匹配及顺序披露、显式 null/缺字段区别、FP32 与完整 forward 计数，以及真实 CLI 收尾不会掩盖原始异常。
