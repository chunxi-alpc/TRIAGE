# 自动结构修复：192 例真实缓存与 GPU 补跑

本轮使用旧实验的同一批 192 例 confirmation（含 24 个标准配对、总计 72 对），保留全部 U、旧解析失败和未发生干预的记录。该数据已被使用，本结果是探索性实验。旧冻结 V7 源码、原始输出和结果均未覆盖。

**结构错误大部分可以通过明确原子命题含义和恢复已生成的最终 JSON 消除。修复后规则正确率从 68.06% 升至 97.22%，直接执行规则的病例正确率从 31.77% 升至 92.19%。这些是结构与确定性执行结果，不能当作 Qwen 注意力干预收益。**

## 固定映射与接口

| 请求中的条件类型 | 模型原始标签 | 原子命题状态 |
|---|---|---|
| 纳入条件 | Met / Not Met | T / F |
| 排除条件（文本本身定义原子命题） | Violated / Not Violated | T / F |
| 任一类型 | Unclear / Irrelevant | U / U |
| 任一类型 | 无可用最终标签 | 复用可恢复缓存，仍缺失则补跑；技术失败保留 U |

例如“患者有暂停标志”作为排除条件时，`Not Violated` 表示该排除命题为 F；规则中的 `UNLESS` 和 `NOT` 之后由逻辑树各计算一次。不能把整个规则是否适用直接当成原子真假，也不能将未知当成 F。原 V7 的数值判断及其优先顺序保留。

每个旧请求只评估一个原子，程序从原请求清单绑定稳定 `condition_id`。旧输出若已有完整最终 JSON，只是后续重复生成被截断，或抄写条件文字有差异，则恢复该完整输出；不从思考文本提取标签。重复完整输出相互矛盾时不强选一个。预测输入只有患者文本、规则和问题；参考结构仅在预测完成后评价，未用于修复。

新补跑接口由程序携带 `condition_id`，Phi-4 真实生成一个受限 T/F/U 词元，再序列化为短 JSON。这样不再要求模型重复条件文本，也不会因长解释达到 2048 词元上限。该约束仅用于结构分类输出，不是给最终路由答案加分，也没有置信度或证据准入门控。

## 结果与分解

| 条件 | 规则正确 / 576 | 确定性执行正确 / 192 | 标准配对正确 / 24 |
|---|---:|---:|---:|
| 原 V7 状态，绕过证据门控 | 392（68.06%） | 61（31.77%） | 2 |
| 仅修正标签映射 | 475（82.47%） | 141（73.44%） | 11 |
| 仅恢复最终 JSON，保留旧标签映射 | 449（77.95%） | 76（39.58%） | 2 |
| 修正映射并恢复最终 JSON | 555（96.35%） | 177（92.19%） | 20 |
| 再补跑 17 个缺失请求：最终 fixed | 560（97.22%） | 177（92.19%） | 20 |
| 参考结构 | 576（100%） | 192（100%） | 24 |

单独改映射改变 85 条规则，纠正 84 条、破坏 1 条。单独恢复 JSON 改变 65 条规则，纠正 58 条、破坏 1 条。两者作用有重叠，不能相加。映射后再恢复 JSON 增加 80 条正确规则；最后 17 个请求的 GPU 补跑再纠正 5 条规则，没有改变本批确定性执行答案。

1090 个唯一请求中，866 个直接复用已接受输出，207 个恢复完整最终 JSON，17 个实际 GPU 补跑。旧请求跨分片存在重复，因此唯一请求统计按首次出现计算；逐例状态优先使用该例原始输出，仅缺失时复用相同输入缓存。最终 fixed 保留 62 个 U 规则，192 例均有结构输出。原 V7 有 155 例包含解析失败，均保留在原状态对照及逐原子来源记录中。

最终仍有 16 个规则状态错误：14 个 F→T、1 个 U→T、1 个 U→F。涉及的错误原子均来自已存在的 Phi 输出，其中 10 次来自已接受标签、8 次来自恢复的最终 JSON（同一错误规则可能含多个错误原子）。这些剩余错误应归于模型对原子命题或上下文的判断，不能声称接口修复已经解决结构理解。逐例明细在 `results/remaining_errors.jsonl`。

## 真实 GPU 运行

- 服务器：`gcx@10.29.128.115`；环境：`/home/gcx/n/.venv`。
- GPU：物理 6 号 A100-SXM4-80GB；模型 `/home/gcx/models/care_module1_v5/phi-4`，原 TrialMatchAI LoRA。
- 仅补跑 17 个缺失请求，batch size 8；无 NLI 重跑。
- 实际用时 32.31 秒（含加载），峰值 allocated 13.174 GiB、reserved 13.375 GiB。生成与运行清单均已保存，任务完成后 GPU 已释放。

## 复现与产物

在新 worktree 根目录运行：

```bash
source /home/gcx/gcx261014/bin/activate
python experiments/causal_attention_20261019/structure/prepare_structures.py
CUDA_VISIBLE_DEVICES=6 /home/gcx/n/.venv/bin/python experiments/causal_attention_20261019/structure/run_compact.py
python experiments/causal_attention_20261019/structure/prepare_structures.py
python experiments/causal_attention_20261019/structure/repair_analysis.py
```

已有补跑记录默认复用，不会再生成。首次 `prepare_structures.py` 需要原服务器的旧缓存目录 `/home/gcx/care_reliability_runs/joint_formal_work_02/structures`；旧 PR #9 同时保存了历史原始产物。当前分支已保存三种结构 JSONL，可直接在同一 192 例上进行后续 Qwen 对照，无需重新生成结构。

- `results/{gold,original_v7_ungated,fixed}.jsonl`：供结构提示、仅注意力、组合三臂读取。
- `results/atomic_mapping.jsonl`：逐例、逐规则、逐原子的原标签、原解析状态和最终映射来源。
- `results/deterministic_predictions.jsonl`：确定性执行逐例结果，明确 `is_model_prediction=false`。
- `results/metrics.json`、`results/repair_decomposition.json`：准确率、状态转移、必要的逻辑检查。
- `results/recovery_examples.json`：恢复依据和真实原始输出示例。
- `compact_generations.jsonl`、`compact_run_manifest.json`、`compact_run.log`：实际 Phi 补跑输出、环境、资源与日志。
- `missing_requests.jsonl`：首次发现的 17 个请求，保留用于复现；`results/unresolved_requests.jsonl`：最终为空。

96 例 dev 用于本轮注意力位置与强度探索；此结构修复比较固定在有完整旧 V7 缓存的 192 例 confirmation，未为 dev 虚构旧 V7 状态，未将 dev 缺失混入评估。
