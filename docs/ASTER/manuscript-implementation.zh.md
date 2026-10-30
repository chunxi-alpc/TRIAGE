# 按论文主稿实现的参考方法

## 版本与范围

规格固定为主稿提交 `60027c426a3155bf55c26273f4c3f49fd563ce1f` 的第 4.2–4.4 节，以及方法补充 A–D 节。新增代码不采用 PR #41 的查询差分主方向、当前键的秩一投影或独立 T/F/U 路由，也不改变 PR #41 的实验。

本实现提供可独立调用的数值核心、数据契约、Qwen3/Llama 采集与缓存参考执行器，以及拟合/逐病例推理入口。它不是已经完成的临床实验，也不是全量实验调度器。主稿中的“性能待验证”仍然成立。

| 文件 | 对应职责 |
|---|---|
| `src/aster/manuscript_core.py` | 最小对统计、谱估计、有向均值、双状态选头、祖先门控和缓存算子 |
| `src/aster/manuscript_runtime.py` | 提示与词元对齐、键/查询采集、拟合流水线、指南缓存及贪心生成 |
| `scripts/run_aster_manuscript.py` | 本地模型拟合与逐病例推理命令；保留失败记录 |
| `tests/test_aster_manuscript.py` | 数值、数据、路由和缓存契约；可选真实架构测试 |
| `examples/aster_manuscript_schema.json` | 输入格式示例；不是临床数据或实验成绩 |

现有 `ActionBlindBranchClassifier`、`HFLabelScorer` 和 `fit_temperature` 保持原样。此处不重新实现概率校准，也不把温度缩放当作分支正确性保证。

## 一、离线方向学习

对同一规则、同一处置、同一问题构造三个经外部独立核验的患者状态：正状态满足目标规则；负状态满足同组兄弟规则；未知状态保留祖先事实但缺失决定性区分事实。标签不能来自在线分类器的预测。

离线顺序为“患者—问题—指南—回答前缀”。在真实模板下对整条输入分词一次。要求三状态总词元数、规则词元标识及绝对位置一致。无法对齐时记录排除原因；不使用截断、自动填充或移动片段伪造对齐。

键采集在键归一化之后、旋转位置编码之前。对每个键值头，严格计算：

```math
d_{i,j}=k^+_{i,j}-k^-_{i,j},\qquad
\Omega=\frac1N\sum_i\frac1{|J_i|}\sum_j d_{i,j}d_{i,j}^{\top},\qquad
\bar d=\frac1N\sum_i\frac1{|J_i|}\sum_j d_{i,j}.
```

必须先逐词元求外积，再在最小对内平均。不能把片段平均差分的外积冒充这一统计量。

```math
U=\operatorname{TopEig}_m(\Omega),\qquad
v=UU^{\top}\bar d,\qquad
\Delta=\frac{v}{\|v\|_2+\epsilon}.
```

累计统计使用 CPU 双精度。数值秩不足、非有限统计或投影均值近零的头排除。`rank` 和 `top_k` 由调用者在开发阶段确定；数值容差不是已经验证的最佳超参数。

专家同时保存完整特征值、子空间基、均值、方向、投影均值范数和两种不同的诊断量：谱能量保留率与平均差分范数保留率。两者不能混称“方差解释率”。`subspace_overlap` 提供分半稳定性统计，但本次没有运行真实数据上的稳定性实验。

## 二、在线响应选头

在来源隔离的选头集上计算主稿的表示分离度。标准差使用总体定义；未知偏移先在片段内平均，再取绝对值。

将提示改成真实在线顺序“指南—患者—问题—回答前缀”，使用未编辑骨干的最后回答前缀查询。Qwen3 查询在原模型精度下完成旋转后再转双精度统计。所有共享查询头都计入读取响应。

保留投影均值范数、分离度、正状态读取和负状态读取均通过要求的头。负状态也要求正读取，因为真正抑制时另乘负增益。按联合分数降序排序，并列按层号、键值头号排序。至多选择 K 个，不用无效头补齐；允许空专家。

拟合与选头按 `source_id` 隔离，并可传入锁定测试来源名单。来源编号必须由数据适配器规范化；代码不会自动识别不同名称下的同源指南。所有头使用同一批完整通过的选头最小对。

## 三、处置盲判断与祖先门控

推理先调用现有分类器。它与生成共享同一个模型对象，但不会接触编辑后缓存。分类器只读取患者与完整条件。每组结果保留预测与弃权，分类接口失败单列为 `None`，不能改写成语义 `NONE`。

层级组需要显式提供 `parent_group_id` 与 `parent_rule_id`。子组的 `ancestor_conditions` 必须包含父组的祖先条件及到达它的父规则条件。根组无父边。禁止循环、重复编号或缺失父边。

祖先弃权、预测 NONE、分类失败或选择了其他分支时，子组关闭。被接受的选中规则获得 `+g_plus`，同组兄弟规则获得 `-g_minus`。置信度只控制接受，不再乘第二次概率。温度、接受阈值及正负幅度均须由非测试资料确定，并作为命令参数显式传入。

## 四、真正的缓存键平移

先对未编辑的指南前缀预填充，再为每个病例复制所有层的键值缓存。实际写入：

```math
k'_j=k_j+g_r\Delta,\qquad
\widetilde k'_j=\widetilde k_j+g_rR_j\Delta.
```

仅改变所选键值头的规则编号及处置词元。不会将 `k` 投影到方向上；不会修改值；不会再次归一化；不会把未旋转的方向直接加到已旋转缓存。

随后执行完整患者/问题后缀及逐词元贪心生成。原指南缓存保持不变。分组查询注意力中，修改一个键值头自然作用于共享它的全部查询头。零增益或空专家跳过数值编辑，但生成仍使用独立缓存副本。零编辑比较基准是相同的分段前向，不预先声称与一次性完整前向逐位相等。

记录计划编辑向量数、实际非零变化向量数与实际最大位移。低精度舍入可能使很小的计划位移变成零，因此计划增益不等于实际编辑。

## 五、输入格式

见 `examples/aster_manuscript_schema.json`。实际文件采用 JSONL，每行一条记录。

拟合记录需要 `pair_id`、规范化 `source_id`、`split="fit"`、`target_rule_id`、`question`、共享 `guideline`、三状态 `states` 和 `verification`。选头记录使用 `split="selection"`。

`verification` 保存外部核验者/规则验证器及可追溯参考，声明正分支、负兄弟分支、未知标签、祖先可达性与标签独立性。**填写这些字段不等于完成语义核验。** 当前代码不提供任意临床自然语言指南的可靠自动标注器。自然病历最小对必须在输入前独立核验。

推理记录需要 `case_id`、`patient`、`question`、`guideline`，以及可选字符串数组 `options`。推理函数不读取测试金标。

词元跨越编号/处置边界会被拒绝。不能为提高覆盖率而静默把相邻条件文字纳入编辑范围。实际分词器上的通过率和排除原因应作为数据构造结果报告。

## 六、安装与运行

从仓库根目录执行。权重必须事先存在于本地；脚本始终使用 `local_files_only=True`，不会联网下载模型。当前参考接口固定为 Transformers 4.57.6。

```bash
python -m pip install -r requirements-aster-manuscript.txt
python -m unittest discover -s tests -p 'test_aster_manuscript.py' -v
```

先准备经核验并按来源隔离的拟合/选头 JSONL。以下环境变量须由实际数据、模型修订及开发配置填写，不能把示例当作最佳配置：

```bash
python -m scripts.run_aster_manuscript fit \
  --model "$MODEL_PATH" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --device "$DEVICE" --dtype "$DTYPE" \
  --fit-data "$FIT_JSONL" --selection-data "$SELECTION_JSONL" \
  --forbidden-sources "$LOCKED_TEST_SOURCE_IDS_JSON" \
  --rank "$RANK" --top-k "$TOP_K" --output "$NEW_FIT_DIR"
```

`expert.json` 保存可读的冻结专家，不使用任意对象 pickle。`audit.json` 保存纳入、排除及无效头记录。拟合失败在新目录的 `status.json` 中标明，不声称存在可部署专家。

温度拟合沿用仓库现有 `fit_temperature`，只使用独立校准资料；弃权阈值与增益另外在开发集确定。然后运行：

```bash
python -m scripts.run_aster_manuscript run \
  --model "$MODEL_PATH" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --device "$DEVICE" --dtype "$DTYPE" \
  --expert "$NEW_FIT_DIR/expert.json" --cases "$CASES_JSONL" \
  --temperature "$TEMPERATURE" \
  --probability-threshold "$PROBABILITY_THRESHOLD" --margin-threshold "$MARGIN_THRESHOLD" \
  --g-plus "$G_PLUS" --g-minus "$G_MINUS" \
  --classifier-max-input-tokens "$CLASSIFIER_INPUT_BUDGET" \
  --max-new-tokens "$OUTPUT_BUDGET" --with-baseline --output "$NEW_INFERENCE_DIR"
```

拟合若限制候选层，推理须给出相同 `--layers`。骨干身份、精度和模板必须与专家一致。`--raw-template` 仅用于没有聊天模板的离线基础模型实验；现有处置盲分类接口仍要求有效聊天模板。

所有输出目录必须不存在。逐病例失败也写入 `predictions.jsonl`，不删除困难病例、不进行按成绩重试。脚本不自动评分临床输出、不搜索测试集强度、不启动远程 GPU、不修改已有运行。

## 七、验收状态与剩余工作

本次实际完成：37 项新增测试中 35 项通过、2 项跳过。通过项使用数值数组、字符分词夹具及随机微型因果注意力模型，覆盖权重、公式、极性、未知惩罚、位置旋转、GQA、采集钩子清理、祖先门控、缓存隔离和零编辑一致性。

当前环境没有 Transformers，且无法取得该依赖，因此两个随机微型真实 Qwen3/Llama 架构测试未运行。也没有运行预训练模型、真实分词器的三状态对齐、真实 HF 缓存端到端验收、临床效能或跨模型实验。不能把随机夹具测试当成这些项目已经通过。

适配器有意限制为单设备、单病例、全注意力、非量化动态缓存。动态 RoPE、YaRN、滑动窗口、混合缓存和并发共享模型不在当前验收范围。超限输入记录失败，不暗中改变位置编码。正式扩容前应先在目标本地权重上完成小样本数值验收，再独立冻结运行配置。

临床自然语言金标、来源同一性核查、自动增益/阈值选择、断点恢复、完整基线矩阵和官方任务评分器不包含在这次参考执行器中。现有 PR #40 的机制结果也不会自动成为这份新代码的实测结果。
