# MEGA train_val 最小开发实现

状态：`planned`。已经写入实际代码及 11 项合成测试，**未在本地或服务器执行**。协调器完成 G1 后，在服务器运行；此目录不包含临床效果、独立评测或已通过测试的声明。

## 数据和方法输入

直接使用 `mega_trainval_reader.iter_method_tasks` 的冻结输入，方法只接收完整指南、原始病史、语言。身份仅在外层记录。开发代码没有打开参考树、参考路径、答案或 test 的入口。原始 MEGA 任务问题和输出格式来自冻结的官方中英文模板；执行阶段完整保留该模板、病史和指南。

1. **指南引用抽取**：同一个 Qwen 只看指南，生成条件、动作、例外和包含它们的连续原文段。每个引用都必须逐字出现在原指南，条件和动作必须包含在该段内；完整动作不能放进条件字段。同一指南在一次运行内只抽取一次，缓存原始输出、核验结果和全部来源偏移。重复引用用第一个完全相同的位置，记录出现次数。抽取错误保留；不重试到“正确”为止。这里验证的是引用，没有证明规则语义、动作对应或覆盖完整性。
2. **条件阶段 A**：输入完整原始病史和所有抽取条件；不提供动作、完整规则段、候选处置或答案。患者既往治疗仍在原病史中。固定英文词项/中文字符交集选择患者证据句，并选择当前条件原文；没有外部模型、标签或效果搜索。每条规则单独进行一次单 token 推理；先确认 T/F/U 各为一个可精确解码的不同 token。保存三个 token 的原词表概率、logit 和**本规则内部**三候选归一化概率，不跨规则归一化，不把生成文本里的数字称为概率。失败回退 U，并明确记录不是概率 1。
3. **执行阶段 B**：预测 T 的规则选择完整条件—动作—例外原文段，重叠/重复 token 自动合并。U 不等于 F；完整原指南始终在输入中。执行阶段不做符号执行或自造准确率，按官方模板生成原生决策步骤和最终结论。

提取模型可能漏掉例外或错误拆分条件/动作，引用检验不能消除这种风险。异常及未解析案例保留在所有方法的固定分母；原生模型输出仍可以生成，记录 `method_fallback`，不能解释为注意力方法成功。

## 聊天模板与同输入控制

所有生成使用同一 tokenizer 的 `system` 聊天模板、`enable_thinking=False`、贪心、seed=0。检查纯文本在聊天模板中唯一且原样出现，把源字段字符偏移整体平移；再比较 `apply_chat_template(tokenize=True)` 和 StageSEKAAdapter 的实际编码结果。A 开关复用相同条件 prompt 和证据。B 开关在固定 A 的同一 TFU 下复用相同病史、原问题、完整指南、TFU 和证据。

默认原生输出 9 组：完整指南、A0B0、A0B1、A1B0、A1B1、固定 A1 TFU 的 B 零强度/错误跨度/错位规则/反向。A0B0 即文本两阶段对照。A1B0 是 B 控制的同信息 no-hook 基准。每条条件另保存 A0 no-hook、A1 normal/zero/wrong/misaligned/reverse 六组首 token 输出；这些条件负控制不替换 B 控制使用的正常 A1 TFU。

wrong 使用原提示内不重叠且等 token 数的其他证据；misaligned 将规则对应关系循环错位，并要求最终 token mask 真正变化且等 token 数。不能构造时记录 `control_unavailable`，保留病例和原始上下文预测，明确注明复用了 A1B0；该行不构成已运行的负控制。全部规则都被选中、导致循环置换后的集合不变时，也不能冒称已实施错位。无预测支持时记录 `no_support_fallback`，不强行添加证据。

匹配零强度进入上游同一算子并把强度改为 0；没有支持跨度时会明确退回无 hook，必须单独统计。逐例保存 zero 与 no-hook 的词表首步最大分数漂移和 argmax 差异。真正数值一致与否要由服务器结果判定。

可选 `--include-text-controls` 增加文本强调和预测裁剪。文本强调复制 A0 预测支持的原文引用；裁剪保留预测 T/U 的完整规则段，没抽出规则或保留集为空时使用完整指南。这是有信息删除的简单对照，抽取覆盖不完整会丢失指南内容，应独立报告；它不属于 B 同信息注意力对照。

## 资源、成本和状态

只允许协调器指定的一张可见 GPU；否则上游 SEKALLM 会自动跨可见 GPU 分片。使用 SDPA、原检查点 dtype、固定 last10 和现有 SEKA 2.4/0 参数，两份正负投影都加载。此参数沿用先前标准设置，未在临床测试结果上优化。

输出 token 上限、规则抽取 token 上限和上下文上限都必须在启动时明确传入，并写入协议。完整输入超限时报错保留病例，不静默裁剪或换样本。默认 16 例、9 个原生输出臂，最多 144 次原生生成；此外每个独立指南一次抽取、每例每条抽取规则最多六次单 token 条件判别。实际规则数在抽取前未知，不能先假报固定条件推理规模。资源和耗时需要协调器根据这些预算向用户报告后启动。

所有真实 forward 次数、生成 token 数、耗时、显存、输入和 mask 哈希、数据/模型/源码/模板/投影指纹写入日志。共享指南抽取和条件判断日志只写一次；每个原生输出通过 `cost_references` 指向相应成本，并给出同指南病例数用于摊销。完整指南基线不应被收费额外规则抽取。比较方法时必须包含适用的抽取与条件判断成本，不能只比较最终生成时间。

原生输出保存为每个方法独立的 JSONL，包含身份、`llm_output`、状态和元数据。解析失败、推理失败和准备失败均保留行；即使某个病例准备阶段异常，也为未生成的方法补错误行。实际数据访问或整体模型加载失败则记录 `blocked` 和清楚错误，不宣称完成。程序结束只标记 `execution_status=finished`、`verification_pending=true`；独立审计才能改 `complete_verified`。

## 服务器执行接口

以下是接口，尚未执行。变量由协调器按真实路径和已汇报预算设置，不在本地运行。

```bash
cd "$CLINICAL_DEV"
PYTHONPATH="$METHOD_AGENT:$PYTHONPATH" python -m unittest -v test_clinical_core

CUDA_VISIBLE_DEVICES=0 python run_mega_development.py \
  --dataset-root "$MEGA_DATA" --input-manifest "$RUN/trainval_manifest.json" \
  --official-root "$MEGA_CODE" --seka-root "$SEKA_ROOT" \
  --adapter-dir "$METHOD_AGENT" --reader-dir "$DATA_AUDIT" \
  --model "$MODEL" --model-fingerprint "$MODEL_FINGERPRINT" \
  --pos "$SEKA_POS" --neg "$SEKA_NEG" \
  --max-new-tokens "$NATIVE_OUTPUT_BUDGET" \
  --extraction-max-new-tokens "$RULE_EXTRACTION_BUDGET" \
  --max-context-tokens "$CONTEXT_BUDGET" \
  --output-dir "$RUN/clinical_seed0" --seed 0
```

每个方法的 `*.predictions.private.jsonl` 可接 `MEGA_NATIVE_INTERFACE.zh.md` 中官方单轮解析、Qwen 裁判和汇总入口；方法进程不访问裁判参考。PCC、官方连续 Acc、分组区间、纠正/退化由独立评价程序计算，这里没有伪造指标。正式评测 seeds 0/1/2 仍需方法冻结后另行执行；此开发入口刻意只接受 seed=0。

运行目录及抽取缓存设为私有权限，包含患者/指南引用和模型输出；只留在服务器研究目录，不把原始数据或缓存原文提交 PR。可提交本目录源码、协议和去除受限原文后的审计摘要。
