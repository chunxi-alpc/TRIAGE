# 数据机制、标签真值、划分与旧包使用约束

## 旧包只是开发材料

本对话此前准备过本地研究包 \`CARE_module2_A1_preexperiments.zip\`，CPU 共 22 项测试通过，包含 600 条合成记录，分为 routing 120、relevance 48、source 120、determinacy 120、joint 192。旧 \`joint\` 不是四因素设计；24 个话题共有较少底层逻辑形式；“source” 可为编造事实，不能就此宣称模型参数先验。因此本目录 **没有把这些样本上传或重命名为正式 test 数据**。Codex 得到旧 ZIP 时可以用于脚手架，但必须完成机制改造、全量真值审计并记录新旧数据 lineage。

## 生成原则和不变量

1. 先建立逻辑语法/事实世界/合法动作的 oracle 程序，再派生患者叙述、可见上下文、控制政策和问题，不从答案文本倒推 gold。
2. 每个 \`root_case_id\` 的所有对照（换序、数值变动、问句改写、不同 source、关键未知补全）固定归同一 split。
3. 独立机制家族不能仅是“不同医疗主题词但相同 \`AND(x>τ,y)\` 语法”；生成器输出 \`underlying_mechanism_id\` 和逻辑 AST hash；结构同构样本不跨 split。
4. N1 的 R 通过 query 范围变化，A 通过规则条件事实变化；不能把模型隐藏的目标控制写进问句导致无条件选对。
5. N2 的中性接收 prompt 在 C/P 两个 intervention 条件逐字相同。真实 P 主轨必须先闭卷稳定测试，且不使用答案候选直接泄漏先验事实。合成 prior 与真实 parametric knowledge 两轨独立。
6. N3 对所有允许未知赋值完成求值，计算 \`S(x)\`；若同一个行动由不同来源策略产生，必须按实际指定来源重新计算。可接受追问属性集合由可能决策分歧确定。
7. N4 16 格完整块先用 oracle 保证可满足、无矛盾，再赋 split；不得在看到模型结果后剔除非法或难例。
8. N5 从 PR #12 frozen A1 已存原始预测和原始病例派生合法下游任务；不根据 gold 或干预效果筛样，完整保留不合资格的统计分母。
9. 训练/dev/test 的拆分必须在任何模型结果和主测试读取前锁定并哈希。重复患者文字、逻辑同构、事实组同义改写不得跨 split。
10. 测试 gold、source target、decision target、donor target 字段只在 scorer 或明确隔离的 oracle 组；不能因为程序变量复用而泄漏入模型输入。

## 字段契约建议

每条记录：\`case_id, family_id, underlying_mechanism_id, root_case_id, pair_id, fact_group_id, split, task_id, prompt_text, action_spans, source_spans, gold_answer, gold_R, gold_A, gold_source_answer_C, gold_source_answer_P, gold_DQ, valid_question_attributes, control_target, origin_dataset, lineage_sha256\`。不适用字段写 JSON null，不伪造值。中性输入带 \`control_target\` 仅供 scorer/外部 z 使用，提示构造器必须剔除它。

建议拆成 \`inputs.jsonl\`、\`labels.jsonl\`、\`splits.jsonl\`、\`manifest.json\`，并有显式 allowlist builder；统一保存 SHA-256。合成临床措辞只能作机制模拟，不能宣称医生验证或现行治疗建议。

## 强制的自动检查

- AST 正规化与同构/同根/同事实组跨 split 不相交；对所有 split 不只是 test。
- N1 每一基础机制四格齐备，且各格 oracle gold 确实体现 R 与 A 的独立性。
- N2 对接收样本 C 和 P 的目标不同且可核实，捐赠者不来自接收者同一事实组；不得用 donor gold 计算 receiver 正确率。
- N3 对补全集合穷举或使用形式等价的符号验证；少量抽样核对穷举，错误的 D/Q 标注应使构建失败。
- N4 16 格每格至少有一个可执行的 gold，规则顺序互换不得改变真值。
- N5 \`case_id/rule_id\` 和文本/规则哈希完全匹配冻结 A1；U、技术失败和空动作不得静默丢失。
- 最终 Qwen tokenizer 的字符到词元映射、offset、动作片段不能指向答案选项或无关文本。
- 固定随机种子；所有用于创建正式数据的代码版本、处理时间和哈希写入 manifest。
- 任何缺失、解析错误、超预算和不合法输出都占分母并标原因，不通过删去失败样本来改善准确率。

## 数据集如何判断是否足以完成实验

N1 至少需充足的**独立逻辑机制**支持按家族不泄漏，N2 真实 P 必须有稳定已知的多个独立事实组，N3 必须存在 D/Q 词面重合且逻辑不同的条件，N4 必须有合法四因素 16 格，N5 必须有原始动作/金标及逐规则冻结 A1 预测。

如果达不到这些前置条件，允许只提交数据/验证器和明确的 \`blocked:reason\` 状态，不允许靠伪标签、删除难例或使用测试筛选继续执行。

