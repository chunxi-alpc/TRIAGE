# CARE 方法与数据审查

本审查在服务器 10.29.128.115 完成。数据生成和确定性模块接口检查已实际执行；未执行任何 GPU 模型推理。

## 用户指定原文件

用户指定的 /mnt/data/粘贴的文本 (1)(20261019-065114).txt 未能访问：本地 /mnt/data 不存在，服务器 /mnt/data 也不存在。只读检索了本地工作目录、Downloads 与服务器 /home/gcx 中的相关文件名，没有找到该文件。Downloads 中另有 CEBU 的“粘贴的文本 (1).txt”，内容属于另一项工作，不能替代指定 CARE 文件。因此不能声称已读取指定原方法文件，也不能据当前实现确认完整 CARE 忠实复现。

## 可核验的现有实现

- 模块一 V6：/home/gcx/TRIAGE/src/module1/care_v6.py。
- 代码仓库提交：2cc2e84d8f80988c456f14b1387892ef46624858。该文件的工作区状态：clean。
- 本次冻结副本：sources/care_v6_frozen.py，SHA256：d2cfe48d5eb8f08da8e3d806982670b0810323c7b98e0246f57025f052bdf997。
- V6 需要预先整理的结构化规则；TrialMatchAI（Phi-4 + LoRA）判断原子 T/F/U，BioLinkBERT-MedNLI 仅定位证据；明确数值由确定性比较覆盖，最后合成三值逻辑和动作门控。
- 最新后续工作目录：/home/gcx/TRIAGE-worktrees/care-reliability-20261019/experiments/care_reliability_20261019。其 V7 修复完整充分证明证据门控，但不等于原完整 CARE。
- /home/gcx/TRIAGE/docs/ACRE/02-method.md 是另一个标为待验证的 ACRE 设计规范，不能当作缺失的 CARE 上传文件。
- /home/gcx/TRIAGE/experiments/care_qwen3_4b_base_20261016 有以前的 Qwen3-4B-Base 三项观察实验和子空间代码。其任务说明也明确“不需实现完整 CARE”。因此实现 D 时必须披露是可运行的适用性 Key 子空间适配。

## 已有数据

1. 模块一 V6 的 data/inputs_192.jsonl：192 条 sample_id、rule_id、sentences；rules_v5.jsonl 是 16 条带 conditions、logic、exceptions 的离线结构化规则。这些既往数据已被开发和分析使用，不能称为新的盲测集。
2. 同目录 new_inputs_100.jsonl/new_rules_100.jsonl 与 new_100_predictions_v6.jsonl：已有 100 例及预测缓存。
3. reliability 实验 data/mechanism_cases.jsonl：576 例、24 家族，含完整 prompt/patient/rules/reference_structure/gold、AND/OR/NOT/例外、单原子翻转及缺失情形。其 README 明确数据措辞已用于 V7 修复，不是全程盲法的独立确证集。
4. reliability 新集主测试包含 8 家族、192 例；该数据不是新的临床指南，路由动作完全虚构。现有结构预测结果位于 /home/gcx/care_reliability_runs/joint_formal_work_02/structures/，本任务没有将其旧结果冒充新实验。

## 本次新数据

data/build_data.py 不读取既往病例、模型输出、模型置信度或干预收益。随机种子 20261019。
共 440 例：12 家族训练/120 例，8 家族开发/80 例，16 家族测试/160 例，另 8 家族设备路由迁移/80 例。每个家族 10 种情形：适用/不适用最小数值翻转、多规则联合动作对、例外有/无、旧规则无效时的新规则适用/不适用、信息缺失、无冲突。每对保持全部规则、动作和选项位置不变，只改变一个患者数值或例外事实。患者在规则前出现，供因果 Key 提取。

数据 SHA256：426364257240ee73c1aedf7e0911fe15ecfeb5c797ded60dec56f36faf95b0bd。
实际执行检查：440 条唯一样例，44 个互斥规则家族，所有 action span 与文本匹配；3 类数值最小对及例外对保持条件；各划分无家族交叉。答案分布：{'D': 103, 'B': 115, 'C': 113, 'A': 109}。

schema 使用 item_id、pair_id、rule_family_id、split、task、condition、applicable、old_valid、conflict、gold、options、prompt、patient、structured_rules、rule_action_span、new_action_label。迁移 split 为 transfer。label 和 generation_metadata 仅用于评分与数据审查；预测只允许 patient、structured_rules，主模型只输入 prompt。

局限：
- 是虚构患者/设备路由，不能推出临床准确率。
- 新旧规则都明确提供在上下文里；只测上下文来源冲突，未测闭卷参数先验。
- 规则实例、阈值与动作组合跨划分隔离，但通用逻辑模板共享。不得声称是未见逻辑结构的泛化。
- 迁移改变领域和保留家族，结构及指令模板仍相同。
- 本轮模型前向前冻结新数据；未在任何模型结果上筛样或修改标准答案。

## C 的缩减模块一接口

data/care_gate.py 直接复用冻结 V6 的数值比较、分句及三值逻辑，增加有限的显式句子布尔解析。接口 predict_item(item) 只读取 patient 和 structured_rules；get_gate(item) 返回 +1（新规则适用）、-1（不适用且旧规则有效）、0（未知、旧规则无效且新规则不适用、或无冲突）。保留 positive_gate 供仅正向门控消融。

该版本未调用 Phi-4/TrialMatchAI、BioLinkBERT，未估计置信度，未做自由文本规则解析。因此只能称“CARE 数值与逻辑组件的缩减实现”，不能称完整 CARE 模块一。

真实 CPU 接口检查：440 条状态均与可执行标注一致，运行 0.0536 秒（包含双次预测和泄漏检查）；所有非白名单字段置为任意字符串后，预测不变。门控分布 {'-1': 132, '0': 132, '1': 176}。这里的 440/440 是受控模板确定性程序检查，绝不是模型或临床准确率。

输出原始证据：data/module1_reduced_outputs.jsonl；汇总 data/module1_reduced_audit.json。所有脚本可在原服务器环境直接重建。
