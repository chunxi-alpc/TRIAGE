# 新规则与新合成事实的探索性验证

本目录提供 4 条完整规则、24 条明确合成病例和 4 对反事实。规则来自 MEGA-CDP `train_val/decision_trees.jsonl` 中此前未在现有结果记录出现的指南 42、44、80、53，选定叶节点依次为 3、10、3、43。输入和标签在模型运行前固定，冻结时间和文件指纹见 `data/FREEZE.json`。未读取患者正确路径、参考答案、数据集原有病例或模型输出。

四条规则分别涉及院内转运监测、乳腺新辅助治疗前活检分类、RhD 阴性孕妇首次产检、无糖尿病人群的指南外痛症处理。11 个原子条件显式组合为 AND/OR/NOT 逻辑：转运且（麻醉或镇静）且保留气道装置；活检且治疗前且 WHO 分类尚未完成；孕妇且 RhD 阴性且首次产检；非糖尿病。源树中复合节点已拆开，并定位其原始局部文字，分支 no 用 NOT 表达。怀孕规则所有合成病例明确为女性，性别是已披露的固定人群背景。共 66 个条件请求。病例分别为完整事实 T、反事实 F、关键条件缺失 U、否定表达 T、其他患者记录加本患者明确事实 T、同一评估的未解决矛盾 U。没有临床专家复核；这是新的合成探索集，不是独立临床效果验证。

`data/inputs_public.jsonl` 是全部自编事实，可公开。`labels_public.jsonl` 包括标签和证据句编号；`pairs_public.jsonl` 包括成对关系。`annotation_audit.jsonl` 记录逐规则人工阅读要点；`history_audit.json` 记录既有实验 JSON/JSONL 和缓存病例编号的完整排除审计摘要。审计只能覆盖服务器现存记录，不能保证模型预训练中从未见过原指南。

完整源树、动作和规范化规则存放 `data/private/`，不提交仓库。第三方 MEGA-CDP 数据卡许可证为 TBD，来源仓库为 `nchennnn/MEGA-CDP`；不重新分发其树或动作原文。`rule_manifest_public.jsonl` 提供 source 文件/规则树/动作/规范规则的 SHA256、节点、分支以及原始字符区间。原始位置以 `decision_trees.jsonl` 每行 `text` 字段解码后的完整 JSON 字符串为坐标，区间是 JSON 字符串内容 `[start,end)`；如其中有转义字符，用 JSON 解码恢复。`source_span` 和 `action_span` 则相对于规范化规则文字，供模块二定位。二者来源明确区分。

在服务器重建与验证（不用 GPU）：

```bash
cd /home/gcx/TRIAGE-v5-pr4
/home/gcx/gcx261014/bin/python experiments/care_module1_v5/exploratory/build.py --output /tmp/care_v5_exploratory_rebuild
/home/gcx/gcx261014/bin/python experiments/care_module1_v5/exploratory/validate.py
```

`build.py` 拒绝覆盖既有冻结数据。源数据须另行按来源要求取得。`validate.py` 检查已冻结文件、规范条件/动作位置、原始条件局部引用和动作引用、24 条输入与标签一一对应、反事实配对和既往指南排除记录。

按任务要求汇报 GPU 资源后，使用主实验同一推理接口运行新集；不在本脚本里下载模型或自动启动 GPU：

```bash
/home/gcx/gcx261014/bin/python experiments/care_module1_v5/scripts/run_experiment.py \
  --expected-count 24 \
  --inputs experiments/care_module1_v5/exploratory/data/inputs_public.jsonl \
  --rules experiments/care_module1_v5/exploratory/data/private/rules.jsonl \
  --device cuda:6 \
  --output experiments/care_module1_v5/exploratory/runs/frozen_v1
/home/gcx/gcx261014/bin/python experiments/care_module1_v5/exploratory/evaluate.py \
  --predictions experiments/care_module1_v5/exploratory/runs/frozen_v1/predictions_public.jsonl \
  --output experiments/care_module1_v5/exploratory/results
```

评价复用主实验指标和严格证据协议，输出逐类错误、各场景正确数、证据定位、联合正确、反事实配对和耗时。本集没有 Qwen3 预测，不生成或假设该模型的比较结果。显存峰值和真实运行状态来自实验运行目录 `run_status.json`。

首次原型包含未拆分的复合节点，规范检查后在任何探索集推理前替换。其冻结指纹保存在 `superseded_freeze.json`，修改理由见 `pre_run_schema_review.json`，完整未运行原型归档服务器 `/home/gcx/care_module1_v5_exploratory_superseded_before_run/`。未依据模型结果选择规则或调整标签。

本集没有数值阈值，因此 A+数值组没有程序数值证据；其已知样本证据得分为 0 或联合指标只包含未知样本协议的得分时，反映缺少句级定位输出，不能与 B 的完整证据及联合指标当作同等能力比较。纯 TrialMatchAI 的证据定位和联合指标不适用。新增 NOT 条件也不是 TrialMatchAI 官方 exclusion criterion；排除侧 Not Violated 映射目前只有单元测试，没有真实排除标准的独立系统验证。
