# TrialGPT 全量专家标签评估：冻结模块一 A1

**当前状态：代码与协议准备中；没有新的 A1 GPU 结果。** 仅在确认 MedGUIDE PR #19 原远端进程和所有子进程已经终止后才能启动。

## 数据来源和任务

- 官方： https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations
- 共 1,015 条医生专家标注的患者—试验标准记录，53 位合成患者说明、103 项试验。数据官方只有一个名为 train 的分区；**这里完整用于冻结 A1 外部评价，不训练、不调参**。
- 患者原文 note、单条 criterion_text 和 criterion_type 可作模型输入；expert_eligibility 仅在独立评分器读取。不得输入原数据的 gpt4_eligibility、gpt4_explanation、expert_sentences、专家 gold。
- 不生成新病例，不手工标注，也不从模型输出伪造金标。所有 1,015 条保留在母集；错误和缺失输出保留分母。

## A1 不变

直接调用仓库现有 experiments/care_module1_semantic_v8/semantic_runner.py，arm=A1。模型 Phi-4 + TrialMatchAI LoRA、原三个 frozen few-shot、NF4/BF16、batch=8、greedy、384 新 token、至多一次格式重试均不变。只将新数据按原字段 question / patient / rules 构造输入，每次一条官方 criterion，原文不改写。

**局限必须保留：** 原 A1 system prompt 描述 invented routing simulation，并没有针对入排标准重训或改提示；每条请求只有一条规则，而原试验多为每例三条规则。这是严格冻结下的领域与输入形式转移。

## 标签与统计

TrialGPT 的原生六类不能无损等同 A1 的三类。预注册映射：

- inclusion: included→T，not included→F；
- exclusion: excluded→T，not excluded→F；
- 任一类型 not enough information→U；
- 任一类型 not applicable→UNMAPPED（保留在总分母，单独报告；不强行映射）。

主要指标只叫“专家资格标签投影一致率”，绝不称为验证了原生三值逻辑真假。特别报告 not excluded 且没有专家证据句的分层——TrialGPT 可能因未提及而给出 not excluded，而冻结 A1 可能按原策略给 U。

输出：1,015 条总母集、可映射条数、UNMAPPED 条数、有效解析率、准确率、T/F/U 混淆矩阵/宏 F1、按 inclusion/exclusion、training 标记、专家证据有无分层；患者 ID 和试验 ID 的聚类不确定性。TrialGPT 论文中的 87.3% 是原生任务六类指标，不能直接拿来当 A1 三类公平基线。

## 真正运行步骤（在授权服务器）

**先停止旧工作：** GitHub PR #19 已关闭，但旧 nohup 不会自动退出。必须先在 10.29.128.115 的授权 SSH 中核验并停止本项目的 run.sh、四个 semantic_runner.py 和 qwen_run.py 子进程。保留旧 private/ 证据，不杀别人的作业。只读查看：

    ps -eo pid,ppid,pgid,args | grep '[m]edguide_full_20261021'
    nvidia-smi

确认旧进程为零后，在仓库根目录：

    source /home/gcx/gcx261014/bin/activate
    export PYTHONPATH=/home/gcx/care_steering_20261019/.deps
    python experiments/trialgpt_a1_full_20261021/prepare.py
    python experiments/trialgpt_a1_full_20261021/tests_cpu.py
    bash experiments/trialgpt_a1_full_20261021/run.sh

prepare 下载官方 Parquet，记录精确数据集 SHA、文件 SHA、真实分母/类别和全量匿名索引；不存入仓库。run 再次复验原 A1 的 prompt 和 few-shot SHA、检查旧 MedGUIDE 运行进程、四卡分片执行原版 A1 推理。结果保存在 results/summary.json、results/predictions.jsonl 和 results/REPORT.zh.md；原始患者文字与模型长输出只保存在 gitignore 的 private/。

未取得全部请求输出或服务器失败必须标记 **INCOMPLETE**。不自动合并主分支，不恢复或重跑已取消的 MedGUIDE 实验。

