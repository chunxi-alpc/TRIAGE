# TrialGPT 全量专家标签评估：冻结模块一 A1

**当前状态：全量评估已完成，退出码 0；1,014/1,014 条请求返回，1 条空文本保留为输入失败。投影一致率 722/960（75.21%），宏 F1 0.7334。详细结果见 [评估报告](results/REPORT.zh.md)。** 仅在确认 MedGUIDE PR #19 原远端进程和所有子进程已经终止后才能启动。

## 数据来源和任务

- 官方： https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations
- 共 1,015 条医生专家标注的患者—试验标准记录，53 位合成患者说明、103 项试验。数据官方只有一个名为 train 的分区；**这里完整用于冻结 A1 外部评价，不训练、不调参**。
- 患者原文 note 和单条 criterion_text 构成模型输入；criterion_type 仅用于评分分层；expert_eligibility 仅在独立评分器读取。不得输入原数据的 gpt4_eligibility、gpt4_explanation、expert_sentences、专家 gold。
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

本分支包含 stop_medguide.py：在服务器先执行 python experiments/trialgpt_a1_full_20261021/stop_medguide.py（只读预览），核对 PID，再由有权限的 Codex 执行同一命令加 --execute；该脚本只按旧项目路径定位进程，保留所有历史日志。确认旧进程为零后，在仓库根目录：

    source /home/gcx/gcx261014/bin/activate
    export PYTHONPATH=/home/gcx/care_steering_20261019/.deps
    python experiments/trialgpt_a1_full_20261021/prepare.py
    python experiments/trialgpt_a1_full_20261021/tests_cpu.py
    bash experiments/trialgpt_a1_full_20261021/run.sh

prepare 下载官方 Parquet，记录精确数据集 SHA、文件 SHA、真实分母/类别和全量匿名索引；不存入仓库。run 再次复验原 A1 的 prompt 和 few-shot SHA、检查旧 MedGUIDE 运行进程、四卡分片执行原版 A1 推理。结果保存在 results/summary.json、results/predictions.jsonl 和 results/REPORT.zh.md；原始患者文字与模型长输出只保存在 gitignore 的 private/。

未取得全部请求输出或服务器失败必须标记 **INCOMPLETE**。不自动合并主分支，不恢复或重跑已取消的 MedGUIDE 实验。


## 本次实际核验（2026-10-21）

旧 MedGUIDE 独立进程组 3226347 已终止，141 个父子进程（含编译子进程），ps 和 nvidia-smi 均核验；未更改其他作业。原日志与中间文件完整保留在旧 worktree/private。核验摘要见 stop_verified.json。

官方数据固定 revision 1cfcafde94a1560a33b4addc1638664fe28fc059，Parquet SHA256 见 dataset_manifest.json。原始字段已实际核验；included=150、not included=23、excluded=10、not excluded=484、not enough information=293、not applicable=55。960 条可做有损 T/F/U 投影，55 条单列不适用，其他未映射=0。annotation_id=883 标准文本为空，不补写、不构造请求，在全母集和投影分母中保留为 invalid_input；实际推理输入为 1,014 条。

评分区分 invalid_input、missing_prediction、parse_failure、invalid_output、valid；保留六类原标签对 A1 输出的交叉计数，以及三类投影混淆矩阵。患者、试验分别提供 accuracy 与 macro F1 的 10,000 次聚类 bootstrap 区间。这两个区间不是同时处理患者与试验交叉相关的双向区间。基础模型或 LoRA 是否曾见过这些数据无法由本次不训练来排除。

启动前预算：GPU 0/1/2/3，每卡约 25–35 GiB，预计 30–60 分钟。稳定后不持续监控。可在服务器仓库根目录执行：

    nohup bash experiments/trialgpt_a1_full_20261021/run.sh > experiments/trialgpt_a1_full_20261021/private/launcher.log 2>&1 < /dev/null &

脚本完成后自动评分并写入 results/，不会自动提交或推送。断线不影响运行；重启通过原 runner 已有 case_id 跳过已返回记录。执行前检查原任务已经退出，不能同时启动重复实例。当前真实完成度以 private/a1_*/manifest.json、private/exit_status.txt 及 results/summary.json 为准。

首批稳定性核验见 startup_verified.json，启动源提交见 launch.json。核验时每卡显存约 18.5–19.3 GiB；这是启动阶段观测值，不是全程峰值。评估结束后生成全部指标，最终指标现已写入 results/REPORT.zh.md；startup_verified.json 与 launch.json 保留为历史启动记录。

## U 分歧逐条分析

已对全部124条专家U、A1预测T/F的分歧逐条复核：明确A1判断错误90条（72.58%），标签定义或上下文差异16条（12.90%），无法确定18条（14.52%）。这是Codex事后文本复核，不是独立专家裁定或新金标；原始标签、预测和全量评分保持不变。详见 [逐条报告](analysis/U_ERROR_REVIEW.zh.md)、[124条复核记录](analysis/u_error_review.jsonl) 和 [汇总](analysis/u_error_review_summary.json)。
