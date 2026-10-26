# 临床开发基线实现审查

审查日期：2026-10-26。范围是源码静态审查、服务器小模型验证结果的协调器回报，以及运行来源绑定；没有在本地运行研究实验或测试，没有读取正式测试问题/标签，没有临床效果结论。

## PASTA 与 AdaSEKA

PASTA 在每个查询块内直接调用固定 SEKA 作者 Qwen 移植版的 `edit_multisection_attention`；全部 key/value 保留，因果 mask 只切查询维度，缓存生成始终使用原始完整 prompt 长度。各证据段分别传入，未合并相邻段，因为作者 exclude 模式会逐段累计偏置，合并会改变生成 key 的相对权重。nohook 使用相同查询分块算子但不编辑 mask；零强度创建 alpha=1 的新作者对象，避免旧 log(alpha) 缓存。

首轮真实服务器合同测试失败：PASTA 全 mask 与查询分块的首步 64 个分数全部不同，最大绝对差约 0.04686。差异来自证据文本到 token 的映射：作者 `find_token_range` 使用包含字符边界的判断，原适配器使用半开区间重叠判断。字符 tokenizer 中 “bc” 的作者范围为 [0,3)，原适配器为 [1,3)。现已直接调用作者函数，并按指定原文位置计算同样的 occurrence。没有放宽容差，没有改注意力算子；原源版本 `clinical_attention_source_v1/` 与失败日志由协调器保留。

协调器随后报告服务器 v2 四项 tiny 合同全部通过，耗时 2.428 秒：PASTA 作者完整路径与查询分块路径首步分数及三 token 缓存生成一致；同核 nohook/zero 一致；真实 Ada 四专家类、BF16、控制及前向计数正确；路由存储适配全部 hidden states 逐位相等、最后 token logits 最大误差 0。测试禁用 EOS 停止并要求实际生成三 token，不能只通过首步即停的路径。本代理已独立核对修复源码，未在本地复跑。上述证据不证明 55k 上下文显存可行或临床有效。

临床 Ada 显式采用 BF16，与此前官方默认 FP32 标准集运行不同，协议已披露。使用冻结的 .5/3 强度、last10、top_k_singular=5、weighted_top_k、temperature=1。路由预计算单独计入 attempts/completed；不把它算作免费。存储适配仅对 `output_hidden_states=True,use_cache=False` 路由调用设置 logits_to_keep=1，所有位置/层 hidden states 保留，作者路由不读取 logits。

attention runner 的 nohook 与零强度比较使用实际 input_ids hash；失败行保留，未把缺失数值参考当通过。额外无干预参考的真实推理保存到单独文件。SEKA A0B1 复用行保留原 generation、源文件 hash，并把新增运行成本记 0；最终成本汇总仍应计入原 B1 输出及共享条件判断成本。

## 模型与投影来源

`verify_clinical_attention_artifacts.py` 验证 PASTA 构建成功、官方 profiler/module/source、实际构建输出目录、完整模型字节指纹、源码和校准输入指纹及 YaRN factor=2/original_max_position_embeddings=32768；再核对真实四专家审计及相同模型映射。`run_attention_baselines.py` 重新计算实际模型及投影文件 hash，将该映射摘要与 primary 指纹绑定，核对 BF16、原配置 YaRN 与 primary runtime 设置、PASTA 构建 manifest 和四专家 audit/config。

这两个文件的来源 schema 一致。PASTA 标签明确是 SEKA 作者 Qwen 移植版及其有限校准头选择，未声称完成原 PASTA 论文的完整搜索；四专家文件检查完成不等于临床指标完成。协调器尚待构建队列结束后在服务器实际执行新来源验证器，本次静态审查不替代该运行。

## A0/A1 同信息融合

`run_mega_additional_baselines.py` 默认仍为 A0。选择 A1 时只读取 primary 的 A1 条件预测，TFU、证据段、支持分支、无指南 prior 与裁剪/直接映射均来自该分支；参考输出改为 A1B0，并核对重建完整提示 hash。不存在 A1 缺失时回退 A0 的路径。协议、逐例记录和成本引用同时保存 condition_variant/reference_arm，没有改变冻结融合公式或参数。

共享提取缓存、所选 A 条件判断均有来源和成本引用。text_two_stage 复用所选 A?B0，新增 forward=0；实际方法成本须从源预测文件恢复原 B0 输出成本，不能把复用记录理解为整个方法免费。PASTA/Ada 当前仍使用 A0 条件输出，可同信息比较 A0B1；不能未经说明称其与 A1B1 使用相同 TFU。

## 队列状态

已按协调器授权修复两个本地队列驱动：同计划状态文件用 open('x') 原子认领；启动前写 running；异常和非零退出写 failed；成功进程只写待独立验证，不宣称 complete_verified；所有待等待的 PID 在开始时即打开 pidfd，避免逐个等待时的 PID 复用。没有修改服务器正在运行的旧进程或状态。

`test_queue_drivers.py` 准备了三个服务器专用伪文件/模拟子进程测试：重复认领不启动、全部 pidfd 先打开且失败不会当完成、校验步骤失败不启动临床。未在本地执行，尚未收到服务器结果。

## 本次所审文件 SHA256

| 文件 | SHA256 |
| --- | --- |
| clinical_attention_baselines/author_attention.py | b5cc3fe44cc0a5b8aa521976d732cbec26c9e299937e428248ae840b2dd8d67d |
| clinical_attention_baselines/run_attention_baselines.py | d9e5c6528286071c6ac498f40abed3234337ff1b2c1ed2292494a53dfebe5b6d |
| clinical_attention_baselines/test_author_attention.py | 1656875a291da9c126b8d205b76593f1548216238464b59f14d5383f0b521594 |
| clinical_dev/run_mega_additional_baselines.py | 230f214cdce2b2d2be9f221a260c148eca4d014c5dea36888d3fb5ab7b918163 |
| verify_clinical_attention_artifacts.py | 1a605c56338f65292feefb8ef33c6bee913d7e89d44ec4cd81fd64cf9047e692 |
| run_stage_transition.py | fb74614eae63a41f5d8d0bf06bdfe3c56ecf3b20d6e952efc3d71eda4b0a152f |
| run_baseline_queue.py | 2a3624105b74caeecb414ff1b1ce9b506253ad7619e726425d3a26a421930760 |
| independent_audit/test_queue_drivers.py | 4ef02cf20f823cf5478dd567ae6cf644cfe8e5ecccae818620bc729cb66cc3b0 |
