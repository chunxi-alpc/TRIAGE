# 官方SEKA当前执行记录的审计说明

日期：2026-10-26。下列运行数据由协调器从115服务器真实执行产物提供；完整产物独立核查状态另列，不把口头摘要替代逐例复算。

## 合成校准投影

官方仓库随附合成文件实际有100行，每行两个triplet。命令max_samples=200是上限，builder使用min(200,100)，因此处理100行、200个triplet，每个triplet三次前向，共600次。它不是200行或1200次前向。

协调器报告全可用校准投影任务退出0，600次前向，墙钟62.6686秒，峰值分配显存30.6123 GiB，79个非零头和1个零头。首次projection_full200_audit.json因外部预期写成1200而失败，失败记录保留。修正预期为真实600后另存v2，不重新运行模型，也不把审计预期错误记作模型失败。verify_projection.py已补充读取校准文件行数，分别登记requested cap、available rows、effective rows及对应前向上限。

此前16行接口验证为96次完成前向，退出0，FP32，墙钟38.064秒，峰值30.560 GiB，80个非零头。全量校准与小试使用不同校准规模，必须分别记录，不能混用投影指纹。

投影中的零头是上游min_diff筛选的合法输出；负投影是独立Un的剩余方向正交投影，不能要求与正投影互补。投影几何校验通过只说明产物可用，不意味着完成官方标准集或论文检查点的性能复现。

## CounterFact接口评测

协调器报告0:16的efficacy评测退出0，16次前向，墙钟25.7467秒，峰值15.2911 GiB。保存指标score.mean=1.0，magnitude.mean=0.0005277313503307717。它是16条接口测试，不能当完整标准集的99.08等发表数字复现，也不能解释为临床答案准确率。

独立脚本export_counterfact.py从原始逐例log probability重新计算胜负、概率差、均值与总体标准差，并与embedded/separate指标核对。公开文件仅包含id、prompt/generation的SHA256、分数及胜负等数值；不导出原文。不保存的token IDs不事后猜测，generation_token_count记null，包括空generation。

## 同输入与成本证据

后续run_official_job.py在完成forward的hook中逐调用保存input_ids和attention_mask的dtype、shape及SHA256到forward_inputs.jsonl，不保存token IDs或文本。缺失张量明确记null，不能把两份null当作已证实同输入。该证据可核对no-hook与matched-zero的实际模型输入；不同GPU设备号不参与输入hash。

记录forward_input_hashing_seconds，并包含于总墙钟。该时间包括GPU到CPU拷贝、同步等待、hash和JSONL写出，不能简单从总时间扣掉后声称获得纯模型计算时间。原先未启用逐调用记录的运行不得追溯宣称已核对token级完全相同输入。
