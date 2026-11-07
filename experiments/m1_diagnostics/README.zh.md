# 冻结 M1 机制诊断

入口位于服务器 /home/gcx/TRIAGE/experiments/m1_diagnostics，结果位于 /home/gcx/m1_diagnostics_20261106。
所有计算在 gcx261014 环境运行。仅使用 GPU 0、1，排在已有任务之后；CPU阶段不加载GPU。

阶段：
- cpu：冻结头集合与样本，读取旧缓存，按来源重采样2000次。
- patch --model qwen|medgemma：固定64对拟合/开发数据的头、归一化与残差诊断。
- baseline --model qwen|medgemma：各200例MedGUIDE与MEGA-CDP发布原文。
- report：根据已完成缓存生成中文报告；未完成阶段显式标记。
- all：先完成CPU阶段，排队后分别在GPU 0/1加载Qwen/MedGemma，结束时自动生成完整报告。

运行 /home/gcx/m1_diagnostics_20261106/run.sh 即可续跑。
阶段与模型可通过 python -m experiments.m1_diagnostics --stage patch --model qwen 单独恢复；单阶段运行前需设置正确CUDA_VISIBLE_DEVICES。
每对修补和每例原文评估原子写入；完成项跳过，失败项保留明细。缓存记录模型、精度、配对、头编号及诊断方式。
heads/grads保持[层,查询头,头维度]，分数为ell；冻结文件保留H_A/H_P/H/T_h。

统计定义：
t=ell右-ell左，e=(IE正向-IE反向)/2。
原口径仅在|t|>=0.1上计算mean(e/t)；加权Σsign(t)e/Σ|t|及回归Σte/Σt²分别报告全部缓存与有效配对。
Qwen旧缓存不含小效应配对的修补结果，因此“全部”覆盖不足时不伪造全体估计。
LSR定义为共同节点的mean(|Δell_A|)/mean(|Δell_F|)；同时列出余量超过比例与实际翻转率。
重采样保持同源的所有节点、配对和正反方向，不将开发/测试差当区间。

原文保持：
MedGUIDE的原prompt已含选项和作答指令，使用原字符串；按实际A–H标签读出。
MEGA保持原病例、完整树、原节点条件和标注路径，仅添加统一任务模板；自主路径偏离后继续评分，不为偏离节点生成标签。
不新增变体，不补充错误样本，不做M1 v2或M3，不做哈希检查。

## 结果与自动上传

本轮已完成，合并结果发布在 [统一结果目录](../../results/m1_diagnostics_20261106/README.md)，
完整表格见 [REPORT.zh.md](../../results/m1_diagnostics_20261106/REPORT.zh.md)。
结果目录内保存原文样本编号、冻结头、统计汇总及压缩逐例/逐对结果。
`*_patch_details.jsonl.gz` 中每行是类别和修补记录；
`*_baseline_details.jsonl.gz` 中保留预测、原标签、节点错误和来源，完整原始输入仍留在服务器。
旧缓存效应位于 `cpu/legacy_effects.jsonl.gz`。压缩文件可用 Python 标准库 gzip 逐行读取。
这些压缩文件替代逐个上传数百份小文件，原始目录不作删除或覆盖。

完整实验结束、报告确认所有阶段完成后，自动调用上传入口：

```bash
python -m experiments.m1_diagnostics.publish
```

上传基于最新 origin/main 的临时独立检出，仅合并本目录代码、统一结果目录和根README中的本轮摘要；
推送普通提交到 main，遇到并发更新先同步再重试，不强制覆盖远端。
上传状态保存在服务器结果目录的 `status/publish.json`，失败后可用上述命令重试。
实验未完成时禁止发布为最终结果。反复上传相同结果不会产生新提交。
后续实验完成后同样必须合并结果、更新根README并推送该GitHub仓库，作为交付步骤。

本轮实际先在0卡运行Qwen、1卡运行MedGemma，Qwen完成后0卡改跑MedGemma原文评估。
总实验完成时间约2小时；原始日志和激活缓存均不上传。
