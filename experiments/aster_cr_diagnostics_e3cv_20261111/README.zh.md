# ASTER-CR 收益拆分与 E3 五折验证

运行位置：服务器 /home/gcx/aster-cr-diagnostics-e3cv-20261111。
使用 source /home/gcx/gcx261014/bin/activate 进入既有环境。

仅新增 E3 五折、6个独立截距和诊断评价。原实验路径在 common.py 的 BASE 中明确指定。
旧模型、数据、原始分数、激活和梯度只读；本轮精确修补写入新目录。

运行顺序（服务器仓库根目录）：

1. python -m experiments.aster_cr_diagnostics_e3cv_20261111.train prepare
2. python -m experiments.aster_cr_diagnostics_e3cv_20261111.train launch
3. python -m experiments.aster_cr_diagnostics_e3cv_20261111.evaluate intercepts
4. python -m experiments.aster_cr_diagnostics_e3cv_20261111.evaluate full
5. python -m experiments.aster_cr_diagnostics_e3cv_20261111.evaluate cv
6. python -m experiments.aster_cr_diagnostics_e3cv_20261111.modules
7. python -m experiments.aster_cr_diagnostics_e3cv_20261111.verify
8. python -m experiments.aster_cr_diagnostics_e3cv_20261111.report

CPU阶段设置 CUDA_VISIBLE_DEVICES 为空，建议 OMP_NUM_THREADS=4、MKL_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4。
full 和 cv 可以并行。GPU队列仅领取缺失的原始配对—头组合，每卡一个进程；用户授权卡为0、1、5、6。
本次双卡补算已在用户追加5、6卡前完成，新增卡未加载模型。

断点恢复：已经完成的E3折、精确配对缓存和评分目录均可复用；异常写入 status/failed_*.json，详细输出写入 logs/。
score_variant.py 统一实现五种分数；各 scope 的 NPZ 使用不同评分版本字段，并保留包含折号与截距拟合来源的清单。
完整路径由全部可达节点分数重新遍历；MedPIC调用上一轮的合法答案集合解码。
五折合并按80个来源执行2,000次配对重采样，不平均折级VR或置信区间。

验证：
python -m unittest experiments.aster_cr_diagnostics_e3cv_20261111.test_diagnostics

输出在 reports/aster_cr_diagnostics_e3cv_20261111/。报告为 REPORT.zh.md；
服务器保留未压缩逐样本JSONL，GitHub提供同名 .jsonl.gz；五折合并结果通过 prediction_references.json 引用五个折。
