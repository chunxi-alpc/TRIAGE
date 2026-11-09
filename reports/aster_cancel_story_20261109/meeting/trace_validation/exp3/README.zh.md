# 跨来源验证

所有实验均在服务器 gcx@10.29.128.115、gcx261014 环境运行。使用 repair-v2 的真实 Qwen3-8B 缓存，原始来源划分保持为 fit 80、dev 20、test 40。所有配对两侧来源与划分一致，三组来源无交集。

## 已完成的缓存复算

`run_cv.py` 优先复用已有 `meeting/real_completion/coordinates.npz` 和 `raw_selected.npy`；若缺失，会自动通过共享加载器从原始真实激活缓存读取选头矩阵，无需另行运行旧实验准备脚本。两条路径均使用同一组原始视图、分数与51个头，训练算法相同。

- `run_cv.py`：固定原51个头，按来源划分五折；每折仅使用64个训练来源重估标准化、C/F证据子空间、A/P干扰子空间、正交化和岭回归，固定维数16、alpha=0.01、无截距。另16个来源用于验证。训练对为A/P，A单独报告。由于原51个头曾使用全部80个fit来源选择，这只是冻结头的条件性诊断。
- `source_diagnostics.py`：从实验1已保存的主16维dev逐对预测复算20个来源指标，并提取表现最差来源的当前条件、指南文件与分数分布。仅作结果解释，不再调参。
- `full_cv.py --prepare`：按来源汇总原加权梯度归因。全部80来源重构的A/P/C/F top64头集合与原选择完全相同。原 `fit_exact.jsonl` 仅44对×64头，无法覆盖五折所有新候选和新抽样对，缺失情况保存在 `full_cv_cache_coverage.csv`。

## 完整流程五折（已完成）

完整重选头五折的A/P总体R²为0.3454、RMSE为2.3641，A类R²为0.3445；各折A/P R²分别为0.4651、0.2166、0.3208、0.3559、0.2911。62/80个来源R²为正，来源中位数0.2466。该结果支持关系能迁移到多数未参与本折选择和训练的来源，同时存在明显来源差异。

两卡合计新增3948次真实最终token前向，复用10826条原干预记录；GPU0/1实测峰值显存17.968/17.962 GiB，各自三折/两折用时合计199.4/153.7秒。全部五折CPU重放的头集合、逐对顺序和预测完全相同，新增模型前向为0，记录见 `replay_verification.json`。

`full_cv.py` 每折重新执行原算法：训练来源上加权梯度归因，各类top64，每类最多12个来源均衡训练对的双向单头干预，交替融合A/P和C/F排名至51头；随后只用本折训练来源估计标准化、子空间和岭回归。验证来源不参与任何选择或拟合。相同pair/head的旧真实干预可复用；新增干预逐条写入 `full/fold*/exact.jsonl`。所有分数均使用既有 Engine 的 BF16 主模型、不可变前缀缓存和 FP32 标签投影。

从仓库根目录执行：

```bash
source /home/gcx/gcx261014/bin/activate
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/run_cv.py
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/source_diagnostics.py
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --prepare
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --folds 0 2 4
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --folds 1 3
python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --aggregate
```

两个GPU命令可以在两个终端同时运行。首次执行完整流程需要Qwen3-8B，预计每卡约18–24 GiB显存。已完成的折会跳过，未完成的折复用已保存的干预。全部干预完成后，以下命令仅使用CPU重新计算选择、子空间与回归，保存到 `full/replay/`，无需再次运行模型：

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full_cv.py --replay --folds 0 1 2 3 4
```

逐对预测与每来源、每折指标保存在同级CSV；冻结诊断模型保存在 `frozen_heads_fold*.npz`，完整流程模型和训练来源列表保存在 `full/fold*/`。种子20261109用于来源分折，子空间沿用仓库种子20261108。

## 来源差异的实测观察

原20个dev留出来源中17个来源的A/P R²为正，来源中位数为0.2860。最差来源train_val:1780为肾病/风湿免疫用药条件，n=30，目标标准差1.474、RMSE1.829、R²=-0.5389；train_val:2420为放射性膀胱损伤条件，n=45，目标标准差1.396、RMSE1.459。它们的来源内分数变化较小，误差已超过该来源变化尺度，因此R²为负。train_val:998为儿童营养和腹泻等条件，n=51，目标标准差4.414，但预测标准差仅1.326，且目标均值-2.680、预测均值-1.361，表现出对该来源较大变化的预测不足。

完整流程最差来源为train_val:304：n=132，R²=-1.3011、RMSE1.7428、目标标准差1.1489。该来源包含较长CAR-T治疗决策树，提示词长度中位数9409字符；目标均值-0.045，预测均值1.153，存在明显来源内偏差。第二差的train_val:772为儿童居家心理行为管理条件：n=36、R²=-0.9686，目标均值-0.472、预测均值0.349；第三差的train_val:1946为英文艰难梭菌感染/肠病条件：n=33、R²=-0.7099，目标均值0.397、预测均值-0.208，预测标准差0.887小于目标标准差1.577。均值偏差、变化尺度预测不足和较少配对是直接观察到的现象；长度、语言和主题仅是可观察差异，当前数据没有识别哪一项导致误差。完整流程结果优先于冻结头诊断用于回答泛化问题。

## 测试历史与新来源

repair-v2原方法在2026-11-08 16:51:33冻结后已完成test评分与路径评价，旧 `meeting/real_completion/metrics.json` 也明确记载已有测试使用。因此本次test只能称为既有留出集。原准备过程的库存中有1346个当时可用来源，仅取140个；其余来源尚无本流程确认的未使用来源激活缓存，不能声称已经完成全新独立验证。外部NLI4CT/MedPIC已有构造文件，但未发现相应Qwen评分缓存，不纳入跨指南结论。
