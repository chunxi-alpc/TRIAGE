# 模块二后续实验运行入口

**已准备：** 十一种干预条件、固定协议、真实模型运行器、按家族统计分析器、CPU 数值与日志完整性检查。

**尚未执行：** PyTorch 适配器数值检查、真实模型钩子验证和 GPU 推断。当前 `checks/` 中的材料仅为 CPU 预检，不能作为模型效果。

实验设计见 [冻结协议](PROTOCOL.zh.md) 和 [机器可读参数](protocol.json)。旧结果不被改写。

## 无 GPU 的准备检查

从仓库根目录执行：

```bash
python experiments/care_reliability_20261019/module2/run.py --help
python experiments/care_reliability_20261019/module2/analyze.py --help
python -m unittest discover -s experiments/care_reliability_20261019/module2 -p 'test_*.py' -v
```

`test_backend_torch.py` 在依赖缺失时明确跳过。其余数值检查使用 NumPy；配置/版本拒绝和日志覆盖检查使用 CPU 测试夹具，均不代表真实模型或 CUDA 验证。

以下命令只做文件、划分与数据格式预检，不加载模型。输出目录必须尚不存在。

```bash
python experiments/care_reliability_20261019/module2/run.py \
  --preflight --mode legacy384_regression --fold 0 \
  --output /tmp/care-m2-preflight-legacy-fold0

python experiments/care_reliability_20261019/module2/run.py \
  --preflight --mode sealed_mechanism --fold 0 \
  --data experiments/care_reliability_20261019/data/module2_sealed.jsonl \
  --data-manifest experiments/care_reliability_20261019/data/manifest.json \
  --output /tmp/care-m2-preflight-sealed-fold0
```

另外两折分别改为 `--fold 1`、`--fold 2` 和新的输出目录。六项 CPU 预检的简表保存在 `checks/preflight_summary.json`。

## 在已确认的真实执行环境续跑

先核对执行模式、服务器与当前授权 GPU，打印并确认资源计划，再运行真实任务。旧记录中的环境为 `/home/gcx/gcx261014`，固定模型历史路径为 `/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base`；两者须在真实环境现场核对，当前云端没有验证该路径可达。

所需 Python 包为 NumPy、PyTorch、Transformers，其中原机制兼容门槛要求安装包版本精确为 PyTorch `2.8.0` / Transformers `4.57.6`。模型目录须含原始权重分片及对应 tokenizer/config。本运行器不下载权重。原权重哈希来自 `experiments/care_qwen3_4b_base_20261016/results/manifest.json`；该 manifest 本身也按冻结 SHA256 核验。21 个关键 config 字段在 AutoConfig 解析后与模型加载后各核对一次。环境差异及配置证据写入 `model_compatibility.json`。

一个 GPU 进程的命令如下。`CUDA_VISIBLE_DEVICES` 必须由协调者设置为当前已授权物理 GPU；`--device cuda:0` 指向该进程可见的逻辑设备。

```bash
M2_MODEL=/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base
M2_RUN_ROOT=/home/gcx/care_reliability_runs/module2_20261019_run1

python experiments/care_reliability_20261019/module2/run.py \
  --mode legacy384_regression --fold 0 --model "$M2_MODEL" \
  --output "$M2_RUN_ROOT/legacy384_regression/fold0"

python experiments/care_reliability_20261019/module2/run.py \
  --mode sealed_mechanism --fold 0 --model "$M2_MODEL" \
  --data experiments/care_reliability_20261019/data/module2_sealed.jsonl \
  --data-manifest experiments/care_reliability_20261019/data/manifest.json \
  --output "$M2_RUN_ROOT/sealed_mechanism/fold0"
```

其余两折使用相同参数、对应折号及各自的新输出目录。不得根据确认集结果修改协议后继续使用同一运行标签。

三折真实模型执行完成后，分别分析两种模式：

```bash
python experiments/care_reliability_20261019/module2/analyze.py \
  --run-root "$M2_RUN_ROOT/legacy384_regression" \
  --output "$M2_RUN_ROOT/legacy_analysis"

python experiments/care_reliability_20261019/module2/analyze.py \
  --run-root "$M2_RUN_ROOT/sealed_mechanism" \
  --output "$M2_RUN_ROOT/sealed_analysis"
```

每例都增加直接偏置零强度前向，验证 mask 头维度扩展后的候选 logit/概率与基线精确相等；十个键条件的零检查仍只在每折首例执行。六任务在没有策略无操作时的前向上限为 8076 次，实际次数写入日志。零控制有差异会停止运行。

分析结果包括主要配对比较、家族置信区间、每家族结果、干预预算/锚点方向/覆盖率、历史冲突子集及新集变体稳定性。分析器会拒绝尚未完成真实推断、配置/环境不兼容或逐病例零控制不完整的日志。

## 联合实验接口

`compatibility.py` 的 `load_frozen_manifest`、`config_audit`、`environment_audit` 均只依赖标准库，供联合运行器复用相同原模型约束；返回的审计应先保存，`compatible=False` 时停止真实推断。

`backend.InterventionGeometry` 接收同一个未干预单层前向所捕获的查询、键、动作词元位置、固定层/KV 头与协议。`key_intervention(direction, calibration_mode, operators, sign, target_rms)` 返回修改后的动作键及审计信息。

联合实验可对同层多个互不重叠的动作词元片段分别计算修改，再用一次钩子写入。必须先检查分词后的片段也不重叠。不同规则的预算可以按各自置信度缩放，但合并后的总均方根需另算；这种组合没有实现证据预算，也不能直接沿用单片段的总预算表述。

`no_eligible_direction` 是明确的无操作，调用方须保留原键并记录覆盖率。不可将未学习到的 B4 资产用随机算子替代为“已学习投影”。
