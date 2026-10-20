# 模块二接口预实验：环境与执行边界

本文件对应仓库参考快照 `7557d3150214a3894f2d340df6d9964de75efce1`。当前本地目录是从 GitHub 校验取得的部分快照；该 SHA 是输入来源，不能解释为本地 `git HEAD` 的检查结果。

**本次环境预检的结论为 `blocked`。未执行新的模型加载、GPU 前向、模块一推理或完整模块二实验。** 本目录提供的统计复核、数值实现和协议不构成已完成的 P0–P6 模型实验套件。

## 1. 本次实测

完整机器可读记录见 [results/preflight.json](results/preflight.json)。该文件由 [preflight.py](preflight.py) 在本地实际执行产生，时间为 `2026-10-20T05:32:21.855958+00:00`，退出码为 `2`，含义为前提未满足。

| 检查 | 本次观测 | 含义 |
| --- | --- | --- |
| Python | 3.12.14 | 本地解释器版本 |
| NumPy / SciPy | 2.3.5 / 1.17.0，可实际导入 | 可做已有结果的统计分析和线性代数检查 |
| PyTorch / Transformers | 未安装，不能导入 | 不能在当前 Python 环境运行模型 |
| CUDA 检查 | 无 PyTorch；`nvidia-smi` 不可用 | 当前环境没有确认可用的 CUDA 执行条件，不是对物理硬件的完整清点 |
| Qwen3-4B-Base | 指定路径不存在 config、索引、权重分片和 tokenizer | 不能加载、核验或执行目标模型 |
| 原始旧 192 条 / 新 100 条输入 | 两个预期路径均不存在 | 不能据此执行冻结模块一驱动的患者级新推理 |
| 冻结模块一预测 | 192 条及 100 条均存在，样本 ID 唯一 | 可在不重算模块一的前提下审计保存的信号 |
| 模型历史 provenance | revision 与三个权重哈希和固定记录一致 | 只是核验历史证据文件；不表示当前拥有权重 |
| 规则动作字段 | 旧集 16/16 条、新集 9/9 条无有效 `action` | 当前数据不提供完整临床决策动作 |

预检同时完成两项 CLI 行为验证：已有输出文件被拒绝覆盖、原文件哈希保持不变；请求物理 GPU `4` 被授权范围检查阻断。这些验证使用临时目录，未运行模型，也不计入方法有效性实验。

## 2. SSH 实测与审批状态

协调执行代理在本次会话已尝试两次连接仓库记录的服务器。逐项事实见 [results/connectivity.json](results/connectivity.json)。本环境审计没有重新扫描网络。

| 次序 | 执行方式 | 观测 | 能得出的结论 |
| --- | --- | --- | --- |
| 1 | 默认沙箱 | socket 阶段 `Operation not permitted` | 本地受限执行环境阻止连接 |
| 2 | 请求提升执行权限，自动审批已允许 | `10.29.128.115:22` 返回 `Connection refused` | 授权后的连接仍未建立，未到身份验证阶段 |

**第二次不是自动审批拒绝，也不是已证实的凭据错误。** 尚未登录服务器，未执行远程命令，未确认当前模型、数据和 GPU 状态，未启动远程任务。没有读取、生成或传送凭据。

## 3. 历史记录：用于恢复时核对

以下内容来自仓库文档与已保存的真实运行记录，不能视为本次环境实测。

| 项目 | 记录值 | 来源 |
| --- | --- | --- |
| 服务器 | `gcx@10.29.128.115` | [运行说明](../../docs/RUNNING.md) |
| 服务器仓库 | `/home/gcx/TRIAGE` | 运行说明 |
| 服务器 Python 环境 | `/home/gcx/gcx261014` | 运行说明 |
| 可使用的物理 GPU | `0,1,2,3,6,7` | 运行说明 |
| 模块二模型 | `Qwen/Qwen3-4B-Base` | [有符号诊断协议](../care_module2_signed_20261019/PROTOCOL.md) |
| 模型 revision | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | [历史 provenance](../care_module2_signed_20261019/results/fold0/provenance.json) |
| 权重目录 | `/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base` | [diagnostic.py](../care_module2_signed_20261019/diagnostic.py) 默认参数 |
| 历史临床诊断数据 | `/home/gcx/care_20261016/CARE_Codex_Pilot/data/clinical/candidates.jsonl` | 同上 |
| 历史模型架构 | 36 层、32 个 query head、8 个 KV head、head dimension 128 | 有符号诊断协议 |
| 历史推理方式 | BF16、SDPA、batch size 1、无 KV cache、单 token 候选评分 | 有符号诊断协议 |
| 历史依赖 | torch 2.8.0、transformers 4.57.6 | 历史 provenance |

旧有符号实验覆盖 384 个 case、192 个配对、12 个规则族；历史记录报告每 fold 约 94–96 秒，峰值 allocated 不高于 7.59 GiB、reserved 不高于 7.60 GiB。旧启动资源计划是四张卡 `0,1,2,3`、每卡预计 10–16 GiB、预计 10–20 分钟。**前者是旧轮运行记录，后者是旧轮启动估计；均不是本次新实验的实测或时长承诺。** 新实验启动前仍须单独记录用卡、显存预算、样本量和时长预算。

不要直接调用旧 `launch.py`：其输出目录固定为旧实验的 `results`。本次没有调用该脚本。不要调用旧 V6 `evaluate_v6.py` 重写历史输出，也不需要重跑已冻结模块一。

## 4. 输入恢复与完整实验的边界

原始输入 SHA256 已由 [new_set_manifest.json](../care_module1_v6/data/new_set_manifest.json) 固定：

| 文件 | SHA256 |
| --- | --- |
| `inputs_192.jsonl` | `32ec4446d80a3f514827da8a5df8a0206aa4eceea63f9940df99876de7c4955b` |
| `new_inputs_100.jsonl` | `3d0c58bd3cadb0b23038954bee95df58e066936d7de21623f09d39e7e4de8602` |

默认在 `experiments/care_module1_v6/data/` 查找这些文件，也可用 CLI 显式指定实际路径。新增 100 条输入的历史来源目录为 `/home/gcx/care_module1_20261017_v3/candidate_data4`。路径存在后，预检还必须验证文件哈希和行数；仅创建同名文件不能通过验证。

旧 192 条 V6 预测是已有 V5 输出的离线重合成；新 100 条保存的是 V6 实际逐例预测。二者均冻结使用。原始患者文本不能从 gold 标签、证据区间或预测结果反向补造。

当前数据主要给出适用性 `T/F/U`。即使恢复权重和患者输入，也不会自动补出以下实验条件：

- **路由控制域。** 需要先定义预测哪类答案、规则动作、词元角色、配对、按规则族切分及 held-out 评价；当前无效的动作字段不能被解释为治疗答案。
- **来源仲裁。** 需要经过闭卷知识核验的上下文/先验冲突最小对比，以及 C/P 目标。适用性标签不足以构建这些目标。
- **可决断仲裁。** 需要关键属性已知/未知的配对，以及回答/追问目标。安全拒答标签不能替代缺失属性的可决断性标签。
- **联合接口。** 需要将以上变量、冻结模块一信号、真实决策位置和输出评价对齐；当前目录未实现完整的新模型 P0–P6 runner。

历史 384 个有符号路由 case 可从 GitHub 恢复，用于重分析已保存结果或复现该历史诊断。它们不提供新的模块一驱动样例、来源/可决断性轴，或关键属性未知条件。因此，运行前提、数据覆盖范围、实现完成状态和实际有效性结果在报告中分别记录。

## 5. 可移植预检命令

`preflight.py` 只使用 Python 标准库组织检查。它实际尝试导入已安装的依赖，在离线模式下读取现有文件，并在可用时读取一次 `nvidia-smi` 设备信息。它不安装依赖、不联网下载、不发起 SSH、不加载模型、不分配推理张量、不重算模块一。唯一有意写入是显式指定的 `--out` JSON，使用排他创建，拒绝已有文件或符号链接。

本地已执行的命令为：

```bash
python -B experiments/care_module2_interface_20261020/preflight.py \
  --repo-root /workspace/scratch/acd4a488f203/TRIAGE \
  --model-path /home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base \
  --gpu 0 \
  --out experiments/care_module2_interface_20261020/results/preflight.json
```

该输出已经存在，原命令再次运行会拒绝覆盖。未来恢复已授权计算环境、并将本实验代码放入服务器工作副本后，可使用新的输出路径执行下面的**预检**命令。此处提供命令不表示已经同步代码、连接成功或模型套件已可运行。

```bash
source /home/gcx/gcx261014/bin/activate
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -B /home/gcx/TRIAGE/experiments/care_module2_interface_20261020/preflight.py \
  --repo-root /home/gcx/TRIAGE \
  --model-path /home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base \
  --gpu 0 \
  --old-input /home/gcx/TRIAGE/experiments/care_module1_v6/data/inputs_192.jsonl \
  --new-input /home/gcx/TRIAGE/experiments/care_module1_v6/data/new_inputs_100.jsonl \
  --out /home/gcx/TRIAGE/experiments/care_module2_interface_20261020/runs/recovery_01/preflight.json
```

`--gpu` 使用物理编号，可传 `0,1` 等逗号分隔列表；允许集合是 `0,1,2,3,6,7`。CUDA 的进程内编号与物理编号不能混用。存在 UUID/MIG 形式的可见性掩码时，本预检会要求进一步核对映射。

退出码 `2` 表示阻塞或输出路径无效。退出码 `0` 的范围仅为 `ready_for_limited_runtime_checks`，表示列出的运行前提通过；报告仍明确 `complete_new_model_suite_runnable=false`，也不会把检查成功写成方法有效。
