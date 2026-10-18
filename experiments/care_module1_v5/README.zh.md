# CARE 模块一 V5

V5 在 [src/module1/care_v5.py](../../src/module1/care_v5.py) 实现。`from src import module1` 后，`module1.run(guideline, patient, question)` 使用 V5；首次调用时加载指定的 TrialMatchAI 适配器和 MedNLI 模型，不导入旧 Qwen 后端。可选地用 `module1.set_default_v5(module1.CAREModule1V5.from_pretrained(device="cuda:6"))` 预先初始化。默认 `module1.Module1` 类同样使用 V5；旧实现只能显式通过 `module1.run_legacy` 或 `module1.LegacyModule1` 使用。

TrialMatchAI 适配器是 Phi-4 上的 PEFT LoRA 生成模型，不是独立分类头。V5 按官方源码固定版本使用聊天模板及 JSON 输出格式，每次请求只含一个原子条件，批量大小为 8、生成上限 2048 token；保存原始生成。仅从目标条件的 Classification 读取标签，截断或解析失败记为 U。批量时间按每批实际请求数分摊，属于吞吐时间分配，不是单请求延迟。入组标签映射为：`Met→T`、`Not Met→F`、`Unclear/Irrelevant→U`。例外条件按排除条件映射：`Violated→T`、`Not Violated→U`。任何未知标签都映射为 U；适配器加载失败会停止，不会回退基础 Phi-4 或 Qwen3。BioLinkBERT 根据实际模型配置读取标签映射；蕴含/矛盾/中立分别保留为支持/反驳/未知。只有找到明确数值及单位时才运行确定性阈值判断；有多个未撤回数值时设为 U。

患者文本按句子拆分并保存 Python 字符位置 `[start,end)`。BioLinkBERT 只接收筛选出的候选句和单个条件。超过 512 token 的候选句会记为未知并跳过，不会静默截断；完整 TrialMatchAI 输入若超出上下文也会停止该样本。明确标记为撤回的旧记录不用于证据筛选；未撤回的相互矛盾报告继续作为冲突保留。

## 数据和限制

当前实验使用 V4 的 192 条探索性回归样本、16 条结构化规则和已保存的 Qwen3 直接判定。V4 数据参与了方法设计，故不能声称独立测试。数据包括 32 个原始合成病例和每个病例的控制变体；标签没有经过临床专家复核。V5 公共目录只保留标签、样本 ID、证据句编号/字符区间和模型分数；病例原文、完整模型生成和带原文引用的模块输出只写入服务器 `runs/<run_id>/private/`，不会提交。

另有 [24 条新规则合成探索样本](exploratory/README.zh.md)，使用此前未用的 4 条完整结构树规则，运行前冻结，具有明确动作位置；这些仍不是临床独立金标。

V4 提供的是研究用谓词改写，没有完整原始指南文本、治疗动作文本或其原文区间。`data/rules_v5.jsonl` 的条件区间指向规范化规则包中的离线条件，不冒充原始指南位置；动作为空并在输出中显式保留。要支持正式下游动作决策，仍需提供有来源许可的原始指南动作片段及独立验证标签。

评价分别保留三组：纯 TrialMatchAI 标签与三值逻辑、TrialMatchAI 加确定性数值比较（方法 A）、以及在方法 A 上加入 BioLinkBERT-MedNLI 的证据关系（方法 B）。这样可以单独看出数值比较和证据模型各自带来的变化。

## 服务器复现

在服务器仓库工作树根目录执行。以下命令只应在服务器运行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
/home/gcx/.local/bin/uv pip install --python /home/gcx/n/.venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple peft bitsandbytes
bash experiments/care_module1_v5/scripts/download_models.sh

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/validate_model_configs.py \
  --base /home/gcx/models/care_module1_v5/phi-4 \
  --adapter /home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora \
  --evidence /home/gcx/models/care_module1_v5/biolinkbert-mednli \
  --output experiments/care_module1_v5/data/model_validation.json

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/prepare_public_data.py \
  --v4-labels /home/gcx/care_module1_20261018_v4/data/labels.jsonl \
  --v4-predictions /home/gcx/care_module1_20261018_v4/data/model/predictions.jsonl \
  --v4-pairs /home/gcx/care_module1_20261018_v4/data/pairs.jsonl \
  --pairs-out experiments/care_module1_v5/data/pairs_public.jsonl \
  --labels-out experiments/care_module1_v5/data/labels_public.jsonl \
  --qwen-reference-out experiments/care_module1_v5/data/v4_qwen_direct_public.jsonl

/home/gcx/n/.venv/bin/python experiments/care_module1_v5/scripts/prepare_rules.py \
  --manifest /home/gcx/care_module1_20261018_v4/data/rules_manifest.jsonl \
  --output experiments/care_module1_v5/data/rules_v5.jsonl

/home/gcx/n/.venv/bin/python -m pytest experiments/care_module1_v5/tests -q
bash experiments/care_module1_v5/scripts/run_all_v5.sh
```

不要把 `runs/*/private/` 或病例原文加入版本控制。模型加载、推理及输出都强制使用 CUDA；没有 CPU、基础 Phi-4 或 Qwen3 回退。结果报告由 `scripts/generate_report.py` 从 `run_status.json`、指标和错误文件生成。

## V4 对照

V4 保存的直接判定为 136/192（70.83%）、宏 F1 0.7041、已知样本证据准确率 87/137（63.50%）、联合正确 81/192、反事实成对正确 22/32、生成耗时 79.2 秒。V5 使用同一批样本进行探索性回归，同时保存 TrialMatchAI 独立判定和加入证据模型后的结果。

运行前先执行 `scripts/resume_phi4_modelscope.py`，可续传与 HF 摘要一致的 Phi-4 权重分片；适配器与 MedNLI 可用 `scripts/download_small_models.py` 下载，均逐文件校验。实际下载记录保存在公开 provenance 和服务器模型目录。完整顺序运行使用 `bash experiments/care_module1_v5/scripts/run_suite.sh`；它先运行 192 条回归，再运行 24 条新合成探针，两组共用 GPU 6。
