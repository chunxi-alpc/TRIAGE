CARE 模块一 v4：证据修复和新规则完整病例验证

先读 REPORT.zh.txt；完整指标 data/metrics.json；逐例预测 data/model/predictions.jsonl；逐字段事实、状态和定位记录 data/model/normalized_records.jsonl；错误 data/errors.jsonl；协议 PROTOCOL.zh.txt。
本目录是可公开交付版本。原始完整病例、含原文引用的模型输出只在服务器 /home/gcx/care_module1_20261018_v4 保留。公开 facts_raw.jsonl 和 verification_raw.jsonl 去除原文引用，保留原输出和记录 SHA256；不能拿删减后的引用重新做证据核对，但可用预测文件原样复算所有公开指标。

服务器复算已保存结果（无 GPU）：
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/care_module1_20261018_v4
python reproduce.py

从有权访问的固定 MedGUIDE 数据重建并检查输入哈希（无 GPU）：
python reproduce.py --source /home/gcx/TRIAGE/data/MedGUIDE

该命令也会恢复缺少的 data/private 输入。report.py 的详细错误分析需要这些私有输入；仅复算公开指标不需要源数据。公开 facts_raw 与 verification_raw 中的 parsed 是归一化后的值、编号和状态，不是完整原始 JSON；原始记录的哈希及服务器路径均保留。

如需从头再次运行模型，在事先报告 GPU 需求后运行（新输出到 replication_data，保留旧结果）：
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduce.py --source /home/gcx/TRIAGE/data/MedGUIDE --model /home/gcx/n/models/Qwen3-8B --run

运行环境：Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.57.6、pandas、numpy。实际环境和资源见 data/model/environment.json；权重指纹见 provenance/model_files.json（复用此前同路径已计算的指纹）；数据版本及文件哈希见 data/dataset_audit.json。
FREEZE.json 是模型启动前保存的代码及私有输入哈希。公开包删去了受限原文，因此在恢复源数据前，不能直接要求全部私有文件都存在。reproduce.py 可检查重建输入和公开指标；GPU 重跑可能受硬件/库版本影响，不承诺跨环境逐 token 一致。

来源：
MedGUIDE-MCQA-8K：https://huggingface.co/datasets/MedGUIDE/MedGUIDE-MCQA-8K ，固定版本 1460ec478280a7295040b92f7eb985327f77dcbe。数据卡许可证待明确，因此不上传整套病例与第三方原文；使用者需按来源许可自行取得数据。
Qwen3-8B：https://huggingface.co/Qwen/Qwen3-8B ，服务器既有权重，不随结果分发。
TrialGPT-Matching：https://github.com/ncbi-nlp/TrialGPT/blob/main/trialgpt_matching/TrialGPT.py ，只参考结构化匹配思路，本实验不是其复现。

范围：这是合成病例加控制修改的研究条件判断，未经独立临床专家裁定，不是临床效果验证。32 个原病例是真实读取的数据集记录，不是 32 位真实患者；160 个修改样本由代码构造并明确标记。
